#%%
# Commom modules
import torch
import numpy as np
from torch import nn
from torch.nn.functional import conv3d


class DIP(nn.Module):
    def __init__(self, tel, device, norm_mode):
        super().__init__()
        
        self.oversampling = 1
        self.norm_mode = norm_mode
        self.img_size  = tel.img_resolution
        self.device    = device

        self.tel = tel
        self.tel_pupil = torch.tensor(self.tel.pupil).to(self.device)

        #flux is [photon/m2/s] per λ  TODO: account for the reflectivity map
        self.flux = torch.tensor( [point['flux']       for point in self.tel.src.spectrum], device=self.device )
        self.λs   = torch.tensor( [point['wavelength'] for point in self.tel.src.spectrum], device=self.device )

        #TODO: redo flux for my sampling!
        #TODO: set oversampling when undersampled
        pixels_λ_D = self.tel.f/self.tel.det.pixel_size * self.λs.cpu().numpy()/self.tel.D
        self.oversampling = self.oversampling + int(self.oversampling%2 != self.img_size%2)*int(self.oversampling!=1) # this is to bin images with odd number of pixels properly

        pad = np.round((self.oversampling*pixels_λ_D-1)*self.tel_pupil.shape[0]/2).astype('int')
        self.φ_size  = self.tel_pupil.shape[0] + 2*pad
        self.photons = self.flux/self.tel_pupil.sum() * self.tel.pupilReflectivity * self.tel.area * self.tel.det.sampling_time
        self.padders = [torch.nn.ZeroPad2d(val.item()) for val in pad]


    def _to_device_recursive(self, obj, device):
        if isinstance(obj, torch.Tensor):
            if obj.device != device:
                if isinstance(obj, nn.Parameter):
                    obj.data = obj.data.to(device)
                    if obj.grad is not None:
                        obj.grad = obj.grad.to(device)
                else:
                    obj = obj.to(device)
                    
        elif isinstance(obj, nn.Module):
            obj.to(device)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._to_device_recursive(item, device)
        elif isinstance(obj, dict):
            for item in obj.values():
                self._to_device_recursive(item, device)
        return obj


    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if self.device == device:
            return self
        self.device = device
        
        for name, attr in self.__dict__.items():
            new_attr = self._to_device_recursive(attr, device)
            if new_attr is not attr:
                # print(f"Transferring '{name}' to device '{device}'")
                setattr(self, name, new_attr)
        return self


    def binning(self, inp, N):
        return torch.nn.functional.avg_pool2d(inp.unsqueeze(1),N,N).squeeze(1) * N**2 if N > 1 else inp


    def OPD2PSF(self, photons, λ, OPD, φ_size, padder, oversampling):  
        amplitude = torch.sqrt(photons)*self.tel_pupil #     V--- conversion of OPD [nm]->[m]
        EMF = padder( amplitude * torch.exp(2j*torch.pi/λ*OPD*1e-9) )

        lin = torch.linspace(0, φ_size-1, steps=φ_size, device=self.device)
        xx, yy = torch.meshgrid(lin, lin, indexing='xy')
        center_aligner = torch.exp(-1j*torch.pi/φ_size*(xx+yy)*(1-self.img_size%2))

        PSF = torch.fft.fftshift(1./φ_size * torch.fft.fft2(EMF*center_aligner, dim=(-2,-1)), dim=(-2,-1)).abs()**2
        cropper = slice(φ_size//2-(self.img_size*oversampling)//2, φ_size//2+round((self.img_size*oversampling+1e-6)/2))

        return self.binning(PSF[...,cropper,cropper], oversampling)


    def forward(self, OPD, obj=None):
        if OPD.ndim == 2:
            OPD = OPD.unsqueeze(0)
        N = OPD.shape[0] # number of PSF samples in the stack

        PSF = torch.zeros([N, self.img_size, self.img_size], dtype=OPD.dtype, device=self.device)
        for i in range(len(self.tel.src.spectrum)):
            PSF += self.OPD2PSF(self.photons[i], self.λs[i], OPD, self.φ_size[i].item(), self.padders[i], self.oversampling)
        
        if obj is not None:
            PSF_conv = conv3d(
                PSF.unsqueeze(1).unsqueeze(0), obj.unsqueeze(1).unsqueeze(1),
                bias=None, stride=1, padding='same', groups=N).squeeze(0).squeeze(1)
            return self.normalize(PSF_conv)
        else:
            return self.normalize(PSF)


    # Normalize a PSF batch depending on the normalization regime
    def normalize(self, inp):
        if self.norm_mode == 'sum':
            return inp / inp.sum(dim=(1,2), keepdim=True)
        elif self.norm_mode == 'max':
            return inp / torch.amax(inp, dim=(1,2), keepdim=True)
        else:
            return inp


try:
    from graphviz import Digraph
except ImportError:
    pass
else:
    def iter_graph(root, callback):
        queue = [root]
        seen = set()
        while queue:
            fn = queue.pop()
            if fn in seen:
                continue
            seen.add(fn)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append(next_fn)
            callback(fn)


    def register_hooks(var):
        fn_dict = {}
        def hook_c_b(fn):
            def register_grad(grad_input, grad_output):
                fn_dict[fn] = grad_input
            fn.register_hook(register_grad)
            
        iter_graph(var.grad_fn, hook_c_b)

        def is_bad_grad(grad_output):
            if grad_output is None:
                return False
            return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

        def make_dot():
            node_attr = dict(style='filled',
                            shape='box',
                            align='left',
                            fontsize='12',
                            ranksep='0.1',
                            height='0.2')
            
            dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

            def size_to_str(size):
                return '('+(', ').join(map(str, size))+')'

            def build_graph(fn):
                if hasattr(fn, 'variable'):  # if GradAccumulator
                    u = fn.variable
                    node_name = 'Variable\n ' + size_to_str(u.size())
                    dot.node(str(id(u)), node_name, fillcolor='lightblue')
                else:
                    def grad_ord(x):
                        mins = ""
                        maxs = ""
                        y = [buf for buf in x if buf is not None]
                        for buf in y:
                            min_buf = torch.abs(buf).min().cpu().numpy().item()
                            max_buf = torch.abs(buf).max().cpu().numpy().item()

                            if min_buf < 0.1 or min_buf > 99:
                                mins += "{:.1e}".format(min_buf) + ', '
                            else:
                                mins += str(np.round(min_buf,1)) + ', '
                            if max_buf < 0.1 or max_buf > 99:
                                maxs += "{:.1e}".format(max_buf) + ', '
                            else:
                                maxs += str(np.round(max_buf,1)) + ', '
                        return mins[:-2] + ' | ' + maxs[:-2]

                    assert fn in fn_dict, fn
                    fillcolor = 'white'
                    if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                        fillcolor = 'red'
                    dot.node(str(id(fn)), str(type(fn).__name__)+'\n'+grad_ord(fn_dict[fn]), fillcolor=fillcolor)
                for next_fn, _ in fn.next_functions:
                    if next_fn is not None:
                        next_id = id(getattr(next_fn, 'variable', next_fn))
                        dot.edge(str(next_id), str(id(fn)))
            iter_graph(var.grad_fn, build_graph)
            return dot

        return make_dot

    # Q = loss_fn(dip(OPD=GetOPD_prob(mu_A, sigma_A)), data)
    # get_dot = register_hooks(Q)
    # Q.backward()
    # dot = get_dot()
    # # #dot.save('tmp.dot') # to get .dot
    # # #dot.render('tmp') # to get SVG
    # dot # in Jupyter, you can just render the variable