#%%
# Commom modules
import torch
from graphviz import Digraph
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, tolerance=1e-1, relative=False):
        self.__patience  = patience
        self.__tolerance = tolerance
        self.__previous_loss = 1e16
        self.__counter = 0
        self.__relative = relative
        self.stop = False

    def __compare(self, a, b):
        return abs(a/b-1) < self.__tolerance if self.__relative else abs(a-b) < self.__tolerance

    def __call__(self, current_loss):
        if self.__compare(self.__previous_loss, current_loss.item()):
            self.__counter += 1 
            self.stop = True if self.__counter >= self.__patience else False
        else: self.__counter = 0
        self.__previous_loss = current_loss.item()


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