### DIfferentiable PSF model (DIP)

DIP is a flexible and versatile framework for focal-plane wavefront sensing based on PyTorch. 

## Requirements

DIP intrinsically depends on LIFT. please download the [LIFT repository](https://github.com/EjjeSynho/LIFT) first. While installing, please ensure the following folder structure:

```
LIFT
├── tests
│   ├── example.ipynb
│   └── ...
├── modules
│   ├── Telescope
⁞   ├── LIFT
    └── ...
DIP
├── tests
│   ├── example.ipynb
│   └── ...
├── DIP.py
⁞
```

## Required modules
```
pytorch
graphviz (optional)

```

## Acknowledgments
This code was developed during PhD program of Arseniy Kuznetsov, funded by ESO, LAM, and ONERA.