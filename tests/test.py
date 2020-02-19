import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from options.continual_options import ContinualOptions
from models import create_model


if __name__ == "__main__":

    opt = ContinualOptions().parse()
    model = create_model(opt)

    G = model.netG_A
    e = G.encoder
    d = G.depth
    t = G.decoder
    r = G.rotation

    x = torch.rand(1, 3, 256, 256)
    z = e(x)
