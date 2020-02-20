import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from options.continual_options import ContinualOptions
from models import create_model
from data import create_dataset


if __name__ == "__main__":

    # local: run test.py --dataroot=../s2w_d --gpu_ids -1 --netG continual

    opt = ContinualOptions().parse()
    model = create_model(opt)
    dataset = create_dataset(opt)

    G = model.netG_A
    e = G.encoder
    d = G.depth
    t = G.decoder
    r = G.rotation

    x = torch.rand(1, 3, 256, 256)
    z = e(x)

    for i, b in enumerate(dataset):
        print(b["angleA"])
        print(b["angleB"])
        if i > 20:
            break
