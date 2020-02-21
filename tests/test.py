import comet_ml
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from options.continual_options import ContinualOptions
from models import create_model
from data import create_dataset
from copy import copy
from eval import eval

if __name__ == "__main__":

    # local: run test.py --dataroot=../s2w_d --gpu_ids -1 --netG continual

    opt = ContinualOptions().parse()
    model = create_model(opt)
    dataset = create_dataset(opt)
    test_opt = copy(opt)
    test_opt.serial_batches = True
    test_opt.phase = "test"
    test_dataset = create_dataset(test_opt)

    G = model.netG_A
    e = G.encoder
    d = G.depth
    t = G.decoder
    r = G.rotation

    x = torch.rand(1, 3, 256, 256)
    z = e(x)

    g_step = False
    eval_model = True

    if g_step:
        b = next(iter(dataset))

        print("Setting input")
        model.set_input(b)
        print("Forward")
        model.forward()
        print("Setting requires_grad : False to netDs")
        model.set_requires_grad(
            [model.netD_A, model.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        print("Backward")
        model.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        model.backward_G()  # calculate gradients for G_A and G_B
        for d in dir(model):
            if d.startswith("loss_") and "names" not in d:
                print("{:15} {:.3f}".format(d, getattr(model, d).item()))
        print("Step")
        model.optimizer_G.step()  # update G_A and G_B's weights

    if eval_model:
        exp = comet_ml.Experiment(project_name="continual-translation")
        eval(model, exp, 1234)
