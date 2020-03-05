import comet_ml
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from options.continual_options import ContinualOptions
from models import create_model
from data import create_dataset
from copy import copy
from eval import eval


def show_tensor(t):
    t = t.detach().cpu().numpy()
    t -= t.min()
    t /= t.max()
    t = np.transpose(t, (1, 2, 0))
    plt.imshow(t)
    plt.show(block=True)


if __name__ == "__main__":

    # local: run test.py --dataroot=../s2w_d --gpu_ids -1 --netG continual
    # beluga: python test.py --dataroot=$SLURM_TMPDIR/s2w_d --netG continual --batch_size=5

    opt = ContinualOptions().parse()
    dataset = create_dataset(opt)
    test_opt = copy(opt)
    test_opt.serial_batches = True
    test_opt.phase = "test"
    test_dataset = create_dataset(test_opt)

    g_step = True
    eval_model = True
    show_rot = False

    if show_rot:
        b = next(iter(dataset))
        c = False
        i = 0
        while c:
            print(b["angleA"][i])
            show_tensor(b["rA"][i])
            c = input("Next Image? (y)") == "y"
            i += 1
        model = create_model(opt)
        model.set_input(b)
        model.forward()

    if g_step:
        model = create_model(opt)
        G = model.netG_A
        if isinstance(G, torch.nn.DataParallel):
            e = G.module.encoder
            d = G.module.depth
            t = G.module.decoder
            r = G.module.rotation
        else:
            e = G.encoder
            d = G.depth
            t = G.decoder
            r = G.rotation

        x = torch.rand(1, 3, 256, 256).to(next(G.parameters()).device)
        z = e(x)
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
        model = create_model(opt)
        exp = comet_ml.Experiment(project_name="continual-translation")
        exp.add_tag(Path(opt.dataroot).name)
        exp.add_tag(opt.model)
        model.exp = exp
        if "task_schedule" in opt:
            exp.add_tag(opt.task_schedule)
        exp.add_tag("functional_test")
        metrics = eval(model, test_dataset, exp, 1234)
