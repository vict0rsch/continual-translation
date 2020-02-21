import torch
import numpy as np
from PIL import Image
from models.continual_model import ContinualModel
from data.unaligned_dataset import UnalignedDataset


def eval(model: ContinualModel, dataset: UnalignedDataset, exp, total_iters):
    model.set_requires_grad(model, requires_grad=False)

    angles = {
        "A": {"target": [], "predictions": []},
        "B": {"target": [], "predictions": []},
    }
    depth_losses = {"A": [], "B": []}
    test_images = {
        "A": {"cycle": [], "idt": [], "real": [], "fake": []},
        "B": {"cycle": [], "idt": [], "real": [], "fake": []},
    }
    for i, b in enumerate(dataset):
        model.set_input(b)
        model.forward()
        if model.should_compute("rotation"):
            angles["A"]["target"] += model.angle_A.cpu().numpy()
            angles["A"]["prediction"] += model.angle_A_pred.detach().cpu().numpy()
            angles["B"]["target"] += model.angle_B.cpu().numpy()
            angles["B"]["prediction"] += model.angle_B_pred.detach().cpu().numpy()
        if model.should_compute("depth"):
            depth_losses["A"].append(
                np.square(
                    model.depth_A.cpu().numpy()
                    - model.depth_A_pred.detach().cpu().numpy()
                ).mean()
            )
            depth_losses["B"].append(
                np.square(
                    model.depth_B.cpu().numpy()
                    - model.depth_B_pred.detach().cpu().numpy()
                ).mean()
            )
        if model.should_compute("translation"):
            if len(test_images["A"]["real"]) < 10:
                test_images["A"]["real"].append(model.real_A.cpu().numpy())
                test_images["A"]["cycle"].append(model.rec_A.detach().cpu().numpy())
                test_images["A"]["idt"].append(model.idt_A.detach().cpu().numpy())
                test_images["A"]["fake"].append(model.fake_A.detach().cpu().numpy())

                test_images["B"]["real"].append(model.real_B.cpu().numpy())
                test_images["B"]["cycle"].append(model.rec_B.detach().cpu().numpy())
                test_images["B"]["idt"].append(model.idt_B.detach().cpu().numpy())
                test_images["B"]["fake"].append(model.fake_B.detach().cpu().numpy())

    model.set_requires_grad(model, requires_grad=True)
