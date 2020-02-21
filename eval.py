import torch
import numpy as np
from PIL import Image
from models.continual_model import ContinualModel


def eval(model: ContinualModel, dataset, exp, total_iters):
    model.set_requires_grad(model, requires_grad=False)

    angles = {
        "A": {"target": [], "predictions": []},
        "B": {"target": [], "predictions": []},
    }
    depth_losses = {"A": [], "B": []}
    test_images = {"A": [], "B": []}
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
            pass

    model.set_requires_grad(model, requires_grad=True)
