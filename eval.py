from comet_ml import Experiment
import numpy as np
import torch
from models.continual_model import ContinualModel
from data.unaligned_dataset import UnalignedDataset


def eval(
    model: ContinualModel, dataset: UnalignedDataset, exp: Experiment, total_iters
):
    print(f"----------- Evaluation {total_iters} ----------")
    with torch.no_grad():
        angles = {
            "A": {"target": [], "prediction": []},
            "B": {"target": [], "prediction": []},
        }
        depth_losses = {"A": [], "B": []}
        test_images = {
            "A": {"cycle": [], "idt": [], "real": [], "fake": []},
            "B": {"cycle": [], "idt": [], "real": [], "fake": []},
        }
        ignores = set()
        print()
        for i, b in enumerate(dataset):
            print(f"\rEval batch {i}", end="")
            model.set_input(b)
            model.forward(ignores)
            if model.should_compute("rotation"):
                angles["A"]["target"] += list(model.angle_A.cpu().numpy())
                angles["A"]["prediction"] += [
                    np.argmax(t) for t in model.angle_A_pred.detach().cpu().numpy()
                ]
                angles["B"]["target"] += list(model.angle_B.cpu().numpy())
                angles["B"]["prediction"] += [
                    np.argmax(t) for t in model.angle_B_pred.detach().cpu().numpy()
                ]
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
                if len(test_images["A"]["real"]) < 15 // model.opt.batch_size:
                    test_images["A"]["real"].append(model.real_A.cpu().numpy())
                    test_images["A"]["cycle"].append(model.rec_A.detach().cpu().numpy())
                    test_images["A"]["idt"].append(model.idt_B.detach().cpu().numpy())
                    test_images["A"]["fake"].append(model.fake_B.detach().cpu().numpy())

                    test_images["B"]["real"].append(model.real_B.cpu().numpy())
                    test_images["B"]["cycle"].append(model.rec_B.detach().cpu().numpy())
                    test_images["B"]["idt"].append(model.idt_A.detach().cpu().numpy())
                    test_images["B"]["fake"].append(model.fake_A.detach().cpu().numpy())
                else:
                    ignores.add("translation")
    print()
    # --------------------------------
    # -----  Create image tiles  -----
    # --------------------------------
    ims_A = []
    ims_B = []
    for i in range(len(test_images["A"]["real"])):
        im_A_real = np.concatenate(test_images["A"]["real"][i], axis=-1)
        im_A_cycle = np.concatenate(test_images["A"]["cycle"][i], axis=-1)
        im_A_idt = np.concatenate(test_images["A"]["idt"][i], axis=-1)
        im_A_fake = np.concatenate(test_images["A"]["fake"][i], axis=-1)
        im_A = np.concatenate([im_A_real, im_A_fake, im_A_cycle, im_A_idt], axis=-2)
        ims_A.append(im_A)

        im_B_real = np.concatenate(test_images["B"]["real"][i], axis=-1)
        im_B_cycle = np.concatenate(test_images["B"]["cycle"][i], axis=-1)
        im_B_idt = np.concatenate(test_images["B"]["idt"][i], axis=-1)
        im_B_fake = np.concatenate(test_images["B"]["fake"][i], axis=-1)
        im_B = np.concatenate([im_B_real, im_B_fake, im_B_cycle, im_B_idt], axis=-2)
        ims_B.append(im_B)

    # ------------------------
    # -----  Comet Logs  -----
    # ------------------------
    for i in range(len(test_images["A"]["real"])):
        exp.log_image(
            (np.transpose(ims_A[i], (1, 2, 0)) + 1) / 2,
            "test_A_{}_{}_{}_rfci".format(total_iters, i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
        exp.log_image(
            (np.transpose(ims_B[i], (1, 2, 0)) + 1) / 2,
            "test_B_{}_{}_{}_rfci".format(total_iters, i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )

    exp.log_metric("test_A_loss_d", np.mean(depth_losses["A"]), step=total_iters)
    exp.log_metric("test_B_loss_d", np.mean(depth_losses["B"]), step=total_iters)

    exp.log_metric(
        "test_A_rot_acc",
        np.mean(
            [p == t for p, t in zip(angles["A"]["prediction"], angles["A"]["target"])]
        ),
        step=total_iters,
    )

    exp.log_metric(
        "test_B_rot_acc",
        np.mean(
            [p == t for p, t in zip(angles["B"]["prediction"], angles["B"]["target"])]
        ),
        step=total_iters,
    )

    exp.log_confusion_matrix(
        get_one_hot(np.array(angles["A"]["target"]), 4),
        get_one_hot(np.array(angles["A"]["prediction"]), 4),
        labels=["0", "90", "180", "270"],
        file_name=f"confusion_A_{total_iters}.json",
        title="Confusion Matrix A",
    )
    exp.log_confusion_matrix(
        get_one_hot(np.array(angles["B"]["target"]), 4),
        get_one_hot(np.array(angles["B"]["prediction"]), 4),
        labels=["0", "90", "180", "270"],
        file_name=f"confusion_B_{total_iters}.json",
        title="Confusion Matrix B",
    )

    print("----------- End Evaluation----------")


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])
