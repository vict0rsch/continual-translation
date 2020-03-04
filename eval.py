from comet_ml import Experiment
import numpy as np
import torch
from models.continual_model import ContinualModel
from data.unaligned_dataset import UnalignedDataset


def eval(
    model: ContinualModel,
    dataset: UnalignedDataset,
    exp: Experiment,
    total_iters: int = 0,
    nb_ims: int = 30,
):
    continual = model.opt.model == "continual"
    print(f"----------- Evaluation {total_iters} ----------")
    with torch.no_grad():
        angles = {
            "A": {"target": [], "prediction": []},
            "B": {"target": [], "prediction": []},
        }
        depth_losses = {"A": [], "B": []}
        gray_losses = {"A": [], "B": []}
        depth_images = {
            "A": {"target": [], "prediction": []},
            "B": {"target": [], "prediction": []},
        }
        test_images = {
            "A": {"cycle": [], "idt": [], "real": [], "fake": []},
            "B": {"cycle": [], "idt": [], "real": [], "fake": []},
        }
        gray_images = {"A": [], "B": []}
        ignore = set()
        force = {"rotation", "depth", "identity", "translation", "gray"}
        print()
        losses = {
            k: []
            for k in dir(model)
            if k.startswith("loss_") and isinstance(getattr(model, k), torch.Tensor)
        }
        for i, b in enumerate(dataset):
            print(f"\rEval batch {i}", end="")
            model.set_input(b)
            model.forward(ignore, force)
            model.backward_G(losses_only=True, ignore=ignore, force=force)

            for k in dir(model):
                if k.startswith("loss_") and isinstance(
                    getattr(model, k), torch.Tensor
                ):
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(getattr(model, k).detach().cpu().item())

            if continual:

                # ----------------------
                # -----  Rotation  -----
                # ----------------------
                angles["A"]["target"] += list(model.angle_A.cpu().numpy())
                angles["A"]["prediction"] += [
                    np.argmax(t) for t in model.angle_A_pred.detach().cpu().numpy()
                ]
                angles["B"]["target"] += list(model.angle_B.cpu().numpy())
                angles["B"]["prediction"] += [
                    np.argmax(t) for t in model.angle_B_pred.detach().cpu().numpy()
                ]
                # -------------------
                # -----  Depth  -----
                # -------------------
                target_A = model.depth_A.cpu().numpy()
                pred_A = model.depth_A_pred.detach().cpu().numpy()
                target_B = model.depth_B.cpu().numpy()
                pred_B = model.depth_B_pred.detach().cpu().numpy()
                if len(depth_images["A"]["target"]) < nb_ims // model.opt.batch_size:
                    depth_images["A"]["target"].append(target_A)
                    depth_images["A"]["prediction"].append(pred_A)
                    depth_images["B"]["target"].append(target_B)
                    depth_images["B"]["prediction"].append(pred_B)

                depth_losses["A"].append(np.square(target_A - pred_A).mean())
                depth_losses["B"].append(np.square(target_B - pred_B).mean())

                gray_losses["A"].append(model.loss_G_gA.item())
                gray_losses["B"].append(model.loss_G_gB.item())

                # ------------------
                # -----  Gray  -----
                # ------------------
                if len(gray_images["A"]) < nb_ims // model.opt.batch_size:
                    gray_images["A"].append(model.fake_gA.detach().cpu().numpy())
                    gray_images["B"].append(model.fake_gB.detach().cpu().numpy())

            # -------------------------
            # -----  Translation  -----
            # -------------------------
            if len(test_images["A"]["real"]) < nb_ims // model.opt.batch_size:
                test_images["A"]["real"].append(model.A_real.cpu().numpy())
                test_images["A"]["cycle"].append(model.A_rec.detach().cpu().numpy())
                test_images["A"]["idt"].append(model.B_idt.detach().cpu().numpy())
                test_images["A"]["fake"].append(model.B_fake.detach().cpu().numpy())

                test_images["B"]["real"].append(model.B_real.cpu().numpy())
                test_images["B"]["cycle"].append(model.B_rec.detach().cpu().numpy())
                test_images["B"]["idt"].append(model.A_idt.detach().cpu().numpy())
                test_images["B"]["fake"].append(model.A_fake.detach().cpu().numpy())
            else:
                ignore.add("translation")
                if "translation" in force:
                    force.remove("translation")
    print()
    # --------------------------------
    # -----  Create image tiles  -----
    # --------------------------------

    # flatten lists
    for domain, dic in test_images.items():
        for im_type, im_list in dic.items():
            tmp = []
            for im in im_list:
                tmp += list(im)
            test_images[domain][im_type] = tmp

    if len(depth_images["A"]["target"]) > 0 and continual:
        for domain, dic_d in depth_images.items():
            for im_type, im_list in dic_d.items():
                tmp = []
                for im in im_list:
                    tmp += list(im)
                depth_images[domain][im_type] = tmp

    if len(gray_images["A"]) > 0 and continual:
        for domain, im_list in gray_images.items():
            tmp = []
            for im in im_list:
                tmp += list(im)
            gray_images[domain] = tmp

    ims_A = []
    ims_B = []
    try:
        for i in range(0, len(test_images["A"]["real"]), 5):
            k = i + 5
            # ---------------
            # -----  A  -----
            # ---------------

            # Translation
            im_A_real = np.concatenate(test_images["A"]["real"][i:k], axis=-1)
            im_A_cycle = (
                np.concatenate(test_images["A"]["cycle"][i:k], axis=-1)
                if len(test_images["A"]["cycle"]) > i
                else None
            )
            im_A_idt = (
                np.concatenate(test_images["A"]["idt"][i:k], axis=-1)
                if len(test_images["A"]["idt"]) > i
                else None
            )
            im_A_fake = (
                np.concatenate(test_images["A"]["fake"][i:k], axis=-1)
                if len(test_images["A"]["fake"]) > i
                else None
            )
            im_A = np.concatenate(
                list(
                    filter(None.__ne__, [im_A_real, im_A_fake, im_A_cycle, im_A_idt],)
                ),
                axis=-2,
            )
            # Depth
            if len(depth_images["A"]["target"]) > i and continual:
                depth_images["A"]["target"][i:k] = [
                    to_min1_1(_im) for _im in depth_images["A"]["target"][i:k]
                ]
                depth_images["A"]["prediction"][i:k] = [
                    to_min1_1(_im) for _im in depth_images["A"]["prediction"][i:k]
                ]
                im_d_A_target = np.concatenate(
                    depth_images["A"]["target"][i:k], axis=-1
                )
                im_d_A_pred = np.concatenate(
                    depth_images["A"]["prediction"][i:k], axis=-1
                )
                ims_d_A = np.repeat(
                    np.concatenate([im_d_A_target, im_d_A_pred], axis=-2), 3, axis=0
                )
                im_A = np.concatenate([im_A, ims_d_A], axis=-2)
            # Gray
            if len(gray_images["A"]) > i and continual:
                im_g_A = np.concatenate(gray_images["A"][i:k], axis=-1)
                im_A = np.concatenate([im_A, im_g_A], axis=-2)

            ims_A.append(im_A)

            # ---------------
            # -----  B  -----
            # ---------------

            # Translation
            im_B_real = np.concatenate(test_images["B"]["real"][i:k], axis=-1)
            im_B_cycle = (
                np.concatenate(test_images["B"]["cycle"][i:k], axis=-1)
                if len(test_images["B"]["cycle"]) > i
                else None
            )
            im_B_idt = (
                np.concatenate(test_images["B"]["idt"][i:k], axis=-1)
                if len(test_images["B"]["idt"]) > i
                else None
            )
            im_B_fake = (
                np.concatenate(test_images["B"]["fake"][i:k], axis=-1)
                if len(test_images["B"]["fake"]) > i
                else None
            )
            im_B = np.concatenate(
                list(
                    filter(None.__ne__, [im_B_real, im_B_fake, im_B_cycle, im_B_idt],)
                ),
                axis=-2,
            )
            # Depth
            if len(depth_images["B"]["target"]) > i and continual:
                depth_images["B"]["target"][i:k] = [
                    to_min1_1(_im) for _im in depth_images["B"]["target"][i:k]
                ]
                depth_images["B"]["prediction"][i:k] = [
                    to_min1_1(_im) for _im in depth_images["B"]["prediction"][i:k]
                ]
                im_d_B_target = np.concatenate(
                    depth_images["B"]["target"][i:k], axis=-1
                )
                im_d_B_pred = np.concatenate(
                    depth_images["B"]["prediction"][i:k], axis=-1
                )
                ims_d_B = np.repeat(
                    np.concatenate([im_d_B_target, im_d_B_pred], axis=-2), 3, axis=0
                )
                im_B = np.concatenate([im_B, ims_d_B], axis=-2)
            # Gray
            if len(gray_images["B"]) > i and continual:
                im_g_B = np.concatenate(gray_images["B"][i:k], axis=-1)
                im_B = np.concatenate([im_B, im_g_B], axis=-2)

            ims_B.append(im_B)
    except ValueError as e:
        print(e)
        import pdb

        pdb.set_trace()

    # ------------------------
    # -----  Comet Logs  -----
    # ------------------------
    for i in range(len(ims_A)):
        exp.log_image(
            (np.transpose(ims_A[i], (1, 2, 0)) + 1) / 2,
            "test_A_{}_{}_{}_rfcidg".format(total_iters, i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
        exp.log_image(
            (np.transpose(ims_B[i], (1, 2, 0)) + 1) / 2,
            "test_B_{}_{}_{}_rfcidg".format(total_iters, i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
    if continual:
        test_A_loss_d = np.mean(depth_losses["A"])
        test_B_loss_d = np.mean(depth_losses["B"])
        test_A_loss_g = np.mean(gray_losses["A"])
        test_B_loss_g = np.mean(gray_losses["B"])
        test_A_rot_acc = np.mean(
            [p == t for p, t in zip(angles["A"]["prediction"], angles["A"]["target"])]
        )
        test_B_rot_acc = np.mean(
            [p == t for p, t in zip(angles["B"]["prediction"], angles["B"]["target"])]
        )
        exp.log_metric("test_A_loss_d", test_A_loss_d, step=total_iters)
        exp.log_metric("test_B_loss_d", test_B_loss_d, step=total_iters)
        exp.log_metric("test_A_loss_g", test_A_loss_g, step=total_iters)
        exp.log_metric("test_B_loss_g", test_B_loss_g, step=total_iters)
        exp.log_metric(
            "test_A_rot_acc", test_A_rot_acc, step=total_iters,
        )
        exp.log_metric(
            "test_B_rot_acc", test_B_rot_acc, step=total_iters,
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
    metrics = {
        "test_A_loss_d": test_A_loss_d,
        "test_B_loss_d": test_B_loss_d,
        "test_A_loss_g": test_A_loss_g,
        "test_B_loss_g": test_B_loss_g,
        "test_A_rot_acc": test_A_rot_acc,
        "test_B_rot_acc": test_B_rot_acc,
    }
    test_losses = {}
    for k, v in losses.items():
        test_losses["test_" + k] = v
    metrics.update({k: np.mean(v) for k, v in test_losses.items()})
    return metrics


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def to_min1_1(im):
    im -= im.min()
    im /= im.max()
    im -= 0.5
    im *= 2
    return im
