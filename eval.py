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
        data = {
            "translation": {
                "A": {"cycle": [], "idt": [], "real": [], "fake": []},
                "B": {"cycle": [], "idt": [], "real": [], "fake": []},
            }
        }
        for t in model.tasks:
            tmp = {}
            if t.eval_visuals_pred:
                tmp["pred"] = []
            if t.eval_visuals_target:
                tmp["target"] = []
            data[t.key] = {domain: tmp.copy() for domain in "AB"}

        force = set(["identity", "translation"] + model.tasks.keys)
        print()
        losses = {
            k: []
            for k in dir(model)
            if k.startswith("loss_") and isinstance(getattr(model, k), torch.Tensor)
        }
        for i, b in enumerate(dataset):
            print(f"\rEval batch {i}", end="")
            model.set_input(b)
            model.forward(force=force)
            model.backward_G(losses_only=True, force=force)

            for k in dir(model):
                if k.startswith("loss_") and isinstance(
                    getattr(model, k), torch.Tensor
                ):
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(getattr(model, k).detach().cpu().item())

            if continual:

                for t in model.tasks:
                    for domain in "AB":
                        for dtype in data[t.key][domain]:
                            value = list(
                                model.get(f"{domain}_{t.key}_{dtype}").cpu().numpy()
                            )
                            if t.log_type == "acc":
                                value = [np.argmax(v) for v in value]
                            else:
                                if (
                                    len(data[t.key][domain][dtype])
                                    >= nb_ims // model.opt.batch_size
                                ):
                                    value = []
                            data[t.key][domain][dtype] += value

            # -------------------------
            # -----  Translation  -----
            # -------------------------
            if len(data["translation"]["A"]["real"]) < nb_ims // model.opt.batch_size:
                data["translation"]["A"]["real"].append(model.A_real.cpu().numpy())
                data["translation"]["A"]["cycle"].append(
                    model.A_rec.detach().cpu().numpy()
                )
                data["translation"]["A"]["idt"].append(
                    model.B_idt.detach().cpu().numpy()
                )
                data["translation"]["A"]["fake"].append(
                    model.B_fake.detach().cpu().numpy()
                )

                data["translation"]["B"]["real"].append(model.B_real.cpu().numpy())
                data["translation"]["B"]["cycle"].append(
                    model.B_rec.detach().cpu().numpy()
                )
                data["translation"]["B"]["idt"].append(
                    model.A_idt.detach().cpu().numpy()
                )
                data["translation"]["B"]["fake"].append(
                    model.A_fake.detach().cpu().numpy()
                )

    print()

    ims = {"A": [], "B": []}
    try:
        for i in range(0, len(data["translation"]["A"]["real"]), 5):
            k = i + 5
            # ---------------
            # -----  A  -----
            # ---------------

            # Translation
            im_A_real = np.concatenate(data["translation"]["A"]["real"][i:k], axis=-1)
            im_B_real = np.concatenate(data["translation"]["B"]["real"][i:k], axis=-1)
            im_A_cycle = (
                np.concatenate(data["translation"]["A"]["cycle"][i:k], axis=-1)
                if len(data["translation"]["A"]["cycle"]) > i
                else None
            )
            im_B_cycle = (
                np.concatenate(data["translation"]["B"]["cycle"][i:k], axis=-1)
                if len(data["translation"]["B"]["cycle"]) > i
                else None
            )
            im_A_idt = (
                np.concatenate(data["translation"]["A"]["idt"][i:k], axis=-1)
                if len(data["translation"]["A"]["idt"]) > i
                else None
            )
            im_B_idt = (
                np.concatenate(data["translation"]["B"]["idt"][i:k], axis=-1)
                if len(data["translation"]["B"]["idt"]) > i
                else None
            )
            im_A_fake = (
                np.concatenate(data["translation"]["A"]["fake"][i:k], axis=-1)
                if len(data["translation"]["A"]["fake"]) > i
                else None
            )
            im_B_fake = (
                np.concatenate(data["translation"]["B"]["fake"][i:k], axis=-1)
                if len(data["translation"]["B"]["fake"]) > i
                else None
            )
            ims["A"] = np.concatenate(
                list(
                    filter(None.__ne__, [im_A_real, im_A_fake, im_A_cycle, im_A_idt],)
                ),
                axis=-2,
            )
            ims["B"] = np.concatenate(
                list(
                    filter(None.__ne__, [im_B_real, im_B_fake, im_B_cycle, im_B_idt],)
                ),
                axis=-2,
            )
            for task_key, task_dic in data.items():
                for domain, domain_dic in task_dic.items():
                    for dtype, dlist in domain_dic.items():
                        values = dlist[i:k]
                        if task_key == "depth":
                            values = [
                                np.repeat(to_min1_1(_im), 3, axis=0) for _im in values
                            ]
                        ims_t = np.concatenate(values, axis=-1)
                        ims[domain] = np.concatenate([ims[domain], ims_t], axis=-2)

    except ValueError as e:
        print(e)
        import pdb

        pdb.set_trace()

    # ------------------------
    # -----  Comet Logs  -----
    # ------------------------
    for i in range(len(ims["A"])):
        exp.log_image(
            (np.transpose(ims["A"][i], (1, 2, 0)) + 1) / 2,
            "test_A_{}_{}_{}_rfcidg".format(total_iters, i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
        exp.log_image(
            (np.transpose(ims["B"][i], (1, 2, 0)) + 1) / 2,
            "test_B_{}_{}_{}_rfcidg".format(total_iters, i * 5, (i + 1) * 5 - 1),
            step=total_iters,
        )
    if continual:

        test_losses = {
            "test_" + ln: np.mean(losses[ln])
            for ln in t.loss_names
            for t in model.tasks
        }

        test_accs = {
            f"test_{domain}_{t.key}_acc": np.mean(
                [data[t.key][domain]["pred"] == data[t.key][domain]["target"]]
            )
            for domain in "AB"
            for t in model.tasks
            if t.log_type == "acc"
        }

        exp.log_metrics(test_losses, step=total_iters)
        exp.log_metrics(test_accs, step=total_iters)

        for t in model.tasks:
            for domain in "AB":
                exp.log_confusion_matrix(
                    get_one_hot(np.array(data[t.key][domain]["target"]), t.output_dim),
                    get_one_hot(
                        np.array(data[t.key][domain]["prediction"]), t.output_dim
                    ),
                    file_name=f"confusion_{domain}_{t.key}_{total_iters}.json",
                    title=f"confusion_{domain}_{t.key}_{total_iters}.json",
                )

    print("----------- End Evaluation----------")
    metrics = test_losses.copy()
    metrics.update(test_accs)
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
