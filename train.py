"""General-purpose training script for image-to-image translation.
This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import comet_ml
import numpy as np
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im, decode_md
from copy import copy


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    test_opt = copy(opt)
    test_opt.phase = "test"
    test_dataset = create_dataset(test_opt)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)
    exp = comet_ml.Experiment(project_name="continual-translation")
    exp.log_parameters(dict(vars(opt)))
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    print("starting")
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # outer loop for different epochs; we save the model by <epoch_count>,
        # <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch
        print("--- epoch starting")

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            if total_iters % opt.display_freq == 0:

                # TODO evaluate on images + rotation accuracy

                # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visuals = model.get_current_visuals()
                for k, v in visuals.items():
                    for _j, imt in enumerate(v):
                        if "depth" in k:
                            imn = np.squeeze(decode_md(imt))
                        else:
                            imn = tensor2im(imt)
                        exp.log_image(imn, f"{k}_{total_iters}_{_j}")
                        if _j > 0:
                            continue

            if total_iters % opt.print_freq == 0:
                # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                exp.log_metrics(losses, step=total_iters)

            if total_iters % opt.save_latest_freq == 0:
                # cache our latest model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)
            if i % 50 == 0:
                print("Iter {} \r".format(i), end="")

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()  # update learning rates at the end of every epoch.