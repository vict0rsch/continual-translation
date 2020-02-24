import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.util import angles_to_tensors
import torch.nn as nn


class ContinualModel(BaseModel):
    """
    This class implements the ContinualModel model, for learning image-to-image
    translation without paired data, using continual surrogate tasks.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this
                flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B,
        and lambda_I for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional):
        lambda_I * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)
        (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--lambda_A",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_B",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_R", type=float, default=1.0, help="weight for rotation",
            )
            parser.add_argument(
                "--lambda_D", type=float, default=1.0, help="weight for depth",
            )
            parser.add_argument(
                "--lambda_I",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_I other than 0 has an\
                    effect of scaling the weight of the identity mapping loss. For \
                    example, if the weight of the identity loss should be 10 times \
                    smaller than the weight of the reconstruction loss, please set \
                    lambda_I = 0.1",
            )
            parser.add_argument(
                "--task_schedule", type=str, default="parallel", help="Tasks schedule",
            )
            # sequential       :  <rotation> then <depth> then <translation>
            #                     without the possibility to come back
            #
            # parallel         :  always <depth, rotation, translation>
            #
            # additional       :  <rotation> then <depth, rotation> then
            #                     <depth, rotation, translation>
            #
            # continual        :  additional with mitigation
            #
            # representational :  <rotation, depth, identity> then <translation>
            parser.add_argument(
                "--r_loss_threshold",
                type=float,
                default=0.2,
                help="minimal rotation classification loss to switch task",
            )
            parser.add_argument(
                "--d_loss_threshold",
                type=float,
                default=0.5,
                help="minimal depth estimation loss to switch task",
            )
            parser.add_argument(
                "--i_loss_threshold",
                type=float,
                default=0.5,
                help="minimal identity loss to switch task (representational only)",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be
                a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A",
            "G_A",
            "cycle_A",
            "idt_A",
            "D_B",
            "G_B",
            "cycle_B",
            "idt_B",
            "G_A_r",
            "G_A_d",
            "G_B_r",
            "G_B_d",
        ]
        # specify the images you want to save/display. The training/test scripts
        # will call <BaseModel.get_current_visuals>
        # visual_names_A = ["real_A", "fake_B", "rec_A"]
        # visual_names_B = ["real_B", "fake_A", "rec_B"]
        # if self.isTrain and self.opt.lambda_I > 0.0:
        #     # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append("idt_B")
        #     visual_names_B.append("idt_A")

        # combine visualizations for A and B
        self.visual_names = set(["real_A", "real_B"])
        # specify the models you want to save to the disk. The training/test scripts
        # will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netG_B = networks.define_G(
            opt.output_nc,
            opt.input_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
            self.netD_B = networks.define_D(
                opt.input_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )

        if self.isTrain:
            if opt.lambda_I > 0.0:
                # only works when input and output images have the same number of
                # channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created
            # by function <BaseModel.setup>.

            self.rotationCriterion = torch.nn.CrossEntropyLoss()
            self.depthCriterion = torch.nn.MSELoss()

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.init_schedule()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.depth_A = input["dA" if AtoB else "dB"].to(self.device).float()
        self.rot_A = input["rA" if AtoB else "rB"].to(self.device)
        self.angle_A = input["angleA" if AtoB else "angleB"].to(self.device)
        self.image_paths_A = input["A_paths" if AtoB else "B_paths"]

        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.depth_B = input["dB" if AtoB else "dA"].to(self.device).float()
        self.rot_B = input["rB" if AtoB else "rA"].to(self.device)
        self.angle_B = input["angleB" if AtoB else "angleA"].to(self.device)
        self.image_paths_B = input["B_paths" if AtoB else "A_paths"]

    def forward(self, ignore=set(), force=set()):
        """Run forward pass; called by both functions
        <optimize_parameters> and <test>."""

        assert not (ignore & force)
        should = {
            "rotation": False,
            "depth": False,
            "identity": False,
            "translation": False,
        }
        for k in should:
            if k in force:
                should[k] = True
            elif k in ignore:
                should[k] = False
            else:
                should[k] = self.should_compute(k)

        # -------------------------------
        # -----  DataParallel Mode  -----
        # -------------------------------
        if isinstance(self.netG_A, nn.DataParallel):
            # --------------------
            # -----  Encode  -----
            # --------------------
            self.z_A = self.netG_A.module.encoder(self.real_A)
            self.z_B = self.netG_B.module.encoder(self.real_B)

            if should["rotation"]:
                self.angle_A_pred = self.netG_A.module.rotation(self.z_A)
                self.angle_B_pred = self.netG_B.module.rotation(self.z_B)

            if should["depth"]:
                self.depth_A_pred = self.netG_A.module.depth(self.z_A)
                self.depth_B_pred = self.netG_B.module.depth(self.z_B)

            if should["identity"]:
                self.idt_A = self.netG_A.module.decoder(self.z_B)
                self.idt_B = self.netG_B.module.decoder(self.z_A)

            if should["translation"]:
                self.fake_B = self.netG_A.module.decoder(self.z_A)  # G_A(A)
                self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
                self.fake_A = self.netG_B.module.decoder(self.z_B)  # G_B(B)
                self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        # ---------------------------
        # -----  Other Devices  -----
        # ---------------------------
        else:
            # --------------------
            # -----  Encode  -----
            # --------------------
            self.z_A = self.netG_A.encoder(self.real_A)
            self.z_B = self.netG_B.encoder(self.real_B)

            if should["rotation"]:
                self.angle_A_pred = self.netG_A.rotation(self.z_A)
                self.angle_B_pred = self.netG_B.rotation(self.z_B)

            if should["depth"]:
                self.depth_A_pred = self.netG_A.depth(self.z_A)
                self.depth_B_pred = self.netG_B.depth(self.z_B)

            if should["identity"]:
                self.idt_A = self.netG_A.decoder(self.z_B)
                self.idt_B = self.netG_B.decoder(self.z_A)

            if should["translation"]:
                self.fake_B = self.netG_A.decoder(self.z_A)  # G_A(A)
                self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
                self.fake_A = self.netG_B.decoder(self.z_B)  # G_B(B)
                self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def should_compute(self, arg):
        if arg == "rotation":
            return self.__should_compute_rotation
        elif arg == "depth":
            return self.__should_compute_depth
        elif arg == "identity":
            return self.__should_compute_identity
        elif arg == "translation":
            return self.__should_compute_translation
        raise ValueError(f"Unknown arg {arg}")

    def update_visuals(self):
        if self.__should_compute_translation:
            self.visual_names.add("fake_A")
            self.visual_names.add("fake_B")
            self.visual_names.add("rec_A")
            self.visual_names.add("rec_B")
        else:
            self.visual_names.remove("fake_A")
            self.visual_names.remove("fake_B")
            self.visual_names.remove("rec_A")
            self.visual_names.remove("rec_B")

        if self.__should_compute_identity:
            self.visual_names.add("idt_B")
            self.visual_names.add("idt_A")
        else:
            self.visual_names.remove("idt_B")
            self.visual_names.remove("idt_A")

        if self.__should_compute_depth:
            self.visual_names.add("depth_B")
            self.visual_names.add("depth_B_pred")
            self.visual_names.add("depth_A")
            self.visual_names.add("depth_A_pred")
        else:
            self.visual_names.remove("depth_B")
            self.visual_names.remove("depth_B_pred")
            self.visual_names.remove("depth_A")
            self.visual_names.remove("depth_A_pred")

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_I
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_D = self.opt.lambda_D
        lambda_R = self.opt.lambda_R

        self.loss_G = 0
        if self.should_compute("depth"):
            # print("depth loss")
            device = self.depth_A_pred.device
            self.loss_G_A_d = self.depthCriterion(
                self.depth_A_pred, self.depth_A.to(device)
            )
            self.loss_G_B_d = self.depthCriterion(
                self.depth_B_pred, self.depth_B.to(device)
            )
            self.loss_G += lambda_D * (self.loss_G_B_d + self.loss_G_A_d)

        if self.should_compute("rotation"):
            # print("rotation loss")
            device = self.angle_A_pred.device
            self.loss_G_A_r = self.rotationCriterion(
                self.angle_A_pred,
                angles_to_tensors(self.angle_A, one_hot=False).to(device),
            )
            self.loss_G_B_r = self.rotationCriterion(
                self.angle_B_pred,
                angles_to_tensors(self.angle_B, one_hot=False).to(device),
            )
            self.loss_G += lambda_R * (self.loss_G_B_r + self.loss_G_A_r)

        if self.should_compute("identity"):
            # print("identity loss")
            # Identity loss
            assert lambda_idt > 0
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )

            self.loss_G += self.loss_idt_A + self.loss_idt_B

        if self.should_compute("translation"):
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
            # combined loss and calculate gradients
            self.loss_G += (
                self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
            )
        self.loss_G.backward()

    def sequential_schedule(self):
        r = self.opt.r_loss_threshold
        if self.__should_compute_rotation:
            # never check again once we've changed task
            if self.loss_G_A_r > r and self.loss_G_B_r > r:
                self.__should_compute_rotation = False
                self.__should_compute_depth = True

        d = self.opt.d_loss_threshold
        if self.__should_compute_depth:
            # never check again once we've changed task
            if self.loss_G_A_d > d and self.loss_G_B_d > d:
                self.__should_compute_depth = False
                self.__should_compute_identity = True
                self.__should_compute_translation = True

    def additional_schedule(self):
        r = self.opt.r_loss_threshold
        if self.loss_G_A_r > r and self.loss_G_B_r > r:
            self.__should_compute_depth = True

        d = self.opt.d_loss_threshold
        if self.loss_G_A_d > d and self.loss_G_B_d > d:
            self.__should_compute_identity = True
            self.__should_compute_translation = True

    def representational_schedule(self):
        r = self.opt.r_loss_threshold
        d = self.opt.d_loss_threshold
        i = self.opt.i_loss_threshold

        if (
            self.loss_G_A_r > r
            and self.loss_G_B_r > r
            and self.loss_G_A_d > d
            and self.loss_G_B_d > d
            and self.loss_idt_A > i
            and self.loss_idt_B > i
        ):
            self.__should_compute_translation = True
            self.__should_compute_identity = True
            self.__should_compute_rotation = False
            self.__should_compute_depth = False
            if not self.repr_is_frozen:
                self.set_requires_grad(
                    [self.netG_A.encoder, self.netG_B.encoder], requires_grad=False
                )
                self.repr_is_frozen = True

    def parallel_schedule(self):
        return

    def init_schedule(self):
        if self.opt.task_schedule == "parallel":
            self.__should_compute_rotation = True
            self.__should_compute_depth = True
            self.__should_compute_identity = True
            self.__should_compute_translation = True
            self.update_task_schedule = self.parallel_schedule

        elif self.opt.task_schedule == "sequential":
            self.__should_compute_rotation = True
            self.__should_compute_depth = False
            self.__should_compute_identity = False
            self.__should_compute_translation = False
            self.update_task_schedule = self.sequential_schedule

        elif self.opt.task_schedule in {"additional", "continual"}:
            self.__should_compute_rotation = True
            self.__should_compute_depth = False
            self.__should_compute_identity = False
            self.__should_compute_translation = False
            self.update_task_schedule = self.additional_schedule

        elif self.opt.task_schedule == "representational":
            self.__should_compute_rotation = True
            self.__should_compute_depth = True
            self.__should_compute_identity = True
            self.__should_compute_translation = False
            self.repr_is_frozen = False
            self.update_task_schedule = self.representational_schedule

        else:
            raise ValueError("Unknown schedule {}".format(self.opt.task_schedule))

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights;
        called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False
        )  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.update_task_schedule()
        self.update_visuals()
