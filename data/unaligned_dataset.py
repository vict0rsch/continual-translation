import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
from pathlib import Path


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.load_depth = self.opt.netG == "continual"
        self.load_rotation = self.opt.netG == "continual"

        self.dir_A = os.path.join(
            opt.dataroot, opt.phase + "A"
        )  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(
            opt.dataroot, opt.phase + "B"
        )  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.max_dataset_size)
        )  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        if self.load_depth:
            self.transform_dA = get_transform(
                self.opt, grayscale=(input_nc == 1), depth=True
            )
            self.transform_dB = get_transform(
                self.opt, grayscale=(input_nc == 1), depth=True
            )
        if self.load_rotation:
            self.transform_rA = get_transform(
                self.opt, grayscale=(input_nc == 1), rotation=True
            )
            self.transform_rB = get_transform(
                self.opt, grayscale=(input_nc == 1), rotation=True
            )

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[
            index % self.A_size
        ]  # make sure index is within then range
        A_img = Image.open(A_path).convert("RGB")
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert("RGB")

        B = self.transform_B(B_img)
        A = self.transform_A(A_img)

        imgs = {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
        }

        if self.load_depth:
            A_d_img = Image.open(
                Path(A_path).parent / "depths" / (Path(A_path).stem + ".png")
            ).convert("L")
            B_d_img = Image.open(
                Path(B_path).parent / "depths" / (Path(B_path).stem + ".png")
            ).convert("L")
            dA = self.transform_dA(A_d_img)
            dB = self.transform_dB(B_d_img)
            imgs.update({"dA": dA, "dB": dB})
        if self.load_rotation:
            rA, angleA = self.transform_rA(A_img)
            rB, angleB = self.transform_rB(B_img)
            imgs.update({"rA": rA, "rB": rB, "angleA": angleA, "angleB": angleB})
        # apply image transformation

        return imgs

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
