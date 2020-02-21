"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from time import time
import torchvision.transforms.functional as TF


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


class _Resize:
    def __init__(self, osize, method):
        super().__init__()
        self.transform = transforms.Resize(osize, method)

    def __call__(self, dic):
        dic.update({k: self.transform(v) for k, v in dic.items() if "angle" not in k})
        return dic


class _ScaleWidth:
    def __init__(self, load_size, method):
        super().__init__()
        self.transform = __scale_width
        self.load_size = load_size
        self.method = method

    def __call__(self, dic):
        dic.update(
            {
                k: self.transform(v, self.load_size, self.method)
                for k, v in dic.items()
                if "angle" not in k
            }
        )
        return dic


class RandomCrop:
    def __init__(self, size):

        self.h = self.w = size
        self.h = int(self.h)
        self.w = int(self.w)

    def __call__(self, dic):
        h, w = dic[[k for k in dic.keys() if "angle" not in k][0]].shape[-2:]
        top = np.random.randint(0, h - self.h)
        left = np.random.randint(0, w - self.w)
        dic.update(
            {
                k: TF.crop(v, top, left, self.h, self.w)
                for k, v in dic.items()
                if "angle" not in k
            }
        )
        return dic


class _Crop:
    def __init__(self, crop_pos, crop_size):
        super().__init__()
        self.transform = __crop
        self.corp_pos = crop_pos
        self.crop_size = crop_size

    def __call__(self, dic):
        dic.update(
            {
                k: self.transform(v, self.corp_pos, self.crop_size)
                for k, v in dic.items()
                if "angle" not in k
            }
        )
        return dic


class _MakePower2:
    def __init__(self, base, method):
        super().__init__()
        self.transform = __make_power_2
        self.base = base
        self.method = method

    def __call__(self, dic):
        dic.update(
            {
                k: self.transform(v, self.base, self.method)
                for k, v in dic.items()
                if "angle" not in k
            }
        )
        return dic


class _RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.flip = TF.hflip
        self.p = p

    def __call__(self, dic):
        if np.random.rand() > self.p:
            return dic
        dic.update({k: self.flip(v) for k, v in dic.items() if "angle" not in k})
        return dic


class _Flip:
    def __init__(self):
        super().__init__()
        self.transform = _RandomHorizontalFlip(0)

    def __call__(self, dic):
        dic.update({k: self.transform(v) for k, v in dic.items() if "angle" not in k})
        return dic


class _Depth:
    def __init__(self):
        super().__init__()
        self.transform = __depth

    def __call__(self, dic):
        dic.update({k: self.transform(v) for k, v in dic.items() if "angle" not in k})
        return dic


class _Rotate:
    def __init__(self):
        super().__init__()

    def __call__(self, dic):
        angle = np.random.choice([0, 90, 180, 270])
        dic.update({k: v.rotate(angle) for k, v in dic.items() if "angle" not in k})
        dic["angle"] = angle
        return dic


class _Normalize:#TODO
    def __init__(self):
        super().__init__()

    def __call__(self, dic):
        pass


def get_dic_transform(
    opt,
    params=None,
    grayscale=False,
    method=Image.BICUBIC,
    convert=True,
    depth=False,
    rotation=False,
):
    pass


def get_transform(
    opt,
    params=None,
    grayscale=False,
    method=Image.BICUBIC,
    convert=True,
    depth=False,
    rotation=False,
):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method))
        )

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __crop(img, params["crop_pos"], opt.crop_size)
                )
            )

    if opt.preprocess == "none":
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if not opt.no_flip and not rotation:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

    if rotation:
        transform_list.append(transforms.Lambda(lambda img: __rotate(img)))

    if depth:
        if rotation:
            transform_list.append(
                transforms.Lambda(
                    lambda img_angle: (__depth(img_angle[0]), img_angle[1])
                )
            )
        else:
            transform_list.append(transforms.Lambda(lambda img: __depth(img)))
    if convert:
        if rotation:
            transform_list += [
                transforms.Lambda(
                    lambda img_angle: (
                        transforms.ToTensor()(img_angle[0]),
                        img_angle[1],
                    )
                )
            ]
        else:
            transform_list += [transforms.ToTensor()]

        if grayscale:
            if rotation:
                transform_list += [
                    transforms.Lambda(
                        lambda img_angle: (
                            transforms.Normalize((0.5,), (0.5,))(img_angle[0]),
                            img_angle[1],
                        )
                    )
                ]
            else:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
        elif depth:
            if rotation:
                transform_list += [
                    transforms.Lambda(
                        lambda img_angle: (
                            transforms.Normalize((2,), (2,))(img_angle[0]),
                            img_angle[1],
                        )
                    )
                ]
            else:
                transform_list += [transforms.Normalize((2,), (2,))]
        else:
            if rotation:
                transform_list += [
                    transforms.Lambda(
                        lambda img_angle: (
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                                img_angle[0]
                            ),
                            img_angle[1],
                        )
                    )
                ]
            else:
                transform_list += [
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __depth(img):
    img = np.array(img)
    if img.max() > 1.0:
        img = img / 255.0
    img = 1 / (img + 1e-6)
    return np.log(img)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True


def __rotate(img):
    # otherwise deterministic for some reason
    np.random.seed(int(str(time()).split(".")[-1]))
    angle = np.random.choice([0, 90, 180, 270])
    img = img.rotate(angle)
    return img, angle
