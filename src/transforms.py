import torch
import numbers
import random
from torchvision.transforms import functional as F
import numpy as np


# TODO:
# - update docstrings


class ArrayCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): CxHxW array

        Returns:
              np.ndarray: cropped array
        """
        _, h, w = img.shape
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[:, i:i+th, j:j+tw]


class ArrayRandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if padding != 0 or pad_if_needed:
            # TODO
            raise NotImplementedError()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = img.shape
        th, tw = output_size
        assert h >= th and w >= tw
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        # if self.padding > 0:
        #     img = F.pad(img, self.padding)

        # pad the width if needed
        # _, h, w = img.shape
        # if self.pad_if_needed and w < self.size[1]:
        #     img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # # pad the height if needed
        # if self.pad_if_needed and h < self.size[0]:
        #     img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return img[:, i:i+h, j:j+w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)



class ArrayToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).float().div(255.)


class ArrayRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return np.flip(img, 2).copy()  # expecting C, H, W
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# TODO
# class RandomVerticalFlip(object):
#     """Vertically flip the given PIL Image randomly with a given probability.
#
#     Args:
#         p (float): probability of the image being flipped. Default value is 0.5
#     """
#
#     def __init__(self, p=0.5):
#         raise NotImplementedError()
#         self.p = p
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): Image to be flipped.
#
#         Returns:
#             PIL Image: Randomly flipped image.
#         """
#         if random.random() < self.p:
#             return F.vflip(img)
#         return img
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(p={})'.format(self.p)


