import torch
import numbers
import random
# from torchvision.transforms import functional as F
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
        _, h, w = img.shape
        th, tw = output_size
        assert h >= th and w >= tw
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)

        return img[:, i:i+h, j:j+w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)



class ArrayToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).float().div(255.)


class ArrayRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
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


