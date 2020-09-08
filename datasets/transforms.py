import copy
#import cupy as cp

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from chainercv import transforms


def mask_resize_with_nearest(mask, size):
    if mask.size == 0:
        raise RuntimeError
    h, w = size
    if not isinstance(mask,np.ndarray):
        import cupy as cp
        mask = cp.asnumpy(mask)
    mask_ = np.array([mask, mask, mask])
    mask__ = np.transpose(mask_, (1, 2, 0))
    mask_r = None
    try:
        import cv2
        mask_r = cv2.resize(mask__, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(str(e))
    if mask_r is None:
        return np.ones(size) * -1
    return mask_r[:, :, 0]


def show_img(img):
    img_show = np.transpose(img, [1, 2, 0])
    img_show = np.array(img_show, dtype=np.int)
    plt.imshow(img_show)
    plt.show()


def show_mask(mask):
    mask = np.array(mask, dtype=int)
    # mask=mask.transpose([1,2,0])
    mask_p = Image.fromarray(np.uint8(mask), 'P')
    palette = np.load('Extra/palette.npy').tolist()
    mask_p.putpalette(palette)

    plt.imshow(mask_p)
    plt.show()


class Transform(object):
    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def _random_expand_mask(self, mask, param):
        x_offset, y_offset, oh, ow = param['x_offset'], param['y_offset'], param['new_height'], param['new_width']
        mask_ = np.zeros((oh, ow))
        h, w = mask.shape
        mask_[y_offset:y_offset + h, x_offset:x_offset + w] = mask
        return mask_

    def _fixed_crop_mask(self, mask, x_slice, y_slice):
        # out = np.crop(mask, begin=(y0, x0), end=(y0 + h, x0 + w))
        out = mask[x_slice.start:x_slice.stop, y_slice.start:y_slice.stop]

        return out

    def _resize_with_nearest(self, mask, size):
        if mask.size == 0:
            raise RuntimeError
        h, w = size

        mask_ = np.array([mask, mask, mask])
        mask__ = np.transpose(mask_, (1, 2, 0))
        mask_r = None
        try:
            import cv2
            mask_r = cv2.resize(mask__, (h, w), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(str(e))
        if mask_r is None:
            return np.ones(size) * -1
        return mask_r[:, :, 0]

    def _random_flip_mask(self, mask, x_flip=False, y_flip=False):
        if x_flip:
            mask = np.flip(mask, axis=1)
        return mask

    def _show_img(self, img):
        img_show = np.transpose(img, [1, 2, 0])
        img_show = np.array(img_show, dtype=np.int)
        plt.imshow(img_show)
        plt.show()

    def _show_mask(self, mask):
        mask = np.array(mask, dtype=int)
        # mask=mask.transpose([1,2,0])
        mask_p = Image.fromarray(np.uint8(mask), 'P')
        palette = np.load('Extra/palette.npy').tolist()
        mask_p.putpalette(palette)

        plt.imshow(mask_p)
        plt.show()

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        # mask = None
        img, bbox, label, mask = in_data

        # TODO: show information

        # self._show_img(img)
        # self._show_mask(mask)
        # 1. Color augmentation
        img = random_distort(img)

        # self._show_img(img)
        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])
            if mask is not None:
                _, new_height, new_width = img.shape
                param['new_height'] = new_height
                param['new_width'] = new_width
                mask = self._random_expand_mask(mask, param)

        # self._show_img(img)
        # self._show_mask(mask)

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)

        # self._show_img(img)

        mask = self._fixed_crop_mask(mask, param['y_slice'], param['x_slice'])

        # self._show_mask(mask)

        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))

        # self._show_img(img)

        if mask is not None:
            if mask.size == 0:
                raise RuntimeError
            mask = self._resize_with_nearest(mask, (self.size, self.size))

        # self._show_mask(mask)

        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])
        if mask is not None:
            mask = self._random_flip_mask(mask, x_flip=params['x_flip'], y_flip=params['y_flip'])

        # self._show_img(img)
        # self._show_mask(mask)

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)
        if mask is None:
            mask = np.ones([self.size, self.size], dtype=np.int32) * -1
        # print("Dtype is :"+str(mask.dtype))
        data_type = str(mask.dtype)
        target_type = 'int32'
        if data_type != target_type:
            mask = mask.astype(np.int32)
        if img is None:
            raise RuntimeError
        return img, mb_loc, mb_label, mask
