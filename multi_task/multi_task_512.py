from chainercv.links.model.ssd import SSD
import numpy as np
from chainercv.links.model.ssd.ssd_vgg16 import VGG16Extractor512, Multibox
from multi_task.segmentation.multimask import Multimask
from chainercv import utils
import chainer

from chainercv import transforms
from datasets.transforms import mask_resize_with_nearest

_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))

import chainer.functions as F
from chainer import initializers
import chainer.links as L

from multi_task.common.attention import AttentionModule


class Multi_task_VGG16Extractor512(VGG16Extractor512):
    def __init__(self, detection=False, segmentation=False, attention=False):
        init = {
            'initialW': initializers.LeCunUniform(),
            'initial_bias': initializers.Zero(),
        }
        super(Multi_task_VGG16Extractor512, self).__init__()
        self.detection = detection
        self.segmentation = segmentation
        if segmentation and detection:
            self.attention = attention
        else:
            self.attention = False

        with self.init_scope():
            if self.detection:
                self.deconv_det_1 = L.Deconvolution2D(256, 256, ksize=2, stride=2, pad=0, outsize=(2, 2), **init)  ##

                self.deconv_conv_det_1 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_det_1 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_det_2 = L.Deconvolution2D(256, 256, ksize=3, stride=1, pad=0, outsize=(4, 4), **init)  ##

                self.deconv_conv_det_2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_det_2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_det_3 = L.Deconvolution2D(256, 256, ksize=2, stride=2, pad=0, outsize=(8, 8), **init)  ##

                self.deconv_conv_det_3 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_det_3 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_det_4 = L.Deconvolution2D(256, 512, ksize=1, stride=2, pad=0, outsize=(16, 16), **init)  ##

                self.deconv_conv_det_4 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_det_4 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_det_5 = L.Deconvolution2D(512, 1024, ksize=2, stride=2, pad=0, outsize=(32, 32), **init)

                self.deconv_conv_det_5 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_det_5 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_det_6 = L.Deconvolution2D(1024, 512, ksize=2, stride=2, pad=0, outsize=(64, 64), **init)
                self.deconv_conv_det_6 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)
                self.skip_conv_det_6 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)

                # conv after skip connection

            if self.segmentation:
                self.deconv_seg_1 = L.Deconvolution2D(256, 256, ksize=2, stride=2, pad=0, outsize=(2, 2), **init)  ##

                self.deconv_conv_seg_1 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_seg_1 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_seg_2 = L.Deconvolution2D(256, 256, ksize=3, stride=1, pad=0, outsize=(4, 4), **init)  ##

                self.deconv_conv_seg_2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_seg_2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_seg_3 = L.Deconvolution2D(256, 256, ksize=2, stride=2, pad=0, outsize=(8, 8), **init)  ##

                self.deconv_conv_seg_3 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_seg_3 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_seg_4 = L.Deconvolution2D(256, 512, ksize=1, stride=2, pad=0, outsize=(16, 16), **init)  ##

                self.deconv_conv_seg_4 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_seg_4 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_seg_5 = L.Deconvolution2D(512, 1024, ksize=2, stride=2, pad=0, outsize=(32, 32), **init)

                self.deconv_conv_seg_5 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, **init)  #
                self.skip_conv_seg_5 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, **init)  #

                self.deconv_seg_6 = L.Deconvolution2D(1024, 512, ksize=2, stride=2, pad=0, outsize=(64, 64), **init)
                self.deconv_conv_seg_6 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)
                self.skip_conv_seg_6 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1, **init)

            if self.attention:
                # TODO: implementation of attention
                self.attention_1 = AttentionModule(256)
                self.attention_2 = AttentionModule(256)
                self.attention_3 = AttentionModule(256)
                self.attention_4 = AttentionModule(512)
                self.attention_5 = AttentionModule(1024)

    def __call__(self, x):

        ys_detection = []
        ys_segmentation = []

        stages = super(Multi_task_VGG16Extractor512, self).__call__(x)
        if self.attention:
            # TODO: if attetion is activated, both segmentation and detection are activated
            h_seg = stages[-1]
            h_det = stages[-1]
            ys_detection.append(h_det)
            ys_segmentation.append(h_seg)
            for i in range(6):
                # for segmentation: deconv then conv
                h_seg = self['deconv_seg_{}'.format(i + 1)](h_seg)
                h_seg = F.relu(h_seg)

                h_seg = self['deconv_conv_seg_{}'.format(i + 1)](h_seg)
                h_seg = F.relu(h_seg)

                # for detection : deconv then conv
                h_det = self['deconv_det_{}'.format(i + 1)](h_det)
                h_det = F.relu(h_det)

                h_det = self['deconv_conv_det_{}'.format(i + 1)](h_det)
                h_det = F.relu(h_det)

                h_seg = stages[-(i + 2)] + h_seg
                h_seg = self['skip_conv_seg_{}'.format(i + 1)](h_seg)

                ys_segmentation.append(h_seg)

                h_det = stages[-(i + 2)] + h_det
                h_det = self['skip_conv_det_{}'.format(i + 1)](h_det)

                ys_detection.append(h_det)

                if i + 1 < 6:
                    attention = self['attention_{}'.format(i + 1)](h_seg, h_det)
                    # TODO: attention is passed to both segmentation and detection
                    attention_det = attention * h_det
                    attention_seg = attention * h_seg

                    h_det = attention_det + h_det
                    h_seg = attention_seg + h_seg
                    del attention_det
                    del attention_seg
                    del attention


        else:
            if self.segmentation:
                h_seg = stages[-1]
                ys_segmentation.append(h_seg)
                for i in range(6):
                    h_seg = self['deconv_seg_{}'.format(i + 1)](h_seg)
                    h_seg = F.relu(h_seg)

                    h_seg = self['deconv_conv_seg_{}'.format(i + 1)](h_seg)
                    h_seg = F.relu(h_seg)

                    h_seg = stages[-(i + 2)] + h_seg

                    ys_segmentation.append(h_seg)

            if self.detection:
                h_det = stages[-1]
                ys_detection.append(h_det)

                for i in range(6):
                    h_det = self['deconv_det_{}'.format(i + 1)](h_det)
                    h_det = F.relu(h_det)

                    h_det = self['deconv_conv_det_{}'.format(i + 1)](h_det)
                    h_det = F.relu(h_det)

                    h_det = stages[-(i + 2)] + h_det

                    ys_detection.append(h_det)

        # ys_segmentation.reverse()
        ys_detection.reverse()
        return ys_detection, ys_segmentation


class Multi_task_512(SSD):
    """docstring for Multi_task_300"""
    _models = {
        'voc0712': {
            'param': {'n_fg_class': 20},
            'url': 'https://chainercv-models.preferred.jp/'
                   'ssd300_voc0712_converted_2017_06_06.npz',
            'cv2': True
        },
        'imagenet': {
            'url': 'https://chainercv-models.preferred.jp/'
                   'ssd_vgg16_imagenet_converted_2017_06_09.npz',
            'cv2': True
        },
    }

    def __init__(self, n_fg_class=None, pretrained_model='imagenet', detection=True, segmentation=False,
                 attention=False):
        param, path = utils.prepare_pretrained_model(
            {'n_fg_class': n_fg_class}, pretrained_model, self._models)

        super(Multi_task_512, self).__init__(
            extractor=Multi_task_VGG16Extractor512(detection=detection, segmentation=segmentation, attention=attention),
            multibox=Multibox(
                n_class=param['n_fg_class'] + 1,
                aspect_ratios=(
                    (2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            steps=(8, 16, 32, 64, 128, 256, 512),
            sizes=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6),
            mean=_imagenet_mean
        )

        with self.init_scope():
            self.multimask = Multimask(n_class=param['n_fg_class'] + 1, scale_num=7)
        self.detection = detection
        self.segmentation = segmentation

        self._prediction_detection = False

        if path:
            chainer.serializers.load_npz(path, self, strict=False)

    @property
    def prediction_detection(self):
        return self._prediction_detection

    @prediction_detection.setter
    def prediction_detection(self, value):
        self._prediction_detection = value

    def predict(self, imgs):
        x = []
        sizes = []
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))
        with chainer.using_config('train', False), \
             chainer.function.no_backprop_mode():
            x = chainer.Variable(self.xp.stack(x))
            result_detection, result_segmentation = self(x)

        if self.prediction_detection:
            mb_locs, mb_confs = result_detection
            # TODO: for detection
            mb_locs, mb_confs = mb_locs.array, mb_confs.array

            bboxes = []
            labels = []
            scores = []
            for mb_loc, mb_conf, size in zip(mb_locs, mb_confs, sizes):
                bbox, label, score = self.coder.decode(
                    mb_loc, mb_conf, self.nms_thresh, self.score_thresh)
                bbox = transforms.resize_bbox(
                    bbox, (self.insize, self.insize), size)
                bboxes.append(chainer.backends.cuda.to_cpu(bbox))
                labels.append(chainer.backends.cuda.to_cpu(label))
                scores.append(chainer.backends.cuda.to_cpu(score))

            return bboxes, labels, scores
        else:
            # TODO: for segmentation
            mask = F.argmax(result_segmentation, axis=1)
            num, _, _ = mask.shape
            mask = mask.array
            masks = []

            for i, size in enumerate(sizes):
                mask_ = mask[i, :, :]
                mask_ = mask_resize_with_nearest(mask_, size)
                masks.append(chainer.backends.cuda.to_cpu(mask_))

            return masks

    def demo(self, imgs, detection=True, segmentation=True):
        if self.segmentation:
            segmentation = segmentation
        else:
            segmentation = self.segmentation

        if self.detection:
            detection = detection
        x = []
        sizes = []
        for img in imgs:
            _, H, W = img.shape
            img = self._prepare(img)
            x.append(self.xp.array(img))
            sizes.append((H, W))
        with chainer.using_config('train', False), \
             chainer.function.no_backprop_mode():
            x = chainer.Variable(self.xp.stack(x))
            result_detection, result_segmentation = self(x)

        bboxes = []
        labels = []
        scores = []
        masks = []
        if detection:
            mb_locs, mb_confs = result_detection
            # TODO: for detection
            mb_locs, mb_confs = mb_locs.array, mb_confs.array

            for mb_loc, mb_conf, size in zip(mb_locs, mb_confs, sizes):
                bbox, label, score = self.coder.decode(
                    mb_loc, mb_conf, self.nms_thresh, self.score_thresh)
                bbox = transforms.resize_bbox(
                    bbox, (self.insize, self.insize), size)
                bboxes.append(chainer.backends.cuda.to_cpu(bbox))
                labels.append(chainer.backends.cuda.to_cpu(label))
                scores.append(chainer.backends.cuda.to_cpu(score))

        if segmentation:
            # TODO: for segmentation
            mask = F.argmax(result_segmentation, axis=1)
            num, _, _ = mask.shape
            mask = mask.array

            for i, size in enumerate(sizes):
                mask_ = mask[i, :, :]
                mask_ = mask_resize_with_nearest(mask_, size)
                masks.append(chainer.backends.cuda.to_cpu(mask_))

        return bboxes, labels, scores, masks

    def to_gpu(self, device=None):
        super(Multi_task_512, self).to_gpu(device)
        self.coder.to_gpu(device)
        self.multimask.to_gpu(device)

    def __call__(self, x):
        # TODO: call for generation of multi-scale features and then detection or segmentation or both
        xs_detection, xs_segmentation = self.extractor(x)

        if self.detection:
            result_detection = self.multibox(xs_detection)
        else:
            result_detection = None

        if self.segmentation:
            result_segmentation = self.multimask(xs_segmentation)
        else:
            result_segmentation = None

        if result_detection is None:
            self.multibox(xs_segmentation)
        if result_segmentation is None:
            xs_detection.reverse()
            self.multimask(xs_detection)

        return result_detection, result_segmentation
