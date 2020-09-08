import chainer

from chainer.functions.loss import softmax_cross_entropy

from chainercv.links.model.ssd import multibox_loss
from chainer import initializers
from chainer import variable


class MultiLoss(chainer.Chain):
    def __init__(self, initial_param1=None, initial_param2=None, alpha=1, gpu=False):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.max_loss_detection = None
        self.max_loss_segmentation = None
        with self.init_scope():
            if initial_param1 is None:
                initial_param1 = 0
            if initial_param2 is None:
                initial_param2 = 0

            param1_initializer = initializers._get_initializer(initial_param1)
            self.param1 = variable.Parameter(param1_initializer)

            param2_initializer = initializers._get_initializer(initial_param2)
            self.param2 = variable.Parameter(param2_initializer)

    def to_gpu(self, device=None):
        super(MultiLoss, self).to_gpu(device)

    def __call__(self, params_detection_loss, params_segmentation_loss):
        mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k = params_detection_loss
        score, mask_gt = params_segmentation_loss
        loc_loss, conf_loss = multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k)
        detection_loss = loc_loss * self.alpha + conf_loss

        segmentation_loss = mask_loss(score, mask_gt)
        loss_pre = detection_loss + segmentation_loss
        if self.param1.array is None:
            self.param1.initialize([1])
        if self.param2.array is None:
            self.param2.initialize([1])

        param1_precision = chainer.functions.exp(-self.param1)
        param2_precision = chainer.functions.exp(-self.param2)
        loss = detection_loss * param1_precision + 2 * self.param1 \
               + segmentation_loss * param2_precision + self.param2

        return loss, (loss, loss_pre, loc_loss, conf_loss, segmentation_loss, self.param1, self.param2)


def mask_loss(score, mask_gt):
    loss = softmax_cross_entropy.softmax_cross_entropy(score, mask_gt)
    return loss


def multi_task_loss(detection_params, segmentation_params):
    return MultiLoss()(detection_params, segmentation_params)
