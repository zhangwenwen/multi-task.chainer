import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from multi_task.extensions import loss_split
from chainercv.datasets import voc_bbox_label_names, coco_bbox_label_names
from chainercv.datasets import VOCBboxDataset, VOCSemanticSegmentationDataset
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss

from multi_task.multi_task_300 import Multi_task_300
from multi_task.multi_task_512 import Multi_task_512
from datasets.multi_task_dataset_voc import Multi_task_VOC

from config.datasets import voc_experiments

from datasets.transforms import Transform
import chainer.functions as F

from multi_task.evaluator.multi_task_evaluator import MultitaskEvaluator

import numpy as np

from multi_task.segmentation.loss.losses import MultiLoss

chainer.set_debug(False)


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, gpu=False, alpha=1, k=3, use_multi_task_loss=False, use_dynamic_loss=False,
                 loss_split=0.5):
        super(MultiboxTrainChain, self).__init__()
        self.gpu = gpu

        with self.init_scope():
            if use_multi_task_loss:
                self.multi_loss = MultiLoss(gpu=self.gpu)
            self.model = model
        self.alpha = alpha
        self.k = k
        self.loss_split = loss_split
        self.multi_task_loss = use_multi_task_loss
        self.use_dynamic_loss = use_dynamic_loss

    def _check_mask(self, mask_batch):
        result = []
        batch_size = mask_batch.shape[0]
        for i in range(batch_size):
            mask_batch = chainer.backends.cuda.to_cpu(mask_batch)
            if np.ndarray.mean(mask_batch[0][i]) == -1:
                result.append(0)
            else:
                result.append(1)
        return_result = np.array(result)
        if self.gpu:
            return_result = chainer.backends.cuda.to_gpu(return_result)
        return return_result

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels, gt_mask):
        detection_result, segmentation_result = self.model(imgs)
        # mb_locs, mb_confs = self.model(imgs)
        detection = False
        segmentation = False
        if detection_result is not None:
            detection = True
        if segmentation_result is not None:
            segmentation = True

        if segmentation and detection:
            self.multi_task_loss = self.multi_task_loss
        else:
            self.multi_task_loss = False

        if detection:
            mb_locs, mb_confs = detection_result
        if segmentation:
            score = segmentation_result

        if not self.multi_task_loss:
            if detection:
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
                loss_detection = loc_loss * self.alpha + conf_loss
            if segmentation:
                loss_segmentation = F.softmax_cross_entropy(score, gt_mask)

            if segmentation and detection:
                if self.use_dynamic_loss:
                    pass
                else:
                    loss = 2 * self.loss_split * loss_detection + 2 * (1 - self.loss_split) * loss_segmentation

                chainer.reporter.report(
                    {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss, 'loss/mask': loss_segmentation,
                     'loss/split': self.loss_split},
                    self)
            elif segmentation:
                loss = loss_segmentation
                chainer.reporter.report(
                    {'loss': loss, 'loss/mask': loss_segmentation},
                    self)

            elif detection:
                loss = loss_detection
                chainer.reporter.report(
                    {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
                    self)
        else:
            loss, report_values = self.multi_loss((mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k),
                                                  (score, gt_mask))
            # loss, (loss_pre, loc_loss, conf_loss, segmentation_loss)
            chainer.reporter.report(
                {'loss': report_values[0].data[0], 'loss/pre': report_values[1], 'loss/loc': report_values[2],
                 'loss/conf': report_values[3],
                 'loss/mask': report_values[4], 'param_det': report_values[5].data[0],
                 'param_seg': report_values[6].data[0]},
                self)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('multi_task_300', 'multi_task_512'), default='multi_task_300')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--iteration', type=int, default=120000)
    parser.add_argument('--eval_step', type=int, nargs='*', default=[80000, 100000, 120000])
    parser.add_argument('--lr_step', type=int, nargs='*', default=[80000, 100000])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--snap_step', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')  # in experiments for real experiment
    parser.add_argument('--resume', type=str)
    parser.add_argument('--detection', action='store_true', default=False)
    parser.add_argument('--segmentation', action='store_true', default=False)
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--dataset', default='voc', type=str)
    parser.add_argument('--experiment', type=str, default='final_voc')
    parser.add_argument('--multitask_loss', action='store_true', default=False)
    parser.add_argument('--dynamic_loss', action='store_true', default=False)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--update_split_interval',type=int,default=100)
    parser.add_argument('--loss_split', type=float,
                        default=0.5)  # in fact for detection, other task(segmentation) is 1-loss_split
    args = parser.parse_args()
    snap_step = args.snap_step
    args.snap_step = []
    for step in range(snap_step, args.iteration + 1, snap_step):
        args.snap_step.append(step)

    # redefine the output path
    import os
    import time
    args.out = os.path.join(args.out, args.experiment, time.strftime("%Y%m%d_%H%M%S", time.localtime()))

    if args.model == 'multi_task_300':
        model = Multi_task_300(n_fg_class=len(voc_bbox_label_names), pretrained_model='imagenet',
                               detection=args.detection, segmentation=args.segmentation, attention=args.attention)
    elif args.model == 'multi_task_512':
        model = Multi_task_512(n_fg_class=len(voc_bbox_label_names), pretrained_model='imagenet',
                               detection=args.detection, segmentation=args.segmentation, attention=args.attention)

    model.use_preset('evaluate')
    if not (args.segmentation or args.detection):
        raise RuntimeError

    train_chain = MultiboxTrainChain(model, gpu=args.gpu >= 0, use_multi_task_loss=args.multitask_loss,
                                     loss_split=args.loss_split)
    train_chain.cleargrads()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train = TransformDataset(Multi_task_VOC(voc_experiments[args.experiment][args.experiment + '_train']),
                             Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.MultiprocessIterator(train, batch_size=args.batchsize)

    test = VOCBboxDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)

    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    test_mask = VOCSemanticSegmentationDataset(split='val')
    test_mask_iter = chainer.iterators.SerialIterator(test_mask, args.batchsize, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    # optimizer.add_hook(GradientClipping(0.1))
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), args.out)
    '''if args.resume:
        serializers.load_npz(args.resume, trainer)'''
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=args.lr),
        trigger=triggers.ManualScheduleTrigger(args.lr_step, 'iteration'))

    if args.dataset == 'voc':
        use_07 = True
        label_names = voc_bbox_label_names
    elif args.dataset == 'coco':
        label_names = coco_bbox_label_names
    if args.detection and not args.debug:
        trainer.extend(MultitaskEvaluator(test_iter, model, args.dataset, use_07, label_names=label_names),
                       trigger=triggers.ManualScheduleTrigger(args.eval_step + [args.iteration], 'iteration'))

    if args.segmentation and not args.debug:
        trainer.extend(
            MultitaskEvaluator(test_mask_iter, model, dataset=args.dataset, label_names=label_names, detection=False),
            trigger=triggers.ManualScheduleTrigger(args.eval_step + [args.iteration], 'iteration'))

    log_interval = args.log_interval, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    if args.segmentation and args.detection and args.dynamic_loss:
        trainer.extend(loss_split.LossSplit(
            trigger=(args.update_split_interval, 'iteration')))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/loss/mask', 'main/loss/loc', 'main/loss/conf', 'main/loss/split']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(
        extensions.snapshot(),
        trigger=triggers.ManualScheduleTrigger(
            args.snap_step + [args.iteration], 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=triggers.ManualScheduleTrigger(
            args.snap_step + [args.iteration], 'iteration'))
    if args.resume:
        if 'model' in args.resume:
            serializers.load_npz(args.resume, model)
        else:
            serializers.load_npz(args.resume,trainer)

    print(args)

    trainer.run()


if __name__ == '__main__':
    main()
