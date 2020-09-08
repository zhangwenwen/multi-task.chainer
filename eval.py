import argparse

import chainer
from chainer import serializers

from chainercv.datasets import voc_bbox_label_names, coco_bbox_label_names
from chainercv.datasets import VOCBboxDataset, VOCSemanticSegmentationDataset

from multi_task.multi_task_300 import Multi_task_300
from multi_task.multi_task_512 import Multi_task_512
from multi_task.evaluator.multi_task_evaluator import MultitaskEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('multi_task_300', 'multi_task_512'), default='multi_task_300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--detection', action='store_true', default=False)
    parser.add_argument('--segmentation', action='store_true', default=False)
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--dataset', default='voc', type=str)
    parser.add_argument('--eval_seg', default=False, action='store_true')
    parser.add_argument('--eval_det', default=False, action='store_true')
    parser.add_argument('--batchsize', type=int, default=32)

    args = parser.parse_args()
    print(args)
    if not (args.segmentation or args.detection):
        raise RuntimeError

    if not args.model_path:
        raise RuntimeError

    if args.model == 'multi_task_300':
        model = Multi_task_300(n_fg_class=len(voc_bbox_label_names), pretrained_model='imagenet',
                               detection=args.detection, segmentation=args.segmentation, attention=args.attention)
    elif args.model == 'multi_task_512':
        model = Multi_task_512(n_fg_class=len(voc_bbox_label_names), pretrained_model='imagenet',
                               detection=args.detection, segmentation=args.segmentation, attention=args.attention)

    model.use_preset('evaluate')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.dataset == 'voc':
        use_07 = True
        label_names = voc_bbox_label_names
    elif args.dataset == 'coco':
        label_names = coco_bbox_label_names

    if args.model_path:
        serializers.load_npz(args.model_path, model)

    if args.detection and args.eval_det:
        test = VOCBboxDataset(
            year='2007', split='test',
            use_difficult=True, return_difficult=True)

        test_iter = chainer.iterators.SerialIterator(
            test, args.batchsize, repeat=False, shuffle=False)
        det_evaluator = MultitaskEvaluator(test_iter, model, use_07_metric=use_07, label_names=label_names,
                                           detection=True)
        result = det_evaluator()
        print('detection result')
        print(result)

    if args.segmentation and args.eval_seg:
        test_mask = VOCSemanticSegmentationDataset(split='val')
        test_mask_iter = chainer.iterators.SerialIterator(test_mask, args.batchsize, repeat=False, shuffle=False)
        seg_evaluator = MultitaskEvaluator(test_mask_iter, model, use_07_metric=use_07, label_names=label_names,
                                           detection=False)
        result_mask = seg_evaluator()
        print('segmentation result')
        print(result_mask)




if __name__ == '__main__':
    main()
