import chainer
import copy
import numpy as np

from chainercv.utils import apply_to_iterator
from chainer import reporter

from chainercv.evaluations import eval_detection_voc, eval_semantic_segmentation


class MultitaskEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, dataset='voc', use_07_metric=False, label_names=None, detection=True):
        super(MultitaskEvaluator, self).__init__(iterator, target)
        self.label_names = label_names
        self.dataset = dataset
        self.use_07_metric = use_07_metric
        self.detection = detection
        if detection:
            MultitaskEvaluator.default_name = "evaluation_det"
        else:
            MultitaskEvaluator.default_name = "evaluation_seg"

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        if hasattr(target, 'prediction_detection'):
            target.prediction_detection = self.detection

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)

        del in_values

        if self.detection:
            # TODO: evaluate the detection result
            pred_bboxes, pred_labels, pred_scores = out_values
            if len(rest_values) == 3:
                gt_bboxes, gt_labels, gt_difficults = rest_values
            elif len(rest_values) == 2:
                gt_bboxes, gt_labels = rest_values
                gt_difficults = None

            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults,
                use_07_metric=self.use_07_metric)

            report = {'map': result['map']}

            if self.label_names is not None:
                for l, label_name in enumerate(self.label_names):
                    try:
                        report['ap/{:s}'.format(label_name)] = result['ap'][l]
                    except IndexError:
                        report['ap/{:s}'.format(label_name)] = np.nan

            observation = {}
            with reporter.report_scope(observation):
                reporter.report(report, target)
            return observation

        else:
            # TODO: evaluate the segmentation result
            pred_labels, = out_values
            gt_labels, = rest_values

            result = eval_semantic_segmentation(pred_labels, gt_labels)

            report = {'miou': result['miou'],
                      'pixel_accuracy': result['pixel_accuracy'],
                      'mean_class_accuracy': result['mean_class_accuracy']}

            if self.label_names is not None:
                for l, label_name in enumerate(self.label_names):
                    try:
                        report['iou/{:s}'.format(label_name)] = result['iou'][l]
                        report['class_accuracy/{:s}'.format(label_name)] = \
                            result['class_accuracy'][l]
                    except IndexError:
                        report['iou/{:s}'.format(label_name)] = np.nan
                        report['class_accuracy/{:s}'.format(label_name)] = np.nan

            observation = {}
            with reporter.report_scope(observation):
                reporter.report(report, target)
            return observation
