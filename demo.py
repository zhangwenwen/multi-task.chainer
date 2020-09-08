import argparse

import chainer
from chainercv import utils
from chainer import serializers

from chainercv.datasets import voc_bbox_label_names, coco_bbox_label_names
from chainercv.datasets import voc_semantic_segmentation_label_colors, voc_semantic_segmentation_label_names

from matplotlib.patches import Patch
from multi_task.multi_task_300 import Multi_task_300
from multi_task.multi_task_512 import Multi_task_512
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from chainercv.visualizations.vis_image import vis_image


def blend(img, mask_img, mask):
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] >= 1:
                img[i, j] = mask_img[i, j]
    return img


def vis_bbox(img, bbox, label=None, score=None, label_names=None,
             instance_colors=None, alpha=1., linewidth=3., ax=None, fontsize=10):
    from matplotlib import pyplot as plt

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10}, fontsize=fontsize)
    return ax


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

    args = parser.parse_args()

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

    model.use_preset('visualize')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.dataset == 'voc':
        use_07 = True
        label_names = voc_bbox_label_names
        seg_label_names = voc_semantic_segmentation_label_names

    elif args.dataset == 'coco':
        label_names = coco_bbox_label_names

    if args.model_path:
        serializers.load_npz(args.model_path, model)

    while True:
        img_path = input("Input image path:")
        img = utils.read_image(img_path, color=True)
        bboxes, labels, scores, masks = model.demo([img], detection=args.eval_det, segmentation=args.eval_seg)

        ax = vis_bbox(
            img, bboxes[0], labels[0], scores[0], label_names=voc_bbox_label_names,
            instance_colors=[(0, 255, 0), (0, 0, 255)], linewidth=3, fontsize=16)

        mask_o = np.uint8(masks[0])

        mask_p = Image.fromarray(mask_o, 'P')

        # blend mask and original image
        # mask_img = np.array(mask_p).transpose([1, 2, 0])

        # img[mask_o > 0] = mask_img[mask_o > 0]

        palette = np.load('Extra/palette.npy').tolist()
        mask_p.putpalette(palette)

        mask_p = mask_p.convert('RGB')
        mask_p = np.array(mask_p)
        img = img.transpose([1, 2, 0])
        mask_blend = blend(img, mask_p, mask_o)

        ax.imshow(mask_blend.astype(np.uint8), alpha=0.85)

        # geterate legend
        legend_handles = []
        n_class = len(voc_semantic_segmentation_label_names)

        all_label_names_in_legend = False
        if not all_label_names_in_legend:
            legend_labels = [l for l in np.unique(masks[0]) if l > 0]
        else:
            legend_labels = range(n_class)

        label_colors = voc_semantic_segmentation_label_colors
        label_colors = np.array(label_colors) / 255
        cmap = matplotlib.colors.ListedColormap(label_colors)
        for l in legend_labels:
            legend_handles.append(
                Patch(color=cmap(l / (n_class - 1)), label=seg_label_names[l]))

        plt.axis('off')

        ax.legend(handles=legend_handles, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=5, fontsize=16)

        plt.show()


if __name__ == '__main__':
    main()
