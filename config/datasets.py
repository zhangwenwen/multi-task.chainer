import os

# common configuration for VOC dataset

VOC_BASE = "/home/andy/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit"
# VOC_BASE = "/Volumes/Developments/Dataset/VOCdevkit"
SBD_BASE = "/home/andy/VOC/benchmark_RELEASE"

COCO_BASE = "/home/andy/coco"

VOC_COMMON = {
    "2007": {
        "JPEGImages": os.path.join(VOC_BASE, "VOC2007/JPEGImages"),
        "Annotations": os.path.join(VOC_BASE, "VOC2007/Annotations"),
        "SegmentationClass": os.path.join(VOC_BASE, 'VOC2007/SegmentationClass'),
        "ImageSets_Main": os.path.join(VOC_BASE, 'VOC2007/ImageSets/Main'),
        "ImageSets_SegmentationClass": os.path.join(VOC_BASE, 'VOC2007/ImageSets/Segmentation')
    },
    "2012": {
        "JPEGImages": os.path.join(VOC_BASE, "VOC2012/JPEGImages"),
        "Annotations": os.path.join(VOC_BASE, "VOC2012/Annotations"),
        "SegmentationClass": "/home/andy/VOC/benchmark_RELEASE/dataset/png_cls",
        "ImageSets_Main": os.path.join(VOC_BASE, 'VOC2012/ImageSets/Main'),
        "ImageSets_SegmentationClass": os.path.join(VOC_BASE, 'VOC2012/ImageSets/Main')
    },
    "sbd": {"ImageSets_SegmentationClass": os.path.join(SBD_BASE, "dataset/png_cls")}
}

# specific experiments with specific datasets


voc_experiments = {
    'final_voc': {
        'final_voc_train': {
            'detection_datasets': ['voc_2007_trainval', 'voc_2012_trainval'],
            'segmentation_datasets': ['voc_2012_trainnoval']
        },
        'final_voc_test': {
            'detection_datasets': ['voc_2007_test'],
            'segmentation_datasets': ['voc_2012_val']
        }
    },
    'debug': {
        'debug_train': {
            'detection_datasets': ['voc_2012_trainnoval'],
            'segmentation_datasets': ['voc_2012_trainnoval']
        },
        'debug_test': {
            'detection_datasets': ['voc_2012_val'],
            'segmentation_datasets': ['voc_2012_val']
        }
    },
    'ablation': {
        'ablation_train': {
            'detection_datasets': ['voc_2012_train'],
            'segmentation_datasets': ['voc_2012_train']
        },
        'ablation_test': {
            'detection_datasets': ['voc_2012_val'],
            'segmentation_datasets': ['voc_2012_val']
        }
    },
    'attention': {
        'attention_train': {
            'detection_datasets': ['voc_2012_trainnoval'],
            'segmentation_datasets': ['voc_2012_trainnoval']
        },
        'attention_test': {
            'detection_datasets': ['voc_2012_val'],
            'segmentation_datasets': ['voc_2012_val']
        }
    },

    'train_det_voc12_train_seg_voc12_train': {

    }
}

COCO_COMMON = {
    "2015": {
        "images": os.path.join(COCO_BASE, "images"),
        "annotations": os.path.join(COCO_BASE, "annotations")
    }

}
coco_experiments = {
    'final_coco': {
        'final_coco_train': {
            'detection_datasets': [],
            'segmentation_datasets': []
        },
    },
    'coco_mini': {
        'coco_mini_train': {
            'coco_mini': {
                'detection_datasets': [],
                'segmentation_datasets': []
            }
        }
    }
}
