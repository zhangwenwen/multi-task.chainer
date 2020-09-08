from chainercv.chainer_experimental.datasets.sliceable import GetterDataset


class Multi_task_COCO(GetterDataset):
    def __init__(self):
        super(Multi_task_COCO, self).__init__()

    def __len__(self):
        pass

    def _get_annotations(self):
        pass

    def _get_image(self):
        pass

    def _get_mask(self):
        pass

    def _parse_dataset_config(self):
        detection_datasets = self.exp_config['detection_datasets']
        segmentation_datasets = self.exp_config['segmentation_datasets']
