from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
import os
from chainercv.datasets.voc import voc_utils
from config.datasets import VOC_COMMON
import xml.etree.ElementTree as ET

from chainercv.utils import read_image
import numpy as np
import chainer


class Multi_task_VOC(GetterDataset):
    """docstring for Multi_task_VOC"""

    def __init__(self, exp_config, use_difficult=False, return_difficult=False):
        super(Multi_task_VOC, self).__init__()
        self.exp_config = exp_config
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

        self.detection_ids = []
        self.segmentation_ids = []

        self.detection_images = []
        self.detection_annos = []

        self.segmentation_images = []
        self.segmentation_masks = []

        self.all_ids = []
        self.all_images = []
        self.all_annos = []
        self.all_masks = []

        self._parse_datasets_config()
        self._merge_detection_segmentation()

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)
        self.add_getter('mask', self._get_mask)
        if not return_difficult:
            self.keys = ('img', 'bbox', 'label', 'mask')

    def _parse_datasets_config(self):
        detection_datasets = self.exp_config['detection_datasets']
        segmentation_datasets = self.exp_config['segmentation_datasets']
        for detection_dataset in detection_datasets:
            dataset_info = detection_dataset.split('_')
            dataset_name = dataset_info[0]
            dataset_year = dataset_info[1]
            dataset_set = dataset_info[2]
            image_set_file = os.path.join(VOC_COMMON[dataset_year]['ImageSets_Main'], dataset_set + '.txt')
            with open(image_set_file, 'r') as f:
                tmp_ids = [line.strip() for line in f]
                self.detection_ids += tmp_ids
                self.detection_images += [os.path.join(VOC_COMMON[dataset_year]["JPEGImages"], line.strip() + '.jpg')
                                          for
                                          line in tmp_ids]
                self.detection_annos += [os.path.join(VOC_COMMON[dataset_year]["Annotations"], line.strip() + '.xml')
                                         for
                                         line in tmp_ids]
        for segmentation_dataset in segmentation_datasets:
            dataset_info = segmentation_dataset.split('_')
            dataset_name = dataset_info[0]
            if dataset_name == 'voc':
                dataset_year = dataset_info[1]
                dataset_set = dataset_info[2]
                image_set_file = os.path.join(VOC_COMMON[dataset_year]['ImageSets_SegmentationClass'],
                                              dataset_set + '.txt')
            else:
                dataset_set = dataset_info[1]
                image_set_file = os.path.join(VOC_COMMON['sbd'][''])
            with open(image_set_file, 'r')  as f:
                tmp_ids = [line.strip() for line in f]
                self.segmentation_ids += tmp_ids
                if dataset_name == 'voc':
                    self.segmentation_images += [
                        os.path.join(VOC_COMMON[dataset_year]['JPEGImages'], line.strip() + '.jpg') for line in tmp_ids]
                    self.segmentation_masks += [
                        os.path.join(VOC_COMMON[dataset_year]['SegmentationClass'], line.strip() + '.png') for line in
                        tmp_ids]
                else:
                    pass

    def _merge_detection_segmentation(self):
        for i in range(len(self.detection_ids)):
            detection_id = self.detection_ids[i]
            self.all_ids.append(detection_id)
            self.all_images.append(self.detection_images[i])
            self.all_annos.append(self.detection_annos[i])

            if detection_id in self.segmentation_ids:
                segmentation_index = self.segmentation_ids.index(detection_id)

                self.all_masks.append(self.segmentation_masks[segmentation_index])
                del self.segmentation_ids[segmentation_index]
                del self.segmentation_masks[segmentation_index]
            else:
                self.all_masks.append(None)
        for i in range(len(self.segmentation_ids)):
            segmentation_id = self.segmentation_ids[i]
            self.all_ids.append(segmentation_id)
            self.all_images.append(self.segmentation_images[i])
            self.all_annos.append(None)
            self.all_masks.append(self.segmentation_masks[i])

        self.segmentation_masks = None
        self.segmentation_images = None
        self.segmentation_ids = None

        self.detection_annos = None
        self.detection_images = None
        self.detection_ids = None

    def __len__(self):
        return len(self.all_ids)

    def _get_image(self, i):
        id_ = self.all_images[i]
        img_path = id_
        img = read_image(img_path, color=True)
        if img is None:
            # TODO: TypeError
            raise TypeError
        return img

    def _get_annotations(self, i):
        id_ = self.all_annos[i]

        if id_ is None:
            # TODO: TypeError
            raise TypeError

            return None

        anno = ET.parse(
            id_)
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_utils.voc_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)

        return bbox, label, difficult

    def _get_mask(self, i):
        id_ = self.all_masks[i]
        if id_ is None or not os.path.exists(id_):
            image_id = self.all_images[i]
            image_ = read_image(image_id, color=True)
            _,h,w=image_.shape
            mask = np.ones([h,w], dtype=np.int32) * -1
            return mask

        # TODO(yuyu2172): Find an option to load properly even with cv2.
        with chainer.using_config('cv_read_image_backend', 'PIL'):
            label = read_image(id_, dtype=np.int32, color=False)
        label[label == 255] = -1
        # (1, H, W) -> (H, W)
        return label[0]
