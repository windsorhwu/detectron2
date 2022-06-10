import os
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.utils.visualizer import Visualizer


class Register:
    """Register a custom dataset"""

    # CLASS_NAMES = ['__background__', 'Ads']
    CLASS_NAMES = [

    ]
    # ROOT = '/data/Surv30'
    ROOT = '/mnt/data4-100.112/huangkun/ymir_data_0519/'
    ROOT = '/data2/huangkun/ymir_data_0519/'
    # ROOT='/mnt/data3-100.112/huangkun/ymir_data_0527/'
    ROOT='/mnt/data5-100.112/huangkun/ymir_data_0605/'
    ROOT='/mnt/data5-100.112/huangkun/ymir_data_0608/'
    ROOT='/mnt/data5-100.112/huangkun/ymir_data_0609/'

    def __init__(self):
        self.CLASS_NAMES = Register.CLASS_NAMES
        self.DATASET_ROOT = Register.ROOT
        self.ANN_ROOT = self.DATASET_ROOT
        self.TRAIN_PATH = os.path.join(self.DATASET_ROOT, 'images', 'surv30_train')
        self.VAL_PATH = os.path.join(self.DATASET_ROOT, 'images', 'surv30_val')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'v1_train.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'v1_val.json')

        self.PREDEFINED_SPLITS_DATASET = {
            'Surv30_cocoformat_train': (self.TRAIN_PATH, self.TRAIN_JSON),
            'Surv30_cocoformat_val': (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(name=key, json_file=json_file, image_root=image_root)

    @staticmethod
    def register_dataset_instances(name, json_file, image_root):
        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, evaluator_type='coco')

    def plain_register_dataset(self):
        # training set
        DatasetCatalog.register('Surv30_cocoformat_train', lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get('Surv30_cocoformat_train').set(thing_classes=self.CLASS_NAMES, evaluator_type='coco', json_file=self.TRAIN_JSON, image_root=self.TRAIN_PATH)

        # validation set
        DatasetCatalog.register('Surv30_cocoformat_val', lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get('Surv30_cocoformat_val').set(thing_classes=self.CLASS_NAMES, evaluator_type='coco', json_file=self.VAL_JSON, image_root=self.VAL_PATH)

    def checkout_dataset_annotation(self, name='Surv30_cocoformat_val'):
        dataset_dicts = load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        print(len(dataset_dicts))

        for i, d in enumerate(dataset_dicts, 0):
            img = cv2.imread(d['file_name'])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imwrite('outputs/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])
            if i == 200:
                break