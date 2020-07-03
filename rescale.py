from albumentations import Compose, BboxParams, LongestMaxSize
import xml.etree.ElementTree as ET
import os
import os.path as osp
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import traceback


class Preprocessor:
    def __init__(self,
                 img_root: str,
                 annot_root: str,
                 img_dst_root: str,
                 annot_dst_root: str,
                 max_size: int = 1000):
        """Rescale the images and annotations (in Pascal VOC format) using
        albumentations.LongestMaxSize, and save the rescaled images along with
        the new notation files into destination folders. Call `process` to start processing.

        Note that this class currently only supports that all images have the same extension.

        Args:
            img_root (str): root where original images are stored
            annot_root (str): root where original 
            img_dst_root (str): destination root where rescaled images will be saved
            annot_dst_root (str): destination root where rescaled annotations will be saved
            max_size (int): max_size argument in albumentations.LongestMaxSize
        """
        self.img_root = img_root
        self.annot_root = annot_root
        self.img_dst_root = img_dst_root
        self.annot_dst_root = annot_dst_root
        self.max_size = max_size

        if not osp.exists(img_dst_root):
            os.mkdir(img_dst_root)
        else:
            if len(os.listdir(img_dst_root)) != 0:
                raise ValueError('img_dst_root is not empty!')
        if not osp.exists(annot_dst_root):
            os.mkdir(annot_dst_root)
        else:
            if len(os.listdir(annot_dst_root)) != 0:
                raise ValueError('annot_dst_root is not empty!')

        self.img_abs_paths = glob(osp.join(img_root, '*'))
        self.annot_abs_paths = glob(osp.join(annot_root, '*'))
        img_extensions = [osp.splitext(p)[1] for p in self.img_abs_paths]
        if any([img_extensions[0] != ext for ext in img_extensions[1:]]):
            raise ValueError('The extensions of images are not the same, convert them to be'
                             'the same first!')
        self.img_ext = img_extensions[0]

        self.img_names = self._extract_base_names(self.img_abs_paths)
        self.annot_names = self._extract_base_names(self.annot_abs_paths)

        self.transform = Compose([LongestMaxSize(max_size=self.max_size, always_apply=True)],
            bbox_params=BboxParams(format='pascal_voc', label_fields=['name', 'difficult']))

    def process(self):
        """Process the images and annotations"""
        count_saved = 0
        for i, annot_name in tqdm(enumerate(self.annot_names)):
            if not annot_name in self.img_names:
                continue
            img_abs_path = osp.join(self.img_root, ''.join([annot_name, self.img_ext]))
            annot_abs_path = self.annot_abs_paths[i]
            is_successed = 0
            try:
                is_successed = self.process_single(img_abs_path, annot_abs_path, annot_name)
            except ValueError:
                traceback.print_exc()
                exit(1)
            if is_successed:
                count_saved += 1

        print('Processed {} images/annotations from {} images and {} annotations'.format(
            count_saved, len(self.img_abs_paths), len(self.annot_abs_paths)))

    def _extract_base_names(self, file_abs_paths):
        return [osp.basename(osp.splitext(p)[0]) for p in file_abs_paths]

    def rescale_single(self, img: np.ndarray, annot:dict):
        data = dict(image=img)
        data.update(annot)
        result = self.transform(**data)
        return result

    @classmethod
    def load_annot_single(cls, tree: ET.ElementTree):
        root = tree.getroot()
        bboxes = []
        name = []
        difficult = []

        for obj in root.findall('object'):
            name_ = obj.find('name').text
            name.append(name_)

            difficult_ = int(obj.find('difficult').text)
            difficult.append(difficult_)

            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            bboxes.append(bbox)

        if not bboxes:
            bboxes = np.zeros((0, 4))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
        return dict(bboxes=bboxes, name=name, difficult=difficult)

    @classmethod
    def update_annot_single(cls, tree: ET.ElementTree, annot:dict, new_img_shape: tuple):
        bboxes = annot['bboxes']
        name = annot['name']
        difficult = annot['difficult']
        if len(bboxes) == 0:
            return tree
        assert len(bboxes) == len(name) and len(bboxes) == len(difficult)

        new_h, new_w = new_img_shape

        root = tree.getroot()
        size_element = root.find('size')
        height_element = size_element.find('height')
        width_element = size_element.find('width')
        height_element.text = str(int(new_h))
        width_element.text = str(int(new_w))

        objects = root.findall('object')
        for obj, bbox, name_, difficult_ in zip(objects, bboxes, name, difficult):
            name_element = obj.find('name')
            name_element.text = str(name_)

            difficult_element = obj.find('difficult')
            difficult_element.text = str(difficult_)

            bnd_box_element = obj.find('bndbox')
            xmin_element = bnd_box_element.find('xmin')
            ymin_element = bnd_box_element.find('ymin')
            xmax_element = bnd_box_element.find('xmax')
            ymax_element = bnd_box_element.find('ymax')

            xmin_element.text = str(int(bbox[0]) + 1)
            ymin_element.text = str(int(bbox[1]) + 1)
            xmax_element.text = str(int(bbox[2]) + 1)
            ymax_element.text = str(int(bbox[3]) + 1)

        return tree

    def process_single(self, img_abs_path, annot_abs_path, file_name):
        try:
            image = cv2.imread(img_abs_path)
            tree = ET.parse(annot_abs_path)
        except:
            traceback.print_exc()
            return 0
        if image is None:
            return 0
        annot = self.load_annot_single(tree)

        data = dict(image=image)
        data.update(annot)

        result = self.transform(**data)
        image = result.pop('image')
        new_img_shape = image.shape[:2]
        img_dst_path = osp.join(self.img_dst_root, ''.join([file_name, self.img_ext]))
        cv2.imwrite(img_dst_path, image)

        annot = result
        tree = self.update_annot_single(tree, annot, new_img_shape)
        annot_dst_path = osp.join(self.annot_dst_root, ''.join([file_name, '.xml']))
        tree.write(annot_dst_path, encoding='utf8')
        return 1