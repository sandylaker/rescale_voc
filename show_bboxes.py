import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rescale import Preprocessor
import xml.etree.ElementTree as ET
import os.path as osp
from PIL import Image


def convert_bbox(bbox):
    xmin, ymin, xmax, ymax = list(map(int, bbox))
    height = xmax - xmin + 1
    width = ymax - ymin + 1
    return xmin, ymin, width, height


def visualize(img, bboxes, ax):
    ax.imshow(img)
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, width, height = convert_bbox(bbox)
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', fill=False)
        ax.add_patch(rect)

def compare_bboxes(file_name):
    origin_annot_path = osp.join('annotations', ''.join([file_name,'.xml']))
    new_annot_path = osp.join('annotations_rescaled', ''.join([file_name,'.xml']))
    origin_img_path = osp.join('images', ''.join([file_name,'.jpg']))
    new_img_path = osp.join('images_rescaled', ''.join([file_name,'.jpg']))

    origin_img = Image.open(origin_img_path)
    new_img = Image.open(new_img_path)
    print('original shape: {}, new shape: {}'.format(origin_img.size, new_img.size))
    origin_tree = ET.parse(origin_annot_path)
    new_tree = ET.parse(new_annot_path)
    origin_annot = Preprocessor.load_annot_single(origin_tree)
    new_annot = Preprocessor.load_annot_single(new_tree)

    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    visualize(origin_img, origin_annot['bboxes'], ax[0])
    visualize(new_img, new_annot['bboxes'], ax[1])
    plt.show()


if __name__ == '__main__':
    compare_bboxes('2012_004331')

