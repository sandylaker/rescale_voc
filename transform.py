import cv2
import numpy as np


class LongestMaxSize:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, **kwargs):
        assert 'image' in kwargs
        assert 'bboxes' in kwargs
        imgs = kwargs.pop('image')
        bboxes = kwargs.pop('bboxes')
        assert isinstance(bboxes, np.ndarray)

        h, w = imgs.shape[:2]
        scale = self._get_scale(h, w)
        if scale <= 0 or scale >= 1:
            pass
        else:
            imgs, bboxes = self._rescale(imgs, bboxes, scale)

        kwargs.update({'image': imgs, 'bboxes': bboxes})
        return kwargs

    def _get_scale(self, h, w):
        long_side = max(h, w)
        if long_side <= self.max_size:
            return -1
        else:
            return self.max_size / long_side

    def _rescale(self, imgs, bboxes, scale):
        imgs = cv2.resize(imgs, dsize=None, fx=scale, fy=scale)
        if bboxes.size == 0:
            pass
        else:
            bboxes = np.floor(bboxes * scale).astype(np.int)
            new_h, new_w = imgs.shape[:2]
            # check validity
            assert self._check_valid_bboxes(bboxes, new_h, new_w)
        return imgs, bboxes

    def _check_valid_bboxes(self, bboxes, h, w):
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]
        return (bboxes[:, [1, 3]] <= h).all() and (bboxes[:, [0, 2]] <= w).all()    # noqa


    def __repr__(self):
        return self.__class__.__name__ + '(max_size = {})'.format(self.max_size)


if __name__ == '__main__':
    image = np.random.randint(0, 255, (100, 100, 3)).astype('uint8')
    bboxes = np.array([[10, 10, 20, 20], [30, 30, 60, 100]])
    name = ['a', 'b']
    difficult = [0, 1]

    # bboxes = np.zeros((0, 4))
    # name = []
    # difficult = []

    scaler = LongestMaxSize(max_size=80)
    data = dict(image=image, bboxes=bboxes, name=name, difficult=difficult)
    result = scaler(**data)
    print('image shape: ', result['image'].shape)
    print('bboxes: ', result['bboxes'])
    print('name: ', result['name'])
    print('difficult: ', result['difficult'])