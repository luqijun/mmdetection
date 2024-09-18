# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale
from .transforms import RandomFlip, RandomCrop
import mmengine

@TRANSFORMS.register_module()
class RandomFlipCrowd(RandomFlip):
    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip points
        if results.get('gt_points', None) is not None:
            results['gt_points'] = self.flip_points(results['gt_points'], img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


    def flip_points(self,
                    points: np.ndarray,
                    img_shape: Tuple[int, int],
                    direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = points
        origin_points = flipped.clone()
        if direction == 'horizontal':
            flipped[..., 0] = img_shape[1] - origin_points[..., 0]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - origin_points[..., 1]
        else:
            flipped[..., 0] = img_shape[1] - origin_points[..., 0]
            flipped[..., 1] = img_shape[0] - origin_points[..., 1]



@TRANSFORMS.register_module()
class RandomChoiceResizeCrowd(BaseTransform):
    """Resize images & bbox & mask from a list of multiple scales.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. Resize scale will be randomly
    selected from ``scales``.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the target scale is sampled from the list
      uniformally.
    - if `scale` is a tuple, the target scale will be set to the tuple.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - scale
    - scale_factor
    - scale_idx
    - keep_ratio


    Args:
        scales (Union[list, Tuple]): Images scales for resizing.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scales: Sequence[Union[int, Tuple]],
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]
        assert mmengine.is_seq_of(self.scales, (tuple, int))

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Resize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    @cache_randomness
    def _random_select(self) -> Tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(self.scales))
        scale = self.scales[scale_idx]
        return scale, scale_idx

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select()
        self.resize.scale = target_scale
        results = self.resize(results)
        results['scale_idx'] = scale_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scales={self.scales}'
        repr_str += f', resize_cfg={self.resize_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCropCrowd(RandomCrop):

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                    crop_x1:crop_x2]

        return results