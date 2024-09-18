# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmdet.visualization.palette import _get_adaptive_scales


@HOOKS.register_module()
class Crowd_DetVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        save_test(outputs, self.test_out_dir)
        return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)



import torchvision.transforms as standard_transforms
import cv2
from PIL import Image
import torch.nn.functional as F
import os

def save_test(outputs, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    out_info = outputs[0]
    img_id = out_info.img_id
    img_path = out_info.img_path
    img = Image.open(img_path)
    gt_points = out_info.gt_instances.bboxes[:, 0:2]
    pre_points = out_info.pred_instances.bboxes
    save_results_points_with_seg_map(os.path.join(out_dir, f'{img_id}.jpg'),
                                     img, pre_points, gt_points)


def save_results_points_with_seg_map(exp_path, img0,
                                     pre_points=None, gt_points=None):  # , flow):

    # gt_cnt = gt_map0.sum().item()
    # pre_cnt = pre_map0.sum().item()
    pil_to_tensor = standard_transforms.ToTensor()
    tensor_to_pil = standard_transforms.ToPILImage()

    # img0 = img0.detach().to('cpu')
    # pil_input0 = tensor_to_pil(img0)
    pil_input0 = img0

    UNIT_W, UNIT_H = pil_input0.size

    # mask_color_map = cv2.applyColorMap((255 * tensor[8]).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    RGB_R = (255, 0, 0)
    RGB_G = (0, 255, 0)

    BGR_R = (0, 0, 255)  # BGR
    BGR_G = (0, 255, 0)  # BGR
    thickness = 2
    pil_input0 = np.array(pil_input0)

    if pre_points is not None:
        for i, point in enumerate(pre_points.detach().cpu().numpy(), 0):
            point = point.astype(np.int32)
            point = (point[0], point[1])
            cv2.drawMarker(pil_input0, point, RGB_G, markerType=cv2.MARKER_CROSS, markerSize=3,
                           thickness=thickness)
            # cv2.drawMarker(pil_input0, point, RGB_R, markerType=cv2.MARKER,markerSize=20,thickness=3)

    if gt_points is not None:
        for i, point in enumerate(gt_points.detach().cpu().numpy(), 0):
            point = point.astype(np.int32)
            point = (point[0], point[1])
            cv2.circle(pil_input0, point, 1, RGB_R, thickness)

    gt_text_loc = (20, 30) if UNIT_W <= 256 else (50, 50)  # (UNIT_W * 0.1, UNIT_H * 0.15)
    pre_text_loc = (20, 70) if UNIT_W <= 256 else (50, 150)  # (UNIT_W * 0.1, UNIT_H * 0.35)
    font_scale = 1 if UNIT_W <= 256 else 2
    text_thickness = 1 if UNIT_W <= 256 else 2
    cv2.putText(pil_input0, 'GT:' + str(len(gt_points)), gt_text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 100, 0), thickness=text_thickness)
    cv2.putText(pil_input0, 'Pre:' + str(len(pre_points)), pre_text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (100, 255, 0), thickness=text_thickness)

    pil_input0 = Image.fromarray(pil_input0)

    imgs = [pil_input0]

    # 保存图片 从左到右，从上到下
    w_num, h_num = 1, 1
    target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
    target = Image.new('RGB', target_shape)
    count = 0
    for img in imgs:
        if count > 0 and count % w_num == 0:
            count += 1  # 第一列不填充
        x, y = int(count % w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
        target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
        count += 1
    target.save(exp_path)

def draw_all_character(visualizer, characters, w):
    start_index = 2
    y_index = 5
    for char in characters:
        if isinstance(char, str):
            visualizer.draw_texts(
                str(char),
                positions=np.array([start_index, y_index]),
                colors=(0, 0, 0),
                font_families='monospace')
            start_index += len(char) * 8
        else:
            visualizer.draw_texts(
                str(char[0]),
                positions=np.array([start_index, y_index]),
                colors=char[1],
                font_families='monospace')
            start_index += len(char[0]) * 8

        if start_index > w - 10:
            start_index = 2
            y_index += 15

    drawn_text = visualizer.get_image()
    return drawn_text
