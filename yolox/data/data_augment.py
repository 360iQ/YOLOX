"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes. Uses Albumentations library for image transformations.
"""
import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh

import albumentations as A
from typing import Callable


def get_augmentation_360() -> Callable:
    """Return a preset of augmentations for 360 images."""
    return A.Compose(
        [
            # Geometrical transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.RandomRotate90(p=1.0),
            A.Affine(scale=(1.0, 1.0), translate_percent=(-0.2, 0.2), rotate=(-15, 15), shear=(-3, 3), cval=0,
                     fit_output=False, keep_ratio=False, p=0.3),
            # Color transformations
            A.SomeOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.OneOf([
                        A.HueSaturationValue(val_shift_limit=10, p=1.0),
                        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                    ], p=1.0),
                    A.CLAHE(p=1.0),
                ], n=2, p=1.0
            ),
            # Noise transformations
            A.OneOf(
                [
                    A.Downscale(scale_min=0.8, scale_max=0.9, p=1.0),
                    A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0),
                    A.ISONoise(p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.8, 1.1), per_channel=True, p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.5),
            # Blur/sharpen transformations
            A.OneOf([
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.Sharpen(alpha=(0.1, 0.4), p=1.0),
            ], p=0.5
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['labels']))


def get_augmentation_default() -> Callable:
    """Return a default preset of augmentations for all images."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['labels']))


def get_augmentation_bullet() -> Callable:
    """Return a preset of augmentations for bullet images."""
    return A.Compose(
        [
            # Geometrical transformations
            A.HorizontalFlip(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomCropFromBorders(crop_left=0.4, crop_right=0.4, crop_top=0.4, crop_bottom=0.4, p=0.3),
            A.Affine(scale=(0.3, 1.8), translate_percent=(-0.5, 0.5), rotate=(0, 0), shear=(-8, 8), cval=0,
                     fit_output=False, keep_ratio=False, p=0.5),
            # Color transformations
            A.SomeOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.OneOf([
                        A.HueSaturationValue(val_shift_limit=10, p=1.0),
                        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                    ], p=1.0),
                    A.CLAHE(p=1.0),
                ], n=2, p=1.0
            ),
            # Noise transformations
            A.OneOf(
                [
                    A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0),
                    A.ISONoise(p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.5),
            # Blur/sharpen transformations
            A.OneOf([
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.Blur(blur_limit=5, p=1.0),
                A.Sharpen(alpha=(0.1, 0.4), p=1.0),
            ], p=0.5
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['labels']))


AUGMENTATION_PRESETS = {
    '360': get_augmentation_360(),
    'default': get_augmentation_default(),
    'bullet': get_augmentation_bullet()
}

def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets

def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale

def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets

def preproc(img, input_size, swap=(2, 0, 1)):
    """
    Preprocess image. Resize to input_size with padding (if necessary) to preserve the aspect ratio and swap axes.
    Return resized image and resize ratio - the ratio of the input_size to the original image size.

    :param img: Image to preprocess.
    :param input_size: Input size of the network.
    :param swap: Axes to swap. Default is (2, 0, 1) for (channel, height, width) from (height, width, channel).
    :return: Preprocessed image and resize ratio.
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    """Applies the augmentations to the image and targets."""
    def __init__(self, max_labels: int = 50, augmentation_pipeline: str = "360") -> None:
        """
        Initializes the TrainTransform object.

        :param max_labels: Maximum number of labels per image. If the number of labels is less than this number, the
        remaining labels are padded with zeros. If the number of labels is greater than this number, the labels are
        truncated.
        :param augmentation_pipeline: Augmentation pipeline to apply. Name of the pipeline in AUGMENTATION_PRESETS.
        """
        self.max_labels = max_labels
        self.augmentation_pipeline = AUGMENTATION_PRESETS.get(augmentation_pipeline) if (
                augmentation_pipeline in AUGMENTATION_PRESETS) else None

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim: tuple[int, int]) -> tuple[
                                                                                              np.ndarray, np.ndarray]:
        """
        Applies the transformations to the image and targets. First, the bounding boxes in TLBR (x1, y1, x2, y2) format
        are filtered to remove boxes with width or height less than 1. Then, the image is passed through the
        augmentation pipeline. Then, the image is resized to the input dimension of the network with padding (if
        necessary) to preserve the aspect ratio and swapped axes (channel, height, width). Finally, bounding boxes are
        converted to (class number, center x, center y, width, height) format and padded with zeros to the maximum
        number of labels. See tests/data/data_augment.py for examples.

        :param image: Image to augment.
        :param targets: Bounding boxes of the image as a numpy array. Shape (num_boxes, 5), where 5 is (x1, y1, x2, y2,
        class_id).
        :param input_dim: Input dimension of the network.

        :return: Augmented image and padded bounding boxes. Shape (3, input_dim, input_dim) and (max_labels, 5) in
        (class number, center x, center y, width, height) format.
        """
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        mask_b = np.minimum(xyxy2cxcywh(boxes.copy())[:, 2], xyxy2cxcywh(boxes.copy())[:, 3]) > 1
        boxes = boxes[mask_b]
        labels = labels[mask_b]

        height, width, _ = image.shape

        result = self.augmentation_pipeline(image=image, bboxes=boxes, labels=labels)
        image = result['image']
        boxes = np.array([list(box) for box in result['bboxes']])
        labels = np.expand_dims(result['labels'], 1)
        # labels = np.expand_dims(labels, 1)

        image_t, r_ = preproc(image, input_dim)
        boxes = xyxy2cxcywh(boxes) if len(boxes) > 0 else np.zeros((0, 4))
        boxes *= r_

        targets_t = np.hstack((labels, boxes))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[:self.max_labels]] = targets_t[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Applies the transformations to the image. First, the image is resized to the input dimension of the network with
    padding (if necessary) to preserve the aspect ratio and swapped axes (channel, height, width). Bounding boxes are
    not considered in this transformation. TODO: check if this is correct.
    """
    def __init__(self, swap: tuple[int, int, int] = (2, 0, 1), legacy: bool = False) -> None:
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img: np.ndarray, res: np.ndarray,
                 input_size: tuple[int, int] = (416, 416)) -> tuple[np.ndarray, np.ndarray]:
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
