"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes. Uses Albumentations library for image transformations.
"""
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
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.RandomRotate90(p=1.0),
            A.CenterCrop(height=800, width=800, p=0.3),
            A.Affine(scale=(0.6, 2.0), translate_percent=(-0.5, 0.5), rotate=(-15, 15), shear=(-8, 8), cval=0,
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
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))


def get_augmentation_default() -> Callable:
    """Return a default preset of augmentations for all images."""
    return A.Compose(
        [
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            A.RandomScale(scale_limit=(0.4, 0.4), interpolation=cv2.INTER_LINEAR, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(hue=0.1, saturation=(0.5, 1.5), brightness=(0.5, 1.5), contrast=0.0, p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))


def get_augmentation_bullet() -> Callable:
    """Return a preset of augmentations for bullet images."""
    return A.Compose(
        [
            # Geometrical transformations
            A.PadIfNeeded(min_height=1280, min_width=1280, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
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
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))


AUGMENTATION_PRESETS = {
    '360': get_augmentation_360(),
    'default': get_augmentation_default(),
    'bullet': get_augmentation_bullet()
}


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
    def __init__(self, max_labels: int = 50, augmentation_pipeline: str = "default") -> None:
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
