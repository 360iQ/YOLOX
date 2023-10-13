#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""
import albumentations as A
import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh, cxcywh2xyxy


def get_augmentation_360():
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
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))


def get_augmentation_default():
    """Return a default preset of augmentations for all images."""
    return A.Compose(
        [
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
            A.RandomScale(scale_limit=(0.4, 0.4), interpolation=cv2.INTER_LINEAR, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(hue=0.1, saturation=(0.5, 1.5), brightness=(0.5, 1.5), contrast=0.0, p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))


def get_augmentation_bullet():
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
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))


AUGMENTATION_PRESETS = {
    '360': get_augmentation_360(),
    'default': get_augmentation_default(),
    'bullet': get_augmentation_bullet()
}


def preproc(img, input_size, swap=(2, 0, 1)):
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
    def __init__(self, albumentations_pipeline=get_augmentation_default):
        self.albumentations_pipeline = albumentations_pipeline()
        self.max_labels = 50

    def __call__(self, image, targets, input_dim):
        height, width, _ = image.shape
        boxes = targets[:, :4]
        labels = targets[:, 4]

        mask = np.minimum(xyxy2cxcywh(boxes.copy())[:, 2], xyxy2cxcywh(boxes.copy())[:, 3]) > 1
        boxes = boxes[mask]
        labels = labels[mask]

        if len(boxes) == 0:
            image = self.albumentations_pipeline(image=image)['image']
            image, _ = preproc(image, input_dim)
            return image, np.zeros((self.max_labels, 5))

        augmentation_result = self.albumentations_pipeline(image=image, bboxes=boxes.tolist(),
                                                           class_labels=labels.tolist())
        image = augmentation_result['image']

        boxes = xyxy2cxcywh(np.array(augmentation_result['bboxes']))
        labels = np.array(augmentation_result['class_labels'])
        labels = np.expand_dims(labels, 1)

        image, r_ = preproc(image, input_dim)
        boxes *= r_

        targets_t = np.hstack((labels, boxes))

        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
