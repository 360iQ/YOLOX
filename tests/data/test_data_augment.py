import pytest

from yolox.data.data_augment import TrainTransform, preproc
import numpy as np
import albumentations as A


def test_preproc():
    image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    image_preprocessed, r = preproc(image, (416, 416))
    assert image_preprocessed.shape == (3, 416, 416)


@pytest.mark.parametrize("pre_input_size, post_input_size, pre_boxes, post_boxes", [
    ((100, 100), (300, 300), np.array([[0., 0., 100., 100., 0.]]), np.array([[0., 150., 150., 300., 300.]])),
    ((100, 200), (300, 300), np.array([[0., 0., 100., 100., 0.]]), np.array([[0., 75., 75., 150., 150.]])),
    ((100, 200), (300, 300), np.array([[50., 50., 100., 100., 0.]]), np.array([[0., 112.5, 112.5, 75., 75.]])),
    ((200, 100), (300, 300), np.array([[10., 10., 30., 30., 0.]]), np.array([[0., 30., 30., 30., 30.]])),
    ((100, 100), (300, 300), np.array([[10., 10., 10., 10., 0.]]), np.array([[0., 0., 0., 0., 0.]])),
])
def test_traintransform(pre_input_size, post_input_size, pre_boxes, post_boxes):
    image = np.ones((*pre_input_size, 3), dtype=np.uint8) * 255
    train_transform = TrainTransform()
    train_transform.augmentation_pipeline = A.Compose(
        [], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['labels']))

    image_t, padded_labels = train_transform(image, pre_boxes, (300, 300))
    assert image_t.shape == (3, *post_input_size)
    assert padded_labels.shape == (50, 5)
    assert np.allclose(padded_labels[0], post_boxes[0])
    assert image_t.max() == 255
