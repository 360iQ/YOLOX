import unittest

import torch
from torch import nn

from yolox.models import IOUloss
from yolox.models.losses import get_smallest_enclosing_box


class TestIOULoss(unittest.TestCase):
    def setUp(self):
        self.gt_bboxes = torch.tensor([[0.5, 0.5, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        self.perfect_bboxes = torch.tensor([[0.5, 0.5, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        self.bad_bboxes = torch.tensor([[1, 0.75, 1, 1], [1, 0.75, 1, 0.5], [0.75, 1, 0.5, 1], [1, 1, 0.5, 0.5]])

    def _test_loss(self, loss_function: nn.Module, gt_loss_values: torch.Tensor):
        loss_values = loss_function(self.perfect_bboxes, self.gt_bboxes)
        assert all([loss_value == 0. for loss_value in loss_values])
        loss_values = loss_function(self.bad_bboxes, self.gt_bboxes)
        assert torch.allclose(loss_values, gt_loss_values, atol=1e-4)

    def test_iuo_loss(self):
        self._test_loss(IOUloss(), torch.tensor([0.9467, 0.75, 0.75, 1.]))

    def test_giou_loss(self):
        self._test_loss(IOUloss(loss_type='giou'), torch.tensor([0.9026, 0.5, 0.5, 1.5]))

    def test_ciou_loss(self):
        self._test_loss(IOUloss(loss_type="ciou"), torch.tensor([0.8512, 0.5345, 0.5345, 1.2500]))

    def test_get_smallest_enclosing_box(self):
        bboxes1 = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        bboxes2 = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
        gt_values = (torch.tensor([[0.5, 0.5], [-0.5, -0.5]]), torch.tensor([[1.5, 1.5], [1.5, 1.5]]))
        values = get_smallest_enclosing_box(bboxes1, bboxes2)
        for value, gt_value in zip(values, gt_values):
            assert (value == gt_value).all(), f"{value} != {gt_value}"
