#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
from loguru import logger
import torch
import torch.nn as nn


def get_smallest_enclosing_box(box1, box2) -> tuple[torch.Tensor, torch.Tensor]:
    box_top_left = torch.min(
        (box1[:, :2] - box1[:, 2:] / 2), (box2[:, :2] - box2[:, 2:] / 2)
    )
    box_bottom_right = torch.max(
        (box1[:, :2] + box1[:, 2:] / 2), (box2[:, :2] + box2[:, 2:] / 2)
    )
    return box_top_left, box_bottom_right


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def __repr__(self):
        return f"{self.__class__.__name__}(loss_type={self.loss_type})"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # pred and target shape = [c_x, c_y, w, h]
        top_left_intersection = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        bottom_right_intersection = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_pred = torch.prod(pred[:, 2:], 1)
        area_target = torch.prod(target[:, 2:], 1)

        en = (top_left_intersection < bottom_right_intersection).type(top_left_intersection.type()).prod(dim=1)
        area_intersection = torch.prod(bottom_right_intersection - top_left_intersection, 1) * en
        area_union = area_pred + area_target - area_intersection
        iou = area_intersection / (area_union + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2

        elif self.loss_type == "giou":
            c_tl, c_br = get_smallest_enclosing_box(pred, target)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - torch.abs(area_c - area_union) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "ciou":
            c_tl, c_br = get_smallest_enclosing_box(pred, target)
            c2 = torch.pow(c_br - c_tl, 2).sum(dim=1).clamp(min=1e-16)  # diagonal length squared
            rho2 = torch.pow(pred[:, :2] - target[:, :2], 2).sum(dim=1)  # center distance squared

            # Calculate v - aspect ratio consistency
            w1, h1 = pred[:, 2].clamp(min=1e-7), pred[:, 3].clamp(min=1e-7)
            w2, h2 = target[:, 2].clamp(min=1e-7), target[:, 3].clamp(min=1e-7)
            v = ((4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)).clamp(min=0.0, max=1.0)

            with torch.no_grad():
                alpha = (v / ((1 - iou + v).clamp(min=1e-7))).clamp(min=0.0, max=1.0)

            ciou = (iou - (rho2 / c2 + alpha * v))
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
