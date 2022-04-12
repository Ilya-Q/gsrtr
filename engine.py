# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import util.misc as utils
from util import box_ops
from typing import Iterable
import tqdm
import json

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]
        
        # model output & calculate loss
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_swig(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # data & target
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        # model output & calculate loss
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets, eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

@torch.no_grad()
def predict_eval_swig(model, data_loader, device, idx_to_verb, idx_to_role, vidx_ridx, idx_to_class):
    model.eval()
    preds = {}
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) if type(v) is not str else v for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)
        for i, info in enumerate(targets):
            image_name = info['img_name'].split('/')[-1]
            pred_verb = outputs['pred_verb'][i]
            pred_noun = outputs['pred_noun'][i]
            pred_bbox = outputs['pred_bbox'][i]
            pred_bbox_conf = outputs['pred_bbox_conf'][i]
            top1_verb = torch.topk(pred_verb, k=1, dim=0)[1].item()
            roles = vidx_ridx[top1_verb]
            num_roles = len(roles)
            verb_label = idx_to_verb[top1_verb]
            role_labels = []
            noun_labels = []
            for i in range(num_roles):
                top1_noun = torch.topk(pred_noun[i], k=1, dim=0)[1].item()
                role_labels.append(idx_to_role[roles[i]])
                noun_labels.append(idx_to_class[top1_noun])
            mw, mh = info['max_width'], info['max_height']
            w, h = info['width'], info['height']
            shift_0, shift_1, scale  = info['shift_0'], info['shift_1'], info['scale']
            pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(pred_bbox.clone(), mw, mh, device=device)
            for i in range(num_roles):
                pb_xyxy[i][0] = max(pb_xyxy[i][0] - shift_1, 0)
                pb_xyxy[i][1] = max(pb_xyxy[i][1] - shift_0, 0)
                pb_xyxy[i][2] = max(pb_xyxy[i][2] - shift_1, 0)
                pb_xyxy[i][3] = max(pb_xyxy[i][3] - shift_0, 0)
                # locate predicted boxes within image (processing w/ image width & height)
                pb_xyxy[i][0] = min(pb_xyxy[i][0], w)
                pb_xyxy[i][1] = min(pb_xyxy[i][1], h)
                pb_xyxy[i][2] = min(pb_xyxy[i][2], w)
                pb_xyxy[i][3] = min(pb_xyxy[i][3], h)
            pb_xyxy /= scale

            preds[image_name] = {
                "verb": top1_verb,
                "nouns": {role:noun for role, noun in zip(role_labels, noun_labels)},
                "boxes": {role_labels[i]: list(pb_xyxy[i]) if pred_bbox_conf[i] >= 0 else None for i in range(num_roles)}
            }
    return preds
