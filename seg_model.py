# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for semantic segmentation."""

import segmentation_models_pytorch as smp
import torch.nn as nn
import torch



def seg_model_consturct(
    model: str = 'unet',
    backbone: str = 'resnet50',
    weights: str  = None,
    in_channels=10,
    num_classes=2, 
):
    model_return = None
    if model == 'unet':
        model_return = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if weights is True else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model == 'deeplabv3+':
        model_return = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights='imagenet' if weights is True else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(
            f"Model type '{model}' is not valid. "
            "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
        )
    if weights:
        state_dict = torch.load(weights)
        state_dict.pop('conv1.weight') # input change 
        model_return.encoder.load_state_dict(state_dict,strict=False)
    return model_return

def get_loss(loss):
    if loss == 'ce':
        return  nn.BCEWithLogitsLoss()
    elif loss == 'jaccard':
        return smp.losses.JaccardLoss(mode='binary')
    elif loss == 'focal':
        return smp.losses.FocalLoss(mode='binary', normalized=True)
    else:
        raise ValueError(
            f"Loss type '{loss}' is not valid. "
            "Currently, supports 'ce', 'jaccard' or 'focal' loss."
        )   

