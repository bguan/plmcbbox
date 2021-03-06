# AUTOGENERATED! DO NOT EDIT! File to edit: 40_subcoco_effdet_lightning.ipynb (unless otherwise specified).

__all__ = ['EffDetModule', 'save_final']

# Cell
import json, os, requests, sys, tarfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pickle
import random

from collections import defaultdict
from functools import reduce
from IPython.utils import io
from pathlib import Path
from PIL import Image
from PIL import ImageStat

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from typing import Hashable, List, Tuple, Union

# Cell
import albumentations as A
import cv2, torch, torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.multiprocessing
from albumentations.pytorch import ToTensorV2
from effdet.config.model_config import get_efficientdet_config
from effdet.factory import create_model
from effdet.bench import DetBenchPredict, DetBenchTrain, unwrap_bench
from effdet.loss import DetectionLoss, loss_fn
from torch import nn
from torch.nn.modules import module
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .subcoco_utils import *
from .subcoco_lightning_utils import *

torch.multiprocessing.set_sharing_strategy('file_system')
print(f"Python ver {sys.version}, torch {torch.__version__}, torchvision {torchvision.__version__}, pytorch_lightning {pl.__version__}, Albumentation {A.__version__}")

if is_notebook():
    from nbdev.showdoc import *

# Cell
class EffDetModule(AbstractDetectorLightningModule):
    def __init__(self, backbone_name:str="tf_efficientdet_lite0", **kwargs):
        AbstractDetectorLightningModule.__init__(self, backbone_name=backbone_name, **kwargs)
        self.config = get_efficientdet_config(model_name=backbone_name)
        self.loss_fn = DetectionLoss(self.config)

    def create_model(self, backbone_name, num_classes=1, **kwargs):
        return create_model(
            backbone_name,
            bench_task='',
            num_classes=num_classes + 1,
            pretrained=False,
            pretrained_backbone=True,
            bench_labeler=True,
        )

    def get_main_model(self):
        main_mod = self.model
        if type(main_mod) is DetBenchPredict or type(main_mod) is DetBenchTrain:
            main_mod = main_mod.model
        return main_mod

    def get_head(self):
        main_mod = self.get_main_model()
        return [main_mod.class_net, main_mod.box_net]

    def get_backbone(self):
        main_mod = self.get_main_model()
        return main_mod.backbone

    def convert_raw_predictions(self, raw_preds: torch.Tensor, detection_threshold: float=0) -> List[dict]:
        #print(f"raw_preds ={raw_preds}")
        dets = raw_preds.detach().cpu().numpy()
        preds = []
        for det in dets:
            if detection_threshold > 0:
                scores = det[:, 4]
                keep = scores > detection_threshold
                det = det[keep]
            pred = {
                "boxes": det[:, :4].clip(0, self.img_sz),
                "scores": det[:, 4],
                "labels": det[:, 5].astype(int),
            }
            preds.append(pred)

        return preds

    def stack_images(self, xs):
        xs_stack = torch.stack([xs[i] if i < len(xs) else torch.zeros((3, self.img_sz, self.img_sz)).cuda() for i in range(self.bs)])
        return xs_stack

    def pack_target(self, ys):
        target = dict(
            bbox=[ys[yi]['boxes'] if yi < len(ys) else torch.zeros((1,4)).cuda() for yi in range(self.bs)],
            cls=[ys[yi]['labels'] if yi < len(ys) else torch.Tensor([-1]).cuda() for yi in range(self.bs)]
        )
        return target

    def training_step(self, train_batch, batch_idx):
        if self.noisy: print('Entering training_step')
        self.model.train()
        bench = DetBenchTrain(unwrap_bench(self.model))
        bench.cuda()
        xs, ys = self.fix_boxes_batch(*train_batch)
        if len(xs) <= 0: return 0

        target = self.pack_target(ys)
        xs_stack = self.stack_images(xs)
        losses = bench(xs_stack, target)['loss']
        if self.noisy: print(f'Exiting training_step, returning {losses}')
        return losses

    def validation_step(self, val_batch, batch_idx):
        if self.noisy: print('Entering validation_step')
        # turn off auto gradient for validation step
        with torch.no_grad():
            xs, ys = val_batch
            predictor = DetBenchPredict(unwrap_bench(self.model))
            predictor.cuda()
            raw_preds = predictor(torch.stack(xs).cuda())
            preds = self.convert_raw_predictions(raw_preds)
            metrics = self.metrics(preds, ys)
            bench = DetBenchTrain(unwrap_bench(self.model))
            bench.cuda()
            target = self.pack_target(ys)
            xs_stack = self.stack_images(xs)
            losses = bench(xs_stack, target)['loss']

        result = { 'val_loss': losses, 'val_acc': metrics[:,0].mean(),  'val_coco': metrics[:,1].mean() }
        if self.noisy: print(f'Exiting validation_step, returning {result}')
        return result

    def forward(self, imgs):
        if self.noisy: print(f'Entering forward, training = {self.training}')
        with torch.no_grad():
            self.model.eval()
            bench = DetBenchPredict(unwrap_bench(self.model))
            raw_preds = bench(torch.stack(imgs))
            preds = self.convert_raw_predictions(raw_preds)
        if self.noisy: print(f'Exiting forward, returning {preds}')
        return preds

# Cell
def save_final(effdet_model, model_save_path):
    torch.save(effdet_model.model.state_dict(), model_save_path)