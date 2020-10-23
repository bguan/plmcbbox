# AUTOGENERATED! DO NOT EDIT! File to edit: 30_subcoco_pl.ipynb (unless otherwise specified).

__all__ = ['run_subcoco_pl']

# Cell
import json, os, requests, sys, tarfile, torch, torchvision
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pickle
import pytorch_lightning as pl
import torch.nn.functional as F

from collections import defaultdict
from IPython.utils import io
from pathlib import Path
from PIL import Image
from PIL import ImageStat

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.core.step_result import TrainResult

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tqdm import tqdm

from .subcoco_utils import *

# Cell
def run_subcoco_pl():
    projroot = Path(os.getcwd())
    print(f"Current Working Directory = {projroot}")
    datadir = projroot/"workspace"
    froot = "coco_sample"
    fname = f"{froot}.tgz"
    url = f"https://s3.amazonaws.com/fast-ai-coco/{fname}"
    json_fname = datadir/froot/'annotations'/'train_sample.json'
    img_dir = datadir/froot/'train_sample'
    if not os.path.isdir(datadir/froot):
        fetch_data(url, datadir, fname, chunk_size=1024*1024)
    with open(json_fname, 'r') as json_f:
        train_json = json.load(json_f)
    stats = load_stats(train_json, img_dir=img_dir)

    subcoco_dm = SubCocoDataModule(img_dir, stats, bs=12, workers=6)
    tdl=subcoco_dm.train_dataloader()
    frcnn_model = FRCNN(lbl2name=stats.lbl2name)
    save_model_fname='models/FRCNN-subcoco-{epoch}-{val_acc:.2f}.ckpt'
    chkpt_cb = ModelCheckpoint(
        filepath=save_model_fname,
        save_last=False,
        monitor='val_acc',
        mode='max'
    )
    head_trainer = Trainer(gpus=1, max_epochs=50, checkpoint_callback=chkpt_cb, accumulate_grad_batches=3)
    head_trainer.fit(frcnn_model, subcoco_dm)
    frcnn_model.unfreeze() # allow finetuning of the backbone
    trainer = Trainer(gpus=1, max_epochs=200, checkpoint_callback=chkpt_cb, accumulate_grad_batches=3)
    trainer.fit(frcnn_model, tiny_coco_dm)
    model_save_path = 'models/FRCNN-subcoco-final.saved'
    torch.save(frcnn_model.model.state_dict(), model_save_path)