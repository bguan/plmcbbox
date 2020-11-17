#! /usr/bin/python
import sys

from mcbbox.subcoco_utils import *
from mcbbox.subcoco_effdet_lightning import *

#datadir, url, froot, img_subdir = 'workspace', 'http://files.fast.ai/data/examples/coco_tiny.tgz', 'coco_tiny', 'train'
datadir, url, froot, img_subdir = 'workspace', 'https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz', 'coco_sample', 'train_sample'

train_json = fetch_subcoco(datadir=datadir, url=url, img_subdir=img_subdir)
img_dir = f'{datadir}/{froot}/{img_subdir}'
stats = load_stats(train_json, img_dir=img_dir, force_reload=False)

img_sz=512
effdet_model, last_save_fname = run_training(
        stats, 'models', img_dir, resume_ckpt_fname='last.ckpt', img_sz=img_sz, bs=4, acc=8, workers=4, head_runs=0, full_runs=100,
        monitor='val_acc', mode='max', save_top=-1, calc_metrics=True, patience=10)
model_save_path = f"models/effdet-{froot}-{img_sz}-last.saved"
save_final(effdet_model, model_save_path)
sys.exit(f'Run ended, model saved to {model_save_path}')
