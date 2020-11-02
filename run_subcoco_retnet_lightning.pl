#! /usr/bin/python
from mcbbox.subcoco_utils import *
from mcbbox.subcoco_retnet_lightning import *

datadir, url, froot, img_subdir = 'workspace', 'http://files.fast.ai/data/examples/coco_tiny.tgz', 'coco_tiny', 'train'
#datadir, url, froot, img_subdir = 'workspace', 'https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz', 'coco_sample', 'train_sample'
train_json = fetch_subcoco(datadir=datadir, url=url, img_subdir=img_subdir)
img_dir = f'{datadir}/{froot}/{img_subdir}'
stats = load_stats(train_json, img_dir=img_dir, force_reload=False)

img_sz=128 #512
frcnn_model, last_save_fname = run_training(
        stats, 'models', img_dir, resume_ckpt_fname=f'retnet-{froot}-{img_sz}-last.ckpt', img_sz=img_sz, 
        bs=2, acc=16, workers=4, head_runs=1, full_runs=1,
        monitor='val_acc', mode='max', save_top=-1)
model_save_path = f"models/retnet-{froot}-{img_sz}-last.saved"
save_final(frcnn_model, model_save_path)
