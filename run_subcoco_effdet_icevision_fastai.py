#!/usr/bin/python
from mcbbox.subcoco_utils import *
from mcbbox.subcoco_effdet_icevision_fastai import *

datadir, url, froot, img_subdir = 'workspace', 'http://files.fast.ai/data/examples/coco_tiny.tgz', 'coco_tiny', 'train'

# datadir, url, froot, img_subdir = 'workspace', 'https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz', 'coco_sample', 'train_sample'
train_json = fetch_subcoco(datadir=datadir, url=url, img_subdir=img_subdir)
img_dir = f'{datadir}/{froot}/{img_subdir}'
stats = load_stats(train_json, img_dir=img_dir, force_reload=False)

train_records, valid_records = parse_subcoco(stats)

img_sz=128
tfms, learn, arch = gen_transforms_and_learner(stats, train_records, valid_records, img_sz=img_sz, bs=4, acc_cycs=8)
run_training(learn, min_lr=0.01, head_runs=1, full_runs=1)

save_model_fpath = f'models/{arch}-subcoco-{img_sz}-220-final.pth'
save_final(learn, save_model_fpath)

