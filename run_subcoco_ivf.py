#!/usr/bin/python
from mcbbox.subcoco_utils import *
from mcbbox.subcoco_ivf import *

datadir, url, img_subdir = 'workspace', 'https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz', 'train_sample'
train_json = fetch_subcoco(datadir=datadir, url=url, img_subdir=img_subdir)
stats = load_stats(train_json, img_dir=f'{datadir}/coco_sample/{img_subdir}', force_reload=False)
train_records, valid_records = parse_subcoco(stats)

img_sz=512
tfms, learn, backbone_name = gen_transforms_and_learner(stats, train_records, valid_records, img_sz=img_sz, bs=3, acc_cycs=12)
run_training(learn, min_lr=0.01, head_runs=20, full_runs=200)

save_model_fpath = f'models/{backbone_name}-subcoco-{img_sz}-220-final.pth'
save_final(learn, save_model_fpath)

