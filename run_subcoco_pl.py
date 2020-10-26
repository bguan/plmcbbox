#! /usr/bin/python
from mcbbox.subcoco_utils import *
from mcbbox.subcoco_pl import *

datadir, url, img_subdir = 'workspace', 'https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz', 'train_sample'
train_json = fetch_subcoco(datadir=datadir, url=url, img_subdir=img_subdir)
img_dir = f'{datadir}/coco_sample/{img_subdir}'
stats = load_stats(train_json, img_dir=img_dir, force_reload=False)

img_sz=512
frcnn_model = run_training(stats, img_dir, img_sz=img_sz, bs=2, acc=16, workers=4, head_runs=20, full_runs=200)
model_save_path = f"models/FRCNN-subcoco-{img_sz}-220-final.saved"
save_final(frcnn_model, model_save_path)
