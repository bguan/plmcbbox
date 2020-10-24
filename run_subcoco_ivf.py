#!/usr/bin/python
from mcbbox.subcoco_ivf import *

img_sz, bs, acc, head_runs, full_runs, lr = 512, 4, 8, 20, 100, 0.01
tfms, learn, backbone_name = gen_transforms_and_learner(img_size=img_sz, bs=bs, acc_cycs=acc)
run_training(learn, min_lr=lr, head_runs=head_runs, full_runs=full_runs)
save_model_fpath = f'models/{backbone_name}-subcoco-{img_sz}-final.pth'
save_final(learn, save_model_fpath)

