#! /usr/bin/python
from mcbbox.subcoco_pl import *

run_training(img_sz=512, bs=8, acc=4, workers=4, head_runs=20, full_runs=200)
save_final()
