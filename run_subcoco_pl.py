from mcbbox.subcoco_pl import *

run_training(img_sz=128, bs=1, acc=1, workers=1, head_runs=1, full_runs=1)
save_final()
