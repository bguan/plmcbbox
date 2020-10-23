from mcbbox.subcoco_pl import *

run_training(img_sz=128, bs=4, acc=2, workers=4, head_runs=1, full_runs=1)
save_final()
