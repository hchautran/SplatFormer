
path="/home/phuc/SplatFormer"
model_path="$path/train-on-objaverse.pth"
compress_mode=tome

torchrun  visualize.py --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3_$compress_mode.gin \
    --gin_file=configs/train/default.gin \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_param="FeaturePredictor.resume_ckpt='$model_path'" 


