
path="$PROJECT_DIR/SplatFormer"
model_path="$path/train-on-objaverse.pth"
compress_mode=$1


CUDA_VISIBLE_DEVICES=$2 python calflops.py \
    --output_dir=outputs_$compress_mode/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3_$compress_mode.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size=1" \
    --only_eval --eval_subdir test --compare_with_input \
    --gin_param="FeaturePredictor.resume_ckpt='$model_path'"
