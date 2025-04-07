
path="/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/llm_quantize/SplatFormer/"
model_path="$path/train-on-objaverse.pth"

torchrun --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3_tome.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size=1" \
    --only_eval --eval_subdir test --compare_with_input \
    --gin_param="FeaturePredictor.resume_ckpt='$model_path'" 


