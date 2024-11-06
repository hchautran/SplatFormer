# ngpus=8
# accumulate_step=4
ngpus=1
accumulate_step=4
batch_size=$((ngpus * accumulate_step))

torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs_debug/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step} \
    --only_eval --eval_subdir test_debug1 --compare_with_input \
    --gin_param="FeaturePredictor.resume_ckpt='../3dgs_multiple-scenes/outputs/objaverse_splat/48k/res384_loss1-lpips1_maxgsnum100000_stride1222-skip-embedMLP_gpu8-acc4_amp/checkpoints/model_00173999.pth'" 


