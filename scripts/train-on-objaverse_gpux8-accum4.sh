ngpus=1
accumulate_step=1
batch_size=$((ngpus * accumulate_step))
echo $batch_size
compress_mode=$1

torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs_$compress_mode/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3_$compress_mode.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step}

