ngpus=8
accumulate_step=4
batch_size=$((ngpus * accumulate_step))

torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_param="SplatfactoDataset.nerfstudio_path='../nerfstudio'" \
    --gin_param="SplatfactoDataset.colmap_path='../code/objaverse-xl/scripts/rendering/OOD'" \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step}

