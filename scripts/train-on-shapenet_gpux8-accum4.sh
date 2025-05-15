ngpus=1
accumulate_step=1
batch_size=$((ngpus * accumulate_step))

# torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29518 \
#     train.py \
#     --output_dir=outputs/shapenet_splatformer \
#     --gin_file=configs/dataset/shapenet.gin \
#     --gin_file=configs/model/ptv3.gin \
#     --gin_file=configs/train/default.gin \
#     --gin_param="FeaturePredictor.input_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
#     --gin_param="FeaturePredictor.output_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
#     --gin_param="FeaturePredictor.sh_degree=0" \
#     --gin_param="build_trainloader.batch_size="${batch_size} \
#     --gin_param="build_trainloader.accumulate_step="${accumulate_step}

python train.py \
    --output_dir=outputs/shapenet_splatformer \
    --gin_file=configs/dataset/shapenet.gin \
    --gin_file=configs/model/ptv3_base.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="FeaturePredictor.input_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
    --gin_param="FeaturePredictor.output_features= ['means','scales', 'opacities', 'quats', 'features_dc']" \
    --gin_param="FeaturePredictor.sh_degree=0" \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step}




