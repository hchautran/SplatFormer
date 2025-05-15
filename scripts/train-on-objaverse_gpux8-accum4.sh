ngpus=1
accumulate_step=1
batch_size=$((ngpus * accumulate_step))
algo=$1
merge_rate=$2

path="/home/phuc/SplatFormer"
model_path="$path/train-on-objaverse.pth"

torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs_$compress_mode/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3_$algo.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step} \
    --gin_param="FeaturePredictor.resume_ckpt='$model_path'" --merge_rate=$merge_rate 
    # --merge_rate=$merge_rate

