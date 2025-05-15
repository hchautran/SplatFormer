
path="/home/phuc/SplatFormer"
model_path="$path/train-on-objaverse.pth"
compress_mode=$1
merge_rate=$2


# for compress_mode in tome pitome tofu prune 
# do
#     for merge_rate in 0.1 0.2 0.3 0.4 0.5 
#     do
#         python calflops.py \
#             --output_dir=outputs_$compress_mode/objaverse_splatformer \
#             --gin_file=configs/dataset/objaverse.gin \
#             --gin_file=configs/model/ptv3_$compress_mode.gin \
#             --gin_file=configs/train/default.gin \
#             --gin_param="build_trainloader.batch_size=1" \
#             --only_eval --eval_subdir test --compare_with_input \
#             --gin_param="FeaturePredictor.resume_ckpt='$model_path'" --merge_rate=$merge_rate 
#     done
# done


for compress_mode in prune 
do
    for merge_rate in 0.6 0.7 0.8 0.9 0.95 
    do
        python calflops.py \
            --output_dir=outputs_$compress_mode/objaverse_splatformer \
            --gin_file=configs/dataset/objaverse.gin \
            --gin_file=configs/model/ptv3_$compress_mode.gin \
            --gin_file=configs/train/default.gin \
            --gin_param="build_trainloader.batch_size=1" \
            --only_eval --eval_subdir test --compare_with_input \
            --gin_param="FeaturePredictor.resume_ckpt='$model_path'" --merge_rate=$merge_rate 
    done
done

# python calflops.py \
#     --output_dir=outputs_base/objaverse_splatformer \
#     --gin_file=configs/dataset/objaverse.gin \
#     --gin_file=configs/model/ptv3_base.gin \
#     --gin_file=configs/train/default.gin \
#     --gin_param="build_trainloader.batch_size=1" \
#     --only_eval --eval_subdir test --compare_with_input \
#     --gin_param="FeaturePredictor.resume_ckpt='$model_path'" --merge_rate=0.0 
