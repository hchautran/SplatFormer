# SplatFormer: Point Transformer for Robust 3D Gaussian Splatting
[Project page]() | [Paper]() <br>
![Teaser image](assets/teaser.png)

This repo contains the official implementation for the paper "SplatFormer: Point Transformer for Robust 3D Gaussian Splatting". Our approach uses a point transformer to refine 3DGS for out-of-distribution novel view synthesis in a single feed-forward.

## Installation
We tested on a server configured with Ubuntu 22.04, cuda 11.8, and gcc 8.5.0. Other similar configurations should also work.
```
git clone git@github.com:ChenYutongTHU/SplatFormer.git
cd SplatFormer
conda create -n splatformer python=3.8 -y
conda activate splatformer

# Install the pytorch version for your cuda version.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
#ninja pip, pytorch-scatter(pip install torch-scatter)
conda install ninja h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm pytorch-cluster pytorch-scatter pytorch-sparse -c anaconda -c conda-forge -c pyg -y
pip install requirements.txt

# Install pointcept for point transformer

# Install nerfstudio


```

## Out-of-distribution (OOD) Novel View Synthesis Test Sets
Our OOD-NVS test sets can be downloaded [here](). There are three object-centric OOD NVS test sets rendered from ShapeNet-core, Objaverse-v1, and GSO, and one real-world iPhone image set captured by us. All scene directories are in colmap-like structure. 

For object-centric, you can follow this [instruction]() to re-render the 3D scenes by yourself. We use the ground-truth camera pose provided by Blender and save the bounding box coordinates of the object, which will be used in initialization for 3DGS training.

For real-world iPhone captures, we use [hloc](https://github.com/cvg/Hierarchical-Localization) to estimate the camera poses. The provided point cloud is estimated from the training cameras and the provided images are already undistorted.

## Training SplatFormer
### Training set generation 
We provide rendering and 3DGS scripts to generate our training datasets. Please see [Dataset_generation.md]() for more details.
After generating the rendered images and initial 3DGS, please put them under [train-set/] as 

    .
    └── train-set                    
        ├── objaverseOOD 
            ├── colmap
               ├── 
            ├── nerfstudio   
               ├──  
We also provide a small subset of our training set [here]().

### Train
```
sh scripts/train-on-objaverse.sh
sh scripts/train-on-shapenet.sh
```
By default, we use 8x4090 gpus or 8x3090 gpus, and a accumulation step of 4. You can change the configurations in the training script. You can download our trained checkpoints [here]().

## Evaluating SplatFormer
To evaluate the trained SplatFormer, please download the [OOD-NVS test sets and the initial 3DGS]() trained using the input views, unextract them and put them under test-set/ as 

    .
    └── test-set                    
        ├── GSOOOD  
            ├── colmap
            └── nerfstudio     
        ├── objaverseOOD         
        └── RealOOD                
You can download models trained on Objaverse-v1 and Shapenetcore [here]() and run the evaluations.
```
sh scripts/train-on-objaverse_test.sh # Evaluate the model trained on Objaverse-v1 on Objaverse-OOD, GSO-OOD, and Real-OOD
sh scripts/train-on-shapenet_test.sh # Evaluate the model trained on ShapeNet on ShapeNet-OOD
```
Then under the output directory (e.g. outputs/objaverse_splatformer/test), you can see the evaluation metrics in eval.log, OOD renderings in objaverse/pred, and compare with 3DGS in objaverse/compare.

* Note: Our SplatFormer takes 3DGS trained for 10k steps as input. In our paper, we report 3DGS trained for 30k steps (default setting) as baselines. The two 3DGS training configurations lead to only small difference in the evaluation performance.


## Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{}
}
```

## LICENSE
The license of training dataset.