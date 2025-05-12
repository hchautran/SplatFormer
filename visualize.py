import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import OrderedDict
import pyviz3d.visualizer as viz
import random
import argparse
from dataset.Loader import build_trainloader, build_testloader
from models.feature_predictor import FeaturePredictor
from pointcept.engines.defaults import (
    # default_argument_parser,
    default_config_parser,
    # default_setup,
)
import gin
from utils import gpu_utils, gs_utils, loss_utils
import torch.distributed as dist
from absl import app, flags
from train import set_seed, training
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import VALID_TOME_MODES
from collections import defaultdict
import gc

VALID_TOME_MODES = ["patch", "tome", "progressive", "pitome", "random_patch", "base", 'important_patch']


def create_color_palette():
    return np.array([
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144),
       (0,0,0),
    ], dtype=np.uint8)


class Model:
    def __init__(self):

        self.additional_info={
            "replace_attn": None,
            "tome": "patch",
            "r": 0.1,
            "stride": 50,
            "tome_mlp": True,
            "tome_attention": True,
            "trace_back": False,
            "single_head_tome": False,
            "margin":0.9,
            "alpha": 1.0,
            "threshold": 0.9
        }

        self.model = FeaturePredictor(additional_info=self.additional_info)
        self.model.eval()
        self.test_loader = build_testloader()
        self.train_loader = build_trainloader()
        self.dict = {}

    def get_model(self):
        return self.model

    def get_loader(self, mode='train'):
        if not mode == 'train':
            return self.test_loader
        else:
            return self.train_loader

    def get_pca_color(self, feat):
        feat_centered = feat - feat.mean(dim=-2, keepdim=True)
        u, s, v = torch.pca_lowrank(feat_centered, center=False, q=3)
        X_pca = feat_centered @ v[:, :3]
        # Normalize PCA values to range [0, 255] for color
        X_pca_min = X_pca.min(dim=0)[0]  # Min per component
        X_pca_max = X_pca.max(dim=0)[0]  # Max per component
        X_pca_scaled = (X_pca - X_pca_min) / (X_pca_max - X_pca_min)  # Normalize to [0,1]
        X_pca_uint8 = (X_pca_scaled * 255).to(torch.uint8).cpu()  # Scale to [0,255] and convert to uint8

        return X_pca_uint8.numpy()


    def visualize(self, positions, colors=None, name='random_obj'):
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().detach().numpy()

        if colors is None:
            colors = np.zeros_like(positions)

        point_size=5
        self.visualizer.add_points(name, positions, colors, point_size=point_size)
        self.visualizer.save('visualization')

    def register_get_feature_hook(self, keywords):
        extracted_features = []
        hooks = []
        hooked_layer_names = []

        def hook_fn(module, input, output):
            point = input[0]

            pad, unpad, cu_seqlens = module.get_padding_and_inverse(point)
            order = point.serialized_order[module.order_index][pad]
            inverse = unpad[point.serialized_inverse[module.order_index]]

            # padding and reshape feat and batch for serialized point patch
            coords = point.coord[order]
            qkv = module.qkv(point.feat)[order]
            H = module.num_heads
            K = module.patch_size
            C = module.channels

            ori_q, ori_k, ori_v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            coords = coords.reshape(-1, K, 3).unsqueeze(1).expand(-1,H, K, 3)


            B, H, T, C = ori_v.shape

            if (module.additional_info is not None) and \
                            module.additional_info["tome"] in VALID_TOME_MODES:

                # print("Initialize patch_tome")
                q, k, v, size, merge, unmerge = module.process_merging(ori_q, ori_k, ori_v, order, inverse)
                coords = merge(coords)


            attn = (q * module.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            ori_attn = (ori_q * module.scale) @ ori_k.transpose(-2, -1)  # (N', H, K, K)

            if (module.additional_info is not None) and \
                            module.additional_info["tome"] in VALID_TOME_MODES:
                attn = attn + size.log()

            attn = module.softmax(attn)
            ori_attn = module.softmax(ori_attn)
            attn = module.attn_drop(attn).to(qkv.dtype)
            ori_attn = module.attn_drop(ori_attn).to(qkv.dtype)
            feat = (attn @ v) # (N', H, K, C // H)
            ori_feat = (ori_attn @ ori_v) # (N', H, K, C // H)

            if (module.additional_info is not None) and \
                (module.additional_info["tome"] in VALID_TOME_MODES) and \
                    module.additional_info["tome_attention"]:

                feat = module.process_unreduction(feat, unmerge) # (N, H, K, C // H)
                merged_infos = []

                merged_colors = torch.rand(B, H, v.shape[2], 3).cuda()
                merged_colors = module.process_unreduction(merged_colors, unmerge)
                coords = module.process_unreduction(coords, unmerge)
#

                for i in range(H):
                    merged_c = merged_colors[:, i].reshape(-1, 3)
                    merged_c = merged_c[inverse]
                    merged_infos.append(merged_c)

                v = module.process_unreduction(v, unmerge)
            else:
                merged_infos = None

            attn_feats = []
            ori_attn_feats = []
            merged_coords =[]
            for i in range(H):
                attn_feat = feat[:, i].reshape(-1, C)[inverse]
                ori_attn_feat = ori_feat[:, i].reshape(-1, C)[inverse]
                coord = coords[:, i].reshape(-1, 3)[inverse]
                attn_feats.append(attn_feat)
                ori_attn_feats.append(ori_attn_feat)
                merged_coords.append(coord)

            feat = feat.transpose(1, 2).reshape(-1, C)
            ori_feat = ori_feat.transpose(1, 2).reshape(-1, C)

            segments_ids = torch.zeros((B, T)).cuda()
            segments_ids = segments_ids + torch.arange(B).cuda().unsqueeze(-1)
            segments_ids = segments_ids.reshape(-1, 1)

            feat = feat[inverse]
            ori_feat = ori_feat[inverse]
            segments_ids = segments_ids[inverse]
            # value = v.transpose(1,2).reshape(-1, C).clone().detach()[inverse]
            value = v[:,0,:,:].reshape(-1, C).clone().detach()[inverse]
            ori_value = ori_v[:,0,:,:].reshape(-1, C).clone().detach()[inverse]

            extracted_features.append(
                {
                    "coords": point.coord.clone().detach(),
                    "input_feat": point.feat.clone().detach(),
                    "value": value,
                    "ori_value": ori_value,
                    "attn_feats": attn_feats,
                    "ori_attn_feats": ori_attn_feats,
                    "segment_ids": segments_ids,
                    "merged_infos": merged_infos,
                    "merged_coords": merged_coords
                }
            )

        for name, module in self.model.named_modules():
            if any(name.endswith(keyword) for keyword in keywords):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                hooked_layer_names.append(name)

        return hooks, extracted_features, hooked_layer_names

    def remove_get_feature_hook(self, hooks):
        for hook in hooks:
            hook.remove()

class Visualizer():
    def __init__(self):
        self.visualizer = viz.Visualizer()

    def __call__(self, positions, colors=None, name='random_obj'):
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().detach().numpy()

        if colors is None:
            colors = np.zeros_like(positions)

        point_size= 10
        self.visualizer.add_points(name, positions, colors, point_size=point_size)
        self.visualizer.save('visualization')


def main(argv):
    dist.init_process_group('nccl')
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    count=0
    all_coords = defaultdict(list)
    all_diffs = defaultdict(list)

    for idx_algo, algo in enumerate(['base', 'progressive', 'patch', 'important_patch' ]):
        model = Model()
        model.additional_info['tome'] = algo
        model.additional_info['r'] = 0.90
        model.additional_info['stride'] = 10
        hooks, features, hooked_layer_names = model.register_get_feature_hook(['attn'])
        loaders = model.get_loader(mode='test')
        model = model.get_model().to(device)

        for name, loader in loaders.items():
            for item in loader:
                if count ==0:
                    with torch.no_grad():
                        test_batch_gs = gpu_utils.move_to_device([data['gs_params'] for data in item], device)
                        test_batch_idx = [data['scene_idx'] for data in item]
                        forward_kwargs = {'batch_normalized_gs': test_batch_gs, 'batch_scene_idx': test_batch_idx}
                        _ = model(**forward_kwargs)
                        for idx, layer in enumerate(hooked_layer_names):
                            if idx == len(hooked_layer_names) - 1:
                                coords = features[idx]['coords']
                                # input_feats = features[idx]['input_feat']
                                attn_feats = features[idx]['attn_feats']
                                ori_attn_feats = features[idx]['ori_attn_feats']
                                merged_coords = features[idx]['merged_coords'][0].detach().cpu().numpy()

                                # color_intput = model.get_pca_color(input_feats)
                                # color_merge = model.get_pca_color(features[idx]['merged_infos'][0])
                                # color_attn = model.get_pca_color(attn_feats[0])
                                # color_ori_attn = model.get_pca_color(ori_attn_feats[0])
                                # value = features[idx]['value'] = features[idx]['ori_value']
                                merged_coords[:,0] = merged_coords[:,0] + idx_algo * .75
                                merged_coords[:,1] = merged_coords[:,1] + idx * .75
                                coords[:,0] = coords[:,0] + idx_algo * 0.75
                                coords[:,1] = coords[:,1] + idx * 0.75

                                diff = torch.abs(attn_feats[0] - ori_attn_feats[0]).mean(-1)
                                all_diffs[f'{layer}'].append(diff)
                                all_coords[f'{layer}'].append(coords)
                break
            break
        del model
        gc.collect()
        torch.cuda.empty_cache()


    visualizer = Visualizer()
    for key in all_diffs.keys():
        diffs = torch.cat(all_diffs[key], dim=0)
        diff_normalized = (diffs - diffs.min())/ (diffs.max() - diffs.min() + 1e-5)
        cmap = plt.cm.get_cmap('RdYlGn')
        color_diff_attn = cmap(1 - diff_normalized.detach().cpu().numpy())[:, :3]
        color_diff_attn = (color_diff_attn * 255).astype(int).reshape(len(all_diffs[key]), -1, 3)
        for idx in range(len(all_diffs[key])):
            visualizer(
                positions=all_coords[key][idx],
                colors=color_diff_attn[idx],
                name=f'{key}_{idx}'
            )



    dist.destroy_process_group()

if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string(
    'gin_param', '', 'Newline separated list of Gin parameter bindings.')
    flags.DEFINE_boolean('only_eval', True, 'eval or train')
    app.run(main)
