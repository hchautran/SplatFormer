import torch, os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import functools
import sys
from torch.optim import SGD
import wandb, json
import argparse
import cv2
from models.feature_predictor import FeaturePredictor

from collections import OrderedDict
import random
import gin 
from absl import app, flags
from dataset.Loader import build_trainloader, build_testloader
from utils import gpu_utils, gs_utils, loss_utils
from utils.optimizers import build_optimizer, build_scheduler
from utils.metrics import MetricComputer
from utils.log_utils import ProcessSafeLogger
from utils.metrics import psnr
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from collections import defaultdict
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    SerializedAttention,
    MLP,
    Point,
    Block
)
from fvcore.nn import FlopCountAnalysis
from train import training




def register_hooks(model):
    gflops =   defaultdict(lambda: 0)
    hooks = []
    hooked_layer_names = []
    

    def hook_fn(module, point: Point, output):
        point = point[0].copy()

        shortcut = point.feat
        point = module.cpe(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if module.pre_norm:
            point = module.norm1(point)

        copied_point = {
            'feat': point.feat,
            'offset': point.offset,
            'serialized_order': point.serialized_order,
            'serialized_inverse': point.serialized_inverse,
        }
        flops = FlopCountAnalysis(module.attn, copied_point)
        flops.unsupported_ops_warnings(
            False).uncalled_modules_warnings(False)
        attn_flops = flops.total() / 1e9
        print('flops attn: {:.4g} G'.format(attn_flops))
        gflops['attn flops'] += attn_flops
        
        point = module.drop_path(module.attn(point))
        point.feat = shortcut + point.feat
        if not module.pre_norm:
            point = module.norm1(point)

        shortcut = point.feat
        if module.pre_norm:
            point = module.norm2(point)

        flops = FlopCountAnalysis(module.mlp, point.feat)
        flops.unsupported_ops_warnings(
            False).uncalled_modules_warnings(False)
        mlp_flops = flops.total() / 1e9
        # print('flops mlp: {:.4g} G'.format(mlp_flops))
        gflops['mlp flops'] += mlp_flops 



    for name, module in model.named_modules():
        if isinstance(module, Block):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
            hooked_layer_names.append(name)

    return hooks, gflops, hooked_layer_names

def remove_get_feature_hook(self, hooks):
    for hook in hooks:
        hook.remove()


def measure_gpu_memory(func):
    """
    Decorator to measure the maximum GPU memory usage during a function call.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Call the wrapped function
        result = func(*args, **kwargs)
        
        # Get the peak memory usage
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        # print(f"Maximum GPU memory used during '{func.__name__}': {max_memory:.2f} MB")
        
        return result, max_memory
    return wrapper





@gin.configurable
def evaluation(model, test_loader, output_dir, output_gt, compare_with_pseudo, 
              compare_with_input=False,
              save_as_single=False,
              save_viewer=False,
              evaluate_input=False):
    model.eval()
    hooks, gflops, hook_layer_names = register_hooks(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
      progress_bar = tqdm(total=len(test_loader))
      for idx, test_batch in enumerate(test_loader): #A scene at a time
        test_batch_gs = gpu_utils.move_to_device([data['gs_params'] for data in test_batch], device)
        test_batch_idx = [data['scene_idx'] for data in test_batch]
        forward_kwargs = {'batch_normalized_gs': test_batch_gs, 'batch_scene_idx': test_batch_idx}
        _ = model(**forward_kwargs)
        print('attn flops', gflops['attn flops']/(idx + 1))
        progress_bar.update(1)





def log_result(metrics, test_dataset, metrics_input, merge_info=None, max_mem=0):
  if os.path.exists('eval.csv'):
    with open('eval.csv', 'a') as f:
      if merge_info is not None:
        algo = merge_info['tome']
        r = merge_info['r']
      else:
        algo = 'base' 
        r = '0.0'
      
      print(metrics)

      f.write(f'{test_dataset},{metrics["psnr"]},{metrics["ssim"]},{metrics["lpips"]},{algo},{r},{max_mem}\n')
  else:
    with open('eval.csv', 'w') as f:
      f.write('dataset,psnr,ssim,lpips,algo,r,max mem\n')
    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, FLAGS.eval_subdir, 'eval.log')).get_logger()
    metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
    logger.info(f'Test-{test_dataset}: {metric_str}')
    if FLAGS.compare_with_input:
      metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics_input.items()])
      logger.info(f'Input 3DGS: Test-{test_dataset}: {metric_str}')



def main(argv):

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    merge_info  = gin.query_parameter("PointTransformerV3Model.additional_info")
    print(merge_info)
    merge_info['r'] = FLAGS.merge_rate
      
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FeaturePredictor(additional_info=merge_info)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if model.resume_ckpt is not None:
      model.load_state_dict(torch.load(model.resume_ckpt,map_location='cpu'))
      print(f'Load model from {model.resume_ckpt}')

    model = model.to(device)
    model.eval()
    for test_dataset, test_loader in build_testloader().items():
        print(f'Evaluating {test_dataset}...')
        if len(test_loader) <10 :
          evaluation(
            model, 
            test_loader = test_loader, 
            output_dir=FLAGS.output_dir+f'/{FLAGS.eval_subdir}/{test_dataset}', 
            compare_with_input=FLAGS.compare_with_input,
            save_as_single=True,
            save_viewer=FLAGS.save_viewer,
            output_gt=True, 
            compare_with_pseudo=False
          )

      


if __name__ == '__main__':

  flags.DEFINE_string('output_dir', 'output', 'Output directory')
  flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
  flags.DEFINE_string('wandb_dir', '/cluster/scratch/chenyut/wandb', 'Wandbs Output directory')
  flags.DEFINE_float('merge_rate', 0.5, 'merge rate')
  flags.DEFINE_boolean('only_eval', False, 'eval or train')
  flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
  flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
  flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.')
  flags.DEFINE_multi_string(
    'gin_param', '', 'Newline separated list of Gin parameter bindings.')

  FLAGS = flags.FLAGS

  app.run(main)