import torch
import gin
from .GS import SplatfactoDataset

@gin.configurable
def GS_collate_fn(data_list):
    return data_list

@gin.configurable
def build_trainloader(dataset, batch_size, num_workers, collate_fn, accumulate_step):
    if dataset in ['shapenet','objaverse']:
        with gin.config_scope('train_dataset'):
            train_dataset = SplatfactoDataset()
    assert batch_size % torch.cuda.device_count() == 0, 'Batch size should be divisible by the number of GPUs'
    assert batch_size % accumulate_step == 0, 'Batch size should be divisible by the number of accumulate steps'
    batch_size_per_gpu = int(batch_size / (torch.cuda.device_count()*accumulate_step))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_gpu, num_workers=num_workers, 
                                             collate_fn=collate_fn)
    return dataloader

@gin.configurable
def build_testloader(dataset, batch_size, num_workers, collate_fn):
    if dataset in ['shapenet','objaverse']:
        test_nerfstudio_folder_list = gin.query_parameter('test_dataset/SplatfactoDataset.nerfstudio_folder_list')
        test_colmap_folder_list = gin.query_parameter('test_dataset/SplatfactoDataset.colmap_folder_list')

    assert type(test_nerfstudio_folder_list) == type(test_colmap_folder_list), 'test_nerfstudio_folder_list and test_colmap_folder_list should have the same type'
    if type(test_nerfstudio_folder_list) == str: #legacy
        test_folder_list_dict = {'default':{'nerfstudio_folder_list':test_nerfstudio_folder_list, 'colmap_folder_list':test_colmap_folder_list}}
    elif type(test_nerfstudio_folder_list) == dict:
        test_folder_list_dict = {}
        for key in test_nerfstudio_folder_list.keys():
            test_folder_list_dict[key] = {'nerfstudio_folder_list':test_nerfstudio_folder_list[key], 'colmap_folder_list':test_colmap_folder_list[key]}
    
    rt_dataloader = {}
    for test_dataset_name, nerfstudio_colmap_folder_list in test_folder_list_dict.items():
        if dataset in ['shapenet','objaverse']:
            with gin.config_scope('test_dataset'):
                test_dataset = SplatfactoDataset(**nerfstudio_colmap_folder_list)
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                                                collate_fn=collate_fn)
        rt_dataloader[test_dataset_name] = dataloader
    return rt_dataloader


