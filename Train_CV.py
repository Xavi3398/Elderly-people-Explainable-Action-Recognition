import torch, torchvision
import mmaction
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmaction.apis import inference_recognizer, init_recognizer

from mmcv import Config
from mmcv.runner import set_random_seed

import os.path as osp

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv

models = dict(timesformer_divST = 'timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth',
            timesformer_spaceOnly = 'timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb-0cf829cd.pth')

configs = dict(timesformer_divST = '/configs/recognition/timesformer/timesformer_divST_8x32x1 - modified pipeline.py',
            timesformer_spaceOnly = '/configs/recognition/timesformer/timesformer_spaceOnly_8x32x1 - modified pipeline.py')

mmaction_dir = 'E:/OneDrive - Universitat de les Illes Balears/2021-2022/TFM/mmaction2'
dataset_dir = 'F:/ETRI-Activity3D/RGB Videos/EtriActivity3D'
dataset_name = 'EtriActivity3D'
pre_trained_model = models['timesformer_divST']
config_file = configs['timesformer_divST']
logs_dir = 'E:/OneDrive - Universitat de les Illes Balears/2021-2022/TFM/training_logs/CV_TRAINING'
n_classes = 55
K_CV_start = 1
K_CV_end = 5

def train_cv(k):

    cfg = Config.fromfile(mmaction_dir + config_file)

    # Modify dataset type and path
    cfg.dataset_type = 'VideoDataset'
    cfg.data_root = dataset_dir + '/all_videos/'
    cfg.data_root_val =  dataset_dir + '/all_videos/'
    cfg.data_root_test =  dataset_dir + '/all_videos/'
    cfg.ann_file_train =  dataset_dir + '/' + dataset_name + '_train_k' + str(k) + '_video.txt'
    cfg.ann_file_val =  dataset_dir + '/' + dataset_name + '_test_k' + str(k) + '_video.txt'
    cfg.ann_file_test =  dataset_dir + '/' + dataset_name + '_test_k' + str(k) + '_video.txt'

    cfg.data.train.type = 'VideoDataset'
    cfg.data.train.ann_file = dataset_dir + '/' + dataset_name + '_train_k' + str(k) + '_video.txt'
    cfg.data.train.data_prefix = dataset_dir + '/all_videos/'

    cfg.data.val.type = 'VideoDataset'
    cfg.data.val.ann_file =dataset_dir + '/' + dataset_name + '_test_k' + str(k) + '_video.txt'
    cfg.data.val.data_prefix = dataset_dir + '/all_videos/'

    cfg.data.test.type = 'VideoDataset'
    cfg.data.test.ann_file = dataset_dir + '/' + dataset_name + '_test_k' + str(k) + '_video.txt'
    cfg.data.test.data_prefix = dataset_dir + '/all_videos/'

    cfg.model.cls_head.num_classes = n_classes # Modify num classes of the model in cls_head
    
    cfg.seed = 777 # Set seed thus the results are more reproducible
    set_random_seed(777, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.setdefault('omnisource', False) # The flag is used to determine whether it is omnisource training
    
    cfg.log_config.interval = 5
    cfg.checkpoint_config.interval = 5
    cfg.data.videos_per_gpu = 10
    cfg.optimizer.lr = cfg.optimizer.lr / 8 # We divide it by 8 since we only use one GPU.
    cfg.total_epochs = 10
    cfg.work_dir = os.path.join(logs_dir, 'K'+str(k)) # Set up working dir to save files and logs.
    cfg.load_from = mmaction_dir + '/checkpoints/' + pre_trained_model # Load checkpoint
    
    # Save the best
    cfg.evaluation.save_best='auto'

    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')


    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the recognizer
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_model(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    for i in range(K_CV_start, K_CV_end + 1):
        print("----------------------STARTING CV_"+str(i)+"-----------------------")
        train_cv(i+1)