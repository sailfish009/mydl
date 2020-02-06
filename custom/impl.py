import os
import torch

from .. config import get_cfg
from .. engine import default_argument_parser, default_setup, launch
from .. utils.logger import setup_logger
from .. utils.collect_env import collect_env_info
from .. checkpoint import DetectionCheckpointer
from .  protein.ml_stratifiers import MultilabelStratifiedShuffleSplit

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def get_device(cfg):
    return torch.device(cfg.MODEL.DEVICE)

def get_cuda_count():
    return torch.cuda.device.count()

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_logger(cfg, save_path="", distributed_rank=0):
    logger = setup_logger("model", save_path, distributed_rank=distributed_rank)
    logger.info("Using {} GPUs".format(torch.cuda.device_count()))
    logger.info(cfg)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    return logger

def get_checkpointer(cfg, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=None, logger=None):
    checkpointer =  DetectronCheckpointer(cfg, model, optimizer, scheduler, save_dir, save_to_disk, logger)
    return checkpointer, checkpointer.load(cfg.MODEL.WEIGHT)
    # return checkpointer.load(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHT))

def mixed_precision(float_type, model, optimizer):
    use_mixed_precision = float_type == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    return amp.initialize(model, optimizer, opt_level=amp_opt_level)

def train_test_split(df, n_splits=4):
    df_backup = df.copy()
    X = df['Id'].tolist()
    y = df['target_vec'].tolist()
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=42)
    train_dfs, valid_dfs = [], []
    for train_index, valid_index in msss.split(X,y):
        train_df = df_backup.iloc[train_index]
        train_dfs.append(train_df[['Id', 'Target']])
        valid_df = df_backup.iloc[valid_index]
        valid_dfs.append(valid_df[['Id', 'Target']])
    return train_dfs, valid_dfs

def get_arguments(extra_checkpoint_data):
    arguments = {}
    arguments["iteration"] = 0
    arguments.update(extra_checkpoint_data)
    return arguments

def get_data_loder_val(cfg, distributed=False):
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    return data_loader_val, test_period, checkpoint_period

def run(main_func):
    args = default_argument_parser().parse_args()
    print('Command Line Args:', args)
    launch(
        main_func,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )



