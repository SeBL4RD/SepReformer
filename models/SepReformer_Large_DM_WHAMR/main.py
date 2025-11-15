import os
import torch
import sys
from loguru import logger
from .dataset import get_dataloaders
from .model import Model
from .engine import Engine
from utils import util_system, util_implement
from utils.decorators import *

# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.remove()  # supprime tout handler existant
logger.add(log_file_path, level="INFO", mode="w")   # log file, mais pas les DEBUG
logger.add(sys.stderr, level="WARNING")             # console, warnings et erreurs seulement

class MainProcessor:
    @logger_wraps()
    @staticmethod
    def run(model_name, engine_mode, sample_file=None, out_wav_dir=None):
        ''' Build Setting '''
        # Call configuration file (configs.yaml)
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
        yaml_dict = util_system.parse_yaml(yaml_path)
        
        # Run wandb and get configuration
        config = yaml_dict["config"]
        
        # Create fake args object
        class Args:
            def __init__(self):
                self.model = model_name
                self.engine_mode = engine_mode
                self.sample_file = sample_file
                self.out_wav_dir = out_wav_dir
        
        args = Args()
        
        # Call DataLoader
        dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])
        
        ''' Build Model '''
        model = Model(**config["model"])
        
        ''' Build Engine '''
        gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
        device = torch.device(f'cuda:{gpuid[0]}')
        
        criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
        optimizers = util_implement.OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()
        schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()
        
        # Call & Run Engine
        engine = Engine(args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device)
        if engine_mode == 'infer_sample':
            engine._inference_sample(sample_file)
        else:
            engine.run()