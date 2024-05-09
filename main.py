import torch
import torch.nn as nn
import random
import numpy as np
import os

from train import train_fn
from test import test_fn
from utils.build_loaders import build_loaders
from utils.build_model import build_model
from configs.settings import get_settings
from models.hungary import HungarianMatcher

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    print("start")
    cfgs = get_settings()##获得配置参数

    set_random_seed(seed=cfgs.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device)
    #加载数据
    train_loader, valid_loader, test_loader = build_loaders(cfgs)
    print("building model")
    hungary_matcher = HungarianMatcher()#匈牙利匹配 计算目标和预测之间的匹配
    print("building model")
    model = build_model(cfgs)
    model = model.float()
    model.to(device)
    model = train_fn(cfgs, cfgs.model_name, model, hungary_matcher, train_loader, test_loader,valid_loader, device)
    model.load_state_dict(torch.load(cfgs.train.save_checkpoints_path))
    model.eval()
    test_fn(cfgs, model, test_loader, device)

