import torch
import numpy as np
import pandas as pd

from src.utils.getter import get_instance
from src.utils.device import detach, move_to

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str,
                    help='path to weight files')
parser.add_argument('--gpus', type=int, default=None,
                    help='(single) GPU to use (default: None)')
parser.add_argument('--data_path', type=str, 
                    help='Test path')
args = parser.parse_args()

# Device
dev_id = 'cuda:{}'.format(args.gpus) \
    if torch.cuda.is_available() and args.gpus is not None \
    else 'cpu'
device = torch.device(dev_id)

# Load model
pretrained_cfg = torch.load(args.weight, map_location=dev_id)
model = get_instance(pretrained_cfg['config']['model']).to(device)
model.load_state_dict(pretrained_cfg['model_state_dict'])

with torch.no_grad():
    model.eval()
    print('Making submission file.......')
    X = pd.read_csv(args.data_path)
    X = X.to_numpy().reshape(-1, 1, 28, 28)
    X = X.repeat(3, 1)
    X = torch.FloatTensor(X)/255
    X = move_to(X, device)
    outs = model(X)
    _, preds = torch.max(outs, dim=1)
    preds = detach(preds)
    cols = ['ImageId', 'Label']
    sub = pd.DataFrame(columns=cols)
    sub['ImageId'] = np.arange(1, preds.size(0)+1)
    sub['Label'] = preds.numpy()
    sub.to_csv('submission.csv', index=False)
    print('Done.')