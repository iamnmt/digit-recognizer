import yaml
import torch
from tqdm import tqdm

from src.utils.getter import get_instance, get_single_data
from src.utils.device import move_to, detach

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', type=str,
                    help='config path')
parser.add_argument('-w', '--weight', type=str,
                    help='path to weight files')
parser.add_argument('-g', '--gpus', type=int, default=None,
                    help='(single) GPU to use (default: None)')
args = parser.parse_args()

config_path = args.config
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config['gpus'] = args.gpus

# Device
dev_id = 'cuda:{}'.format(args.gpus) \
    if torch.cuda.is_available() and args.gpus is not None \
    else 'cpu'
device = torch.device(dev_id)

# Load model
pretrained_cfg = torch.load(args.weight, map_location=dev_id)
model = get_instance(pretrained_cfg['config']['model']).to(device)
model.load_state_dict(pretrained_cfg['model_state_dict'])

# Load metric
metric = {
    mcfg['name']: get_instance(mcfg)
    for mcfg in config['metric']
}

# Load data
val_dataloader = get_single_data(config['dataset'], with_dataset=False)

with torch.no_grad():
    for m in metric.values():
        m.reset()

    model.eval()
    print('Evaluating........')
    progress_bar = tqdm(val_dataloader)
    for i, (inp, lbl) in enumerate(progress_bar):
        # Load inputs and labels
        inp = move_to(inp, device)
        lbl = move_to(lbl, device)

        # Get network outputs
        outs = model(inp)

        # Update metric
        outs = detach(outs)
        lbl = detach(lbl)
        for m in metric.values():
            m.update(outs, lbl)

    print('--- Evaluation result')
    for k in metric.keys():
        m = metric[k].value()
        metric[k].summary()
