import os
import yaml
import argparse
import numpy as np
import pandas as pd

import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import seed_everything

from src.utils.random_seed import SEED
from src.utils.getter import get_instance

import argparse

class Pipeline(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_instance(config['model'])
    
    def prepare_data(self):
        self.predict_dataset = get_instance(self.config['dataset'])

    def predict_dataloader(self):
        predict_dataloader = get_instance(
            config=config['dataset']['loader'],
            dataset=self.predict_dataset,
            num_workers=4
        )
        return predict_dataloader

    def forward(self, inp):
        return self.model(inp)

    def predict_step(self, batch, batch_idx):
        inps, ids = batch
        outs = self.forward(inps)
        preds = outs.argmax(-1)
        return (
            preds.int().cpu().detach().numpy(), 
            ids.int().cpu().detach().numpy()
        )


def infer(config, fast_dev_run=False):

    import pprint
    pprint.PrettyPrinter(indent=2).pprint(config)

    pipeline = Pipeline(config)
    
    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        gpus=(1 if config['use_gpu'] else None),
    )

    predictions = trainer.predict(
        model=pipeline, 
        ckpt_path=config['weight']
    )

    data = {
        'ImageId': np.array([], dtype=int),
        'Label': np.array([], dtype=int)
    }

    for preds, ids in predictions:
        data['Label'] = np.concatenate((data['Label'], preds), dtype=int)
        data['ImageId'] = np.concatenate((data['ImageId'], ids), dtype=int)

    sub = pd.DataFrame(data)
    sub.to_csv('submission.csv', index=False)
    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config')
    parser.add_argument('--use_gpu', action="store_true", default=False)
    parser.add_argument("--save_path", default="submission.csv")
    parser.add_argument("--weight", type=str)

    args = parser.parse_args()

    seed_everything(seed=SEED)
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['use_gpu'] = args.use_gpu
    config["save_path"] = args.save_path
    config["weight"] = args.weight

    infer(config)