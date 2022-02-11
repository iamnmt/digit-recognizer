import os
import yaml
import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import seed_everything

from src.utils.random_seed import SEED
from src.utils.getter import get_instance

class Pipeline(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_instance(config['model'])
        self.loss = get_instance(config['loss'])

    def prepare_data(self):
        self.train_dataset = get_instance(self.config['dataset']['train'])
        self.val_dataset = get_instance(self.config['dataset']['val'])

    def train_dataloader(self):
        train_dataloader = get_instance(
            config=config['dataset']['train']['loader'],
            dataset=self.train_dataset,
            num_workers=4
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = get_instance(
            config=config['dataset']['val']['loader'],
            dataset=self.val_dataset,
            num_workers=4
        )
        return val_dataloader

    def forward(self, inp):
        return self.model(inp)

    def training_step(self, batch, batch_idx):
        inps, lbls = batch
        outs = self.forward(inps)
        loss = self.loss(outs, lbls)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inps, lbls = batch
        outs = self.forward(inps)
        loss = self.loss(outs, lbls)
        acc = (outs.argmax(-1) == lbls).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = acc = torch.tensor([], device=self.device)
        for o in outputs:
            loss = torch.cat([loss, o['loss'].unsqueeze(0)], dim=0)
            acc = torch.cat([acc, o['acc']], dim=0)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_instance(config['optimizer'], params=self.model.parameters())
        lr_scheduler = get_instance(config['scheduler'], optimizer=optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

def train(config, fast_dev_run=False):

    pipeline = Pipeline(config)

    cp_dir = config['save_dir']
    cp_dir = os.path.join(cp_dir, config['model']['name']) + str(
        config.get('id', 'None')
    )

    logger = pl_loggers.TensorBoardLogger(
        save_dir='runs',
        name=config['model']['name'],
        version=config.get('id', 'None')
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cp_dir,
        filename='{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        verbose=config['verbose']
    )

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        tpu_cores=(8 if config['use_tpu'] else None),
        gpus=config['gpus'],
        check_val_every_n_epoch=config['trainer']['val_step'],
        max_epochs=config['trainer']['nepochs'],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=logger,
        default_root_dir = 'runs',
    )
    trainer.fit(model=pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument("--use_tpu", action="store_true", default=False)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument("--save_dir", default="runs_lightning")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    seed_everything(seed=SEED)
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config["use_tpu"] = args.use_tpu
    config['gpus'] = args.gpus
    config["save_dir"] = args.save_dir
    config["verbose"] = args.verbose

    train(config)

