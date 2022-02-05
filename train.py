import os
import yaml
import pprint
import argparse

from src.utils.random_seed import set_determinism
from src.utils.getter import get_data
from src.trainers import SupervisedTrainer


def train(config):
    assert config is not None, "Do not have config file!"

    pprint.PrettyPrinter(indent=2).pprint(config)

    # 1: Load datasets
    train_dataloader, val_dataloader = \
        get_data(config['dataset'], config['seed'])

    # 2: Create trainer
    trainer = SupervisedTrainer(config=config)

    # 3: Start trainining
    trainer.train(train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader)

def train_folds(config):
    assert config["dataset"]["num_folds"] != 0, "Num folds can not equal with zero"

    num_folds = config["dataset"]["num_folds"]
    folds_train_dir = config["dataset"]["folds_train_dir"]
    folds_val_dir = config["dataset"]["folds_val_dir"]

    folds_train_ls = [
        os.path.join(folds_train_dir, x) for x in os.listdir(folds_train_dir)
    ]
    folds_test_ls = [
        os.path.join(folds_val_dir, x) for x in os.listdir(folds_val_dir)
    ]
    folds_train_ls.sort(), folds_test_ls.sort()

    assert len(folds_train_ls) == len(folds_test_ls), "Folds are not match"
    id = str(config.get("id", "None"))

    for idx, paths in enumerate(
        zip(folds_train_ls[:num_folds], folds_test_ls[:num_folds])
    ):
        config["id"] = id + "/checkpoint_fold{}".format(idx)
        (
            config["dataset"]["train"]["args"]["csv_path"],
            config["dataset"]["val"]["args"]["csv_path"],
        ) = paths
        train(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus
    config['debug'] = args.debug

    set_determinism()
    if config['dataset']['num_folds'] is not None:
        train_folds(config)
    else:
        train(config)
