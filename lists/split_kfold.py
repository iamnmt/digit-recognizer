import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils.random_seed import SEED

SAVE_DIR = './lists/folds/'
DATA_PATH = './data/train.csv'
NUM_FOLDS = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', default=NUM_FOLDS)
    parser.add_argument('--seed', default=SEED)

    args = parser.parse_args()

    num_folds = int(args.num_folds)
    seed = int(args.seed)

    df = pd.read_csv(DATA_PATH)
    kf = StratifiedKFold(num_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df.label.values)):
        df.loc[train_idx, :].to_csv(f'{SAVE_DIR}train/train_{fold}.csv', index=False)
        df.loc[val_idx, :].to_csv(f'{SAVE_DIR}val/val_{fold}.csv', index=False)
