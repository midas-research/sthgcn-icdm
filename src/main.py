import os, time, argparse
import numpy as np
import pandas as pd
from logger import set_logger
from config import get_args
from dataset import StockDataset
from trainer import Trainer
from evaluator import Evaluator

def main():
    config = get_args()
    logger = set_logger(config)
    dataset = StockDataset(config)
    trainer = Trainer(dataset, config)
    trainer.train()
    exit()

if __name__ == '__main__':
    main()
