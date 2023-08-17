"""
Trains a character-level language model.
"""

import os
import sys
from typing import BinaryIO

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT, C2GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    #C.model.model_type = 'gpt-mini'
    C.model.model_type = 'gpt2-medium'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class ByteStreamDataset(Dataset):
    """
    Emits batches of bytes.
    """
    def __init__(self, file_handle: BinaryIO, sequence_length: int):
        self.file_handle = file_handle
        self.sequence_length = sequence_length
        self.input_length = 0
        file_handle.seek(0, 2)  # Seek to EOF.
        self.input_length = file_handle.tell()
        assert file_handle.mode == 'rb'

    @staticmethod
    def get_vocab_size():
        return 256

    def get_block_size(self):
        return self.sequence_length

    def __len__(self):
        return self.input_length - self.get_block_size() - 1

    def __getitem__(self, idx):
        self.file_handle.seek(idx, 0)
        # Grab a chunk of (block_size + 1) characters from the data
        #chunk = self.data[idx:idx + self.config.block_size + 1]
        chunk = self.file_handle.read(self.get_block_size() + 1)
        # Chunk is already a byte array so we need to do no conversion.
        x = torch.tensor([c for c in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([c for c in chunk[1:]], dtype=torch.long)
        return x, y





# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[2:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    sequence_length = 1024

    # construct the training dataset
    tin = open(sys.argv[1], 'rb')
    train_dataset = ByteStreamDataset(tin, sequence_length=sequence_length)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = sequence_length // 4
    #model = GPT(config.model)
    model = C2GPT(
        input_sequence_length=sequence_length,
        ngram_embedding_size=512,
        ngram_length=4,
        num_layers=8,
        num_heads=8,
    )

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = b"In a surprising twist,"
                x = torch.tensor(context, dtype=torch.long)[None, ...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion_bytes = list()
                for i in y:
                    completion_bytes.append(i)
                completion = bytes(completion_bytes).decode('utf-8', errors='ignore')
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
