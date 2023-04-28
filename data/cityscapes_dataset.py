import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from pathlib import Path
import natsort
from natsort import natsorted


class cityscapes_Dataloader():
    def __init__(self,  exp_config):

        # Setup Dataset
        self.train_ds = cityscapes_Dataset(os.path.join(exp_config.data_folder, 'train')
                                           , exp_config.transform['train'], exp_config.use_bbpred
                                           )
        self.val_ds = cityscapes_Dataset(os.path.join(exp_config.data_folder, 'val'),
                                         exp_config.transform['val'], exp_config.use_bbpred)

        self.test_ds = cityscapes_Dataset(os.path.join(exp_config.data_folder, 'test'),
                                          exp_config.transform['test'], exp_config.use_bbpred)

        # Setup Datalaoder

        self.train = DataLoader(self.train_ds, shuffle=True, batch_size=exp_config.train_batch_size,
                                drop_last=True, pin_memory=True, num_workers=exp_config.num_w,
                                )

        self.validation = DataLoader(self.val_ds, shuffle=False, batch_size=exp_config.val_batch_size,
                                drop_last=True, pin_memory=True, num_workers=exp_config.num_w)

        self.test = DataLoader(self.test_ds, shuffle=False, batch_size=exp_config.test_batch_size,
                                drop_last=False, pin_memory=True, num_workers=exp_config.num_w)


class cityscapes_Dataset(Dataset):
    def __init__(self, data_file, transform = None, use_bbpred = True):
        self.img_file = natsorted((Path(data_file) / 'images').iterdir(), alg=natsort.PATH)
        self.label_file = natsorted((Path(data_file) / 'labels').iterdir(), alg=natsort.PATH)
        self.pred_file = natsorted((Path(data_file) / 'bbpreds').iterdir(), alg=natsort.PATH)
        self.prob_file = natsorted((Path(data_file) / 'probs').iterdir(), alg=natsort.PATH)
        self.transform = transform
        self.use_bbpred = use_bbpred

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, index: int):

        x = np.load(self.img_file[index])
        y = np.load(self.label_file[index])
        y_prob = np.load(self.prob_file[index])

        if self.use_bbpred:
            bbpred = np.load(self.pred_file[index])
            bbpred[y[0]==0] = 0 # ignored labels
            bbpred = np.expand_dims(bbpred, 0).astype(x.dtype)

            sample = {
                'image': x,
                'label': y,
                'bb_preds': bbpred
            }
            if self.transform is not None:
                sample = self.transform(sample)

            # Concat input and bbpred
            input = (np.concatenate([sample['image'], sample['bb_preds']], 0)).astype(np.float32)
        else:
            sample = {
                'image': x,
                'label': y,
            }
            if self.transform is not None:
                sample = self.transform(sample)

            input = sample['image']

        return input, sample['label'], y_prob,
