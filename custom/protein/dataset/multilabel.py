import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


class ProteinDataset(Dataset):
    def __init__(self, cfg, label_file, root_folder, transforms, is_train, file_ext=".jpg"):
        df_train = pd.read_csv(label_file)
        self.ids = df_train["Id"].tolist()
        self.is_train = is_train
        self.file_ext = file_ext
        if is_train:
            raw_labels = df_train['Target'].tolist()
            self.labels = [list(map(int, item.split(' '))) for item in raw_labels]
        self.num_classes = cfg.MODEL.NUM_CLASS
        self.root_folder = root_folder
        self.transforms = transforms
        if cfg.MODEL.NAME in ['resnet343c', 'resnet18_3c']:
            self.channels = 3
        else:
            self.channels = 4

        self.width = cfg.INPUT.MAX_SIZE_TRAIN
        self.height = cfg.INPUT.MAX_SIZE_TRAIN

    def __getitem__(self, index):
        img_names = [self.ids[index] + "_" + color + self.file_ext for color in ["red", "green", "blue", "yellow"]]
        R = cv2.imread(os.path.join(self.root_folder, img_names[0]), cv2.IMREAD_GRAYSCALE)
        G = cv2.imread(os.path.join(self.root_folder, img_names[1]), cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(os.path.join(self.root_folder, img_names[2]), cv2.IMREAD_GRAYSCALE)
        Y = cv2.imread(os.path.join(self.root_folder, img_names[3]), cv2.IMREAD_GRAYSCALE)

        try:
            if self.channels == 4:
                R = cv2.resize(R, (self.width, self.height), interpolation=cv2.INTER_CUBIC )
                G = cv2.resize(G, (self.width, self.height), interpolation=cv2.INTER_CUBIC )
                B = cv2.resize(B, (self.width, self.height), interpolation=cv2.INTER_CUBIC )
                Y = cv2.resize(Y, (self.width, self.height), interpolation=cv2.INTER_CUBIC )
                img = np.stack([R, G, B, Y], axis=-1)
                # img = np.stack([R, G, B, Y], axis=2)
                img = self.transforms(img)
            elif self.channels == 3:
                img = np.stack([R, G, B], axis=-1)
                img = self.transforms(img)
            else:
                raise KeyError("channel number not supported")

            label_vec = torch.zeros(self.num_classes)
            if self.is_train:
                labels = self.labels[index]
                label_vec[labels] = 1
            return img, label_vec, index
        # except ValueError:
        except:
            print("error on %s" % self.ids[index])

    def __len__(self):
        return len(self.ids)
