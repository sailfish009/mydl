import os
import pandas as pd
import cv2
from PIL import Image 
import numpy as np
import torch
from torch.utils.data import Dataset


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

    def __getitem__(self, index):
        img_names = [self.ids[index] + "_" + color + self.file_ext for color in ["red", "green", "blue", "yellow"]]
        # R = cv2.imread(os.path.join(self.root_folder, img_names[0]), cv2.IMREAD_GRAYSCALE)
        # G = cv2.imread(os.path.join(self.root_folder, img_names[1]), cv2.IMREAD_GRAYSCALE)
        # B = cv2.imread(os.path.join(self.root_folder, img_names[2]), cv2.IMREAD_GRAYSCALE)
        # Y = cv2.imread(os.path.join(self.root_folder, img_names[3]), cv2.IMREAD_GRAYSCALE)

        R = np.array(Image.open(os.path.join(self.root_folder, img_names[0])))
        G = np.array(Image.open(os.path.join(self.root_folder, img_names[1])))
        B = np.array(Image.open(os.path.join(self.root_folder, img_names[2])))
        Y = np.array(Image.open(os.path.join(self.root_folder, img_names[3])))
        try:
            if self.channels == 4:
                images = np.zeros(shape=(512,512,4))
                images[:,:,0] = R.astype(np.uint8) 
                images[:,:,1] = G.astype(np.uint8)
                images[:,:,2] = B.astype(np.uint8)
                images[:,:,3] = Y.astype(np.uint8)
                img = images.astype(np.uint8)

                # orig
                # # img = np.stack([R, G, B, Y], axis=-1)
                # img = np.stack([R, G, B, Y], axis=2)
                # img = self.transforms(img)
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
        except ValueError:
            print("error on %s" % self.ids[index])

    def __len__(self):
        return len(self.ids)
