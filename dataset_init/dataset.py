import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms, models
import sys
import os
import numpy as np
import pandas as pd
import time

from torch.utils.data import Dataset
import glob
import pandas as pd
from PIL import Image

import json

class DatasetCustom(Dataset):
    def __init__(self, dataset_dir, labels_dir, alphabet, size, amount_position):

        self.interpolation = Image.BILINEAR
        self.transfroms_train = transforms.ToTensor()

        self.dataset_dir = dataset_dir
        self.AMOUNT_POSITION = amount_position
        self.size = size
        with open(labels_dir, "r") as read_file:
            self.labels_file = json.load(read_file)
        self.numbers_list = list(self.labels_file.keys())
        # найдем общие изображения в dataset и json
        dataset_list = []
        for type_id in os.listdir(dataset_dir):

            for card_id in os.listdir( os.path.join(dataset_dir, type_id)):

                dataset_list.append(card_id)

        self.intersection_images = list(set(dataset_list).intersection(set(self.numbers_list )))

        # создание алфавита
        self.alphabet = {}
        for i, elem in enumerate(list(alphabet)):
            self.alphabet[elem] = i +1


    def __len__(self):
        return len(self.intersection_images )

    def convert_labes(self, label_text):
        list_labels_text = list(label_text)

        len_labels = torch.tensor([len(list_labels_text)], dtype=torch.long )
        list_labels = []
        for elem in list_labels_text:
            try:
                list_labels.append(self.alphabet[elem])
            except:
                pass
        image_labels = torch.tensor(list_labels, dtype = torch.long)

        # make image label with length AMOUNT_POSITION
        image_labels_wide = torch.zeros(self.AMOUNT_POSITION, dtype=torch.long)
        image_labels_wide[:image_labels.size(0)] = image_labels_wide[:image_labels.size(0)] + image_labels

        return len_labels, image_labels_wide



    def __getitem__(self, item):
        image_name = self.intersection_images[item]

        img = Image.open(os.path.join(self.dataset_dir, self.labels_file[image_name]["type_id"], image_name))
        img = img.resize(self.size, self.interpolation)



        img = self.transfroms_train(img)
        img.sub_(0.5).div_(0.5)

        label_text = self.labels_file[image_name]["number"]

        # delite space from label
        label_text = label_text.replace("-", "").replace(" ", "").replace(".", "").replace("_", "").replace("№", "")

        len_labels, image_labels = self.convert_labes(label_text)


        return img, image_labels, len_labels, label_text, image_name, self.labels_file[image_name]["type_id"]


