from pymongo import MongoClient
import json
import ast
import pandas as pd
import os
import sys
from datetime import datetime

class ParametersLoader:
    def __init__(self, config):

        self.config = config
        self.general_params = self.return_parameters("general")
        self.load_save_params = self.return_parameters("load_and_save")
        self.model_params = self.return_parameters("model")

        '''add groups of sub parameters to model parameters'''
        list_sub_params = ["crnn_big_size", "cp2rnn", "adam", "adadelta", "rmsprop"]
        for sub_params in list_sub_params:
            try:
                self.model_params[sub_params] = self.return_parameters(sub_params)
            except:
                pass

        '''path to train dataset'''
        self.path_train_dataset = os.path.join(self.load_save_params["dataset_dir"],
                                               self.load_save_params["dataset_name"],
                                               "train")
        self.path_to_train_labels = os.path.join(self.load_save_params["dataset_dir"],
                                               self.load_save_params["train_labels_name"])

        '''path to val dataset'''
        self.path_val_dataset = os.path.join(self.load_save_params["dataset_dir"],
                                               self.load_save_params["dataset_name"],
                                               "val")
        self.path_to_val_labels = os.path.join(self.load_save_params["dataset_dir"],
                                               self.load_save_params["val_labels_name"])


        self.path_to_metrics = self.create_path(home_path=self.load_save_params["path_to_save_results"],
                                               dirs=["metrics",
                                                     self.model_params["model_name"],
                                                    self.load_save_params["dataset_name"]],
                                                file_name="metrics.csv"
                                                )
        self.path_to_model = self.create_path(home_path=self.load_save_params["path_to_save_results"],
                                               dirs=["models",
                                                     self.model_params["model_name"],
                                                    self.load_save_params["dataset_name"]]
                                                )

        self.nclass = len(self.model_params["alphabet"]) + 1


    def create_path(self, home_path, dirs, file_name=None):
        if os.path.isdir(home_path):
            for dir in dirs:
                home_path = os.path.join(home_path, dir)
                if not os.path.isdir(home_path):
                    os.mkdir(home_path)
                    os.chmod(home_path, 0o777)
        if file_name:
            home_path = os.path.join(home_path, file_name)

        return home_path


    def return_parameters(self, name):
        params = dict(self.config.items(name))
        keys = params.keys()
        for key in keys:
            try:
                params[key] = ast.literal_eval(params[key])
            except:
                pass
        return params