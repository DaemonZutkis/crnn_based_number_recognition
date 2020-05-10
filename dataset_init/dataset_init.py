import torch

from utils.parameters_loader import ParametersLoader
from dataset_init.dataset import DatasetCustom


class DatasetInit(ParametersLoader):
    def __init__(self, *args, **kwargs):
        super(DatasetInit, self).__init__(*args, **kwargs)

        self.train_dataset = DatasetCustom(
            dataset_dir=self.path_train_dataset,
            labels_dir=self.path_to_train_labels,
            alphabet=self.model_params["alphabet"],
            size=self.model_params[
                self.model_params["model_type"]]["image_size"],
            amount_position=self.model_params[self.model_params["model_type"]]["amount_position"]
        )

        self.val_dataset = DatasetCustom(dataset_dir=self.path_val_dataset,
                                   labels_dir=self.path_to_val_labels,
                                   alphabet=self.model_params["alphabet"],
                                   size=self.model_params[
                                       self.model_params["model_type"]]["image_size"],
                                   amount_position=self.model_params[
                                       self.model_params["model_type"]]["amount_position"])


        self.dataloaders = {}

        self.dataloaders["train"] = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.model_params["batch_size"],
                                                                shuffle=True,
                                                                num_workers=4,
                                                                drop_last=True)

        self.dataloaders["val"] = torch.utils.data.DataLoader(self.val_dataset,
                                                              batch_size=self.model_params["batch_size"],
                                                              shuffle=True,
                                                              num_workers=4,
                                                              drop_last=True)
