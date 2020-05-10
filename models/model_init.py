from torch.nn import CTCLoss
import torch
import torch.optim as optim
import crnn_utils
from utils.parameters_loader import ParametersLoader
from models import crnn_big_size
import os

class ModelIinit(ParametersLoader):
    def __init__(self, *args, **kwargs):
        super(ModelIinit, self).__init__(*args, **kwargs)

        if self.model_params["model_type"] == "crnn_big_size":

            self.model = crnn_big_size.CRNN(nc=self.model_params["num_input_channels"],
                                     nclass=self.nclass,
                                     nh=self.model_params["hid_layer_size"])

        self.converter = crnn_utils.strLabelConverter(self.model_params["alphabet"])
        self.criterion = CTCLoss(zero_infinity=True).to(self.general_params["device"])
        self.model.apply(self.weights_init)

        '''load pretrained weigths'''
        path_to_pretrained_model = self.model_params[self.model_params["model_type"]]["path_pretrained"]

        if path_to_pretrained_model and os.path.isfile(path_to_pretrained_model):
            print('loading pretrained model')
            self.model.load_state_dict(path_to_pretrained_model)

        self.model.to(self.general_params["device"])
        self.model = torch.nn.DataParallel(self.model,
                                           device_ids=range(self.general_params["num_gpu"]))

        '''optimizer initialise'''
        if self.model_params["optimizer"] == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.model_params["adam"]["lr"],
                betas=(self.model_params["adam"]["lr"],
                0.999))

        elif self.model_params["optimizer"] == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.model_params["adam"]["lr"])
        else:
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.model_params["adam"]["lr"])


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def normalise_softmax(self, data):
        data_min, _ = data.min(2)
        data_min = torch.unsqueeze(data_min, 2)
        data_norm = data - data_min
        soft = torch.nn.Softmax(dim=2)
        data_soft = soft(data_norm)
        return data_soft