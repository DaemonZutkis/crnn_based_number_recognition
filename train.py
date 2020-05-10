import os
import torch
from torch.autograd import Variable
from utils.parameters_loader import ParametersLoader
from utils.save import Save
from dataset_init.dataset_init import DatasetInit
from models.model_init import ModelIinit
import configparser
import time

class Train(ParametersLoader):
    def __init__(self, *args, **kwargs):
        super(Train, self).__init__(*args, **kwargs)
        self.config = config
        self.dataset_init = DatasetInit(self.config)
        self.model_init = ModelIinit(self.config)
        self.save = Save()

    def Train(self):
        self.dataloaders = self.dataset_init.dataloaders

        '''training process'''

        n_correct = 0

        accuracy_old = {"train": 0, "val": 0}

        for epoch in range(self.model_params["num_epochs"]):
            print("epoch:{}, time:{}".format(epoch, time.strftime("%X")))

            type_id_accuracy = {"train": {}, "val": {}}

            loss_average = {"train": 0, "val": 0}
            accuracy = {"train": 0, "val": 0}

            for phase in ["train", "val"]:

                type_id_accuracy[phase] = {}

                loss_average[phase] = 0
                accuracy[phase] = 0

                if phase == 'train':
                    '''Set model to training mode'''
                    self.model_init.model.train()
                else:
                    '''Set model to evaluate mode'''
                    self.model_init.model.eval()

                for next_data in self.dataloaders[phase]:

                    '''calculate gradients only over train phase'''
                    with torch.set_grad_enabled(phase == 'train'):
                        images = next_data[0]
                        labels = next_data[1]
                        length = next_data[2]
                        label_text = next_data[3]
                        cards_id = next_data[4]
                        types_id = next_data[5]

                        images = images.cuda()
                        length = length.cuda()
                        labels = labels.cuda()

                        preds = self.model_init.model(images)

                        preds_size = Variable(torch.IntTensor([preds.size(0)] *
                                                              self.model_params["batch_size"]))
                        length = torch.squeeze(length)

                        preds_size = preds_size.cuda()

                        probs, preds_copy = preds.max(2)

                        preds = preds.log_softmax(2)

                        cost = self.model_init.criterion(preds, labels, preds_size,
                                         length) / self.model_params["batch_size"] # length - real target length

                        loss_average[phase] = loss_average[phase] + cost

                        if phase == "train":
                            self.model_init.model.zero_grad()
                            cost.backward()
                            self.model_init.optimizer.step()

                        '''evaluate accuracy'''
                        preds_copy = preds_copy.to("cpu")
                        preds_copy = preds_copy.transpose(1, 0).contiguous().view(-1)
                        sim_preds = self.model_init.converter.decode(preds_copy.data, preds_size.data, raw=False)

                        for sample, (logit, label, card_id, type_id) in enumerate(
                                zip(sim_preds, label_text, cards_id, types_id)):
                            if logit == label:
                                n_correct += 1


                loss_average[phase] = loss_average[phase]/\
                                      (len(self.dataloaders[phase])*self.model_params["batch_size"])
                accuracy[phase] = n_correct/( len(self.dataloaders[phase])*self.model_params["batch_size"])


                if phase == "val":
                    if accuracy["val"] > accuracy_old["val"]:
                        print("saving_model")
                        torch.save(
                            self.model_init.model.state_dict(),
                            os.path.join(self.path_to_model, 'crnn_number.pth'))
                        accuracy_old["val"] = accuracy["val"]

                    '''save metrics of loss and accuracy'''
                    metrics = {}
                    for param_name, param in zip(
                            ["loss_average", "accuracy"],
                            [loss_average, accuracy]):

                        for phase_ in ["train", "val"]:
                            metrics[param_name + "_" + phase_] = [round(float(param[phase_]), 6)]

                    self.save.save_metrix(self.path_to_metrics, metrics)

                n_correct = 0


CONFIG_FILENAME = 'number_recognition.ini'
config = configparser.ConfigParser()
config.read(CONFIG_FILENAME)
train = Train(config=config)

'''start model training'''
train.Train()
