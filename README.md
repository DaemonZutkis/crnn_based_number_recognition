#Convolution recurrent neural network (CRNN)

Extended implementaion of CRNN network on Pytorch, for bigger image input size processing.

Original implementation: [CRNN](https://github.com/bgshih/crnn)  
PyTorch implementation of original crnn: [Pytorch CRNN](https://github.com/meijieru/crnn.pytorch)

## Model parameters

You are able to change training parameters by editing config file number_recognition.ini
 
path_to_save_results - the directories for model and metrics will be created automatically in this location 

model_type - this is name of model's current architecture  
model_name - this is name of model with type model_type, that have been trained at dataset dataset_name 

##Dataset structure  

Dataset structure is showed as the image below. 

<a href="https://habrastorage.org/webt/si/8t/y4/si8ty4lqvsjlkkkq5vak5x952l8.png">
<img src="https://habrastorage.org/webt/si/8t/y4/si8ty4lqvsjlkkkq5vak5x952l8.png" width=500></a>  

##labels file structure

Labels information is contained in labels files: train_labels_name, val_labels_name.  
Example of labels file with two images:   
{file_name_1.jpg": {"number": "959205766", "type_id": "type_id_1"},  
 "file_name_2.jpg": {"number": "5329147", "type_id": "type_id_m"}}