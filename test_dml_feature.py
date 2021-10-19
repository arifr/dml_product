from pytorch_metric_learning import losses, miners, trainers, testers
import numpy as np
import pandas as pd
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim
import logging
from torch.utils.data import Dataset
from PIL import Image
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import accuracy_calculator
import argparse
import models

#from cub2011 import Cub2011
#from mobilenet import mobilenet_v2 
#from efficientnet_pytorch import EfficientNet



logging.getLogger().setLevel(logging.INFO)



parser = argparse.ArgumentParser("Test CNN Feature")
# dataset
#parser.add_argument('-m', '--miner', type=str, default='semihard', choices=['all','semihard','easy_positif','batch_hard'])
#parser.add_argument('-e', '--embedding', type=int, default=64, choices=[64,128,256,512])
parser.add_argument('-f', '--model-fname', type=str, default='')


#parser.add_argument('-j', '--workers', default=4, type=int,
#                    help="number of data loading workers (default: 4)")

args = parser.parse_args()



FEATURES=()


def hook_fn(self,input,output):
    FEATURES = input    




#####################
### tambahan ########
#####################

class StanfordProducts(Dataset) :
    def __init__(self,root,transform,train=True):
        if train:
            info_path = "/Ebay_train.txt"
        else:
            info_path = "/Ebay_test.txt"
        files = pd.read_csv(root+info_path, header=0, delimiter=' ',usecols=['path','class_id'])[['path','class_id']]
        self.data = files.to_dict(orient='record')
        self.transform = transform
        self.root = root
    def __getitem__(self,index):
        image = Image.open(self.root +'/'+ self.data[index]['path'])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')

        trans = self.transform(image)
        return  trans, self.data[index]['class_id']-1
        
    def __len__(self):
        return len(self.data)




class CustomerToShop(Dataset) :

    def __init__(self,root,transform,train=True):

        files = pd.read_csv(root+'/Eval/list_eval_partition_new.txt', header=0, delimiter='\t',skiprows=1)[['image_path','item_id','evaluation_status']]

        ##image_name	item_id	evaluation_status

        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'test' " #or evaluation_status == 'val' "


        #print(files.to_dict(orient='records'))
        #print (files.to_dict(orient='record'))

        self.data = files.query(str_query).to_dict(orient='record')
        #self.image_path = image_path
        for dt in self.data :
            dt['item_id'] = int(dt['item_id'][3:].strip('0'))

        self.transform = transform
        #print(type(self.data['item_id']))
        #print(len(self.data))

        
        #def
    def __getitem__(self,index):
        image = Image.open(root + '/'+ self.data[index]['image_path'])
        #image.show()
        #print (self.data[index])
        if (image.mode != 'RGB'):
            image = image.convert('RGB')
        trans = self.transform(image)
        #image = trans(image)
        
        #print('from get: \n') 
        #print(type(itemid))
        
                
        return  trans, self.data[index]['item_id']

        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}

    def __len__(self):
        return len(self.data)


class InShop(Dataset) :

    def __init__(self,root,transform,train=True):


        files = pd.read_csv(root+'/Eval/list_eval_partition.txt', header=0, delimiter='\s+',skiprows=1)[['image_name','item_id','evaluation_status']]

    
        if train:
            str_query = "evaluation_status == 'train'"
        else:
            str_query = "evaluation_status == 'query'"


        #print(files.to_dict(orient='records'))
        

        self.data = files.query(str_query).to_dict(orient='record')
        for dt in self.data :
            dt['item_id'] = int(dt['item_id'][3:].strip('0'))
        self.transform = transform

        #def
    def __getitem__(self,index):
        image = Image.open(root + '/'+ self.data[index]['image_name'])
        #image.show()
        #print (self.data[index])
        #trans = transforms.ToTensor()
        #image = trans(image)
        return  self.transform(image), self.data[index]['item_id']

        #return  self.transform(image), self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}


    def __len__(self):
        return len(self.data)



class GetPrecisions(AccuracyCalculator):
    def calculate_precision_at_k(self, knn_labels, query_labels,**kwargs):
        precisions_at_k=[]
        for i in range(50):
            precisions_at_k.append(accuracy_calculator.precision_at_k_2(knn_labels, query_labels[:, None], i+1,False))
        return precisions_at_k

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_k"] 


##############################
########## Test ##########
##############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#trunk = torch.hub.load('pytorch/vision:v0.5.0','shufflenet_v2_x1_0',pretrained = True)
#trunk = EfficientNet.from_pretrained('efficientnet-b0')
# Set trunk model and replace the softmax layer with an identity function
#trunk = models.resnet50(pretrained=True)
#trunk = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
#trunk = mobilenet_v2(pretrained=True)
#print(trunk.last_channel)
#trunk = torch.load('online_product_trunk.pth')
#trunk_output_size = trunk.last_channel
#trunk_output_size = trunk._global_params.num_classes
#trunk_output_size = 1280
#trunk_output_size = trunk._stage_out_channels[-1]
#print (trunk_output_size)
#trunk.fc = Identity()
#trunk.fc = Identity()



#trunk = models.resnet18(pretrained=True)
#trunk_output_size = trunk.fc.in_features
#trunk.fc = common_functions.Identity()

#trunk = models.mobilenet_v2(pretrained=False)

#print("Model Name: ",args.model_fname)
model = torch.load(args.model_fname)
trunk = model.base_model

#print("Trunk :", trunk)

#trunk_output_size = trunk.last_channel
trunk.classifier[1] = common_functions.Identity()
trunk = trunk.to(device)

#trunk.load_state_dict(torch.load("Class_MobileNetV2_semihard_StanfordProducts_100_256_saved_models/trunk_best100.pth",map_location=device))



#trunk = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
#trunk.fc = Identity()
#trunk.fc = Normalize(trunk_output_size)
#trunk = torch.nn.DataParallel(trunk.to(device))



#trunk = trunk.to(device)



# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embedding#embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))
#print("output size:{0}".format(trunk_output_size))
#embedder = MLP([trunk_output_size, 1494, 128]).to(device)
#embedder = MLP([trunk_output_size, args.embedding]).to(device).half()

#embedder = MLP([trunk_output_size, args.embedding]).to(device)
#embedder = torch.load("Class_MobileNetV2_semihard_StanfordProducts_100_256_saved_models/embedder_best100.pth")
#embedder.load_state_dict(torch.load("Class_MobileNetV2_semihard_StanfordProducts_100_256_saved_models/embedder_best100.pth",map_location=device))



#embedder = MLP([trunk_output_size, args.embedding]).half()

#embedder = torch.nn.Linear(trunk_output_size,512).to(device)
#embedder = torch.load('online_product_embedder.pth')
#embedder = AutoEncoder(trunk_output_size,128).to(device)


# Set optimizers
#trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.0001, weight_decay=0.0005)
#embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0005)

# Set the image transform
'''
img_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=227),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
'''


img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

'''
img_transform_train = transforms.Compose([transforms.RandomResizedCrop(size=227),
                                    #transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


'''

img_transform_test = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



# Set the datasets
#train_dataset = datasets.CIFAR100(root="CIFAR100_Dataset", train=True, transform=img_transform, download=True)
#val_dataset = datasets.CIFAR100(root="CIFAR100_Dataset", train=False, transform=img_transform, download=True)

#print(train_dataset)
#print(type(train_dataset))

    
#train_dataset = getOnlineProducts(conf, train=True)
#val_dataset = getOnlineProducts(conf,train=False)
#
#

root = '/home/m405305/dataset/Stanford_Online_Products'
train_dataset = StanfordProducts(root,transform=img_transform_train,train=True)
val_dataset = StanfordProducts(root,transform=img_transform_test,train=False)




'''
root = '/home/m405305/Deep-Metric-Learning-Baselines/Datasets/online_products'
image_path = 'images'
train_dataset = StandfordProducts(root,image_path,transform=img_transform_train,train=True)
val_dataset = StandfordProducts(root,image_path,transform=img_transform_test,train=False)
'''

'''
root = '/home/m405305/dataset/inshop'
#image_path = 'images'
train_dataset = InShop(root,transform=img_transform_train,train=True)
val_dataset = InShop(root,transform=img_transform_test,train=False)
'''

'''
root = '/home/m405305/dataset'
train_dataset = Cub2011(root,transform=img_transform_train,train=True,download=False)
val_dataset = Cub2011(root,transform=img_transform_test,train=False,download=False)
'''

'''
root = '/home/m405305/dataset/cust-shop'
#image_path = 'images'
train_dataset = CustomerToShop(root,transform=img_transform_train,train=True)
val_dataset = CustomerToShop(root,transform=img_transform_test,train=False)
'''

#print (type(val_dataset.__getitem__(10)))



print(type(train_dataset).__name__)

#models = {"trunk": trunk, "embedder": embedder}
#optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
#loss_funcs = {"metric_loss": loss}
#mining_funcs = {"tuple_miner": miner}
dataset_dict = {"train": train_dataset, "val": val_dataset} 
#dataset_dict = {"val": val_dataset} 
num_epochs = 1


#log_model_name = "Test_class_{}_{}_{}_{}_{}".format(trunk.__class__.__name__, args.miner, train_dataset.__class__.__name__, num_epochs, args.embedding) 




#model_folder = log_model_name + "_saved_models"

#record_keeper, _, _ = logging_presets.get_record_keeper(log_model_name+"_logs", log_model_name+"_tensorboard")
#hooks = logging_presets.get_hook_container(record_keeper)

#acc_calc = AccuracyCalculator(include=("NMI", "precision_at_1", "r_precision","mean_average_precision_at_r","recall_at_1","recall_at_10","recall_at_20",
#"recall_at_30","recall_at_40", "recall_at_50"), k=51)
#acc_calc = AccuracyCalculator(exclude=('AMI',),k=1000)
#acc_calc = GetPrecisions(exclude=('AMI','NMI','precision_at_1', 'r_precision', 'mean_average_precision_at_r'),k=1000)
acc_calc = GetPrecisions(exclude=('AMI','NMI','r_precision', 'mean_average_precision_at_r'),k=1000)




#tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook, accuracy_calculator = acc_calc, normalize_embeddings=True) 
tester = testers.GlobalEmbeddingSpaceTester(accuracy_calculator = acc_calc, normalize_embeddings=True) 

#all_acuracies = tester.test(dataset_dict,num_epochs,trunk,embedder)
all_acuracies = tester.test(dataset_dict,num_epochs,trunk)


print(all_acuracies)

#end_of_epoch_hook = hooks.end_of_epoch_hook(tester,dataset_dict,model_folder, test_interval=10)





#torch.save(trainer.models['trunk'],'online_product_trunk.pth')
#torch.save(trainer.models['embedder'],'online_product_embedder.pth')


#############################
########## Testing ##########
############################# 

# The testing module requires faiss and scikit-learn
# So if you don't have these, then this import will break
#from pytorch_metric_learning import testers

#tester = testers.GlobalEmbeddingSpaceTester(reference_set="compared_to_sets_combined", record_keeper=record_keeper)
#tester = testers.GlobalEmbeddingSpaceTester(normalize_embeddings=True,record_keeper=record_keeper)

#epoch = 1

#tester.test(dataset_dict, epoch, trunk, embedder)

#if record_keeper is not None:
#    record_keeper.pickler_and_csver.save_records()
    
