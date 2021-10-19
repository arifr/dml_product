from pytorch_metric_learning import losses, miners, trainers
import numpy as np
import pandas as pd
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim
import logging
from torch.utils.data import Dataset
from PIL import Image
from cub2011 import Cub2011
from mobilenet import mobilenet_v2 


logging.getLogger().setLevel(logging.INFO)




# This is a basic multilayer perceptron
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker



class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x




class Normalize(nn.Module):
    def __init__(self,num_feat):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(num_feat)
    def forward(self,x):
        #orm = nn.BatchNorm1d(self.num_feat)
        return self.bn1(x)




#####################
### tambahan ########
#####################


class StandfordProducts(Dataset) :
    def __init__(self,root,image_path,transform,train=True):
        if train:
            info_path = '/Info_Files/Ebay_train.txt'
        else:
            info_path = '/Info_Files/Ebay_test.txt'
        files = pd.read_csv(root+info_path, header=0, delimiter=' ',usecols=['path','class_id'])[['path','class_id']]
        #print(files.to_dict(orient='records'))
        self.data = files.to_dict(orient='record')
        self.image_path = image_path
        self.transform = transform
        #print(type(self.data[1]['class_id']))
        #def
    def __getitem__(self,index):
        image = Image.open(root + '/'+ self.image_path + '/' + self.data[index]['path'])
        #print ('{0}=>{1},{2}'.format(self.data[index]['path'],image.size,image.mode))
        #print ('{0}=>{1}'.format(self.data[index]['path'],image.size))
        if (image.mode != 'RGB'):
            #print ('{0}=>{1}'.format(self.data[index]['path'],image.mode))
            image = image.convert('RGB')

        trans = self.transform(image)
        #image = trans(image)
        #print (trans.size())
        #print('from get: \n') 
        #print(type(self.data[index]) )
        return  trans, self.data[index]['class_id']
        #{'image':im, 'target':self.data[index]['class_id']}

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





#class DatasetConfig:
#    source_path=''
#    image_path=''
#
#
#def getOnlineProducts(conf, train=True) :
#    #read text flie
#    if train :
#        #files = pd.read_table(conf.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ',usecols=['path','class_id'])
#        files = pd.read_csv(conf.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ',usecols=['path','class_id'])[['path','class_id']]
#    
#    else: 
#        files = pd.read_table(conf.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ', usecols=['path','class_id'])
#    #print("training files :\n {0}".format(training_set['path'][0]))
#    #print("test files :\n {0}".format(test_files))
##    with open(conf.source_path+'/Info_Files/Ebay_train.txt',newline='') as csvfile:
##        training_set = csv.DictReader(csvfile)
##        for row in training_set:
##            print("training dict :\n {0}".format(row))
##        
#    #training_set = training_files['path']['class_id'][:]
#    return files.values.tolist()




# record_keeper is a useful package for logging data during training and testing
# You can use the trainers and testers without record_keeper.
# But if you'd like to install it, then do pip install record_keeper
# See more info about it here https://github.com/KevinMusgrave/record_keeper
try:
    import os
    import errno
    import record_keeper as record_keeper_package
    from torch.utils.tensorboard import SummaryWriter

    def makedir_if_not_there(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    pkl_folder = "dml_dist_margin_logs"
    tensorboard_folder = "dml_dist_margin_tensorboard"
    makedir_if_not_there(pkl_folder)
    makedir_if_not_there(tensorboard_folder)
    pickler_and_csver = record_keeper_package.PicklerAndCSVer(pkl_folder)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_folder)
    record_keeper = record_keeper_package.RecordKeeper(tensorboard_writer, pickler_and_csver, ["record_these", "learnable_param_names"])

except ModuleNotFoundError:
    record_keeper = None


##############################
########## Training ##########
##############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(type(device))

# Set trunk model and replace the softmax layer with an identity function
#trunk = models.resnet50(pretrained=True)
#trunk = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
trunk = mobilenet_v2(pretrained=True)
#print(trunk.last_channel)
#trunk = torch.load('online_product_trunk.pth')
trunk_output_size = trunk.last_channel
#trunk.fc = Identity()


#trunk = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
#trunk_output_size = trunk.fc.in_features
#trunk.fc = Identity()
#trunk.fc = Normalize(trunk_output_size)
#trunk = torch.nn.DataParallel(trunk.to(device))
trunk = trunk.to(device)



# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
#embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))
embedder = MLP([trunk_output_size, 512]).to(device)
#embedder = torch.nn.Linear(trunk_output_size,512).to(device)
#embedder = torch.load('online_product_embedder.pth')


# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.00001, weight_decay=0.00005)

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


root = '/home/m405305/Deep-Metric-Learning-Baselines/Datasets/online_products'
image_path = 'images'
train_dataset = StandfordProducts(root,image_path,transform=img_transform_train,train=True)
val_dataset = StandfordProducts(root,image_path,transform=img_transform_test,train=False)


#root = '/home/m405305/dataset'
#train_dataset = Cub2011(root,transform=img_transform_train,train=True,download=False)
#val_dataset = Cub2011(root,transform=img_transform_test,train=False,download=False)

'''
root = '/home/m405305/Deep-Metric-Learning-Baselines/Datasets/cust-shop'
image_path = 'images'
train_dataset = CustomerToShop(root,transform=img_transform_train,train=True)
val_dataset = CustomerToShop(root,transform=img_transform_test,train=False)
'''

#print (type(val_dataset.__getitem__(10)))




# Set the loss function
loss = losses.TripletMarginLoss(margin=0.01)
#loss = losses.MarginLoss(margin=0.01,nu=1.2,beta=0)
#loss = losses.ContrastiveLoss()

# Set the mining function
#miner = miners.MultiSimilarityMiner(epsilon=0.1)
#miner = miners.DistanceWeightedMiner(cutoff=0, nonzero_loss_cutoff=0.5)
miner = miners.TripletMarginMiner(margin=0.01,type_of_triplets='semihard')





# Set other training parameters
batch_size = 40
num_epochs =  1
iterations_per_epoch = 10

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"post_gradient_miner": miner}

trainer = trainers.MetricLossOnly(models,
                                optimizers,
                                batch_size,
                                loss_funcs,
                                mining_funcs,
                                iterations_per_epoch,
                                train_dataset,
                                record_keeper=record_keeper)

trainer.train(num_epochs=num_epochs)

#torch.save(trainer.models['trunk'],'online_product_trunk.pth')
#torch.save(trainer.models['embedder'],'online_product_embedder.pth')


#############################
########## Testing ##########
############################# 

# The testing module requires faiss and scikit-learn
# So if you don't have these, then this import will break
from pytorch_metric_learning import testers

#tester = testers.GlobalEmbeddingSpaceTester(reference_set="compared_to_sets_combined", record_keeper=record_keeper)
tester = testers.GlobalEmbeddingSpaceTester(record_keeper=record_keeper)
dataset_dict = {"train": train_dataset, "val": val_dataset}
epoch = 1

tester.test(dataset_dict, epoch, trunk, embedder)

if record_keeper is not None:
    record_keeper.pickler_and_csver.save_records()
    