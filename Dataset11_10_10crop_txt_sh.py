import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch


class dataset(Dataset):
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        self.args = args  
        self.dataset_path = args.dataset_path  
        self.dataset_name = args.dataset_name  


        self.feature_path = r"your path"
    
        self.trainlist = self.txt2list(
            txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split_10crop.txt')) 
    
        self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split_10crop.txt'))  

        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label_10crop.pickle'))  



        self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict,
                                                                                   self.trainlist) 

        self.train = train 
        self.t_max = args.max_seqlen  
    def txt2list(self, txtpath=''):
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()  
        return filelist 

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)  
        return video_label_dict  

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            video_name = t.replace('\n', '').replace('Ped', 'ped')  
            if video_label_dict[video_name] == '[1.0]': 
                anomaly_video_train.append(video_name)
            else:
                normal_video_train.append(video_name)

        return normal_video_train, anomaly_video_train

    def __getitem__(self, index):
        if self.train:
            normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)  
            anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size) 

            anomaly_features = torch.zeros(0)  
            normaly_features = torch.zeros(0)  


            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                anomaly_data_video_name = a_i.replace('\n', '')
                normaly_data_video_name = n_i.replace('\n', '')  


                anomaly_feature = np.load(file=os.path.join(self.feature_path, anomaly_data_video_name + '.npy'))  
                anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max) 
                anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)  

                normaly_feature = np.load(file=os.path.join(self.feature_path, normaly_data_video_name + '.npy'))  
                normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)  
                normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)  

                anomaly_features = torch.cat((anomaly_features, anomaly_feature), dim=0) 
                normaly_features = torch.cat((normaly_features, normaly_feature), dim=0) 

            normaly_label = torch.zeros((self.args.sample_size, 1)) 
            anomaly_label = torch.ones((self.args.sample_size, 1))  

            anomaly_features = anomaly_features.float()
            normaly_features = normaly_features.float()

            return [anomaly_features, normaly_features], [anomaly_label, normaly_label]
        else:
            data_video_name = self.testlist[index].replace('\n', '') 
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name + '.npy'))  
            self.feature = self.feature.astype(np.float32)

            return self.feature, data_video_name  

    def __len__(self):
        if self.train:
            return len(self.trainlist)  
        else:
            return len(self.testlist)  

if __name__ == "__main__":
    args = options.parser.parse_args() 
    train_dataset = dataset(args=args, train=True)  

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=0, shuffle=True)  
    test_dataset = dataset(args=args, train=False)  
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=0, shuffle=False)  