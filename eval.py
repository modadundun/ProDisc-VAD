import pickle 
import os  
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix  
from utils import scorebinary, anomap  
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve


def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):

    global label_dict_path 
    if manual: 
        save_root = './manul_test_result'  
    else:  
        save_root = './result' 

    if dataset == 'shanghaitech':  
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)  
        with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f: 
            frame_label_dict = pickle.load(f) 
        with open(file=os.path.join(label_dict_path, 'video_label_10crop.pickle'),
                  mode='rb') as f: 
            video_label_dict = pickle.load(f) 
        all_predict_np = np.zeros(0)
        all_label_np = np.zeros(0)  
        for k, v in predict_dict.items(): 
            base_video_name = k[:-2]
            if video_label_dict[k] == '[1.0]': 
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  
            elif video_label_dict[k] == '[0.0]': 
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))])) 
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)  
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score)) 
        if plot: 
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5) 
        if os.path.exists(os.path.join(save_root, save_path)) == 0: 
            os.makedirs(os.path.join(save_root, save_path))  
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:  
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score)) 




    if dataset == 'ucf-crime':  
        label_dict_path = '{}/ucf-crime/GT'.format(args.dataset_path)  
        with open(file=os.path.join(label_dict_path, 'ucf_gt_upgate.pickle'), mode='rb') as f:  
            frame_label_dict = pickle.load(f)  
        with open(file=os.path.join(label_dict_path, 'video_label_10crop.pickle'),
                  mode='rb') as f: 
            video_label_dict = pickle.load(f) 
        all_predict_np = np.zeros(0)  
        all_label_np = np.zeros(0) 
        for k, v in predict_dict.items(): 
            base_video_name = k[:-2]
            if video_label_dict[k] == '[1.0]': 
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16))) 
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  
            elif video_label_dict[k] == '[0.0]':  
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np) 
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score))  
        if plot:  
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5) 
        if os.path.exists(os.path.join(save_root, save_path)) == 0: 
            os.makedirs(os.path.join(save_root, save_path))  
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f: 
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score))  


