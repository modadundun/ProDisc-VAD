import argparse
parser = argparse.ArgumentParser(description='ProDisc_VAD')
parser.add_argument('--device', type=int, default=0, help='GPU ID')  
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0001)')  

parser.add_argument('--model_name', default='ProDisc_VAD', help=' ') 

parser.add_argument('--loss_type', default='MIL', type=str, help="")
parser.add_argument('--pretrain', type=int, default=0)  
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')

parser.add_argument('--feature_size', type=int, default=512, help='size of feature ')

parser.add_argument('--batch_size', type=int, default=1, help='number of samples in one iteration')  
parser.add_argument('--sample_size', type=int, default=30, help='number of samples in one iteration')  
parser.add_argument('--sample_step', type=int, default=1, help='')  
parser.add_argument('--dataset_name', type=str, default='ucf-crime', help='')  
parser.add_argument('--dataset_path', type=str, default='../dataset', help='path to dir contains anomaly datasets') 


parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750)')  
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')  
parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train (default: 50000)')  
parser.add_argument('--k', type=int, default=6, help='value of k')
parser.add_argument('--plot', type=int, default=0, help='whether plot the video anomalous map on testing')

parser.add_argument('--snapshot', type=int, default=1, help='anomaly sample threshold')
