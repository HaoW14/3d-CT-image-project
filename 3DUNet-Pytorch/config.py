import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')

parser.add_argument('--device', default='cuda:0', help='set device')

parser.add_argument('--seed', type=int, default=1, help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path',default = './fixed_data',help='fixed trainset root path')

parser.add_argument('--save',default='model1',help='save path of trained model')

parser.add_argument('--resize_scale', type=list, default=[1,1,1],help='resize scale for input data')

parser.add_argument('--crop_size', type=list, default=[64, 128, 128],help='patch size of train samples after resize')

parser.add_argument('--batch_size', type=list, default=6,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=500, metavar='N',help='number of epochs to train')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate ')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum') #影响

parser.add_argument('--early-stop', default=100, type=int, help='early stopping')

args = parser.parse_args()