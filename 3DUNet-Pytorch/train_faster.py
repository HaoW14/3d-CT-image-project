from dataset import dataset_lits_faster
from torch.utils.data import DataLoader
import torch,os
import torch.optim as optim
import config
from models.Unet import UNet, ResBlock
from utils import logger, metrics,common
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

#torch.set_num_threads(2)  #用于防止cpu占用过高影响速度
def val(model, val_loader):
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = metrics.Myloss()(output, target)

            dice = metrics.dice(output, target,1)

            val_loss += float(loss)
            val_dice += float(dice)

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    return OrderedDict({'Val Loss': val_loss, 'Val dice': val_dice})

def train(model, train_loader):
    print("=======Epoch:{}=======".format(epoch))
    model.train()  #模型中如果有bn层或者dropout层，就要指定 model.train() 语句
    train_loss = 0
    train_dice0 = 0
    train_dice1 = 0
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)): #tqdm实现进度条
        data, target = data.float(),target.long()
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()  #将梯度归0

        loss = metrics.Myloss()(output, target)

        loss.backward()  #反向传播求梯度
        optimizer.step()  #更新参数

        train_loss += float(loss)
        train_dice0 += float(metrics.dice(output, target,0))
        train_dice1 += float(metrics.dice(output, target, 1))
    train_loss /= len(train_loader)
    train_dice0 /= len(train_loader)
    train_dice1 /= len(train_loader)

    return OrderedDict({ 'Train Loss': train_loss, 'Train dice0': train_dice0 , 'Train dice1': train_dice1})


if __name__ == '__main__':
    args = config.args
    device = torch.device(args.device)
    # data info
    train_set = dataset_lits_faster.Lits_DataSet(args.crop_size, args.dataset_path, mode='train')
    val_set = dataset_lits_faster.Lits_DataSet(args.crop_size, args.dataset_path, mode='val')

    train_loader = DataLoader(dataset=train_set, num_workers= 4, batch_size=args.batch_size)
    val_loader = DataLoader(dataset=val_set, num_workers= 4, batch_size=args.batch_size) #numworker可以加快速度，但是耗内存，一般设置为cpu的核心数

    # model info
    model = UNet(1, [16, 32, 48,64,128], 2,conv_block=None).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum = args.momentum)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0.001)  #余弦退火
    #init_util.print_network(model)
    #model = nn.DataParallel(model, device_ids=[0,1,2,3])   # multi-GPU

    log = logger.Logger('./output/{}'.format(args.save))  #输出用文件保存

    #checkpoint = torch.load(os.path.join('./output/{}'.format(args.save), 'best_model.pth'))  #用于加载上次的最优模型
    #model.load_state_dict(checkpoint['net'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    best = [0,np.inf] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    for epoch in range(1, args.epochs + 1):
        #common.adjust_learning_rate(optimizer, epoch, args)  #训练过程调节学习率
        train_log = train(model, train_loader)  #训练
        val_log = val(model, val_loader)       #验证    返回各种评价
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}  #保存3个参数
        torch.save(state, os.path.join('./output/{}'.format(args.save), 'latest_model.pth'))
        trigger += 1
        if val_log['Val Loss'] < best[1]:  #如果损失更小
            print('Saving best model')
            torch.save(state, os.path.join('./output/{}'.format(args.save), 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val Loss']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))
        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()