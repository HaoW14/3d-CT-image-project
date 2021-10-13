import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.common import *
import config

args = config.args
device = torch.device(args.device)
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)
class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, target):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        #net_output = softmax_helper(net_output)
        # print('net_output shape: ', net_output.shape)
        bound = ndimage.distance_transform_edt(target.cpu())#得到distance map
        bound = np.trunc(bound) #取整
        bound = torch.from_numpy(bound).to(device)
        #bound = bound.astype(torch.float32)

        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()

        return bd_loss


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def forward(self, logits, targets):# logits:b * c * s * h * w     targets: b * s * h * w
        weight = torch.tensor([0.3,0.7]).to(args.device)
        loss1 = torch.nn.CrossEntropyLoss(weight=weight)(logits, targets)
        loss2 = DiceLoss()(logits, targets)

        #loss3 = BDLoss()(logits,targets)
        return loss1+loss2


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):# logits:b * c * s * h * w     targets: b * s * h * w

        loss = 1 - dice(logits, targets,1) #一个类别中，平均每个批次的dice
        return  loss


class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):

        input = flatten(net_output)
        target = flatten(gt)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1 - 2. * intersect / denominator.clamp(min=self.smooth)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, target_onehot):
        N = pred.shape[0]
        n_classes = pred.shape[1]
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()

class FocalLoss(nn.Module): #可能是对极不平衡的比较管用
    def __init__(self, gamma = 2, alpha = 0.8, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001 

    def forward(self, logits, targets):
        """
        logits: batch_size * labels_num * s*h*w
        labels: batch_size * labels_num * s*h*w (onehot)
        """


        pt = (targets * logits).sum(1).view(-1,1)
        log_p = pt.log()
        sub_pt = 1 - pt
        fl = -self.alpha * ((sub_pt)**self.gamma) * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

#返回标签为class_index的dice
def dice(logits, targets,classindex):# logits:b * c * s * h * w     targets: b  s * h * w
    logits = nn.Softmax(1)(logits)
    b = logits.size()[0]
    smooth = 1
    targets = torch.unsqueeze(targets,1)
    targets = torch.zeros(logits.shape).to(args.device).scatter_(1, targets, 1)
    input_flat = logits[:,classindex,:,:,:].view(b,-1) # contiguous使内存地址连续，才能用view
    targets_flat = targets[:,classindex,:,:,:].view(b,-1)

    intersection = input_flat * targets_flat
    union = input_flat.sum(1) + targets_flat.sum(1)
    total_batch_dice = (2. * intersection.sum(1) + smooth) / (union + smooth)
    mean_batch_dice = total_batch_dice.mean() # 一个类别中，平均每个批次的dice

    return mean_batch_dice

