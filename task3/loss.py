import torch.nn as nn
import utils


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        y_true = y_true.int()
        utils.produce_gif(y_pred[0].int().permute(1, 2, 0).detach().numpy(), f"ypred.gif")
        utils.produce_gif(y_true[0].permute(1, 2, 0).detach().numpy(), f"ytrue.gif")
        assert y_pred.size() == y_true.size()
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        # print(f"intersection: {intersection}")
        # print(f"divider: {(y_pred.sum() + y_true.sum() - intersection)}")
        dsc = (intersection) / (y_pred.sum() + y_true.sum() - intersection)
        return 1.0 - dsc


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1.0 - dsc
