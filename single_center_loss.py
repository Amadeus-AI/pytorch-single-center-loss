import torch
import torch.nn as nn

class SingleCenterLoss(nn.Module):
    """
    Single Center Loss
    
    Reference:
    J Li, Frequency-aware Discriminative Feature Learning Supervised by Single-CenterLoss for Face Forgery Detection, CVPR 2021.
    
    Parameters:
        m (float): margin parameter. 
        D (int): feature dimension.
        C (vector): learnable center.
    """
    def __init__(self, m = 0.3, D = 1000, use_gpu=True):
        super(SingleCenterLoss, self).__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.C = nn.Parameter(torch.randn(self.D).cuda())
        else:
            self.C = nn.Parameter(torch.randn(self.D))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True) + torch.pow(self.C, 2).sum().expand(batch_size, 1)

        labels = labels.unsqueeze(1)

        real_count = labels.sum()

        dist_mat = torch.sqrt(dist_mat)
        dist_real = (dist_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()
        dist_fake = (dist_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()

        if real_count != 0:
            dist_real /= real_count

        if real_count != batch_size:
            dist_fake /= (batch_size - real_count)

        max_margin = dist_real - dist_fake + self.margin

        if max_margin < 0:
            max_margin = 0

        loss = dist_real + max_margin

        return loss