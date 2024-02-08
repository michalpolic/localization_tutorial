# written by Seongwon Lee (won4113@yonsei.ac.kr)
import torch.nn as nn

import third_party.SENet.core.net as net
from third_party.SENet.model.resnet import ResStemIN, ResStage, GlobalHead
from third_party.SENet.model.self_similarity import SSM


class SENet(nn.Module):
    """ResNet with Self-Similairty Encoding Module model."""

    def __init__(self, resnet_size):
        super(SENet, self).__init__()
        self.RESNET_DEPTH = resnet_size
        self.REDUCTION_DIM = 2048
        self.SSM_MID_DIM = 256
        self.UNFOLD_SIZE = 7
        self.SSE_KERNEL_SIZE = 3
        self._construct()
        self.apply(net.init_weights)

    def _construct(self):

        _IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
        NUM_GROUPS = 1
        WIDTH_PER_GROUP = 64
        g, gw = NUM_GROUPS, WIDTH_PER_GROUP

        (d1, d2, d3, d4) = _IN_STAGE_DS[self.RESNET_DEPTH]
        w_b = gw * g

        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g)
        self.SSM = SSM(in_ch=2048, mid_ch=self.SSM_MID_DIM, unfold_size=self.UNFOLD_SIZE, ksize=self.SSE_KERNEL_SIZE)
        self.head = GlobalHead(2048, nc=self.REDUCTION_DIM)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        x4 = self.SSM(x4)
        x4_p = self.head.pool(x4)
        x4_p = x4_p.view(x4_p.size(0), -1)
        x = self.head.fc(x4_p)
        return x
