import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """implements fully connected sublayer
    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
