import torch
import einops

class SaintAndre(torch.nn.Module):
    def __init__(self, latent_size=512, dropout_ratio=0.5):
        super(SaintAndre, self).__init__()
        self.latent_size = latent_size

        self.linear1 = torch.nn.Linear(2 * 3 * 3, self.latent_size)
        self.linear2 = torch.nn.Linear(self.latent_size, 1)
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 2 * 3 * 3)
        act1 = self.linear1(input_tsr)  # (N, L)
        act2 = torch.nn.functional.relu(act1)  # (N, L)
        act3 = self.dropout(act2)  # (N, L)
        act4 = self.linear2(act3)  # (N, 1)
        return act4

class Coptic(torch.nn.Module):
    def __init__(self, number_of_channels=512, dropout_ratio=0.5):
        super(Coptic, self).__init__()
        self.number_of_channels = number_of_channels

        self.conv1 = torch.nn.Conv2d(2, self.number_of_channels, kernel_size=(3, 3))
        self.dropout = torch.nn.Dropout1d(p=dropout_ratio)
        self.linear1 = torch.nn.Linear(self.number_of_channels, 1)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 2 * 3 * 3)
        act1 = einops.rearrange(input_tsr, 'N (C H W) -> N C H W', C=2, H=3, W=3)  # (N, 2, 3, 3)
        act2 = self.conv1(act1)  # (N, L, 1, 1)
        act3 = einops.rearrange(act2, 'N L 1 1 -> N L')
        act4 = self.dropout(act3)  # (N, L)
        act5 = self.linear1(act4)  # (N, 1)
        return act5
