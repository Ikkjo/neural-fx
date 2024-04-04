import torch


class DilatedConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)


class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WaveNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = DilatedConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.activation(out)
        return out


class ResidualWaveNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualWaveNetBlock, self).__init__()
        self.wavenet_block = WaveNetBlock(
            in_channels, out_channels, kernel_size, dilation
        )
        self.conv = Conv1d(out_channels, out_channels)

    def forward(self, x):
        out = self.wavenet_block(x)
        out = self.conv(out)
        return out + x[:, :, -out.size(2) :]


class WaveNet(torch.nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, dilation_array):
        super(WaveNet, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation_array = dilation_array
        self.receptive_field = self.get_receptive_field()
        self.first_conv = WaveNetBlock(
            1, self.num_channels, self.kernel_size, self.dilation_array[0]
        )
        self.residual_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers - 1):
            self.residual_layers.append(
                ResidualWaveNetBlock(
                    self.num_channels,
                    self.num_channels,
                    kernel_size,
                    self.dilation_array[i],
                )
            )
        self.residual_layers = torch.nn.Sequential(*self.residual_layers)
        self.last_conv = WaveNetBlock(
            self.num_channels, 1, kernel_size, self.dilation_array[-1]
        )
        self.out = WaveNetBlock(
            in_channels=1, out_channels=1, kernel_size=1, dilation=1
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[-1] < self.receptive_field:
            x = torch.cat(
                [
                    torch.zeros(
                        x.shape[0],
                        1,
                        self.receptive_field - x.shape[-1],
                        device=x.device,
                    ),
                    x,
                ],
                dim=-1,
            )
        else:
            x = x[:, :, -self.receptive_field :]
        out = self.first_conv(x)
        out = self.residual_layers(out)
        out = self.last_conv(out)
        out = self.out(out)
        return out

    def get_receptive_field(self):
        return (self.kernel_size - 1) * torch.sum(torch.tensor(self.dilation_array)) + 1


def get_dilation_array(num_layers, max_dilation_parameter):
    dilation_array = []
    for i in range(num_layers):
        dilation_array.append(2 ** (i % max_dilation_parameter))
    return dilation_array
