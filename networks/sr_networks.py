import math
import torch
from torch import nn
import torch.nn.functional as F
from . import sr_networks_edsr_common


# https://github.com/Lornatang/SRCNN-PyTorch
class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )
        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )
        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out
    
    def embed(self, x):
        # for DM
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.firstPart = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.midPart = [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),nn.ReLU(inplace=True)]
        for _ in range(7):  # 17
            self.midPart.extend([nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True)])
        self.midPart = nn.Sequential(*self.midPart)
        self.lastPart = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.firstPart(x))
        x = self.midPart(x)
        x = self.lastPart(x)
        out = torch.add(x, residual)
        return out


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    def __init__(self, args, conv=sr_networks_edsr_common.default_conv):
        super(EDSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define body module
        m_body = [
            sr_networks_edsr_common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        # define tail module
        m_tail = [
            sr_networks_edsr_common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class S_CTSRN(nn.Module):
    """
    Single-Slice CT Super-Resolution Network (S-CTSRN)
    The architecture consists of two parallel branches:
        1. Upper residual branch: extracts high-frequency detail features.
        2. Lower upsampling branch: upsamples the input image to provide a base reconstruction.
    """
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale = scale_factor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, scale_factor**2, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Input: Low-resolution CT slices [B, 1, H, W]
        Output: Reconstructed high-resolution images [B, 1, sH, sW]
        """
        residual = F.relu(self.conv1(x))
        residual = F.relu(self.conv2(residual))
        residual = self.conv3(residual)
        # Pixel shuffle (ESPCN method)
        residual = F.pixel_shuffle(residual, self.scale)  # [B, 1, (H-8)*s, (W-8)*s]
        # Replace transposed convolution with a fixed kernel with interpolation.
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        return base + residual


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EDSR and MDSR')
    parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
    parser.add_argument('--scale', type=str, default='4', help='super resolution scale')
    parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
    parser.add_argument('--n_colors', type=int, default=1, help='number of color channels to use')
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    # model = SRCNN()
    # model = VDSR()
    # model = EDSR(args)
    model = S_CTSRN()
    input_tensor = torch.rand(2, 1, 128, 128)
    output = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    print(torch.max(input_tensor))
    print(torch.min(input_tensor))
    print(torch.max(output))
    print(torch.min(output))