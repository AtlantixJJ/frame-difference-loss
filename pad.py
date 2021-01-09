import torch
import torch.nn.functional as F

class InterpolationPadding2d(torch.nn.Module):
    """
    Only support 1 px padding
    """
    def __init__(self, detach=False):
        super(InterpolationPadding2d, self).__init__()

        self.detach = detach
    
    def forward(self, x):
        n, c, h, w = x.shape
        y = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        y[:, :, 0, 1:-1] = x[:, :, 0, :] + x[:, :, 0, :] - x[:, :, 1, :]
        y[:, :, 1:-1, 0] = x[:, :, :, 0] + x[:, :, :, 0] - x[:, :, :, 1]
        y[:, :, -1, 1:-1] = x[:, :, -1, :] + x[:, :, -1, :] - x[:, :, -2, :]
        y[:, :, 1:-1, -1] = x[:, :, :, -1] + x[:, :, :, -1] - x[:, :, :, -2]
        y[:, :, 0, 0] = (y[:, :, 0, 1] + y[:, :, 1, 0]) / 2
        y[:, :, 0, -1] = (y[:, :, 0, -2] + y[:, :, 1, -1]) / 2
        y[:, :, -1, 0] = (y[:, :, -1, 1] + y[:, :, -2, 0]) / 2
        y[:, :, -1, -1] = (y[:, :, -2, -1] + y[:, :, -1, -2]) / 2
        if self.detach:
            y[:, :, 0, :] = y[:, :, 0, :].detach()
            y[:, :, -1, :] = y[:, :, -1, :].detach()
            y[:, :, :, 0] = y[:, :, :, 0].detach()
            y[:, :, :, -1] = y[:, :, :, -1].detach()
        return y

class Padding2d(torch.nn.Module):
    def __init__(self, padding, pad_type):
        super(Padding2d, self).__init__()
        self.sh = padding[0]
        self.sw = padding[1]
        self.pad_type = pad_type
        self.standard_types = ["constant", "reflect", "replicate"]
        self.offset_left = (torch.rand(1).numpy()[0] < 0.5)
    
    def forward(self, x):
        n, c, h, w = x.shape
        h1, w1 = self.sh // 2, self.sw // 2
        h2, w2 = self.sh - h1, self.sw - w1
        for stype in self.standard_types:
            if stype in self.pad_type:
                y = F.pad(x, (h1, h2, w1, w2), stype)
        if "interpolate" in self.pad_type:
            y = F.interpolate(x, (self.sh + h, self.sw + w),
                mode='bilinear', align_corners=True)
        if "gaussian" in self.pad_type:
            y = F.pad(x, (h1, h2, w1, w2), "constant")
            flt = x.view(n, c, -1)
            mean, var = flt.mean(2, keepdim=True), flt.var(2, keepdim=True)
            randvec = torch.Tensor(n, c, 2 * (h + w) + 8).normal_().type_as(x)
            randvec = randvec * var + mean
            l, r = 0, w+self.sw
            y[:, :, 0, :]   = randvec[:, :, l:r]
            l = r
            r = l + w+self.sw
            y[:, :, -1, :] = randvec[:, :, l:r]
            l = r
            r = l + h+self.sh
            y[:, :, :, 0]   = randvec[:, :, l:r]
            l = r
            r = l + h+self.sh
            y[:, :, :, -1]     = randvec[:, :, l:r]
        if "detach" in self.pad_type:
            y = y.detach()
        if self.offset_left:
            y[:, :, h1 : h1 + h, w1 : w1 + w] = x
        else:
            y[:, :, -h2 - h : -h2, -w2 - w : -w2] = x
        return y

if __name__ == "__main__":
    # test the differential relation

    # detached
    a = torch.Tensor(1, 1, 7, 7)
    a.normal_()
    a.requires_grad = True
    a_pad = F.pad(a, (1, 1, 1, 1), 'reflect').detach()
    a_pad[:, :, 1:-1, 1:-1] = a
    a_pad.sum().backward()
    print("=> reflect detached")
    print(a.grad)

    # normal
    a = torch.Tensor(1, 1, 7, 7)
    a.normal_()
    a.requires_grad = True
    a_pad = F.pad(a, (1, 1, 1, 1), 'reflect')
    a_pad.sum().backward()
    print("=> reflect normal")
    print(a.grad)

    # detached
    a = torch.Tensor(1, 1, 7, 7)
    a.normal_()
    a.requires_grad = True
    a_pad = F.pad(a, (1, 1, 1, 1), 'replicate').detach()
    a_pad[:, :, 1:-1, 1:-1] = a
    a_pad.sum().backward()
    print("=> replicate detached")
    print(a.grad)

    # normal
    a = torch.Tensor(1, 1, 7, 7)
    a.normal_()
    a.requires_grad = True
    a_pad = F.pad(a, (1, 1, 1, 1), 'replicate')
    a_pad.sum().backward()
    print("=> replicate normal")
    print(a.grad)

    # detached
    a = torch.Tensor(1, 1, 7, 7)
    a.normal_()
    a.requires_grad = True
    a_pad = F.pad(a, (1, 1, 1, 1), 'constant').detach()
    a_pad[:, :, 1:-1, 1:-1] = a
    a_pad.sum().backward()
    print("=> constant detached")
    print(a.grad)

    # normal
    a = torch.Tensor(1, 1, 7, 7)
    a.normal_()
    a.requires_grad = True
    a_pad = F.pad(a, (1, 1, 1, 1), 'constant')
    a_pad.sum().backward()
    print("=> constant normal")
    print(a.grad)