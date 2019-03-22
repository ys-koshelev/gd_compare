import torch

# function to reverse the kernel, since torch computes cross-correlation instead of convolution
def ReverseKernel(kernel):
    return torch.flip(kernel, (-1,-2));

# convolution of single image with single kernel
def Conv2d(input, kernel):
    out = torch.conv2d(input.unsqueeze(0).transpose(1,0), ReverseKernel(kernel).unsqueeze(0), groups=1).transpose(1,0).squeeze(0);
    return out;  

# measure PSNR, assuming the maximum image intensity of 1
def PSNR(x, y):
    MSE = torch.nn.functional.mse_loss(x, y);
    return -10*torch.log10(MSE);