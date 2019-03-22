import torch
from utils import Conv2d
import copy

# 2D parabolic function
class SquareFunction(torch.nn.Module):
    def __init__(self, Q=torch.FloatTensor([[1.,0.], [0.,1.]]), c=torch.FloatTensor([1.,1.])):
        super(SquareFunction, self).__init__()
        self.num_eval_calls = 0;
        self.c = c;
        self.Q = Q;
        self.x_min = -torch.inverse(Q)@c;
        self.fun_min = self.forward(self.x_min);
        self.x = torch.nn.Parameter(torch.rand(2));
        
    def constraint(self, x):
        return True;
    
    def forward_parameter(self):
        return self.forward(self.x);
    
    def forward(self, x):
        self.num_eval_calls += 1;
        if self.constraint(x):
            f = torch.transpose(x, 0, -1)@((self.Q@x)/2 + self.c);
            return f;
        else:
            return torch.inf;
    
# objective function for image Total Variation denoising
class ObjectiveFunction(torch.nn.Module):
    def __init__(self, lam, norm, y):
        super(ObjectiveFunction, self).__init__()
        self.lam = torch.FloatTensor([lam])[0];
        self.norm = norm;
        self.x = torch.nn.Parameter(copy.deepcopy(torch.empty(y.shape)));
        self.y = y;
        self.x.data = copy.deepcopy(self.y);
        self.grad_x = torch.FloatTensor([[0,0,0],[1,-1,0],[0,0,0]]).unsqueeze(0);
        self.grad_y = torch.FloatTensor([[0,1,0],[0,-1,0],[0,0,0]]).unsqueeze(0);
    
    def init(self, data):
        self.x.data = data;
        pass;
    
    def forward(self):
        ret = (torch.norm(self.x - self.y)**2)/2 + self.lam*(torch.norm(Conv2d(self.x, self.grad_x), p=self.norm) + torch.norm(Conv2d(self.x, self.grad_y), p=self.norm));
        return ret;