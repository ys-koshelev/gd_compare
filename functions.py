import torch

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