import copy
import torch

def DescentOptimizer(fun, opt, init, max_iter=100, eps=1e-5):
    fun.x.data = init;
    x_steps = [init.numpy()];
    y_steps = [fun(init).numpy()];
    f = fun.forward_parameter();
    for i in range(max_iter):
        opt.zero_grad();
        f.backward();
        opt.step();
        f = fun.forward_parameter();
        x_steps.append(copy.deepcopy(fun.x.data.numpy()));
        y_steps.append(f.detach().numpy());
        if (torch.abs(f.data - fun.fun_min) < eps):
            print('Converged with desired accuracy in {} iterations.'.format(i));
            return x_steps, y_steps;
    print('The desired accuracy was not achieved.')
    return x_steps, y_steps;