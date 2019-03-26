import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_through_numpy(values, fun, param):
    for ind in range(len(values)):
        values[ind][0] = np.array(values[ind][0])
        values[ind][1] = np.array(values[ind][1])
    plt.figure(figsize=(18,15));
    plt.tight_layout()
    
    # plot contours
    if param == 'convex':
        X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 100))
    else:
        X, Y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1,1, 100))
    Z = np.empty_like(X);
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fun(torch.FloatTensor([X[i,j], Y[i,j]])).numpy()

    # levels for contour lines
    if param == "convex":
        levels = fun.fun_min.numpy() + np.linspace(0, 3, 10)**2
    else:
        levels = fun.fun_min.numpy() + np.linspace(0, 1, 10)
    ax1 = plt.subplot(2,3,1)
    ax1.plot(fun.x_min[0].numpy(), fun.x_min[1].numpy(), '*')
    ax1.contour(X, Y, Z, levels)
    
    ax2 = plt.subplot(2,3,2)
    ax2.set_title('Function residual')
    ax2.grid()
    
    ax3 = plt.subplot(2,3,3)
    ax3.set_title('Logarithmic function residual')
    ax3.grid()
    
    ax4 = plt.subplot(2,3,4)
    ax4.set_title('Value residual (Euclidean norm)')
    ax4.grid()
    
    ax5 = plt.subplot(2,3,5)
    ax5.set_title('Logarithmic value residual (Euclidean norm)')
    ax5.grid()
    
    # minimum
    for ind in range(len(values)):
        x_steps = values[ind][0]
        y_steps = values[ind][1]
        name = values[ind][2]
        
        # initial point    
        ax1.plot(x_steps[0][0], x_steps[0][1], 'or')
    
        # plot on second image
        ax1.plot(x_steps[:, 0], x_steps[:, 1],'.', label = name)

        # residual f(x_k) - f* 
        f_k_residual = np.abs(fun.fun_min.numpy() - np.array(y_steps));
        N = len(f_k_residual)

        x_k_residual = np.array(x_steps) - fun.x_min.numpy();

        ax2.plot(f_k_residual, label = name)
        
        ax3.plot(np.log(f_k_residual), label = name) 
        
        ax4.plot(np.arange(N), np.linalg.norm(x_k_residual, axis=1), label = name)
        
        ax5.plot(np.arange(N), np.log(np.linalg.norm(x_k_residual, axis=1)), label = name );
        
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    plt.show()
    pass