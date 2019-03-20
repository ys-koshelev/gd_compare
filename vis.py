import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_through_numpy(x_steps, fun, y_steps):
    x_steps = np.array(x_steps);
    
    # prepare image
    plt.figure(figsize=(18,15));
    plt.tight_layout()
    ax1 = plt.subplot(1,2,2)
    
    # plot contours
    X, Y = np.meshgrid(np.linspace(-7, 3, 400), np.linspace(-9, 9, 100))
    Z = np.empty_like(X);
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fun(torch.FloatTensor([X[i,j], Y[i,j]])).numpy()

    # levels for contour lines
    levels = fun.fun_min.numpy() + np.linspace(0, 12, 10)**2

    # on both sub-plots plot:
    # having equal axes is important to see if lines orthogonal or not
    ax1.axis('equal')
    
    # minimum
    ax1.plot(fun.x_min[0].numpy(), fun.x_min[1].numpy(), '*')
    
    # initial point
    ax1.plot(x_steps[0][0], x_steps[0][1], 'or')
    
    # plot contour lines
    ax1.contour(X, Y, Z, levels)
    
    # plot on second image
    ax1.plot(x_steps[:, 0], x_steps[:, 1])

    # residual f(x_k) - f* 
    f_k_residual = np.abs(fun.fun_min.numpy() - np.array(y_steps));
    N = len(f_k_residual)

    x_k_residual = np.array(x_steps) - fun.x_min.numpy();

    ax2 = plt.subplot(4,2,5)
    ax2.set_title('Function residual')
    ax2.plot(f_k_residual, )
    ax2.grid()

    ax3 = plt.subplot(4,2,1)
    ax3.set_title('Logarithmic function residual')
    ax3.plot(np.log(f_k_residual), ) 
    ax3.grid()

    ax4 = plt.subplot(4,2,3)
    ax4.set_title('Value residual (Euclidean norm)')
    ax4.plot(np.arange(N), np.linalg.norm(x_k_residual, axis=1))
    ax4.grid()

    ax5 = plt.subplot(4,2,7)
    ax5.set_title('Logarithmic value residual (Euclidean norm)')
    ax5.plot(np.arange(N), np.log(np.linalg.norm(x_k_residual, axis=1)) );
    ax5.grid()

    plt.show()
    pass