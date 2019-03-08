import sys
#sys.path.append('..')
from GPy.kern import LFM
import numpy as np
from GPy import models
# import GPy
# %pylab inline
import pylab as pb
#import .. model
#pylab.ion()
import matplotlib.pyplot as plt
from GPy import kern


def plot_2outputs(m,xlim,ylim):
    fig = pb.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,100),ax=ax1)#, predict_kw={"full_cov": True})
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(100,200),ax=ax2)#, predict_kw={"full_cov": True})
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)
    return fig, ax1, ax2



if __name__ == '__main__':
    # This functions generate data corresponding to two outputs
    f_output1 = lambda x: 4. * np.sin(x / 2.) + np.random.rand(x.size)[:, None] * 2.
    f_output2 = lambda x: 6. * np.cos(x / 2.) + np.random.rand(x.size)[:, None] * 8.
    X1 = np.random.rand(100)[:, None] * 75
    X2 = np.random.rand(100)[:, None] * 70
    Y1 = f_output1(X1)
    # Y1 = Y1 - Y1.mean()
    Y2 = f_output2(X2)
    # Y2 = Y2 - Y2.mean()
    # {X,Y} test set for each output
    Xt1 = np.random.rand(100)[:, None] * 100
    Xt2 = np.random.rand(100)[:, None] * 100
    Yt1 = f_output1(Xt1) #- Y1.mean()
    Yt2 = f_output2(Xt2) #- Y2.mean()


    # xlim = (0, 100);
    # ylim = (0, 50)
    # fig = pb.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # ax1.set_xlim(xlim)
    # ax1.set_title('Output 1')
    # ax1.plot(X1[:, :1], Y1, 'kx', mew=1.5, label='Train set')
    # ax1.plot(Xt1[:, :1], Yt1, 'rx', mew=1.5, label='Test set')
    # ax1.legend()
    # ax2 = fig.add_subplot(212)
    # ax2.set_xlim(xlim)
    # ax2.set_title('Output 2')
    # ax2.plot(X2[:, :1], Y2, 'kx', mew=1.5, label='Train set')
    # ax2.plot(Xt2[:, :1], Yt2, 'rx', mew=1.5, label='Test set')
    # ax2.legend()
    # pb.show()

    # kernel = LFM(input_dim=, output_dim=2)

    # import pkgutil
    #
    # search_path = '/home/chitianqilin/PycharmProjects/GPy/GPy/models.'  # set to None to see all modules importable from sys.path
    # all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
    # print(all_modules)

   # kernel = kern.LFM(input_dim=X.shape[1], output_dim=Ny, scale=5 * np.ones(Ny), active_dims=None)

    m = models.lfm_regression.LFMRegression([X1, X2], [Y1, Y2])#, kernel)
    # m.optimize(max_iters=10000)

    m.optimize_restarts(1000, robust=True)#, parallel=True) #
    fig, ax1, ax2=plot_2outputs(m, xlim=(0, 100), ylim=(-20, 60))
    # ax1.scatter(m.X[:100, 0], m.Y[:100])
    # ax2.scatter(m.X[100:, 0], m.Y[100:])
    plt.show()

    print(m)