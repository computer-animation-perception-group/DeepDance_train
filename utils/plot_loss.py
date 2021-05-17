import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import os



def run():
    a = OrderedDict()
    a[0] = 0.9
    a[1] = 0.8
    a[2] = 0.6
    save_plot_loss('', a, a, a)
    plt.show()


def save_plot_loss(save_path, d_loss_dict, g_mse_loss_dict, g_loss_dict):
    d_loss = [k for k in d_loss_dict.values()]
    g_mse_loss = [k for k in g_mse_loss_dict.values()]
    g_loss = [k for k in g_loss_dict.values()]
    x = list(range(len(d_loss)))

    plt.plot(x, d_loss, 'g-', label='d_loss')
    plt.plot(x, g_mse_loss, 'r-', label='g_mse_loss')
    plt.plot(x, g_loss, 'b-', label='g_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.savefig(save_path)
    plt.close()
