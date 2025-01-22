# Some plotting functions to help analyze RNN after training.

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import scipy as sy
from net import *

plt.rcParams["axes.grid"] = False


def pf(x, alpha, beta):
    return 1. / (1 + np.exp(-(x - alpha) / beta))


def prob_right(x):
    return np.sum(x > 0) / len(x)



def psychometric(net,u,conditions):
    par0 = np.array([0., 1.])
    contrasts = np.linspace(-1, 1, 15)
    x = net.forward(u)
    output = net.output_layer(x)

    rows = []
    for trial in range(u.shape[0]):
        rows.append({'context': conditions[trial]['context'],
                    'motion_coh': conditions[trial]['motion_coh'],
                     'color_coh': conditions[trial]['color_coh'],
                     'choice': torch.relu(output[trial, -1, 0] - output[trial, -1, 1])})
    df = pd.DataFrame(rows)

    motion_df = df.groupby(['context','motion_coh'])['choice'].apply(prob_right).reset_index(name='prob_right')
    color_df = df.groupby(['context','color_coh'])['choice'].apply(prob_right).reset_index(name='prob_right')



    fig = plt.figure(figsize=(4, 1.5))
    gs = gridspec.GridSpec(1, 2,wspace=.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    par, mcov = sy.optimize.curve_fit(pf, motion_df[(motion_df.context=="motion")].motion_coh.values, motion_df[(motion_df.context=="motion")].prob_right.values, par0)
    ax0.plot( 100 * contrasts,100 * pf(contrasts, par[0], par[1]),color='black', lw=1,marker ='.',label = 'Motion')

    par, mcov = sy.optimize.curve_fit(pf, motion_df[(motion_df.context == "color")].motion_coh.values,
                                      motion_df[(motion_df.context == "color")].prob_right.values, par0)
    ax0.plot(100 * contrasts,100 * pf(contrasts, par[0], par[1]), color='lightgray', lw=1,marker = '.',label = 'Color')

    par, mcov = sy.optimize.curve_fit(pf, color_df[(color_df.context=="motion")].color_coh.values, color_df[(color_df.context=="motion")].prob_right.values, par0)
    ax1.plot( 100 * contrasts,100 * pf(contrasts, par[0], par[1]),color='black', lw=1,marker ='.',label = 'Motion')

    par, mcov = sy.optimize.curve_fit(pf, color_df[(color_df.context == "color")].color_coh.values,
                                      color_df[(color_df.context == "color")].prob_right.values, par0)
    ax1.plot(100 * contrasts,100 * pf(contrasts, par[0], par[1]), color='lightgray', lw=1,marker = '.',label = 'Color')


    ax1.legend()
    for ax in [ax0,ax1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=7, bottom=True)
        ax.yaxis.set_tick_params(labelsize=7, left=True)
    ax0.set_ylabel("Choice to right (%)", fontsize=8)
    ax0.set_xlabel("Motion coherence (%)", fontsize=8)
    ax1.set_xlabel("Color coherence (%)", fontsize=8)
