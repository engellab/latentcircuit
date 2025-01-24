## Functions for preprocessing PFC data from Mante et al 2013

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from scipy import io as spio

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict




def contrast_means(df):
    df.stim_dir = df.stim_dir.astype(float)
    df.stim_col2dir = df.stim_col2dir.astype(float)
    dir_means = np.mean(np.stack(df.groupby('unit_id')['stim_dir'].apply(np.unique).reset_index(
        name='unique_contrasts').unique_contrasts.values), axis=0)
    col2dir_means = np.mean(np.stack(df.groupby('unit_id')['stim_col2dir'].apply(np.unique).reset_index(
        name='unique_contrasts').unique_contrasts.values), axis=0)
    return np.round(dir_means, 2), np.round(col2dir_means, 2)


def replace_dir(x, dir_means):
    """
    Replace each contrast
    """
    unique_dir = np.unique(x)

    def fn(dir):
        return dir_means[np.argwhere(unique_dir == dir)].item()

    return list(map(fn, x))


def replace_col2dir(x, col2dir_means):
    """
    Replace each contrast
    """
    unique_col2dir = np.unique(x)

    def fn(col2dir):
        return col2dir_means[np.argwhere(unique_col2dir == col2dir)].item()

    return list(map(fn, x))




animal = 'ar'
matdata = loadmat('Data/Mante/Monkey_' + animal + '/dataT.mat')

# Reorganize into pandas frame.
rows = []
nun = len(matdata['dataT']['unit'])
for unit in range(nun):
    n_trials = matdata['dataT']['unit'][unit].response.shape[0]
    for k in range(n_trials):
        # if matdata['dataT']['unit'][unit].task_variable.correct[k]==1:
        rows.append({'unit_id': unit,
                     'trial': k,
                     'stim_dir': matdata['dataT']['unit'][unit].task_variable.stim_dir[k],
                     'stim_col2dir': matdata['dataT']['unit'][unit].task_variable.stim_col2dir[k],
                     'context': matdata['dataT']['unit'][unit].task_variable.context[k],
                     'choice': matdata['dataT']['unit'][unit].task_variable.targ_dir[k],
                     'correct': matdata['dataT']['unit'][unit].task_variable.correct[k],
                     'response': matdata['dataT']['unit'][unit].response[k]})
df = pd.DataFrame(rows)

# Rename context
dct = {1: 'motion', -1: 'color'}
df.context = [*map(dct.get, df.context.values)]

# Group contrasts
dir_means, col2dir_means = contrast_means(df)
df.stim_dir = df.groupby('unit_id')['stim_dir'].transform(lambda x: replace_dir(x, dir_means))
df.stim_col2dir = df.groupby('unit_id')['stim_col2dir'].transform(lambda x: replace_col2dir(x, col2dir_means))

# Get units that have at least 10 trials for all correct conditions
df_correct = df[df.correct == 1]
df_correct["Condition"] = \
    df_correct.groupby(['stim_dir', 'stim_col2dir', 'context', 'choice']).grouper.group_info[0]
counts = df_correct.groupby(['unit_id', 'Condition']).size().reset_index(name="counts")
min_counts = counts.groupby('unit_id').counts.min().reset_index()
good_units = min_counts[min_counts.counts >= 4]['unit_id'].unique()
df = df[df.unit_id.isin(good_units)]

# Define condition column
df["condition"] = df.groupby(['stim_dir', 'stim_col2dir', 'context', 'choice', 'correct']).grouper.group_info[0]


## Bootstrapped mean responses for each condition
n_boot = 5
p  = .8
rows = []
for unit_id in df.unit_id.unique():
    unit_df = df[df.unit_id==unit_id]
    for condition in unit_df.condition.unique():
        for bootstrap in range(n_boot):
            sample_df = (unit_df[unit_df.condition == condition]).sample( frac=p, replace=True)
            rows.append({'unit_id': unit_id,
                         'bootstrap': bootstrap,
                         'condition':condition,
                         'stim_dir': sample_df.stim_dir.unique().item(),
                         'stim_col2dir': sample_df.stim_col2dir.unique().item(),
                         'context': sample_df.context.unique().item(),
                         'choice': sample_df.choice.unique().item(),
                         'correct':sample_df.correct.unique().item(),
                         'response':np.mean(np.stack(sample_df.response.values,axis=0),axis=0)})
df = pd.DataFrame(rows)

# Save
df.to_pickle("pfc_"+ animal + ".pkl")

# Smooth responses
sigma = 1
df['response'] = df.response.apply(gaussian_filter1d, args=[sigma])

# Z-score
df['center'] = df.groupby(['unit_id']).response.transform(lambda x: np.mean(np.stack(x)))
df['std'] = df.groupby(['unit_id']).response.transform(lambda x: np.std(np.stack(x)))
df['response'] = (df['response'] - df['center']) / df['std']

# Restrict to intersection of conditions across units
condition_sets = df.groupby('unit_id').condition.apply(lambda x: set(np.stack(x.values))).reset_index()
u = set.intersection(*list(condition_sets.condition.values))
df = df[df.condition.isin(u)]
df = df[df.correct==1]

train = df.groupby(['bootstrap','stim_dir', 'stim_col2dir', 'context', 'choice', 'correct'])['response'].apply(
        lambda x: np.stack(x)).reset_index()

# Remove condition independent responses
x_t = np.mean(np.stack(train.response.values), axis=0)
train.response = train.response.apply(lambda x: x - x_t)

# PCA
n_components = .5
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components, svd_solver='full')
N = train.response.values[0].shape[1]
pca.fit(np.reshape(np.stack(train.response.values, axis=0), (-1, N)))
train.response = train.response.apply(lambda x: pca.inverse_transform(pca.transform(x)))

train.to_pickle("pfc_trajectory_"+ animal + ".pkl")