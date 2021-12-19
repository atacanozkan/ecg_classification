
##############################################################
####################### Libraries ############################
##############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from treelib import Tree
import pywt
import scipy.io as sio
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


##############################################################
####################### Functions ############################
##############################################################

os.getcwd()
os.chdir("../datasets")
file_path = os.getcwd() + "\\"
def load_dataset(filename, extension = '.csv'):
    """
    Iports the dataset
    Parameters
    ----------
    dataset

    Returns
    -------
    dataframe
    """
    if 'csv' in extension:
        data = pd.read_csv(file_path+filename+extension)
    elif 'xls' in extension:
        data = pd.read_excel(file_path+filename+extension)
    elif 'pkl' in extension:
        data = pd.DataFrame(pickle.load(open(file_path+filename+extension, 'rb')))
    return data

def save_dataset(data, filename, extension = '.csv'):
    """
    Iports the dataset
    Parameters
    ----------
    dataset

    Returns
    -------
    dataframe
    """
    if 'csv' in extension:
        data.to_csv(file_path+filename+extension)
    elif 'xls' in extension:
        data.to_excel(file_path+filename+extension, index=False)
    elif 'pkl' in extension:
        pickle.dump(data, open(file_path+filename+extension, 'wb'))


def plot_time_freq_scaleogram(y, f_s):
    y = list(y)

    scales = np.arange(1, 256)
    title = 'CWT (Power Spectrum)'
    ylabel = 'Frequency'
    xlabel = 'Time'

    fig = plt.figure(figsize=(16, 16))
    spec = gridspec.GridSpec(ncols=6, nrows=6)
    top_ax = fig.add_subplot(spec[0, 0:5])
    bottom_left_ax = fig.add_subplot(spec[1:, 0:5])
    bottom_right_ax = fig.add_subplot(spec[1:, 5])
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])

    plot_signal(top_ax, y, f_s)
    yticks, ylim, im = plot_wavelet(bottom_left_ax, y, f_s, scales, waveletname='cmor1.5-1.0', xlabel=xlabel, ylabel=ylabel, title=title, depth=4)

    fig.colorbar(im, cax=cbar_ax, orientation="vertical")

    plot_fft(bottom_right_ax, y, f_s, plot_direction='vertical', yticks=yticks, ylim=ylim)
    bottom_right_ax.set_ylabel('Period', fontsize=14)
    # plt.tight_layout()
    plt.show()

def plot_signal(ax, y, f_s):
    N = len(y)
    time = np.arange(0, N) * (1/f_s)
    ax.plot(time, y, label='signal')
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.set_title('Signal', fontsize=16)
    ax.legend(loc='upper right')

def plot_fft(ax, y, f_s, plot_direction='horizontal', yticks=None, ylim=None):
    variance = np.std(y) ** 2
    f, fft = get_fft(y, f_s)
    if plot_direction == 'horizontal':
        ax.plot(f, fft, 'r-', label='Fourier Transform')
    elif plot_direction == 'vertical':
        scales_log = np.log2(f)
        ax.plot(fft, scales_log, 'r-', label='Fourier Transform')
        ax.set_yticks(np.log2(yticks))
        #ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        # ax.invert_yaxis()
        ax.set_ylim(ylim[0], -1)
    ax.legend()

def plot_wavelet(ax, y, f_s, scales, waveletname='cmor', cmap=plt.cm.seismic, title='', ylabel='', xlabel='', depth = 1):
    time = np.arange(0, N) * (1 / f_s)
    coefficients, frequencies = pywt.cwt(y, scales, waveletname, 1/f_s)

    power = (abs(coefficients)) ** 2
    period = frequencies
    levels = [0.015625,0.03125,0.0625, 0.125, 0.25, 0.5, 1]
    levels = [level / depth for level in levels]
    # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()

    return yticks, ylim, im

def plot_tree(wp):
    tree = Tree()
    tree.create_node("data", "data")  # No parent means its the root node
    for level in range(1, wp.maxlevel + 1):
        nodes = wp.get_level(level, order='natural', decompose=False)
        for i, node in enumerate(nodes):
            if level == 1:
                parent = "data"
            else:
                parent = nodes_prev[int(i / 2)].path
            tree.create_node(str(node.path), str(node.path), parent=parent)
        nodes_prev = nodes
    tree.show()

def wpd_plt(y, wave='sym5', n=None, best_basis = None):
    # wpd decompose
    wp = pywt.WaveletPacket(data=y, wavelet=wave, mode='symmetric', maxlevel=n)
    n = wp.maxlevel
    # Calculate the coefficients for each node, where map Medium, key by'aa'Wait, value For List
    map = {}
    map[1] = y
    for row in range(1, n + 1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data

    # Mapping
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(n + 1, 1, 1)  # Draw the first graph
    ax.set_title('data')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.plot(map[1])
    for i in range(2, n + 2):
        level_num = pow(2, i - 1)  # Starting with the second row, the power of 2 of the previous row is calculated
        # Getting decomposed at each level node: For example, the third layer['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i - 1, 'freq')]
        for j in range(1, level_num + 1):
            if best_basis != None:
                if True in [(re[j - 1].startswith(b) & (re[j - 1] != b)) for b in best_basis]:
                    continue
            ax = plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
            plt.plot(map[re[j - 1]])  # List starts at 0
            ax.set_title(re[j - 1])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
    plt.show()


def get_best_basis(data, wave='sym5', thresh=1., **_):
    wp = pywt.WaveletPacket(data, wave)
    wp.get_leaf_nodes(decompose=True)
    levels = pywt.dwt_max_level(len(data), pywt.Wavelet(wave))

    for level in range(levels, 1, -1):
        nodes = wp.get_level(level, order='natural', decompose=False)
        paths = [n.path for n in nodes]
        n = len(paths)
        for i in range(0, n, 2):
            child_vals = np.hstack([wp[paths[i]].data, wp[paths[i + 1]].data])  # child nodelarının approximation ve details verilerini birleştirir
            parent_val = wp[wp[paths[i]].parent.path].data  # parent nodunun verileri
            # child nodelarının verilerinin entropisi parent nodu verisinin entropisinden büyükse child nodeları sil
            # aksi durumda parent nodu verisini parent nod verisinin entropi değeri olarak değiştir
            if shannon_entropy(child_vals) > shannon_entropy(parent_val) * thresh:
                wp.__delitem__(paths[i])
                wp.__delitem__(paths[i + 1])
            else:
                wp[wp[paths[i]].parent.path].data = min(shannon_entropy(child_vals), shannon_entropy(parent_val))
    leaves = wp.get_leaf_nodes()
    best_basis = leaves

    return wp, best_basis

def get_features(data):
    entropy = shannon_entropy(data)
    crossings = get_crossings(data)
    statistics = get_statistics(data)
    return [entropy] + crossings + statistics

def shannon_entropy(data):
    if len(data) == 1:
        s = data
    else:
        e = data ** 2 / len(data)
        p = e / sum(e)
        s = -sum(p * np.log(p))
    return s

def get_crossings(data):
    zero_crossing_indices = np.nonzero(np.diff(np.array(data) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(data) > np.nanmean(data)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_statistics(data):
    n5 = np.nanpercentile(data, 5)
    n25 = np.nanpercentile(data, 25)
    n75 = np.nanpercentile(data, 75)
    n95 = np.nanpercentile(data, 95)
    median = np.nanpercentile(data, 50)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    var = np.nanvar(data)
    rms = np.nanmean(np.sqrt(data**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def get_fft(y, f_s):
    N = len(y)
    f = np.linspace(0.0, 1.0 / (2.0 * (1/f_s)), N // 2)
    fft_ = fft(y)
    fft_ = 2.0 / N * np.abs(fft_[0:N // 2])
    return f, fft_

def get_autocorr(y, f_s):
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    x_values = np.array([(1/f_s) * jj for jj in range(0, N)])
    return x_values, autocorr


##############################################################
################# Data parameteters ##########################
##############################################################

wave = 'sym5'
N = 2048 # data points of each observation
f_s = 128 # data sampling frequency (Hz)
dt = 1/f_s # data sampling period (sec)
N_t = N*dt # data recording time (sec)


##############################################################
################## Data preparation ##########################
##############################################################

ecg = sio.loadmat('D:/Calisma/datasets/ecg_data/ECGData.mat')
ecg_labels = list(map(lambda x: x[0][0], ecg['ECGData'][0][0][1]))

ecg_data = np.array([])
ecg_label = np.array([])

div = 32
for e, ecg_signal in enumerate(ecg['ECGData'][0][0][0]):
    ecg_label = np.append(ecg_label, [ecg_labels[e] for i in range(div)])
    if ecg_data.size == 0:
        ecg_data = np.vstack([np.array_split(ecg_signal, div)])
    else:
        ecg_data = np.vstack([ecg_data, np.array_split(ecg_signal, div)])

# Removing mean component of the data
ecg_data = np.array([data - np.mean(data) for data in ecg_data])

df_data = pd.DataFrame()
for i, data in enumerate(ecg_data):
    df_data['data_' + str(i)] = data

save_dataset(df_data, 'ecg_data', extension = '.pkl')
df_data = load_dataset('ecg_data', extension = '.pkl')


##############################################################
################# Data visualization #########################
##############################################################

# Sample observation time, frequency ve autocorrelation graphs
y = list(df_data['data_0'])

time = np.arange(0, len(y)) * (1 / f_s)
plt.plot(time, y, linestyle='-', color='blue')
plt.xlabel('Time [s]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title('Signal', fontsize=16)
plt.show()

f, fft_= get_fft(y, f_s)
plt.plot(f, fft_, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title('Frequency Transform', fontsize=16)
plt.show()

t, autocorr = get_autocorr(y, f_s)
plt.plot(t, autocorr, linestyle='-', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Autocorrelation', fontsize=16)
plt.show()

# Sample observation time, frequency ve scaleogram graphs
plot_time_freq_scaleogram(y, f_s)

# Sample observation graphs
wp = pywt.WaveletPacket(y,wave)
plot_tree(wp)
wpd_plt(df_data['data_0'])


##############################################################
######### Discrete wavelet transform features ################
##############################################################

df = pd.DataFrame()
features = []
for i in range(df_data.shape[1]):
    f = []
    coeff = pywt.wavedec(df_data['data_' + str(i)], wave)
    for c in coeff:
        f += get_features(c)
    features.append(f)
    print(str(i+1) + '/' + str(df_data.shape[1]) + ' dwt coefficient features', end="\r")
df = pd.DataFrame(features)
df.columns = ['dwt_' + str(i) for i in range(df.shape[1])]

save_dataset(df, 'ecg_feature', extension = '.pkl')
df = load_dataset('ecg_feature', extension = '.pkl')


##############################################################
###### Wavelet packet transform best basis features ##########
##############################################################

best_basis = []
for i in range(df_data.shape[1]):
    wp, basis = get_best_basis(df_data['data_' + str(i)], wave)
    best_basis.append(basis)
    print(str(i + 1) + '/' + str(df_data.shape[1]) + ', number of best basis: ' + str(len(basis)), end="\r")

len(best_basis)

wpbb_ = []
for i, bb in enumerate(best_basis):
    print(str(i + 1) + '/' + str(len(best_basis)) + ' wpt coefficient features', end="\r")
    wpbb = []
    entropy_list = []
    entropy_list = np.array([shannon_entropy(p.data) for p in bb])
    wpbb.append(np.sum(entropy_list))
    wpbb.append(np.average(entropy_list))
    wpbb.append(np.std(entropy_list))
    wpbb.append(np.var(entropy_list))
    for level in range(1,8):
        entropy_list = []
        entropy_list = np.array([shannon_entropy(p.data) for p in bb if len(p.path) == level])
        if(len(entropy_list) ==0):
            entropy_list = [0]
        wpbb.append(np.sum(entropy_list))
        wpbb.append(np.average(entropy_list))
        wpbb.append(np.std(entropy_list))
        wpbb.append(np.var(entropy_list))

    power_list = []
    power_list = np.array([np.sum((abs(p.data)) ** 2) for p in bb])
    wpbb.append(np.sum(power_list))
    wpbb.append(np.average(power_list))
    wpbb.append(np.std(power_list))
    wpbb.append(np.var(power_list))
    for level in range(1,8):
        power_list = []
        power_list = np.array([np.sum((abs(p.data))) ** 2 for p in bb if len(p.path) == level])
        if(len(power_list) ==0):
            power_list = [0]
        wpbb.append(np.sum(power_list))
        wpbb.append(np.average(power_list))
        wpbb.append(np.std(power_list))
        wpbb.append(np.var(power_list))
    wpbb_.append(wpbb)

wpbb_a = np.array(wpbb_).T
for i in range(wpbb_a.shape[0]):
    df['wpt_' + str(i)] = wpbb_a[i]

save_dataset(df, 'ecg_feature', extension = '.pkl')
df = load_dataset('ecg_feature', extension = '.pkl')

# Plotting sample wavelet packet best basis tree
y = list(df_data['data_0'])
wp, best_basis = get_best_basis(y, wave)
wpd_plt(df_data['data_0'], best_basis = [b.path for b in best_basis])

df.head()
df

##############################################################
###### Principal component analysis of the features ##########
##############################################################

# PCA model and fitting
df_pca = StandardScaler().fit_transform(df)
pca = PCA()
pca_fit = pca.fit_transform(df_pca)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

# Optimum component numbers
pca = PCA().fit(df_pca)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Component number")
plt.ylabel("Cumulative variance ratio")
plt.show()

#Final PCA
pca = PCA(n_components=60) # optimum component number: 60
pca_fit = pca.fit_transform(df_pca)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_) # explanability of 60 components: %95.

for i, pca in enumerate(pca_fit.T):
    df['pca_' + str(i)] = pca

##############################################################
################## Adding target labels ######################
##############################################################

df['y'] = LabelEncoder().fit_transform(ecg_label)
df['y_label'] = ecg_label

save_dataset(df, 'ecg_feature', extension = '.pkl')
df = load_dataset('ecg_feature', extension = '.pkl')
