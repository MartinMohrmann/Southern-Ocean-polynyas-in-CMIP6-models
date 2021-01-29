# this is the unaltered version for the monthly data seasonality plots "hotbars"
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import colormap_alpha
import utils
import pandas
import numpy

plt.rcParams.update({'font.size': 18})
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False

variable = 'sivol'
freq = 'mon'
threshold = '30'
scenario = 'historical'
resultsdictionary = utils.create_resultsdictionary()

fig, axs = plt.subplots(figsize=[17,12], nrows=len(resultsdictionary[variable][freq][threshold].keys()), sharex=True)
fig.subplots_adjust(hspace=0.3)
models = list(resultsdictionary[variable][freq][threshold].keys())
models.sort()
models.remove('OBS')
models.append('OBS')

for index, model in enumerate(models):#resultsdictionary[threshold].keys()):
    #if model == 'OBS':
    #    continue
    df = resultsdictionary[variable][freq][threshold][model]
    df = df[(df.index.month >= 5) & (df.index.month <= 11) & (
            df.index.dayofyear < 320)].groupby(pandas.Grouper(freq='A')).mean()
    dfwinter_stacked = numpy.vstack((
        df['pa_we_op'], 
        df['pa_we_co']))
    vmax = 1e11
    vmin = 1e8
    im = axs[index].imshow(dfwinter_stacked+0.2, aspect=2, cmap='hot', extent=[df.index.min().year, df.index.max().year, 0,2], vmin=vmin, vmax=vmax)
    axs[index].set_yticklabels([])
    axs[index].set_yticks([])
    
    if index==-1:
        axs[index].xaxis.set_tick_params(labeltop='on', labelbottom='off')
        axs[index].set_xticks(ticks=range(0,165,20))
        axs[index].set_xticklabels(range(0,165,20))
        axs[index].set_xlabel('year')
    else:
        axs[index].set_xticks([])
        axs[index].set_xticklabels([])
        axs[index].minorticks_on()#(range(1850,2015))
        axs[index].tick_params(axis='y',which='minor',bottom=False)    
    cmap = plt.cm.get_cmap('hot')
    axs[index].text(x=2022, y=0, s="%s"%model)

plt.xlim(1850, 2020)
axs[-1].set_xticks(range(1850,2020,20))
axs[-1].set_xticklabels(range(1850,2020,20))

cbar = fig.colorbar(im, ax=[axs[:]], location='left', extend='max', shrink=1, pad=0.02)
cbar.ax.set_yticklabels(cbar.get_ticks()/1e12)

fig.text(x=0.15, y=0.59, s='polynya area in million km²', va='center', rotation='vertical', fontsize=18)

plt.savefig("./plt_hotbars/hotbars_%s_%s_%s.png"%(variable, freq, threshold), bbox_inches = 'tight',
    pad_inches = 0, dpi=300)
