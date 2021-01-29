import utils
from matplotlib import pyplot as plt
import numpy 
import pandas

resultsdictionary = utils.create_resultsdictionary()

#columns
# Ein fuer alle mal symbole und farben festlegen
monthdict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

styledict = {'color': {'CanESM5': 'C0',
  'CESM2-WACCM-FV2': 'C1',
  'CESM2-WACCM': 'C2',
  'NorCPM1': 'C3',
  'EC-Earth3': 'C4',
  'BCC-CSM2-MR': 'C5',
  'MIROC6': 'C6',
  'MPI-ESM1-2-LR': 'C7',
  'UKESM1-0-LL': 'C8',
  'EC-Earth3-Veg': 'C9',
  'MPI-ESM-1-2-HAM': 'C0',
  'NorESM2-LM': 'C1',
  'CESM2': 'C2',
  'GFDL-CM4': 'C3',
  'ACCESS-CM2': 'C4',
  'MIROC-ES2L': 'C5',
  'MPI-ESM1-2-HR': 'C6',
  'MRI-ESM2-0': 'C7',
  'IPSL-CM6A-LR': 'C8',
  'GISS-E2-1-H': 'C9',
  'NorESM2-MM': 'C0',
  'BCC-ESM1': 'C1',
  'HadGEM3-GC31-LL': 'C2',
  'ACCESS-ESM1-5': 'C3',
  'CAMS-CSM1-0': 'C4',
  'CNRM-ESM2-1': 'C5',
  'CNRM-CM6-1': 'C6',
  'SAM0-UNICON': 'C7',
  'CESM2-FV2': 'C8',
  'GFDL-ESM4': 'C9',
  'OBS': 'C0'},
 'marker': {'CanESM5': 'o',
  'CESM2-WACCM-FV2': 'v',
  'CESM2-WACCM': '^',
  'NorCPM1': '<',
  'EC-Earth3': '>',
  'BCC-CSM2-MR': '8',
  'MIROC6': 's',
  'MPI-ESM1-2-LR': 'p',
  'UKESM1-0-LL': '*',
  'EC-Earth3-Veg': 'h',
  'MPI-ESM-1-2-HAM': 'H',
  'NorESM2-LM': 'D',
  'CESM2': 'd',
  'GFDL-CM4': 'P',
  'ACCESS-CM2': 'X',
  'MIROC-ES2L': 'o',
  'MPI-ESM1-2-HR': 'v',
  'MRI-ESM2-0': '^',
  'IPSL-CM6A-LR': '<',
  'GISS-E2-1-H': '>',
  'NorESM2-MM': '8',
  'BCC-ESM1': 's',
  'HadGEM3-GC31-LL': 'p',
  'ACCESS-ESM1-5': '*',
  'CAMS-CSM1-0': 'h',
  'CNRM-ESM2-1': 'H',
  'CNRM-CM6-1': 'D',
  'SAM0-UNICON': 'd',
  'CESM2-FV2': 'P',
  'GFDL-ESM4': 'X',
  'OBS':'o'}}

variable = 'sivol'
period = 'mon'
threshold = '30'

models = list(resultsdictionary[variable][period][threshold].keys())
models.sort()

import itertools
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 14})
my_marker_cycle = itertools.cycle(list(Line2D.filled_markers))
my_color_cycle = itertools.cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
my_marker_cycle_list = []
my_color_cycle_list = []
for i in range(0,len(models)):
    my_marker_cycle_list.append(my_marker_cycle.__next__())
    my_color_cycle_list.append(my_color_cycle.__next__())
#mt2['color'] = my_color_cycle_list
#mt2['marker'] = my_marker_cycle_list
# defining the coastal polynyas




fig, axs = plt.subplots(nrows=2, ncols=3, figsize=[20,20], sharex=True)

axs = axs.flatten()
for j,col in enumerate(['sa_we', 'pa_we_co', 'pa_we_op']):#columns:
    iterative_mean = 0
    
    # PLOT MODEL RESULTS
    for index, model in enumerate(models):
        label = model if j==0 else None
        ma = resultsdictionary[variable][period][threshold][model][col].groupby(
             resultsdictionary[variable][period][threshold][model][col].index.month).describe()
        iterative_mean += ma['mean']
        axs[j].plot(
            ma['mean']/10**12, 
            alpha=0.2, 
            c=styledict['color'][model], 
            marker=styledict['marker'][model],)
        points = axs[j].scatter(
            x = ma.index, 
            y = ma['mean']/10**12, 
            c=styledict['color'][model], 
            marker=styledict['marker'][model],
            edgecolor='k', 
            s=100, 
            label=label)
    # plotting the iterative multi-model-mean
    label = 'multi model mean' if j==0 else None 
    axs[j].plot(iterative_mean/len(models)/10**12, lw=5, color="k", ls='--', label=label)

    """
    # PLOT OSI RESULTS
    if col in dfre[threshold].keys():
        label = 'OSI-450' if j==0 else None
        ma = dfre[threshold][col].groupby(dfre[threshold][col].index.month).describe()
        lines = axs[j].plot(range(1,13),ma['mean']/10**12, label=label, lw=5, color="k")
        color = lines[0].get_color()
        axs[j].fill_between(x=range(1,13),y1=numpy.array((ma['mean']-2*ma['std'])/10**12).clip(min=0), y2=(ma['mean']+2*ma['std'])/10**12, alpha=0.2, color='grey')
    else:
        print('dfre is missing the %s column'%col)
    if col in descriptions.keys():
        axs[j].set_title('%s) '%abc[j]+descriptions[col])
    else:
        axs[j].set_title(col)
    """

    #plt.legend(loc=(1.1,0))
    # plt.title(pandas_dfs[col]['description'][0])
    axs[j].set_xlabel('month')
    #axs[j].set_xticks(numpy.array(list(monthdict.keys()))+1,list(monthdict.values()))#monthdict.keys, monthdict.values)
    axs[j].set_xticks(range(1,13))
    axs[j].set_xticklabels(list(monthdict.values()))
    axs[j].set_ylabel('area in mio km²')
    axs[j].set_xlim(0,13)
    axs[0].set_title('Sea ice area in \n the Weddell Sea')
    axs[1].set_title('Area of coastal polynyas \n in the Weddell Sea')
    axs[2].set_title('Area of OWP in \n the Weddell Sea')
    # axs[j].set_yscale('log')
    if col in ['pa_we_co', 'pa_we_op']:
        axs[j].set_ylim(0,0.15)




##########
variable = 'siconc'
period = 'day'
threshold = '30'

models = list(resultsdictionary[variable][period][threshold].keys())
models.sort()
axs = axs.flatten()
for j,col in enumerate(['sa_we', 'pa_we_co', 'pa_we_op']):#columns:
    iterative_mean = 0
    
    # PLOT MODEL RESULTS
    for index, model in enumerate(models):
        label = model if j==0 else None
        ma = resultsdictionary[variable][period][threshold][model][col].groupby(
             resultsdictionary[variable][period][threshold][model][col].index.month).describe()
        iterative_mean += ma['mean']
        axs[j+3].plot(
            ma['mean']/10**12, 
            alpha=0.2, 
            c=styledict['color'][model], 
            marker=styledict['marker'][model],)
        points = axs[j+3].scatter(
            x = ma.index, 
            y = ma['mean']/10**12, 
            c=styledict['color'][model], 
            marker=styledict['marker'][model],
            edgecolor='k', 
            s=100, 
            label=label)
    # plotting the iterative multi-model-mean
    label = 'multi model mean' if j==0 else None 
    axs[j+3].plot(iterative_mean/len(models)/10**12, lw=5, color="k", ls='--', label=label)

    """
    # PLOT OSI RESULTS
    if col in dfre[threshold].keys():
        label = 'OSI-450' if j==0 else None
        ma = dfre[threshold][col].groupby(dfre[threshold][col].index.month).describe()
        lines = axs[j].plot(range(1,13),ma['mean']/10**12, label=label, lw=5, color="k")
        color = lines[0].get_color()
        axs[j].fill_between(x=range(1,13),y1=numpy.array((ma['mean']-2*ma['std'])/10**12).clip(min=0), y2=(ma['mean']+2*ma['std'])/10**12, alpha=0.2, color='grey')
    else:
        print('dfre is missing the %s column'%col)
    if col in descriptions.keys():
        axs[j].set_title('%s) '%abc[j]+descriptions[col])
    else:
        axs[j].set_title(col)
    """

    #plt.legend(loc=(1.1,0))
    # plt.title(pandas_dfs[col]['description'][0])
    axs[j+3].set_xlabel('month')
    #axs[j].set_xticks(numpy.array(list(monthdict.keys()))+1,list(monthdict.values()))#monthdict.keys, monthdict.values)
    axs[j+3].set_xticks(range(1,13))
    axs[j+3].set_xticklabels(list(monthdict.values()))
    axs[j+3].set_ylabel('area in mio km²')
    axs[j+3].set_xlim(0,13)
    axs[0+3].set_title('Sea ice area in \n the Weddell Sea')
    axs[1+3].set_title('Area of coastal polynyas \n in the Weddell Sea')
    axs[2+3].set_title('Area of OWP in \n the Weddell Sea')
    # axs[j].set_yscale('log')
    if col in ['pa_we_co', 'pa_we_op']:
        axs[j+3].set_ylim(0,0.15)
##########





fig.subplots_adjust(bottom=0.2)
fig.legend(loc='lower center', bbox_to_anchor=(0.4,-0.00),
          fancybox=True, shadow=True, ncol=5, fontsize=10)
plt.savefig('./plt_seasonality_curves/seasonality_%s_%s_%s_multivariable.png'%(variable, period, threshold), dpi=100)
#plt.tight_layout(rect=(0, 0.15, 1, 1), h_pad=0.2, w_pad=0.2)
    #if col in ['pa_we', 'pa_we_op']:
    #    plt.ylim(0,1e11)