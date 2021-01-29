# Attention: This does not work with the GISS model yet!
# Fully working '12month climatology' of all polynyas
import pickle
import cmocean
import utils
from matplotlib import pyplot as plt
import cartopy
import cartopy.crs as ccrs
import numpy

monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 
    6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 
    11:'Nov', 12:'Dec'}
tsDict={'monthly':{1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 
        8:8, 9:9, 10:10, 11:11, 12:12},
        'daily':{1:15, 2:45, 3:75, 4:105, 5:135, 6:165, 
        7:195, 8:225, 9:255, 10:285, 11:315, 12:345}}
freq_dict = {'daily':'day', 'monthly':'mon'}

plt.rcParams.update({'font.size': 25})
variable = 'siconc'
freq = 'daily'
threshold = '30'
scenario = 'historical'
if freq=='daily':
    divider = 30
else:
    divider = 1
resultsdictionary = utils.create_resultsdictionary()
models = list(resultsdictionary[variable][freq_dict[freq]]['30'].keys())
models.sort()
models.remove('OBS')
#breakpoint()
if variable == 'sithick':
    models.remove('CESM2-WACCM'); models.remove('CESM2-WACCM-FV2'); models.remove('MPI-ESM1-2-LR')
    models.remove('BCC-ESM1'); models.remove('MPI-ESM-1-2-HAM')
print('plotting for %s models'%len(models))

fig = plt.figure(figsize=[25,25])
fig.subplots_adjust(wspace=0.1, hspace=0.1)
#print(source_id)
print('List of models:', models)
for index, source_id in enumerate(models):
    print(source_id)
    if source_id in ['GISS-E2-1-H', 'OBS']:#, 'OBS']:
        continue
    dataset_areacello = utils.open_CMIP_variable(
        model=source_id, variable='areacello', scenario='historical', freq='daily')
    dataset_areacello = utils.homogenize_CMIP_variable(dataset_areacello)
    for i in range(9,10):
        timestep = i#tsDict[freq][i]
        sa_tot_file = open('../sa_tot_alltime/sa_tot_alltime_%s_%s_%s_%s_%s_%s'%(
            freq, scenario,threshold,source_id,variable,timestep), "rb" )
        pa_tot_file = open('../sa_tot_alltime/pa_tot_alltime_%s_%s_%s_%s_%s_%s'%(
            freq, scenario,threshold,source_id, variable, timestep), "rb" )
        print('using file  ../sa_tot_alltime/sa_tot_alltime_%s_%s_%s_%s_%s_%s'%(
            freq, scenario,threshold,source_id,variable,timestep))
        #../sa_tot_alltime/sa_tot_alltime_daily_historical_60_SAM0-UNICON_siconc_97
        # the divided by 30 I have to do here because it is not implemented correctly in the algorithm yet
        if source_id in ['ACCESS-CM2', 'MRI-ESM2-0']: #thos are the two that might not have to be divided!
            sa_tot_alltime = pickle.load(sa_tot_file)
            pa_tot_alltime = pickle.load(pa_tot_file)
        else:
            sa_tot_alltime = pickle.load(sa_tot_file)/divider
            pa_tot_alltime = pickle.load(pa_tot_file)/divider
        sa_tot_file.close()
        pa_tot_file.close()

        print('index:', index, "source_id:", source_id)

        ax = plt.subplot(5,5,index+1,projection=ccrs.Orthographic(central_latitude=-90))
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.background_patch.set_facecolor('grey')
        cmap=utils.colormap_alpha(cmocean.cm.ice)#plt.cm.Greys_r)
        colors1 = plt.pcolormesh(
                dataset_areacello.lon, 
                dataset_areacello.lat,
                sa_tot_alltime, 
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=0,
                vmax=100)

        cmap = utils.colormap_alpha(plt.cm.get_cmap('autumn', 21))
        colors2 = plt.pcolormesh(
                dataset_areacello.lon, 
                dataset_areacello.lat,
                pa_tot_alltime, 
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=0,
                vmax=20)#20

        ax.set_title(source_id, fontsize=25)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, edgecolor='k')
        ax.gridlines()
    
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
cbar_ax2 = fig.add_axes([0.90, 0.15, 0.025, 0.7])
cbar1 = fig.colorbar(colors1, cax=cbar_ax)
cbar2 = fig.colorbar(colors2, cax=cbar_ax2, extend='max')

#if variable == 'siconc':
cbar1.set_label('% probability of sea ice coverage > threshold',labelpad=-110)
#elif variable == 'sivol':
#    cbar1.set_label('mean sea ice thickness', labelpad=-110)
#elif variable == 'sithick':
#    cbar1.set_label('')
cbar2.set_label('% propability of polynya occurence',labelpad=0)
cbar_ax.yaxis.set_ticks_position('left')
plt.savefig('./plt_locations/%s_%s_%s_%s.png'%(variable, threshold, freq, monthDict[i]))#, dpi=300)
# plt.show()
    
#open('./sa_tot_alltime/sa_tot_alltime/')