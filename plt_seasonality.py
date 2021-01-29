# Attention: This does not work with the GISS model yet!
# Fully working '12month climatology' of all polynyas
import pickle
import cmocean
from utils import open_CMIP_variable, homogenize_CMIP_variable#, colormap_alpha
import utils
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy

monthDict={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
#tsDict={'monthly':{1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12},
#        'daily':{1:15, 2:45, 3:75, 4:105, 5:135, 6:165, 7:195, 8:225, 9:255, 10:285, 11:315, 12:345}}
plt.rcParams.update({'font.size': 25})

freq_dict = {'daily':'day', 'monthly':'mon'}
variable = 'sivol'
freq = 'monthly' # 'monthly'
threshold = '30'
scenario = 'historical'
resultsdictionary = utils.create_resultsdictionary()
models = list(resultsdictionary[variable][freq_dict[freq]]['30'].keys())
models.sort()
for source_id in models:
    if source_id in ['GISS-E2-1-H', 'OBS']:
        continue
    dataset_areacello = utils.open_CMIP_variable(model=source_id, variable='areacello', scenario='historical', freq='daily')
    dataset_areacello = homogenize_CMIP_variable(dataset_areacello)
    fig, axs = plt.subplots(nrows=2, ncols=6,figsize=[25, 17])
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(source_id)
    for i in range(1,13):
        timestep = i#tsDict[freq][i]
        outfile = open('../sa_tot_alltime/sa_tot_alltime_%s_%s_%s_%s_%s_%s'%(freq, scenario,threshold,source_id,variable,timestep), "rb" )
        sa_tot_alltime = pickle.load(outfile)
        outfile.close()
        outfile = open('../sa_tot_alltime/pa_tot_alltime_%s_%s_%s_%s_%s_%s'%(freq, scenario,threshold,source_id,variable,timestep), "rb" )
        pa_tot_alltime = pickle.load(outfile)
        #if pa_tot_alltime == 0.0:
        #    print('File not found for model ../sa_tot_alltime/pa_tot_alltime_%s_%s_%s_%s_%s_%s'%(freq, scenario,threshold,source_id,variable,timestep))
        #    continue
        
        if freq == 'daily':
            sa_tot_alltime = sa_tot_alltime/30
            pa_tot_alltime = pa_tot_alltime/30
        
        # breakpoint()

        print(i)

        ax = plt.subplot(3,4,i,projection=ccrs.Orthographic(central_latitude=-90))
        ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
        ax.background_patch.set_facecolor('grey')
        cmap=utils.colormap_alpha(cmocean.cm.ice)#plt.cm.Greys_r)
        # breakpoint()
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
                vmax=20)

        ax.set_title(monthDict[i], fontsize=25)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, edgecolor='k')
        ax.gridlines()
        outfile.close()
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
    cbar_ax2 = fig.add_axes([0.90, 0.15, 0.025, 0.7])
    cbar1 = fig.colorbar(colors1, cax=cbar_ax)
    cbar2 = fig.colorbar(colors2, cax=cbar_ax2, extend='max')
    
    cbar1.set_label('% mean sea ice concentration',labelpad=-110)#horizontalalignment='left')
    cbar2.set_label('% propability of polynya occurence',labelpad=0)
    cbar_ax.yaxis.set_ticks_position('left')
    # print('YEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEES', freq)
    if freq == 'monthly':
        plt.savefig('./plt_seasonality/%s/%s_%s.png'%(freq, variable, source_id))
    if freq == 'daily':
        plt.savefig('./plt_seasonality/%s/%s_%s.png'%(freq, variable, source_id))
    # plt.show()
    