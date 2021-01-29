from skimage import data, filters
from skimage.segmentation import flood, flood_fill
from matplotlib import pyplot as plt
import cartopy
import xarray
import numpy
import resource
from mpl_toolkits.axes_grid1 import AxesGrid
import warnings
import datetime
import pickle
import os
import glob
import pandas
import cmocean
import pickle
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib.patches import Patch
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

months_length = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30,
                 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

def colormap_alpha(cmap):
    my_cmap = cmap(numpy.arange(cmap.N))
    my_cmap[:,-1] = numpy.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    return my_cmap

def limit_memory(maxsize):
    """Limit memory to the given value (MB's) """
    maxsize = maxsize * 1024 * 1024 # function works with bytes
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def open_CMIP_variable(model, variable, scenario, freq):
    freq = 'day' if freq == 'daily' else 'mon'
    # for the GISS-models, areacella available only and is the sea ice grid
    if variable in ['siconc', 'sivol', 'sithick']:
        variantsdict = {'CNRM-CM6-1':'r1i1p1f2','CNRM-ESM2-1':'r1i1p1f2','UKESM1-0-LL':'r1i1p1f2', 'MIROC-ES2L':'r1i1p1f2','HadGEM3-GC31-LL':'r1i1p1f3'}
        if model in variantsdict:
            variant = variantsdict[model]
        else:
            variant = 'r1i1p1f1'
        filelist = glob.glob('../%s/%s/%s_SI%s*%s*.nc'%(model,scenario,variable,freq[0:3],variant))
        print('matching files with: ../%s/%s/%s_SI%s*%s*.nc'%(model,scenario,variable,freq[0:3],variant))
        if filelist:
            #print('progressing with the following input files:\n%', sorted(filelist))
            dataset = xarray.open_mfdataset(sorted(filelist))
        else:
            #print('no input files found with are matching ../%s/%s/%s_SI%s*%s*.nc'%(model,scenario,variable,freq[0:3],variant))
            return 0,0
    elif variable == 'areacello':
            try:
                dataset = xarray.open_mfdataset('../%s/areacello*historical*.nc'%(model))
            except:
                dataset = xarray.open_mfdataset('../%s/areacello*.nc'%(model))
    else:
        try:
            dataset = xarray.open_mfdataset('../%s/%s/%s*.nc'%(model, scenario, variable))
        except:
            print('no sourcefiles for variable %s, model %s, scenario %s'%(variable, model, scenario))
            return none
    return dataset


def homogenize_CMIP_variable(dataset):
    # print(dataset)
    # homogenize the naming schemes to lat, lon and lev
    lat_key = list({'lat', 'nav_lat', 'latitude'} & (set(dataset.data_vars) | set(dataset.coords)))[0]
    lon_key = list({'lon', 'nav_lon', 'longitude'} & (set(dataset.data_vars) | set(dataset.coords)))[0]
    lev_key = list({'lev', 'olevel'} & (set(dataset.data_vars) | set(dataset.coords)))
    if lon_key and dataset.source_id not in ['BCC-CSM2-MR','BCC-ESM1', 'GISS-E2-1-H']:
        dataset = dataset.rename(
            {lat_key: 'lat', lon_key: 'lon'}).set_coords(
            ['lon', 'lat'])
    # is there a depth coordinate also in dataset (i.e. areacello wont have one)
    if lev_key:
        dataset = dataset.rename(
            {'olevel': 'lev'}).set_coords(s
            ['lev'])
    try:
        dataset_areacello = dataset.rename(
            {'nj': 'j', 'ni': 'i'}).set_coords(
            ['i', 'j'])
    except:
        pass
    # This might be a bit of a brave change, but rolling all lon coordinates to -180,180
    if dataset is not None:
        dataset['lon'] = dataset.lon.where(dataset.lon<180, dataset.lon-360)
    return dataset

def compute_CMIP_areas(
    dataset, 
    dataset_areacello,
    scenario,
    threshold_conc, 
    threshold_thick,
    freq,
    timelimit=1980, 
    createplots=True,
    seasonalmaps=True):
    
    if freq == 'monthly':
        sa_tot_alltime = dict.fromkeys(range(1,13), 0)
        pa_tot_alltime = dict.fromkeys(range(1,13), 0)
    elif freq == 'daily':
        sa_tot_alltime = dict.fromkeys(range(1,365), 0)
        pa_tot_alltime = dict.fromkeys(range(1,365), 0)
    else:
        print('invalid value for freq!')

    resultsdictionary = dict(date=[])
    areas = {'pa_tot':'Wistia', 'sa_tot':'Reds', 'pa_we':'Wistia', 
             'sa_we':'Greens', 'pa_we_op':'winter', 'pa_op':'winter'}
    colorscheme = ["Wistia","Greens","Wistia","winter","winter"]
    for area in areas.keys():
        resultsdictionary[area] = []
        resultsdictionary[area+'_hf'] = []
        resultsdictionary[area+'_mlotst'] = []

    # find a spot in the open ocean (and fastland antarctica) to start the freezing over floodfill
    lat = -40; lon = 90;
    variable_id = dataset['seaice'].variable_id
    source_id = dataset['seaice'].source_id
    
    if source_id == 'GISS-E2-1-H':
        areacellvar = 'areacella'
    else:
        areacellvar = 'areacello'
        
    newarray =  (numpy.abs(dataset['seaice']['lat']-lat)**2 + numpy.abs(dataset['seaice']['lon']-lon)**2)
    j,i = numpy.unravel_index(numpy.argmin(newarray, axis=None), newarray.shape)
    
    if variable_id == 'siconc':
        floodfillvalue = 100
        tolerance = threshold_conc
        tolerance_sea_ice = threshold_conc
        vmin=0; vmax=100
    elif variable_id in ['sivol', 'sithick']:
        floodfillvalue = 1
        tolerance = threshold_thick
        tolerance_sea_ice = threshold_thick
        vmin=0; vmax=0.5

    for t in range(0,timelimit):
        # print(t)
        # progress print
        if t%100 == 0:
            print('model:%s, timestep:%s'%(source_id, t))

        date = datetime.datetime(dataset['seaice']['time'][t].dt.year, 
                      dataset['seaice']['time'][t].dt.month, 
                      dataset['seaice']['time'][t].dt.day)
        resultsdictionary['date'].append(date)
        if createplots:
            fig = plt.figure(figsize=[20, 7.5])
            ax = plt.subplot(1,4,1,projection=ccrs.Orthographic(central_latitude=-90))
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
            ax.background_patch.set_facecolor('grey')
            plot_dataset(dataset['seaice'][variable_id][t], dataset_areacello, 
                         tolerance, cmap=colormap_alpha(cmocean.cm.ice), #my_cmap,#cmap=cmocean.cm.ice, 
                         title="sea ice plot (%s)"%variable_id, 
                         ax=ax,
                         vmin=vmin, vmax=vmax)
            ax.add_feature(cartopy.feature.LAND, edgecolor='k')
            legend_elements = [Patch(facecolor='azure', edgecolor='k',label='sea ice covered'),
                               Patch(facecolor='cornflowerblue', edgecolor='k', label='partly covered'),
                               Patch(facecolor='black', edgecolor='k', label='open water')]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=14)

            ax = plt.subplot(1,4,2,projection=ccrs.Orthographic(central_latitude=-90))
            ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
            ax.background_patch.set_facecolor('grey')
            plot_dataset(dataset['seaice'][variable_id][t], dataset_areacello, 
                         tolerance, cmap=colormap_alpha(cmocean.cm.ice), 
                         title="sea ice plot (%s)"%variable_id, 
                         ax=ax,
                         vmin=vmin, vmax=vmax)
            ax.add_feature(cartopy.feature.LAND, edgecolor='k')
        
        floodedarray =  flood_fill(
            dataset['seaice'][variable_id][t].values, (j, i), floodfillvalue, tolerance=tolerance)
        if source_id == 'GISS-E2-1-H':
            antarcticamask=flood_fill(image=dataset['seaice'][variable_id][9].values, seed_point=(0,0), new_value=numpy.nan)
            # Fill antartic continent with numpy.nan
            siconc2 = numpy.where(numpy.isnan(antarcticamask), antarcticamask, floodedarray)
        else:
            siconc2 = floodedarray
        # list of models with one dimensional latitude/longitude errors instead of array
        if t==0:
            if source_id in ['BCC-CSM2-MR','BCC-ESM1', 'GISS-E2-1-H']:
                latitudes, longitudes = [a.T for a in numpy.meshgrid(dataset_areacello.lat, dataset_areacello.lon)]
            else:
                longitudes, latitudes = dataset_areacello.lon, dataset_areacello.lat
        
        masks = {}

        with numpy.errstate(invalid='ignore'):
            numpy.seterr(divide='ignore', invalid='ignore')
            # some sea ice distributions will be cut of at 55°S, but sea ice that far north would
            # be unrealistic anyway
            # create all the different masks for polynyas and different sea ice sectors

            if t==0:
                if source_id in ['BCC-CSM2-MR','BCC-ESM1', 'GISS-E2-1-H']:
                    weddellsection = ((latitudes<-55) & ((longitudes>-65) & (longitudes<30)))
                    southernbound = (latitudes<-55)
                else:
                    weddellsection = ((latitudes<-55) & ((longitudes>-65) & (longitudes<30))).values
                    southernbound = (latitudes<-55).values
                
            masks['sa_tot'] = numpy.bitwise_and(dataset['seaice'][variable_id][t].values>tolerance_sea_ice, southernbound)
            masks['sa_we'] = (dataset['seaice'][variable_id][t].values>tolerance_sea_ice) & southernbound & weddellsection
            
            masks['pa_tot'] = (siconc2<tolerance) & southernbound #& southernbound
            masks['pa_we'] = (siconc2<tolerance) & southernbound & weddellsection

            # Now turn all NaN areas of central antarctica to zero, 
            # but keep other continents NaN
            siconc2 = numpy.where((siconc2>0) & (latitudes<-25),siconc2,0)#-55),siconc2,0)
            # fill Antarctica inclusive coastal polynyas with sea ice 
            # The MPI count the y-index from the north instead of from the south upwards
            if source_id in ['MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR']:
                siconc2 =  flood_fill(numpy.array(siconc2), (len(siconc2[:-1,0]), 0), floodfillvalue, tolerance=tolerance)
            else:
                siconc2 =  flood_fill(numpy.array(siconc2), (0, 0), floodfillvalue, tolerance=tolerance)
            
            masks['pa_op'] = (siconc2<tolerance) & southernbound #& southernbound
            masks['pa_we_op'] = (siconc2<tolerance) & southernbound & weddellsection
            
            for mask in masks.keys():
                area = dataset_areacello[areacellvar].where(masks[mask])
                area_sum = area.sum(skipna=True)
                resultsdictionary[mask].append(float(area_sum.values))
            
            if seasonalmaps:
                pa_tot_alltime[resultsdictionary['date'][t].month] += masks['pa_tot'].astype(int)
                sa_tot_alltime[resultsdictionary['date'][t].month] += masks['sa_tot'].astype(int)

            if createplots:
                # sea ice
                plot_dataset(dataset_areacello['areacello'].where(masks['sa_tot']), dataset_areacello, tolerance, cmap=areas['sa_tot'], title="", ax=ax)
                plot_dataset(dataset_areacello['areacello'].where(masks['sa_we']), dataset_areacello, tolerance, cmap=areas['sa_we'], title="", ax=ax)
                # plot polynyas
                plot_dataset(dataset_areacello['areacello'].where(masks['pa_tot']), dataset_areacello, tolerance, cmap=areas['pa_tot'], title="", ax=ax)
                # plot OWPs
                plot_dataset(dataset_areacello['areacello'].where(masks['pa_op']), dataset_areacello, tolerance, cmap=areas['pa_op'], title="", ax=ax)                
                legend_elements = [Patch(facecolor='orange', edgecolor='k',label='coastal polynya'),
                                   Patch(facecolor='darkred', edgecolor='k', label='sea ice area'),
                                   Patch(facecolor='darkgreen', edgecolor='k', label='sea ice area Wed. Sea'),
                                   Patch(facecolor='lightgreen', edgecolor='k', label='open water Polynya')]
                ax.legend(handles=legend_elements, loc='lower left', fontsize=14)
                plt.title(resultsdictionary['date'][t].month)
                plt.savefig('./areafigures/%s/%s/%s_thresh%sthick%s_%s_%s-%s.png'%(
                    scenario,source_id,source_id, threshold_conc, threshold_thick ,variable_id,resultsdictionary['date'][t].year,
                             str(resultsdictionary['date'][t].month).zfill(2)))
                plt.show()
                plt.close('all')
                
    print('finished mainloop')
    if freq == 'daily':
        month_or_yeardays = range(1,365)
        divider = 365/months_length[i]
    elif freq == 'monthly':
        month_or_yeardays = range(1,13)
        divider = 12
    
    if seasonalmaps:
        for i in month_or_yeardays:
            # Save a dictionary into a pickle file, the maps could be plotted imidiately alternatively
            outfile = open('./sa_tot_alltime/sa_tot_alltime_%s_%s_%s_%s_%s_%s'%(freq, scenario,threshold_conc,source_id,variable_id,str(i)), "wb" )
            pickle.dump(sa_tot_alltime[i]/(
                timelimit/divider)*100,outfile)
            # 30/ *100
            outfile.close()
            outfile = open('./sa_tot_alltime/pa_tot_alltime_%s_%s_%s_%s_%s_%s'%(freq, scenario,threshold_conc,source_id,variable_id,str(i)), "wb" )
            pickle.dump(pa_tot_alltime[i]/(timelimit/divider)*100,outfile)
            outfile.close()
    
    print('finished processing %s'%source_id)
    for key in dataset.keys():
        if dataset[key]:
            dataset[key].close()
    dataset_areacello.close()
    return resultsdictionary


def plot_dataset(dataset, dataset_areacello, tolerance, cmap, title ,ax, vmin=0, vmax=1):
    colors1 = plt.pcolormesh(
            dataset_areacello.lon, 
            dataset_areacello.lat,
            dataset, 
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax)
    
    ax.set_title(title, fontsize=30)
    ax.coastlines()
    if not title=='d':
        ax.add_feature(cartopy.feature.LAND, edgecolor='k')
    ax.gridlines()
    return colors1


def load_CMIP_results(model):
    """find the Polynya and Seaice areas for the CMIP6 models"""
    if os.path.isfile('./polynyaareapickles1/complete_resultsdictfile'+model):
        with open('./polynyaareapickles/complete_resultsdictfile'+model, 'rb') as resultsdictionaryfile:
            resultsdictionary = pickle.load(resultsdictionaryfile)

            
def save_CMIP_results(dataset, model, scenario, resultsdictionary, threshold_conc, threshold_thick, freq):
    with open('./polynyaareapickles/complete_resultsdictfile{scenario}_{model}_{threshold_conc}_{threshold_thick}_{variable}_{freq}'.format(
            scenario=scenario, 
            model=model, 
            threshold_conc=threshold_conc, 
            threshold_thick=threshold_thick, 
            variable=dataset['seaice'].variable_id, 
            freq=freq), 'wb') as resultsdictionaryfile:
        pickle.dump(resultsdictionary, resultsdictionaryfile)

def polynya_area_ecmwf(variable, freq, threshold_conc, threshold_thick, createplots, saveresults):
    # This is very similar to compute_CIMIP_areas() but works for the observational datasets instead of CMIP
    import netCDF4
    from matplotlib import pyplot as plt
    import numpy
    from skimage import data, filters
    from skimage.segmentation import flood, flood_fill
    import glob

    if variable == 'siconc':
        floodfillvalue = 100
        tolerance = threshold_conc
    elif variable in ['sivol', 'sithick']:
        floodfillvalue = 100 
        tolerance = threshold_thick*100

    pa_tot_list = []
    sa_tot_list  = []
    pa_we_list = []
    sa_we_list = []
    pa_op_list = []
    pa_we_op_list = []
    date_list = []
    sa_tot_alltime = dict.fromkeys([1,2,3,4,5,6,7,8,9,10,11,12], 0)
    pa_tot_alltime = dict.fromkeys([1,2,3,4,5,6,7,8,9,10,11,12], 0)

    dataset = netCDF4.Dataset('../ECMWF/historical/ice_conc_sh_ease-125_cont-reproc_201509151200.nc')
    model = 'ecmwf'
    startyear = 1979 if variable == 'siconc' else 2010
    for year in numpy.arange(startyear,2021,1):
        print(year)
        for month in numpy.arange(1,13,1):
            for day in numpy.arange(1,32,1):
                if freq == 'daily':
                    datefilter = str(year)+str(month).zfill(2)+str(day).zfill(2)
                elif freq == 'monthly':
                    datefilter = str(year)+str(month).zfill(2)+'*'
                else:
                    print('invalid value for freq!')
                if variable == 'siconc':
                    minlat = -58
                    key = 'ice_conc' # original key from the OSI450 data
                    filelist = glob.glob('../ECMWF/historical/ice_conc_sh_ease2*_%s*.nc'%datefilter)
                elif variable in ['sivol', 'sithick']:
                    minlat = -63
                    key = 'thickness' # original from the thin ice product data
                    filelist = glob.glob('../ECMWF/historical/%s*_hvsouth_rfi_l1c.nc'%datefilter)
                if filelist:
                    # some dates are note available in the early years, but about every other one is
                    dataset = xarray.open_mfdataset(filelist, concat_dim='time')
                    if variable in ['sivol', 'sithick']:
                        dataset['lat'] = xarray.open_dataset('../ECMWF/south_lat_12km.hdf')['Latitude [degrees]']
                        dataset['lon'] = xarray.open_dataset('../ECMWF/south_lon_12km.hdf')['Longitude [degrees]']
                        dataset['area']= xarray.open_dataset('../ECMWF/PolStereo_GridCellArea_s12.5km_Antarctic.nc').data
                else:
                    # print('DID NOT SUCCESS WITH OPENING:')
                    # print('../ECMWF/historical/%s*_hvsouth_rfi_l1c.nc'%datefilter)
                    if freq == 'daily':
                        continue 
                    else:
                        break

                siconc = dataset[key][:].mean(axis=0) if freq == 'monthly' else dataset[key][0]
                
                # this is changing the masked regions in the sea ice thickness product from -2 to numpy.nan
                if variable in ['sivol', 'sithick']:
                    siconc=numpy.where(
                        numpy.array(siconc)<-1.8,
                        numpy.nan,numpy.array(siconc))

                # --- plotting part
                if createplots:
                    fig = plt.figure(figsize=[10, 7])
                    ax = plt.subplot(1,2,1,projection=ccrs.Orthographic(central_latitude=-90))
                    ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
                    ax.background_patch.set_facecolor('grey')
                    ax.stock_img()
                    ax.coastlines()
                    colors1 = plt.pcolormesh(
                            dataset['lon'], 
                            dataset['lat'],
                            siconc,
                            transform=ccrs.PlateCarree(),
                            cmap=colormap_alpha(cmocean.cm.ice))
                    # This second pcolormap is completely invisble (zorder), but it produces the non-
                    # transparent artifact free colormap
                    colors2 = plt.pcolormesh(
                            dataset['lon'], 
                            dataset['lat'],
                            siconc,
                            transform=ccrs.PlateCarree(),
                            cmap=cmocean.cm.ice,
                            alpha=1,
                            zorder=-1000)

                    cax = fig.add_axes([0.1, 0.2, 0.4, 0.04])
                    cbar = plt.colorbar(colors2, cax=cax, orientation="horizontal")#ax=[ax], location='left')
                    cbar.ax.set_xlabel('sea ice concentration in %')
                    #ax.legend(handles=legend_elements, loc='lower left', fontsize=14)
                    ax2 = plt.subplot(1,2,2,projection=ccrs.Orthographic(central_latitude=-90))
                    ax2.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
                    ax2.background_patch.set_facecolor('grey')
                    ax2.coastlines()
                    colors2 = plt.pcolormesh(
                            dataset['lon'], 
                            dataset['lat'],
                            siconc,
                            transform=ccrs.PlateCarree(),
                            cmap=colormap_alpha(cmocean.cm.ice))
                    # --- end plotting part

                # Here I flod from the outside, the world ocean becomes ice
                if variable == 'siconc':
                    i, j = 0, 400
                    floodedarray =  flood_fill(
                        siconc.values, (j, i), floodfillvalue, tolerance=tolerance)
                else:
                    i, j = 0, 300
                    # breakpoint()
                    floodedarray =  flood_fill(
                        siconc, (j, i), floodfillvalue, tolerance=tolerance)
                # in the Lambert_Azimuthal_Grid, it is dA=dx*dy
                # Since areas don't change between grid cells, 
                # visualization and intercomparison operations are greatly 
                # simplified and analysis is more convenient. (https://nsidc.org/ease/ease-grid-projection-gt)

                # uniform grid cell area for the siconc product:
                cellarea_siconc = 25e3**2
                # compute pa_tot
                pa_tot_mask = ((floodedarray <= tolerance) & 
                    (numpy.array(dataset['lat'])<minlat)) * 1
                pa_tot_pixels = pa_tot_mask.sum()
                pa_tot = pa_tot_pixels*cellarea_siconc

                # compute sa_tot
                sa_tot_mask = (siconc >= tolerance) * 1
                sa_tot_pixels = sa_tot_mask.sum()
                sa_tot = sa_tot_pixels*cellarea_siconc

                # compute sa_we
                sa_we_mask = ((siconc >= tolerance) & 
                  (numpy.array(dataset['lon']) < 30) & 
                  (numpy.array(dataset['lon']) > -65)) * 1
                sa_we_pixels = sa_we_mask.sum()
                sa_we = sa_we_pixels*cellarea_siconc

                # compute pa_we
                pa_we_mask = ((floodedarray <= tolerance) & 
                  (numpy.array(dataset['lon']) < 30) & 
                  (numpy.array(dataset['lon']) > -65) &
                  (numpy.array(dataset['lat']) < minlat)) * 1
                pa_we_pixels = pa_we_mask.sum()
                pa_we = pa_we_pixels*cellarea_siconc

                if variable == 'siconc':
                    i, j = 216, 216
                else:
                    i, j = 300, 300

                booleanarray = floodedarray>0
                siconc2 = floodedarray*booleanarray
                if variable == 'siconc':
                    siconc2 = numpy.where((floodedarray>0),floodedarray,0)
                else:
                    siconc2 = numpy.where(numpy.isnan(floodedarray), 0, floodedarray)
                siconc2 =  flood_fill(numpy.array(siconc2), (i, j), floodfillvalue, tolerance=tolerance)

                # compute pa_we_op
                pa_we_op_mask = ((siconc2 < tolerance) & 
                  (numpy.array(dataset['lon']) < 30) & 
                  (numpy.array(dataset['lon']) > -65) & 
                  (numpy.array(dataset['lat']) < minlat) &
                  numpy.invert(numpy.isnan(dataset[key][0]))) * 1
                pa_we_op_pixels = pa_we_op_mask.sum()
                pa_we_op = pa_we_op_pixels*cellarea_siconc

                # compute pa_op
                pa_op_mask = ((siconc2 < tolerance) & 
                  (numpy.array(dataset['lat'])<minlat) &
                   numpy.invert(numpy.isnan(dataset[key][0]))) * 1
                pa_op_pixels = pa_op_mask.sum()
                pa_op = pa_op_pixels*cellarea_siconc

                
                if variable=='sivol':
                    pa_tot = dataset['area'].where(pa_tot_mask).sum(skipna=True)*1e6
                    sa_tot = dataset['area'].where(sa_tot_mask).sum(skipna=True)*1e6
                    sa_we = dataset['area'].where(sa_we_mask).sum(skipna=True)*1e6
                    pa_we = dataset['area'].where(pa_we_mask).sum(skipna=True)*1e6
                    pa_we_op = dataset['area'].where(numpy.array(pa_we_op_mask)).sum(skipna=True)*1e6
                    pa_op = dataset['area'].where(numpy.array(pa_op_mask)).sum(skipna=True)*1e6

                pa_tot_alltime[month] += pa_tot_mask.astype(int)
                sa_tot_alltime[month] += sa_tot_mask.astype(int)

                if createplots:
                    # --- start second part of plotting routine
                    ax2.pcolormesh(dataset['lon'], 
                                   dataset['lat'], 
                                   sa_tot_mask, 
                                   transform=ccrs.PlateCarree(),
                                   cmap=colormap_alpha(cmap=plt.cm.Reds))
                    ax2.pcolormesh(dataset['lon'], 
                                   dataset['lat'], 
                                   sa_we_mask, 
                                   transform=ccrs.PlateCarree(),
                                   cmap=colormap_alpha(cmap=plt.cm.Greens))
                    ax2.pcolormesh(dataset['lon'], 
                                   dataset['lat'], 
                                   pa_tot_mask, 
                                   transform=ccrs.PlateCarree(),
                                   cmap=colormap_alpha(cmap=plt.cm.Wistia))
                    ax2.pcolormesh(dataset['lon'], 
                                   dataset['lat'], 
                                   pa_op_mask, 
                                   transform=ccrs.PlateCarree(),
                                   cmap=colormap_alpha(cmap=plt.cm.winter))

                    legend_elements = [Patch(facecolor='orange', edgecolor='k',label='coastal polynya'),
                            Patch(facecolor='darkred', edgecolor='k', label='sea ice area'),
                            Patch(facecolor='darkgreen', edgecolor='k', label='sea ice area Wed. Sea'),
                            Patch(facecolor='lightgreen', edgecolor='k', label='open water Polynya')]
                    ax2.legend(handles=legend_elements, loc='lower left', fontsize=14, bbox_to_anchor=(0.2, -0.3))

                    plt.savefig('./areafigures/%s/%s/%s_thresh%sthick%s_%s_freq%s_%s-%s-%s.png'%(
                        'historical','ecmwf', 'ecmwf', 
                        threshold_conc, threshold_thick, variable, freq ,year,str(int(month)).zfill(2), str(int(day)).zfill(2)))

                    plt.show()
                    plt.close('all')
                    # --- end second part of plotting routine
                
                #polynyaarealist.append(polynyaarea)
                #seaicearealist.append(seaicearea)
                date_list.append(datetime.datetime(year,month,day))
                pa_tot_list.append(float(pa_tot))
                sa_tot_list.append(float(sa_tot))
                pa_we_list.append(float(pa_we))
                sa_we_list.append(float(sa_we))
                pa_we_op_list.append(float(pa_we_op))
                pa_op_list.append(float(pa_op))
                
                dataset.close()
                if freq == 'monthly':
                    break


    for i in range(1,13):
        pickle.dump(sa_tot_alltime[i]/len(numpy.arange(startyear,2020,1))*100, open(
            './sa_tot_alltime/sa_tot_alltime_%s_historical_%s_%s_%s_%s'%(freq, threshold_conc, model,variable, str(i)), "wb" ))
        pickle.dump(pa_tot_alltime[i]/len(numpy.arange(startyear,2020,1))*100, open(
            './sa_tot_alltime/pa_tot_alltime_%s_historical_%s_%s_%s_%s'%(freq, threshold_conc, model,variable, str(i)), "wb" ))

    if saveresults:
        for res_var in ['pa_op_list','pa_tot_list', 'sa_tot_list', 'pa_we_list', 'sa_we_list', 'pa_we_op_list', 'date_list']:
            with open('./polynyaareapickles/%s_ecmwf_%s_%s_%s_%s'%(res_var, variable, freq, str(threshold_conc), str(threshold_thick)), 'wb') as file:
                pickle.dump(eval(res_var), file)

from multiprocessing import Pool

def jobfunction(model):
    print("start processing model %s, scenario %s now"%(model, scenario))
    dataset = {}
    dataset['seaice'] = open_CMIP_variable(model, variable, scenario, freq)
    dataset_areacello = open_CMIP_variable(model, 'areacello', scenario, freq)
    
    if timesteps == 'max':
        timelimit = len(dataset['seaice'].time)
    else: 
        timelimit = timesteps
    print('TIMELIMIT:', timelimit)
    if not dataset['seaice']:
        print('ERROR: no sea ice data for %s or no success while homogenizing'%model)
        return
    try:
        dataset['seaice'] = homogenize_CMIP_variable(dataset['seaice'])
    except:
        breakpoint()
    if not dataset_areacello:
        print('ERROR: no dataset_areacello available for %s'%model)
        return
    dataset_areacello = homogenize_CMIP_variable(dataset_areacello)
    if dryrun:
        dataset_areacello.close()
        dataset['seaice'].close()
        return

    print("proceding with %s timesteps"%timelimit)
    resultsdictionary = compute_CMIP_areas(
            dataset,
            dataset_areacello,
            scenario,
            threshold_conc,
            threshold_thick,
            freq=freq, 
            timelimit=timelimit,
            createplots=createplots,
            seasonalmaps=seasonalmaps)
    if saveresults:
        save_CMIP_results(dataset, model, scenario, resultsdictionary, threshold_conc, threshold_thick, freq)
        

createplots = False
seasonalmaps = True
saveresults = True
scenario = 'historical'
variable = 'siconc'
threshold_conc = 30
threshold_thick = 0.12
freq = 'monthly'
dryrun = False
timesteps = 'max'
        
if __name__ == '__main__':
    print('Run with: %s %s %s %s'%(variable, freq, threshold_conc, threshold_thick))

    if (freq == 'monthly' and variable == 'sivol'):
        models = ['CAMS-CSM1-0','CNRM-CM6-1', 'CNRM-ESM2-1', 'ACCESS-CM2','ACCESS-ESM1-5', 'BCC-CSM2-MR', 'BCC-ESM1', 
                  'CESM2', 'CESM2-FV2', 'CESM2-WACCM', 'CESM2-WACCM-FV2', 'CNRM-CM6-1', 'GFDL-CM4', 'GFDL-ESM4', 
                  'HadGEM3-GC31-LL', 'UKESM1-0-LL'
                  'EC-Earth3', 'EC-Earth3-Veg', 'IPSL-CM6A-LR', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR', 
                  'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'SAM0-UNICON']
    elif (freq == 'monthly' and variable == 'siconc'):
        models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'BCC-ESM1', 'CAMS-CSM1-0', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 
                  'CESM2', 'CESM2-FV2', 'CESM2-WACCM', 'CESM2-WACCM-FV2', 
                  'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 
                  'MIROC-ES2L', 'MIROC6', 'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 
                  'MRI-ESM2-0', 'NorCPM1', 'SAM0-UNICON', 'UKESM1-0-LL']  
    elif (freq == 'daily' and variable == 'siconc'):
        models = ['CNRM-CM6-1', 'CNRM-ESM2-1', 'ACCESS-CM2', 'BCC-CSM2-MR', 'BCC-ESM1', 
                  'CanESM5', 'CESM2', 'CESM2-FV2', 'CESM2-WACCM', 'CESM2-WACCM-FV2', 
                  'CNRM-CM6-1', 'EC-Earth3', 'EC-Earth3-Veg', 'IPSL-CM6A-LR', 'MPI-ESM-1-2-HAM', 
                  'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'SAM0-UNICON']
    elif (freq == 'daily' and variable == 'sithick'):
        models = ['ACCESS-CM2','BCC-CSM2-MR','BCC-ESM1',
                  'CESM2',
                  'CESM2-FV2','CESM2-WACCM','CESM2-WACCM-FV2',
                  'EC-Earth3',
                  'IPSL-CM6A-LR','MIROC6','MPI-ESM-1-2-HAM','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','SAM0-UNICON']

    # The Pool-stuff is just for parallelisation (n threads)
    # run jobfunction(model) to continue without parallelisation
    
    p = Pool(7)
    p.map(jobfunction, models)
    
    
    