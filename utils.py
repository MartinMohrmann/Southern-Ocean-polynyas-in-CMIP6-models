import numpy

def create_resultsdictionary():

    import pandas
    import pickle
    import glob

    scenario = 'historical'
    thresholds = ['30', '45','60']


    models = ['CAMS-CSM1-0','CNRM-CM6-1', 'CNRM-ESM2-1', 
          'ACCESS-CM2','ACCESS-ESM1-5', 'BCC-CSM2-MR', 
          'BCC-ESM1', 'CESM2', 'CESM2-FV2', 'CESM2-WACCM', 
          'CESM2-WACCM-FV2', 'CNRM-CM6-1', 'GFDL-CM4', 
          'GFDL-ESM4', 'HadGEM3-GC31-LL', 'UKESM1-0-LL',
          'EC-Earth3', 'EC-Earth3-Veg', 'IPSL-CM6A-LR', 
          'MPI-ESM-1-2-HAM', 'MPI-ESM1-2-HR', 'NorCPM1',
          'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'SAM0-UNICON',
          'CanESM5', 'MIROC6', 'MIROC-ES2L']

    models.append('OBS')

    periods = ['mon', 'day']
    periodnames = {'mon':'monthly', 'day':'daily'}
    variables = ['siconc', 'sivol', 'sithick']

    resultsdictionary = {}
    resultsdictionarythick = {}
    resultsdictionarysivol = {}

    for variable in variables:
        resultsdictionary[variable] = {}
        for period in periods:
            resultsdictionary[variable][period] = {}
            for threshold in thresholds:
                resultsdictionary[variable][period][threshold] = {}



    for model in models:
        print(model)
        for threshold in thresholds:
            for variable in variables:
                for period in periods:
                    if model == 'OBS':
                        if variable in ['sithick']:
                            continue
                        resultsdictionary[variable][period][threshold][model] = {}
                        for res_var in ['pa_op_list', 'pa_tot_list', 'sa_tot_list', 'pa_we_list', 'sa_we_list', 'pa_we_op_list', 'date_list']:
                            with open('../polynyaareapickles/%s_ecmwf_%s_%s_%s_0.12'%(res_var, variable, periodnames[period], threshold), 'rb') as resultsdictfile:
                                #'./polynyaareapickles/pa_we_op_list_ecmwf_sivol_monthly_30_0.12'
                                resultsdictionary[variable][period][threshold][model][res_var[:-5]] = pickle.load(resultsdictfile)
                                #resultsdictionary[variable][period][threshold][model][res_var] = pandas.DataFrame(pickle.load(resultsdictfile)
                        resultsdictionary[variable][period][threshold][model] = pandas.DataFrame(resultsdictionary[variable][period][threshold][model])
                    else:
                        if variable == 'sivol':
                            filelist = glob.glob('../polynyaareapickles/complete_resultsdictfilehistorical_%s_%s*%s_%s'%(model, '30', variable, periodnames[period]))
                        else:
                            filelist = glob.glob('../polynyaareapickles/complete_resultsdictfilehistorical_%s_%s*%s_%s'%(model, threshold, variable, periodnames[period]))
                        if not filelist:
                            #print('no file found for:',variable ,model, period)
                            continue
                        else:
                            # print(variable, model, period)
                            pass
                        with open(filelist[0], 'rb') as resultsdictfile:
                            resultsdictionary[variable][period][threshold][model] = pandas.DataFrame(
                                { key:pandas.Series(value) for key, value in pickle.load(resultsdictfile).items() })
                    resultsdictionary[variable][period][threshold][model] = resultsdictionary[variable][period][threshold][model].set_index(resultsdictionary[variable][period][threshold][model].date)
                    # Ausgleich methodischer Fehler
                    #print(model, period, variable, threshold)
                    if model != 'OBS':
                        resultsdictionary[variable][period][threshold][model]['pa_op'] = resultsdictionary[variable][period][threshold][model]['pa_op'] - min(resultsdictionary[variable][period][threshold][model]['pa_op'])
                    resultsdictionary[variable][period][threshold][model]['pa_we_op'] = resultsdictionary[variable][period][threshold][model]['pa_we_op'] - min(resultsdictionary[variable][period][threshold][model]['pa_we_op'])
                    # compute 'pa_we_co' from available variables
                    resultsdictionary[variable][period][threshold][model]['pa_we_co'] = resultsdictionary[variable][period][threshold][model]['pa_we'] - resultsdictionary[variable][period][threshold][model]['pa_we_op']
                    resultsdictionary[variable][period][threshold][model]['pa_co'] = resultsdictionary[variable][period][threshold][model]['pa_tot'] - resultsdictionary[variable][period][threshold][model]['pa_op']
    return resultsdictionary

    """
    for model in models:
        for threshold in thresholds:
            for variable in variables:
                for period in periods:
                    if model == 'OBS':
                        #if variable in ['sivol', 'sithick'] and period == 'day':
                        #    continue
                        resultsdictionary[variable][period][threshold][model] = {}
                        for res_var in ['pa_op_list','pa_tot_list', 'sa_tot_list', 'pa_we_list', 'sa_we_list', 'pa_we_op_list', 'date_list']:
                            with open('../polynyaareapickles/%s_ecmwf_%s_%s_30_0.12'%(res_var, variable, periodnames[period]), 'rb') as resultsdictfile:
                                resultsdictionary[variable][period][threshold][model][res_var[:-5]] = pickle.load(resultsdictfile)
                                #resultsdictionary[variable][period][threshold][model][res_var] = pandas.DataFrame(pickle.load(resultsdictfile)
                        resultsdictionary[variable][period][threshold][model] = pandas.DataFrame(resultsdictionary[variable][period][threshold][model])
                    else:
                        filelist = glob.glob('../polynyaareapickles/complete_resultsdictfilehistorical_%s_%s*%s_%s'%(model, threshold, variable, periodnames[period]))
                        if not filelist:
                            #print('no file found for:',variable ,model, period)
                            continue
                        else:
                            # print(variable, model, period)
                            pass
                        with open(filelist[0], 'rb') as resultsdictfile:
                            # breakpoint()
                            resultsdictionary[variable][period][threshold][model] = pandas.DataFrame(
                                {key:pandas.Series(value) for key, value in pickle.load(resultsdictfile).items()})
                    resultsdictionary[variable][period][threshold][model] = resultsdictionary[variable][period][threshold][model].set_index(resultsdictionary[variable][period][threshold][model].date)
                    # Ausgleich methodischer Fehler
                    # if model != 'OBS':
                    resultsdictionary[variable][period][threshold][model]['pa_op'] = resultsdictionary[variable][period][threshold][model]['pa_op'] - min(resultsdictionary[variable][period][threshold][model]['pa_op'])
                    resultsdictionary[variable][period][threshold][model]['pa_we_op'] = resultsdictionary[variable][period][threshold][model]['pa_we_op'] - min(resultsdictionary[variable][period][threshold][model]['pa_we_op'])
                    resultsdictionary[variable][period][threshold][model]['pa_we_co'] = resultsdictionary[variable][period][threshold][model]['pa_we'] - resultsdictionary[variable][period][threshold][model]['pa_we_op']

    return resultsdictionary
    """

def open_CMIP_variable(model, variable, scenario, freq):
    import glob
    import xarray
    freq = 'day' if freq == 'daily' else 'mon'
    # for the GISS-models, areacella available only and is the sea ice grid
    if variable in ['siconc', 'sivol', 'sithick']:
        variantsdict = {'CNRM-CM6-1':'r1i1p1f2','CNRM-ESM2-1':'r1i1p1f2','UKESM1-0-LL':'r1i1p1f2', 'MIROC-ES2L':'r1i1p1f2','HadGEM3-GC31-LL':'r1i1p1f3'}
        if model in variantsdict:
            variant = variantsdict[model]
        else:
            variant = 'r1i1p1f1'
        filelist = glob.glob('../../%s/%s/%s_SI%s*%s*.nc'%(model,scenario,variable,freq[0:3],variant))
        #print('matching files with: ../%s/%s/%s_SI%s*%s*.nc'%(model,scenario,variable,freq[0:3],variant))
        if filelist:
            #print('progressing with the following input files:\n%', sorted(filelist))
            dataset = xarray.open_mfdataset(sorted(filelist))
        else:
            #print('no input files found with are matching ../%s/%s/%s_SI%s*%s*.nc'%(model,scenario,variable,freq[0:3],variant))
            return 0,0
    elif variable == 'areacello':
            try:
                dataset = xarray.open_mfdataset('../../%s/areacello*historical*.nc'%(model))
            except:
                # breakpoint()
                dataset = xarray.open_mfdataset('../../%s/areacello*.nc'%(model))
    else:
        try:
            dataset = xarray.open_mfdataset('../../%s/%s/%s*.nc'%(model, scenario, variable))
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

def colormap_alpha(cmap):
    from matplotlib.colors import ListedColormap
    my_cmap = cmap(numpy.arange(cmap.N))
    my_cmap[:,-1] = numpy.linspace(0, 1, cmap.N) # inactive
    
    my_cmap[:,-1] = 1
    my_cmap[0,-1] = 0
    my_cmap = ListedColormap(my_cmap)
    return my_cmap
