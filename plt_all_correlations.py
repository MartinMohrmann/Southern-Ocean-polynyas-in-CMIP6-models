from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels
import numpy as np
import numpy
import pandas
import utils

resultsdictionary = utils.create_resultsdictionary()
periods=['mon']
variables = ['sivol', 'siconc']
slopedict = {}
rvdict = {}
interceptdict = {}

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=[10,15],constrained_layout=True) #sharex=True, sharey=True, constrained_layout=True)
#fig2, axs2 = plt.subplots(nrows=4, ncols=5, figsize=[13,10], sharex=True, sharey=True, constrained_layout=True)
axs = axs.flat
#axs2 = axs2.flat
month = 8
steplength = 12 # should be eiter 12 or 1 (yearly or monthly) # end CESM daily one step earlier!
threshold = '30'
variable1, period1 = 'sivol', 'mon'
variable2, period2= 'siconc', 'mon'
#print(threshold)
models = list(set(resultsdictionary[variable1][period1][threshold].keys()) & 
              set(resultsdictionary[variable2][period2][threshold].keys()))
unused_axes = set(range(0,25)) - set(range(0,len(models)))
for i in unused_axes:
    fig.delaxes(axs[i])
print('List of models:', models)
models.sort()
for index, model in enumerate(models):

    if model == 'OBS':
        continue
    # print(model)
    axs[index].set_aspect('equal', adjustable='box')
    endstep = len(resultsdictionary['siconc']['mon']['30'][model]['pa_tot'])
    if model == 'MRI-ESM2-0':
        # because the MRI models submit data only after 1919 in daily resolution
        year=0#69
    elif model == 'SAM0-UNICON':
        year=0#100
    elif model in ['CESM2', 'CESM2-WACCM', 'CESM2-FV', 'CESM2-WACCM-FV']:
        endstep = len(resultsdictionary['siconc']['mon']['30'][model]['pa_tot'])-1
    #elif model == 'OBS':
        #year=31
    else:
        year=0

    x = resultsdictionary[variable1][period1][threshold][model]['pa_tot'].groupby(pandas.Grouper(freq='M')).mean()[month:endstep:steplength].values
    X = x
    #y = resultsdictionary['siconc']['mon'][threshold][model]['pa_tot'][12*year+month:endstep:steplength].values
    y = resultsdictionary[variable2][period2][threshold][model]['pa_tot'][12*year+month:endstep:steplength].values
    print(model, len(x), len(y))
    axs[index].scatter(x, y, alpha=0.5)
    #plt.xlabel('polynya areas computed from "sivol"')
    #plt.ylabel('polynya areas computed from "siconc"')
    axs[index].set_title(model, pad=15)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # print('sterr:', std_err)
    xmax = max(max(x), max(y))#0.5e12
    x_coord = numpy.linspace(0,xmax,100)
    axs[index].plot(x,intercept+x*slope+intercept, color='blue', ls=':', lw=2)

    x = sm.add_constant(x)
    res_ols = sm.OLS(y, x).fit()
    # print(res_ols.summary())
    slope = res_ols.params[1]
    slopedict[model] = slope
    rvdict[model] = r_value
    interceptdict[model] = intercept
    
    slopelowest = res_ols.conf_int(alpha=0.05, cols=None)[1][0]
    slopehighest= res_ols.conf_int(alpha=0.05, cols=None)[1][1]
    intercept = res_ols.params[0]
    interceptlowest = res_ols.conf_int(alpha=0.05, cols=None)[0][0]
    intercepthighest= res_ols.conf_int(alpha=0.05, cols=None)[0][1]
    
    axs[index].plot(x_coord,intercept+x_coord*slope, color='green')
    axs[index].fill_between(x_coord, interceptlowest+x_coord*slopelowest, intercepthighest+x_coord*slopehighest, alpha=0.2)

    axs[index].set_xlim(-.1*xmax, xmax)
    axs[index].set_ylim(-.1*xmax, xmax)
    
    axs[index].grid()
    
    #residual = res_ols.resid
    residual = y-x[:,1]
    #residual = y-x[:,1]
    relative_residual = abs((residual/numpy.maximum(X,y))[~numpy.isnan(residual/numpy.maximum(x[:,1],y))])
    #print(model, '90% percentile:',numpy.percentile(relative_residual, 90))
    #print(model, mean(abs(relative_residual)))#numpy.std(relative_residual))
    
    #axs2[index].hist(residual, bins=40)
    #axs2[index].set_title(model)
    
fig.suptitle('polynya areas computed from "sivol monthly" (x) vs "siconc monthly (y)" month:%s, stepl:%s'%(month, steplength))
#fig.delaxes(axs[17])
#fig.delaxes(axs[18])
#fig.delaxes(axs[19])
plt.savefig('./all_correlations/all_correlations_%s%s_%s%s_%s.png'%(variable1, period1, variable2, period2, threshold))

fig, axs = plt.subplots(ncols=3, figsize=[20,6])
df = pandas.DataFrame([interceptdict.values(), slopedict.values(), rvdict.values()], columns=interceptdict.keys(), index=['intercept', 'slope', 'r-value'])
axi0 = df.T['intercept'].plot(kind='bar', title='intercept', ax=axs[0])
axi1 = df.T['slope'].plot(kind='bar', title='slope', ax=axs[1])
axi2 = df.T['r-value'].plot(kind='bar', title='r-value', ax=axs[2])

print('mean slope:',df.T['slope'].mean(), 'mean r-value:',df.T['r-value'].mean())

x_offset = 0.02
y_offset = 0.04

for p in axi0.patches:
    b = p.get_bbox()
    x = b.y1 + b.y0
    val = f"{x:.3}"        
    axi0.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), rotation='90')

for p in axi1.patches:
    b = p.get_bbox()
    val = "{:.2f}".format(b.y1 + b.y0)        
    axi1.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), rotation='90')

for p in axi2.patches:
    b = p.get_bbox()
    val = "{:.2f}".format(b.y1 + b.y0)        
    axi2.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), rotation='90')

plt.savefig('./all_correlations/all_correlations_%s%s_%s%s_%sbarchart'%(variable1, period1, variable2, period2, threshold))
