import utils
from scipy import stats
from matplotlib import pyplot as plt
import pandas
import numpy
import seaborn as sns
plt.rcParams.update({'font.size': 18})
resultsdictionary = utils.create_resultsdictionary()
#fig, axs = plt.subplots(ncols=, figsize=[15,8])

period = 'mon'
variable = 'sivol'
threshold = '30'


print(variable, period, threshold)
models = resultsdictionary[variable][period][threshold]
#models.sort()
dfa = pandas.concat([resultsdictionary[variable][period][threshold][model]['pa_we_co'] for model in models], axis=1, keys=models)
dfb = pandas.concat([resultsdictionary[variable][period][threshold][model]['pa_we_op'] for model in models], axis=1, keys=models)
#df
#breakpoint()
#sns.load_dataset(df)
#complete = agg(ice_var='siconc', area_var='pa_we', period=period, method='mean').T['mean']
if variable == 'siconc':
    startyear = 1979
elif variable == 'sivol':
    startyear = 1850
df2a = dfa[(dfa.index.month >= 5) & (dfa.index.month <= 11) & (dfa.index.year >= startyear) & (
            dfa.index.dayofyear < 320)].groupby(pandas.Grouper(freq='A')).mean()
df2b = dfb[(dfb.index.month >= 5) & (dfb.index.month <= 11) & (dfb.index.year >= startyear) & (
            dfb.index.dayofyear < 320)].groupby(pandas.Grouper(freq='A')).mean()

df2a['seab_area'] = 'pa_we_co' 
df2b['seab_area'] = 'pa_we_op'

df2a = df2a.sort_index(axis = 1) 
df2b = df2b.sort_index(axis = 1) 

dfc = pandas.concat([df2a, df2b], axis=0)
#breakpoint()
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=[15,15], gridspec_kw=dict(width_ratios=[len(df2a.keys()),1]), 
    sharey=True)

df2a = df2a.drop(['seab_area'], axis=1)
df2b = df2b.drop(['seab_area'], axis=1)

yscaling = 10e9
ymax = 3e11/yscaling
ymin = -0.01e11/yscaling

g = sns.violinplot(data=df2a/yscaling, scale='width', inner="points", ax=axs[0][0], order=df2a.keys()[:-1])#,
g.scatter(x=range(0,len(models)), y=df2a.mean()/yscaling, color='white', zorder=100, s=50)
g.set_xticklabels(labels=df2a.keys()[:-1] ,rotation=90)
g.set_title('coastal')
g.set_ylabel('polynya areas in 10³km²')
g.set_ylim(ymin, ymax)

g = sns.violinplot(data=df2a['OBS']/yscaling, scale='width', inner="points", ax=axs[0][1])#,
g.set_xticklabels(labels=['OBSERVATIONS\n(2010-2020)'] ,rotation=90, color="red")
g.scatter(x=0, y=df2a['OBS'].mean()/yscaling, color='white', zorder=100, s=50, edgecolor='black')
#g.set_title('coastal')
g.set_ylim(ymin, ymax)

g = sns.violinplot(data=df2b/yscaling, scale='width', inner="points", ax=axs[1][0])#, color='white')
g.scatter(x=range(0,len(models)), y=df2b.mean()/yscaling, color='white', edgecolor='black', zorder=100, s=50)
g.set_xticklabels(labels=df2a.keys()[:-1] ,rotation=90)
g.set_title('open water')
g.set_ylim(ymin, ymax)

g = sns.violinplot(data=df2b['OBS']/yscaling, scale='width', inner="points", ax=axs[1][1])#,
g.set_xticklabels(labels=['OBSERVATIONS\n(2010-2020)'] ,rotation=90, color="red")
g.scatter(x=0, y=df2b['OBS'].mean()/yscaling, color='white', zorder=100, s=50)
#g.set_title('coastal')
g.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig('violinplots_%s_%s_%s.png'%(variable, period, threshold), dpi=100)