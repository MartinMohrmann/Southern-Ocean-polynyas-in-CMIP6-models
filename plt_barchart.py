import utils
resultsdictionary = utils.create_resultsdictionary()

# Besonders spannend ist hier das ganze mal mit max() und mal mit mean() zu aggreggieren und die Monate zu ändern
# beachte das die ganze Statistik für die gesamte südliche Hemisphere oder alternativ für die Weddell Sea gemacht werden 
# kann

threshold = '30'
def agg(ice_var, area_var, period, method):
    models = resultsdictionary[ice_var][period][threshold].keys()
    for model in models:
        if area_var in ['pa_co']:
            resultsdictionary[ice_var][period][threshold][model]['pa_co'] = resultsdictionary[ice_var][period][threshold][model]['pa_tot'] - resultsdictionary[ice_var][period][threshold][model]['pa_op']
        if area_var in ['pa_we_co']:
            resultsdictionary[ice_var][period][threshold][model]['pa_we_co'] = resultsdictionary[ice_var][period][threshold][model]['pa_we'] - resultsdictionary[ice_var][period][threshold][model]['pa_we_op']
            
    df = pandas.concat([resultsdictionary[ice_var][period][threshold][model][area_var] for model in models], axis=1, keys=models)
    df
    if method=='mean':
        df = df[(df.index.month >= 5) & (df.index.month <= 11) & (df.index.year >= 1979) & (
            df.index.dayofyear < 320)].groupby(pandas.Grouper(freq='A')).mean()
    elif method=='max':
        df = df[(df.index.month >= 5) & (df.index.month <= 11) & (df.index.year >= 1979) & (
            df.index.dayofyear < 320)].groupby(pandas.Grouper(freq='A')).max()
    #breakpoint()
    df = df.describe()
    # breakpoint()
    return df

def barcharts(resultsdictionary, period, ice_var, ax, method, text):
    complete = {}
    #if period == 'mon' and method == 'mean' and ice_var == 'siconc':
    #    yerr = resultsdictionary[ice_var][period]['30'][model]['pa_we'].describe()['25%']
    yerr = agg(ice_var=ice_var, area_var='pa_we', period=period, method=method).T['50%']
    # breakpoint()
    # breakpoint()
    complete['pa_we_tot'] = agg(ice_var=ice_var, area_var='pa_we', period=period, method=method).T['mean'] #+ numpy.random.rand(len(complete))
    # ACHTUNG: HIER SITZT NOCH EIN FEHLER! Ich kann in folgender Zeile nicht die Differenz aus möglicherweise unterschiedlichen Zeitschritten ziehen
    complete['pa_we_co'] = agg(ice_var=ice_var, area_var='pa_we_co', period=period, method=method).T['mean']#-agg(ice_var=ice_var, area_var='pa_we_op', period=period, method=method)
    #resultsdictionary[ice_var][period][threshold]
    complete['pa_we_op'] = agg(ice_var=ice_var, area_var='pa_we_op',  period=period, method=method).T['mean']
    #complete['pa_tot_max'] = aggmax(ice_var=ice_var, area_var='pa_tot',  period=period)
    complete = pandas.DataFrame([complete['pa_we_tot'], complete['pa_we_co'], complete['pa_we_op']], index=['pa_we_tot', 'pa_we_co', 'pa_we_op']).transpose()

    alpha = 0.2 if method=='max' else 1
    # complete['pa_tot'] = complete['pa_tot'] + numpy.random.rand(len(complete))
    # complete['name'] = complete.index
    # import pdb; pdb.set_trace()
    # complete.sort_values(by=['name','pa_tot', ])
    # complete['randNumCol'] = numpy.random.sample(len(complete))
    complete = complete.sort_values(by=['pa_we_tot'], ascending=False)
    #complete['pa_we_tot'].max()+0.2*complete['pa_we_tot'].max()
    if method == 'mean':
        #breakpoint()
        (complete['pa_we_co']+complete['pa_we_op']).plot(kind='bar', yerr=yerr, ylim=(-.1e11,ymax),title=text+'%s %s'%(ice_var, period), ax=ax, alpha=0.1)#, yerr=complete['pa_tot_max'])
    complete[['pa_we_op', 'pa_we_co']].plot(kind='bar', stacked=True, ylim=(-.1e11,ymax),title=text+'%s %s'%(ice_var, period), ax=ax, alpha=alpha)#, yerr=complete['pa_tot_max'])

    #complete['pa_tot_max'].plot(kind="bar", ax=ax, alpha=0.2)
    #ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes, fontsize=20)
    return complete
  
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=[20,15])
axs = axs.flat
ymax = 2e11
# !!!!!!!!!!!!!!add aggregate = 'full' to all the following lines!
complete_siconc_mon = barcharts(resultsdictionary=resultsdictionary, period='mon', ice_var='siconc', ax=axs[0],text='',method='max')
complete_siconc_mon = barcharts(resultsdictionary=resultsdictionary, period='mon', ice_var='siconc', ax=axs[0],text='(a) ',method='mean')#
complete_siconc_day = barcharts(resultsdictionary=resultsdictionary, period='day', ice_var='siconc', ax=axs[1],text='', method='max')
complete_siconc_day = barcharts(resultsdictionary=resultsdictionary, period='day', ice_var='siconc', ax=axs[1],text='(b) ',method='mean')
complete_sivol_mon  = barcharts(resultsdictionary=resultsdictionary, period='mon', ice_var='sivol', ax=axs[2], text='',method='max')
complete_sivol_mon  = barcharts(resultsdictionary=resultsdictionary, period='mon', ice_var='sivol', ax=axs[2], text='(c) ',method='mean')

(complete_siconc_day-complete_siconc_mon)[['pa_we_op', 'pa_we_co']].plot(kind='bar', stacked=True, title='(d) PA diff siconc daily/monthly', ax=axs[4], ylim=(-0.5*ymax,0.5*ymax))
(complete_sivol_mon-complete_siconc_mon)[['pa_we_op', 'pa_we_co']].plot(kind='bar', stacked=True, title='(e) PA diff siconc/sivol monthly', ax=axs[5], ylim=(-0.5*ymax,0.5*ymax))

#axs[0].text(0.5, 0.5,'a',transform=ax.transAxes)
#(complete_sivol_mon-complete_siconc_mon)[['pa_op', 'pa_co']].plot(kind='bar', stacked=True, title='PA diff siconc/sivol monthly', ax=axs[6], ylim=(-2.5e11,2.5e11))
plt.subplots_adjust(hspace=0.5)
fig.delaxes(axs[3])
plt.savefig('barplot.png')
plt.show()


#plt.text()
#plot_clustered_stacked([complete_siconc_mon, complete_sivol_mon, complete_siconc_day],["df1", "df2", "df3"])
#plt.ylim(-1e11,1e11)