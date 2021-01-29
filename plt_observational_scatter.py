import utils
from scipy import stats
from matplotlib import pyplot as plt
import pandas
import numpy

resultsdictionary = utils.create_resultsdictionary()
fig, axs = plt.subplots(ncols=3, figsize=[10,10])

month=11
sivol_mon = resultsdictionary['sivol']['mon']['30']['OBS'][resultsdictionary['sivol']['mon']['30']['OBS'].date.dt.month == month]
sivol_day = resultsdictionary['sivol']['day']['30']['OBS'].groupby(pandas.Grouper(freq='M')).mean() 
sivol_day = sivol_day[sivol_day.index.month==month]
nmin = -0.1e11
nmax = 7e11
x,y = sivol_day['pa_tot'], sivol_mon['pa_tot']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(slope, r_value)
titlepad = 15
titlex, titley, titlefontsize = 0.5e11, 6e11, 16

axs[0].scatter(x,y, color='blue');
axs[0].set_xlabel("A(daily sivol)"); axs[0].set_ylabel("A(monthly sivol)"); 
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_title('daily vs monthly sivol', pad=titlepad)
axs[0].set_ylim(nmin, nmax); axs[0].set_xlim(nmin, nmax);
axs[0].text(x=titlex, y=titley, s='a', fontsize=titlefontsize)
xcoord = numpy.linspace(nmin, nmax, 100)
axs[0].plot(xcoord, xcoord*slope+intercept, alpha=0.5)

for index, threshold in enumerate(['30','60']):

    siconc_mon = resultsdictionary['siconc']['mon'][threshold]['OBS'][resultsdictionary['siconc']['mon'][threshold]['OBS'].date.dt.month == month]
    siconc_day = resultsdictionary['siconc']['day'][threshold]['OBS'].groupby(pandas.Grouper(freq='M')).mean()
    siconc_day = siconc_day[siconc_day.index.month==month]

    x,y = siconc_day['pa_tot'], siconc_mon['pa_tot'] 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    xcoord = numpy.linspace(nmin, nmax, 100)
    axs[index+1].plot(xcoord, xcoord*slope+intercept, alpha=0.5)
    print(slope, r_value)
    axs[index+1].scatter(x,y, color='grey')#facecolor='white', edgecolor='blue')
    axs[index+1].scatter(x[-10:],y[-10:], color='blue');
    axs[index+1].set_xlabel("A(daily siconc)"); axs[index+1].set_ylabel("A(monthly siconc)"); 
    axs[index+1].set_aspect('equal', adjustable='box')
    axs[index+1].set_title('daily vs monthly siconc', pad=titlepad)
    axs[index+1].set_ylim(nmin, nmax); axs[index+1].set_xlim(nmin, nmax);

axs[1].text(x=titlex, y=titley, s='b', fontsize=titlefontsize)
axs[2].text(x=titlex, y=titley, s='c', fontsize=titlefontsize)

    
#plt.show()

plt.savefig('observational_scatter.png')
#plt.show()
