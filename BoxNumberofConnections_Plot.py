#### In this code, we show how well-connected ballooning modes are for a gs2 simulation. Plots ballooning modes onto a hat{theta}_0 ky rho_i grid.
#### Questions? Contact jasonfrancisparisi AT gmail.com

from matplotlib import colors, ticker, cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mc
from matplotlib import colors
import scipy.special as sp
from scipy import optimize
from scipy import special
import scipy.interpolate
import scipy.ndimage
from pylab import *
import numpy as np

#------------------------------------#
#      plotting parameters		     #
#------------------------------------#

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

label_size = 30
modlabel_size = 0.9*label_size
ticklabelsize=25
marker_size = 70

#------------------------------------#
#       function definitions	     #
#------------------------------------#

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

#------------------------------------#
#    hypothetical gs2 file inputs    #
#------------------------------------#

# setupdescrip = 'April23runlowres'
# setupdescrip = 'April23manyconnections'
# setupdescrip = 'April23manyconnectionsjtwist1'
# setupdescrip = 'April26NLNT24'
# setupdescrip = 'May4NLNT27'
# setupdescrip = 'May27NLNT28full'
# setupdescrip = 'May27NLNT28slab'
# setupdescrip = 'May30Thesisexamplewellresolvedforcomp'
# setupdescrip = 'Jun32020Nolocalshearcheap'
# setupdescrip = 'Jun32020Nolocalshearcheapjtwist18'
# setupdescrip = 'Jun32020Nolocalshearcheapjtwist12'
# setupdescrip = 'Jun172020thesisstandard1'
# setupdescrip = 'Jun172020thesisstandard2'
# setupdescrip = 'Jun172020thesisstandard3'
# setupdescrip = 'Jun172020thesisstandard4'
# setupdescrip = 'Jun172020thesisstandard4BIG'
# setupdescrip = 'Jun202020thesisnonlinear'
# setupdescrip = 'Jul62020thesisnonlinear'
# setupdescrip = 'Jul132020thesisnonlinear'
# setupdescrip = 'Jul142020thesisnonlinearBase1'
# setupdescrip = 'Jul142020thesisnonlinearHero1'
# setupdescrip = 'Aug12020thesisnonlinearshat0p8'
# setupdescrip = 'Aug12020thesisnonlinearshatnormal'

# setupdescrip = 'Aug32020thesisstandard1'
# setupdescrip = 'Aug32020thesisstandard2'
# setupdescrip = 'Aug32020thesisstandard3'
# setupdescrip = 'Aug372020thesisstandard4'
# setupdescrip = 'Geo2mod'
# setupdescrip = 'NL25'
# setupdescrip = 'NL25higherjtwist'
# setupdescrip = 'NL25smallery0'
# setupdescrip = 'testcircle'
# setupdescrip = 'fineDeltaky'
# setupdescrip = 'CBC_electron_scale'
# setupdescrip = 'CBC_electron_scale_2'
setupdescrip = 'CBC_electron_scale_cheapest_nper2'
# setupdescrip = 'CBC_electron_scale_cheapest_nper1'
# setupdescrip = 'CBC_electron_scale_cheapest_nper2'

nperiod = 2
shat = 0.8
# shat = 3.3594
# rhoi_over_rhos = 1.41
rhoi_over_rhos = 1

y0 = 0.3
# jtwist = 15
# jtwist = 2
jtwist = 10
# naky = 24
naky = 16
# nakx = 179 ### should be an odd number...
# nakx = 91 ### should be an odd number...
nakx = 121 ### should be an odd number...

#------------------------------------#
#  calculating ballooning chains     #
#------------------------------------#

kxspacing = (2*nperiod -1)*2*pi*shat/(y0*jtwist) ### in rho_s units

maxkx = np.floor(nakx/2)*kxspacing
minkx = -maxkx

kxgrid = np.linspace(minkx,maxkx,nakx)
kx = np.roll(np.fft.fftshift(kxgrid),1)
ky = np.linspace(0,(1/y0)*(naky-1),naky)

theta0 = np.zeros((naky,nakx),dtype=float)
for kyit in np.arange(1,naky):
	for kxit in np.arange(nakx):
		theta0here = kx[kxit]/(shat*ky[kyit])
		if (abs(theta0here - np.pi) < 1e-5):
			theta0[kyit,kxit] = pi
		else:
			theta0[kyit,kxit] = theta0here

connectedthetasatky = [] ### This contains all of the connected chains for each ky.

for kyit in np.arange(1,naky):

	#### We now find the theta0 values that are connected: produce all chains of theta0 that are connected by pm 2pi
	#### start with the minimum theta0 in the theta0 array... then finding all the connected chains increasing in increments of 2pi
	connectedtheta0sarray = [] #### For each connectedchainit, puts in the theta0 values that are connected...
	indicesconnectedthusfar = np.zeros(nakx,dtype=float) ### if mode is unconnected, zero. If it has already been connected, a 1.
	connectedchainit = 0 # this labels the connected chain it... for first nonzero ky, there should be jtwist of these...

	for theta0it in np.arange(nakx):
		connectedtheta0sarray.append([])		
		### search for a connected first mode

		connectedchainbreak = 0

		### first, make sure that the mode is unconnected... otherwise, don't search for mode connections.. 
		if np.int(indicesconnectedthusfar[theta0it]) == 0:
			### mode exists if np.where(theta0[0][0][1]==2*np.pi)[0].shape == 1

			firstconnectingv = (2*nperiod-1)*2*np.pi + theta0[kyit,theta0it] #### Should this be a -ve?

			firstconnectedindexcontainer = np.where(np.logical_and(theta0[kyit]<firstconnectingv+0.0001, theta0[kyit]>firstconnectingv-0.0001))[0] ### looking for further connections

			# if np.where(theta0[uu][vv][kyit]==2*np.pi+theta0[uu][vv][kyit,theta0it])[0].shape[0] == 1: ### if there's a connection at + 2pi.
			if firstconnectedindexcontainer.shape[0] == 1: ### if there's a connection at + 2pi
				firstconnectedindex = firstconnectedindexcontainer[0]
				indicesconnectedthusfar[np.int(firstconnectedindex)] = 1 # saying this guy is connected.
				indicesconnectedthusfar[theta0it] = 1
				connectedtheta0sarray[connectedchainit].append(theta0it)
				connectedtheta0sarray[connectedchainit].append(np.int(firstconnectedindex))
				### we're connected! Now, search for further connections.
				for connectionit in np.arange(2,10000): 	

					connectingv = connectionit*(2*nperiod-1)*2*np.pi+theta0[kyit,theta0it]
					### Checking we're between two values	
					nextconnectedindexcontainer = np.where(np.logical_and(theta0[kyit]<connectingv+0.0001, theta0[kyit]>connectingv-0.0001))[0] ### looking for further connections

					if nextconnectedindexcontainer.shape[0] == 1: ### if there's a connection at + connectionit*2pi
						nextconnectedindex = nextconnectedindexcontainer[0]
						indicesconnectedthusfar[np.int(nextconnectedindex)] = 1 # saying this guy is connected.
						connectedtheta0sarray[connectedchainit].append(np.int(nextconnectedindex))
					else: ### if there are no more connections, break it up.
						connectedchainit = connectedchainit + 1
						connectedchainbreak = 1
						break
				### Now, we are going backwards!!! 
				for connectionit in np.arange(1,10000): 	

					print('Im here and theta0it is {}'.format(theta0it))
					connectingv = -connectionit*(2*nperiod-1)*2*np.pi+theta0[kyit,theta0it]
					### Checking we're between two values	
					nextconnectedindexcontainer = np.where(np.logical_and(theta0[kyit]<connectingv+0.0001, theta0[kyit]>connectingv-0.0001))[0] ### looking for further connections

					if nextconnectedindexcontainer.shape[0] == 1: ### if there's a connection at + connectionit*2pi
						nextconnectedindex = nextconnectedindexcontainer[0]
						indicesconnectedthusfar[np.int(nextconnectedindex)] = 1 # saying this guy is connected.
						connectedtheta0sarray[connectedchainit-connectedchainbreak].append(np.int(nextconnectedindex))
					else: ### if there are no more connections, break it up.
						if connectedchainbreak == 0:
							connectedchainit = connectedchainit + 1
						break

	### what do we do with the stuff that is not connected? ### These are just unconnected modes. We need to add these unconnected modes too!

	### FINDING UNCONNECTED MODES...
	#### Sort through all modes that have been unconnected, and write them out
	for it in np.arange(len(theta0[kyit])):
		if indicesconnectedthusfar[it] == 0: ## if mode unconnected
			connectedtheta0sarray.append([])
			connectedtheta0sarray[connectedchainit].append(it)
			connectedchainit = connectedchainit + 1

	### remove all empty arrays
	list2 = [x for x in connectedtheta0sarray if x]

	### Now, we wish to calculate theta0 hat for each array... and sort by theta0hat...

	theta0sforthisky = theta0[kyit]
	theta0hatandballooningchainarray = []
	for it in np.arange(len(list2)):

		if len(list2[it])>1:
			theta0values = np.array([ theta0sforthisky[i] for i in list2[it]])

			testvals = np.where(np.logical_and(theta0values>=-(2*nperiod-1)*np.pi, theta0values<=(2*nperiod-1)*np.pi-0.001))

			if len(testvals[0]) == 0: ### if the edge case, where -pi and pi are in here..
				theta0hatandballooningchainarray.append([pi,list2[it]])

			else:
				theta0hatindex = np.where(np.logical_and(theta0values>=-(2*nperiod-1)*np.pi, theta0values<=(2*nperiod-1)*np.pi-0.001))[0][0] ### finding all values between-pi and pi -.001 (for the edge case)
				### Now, the associated theta0hat value is found
				theta0hatindexhere = list2[it][theta0hatindex]
				theta0hatval = theta0[kyit,theta0hatindexhere]
				theta0hatandballooningchainarray.append([theta0hatval,list2[it]])
		else:
			theta0hatval = theta0[kyit,list2[it][0]]
			theta0hatandballooningchainarray.append([theta0hatval,list2[it]])

	### NEXT! We sort according to the value of theta0hat.... in ascending order...

	list3 = sorted(theta0hatandballooningchainarray,key=lambda x: x[0])
	connectedthetasatky.append(list3)

#### Now we have connectedthetasatky.

label_size = 30
ticklabelsize = 25
numtheta0s = 20
theta0range = np.linspace(0,np.pi,numtheta0s)
plt.rcParams['xtick.labelsize'] = ticklabelsize
plt.rcParams['ytick.labelsize'] = ticklabelsize
xpad = 30
ypad = 20
linewidth1 = 1.15
lwidth=4
ytickpad=20

sequentialcolorrange = ['k','darkblue','royalblue','deepskyblue','turquoise','lightgreen','lawngreen','forestgreen','olive','goldenrod','darkorange','indianred','crimson','red','mediumvioletred','fuchsia']
fig =plt.figure(figsize=(16,16),dpi=100,facecolor='w')
ax1 = plt.subplot(111) #

for kyit in np.arange(naky-1):
	n = (kyit+1)*jtwist
	color=iter(cm.rainbow(np.linspace(0,1,n))) ### colors to cycle over
	numberofballoningchains = len(connectedthetasatky[kyit])
	# if numberofballoningchains != len(kx[uu][vv]): ### all chains are unconnected
	if numberofballoningchains != nakx: ### all chains are unconnected
		for chainit in np.arange(numberofballoningchains): ### over the chains
			if len(connectedthetasatky[kyit][chainit][1]) > 1: ### if
				try:
					colornow=next(color)
					for chainconnectionit in np.arange(len(connectedthetasatky[kyit][chainit][1])): ### iterating over the theta0s that are connected within a chain!
						theta0it = connectedthetasatky[kyit][chainit][1][chainconnectionit]
						theta0location = theta0[kyit+1,theta0it]
						ax1.scatter(theta0location,ky[kyit+1]*rhoi_over_rhos,color=colornow,marker='s',s=4)
				except:
					print('exception')
					pass
			else:
				theta0it = connectedthetasatky[kyit][chainit][1][0]
				theta0location = theta0[kyit+1,theta0it]
				ax1.scatter(theta0location,ky[kyit+1]*rhoi_over_rhos,color='k',marker='s',s=4) ### all black unconnected
	# if numberofballoningchains == len(kx[uu][vv]): ### all chains are unconnected
	if numberofballoningchains == nakx: ### all chains are unconnected
		for chainit in np.arange(numberofballoningchains): ### over the chains
			theta0it = connectedthetasatky[kyit][chainit][1][0]
			theta0location = theta0[kyit+1,theta0it]
			ax1.scatter(theta0location,ky[kyit+1]*rhoi_over_rhos,color='k',marker='s',s=4) ### all black unconnected

ax1.set_ylabel('$k_y \\rho_i$',fontsize=label_size)
ax1.set_xlabel('$\\hat{{\\theta}}_0$',fontsize=label_size)

box = ax1.get_position()
ax1.set_position([box.x0, box.y0+0.025, box.width*0.92, box.height])

ax1.set_ylim(ymin = -0.05)

for it in [0,1,2,3]:
	ax1.axvline(( (2*nperiod-1))*pi + 2*it*(2*nperiod-1)*pi,color='pink',ls = '--')
	ax1.axvline(-((2*nperiod-1))*pi - 2*it*(2*nperiod-1)*pi,color='pink',ls = '--')

plt.suptitle('nakx = {}, naky = {}, jtwist = {}, $y_0$ = {}, nperiod = {}, $\\hat{{s}}$ = {:.2f}'.format(nakx,naky,jtwist,y0,nperiod,shat),fontsize=0.78*label_size)

ax1.set_xlim(-(2*nperiod-1)*25,(2*nperiod-1)*25)
ax1.axvline(0,ls='--',c='k')

plt.savefig('ConnectedModeswiththeta0HATcutoff{}.eps'.format(setupdescrip),bbox_inches = 'tight', pad_inches = 0.1)

plt.clf()
plt.close(fig)




