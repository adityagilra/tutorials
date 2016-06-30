#!/usr/bin/env python
'''
Based on:
Zenke, Friedemann, Guillaume Hennequin, and Wulfram Gerstner.
 "Synaptic Plasticity in Neural Networks Needs Homeostasis with a Fast Rate Detector."
 PLoS Comput Biol 9, no. 11 (November 14, 2013): e1003330. doi:10.1371/journal.pcbi.1003330.

Zenke's fast homeostasis embedded in Brunel 2000 / Ostojic 2014 network.

author: Aditya Gilra, Jun 2016.
 in Brian2rc3 for CAMP 2016.
'''

'''
Tutorial steps:
1.
'''

#import modules and functions to be used
from brian2 import *    # importing brian also does:
                        # 'from pylab import *' which imports:
                        # matplot like commands into the namespace, further
                        # also can use np. for numpy and mpl. for matplotlib
from data_utils import *

stand_alone = True
if stand_alone: set_device('cpp_standalone', build_on_run=False)
else:
    #prefs.codegen.target = 'numpy'
    #prefs.codegen.target = 'weave'
    prefs.codegen.target = 'cython'
import random
import time

np.random.seed(100) # set seed for reproducibility of simulations
random.seed(100) # set seed for reproducibility of simulations

# ###########################################
# Simulation parameters
# ###########################################

simdt = 0.1*ms
simtime = 0.5*second
defaultclock.dt = simdt     # set Brian's sim time step
dt = simdt/second           # convert to value in seconds

# ###########################################
# Neuron model
# ###########################################

# equation: dv/dt=(1/taum)*(-(v-el))
# with spike when v>vt, reset to vr

vt = 20.*mV         # Spiking threshold
taum = 20.*ms       # Membrane time constant
vr = 10.*mV         # Reset potential
muext0 = 24*mV      # external input to each neuron
taur = 0.5*ms       # Refractory period
taudelay = 0.75*ms  # synaptic delay

eqs_neurons='''
muext : volt
dv/dt=-v/taum + muext/taum : volt
'''

# ###########################################
# Network parameters: numbers
# ###########################################

N = 4096+1024       # Total number of neurons
fexc = 0.8          # Fraction of exc neurons
NE = int(fexc*N)    # Number of excitatory cells
NI = N-NE           # Number of inhibitory cells 

# ###########################################
# Network parameters: synapses
# ###########################################

rescale = 2         # rescale C and J to maintain total input
C = 1000/rescale    # Number of incoming connections on each neuron (exc or inh)
J = 0.02*mV*rescale # exc strength is J (in mV as we add to voltage)
                    # Critical J is ~ 0.45 mV in paper for N = 10000, C = 1000
g = 5.0             # -gJ is the inh strength. For exc-inh balance g>~f(1-f)=4

# ###########################################
# Network parameters: synaptic plasticity
# ###########################################

wmax = 5.               # hard bound on synaptic weight
Apre_tau = 20*ms        # STDP Apre LTP time constant
Apost_tau = 20*ms       # STDP Apost LTD time constant
Apostslow_tau = 100*ms
Aposthomeo_tau = 1*second
#Aposthomeo_tau = 1*second
Apre0 = 1.0             # incr in Apre, on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
Apostslow0 = 1.0        # incr in Apostslow on post spike
#rate0 = 50*Hz
rate0 = 25*Hz
Apost0 = Apre0*Apre_tau*(1+Apostslow0*Apostslow_tau*rate0)/Apost_tau
                        # incr in Apost on post spike for triplet rule
                        # only initial value, later calculated
print 'Apost0 =',Apost0
#beta = 1.2              # heterosynaptic plasticity strength parameter
beta = 0.5              # heterosynaptic plasticity strength parameter
eta = 1e-3              # learning rate
stdp_eqns = ''' wsyn : 1
                dApre/dt=-Apre/Apre_tau : 1 (event-driven)
                dApost/dt=-Apost/Apost_tau : 1 (event-driven)
                dApostslow/dt=-Apostslow/Apostslow_tau : 1 (event-driven)
                dAposthomeo/dt=-Aposthomeo/Aposthomeo_tau : 1 (event-driven) '''
pre_eqns = '''Apre += Apre0
                wsyn += -Apost*eta
                wsyn=clip(wsyn,0,wmax)
                v+=wsyn*J'''
#post_eqns = '''Apost += Apost0
#                wsyn += eta*(Apre + Apre*Apostslow)
#                Apostslow+=Apostslow0
#                wsyn=clip(wsyn,0,wmax)'''
post_eqns = '''Apost += Apost0
                wsyn += eta*Apre*(1 + Apostslow + beta*(Apostslow/(rate0*Apostslow_tau))**2)
                Apostslow+=Apostslow0
                wsyn=clip(wsyn,0,wmax)'''
#post_eqns = '''Apost += Apre0*Apre_tau*(1+Apostslow0*Apostslow_tau*(Aposthomeo/Aposthomeo_tau)**2/rate0)/Apost_tau
#                wsyn += eta*(Apre + Apre*Apostslow)
#                Apostslow+=Apostslow0
#                wsyn=clip(wsyn,0,wmax)'''

def dwbydt(r):
    return eta*(Apre0*Apre_tau/second-Apost0*Apost_tau/second)*r**2 + \
               eta*Apre0*Apre_tau/second*Apostslow0*Apostslow_tau/second*r**3 -\
               eta*beta*Apre0*Apre_tau/second * Apostslow0**2 * r**4 / (rate0/Hz)**2

figure()
rrange = arange(0,100,0.1)
plot(rrange,dwbydt(rrange))

#show()
#sys.exit(0)

# ###########################################
# Initialize neuron (sub)groups
# ###########################################

P=NeuronGroup(N,model=eqs_neurons,\
    threshold='v>=vt',reset='v=vr',refractory=taur,method='euler')
P.v = uniform(0.,vt/mV,N)*mV
PE = P[:NE]
PI = P[NE:]

# ###########################################
# Connecting the network 
# ###########################################

sparseness = C/float(N)
# E to E connections
#conEE = Synapses(PE,PE,'wsyn:1',on_pre='v_post+=wsyn*J',method='euler')
conEE = Synapses(PE,PE,stdp_eqns,\
                on_pre=pre_eqns,on_post=post_eqns,\
                method='euler')
#conEE.connect(condition='i!=j',p=sparseness)
# need exact connection indices for weight monitor in standalone mode
conEE_idxs_pre = []
conEE_idxs_post = []
Ce = int(fexc*C)
for k in range(NE):
    conEE_idxs_pre.extend(Ce*[k])
    idxs = range(NE)
    idxs.remove(k)      # no autapses i.e. no self-connections
    l = np.random.permutation(idxs)[:Ce]
    conEE_idxs_post.extend(l)
conEE_idxs_assembly = where(array(conEE_idxs_post)[:Ce*400]<400)[0]
conEE_idxs_cross = where(array(conEE_idxs_post)[:Ce*400]>400)[0]
conEE_idxs_bgnd = where(array(conEE_idxs_post)[Ce*400:]>400)[0]
conEE.connect(i=conEE_idxs_pre,j=conEE_idxs_post)
conEE.delay = taudelay
# Follow Dale's law -- exc (inh) neurons only have +ve (-ve) synapses
#  hence need to set w correctly (always set after creating connections)
conEE.wsyn = 1.

# E to I connections
conIE = Synapses(PE,PI,'wsyn:1',on_pre='v_post+=wsyn*J',method='euler')
conIE.connect(p=sparseness)
conIE.delay = taudelay
conIE.wsyn = 1

# I to E connections
conEI = Synapses(PI,PE,'wsyn:1',on_pre='v_post+=wsyn*J',method='euler')
conEI.connect(p=sparseness)
conEI.delay = taudelay
conEI.wsyn = -g

# I to I connections
conII = Synapses(PI,PI,'wsyn:1',on_pre='v_post+=wsyn*J',method='euler')
conII.connect(condition='i!=j',p=sparseness)
conII.delay = taudelay
conII.wsyn = -g

# ###########################################
# Stimuli
# ###########################################

P.muext = muext0
# 400 neurons (~10%) receive stimulus current to increase firing
Pstim = P[:400]
Pstim.muext = muext0 + 5*mV

# ###########################################
# Setting up monitors
# ###########################################

Nmon = N
sm = SpikeMonitor(P)

# Population monitor
popm = PopulationRateMonitor(P)

# voltage monitor
sm_vm = StateMonitor(P,'v',record=range(10)+range(NE,NE+10))

# weights monitor
wm = StateMonitor(conEE,'wsyn', record=range(Ce*NE), dt=simdt*1000)

# ###########################################
# Simulate
# ###########################################

print "Setup complete, running for",simtime,"at dt =",dt,"s."
t1 = time.time()
run(simtime,report='text')
device.build(directory='output', compile=True, run=True, debug=False)
print 'inittime + runtime, t = ', time.time() - t1

#print "For g,J =",g,J,"mean exc rate =",\
#    sm_e.num_spikes/float(NE)/(simtime/second),'Hz.'
#print "For g,J =",g,J,"mean inh rate =",\
#    sm_i.num_spikes/float(NI)/(simtime/second),'Hz.'

# ###########################################
# Make plots
# ###########################################

# always convert spikemon.t and spikemon.i to array-s before indexing
# spikemon.i[] indexing is extremely slow!
spiket = array(sm.t/second) # take spiketimes of all neurons
spikei = array(sm.i)

fig = figure()
# Vm plots
timeseries = arange(0,simtime/second+dt,dt)
for j in range(3):
    plot(timeseries[:len(sm_vm.t)],sm_vm[j].v)

fig = figure()
# raster plot
subplot(231)
plot(sm.t,sm.i,',')
title(str(N)+" exc & inh neurons")
xlim([0,simtime/second])
xlabel("")

# CV histogram
subplot(234)
CVarray = CV_spiketrains(spiket,spikei,0.3,range(NE))
print 'CV distribution: mean, min, and max =',\
            mean(CVarray),min(CVarray),max(CVarray)
hist(CVarray,bins=100) # from 0.3s on
xlabel('CV of ISI distribution')
ylabel('# of neurons')

print "plotting firing rates"
subplot(232)
tau=50e-3
sigma = tau/2.
# firing rates
timeseries = arange(0,simtime/second+dt,dt)
rate = np.zeros(int(simtime/simdt))
for nrni in range(400):
    rate += rate_from_spiketrain(spiket,spikei,simtime/second,sigma,dt,nrni)
plot(timeseries[:len(rate)],rate/400.,'r')
rate = np.zeros(int(simtime/simdt))
for nrni in range(400,800):
    rate += rate_from_spiketrain(spiket,spikei,simtime/second,sigma,dt,nrni)
plot(timeseries[:len(rate)],rate/400.,'b')
title("exc rates: assembly (r), bgnd (b)")
ylabel("Hz")
ylim(0,300)
subplot(235)
num_to_plot = 10
for nrni in range(NE,NE+num_to_plot):
    rate = rate_from_spiketrain(spiket,spikei,simtime/second,sigma,dt,nrni)
    plot(timeseries[:len(rate)],rate)
    #print mean(rate),len(sm_i[nrni])
    #rates.append(rate)
title(str(num_to_plot)+" inh rates")
ylim(0,300)
#print "Mean rate = ",mean(rates)
xlabel("time (s)")
ylabel("Hz")

print "plotting pop firing rates"
# Population firing rates
subplot(233)
timeseries = arange(0,simtime/second,dt)
plot(popm.t/second,popm.smooth_rate(width=50.*ms,window="gaussian")/Hz,
                            color='grey')
rate = rate_from_spiketrain(spiket,spikei,simtime/second,sigma,dt)/float(N)
plot(timeseries[:len(rate)],rate)
title("population rate")
ylabel("Hz")
xlabel("time (s)")

print "plotting weights"
subplot(236)
plot(wm.t/second,mean(wm.wsyn[conEE_idxs_assembly,:],axis=0),color='r')
plot(wm.t/second,mean(wm.wsyn[conEE_idxs_cross,:],axis=0),color='m')
plot(wm.t/second,mean(wm.wsyn[conEE_idxs_bgnd,:],axis=0),color='b')
title("assembly weights")
ylabel("arb")
xlabel("time (s)")

print conEE.wsyn

fig.tight_layout()

show()
