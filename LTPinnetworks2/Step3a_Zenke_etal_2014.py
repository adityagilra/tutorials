#!/usr/bin/env python
'''
Based on:
Zenke, Friedemann, Everton J. Agnes, and Wulfram Gerstner.
 "Diverse Synaptic Plasticity Mechanisms Orchestrated to Form and Retrieve Memories in Spiking Neural Networks."
 Nature Communications 6 (April 21, 2015).

Part of Zenke's rule embedded in modified Brunel 2000 / Ostojic 2014 network

author: Aditya Gilra, Jun 2016.
 in Brian2rc3 for CAMP 2016.
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

np.random.seed(100)     # set seed for reproducibility of simulations
random.seed(100)        # set seed for reproducibility of simulations

# ###########################################
# Simulation parameters
# ###########################################

simdt = 0.1*ms
simtime = 10*second
defaultclock.dt = simdt # set Brian's sim time step
dt = simdt/second       # convert to value in seconds

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
J = 0.01*mV*rescale # exc strength is J (in mV as we add to voltage)
                    # Critical J is ~ 0.45 mV in paper for N = 10000, C = 1000
g = 5.0             # -gJ is the inh strength. For exc-inh balance g>~f(1-f)=4

# ###########################################
# Network parameters: synaptic plasticity
# ###########################################

wmax = 10.              # hard bound on synaptic weight
Apre_tau = 20*ms        # STDP Apre LTP time constant; tauplus
Apost_tau = 20*ms       # STDP Apost LTD time constant; tauminus
Apre0 = 1.0             # incr in Apre, on pre-spikes; Aplus for LTP
                        # at spike coincidence, delta w = -Apre0*eta
Apost0 = 1.0            # incr in Apost on post-spikes; Aminus for LTD
eta = 5e-2              # learning rate
Apostslow0 = 1.0        # incr in Apostslow on post spike
Apostslow_tau = 100*ms

pre_eqns = '''Apre += Apre0
                wsyn += -Apost*eta
                wsyn=clip(wsyn,0,wmax)
                v+=wsyn*J'''
stdp_eqns = ''' wsyn : 1
                dApre/dt=-Apre/Apre_tau : 1 (event-driven)
                dApost/dt=-Apost/Apost_tau : 1 (event-driven)'''
post_eqns = '''Apost += Apost0
                wsyn += eta*Apre
                wsyn=clip(wsyn,0,wmax)'''
def dwbydt(r):
    return eta*(Apre0*Apre_tau/second - Apost0*Apost_tau/second)*r**2

figure()
rrange = arange(0,90,0.1)
plot(rrange,dwbydt(rrange))

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
## 400 neurons (~10%) receive stimulus current to increase firing
#Pstim = P[:400]
#Pstim.muext = muext0 + 7*mV

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
wm = StateMonitor(conEE,'wsyn', record=range(Ce*NE), dt=simtime/20.)

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
# raster plot
subplot(231)
plot(sm.t,sm.i,',')
title(str(N)+" exc & inh neurons")
xlim([simtime/second-1,simtime/second])
xlabel("")

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

subplot(233)
hist(wm.wsyn[:,-1],bins=500,edgecolor='none')
xlabel('weight')
ylabel('count')

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

print "plotting weights"
subplot(236)
plot(wm.t/second,mean(wm.wsyn[conEE_idxs_assembly,:],axis=0),color='r')
plot(wm.t/second,mean(wm.wsyn[conEE_idxs_cross,:],axis=0),color='m')
plot(wm.t/second,mean(wm.wsyn[conEE_idxs_bgnd,:],axis=0),color='b')
title("assembly weights (cross=m)")
ylabel("arb")
xlabel("time (s)")

print conEE.wsyn

fig.tight_layout()

show()
