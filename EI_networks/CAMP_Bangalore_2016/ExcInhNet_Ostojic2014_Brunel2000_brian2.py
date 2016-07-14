#!/usr/bin/env python
'''
The LIF network is based on:
Ostojic, S. (2014).
 Two types of asynchronous activity in networks of excitatory and inhibitory spiking neurons.
 Nat Neurosci 17, 594-600.
which is in turn based on Brunel (2000).

author: Aditya Gilra.
Original version in Brian 1.4.1 for CAMP 2014, Bangalore, 20 June, 2014.
Updated to Brian2rc3 for CAMP 2016 by Aditya Gilra, Jun 2016.
'''

'''
1. Change mu,g to observe 'phase diagram' of Brunel (2000)
2. Change synaptic coupling J = 0.1, 0.2, 0.4, 0.6, 0.8 (mV).
3. Reduce N to observe finite size effects
 (don't forget to scale J to maintain original mean input in each neuron)
'''

#import modules and functions to be used
from brian2 import *    # importing brian also does:
                        # 'from pylab import *' which imports:
                        # matplot like commands into the namespace, further
                        # also can use np. for numpy and mpl. for matplotlib
from data_utils import *

stand_alone = False#True
if stand_alone: set_device('cpp_standalone', build_on_run=False)
else:
    #prefs.codegen.target = 'numpy'
    prefs.codegen.target = 'weave'
    #prefs.codegen.target = 'cython'
import random
import time

np.random.seed(100) # set seed for reproducibility of simulations
random.seed(100) # set seed for reproducibility of simulations

# ###########################################
# Simulation parameters
# ###########################################

simdt = 0.1*ms
simtime = 1.0*second
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
muext = 24*mV       # input to each neuron
taur = 0.5*ms       # Refractory period
taudelay = 0.75*ms  # synaptic delay

eqs_neurons='''
dv/dt=-v/taum+muext/taum : volt
'''

# ###########################################
# Network parameters: numbers
# ###########################################

N = 10000           # Total number of neurons
fexc = 0.8          # Fraction of exc neurons
NE = int(fexc*N)    # Number of excitatory cells
NI = N-NE           # Number of inhibitory cells 

# ###########################################
# Network parameters: synapses
# ###########################################

C = 1000            # Number of incoming connections on each neuron (exc or inh)
J = 0.2*mV          # exc strength is J (in mV as we add to voltage)
                    # Critical J is ~ 0.45 mV in paper for N = 10000, C = 1000
g = 5.0             # -gJ is the inh strength. For exc-inh balance g>~f(1-f)=4

# ###########################################
# Initialize neuron (sub)groups
# ###########################################

P=NeuronGroup(N,model=eqs_neurons,\
    threshold='v>=vt',reset='v=vr',refractory=taur,method='euler')
P.v = uniform(0.,vt/mV,N)*mV

# ###########################################
# Connecting the network 
# ###########################################

sparseness = C/float(N)
con = Synapses(P,P,'w:volt (constant)',on_pre='v_post+=w',method='euler')
con.connect(condition='i!=j',p=sparseness)
con.delay = taudelay
# Follow Dale's law -- exc (inh) neurons only have +ve (-ve) synapses
#  hence need to set w correctly (always set after creating connections)
con.w['i<NE'] = J
con.w['i>=NE'] = -g*J

# ###########################################
# Setting up monitors
# ###########################################

Nmon = N
sm = SpikeMonitor(P)

# Population monitor
popm = PopulationRateMonitor(P)

# voltage monitor
sm_vm = StateMonitor(P,'v',record=range(10)+range(NE,NE+10))

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
num_to_plot = 10
#rates = []
for nrni in range(num_to_plot):
    rate = rate_from_spiketrain(spiket,spikei,simtime/second,sigma,dt,nrni)
    plot(timeseries[:len(rate)],rate)
    #print mean(rate),len(sm_e[nrni])
    #rates.append(rate)
title(str(num_to_plot)+" exc rates")
ylabel("Hz")
ylim(0,300)
subplot(235)
for nrni in range(NE,NE+num_to_plot):
    rate = rate_from_spiketrain(spiket,spikei,simtime/second,sigma,dt,nrni)
    plot(timeseries[:len(rate)],rate)
    #print mean(rate),len(sm_i[nrni])
    #rates.append(rate)
title(str(num_to_plot)+" inh rates")
ylim(0,300)
#print "Mean rate = ",mean(rates)
xlabel("Time (s)")
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
xlabel("Time (s)")

fig.tight_layout()

show()
