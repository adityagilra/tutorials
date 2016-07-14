#!/usr/bin/env python
'''
Author: Aditya Gilra, Jun 2016. (with inputs from Matthieu Gilson)
 in Brian2rc3 for CAMP 2016.
'''

#import modules and functions to be used
from brian2 import *    # importing brian also does:
                        # 'from pylab import *' which imports:
                        # matplot like commands into the namespace, further
                        # also can use np. for numpy and mpl. for matplotlib

stand_alone = True
if stand_alone: set_device('cpp_standalone', build_on_run=False)
else:
    #prefs.codegen.target = 'numpy'
    prefs.codegen.target = 'weave'
    #prefs.codegen.target = 'cython'
import random
import time

# http://stackoverflow.com/questions/31057197/should-i-use-random-seed-or-numpy-random-seed-to-control-random-number-gener
np.random.seed(0)       # set seed for reproducibility of simulations
random.seed(0)          # set seed for reproducibility of simulations

# ###########################################
# Simulation parameters
# ###########################################

simdt = 0.1*ms
simtime = 3.0*second
defaultclock.dt = simdt # set Brian's sim time step
simdtraw = simdt/second # convert to value in seconds

# ###########################################
# Neuron model
# ###########################################

taudelay = 0.75*ms      # synaptic delay
tauA = 1*ms             # synaptic epsp tauA
tauB = 5*ms             # synaptic epsp tauB

eqs_neurons='''
dA/dt=-A/tauA : 1
dB/dt=-B/tauB : 1
rho_out = (A-B)/(tauA-tauB) : Hz
'''

# ###########################################
# Network parameters: numbers
# ###########################################

Ninp = 2000             # Number of neurons per pool
nu0 = 10*Hz             # spiking rate of inputs

# ###########################################
# Network parameters: synaptic plasticity
# ###########################################

eta = 1e-2              # learning rate (as in paper)
Apre_tau = 17*ms        # STDP Apre (LTP) time constant
Apost_tau = 34*ms       # STDP Apost (LTD) time constant
stdp_eqns = ''' w : 1
                dApre/dt=-Apre/Apre_tau : 1 (event-driven)
                dApost/dt=-Apost/Apost_tau : 1 (event-driven)
            '''
Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                    # at spike coincidence, delta w = -Apre0*eta
Apost0 = 0.55       # incr in Apost (LTD) on post spike
wmax = 0.04         # max weight (hard bound)
winit = wmax/2.0    # initial weights are from 0 to winit
w0 = wmax/2.0
pre_eqns = 'Apre+=Apre0; w+=-eta*Apost;'\
                ' w=clip(w,0,wmax)'
post_eqns = 'Apost+=Apost0; w += eta*Apre;'\
                ' w=clip(w,0,wmax)'

# ###########################################
# Initialize neuron (sub)groups
# ###########################################

# post-synaptic neuron
P=NeuronGroup(1,model=eqs_neurons,\
                threshold='rand()<rho_out*dt',method='euler')

# ###########################################
# Stimuli
# ###########################################

Pinp1 = PoissonGroup(Ninp,rates=nu0)

# ###########################################
# Connecting the network 
# ###########################################

con = Synapses(Pinp1,P,stdp_eqns,\
                on_pre='A+=w*0.1;B+=w*0.1;'+pre_eqns,on_post=post_eqns,
                method='euler')
con.connect(True)
con.delay = uniform(size=(Ninp,))*1.*ms + 4.*ms
con.w = uniform(size=(Ninp,))*2*winit

# ###########################################
# Setting up monitors
# ###########################################

sm = SpikeMonitor(P)
sminp1 = SpikeMonitor(Pinp1)

# Population monitor
popm = PopulationRateMonitor(P)
popminp1 = PopulationRateMonitor(Pinp1)

# voltage monitor
sm_rho = StateMonitor(P,'rho_out',record=[0])

# weights monitor
wm = StateMonitor(con,'w',record=range(Ninp), dt=simtime/100.)

# ###########################################
# Simulate
# ###########################################

# a simple run would not include the monitors
net = Network(collect())            # collects Brian2 objects in current context

print "Setup complete, running for",simtime,"at dt =",simdtraw,"s."
t1 = time.time()
net.run(simtime,report='text')
device.build(directory='output', compile=True, run=True, debug=False)

# ###########################################
# Make plots
# ###########################################

# always convert spikemon.t and spikemon.i to array-s before indexing
# spikemon.i[] indexing is extremely slow!
spiket = array(sm.t/second) # take spiketimes of all neurons
spikei = array(sm.i)

fig = figure()

subplot(131)
plot(popm.t/second,popm.smooth_rate(width=50.*ms,window="gaussian")/Hz,',-')
xlabel('time (s)')
ylabel('post-rate (Hz)')

# weight evolution
subplot(132)
plot(wm.t/second,transpose(wm.w[:20,:]),',-')
xlabel('time (s)')
ylabel("weight (arb)")
yticks([0,1,2])
title('weights evolution')

subplot(133)
hist(wm.w[:,-1],bins=50,edgecolor='none')
xlabel('weight')
ylabel('count')

show()
