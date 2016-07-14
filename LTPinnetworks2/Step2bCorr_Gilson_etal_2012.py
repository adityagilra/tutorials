#!/usr/bin/env python
'''
Based on:
Gilson, Matthieu, Tomoki Fukai, and Anthony N. Burkitt.
Spectral Analysis of Input Spike Trains by Spike-Timing-Dependent Plasticity.
PLoS Comput Biol 8, no. 7 (July 5, 2012): e1002584. doi:10.1371/journal.pcbi.1002584.

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
simtime = 100.0*second
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

Npools = 4              # Number of correlated pools
Ninp = 500               # Number of neurons per pool
nu0 = 10*Hz             # spiking rate of inputs

# ###########################################
# Network parameters: synapses
# ###########################################

Q = array([[sqrt(0.4),sqrt(0.1),0.,0.],\
            [0.,sqrt(0.2),sqrt(0.2),0.],\
            [0.,0.,sqrt(0.1),sqrt(0.1)]])
corr = dot(transpose(Q),Q)
print "Correlation matrix between pools is\n",corr

# ###########################################
# Network parameters: synaptic plasticity
# ###########################################

eta = 2e-4              # learning rate (as in paper)
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
winit = wmax/2.     # initial weights are from 0 to winit
#w0 = wmax/2.
w0 = wmax
pre_eqns = 'Apre+=Apre0; w+=-eta*Apost*w/w0;'\
                ' w=clip(w,0,inf)'
post_eqns = 'Apost+=Apost0; w += eta*Apre;'\
                ' w=clip(w,0,inf)'

# ###########################################
# Initialize neuron (sub)groups
# ###########################################

# post-synaptic neuron
P=NeuronGroup(1,model=eqs_neurons,\
                threshold='rand()<rho_out*dt',method='euler')

# ###########################################
# Stimuli
# ###########################################

#inputs rates for absence of correlated events such that all neurons have same firing rate F
baseline_input_rates = np.zeros(Npools*Ninp)*Hz
for i_gp in range(Npools):
    baseline_input_rates[i_gp*Ninp:(i_gp+1)*Ninp] = nu0*(1.-Q[:,i_gp].sum())
    print baseline_input_rates[i_gp*Ninp]

# 3 driving spike trains
Pinp0=NeuronGroup(3,model='spikerate : Hz',\
                threshold='rand()<spikerate*dt')
Pinp0.spikerate = nu0
Pinps = [Pinp0[:1],Pinp0[1:2],Pinp0[2:]]

# Npools number of Ninp spike trains
Pinp1=NeuronGroup(Npools*Ninp,model='spikerate : Hz',\
                threshold='rand()<spikerate*dt')
Pinp1.spikerate = baseline_input_rates
Ppools = []
for k in range(Npools):
    Ppools.append(Pinp1[k*Ninp:(k+1)*Ninp])

inpconns = []
def correlate_spike_trains(PR,P,l,csqrt):
    con = Synapses(PR,P,'',on_pre='spikerate+='+str(csqrt)+'/dt')
    con.connect(True)
    con.delay = 0.*ms
    con1 = Synapses(PR,P,'',on_pre='spikerate='+str(baseline_input_rates[l*Ninp]/Hz)+'*Hz')
    con1.connect(True)
    con1.delay = simdt
    inpconns.append((con,con1))
for k in range(3):
    for l in range(Npools):
        if Q[k,l]!=0.:
            correlate_spike_trains(Pinps[k],Ppools[l],l,Q[k,l])

# ###########################################
# Connecting the network 
# ###########################################

con = Synapses(Pinp1,P,stdp_eqns,\
                on_pre='A+=w*0.05;B+=w*0.05;'+pre_eqns,on_post=post_eqns,
                method='euler')
con.connect(True)
con.delay = uniform(size=(Npools*Ninp,))*1.*ms + 4.*ms
con.w = uniform(size=(Npools*Ninp,))*2*winit

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
wm = StateMonitor(con,'w',record=range(Npools*Ninp), dt=1*second)

# ###########################################
# Simulate
# ###########################################

# a simple run would not include the monitors
net = Network(collect())            # collects Brian2 objects in current context
net.add(inpconns)

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
cols = ['r','b','g','c']

# raster plot
subplot(231)
plot(sminp1.t/second,sminp1.i,',')
xlim([0,1])
xlabel("time (s)")

# weight evolution 
subplot(232)
meanpoolw = []
for k in range(Npools):
    meanpoolw.append(mean(wm.w[k*Ninp:(k+1)*Ninp,:],axis=0)/w0)
    plot(wm.t/second,meanpoolw[-1],'-'+cols[k],lw=(4-k))
xlabel("time (s)")
ylabel("PCA-weight")
title('weight evolution')

# plot output firing rate sm_rho.rho_out[nrn_idx,time_idx]
subplot(234)
plot(sm_rho.t/second,sm_rho.rho_out[0]/Hz,'-')
xlim([0,simtime/second])
xlabel("")

# plot final weights wm.w[syn_idx,time_idx]
subplot(233)
plot(range(Npools*Ninp),wm.w[:,-1],'.')
for k in range(Npools):
    meanpoolw_end = mean(wm.w[k*Ninp:(k+1)*Ninp,-1])
    plot([k*Ninp,(k+1)*Ninp],[meanpoolw_end,meanpoolw_end],'-'+cols[k],lw=3)
xlabel("pre-neuron #")
ylabel("weight (/w0)")
title("end wts")

# plot averaged weights over last 50s (weights are sampled per second)
subplot(236)
plot(range(Npools*Ninp),mean(wm.w[:,-50:],axis=1),'.')
for k in range(Npools):
    meanpoolw_end = mean(wm.w[k*Ninp:(k+1)*Ninp,-50:])
    plot([k*Ninp,(k+1)*Ninp],[meanpoolw_end,meanpoolw_end],'-'+cols[k],lw=3)
xlabel("pre-neuron #")
ylabel("weight (/w0)")
title("mean (50s) end wts")

fig = figure()

# plot eigenvectors of corr = Q^T Q matrix
w,v = np.linalg.eig(corr)
subplot(131)
#plot(v)
for k in range(Npools):
    plot(v[:,k],'.-'+cols[k],lw=4-k)
xlabel("pre-neuron #")
ylabel("weight (/w0)")
title('eigenvectors of corr matrix')

# weight evolution along eigenvectors of corr matrix
subplot(132)
for k in range(Npools):
    plot(wm.t/second,dot(v[:,k],meanpoolw),'-'+cols[k],lw=(4-k))
xlabel('time (s)')
ylabel("weight (/w0)")
title('weights along PCs')

subplot(133)
hist(wm.w[:,-1],bins=200)

#fig.tight_layout()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

show()
