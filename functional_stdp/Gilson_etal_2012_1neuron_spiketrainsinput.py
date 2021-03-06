#!/usr/bin/env python
'''
Based on:
Gilson, Matthieu, Tomoki Fukai, and Anthony N. Burkitt.
Spectral Analysis of Input Spike Trains by Spike-Timing-Dependent Plasticity.
PLoS Comput Biol 8, no. 7 (July 5, 2012): e1002584. doi:10.1371/journal.pcbi.1002584.

Author: Aditya Gilra, Jun 2016. (with inputs from Matthieu Gilson)
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

Npools = 4              # Number of correlated pools
Ninp = 500               # Number of neurons per pool
nu0 = 10*Hz             # spiking rate of inputs

# ###########################################
# Network parameters: synapses
# ###########################################

Q = array([[sqrt(0.4),sqrt(0.1),0.,0.],\
            [0.,sqrt(0.2),sqrt(0.2),0.],\
            [0.,0.,sqrt(0.1),sqrt(0.1)]])
Q = np.zeros(shape=(3,4))
corr = dot(transpose(Q),Q)
print "Correlation matrix between pools is\n",corr

# ###########################################
# Network parameters: synaptic plasticity
# ###########################################

eta = 2e-4              # learning rate (as in paper)
#eta = 1e-3              # learning rate (problem!)
Apre_tau = 17*ms        # STDP Apre (LTP) time constant
Apost_tau = 34*ms       # STDP Apost (LTD) time constant
stdp_eqns = ''' w : 1
                dApre/dt=-Apre/Apre_tau : 1 (event-driven)
                dApost/dt=-Apost/Apost_tau : 1 (event-driven)
            '''
#stdp_type = 'log-stdp'
#stdp_type = 'log-stdp-post'
#stdp_type = 'log-stdp-noise'
#stdp_type = 'log-stdp2'
#stdp_type = 'log-stdp2-post'
#stdp_type = 'log-stdp2-noise'
#stdp_type = 'log-stdp2-post-noise'
#stdp_type = 'mlt-stdp'
#stdp_type = 'nlta-stdp'
stdp_type = 'add-stdp'
if stdp_type == 'log-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.5        # incr in Apost (LTD) on post spike
    beta = 50           # LTP decay factor
    alpha = 5           # LTD curvature factor
    w0 = 0.005          # reference weight
    pre_eqns = 'Apre+=Apre0; w = clip(w - eta*Apost*((w<=w0)*w/w0 +'\
                    ' (w>w0)*(1+log(1+(w>w0)*alpha*(w/w0-1))/alpha)), 0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + eta*Apre*exp(-w/w0/beta), 0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
if stdp_type == 'log-stdp-noise':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.5        # incr in Apost (LTD) on post spike
    beta = 50           # LTP decay factor
    alpha = 5           # LTD curvature factor
    w0 = 0.005          # reference weight
    std_noise = 1.5     # std deviation of noise
    pre_eqns = 'Apre+=Apre0; w = clip(w - eta*Apost*((w<=w0)*w/w0 +'\
                    ' (w>w0)*(1+log(1+(w>w0)*alpha*(w/w0-1))/alpha))*(randn()*std_noise+1), 0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + eta*Apre*exp(-w/w0/beta), 0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
if stdp_type == 'log-stdp-post':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.5        # incr in Apost (LTD) on post spike
    beta = 50           # LTP decay factor
    alpha = 5           # LTD curvature factor
    w0 = 0.005          # reference weight
    ain = 0.1           # purely pre-synaptic contribution
    aout = -0.05        # purely post-synaptic contribution
    pre_eqns = 'Apre+=Apre0; w = clip(w - eta*Apost*((w<=w0)*w/w0 +'\
                    ' (w>w0)*(1+log(1+(w>w0)*alpha*(w/w0-1))/alpha)) + eta*ain, 0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + eta*Apre*exp(-w/w0/beta) + eta*aout, 0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
elif stdp_type == 'log-stdp2':
    w0 = 0.05           # reference weight
    Apre0 = 0.05*w0     # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = Apre0 * Apre_tau / Apost_tau
                        # incr in Apost (LTD) on post spike
    beta = 50           # LTP decay factor
    alpha = 5           # LTD curvature factor
    pre_eqns = 'Apre+=Apre0; w = clip(w - Apost*log(1+w/w0*alpha)/log(1+alpha), 0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + Apre*exp(-w/w0/beta), 0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
elif stdp_type == 'log-stdp2-post':
    w0 = 0.05           # reference weight
    Apre0 = 0.05*w0     # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = Apre0 * Apre_tau / Apost_tau
                        # incr in Apost (LTD) on post spike
    ain = 0.1*w0        # purely pre-synaptic contribution
    aout = -0.05*w0     # purely post-synaptic contribution
    beta = 50           # LTP decay factor
    alpha = 20          # LTD curvature factor
    pre_eqns = 'Apre+=Apre0; w = clip(w - Apost*log(1+w/w0*alpha)/log(1+alpha) + ain,0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + Apre*exp(-w/w0/beta) + aout,0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
elif stdp_type == 'log-stdp2-noise':
    w0 = 0.05           # reference weight
    Apre0 = 0.05*w0     # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = Apre0 * Apre_tau / Apost_tau
                        # incr in Apost (LTD) on post spike
    beta = 50           # LTP decay factor
    alpha = 20          # LTD curvature factor
    std_noise = 0.6     # std deviation of noise
    pre_eqns = 'Apre+=Apre0; w = clip(w - Apost*log(1+w/w0*alpha)/log(1+alpha)*(randn()*std_noise+1),0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + Apre*exp(-w/w0/beta),0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
elif stdp_type == 'log-stdp2-post-noise':
    w0 = 0.05           # reference weight
    Apre0 = 0.05*w0     # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = Apre0 * Apre_tau / Apost_tau
                        # incr in Apost (LTD) on post spike
    ain = 0.1*w0        # purely pre-synaptic contribution
    aout = -0.05*w0     # purely post-synaptic contribution
    beta = 50           # LTP decay factor
    alpha = 20          # LTD curvature factor
    std_noise = 0.6     # std deviation of noise
    pre_eqns = 'Apre+=Apre0; w = clip(w - Apost*log(1+w/w0*alpha)/log(1+alpha)*(randn()*std_noise+1) + ain,0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + Apre*exp(-w/w0/beta) + aout,0,inf)'
    winit = 2*w0        # initial weights are from 0 to winit
elif stdp_type == 'mlt-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 100        # incr in Apost (LTD) on post spike
    pre_eqns = 'Apre+=Apre0; w = clip(w - eta*Apost*w,0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + eta*Apre,0,inf)'
    winit = 0.04        # initial weights are from 0 to winit
    w0 = 0.04
elif stdp_type == 'nlta-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.8        # incr in Apost (LTD) on post spike
    wmax = 0.04         # max weight (soft bound)
    gamma = 0.1         # weight dependence exponent
    pre_eqns = 'Apre+=Apre0; w = clip(w - eta*Apost*(w/wmax)**gamma, 0,inf)'
    post_eqns = 'Apost+=Apost0; w = clip(w + eta*Apre*(1-w/wmax)**gamma, 0,inf)'
    winit = wmax        # initial weights are from 0 to winit
elif stdp_type == 'add-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.55       # incr in Apost (LTD) on post spike
    wmax = 0.04         # max weight (hard bound)
    pre_eqns = 'Apre+=Apre0; w+=-eta*Apost;'\
                    ' w=clip(w,0,wmax)'
    post_eqns = 'Apost+=Apost0; w += eta*Apre;'\
                    ' w=clip(w,0,wmax)'
    winit = wmax        # initial weights are from 0 to winit
    w0 = wmax

# ###########################################
# Initialize neuron (sub)groups
# ###########################################

# post-synaptic neuron
P=NeuronGroup(1,model=eqs_neurons,\
                threshold='rand()<rho_out*dt',method='euler')

# ###########################################
# Stimuli
# ###########################################

# generate Poisson spike trains of size rate*T
num_events = np.random.poisson(nu0*simtime,size=Npools*Ninp)
spiketimes = []
for k in range(Npools*Ninp):
    spiketimes.append(simtime/second*np.random.rand(num_events[k]))

# for the 3 driving neurons, generate spike trains
for k in range(3):
    num_events = np.random.poisson(nu0*simtime)
    spiketimes_drive = simtime/second*np.random.rand(num_events)
    # for each pool, and each neuron within a pool,
    #  select some of these spikes based on the correlation
    for l in range(Npools):
        if Q[k,l]!=0.:
            for m in range(Ninp):
                spikes_selected = spiketimes_drive[where(np.random.rand(len(spiketimes_drive))<Q[k,l])]
                spiketimes[l*Ninp+m] = np.append(spiketimes[l*Ninp+m],spikes_selected)

# flatten spiketimes and remove spikes in the same timebin for each neuron
indices = []
spiketimes_flat = array([])
for k in range(Npools*Ninp):
    spiketimes[k]=np.sort(spiketimes[k])
    # keep removing spikes that overlap until none do
    while True:
        diffs = np.diff(spiketimes[k])
        good_idxs = where(abs(diffs)>simdt/second)[0]
        if len(good_idxs)+1==len(spiketimes[k]): break
        spiketimes[k] = spiketimes[k][np.append(good_idxs,-1)]
    indices.extend([k]*len(spiketimes[k]))
    spiketimes_flat = append(spiketimes_flat,spiketimes[k])

#Pinp1=SpikeGeneratorGroup(Npools*Ninp,np.array(indices),np.array(spiketimes_flat)*second)
Pinp1 = PoissonGroup(Npools*Ninp,rates=nu0)

# ###########################################
# Connecting the network 
# ###########################################

con = Synapses(Pinp1,P,stdp_eqns,\
                on_pre='A+=w*0.1;B+=w*0.1;'+pre_eqns,on_post=post_eqns,
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
subplot(221)
plot(sminp1.t/second,sminp1.i,',')
xlim([0,1])
xlabel("time (s)")

# weight evolution 
subplot(222)
meanpoolw = []
for k in range(Npools):
    meanpoolw.append(mean(wm.w[k*Ninp:(k+1)*Ninp,:],axis=0)/w0)
    plot(wm.t/second,meanpoolw[-1],'-'+cols[k],lw=(4-k))
xlabel("time (s)")
ylabel("PCA-weight")
title('weight evolution')

# plot output firing rate sm_rho.rho_out[nrn_idx,time_idx]
subplot(232)
plot(sm_rho.t/second,sm_rho.rho_out[0]/Hz,'-')
xlim([0,simtime/second])
xlabel("")

# plot final weights wm.w[syn_idx,time_idx]
subplot(223)
plot(range(Npools*Ninp),wm.w[:,-1],'.')
for k in range(Npools):
    meanpoolw_end = mean(wm.w[k*Ninp:(k+1)*Ninp,-1])
    plot([k*Ninp,(k+1)*Ninp],[meanpoolw_end,meanpoolw_end],'-'+cols[k],lw=3)
xlabel("pre-neuron #")
ylabel("weight (/w0)")
title("end weights")

# plot averaged weights over last 50s (weights are sampled per second)
subplot(224)
plot(range(Npools*Ninp),mean(wm.w[:,:50],axis=1),'.')
for k in range(Npools):
    meanpoolw_end = mean(wm.w[k*Ninp:(k+1)*Ninp,-50:])
    plot([k*Ninp,(k+1)*Ninp],[meanpoolw_end,meanpoolw_end],'-'+cols[k],lw=3)
xlabel("pre-neuron #")
ylabel("weight (/w0)")
title("mean (50s) end weights")

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
title('weights along PCA components')

# weight evolution along original drivers
subplot(133)
for k in range(3):
    plot(wm.t/second,dot(Q[k],meanpoolw),'-'+cols[k],lw=(4-k))
xlabel('time (s)')
ylabel("driver neuron #")
yticks([0,1,2])
title('weights along ICA components')

#fig.tight_layout()
#fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

figure()
hist(wm.w[:,-1],bins=200)

show()
