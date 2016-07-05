#!/usr/bin/env python
'''
Based on:
Gilson, Matthieu, Tomoki Fukai, and Anthony N. Burkitt.
Spectral Analysis of Input Spike Trains by Spike-Timing-Dependent Plasticity.
PLoS Comput Biol 8, no. 7 (July 5, 2012): e1002584. doi:10.1371/journal.pcbi.1002584.

Author: Aditya Gilra, Jun 2016.
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
    prefs.codegen.target = 'numpy'
    #prefs.codegen.target = 'weave'
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
simtime = 500.0*second
defaultclock.dt = simdt # set Brian's sim time step
simdtraw = simdt/second # convert to value in seconds

# ###########################################
# Neuron model
# ###########################################

# equation: dv/dt=(1/taum)*(-(v-el))
# with spike when v>vt, reset to vr

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
Ninp = 50               # Number of neurons per pool
nu0 = 10*Hz             # spiking rate of inputs
corr_ev_duration = 4*ms #duration of correlated events
corr_ev_freq = 20*Hz    #frequency of correlated events

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
#eta = 1e-3              # learning rate (problem!)
Apre_tau = 17*ms        # STDP Apre (LTP) time constant
Apost_tau = 34*ms       # STDP Apost (LTD) time constant
stdp_eqns = ''' w : 1
                dApre/dt=-Apre/Apre_tau : 1 (event-driven)
                dApost/dt=-Apost/Apost_tau : 1 (event-driven)
            '''
stdp_type = 'log-stdp'
#stdp_type = 'mlt-stdp'
#stdp_type = 'nlta-stdp'
#stdp_type = 'add-stdp'
if stdp_type == 'log-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.5        # incr in Apost (LTD) on post spike
    beta = 50           # LTP decay factor
    alpha = 5           # LTD curvature factor
    w0 = 0.005          # reference weight
    pre_eqns = 'Apre+=Apre0; w+=-eta*Apost*((w<=w0)*w/w0 +'\
                    ' (w>w0)*(1+log(1+(w>w0)*alpha*(w/w0-1))/alpha))'
    post_eqns = 'Apost+=Apost0; w += eta*Apre*exp(-w/w0/beta)'
    winit = 2*w0        # initial weights are from 0 to winit
elif stdp_type == 'mlt-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 100        # incr in Apost (LTD) on post spike
    pre_eqns = 'Apre+=Apre0; w+=-eta*Apost*w'
    post_eqns = 'Apost+=Apost0; w += eta*Apre'
    winit = 0.04        # initial weights are from 0 to winit
elif stdp_type == 'nlta-stdp':
    Apre0 = 1.0         # incr in Apre (LTP), on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
    Apost0 = 0.8        # incr in Apost (LTD) on post spike
    wmax = 0.04         # max weight (soft bound)
    gamma = 0.1         # weight dependence exponent
    pre_eqns = 'Apre+=Apre0; w+=-eta*Apost*(w/wmax)**gamma'
    post_eqns = 'Apost+=Apost0; w += eta*Apre*(1-w/wmax)**gamma'
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
baseline_input_rates = np.zeros(Npools*Ninp)
for i_gp in range(Npools):
    baseline_input_rates[i_gp*Ninp:(i_gp+1)*Ninp] = nu0/Hz*(1.-Q[:,i_gp].sum()*corr_ev_freq*corr_ev_duration)
    #print baseline_input_rates[i_gp*N1]
if baseline_input_rates.min()<nu0/2/Hz:
    print 'too strong corr'
    xxx

correlated_inputs = True
#correlated input spike trains with 4 groups and mixing matrix A between groups
if correlated_inputs:
    #correlated inputs
    Pinp1 = PoissonGroup(Npools*Ninp,rates=nu0)
    Ppools = []
    for k in range(Npools):
        Ppools.append(Pinp1[k*Ninp:(k+1)*Ninp])
    #driver inputs
    Pinp0 = PoissonGroup(3,rates=corr_ev_freq,dt=corr_ev_duration)
    Pinps = [Pinp0[:1],Pinp0[1:2],Pinp0[2:]]

    inpconns = []
    def correlate_spike_trains(PR,P,k,l):
        con = Synapses(PR,P,'',on_pre='spikerate='+\
                    str(baseline_input_rates[k]+Q[k,l]*nu0/Hz)+'*Hz')
        con.connect(True)
        con.delay = 0.*ms
        con1 = Synapses(PR,P,'',on_pre='spikerate=nu0')
        con1.connect(True)
        con1.delay = corr_ev_duration
        inpconns.append((con,con1))
    for k in range(3):
        for l in range(Npools):
            if Q[k,l]!=0.:
                correlate_spike_trains(Pinps[k],Ppools[l],k,l)
else:
    #uncorrelated inputs
    Pinp1 = PoissonGroup(Npools*Ninp, rates=nu0)

# ###########################################
# Connecting the network 
# ###########################################

con = Synapses(Pinp1,P,stdp_eqns,\
                on_pre='A+=w;B+=w;'+pre_eqns,on_post=post_eqns,
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

# raster plot
subplot(234)
plot(sminp1.t/second,sminp1.i,',')
xlim([0,1])
xlabel("time (s)")

## plot output firing rate sm_rho.rho_out[nrn_idx,time_idx]
#subplot(232)
#plot(sm_rho.t/second,sm_rho.rho_out[0]/Hz,'-')
#xlim([0,simtime/second])
#xlabel("")

# plot final weights wm.w[syn_idx,time_idx]
subplot(233)
plot(range(Npools*Ninp),wm.w[:,-1],'.')
for k in range(Npools):
    meanpoolw = mean(wm.w[k*Ninp:(k+1)*Ninp,-1])
    plot([k*Ninp,(k+1)*Ninp],[meanpoolw,meanpoolw],'-k')
xlabel("")

# plot eigenvectors of corr = Q^T Q matrix
w,v = np.linalg.eig(corr)
subplot(235)
plot(v)

# plot averaged weights over last 50s
subplot(236)
plot(range(Npools*Ninp),mean(wm.w[:,-50:],axis=1),'.')
for k in range(Npools):
    meanpoolw = mean(wm.w[k*Ninp:(k+1)*Ninp,-50:])
    plot([k*Ninp,(k+1)*Ninp],[meanpoolw,meanpoolw],'-k')
xlabel("")

fig.tight_layout()

show()
