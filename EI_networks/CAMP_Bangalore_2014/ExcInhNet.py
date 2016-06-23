#!/usr/bin/env python
'''
The LIF network is based on:
Ostojic, S. (2014).
 Two types of asynchronous activity in networks of
 excitatory and inhibitory spiking neurons.
 Nat Neurosci 17, 594-600.
 
Key parameter to change is synaptic coupling J (mV).
Tested with Brian 1.4.1
'''

from brian import *
from pylab import * # imports matplot like commands into the namespace
                    # also can use np. for numpy and mpl. for matplotlib

np.random.seed(100) # set seed for reproducibility of simulations

# ###########################################
# Defining network model parameters
# ###########################################

N = 1000         # Total number of neurons
f = 0.8           # Fraction of exc neurons
NE = int(f*N)     # Number of excitatory cells
NI = N-NE         # Number of inhibitory cells 

C = 100          # Number of incoming connections on each neuron (exc or inh)
fC = f            # fraction fC incoming connections are exc, rest inhibitory
J = 0.2*mV        # exc strength is J.
                  # Critical J is ~ 0.45 mV in paper for N = 1000, C = 1000
                  # Here, N = 1000, C = 100, I get critical J similar!
                  # Using N = 10000, C = 1000, I get critical J of
                  # I'm defining critical J as the J at which mean pop rate
                  # starts to increase beyond that at rest (~38 Hz) after the dip.
                  # But going by firing rate fluctuations of neurons,
                  # should define it as the minimum point of the dip (Fig 1a).
g = 5.0           # -gJ is the inh strength. For exc-inh balance g>~f(1-f)=4

#eta = 1e-2           # Learning rate
#tau_stdp = 20*ms     # STDP time constant

simtime = 1.0*second # Simulation time
dt = defaultclock.dt/second

# ###########################################
# Neuron model
# ###########################################

el = 24*mV#-41*mV         # Resting potential, same as mu0, spontaneously spiking
#el = -65*mV         # Resting potential, same as mu0
vt = 20*mV#-45.*mV        # Spiking threshold
taum = 20*ms        # Membrane time constant
vr = 10*mV#-55*mV         # Reset potential
taur = 0.5*ms       # Refractory period
#taudelay = 0.*ms    # as long as the initial spikes are asynchronous,
                    # no need for synaptic delay.
taudelay = 0.5*ms + dt*second   # Synaptic delay, must be > refractory period
                                # else no 'chaotic' async state
                                # also at least >= taur + dt else missed
eqs_neurons='''
dv/dt=(1/taum)*(-(v-el)) : volt
'''

# ###########################################
# Initialize neuron group
# ###########################################

neurons=NeuronGroup(N,model=eqs_neurons,\
    threshold=vt,reset=vr,refractory=taur)
Pe=neurons.subgroup(NE)
Pi=neurons.subgroup(NI)
#Pe.v = uniform(el,vt+10*mV,NE)
#Pi.v = uniform(el,vt+10*mV,NI)
Pe.v = uniform(0.,vt+10*mV,NE)
Pi.v = uniform(0.,vt+10*mV,NI)

# ###########################################
# Connecting the network 
# ###########################################

sparseness_e = fC*C/float(NE)
sparseness_i = (1-fC)*C/float(NI)
# Follow Dale's law -- exc (inh) neurons only have +ve (-ve) synapses.
con_e = Synapses(Pe,neurons,'',pre='v_post+=J')
con_e.connect_random(sparseness=sparseness_e)
con_e.delay = taudelay
con_i = Synapses(Pi,neurons,'',pre='v_post+=-g*J')
con_i.connect_random(sparseness=sparseness_i)
con_i.delay = taudelay

# Obsolete and inflexible method of creating synapses
#con_e = Connection(Pe,neurons,'v',delay=taudelay)
#con_e.connect_random(Pe,neurons,p=sparseness_e,\
#    fixed=True,weight=1.0,seed=100)
#con_i = Connection(Pi,neurons,'v',delay=taudelay)
#con_i.connect_random(Pi,neurons,p=sparseness_i,\
#    fixed=True,weight=-g,seed=200)
# Can avoid autapses with string based synapse creation:
# something like S[:, :] = '(i != j) * (rand() > 0.15)' 
# will be slow as not a sparse matrix

# ###########################################
# Setting up monitors
# ###########################################

Nmon = 1000
Nmon_exc = int(f*Nmon)
Pe_mon = Pe.subgroup(Nmon_exc)
sm_e = SpikeMonitor(Pe_mon)
Pi_mon = Pi.subgroup(Nmon-Nmon_exc)
sm_i = SpikeMonitor(Pi_mon)

# Population monitor
popm_e = PopulationRateMonitor(Pe,bin=1.*ms)
popm_i = PopulationRateMonitor(Pi,bin=1.*ms)

# ###########################################
# Run
# ###########################################

print "Setup complete, running for",simtime,"at dt =",dt,"s."
run(simtime,report='text')

print "For g,J =",g,J,"mean exc rate =",\
    sm_e.nspikes/float(Nmon_exc)/(simtime/second),'Hz.'
print "For g,J =",g,J,"mean inh rate =",\
    sm_i.nspikes/float(Nmon-Nmon_exc)/(simtime/second),'Hz.'

# ###########################################
# Analysis functions
# ###########################################

def rate_from_spiketrain(spiketimes,fulltime,dt,tau=50e-3):
    """
    Returns a rate series of spiketimes convolved with a Gaussian kernel;
    all times must be in SI units,
    remember to divide fulltime and dt by second
    """
    sigma = tau/2.
    # normalized Gaussian kernel, integral with dt is normed to 1
    # to count as 1 spike smeared over a finite interval
    norm_factor = 1./(sqrt(2.*pi)*sigma)
    gauss_kernel = array([norm_factor*exp(-x**2/(2.*sigma**2))\
        for x in arange(-5.*sigma,5.*sigma+dt,dt)])
    kernel_len = len(gauss_kernel)
    # need to accommodate half kernel_len on either side of fulltime
    rate_full = zeros(int(fulltime/dt)+kernel_len)
    for spiketime in spiketimes:
        idx = int(spiketime/dt)
        rate_full[idx:idx+kernel_len] += gauss_kernel
    # only the middle fulltime part of the rate series
    # This is already in Hz,
    # since should have multiplied by dt for above convolution
    # and divided by dt to get a rate, so effectively not doing either.
    return rate_full[kernel_len/2:kernel_len/2+int(fulltime/dt)]

# ###########################################
# Make plots
# ###########################################

fig = figure()
# raster plots
subplot(231)
raster_plot(sm_e,ms=1.)
title(str(Nmon_exc)+" exc neurons")
xlabel("")
subplot(234)
raster_plot(sm_i,ms=1.)
title(str(Nmon-Nmon_exc)+" inh neurons")
subplot(232)

# firing rates
timeseries = arange(0,simtime/second,dt)*1000
num_to_plot = 10
#rates = []
for nrni in range(num_to_plot):
    rate = rate_from_spiketrain(sm_e[nrni],simtime/second,dt)
    plot(timeseries,rate)
    #print mean(rate),len(sm_e[nrni])
    #rates.append(rate)
title(str(num_to_plot)+" exc rates")
ylabel("Hz")
ylim(0,300)
subplot(235)
for nrni in range(num_to_plot):
    rate = rate_from_spiketrain(sm_i[nrni],simtime/second,dt)
    plot(timeseries,rate)
    #print mean(rate),len(sm_i[nrni])
    #rates.append(rate)
title(str(num_to_plot)+" inh rates")
ylim(0,300)
#print "Mean rate = ",mean(rates)
xlabel("Time (ms)")
ylabel("Hz")

# Population firing rates
subplot(233)
timeseries = arange(0,simtime/second,1e-3)*1000
plot(timeseries,popm_e.smooth_rate(width=50.*ms,filter="gaussian"))
title("Exc population rate")
ylabel("Hz")
subplot(236)
timeseries = arange(0,simtime/second,1e-3)
plot(timeseries,popm_i.smooth_rate(width=50.*ms,filter="gaussian"))
title("Inh population rate")
xlabel("Time (ms)")
ylabel("Hz")

fig.tight_layout()

show()
