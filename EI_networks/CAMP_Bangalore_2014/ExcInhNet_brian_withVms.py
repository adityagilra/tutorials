#!/usr/bin/env python
'''
The LIF network is based on:
Ostojic, S. (2014).
 Two types of asynchronous activity in networks of
 excitatory and inhibitory spiking neurons.
 Nat Neurosci 17, 594-600.
 
Key parameter to change is synaptic coupling J (mV).
Tested with Brian 1.4.1

Written by Aditya Gilra, CAMP 2014, Bangalore, 20 June, 2014.
'''

#import modules and functions to be used
from brian import * # importing brian also does:
                    # 'from pylab import *' which imports:
                    # matplot like commands into the namespace, further
                    # also can use np. for numpy and mpl. for matplotlib

np.random.seed(100) # set seed for reproducibility of simulations

# ###########################################
# Simulation parameters
# ###########################################

simtime = 1.0*second            # Simulation time
defaultclock.dt = 0.1*ms        # Brian's default sim time step is 1 ms
dt = defaultclock.dt/second     # convert to value in seconds

# ###########################################
# Neuron model
# ###########################################

# equation: dv/dt=(1/taum)*(-(v-el))
# with spike when v>vt, reset to vr

el = -65.*mV          # Resting potential
vt = -45.*mV          # Spiking threshold
taum = 20.*ms         # Membrane time constant
vr = -55.*mV          # Reset potential
inp = 24.*mV/taum     # input I/C to each neuron
                      # same as setting el=-41 mV and inp=0
taur = 0.5*ms         # Refractory period: 0 ms gives similar results
taudelay = 0.5*ms + dt*second     # synaptic delay
                                  # 0 ms gives similar results
                                  # contrary to Ostojic?!

eqs_neurons='''
dv/dt=(1/taum)*(-(v-el))+inp : volt
'''

# ###########################################
# Network parameters: numbers
# ###########################################

N = 1000          # Total number of neurons
fexc = 0.8        # Fraction of exc neurons
NE = int(fexc*N)  # Number of excitatory cells
NI = N-NE         # Number of inhibitory cells 

# ###########################################
# Network parameters: synapses
# ###########################################

C = 100           # Number of incoming connections on each neuron (exc or inh)
fC = fexc         # fraction fC incoming connections are exc, rest inhibitory
J = 2.0*mV        # exc strength is J (in mV as we add to voltage)
                  # Critical J is ~ 0.45 mV in paper for N = 1000, C = 1000
g = 5.0           # -gJ is the inh strength. For exc-inh balance g>~f(1-f)=4

# ###########################################
# Initialize neuron (sub)groups
# ###########################################

neurons=NeuronGroup(N,model=eqs_neurons,\
    threshold=vt,reset=vr,refractory=taur)
Pe=neurons.subgroup(NE)
Pi=neurons.subgroup(NI)
Pe.v = uniform(el,vt+10*mV,NE)
Pi.v = uniform(el,vt+10*mV,NI)

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

# ###########################################
# Setting up monitors
# ###########################################

Nmon = N
Nmon_exc = int(fexc*Nmon)
Pe_mon = Pe.subgroup(Nmon_exc)
sm_e = SpikeMonitor(Pe_mon)
Pi_mon = Pi.subgroup(Nmon-Nmon_exc)
sm_i = SpikeMonitor(Pi_mon)

# Population monitor
popm_e = PopulationRateMonitor(Pe,bin=1.*ms)
popm_i = PopulationRateMonitor(Pi,bin=1.*ms)

# State monitors
print len(Pe)
Pe_v = Pe_mon.subgroup(5)
sm_ev = StateMonitor(Pe_v,'v',record=True)
Pi_v = Pi_mon.subgroup(5)
sm_iv = StateMonitor(Pi_v,'v',record=True)

# ###########################################
# Simulate
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

fig = figure()
subplot(121)
for i in range(5):
    plot(sm_ev.times,sm_ev[i])
title("Exc Vm")
xlabel("Time (ms)")
ylabel("Vm (mV)")
subplot(122)
for i in range(5):
    plot(sm_iv.times,sm_iv[i])
title("Inh Vm")
xlabel("Time (ms)")

show()
