# -*- coding: utf-8 -*-
"""
Strong Sparse Weak Dense (SSWD) network from Teramae, Tsubo, and Fukai, 2012.
author: Aditya Gilra, 20 June, 2016, EPFL.
"""

import sys
simulate = True

from brian2 import *
stand_alone = True
if simulate:
    if stand_alone: set_device('cpp_standalone', build_on_run=False)
    else:
        prefs.codegen.target = 'numpy'
        #prefs.codegen.target = 'weave'
        #prefs.codegen.target = 'cython'

seed(100)
np.random.seed(100)

# neuronal constants
tauE = 20.*ms                   # exc membrane tau
tauI = 10.*ms                   # inh membrane tau
tauS = 2.0*ms                   # synaptic tau
taurefr = 1.*ms                 # refractory period
syndelayE = 2.*ms               # exc synaptic delay
syndelayI = 1.*ms               # inh synaptic delay
Vth = -50.*mV                   # C threshold
VL = -70.*mV                    # V leak
VEr = 0.*mV                     # exc syn reversal potential
VIr = -80.*mV                   # inh syn reversal potential
Va = 0.1*mV                     # threshold for syn transmission failure

# network constants
NE = 10000                      # number of exc neurons
NI = 2000                       # number of inh neurons
Ntot = NE+NI                    # total number of neurons
cE = 0.1                        # exc connection probability
cI = 0.5                        # inh connection probability
GIE = 0.017*ms**-1              # E to I
GEI = 0.0018*ms**-1             # I to E
GII = 0.0025*ms**-1             # I to I
lognEEsigma = 1.0               # E to E epsp (mV), std dev of lognormal
lognEEmean = lognEEsigma**2+log(0.2)
                                # E to E epsp (mV), mean of lognormal
Vmax = 20.*mV                   # upper limit of EPSP
afail = 0.1*mV                  # spike transmission failure param
gfail = afail/tauS/(VEr-VL)     # afail but in terms of conductance g

# stimulation constants
duration_bg = 50*ms             # duration of initial background 'kick'
inp_bg = 25*Hz                  # initial kick
epsp_bg = 10*mV                 # EPSP size
g_bg = epsp_bg/(VEr-VL)/tauS    # epsp ~= g*tauS*(VEr-VL)
                                # approx conductance for above EPSP size
                                # in simulations, in exc neurons,
                                # g(10mV) gave 7.3 mV; g(1mV) gave 0.77mV;
                                # g(0.1mV) gave 0.077mV
approx_g_factor = 0.75          # a factor to partially rectify above approx
g_bg = g_bg/approx_g_factor

duration_settle = 50*ms         # network settles to bgnd activity after kick

# simulation constants
tstep = 0.1*ms                  # time step of simulation
runtime = duration_bg+duration_settle

model_eqns = """
    taum : second
    u0 : volt
    du/dt = -(u-VL)/taum - gE*(u-VEr) - gI*(u-VIr) + u0/taum : volt
    dgE/dt = -gE/tauS : second**-1
    dgI/dt = -gI/tauS : second**-1
"""
threshold_eqns = "u>=Vth"

# One E+I neuron group is faster to simulate than two separate E & I
Nrns = NeuronGroup(Ntot, model_eqns, method='euler',\
                    threshold=threshold_eqns,\
                    reset="u=VL",refractory=taurefr)
Nrns.u = VL                     # initially, all neurons are as if just reset
NrnsE = Nrns[:NE]
NrnsI = Nrns[NE:]
NrnsE.taum = tauE
NrnsI.taum = tauI

###
# brian2 code to make, connect, weight synapses
###
SynsEE = Synapses(NrnsE, NrnsE, 'g : second**-1', \
                        on_pre='gE += g*(rand()>gfail/(gfail+g))') # E to E
                                                # some spikes fail to transmit
SynsEE.delay = '(rand()-0.5)*2*ms + syndelayE'  # +/-1ms variability in delay
SynsIE = Synapses(NrnsE, NrnsI, 'g : second**-1', on_pre='gE += g') # E to I
SynsIE.delay = '(rand()-0.5)*2*ms + syndelayE'
SynsEI = Synapses(NrnsI, NrnsE, 'g : second**-1', on_pre='gI += g') # I to E
SynsEI.delay = '(rand()-0.5)*2*ms + syndelayI'
SynsII = Synapses(NrnsI, NrnsI, 'g : second**-1', on_pre='gI += g') # I to E
SynsII.delay = '(rand()-0.5)*2*ms + syndelayI'

# excitatory to excitatory
connEE = where(uniform(size=(NE,NE))>=cE)
                                        # keep only cE prob of connections
SynsEE.connect(i=connEE[0],j=connEE[1])

# choose lognormally distributed values of EPSPs ensuring that EPSP <= 20mV
# create full array directly and then resample the too high ones
#  faster than in a python loop
weightsEE = lognormal(mean=lognEEmean, sigma=lognEEsigma, size=(len(connEE[0])))
bad_idxs = where(weightsEE>Vmax/mV)[0]
for k in bad_idxs:
    while True:
        val = lognormal(mean=lognEEmean, sigma=lognEEsigma)
        if val <= Vmax/mV: break
    weightsEE[k] = val

#SynsEE[:,:].w = weightsEE   # Note that w is 1D, but you can index SynsEE using [i,j]
                            #  but this doesn't work in stand-alone mode,
                            #  brian2 cannot access state variables before running
SynsEE.g = weightsEE/tauS/(VEr-VL)*mV/approx_g_factor
#SynsEE.g = lognEEmean/tauS/(VEr-VL)*mV
                            # set to conductance g instead of EPSP
                            #  epsp ~= g*tauS*(VEr-VL)

# excitatory to inhibitory
connIE = where(uniform(size=(NE,NI))<cE)
SynsIE.connect(i=connIE[0],j=connIE[1])
SynsIE.g = GIE

# inhibitory to excitatory
connEI = where(uniform(size=(NI,NE))<cI)
SynsEI.connect(i=connEI[0],j=connEI[1])
SynsEI.g = GEI

# inhibitory to excitatory
connII = where(uniform(size=(NI,NI))<cI)
SynsII.connect(i=connII[0],j=connII[1])
SynsII.g = GII

# bug in PopulationRateMonitor(NrnsI) with NrnsI = NeuronGroup[N:]
# gives higher firing rate than actual?!
ratesI = PopulationRateMonitor(NrnsI)
MuInh = StateMonitor(Nrns, 'u', record=range(NE,NE+5))

rates = PopulationRateMonitor(NrnsE)
spikes = SpikeMonitor(Nrns)
Mu = StateMonitor(Nrns, 'u', record=range(5))

# Poisson input
NrnsIn = NeuronGroup(Ntot, 'rate : Hz', threshold='rand()<rate*dt')
SynsEIn = Synapses(NrnsIn, Nrns, '', on_pre='gE += g_bg') # In to E
SynsEIn.connect(condition='i==j')

# ###########################################
# Simulation
# ###########################################

defaultclock.dt = tstep
print "Kickstart the network"
NrnsIn.rate = inp_bg
if simulate: run(duration_bg,report='text')

print "Let the network settle into background firing"
NrnsIn.rate = 0*Hz
if simulate: run(duration_settle,report='text')

if simulate and stand_alone:
    device.build(directory='output', compile=True,\
                                    run=True, debug=False)

if not simulate: sys.exit()

# ###########################################
# Analysis & plotting
# ###########################################

# spike raster
figure()
plot(spikes.t/second, spikes.i, '.')

print "plotted spike raster"

# overall rate
ratefig = figure()
rateax = ratefig.add_subplot(111)
binunits = 100
bindt = tstep*binunits
bins = range(int(runtime/bindt))
Nbins = len(bins)
rateax.plot([rates.t[i*binunits]/ms+bindt/2.0/ms for i in bins],\
    [sum(rates.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits) for i in bins],
    '.-b',label="exc nrns' rate")
rateax.plot([rates.t[i*binunits]/ms+bindt/2.0/ms for i in bins],\
    [sum(rates.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits) for i in bins],
    '.-r',label="inh nrns' rate")
rateax.set_ylabel("rate (Hz)")
rateax.set_xlabel("time (ms)")
legend()

print "plotted rates"

# voltage
vfig = figure()
v_ax = vfig.add_subplot(111)
v_ax.set_ylabel('voltage (mV)')
v_ax.set_title('Exc on (b), Exc on incompl (c) Inh (r)')
# Exc on and on_incompl
for idx in range(5):
    v_ax.plot(Mu.t/ms, Mu.u[idx]/mV,'-,b')
    # Inh
    v_ax.plot(Mu.t/ms, MuInh.u[idx]/mV,'-,r')

show()
