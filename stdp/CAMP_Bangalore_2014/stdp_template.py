#!/usr/bin/env python
'''
Implement and plot STDP rule.
'''

from brian import *
# For reference:
# http://brian.readthedocs.org/en/1.4.1/

# ###########################################
# Defining network model parameters
# ###########################################

NE = 2                  # Number of excitatory cells
er_e = 0*mV             # Excitatory reversal potential
g_ampa = 1*nS           # AMPA conductance
tau_ampa = 1*ms         # AMPA decay constant

# ###########################################
# Neuron models
# ###########################################

## Leaky integrate and fire neuron
Vrest = -65e-3*volt # V
Vt_base = -45e-3*volt
Vreset = -55e-3*volt # V # in current steps, Vreset is same as pedestal
R = 1e8*ohm
tau = 10e-3*second
IaF_eqns = """
    dV/dt = ( R*(bgnd_I + g_ampa*wsyntot*(er_e-V)) - (V-Vrest)) / tau : volt
    dwsyntot/dt = -wsyntot/tau_ampa : 1
    bgnd_I : amp
"""

# ###########################################
# Initialize neuron group
# ###########################################

# neuron 1 (later we make this presynaptic)
P1_stdptest = NeuronGroup(1,model=IaF_eqns,threshold=Vt_base,\
                        reset=Vreset,refractory=2*ms)
P1_stdptest.V = Vrest
P1_stdptest.wsyntot = 0.
P1_stdptest.bgnd_I = 0.*amp
# neuron 2 (later we make this postsynaptic)
P2_stdptest = NeuronGroup(1,model=IaF_eqns,threshold=Vt_base,\
                        reset=Vreset,refractory=2*ms)
P2_stdptest.V = Vrest
P2_stdptest.wsyntot = 0.
P2_stdptest.bgnd_I = 0.*amp

# ###########################################
# Synaptic model: dynamical/online STDP at each pre and post spike
# ###########################################

# Values approx from figure in Scholarpedia article (following Bi and Poo 1998):
# Jesper Sjoestroem and Wulfram Gerstner (2010) Spike-timing dependent plasticity.
# Scholarpedia, 5(2):1362., revision #137369
Apre_tau = 10*msecond     # Apre time constant
Apost_tau = 20*msecond    # Apost time constant
Apre0 = 1.0               # at pre, Apre += Apre0
Apost0 = 0.25             # at post, Apost += Apost0
stdp_eqns = ''' wsyn : 1
            dApre/dt = -Apre/Apre_tau : 1
            dApost/dt = -Apost/Apost_tau : 1           
            '''
# implement stdp at each pre and post spike
# and postsynaptic activation on pre spike
# not bounding wsyn below and above
# wsyntot defaults to wsyntot_post
pre_eqns = 'Apre+=Apre0; wsyn-=Apost; wsyntot_post+=wsyn'
post_eqns = 'Apost+=Apost0; wsyn+=Apre'

con_ee = Synapses(P1_stdptest, P2_stdptest, stdp_eqns,
                    pre=pre_eqns, post=post_eqns)
con_ee[:,:]=True  # connect every pair
## ALWAYS set the synaptic variables AFTER making the connections!
con_ee.wsyn = 1.0 # sets the full array, here 1x1

# ###########################################
# Setting up monitors
# ###########################################

spm1 = SpikeMonitor(P1_stdptest)
spm2 = SpikeMonitor(P2_stdptest)
stm1 = StateMonitor(P1_stdptest, 'V', record=True)
stm2 = StateMonitor(P2_stdptest, 'V', record=True)

# ###########################################
# Simulate the STDP curve with spaced pre-post spike pairs
# ###########################################

# Settletime is not necessary here as we manually reset relevant params
# But in other contexts you may want state variables
#  to settle to equilibrium values
settletime = 100*ms
def reset_settle():
    """ Call this between every pre-post pair
    to reset the neurons and make them settle to rest.
    """
    con_ee.wsyn = 1.0 # sets the full array
    P1_stdptest.wsyntot = 0.0
    P2_stdptest.wsyntot = 0.0
    con_ee.Apre = 0.0
    con_ee.Apost = 0.0
    run(settletime)

def make_neuron_spike(nrngrp,I=1e-7*amp,duration=2.*ms):
    """ Inject a brief current pulse to 
    make a neuron spike
    """
    nrngrp.bgnd_I = I
    run(duration)
    nrngrp.bgnd_I = 0.*amp

dwlist_neg = []
ddt = 4*ms
t_extent = 20.*ms
# dt = tpost - tpre
# negative dt corresponds to post before pre
for deltat in arange(t_extent,0*second,-ddt):
    reset_settle()
    # post neuron spike
    make_neuron_spike(P2_stdptest)
    run(deltat*second)
    ## pre neuron spike after deltat
    make_neuron_spike(P1_stdptest)
    run(1e-3*second)
    dw = con_ee.wsyn[0,0][0] - 1.
    print 'post before pre, dt = %1.3f s, dw = %1.3f'%(-deltat,dw)
    dwlist_neg.append(dw)
# positive dt corresponds to pre before post
dwlist_pos = []
for deltat in arange(0*second+ddt,t_extent+ddt,ddt):
    reset_settle()
    # post neuron spike
    make_neuron_spike(P1_stdptest)
    run(deltat*second)
    ## pre neuron spike after deltat
    make_neuron_spike(P2_stdptest)
    run(1e-3*second)
    dw = con_ee.wsyn[0,0][0] - 1.
    print 'pre before post, dt = %1.3f s, dw = %1.3f'%(deltat,dw)
    dwlist_pos.append(dw)

# ###########################################
# Plot the simulated Vm-s and STDP curve
# ###########################################

# Voltage plots
# insert spikes from Spike Monitor so that Vm doesn't look weird
stm1.insert_spikes(spm1)
stm2.insert_spikes(spm2)
figure(facecolor='w')
plot(stm1[0],color='r') # pre neuron's vm
plot(stm2[0],color='b') # post neuron's vm
xlabel('time (ms)')
ylabel('Vm (mV)')
title("pre (r) and post (b) neurons' Vm")

# STDP curve
fig = figure(facecolor='w')
ax = fig.add_subplot(111)
ax.plot(arange(-t_extent/ms,0,ddt/ms),array(dwlist_neg),'.-r')
ax.plot(arange(0+ddt/ms,(t_extent+ddt)/ms,ddt/ms),array(dwlist_pos),'.-b')
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ax.set_xticks([xmin,0,xmax])
ax.set_yticks([ymin,0,ymax])
ax.plot((0,0),(ymin,ymax),linestyle='dashed',color='k')
ax.plot((xmin,xmax),(0,0),linestyle='dashed',color='k')
ax.set_xlabel('$t_{post}-t_{pre}$ (ms)')
ax.set_ylabel('$\Delta w$ (arb)')
fig.tight_layout()
#fig.subplots_adjust(hspace=0.3,wspace=0.5) # has to be after tight_layout()

show()
