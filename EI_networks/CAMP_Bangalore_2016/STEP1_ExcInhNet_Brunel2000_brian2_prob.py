from brian2 import *

set_device('cpp_standalone', build_on_run=False)

# neuronal parameters
N = 12500 # total number of neurons
NE = 10000 # number of excitatory neurons
vth = 20*mV # threshold potential
vr = 10*mV # reset potential
tau = 20*ms # membrane time constant

eqs_neurons='''
inp : volt
dv/dt = (-v + inp)/tau : volt
'''
P=NeuronGroup(N=N,model=eqs_neurons,\
                threshold='v>=vth',reset='v=vr',\
                refractory=2*ms,method='euler')
P.v = uniform(size=12500)*vth

# synaptic parameters
g = 5 # ratio of inh to exc
J = 0.1*mV # synaptic weight
p = 0.1 # connection probability
delay = 1.5*ms # synaptic delay

# delta-function synapses
con = Synapses(P,P,'w:volt (constant)',on_pre='v_post+=w',method='euler')
con.connect(condition='i!=j',p=p)
con.delay = delay
con.w['i<NE'] = J
con.w['i>=NE'] = -g*J

# input parameters
inpfactor = 2
nu_thr = vth/(p*NE*J*tau)
Pinp = PoissonGroup(N=N,rates=inpfactor*nu_thr)
###
# connect the Pinp neurons to P neurons here with prob p*NE/N
###

sm = SpikeMonitor(P)
sr = PopulationRateMonitor(P)
sm_vm = StateMonitor(P,'v',record=range(5))

run(0.25*second, report='text')
device.build(directory='output', compile=True, run=True, debug=False);

print "mean activity (Hz) =",mean(sr.rate/Hz)

figure()
plot(sm.t/ms,sm.i,'.')
#ylim([1350,1400])
figure()
plot(sr.t/ms,sr.rate/Hz,',-')

show()
