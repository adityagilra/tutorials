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
PE = P[:NE]
PI = P[NE:]
cons = []
def connect_prob(P1,P2,wt,recurrent):
    con = Synapses(P1,P2,'w:volt (constant)',on_pre='v_post+=w',method='euler')
    if recurrent: con.connect(condition='i!=j',p=p)
    else: con.connect(p=p)
    con.delay = delay
    con.w = wt
    cons.append(con)
connect_prob(PE,PE,J,True)
connect_prob(PE,PI,J,False)
connect_prob(PI,PE,-g*J,False)
connect_prob(PI,PI,-g*J,True)

# input parameters
inpfactor = 2
nu_theta = vth/(p*NE*J*tau)
Pinp = PoissonGroup(N=N,rates=inpfactor*nu_theta)
con_ext = Synapses(Pinp, P, on_pre='v += J')
con_ext.connect(True, p=p*NE/float(N))
con_ext.delay = delay

sm = SpikeMonitor(P)
sr = PopulationRateMonitor(P)
sm_vm = StateMonitor(P,'v',record=range(5))

net = Network(collect())  # automatically include brian objects in scope
net.add(cons)
net.run(0.25*second, report='text')
device.build(directory='output', compile=True, run=True, debug=False);

print "mean activity (Hz) =",mean(sr.rate/Hz)

figure()
plot(sm.t/ms,sm.i,'.')
ylim([1350,1400])
figure()
plot(sr.t/ms,sr.rate/Hz,',-')

show()
