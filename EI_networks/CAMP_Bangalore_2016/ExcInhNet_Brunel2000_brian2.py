from brian2 import *
from data_utils import *

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
J = 0.2*mV # synaptic weight
p = 0.1 # connection probability
delay = 1.5*ms # synaptic delay

# delta-function synapses
con = Synapses(P,P,'w:volt (constant)',on_pre='v_post+=w',method='euler')
#con.connect(condition='i!=j',p=p)
print 'computing connection matrix'
CE = int(p*NE)
CI = int(p*(N-NE))
C = CE+CI
conn_i = np.zeros(C*N,dtype=int)
preneuronsE = arange(0,NE,dtype=int)
preneuronsI = arange(NE,N,dtype=int)
for j in range(N): # loop over post-synaptic neurons
    # draw CE number of neuron indices out of NE neurons, no autapses
    if j<NE: preneurons = np.delete(preneuronsE,j)
    else: preneurons = preneuronsE
    conn_i[j*C:j*C+CE] = np.random.permutation(preneurons)[:CE]
    # draw CI number of neuron indices out of inhibitory neurons, no autapses
    if j>NE: preneurons = np.delete(preneuronsI,j-NE)
    else: preneurons = preneuronsI
    conn_i[j*C+CE:(j+1)*C] = np.random.permutation(preneurons)[:CI]
conn_j = np.repeat(range(N),C)
print 'connecting network'
con.connect(i=conn_i,j=conn_j)
con.delay = delay
con.w['i<NE'] = J
con.w['i>=NE'] = -g*J

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

print 'compiling/running'
run(0.25*second, report='text')
device.build(directory='output', compile=True, run=True, debug=False);

print "mean activity (Hz) =",mean(sr.rate/Hz)

figure()
plot(sm.t/ms,sm.i,'.')
#ylim([1350,1400])
figure()
plot(sr.t/ms,sr.rate/Hz,',-')
#figure()
#hist(CV_spiketrains(array(sm.t),array(sm.i),0.,range(N)),bins=100)

show()
