#!/usr/bin/env python
'''
The LIF network is based on:
Ostojic, S. (2014).
 Two types of asynchronous activity in networks of
 excitatory and inhibitory spiking neurons.
 Nat Neurosci 17, 594-600.
 
Key parameter to change is synaptic coupling J (mV).

Written by Aditya Gilra, CAMP 2014, Bangalore, 20 June, 2014.
Partly based on a tutorial by Malte Rasch, CCN, Beijing, July 2013.
'''

#import modules and functions to be used
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(100) # set seed for reproducibility of simulations
random.seed(100) # set seed for reproducibility of simulations

# ###########################################
# Neuron model
# ###########################################

# equation: dv/dt=(1/taum)*(-(v-el))
# with spike when v>vt, reset to vr

el = -65.  #mV        # Resting potential
vt = -45.  #mV        # Spiking threshold
taum = 20. #ms        # Membrane time constant
vr = -55.  #mV        # Reset potential
inp = 24./taum #mV/ms # input I/C to each neuron
                      # same as setting el=-41 mV and inp=0

# ###########################################
# Network parameters: numbers
# ###########################################

N = 1000          # Total number of neurons
fexc = 0.8        # Fraction of exc neurons
NE = int(fexc*N)  # Number of excitatory cells
NI = N-NE         # Number of inhibitory cells 

# ###########################################
# Simulation parameters
# ###########################################

simtime = 1000.0  #ms # Simulation time
dt = 1.           #ms # time step

# ###########################################
# Network parameters: synapses (not for ExcInhNetBase)
# ###########################################

C = 100           # Number of incoming connections on each neuron (exc or inh)
fC = fexc         # fraction fC incoming connections are exc, rest inhibitory
J = 0.2 #mV       # exc strength is J (in mV as we add to voltage)
                  # Critical J is ~ 0.45 mV in paper for N = 10000, C = 1000
g = 5.0           # -gJ is the inh strength. For exc-inh balance g>~f(1-f)=4

# ###########################################
# Exc-Inh network base class without connections
# ###########################################

class ExcInhNetBase:
    """Simulates and plots LIF neurons (exc and inh separate).
    code by Aditya Gilra, CAMP, 2014-6-17;
    structure from Malte J Rasch, CCN, 2013-6-19."""

    def __init__(self,N=N,fexc=fexc,el=el,vt=vt,taum=taum,vr=vr):
        """ Constructor of the class """
        
        self.N = N                 # Total number of neurons
        self.fexc = fexc           # Fraction of exc neurons
        self.NmaxExc = int(fexc*N) # max idx of exc neurons, rest inh

        self.el = el        # Resting potential
        self.vt = vt        # Spiking threshold
        self.taum = taum    # Membrane time constant
        self.vr = vr        # Reset potential
        
        self.simif = False  # whether the simulation is complete
        
        self._setup_network()

    def __str__(self):
         return "LIF network of %d neurons "\
             "having %d exc." % (self.N,self.NmaxExc)
    
    def _setup_network(self):
        """Sets up the network (_init_network is enough)"""
        pass

    def _init_network(self,v0=el):
        """Initialises the network variables before simulation"""        
        
        self.v = np.zeros((self.T,self.N)) # row is time    
        self.v[0] = v0 # set the whole row vector

    def _update_network(self,t,fired,inp=inp):
        """ Updates the network dynamics from t->t+1 """
        
        # update all
        # inp is I/C
        self.v[t+1] = self.v[t] + \
            ((self.el-self.v[t])/self.taum+inp)*self.dt
        
        # reset those that fired in the previous step
        self.v[t+1,fired] = self.vr
        
    def simulate(self,simtime=simtime,dt=dt,plotif=False,**kwargs):
        
        self.dt = dt
        self.simtime = simtime
        self.T = np.ceil(simtime/dt)
        
        self._init_network(**kwargs)
        
        # integration loop
        for t in np.arange(self.T-1):            
            fired = self.v[t]>=self.vt           
            self._update_network(t,fired)
            
        self.simif = True
        
        if plotif:
            self.plot()
            
    def get_spks(self):
        """ Return spikes in a dictionary"""
        
        if not self.simif:
            raise "Not simulated yet"
        else:
            # time and neuronidx of spikes as two tuples
            tspk,nspk = np.nonzero(self.v>=self.vt)
            # get the tuple indices of the exc neurons
            exc_idx = np.nonzero(nspk<self.NmaxExc)
            # tuple indices of the inh  neurons
            inh_idx = np.nonzero(nspk>=self.NmaxExc)
            tespk = tspk[exc_idx]*self.dt
            tispk = tspk[inh_idx]*self.dt
            espk = nspk[exc_idx] # get the exc neuron idx
            ispk = nspk[inh_idx]
            
            # nicer output format
            dic = {}
            dic['espkt'] = (tespk,espk)
            dic['ispkt'] = (tispk,ispk)      

            return dic 

    def plot(self):
        """ plots the simulated net"""
         
        d = self.get_spks() 
        self.trange = np.arange(0,self.simtime)   
        
        plt.figure()
        plt.plot(d['espkt'][0],d['espkt'][1],'b.',marker=',',\
            label='Exc. spike trains')
        plt.plot(d['ispkt'][0],d['ispkt'][1],'r.',marker=',',\
            label='Inh. spike trains')           
        plt.xlabel('Time [ms]')
        plt.ylabel('Neuron number [#]')
        plt.xlim([0,self.simtime])
        plt.title("%s" % self, fontsize=14,fontweight='bold')
        plt.legend(loc='upper left')

# ###########################################
# Exc-Inh network class with connections (inherits from ExcInhNetBase)
# ###########################################

class ExcInhNet(ExcInhNetBase):
    """ Recurrent network simulation """
    
    def __init__(self,J=J,incC=C,fC=fC,scaleI=g,**kwargs):
        """Overloads base (parent) class"""

        self.J = J              # exc connection weight
        self.incC = incC         # number of incoming connections per neuron
        self.fC = fC            # fraction of exc incoming connections
        self.excC = int(fC*incC)# number of exc incoming connections
        self.scaleI = scaleI    # inh weight is scaleI*J
                 
        # call the parent class constructor
        ExcInhNetBase.__init__(self,**kwargs) 
    
    def __str__(self):
         return "LIF network of %d neurons "\
             "of which %d are exc." % (self.N,self.NmaxExc) 
 
    def _init_network(self,**args):
        ExcInhNetBase._init_network(self,**args)
        
    def _setup_network(self):

        ExcInhNetBase._setup_network(self)  

        # recurrent connections    
        # set the weight matrix
        W = np.zeros((self.N,self.N))
        Nrange_exc = range(0,self.NmaxExc)
        Nrange_inh = range(self.NmaxExc,N)
        for nrnidx in range(self.N):
            # draw values without replacement
            # np.random.choice() is only in numpy 1.7 (we use 1.6.1)
            pre_exc_nrns = random.sample(Nrange_exc,self.excC)
            W[nrnidx,pre_exc_nrns] = self.J
            pre_inh_nrns = random.sample(\
                Nrange_inh,self.incC-self.excC)
            W[nrnidx,pre_inh_nrns] = -self.J*self.scaleI

        self.W = W
        
    def _update_network(self,t,fired,inp=inp):
        """ Updates the network dynamics from t->t+1 """
         
        # update the recurrent inputs
        #fired_array = np.zeros(self.N)
        #fired_array[fired] = 1. # only those are 1 which fired
        syn_inp = np.dot(self.W,fired)/self.dt
        inp_tot = inp + syn_inp # add to input vector
        
        ExcInhNetBase._update_network(self,t,fired,inp_tot)
   
if __name__=='__main__':
        
    net = ExcInhNet(N=N)
    net.simulate(simtime,v0=np.random.uniform(el,vt,size=N))    
    net.plot()
    print net

    plt.figure()
    plt.plot(net.trange,net.v[:,0])
    plt.plot(net.trange,net.v[:,1])
    plt.plot(net.trange,net.v[:,2])
    plt.xlabel('time (ms)')
    plt.ylabel('Vm (mV)')

    plt.show()
    
