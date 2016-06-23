from pylab import *

# 1) initialize parameters
tmax = 1000.
dt   = 0.5

# 1.1) Neuron / Network pars
a = 0.02
b = 0.2
c = -65
d = 8.

# 1.2) Input pars
Iapp=7
tr=array([200.,700])/dt # stm time

# 2) reserve memory
T = ceil(tmax/dt)
v = zeros(T)
u = zeros(T)
v[0] = -70 # resting potential
u[0] = -14 # steady state

# 3) for-loop over time
for t in arange(T-1):
    # 3.1) get input 
    if t>tr[0] and t<tr[1]:
        I = Iapp
    else:
        I = 0
         
         
    if v[t]<35:
        # 3.2) update ODE
        dv = (0.04*v[t]+5)*v[t]+140-u[t]
        v[t+1] = v[t] + (dv+I)*dt
        du = a*(b*v[t]-u[t])
        u[t+1] = u[t] + dt*du
    else:
        # 3.3) spike !
        v[t] = 35
        v[t+1] = c
        u[t+1] = u[t]+d
    
# 4) plot voltage trace
rcParams['figure.figsize'] = (4,3)
rcParams['legend.fancybox'] = True
rcParams['legend.fontsize'] = 'small'
rcParams['axes.titlesize'] = 'medium'
rcParams['text.usetex'] = True
rcParams['font.size'] = 14   
rcParams['ps.usedistiller'] = 'xpdf'
rc('figure.subplot',
   **{'bottom':0.15,'left':0.2,
      'right':0.9,'top':0.85})
figure()
tvec = arange(0.,tmax,dt)   
plot(tvec,v,'b',label='Voltage trace')
xlabel('Time [ms]')
ylabel('Membrane voltage [mV]')
title("""A single qIF neuron
with current step input""")
savefig('singleneuron.eps',format='eps',
        transparent=True)
show()
    
   