

wmax = 5.               # hard bound on synaptic weight
Apre_tau = 20*ms        # STDP Apre time constant
Apost_tau = 20*ms       # STDP Apost time constant
Apostslow_tau = 100*ms
Apre0 = 0.5             # incr in Apre, on pre-spikes;
                        # at spike coincidence, delta w = -Apre0*eta
Apost0 = 1.0            # incr in Apost on post spike
Apostslow0 = 0.25       # incr in Apost on post spike
beta = 5e-10             # heterosynaptic plasticity strength parameter
eta = 1e-3              # learning rate
stdp_eqns = ''' wsyn : 1
                dApre/dt=-Apre/Apre_tau : 1 (event-driven)
                dApost/dt=-Apost/Apost_tau : 1 (event-driven)
                dApostslow/dt=-Apostslow/Apostslow_tau : 1 (event-driven)
            '''
pre_eqns = 'Apre+=Apre0; wsyn+=-Apost*eta; wsyn=clip(wsyn,0,wmax); v+=wsyn*J'
post_eqns = 'Apost+=Apost0; wsyn += eta*(Apre + Apre*Apostslow - beta*Apost**3);'\
                'Apostslow+=Apostslow0; wsyn=clip(wsyn,0,wmax)'

def dwbydt(r):
    return eta*(Apre0*Apre_tau/second-Apost0*Apost_tau/second)*r**2 + \
               eta*Apre0*Apre_tau/second*Apostslow0*Apostslow_tau/second*r**3 - \
               beta*Apost0**3*(Apost_tau/second)**3*r**4

figure()
rrange = arange(0,100,0.1)
plot(rrange,dwbydt(rrange))

show()
sys.exit(0)
