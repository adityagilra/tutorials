from pylab import *

# ###########################################
# Analysis functions
# ###########################################

def rate_from_spiketrain(spiket,spikei,fulltime,sigma,dt,nrnidx=None):
    """
    Returns a rate series of spiketimes convolved with a Gaussian kernel;
    all times must be in SI units,
    remember to divide fulltime and dt by second
    """
    # normalized Gaussian kernel, integral with dt is normed to 1
    # to count as 1 spike smeared over a finite interval
    norm_factor = 1./(sqrt(2.*pi)*sigma)
    gauss_kernel = array([norm_factor*exp(-x**2/(2.*sigma**2))\
        for x in arange(-5.*sigma,5.*sigma+dt,dt)])

    if nrnidx is None:
        spiketimes = spiket
    else:
        # take spiketimes of only neuron index nrnidx
        spiketimes = spiket[where(spikei==nrnidx)]
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

def CV_spiketrains(spiket,spikei,tinit,nrnidxs):
    """ calculate CV of excitatory neurons
     a la Lim and Goldman 2013 """
    CV = []
    for j in nrnidxs:
        indices = where(spikei == j)
        spiketimes = spiket[indices]
        spiketimes = spiketimes[where(spiketimes>tinit)]
        ISI = diff(spiketimes)

        # at least 5 spikes in this neuron to get CV
        if len(spiketimes)>5:
            CV.append( std(ISI)/mean(ISI) )
    return array(CV)

