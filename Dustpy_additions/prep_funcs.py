from astropy.constants import M_sun
import numpy as np
from scipy.special import iv, erf
import dustpy.constants as c


####------------------- Defining gaussian source term decreasing in time-------------------###

def source(r, r_0, δ_0):
    r_0 = r_0 * c.au; δ_0 = δ_0 * c.au
    return np.exp(- (r - r_0)**2 / δ_0**2) # [Unitless]

# Insert Mdot in units of M_sun/yr
# r_0 indicates the radius of infall
# δ_0 is σ in the gaussian hence the width of the infall
# r_in and r_out is the range of the radial grid
# The dimension of the normalization factor A is [M/TL^2]
def norm_Cinfall(sim, Minfall, r_0, δ_0):
    r_0 = r_0 * c.au; δ_0 = δ_0 * c.au
    r_in = sim.grid.r[0]
    r_out = sim.grid.r[-1]
    infall_cgs = Minfall * M_sun.to('g').value / c.year
    int = lambda r:  δ_0 * np.pi * (r_0 * np.sqrt(np.pi) * erf((r - r_0) / δ_0) - δ_0 * np.exp(-(r - r_0)**2 / δ_0**2))
    return (int(r_out) - int(r_in))**(-1) * infall_cgs # [g/cm2 /s]

def norm_source(sim, M_infall, r_0, δ_0):
    norm = norm_Cinfall(sim, M_infall, r_0, δ_0)
    return (norm * source(sim.grid.r, r_0, δ_0)) # [g/cm2 /s]

#### Adding time dependence ####
def time_dependence(sim, t0):
    t0 = t0 * c.year
    if sim.t < t0:
        return 0
    else: 
        return t0 * sim.t**(-1)

def time_gaussian_source(sim, M_infall = 1.1624056e-6, r_0 = 10, δ_0 = 1, t0 = 10000):
    constant_source = norm_source(sim, M_infall, r_0, δ_0)
    return time_dependence(sim, t0) * constant_source

####------------------- Defining dust source term dependent on an already defined gas source term-------------------###

# No boundary conditions is applied when defining Σ_dust

# max_a is the maximum grainsize in [cm] - if set to none the function uses the one specified in the simulation frame
# new_distExp is the size distribution   - if set to none the function uses the one specified in the simulation frame
def dust_source(sim, max_a = 0.001, new_distExp = None, D2Gratio = None):
    if new_distExp == None:
        new_distExp = sim.ini.dust.distExp
    if max_a == None:
        max_a = sim.ini.dust.aIniMax
    if D2Gratio == None:
        D2Gratio = sim.ini.dust.d2gRatio

    m = np.where(sim.dust.a <= max_a, sim.dust.a**(new_distExp + 4), 0.)
    m_integral = np.sum(m, axis=1)[..., None]
    m_integral = np.where(m_integral > 0., m_integral, 1.)
    # Normalize to mass
    Sigma = m / m_integral * sim.gas.S.ext[..., None] * D2Gratio
    Sigma = np.where(Sigma <= sim.dust.SigmaFloor,
                             0.1*sim.dust.SigmaFloor,
                             Sigma)
    return Sigma




