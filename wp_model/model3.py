import numpy as np
from sympy import symbols, pi, exp, sqrt, N
from scipy.integrate import quad
from functools import lru_cache
import matplotlib.pyplot as plt

# Parameters for the model
parameters = {
    'minR': 0,
    'maxR': 100,
    'minRecur': 5,
    'rho': 0.01,
    'b': 1,
    'S': 1,
    'R': 100,
    'gamma': 1,
    'c': 0.5,
    'd': 0.5,
    'PO': 1000,
    'Q': 10,
    'alpha': 10000,
    'beta': 10,
    'xi': 1,
    'psi': 0.5,
    'omega': 0.5,
    'lambdaz': 1,
    'U': 5,
    'lambda0': 0.2,
    'StigmaArea': 0.0001
}



# Total pollen and ovule output
@lru_cache(maxsize=None)
def resources(Ai):
    return 1 - Ai

@lru_cache(maxsize=None)
def XM(Ai):
    parameters['alpha'] = parameters['PO'] * parameters['Q']
    return parameters['alpha'] * (resources(Ai) / 2)

@lru_cache(maxsize=None)
def XF(Ai):
    parameters['beta'] = parameters['Q']
    return parameters['beta'] * (resources(Ai) / 2)

# Define dispersal kernel functions for animal pollination
@lru_cache(maxsize=None)
def KAP(r):
    return ((parameters['omega'] + parameters['psi'] * (1 - parameters['psi'])) / parameters['xi']) * exp(-((parameters['omega'] + parameters['psi'] * (1 - parameters['omega'])) * r) / parameters['xi'])

# Define dispersal kernel functions for wind pollination
@lru_cache(maxsize=None)
def KWP(r):
    return sqrt(parameters['lambda0'] / (pi * r)) * exp(-parameters['lambda0'] * r)

# Probability of a pollen grain being dispersed by animal pollinators
@lru_cache(maxsize=None)
def PAP(Ai):
    return Ai**(parameters['gamma'] / parameters['rho'])

# Fraction of pollen available for wind pollination
@lru_cache(maxsize=None)
def PWP(Ai):
    return (1 - PAP(Ai)) * ((parameters['U']**2 * (1 - Ai)) / (parameters['U']**2 + Ai))

# Amount of pollen transferred from a focal plant to another plant at distance r
@lru_cache(maxsize=None)
def t(Ai, r):
    return XM(Ai) * (PAP(Ai) * KAP(r) + PWP(Ai) * KWP(r) * parameters['rho'] * parameters['StigmaArea'] * XF(Ai))

# Total number of pollen grains from all other plants in the population
@lru_cache(maxsize=None)
def T(Ai):
    return 2 * pi * parameters['rho'] * XM(Ai) * quad(lambda rp: (PAP(Ai) * KAP(rp) + PWP(Ai) * KWP(rp) * parameters['rho'] * parameters['StigmaArea'] * XF(Ai)) * rp, 0, parameters['R'], epsabs=1E-9, epsrel=1E-9, limit=50)[0]

# Probability of fertilization
@lru_cache(maxsize=None)
def Fmut(Amut, Ares, r):
    return t(Amut, r) / (t(Amut, r) + T(Ares))

@lru_cache(maxsize=None)
def Fres(Amut, Ares, r):
    return t(Ares, r) / (t(Ares, r) + T(Ares))

# Probability that the number of pollen grains exceeds the threshold for fertilization
@lru_cache(maxsize=None)
def ExceedsThreshold(x):
    return 1 - exp(-parameters['b'] * x)

# Male fitness function for the mutant plant
@lru_cache(maxsize=None)
def WmutM(Amut, Ares):
    return 2 * pi * parameters['rho'] * XF(Ares) * quad(lambda r: ExceedsThreshold(t(Amut, r) + T(Ares)) * Fmut(Amut, Ares, r) * r, 0, parameters['R'], epsabs=1E-9, epsrel=1E-9, limit=50)[0]

# Male fitness function for the resident plant
@lru_cache(maxsize=None)
def WresM(Amut, Ares):
    return 2 * pi * parameters['rho'] * XF(Ares) * quad(lambda r: ExceedsThreshold(t(Ares, r) + T(Ares)) * Fres(Amut, Ares, r) * r, 0, parameters['R'], epsabs=1E-9, epsrel=1E-9, limit=50)[0]

# Female fitness function for the mutant plant
@lru_cache(maxsize=None)
def WmutF(Amut, Ares):
    return XF(Amut) * ExceedsThreshold(T(Ares))

# Female fitness function for the resident plant
@lru_cache(maxsize=None)
def WresF(Amut, Ares):
    return XF(Ares) * ExceedsThreshold(T(Ares))

# Total fitness function for the mutant plant
@lru_cache(maxsize=None)
def Wmut(Amut, Ares):
    return WmutM(Amut, Ares) + WmutF(Amut, Ares)

# Total fitness function for the resident plant
@lru_cache(maxsize=None)
def Wres(Amut, Ares):
    return WresM(Amut, Ares) + WresF(Amut, Ares)

# Relative fitness function for the mutant plant
@lru_cache(maxsize=None)
def Wrel(Amut, Ares):
    return N(Wmut(Amut, Ares) / Wres(Amut, Ares))

def clear_all_cache():
    resources.cache_clear()
    XM.cache_clear()
    XF.cache_clear()
    KAP.cache_clear()
    KWP.cache_clear()
    PAP.cache_clear()
    PWP.cache_clear()
    t.cache_clear()
    T.cache_clear()
    Fmut.cache_clear()
    Fres.cache_clear()
    ExceedsThreshold.cache_clear()
    WmutM.cache_clear()
    WresM.cache_clear()
    WmutF.cache_clear()
    WresF.cache_clear()
    Wmut.cache_clear()
    Wres.cache_clear()
    Wrel.cache_clear()
"""
# Set your values
# Amut_value = 0.3  # for example
# Ares_value = 0.3  # for example

# Calculate the relative fitness
# relative_fitness = Wrel(Amut_value, Ares_value)
# relative_fitness_value = N(relative_fitness)
# print(f"Relative fitness is: {relative_fitness_value}")


# Define ranges of values for density and PO
#density_values = np.linspace(0.01, 100, 5)
#PO_values = np.linspace(100, 100000, 10)
density_values = [0.01, 0.1, 1, 10, 100]
PO_values = [100,1000, 10000, 100000, 1000000]
# Define range of values for Amut and Ares
Amut_values = [0, 0.25, 0.5, 0.75, 0.9]
Ares_values = [0, 0.25, 0.5, 0.75, 0.9]

# Create plot of relative fitness vs density
fig, axs = plt.subplots(len(Ares_values), 1, figsize=(10, 15), sharex=True)
for i, Ares in enumerate(Ares_values):
    for Amut in Amut_values:
        rel_fitness_values = []
        for rho in density_values:
            parameters['rho'] = rho
            clear_all_cache()
            rel_fitness = Wrel(Amut, Ares)
            rel_fitness_values.append(rel_fitness)
        axs[i].plot(density_values, rel_fitness_values, label=f'Amut={Amut}')
    axs[i].set_title(f"Ares={Ares}")
    axs[i].set_ylabel("Relative Fitness")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Density")
plt.tight_layout()
#plt.show()
plt.savefig('figure_rho.png')
# Create plot of relative fitness vs PO
fig, axs = plt.subplots(len(Ares_values), 1, figsize=(10, 15), sharex=True)
for i, Ares in enumerate(Ares_values):
    for Amut in Amut_values:
        rel_fitness_values = []
        for PO in PO_values:
            parameters['PO'] = PO
            clear_all_cache()
            rel_fitness = Wrel(Amut, Ares)
            rel_fitness_values.append(rel_fitness)
        axs[i].plot(PO_values, rel_fitness_values, label=f'Amut={Amut}')
    axs[i].set_title(f"Ares={Ares}")
    axs[i].set_ylabel("Relative Fitness")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Pollen Output (PO)")
plt.tight_layout()
#plt.show()
plt.savefig('figure_PO.png')


Ares_values = np.arange(0, 0.99, 0.2)
Amut_values = np.arange(0, 1, 0.1)

Ares_values = np.arange(0, 0.99, 0.2)
Amut_values = np.arange(0, 1, 0.1)
rho_values = [0.01, 0.1, 1, 10, 100]

# Defining figure and subplots
fig, axs = plt.subplots(len(rho_values), 1, figsize=(10, 15), sharex=True)

for idx, rho in enumerate(rho_values):
    parameters['rho'] = rho
    clear_all_cache()
    for Ares in Ares_values:
        data = []
        for Amut in Amut_values:
            rel_fitness = Wrel(Amut, Ares)
            data.append([Amut, rel_fitness])

        data = np.array(data)
        axs[idx].plot(data[:, 0], data[:, 1], label=f"Ares={Ares:.2f}")

        axs[idx].set_yscale('log')
        axs[idx].grid(True, which="both", ls="--")
        axs[idx].axhline(y=1, color='k', linestyle='--')
        axs[idx].legend()
        axs[idx].set_ylabel("Relative Fitness")
        axs[idx].set_title(f"rho={rho}")

axs[-1].set_xlabel("Mutant's investment in pollinator attraction (Amut)")

plt.tight_layout() 
#plt.show()
plt.savefig('figure3.png')
"""