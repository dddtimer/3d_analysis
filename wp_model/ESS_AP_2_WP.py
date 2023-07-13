"""
The primary objective of the model is to calculate the relative fitness of a given 'mutant' strategy compared to a 'resident' strategy
The core function that achieves this is `Wrel(Amut, Ares)`. 

Key concepts:

- Mutant: This is a theoretical individual or strategy that varies from the 'resident', or commonly found strategy, in the system. 
The variant is a result of mutation to the secondary sex allocation strategy.

- Resident: This is the most commonly found individual or strategy in the system. It represents the dominant strategy 
before the appearance of the mutant.

In this model, both mutant and resident strategies are represented by values (`Amut` for mutant and `Ares` for resident). 
These values can represent different allocations to resources conferring pollinator attraction. 

The function `Wrel(Amut, Ares)` computes the relative fitness of the mutant strategy compared to the resident strategy. 
We expect that if the mutant strategy is the same as the resident strategy (i.e., `Amut` equals `Ares`), the relative fitness 
should be 1. 
The program also includes functionality to plot the relative fitness as a function of various parameters.
"""

from sympy import  pi, exp, sqrt, N
from scipy.integrate import quad
from functools import lru_cache
import matplotlib.pyplot as plt

# Parameters for the model
PARAMETERS = {
    'epsabs': 1E-9, # Absolute integration error tolerance. The desired absolute error in the result of the integral.
    'epsrel': 1E-9, # Relative integration error tolerance. The desired relative error in the result of the integral.
    'limit': 50, # An upper limit for the number of integration steps. Determines the accuracy of numerical integration.
    'R': 100, # Radius or range over which plants and pollinators interact.
    'rho': 0.01, # Plant density. It determines the number of plants in a given area.
    'b': 1, # Min number of pollen grains needed to fertilize an ovule
    'gamma': 1, # Parameter related to the attractiveness of the plants.   
    'PO': 1000, # Pollen to ovule ratio
    'Q': 10, # Conversion factor for absolute pollen and ovule production (e.g., Q=10->pollen=1000*10, ovules=10)
    'alpha': 10000, # Pollen production if 100% of resources went to male function
    'beta': 10, # Ovule production if 100% of resources went to female function
    'xi': 1, # 
    'psi': 0.5, # 
    'omega': 0.5, 
    'lambdaz': 1, 
    'U': 5, # Average wind speed of the environment
    'lambda0': 0.2,    
    'StigmaArea': 0.0001 # Size of the receptive area for pollen on the plant.
}


# Define a FitnessModel class that encapsulates the computation of relative fitness
class FitnessModel:

    def __init__(self, parameters):
        self.parameters = parameters

    # Calculate the resources available for a plant based on its attractiveness Ai
    @lru_cache(maxsize=None)
    def resources(self, Ai):
        return (1 - Ai) / 2

    # Calculate the pollen production (XM) for a plant based on its attractiveness (Amut) and the attractiveness of the resident population (Ares)  
    @lru_cache(maxsize=None)
    def XM(self, Amut, Ares):
        male_resources = self.resources(Ares) + (Ares - Amut)
        alpha = self.parameters['PO'] * self.parameters['Q']
        print(f"Amut:{Amut} Ares:{Ares} M_mut:{male_resources} M_res:{self.resources(Ares)} XM_mut:{alpha * male_resources} XM_res:{alpha*self.resources(Ares)}")
        if alpha*male_resources < 1:
            raise ValueError(f"Pollen production cannot be less than 1; got {alpha*male_resources}")
        return alpha * male_resources
    
    # Calculate the ovule production (XF) for a plant based on its attractiveness Ares
    @lru_cache(maxsize=None)
    def XF(self, Ares):
        beta = self.parameters['Q']
        return beta * (self.resources(Ares))

    # Calculate the animal pollination dispersal kernel (KAP) based on the distance r
    @lru_cache(maxsize=None)
    def KAP(self, r):
        return ((self.parameters['omega'] + self.parameters['psi'] * (1 - self.parameters['psi'])) / self.parameters['xi']) * exp(-((self.parameters['omega'] + self.parameters['psi'] * (1 - self.parameters['omega'])) * r) / self.parameters['xi'])

    # Calculate the wind pollination dispersal kernel (KWP) based on the distance r
    @lru_cache(maxsize=None)
    def KWP(self, r):
        return sqrt(self.parameters['lambda0'] / (pi * r)) * exp(-self.parameters['lambda0'] * r)

    # Calculate the probability of animal pollination (PAP) based on the attractiveness Ai
    @lru_cache(maxsize=None)
    def PAP(self, Ai):
        return Ai**(self.parameters['gamma'] / self.parameters['rho'])

    # Calculate the probability of wind pollination (PWP) based on the attractiveness Ai
    @lru_cache(maxsize=None)
    def PWP(self, Ai):
        return (1 - self.PAP(Ai)) * ((self.parameters['U']**2 * (1 - Ai)) / (self.parameters['U']**2 + Ai))

    # Calculate the amount of pollen transferred per unit area (t) from a focal plant, based on its attractiveness Ai, pollen production XM, and distance r
    @lru_cache(maxsize=None)
    def t(self, Ai, XM, r):
        return XM * (self.PAP(Ai) * self.KAP(r) + self.PWP(Ai) * self.KWP(r) * self.parameters['rho'] * self.parameters['StigmaArea'] * self.XF(Ai))

    # Calculate the total amount of pollen transferred from all other plants in the population to a focal plant (T), based on its attractiveness Ai and pollen production XM
    @lru_cache(maxsize=None)
    def T(self, Ai,XM):
        return 2 * pi * self.parameters['rho'] * XM * quad(lambda rp: (self.PAP(Ai) * self.KAP(rp) + self.PWP(Ai) * self.KWP(rp) * self.parameters['rho'] * self.parameters['StigmaArea'] * self.XF(Ai)) * rp, 0, self.parameters['R'], epsabs=self.parameters['epsabs'], epsrel=self.parameters['epsrel'], limit=self.parameters['limit'])[0]

    # Calculate the probability that a seed on a focal plant is fathered by a plant with attractiveness Amut, given the pollen production of both mutant (XM_mut) and resident (XM_res) plants and the distance r
    @lru_cache(maxsize=None)
    def Fmut(self, Amut, Ares, XM_mut, XM_res, r):
        return self.t(Amut, XM_mut, r) / (self.t(Amut, XM_mut, r) + self.T(Ares, XM_res))

    # Calculate the probability that a seed on a focal plant is fathered by a resident plant, given the pollen production of resident plants (XM_res) and the distance r
    @lru_cache(maxsize=None)
    def Fres(self, Ares, XM_res, r):
        return self.t(Ares, XM_res, r) / (self.t(Ares, XM_res, r) + self.T(Ares, XM_res))

    # Calculate whether a certain value of pollen production (x) exceeds a certain threshold for ovule fertilization (b)
    def ExceedsThreshold(self, x):
        return 1 - exp(-self.parameters['b'] * x)

    # Calculate the male fitness of a mutant plant (WmutM), based on the attractiveness of the mutant plant Amut, the attractiveness of the resident plants Ares, and the pollen production of the mutant plants XM_mut and resident plants XM_res
    @lru_cache(maxsize=None)
    def WmutM(self, Amut, Ares, XM_mut, XM_res):
        return 2 * pi * self.parameters['rho'] * self.XF(Ares) * quad(lambda r: self.ExceedsThreshold(self.t(Amut, XM_mut, r) + self.T(Ares, XM_res)) * self.Fmut(Amut, Ares,XM_mut, XM_res, r) * r, 0, self.parameters['R'], epsabs=self.parameters['epsabs'], epsrel=self.parameters['epsrel'], limit=self.parameters['limit'])[0]

    # Calculate the male fitness of resident plants (WresM), based on the attractiveness of the resident plants Ares and their pollen production XM_res
    @lru_cache(maxsize=None)
    def WresM(self, Ares, XM_res):
        return 2 * pi * self.parameters['rho'] * self.XF(Ares) * quad(lambda r: self.ExceedsThreshold(self.t(Ares, XM_res, r) + self.T(Ares, XM_res)) * self.Fres(Ares, XM_res, r) * r, 0, self.parameters['R'], epsabs=self.parameters['epsabs'], epsrel=self.parameters['epsrel'], limit=self.parameters['limit'])[0]

    # Calculate the female fitness of a mutant plant (WmutF), based on the attractiveness of the resident plants Ares and their pollen production XM_res
    @lru_cache(maxsize=None)
    def WmutF(self, Ares, XM_res):
        return self.XF(Ares) * self.ExceedsThreshold(self.T(Ares,XM_res))

    # Calculate the female fitness of resident plants (WresF), based on the attractiveness of the resident plants Ares and their pollen production XM_res
    @lru_cache(maxsize=None)
    def WresF(self, Ares, XM_res):
        return self.XF(Ares) * self.ExceedsThreshold(self.T(Ares, XM_res))

    # Calculate the total fitness of a mutant plant (Wmut), based on its attractiveness Amut, the attractiveness of the resident plants Ares, and the pollen production of the mutant plants XM_mut and resident plants XM_res
    @lru_cache(maxsize=None)
    def Wmut(self, Amut, Ares, XM_mut, XM_res):
        return self.WmutM(Amut, Ares, XM_mut, XM_res) + self.WmutF(Ares, XM_res)

    # Calculate the total fitness of resident plants (Wres), based on their attractiveness Ares and their pollen production XM_res
    @lru_cache(maxsize=None)
    def Wres(self, Ares, XM_res):
        return self.WresM(Ares, XM_res) + self.WresF(Ares, XM_res)

    # Calculate the relative fitness of a mutant plant compared to resident plants, based on the attractiveness of the mutant plant Amut and the attractiveness of the resident plants Ares
    @lru_cache(maxsize=None)
    def Wrel(self, Amut, Ares):
        XM_res=self.XM(Ares,Ares)
        XM_mut=self.XM(Amut,Ares)
        return N(self.Wmut(Amut, Ares, XM_mut, XM_res) / self.Wres(Ares, XM_res))

    # Clear all caches from memoization
    def clear_all_cache(self):
        self.KAP.cache_clear()
        self.KWP.cache_clear()
        self.PAP.cache_clear()
        self.PWP.cache_clear()
        self.t.cache_clear()
        self.T.cache_clear()
        self.Fmut.cache_clear()
        self.Fres.cache_clear()
        self.WmutM.cache_clear()
        self.WresM.cache_clear()
        self.WmutF.cache_clear()
        self.WresF.cache_clear()
        self.Wmut.cache_clear()
        self.Wres.cache_clear()
        self.Wrel.cache_clear()

    # Checks whether the value of Amut is valid, given the current allocation to male function
    def is_valid_amut(Amut, Ares, alpha):
        if Amut < Ares:
            return True
        else:
            upper_bound = Ares + (1 - Ares) / 2 - 1 / alpha
            return Ares < Amut <= upper_bound

    '''Graphing function: explores how a chosen parameter impacts the relative fitness of the mutant plant.
     Plots the relative fitness for a range of values of the chosen parameter (e.g., rho), and for a 
    variety of different strategies for the mutant plant (i.e., different values of Amut). The plots are separated by 
    different strategies of the resident plants (i.e., different values of Ares).'''
    def plot_fitness(self, param_name, param_values, Amut_values, Ares_values, filename=None):
        if param_name not in self.parameters:
            raise ValueError(f"Invalid parameter name: {param_name}")

        fig, axs = plt.subplots(len(Ares_values), 1, figsize=(10, 15), sharex=True)
        for i, Ares in enumerate(Ares_values):
            for Amut in Amut_values:
                rel_fitness_values = []
                for value in param_values:
                    try:
                        self.parameters[param_name] = value
                        self.clear_all_cache()
                        # this call will raise a ValueError if Amut is invalid
                        XM_check = self.XM(Amut, Ares)
                        rel_fitness = self.Wrel(Amut, Ares)
                        rel_fitness_values.append(rel_fitness)
                    except ValueError as e:
                        print(f"Skipping invalid value: {e}")
                        continue
                if rel_fitness_values:  # only plot if there are valid fitness values
                    axs[i].plot(param_values[:len(rel_fitness_values)], rel_fitness_values, label=f'Amut={round(Amut, 2)}')
                    axs[i].scatter(param_values[:len(rel_fitness_values)], rel_fitness_values, color='black', s=10)  # add data points as dots

            axs[i].set_title(f"Ares={Ares}")
            axs[i].set_ylabel("Relative Fitness")
            axs[i].grid(True)
            axs[i].legend()

        axs[-1].set_xlabel(param_name.capitalize())
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        else:
            plt.show() 


model = FitnessModel(PARAMETERS)
rho_values = [0.01, 0.1, 1, 10, 100]
Amut_values = [0, 0.25, 0.5, 0.75, 0.9]
Ares_values = [0, 0.25, 0.5, 0.75, 0.9]

# calculate two values of relative fitness
model.Wrel(0,0.9)
model.Wrel(0.95,0.9)

model.plot_fitness('rho', rho_values, Amut_values, Ares_values, filename='figure_rho.png')

