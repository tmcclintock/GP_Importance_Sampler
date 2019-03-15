import numpy as np
import george as geo
import sample_generator as sg

class ImportanceSampler(object):
    """A class used for enabling importance sampling of an MCMC chain.

    Args:
        chain: an MCMC chain that we want to draw samples from

    """
    def __init__(self, chain):
        chain = np.asarray(chain)
        if chain.ndim > 2:
            raise Exception("chain cannot be more than two dimensional.")
        if chain.ndim < 1:
            raise Exception("chain must be a list of parameters.")
        self.chain = np.atleast_2d(chain)

    
