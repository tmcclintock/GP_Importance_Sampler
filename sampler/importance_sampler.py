import numpy as np
import george as geo
import sample_generator as sg

class ImportanceSampler(object):
    """A class used for enabling importance sampling of an MCMC chain.

    Args:
        chain: an MCMC chain that we want to draw samples from
        likes: the likelihoods of the samples in the MCMC chain
    """
    def __init__(self, chain, likes):
        chain = np.asarray(chain)
        likes = np.asarray(likes)
        
        if chain.ndim > 2:
            raise Exception("chain cannot be more than two dimensional.")
        if chain.ndim < 1:
            raise Exception("chain must be a list of parameters.")
        if likes.ndim > 1:
            raise Exception("likes must be a 1D array.")
        self.chain = np.atleast_2d(chain)
        self.likes = likes
        self.lnlikes = np.log(likes)
        self.sample_generator = sg.SampleGenerator(self.chain, scale=5)

    def assign_new_sample_generator(self, scale=5, sample_generator=None):
        """Make a new SampleGenerator object and assign it to this sampler.
        
        Args:
            scale: scale level of the SampleGenerator. Samples are drawn from
                roughly this number of sigmas. Optional; default is 5
            sample_generator: a SampleGenerator object. Optional
        """
        if sample_generator is not None:
            self.sample_generator = sample_generator(self.chain, scale=scale)
        else:
            self.sample_generator = sg.SampleGenerator(self.chain, scale=scale)
        return
        
    def select_training_points(self, Nsamples=40, method="LHMDU", **kwargs):
        samples = self.sample_generator.get_samples(Nsamples, method, **kwargs)
        cov = self.sample_generator.covariance
        icov = np.linalg.inv(cov)
        def sqdists(chain, s, icov):
            X = chain[:] - s
            r = np.dot(icov, X.T).T
            d = np.sum(X * r, axis=1)
            return d
        indices = np.array([np.argmin(sqdists(self.chain, s, icov)) for s in samples])
        self.training_inds = indices
        return

    def get_training_data(self):
        inds = self.training_inds
        return (self.chain[inds], self.likes[inds], self.lnlikes[inds])
    
if __name__ == "__main__":
    chain = np.random.multivariate_normal(mean=[10, 0], cov=[[1,0.1],[0.1,0.5]], size=(10000))
    #chain = np.random.randn(10000, 2)
    x, y = chain.T
    likes = np.exp(-0.5*(x*x+y*y))/np.sqrt(2*np.pi)
    IS = ImportanceSampler(chain, likes)
    IS.select_training_points(100, method="circular")

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='b', s=0.5, alpha=0.2)
    points, _, _ = IS.get_training_data()
    plt.scatter(points[:,0], points[:,1], c='k', s=10)
    plt.show()
