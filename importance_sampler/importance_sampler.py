import numpy as np
import george
from george import kernels
import sample_generator as sg
from scipy.optimize import minimize

class ImportanceSampler(object):
    """A class used for enabling importance sampling of an MCMC chain.

    Args:
        chain: an MCMC chain that we want to draw samples from
        lnlikes: the log-likelihoods of the samples in the MCMC chain
        scale: 'spread' of the training points. Default is 6.
    """
    def __init__(self, chain, lnlikes, scale=10):
        chain = np.asarray(chain)
        lnlikes = np.asarray(lnlikes).copy()
        
        if chain.ndim > 2:
            raise Exception("chain cannot be more than two dimensional.")
        if chain.ndim < 1:
            raise Exception("chain must be a list of parameters.")
        if lnlikes.ndim > 1:
            raise Exception("lnlikes must be a 1D array.")
        self.chain = np.atleast_2d(chain)
        if len(self.chain) < len(self.chain[0]):
            raise Exception("More samples than parameters in chain.")

        #Most lnlikelihoods come in incorrectly normalized. Make an attempt to fix that
        #lnlikes -= np.min(lnlikes) #subtract off the minimum
        #lnlikes /= np.max(lnlikes) #range of values now goes from 0 to 1
        #lnlikes -= 10 #pretend that the minimum value in the chain is a 10 sigma fluctuation
        self.lnlike_max = np.max(lnlikes)
        lnlikes -= self.lnlike_max
        
        self.lnlikes = lnlikes
        self.sample_generator = sg.SampleGenerator(self.chain, scale=scale)
        self.chain_means = np.mean(self.chain, 0)
        self.chain_stddevs = np.sqrt(self.sample_generator.covariance.diagonal())
        #self.select_training_points(method="circular")
        #self.train()

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
        
    def select_training_points(self, Nsamples=40, method="LH", **kwargs):
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
        return (self.chain[inds], self.lnlikes[inds])

    def train(self, kernel=None):
        """Train a Gaussian Process to interpolate the log-likelihood
        of the training samples.
        """
        x, lnL = self.get_training_data()
        #Remove the mean and standard deviation from the training data
        x[:] -= self.chain_means
        x[:] /= self.chain_stddevs
        _guess = 0.5*np.ones(len(self.sample_generator.covariance))
        if kernel is None:
            kernel = kernels.ExpSquaredKernel(metric=_guess, ndim=len(_guess))
            #kernel = kernels.ExpSquaredKernel(metric=self.sample_generator.covariance, ndim=len(_guess))
        #gp = george.GP(kernel, mean=20*np.min(self.lnlikes)) #Extrapolate to a 20sigma fluctuation
        lnPmin = np.min(self.lnlikes)
        gp = george.GP(kernel, mean=lnPmin-np.fabs(lnPmin)*3) #Extrapolate to a 20sigma fluctuation
        gp.compute(x)
        def neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(lnL)

        def grad_neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(lnL)

        #try:
        result = minimize(neg_ln_likelihood, gp.get_parameter_vector(),
                          jac=grad_neg_ln_likelihood)
        gp.set_parameter_vector(result.x)
        self.gp = gp
        self.lnL_training = lnL
        return

    def predict(self, x, return_var=False):
        #Remove the chain mean and standard dev from the predicted point
        x = np.atleast_2d(x).copy()
        x[:] -= self.chain_means
        x[:] /= self.chain_stddevs
        pred, pred_var = self.gp.predict(self.lnL_training, x)
        return pred + self.lnlike_max #re-add on the max that we took of when building
        
if __name__ == "__main__":
    import scipy.stats
    x_mean, y_mean = 3, 0
    means = np.array([x_mean, y_mean])
    cov = np.array([[1,0.1],[0.1,0.5]])
    icov = np.linalg.inv(cov)
    chain = np.random.multivariate_normal(mean=means,
                                          cov=cov,
                                          size=(1000))
    likes = scipy.stats.multivariate_normal.pdf(chain, mean=means, cov=cov)
    lnlikes = np.log(likes)
    IS = ImportanceSampler(chain, lnlikes)
    IS.select_training_points(100, method="LH")
    IS.train()
    x, y = chain.T
    xp = np.linspace(np.min(x),np.max(x))
    yp = np.zeros_like(xp) + y_mean

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='b', s=0.5, alpha=0.2)
    points, _ = IS.get_training_data()
    plt.scatter(points[:,0], points[:,1], c='k', s=5)
    plt.plot(xp, yp, c='r')
    plt.show()
    
    plt.hist(x, density=True, label=r"$P(x)$")
    p = np.vstack((xp,yp)).T
    lnLp = IS.predict(p)
    Lp = np.exp(lnLp)
    plt.plot(xp, Lp, label=r"$P(x|y=\mu_y)$")
    plt.legend()
    plt.show()
