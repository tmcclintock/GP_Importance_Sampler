import numpy as np
import george
from george import kernels
import sample_generator as sg
from scipy.optimize import minimize

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
        self.sample_generator = sg.SampleGenerator(self.chain, scale=3)
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

    def train(self, kernel=None):
        """Train a Gaussian Process to interpolate the log-likelihood
        of the training samples.
        """
        x, L, lnL = self.get_training_data()
        if kernel is None:
            cov = self.sample_generator.covariance
            #m = george.Metric(cov, ndim=len(cov))
            kernel = kernels.ExpSquaredKernel(metric=cov, ndim=2)
        gp = george.GP(kernel, mean=10*np.min(lnL))
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
        #except np.linalg.linalg.LinAlgError:
        #    print("LinAlg error. Select new training points.")
        #    return
        gp.set_parameter_vector(result.x)
        self.gp = gp
        self.lnL = lnL
        return

    def predict(self, x, return_var=False):
        pred, pred_var = self.gp.predict(self.lnL, x)
        return pred
        
if __name__ == "__main__":
    import scipy.stats
    x_mean, y_mean = 0, 0
    means = np.array([x_mean, y_mean])
    cov = np.array([[1,0.1],[0.1,0.5]])
    icov = np.linalg.inv(cov)
    chain = np.random.multivariate_normal(mean=means,
                                          cov=cov,
                                          size=(1000))
    likes = scipy.stats.multivariate_normal.pdf(chain, mean=means, cov=cov)
    IS = ImportanceSampler(chain, likes)
    IS.select_training_points(100, method="circular")
    IS.train()

    x, y = chain.T
    xp = np.linspace(np.min(x),np.max(x))
    yp = np.zeros_like(xp) + y_mean

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='b', s=0.5, alpha=0.2)
    points, _, _ = IS.get_training_data()
    plt.scatter(points[:,0], points[:,1], c='k', s=10)
    plt.plot(xp, yp, c='r')
    plt.show()
    
    plt.hist(x, density=True, label=r"P(x)")
    p = np.vstack((yp,xp)).T
    lnLp = IS.predict(p)
    Lp = np.exp(lnLp)
    plt.plot(xp, Lp, label=r"$P(x|y=\mu_y)$")
    plt.legend()
    plt.show()
