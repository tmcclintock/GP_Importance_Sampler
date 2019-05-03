import numpy as np
import george
from george import kernels
import sample_generator as sg
from scipy.optimize import minimize

class ImportanceSampler(object):
    """A class used for enabling importance sampling of an MCMC chain.

    Note: the 'scale' parameter is not used if you use the `lnlikes_binning`
    method of choosing training points.

    Args:
        chain: an MCMC chain that we want to draw samples from
        lnlikes: the log-likelihoods of the samples in the MCMC chain
        scale: 'spread' of the training points. Default is 6.
    """
    def __init__(self, chain, lnlikes, scale=5):
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

        #Remove the max lnlike. This can help numerical stability
        self.lnlike_max = np.max(lnlikes)
        lnlikes -= self.lnlike_max
        
        self.lnlikes = lnlikes
        self.sample_generator = sg.SampleGenerator(self.chain, scale=scale)
        self.chain_means = np.mean(self.chain, 0)
        self.chain_stddevs = np.sqrt(self.sample_generator.covariance.diagonal())
        self.chain_cov = np.cov(self.chain.T)
        w, R = np.linalg.eig(self.chain_cov)
        self.eigenvalues = w
        self.rotation_matrix = R
        

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

    def _select_lnlike_based_training_points(self, Nsamples, Nbins=5):
        """Select samples from the histogramed loglikelihoods.

        Args:
            Nsamples (int): number of samples to use
            Nbins (int): number of histogram bins to chop up the likelihoods; default is 5

        Returns:
            (array-like): indices of the chain points to use for trianing

        """
        if Nbins > Nsamples:
            raise Exception("Cannot have more bins than samples.")

        N_per_bin = Nsamples / Nbins
        N_extra = Nsamples % Nbins #extra for the last bin

        lnlikes = self.lnlikes
        N_in_bins, edges = np.histogram(lnlikes, bins=Nbins)
        all_inds = np.arange(len(lnlikes)) #individual indices

        #indices to return
        ret_inds = np.array([])

        #Sort the lnlikes
        sorted_inds = np.argsort(lnlikes)
        llsorted = lnlikes[sorted_inds]
        for i in range(0, Nbins):
            ii = (llsorted >= edges[i]) * (llsorted < edges[i+1])
            sii = sorted_inds[ii]
            #If the bin is not populated (the tail), use the whole bin
            if len(sii) < N_per_bin:
                ret_inds = np.append(ret_inds, sii)
            else:
                #Otherwise, take a random selection from the bin
                ret_inds = np.append(ret_inds, np.random.choice(sii, N_per_bin, replace=False))
            continue
        #If we don't have enough samples, take more samples but make sure we have no duplicates
        if len(ret_inds) < Nsamples:
            Nleft = Nsamples - len(ret_inds)
            for i in range(0, Nleft):
                new_sample = np.random.choice(all_inds)
                if new_sample not in ret_inds: #Enforce no replacement
                    ret_inds = np.append(ret_inds, new_sample)
                else:
                    i -= 1
                continue
        print Nbins#ret_inds
        return ret_inds.astype(int)
        
    def select_training_points(self, Nsamples=40, method="LH", **kwargs):
        """Select training points from the chain to train the GPs.

        Note: this method does not use the "scale" parameter.
        
        Args:
            Nsamples (int): number of samples to use; defualt is 40
            method (string): design for training points; defualt is Latin
                Hypercube, or 'LH'
            kwargs: keywords to pass the sample generator.get_samples() method

        """
        if method == "lnlikes_binning":
            Nbins = 5
            if kwargs is not None:
                for key, value in kwargs.iteritems():
                    if key == "Nbins":
                        Nbins = value
                    continue
            indices = self._select_lnlike_based_training_points(Nsamples, Nbins)
        else:
            samples = self.sample_generator.get_samples(Nsamples, method, **kwargs)
            cov = self.sample_generator.covariance
            def sqdists(chain, s, cov):
                X = chain[:] - s
                r = np.linalg.solve(cov, X.T).T
                d = np.sum(X * r, axis=1)
                return d
            indices = np.array([np.argmin(sqdists(self.chain, s, cov)) for s in samples])
            
        #Include the max liklihood point
        best_ind = np.argmax(self.lnlikes)
        self.training_inds = np.append(indices, best_ind)
        return
    
    def get_training_data(self):
        """Obtain the currently used training points

        return:
            Tuple of chain and lnlikelihood values.
        """
        inds = self.training_inds
        return (self.chain[inds], self.lnlikes[inds])

    def train(self, kernel=None):
        """Train a Gaussian Process to interpolate the log-likelihood
        of the training samples.

        Args:
            kernel (george.kernels.Kernel object): kernel to use, or any 
                acceptable object that can be accepted by the george.GP object

        """
        x, lnL = self.get_training_data()
        #Remove the mean and standard deviation from the training data
        x[:] -= self.chain_means
        x[:] /= self.chain_stddevs
        x = np.array([np.dot(self.rotation_matrix.T, xi) for xi in x])
        #x[:] = np.dot(self.rotation_matrix.T, x[:])
        _guess = 0.5#*np.ones(1)#len(self.sample_generator.covariance))
        if kernel is None:
            kernel = kernels.ExpSquaredKernel(metric=_guess, ndim=len(x[0]))
            #kernel = kernels.ExpSquaredKernel(metric=self.sample_generator.covariance, ndim=len(_guess))
        #Note: the mean is set slightly lower that the minimum lnlike
        #gp = george.GP(kernel, mean=20*np.min(self.lnlikes))
        lnPmin = np.min(self.lnlikes)
        gp = george.GP(kernel, mean=lnPmin-np.fabs(lnPmin)*3)
        gp.compute(x)
        def neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(lnL)

        def grad_neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(lnL)

        #try:
        #method = "Nelder-Mead"
        result = minimize(neg_ln_likelihood, gp.get_parameter_vector(),
                          jac=grad_neg_ln_likelihood)#, method=method)
        print result
        gp.set_parameter_vector(result.x)
        self.gp = gp
        self.lnL_training = lnL
        return

    def predict(self, x):
        """Given a set of parameters, return the predicted log probability.

        Args:
            x (array-like): parameters to interpolate at
        
        Returns:
            interpolated log probability at x.

        """
        #Remove the chain mean and standard dev from the predicted point
        x = np.atleast_2d(x).copy()
        x[:] -= self.chain_means
        x[:] /= self.chain_stddevs
        x = np.array([np.dot(self.rotation_matrix.T, xi) for xi in x])
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
                                          size=(10000))
    likes = scipy.stats.multivariate_normal.pdf(chain, mean=means, cov=cov)
    lnlikes = np.log(likes)
    IS = ImportanceSampler(chain, lnlikes)
    IS.select_training_points(1000, method="lnlikes_binning", Nbins = 5)
    IS.train()
    x, y = chain.T

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='b', s=0.5, alpha=0.2)
    points, _ = IS.get_training_data()
    plt.scatter(points[:,0], points[:,1], c='k', s=5)
    plt.show()

    """
    plt.hist(x, density=True, label=r"$P(x)$")
    p = np.vstack((xp,yp)).T
    lnLp = IS.predict(p)
    Lp = np.exp(lnLp)
    plt.plot(xp, Lp, label=r"$P(x|y=\mu_y)$")
    plt.legend()
    plt.show()
    """
