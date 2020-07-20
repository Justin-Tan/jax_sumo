import jax.numpy as jnp
import jax.scipy.stats.norm as norm

from jax import lax, random
from jax.lax import scan

# ======== Operations on diagonal Gaussians ========


def diag_gaussian_logpdf(x, mean=None, logvar=None):
    # Log of PDF for diagonal Gaussian for a single point
    D = x.shape[0]
    if (mean == None) and (logvar == None):
        # Standard Gaussian
        mean, logvar = jnp.zeros_like(x), jnp.zeros_like(x)
    # manual = -0.5 * (jnp.log(2*jnp.pi) + logvar + (x-mean)**2 * jnp.exp(-logvar))
    # print('manual check', jnp.sum(manual))
    return jnp.sum(norm.logpdf(x, loc=mean, scale=jnp.exp(0.5 * logvar)))

def diag_gaussian_entropy(mean, logvar):
    # Entropy of diagonal Gaussian
    D = mean.shape[0]
    diff_entropy = 0.5 * (D * jnp.log(2 * jnp.pi) + jnp.sum(logvar) + D)
    return diff_entropy

def diag_gaussian_sample(rng, mean, logvar):
    # Sample single point from a diagonal multivariate distribution
    return mean + jnp.exp(0.5 * logvar) * random.normal(rng, mean.shape)

def diag_gaussian_kl_divergence(mean0, logvar0, mean1=None, logvar1=None):
    # Compute KL divergence between two diagonal-covariance Gaussians
    # D_KL(N(mu_0, Sigma_0) || N(mu_1, Sigma_1))

    if (mean1 == None) and (logvar1 == None):
        # D_KL(N(mu0, Sigma_0) || N(0, 1))
        mean1, logvar1 = jnp.zeros_like(mean0), jnp.zeros_like(logvar0)

    D = mean0.shape[0]
    trace_part = jnp.exp(logvar0 - logvar1)
    quadratic_part = (mean1 - mean0)**2 * jnp.exp(-logvar1)
    return 0.5 * (-D + jnp.sum(trace_part + quadratic_part + (logvar1 - logvar0)))


# ========== Unnormalized log density ===========

def funnel_log_density(x):
    return norm.logpdf(x[0], 0, jnp.exp(x[1])) + \
           norm.logpdf(x[1], 0, 1.35)


# ========== SUMO CDF Sampling ==========

def reverse_cdf(k, alpha=80, b=0.1):
    return jnp.where(k < alpha, 1./k, 1./alpha * (1-b)**(k-alpha))

def sampling_tail(rng, start_k=4, cap=100, alpha=80):

    def _body_fun(x):
        k, u, threshold, rng = x
        u = random.uniform(random.fold_in(rng, k))
        threshold = reverse_cdf(k, alpha=80)
        k = lax.cond(u < threshold, lambda x: x + 1, lambda x: x, k)
        return k, u, threshold ,rng

    def _cond_fun(x):
        k, u, threshold, rng = x
        return u < threshold

    k = start_k
    k, *_ = lax.while_loop(cond_fun=_cond_fun, body_fun=_body_fun, 
                               init_val=(start_k,0.,1.,rng))
    
    return k.astype(jnp.int32)

# ========= logcumsumexp ==========
def logcumsumexp(x):
    # Prefix scan - faster on CPU but slower on GPU
    def _logaddexpcarry(carry, x):
        out = jnp.logaddexp(carry, x)
        return out, out
    cumsum_final, _logcumsumexp = scan(_logaddexpcarry, init=-jnp.inf, xs=x)
    return _logcumsumexp

def naive_logcumsumexp(x):
    x_max = jnp.max(x)
    return x_max + jnp.log(jnp.cumsum(jnp.exp(x-x_max)))