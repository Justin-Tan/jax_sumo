import jax
import numpy as onp  # original CPU-backed NumPy
import jax.numpy as jnp

from jax import random
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp

from jax.experimental import stax
from jax.experimental import optimizers

import os, time, argparse
from functools import partial

import matplotlib.pyplot as plt

# Custom
from distributions import *

# ===== Encoder =====

def encoder(hidden_dim, z_dim, activation='Tanh'):
    activation = getattr(stax, activation)
    encoder_init, encode = stax.serial(
        stax.Dense(hidden_dim), activation,
        # stax.Dense(hidden_dim), activation,
        stax.FanOut(2),
        stax.parallel(stax.Dense(z_dim), stax.Dense(z_dim)),
    )
    return encoder_init, encode

# ===== Decoder =====

def decoder(hidden_dim, x_dim=2, activation='Tanh'):
    activation = getattr(stax, activation)
    decoder_init, decode = stax.serial(
        stax.Dense(hidden_dim), activation,
        # stax.Dense(hidden_dim), activation,
        stax.FanOut(2),
        stax.parallel(stax.Dense(x_dim), stax.Dense(x_dim)),
    )
    return decoder_init, decode

# ========= Helper functions for plotting. ==========

@partial(jit, static_argnums=(0, 1, 2, 4))
def _mesh_eval(func, x_limits, y_limits, params, num_ticks):
    # Evaluate func on a 2D grid defined by x_limits and y_limits.
    x = jnp.linspace(*x_limits, num=num_ticks)
    y = jnp.linspace(*y_limits, num=num_ticks)
    X, Y = jnp.meshgrid(x, y)
    xy_vec = jnp.stack([X.ravel(), Y.ravel()]).T
    zs = vmap(func, in_axes=(0, None))(xy_vec, params)
    return X, Y, zs.reshape(X.shape)

def mesh_eval(func, x_limits, y_limits, params, num_ticks=101):
    return _mesh_eval(func, x_limits, y_limits, params, num_ticks)

def callback(t, params, model, objective, num_samples=256):

    rng = random.PRNGKey(t)
    target_dist = lambda x, _: jnp.exp(funnel_log_density(x))
    approx_dist = lambda x, params: jnp.exp(iwelbo_amortized(x, rng, *params))

    print('Iteration: {} | Reverse-KL: {:.3f}'.format(t, objective(t, *params, model, False, num_samples)))
    x_limits, y_limits = [-2, 2], [-4,2]

    # Contours of true distribution
    plt.figure(figsize=(12, 4))
    X, Y, Z  = mesh_eval(target_dist, x_limits, y_limits, 500)
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, Z, 20, vmin=0., vmax=0.32, cmap=plt.cm.viridis)
    plt.colorbar(label='Density')
    plt.xlabel('$x$'); plt.ylabel('$y$'); plt.title('Target')

    X_v, Y_v, Z_v = mesh_eval(approx_dist, x_limits, y_limits, params)
    plt.subplot(1, 2, 2)
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.contourf(X_v, Y_v, Z_v, 20, vmin=0., vmax=0.32, cmap=plt.cm.viridis)
    plt.colorbar(label='Density')
    plt.xlabel('$x$'); plt.ylabel('$y$'); plt.title('Surrogate | Iteration {}'.format(t))

    # Samples from learned model
    rngs = random.split(rng, int(num_samples))
    z_samples, x_samples, *_ = vmap(generate_samples, in_axes=(0, None, None, None))(rngs, *params, None)
    plt.plot(x_samples[:, 0], x_samples[:, 1], 'b.')
    plt.xlim(x_limits)
    plt.ylim(y_limits)

    # plt.savefig('density_{}.pdf'.format(t), bbox_inches='tight', format='pdf', dpi=64)
    plt.draw()
    plt.pause(1.0/60.0)
    plt.clf()

# ===== IW-ELBO estimation =====

def generate_samples(rng, enc_params, dec_params, x):
    x_rng, z_rng = random.split(rng, 2)

    if x is None:
        # Sample from standard Gaussian prior
        mu_z, logvar_z = jnp.zeros(args.latent_dim), jnp.zeros(args.latent_dim)
    else:
        mu_z, logvar_z = encoder_forward(enc_params, x)
    z_sample = diag_gaussian_sample(z_rng, mu_z, logvar_z)
    mu_x, logvar_x = decoder_forward(dec_params, z_sample)
    x_sample = diag_gaussian_sample(x_rng, mu_x, logvar_x)
    qzCx_stats = (mu_z, logvar_z)
    pxCz_stats = (mu_x, logvar_x)
    return z_sample, x_sample, qzCx_stats, pxCz_stats


def iw_estimator(x, rng, enc_params, dec_params):

    z_sample, x_sample, qzCx_stats, pxCz_stats = generate_samples(rng, enc_params, dec_params, x)
    log_pxCz = diag_gaussian_logpdf(x, *pxCz_stats)
    log_qz = diag_gaussian_logpdf(z_sample, *qzCx_stats)
    log_pz = diag_gaussian_logpdf(z_sample)
    iw_log_summand = log_pxCz + log_pz - log_qz
    
    return iw_log_summand

def iwelbo_amortized(x, rng, enc_params, dec_params, num_samples=32, *args, **kwargs):

    rngs = random.split(rng, num_samples)
    vec_iw_estimator = vmap(iw_estimator, in_axes=(None, 0, None, None))
    iw_log_summand = vec_iw_estimator(x, rngs, enc_params, dec_params)

    assert num_samples == len(iw_log_summand)
    K = num_samples
    iwelbo_K = logsumexp(iw_log_summand) - jnp.log(K)
    return iwelbo_K


@partial(jit, static_argnums=(4, 5,))
def SUMO(x, rng, enc_params, dec_params, K, m=16, *args, **kwargs):
    # K = sampling_tail(rng)
    K_range = lax.iota(dtype=jnp.int32, size=K+m)+1
    rngs = random.split(rng, K+m)
    vec_iw_estimator = vmap(iw_estimator, in_axes=(None, 0, None, None))
    log_iw = vec_iw_estimator(x, rngs, enc_params, dec_params)
    iwelbo_K = logcumsumexp(log_iw) - jnp.log(K_range)

    vec_reverse_cdf = vmap(reverse_cdf, in_axes=(0,))
    inv_weights = jnp.divide(1., vec_reverse_cdf(K_range[m:]))
    return iwelbo_K[m-1] + jnp.sum(inv_weights * (iwelbo_K[m:] - iwelbo_K[m-1:-1]))

# ===== Objectives =====

def reverse_kl(log_prob, log_px_estimator, rng, enc_params, dec_params):
    """
    Single-sample Monte Carlo estimate of reverse-KL, used to 
    approximate the target density p*(x)

    D_KL(p(x) || p*(x)) = E_p[log p(x) - log p*(x)]
    """
    z_sample, x_sample, qzCx_stats, pxCz_stats = generate_samples(rng, enc_params, dec_params, x=None)
    log_px = log_px_estimator(x_sample, rng, enc_params, dec_params, K, args.min_terms)
    reverse_kl = log_px - log_prob(x_sample)

    return log_px, reverse_kl

def batch_reverse_kl(logprob, log_px_estimator, rng, enc_params, dec_params, num_samples):
    # Average over a batch of random samples.
    rngs = random.split(rng, int(num_samples))
    vectorized_rkl = vmap(partial(reverse_kl, logprob, log_px_estimator), in_axes=(0, None, None))
    log_px_batch, reverse_kl_batch = vectorized_rkl(rngs, enc_params, dec_params)
    return jnp.mean(reverse_kl_batch)


#@partial(jit, static_argnums=(3,4))
def objective(t, enc_params, dec_params, log_px_estimator, maximize=False, num_samples=32):
    rng = random.PRNGKey(t)
    reverse_kl_batch = batch_reverse_kl(funnel_log_density, log_px_estimator, rng, 
                                        enc_params, dec_params, num_samples)
    if maximize is True:
        return jnp.negative(reverse_kl_batch)
    return reverse_kl_batch

# ===== Update functions =====

@partial(jit, static_argnums=(3,4))
def update(t, dec_opt_state, enc_opt_state, model, batch_size=32):
    # Minimize objective w.r.t. decoder parameters
    enc_params = get_enc_params(enc_opt_state)
    dec_params = get_dec_params(dec_opt_state)
    gradient = jax.grad(objective, argnums=2)(t, enc_params, dec_params, model, False, batch_size)
    dec_opt_state = dec_opt_update(t, gradient, dec_opt_state)

    return get_dec_params(dec_opt_state), dec_opt_state

@partial(jit, static_argnums=(3))
def enc_update_iwelbo(t, dec_opt_state, enc_opt_state, batch_size=32):
    # Maximize objective w.r.t. encoder parameters to reduce bias
    enc_params = get_enc_params(enc_opt_state)
    dec_params = get_dec_params(dec_opt_state)
    enc_gradient = jax.grad(objective, argnums=1)(t, enc_params, dec_params, iwelbo_amortized, True, batch_size)
    enc_opt_state = enc_opt_update(t, enc_gradient, enc_opt_state)

    return get_enc_params(enc_opt_state), enc_opt_state 

@partial(jit, static_argnums=(3))
def enc_update_sumo(t, dec_opt_state, enc_opt_state, num_samples=32):
    # Minimize variance of SUMO w.r.t. encoder parameters
    rng = random.PRNGKey(t)
    rngs = random.split(rng, int(num_samples))
    enc_params = get_enc_params(enc_opt_state)
    dec_params = get_dec_params(dec_opt_state)
    
    def _sumo_sq(rngs, enc_params, dec_params):
        vectorized_rkl = vmap(partial(reverse_kl, funnel_log_density, SUMO), in_axes=(0,None, None))
        log_px_batch, _ = vectorized_rkl(rngs, enc_params, dec_params)
        return jnp.mean(jnp.square(log_px_batch))
        
    enc_gradient = jax.grad(_sumo_sq, argnums=1)(rngs, enc_params, dec_params)
    enc_opt_state = enc_opt_update(t, enc_gradient, enc_opt_state)

    return get_enc_params(enc_opt_state), enc_opt_state 

if __name__ == '__main__':

    description = "Density matching in Jax."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-z", "--latent_dim", type=int, default=8, help="Dimension of latent space.")
    parser.add_argument("-hd", "--hidden_dim", type=int, default=128, help="Number of hidden units in encoder/decoder.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("-m", "--min_terms", type=int, default=4, help="Minimum number of terms to evaluate in series.")
    parser.add_argument("-t", "--total_iterations", type=int, default=int(1e5), help="Number of training iterations.")
    parser.add_argument("-model", "--model", type=str, default='SUMO', choices=('SUMO','IWELBO'), help="Surrogate model.")
    args = parser.parse_args()

    # ===== Build models =====
    feature_dim = 2
    encoder_init, encoder_forward = encoder(hidden_dim=args.hidden_dim, z_dim=args.latent_dim)
    decoder_init, decoder_forward = decoder(hidden_dim=args.hidden_dim, x_dim=feature_dim)

    rng = random.PRNGKey(42)
    rng, dec_init_rng, enc_init_rng = random.split(rng, 3)
    _, init_encoder_params = encoder_init(enc_init_rng, (-1, feature_dim))
    output_shape, init_decoder_params = decoder_init(dec_init_rng, (-1, args.latent_dim))

    dec_opt_init, dec_opt_update, get_dec_params = optimizers.adam(step_size=5e-5)
    dec_opt_state = dec_opt_init(init_decoder_params)

    enc_opt_init, enc_opt_update, get_enc_params = optimizers.adam(step_size=5e-5)
    enc_opt_state = enc_opt_init(init_encoder_params)

    model = SUMO if args.model == 'SUMO' else iwelbo_amortized

    print('Optimizing variational parameters under {} model ...'.format(args.model))
    for t in range(int(args.total_iterations)):

        # Sample concrete value of K
        K = jit(sampling_tail)(random.PRNGKey(t))
        dec_params, dec_opt_state = update(t, dec_opt_state, enc_opt_state, model, args.batch_size)

        if args.model == 'SUMO':
            enc_params, enc_opt_state = enc_update_sumo(t, dec_opt_state, enc_opt_state, args.batch_size)
        elif args.model == 'IWELBO':
            enc_params, enc_opt_state = enc_update_iwelbo(t, dec_opt_state, enc_opt_state, args.batch_size)


        if t % 2000 == 0:
            callback(t, params=(enc_params, dec_params), objective=objective, model=model)
