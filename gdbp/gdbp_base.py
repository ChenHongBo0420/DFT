from jax import numpy as jnp, random, jit, value_and_grad, nn
import flax
from commplax import util, comm, cxopt, op, optim
from commplax.module import core, layer
import numpy as np
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from typing import Any, Optional, Union
from . import data as gdat
import jax
from scipy import signal
from flax import linen as nn

Model = namedtuple('Model', 'module initvar overlaps name')
Array = Any
Dict = Union[dict, flax.core.FrozenDict]


# def make_base_module(steps: int = 3,
#                      dtaps: int = 261,
#                      ntaps: int = 41,
#                      rtaps: int = 61,
#                      init_fn: tuple = (core.delta, core.gauss),
#                      w0 = 0.,
#                      mode: str = 'train'):

#     _assert_taps(dtaps, ntaps, rtaps)

#     d_init, n_init = init_fn

#     if mode == 'train':
#         mimo_train = True
#     elif mode == 'test':
#         mimo_train = cxopt.piecewise_constant([200000], [True, False])
#     else:
#         raise ValueError('invalid mode %s' % mode)
 
#     base_layers = [
#         layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps, d_init=d_init, n_init=n_init),
#         layer.BatchPowerNorm(mode=mode),
#         layer.MIMOFOEAf(name='FOEAf', w0=w0, train=mimo_train, preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
#         layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
#         layer.MIMOAF(train=mimo_train)
#     ]
      
#     base = layer.Serial(*base_layers)
#     return base
 
class Decoder(nn.Module):
    base_layers: list

    def setup(self):
        self.base_module = nn.Sequential(self.base_layers)

    def __call__(self, z, x):
        combined = jnp.concatenate([x, z], axis=-1)
        return self.base_module(combined)
     
def reparameterize(key, mu, logvar):
    std = jnp.exp(logvar)
    eps = jax.random.normal(key, std.shape)
    return mu + eps * std

# class VAE(nn.Module):
#     steps: int = 3
#     dtaps: int = 261
#     ntaps: int = 41
#     rtaps: int = 61
#     init_fn: tuple = (core.delta, core.gauss)
#     w0: float = 0.0
#     mode: str = 'train'

#     def setup(self):
#         _assert_taps(self.dtaps, self.ntaps, self.rtaps)

#         d_init, n_init = self.init_fn

#         if self.mode == 'train':
#             self.mimo_train = True
#         elif self.mode == 'test':
#             self.mimo_train = cxopt.piecewise_constant([200000], [True, False])
#         else:
#             raise ValueError('invalid mode %s' % self.mode)

#         self.conv1d = layer.vmap(layer.Conv1d)(name='Conv1d', taps=self.rtaps)
#         self.conv1d1 = layer.vmap(layer.Conv1d)(name='Conv1d1', taps=self.rtaps)

#         self.base_layers = [
#             layer.FDBP(steps=self.steps, dtaps=self.dtaps, ntaps=self.ntaps, d_init=d_init, n_init=n_init),
#             layer.BatchPowerNorm(mode=self.mode),
#             layer.MIMOFOEAf(name='FOEAf', w0=self.w0, train=self.mimo_train, preslicer=core.conv1d_slicer(self.rtaps), foekwargs={}),
#             layer.vmap(layer.Conv1d)(name='RConv', taps=self.rtaps),
#             layer.MIMOAF(train=self.mimo_train)
#         ]

#         self.decoder = Decoder(base_layers=self.base_layers)

#     def __call__(self, x, key):
#         z_mean, z_logvar = self.conv1d(x), self.conv1d1(x)
#         z = reparameterize(key, z_mean, z_logvar)
#         reconstructed_x = self.decoder(z, x)
#         return reconstructed_x, z_mean, z_logvar

def make_base_module(steps: int = 3,
                     dtaps: int = 261,
                     ntaps: int = 41,
                     rtaps: int = 61,
                     init_fn: tuple = (core.delta, core.gauss),
                     w0 = 0.,
                     mode: str = 'train'):

    _assert_taps(dtaps, ntaps, rtaps)

    d_init, n_init = init_fn

    if mode == 'train':
        mimo_train = True
    elif mode == 'test':
        mimo_train = cxopt.piecewise_constant([200000], [True, False])
    else:
        raise ValueError('invalid mode %s' % mode)
    
    base_layers = [
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps, d_init=d_init, n_init=n_init),
        layer.BatchPowerNorm(mode=mode),
        layer.MIMOFOEAf(name='FOEAf', w0=w0, train=mimo_train, preslicer=core.conv1d_slicer(rtaps), foekwargs={}),
        layer.vmap(layer.Conv1d)(name='RConv', taps=rtaps),
        layer.MIMOAF(train=mimo_train)
    ]
    base_layers1 = [
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps, d_init=d_init, n_init=n_init),
    ]
    base_layers2 = [
        layer.FDBP(steps=steps, dtaps=dtaps, ntaps=ntaps, d_init=d_init, n_init=n_init),
    ]
    base = layer.Serial(*base_layers)
    base1 = layer.Serial(*base_layers1)
    base2 = layer.Serial(*base_layers2)
    return base, base1, base2

def _assert_taps(dtaps, ntaps, rtaps, sps=2):
    ''' we force odd taps to ease coding '''
    assert dtaps % sps, f'dtaps must be odd number, got {dtaps} instead'
    assert ntaps % sps, f'ntaps must be odd number, got {ntaps} instead'
    assert rtaps % sps, f'rtaps must be odd number, got {rtaps} instead'


def fdbp_init(a: dict,
              xi: float = 1.1,
              steps: Optional[int] = None):
    '''
        initializer for the base module

        Args:
            xi: NLC scaling factor
            steps: GDBP steps, used to calculate the theoretical profiles of D- and N-filters

        Returns:
            a pair of functions to initialize D- and N-filters
    '''

    def d_init(key, shape, dtype=jnp.complex64):
        dtaps = shape[0]
        d0, _ = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # rescale as input power which has been norm to 2 in dataloader
            virtual_spans=steps)
        return d0[0, :, 0]

    def n_init(key, shape, dtype=jnp.float32):
        dtaps = shape[0]
        _, n0 = comm.dbp_params(
            a['samplerate'],
            a['distance'] / a['spans'],
            a['spans'],
            dtaps,
            a['lpdbm'] - 3,  # rescale
            virtual_spans=steps)

        return xi * n0[0, 0, 0] * core.gauss(key, shape, dtype)

    return d_init, n_init


# def model_init(data: gdat.Input,
#                base_conf: dict,
#                sparams_flatkeys: list,
#                n_symbols: int = 4000,
#                sps : int = 2,
#                name='Model'):
#     ''' initialize model from base template, generating CDC, DBP, EDBP, FDBP, GDBP
#     depending on given N-filter length and trainable parameters

#     Args:
#         data:
#         base_conf: a dict of kwargs to make base module, see `make_base_module`
#         sparams_flatkeys: a list of keys contains the static(nontrainable) parameters.
#             For example, assume base module has parameters represented as nested dict
#             {'color': 'red', 'size': {'width': 1, 'height': 2}}, its flatten layout is dict
#              {('color',): 'red', ('size', 'width',): 1, ('size', 'height'): 2}, a sparams_flatkeys
#              of [('color',): ('size', 'width',)] means 'color' and 'size/width' parameters are static.
#             regexp key is supportted.
#         n_symbols: number of symbols used to initialize model, use the minimal value greater than channel
#             memory
#         sps: sample per symbol. Only integer sps is supported now.

#     Returns:
#         a initialized model wrapped by a namedtuple
#     '''
    
#     mod = make_base_module(**base_conf, w0=data.w0)
#     y0 = data.y[:n_symbols * sps]
#     rng0 = random.PRNGKey(0)
#     z0, v0 = mod.init(rng0, core.Signal(y0))
#     ol = z0.t.start - z0.t.stop
#     sparams, params = util.dict_split(v0['params'], sparams_flatkeys)
#     state = v0['af_state']
#     aux = v0['aux_inputs']
#     const = v0['const']
#     return Model(mod, (params, state, aux, const, sparams), ol, name)

def model_init(data: gdat.Input,
               base_conf: dict,
               sparams_flatkeys: list,
               n_symbols: int = 4000,
               sps : int = 2,
               name='Model'):
    
    mod, mod1, mod2 = make_base_module(**base_conf, w0=data.w0)
    print(mod1)
    y0 = data.y[:n_symbols * sps]
    rng0 = random.PRNGKey(0)
    z = reparameterize(rng0, mod1, mod2)
    mod = jnp.concatenate([mod, z], axis=-1)
    z0, v0 = mod.init(rng0, core.Signal(y0))
    ol = z0.t.start - z0.t.stop
    sparams, params = util.dict_split(v0['params'], sparams_flatkeys)
    state = v0['af_state']
    aux = v0['aux_inputs']
    const = v0['const']
    return Model(mod, (params, state, aux, const, sparams), ol, name)


def l2_normalize(x, axis=None, epsilon=1e-12):
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jnp.sqrt(jnp.maximum(square_sum, epsilon))
    return x / x_inv_norm

def energy(x):
    return jnp.sum(jnp.square(x))
  
def si_snr(target, estimate, eps=1e-8):
    target_energy = energy(target)
    dot_product = jnp.sum(target * estimate)
    s_target = dot_product / (target_energy + eps) * target
    e_noise = estimate - s_target
    target_energy = energy(s_target)
    noise_energy = energy(e_noise)
    si_snr_value = 10 * jnp.log10((target_energy + eps) / (noise_energy + eps))
    return -si_snr_value  


def loss_fn(module: layer.Layer,
            params: Dict,
            state: Dict,
            y: Array,
            x: Array,
            aux: Dict,
            const: Dict,
            sparams: Dict,):
    params = util.dict_merge(params, sparams)
    
    z_original, updated_state = module.apply(
        {'params': params, 'aux_inputs': aux, 'const': const, **state}, core.Signal(y))
      

    aligned_x = x[z_original.t.start:z_original.t.stop]
    # mse_loss = jnp.mean(jnp.abs(z_original.val - aligned_x) ** 2)
    snr = si_snr(jnp.abs(z_original.val), jnp.abs(aligned_x))
    total_loss = snr
    return total_loss, updated_state

@partial(jit, backend='cpu', static_argnums=(0, 1))
def update_step(module: layer.Layer,
                opt: cxopt.Optimizer,
                i: int,
                opt_state: tuple,
                module_state: Dict,
                y: Array,
                x: Array,
                aux: Dict,
                const: Dict,
                sparams: Dict):
    ''' single backprop step

        Args:
            model: model returned by `model_init`
            opt: optimizer
            i: iteration counter
            opt_state: optimizer state
            module_state: module state
            y: transmitted waveforms
            x: aligned sent symbols
            aux: auxiliary input
            const: contants (internal info generated by model)
            sparams: static parameters

        Return:
            loss, updated module state
    '''

    params = opt.params_fn(opt_state)
    (loss, module_state), grads = value_and_grad(
        loss_fn, argnums=1, has_aux=True)(module, params, module_state, y, x,
                                          aux, const, sparams)
    opt_state = opt.update_fn(i, grads, opt_state)
    return loss, opt_state, module_state


def get_train_batch(ds: gdat.Input,
                    batchsize: int,
                    overlaps: int,
                    sps: int = 2):
    ''' generate overlapped batch input for training

        Args:
            ds: dataset
            batchsize: batch size in symbol unit
            overlaps: overlaps in symbol unit
            sps: samples per symbol

        Returns:
            number of symbols,
            zipped batched triplet input: (recv, sent, fomul)
    '''

    flen = batchsize + overlaps
    fstep = batchsize
    ds_y = op.frame_gen(ds.y, flen * sps, fstep * sps)
    ds_x = op.frame_gen(ds.x, flen, fstep)
    n_batches = op.frame_shape(ds.x.shape, flen, fstep)[0]
    return n_batches, zip(ds_y, ds_x)


def train(model: Model,
          data: gdat.Input,
          batch_size: int = 500,
          n_iter = None,
          opt: optim.Optimizer = optim.sgd(optim.piecewise_constant([500, 1000], [1e-4, 1e-5, 1e-6]))):
    ''' training process (1 epoch)

        Args:
            model: Model namedtuple return by `model_init`
            data: dataset
            batch_size: batch size
            opt: optimizer

        Returns:
            yield loss, trained parameters, module state
    '''

    params, module_state, aux, const, sparams = model.initvar
    opt_state = opt.init_fn(params)

    n_batch, batch_gen = get_train_batch(data, batch_size, model.overlaps)
    n_iter = n_batch if n_iter is None else min(n_iter, n_batch)

    for i, (y, x) in tqdm(enumerate(batch_gen),
                             total=n_iter, desc='training', leave=False):
        if i >= n_iter: break
        aux = core.dict_replace(aux, {'truth': x})
        loss, opt_state, module_state = update_step(model.module, opt, i, opt_state,
                                                   module_state, y, x, aux,
                                                   const, sparams)
        yield loss, opt.params_fn(opt_state), module_state


def test(model: Model,
         params: Dict,
         data: gdat.Input,
         eval_range: tuple=(300000, -20000),
         metric_fn=comm.qamqot):
    ''' testing, a simple forward pass

        Args:
            model: Model namedtuple return by `model_init`
        data: dataset
        eval_range: interval which QoT is evaluated in, assure proper eval of steady-state performance
        metric_fn: matric function, comm.snrstat for global & local SNR performance, comm.qamqot for
            BER, Q, SER and more metrics.

        Returns:
            evaluated matrics and equalized symbols
    '''

    state, aux, const, sparams = model.initvar[1:]
    aux = core.dict_replace(aux, {'truth': data.x})
    if params is None:
      params = model.initvar[0]

    z, _ = jit(model.module.apply,
               backend='cpu')({
                   'params': util.dict_merge(params, sparams),
                   'aux_inputs': aux,
                   'const': const,
                   **state
               }, core.Signal(data.y))
    metric = metric_fn(z.val,
                       data.x[z.t.start:z.t.stop],
                       scale=np.sqrt(10),
                       eval_range=eval_range)
    return metric, z
