---
title: Implementation of ROPE in Jax
tags: engineering, LLM
date: 2025-04-02 22:42:39
categories: Jax, Artificial Intelligence, Gsoc
author: codecason
---

Hello guys:

Since the form input space is limited, please allow me to add more content.

I'm currently a master student in Peking University, majoring in artificial intelligence. My current research field includes code generation in LLM.

I'm interested in Jax, cause it is different from Pytorch and Tensorflow(Although it is more like numpy currently).And I quickly learned the docs, and found the features like grad, jit, and vmap.But as I studied the docs, I found there was still some space for improvement for the coherence and user-friendly guide. Besides, many new LLMs need support, so I thought it was a good opportunity to contribute to the community.

Of course, this experience will enrich my resume and my future career as a reward.But I think the community could also benefit from my contribution and I could gain much more than that.

My skills are as follows:
- Basic Computer Knowledge
- Python 99%
- C++ 90%
- Pytorch 80% Tensorflow 20%
- Deep Learning 80%
- PaddlePaddle 70%
- CUDA 60%
- LLM 80%, I have learned a lot about the architecture of LLM.

And here is an example of implementing ROPE in Jax.
~~~python
import jax
import jax.numpy as jnp

def precompute_freqs_cis(dim: int, end: int, constant: float = 10000.0):
    '''
    calculate the value of cos and sin, where cos is real part, and sin is imaginary, like cosx + j*sinx
    :param dim: the last dim of q,k,v，generally emb_dim/head_num
    :param end: the length of sequence
    :param constant： 10000
    :return:
    complex calculation torch.polar(a, t) output， a*(cos(t)+j*sin(t))
    '''
    freqs = 1.0 / (constant ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))  # [d/2]
    t = jnp.arange(end)  # [length]

    freqs = jnp.outer(t, freqs)  # [length, d/2]
    # freqs format is [m*theta_0, m*theta_1, ..., m*theta_(d/2-1)]
    # freqs_cis: [cos(m*theta_0)+j*sin(m*theta_0),  ..., cos(m*theta_(d/2-1))+j*sin(m*theta_(d/2-1))]
    # torch.polar(a, t) return a*(\theta^(i*t))
    freqs_cis = jnp.exp(1j * freqs)  # Equivalent to cos + j*sin
    return freqs_cis

def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)  # [1, length, 1, d/2]

def apply_rotary_emb(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray):
    # change xq shape to [bs, length, head,  d/2, 2]
    # xq:[q0, q1, .., q(d-1)] (ignore dims before -1)
    # turns into xq_: [q0+j*q1, q2+j*q3, ..., q(d-2)+j*q(d-1)]

    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    xq_complex = xq_[..., 0] + 1j * xq_[..., 1]
    xk_complex = xk_[..., 0] + 1j * xk_[..., 1]
    
    # mainly brodcast the last but two dimension to head
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)  # [1, length, 1, d/2]

    # (q0+j*q1)(cos(m*theta_0)+j*sin(m*theta_0)) = q0*cos(m*theta_0)-q1*sin(m*theta_0) + j*(q1*cos(m*theta_0)+q0*sin(m*theta_0))
    # then throught torch.view_as_real function, the shape changes from [bs, length, head, d/2] to [bs, length, head, d/2, 2],
    # -> in which the last dim are real and imag parts.
    # and finally flatten the dimensions, the shape becomes [bs, length, head, d]
    # now the xq_out format is [real0, imag0, real1, imag1, ..., real(d/2-1), imag(d/2-1)]

    xq_out = (xq_complex * freqs_cis).view(jnp.float32).reshape(*xq.shape)
    xk_out = (xk_complex * freqs_cis).view(jnp.float32).reshape(*xk.shape)
    return xq_out, xk_out

if __name__ == '__main__':
    q = jnp.arange(24).reshape((1, 3, 2, 4))
    k = jnp.arange(24).reshape((1, 3, 2, 4))
    v = jnp.arange(24).reshape((1, 3, 2, 4))
    print(q.shape)

    freqs_cis = precompute_freqs_cis(dim=q.shape[-1], end=q.shape[1], constant=10000.0)
    # freqs_cis [seq_len, d / 2], m = (0, 1, ..., seq_len - 1) d/2 from theta_0 to theta_(d/2 - 1)
    # print(freqs_cis.detach().numpy())

    q_new, k_new = apply_rotary_emb(xq=q, xk=k, freqs_cis=freqs_cis)
    print(q_new, '\n', k_new)

~~~

Thanks for your reading.
