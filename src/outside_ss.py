# after calculating Q^b, Q^m and Q, we can obtain W.
# finally we can calculate the pseudo basepairing probability P^*_{i,j} = E[Q^b_{i, j}] * E[W_{i, j}] / E[Z_{1, N}]

import unittest
import jax
import jax.numpy as jnp
import numpy as onp
from .ss import get_ss_partition_fn
from . import energy
import pandas as pd

from . import vienna
from . import vienna_rna
import RNA
from .checkpoint import checkpoint_scan



def get_outside_ss_pf(em, n, Zss=None, E=None, Qm=None, Qb=None):
  pf = lambda pseq: jnp.zeros((n, n))
  return pf




class TestOutsidePartitionFunction(unittest.TestCase):
  def _random_onehot_seq(self, n):
    p_seq = onp.empty((n, 4), dtype=onp.float64)
    for i in range(n):
        p_seq[i] = onp.random.random_sample(4)
        print(p_seq[i])
        # p_seq[i] /= onp.sum(p_seq[i])
    return p_seq
  def _random_p_seq(self, n):
      p_seq = onp.empty((n, 4), dtype=onp.float64)
      for i in range(n):
          p_seq[i] = onp.random.random_sample(4)
          p_seq[i] /= onp.sum(p_seq[i])
      return p_seq


  def _bpp_vienna(self, seq_onehot):
    seq = ""
    for i in range(len(seq_onehot)):
      # A, U, G, C
      base_index = jnp.argmax(seq_onehot[i])
      seq += list("ACGU")[base_index]
    fc = RNA.fold_compound(seq)
    (propensity, ensemble_energy) = fc.pf()
    basepair_probs = fc.bpp()
    # 0行目、0 列目は削除する。padding してるので
    # basepair_probs = basepair_probs
    return pd.DataFrame(basepair_probs).values[1:, 1:]
  
  def _bpp_jax(self, p_seq, em):
    n = len(p_seq)
    ss_partition_fn = get_ss_partition_fn(em, n)
    Zss, E, Qm, Qb = ss_partition_fn(p_seq)
    outside_partition_fn = get_outside_ss_pf(em, n, Zss, E, Qm, Qb)
    W = outside_partition_fn(p_seq)
    print("Zss is ", Zss)
    # print("E is ", E)
    # print("Qm is ", Qm)
    # print("Qb is ", Qb)
    # print("W is \n", pd.DataFrame(W))
    bpp_jax = jnp.zeros((n, n))
    # bpp_jax = W * P / Zss
    return bpp_jax
    

  def test_outside_partition_function(self): 
    print("test outside partition function")
    n = 10
    p_seq_onehot = self._random_p_seq(n)
    em = energy.JaxNNModel()
    outside_ss_fn = get_outside_ss_pf(em, n)
    W = outside_ss_fn(p_seq_onehot)
    print(W)
    self.assertAlmostEqual(W, 1.0)

  def test_bpp_vienna(self):
    print("test bpp vienna")
    n = 10
    em = energy.JaxNNModel()
    p_seq_onehot = self._random_onehot_seq(n)
    bpp_vienna = self._bpp_vienna(p_seq_onehot)
    bpp_jax = self._bpp_jax(p_seq_onehot, em)
    bpp_jax_ = pd.DataFrame(bpp_jax).values
    print("bpp vienna")
    print(bpp_vienna)
    print("bpp jax")
    print(bpp_jax)
    # self.assertAlmostEqual(bpp_vienna, bpp_jax, places=3) <-- .all を使う
    self.assertTrue(onp.allclose(bpp_vienna, bpp_jax_, atol=1e-3))




