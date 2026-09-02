import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from liblns.lns16_i5f10 import *

def test_fp32_mul(
    a: np.ndarray,
    b: np.ndarray,
):
   y_fp32 = a * b

   a_lns = fp32_to_lns16_i5f10(a)
   b_lns = fp32_to_lns16_i5f10(b)

   y_lns = lns16_i5f10_to_fp32(lns16_i5f10_mul(a_lns, b_lns))

   eps = np.finfo(np.float32).eps
   rel_error = np.abs(y_lns - y_fp32) / np.maximum(np.abs(y_fp32), eps)

   print(f"relative error mean   : {rel_error.mean()}")
   print(f"relative error std    : {rel_error.std()}")
   print(f"relative error median : {np.median(rel_error)}")

if __name__ == "__main__":
    rng = np.random.default_rng()

    a = rng.normal(-32, 32, 1024 * 1024)
    b = rng.normal(-32, 32, 1024 * 1024)

    test_fp32_mul(a, b)