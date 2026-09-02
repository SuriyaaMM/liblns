import numpy as np

LNS8_I4F3_NINF = np.array(0xFF, dtype=np.uint8)
LNS8_I4F3_INF  = np.array(0x7F, dtype=np.uint8)
LNS8_I4F3_ZERO = np.array(0x00, dtype=np.uint8)
# 127 is reserved for INF & NINF
LNS8_I4F3_MAXE = np.array(126,  dtype=np.uint8)
LNS8_I4F3_BIAS = np.array(7,    dtype=np.uint8)
LNS8_I4F3_FBIT = np.array(3,    dtype=np.uint8)

LNS8_I4F3_LUT_ADD = np.round(
    np.log2(1 + 2 ** (-(np.arange(0, 128, dtype=np.float32) / 8.0))) * 8
).astype(np.int16)
LNS8_I4F3_LUT_SUB = np.round(
    np.log2(1 - 2 ** (-(np.arange(0, 128, dtype=np.float32) / 8.0))) * 8
).astype(np.int16)
# log2(0) = 0, piecewise function
LNS8_I4F3_LUT_SUB[0] = 0


def __lns8_i4f3_get_sign(
    x: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    # MSB is the sign bit
    return (x >> 7)

def __lns8_i4f3_get_exponent(
    x: np.ndarray[np.uint8]
) -> np.ndarray[np.uint8]:
    # mask with 0111 1111
    return (x & (0x7F))

def fp32_to_lns8_i4f3(
    x: np.ndarray[np.float32]
) -> np.ndarray[np.uint8]:

    # extract the sign bit for all numbers
    s: np.ndarray[np.uint8] = np.where(
        (x > 0), 
        0, 
        1
    ).astype(np.uint8)

    # unbiased exponent
    e_raw: np.ndarray[np.float32] = np.log2(np.abs(x))
    # clip the exponent between (0, MAXE)
    e: np.ndarray[np.uint8] = np.clip(
        # extract the exponent & shift up the fractional part
        a=np.round((e_raw + LNS8_I4F3_BIAS) * (1 << LNS8_I4F3_FBIT)),
        a_min=0,
        a_max=LNS8_I4F3_MAXE
    ).astype(np.uint8)
    # pack into 8-bit integer
    lns: np.ndarray[np.uint8] = (s << 7) | (e)

    # handle special values
    lns = np.where(
        (x == 0),
        LNS8_I4F3_ZERO,
        lns
    )
    lns = np.where(
        np.isposinf(x),
        LNS8_I4F3_INF,
        lns
    )
    lns = np.where(
        np.isneginf(x),
        LNS8_I4F3_NINF,
        lns
    )

    return lns

def lns8_i4f3_to_fp32(
    x: np.ndarray[np.uint8]
) -> np.ndarray[np.float32]:

    # extract the sign bit
    s: np.ndarray[np.float32] = __lns8_i4f3_get_sign(x).astype(np.float32)

    # extract the exponents
    e: np.ndarray[np.float32] = \
        (
            # extract the exponent bit 
            (x & 0x7F)  / 
            # fractional scale factor (2 ^ LNBS8_I4F3_FBIT)
            np.array(1 << LNS8_I4F3_FBIT, dtype=np.float32)

        # rebase
        ) - np.array(LNS8_I4F3_BIAS, dtype=np.float32)

    # reconstruct the fp32
    fp32: np.ndarray[np.float32] = ((-1) ** s) * (2 ** e)

    # handle special values
    fp32 = np.where(
        (x == LNS8_I4F3_ZERO),
        0.0,
        fp32
    )
    fp32 = np.where(
        (x == LNS8_I4F3_INF),
        np.inf,
        fp32
    )
    fp32 = np.where(
        (x == LNS8_I4F3_NINF),
        -np.inf,
        fp32
    )

    return fp32

def lns8_i4f3_mul(
    x: np.ndarray[np.uint8],
    y: np.ndarray[np.uint8],
):
    s_x: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(x)
    s_y: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(y)

    e_x: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(x).astype(np.int16)
    e_y: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(y).astype(np.int16)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in np.uint8
    e_xy = np.clip(
        ((e_x) + (e_y) - (LNS8_I4F3_BIAS << LNS8_I4F3_FBIT)), 
        0, 
        LNS8_I4F3_MAXE
    ).astype(np.uint8)

    result = (s_xy << 7) | e_xy
    # explicitly handle zero propagation
    result = np.where(
        (x == LNS8_I4F3_ZERO) | (y == LNS8_I4F3_ZERO),
        LNS8_I4F3_ZERO,
        result
    )

    return result

def lns8_i4f3_div(
    x: np.ndarray[np.uint8],
    y: np.ndarray[np.uint8],
):
    s_x: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(x)
    s_y: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(y)

    e_x: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(x).astype(np.int16)
    e_y: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(y).astype(np.int16)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in np.uint8
    e_xy = np.clip(
        ((e_x) - (e_y) + (LNS8_I4F3_BIAS << LNS8_I4F3_FBIT)), 
        0, 
        LNS8_I4F3_MAXE
    ).astype(np.uint8)

    result = (s_xy << 7) | e_xy

    # numerator = 0
    result = np.where(
        x == LNS8_I4F3_ZERO,
        LNS8_I4F3_ZERO,
        result
    ) 
    # denominator = 0
    result = np.where(
        y == LNS8_I4F3_ZERO,
        np.where(
            s_xy == 0,
            LNS8_I4F3_INF,
            LNS8_I4F3_NINF
        ),
        result
    )

    return result

def lns8_i4f3_add(
    x: np.ndarray[np.uint8],
    y: np.ndarray[np.uint8],
):
    s_x: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(x)
    s_y: np.ndarray[np.uint8] = __lns8_i4f3_get_sign(y)

    e_x: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(x).astype(np.int16)
    e_y: np.ndarray[np.int16] = __lns8_i4f3_get_exponent(y).astype(np.int16)

    s_xy = np.where(e_x >= e_y, s_x, s_y)

    d = np.abs(e_x - e_y)

    # lookuptable addition & subtraction
    e_xy = np.maximum(e_x, e_y) + np.where(
        s_x == s_y,
        LNS8_I4F3_LUT_ADD[d],
        LNS8_I4F3_LUT_SUB[d]
    )

    e_xy = np.clip(
        e_xy, 
        0, 
        LNS8_I4F3_MAXE
    ).astype(np.uint8)

    result = (s_xy << 7) | e_xy

    # exact cancellation X + (-X) = 0
    result = np.where(
        (s_x != s_y) & (d == 0),
        LNS8_I4F3_ZERO,
        result
    )

    # x == 0, then answer is simply y 
    result = np.where(
        x == LNS8_I4F3_ZERO,
        y, 
        result
    )
    # y == 0, then answer is simply x
    result = np.where(
        y == LNS8_I4F3_ZERO,
        x,
        result
    )
    return result

def lns8_i4f3_sub(
    x: np.ndarray[np.uint8],
    y: np.ndarray[np.uint8],
):
    y_neg = np.where(
        y == LNS8_I4F3_ZERO, 
        LNS8_I4F3_ZERO, 
        # sign bit mask = 0x80 = 1000 0000
        y ^ 0x80
    )

    return lns8_i4f3_add(x, y_neg)

if __name__ == "__main__":
    lns_x = fp32_to_lns8_i4f3(11.101)
    lns_y = fp32_to_lns8_i4f3(12.121)

    lnsaxy = lns8_i4f3_sub(lns_x, lns_y)

    print(f"mul = {lns8_i4f3_to_fp32(lnsaxy)}")