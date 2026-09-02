import numpy as np

LNS16_I5F10_NINF = np.array(0xFFFF, dtype=np.uint16)
LNS16_I5F10_INF  = np.array(0x7FFF, dtype=np.uint16)
LNS16_I5F10_ZERO = np.array(0x0000, dtype=np.uint16)
# 127 is reserved for INF & NINF
LNS16_I5F10_MAXE = np.array(32766,  dtype=np.uint16)
LNS16_I5F10_BIAS = np.array(15,    dtype=np.uint16)
LNS16_I5F10_FBIT = np.array(10,    dtype=np.uint16)

LNS16_I5F10_LUT_ADD = np.round(
    np.log2(1 + 2 ** (-(np.arange(0, 32768, dtype=np.float32) / 1024.0))) * 1024.0
).astype(np.int32)
LNS16_I5F10_LUT_SUB = np.round(
    np.log2(1 - 2 ** (-(np.arange(0, 32768, dtype=np.float32) / 1024.0))) * 1024.0
).astype(np.int32)
# log2(0) = 0, piecewise function
LNS16_I5F10_LUT_SUB[0] = 0

def __lns16_i5f10_get_sign(
    x: np.ndarray[np.uint16]
) -> np.ndarray[np.uint16]:
    # MSB is the sign bit
    return (x >> 15)

def __lns16_i5f10_get_exponent(
    x: np.ndarray[np.uint16]
) -> np.ndarray[np.uint16]:
    # mask with 0111 1111 1111 1111
    return (x & (0x7FFF))

def fp32_to_lns16_i5f10(
    x: np.ndarray[np.float32]
) -> np.ndarray[np.uint16]:

    # extract the sign bit for all numbers
    s: np.ndarray[np.uint16] = np.where(
        (x > 0), 
        0, 
        1
    ).astype(np.uint16)

    # unbiased exponent
    e_raw: np.ndarray[np.float32] = np.log2(np.abs(x))
    # clip the exponent between (0, MAXE)
    e: np.ndarray[np.uint16] = np.clip(
        # extract the exponent & shift up the fractional part
        np.round((e_raw + LNS16_I5F10_BIAS) * (1 << LNS16_I5F10_FBIT)),
        0,
        LNS16_I5F10_MAXE
    ).astype(np.uint16)
    # pack into 16-bit integer
    lns: np.ndarray[np.uint16] = (s << 15) | (e)

    # handle special values
    lns = np.where(
        (x == 0),
        LNS16_I5F10_ZERO,
        lns
    )
    lns = np.where(
        np.isposinf(x),
        LNS16_I5F10_INF,
        lns
    )
    lns = np.where(
        np.isneginf(x),
        LNS16_I5F10_NINF,
        lns
    )

    return lns

def lns16_i5f10_to_fp32(
    x: np.ndarray[np.uint16]
) -> np.ndarray[np.float32]:

    # extract the sign bit
    s: np.ndarray[np.float32] = __lns16_i5f10_get_sign(x).astype(np.float32)

    # extract the exponents
    e: np.ndarray[np.float32] = \
        (
            # extract the exponent bit 
            (x & 0x7FFF)  / 
            # fractional scale factor (2 ^ LNS16_I5F10_FBIT)
            np.array(1 << LNS16_I5F10_FBIT, dtype=np.float32)

        # rebase
        ) - np.array(LNS16_I5F10_BIAS, dtype=np.float32)

    # reconstruct the fp32
    fp32: np.ndarray[np.float32] = ((-1) ** s) * (2 ** e)

    # handle special values
    fp32 = np.where(
        (x == LNS16_I5F10_ZERO),
        0.0,
        fp32
    )
    fp32 = np.where(
        (x == LNS16_I5F10_INF),
        np.inf,
        fp32
    )
    fp32 = np.where(
        (x == LNS16_I5F10_NINF),
        -np.inf,
        fp32
    )

    return fp32

def lns16_i5f10_mul(
    x: np.ndarray[np.uint16],
    y: np.ndarray[np.uint16],
):
    s_x: np.ndarray[np.uint16] = __lns16_i5f10_get_sign(x)
    s_y: np.ndarray[np.uint16] = __lns16_i5f10_get_sign(y)

    e_x: np.ndarray[np.int32] = __lns16_i5f10_get_exponent(x).astype(np.int32)
    e_y: np.ndarray[np.int32] = __lns16_i5f10_get_exponent(y).astype(np.int32)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in np.uint16
    e_xy = np.clip(
        ((e_x) + (e_y) - (LNS16_I5F10_BIAS << LNS16_I5F10_FBIT)), 
        0, 
        LNS16_I5F10_MAXE
    ).astype(np.uint16)

    result = (s_xy << 15) | e_xy
    # explicitly handle zero propagation
    result = np.where(
        (x == LNS16_I5F10_ZERO) | (y == LNS16_I5F10_ZERO),
        LNS16_I5F10_ZERO,
        result
    )

    return result

def lns16_i5f10_div(
    x: np.ndarray[np.uint16],
    y: np.ndarray[np.uint16],
):
    s_x: np.ndarray[np.uint16] = __lns16_i5f10_get_sign(x)
    s_y: np.ndarray[np.uint16] = __lns16_i5f10_get_sign(y)

    e_x: np.ndarray[np.int32] = __lns16_i5f10_get_exponent(x).astype(np.int32)
    e_y: np.ndarray[np.int32] = __lns16_i5f10_get_exponent(y).astype(np.int32)

    s_xy = (s_x) ^ (s_y)
    # this operation doesn't fit entirely in np.uint16
    e_xy = np.clip(
        ((e_x) - (e_y) + (LNS16_I5F10_BIAS << LNS16_I5F10_FBIT)), 
        0, 
        LNS16_I5F10_MAXE
    ).astype(np.uint16)

    result = (s_xy << 15) | e_xy
    # numerator = 0
    result = np.where(
        x == LNS16_I5F10_ZERO,
        LNS16_I5F10_ZERO,
        result
    ) 
    # denominator = 0
    result = np.where(
        y == LNS16_I5F10_ZERO,
        np.where(
            s_xy == 0,
            LNS16_I5F10_INF,
            LNS16_I5F10_NINF
        ),
        result
    )

    return result

def lns16_i5f10_add(
    x: np.ndarray[np.uint16],
    y: np.ndarray[np.uint16],
):
    s_x: np.ndarray[np.uint16] = __lns16_i5f10_get_sign(x)
    s_y: np.ndarray[np.uint16] = __lns16_i5f10_get_sign(y)

    e_x: np.ndarray[np.int32] = __lns16_i5f10_get_exponent(x).astype(np.int32)
    e_y: np.ndarray[np.int32] = __lns16_i5f10_get_exponent(y).astype(np.int32)

    # determine the sign bit
    s_xy = np.where(e_x >= e_y, s_x, s_y)
    
    d = np.abs(e_x - e_y)

    # lookuptable addition & subtraction
    e_xy = np.maximum(e_x, e_y) + np.where(
        s_x == s_y,
        LNS16_I5F10_LUT_ADD[d],
        LNS16_I5F10_LUT_SUB[d]
    )

    e_xy = np.clip(
        e_xy, 
        0, 
        LNS16_I5F10_MAXE
    ).astype(np.uint16)

    result = (s_xy << 15) | e_xy

    # exact cancellation X + (-X) = 0
    result = np.where(
        (s_x != s_y) & (d == 0),
        LNS16_I5F10_ZERO,
        result
    )

    # x == 0, then answer is simply y 
    result = np.where(
        x == LNS16_I5F10_ZERO,
        y, 
        result
    )
    # y == 0, then answer is simply x
    result = np.where(
        y == LNS16_I5F10_ZERO,
        x,
        result
    )
    return result

def lns16_i5f10_sub(
    x: np.ndarray[np.uint16],
    y: np.ndarray[np.uint16],
):
    y_neg = np.where(
        y == LNS16_I5F10_ZERO, 
        LNS16_I5F10_ZERO, 
        # sign bit mask = 0x80 = 1000 0000 0000 0000
        y ^ 0x8000
    )

    return lns16_i5f10_add(x, y_neg)

if __name__ == "__main__":
    lns_x = fp32_to_lns16_i5f10(11.101)
    lns_y = fp32_to_lns16_i5f10(12.121)

    lnsaxy = lns16_i5f10_sub(lns_x, lns_y)
    lnaaxy = lns16_i5f10_add(lns_x, lns_y)
    lnmaxy = lns16_i5f10_mul(lns_x, lns_y)
    lndaxy = lns16_i5f10_div(lns_x, lns_y)

    print(f"sub = {lns16_i5f10_to_fp32(lnsaxy)}")
    print(f"add = {lns16_i5f10_to_fp32(lnaaxy)}")
    print(f"mul = {lns16_i5f10_to_fp32(lnmaxy)}")
    print(f"div = {lns16_i5f10_to_fp32(lndaxy)}")