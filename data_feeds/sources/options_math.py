"""
Black–Scholes utilities with dividend yield q, prices, Greeks, and implied volatility solver.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float("nan")
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (
        sigma * math.sqrt(T)
    )


def bs_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    d1 = bs_d1(S, K, r, q, sigma, T)
    if math.isnan(d1):
        return float("nan")
    return d1 - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    put_call: str = "call",
) -> float:
    if T <= 0:
        return max(0.0, S - K) if put_call == "call" else max(0.0, K - S)
    if sigma <= 0:
        f = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(0.0, f) if put_call == "call" else max(0.0, -f)
    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    if put_call == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(
            d2
        )
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    put_call: str = "call",
) -> Dict[str, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)
    if put_call == "call":
        delta = math.exp(-q * T) * _norm_cdf(d1)
    else:
        delta = -math.exp(-q * T) * _norm_cdf(-d1)
    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * pdf_d1 * math.sqrt(T)
    if put_call == "call":
        theta = (
            -S * math.exp(-q * T) * pdf_d1 * sigma / (2.0 * math.sqrt(T))
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
            + q * S * math.exp(-q * T) * _norm_cdf(d1)
        )
    else:
        theta = (
            -S * math.exp(-q * T) * pdf_d1 * sigma / (2.0 * math.sqrt(T))
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
            - q * S * math.exp(-q * T) * _norm_cdf(-d1)
        )
    if put_call == "call":
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    else:
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega / 100.0),
        "theta": float(theta / 365.0),
        "rho": float(rho / 100.0),
        "contract_multiplier": 100.0,
    }


def _iv_objective(
    sigma: float,
    target: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    put_call: str,
) -> float:
    return bs_price(S, K, r, q, sigma, T, put_call) - target


def implied_vol(
    price_mkt: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    put_call: str = "call",
    bracket: Tuple[float, float] = (1e-6, 5.0),
    tol: float = 1e-6,
    maxiter: int = 100,
) -> Optional[float]:
    if price_mkt is None or price_mkt < 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if put_call == "call":
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r
    if price_mkt < lower - 1e-12 or price_mkt > upper + 1e-12:
        return None
    a, b = bracket
    fa = _iv_objective(a, price_mkt, S, K, r, q, T, put_call)
    fb = _iv_objective(b, price_mkt, S, K, r, q, T, put_call)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb < 0:
        left, right = a, b
        f_left, f_right = fa, fb
        c, f_c = left, f_left
        d = e = right - left
        for _ in range(maxiter):
            if f_right * f_c > 0:
                c, f_c = left, f_left
                d = e = right - left
            if abs(f_c) < abs(f_right):
                left, right, c = right, c, right
                f_left, f_right, f_c = f_right, f_c, f_right
            tol1 = 2.0 * tol * max(1.0, abs(right)) + 0.5 * tol
            xm = 0.5 * (c - right)
            if abs(xm) <= tol1 or f_right == 0.0:
                return max(1e-12, float(right))
            if abs(e) >= tol1 and abs(f_left) > abs(f_right):
                s = f_right / f_left
                if left == c:
                    p = 2.0 * xm * s
                    qn = 1.0 - s
                else:
                    qn = f_left / f_c
                    r3 = f_right / f_c
                    p = s * (2.0 * xm * qn * (qn - r3) - (right - left) * (r3 - 1.0))
                    qn = (qn - 1.0) * (r3 - 1.0) * (s - 1.0)
                if p > 0:
                    qn = -qn
                p = abs(p)
                min1 = 3.0 * xm * qn - abs(tol1 * qn)
                min2 = abs(e * qn)
                if 2.0 * p < min(min1, min2):
                    e = d
                    d = p / qn
                else:
                    d = xm
                    e = d
            else:
                d = xm
                e = d
            left, f_left = right, f_right
            right = right + (d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1))
            f_right = _iv_objective(right, price_mkt, S, K, r, q, T, put_call)
        return max(1e-12, float(right))
    # Expand bracket then bisection
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(10):
        if flo * fhi < 0:
            break
        hi = min(hi * 1.5, 10.0)
        fhi = _iv_objective(hi, price_mkt, S, K, r, q, T, put_call)
    if flo * fhi > 0:
        return None
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = _iv_objective(mid, price_mkt, S, K, r, q, T, put_call)
        if abs(fmid) < tol or (hi - lo) < tol:
            return max(1e-12, float(mid))
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return max(1e-12, float(0.5 * (lo + hi)))
