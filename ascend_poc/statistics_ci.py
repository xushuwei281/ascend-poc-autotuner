"""Small standard-library numerical primitives for the optional block CI.

Student's t survival probabilities use the regularized incomplete beta
function; quantiles invert the survival probability directly, avoiding loss of
precision in ``1 - cdf`` in the upper tail. Nonconvergence fails closed.
"""
from __future__ import annotations

import math


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction with modified Lentz."""
    tiny = 1e-300
    denominator = 1.0 - (a + b) * x / (a + 1.0)
    if abs(denominator) < tiny:
        denominator = tiny
    denominator = 1.0 / denominator
    numerator = 1.0
    fraction = denominator
    for step in range(1, 1001):
        twice = 2.0 * step
        coefficients = (
            step * (b - step) * x / ((a + twice - 1.0) * (a + twice)),
            -(a + step) * (a + b + step) * x / ((a + twice) * (a + twice + 1.0)),
        )
        last_change = 1.0
        for coefficient in coefficients:
            denominator = 1.0 + coefficient * denominator
            numerator = 1.0 + coefficient / numerator
            if abs(denominator) < tiny:
                denominator = tiny
            if abs(numerator) < tiny:
                numerator = tiny
            denominator = 1.0 / denominator
            last_change = denominator * numerator
            fraction *= last_change
        if abs(last_change - 1.0) <= 2e-14:
            if not math.isfinite(fraction) or fraction <= 0:
                raise ValueError('Incomplete beta calculation was not finite and positive')
            return fraction
    raise ValueError('Incomplete beta calculation did not converge')


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    scale = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        value = scale * _beta_fraction(a, b, x) / a
    else:
        value = 1.0 - scale * _beta_fraction(b, a, 1.0 - x) / b
    if not math.isfinite(value) or not -1e-13 <= value <= 1.0 + 1e-13:
        raise ValueError('Incomplete beta probability is invalid')
    return min(1.0, max(0.0, value))


def student_t_survival(value: float, degrees_of_freedom: int) -> float:
    """P(T >= value) for a positive Student t value, with integer df >= 1."""
    if (type(degrees_of_freedom) is not int or degrees_of_freedom < 1
            or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError('Student t requires a finite nonnegative value and positive integer df')
    if value == 0:
        return 0.5
    df = float(degrees_of_freedom)
    scaled = value / math.sqrt(df)
    x = 0.0 if scaled > 1e154 else 1.0 / (1.0 + scaled * scaled)
    return 0.5 * _regularized_beta(x, df / 2.0, 0.5)


def student_t_critical(alpha: float, degrees_of_freedom: int) -> float:
    """Positive one-sided critical value: P(T >= critical) == alpha.

    This is a t quantile with the actual block-level degrees of freedom, not a
    fixed normal 1.96 multiplier. The caller owns the sampling assumptions and
    any allocation of alpha across scenarios or adaptive comparisons.
    """
    if (type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < .5
            or type(degrees_of_freedom) is not int or degrees_of_freedom < 1):
        raise ValueError('Student t critical value requires 0 < alpha < .5 and integer df >= 1')
    low, high = 0.0, 1.0
    for _ in range(1024):
        if student_t_survival(high, degrees_of_freedom) <= alpha:
            break
        high *= 2.0
        if not math.isfinite(high):
            raise ValueError('Student t critical value exceeded finite range')
    else:
        raise ValueError('Could not bracket the Student t critical value')
    for _ in range(160):
        middle = low / 2.0 + high / 2.0
        if student_t_survival(middle, degrees_of_freedom) > alpha:
            low = middle
        else:
            high = middle
        if high - low <= 2e-13 * max(1.0, high):
            return high  # Conservative upper end of the numerical bracket.
    raise ValueError('Student t critical value did not converge')
