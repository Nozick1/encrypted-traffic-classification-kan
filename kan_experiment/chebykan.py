"""Chebyshev-KAN layer (Kolmogorov-Arnold Network using Chebyshev polynomials).

Genuine KAN variant: learnable univariate activation functions placed on the
*edges* (each edge i->o is a degree-`degree` Chebyshev polynomial); there are no
fixed linear weights on the edges.

Numerical note: the original ChebyKAN repo evaluates T_k via cos(k * arccos(x)).
That form is unstable when tanh saturates to exactly +/-1 (arccos' -> 0/0 -> NaN
gradients). Here we evaluate the Chebyshev polynomials with the stable recurrence
    T_0(x)=1,  T_1(x)=x,  T_k(x)=2x*T_{k-1}(x)-T_{k-2}(x),
which is well-behaved on the whole interval [-1, 1].
"""

import torch
import torch.nn as nn


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # One Chebyshev coefficient vector per (input, output) edge.
        self.cheby_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Chebyshev polynomials are defined on [-1, 1]; squash inputs into range.
        x = torch.tanh(x)                       # (B, inputdim)
        # Stable recurrence for T_0..T_degree.
        Ts = [torch.ones_like(x), x]            # T0 = 1, T1 = x
        for _ in range(2, self.degree + 1):
            Ts.append(2 * x * Ts[-1] - Ts[-2])
        T = torch.stack(Ts, dim=-1)             # (B, inputdim, degree+1)
        # Sum over input dimension with the learnable coefficients.
        y = torch.einsum("bid,iod->bo", T, self.cheby_coeffs)
        return y.view(-1, self.outdim)


class ChebyKAN_Head(nn.Module):
    """A small multi-layer Chebyshev-KAN classifier head."""

    def __init__(self, in_dim, hidden, out_dim, degree=3):
        super().__init__()
        # LayerNorm before KAN keeps inputs away from tanh saturation.
        self.ln1 = nn.LayerNorm(in_dim)
        self.kan1 = ChebyKANLayer(in_dim, hidden, degree)
        self.ln2 = nn.LayerNorm(hidden)
        self.kan2 = ChebyKANLayer(hidden, out_dim, degree)

    def forward(self, x):
        x = self.ln1(x)
        x = self.kan1(x)
        x = self.ln2(x)
        x = self.kan2(x)
        return x
