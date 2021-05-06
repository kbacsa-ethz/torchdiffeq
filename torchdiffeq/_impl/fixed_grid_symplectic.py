# Implemention of symplectic integrator.
# Whole input as (q,p) and  the shape is (bath_size, 2N)
# where N is the number of paritcles 

import torch
from .solvers import FixedGridODESolver

# symplectic integrators constatn 
_c1 = 1.0/(4.0 - pow(2.0,4.0/3.0))
_c2 = (1.0 - pow(2.0,1.0/3.0))/(4.0 - pow(2.0,4.0/3.0))
_b1 = 1.0 / (2.0 - pow(2.0,1.0/3.0))
_b2 = 1.0 / (1.0 - pow(2.0,2.0/3.0))


class Yoshida4th(FixedGridODESolver):
    "support only H = p^2/2 + V(q,theta) form"
    order = 4

    def __init__(self, eps=0., **kwargs):
        super(Yoshida4th, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(),dtype=self.dtype,device=self.device)
        n = y.size(-1) // 2

        dy[..., :n] = h*_c1*y[..., n:]
        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = h*_b1*k_[..., n:]

        dy[..., :n] = dy[..., :n] + h*_c2*(y[..., n:] + dy[..., n:])
        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = dy[..., n:] + h*_b2*k_[..., n:]

        dy[..., :n] = dy[..., :n] + h*_c2*(y[..., n:] + dy[..., n:])
        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = dy[..., n:] + h*_b1*k_[..., n:]

        dy[..., :n] = dy[..., :n] + h*_c1*(y[..., n:] + dy[..., n:])

        return dy

    def _step_func(self, func, t, dt, y):
        return self._step_symplectic(func, y, t, dt)

    def integrate(self, t):
        n = len(self.y0) // 2
        reverse = False
        if abs(t[0]) > abs(t[-1]):
            reverse = True

        if reverse:
            self.y0[n:] = -self.y0[n:]

        solution = super().integrate(t)

        if reverse:
            self.y0[n:] = -self.y0[n:]
            solution[:,n:] = -solution[:,n:]

        return solution
 
