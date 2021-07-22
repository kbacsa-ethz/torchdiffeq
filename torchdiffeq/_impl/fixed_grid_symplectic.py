# Implemention of symplectic integrator.
# Whole input as (q,p) and  the shape is (bath_size, 2N)
# where N is the number of paritcles 

import torch
from .solvers import FixedGridODESolver

# symplectic integrators constants
_c1 = 1.0 / (4.0 - pow(2.0, 4.0 / 3.0))
_c2 = (1.0 - pow(2.0, 1.0 / 3.0)) / (4.0 - pow(2.0, 4.0 / 3.0))
_b1 = 1.0 / (2.0 - pow(2.0, 1.0 / 3.0))
_b2 = 1.0 / (1.0 - pow(2.0, 2.0 / 3.0))


class SymplecticSolver(FixedGridODESolver):
    def __init__(self, eps=0., **kwargs):
        super(SymplecticSolver, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_symplectic(self, func, y, t, dt):
        pass

    def _step_func(self, func, t, dt, y):
        return self._step_symplectic(func, y, t, dt)

    def integrate(self, t):
        n = len(self.y0) // 2
        reverse = False
        if abs(t[0]) > abs(t[-1]):
            reverse = True

        if reverse:
            self.y0[..., n:] = -self.y0[..., n:]

        solution = super().integrate(t)

        if reverse:
            self.y0[..., n:] = -self.y0[..., n:]
            solution[:, n:] = -solution[:, n:]

        return solution
       

class StoermerVerlet(SymplecticSolver):
    order = 1

    def __init__(self, **kwargs):
        super(StoermerVerlet, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        dy[..., n:] = y[..., :n] - y[..., n:]
        k_ = func(t + self.eps, y[..., :n])
        dy[..., :n] = dy[..., n:] + (h**2) * k_

        return dy
        
        
class SO2(SymplecticSolver):
    order = 1

    def __init__(self, **kwargs):
        super(SO2, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        dy[..., n:] = y[..., :n] - y[..., n:]

        k_ = func(t + self.eps, y[..., :n])

        sin_q_delta = torch.sin(y[..., :n] - y[..., n:]) + (h**2) * k_
        dy[..., :n] = torch.arcsin(torch.clip(sin_q_delta, -(1.-1e-4), 1-1e-4))

        return dy


class Yoshida4th(SymplecticSolver):
    "support only H = p^2/2 + V(q,theta) form"
    order = 4

    def __init__(self, **kwargs):
        super(Yoshida4th, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        dy[..., :n] = h * _c1 * y[..., n:]
        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = h * _b1 * k_[..., n:]

        dy[..., :n] = dy[..., :n] + h * _c2 * (y[..., n:] + dy[..., n:])
        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = dy[..., n:] + h * _b2 * k_[..., n:]

        dy[..., :n] = dy[..., :n] + h * _c2 * (y[..., n:] + dy[..., n:])
        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = dy[..., n:] + h * _b1 * k_[..., n:]

        dy[..., :n] = dy[..., :n] + h * _c1 * (y[..., n:] + dy[..., n:])

        return dy


class VelocityVerlet(SymplecticSolver):
    "support only H = p^2/2 + V(q,theta) form"
    order = 2

    def __init__(self, **kwargs):
        super(VelocityVerlet, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        k_ = func(t + self.eps, y[..., :n])
        dy[..., :n] = h * (y[..., n:] - 0.5 * h * k_)

        k_ += func(t + self.eps, y[..., :n] + dy[..., :n])
        dy[..., n:] = - 0.5 * h * k_
        return dy


class VelocityVerletDissipative(SymplecticSolver):
    "support only H = p^2/2 + V(q,theta) form"
    order = 2

    def __init__(self, **kwargs):
        super(VelocityVerletDissipative, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        yt = torch.zeros(tuple(map(sum, zip(y.size(), (0, 1)))), dtype=self.dtype, device=self.device)
        dyt = torch.zeros(tuple(map(sum, zip(y.size(), (0, 1)))), dtype=self.dtype, device=self.device)

        yt[..., 0] = t
        yt[..., 1:] = y

        k_ = func(t + self.eps, yt[..., :n+1])
        dyt[..., 1:n+1] = h * (yt[..., n+1:] - 0.5 * h * k_[..., 1:])

        dyt[..., 0] = h

        k_ += func(t + self.eps, yt[..., :n+1] + dyt[..., :n+1])
        dyt[..., n+1:] = - 0.5 * h * k_[..., 1:]

        dy[..., :] = dyt[..., 1:]

        return dy
        
        
class LeapFrog(SymplecticSolver):
    "support only H = p^2/2 + V(q,theta) form"
    order = 2

    def __init__(self, **kwargs):
        super(LeapFrog, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = -0.5 * h * k_[..., n:]

        k_ = func(t + self.eps, y + dy)
        dy[..., :n] = h * k_[..., :n]

        k_ = func(t + self.eps, y + dy)
        dy[..., n:] = dy[..., n:] - 0.5 * h * k_[..., n:]

        return dy


class LeapFrogDissipative(SymplecticSolver):
    "A naive attempt..."
    order = 2

    def __init__(self, **kwargs):
        super(LeapFrogDissipative, self).__init__(**kwargs)

    def _step_symplectic(self, func, y, t, h):
        dy = torch.zeros(y.size(), dtype=self.dtype, device=self.device)
        n = y.size(-1) // 2

        # augmented variables
        yt = torch.zeros(tuple(map(sum, zip(y.size(), (0, 1)))), dtype=self.dtype, device=self.device)
        dyt = torch.zeros(tuple(map(sum, zip(y.size(), (0, 1)))), dtype=self.dtype, device=self.device)

        yt[..., 0] = t
        yt[..., 1:] = y

        k_ = func(t + self.eps, yt + dyt)
        dyt[..., n + 1:] = -0.5 * h * k_[..., n + 1:]

        k1_ = func(t + self.eps, yt + dyt)
        dyt[..., 0] = h
        k2_ = func(t + self.eps, yt + dyt)

        dyt[..., 1:n + 1] = 0.5 * h * (k1_[..., 1:n + 1] + k2_[..., 1:n + 1])

        k_ = func(t + self.eps, yt + dyt)
        dyt[..., n + 1:] = dyt[..., n + 1:] - 0.5 * h * k_[..., n + 1:]

        dy[..., :] = dyt[..., 1:]

        return dy
