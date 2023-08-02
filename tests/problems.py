import math
import numpy as np
import scipy.linalg
import torch


class ConstantODE(torch.nn.Module):

    def __init__(self):
        super(ConstantODE, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.2))
        self.b = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, t, y):
        return self.a + (y - (self.a * t + self.b))**5

    def y_exact(self, t):
        return self.a * t + self.b


class ConstantODEforSymplectic(torch.nn.Module):

    def __init__(self):
        super(ConstantODEforSymplectic, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.2))
        self.q_0 = torch.tensor(0.5)        
        self.p_0 = torch.tensor(1.0)

    def forward(self, t, y):
        auxiliy_fp = self.a * torch.ones(1).to(y) 
        auxiliy_fq = y[1:]
        return torch.cat([auxiliy_fq,auxiliy_fp])

    def y_exact(self, t):
        t_numpy = t.detach().cpu().numpy()
        ans = []
        for t_i in t_numpy:
            auxiliy_q = 0.5 * self. a * t_i ** 2 + self.p_0 * t_i + self.q_0
            auxiliy_p = self. a * t_i + self.p_0 
            ans.append([auxiliy_q, auxiliy_p])
        return torch.stack([torch.tensor(ans_) for ans_ in ans])\
                                                .reshape(len(t_numpy), 2).to(t)


class SineODE(torch.nn.Module):
    def forward(self, t, y):
        return 2 * y / t + t**4 * torch.sin(2 * t) - t**2 + 4 * t**3

    def y_exact(self, t):
        return -0.5 * t**4 * torch.cos(2 * t) + 0.5 * t**3 * torch.sin(2 * t) + 0.25 * t**2 * torch.cos(
            2 * t
        ) - t**3 + 2 * t**4 + (math.pi - 0.25) * t**2


class LinearODE(torch.nn.Module):

    def __init__(self, dim=10):
        super(LinearODE, self).__init__()
        self.dim = dim
        U = torch.randn(dim, dim) * 0.1
        A = 2 * U - (U + U.transpose(0, 1))
        self.A = torch.nn.Parameter(A)
        self.initial_val = np.ones((dim, 1))

    def forward(self, t, y):
        return torch.mm(self.A, y.reshape(self.dim, 1)).reshape(-1)

    def y_exact(self, t):
        t_numpy = t.detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        ans = []
        for t_i in t_numpy:
            ans.append(np.matmul(scipy.linalg.expm(A_np * t_i), self.initial_val))
        return torch.stack([torch.tensor(ans_) for ans_ in ans]).reshape(len(t_numpy), self.dim).to(t)


class HarmonicOscillator(torch.nn.Module):

    def __init__(self, dim=6):
        super(HarmonicOscillator, self).__init__()
        n = dim//2
        kappa = 0.1 * torch.abs(torch.randn(n))
        D_H_upper = torch.cat((torch.zeros(n,n) , torch.diag(-kappa)), 1)
        D_H_lower = torch.cat((torch.eye(n), torch.zeros(n,n)), 1)
        self.D_H = torch.cat((D_H_upper, D_H_lower),0)

        self.dim = dim
        self.n = n
        self.kappa = torch.nn.Parameter(kappa)
        self.initial_val = 0.5* np.ones((1, dim))

    def forward(self, t, y):
        n = self.n
        return torch.cat([y[n:], -self.kappa*y[:n]]) 

    def y_exact(self, t):
        t_numpy = t.detach().cpu().numpy()
        D_H_np = self.D_H.detach().cpu().numpy()
        ans = []
        for t_i in t_numpy:
            ans.append(np.matmul(self.initial_val,scipy.linalg.expm(D_H_np * t_i)))
        return torch.stack([torch.tensor(ans_) for ans_ in ans]).reshape(len(t_numpy), self.dim).to(t)


PROBLEMS = {'constant': ConstantODE,'constant_symplectic':ConstantODEforSymplectic,
            'linear': LinearODE, 'sine': SineODE, 'harmonic':HarmonicOscillator}

DTYPES = (torch.float32, torch.float64)
DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda:0')
FIXED_METHODS = ('euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams')
FIXED_SYMPLECTIC_METHODS = ('yoshida4th',)
ADAPTIVE_METHODS = ('dopri5', 'bosh3', 'adaptive_heun', 'dopri8')  # TODO: add in adaptive adams and tsit5 if/when they're fixed
METHODS = FIXED_METHODS + ADAPTIVE_METHODS + FIXED_SYMPLECTIC_METHODS


def construct_problem(device, npts=10, ode='constant', reverse=False, dtype=torch.float64):

    f = PROBLEMS[ode]().to(dtype=dtype, device=device)

    t_points = torch.linspace(1, 8, npts, dtype=dtype, device=device, requires_grad=True)
    sol = f.y_exact(t_points)

    def _flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=device)
        return x[tuple(indices)]

    if reverse:
        t_points = _flip(t_points, 0).clone().detach()
        sol = _flip(sol, 0).clone().detach()

    return f, sol[0].detach().requires_grad_(True), t_points, sol


if __name__ == '__main__':
    f = SineODE().cpu()
    t_points = torch.linspace(1, 8, 100, device='cpu')
    sol = f.y_exact(t_points)

    import matplotlib.pyplot as plt
    plt.plot(t_points, sol)
    plt.show()
