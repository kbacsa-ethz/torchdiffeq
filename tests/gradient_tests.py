import unittest
import torch
import torchdiffeq

from problems import construct_problem, PROBLEMS, DEVICES, METHODS, \
                     FIXED_SYMPLECTIC_METHODS


def max_abs(tensor):
    return torch.max(torch.abs(tensor))


class TestGradient(unittest.TestCase):
    def test_odeint(self):
        for device in DEVICES:
            for method in METHODS:

                with self.subTest(device=device, method=method):
                    if method in FIXED_SYMPLECTIC_METHODS:
                        f, y0, t_points, _ = construct_problem(device=device,ode='constant_symplectic')
                    else:
                        f, y0, t_points, _ = construct_problem(device=device)
                    func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method=method)
                    self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adjoint(self):
        """
        Test against dopri5
        """
        for device in DEVICES:
            for ode in PROBLEMS:
                for t_grad in (True, False):
                    if ode == 'constant':
                        eps = 1e-12
                    elif ode == 'linear':
                        eps = 1e-5
                    elif ode == 'sine':
                        eps = 5e-3
                    elif ode in ['harmonic','constant_symplectic']:
                        eps = 1e-5
                    else:
                        raise RuntimeError

                    with self.subTest(device=device, ode=ode, t_grad=t_grad):
                        torch.manual_seed(1234)
                        f, y0, t_points, _ = construct_problem(device=device, ode=ode)
                        t_points.requires_grad_(t_grad)

                        ys = torchdiffeq.odeint(f, y0, t_points, rtol=1e-9, atol=1e-12)
                        gradys = torch.rand_like(ys)
                        ys.backward(gradys)

                        reg_y0_grad = y0.grad.clone()
                        reg_t_grad = t_points.grad.clone() if t_grad else None
                        reg_params_grads = []
                        for param in f.parameters():
                            reg_params_grads.append(param.grad.clone())

                        y0.grad.zero_()
                        if t_grad:
                            t_points.grad.zero_()
                        for param in f.parameters():
                            param.grad.zero_()

                        ys = torchdiffeq.odeint_adjoint(f, y0, t_points, rtol=1e-9, atol=1e-12)
                        ys.backward(gradys)

                        adj_y0_grad = y0.grad
                        adj_t_grad = t_points.grad if t_grad else None
                        adj_params_grads = []
                        for param in f.parameters():
                            adj_params_grads.append(param.grad)

                        self.assertLess(max_abs(reg_y0_grad - adj_y0_grad), eps)
                        if t_grad:
                            self.assertLess(max_abs(reg_t_grad - adj_t_grad), eps)
                        for reg_grad, adj_grad in zip(reg_params_grads, adj_params_grads):
                            self.assertLess(max_abs(reg_grad - adj_grad), eps)


class TestCompareAdjointGradient(unittest.TestCase):

    def problem(self, device):
        class Odefunc(torch.nn.Module):
            def __init__(self):
                super(Odefunc, self).__init__()
                self.A = torch.nn.Parameter(torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]))
                self.unused_module = torch.nn.Linear(2, 5)

            def forward(self, t, y):
                return torch.mm(y**3, self.A)

        y0 = torch.tensor([[2., 0.]], device=device, requires_grad=True)
        t_points = torch.linspace(0., 25., 10, device=device, requires_grad=True)
        func = Odefunc().to(device)
        return func, y0, t_points

    def test_against_dopri5(self):
        method_eps = {'dopri5': (3e-4, 1e-4, 2e-3)}  # TODO: add in adaptive adams if/when it's fixed.
        for device in DEVICES:
            for method, eps in method_eps.items():
                for t_grad in (True, False):
                    with self.subTest(device=device, method=method):
                        func, y0, t_points = self.problem(device=device)
                        t_points.requires_grad_(t_grad)

                        ys = torchdiffeq.odeint_adjoint(func, y0, t_points, method=method)
                        gradys = torch.rand_like(ys) * 0.1
                        ys.backward(gradys)

                        adj_y0_grad = y0.grad
                        adj_t_grad = t_points.grad if t_grad else None
                        adj_A_grad = func.A.grad
                        self.assertEqual(max_abs(func.unused_module.weight.grad), 0)
                        self.assertEqual(max_abs(func.unused_module.bias.grad), 0)

                        func, y0, t_points = self.problem(device=device)
                        ys = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
                        ys.backward(gradys)

                        self.assertLess(max_abs(y0.grad - adj_y0_grad), eps[0])
                        if t_grad:
                            self.assertLess(max_abs(t_points.grad - adj_t_grad), eps[1])
                        self.assertLess(max_abs(func.A.grad - adj_A_grad), eps[2])


class TestCompareAdjointGradientforSymplectic(unittest.TestCase):
    def test_adjoint_sympelctic(self):

        for device in DEVICES:
            for ode in ['harmonic','constant_symplectic']:
                eps = 3*1e-4
                dt = 0.01
                for t_grad in (True,False):
                    torch.manual_seed(1234)
                    with self.subTest(device=device, ode=ode, t_grad=t_grad):
                        f, y0, t_points, _ = construct_problem(device=device, ode=ode)
                        t_points.requires_grad_(t_grad)

                        ys = torchdiffeq.odeint(f, y0, t_points, method='dopri5')
                        gradys = torch.rand_like(ys)
                        ys.backward(gradys)

                        reg_y0_grad = y0.grad.clone()
                        reg_t_grad = t_points.grad.clone() if t_grad else None
                        reg_params_grads = []
                        for param in f.parameters():
                            reg_params_grads.append(param.grad.clone())

                        y0.grad.zero_()
                        if t_grad:
                            t_points.grad.zero_()
                        for param in f.parameters():
                            param.grad.zero_()
                        ys = torchdiffeq.odeint_adjoint(f, y0, t_points, 
                                                     method='yoshida4th',
                                                     options={'step_size':dt})
                        ys.backward(gradys)

                        adj_y0_grad = y0.grad
                        adj_t_grad = t_points.grad if t_grad else None
                        adj_params_grads = []
                        for param in f.parameters():
                            adj_params_grads.append(param.grad)

                        self.assertLess(max_abs(reg_y0_grad - adj_y0_grad), eps)
                        if t_grad:
                            self.assertLess(max_abs(reg_t_grad - adj_t_grad), eps)
                        for reg_grad, adj_grad in zip(reg_params_grads, adj_params_grads):
                            self.assertLess(max_abs(reg_grad - adj_grad), eps)

    def test_adjoint_sympelctic_quadratic_conservations(self):

        for device in DEVICES:
            for ode in ['harmonic']:#'constant_symplectic']:
                eps = 3*1e-4
                dt = 0.01
                for t_grad in (True,False):
                    torch.manual_seed(1234)
                    with self.subTest(device=device, ode=ode, t_grad=t_grad):
                        f, y0, t_points, _ = construct_problem(device=device, ode=ode)
                        t_points.requires_grad_(t_grad)
                        f0 = f(t_points[0], y0)

                        ys = torchdiffeq.odeint(f, y0, t_points, method='dopri5')
                        f1 = f(t_points[-1], ys[-1])

                        gradys = torch.zeros_like(ys) # initial condition of adjoint
                        gradys[-1] = torch.rand_like(ys[-1])
                        ys.backward(gradys)

                        reg_y0_grad = y0.grad.clone()

                        reg_quad_init = torch.dot(f0,reg_y0_grad)
                        reg_quad_fin = torch.dot(f1,gradys[-1])
                        self.assertLess(max_abs(reg_quad_init - reg_quad_fin), eps)

                        y0.grad.zero_()
                        if t_grad:
                            t_points.grad.zero_()
                        for param in f.parameters():
                            param.grad.zero_()

                        ys = torchdiffeq.odeint_adjoint(f, y0, t_points, 
                                                     method='yoshida4th',
                                                     options={'step_size':dt})
                        f1 = f(t_points[-1], ys[-1])

                        ys.backward(gradys)

                        adj_y0_grad = y0.grad

                        adj_quad_init = torch.dot(f0,adj_y0_grad)
                        adj_quad_fin = torch.dot(f1,gradys[-1])
                        self.assertLess(max_abs(adj_quad_init - adj_quad_fin), eps)

                        self.assertLess(max_abs(reg_y0_grad - adj_y0_grad), eps)
                        self.assertLess(max_abs(reg_quad_init - adj_quad_init), eps)
                        self.assertLess(max_abs(reg_quad_fin - adj_quad_fin), eps)


if __name__ == '__main__':
    unittest.main()
