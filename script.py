import numpy as np
import sunode
import sunode.wrappers.as_aesara
import pymc as pm
import matplotlib.pyplot as plt
lib = sunode._cvodes.lib


def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'hares': p.alpha * y.hares - p.beta * y.lynx * y.hares,
        'lynx': p.delta * y.hares * y.lynx - p.gamma * y.lynx,
    }

# initialize problem
problem = sunode.symode.SympyProblem(
    params={
        # We need to specify the shape of each parameter.
        # Any empty tuple corresponds to a scalar value.
        'alpha': (),
        'beta': (),
        'gamma': (),
        'delta': (),
        'hares0': ()
    },
    states={
        # The same for all state variables
        'hares': (),
        'lynx': (),
    },
    rhs_sympy=lotka_volterra,
    derivative_params=[
        # We need to specify with respect to which variables
        # gradients should be computed.
        ('alpha',),
        ('beta',),
        ('hares0',),
    ],
)

tvals = np.linspace(0, 10, 3)

y0 = np.zeros((), dtype=problem.state_dtype)
y0['hares'] = 1e0
y0['lynx'] = 0.1
params_dict = {
    'alpha': 0.1,
    'beta': 0.2,
    'gamma': 0.3,
    'delta': 0.4,
    'hares0': 1e0
}


sens0 = np.zeros((3, 2))
sens0[2,0] = np.log(10)*1e0

solver = sunode.solver.Solver(problem, solver='BDF', sens_mode='simultaneous')
yout, sens_out = solver.make_output_buffers(tvals)


# gradient via fwd senstivity
solver.set_params_dict(params_dict)
output = solver.make_output_buffers(tvals)
solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0=sens0, sens_out=sens_out)

grad_out_fwd = [ sens_out[:,j,:].sum() for j in range(3)]
print(grad_out_fwd)

# gradient via adj senstivity
solver = sunode.solver.AdjointSolver(problem, solver='BDF')
solver.set_params_dict({
    'alpha': 0.1,
    'beta': 0.2,
    'gamma': 0.3,
    'delta': 0.4,
    'hares0': 1e0
})
tvals_expanded = tvals #np.linspace(0, 10, 21)
yout, grad_out, lambda_out = solver.make_output_buffers(tvals_expanded)
# lib.CVodeSStolerances(solver._ode, 1e-8, 1e-8)
# lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-10, 1e-10)
# lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-10, 1e-10)
# lib.CVodeSetMaxNumSteps(solver._ode, 10000)
# lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 10000)

solver.solve_forward(t0=0, tvals=tvals, y0=y0, y_out=yout)
grads = np.ones_like(yout)
# grads[::10,:] = 1
solver.solve_backward(t0=tvals_expanded[-1], tend=tvals_expanded[0], tvals=tvals_expanded[1:-1],
                      grads=grads, grad_out=grad_out, lamda_out=lambda_out)
grad_out_adj = -np.matmul(sens0, lambda_out  -grads[0, :]) + grad_out
print(grad_out_adj)
