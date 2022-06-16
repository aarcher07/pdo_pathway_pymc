import sunode
import matplotlib.pyplot as plt
import numpy as np

def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'hares': p.alpha * y.hares - p.beta * y.lynx * y.hares,
        'lynx': p.delta * y.hares * y.lynx - p.gamma * y.lynx,
    }


problem = sunode.symode.SympyProblem(
    params={
        # We need to specify the shape of each parameter.
        # Any empty tuple corresponds to a scalar value.
        'alpha': (),
        'beta': (),
        'gamma': (),
        'delta': (),
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
    ],
)

# The solver generates uses numba and sympy to generate optimized C functions
# for the right-hand-side and if necessary for the jacobian, adoint and
# quadrature functions for gradients.
solver = sunode.solver.Solver(problem, solver='BDF', sens_mode='staggered')


tvals = np.linspace(0, 2000)
# We can use numpy structured arrays as input, so that we don't need
# to think about how the different variables are stored in the array.
# This does not introduce any runtime overhead during solving.
y0 = np.zeros((), dtype=problem.state_dtype)
y0['hares'] = 1
y0['lynx'] = 0.1

# We can also specify the parameters by name:
solver.set_params_dict({
    'alpha': 0.1,
    'beta': 0.2,
    'gamma': 0.3,
    'delta': 0.4,
})

yout, sens_out = solver.make_output_buffers(tvals)
print(sens_out.shape)
solver.solve(t0=0, tvals=tvals, y0=y0, y_out=yout, sens0 = np.zeros((2,2)), sens_out=sens_out)

# We can convert the solution to an xarray Dataset
plt.plot(tvals, yout[:,0])
plt.title('Hares')
plt.show()
plt.plot(tvals, sens_out[:,0,0])
plt.plot(tvals, sens_out[:,0,1])
plt.title('Hares Sens')
plt.show()
# Or we can convert it to a numpy record array
plt.plot(tvals, yout[:,1])
plt.title('Lynx')
plt.show()
plt.plot(tvals, sens_out[:,1,0])
plt.plot(tvals, sens_out[:,1,1])
plt.title('Lynx Sens')

plt.show()
