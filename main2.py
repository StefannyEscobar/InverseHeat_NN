from dolfin import *
from fenics import *
from fenics_adjoint import *
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# SECTION 1: FENICS SETUP

# Constants and parameters
rho = Constant(1.46555e6)
C = Constant(20.0)
dt = 1
num_steps = 10

# Mesh and Function Spaces
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 18, 18)
V = FunctionSpace(mesh, "Lagrange", 1)
x = SpatialCoordinate(mesh)
M = FunctionSpace(mesh, "CG", 1)

# Heat source and boundary conditions
Q_heat = 20000.0
Q = Constant(0.0)
g = Expression('Q_heat', degree=1, Q_heat=Constant(Q_heat))

u_0 = Constant(0.0)
u0 = interpolate(u_0, V)

def left_boundary(x, on_boundary):
    return on_boundary and (abs(x[0]) < DOLFIN_EPS)

bc = DirichletBC(V, Constant(0.0), left_boundary)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((abs(x[0] - 1.) < DOLFIN_EPS))

vertical = right()
tdim = mesh.topology().dim()
boundaries = MeshFunction('size_t', mesh, tdim-1)
boundaries.set_all(0)
vertical.mark(boundaries, 1)
dsB = Measure("ds", subdomain_id=1, subdomain_data=boundaries)

# Target and initial guesses for parameters


d_target = interpolate(Constant(200.0), M)
d_guess = interpolate(Constant(500.0), M)
e_target = interpolate(Constant(130.0), M)
e_guess = interpolate(Constant(1000.0), M)

# Function to define the forward problem
def forward(d, e):
    u = TrialFunction(V)
    v = TestFunction(V)
    u0 = Function(V)
    def k():
        return d + e * x[1]

    a = rho * C * u * v * dx + dt * inner(grad(v), k() * grad(u)) * dx
    L = dt * g * v * dsB + rho * C * inner(u0, v) * dx + dt * Q * v * dx

    u = Function(V)
    vtkfile = File('Heat_Transfer/solution.pvd')
    t = 0
    for n in range(num_steps):
        t += dt
        solve(a == L, u, bc)
        vtkfile << (u, t)
        plot(u)
    u0.assign(u)
    return u, d, e

# SECTION 2: OPTIMIZATION


[u_target, d_target_1, e_target_1] = forward(d_target, e_target)
File("forward_modified/targetTempereature_modified.pvd") << u_target
File("forward_modified/target_d_modified.pvd") << d_target_1
File("forward_modified/target_e_up.pvd") << e_target_1


[u, d, e] = forward(d_guess, e_guess)
output = File("inverse_Up/inverse_result_up.pvd")


alpha = Constant(1e-2)
J = assemble((0.5 * inner((u - u_target), u - u_target)) * dx +
             0.5 * alpha * alpha*(dot(grad(d), grad(d))) * dx)


d_c = Control(d)
J_rf = ReducedFunctional(J, d_c, eval_cb_post=lambda j, d: output << d)
d_opt = minimize(J_rf, method='L-BFGS-B', bounds=(5.0, 2000.0), options={"gtol": 1e-15, "ftol": 1e-15})

optimized_d_value = d_opt.vector().get_local()[0]
print("Después de la optimización con FEniCS:")
print("Valor optimizado de d:", optimized_d_value)

[u_opt, _, _] = forward(d_opt, e_guess)
File("inverse_Up/optimized_solution.pvd") << u_opt


# SECTION 3: INTEGRATION WITH NEURAL NETWORK

def objective_function_tf(params):
    params_np = np.array(params)
    u_pred, _, _ = forward(params_np[0], params_np[1])
    
    J_expr = 0.5 * inner((u_pred - u_target), (u_pred - u_target)) * dx
    
    d_function = interpolate(Constant(params_np[0]), M)
    e_function = interpolate(Constant(params_np[1]), M)

    dJ_expr = derivative(J_expr, d_function)
    eJ_expr = derivative(J_expr, e_function)

    J = assemble(J_expr)
    dJ = np.array(assemble(dJ_expr))
    eJ = np.array(assemble(eJ_expr))

    return J, np.array([dJ, eJ])

#Implementación

initial_parameters = np.array([400.0, 1000.0])  # initial guess
optimized_parameters = tfp.optimizer.lbfgs_minimize(objective_function_tf, initial_parameters)

print("Optimized Parameters with TensorFlow:", optimized_parameters.position.numpy())


[u_tf, _, _] = forward(optimized_parameters.position.numpy()[0], optimized_parameters.position.numpy()[1])
File("inverse_Up/optimized_solution_tf.pvd") << u_tf
print(f"U target {np.max(u.vector().get_local())}, U encontrada con NN {np.max(u_tf.vector().get_local())}, U con FENICS {np.max(u_opt.vector().get_local())}")

# Visualización de resultados

plot(u_opt, title="Optimized Solution Fenics")
fenics.plot(mesh)
plt.show()

# Visualización de la solución optimizada con TensorFlow Probability
plot(u_tf, title="Optimized Solution with TensorFlow")
fenics.plot(mesh)
plt.show()

