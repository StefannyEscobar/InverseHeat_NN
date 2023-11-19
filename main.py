from dolfin import *
from fenics import *
import numpy as np
import tensorflow as tf
from fenics_adjoint import *
import matplotlib.pyplot as plt

# SECTION 1: FENICS SETUP

# Constants and parameters
rho = Constant(1.46555e6)
C = Constant(20.0)
dt = 1
num_steps = 10

# Mesh and Function Spaces
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 10, 10)
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
    # Define Thermal conductivity
    def k():
        return d + e*x[1]           
    a = rho*C*u*v*dx + dt*inner(grad(v), k()*grad(u))*dx
    L = dt*g*v*dsB + rho*C*inner(u0, v)*dx + dt*Q*v*dx
    u = Function(V)                                                   
    vtkfile = File('Heat_Transfer/solution.pvd')  
    t = 0
    for n in range(num_steps):
        t += dt
        # Compute solution
        solve(a == L, u, bc)  
        # Save solution
        vtkfile << (u, t)
        plot(u)
    # Update solution
    u0.assign(u)
    return u, d, e

# SECTION 2: OPTIMIZATION

# Solve forward problem for target parameters
[u_target, d_target_1, e_target_1] = forward(d_target, e_target)
File("forward_modified/targetTempereature_modified.pvd") << u_target
File("forward_modified/target_d_modified.pvd") << d_target_1
File("forward_modified/target_e_up.pvd") << e_target_1

plot(u_target, title="Forward Solution target")
plt.show()

# Solve forward problem for initial guesses
[u, d, e] = forward(d_guess, e_guess)

# File for saving optimization results
output = File("inverse_Up/inverse_result_up.pvd")
output << u
# Plot and show the solution
plot(u, title="Forward Solution aprox")
plt.show()

# Define objective function for optimization
alpha = Constant(0.0)
beta = Constant(1e-2)
J = assemble((0.5 * inner((u - u_target), u - u_target)) * dx +
             0.5 * alpha * sqrt(beta * beta + dot(grad(d), grad(d))) * dx)

# Optimize variables using L-BFGS-B method
d_c = Control(d)
J_rf = ReducedFunctional(J, d_c, eval_cb_post=lambda j, d: output << d)
d_opt = minimize(J_rf, method='L-BFGS-B', bounds=(5.0, 200.0), options={"gtol": 1e-15, "ftol": 1e-15})

optimized_d_value = d_opt.vector().get_local()[0]
print("Después de la optimización:")
print("Valor optimizado de d:", optimized_d_value)

# SECTION 3: INTEGRATION WITH NEURAL NETWORK

print("Loading the trained neural network model...")
model = tf.keras.models.load_model('trained_model')
print("Model loaded successfully.")

# Function to get predicted parameters from the neural network
def get_predicted_parameters(x):
    predicted_params = model.predict(np.array([x]))
    return predicted_params[0]

# Function to run FEniCS simulation with predicted parameters
def run_simulation_with_predicted_parameters(x):
    predicted_params = get_predicted_parameters(x)
    [u_result, d_result, e_result] = forward(predicted_params[0], predicted_params[1])
    return u_result, d_result, e_result


# Example usage
input_parameters = np.array([40.0, 100.0]) 
u_result, d_result, e_result = run_simulation_with_predicted_parameters(input_parameters)

# Save the results to files
u_result_function = interpolate(u_result, V)

File("inverse_results/predicted_solution.pvd") << u_result_function
# File("inverse_results/predicted_d.pvd") << d_result
# File("inverse_results/predicted_e.pvd") << e_result

plot(u_result_function, title="Predicted u")
plt.show()
print(f"U target {np.max(u.vector().get_local())}, U encontrada con el valor de d {np.max(u_result.vector().get_local())}")

print("Results saved to 'inverse_results' folder.")
