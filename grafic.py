from fenics import plot
import matplotlib.pyplot as plt

# Cargar el resultado de FEniCS
u_result = Function(V)
File("inverse_results/predicted_solution.pvd") >> u_result

# Graficar el resultado
plot(u_result, title="Predicted Solution")
plt.show()
