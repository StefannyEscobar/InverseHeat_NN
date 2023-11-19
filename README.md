# Finite Element Method (FEM) and Neural Network Integration for Inverse Heat Transfer Problem

## Overview

This repository contains the implementation of a methodology that combines the Finite Element Method (FEM) with Neural Networks (NN) for efficiently solving the inverse heat transfer problem. The study explores the use of the weak formulation within FEM, implemented using the FEniCS framework, along with the Dolfin Adjoint algorithmic differentiation tool. Additionally, a Neural Network is incorporated to enhance the computational efficiency of gradient calculations.

## Methodology Highlights

- **FEM Implementation**: Utilizes the weak formulation in FEM, providing advantages in problem-solving and making it more adaptable to computational tools.

- **Dolfin Adjoint**: Implements FEM using the FEniCS framework along with Dolfin Adjoint for algorithmic differentiation, contributing to the effectiveness of the methodology.

- **Neural Network Integration**: Inspired by Mitusch et al. (2023), a Neural Network is incorporated to efficiently calculate gradients, enhancing computational efficiency.

- **Optimization**: The BFGS iterative method is employed for solving unconstrained nonlinear optimization problems, contributing to the overall robustness of the approach.

## Results and Insights

- **Consistency in Results**: Results from both adjoint methods and Neural Networks are nearly identical, highlighting the robustness of the approach.

- **Stability Under Varying Conditions**: The methodology remains stable and resilient even with different initial guesses, providing a reliable solution to the inverse problem.

- **Insights for Future Research**: Although anticipated results were not achieved, valuable insights and lessons learned pave the way for future refinements and explorations.

## Graphical Representations

- [Figure 1: Results of the adjoint method, Neural Networks (NN), and FEM.](path/to/a.png)
- [Figure 2: Results of the adjoint method and FEM.](path/to/b.png)
- [Figure 3: Loss and training for pre-trained model.](path/to/Figure_1.png)
