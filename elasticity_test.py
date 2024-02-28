"""
This is a monolithic solver for pressure-driven flow around a
neo-Hookean particle.  The particle is **free** to move with
the flow.  However, equations are formulated in a moving
frame that travels with the particle

The problem is formulated using the ALE
method, which maps the deformed geometry to the initial
geometry.  The problem is solved using the
initial geometry; the deformed geometry can by using the
WarpByVector filter in Paraview using the displacement
computed in the fluid domain.

The problem uses Lagrange multipliers to ensure the centre of
mass of the particle remains fixed as well as to impose
continuity of stress

The code works by initially solving the problem with a small
value of epsilon (ratio of fluid stress to elastic stiffness)
and then gradually ramping up epsilon.  If convergence is
not obtained then the code tries again using a smaller value
of epsilon.

This code implements to axisymmetric verion, with changes to
the deformation gradient tensor, determinant, and divergence.

"""

from dolfin import *
from helpers import *
import matplotlib.pyplot as plt
from csv import writer
import numpy as np


# Python function for axisymmetric divergence
def diva(vec, r):
    return div(vec) + vec[1] / r


# directory for file output
dir = '/home/simon/data/fenics/elasticity_axisym/'


def generate_output_files():
    output_s = XDMFFile(dir + "solid.xdmf")
    output_s.parameters['rewrite_function_mesh'] = False
    output_s.parameters["functions_share_mesh"] = True
    output_s.parameters["flush_output"] = True
    return output_s


def get_mesh():
    meshname = 'sphere'
    mesh = Mesh('mesh/' + meshname + '.xml')
    subdomains = MeshFunction("size_t", mesh, 'mesh/' + meshname + '_physical_region.xml')
    bdry = MeshFunction("size_t", mesh, 'mesh/'
                        + meshname + '_facet_region.xml')
    return mesh, subdomains, bdry


def bottom(x, on_boundary):
    return (on_boundary and near(x[0], 0.0))


"""
Solver parameters
"""
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "absolute_tolerance": 1e-8,
                                          "error_on_nonconvergence": False}}
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}


output_s = generate_output_files()

# mesh has been created with Gmsh
mesh, subdomains, bdry = get_mesh()
plot(mesh)
plt.show()

"""
    Physical parameters
"""

# the initial value of epsilon to try solving the problem with
eps_try = 0.001
eps = Constant(eps_try)

# the max value of epsilon
eps_max = 0.001

# the incremental increase in epsilon
de = 0.05

# the min and max values of the increments to make.
de_min = 1e-3
de_max = 1e-1

# elastic constant
lambda_s = 1

# define the boundaries (values from the gmsh file)
circle = 1
solid_axis = 6

# define the domains
solid = 11

# dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
# ds = Measure("ds", domain=mesh, subdomain_data=bdry)
# dS = Measure("dS", domain=mesh, subdomain_data=bdry)
# dS = dS(circle)

P1 = VectorElement('CG', mesh.ufl_cell(), 1)
R = FiniteElement('R', mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, P1)


# state = Function(V)
# u_s, f_0 = split(state)
#
# trial_state = TrialFunctions(V)
# u_tr, f_0_tr = trial_state
#
# test_state = TestFunctions(V)
# v_s, g_0 = test_state
#
# # placeholder for last converged solution
# state_old = Function(V)
# u_s_old, f_0_old = split(state_old)

u_s = Function(V)
u_tr = TrialFunction(V)
v_s = TestFunction(V)

# # Definition of Neumann condition domain
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

top = AutoSubDomain(lambda x: near(x[1] ** 2 + x[0] ** 2, 1.0))

top.mark(boundaries, 1)
ds = ds(subdomain_data=boundaries)

# normal and tangent vectors
nn = FacetNormal(mesh)
tt = as_vector((-nn[1], nn[0]))

ez = as_vector([1, 0])
ex = as_vector([0, 1])

# define the surface differential on the circle
x = SpatialCoordinate(mesh)
r = x[1]

# impose zero vertical solid displacement at the centreline axis
# bc = DirichletBC(V.sub(0).sub(1), Constant(0.0), bdry, solid_axis)
bc = DirichletBC(V.sub(1), Constant(0.0), bdry, solid_axis)

p_mag = 0.1
# 1/eps*2*p_mag*(4 + 13*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)) + 16*pow(p_mag, 2)*pow(sqrt(pow(x[0], 2)+pow(x[1], 2)), 2) + 8*pow(p_mag, 3)*pow(sqrt(pow(x[0], 2)+pow(x[1], 2)), 3) + pow(1+p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 4)*pow(1+2*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 2)*(2+3*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)))*lambda_s)/((1+p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)))*pow(1+2*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 2))
# solid_pressure = Expression((
#     '1/eps*2*p_mag*(4 + 13*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)) + 16*pow(p_mag, 2)*pow(sqrt(pow(x[0], 2)+pow(x[1], 2)), 2) + 8*pow(p_mag, 3)*pow(sqrt(pow(x[0], 2)+pow(x[1], 2)), 3) + pow(1+p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 4)*pow(1+2*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 2)*(2+3*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)))*lambda_s)/((1+p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)))*pow(1+2*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 2)) * x[0] / sqrt(pow(x[0], 2) + pow(x[1], 2))',
#     '1/eps*2*p_mag*(4 + 13*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)) + 16*pow(p_mag, 2)*pow(sqrt(pow(x[0], 2)+pow(x[1], 2)), 2) + 8*pow(p_mag, 3)*pow(sqrt(pow(x[0], 2)+pow(x[1], 2)), 3) + pow(1+p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 4)*pow(1+2*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 2)*(2+3*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)))*lambda_s)/((1+p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)))*pow(1+2*p_mag*sqrt(pow(x[0], 2)+pow(x[1], 2)), 2)) * x[1] / sqrt(pow(x[0], 2) + pow(x[1], 2))'),
#     degree=0, p_mag=p_mag, lambda_s=lambda_s, eps=eps)
# solid_pressure = Expression((
#                             'p_mag / eps * x[0] / sqrt(pow(x[0], 2) + pow(x[1], 2))',
#                             'p_mag / eps * x[1] / sqrt(pow(x[0], 2) + pow(x[1], 2))'),
#                             degree=0, p_mag=p_mag, lambda_s=lambda_s, eps=eps)
# 1/eps*p_mag*(1+p_mag)*(4+(4+17*p_mag+25*pow(p_mag, 2)+16*pow(p_mag, 3)+4*pow(p_mag, 4))*lambda_s)/(1+2*p_mag)*x[0]/sqrt(pow(x[0], 2) + pow(x[1], 2))
# solid_traction = Expression(('1/eps*p_mag*(1+p_mag)*(4+(4+17*p_mag+25*pow(p_mag, 2)+16*pow(p_mag, 3)+4*pow(p_mag, 4))*lambda_s)/(1+2*p_mag)*x[0]/sqrt(pow(x[0], 2) + pow(x[1], 2))',
#                              '1/eps*p_mag*(1+p_mag)*(4+(4+17*p_mag+25*pow(p_mag, 2)+16*pow(p_mag, 3)+4*pow(p_mag, 4))*lambda_s)/(1+2*p_mag)*x[1]/sqrt(pow(x[0], 2) + pow(x[1], 2))'),
#                             degree=0, p_mag=p_mag, eps=eps, lambda_s=lambda_s)
# 1+p_mag-(1/(1+p_mag))+pow(1+p_mag, 2)*(pow(1+p_mag, 3)-1)*lambda_s
solid_traction = Expression(('p_mag / eps * (1+p_mag-(1/(1+p_mag))+pow(1+p_mag, 2)*(pow(1+p_mag, 3)-1)*lambda_s) * x[0]',
                             'p_mag / eps * (1+p_mag-(1/(1+p_mag))+pow(1+p_mag, 2)*(pow(1+p_mag, 3)-1)*lambda_s) * x[1]'),
                            degree=0, p_mag=p_mag, eps=eps, lambda_s=lambda_s)
# ---------------------------------------------------------------------
# Define the model
# ---------------------------------------------------------------------

I = Identity(2)

"""
Solids problem
"""
# deformation gradient tensor
F = I + grad(u_s)
H = inv(F.T)

# (non-dim) compressible PK1 stress tensor
lambda_s = 1  # nu_s = lambda_s/(2(lambda_s + mu_s))
# axisym
J_s = det(F) * (1 + u_s[1] / r)
# reg
def F_func(u_s):
    return I + grad(u_s)

def H_func(u_s):
    return inv(F_func(u_s))

def J_s_func(u_s):
    return det(F_func(u_s)) * (1 + u_s[1] / r)

def Sigma_s_func(u_s, eps):
    return 1 / eps * (F_func(u_s) - H_func(u_s)) + lambda_s / eps * (J_s_func(u_s) - 1) * J_s_func(u_s) * H_func(u_s)

# J_s = det(F)
Sigma_s = 1 / eps * (F - H) + lambda_s / eps * (J_s - 1) * J_s * H

# (non-dim) PK1 stress tensor and incompressibility condition
# Sigma_s = 1 / eps * (F - H) - p_s * H
# ic_s = det(F) * (1 + u_s[1] / r) - 1

# solid deformation linear
# Sigma_s = -p_s * I + (grad(u_s) + grad(u_s).T) / eps
# ic_s = diva(u_s, r)


# ---------------------------------------------------------------------
# build equations
# ---------------------------------------------------------------------

# compressible
FUN3 = (-inner(Sigma_s, grad(v_s)) * r * dx
        - 1 / eps * (1 + u_s[1] / r) * v_s[1] * dx
        + (1 / eps - lambda_s / eps * (J_s - 1) * J_s) * v_s[1] * r / (r + u_s[1]) * dx
        # + inner(as_vector([f_0, 0]), v_s) * r * dx(solid)  # net z disp = 0
        # + dot(ez, u_s) * g_0 * r * dx(solid)  # net z disp = 0
        # + inner(solid_pressure, v_s) * r * dx  # artificial pressure traction on disc
        + inner(solid_traction, v_s) * r * ds(0))  # artificial traction

# mean axial solid displacement is zero
# FUN9 = dot(ez, u_s) * g_0 * r * dx

# FUN3 = (-inner(Sigma_s, grad(v_s)) * dx
#         + inner(solid_pressure, v_s) * dx  # artificial pressure traction on disc
#         - inner(solid_traction, v_s) * ds)  # artificial traction

# linear elasticity for the solid
# FUN3 = (-inner(Sigma_s, grad(v_s)) * r * dx(solid)
#         - 2 / eps * u_s[1] * v_s[1] / r * dx(solid)
#         + p_s * v_s[1] * dx(solid)
#         + inner(as_vector([f_0, 0]), v_s) * r * dx(solid)
#         - inner(lam("-"), v_s("-")) * r("-") * dS)

# Incompressibility for the solid
# FUN4 = ic_s * q_s * r * dx(solid)
# FUN = [FUN3, FUN9]
# problem = fem.petsc.NonlinearProblem(FUN3, state, bc)
# solver = nls.petsc.NewtonSolver(mesh.comm, problem)
JAC = derivative(FUN3, u_s, u_tr)

# ---------------------------------------------------------------------
# set up the solver
# ---------------------------------------------------------------------

# Initialize solver
solve(FUN3 == 0, u_s, bc, J=JAC,
      form_compiler_parameters=ffc_options)
# problem = NonlinearVariationalProblem(FUN3, u_s, bc, JAC)
# solver = NonlinearVariationalSolver(problem)
# prm = solver.parameters
# solver = PETScSNESSolver(problem)
# solver.parameters.update(snes_solver_parameters["snes_solver"])

# ---------------------------------------------------------------------
# Set up code to save solid quanntities only on the solid domain and
# fluid quantities only on the fluid domain
# ---------------------------------------------------------------------

# Create function spaces for the velocity and displacement
Vs = VectorFunctionSpace(mesh, "CG", 1)
P1v = VectorFunctionSpace(mesh, "DG", 0)
cs = x[0] / sqrt(x[0] ** 2 + x[1] ** 2)
sn = x[1] / sqrt(x[0] ** 2 + x[1] ** 2)
A = as_tensor([[cs, sn], [-sn, cs]])
sig_s = project((A * Sigma_s_func(u_s, eps) * A.T) * as_vector([1, 0]), P1v)

u_s_only = Function(Vs)


# Python function to save solution for a given value
# of epsilon
def save(eps):
    u_s_only = project(u_s, Vs)
    u_s_only.rename("u_s", "u_s")
    sig_s.rename("sigma", "sigma")
    print(u_s(0, 1)[1])
    output_s.write(u_s_only, eps)
    output_s.write(sig_s, eps)


save(eps)
# ---------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------

# n = 0

# last converged value of epsilon
# eps_conv = 0

# increment epsilon and solve
# while eps_conv < eps_max:
#
#     print('-------------------------------------------------')
#     print(f'attempting to solve problem with eps = {float(eps):.4e}')
#
#     # make a prediction of the next solution based on how the solution
#     # changed over the last two increments, e.g. using a simple
#     # extrapolation formula
#     if n > 1:
#         u_s[:] = u_s_old[:] + (eps_try - eps_conv) * du_deps
#
#     (its, conv) = solver.solve()
#
#     # if the solver converged...
#     if conv:
#
#         n += 1
#         # update value of eps_conv and save
#         eps_conv = float(eps)
#         save(eps_conv)
#
#         # update the value of epsilon to try to solve the problem with
#         eps_try += de
#
#         # copy the converged solution into old solution
#         assign(u_s_old, u_s)
#
#         print('Disp1 = ', u_s(0, 1)[1])
#         print('Disp2 = ', u_s(1, 0)[0])
#
#         # approximate the derivative of the solution wrt epsilon
#         if n > 0:
#             du_deps = (u_s[:] - u_s_old[:]) / de
#
#     # if the solver diverged...
#     if not (conv):
#         # halve the increment in epsilon if not at smallest value
#         # and use the previously converged solution as the initial
#         # guess
#         if de > de_min:
#             de /= 2
#             eps_try = eps_conv + de
#             assign(u_s, u_s_old)
#         else:
#             print('min increment reached...aborting')
#             save(eps_try)
#             break
#
#     # update the value of epsilon
#     eps.assign(eps_try)