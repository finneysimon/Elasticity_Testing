"""
Non-linear solver for an elastic particle under a surface traction/body force.
"""

from dolfin import *
# from helpers import *
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


# def bottom(x, on_boundary):
#     return (on_boundary and near(x[1], 0.0))
#
# def circle(x, on_boundary):
#     if on_boundary and near(x[0] ** 2 + x[1] ** 2, 1.0):
#         print(x[0], x[1])
#     return (on_boundary and near(x[0] ** 2 + x[1] ** 2, 1.0))
#
# Sub domain for inflow (right)
# class Bottom(SubDomain):
#     def inside(self, x, on_boundary):
#         return (near(x[1], 0.0) and on_boundary)
#
# class Circle(SubDomain):
#     def inside(self, x, on_boundary):
#         return (near(sqrt(x[1] ** 2 + x[0] ** 2), 1.0) and on_boundary)
#

"""
Solver parameters
"""
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}


output_s = generate_output_files()

# mesh has been created with Gmsh
mesh, _, bdry = get_mesh()
plot(mesh)
plt.show()

"""
    Physical parameters
"""

lambda_s = 1  # elastic constant
eps = 0.1  # strain magnitude

# define the boundaries (values from the gmsh file)
circle = 1
solid_axis = 6

# define the domains
solid = 11

# dx = Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = Measure("ds", domain=mesh, subdomain_data=bdry)

P1 = VectorElement('CG', mesh.ufl_cell(), 1)
R = FiniteElement('R', mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, P1*R)

# Include extra constraint to pin net z disp
state = Function(V)
u_s, f_0 = split(state)

trial_state = TrialFunctions(V)
u_tr, f_0_tr = trial_state

test_state = TestFunctions(V)
v_s, g_0 = test_state

# u_s = Function(V)
# u_tr = TrialFunction(V)
# v_s = TestFunction(V)

### Define solid axis as 0, circle as 1
# boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# boundaries.set_all(0)
#
# circ = Circle()
# circ.mark(boundaries, 1)
#
# bot = Bottom()
# bot.mark(boundaries, 2)
# top = AutoSubDomain(lambda x, on_bdry: circle(x, on_bdry))
# top.mark(boundaries, 1)
# File(dir + "mf.pvd") << boundaries
# ds = ds(subdomain_data=boundaries)

# normal and tangent vectors
nn = FacetNormal(mesh)
tt = as_vector((-nn[1], nn[0]))

ez = as_vector([1, 0])
ex = as_vector([0, 1])

# define the surface differential on the circle
x = SpatialCoordinate(mesh)
r = x[1]

# impose zero vertical solid displacement at the centreline axis
bc = DirichletBC(V.sub(0).sub(1), Constant(0.0), bdry, solid_axis)
# bc = DirichletBC(V.sub(1), Constant(0.0), bottom)

solid_traction = Expression(('(1+eps-(1/(1+eps))+pow(1+eps, 2)*(pow(1+eps, 3)-1)*lambda_s) * x[0]',
                             '(1+eps-(1/(1+eps))+pow(1+eps, 2)*(pow(1+eps, 3)-1)*lambda_s) * x[1]'),
                            degree=0, eps=eps, lambda_s=lambda_s)
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
J_s = det(F) * (1 + u_s[1] / r)
# J_s = det(F)  # non-axisym
Sigma_s = (F - H) + lambda_s * (J_s - 1) * J_s * H

def F_func(u_s):
    return I + grad(u_s)

def H_func(u_s):
    return inv(F_func(u_s))

def J_s_func(u_s):
    return det(F_func(u_s)) * (1 + u_s[1] / r)

def Sigma_s_func(u_s):
    return (F_func(u_s) - H_func(u_s)) + lambda_s * (J_s_func(u_s) - 1) * J_s_func(u_s) * H_func(u_s)

# (non-dim) PK1 stress tensor and incompressibility condition
# Sigma_s = (F - H) - p_s * H
# ic_s = det(F) * (1 + u_s[1] / r) - 1

# solid deformation linear
# Sigma_s = -p_s * I + (grad(u_s) + grad(u_s).T)
# ic_s = diva(u_s, r)


# ---------------------------------------------------------------------
# build equations
# ---------------------------------------------------------------------

# compressible
FUN3 = (-inner(Sigma_s, grad(v_s)) * r * dx
        - 1 * (1 + u_s[1] / r) * v_s[1] * dx
        + (1 - lambda_s * (J_s - 1) * J_s) * v_s[1] * r / (r + u_s[1]) * dx
        + inner(as_vector([f_0, 0]), v_s) * r * dx(solid)  # net z disp = 0
        + dot(ez, u_s) * g_0 * r * dx(solid)  # net z disp = 0
        # + inner(solid_pressure, v_s) * r * dx  # artificial pressure traction on disc
        + inner(solid_traction, v_s) * r * ds(circle))  # artificial traction

# non-axisym
# FUN3 = (-inner(Sigma_s, grad(v_s)) * dx
#         + inner(solid_pressure, v_s) * dx  # artificial pressure traction on disc
#         - inner(solid_traction, v_s) * ds)  # artificial traction

# Incompressibility for the solid
# FUN4 = ic_s * q_s * r * dx(solid)
JAC = derivative(FUN3, u_s, u_tr)

# ---------------------------------------------------------------------
# set up the solver
# ---------------------------------------------------------------------
problem = NonlinearVariationalProblem(FUN3, state, bc, JAC)
solver = NonlinearVariationalSolver(problem)
solver.solve()

# solve(FUN3 == 0, state, bc, J=JAC,
#       form_compiler_parameters=ffc_options)

# Create function spaces for the displacement and stress
Vs = VectorFunctionSpace(mesh, "CG", 1)
P1v = VectorFunctionSpace(mesh, "DG", 0)
cs = x[0] / sqrt(x[0] ** 2 + x[1] ** 2)
sn = x[1] / sqrt(x[0] ** 2 + x[1] ** 2)
A = as_tensor([[cs, sn], [-sn, cs]])
sig_s = project((A * Sigma_s_func(u_s) * A.T) * as_vector([1, 0]), P1v)

u_s_only = Function(Vs)


# Python function to save solution for a given value of epsilon
def save(eps):
    u_s_only = project(u_s, Vs)
    u_s_only.rename("u_s", "u_s")
    sig_s.rename("sigma", "sigma")
    print(u_s(0, 1)[1])
    output_s.write(u_s_only, eps)
    output_s.write(sig_s, eps)

save(eps)