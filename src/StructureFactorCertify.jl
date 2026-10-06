# -------------------------------------------
# The goal of this package is to provide a structured framework to deal with spin lattices in the thermodynamic limit, and use semidefinite relaxation tools (based on the NPA hierarchy) to compute bounds on relevant quantities, such as the ground state energy, expectation value of observables in the ground state, or dynamical quantities such as the structure factor
# -------------------------------------------

module StructureFactorCertify

using JuMP
using Combinatorics
using LinearAlgebra
using Printf

# Pauli algebra
export PauliMonomial, PauliPolynomial
export unique_id, pauli_from_id, from_monomial, commutator

# Interactions, lattices and Hamiltonians
export Interaction, TFIM_1D_Interaction, XY_1D_Interaction, Heisenberg_1D_Interaction, TFIM_2D_Interaction
export interaction_range, get_hamiltonian, local_term
export Spin_Lattice, Spin_Lattice_1D, Square_Lattice_2D, num_sites

# Symmetries and moment variables
export Z2Symmetry, Symmetries, detect_symmetries
export MomentMap

# Relaxation
export Relaxation, KMSCondition, CommutatorKMS, AnticommutatorKMS
export Abstract_Problem, compute_bounds

# Problems
export GS_Energy_Problem, GS_Observable_Problem
export StructureFactor_Problem, TestFunction, phi_at, build_structure_factor_observable
export tf_dirichlet, tf_point_mass, tf_gaussian, tf_custom, tf_single_correlator

# Reference values
export tfim_nn_correlator_numerical, dense_matrix, exact_ground_energy

include("pauli.jl")
include("interactions.jl")
include("symmetries.jl")
include("moment_map.jl")
include("relaxation.jl")
include("reduced_density_matrix.jl")
include("problems/gs_energy.jl")
include("problems/gs_observable.jl")
include("problems/structure_factor.jl")
include("reference_values.jl")

end
