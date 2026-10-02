# StructureFactorCertify

Package used to produce numerical results and plots for the project "Convergent hierarchy of bounds for the structure factor of spin systems".

We acknowledge the use of generative AI (Anthropic Claude Opus 5.5 "Medium") for building the package from unpolished code and notes. While most functionalities were working in our test environment before giving it to the AI, the optimizations found by the agent (the `symmetries.jl` part) allowed for considerable speed-up for the plot generation.

## Package layout

This repository follows a standard Julia package structure:

- `Project.toml` — package metadata and test targets
- `src/` — package source code
  - `pauli.jl` — `PauliMonomial`, `PauliPolynomial`
  - `interactions.jl` — interactions, lattices, Hamiltonians
  - `symmetries.jl` — detection of time reversal and global Z₂ symmetries
  - `moment_map.jl` — moment variables ρ(Q), created lazily, and symmetry-blocked PSD constraints
  - `relaxation.jl` — `Relaxation` parameters, KMS conditions, construction of the SDP
  - `problems/` — `GS_Energy_Problem`, `GS_Observable_Problem`, `StructureFactor_Problem`
- `examples/` — runnable example scripts (own environment, with the solvers)
- `test/` — unit tests (require a Mosek license)

## Quick start

```julia
using StructureFactorCertify
using Dualization, Mosek, MosekTools

system = Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, 0.5))
prob   = StructureFactor_Problem(system, tf_single_correlator(1))   # bounds on ⟨Z₀Z₁⟩
relax  = Relaxation(L = 5, d = 2, kms = CommutatorKMS())            # or AnticommutatorKMS()

(lo, hi), model = compute_bounds(prob, relax; optimizer = Dualization.dual_optimizer(Mosek.Optimizer))
```

To run the examples, from the repository root:

```bash
julia --project=examples -e "using Pkg; Pkg.instantiate()"
julia --project=examples examples/structure_factor.jl
```

