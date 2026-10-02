using Test
using StructureFactorCertify
using Dualization, Mosek, MosekTools

const OPTIMIZER = Dualization.dual_optimizer(Mosek.Optimizer)

@testset "StructureFactorCertify" begin

    @testset "Pauli algebra" begin
        X, Y = PauliMonomial(1, [1]), PauliMonomial(1, [2])
        phase, Q = X * Y
        @test phase == 1.0im && Q.term == [3]     # XY = iZ
        phase, Q = Y * X
        @test phase == -1.0im && Q.term == [3]    # YX = -iZ
        P = PauliMonomial(3, [0, 2, 3])
        @test pauli_from_id(3, unique_id(P)).term == P.term
    end

    @testset "Symmetry detection, TFIM" begin
        H = get_hamiltonian(TFIM_1D_Interaction(1.0, 0.5), 5)
        sym = detect_symmetries([H, build_structure_factor_observable(tf_single_correlator(1), 5)])
        @test sym.time_reversal
        @test [s.name for s in sym.z2] == ["∏X"]
    end

    # 2-site chain, d = 2: the basis spans the whole algebra, so the bound is the exact minimal eigenvalue
    @testset "GS energy, TFIM, L=2" begin
        g = 0.75
        E_min, _ = compute_bounds(GS_Energy_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g))),
                                  Relaxation(L = 2, d = 2); optimizer = OPTIMIZER, verbose = false)
        @test E_min ≈ -sqrt(1 + 4g^2) atol = 1e-6
    end

    @testset "GS correlator ⟨Z₂Z₃⟩, TFIM, L=4" begin
        g, L = 0.8, 4
        prob = GS_Observable_Problem(Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g)),
                                     from_monomial(PauliMonomial(L, [0, 3, 3, 0])))
        (lo, hi), _ = compute_bounds(prob, Relaxation(L = L, d = 2); optimizer = OPTIMIZER, verbose = false)
        @test lo ≤ tfim_nn_correlator_numerical(g) ≤ hi
    end

    system = Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, 0.5))
    prob = StructureFactor_Problem(system, tf_single_correlator(1))
    exact = tfim_nn_correlator_numerical(0.5)

    @testset "Structure factor ⟨Z₀Z₁⟩, TFIM g=0.5, L=5, d=2" begin
        (lo, hi), _ = compute_bounds(prob, Relaxation(L = 5, d = 2); optimizer = OPTIMIZER, verbose = false)
        @test lo ≤ exact ≤ hi
        @test lo ≈ 0.2879368 atol = 1e-5   # values of structure_factor_optimized.jl
        @test hi ≈ 0.9372561 atol = 1e-5
    end

    @testset "Symmetry reduction leaves the bounds unchanged" begin
        relax = Relaxation(L = 3, d = 2)
        (lo, hi), _ = compute_bounds(prob, relax; optimizer = OPTIMIZER, verbose = false)
        (lo0, hi0), _ = compute_bounds(prob, Relaxation(L = 3, d = 2, use_time_reversal = false, use_parity = false);
                                       optimizer = OPTIMIZER, verbose = false)
        @test lo ≈ lo0 atol = 1e-5
        @test hi ≈ hi0 atol = 1e-5
    end

    # Both KMS matrices agree up to a factor 2; the double-monomial stationarity contains the single one
    @testset "Anticommutator KMS is at least as tight as commutator KMS" begin
        (lo, hi), _ = compute_bounds(prob, Relaxation(L = 5, d = 2, kms = CommutatorKMS()); optimizer = OPTIMIZER, verbose = false)
        (lo_a, hi_a), _ = compute_bounds(prob, Relaxation(L = 5, d = 2, kms = AnticommutatorKMS()); optimizer = OPTIMIZER, verbose = false)
        @test lo_a ≥ lo - 1e-6
        @test hi_a ≤ hi + 1e-6
        @test lo_a ≤ exact ≤ hi_a
    end

end
