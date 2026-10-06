using Test
using StructureFactorCertify
using JuMP
using Dualization, Mosek, MosekTools

const SFC = StructureFactorCertify
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

    g = 0.5
    system = Spin_Lattice_1D(2, TFIM_1D_Interaction(1.0, g))
    prob   = StructureFactor_Problem(system, tf_single_correlator(1))   # ⟨Z₀Z₁⟩
    exact  = tfim_nn_correlator_numerical(g)
    bounds(relax) = compute_bounds(prob, relax; optimizer = OPTIMIZER, verbose = false)

    @testset "Structure factor ⟨Z₀Z₁⟩, TFIM g=0.5, L=5, d=2" begin
        (lo, hi), _ = bounds(Relaxation(L = 5, d = 2))
        @test lo ≤ exact ≤ hi
        @test lo ≈ 0.2879368 atol = 1e-5   # values of structure_factor_optimized.jl
        @test hi ≈ 0.9372561 atol = 1e-5
    end

    @testset "Symmetry reduction leaves the bounds unchanged" begin
        (lo, hi), _   = bounds(Relaxation(L = 3, d = 2))
        (lo0, hi0), _ = bounds(Relaxation(L = 3, d = 2, use_time_reversal = false, use_parity = false))
        @test lo ≈ lo0 atol = 1e-5
        @test hi ≈ hi0 atol = 1e-5
    end

    # Both KMS matrices agree up to a factor 2; the double-monomial stationarity contains the single one
    @testset "Anticommutator KMS is at least as tight as commutator KMS" begin
        (lo, hi), _     = bounds(Relaxation(L = 5, d = 2, kms = CommutatorKMS()))
        (lo_a, hi_a), _ = bounds(Relaxation(L = 5, d = 2, kms = AnticommutatorKMS()))
        @test lo_a ≥ lo - 1e-6
        @test hi_a ≤ hi + 1e-6
        @test lo_a ≤ exact ≤ hi_a
    end

    ## Translation invariance and contiguous basis, 1D TFIM

    @testset "translate_to_origin" begin
        @test SFC.translate_to_origin(system, 5, [0, 3, 3, 0, 0]) == [3, 3, 0, 0, 0]
        @test SFC.translate_to_origin(system, 5, [0, 0, 1, 0, 2]) == [1, 0, 2, 0, 0]
        @test SFC.translate_to_origin(system, 5, [1, 0, 0, 0, 3]) == [1, 0, 0, 0, 3]
        @test SFC.translate_to_origin(system, 5, zeros(Int, 5))   == zeros(Int, 5)
    end

    @testset "Contiguous basis" begin
        L = 5
        full       = SFC.build_monomial_basis(system, L, 2; basis = "full")
        contiguous = SFC.build_monomial_basis(system, L, 2; basis = "contiguous")
        @test length(full) == 1 + 5*3 + 10*9         # identity, 1-site, all pairs
        @test length(contiguous) == 1 + 5*3 + 4*9    # identity, 1-site, nearest-neighbour pairs
        @test length(SFC.build_monomial_basis(system, L, 2; edge = false, basis = "contiguous")) == 1 + 3*3 + 2*9
        # supports fit in d consecutive sites
        for d in 1:3, P in SFC.build_monomial_basis(system, L, d; basis = "contiguous")
            occupied = findall(!=(0), P.term)
            @test isempty(occupied) || last(occupied) - first(occupied) < d
        end
        # with d = L, "contiguous" and "full" coincide
        @test Set(unique_id.(SFC.build_monomial_basis(system, L, L; basis = "contiguous"))) ==
              Set(unique_id.(SFC.build_monomial_basis(system, L, L; basis = "full")))
        @test_throws ErrorException SFC.build_monomial_basis(system, L, 2; basis = "unknown")
    end

    @testset "Translation invariance: fewer variables, tighter or equal bounds" begin
        (lo, hi), model          = bounds(Relaxation(L = 5, d = 2))
        (lo_ti, hi_ti), model_ti = bounds(Relaxation(L = 5, d = 2, use_translation = true))
        @test num_variables(model_ti) < num_variables(model)
        @test lo_ti ≥ lo - 1e-6
        @test hi_ti ≤ hi + 1e-6
        @test lo_ti ≤ exact ≤ hi_ti
    end

    @testset "Contiguous basis: valid bounds" begin
        (lo, hi), _ = bounds(Relaxation(L = 5, d = 2, basis = "contiguous"))
        @test lo ≤ exact ≤ hi
        # with d = L, same bounds as the full basis
        (lo_c, hi_c), _ = bounds(Relaxation(L = 4, d = 4, basis = "contiguous"))
        (lo_f, hi_f), _ = bounds(Relaxation(L = 4, d = 4, basis = "full"))
        @test lo_c ≈ lo_f atol = 1e-5
        @test hi_c ≈ hi_f atol = 1e-5
    end

    @testset "Translation invariance + contiguous basis, both KMS conditions" begin
        for kms in (CommutatorKMS(), AnticommutatorKMS())
            (lo, hi), _ = bounds(Relaxation(L = 7, d = 3, basis = "contiguous", kms = kms, use_translation = true))
            @test lo ≤ exact ≤ hi
        end
    end

    @testset "Translation invariance with symmetries disabled gives the same bounds" begin
        (lo, hi), _   = bounds(Relaxation(L = 3, d = 2, use_translation = true))
        (lo0, hi0), _ = bounds(Relaxation(L = 3, d = 2, use_translation = true, use_time_reversal = false, use_parity = false))
        @test lo ≈ lo0 atol = 1e-5
        @test hi ≈ hi0 atol = 1e-5
    end

    ## Reduced density matrices

    @testset "RDM boxes" begin
        @test length(SFC.rdm_boxes(system, 5, 3, false)) == 3                      # segments 1:3, 2:4, 3:5
        @test length(SFC.rdm_boxes(system, 5, 3, true)) == 1
        system_2d = Square_Lattice_2D(2, TFIM_2D_Interaction(1.0, g))
        @test sort(length.(SFC.rdm_boxes(system_2d, 3, 2, false))) == [4, 4, 4, 4]   # four 2x2 squares
        @test_throws ErrorException SFC.rdm_boxes(system, 3, 4, false)
    end

    # The full d=2 basis contains every string on 2 sites, so M ≽ 0 already implies ρ_S ≽ 0
    @testset "RDM covered by the basis is redundant" begin
        (lo, hi), _         = bounds(Relaxation(L = 5, d = 2))
        (lo_r, hi_r), _     = bounds(Relaxation(L = 5, d = 2, rdm_size = 2))
        @test lo_r ≈ lo atol = 1e-5
        @test hi_r ≈ hi atol = 1e-5
    end

    @testset "RDM larger than the basis: tighter or equal, valid" begin
        (lo, hi), _     = bounds(Relaxation(L = 5, d = 2))
        (lo_r, hi_r), _ = bounds(Relaxation(L = 5, d = 2, rdm_size = 4))
        @test lo_r ≥ lo - 1e-6
        @test hi_r ≤ hi + 1e-6
        @test lo_r ≤ exact ≤ hi_r
    end

    @testset "RDM + translation invariance + contiguous basis" begin
        (lo, hi), _ = bounds(Relaxation(L = 7, d = 2, basis = "contiguous", rdm_size = 4, use_translation = true))
        @test lo ≤ exact ≤ hi
    end

    # RDM on the whole 2x2 window: the bound is the exact ground energy of the plaquette
    @testset "RDM on the whole window gives the exact energy, 2D TFIM 2x2" begin
        system_2d = Square_Lattice_2D(2, TFIM_2D_Interaction(1.0, 0.7))
        E_min, _ = compute_bounds(GS_Energy_Problem(system_2d), Relaxation(L = 2, d = 1, rdm_size = 2);
                                  optimizer = OPTIMIZER, verbose = false)
        @test E_min ≈ exact_ground_energy(get_hamiltonian(system_2d.interaction, 4)) atol = 1e-6
    end

end
