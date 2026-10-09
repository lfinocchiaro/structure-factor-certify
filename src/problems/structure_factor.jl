## -------
# Structure factor
#
# For a test function φ̂ ∈ S_N (its inverse Fourier transform φ is supported in {-N,…,N}), bound
#   ⟨f̂, φ̂⟩ = ∑_{x=-N}^{N} φ(-x) ρ(Z_0 Z_x),
# where f(x) = ρ(Z_0 Z_x) is the two-point correlation function of the ground state.
# --------

"""
    TestFunction(N, phi)

Test function φ̂ ∈ S_N, stored through its inverse Fourier transform: `phi[k] = φ(k - N - 1)` for
x = -N,…,N (so `phi[N+1] = φ(0)`).
"""
struct TestFunction
    N::Int
    phi::Vector{Float64}
end

"φ(x) for any integer x (zero outside {-N,…,N})."
phi_at(tf::TestFunction, x::Int) = abs(x) > tf.N ? 0.0 : tf.phi[x + tf.N + 1]

function Base.show(io::IO, ::MIME"text/plain", tf::TestFunction)
    println(io, "TestFunction with N=$(tf.N)")
    for x in -tf.N:tf.N
        @printf(io, "  φ(%3d) = %+.6f\n", x, phi_at(tf, x))
    end
end

## Test function constructors

"`tf_dirichlet(N)`: φ(x) = 1 for |x| ≤ N, i.e. φ̂ is the Dirichlet kernel of order N."
tf_dirichlet(N::Int) = TestFunction(N, ones(Float64, 2N+1))

"`tf_point_mass(N, p0)`: φ(x) = cos(2π p0 x), approximating a point mass at momentum p0 ∈ [0,1]."
tf_point_mass(N::Int, p0::Real) = TestFunction(N, [cos(2π * p0 * x) for x in -N:N])

"""
`tf_fejer(N, p₀=0.0, φ₀=1.0)`: N-th Fejér kernel centred at ±p₀,
  φ(x) = φ₀ (1-|x|/(N+1))/(N+1) cos(2πp₀x),   φ̂(p) = φ₀/2 [K_N(p-p₀) + K_N(p+p₀)],
with K_N(p) = [sin((N+1)πp) / ((N+1) sin πp)]² ≥ 0 and K_N(0) = 1. For p₀ ∈ {0, 1/2}, φ̂(p₀) = φ₀.
Since f is even, ⟨f̂, φ̂⟩ equals the pairing with the complex kernel φ₀ K_N(p-p₀) alone.
"""
tf_fejer(N::Int, p₀::Real = 0.0, φ₀::Real = 1.0) = TestFunction(N, [φ₀ * (1 - abs(x)/(N+1))/(N+1) * cospi(2p₀ * x) for x in -N:N])

"`tf_custom(phi_vec)`: test function from a vector of odd length 2N+1, with `phi_vec[k] = φ(k-N-1)`."
function tf_custom(phi_vec::Vector{Float64})
    isodd(length(phi_vec)) || error("phi_vec must have odd length")
    return TestFunction((length(phi_vec) - 1) ÷ 2, phi_vec)
end

"`tf_single_correlator(x)`: φ(x) = 1 at a single point, so that ⟨f̂, φ̂⟩ = ρ(Z_0 Z_x)."
function tf_single_correlator(x::Int)
    phi = zeros(Float64, 2x+1)
    phi[end] = 1.0   # φ(x) = 1: ⟨f̂, φ̂⟩ = ρ(Z_0 Z_{-x}) = ρ(Z_0 Z_x) by translation invariance
    return TestFunction(x, phi)
end


"""
    StructureFactor_Problem(system, tf)

Upper and lower bounds on ⟨f̂, φ̂⟩ = ∑_x φ(-x) ρ(Z_0 Z_x) in the ground state of `system`, for the test function `tf`.
"""
struct StructureFactor_Problem{T<:Spin_Lattice} <: Abstract_Problem
    system::T
    tf::TestFunction
end

"""
    build_structure_factor_observable(tf, L) -> PauliPolynomial

The observable ∑_{x=-N}^{N} φ(-x) Z_{x₀} Z_{x₀+x} on a chain of `L` ≥ 2N+1 sites, centred at x₀ = L÷2 + 1.
"""
function build_structure_factor_observable(tf::TestFunction, L::Int)
    N  = tf.N
    x0 = L ÷ 2 + 1   # reference site
    x0 - N ≥ 1 && x0 + N ≤ L || error("Chain too short for test function with N=$N. Need L ≥ $(2N+1), got L=$L.")
    coefs = Dict{Int,ComplexF64}()
    for x in -N:N
        c = phi_at(tf, -x)
        iszero(c) && continue
        term = zeros(Int, L)
        if x != 0          # for x = 0, Z_{x₀}² = I
            term[x0]   = 3
            term[x0+x] = 3
        end
        uid = unique_id(PauliMonomial(L, term))
        coefs[uid] = get(coefs, uid, 0.0 + 0im) + c
    end
    return PauliPolynomial(L, coefs)
end

"""
    compute_bounds(prob::StructureFactor_Problem, relax; optimizer, verbose = true) -> ((α_min, α_max), model)

Certified interval [α_min, α_max] ∋ ⟨f̂, φ̂⟩ from the relaxation `relax` (needs L ≥ 2N+1), solved with `optimizer`.
"""
function compute_bounds(prob::StructureFactor_Problem{<:Spin_Lattice_1D}, relax::Relaxation; optimizer, verbose::Bool = true)
    verbose && println("── Structure factor, $(prob.system.interaction), N=$(prob.tf.N), L=$(relax.L), d=$(relax.d) ──")
    observable = build_structure_factor_observable(prob.tf, relax.L)
    return observable_bounds(prob.system, observable, relax; optimizer, verbose)
end
