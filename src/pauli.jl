## ---------------
# Pauli algebra for working with qubit lattices
# Letters are encoded as 0,1,2,3 ↔ I,X,Y,Z
# ----------------

"Numerical tolerance under which a coefficient is treated as zero."
const ZERO_TOL = 1e-9

"""
    pauli_algebra_rule(a, b) -> (phase::ComplexF64, c::Int)

Single-site product σ_a σ_b = phase · σ_c. For example X*Y = iZ, so `pauli_algebra_rule(1, 2) == (im, 3)`.
"""
function pauli_algebra_rule(a::Int, b::Int)
    (a == 0 || b == 0) && return (1.0 + 0im, a + b)
    a == b && return (1.0 + 0im, 0)
    return ((a, b) in ((1, 2), (2, 3), (3, 1)) ? 1.0im : -1.0im, 6 - (a + b))
end


# -------------------------
# PauliMonomial
# -------------------------

"""
    PauliMonomial(n, term)

Pauli string σ_{term[1]} ⊗ … ⊗ σ_{term[n]} on `n` qubits, with `term[i] ∈ {0,1,2,3}`.
"""
struct PauliMonomial
    n::Int              # number of qubits
    term::Vector{Int}   # length n, entries in {0,1,2,3} for I,X,Y,Z
end

Base.string(p::PauliMonomial) = join(("IXYZ"[t + 1] for t in p.term), " ⊗ ")

"""
    p1 * p2 -> (phase::ComplexF64, q::PauliMonomial)

Product of two Pauli strings, p1 p2 = phase · q.
"""
function Base.:*(p1::PauliMonomial, p2::PauliMonomial)
    p1.n == p2.n || error("PauliMonomials must have same number of qubits")
    phase = 1.0 + 0im
    term = Vector{Int}(undef, p1.n)
    for i in 1:p1.n
        k, t = pauli_algebra_rule(p1.term[i], p2.term[i])
        phase *= k
        term[i] = t
    end
    return phase, PauliMonomial(p1.n, term)
end

"""
    unique_id(p::PauliMonomial) -> Int

Base-4 integer encoding of a Pauli string (the identity has id 0). Inverse of `pauli_from_id`.
"""
function unique_id(p::PauliMonomial)
    s = 0
    for i in 1:p.n
        s += p.term[i] * (4^(i-1))
    end
    return s
end

"""
    pauli_from_id(n, uid) -> PauliMonomial

Pauli string on `n` qubits with id `uid`. Inverse of `unique_id`.
"""
function pauli_from_id(n::Int, uid::Int)
    term = Vector{Int}(undef, n)
    for i in 1:n
        term[i] = uid % 4
        uid ÷= 4
    end
    return PauliMonomial(n, term)
end

Base.copy(p::PauliMonomial) = PauliMonomial(p.n, copy(p.term))
Base.conj(p::PauliMonomial) = copy(p)  # self-adjoint


# -------------------------
# PauliPolynomial
# -------------------------

"""
    PauliPolynomial(n, coefs)

Linear combination ∑ c_Q Q of Pauli strings on `n` qubits, stored as `coefs[unique_id(Q)] = c_Q`.
"""
struct PauliPolynomial
    n::Int                          # number of qubits
    coefs::Dict{Int,ComplexF64}     # uid -> coefficient
end

function Base.string(p::PauliPolynomial)
    isempty(p.coefs) && return "0"
    return join(("($c) * $(string(pauli_from_id(p.n, uid)))" for (uid, c) in p.coefs), " + ")
end

"""
    from_monomial(m::PauliMonomial, coef = 1) -> PauliPolynomial

The polynomial `coef · m`.
"""
from_monomial(m::PauliMonomial, coef::Number = 1.0) = PauliPolynomial(m.n, Dict(unique_id(m) => ComplexF64(coef)))

"Drop the coefficients of `p` whose modulus is below `tol`."
function remove_zeros(p::PauliPolynomial; tol = ZERO_TOL)
    return PauliPolynomial(p.n, Dict(uid => c for (uid, c) in p.coefs if abs(c) ≥ tol))
end

# Convert a number, monomial or polynomial into a polynomial with the same number of qubits as `p`
_as_polynomial(p::PauliPolynomial, x::Number) = PauliPolynomial(p.n, Dict(0 => ComplexF64(x)))
function _as_polynomial(p::PauliPolynomial, m::PauliMonomial)
    p.n == m.n || error("PauliPolynomials must have the same number of qubits")
    return from_monomial(m)
end
function _as_polynomial(p::PauliPolynomial, q::PauliPolynomial)
    p.n == q.n || error("PauliPolynomials must have the same number of qubits")
    return q
end

function Base.:+(p::PauliPolynomial, q::PauliPolynomial)
    coefs = copy(p.coefs)
    for (uid, c) in q.coefs
        coefs[uid] = get(coefs, uid, 0.0 + 0im) + c
    end
    return remove_zeros(PauliPolynomial(p.n, coefs))
end

Base.:+(p::PauliPolynomial, x) = p + _as_polynomial(p, x)
Base.:+(x, p::PauliPolynomial) = p + x

function Base.:*(p::PauliPolynomial, q::PauliPolynomial)
    coefs = Dict{Int,ComplexF64}()
    for (uid1, c1) in p.coefs
        m1 = pauli_from_id(p.n, uid1)
        for (uid2, c2) in q.coefs
            k, m = m1 * pauli_from_id(p.n, uid2)
            uid = unique_id(m)
            coefs[uid] = get(coefs, uid, 0.0 + 0im) + k * c1 * c2
        end
    end
    return PauliPolynomial(p.n, coefs)
end

Base.:*(p::PauliPolynomial, x) = p * _as_polynomial(p, x)
Base.:*(x, p::PauliPolynomial) = _as_polynomial(p, x) * p

Base.:-(p::PauliPolynomial, x) = p + (-1) * _as_polynomial(p, x)

"""
    commutator(A, B) -> PauliPolynomial

The commutator [A, B] = AB - BA of two Pauli polynomials.
"""
commutator(A::PauliPolynomial, B::PauliPolynomial) = A * B - B * A
