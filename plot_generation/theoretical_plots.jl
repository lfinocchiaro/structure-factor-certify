#=
Exact thermodynamic-limit benchmarks for the 1D TFIM  H = Σ_x (-Z_x Z_{x+1} + g X_x).

  f(x)  = ⟨Z_0 Z_x⟩_GS                       Toeplitz determinant (Pfeuty 1970, Barouch–McCoy 1971)
  f̂(p)  = m² δ(p) + f̂_reg(p),  m² = (1-g²)^{1/4} for g<1, 0 otherwise
  O_N   = (f̂, φ̂_N) = Σ_{|x|≤N} φ_N(x) f(x)    exact finite sum = SDP target value

Test function: Fejér kernel, normalised so that φ̂_N(0) = 1,
  φ_N(x) = (1 - |x|/(N+1)) / (N+1),   φ̂_N(p) = [sin((N+1)πp) / ((N+1) sin πp)]².

Dependencies: Plots, LaTeXStrings  (] add Plots LaTeXStrings)
=#
using LinearAlgebra, Printf, Plots, LaTeXStrings
using DelimitedFiles

const SPIN_FACTOR = 1.0   # 1.0 for Pauli Z; set to 0.25 if Z = S^z = σ^z/2

# ---------------- exact correlation function ----------------
"""a_n for n = -nmax:nmax, returned as a vector indexed by n + nmax + 1.
a_n = (1/2π) ∫ e^{-ink} (1 - g e^{ik})/|1 - g e^{ik}| dk  (trapezoid rule, spectrally accurate for g ≠ 1)."""
function symbol_coeffs(g, nmax; M = 2^16)
    n = -nmax:nmax
    isapprox(g, 1.0) && return [2 / (π * (1 - 2m)) for m in n]   # exact at criticality
    k = 2π .* (0:M-1) ./ M
    s = @. (1 - g * cis(k)) / abs(1 - g * cis(k))
    a = zeros(length(n))
    a[nmax+1] = real(sum(s)) / M
    zp, zm = copy(s), copy(s)
    em, ep = cis.(-k), cis.(k)
    for m in 1:nmax                       # build s·e^{∓imk} incrementally (avoids cis in the loop)
        zp .*= em; a[nmax+1+m] = real(sum(zp)) / M
        zm .*= ep; a[nmax+1-m] = real(sum(zm)) / M
    end
    return a
end

"""f(r) = ⟨Z_0 Z_r⟩ for r = 0:R (returned with f[r+1] = f(r))."""
function corr(g, R)
    a = symbol_coeffs(g, R)
    T = [a[(i - j) + R + 1] for i in 1:R, j in 1:R]      # T_ij = a_{i-j}
    f = [1.0; [det(T[1:r, 1:r]) for r in 1:R]]
    return SPIN_FACTOR .* f
end

m2(g) = g < 1 ? SPIN_FACTOR * (1 - g^2)^(1/4) : 0.0

# ---------------- structure factor (regular part) ----------------
"""Σ_x (f(x) - m²) cos(2πpx), |x| ≤ R. Cesàro (Fejér) window removes Gibbs ringing (used at g = 1)."""
function S_reg(g, p; R = 400, cesaro = false, f = corr(g, R))
    x = 0:R
    w = [r == 0 ? 1.0 : 2.0 for r in x]
    cesaro && (w .*= 1 .- x ./ (R + 1))
    fr = f .- m2(g)
    return [sum(w .* fr .* cos.(2π * pp .* x)) for pp in p]
end

# ---------------- Fejér test function ----------------
fejer(N) = (-N:N, [(1 - abs(x) / (N + 1)) / (N + 1) for x in -N:N])
fejer_hat(N, p) = (xs = fejer(N); [sum(xs[2] .* cos.(2π * pp .* xs[1])) for pp in p])

function overlap(g, N; f = corr(g, N))
    xs, c = fejer(N)
    return sum(c[i] * f[abs(xs[i]) + 1] for i in eachindex(xs))
end

# ---------------- self-check against the Python reference ----------------
function selfcheck()
    ref = [(0.5, 4, 0.9457450592), (0.5, 16, 0.9351347149), (1.0, 4, 0.6481440454),
           (1.0, 16, 0.4802440636), (1.5, 4, 0.3789264881), (1.5, 16, 0.1394296768)]
    for (g, N, v) in ref
        o = overlap(g, N) / SPIN_FACTOR
        @printf("g=%.1f N=%2d  O_N=%.10f  ref=%.10f  %s\n", g, N, o, v, abs(o - v) < 1e-8 ? "ok" : "MISMATCH")
    end
end

# ---------------- Figure 1: three panels ----------------
function panel(g; N = 8, R = 400)
    p = range(-0.5, 0.5, length = 801)
    crit = isapprox(g, 1.0)
    f = corr(g, R)
    S = S_reg(g, p; R = R, cesaro = crit, f = f)
    ymax = 3.0 # crit ? 6.0 : 1.3 * maximum(S)
    O = overlap(g, N; f = f)

    pl = plot(p, S; lw = 2, color = :firebrick, xlabel = L"p", ylabel = L"\hat f(p)",
              ylims = (0, ymax), xlims = (-0.5, 0.5), legend = :topright, framestyle = :box,
              label = L"\hat f(p)", #crit ? "Cesàro-summed, R=$R (true: ∼|p|^-3/4)" : L"\hat f_{\rm reg}(p)",
              title = @sprintf("g = %.1f    O_%d = %.4f", g, N, O))
    # Fejér kernel, rescaled to 80% of the panel height (true peak value: 1)
    plot!(pl, p, 2.0 .* fejer_hat(N, p); fillrange = 0, fillalpha = 0.15, lw = 1.5,
          ls = :dash, color = :black, label = "Fejér kernel N=$N (×2)")
    if g < 1   # Dirac peak: arrow, weight m² written next to it
        plot!(pl, [0, 0], [0, 0.92ymax]; arrow = true, lw = 3, color = :firebrick,
        label = @sprintf("m² δ(p),  m² = %.4f", m2(g)))
    end
    return pl
end

function figure_panels(; N = 8, g_vals = [0.9, 1.0, 2.0])
    plt = plot(panel(g_vals[1]; N = N), panel(g_vals[2]; N = N), panel(g_vals[3]; N = N);
               layout = (1, 3), size = (1600, 480), margin = 6Plots.mm)
    savefig(plt, joinpath(@__DIR__, "..", "plots", "tfim_panels_fejer_$(Libc.strftime("%d%m_%Hh%M", time())).png"))
    return plt
end

# ---------------- Figure 2: target overlaps vs g ----------------
function figure_overlap(; Ns = [2, 4, 8, 16, 32, 64], gg = range(0.05, 2.5, length = 120))
    Nmax = maximum(Ns)
    F = [corr(g, Nmax) for g in gg]                  # one determinant sweep per g, reused for all N
    plt = plot(xlabel = L"g", ylabel = L"O_N(g) = \sum_x \varphi_N(x) f(x)",
               title = "SDP target values (Fejér)", size = (700, 500), framestyle = :box)
    cols = palette(:viridis, length(Ns) + 1)
    for (i, N) in enumerate(Ns)
        plot!(plt, gg, [overlap(g, N; f = F[j]) for (j, g) in enumerate(gg)];
              lw = 2, color = cols[i], label = "N = $N")
    end
    plot!(plt, gg, m2.(gg); lw = 2.5, ls = :dash, color = :black, label = L"N\to\infty:\ (1-g^2)^{1/4}")
    savefig(plt, joinpath(@__DIR__, "..", "plots", "tfim_overlap_vs_g_$(Libc.strftime("%d%m_%Hh%M", time())).png"))
    return plt
end


# -------------- Generate table with theoretical values for the joint plot with numerical bounds --------------
# Store overlaps (f^, φ̂_N) = Σ_{|x|≤N} φ_N(x) f(x) for several N and g, and store it in a file
function table_theoretical(; N_list=[1,2,3], g_list=range(0.05, 2.5, length=30))
    filename = joinpath(@__DIR__, "table_theoretical_$(Libc.strftime("%d%m_%Hh%M", time())).csv")

    F = [corr(g, maximum(N_list)) for g in g_list]                  # one determinant sweep per g, reused for all N
    data = [(overlap(g, N; f = F[j])) for (j, g) in enumerate(g_list), N in N_list]


    # --- Write ---
    open(filename, "w") do io
        writedlm(io, permutedims(["g/N"; string.(N_list)]), ',')   # header: corner label + y values
        writedlm(io, hcat(g_list, data), ',')                      # each row: x value, then data[i, :]
    end

    # # --- Read back ---
    # M, hdr = readdlm(filename, ',', Float64; header=true)
    # g_read    = M[:, 1]
    # N_read    = parse.(Float64, vec(hdr)[2:end])
    # data_read = M[:, 2:end]
end




#selfcheck()
#figure_panels(N=12, g_vals=[0.8, 1.0, 2.5])
#figure_overlap()

# table_theoretical()
