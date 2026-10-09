#=
Exact thermodynamic-limit benchmarks for f(x) = ⟨Z_0 Z_x⟩_GS of two free-fermion chains.

TFIM  H = Σ_x (-Z_x Z_{x+1} + g X_x)                       (model = :TFIM, parameter θ = g)
  f(x) = Toeplitz determinant (Pfeuty 1970, Barouch–McCoy 1971)
  f̂(p) = m² δ(p) + f̂_reg(p),  m² = (1-g²)^{1/4} for g<1, 0 otherwise

XY    H = Σ_x ((1+γ) X_x X_{x+1} + (1-γ) Y_x Y_{x+1})       (model = :XY, parameter θ = γ)
  f(x) = -G_x G_{-x} for x ≠ 0 (Wick, Lieb–Schultz–Mattis 1961 §4, Barouch–McCoy 1971 at h = 0)
  G_r  = (1/2π) ∫ e^{-ikr} (cos k - iγ sin k)/Λ_k dk,  Λ_k = √(cos²k + γ² sin²k)
  f(x) = 0 for even x ≠ 0, f is even in γ, and f̂ has no Dirac part (gapped for γ ≠ 0, f ∼ 1/x² at γ = 0)

O_N = (f̂, φ̂_N) = Σ_{|x|≤N} φ_N(x) f(x)    exact finite sum = SDP target value

Test function: Fejér kernel centred at ±p0 (same as tf_fejer in src), normalised so that φ̂_N(p0) = 1 for p0 ∈ {0, 1/2},
  φ_N(x) = (1 - |x|/(N+1)) / (N+1) · cos(2π p0 x),   φ̂_N(p) = [K_N(p-p0) + K_N(p+p0)]/2,
  K_N(p) = [sin((N+1)πp) / ((N+1) sin πp)]².

Dependencies: Plots, LaTeXStrings  (] add Plots LaTeXStrings)
=#
using LinearAlgebra, Printf, Plots, LaTeXStrings
using DelimitedFiles

const SPIN_FACTOR = 1.0   # 1.0 for Pauli Z; set to 0.25 if Z = S^z = σ^z/2

# Per-model plotting data: parameter name, critical point, default sweep, default test-function centre
const MODELS = Dict(
    :TFIM => (param = "g", label = L"g", crit = 1.0, range = range(0.05, 2.5, length = 30), p0 = 0.0),
    :XY   => (param = "γ", label = L"\gamma", crit = 0.0, range = range(0.0, 1.5, length = 31), p0 = 0.5),
)

# ---------------- Fourier coefficients of a symbol ----------------
"""c_n = (1/2π) ∫ e^{-ink} s(k) dk for n = -nmax:nmax, returned as a real vector indexed by n + nmax + 1
(trapezoid rule, spectrally accurate for smooth s). The symbols used here have real coefficients."""
function fourier_coeffs(s, nmax; M = 2^16)
    k = 2π .* (0:M-1) ./ M
    sk = s.(k)
    a = zeros(2nmax + 1)
    a[nmax+1] = real(sum(sk)) / M
    zp, zm = copy(sk), copy(sk)
    em, ep = cis.(-k), cis.(k)
    for m in 1:nmax                       # build s·e^{∓imk} incrementally (avoids cis in the loop)
        zp .*= em; a[nmax+1+m] = real(sum(zp)) / M
        zm .*= ep; a[nmax+1-m] = real(sum(zm)) / M
    end
    return a
end

# ---------------- TFIM: exact correlation function ----------------
"""a_n for n = -nmax:nmax, a_n = (1/2π) ∫ e^{-ink} (1 - g e^{ik})/|1 - g e^{ik}| dk."""
function tfim_symbol_coeffs(g, nmax; M = 2^16)
    isapprox(g, 1.0) && return [2 / (π * (1 - 2m)) for m in -nmax:nmax]   # exact at criticality
    return fourier_coeffs(k -> (1 - g * cis(k)) / abs(1 - g * cis(k)), nmax; M)
end

"""f(r) = ⟨Z_0 Z_r⟩ for r = 0:R (returned with f[r+1] = f(r))."""
function tfim_corr(g, R)
    a = tfim_symbol_coeffs(g, R)
    T = [a[(i - j) + R + 1] for i in 1:R, j in 1:R]      # T_ij = a_{i-j}
    f = [1.0; [det(T[1:r, 1:r]) for r in 1:R]]
    return SPIN_FACTOR .* f
end

tfim_m2(g) = g < 1 ? SPIN_FACTOR * (1 - g^2)^(1/4) : 0.0

# ---------------- XY: exact correlation function ----------------
"""G_r for r = -rmax:rmax (indexed by r + rmax + 1), G_r = (1/2π) ∫ e^{-ikr} (cos k - iγ sin k)/Λ_k dk.
At γ = 0 the symbol is sign(cos k) and G_r = 2 sin(πr/2)/(πr) in closed form. For small γ ≠ 0 the
trapezoid error is ∼ exp(-γM), negligible for γ ≳ 1e-3 at the default M."""
function xy_G(γ, rmax; M = 2^16)
    iszero(γ) && return [r == 0 ? 0.0 : 2 * sin(π * r / 2) / (π * r) for r in -rmax:rmax]
    return fourier_coeffs(k -> (cos(k) - im * γ * sin(k)) / sqrt(cos(k)^2 + γ^2 * sin(k)^2), rmax; M)
end

"""f(r) = ⟨Z_0 Z_r⟩ for r = 0:R (f[r+1] = f(r)); f(r) = ⟨Z⟩² - G_r G_{-r} with ⟨Z⟩ = 0 at zero field."""
function xy_corr(γ, R)
    G = xy_G(γ, R)
    f = [1.0; [-G[R+1+r] * G[R+1-r] for r in 1:R]]
    return SPIN_FACTOR .* f
end

xy_m2(γ) = 0.0   # no long-range order in Z: no Dirac part in f̂

# ---------------- model dispatch ----------------
corr(model::Symbol, θ, R) = model == :TFIM ? tfim_corr(θ, R) : model == :XY ? xy_corr(θ, R) : error("Unknown model $model")
m2(model::Symbol, θ)      = model == :TFIM ? tfim_m2(θ)      : model == :XY ? xy_m2(θ)      : error("Unknown model $model")

# ---------------- structure factor (regular part) ----------------
"""Σ_x (f(x) - m²) cos(2πpx), |x| ≤ R. Cesàro (Fejér) window removes Gibbs ringing (used at criticality)."""
function S_reg(model, θ, p; R = 400, cesaro = false, f = corr(model, θ, R))
    x = 0:R
    w = [r == 0 ? 1.0 : 2.0 for r in x]
    cesaro && (w .*= 1 .- x ./ (R + 1))
    fr = f .- m2(model, θ)
    return [sum(w .* fr .* cos.(2π * pp .* x)) for pp in p]
end

# ---------------- Fejér test function ----------------
fejer(N; p0 = 0.0) = (-N:N, [(1 - abs(x) / (N + 1)) / (N + 1) * cospi(2p0 * x) for x in -N:N])
fejer_hat(N, p; p0 = 0.0) = (xs = fejer(N; p0); [sum(xs[2] .* cos.(2π * pp .* xs[1])) for pp in p])

function overlap(model, θ, N; p0 = MODELS[model].p0, f = corr(model, θ, N))
    xs, c = fejer(N; p0)
    return sum(c[i] * f[abs(xs[i]) + 1] for i in eachindex(xs))
end

# ---------------- self-checks ----------------
function tfim_selfcheck()
    ref = [(0.5, 4, 0.9457450592), (0.5, 16, 0.9351347149), (1.0, 4, 0.6481440454),
           (1.0, 16, 0.4802440636), (1.5, 4, 0.3789264881), (1.5, 16, 0.1394296768)]
    for (g, N, v) in ref
        o = overlap(:TFIM, g, N; p0 = 0.0) / SPIN_FACTOR
        @printf("g=%.1f N=%2d  O_N=%.10f  ref=%.10f  %s\n", g, N, o, v, abs(o - v) < 1e-8 ? "ok" : "MISMATCH")
    end
end

"""Closed forms (γ = 0: f(r) = -4/(πr)² for odd r; γ = ±1: f(r) = δ_{r0}), the γ → 0 limit of the
trapezoid rule, and the symmetry γ → -γ."""
function xy_selfcheck(; R = 9)
    ok(b) = b ? "ok" : "MISMATCH"
    xx = [r == 0 ? 1.0 : isodd(r) ? -4 / (π * r)^2 : 0.0 for r in 0:R] .* SPIN_FACTOR
    @printf("γ=0     closed form XX          %s\n", ok(maximum(abs.(xy_corr(0.0, R) .- xx)) < 1e-12))
    ising = [r == 0 ? SPIN_FACTOR : 0.0 for r in 0:R]
    @printf("γ=±1    δ_{r0}                  %s\n", ok(maximum(abs.(xy_corr(1.0, R) .- ising)) < 1e-12 &&
                                                       maximum(abs.(xy_corr(-1.0, R) .- ising)) < 1e-12))
    @printf("γ=0.01  close to γ=0            %s\n", ok(maximum(abs.(xy_corr(0.01, R) .- xx)) < 0.02))
    @printf("γ=0.6   even in γ               %s\n", ok(xy_corr(0.6, R) ≈ xy_corr(-0.6, R)))
    @printf("γ=0.6   f(1) = %.6f  (ED, 12 sites periodic: -0.12423)\n", xy_corr(0.6, 1)[2] / SPIN_FACTOR)
end

# ---------------- Figure 1: three panels ----------------
function tfim_panel(g; N = 8, R = 400)
    p = range(-0.5, 0.5, length = 801)
    crit = isapprox(g, 1.0)
    f = tfim_corr(g, R)
    S = S_reg(:TFIM, g, p; R = R, cesaro = crit, f = f)
    ymax = 3.0 # crit ? 6.0 : 1.3 * maximum(S)
    O = overlap(:TFIM, g, N; p0 = 0.0, f = f)

    pl = plot(p, S; lw = 2, color = :firebrick, xlabel = L"p", ylabel = L"\hat f(p)",
              ylims = (0, ymax), xlims = (-0.5, 0.5), legend = :topright, framestyle = :box,
              label = L"\hat f(p)", #crit ? "Cesàro-summed, R=$R (true: ∼|p|^-3/4)" : L"\hat f_{\rm reg}(p)",
              title = @sprintf("g = %.1f    O_%d = %.4f", g, N, O))
    # Fejér kernel, rescaled to 80% of the panel height (true peak value: 1)
    plot!(pl, p, 2.0 .* fejer_hat(N, p); fillrange = 0, fillalpha = 0.15, lw = 1.5,
          ls = :dash, color = :black, label = "Fejér kernel N=$N (×2)")
    if g < 1   # Dirac peak: arrow, weight m² written next to it
        plot!(pl, [0, 0], [0, 0.92ymax]; arrow = true, lw = 3, color = :firebrick,
        label = @sprintf("m² δ(p),  m² = %.4f", tfim_m2(g)))
    end
    return pl
end

"""XY panel on p ∈ [0, 1], centred on the kernel bump at p0 = 1/2. f̂ has no Dirac part, so
O_N = ∫ f̂ φ̂_N dp exactly (Parseval): the shaded area under f̂·φ̂_N is O_N. No Cesàro window is needed,
since f ∼ 1/x² is absolutely summable even at γ = 0, where f̂(p) = 4|p| on [-1/2, 1/2]."""
function xy_panel(γ; N = 8, R = 400, p0 = 0.5)
    p = range(0, 1, length = 801)
    f = xy_corr(γ, R)
    S = S_reg(:XY, γ, p; R = R, f = f)          # m² = 0: this is the full f̂
    φ̂ = fejer_hat(N, p; p0)
    O = overlap(:XY, γ, N; p0, f = f)

    pl = plot(p, S; lw = 2, color = :firebrick, xlabel = L"p", ylabel = L"\hat f(p)",
              ylims = (0, 2.2), xlims = (0, 1), legend = :topright, framestyle = :box,
              label = L"\hat f(p)", title = @sprintf("γ = %.1f    O_%d = %.4f", γ, N, O))
    plot!(pl, p, φ̂; fillrange = 0, fillalpha = 0.10, lw = 1.5, ls = :dash, color = :black,
          label = "Fejér kernel N=$N, p0=$p0")
    plot!(pl, p, S .* φ̂; fillrange = 0, fillalpha = 0.35, lw = 0, color = :firebrick,
          label = L"\hat f\,\hat\varphi_N\ \ (\mathrm{area} = O_N)")
    return pl
end

function figure_panels(model = :TFIM; N = 8, vals = model == :XY ? [0.0, 0.5, 1.5] : [0.9, 1.0, 2.0])
    panel = model == :TFIM ? tfim_panel : model == :XY ? xy_panel : error("Unknown model $model")
    plt = plot((panel(θ; N = N) for θ in vals)...;
               layout = (1, length(vals)), size = (530 * length(vals), 480), margin = 6Plots.mm)
    savefig(plt, joinpath(@__DIR__, "..", "plots", "$(lowercase(string(model)))_panels_fejer_$(Libc.strftime("%d%m_%Hh%M", time())).png"))
    return plt
end

# ---------------- Figure 2: target overlaps vs parameter ----------------
"""O_N(θ) for several N. For N → ∞ the overlap tends to the Dirac weight at p0 (φ̂_N(p0) = 1 but ∫ φ̂_N → 0):
m² for the TFIM at p0 = 0, and 0 for the XY model, which is not drawn."""
function figure_overlap(model = :TFIM; Ns = [2, 4, 8, 16, 32, 64], θs = range(MODELS[model].range[1], MODELS[model].range[end], length = 120),
                        p0 = MODELS[model].p0)
    info = MODELS[model]
    Nmax = maximum(Ns)
    F = [corr(model, θ, Nmax) for θ in θs]           # one correlation sweep per θ, reused for all N
    plt = plot(xlabel = info.label, ylabel = L"O_N = \sum_x \varphi_N(x) f(x)",
               title = "$model: SDP target values (Fejér, p0 = $p0)", size = (700, 500), framestyle = :box)
    cols = palette(:viridis, length(Ns) + 1)
    for (i, N) in enumerate(Ns)
        plot!(plt, θs, [overlap(model, θ, N; p0, f = F[j]) for (j, θ) in enumerate(θs)];
              lw = 2, color = cols[i], label = "N = $N")
    end
    if model == :TFIM && iszero(p0)
        plot!(plt, θs, tfim_m2.(θs); lw = 2.5, ls = :dash, color = :black, label = L"N\to\infty:\ (1-g^2)^{1/4}")
    end
    savefig(plt, joinpath(@__DIR__, "..", "plots", "$(lowercase(string(model)))_overlap_p0_$(p0)_$(Libc.strftime("%d%m_%Hh%M", time())).png"))
    return plt
end


# -------------- Generate table with theoretical values for the joint plot with numerical bounds --------------
# Store overlaps (f^, φ̂_N) = Σ_{|x|≤N} φ_N(x) f(x) for several N and θ in table_theoretical_<model>_p0_<p0>_<date>.csv
function table_theoretical(model = :TFIM; N_list = [1,2,3], θ_list = MODELS[model].range, p0 = MODELS[model].p0)
    filename = joinpath(@__DIR__, "table_theoretical_$(model)_p0_$(p0)_$(Libc.strftime("%d%m_%Hh%M", time())).csv")

    F = [corr(model, θ, maximum(N_list)) for θ in θ_list]       # one correlation sweep per θ, reused for all N
    data = [overlap(model, θ, N; p0, f = F[j]) for (j, θ) in enumerate(θ_list), N in N_list]

    # --- Write ---
    open(filename, "w") do io
        writedlm(io, permutedims(["$(MODELS[model].param)/N"; string.(N_list)]), ',')   # header: corner label + N values
        writedlm(io, hcat(θ_list, data), ',')                                          # each row: θ, then data[i, :]
    end
    return filename
end




#tfim_selfcheck()
#xy_selfcheck()
#figure_panels(:TFIM; N=12, vals=[0.8, 1.0, 2.5])
#figure_panels(:XY; N=8)
#figure_overlap(:TFIM)
#figure_overlap(:XY)

# table_theoretical(:TFIM)
# table_theoretical(:XY)
