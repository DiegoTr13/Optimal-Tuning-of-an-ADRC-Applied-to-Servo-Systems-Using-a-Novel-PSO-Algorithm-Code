# PSO-ND-I for ADRC tuning.

module PSONDIADRC

using MAT
using Random
using Statistics
using Printf

include("ADRC_Simulation.jl")
import .ADRCSimulation as SD

export optimize_adrc

# Swarm distance from the centroid.
function swarm_diversity(X::AbstractMatrix{<:Real})
    c = mean(X, dims=1)
    d = sqrt.(sum((X .- c).^2, dims=2))
    return vec(d)  # nP
end

# Round controller gains.
function quantize_gains!(A; digits::Int=4)
    A .= round.(A; digits=digits)
    return A
end

"""
    optimize_adrc(id=1, verbose=1; kwargs...)

Optimize ADRC gains using PSO-ND-I.
Returns `(gBest, JgBest, JHist, disHist, zHist)`.
"""
function optimize_adrc(id::Integer=1, verbose::Integer=1;
                     nP::Int=50,
                     nIt::Int=150,
                     w::Real=0.8,
                     c1::Real=0.6,
                     c2::Real=0.6,
                     k::Real = 6.0,
                     c::Real = 0.4,
                     Amp1::Real = 0.2,
                     Amp2::Real = 0.6,
                     Amp3::Real = 0.6,
                     save_dir::AbstractString="pso-nd-i-logs",
                     save_dir_fin::AbstractString="pso-nd-i-results",
                     save_logs::Bool = true,
                     seed::Int = round(Int, time()) + id,
                     sim_kwargs::NamedTuple=NamedTuple())

    rng = MersenneTwister(seed)

    # Parameter bounds for initialization.
    lb = [0.0, 0.0, 0.0, 0.0, 0.0]
    ub = [100.0, 100.0, 100.0, 100.0, 100.0]

    nVar = length(lb)
    nVar == length(ub) || error("lb y ub deben tener la misma longitud.")

    err = zeros(Float64, nP, nVar)

    # Rounding precision and constraint margin.
    offset = 1e-3
    digits_ganancias = 4

    # Initialize particles.
    X = rand(rng, nP, nVar) .* 100.0

    # Apply beta <= g1.
    for p in 1:nP
        if nVar >= 5 && X[p, 5] > X[p, 3]
            X[p, 5] = X[p, 3] - offset
            if X[p, 5] < 0
                X[p, 5] = 0.0
            end
        end
    end

    # Round initial gains.
    quantize_gains!(X; digits=digits_ganancias)

    # Enforce beta <= g1.
    for p in 1:nP
        if nVar >= 5 && X[p, 5] > X[p, 3]
            X[p, 5] = round(max(X[p, 3] - offset, 0.0); digits=digits_ganancias)
        end
    end

    V = zeros(Float64, nP, nVar)

    pBest = copy(X)
    pBestVal = fill(Inf, nP)

    Jprev = fill(Inf, nP)

    # Evaluate initial fitness.
    for p in 1:nP
        quantize_gains!(view(X, p, :); digits=digits_ganancias)

        if isempty(sim_kwargs)
            Jp, _, _, _, _, _, _ = SD.simulate_adrc(view(X, p, :))
            pBestVal[p] = float(Jp)
        else
            Jp, _, _, _, _, _, _ = SD.simulate_adrc(view(X, p, :); sim_kwargs...)
            pBestVal[p] = float(Jp)
        end

        Jprev[p] = pBestVal[p]
    end

    g_idx = argmin(pBestVal)
    gBest = copy(view(pBest, g_idx, :))
    quantize_gains!(gBest; digits=digits_ganancias)

    JgBest = pBestVal[g_idx]

    # Allocate histories.
    JHist   = fill(NaN, nIt)
    disHist = fill(NaN, nP, nIt - 1)
    zHist   = fill(NaN, nP, nVar, nIt)

    zHist[:, :, 1] .= X
    JHist[1] = JgBest

    # Create output directories.
    proc_dir = joinpath(save_dir, "proceso_$(id)")

    if save_logs
        mkpath(proc_dir)
        mkpath(save_dir_fin)
    end

    # Optimization iterations.
    for it in 1:(nIt - 1)

        # Distance to global best.
        gBest_row = reshape(gBest, 1, :)
        disnorm = sqrt.(sum((X .- gBest_row).^2, dims=2))
        m = maximum(disnorm)

        dis0 = m > 0 ? vec(disnorm ./ m) : zeros(nP)

        # Normalized componentwise distance.
        epsDen = 1e-12
        denom = max.(abs.(gBest), epsDen)

        DW = abs.((X .- reshape(gBest, 1, :)) ./ reshape(denom, 1, :))
        DWimax = vec(maximum(DW, dims=2))

        m = maximum(DWimax)
        dis0 = m > 0 ? DWimax ./ m : zeros(nP)

        # Sigmoid distance mapping.
        σ(x) = 1 / (1 + exp(-k * (x - c)))
        σ0 = σ(0.0)
        σ1 = σ(1.0)

        dis = (σ.(dis0) .- σ0) ./ (σ1 - σ0)

        boost = max(0.0, 1.0 - it / 40)
        dis_eff = clamp.(dis .+ 0.0 * boost, 0.0, 1.0)

        # Distance-dependent PSO coefficients.
        w_it = w - 0.0 * ((it - 1) / (nIt - 1))

        wp  = Amp1 .* dis0
        c1p = Amp2 .* dis0
        c2p = Amp3 .* dis0

        gscale = 1.0 - 0.6 * (it / (nIt - 1))

        Xmax = vec(maximum(abs.(X), dims=1))
        Xmax = max.(Xmax, eps(Float64))
        Kelite = max(3, round(Int, 0.20 * nP))
        idxElite = partialsortperm(pBestVal, 1:Kelite)
        eliteCenter = vec(median(pBest[idxElite, :], dims=1))
        target = 0.80 .* pBest .+ 0.20 .* reshape(eliteCenter, 1, :)
        # Accumulate integral correction.
        err = err .+ ((target .- X)) .* 5.0

        # Record swarm distances.
        disHist[:, it] .= dis

        for p in 1:nP
            r1 = rand(rng, nVar)
            r2 = rand(rng, nVar)
            r3 = rand(rng, nVar)
            # Update particle velocity.
            V[p, :] .= (w_it + wp[p]) .* V[p, :] .+
                       ((c1 + c1p[p])) .* r1 .* (pBest[p, :] .- X[p, :]) .+
                       ((c2 + c2p[p])) .* r2 .* (gBest .- X[p, :])

            # Update position with integral action.
            X[p, :] .= X[p, :] .+ V[p, :] .+ dis0[p] .* err[p, :]*gscale

            # Enforce positive gains.
            for j in 1:nVar
                if X[p, j] <= 0
                    X[p, j] = offset
                    V[p, j] = 0.0
                end
            end

            # Enforce beta <= g1.
            if nVar >= 5 && X[p, 5] > X[p, 3]
                X[p, 5] = X[p, 3] - offset
                if X[p, 5] < 0
                    X[p, 5] = 0.0
                end
                V[p, 5] = 0.0
            end

            # Round candidate gains.
            quantize_gains!(view(X, p, :); digits=digits_ganancias)

            # Enforce beta <= g1.
            if nVar >= 5 && X[p, 5] > X[p, 3]
                X[p, 5] = round(max(X[p, 3] - offset, 0.0); digits=digits_ganancias)
                V[p, 5] = 0.0
            end

            # Evaluate ADRC cost.
            local Jp::Float64

            if isempty(sim_kwargs)
                Jtmp, _, _, _, _, _, _ = SD.simulate_adrc(view(X, p, :))
                Jp = float(Jtmp)
            else
                Jtmp, _, _, _, _, _, _ = SD.simulate_adrc(view(X, p, :); sim_kwargs...)
                Jp = float(Jtmp)
            end

            # Reset correction after cost increase.
            if Jp > 2 * Jprev[p]
                err[p, :] .= 0.0
            end

            Jprev[p] = Jp

            # Update personal best.
            if Jp < pBestVal[p]
                pBestVal[p] = Jp
                pBest[p, :] .= X[p, :]
                quantize_gains!(view(pBest, p, :); digits=digits_ganancias)
            end

            # Update global best.
            if Jp < JgBest
                gBest .= X[p, :]
                quantize_gains!(gBest; digits=digits_ganancias)

                JgBest = Jp
                g_idx = p

                V[p, :] .= 0.0
            end
        end

        # Record iteration history.
        JHist[it + 1] = JgBest
        zHist[:, :, it + 1] .= X

        # Save iteration results.
        if save_logs
            iter_file = joinpath(proc_dir, @sprintf("iter_%03d.mat", it - 1))

            matwrite(iter_file, Dict(
                "iteration" => it - 1,
                "X" => X,
                "V" => V,
                "err" => err,
                "Jprev" => Jprev,
                "pBest" => pBest,
                "pBestVal" => pBestVal,
                "gBest" => collect(gBest),
                "JgBest" => JgBest,
                "JHist" => JHist[1:(it + 1)],
                "disHist" => disHist[:, 1:it],
                "zHist" => zHist[:, :, 1:(it + 1)],
                "nP" => nP,
                "nIt" => nIt,
                "nVar" => nVar,
                "lb" => lb,
                "ub" => ub,
                "w" => float(w),
                "c1" => float(c1),
                "c2" => float(c2),
                "Amp1" => float(Amp1),
                "Amp2" => float(Amp2),
                "Amp3" => float(Amp3),
                "digits_ganancias" => digits_ganancias,
                "proc_dir" => proc_dir
            ))
        end

        if verbose == 1
            println("Proceso $(id) | Iter $(it)/$(nIt - 1): J = $(JgBest)")
        end
    end

    # Save final results.
    if save_logs
        final_file = joinpath(save_dir_fin, "proceso_$(id).mat")

        matwrite(final_file, Dict(
            "gBest" => collect(gBest),
            "JgBest" => JgBest,
            "JHist" => JHist,
            "disHist" => disHist,
            "zHist" => zHist,
            "err" => err,
            "Jprev" => Jprev,
            "pBest" => pBest,
            "pBestVal" => pBestVal,
            "nP" => nP,
            "nIt" => nIt,
            "nVar" => nVar,
            "lb" => lb,
            "ub" => ub,
            "w" => float(w),
            "c1" => float(c1),
            "c2" => float(c2),
            "Amp1" => float(Amp1),
            "Amp2" => float(Amp2),
            "Amp3" => float(Amp3),
            "digits_ganancias" => digits_ganancias,
            "proc_dir" => proc_dir
        ))
    end

    return collect(gBest), JgBest, JHist, disHist, zHist
end

end # module
