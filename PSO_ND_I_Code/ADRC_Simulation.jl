# ADRC simulation and fitness evaluation.

module ADRCSimulation

using Distributed: myid
using DifferentialEquations
using Symbolics
using RuntimeGeneratedFunctions
using Polynomials

export simulate_adrc, controller_poles, observer_poles

# Initialize symbolic functions.
RuntimeGeneratedFunctions.init(@__MODULE__)

# Numerical utilities.
@inline saturate(u::Real, lSup::Real=10.0, lInf::Real=-10.0) = clamp(u, lInf, lSup)

# Round ties away from zero.
@inline round_away_from_zero(x::Real) = round(x, RoundNearestTiesAway)

@inline quantize_position(x::Real, q::Real) = q * round_away_from_zero(x / q)

# Trapezoidal integration sum.
function trapezoidal_sum(x::AbstractVector{<:Real})
    s = 0.0
    @inbounds for k in 1:(length(x)-1)
        s += 0.5 * (x[k] + x[k+1])
    end
    return s
end

# Reference and derivatives.
@variables t

const r_sym =
    (1 - exp(-5t)) +
    (1//15) * (1 - exp(-t/5)) *
    (sin(2π*t) + sin(3π*t) + (1//2) * sin(5π*t))

const rp_sym  = Symbolics.derivative(r_sym, t)
const rpp_sym = Symbolics.derivative(rp_sym, t)

@inline function _first_build(bf)
    if bf isa Tuple
        return bf[1]
    elseif bf isa AbstractVector
        return bf[1]
    else
        return bf
    end
end

function _make_callable(symexpr)
    # Build callable function.
    bf = Symbolics.build_function(symexpr, t; expression=Val(false))
    obj = _first_build(bf)

    if obj isa Expr
        return @RuntimeGeneratedFunction(obj)
    elseif obj isa Function
        return obj
    else
        return obj
    end
end

const r_fun   = _make_callable(r_sym)
const rp_fun  = _make_callable(rp_sym)
const rpp_fun = _make_callable(rpp_sym)

# External disturbance.
@inline function disturbance_signal(tt::Real)
    return 0.1*sin(0.3*tt*π*2) + 0.1*sin(0.2*tt*π*2) + 0.1*sin(0.5*tt*π*2)
end

# Controller and observer poles.
function controller_poles(d1::Real, d2::Real)
    # Controller polynomial.
    p = Polynomial([float(d2), float(d1), 1.0])
    return roots(p)
end

function observer_poles(d1::Real, d2::Real, d3::Real)
    # Observer polynomial.
    p = Polynomial([float(d2)*float(d3), float(d2), float(d1), 1.0])
    return roots(p)
end

# ADRC simulation.
"""
    simulate_adrc(params)

Simulate ADRC for `[a1, a2, g1, g2, beta]`.
Returns `(J, y, x1q, rv, uVal, duVal, t)`.
"""
function simulate_adrc(params::AbstractVector{<:Real})
    p = collect(float.(params))
    length(p) >= 5 || error("params debe tener al menos 5: [a1,a2,g1,g2,beta].")

    h  = 0.001
    tf = 10.0
    tspan = (0.0, tf)
    t_eval = 0.0:h:tf
    q = 1/4096

    a = 0.9
    b = 50.0

    x0 = zeros(5)

    # Controller parameters.
    a1, a2, g1, g2, beta = p[1:5]

    # Plant and observer states.
    function f!(dx, x, _, tt)
        x1, x2, x1g, x2g, z = x

        x1qv = quantize_position(x1, q)

        # Plant disturbance.
        dv  = disturbance_signal(tt) - a*x2

        # Estimated disturbance.
        dgv = z + beta*x2g

        rr   = r_fun(tt)
        rrp  = rp_fun(tt)
        rrpp = rpp_fun(tt)

        # Nominal control.
        unom = rrpp + a1*(rrp - x2g) + a2*(rr - x1g)

        # Compensated control.
        u  = (unom - dgv) / b
        us = saturate(u)

        dx[1] = x2
        dx[2] = dv + b*us
        dx[3] = x2g + g1*(x1qv - x1g)
        dx[4] = unom + g2*(x1qv - x1g)
        dx[5] = -beta*(dgv + b*u)

        return nothing
    end

    prob = ODEProblem(f!, x0, tspan)

    # Solve the dynamics.
    sol = solve(prob, DP5();
        reltol=1e-4, abstol=1e-6,
        dtmax=h,
        saveat=t_eval,
        tstops=t_eval,           
        dense=false)
    # Check solver status.
    if string(sol.retcode) != "Success"
        params_str = "[" * join(string.(round.(p; digits=8)), ", ") * "]"

        println(stderr,
            "WARNING simulate_adrc: integración no exitosa | " *
            "retcode=$(sol.retcode) | " *
            "params=$(params_str)"
        )
    end

    tvec = Array(sol.t)
    Y = hcat(sol.u...)          # 5×N
    y = permutedims(Y, (2,1))   # N×5

    # Compute output signals.
    rv   = [r_fun(tt) for tt in tvec]
    rpv  = [rp_fun(tt) for tt in tvec]
    rppv = [rpp_fun(tt) for tt in tvec]
    wv   = [disturbance_signal(tt) for tt in tvec]

    x1qv = [quantize_position(y[k,1], q) for k in eachindex(tvec)]

    dvv  = wv .- a .* y[:,2]
    dgvv = y[:,5] .+ beta .* y[:,4]

    uVal  = (rppv .+ a1 .* (rpv .- y[:,4]) .+ a2 .* (rv .- y[:,3]) .- dgvv) ./ b
    duVal = diff(uVal) ./ h

    # Evaluate objective function.
    expr =
        15.0 .* (abs.(rv .- y[:,1]) .+ abs.(rv .- y[:,3])) .+
        10.0 .* (abs.(rpv .- y[:,2]) .+ abs.(rpv .- y[:,4])) .+
        0.5 .* abs.(uVal) .+
        1.0 .* abs.(dvv .- dgvv)

    Int1 = trapezoidal_sum(expr) * h
    Int1 += 0.5 * trapezoidal_sum(abs.(duVal)) * h

    eigsC = controller_poles(a1, a2)
    eigsO = observer_poles(g1, g2, beta)

    errEigOC = 0.0
    if maximum(real.(eigsO)) > minimum(real.(eigsC)) * 5.0
        errEigOC = maximum(real.(eigsO)) - minimum(real.(eigsC)) * 5.0
    end

    xiC = a1 / (2*sqrt(a2))
    errImag = abs(1.0 - xiC)

    J = Int1 + 10.0*errEigOC + 10.0*errImag

    return J, y, x1qv, rv, uVal, duVal, tvec
end

end # module
