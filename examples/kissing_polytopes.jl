using FrankWolfe
using LinearAlgebra
using ProgressMeter
using SCIP
import MathOptInterface as MOI


include("../examples/plot_utils.jl")
include("no_good_counter.jl")

k = 1
d = 64
σ = 7 #Int(floor(sqrt(d)))
δ = Int(floor(d/(σ+1)))

a = float.([k*(1-δ)^(i-1) for i =1:σ+1])
a_bar = [a[Int(floor((i-1)/δ))+1] for i=1:d]


function compute_lattices(rhs)

    optimizer = SCIP.Optimizer(; display_verblevel=0, presolving_maxrounds=0)

    allow_dual_reductions =
        MOI.RawOptimizerAttribute("misc/allowstrongdualreds")
    MOI.set(optimizer, allow_dual_reductions, SCIP.FALSE)

    # add binary variables
    x = MOI.add_variables(optimizer, d)
    MOI.add_constraints(optimizer, [xi for xi in x], MOI.ZeroOne())
    MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(a_bar, x), 0.0), MOI.EqualTo(rhs))

    counter = NoGoodCounter.Counter(optimizer, x)
    SCIP.include_conshdlr(optimizer, counter; needs_constraints=false)

    MOI.optimize!(optimizer)

    return collect(counter.solutions)
end

function compute_lattices_reduced(rhs)

    optimizer = SCIP.Optimizer(; display_verblevel=0, presolving_maxrounds=0)

    allow_dual_reductions =
        MOI.RawOptimizerAttribute("misc/allowstrongdualreds")
    MOI.set(optimizer, allow_dual_reductions, SCIP.FALSE)

    # add binary variables
    x = MOI.add_variables(optimizer, σ+1)
    MOI.add_constraints(optimizer, [xi for xi in x], MOI.Integer())
    MOI.add_constraints(optimizer, [xi for xi in x], MOI.Interval(0.0, float(δ)))
    MOI.add_constraint(optimizer, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(a, x), 0.0), MOI.EqualTo(rhs))

    counter = NoBadCounter.Counter(optimizer, x)
    SCIP.include_conshdlr(optimizer, counter; needs_constraints=false)

    MOI.optimize!(optimizer)

    # Generate all binary vectors of length L with s-many ones
    function generate_binary_vectors(l, s, prefix=[])
        if l == 0 && s == 0
            return [prefix]
        end
        if l == 0 || s < 0
            return []
        end
    
        # Append 0 and recursively call with reduced length and sum
        vectors_0 = generate_binary_vectors(l - 1, s, [prefix..., 0])
    
        # Append 1 and recursively call with reduced length and sum
        vectors_1 = generate_binary_vectors(l - 1, s - 1, [prefix..., 1])
    
        return append!(vectors_0, vectors_1)
    end

    lattices = []

    for sol in counter.solutions
        temp_lattices = [Float64[]]
        for i=1:σ+1
            new_lattices = Vector{Vector{Float64}}()
            combinations = generate_binary_vectors(δ, sol[i])
            for l in temp_lattices
                for c in combinations                    
                    push!(new_lattices, vcat(l, c))
                end
            end
            temp_lattices = copy(new_lattices)
        end
        append!(lattices, temp_lattices)
    end

    return lattices
end

#@time p_lattices = compute_lattices(0.0)
#println(length(p_lattices))
@time p_lattices = compute_lattices_reduced(0.0)
println(length(p_lattices))
#@time q_lattices = compute_lattices(1.0)
#println(length(q_lattices))
@time q_lattices = compute_lattices_reduced(1.0)
println(length(q_lattices))

f(x) = 0

function grad!(storage, x)
    @. storage = zero(x)
end

xp = p_lattices[1]
xq = q_lattices[1]

diff = (xq - xp)/norm(xq - xp)

p_max = -Inf
@time for v in p_lattices
    score = FrankWolfe.fast_dot(diff, v - xp)/norm(v - xp)
    if score > p_max
        global p_max = score
    end
end
println("pmax: ", p_max)

q_min = Inf
@time for v in q_lattices
    score = FrankWolfe.fast_dot(diff, v - xq)/norm(v - xq)
    if score < q_min
        global q_min = score
    end
end
println("qmin: ", q_min)

@time filter!(v -> FrankWolfe.fast_dot(diff, v - xq)/norm(v - xq) < p_max, q_lattices)
println(length(q_lattices))
@time filter!(v -> FrankWolfe.fast_dot(diff, v - xp)/norm(v - xp) > q_min, p_lattices)
println(length(p_lattices))


#=
@time p_scores = [FrankWolfe.fast_dot(diff, v) for v in p_lattices]
@time q_scores = [FrankWolfe.fast_dot(diff, v) for v in q_lattices]

p_max = maximum(p_scores)
q_min = minimum(q_scores)

q_lattices_new = q_lattices[q_scores .> p_max]
p_lattices_new = p_lattices[p_scores .< q_min]
println(length(p_lattices_new))
println(length(q_lattices_new))
=#
lmo_p = FrankWolfe.ConvHull(p_lattices)
lmo_q = FrankWolfe.ConvHull(q_lattices)


x, _, _, _, infeas, traj_data = FrankWolfe.alternating_linear_minimization(
    FrankWolfe.away_frank_wolfe,
    f,
    grad!,
    (lmo_p, lmo_q),
    zeros(d);
    verbose=true,
    trajectory=true,
)

println(1/sqrt(d*(d-1)), " ", sqrt(infeas), " ", sqrt(δ*σ)/((k*(δ-1))^σ))

plot_trajectories([traj_data], ["cyclic"])