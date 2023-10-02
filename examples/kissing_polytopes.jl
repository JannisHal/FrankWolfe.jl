using FrankWolfe
using LinearAlgebra
using ProgressMeter
using JuMP
import HiGHS
import MultiObjectiveAlgorithms as MOA


include("../examples/plot_utils.jl")

k = 1
d = 25

σ = 4 #Int(floor(sqrt(d)))
δ = Int(floor(d/(σ+1)))

a = [k*(1-δ)^(i-1) for i =1:σ+1]
a_bar = [a[Int(floor((i-1)/δ))+1] for i=1:d]


p_lattices = []
q_lattices = []

m1 = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
@variable(m1, x[1:d], Bin)
@constraint(m1, dot(x, a_bar) == 0)
set_attribute(m1, MOA.Algorithm(), MOA.Dichotomy())
set_silent(m1)
optimize!(m1)

m2 = Model(() -> MOA.Optimizer(HiGHS.Optimizer))
@variable(m2, y[1:d], Bin)
@constraint(m2, dot(y, a_bar) == 1)
set_attribute(m2, MOA.Algorithm(), MOA.Dichotomy())
set_silent(m2)
optimize!(m2)

for i in 1:result_count(m1)
    push!(p_lattices, value.(x; result = i))
end

for i in 1:result_count(m2)
    push!(q_lattices, value.(y; result = i))
end


println(length(p_lattices))
println(length(q_lattices))

lmo_p = FrankWolfe.ConvHull(p_lattices)
lmo_q = FrankWolfe.ConvHull(q_lattices)

f(x) = 0

function grad!(storage, x)
    @. storage = zero(x)
end

x, _, _, _, infeas, traj_data = FrankWolfe.alternating_linear_minimization(
    FrankWolfe.away_frank_wolfe,
    f,
    grad!,
    (lmo_p, lmo_q),
    zeros(d);
    lambda=1.0,
    verbose=true,
    trajectory=true,
    #max_iteration=1e6,
    #print_iter=1e5,
)

plot_trajectories([traj_data], ["cyclic"])

println(1/sqrt(d*(d-1)), " ", sqrt(infeas), " ", sqrt(δ*σ)/((k*(δ-1))^σ))