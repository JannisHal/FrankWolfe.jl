using FrankWolfe
using LinearAlgebra

include("../examples/plot_utils.jl")

d = 1000

p1 = zeros(d)
p2 = ones(d)

Q = diagm([i==d ? 0 : 1 for i=1:d])

lmo_p = FrankWolfe.ConvHull([p1, p2])
lmo_q = FrankWolfe.ConvHull([Q[:,i] for i=1:d-1])#

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
    max_iteration=1e5,
    print_iter=1e3,
)

plot_trajectories([traj_data], ["cyclic"])

println(1/sqrt(d*(d-1)), " ", sqrt(infeas))