using FrankWolfe
using Test
using LinearAlgebra

include("lmo.jl")
include("function_gradient.jl")
include("active_set_tests.jl")
include("utils.jl")

@testset "Line Search methods" begin
    a = [-1.0, -1.0, -1.0]
    b = [1.0, 1.0, 1.0]
    grad(x) = 2x
    f(x) = norm(x)^2
    @test FrankWolfe.backtrackingLS(f, grad, a, b) == (1, 0.5)
    @test abs(FrankWolfe.segmentSearch(f, grad, a, b)[2] - 0.5) < 0.0001
end

@testset "FrankWolfe.jl" begin
    @testset "Testing vanilla Frank-Wolfe with various step size and momentum strategies" begin
        f(x) = LinearAlgebra.norm(x)^2
        grad(x) = 2x
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.agnostic,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-5
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.goldenratio,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-5
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.backtracking,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-5
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.nonconvex,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-2
        @test FrankWolfe.fw(
            f,
            grad,
            lmo_prob,
            x0,
            maxIt=1000,
            step_size=FrankWolfe.shortstep,
            L=2,
            verbose=true,
        )[3] ≈ 0.2
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.nonconvex,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-2
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.agnostic,
                verbose=false,
                momentum=0.9,
            )[3] - 0.2,
        ) < 1.0e-3
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.agnostic,
                verbose=false,
                momentum=0.5,
            )[3] - 0.2,
        ) < 1.0e-3
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.agnostic,
                verbose=false,
                momentum=0.9,
                emph=FrankWolfe.memory,
            )[3] - 0.2,
        ) < 1.0e-3
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.adaptive,
                L=100,
                verbose=false,
                momentum=0.9,
            )[3] - 0.2,
        ) < 1.0e-3
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.adaptive,
                L=100,
                verbose=false,
                momentum=0.5,
            )[3] - 0.2,
        ) < 1.0e-3
        @test abs(
            FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.adaptive,
                L=100,
                verbose=false,
                momentum=0.9,
                emph=FrankWolfe.memory,
            )[3] - 0.2,
        ) < 1.0e-3
    end
    @testset "Testing Lazified Conditional Gradients with various step size strategies" begin
        f(x) = LinearAlgebra.norm(x)^2
        grad(x) = 2x
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(5))
        @test abs(
            FrankWolfe.lcg(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.goldenratio,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-5
        @test abs(
            FrankWolfe.lcg(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.backtracking,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-5
        @test abs(
            FrankWolfe.lcg(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=1000,
                step_size=FrankWolfe.shortstep,
                L=2,
                verbose=true,
            )[3] - 0.2,
        ) < 1.0e-5
    end

    @testset "Testing Lazified Conditional Gradients with cache strategies" begin
        n = Int(1e5)
        L = 2
        k = 1000
        bound = 16 * L * 2 / (k + 2)

        f(x) = LinearAlgebra.norm(x)^2
        grad(x) = 2x
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
        x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

        @time x, v, primal, dualGap, trajectory = FrankWolfe.lcg(
            f,
            grad,
            lmo_prob,
            x0,
            maxIt=k,
            step_size=FrankWolfe.shortstep,
            L=2,
            verbose=true,
        )

        @test primal - 1 // n <= bound

        @time x, v, primal, dualGap, trajectory = FrankWolfe.lcg(
            f,
            grad,
            lmo_prob,
            x0,
            maxIt=k,
            step_size=FrankWolfe.shortstep,
            L=2,
            cacheSize=100,
            verbose=true,
        )

        @test primal - 1 // n <= bound

        @time x, v, primal, dualGap, trajectory = FrankWolfe.lcg(
            f,
            grad,
            lmo_prob,
            x0,
            maxIt=k,
            step_size=FrankWolfe.shortstep,
            L=2,
            cacheSize=100,
            greedyLazy=true,
            verbose=true,
        )

        @test primal - 1 // n <= bound
    end

    @testset "Testing emphasis blas vs memory" begin
        n = Int(1e5)
        k = 100
        xpi = rand(n)
        total = sum(xpi)
        xp = xpi ./ total
        f(x) = LinearAlgebra.norm(x - xp)^2
        grad(x) = 2 * (x - xp)
        @testset "Using sparse structure" begin
            lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
            x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

            x, v, primal, dualGap, trajectory = FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=k,
                step_size=FrankWolfe.backtracking,
                printIt=k / 10,
                verbose=true,
                emph=FrankWolfe.blas,
            )

            @test x !== nothing

            x, v, primal, dualGap, trajectory = FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=k,
                step_size=FrankWolfe.backtracking,
                printIt=k / 10,
                verbose=true,
                emph=FrankWolfe.memory,
            )

            @test x !== nothing
        end
        @testset "Using dense structure" begin
            lmo_prob = FrankWolfe.L1ballDense{Float64}(1)
            x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

            x, v, primal, dualGap, trajectory = FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=k,
                step_size=FrankWolfe.backtracking,
                printIt=k / 10,
                verbose=true,
                emph=FrankWolfe.blas,
            )

            @test x !== nothing

            x, v, primal, dualGap, trajectory = FrankWolfe.fw(
                f,
                grad,
                lmo_prob,
                x0,
                maxIt=k,
                step_size=FrankWolfe.backtracking,
                printIt=k / 10,
                verbose=true,
                emph=FrankWolfe.memory,
            )

            @test x !== nothing
        end
    end
    @testset "Testing rational variant" begin
        rhs = 1
        n = 100
        k = 1000

        xpi = rand(n)
        total = sum(xpi)
        xp = xpi ./ total

        f(x) = LinearAlgebra.norm(x - xp)^2
        grad(x) = 2 * (x - xp)

        lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(rhs)
        direction = rand(n)
        x0 = FrankWolfe.compute_extreme_point(lmo, direction)

        @time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
            f,
            grad,
            lmo,
            x0,
            maxIt=k,
            step_size=FrankWolfe.agnostic,
            printIt=k / 10,
            emph=FrankWolfe.blas,
            verbose=true,
        )

        @test eltype(x0) == Rational{BigInt}

        @time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
            f,
            grad,
            lmo,
            x0,
            maxIt=k,
            step_size=FrankWolfe.agnostic,
            printIt=k / 10,
            emph=FrankWolfe.memory,
            verbose=true,
        )
        @test eltype(x0) == Rational{BigInt}

    end
    @testset "Multi-precision tests" begin
        rhs = 1
        n = 100
        k = 1000

        xp = zeros(n)

        L = 2
        bound = 2 * L * 2 / (k + 2)

        f(x) = LinearAlgebra.norm(x - xp)^2
        grad(x) = 2 * (x - xp)
        test_types = [Float16, Float32, Float64, BigFloat, Rational{BigInt}]

        @testset "Multi-precision test for $T" for T in test_types
            println("\nTesting precision for type: ", T)
            lmo = FrankWolfe.ProbabilitySimplexOracle{T}(rhs)
            direction = rand(n)
            x0 = FrankWolfe.compute_extreme_point(lmo, direction)

            @time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
                f,
                grad,
                lmo,
                x0,
                maxIt=k,
                step_size=FrankWolfe.agnostic,
                printIt=k / 10,
                emph=FrankWolfe.blas,
                verbose=true,
            )

            @test eltype(x0) == T
            @test primal - 1 // n <= bound

            @time x, v, primal, dualGap, trajectory = FrankWolfe.fw(
                f,
                grad,
                lmo,
                x0,
                maxIt=k,
                step_size=FrankWolfe.agnostic,
                printIt=k / 10,
                emph=FrankWolfe.memory,
                verbose=true,
            )

            @test eltype(x0) == T
            @test primal - 1 // n <= bound
        end
    end

    @testset "Stochastic FW linear regression" begin
        function simple_reg_loss(θ, data_point)
            (xi, yi) = data_point
            (a, b) = (θ[1:end-1], θ[end])
            pred = a ⋅ xi + b
            return (pred - yi)^2 / 2
        end

        function ∇simple_reg_loss(θ, data_point)
            (xi, yi) = data_point
            (a, b) = (θ[1:end-1], θ[end])
            pred = a ⋅ xi + b
            grad_a = xi * (pred - yi)
            grad = push!(grad_a, pred - yi)
            return grad
        end

        xs = [10 * randn(5) for i in 1:20000]
        params = rand(6) .- 1 # start params in (-1,0)
        bias = 2π
        params_perfect = [1:5; bias]

        params = rand(6) .- 1 # start params in (-1,0)

        data_perfect = [(x, x ⋅ (1:5) + bias) for x in xs]
        f_stoch = FrankWolfe.StochasticObjective(simple_reg_loss, ∇simple_reg_loss, data_perfect)
        lmo = FrankWolfe.LpNormLMO{2}(1.1 * norm(params_perfect))

        θ, _, _, _, _ = FrankWolfe.stochastic_frank_wolfe(
            f_stoch,
            lmo,
            params,
            momentum=0.95,
            verbose=false,
            step_size=FrankWolfe.nonconvex,
            maxIt=10000,
            batch_size=length(f_stoch.xs) ÷ 100,
            trajectory=false,
        )
        @test norm(θ - params_perfect) ≤ 0.02 * length(θ)
    end
end
