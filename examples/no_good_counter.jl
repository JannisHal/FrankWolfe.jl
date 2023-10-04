module NoGoodCounter

    using MathOptInterface
    using SCIP

    const MOI = MathOptInterface

    mutable struct Counter <: SCIP.AbstractConstraintHandler
        scip::SCIP.Optimizer # for SCIP* and var maps
        variables::Vector{MOI.VariableIndex}
        solutions::Set{Array{Float64}}

        Counter(scip, variables) = new(scip, variables, Set())
    end

    function SCIP.check(
        ch::Counter,
        constraints,
        sol,
        checkintegrality,
        checklprows,
        printreason,
        completely,
    )
        return SCIP.SCIP_INFEASIBLE
    end

    function enforce(ch::Counter)
        values = SCIP.sol_values(ch.scip, ch.variables)

        # Store solution
        if values in ch.solutions
            # Getting the same solution twice?
            return SCIP.SCIP_INFEASIBLE
        end
        push!(ch.solutions, values)

        # Add no-good constraint: sum_zeros(x) + sum_ones(1-x) >= 1
        zeros = isapprox.(values, 0.0, atol=1e-6)
        ones = isapprox.(values, 1.0, atol=1e-6)
        others = .!zeros .& .!ones
        if any(others)
            println(values[others])
            error("Found non-binary solution value for ")#$(vars[others])")
        end

        terms = vcat(
            [MOI.ScalarAffineTerm(1.0, vi) for vi in ch.variables[zeros]],
            [MOI.ScalarAffineTerm(-1.0, vi) for vi in ch.variables[ones]],
        )
        ci = MOI.add_constraint(
            ch.scip,
            MOI.ScalarAffineFunction(terms, 0.0),
            MOI.GreaterThan(1.0 - sum(ones)),
        )

        return SCIP.SCIP_CONSADDED
    end

    function SCIP.enforce_lp_sol(
        ch::Counter,
        constraints,
        nusefulconss,
        solinfeasible,
    )
        @assert length(constraints) == 0
        return enforce(ch)
    end

    function SCIP.enforce_pseudo_sol(
        ch::Counter,
        constraints,
        nusefulconss,
        solinfeasible,
        objinfeasible,
    )
        @assert length(constraints) == 0
        return enforce(ch)
    end

    function SCIP.lock(ch::Counter, constraint, locktype, nlockspos, nlocksneg) end

end # module NoGoodCounter

module NoBadCounter

    using MathOptInterface
    using SCIP

    const MOI = MathOptInterface

    mutable struct Counter <: SCIP.AbstractConstraintHandler
        scip::SCIP.Optimizer # for SCIP* and var maps
        variables::Vector{MOI.VariableIndex}
        solutions::Set{Array{Float64}}

        Counter(scip, variables) = new(scip, variables, Set())
    end

    function SCIP.check(
        ch::Counter,
        constraints,
        sol,
        checkintegrality,
        checklprows,
        printreason,
        completely,
    )
        return SCIP.SCIP_INFEASIBLE
    end

    function enforce(ch::Counter)

        values = SCIP.sol_values(ch.scip, ch.variables)

        # Store solution
        if values in ch.solutions
            # Getting the same solution twice?
            return SCIP.SCIP_INFEASIBLE
        end
        push!(ch.solutions, values)

        return SCIP.SCIP_CONSADDED
    end

    function SCIP.enforce_lp_sol(
        ch::Counter,
        constraints,
        nusefulconss,
        solinfeasible,
    )
        @assert length(constraints) == 0
        return enforce(ch)
    end

    function SCIP.enforce_pseudo_sol(
        ch::Counter,
        constraints,
        nusefulconss,
        solinfeasible,
        objinfeasible,
    )
        @assert length(constraints) == 0
        return enforce(ch)
    end

    function SCIP.lock(ch::Counter, constraint, locktype, nlockspos, nlocksneg) end

end # module NoBadCounter