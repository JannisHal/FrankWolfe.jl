
"""
    ActiveSet{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
"""
struct ActiveSet{AT, R <: Real, IT} <: AbstractVector{Tuple{R,AT}}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
end

ActiveSet{AT,R}() where {AT,R} = ActiveSet{AT,R,Vector{float(eltype(AT))}}([], [])

ActiveSet{AT}() where {AT} = ActiveSet{AT,Float64,Vector{float(eltype(AT))}}()

function ActiveSet(tuple_values::AbstractVector{Tuple{R,AT}}) where {AT,R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(atoms[1], float(eltype(atoms[1])))
    as = ActiveSet{AT,R,typeof(x)}(weights, atoms, x)
    compute_active_set_iterate!(as)
    return as
end

function ActiveSet{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}) where {AT,R}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(tuple_values[1][2], float(eltype(tuple_values[1][2])))
    as = ActiveSet{AT,R,typeof(x)}(weights, atoms, x)
    compute_active_set_iterate!(as)
    return as
end

Base.getindex(as::ActiveSet, i) = (as.weights[i], as.atoms[i])
Base.size(as::ActiveSet) = size(as.weights)

# these three functions do not update the active set iterate

function Base.push!(as::ActiveSet, (λ, a))
    push!(as.weights, λ)
    push!(as.atoms, a)
    return as
end

function Base.deleteat!(as::ActiveSet, idx)
    deleteat!(as.weights, idx)
    deleteat!(as.atoms, idx)
    return as
end

function Base.setindex!(as::ActiveSet, tup::Tuple, idx)
    as.weights[idx] = tup[1]
    as.atoms[idx] = tup[2]
    return tup
end

function Base.empty!(as::ActiveSet)
    empty!(as.atoms)
    empty!(as.weights)
    as.x .= 0
    return as
end

function Base.isempty(as::ActiveSet)
    return isempty(as.atoms)
end

"""
Copies an active set, the weight and atom vectors and the iterate.
Individual atoms are not copied.
"""
function Base.copy(as::ActiveSet{AT,R,IT}) where {AT,R,IT}
    return ActiveSet{AT,R,IT}(copy(as.weights), copy(as.atoms), copy(as.x))
end

"""
    active_set_update!(active_set::ActiveSet, lambda, atom)

Adds the atom to the active set with weight lambda or adds lambda to existing atom.
"""
function active_set_update!(active_set::ActiveSet, lambda, atom, renorm=true, idx=nothing; add_dropped_vertices=false, vertex_storage=nothing)
    # rescale active set
    active_set.weights .*= (1 - lambda)
    # add value for new atom
    if idx === nothing
        idx = find_atom(active_set, atom)
    end
    updating = false
    if idx > 0
        @inbounds active_set.weights[idx] = active_set.weights[idx] + lambda
        updating = true
    else
        push!(active_set, (lambda, atom))
    end
    if renorm
        add_dropped_vertices = add_dropped_vertices ? vertex_storage !== nothing : add_dropped_vertices
        active_set_cleanup!(active_set, update=false, add_dropped_vertices=add_dropped_vertices, vertex_storage=vertex_storage)
        active_set_renormalize!(active_set)
    end
    active_set_update_scale!(active_set.x, lambda, atom)
    return active_set
end

"""
    active_set_update_scale!(x, lambda, atom)

Operates `x ← (1-λ) x + λ a`.
"""
function active_set_update_scale!(x::IT, lambda, atom) where {IT}
    @. x = x * (1 - lambda) + lambda * atom
    return x
end

function active_set_update_scale!(x::IT, lambda, atom::SparseArrays.SparseVector) where {IT}
    @. x *= (1 - lambda)
    nzvals = SparseArrays.nonzeros(atom)
    nzinds = SparseArrays.nonzeroinds(atom)
    for idx in eachindex(nzvals)
        x[nzinds[idx]] += lambda * nzvals[idx]
    end
    return x
end

"""
    active_set_update_iterate_pairwise!(x, lambda, fw_atom, away_atom)

Operates `x ← x + λ a_fw - λ a_aw`.
"""
function active_set_update_iterate_pairwise!(x::IT, lambda::Real, fw_atom::A, away_atom::A) where {IT, A}
    @. x += lambda * fw_atom - lambda * away_atom
    return x
end

function active_set_validate(active_set::ActiveSet)
    return sum(active_set.weights) ≈ 1.0 && all(>=(0), active_set.weights)
end

function active_set_renormalize!(active_set::ActiveSet)
    renorm = sum(active_set.weights)
    active_set.weights ./= renorm
    return active_set
end

function weight_from_atom(active_set::ActiveSet, atom)
    idx = find_atom(active_set, atom)
    if idx > 0
        return active_set.weights[idx]
    else
        return nothing
    end
end

"""
    get_active_set_iterate(active_set)

Return the current iterate corresponding. Does not recompute it.
"""
function get_active_set_iterate(active_set)
    return active_set.x
end

"""
    compute_active_set_iterate!(active_set::ActiveSet) -> x

Recomputes from scratch the iterate `x` from the current weights and vertices of the active set.
Returns the iterate `x`.
"""
function compute_active_set_iterate!(active_set)
    active_set.x .= 0
    for (λi, ai) in active_set
        @. active_set.x += λi * ai
    end
    return active_set.x
end

# specialized version for sparse vector
function compute_active_set_iterate!(active_set::ActiveSet{<:SparseArrays.SparseVector})
    active_set.x .= 0
    for (λi, ai) in active_set
        nzvals = SparseArrays.nonzeros(ai)
        nzinds = SparseArrays.nonzeroinds(ai)
        @inbounds for idx in eachindex(nzvals)
            active_set.x[nzinds[idx]] += λi * nzvals[idx]
        end
    end
    return active_set.x
end

function active_set_cleanup!(active_set; weight_purge_threshold=1e-12, update=true, add_dropped_vertices=false, vertex_storage=nothing)
    if add_dropped_vertices && vertex_storage !== nothing 
        for (weight, v) in zip(active_set.weights, active_set.atoms) 
            if weight <= weight_purge_threshold
                push!(vertex_storage, v)
            end
        end
    end

    filter!(e -> e[1] > weight_purge_threshold, active_set)
    if update
        compute_active_set_iterate!(active_set)
    end
    return nothing
end

function find_atom(active_set::ActiveSet, atom)
    @inbounds for idx in eachindex(active_set)
        if _unsafe_equal(active_set.atoms[idx], atom)
            return idx
        end
    end
    return -1
end

"""
    active_set_argmin(active_set::ActiveSet, direction)

Computes the linear minimizer in the direction on the active set.
Returns `(λ_i, a_i, i)`
"""
function active_set_argmin(active_set::ActiveSet, direction)
    val = dot(active_set.atoms[1], direction)
    idx = 1
    temp = 0
    for i in 2:length(active_set)
        temp = fast_dot(active_set.atoms[i], direction)
        if temp < val
            val = temp
            idx = i
        end
    end
    # return lambda, vertex, index
    return (active_set[idx]..., idx)
end

"""
    active_set_argminmax(active_set::ActiveSet, direction)

Computes the linear minimizer in the direction on the active set.
Returns `(λ_min, a_min, i_min, val_min, λ_max, a_max, i_max, val_max, val_max-val_min ≥ Φ)`
"""
function active_set_argminmax(active_set::ActiveSet, direction; Φ=0.5)
    val = Inf
    valM = -Inf
    idx = -1
    idxM = -1
    for i in eachindex(active_set)
        temp_val = fast_dot(active_set.atoms[i], direction)
        if temp_val < val
            val = temp_val
            idx = i
        end
        if valM < temp_val
            valM = temp_val
            idxM = i
        end
    end
    return (active_set[idx]..., idx, val, active_set[idxM]..., idxM, valM, valM - val ≥ Φ)
end


"""
    active_set_initialize!(as, v)

Resets the active set structure to a single vertex `v` with unit weight.
"""
function active_set_initialize!(as::ActiveSet{AT,R}, v) where {AT,R}
    empty!(as)
    push!(as, (one(R), v))
    compute_active_set_iterate!(as)
    return as
end

function compute_active_set_iterate!(active_set::ActiveSet{<:ScaledHotVector, <:Real, <:AbstractVector})
    active_set.x .= 0
    @inbounds for (λi, ai) in active_set
        active_set.x[ai.val_idx] += λi * ai.active_val
    end
    return active_set.x
end
