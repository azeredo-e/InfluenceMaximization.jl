
"""
    lazy_forward(g, costs, B, type; n_iters, rng) -> Vector{Int}
 
CELF lazy-forward seed selection (Leskovec et al., KDD 2007).
"""
function lazy_forward(
    g     :: AbstractSimpleWeightedGraph,
    costs :: Vector{Float64},
    B     :: Float64,
    type  :: Symbol;
    n_iters :: Int = STD_N_ITERS,
    rng     :: Union{AbstractRNG, UnivariateDistribution} = Uniform(0, 1),
) :: Vector{Int}
    n = nv(g)
    A = Int[]
    sizehint!(A, n)
    δ = fill(Inf, n)
    cur = falses(n)
    in_A = falses(n)
    budget_used = 0.0
    tmp_seeds   = Vector{Int}(undef, n)
    use_uc = (type === :UC)

    while true # while ∃ affordable s ∉ A
        # Quick check for performance: if no candidate is affordable, we can break early without recomputing the candidates list
        any_candidate = false
        @inbounds for s in 1:n
            if !in_A[s] && budget_used + costs[s] ≤ B
                any_candidate = true
                break
            end
        end
        any_candidate || break

        fill!(cur, false) # foreach s ∈ V\A do  curs ← false
        R_A = isempty(A) ? 0.0 : independent_cascade(g, A; n_iters, rng) # Calculate the gain from seed nodes A
        len_A = length(A)
        copyto!(tmp_seeds, 1, A, 1, len_A)

        while true
            s_star = 0
            best_p = -Inf
            @inbounds for s in 1:n
                (in_A[s] || budget_used + costs[s] > B) && continue
                p = use_uc ? δ[s] : δ[s] / costs[s]
                if p > best_p
                    best_p = p
                    s_star = s
                end
            end

            if @inbounds cur[s_star]  # if curs*  then  A ← A ∪ {s*};  break
                push!(A, s_star)
                @inbounds in_A[s_star] = true
                budget_used += costs[s_star]
                break 
            else # else  δs* ← R(A ∪ {s*}) − R(A);  curs* ← true
                @inbounds tmp_seeds[len_A + 1] = s_star
                @inbounds δ[s_star] = independent_cascade(g, @view(tmp_seeds[1:len_A + 1]); n_iters, rng) - R_A
                @inbounds cur[s_star] = true
            end
        end
    end # while ∃ affordable candidate (outer loop)
 
    return A
end


"""
    celf(g, costs, B, prob; n_iters, rng, verbose) -> NamedTuple
"""
function celf(
    g::AbstractSimpleWeightedGraph,
    costs::Vector{Float64},
    B::Float64;
    n_iters::Int = STD_N_ITERS,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
    verbose::Bool = false,
)::NamedTuple
    verbose && println("CELF ▸ LazyForward [UC] …")
    A_UC = lazy_forward(g, costs, B, :UC; n_iters, rng)
    R_UC = isempty(A_UC) ? 0.0 : independent_cascade(g, A_UC; n_iters, rng)

    verbose && println("CELF ▸ LazyForward [CB] …")
    A_CB = lazy_forward(g, costs, B, :CB; n_iters, rng)
    R_CB = isempty(A_CB) ? 0.0 : independent_cascade(g, A_CB; n_iters, rng)

    solution, spread, winner = R_UC ≥ R_CB ? (A_UC, R_UC, :UC) : (A_CB, R_CB, :CB)

    if verbose
        @printf("  UC → nodes %-20s  spread = %.3f\n", string(A_UC), R_UC)
        @printf("  CB → nodes %-20s  spread = %.3f\n", string(A_CB), R_CB)
        @printf("  Winner: %s\n", winner)
    end

    return (
        solution = solution,
        spread = spread,
        winner = winner,
        A_UC = A_UC,
        R_UC = R_UC,
        A_CB = A_CB,
        R_CB = R_CB,
    )
end
function celf(
    g::AbstractSimpleWeightedGraph,
    k::Int;
    n_iters::Int = STD_N_ITERS,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
    verbose::Bool = false,
)::NamedTuple
    costs = ones(Float64, nv(g))
    return celf(g, costs, Float64(k); n_iters, rng, verbose)
end
