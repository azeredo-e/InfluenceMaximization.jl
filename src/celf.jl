
"""
    lazy_forward(g, costs, B, type, prob; n_iters, rng) -> Vector{Int}
"""
function lazy_forward(
    g::AbstractSimpleWeightedGraph,
    costs::Vector{Float64},
    B::Float64,
    type::Symbol;
    n_iters::Int = STD_N_ITERS,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
)::Vector{Int}
    # Helper variables and functions
    n_nodes = nv(g) # TODO: I think i need to fix the case where cost=0
    @inline cost_of_A() = isempty(A) ? 0.0 : sum(costs[s] for s in A) # Cost already spent on the current placement A
    @inline affordable(s) = cost_of_A() + costs[s] ≤ B # Is adding node s to A still within budget?
    @inline candidates()::Vector{Int} = [s for s in 1:n_nodes if s ∉ A && affordable(s)] # All nodes not yet in A that fit within the remaining budget.
    priority(s)::Float64 = type == :UC ? δ[s] : δ[s] / costs[s] # Do either UC or CB for argmax

    A = Int[]
    δ = fill(Inf, n_nodes) # δs for each node s
    cur = Vector{Bool}(undef, n_nodes) # curs flag: was δs computed this outer iteration?

    while !isempty(candidates()) # while ∃s ∈ V\A : c(A ∪ {s}) ≤ B  do
        for s = 1:n_nodes # foreach s ∈ V\A do  curs ← false
            cur[s] = false
        end

        R_A = isempty(A) ? 0.0 : independent_cascade(g, A; n_iters, rng) # Calculate the gain from seed nodes A

        while true
            cands = candidates() #? This list could be a statict vedctor

            #! Optimze here, broadcast is hurting performance
            s_star = cands[argmax(priority.(cands))] # Calculate for both UC and CB

            if cur[s_star] # if curs*  then  A ← A ∪ {s*};  break
                push!(A, s_star)
                break
            else # else  δs* ← R(A ∪ {s*}) − R(A);  curs* ← true
                δ[s_star] = independent_cascade(g, [A; s_star]; n_iters, rng) - R_A
                cur[s_star] = true
            end
        end   # while true (inner loop)
    end   # while ∃ affordable candidate (outer loop)

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
