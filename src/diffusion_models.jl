
"""
    independent_cascade(g, seed_nodes, prob; n_iters, rng) -> Float64
"""
function independent_cascade(
    g::AbstractSimpleWeightedGraph,
    seed_nodes::AbstractVector{Int};
    n_iters::Int = STD_N_ITERS,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
)::Float64
    n = nv(g)
    total_activated = 0
    activated = falses(n)
    frontier = Vector{Int}(undef, n)
    next_frontier = Vector{Int}(undef, n)

    for _ in 1:n_iters
        fill!(activated, false)
        f_len = 0
        n_activated = 0
        for s in seed_nodes
            activated[s] = true
            f_len += 1
            frontier[f_len] = s
            n_activated += 1
        end
        while f_len > 0
            nf_len = 0

            for i in 1:f_len
                u = frontier[i]
                for v in outneighbors(g, u)
                    if !activated[v] && rand(rng) < get_weight(g, u, v)
                        activated[v] = true
                        nf_len += 1
                        next_frontier[nf_len] = v
                        n_activated += 1
                    end
                end
            end
            frontier, next_frontier = next_frontier, frontier
            f_len = nf_len
        end

        total_activated += n_activated
    end

    return total_activated / n_iters
end
