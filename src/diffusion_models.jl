
"""
    independent_cascade(g, seed_nodes, prob; n_iters, rng) -> Float64

Estimate the expected number of nodes activated by `seed_nodes` under the
**Independent Cascade (IC)** model via Monte Carlo simulation.

### IC model
Each active node `u` tries — exactly once — to activate every inactive
out-neighbour `v` with probability `prob`, independently of all other edges.
Newly activated nodes attempt to activate *their* neighbours in the next round.
The process terminates when no new activations occur.

### Arguments
- `g`          - Directed graph (`AbstractSimpleWeightedGraph` from Graphs.jl).
- `seed_nodes` - Initial (active) seed set; a `Vector{Int}` of node indices.
- `prob`       - Per-edge activation probability ∈ (0, 1].
- `n_iters`    - Monte Carlo simulations to average over (default: 1 000).
- `rng`        - RNG instance for reproducibility (default: `Random.GLOBAL_RNG`).

### Returns
Mean number of ultimately activated nodes across all `n_iters` simulations.
"""
function independent_cascade(
    g::AbstractSimpleWeightedGraph,
    seed_nodes::AbstractVector{Int};
    n_iters::Int = 1_000,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
)::Float64

    total_activated = 0

    for _ = 1:n_iters
        activated = Set{Int}(seed_nodes)   # all nodes active so far
        frontier = collect(seed_nodes)     # nodes activated in this time-step

        while !isempty(frontier)
            next_frontier = Int[]
            for u in frontier
                for v in outneighbors(g, u)
                    prob = get_weight(g, u, v)
                    if !(v in activated) && rand(rng) < prob
                        push!(activated, v)
                        push!(next_frontier, v)
                    end
                end
            end
            frontier = next_frontier
        end

        total_activated += length(activated)
    end

    return total_activated / n_iters
end

