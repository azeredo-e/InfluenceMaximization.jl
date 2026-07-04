
"""
    greedy(g, K; n_iters, rng, verbose) -> NamedTuple

### Arguments
- `g`       - Weighted directed graph (`AbstractSimpleWeightedGraph`).
              Edge weight w(u,v) is the activation probability p_{u,v}.
- `K`       - Number of seeds to select.
- `n_iters` - Monte Carlo runs per σ evaluation (default: $(STD_N_ITERS)).
- `rng`     - RNG or `Uniform(0,1)` distribution (default: `Uniform(0,1)`).
- `verbose` - Print per-step diagnostics (default: `false`).
 
### Returns
`NamedTuple` with fields:
- `solution :: Vector{Int}`    — selected seeds in order of selection
- `spreads  :: Vector{Float64}`— cumulative σ(φ₀) after each step
- `elapsed  :: Float64`        — total wall-clock time in seconds
"""
function greedy(
    g       :: AbstractSimpleWeightedGraph,
    K       :: Int;
    n_iters :: Int = STD_N_ITERS,
    rng     :: Union{AbstractRNG, UnivariateDistribution} = Uniform(0, 1),
    verbose :: Bool = false,
) :: NamedTuple
    N = nv(g)
    @assert 1 ≤ K ≤ N "K must satisfy 1 ≤ K ≤ nv(g)  (got K=$K, N=$N)"

    in_φ₀ = falses(N)
    tmp_seeds = Vector{Int}(undef, N)

    # φ₀ = ∅
    φ₀ = Int[];     sizehint!(φ₀,      K)
    spreads = Float64[]; sizehint!(spreads, K)
    t0 = time()

    for q in 1:K # for q = 1 to K
        len_φ₀ = length(φ₀)
        σ_φ₀ = len_φ₀ == 0 ? 0.0 : independent_cascade(g, φ₀; n_iters, rng)
        copyto!(tmp_seeds, 1, φ₀, 1, len_φ₀)        
        best_node = 0
        best_gain = -Inf

        # i = argmax_{j ∈ V\φ₀} {σ(φ₀ ∪ {j}) − σ(φ₀)}
        @inbounds for j in 1:N
            in_φ₀[j] && continue # O(1) bit-test; original was O(|φ₀|) scan

            tmp_seeds[len_φ₀ + 1] = j
            gain = independent_cascade(g, @view(tmp_seeds[1:len_φ₀ + 1]); n_iters, rng) - σ_φ₀ 

            if gain > best_gain
                best_gain = gain
                best_node = j
            end
        end

        push!(φ₀, best_node) # φ₀ = φ₀ ∪ {i}
        in_φ₀[best_node] = true
        push!(spreads, σ_φ₀ + best_gain)

        verbose && @printf(
            "  step %2d | node %4d | marginal gain %7.3f | σ(φ₀) = %7.3f\n",
            q, best_node, best_gain, last(spreads)
        )
    end

    return (solution = φ₀, spreads = spreads, elapsed = time() - t0)
end
