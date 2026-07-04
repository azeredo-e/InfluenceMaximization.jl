
"""
    select_top_k(g, K; n_iters, rng) -> Vector{Int}

### Arguments
- `g`       - Weighted directed graph (`AbstractSimpleWeightedGraph`).
              Edge weight w(u,v) is the activation probability p_{u,v}.
- `K`       - Number of seeds to select.
- `n_iters` - Monte Carlo runs per σ evaluation (default: $(STD_N_ITERS)).
- `rng`     - RNG or `Uniform(0,1)` distribution (default: `Uniform(0,1)`).
 
### Returns
`Vector{Int}` of K node indices, sorted by descending solo spread.
"""
function select_top_k(
    g       :: AbstractSimpleWeightedGraph,
    K       :: Int;
    n_iters :: Int = STD_N_ITERS,
    rng     :: Union{AbstractRNG, UnivariateDistribution} = Uniform(0, 1),
    verbose :: Bool = false,
) :: Vector{Int}
    N = nv(g)
    @assert 1 ≤ K ≤ N "K must satisfy 1 ≤ K ≤ nv(g)  (got K=$K, N=$N)"

    # Compute σ({j}) for every node j ∈ V
    solo_spread = Vector{Float64}(undef, N)
    single_seed = Vector{Int}(undef, 1)
    for j in 1:N
        single_seed[1] = j
        solo_spread[j] = independent_cascade(g, single_seed; n_iters, rng)
    end

    # Select K nodes with the highest σ(·) as seed set φ₀
    φ₀ = partialsortperm(solo_spread, 1:K; rev = true)

    return collect(φ₀)
end
 