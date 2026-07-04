
"""
    ranked_replace(g, K; n_iters, rng) -> Vector{Int}

### Arguments
- `g`       - Weighted directed graph (`AbstractSimpleWeightedGraph`).
              Edge weight w(u,v) is the activation probability p_{u,v}.
- `K`       - Number of seeds.
- `n_iters` - Monte Carlo runs per σ evaluation (default: $(STD_N_ITERS)).
- `rng`     - RNG or `Uniform(0,1)` distribution (default: `Uniform(0,1)`).
 
### Returns
`Vector{Int}` of K selected node indices.
"""
function ranked_replace(
    g       :: AbstractSimpleWeightedGraph,
    K       :: Int;
    n_iters :: Int = STD_N_ITERS,
    rng     :: Union{AbstractRNG, UnivariateDistribution} = Uniform(0, 1),
    verbose :: Bool = false,
) :: Vector{Int}
    N = nv(g)
    @assert 1 ≤ K ≤ N "K must satisfy 1 ≤ K ≤ nv(g)  (got K=$K, N=$N)"

    single_seed = Vector{Int}(undef, 1) # reused for every [j] solo-spread call
    φ_candidate = Vector{Int}(undef, K) # φ₀ ∪ {j} \ {i}  — always K elements
    seeds_asc   = Vector{Int}(undef, K) # φ₀ sorted ascending by solo_spread

    # Compute σ({j}) for every j ∈ V
    solo_spread = Vector{Float64}(undef, N)
    for j in 1:N
        single_seed[1] = j
        solo_spread[j] = independent_cascade(g, single_seed; n_iters, rng)
    end

    # Initialise φ₀ with K highest solo-spread nodes
    φ₀ = collect(partialsortperm(solo_spread, 1:K; rev = true))
    in_φ₀ = falses(N)
    @inbounds for s in φ₀
        in_φ₀[s] = true
    end

    σ_φ₀ = independent_cascade(g, φ₀; n_iters, rng)

    # Sort V\φ₀ descending by solo_spread — computed once
    candidates = sort(
        [j for j in 1:N if !in_φ₀[j]];
        by  = j -> solo_spread[j],
        rev = true,
    )

    # for j ∈ V\φ₀ in descending order of σ(j)
    for j in candidates 
        # Sort φ₀ ascending by solo_spread
        copyto!(seeds_asc, φ₀)
        sort!(seeds_asc; by = i -> solo_spread[i]) # ascending, in-place

        # for i ∈ φ₀ in ascending order of σ({i})
        for i in seeds_asc
            # σ(φ₀ ∪ {j} \ {i}) > σ(φ₀)?
            k_idx = 0
            @inbounds for s in φ₀
                if s != i
                    k_idx += 1
                    φ_candidate[k_idx] = s
                end
            end
            φ_candidate[K] = j 
            σ_candidate = independent_cascade(g, φ_candidate; n_iters, rng)

            if σ_candidate > σ_φ₀
                # φ₀ = φ₀ ∪ {j} \ {i}
                @inbounds for idx in 1:K
                    if φ₀[idx] == i
                        φ₀[idx] = j
                        break
                    end
                end
                in_φ₀[i] = false
                in_φ₀[j] = true
                σ_φ₀ = σ_candidate
                break
            end
        end
    end

    return φ₀
end
 