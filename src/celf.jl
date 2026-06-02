
"""
    lazy_forward(g, costs, B, type, prob; n_iters, rng) -> Vector{Int}

Implements the `LazyForward` function from Algorithm 1 of Leskovec et al. 2007.
what node to pick as seed that gives the best result

### Why the algorithm is correct

δs is initialised to +∞ for every node and is **never reset** between outer
iterations — only the `cur` flag is reset.  This means δs always holds either
+∞ (never evaluated) or the marginal gain computed against a *past*, smaller
seed set A.  By submodularity the true current gain can only be ≤ that cached
value, so every stored δs is a valid **upper bound**.

Inside the inner loop the algorithm always examines the node s* with the
highest upper-bound δs.  It recomputes the true gain and marks cur=true.
If s* is still the argmax after the recomputation it is immediately selected:
every other node's stored δ is an overestimate of its true gain, so none of
them can beat s*'s freshly verified value.  If s* dropped, the new argmax
(still holding a stale upper bound) is examined next.

This lazy deferral means subsequent outer iterations often need only 1-2
recomputations instead of the full n required by plain greedy.

### Arguments
- `g`       - `AbstractSimpleWeightedGraph` from Graphs.jl (directed or undirected).
- `costs`   - Per-node cost vector; `costs[s]` is c(s) in the paper.
- `B`       - Total budget (scalar).
- `type`    - `:UC` for unit-cost greedy (ignores costs),
              `:CB` for cost-benefit greedy (divides gain by cost).
- `prob`    - Per-edge activation probability ∈ (0, 1].
- `n_iters` - Monte Carlo runs per R(·) evaluation.
- `rng`     - RNG for reproducibility.
"""
function lazy_forward(
    g::AbstractSimpleWeightedGraph,
    costs::Vector{Float64},
    B::Float64,
    type::Symbol;
    n_iters::Int = 1_000,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
)::Vector{Int}
    # Helper variables and functions
    n_nodes = nv(g) # TODO: I think i need to fix the case where cost=0
    @inline cost_of_A() = isempty(A) ? 0.0 : sum(costs[s] for s in A) # Cost already spent on the current placement A
    @inline affordable(s) = cost_of_A() + costs[s] ≤ B # Is adding node s to A still within budget?
    @inline candidates() = [s for s = 1:n_nodes if s ∉ A && affordable(s)] # All nodes not yet in A that fit within the remaining budget.
    @inline priority(s) = type == :UC ? δ[s] : δ[s] / costs[s] # Do either UC or CB for argmax

    A = Int[]
    δ = fill(Inf, n_nodes) # δs for each node s
    cur = Vector{Bool}(undef, n_nodes) # curs flag: was δs computed this outer iteration?

    while !isempty(candidates()) # while ∃s ∈ V\A : c(A ∪ {s}) ≤ B  do
        for s = 1:n_nodes # foreach s ∈ V\A do  curs ← false
            cur[s] = false
        end

        R_A = isempty(A) ? 0.0 : independent_cascade(g, A; n_iters, rng) # Calculate the gain from seed nodes A

        while true
            cands = candidates()

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


# =============================================================================
# §3  CELF  — the top-level algorithm (bottom box in Algorithm 1)
# =============================================================================

"""
    celf(g, costs, B, prob; n_iters, rng, verbose) -> NamedTuple

Implements the `CELF` algorithm from Algorithm 1 of Leskovec et al. 2007.

Pseudocode (verbatim from the paper):

    Algorithm: CELF(G=(V,E), R, c, B)
      A_UC ← LazyForward(G, R, c, B, UC)
      A_CB ← LazyForward(G, R, c, B, CB)
      return argmax{ R(A_UC), R(A_CB) }

### Why two calls?

With non-uniform costs the unit-cost greedy (UC) can perform arbitrarily
badly — it ignores cost and may waste the entire budget on one expensive node
(see the ε-example in Section 3 of the paper).  The cost-benefit greedy (CB)
fixes this but can in turn fail when one very high-reward node is cheap enough
to dominate.  Theorem 3 in the paper proves that the *better* of the two
solutions always achieves at least ½(1 - 1/e) ≈ 31.6 % of optimal.

### Arguments
- `g`       - `AbstractSimpleWeightedGraph` from Graphs.jl.
- `costs`   - Per-node cost vector; use `ones(nv(g))` for unit cost.
- `B`       - Budget scalar.
- `prob`    - Per-edge activation probability.
- `n_iters` - Monte Carlo runs per R(·) evaluation.
- `rng`     - RNG for reproducibility.
- `verbose` - Print intermediate results.

### Returns
A `NamedTuple` with fields:
  `solution` - winning seed set (Vector{Int})
  `spread`   - estimated spread of the winning set (Float64)
  `winner`   - which variant won: `:UC` or `:CB`
  `A_UC`, `R_UC`, `A_CB`, `R_CB` - both solutions and their spreads
"""
function celf(
    g::AbstractSimpleWeightedGraph,
    costs::Vector{Float64},
    B::Float64;
    n_iters::Int = 1_000,
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

