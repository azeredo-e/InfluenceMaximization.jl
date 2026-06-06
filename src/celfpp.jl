
"""
    celfpp(g, k, prob; n_iters, rng, verbose) -> NamedTuple
"""
function celfpp(
    g::AbstractGraph,
    k::Int;
    n_iters::Int = STD_N_ITERS,
    rng::Union{AbstractRNG, UnivariateDistribution} = Uniform(0,1),
    verbose::Bool = false,
    show_result = true
) :: NamedTuple
    n = nv(g)
    @assert 1 ≤ k ≤ n   "k must satisfy 1 ≤ k ≤ nv(g)"

    mg1 = zeros(Float64, n)
    mg2 = zeros(Float64, n)
    prev_best = zeros(Int, n)
    flag = zeros(Int, n)
    S = Int[]
    last_seed = 0
    cur_best = 0
    lookups = 0
    Q = PriorityQueue{Int, Float64}(Base.Order.Reverse)

    for u in 1:n
        mg1[u]  = independent_cascade(g, [u]; n_iters, rng) # u.mg1 = σ({u})
        lookups += 1

        prev_best[u] = cur_best # u.prev_best = cur_best 

        if cur_best == 0 # u.mg2 = σ({cur_best})
            mg2[u] = mg1[u]
        else
            mg2[u]  = independent_cascade(g, [u, cur_best]; n_iters, rng) #- mg1[cur_best]
            lookups += 1
        end

        flag[u] = 0
        enqueue!(Q, u, mg1[u])

        if cur_best == 0 || mg1[u] > mg1[cur_best]
            cur_best = u
        end
    end

    spread_S = 0.0
    spread_S_valid = true

    while length(S) < k # |S| < k
        u = peek(Q)[1] # u = top (root) element in Q

        if flag[u] == length(S) # if u.flag == |S|
            dequeue!(Q) # S ← S ∪ {u}; Q ← Q − {u}
            push!(S, u)
            last_seed = u # last_seed = u
            cur_best = 0
            spread_S_valid = false

            verbose && @printf(
                "  [CELF++] step %2d | node %4d | total lookups: %d\n",
                length(S), u, lookups
            )
            continue
        elseif prev_best[u] == last_seed # else if u.prev_best == last_seed
            mg1[u] = mg2[u]
        else
            if !spread_S_valid
                spread_S = independent_cascade(g, S; n_iters, rng)
                spread_S_valid = true
                lookups       += 1
            end

            mg1[u]  = independent_cascade(g, [S; u]; n_iters, rng) - spread_S # u.mg1 = Δu(S) = σ(S ∪ {u}) − σ(S)
            lookups += 1

            prev_best[u] = cur_best # u.prev_best = cur_best

            if cur_best == 0 # u.mg2 = Δu(S ∪ {cur_best}) = σ(S ∪ {u, cur_best}) − σ(S ∪ {cur_best})
                mg2[u] = mg1[u]
            else
                spread_S_cb = independent_cascade(g, [S; cur_best]; n_iters, rng)
                mg2[u] = independent_cascade(g, [S; u; cur_best]; n_iters, rng) - spread_S_cb
                lookups += 2
            end
        end

        flag[u] = length(S) # u.flag = |S|; Update cur best.
        if cur_best == 0 || mg1[u] > mg1[cur_best]
            cur_best = u
        end

        Q[u] = mg1[u] # Reinsert u into Q and heapify
    end

    show_result && @printf(
        "  [CELF++] optimal seed: %s | final spread: %.4f | total lookups: %d\n",
        string(S), spread_S, lookups
    )

    return (solution=S, lookups=lookups, spread=spread_S)
end

