module InfluenceMaximization

using Printf
using Random
using Distributions
using Graphs
using SimpleWeightedGraphs
using DataStructures

const STD_N_ITERS = 10_000

include("diffusion_models.jl")
include("celf.jl")
include("celfpp.jl")
include("greedy.jl")
include("ranked_replace.jl")
include("select_top_k.jl")
include("synthetic_data.jl")


# Algorithms
export celf, celfpp, select_top_k, greedy, ranked_replace

export celf, celfpp

end # module InfluenceMaximization
