module InfluenceMaximization

using Printf
using Random
using Distributions
using Graphs
using SimpleWeightedGraphs
using DataStructures


include("diffusion_models.jl")
include("celf.jl")
include("celfpp.jl")

const STD_N_ITERS = 10_000

export celf, celfpp

end # module InfluenceMaximization
