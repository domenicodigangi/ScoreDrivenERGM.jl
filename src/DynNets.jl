module DynNets


using Distributions
using StatsBase
using Optim
using LineSearches
using StatsFuns
using Statistics
using Roots
using MLBase
using GLM
using LinearAlgebra
using JLD2
using DataFrames
using ForwardDiff
using NLSolversBase
using RCall
using PyPlot
using Logging
using Distributed
using SharedArrays


using ..AReg
using ..Utilities
using ..Scalings
using ..ErgmRcall
using ..StaticNets



#constants
const targetErrValDynNets = 0.01


include("./DynNets_models/DynNets_Abstract_GasNetModel.jl")

include("./DynNets_models/DynNets_GasNetModelDirBin1.jl")

include("./DynNets_models/DynNets_SdErgmPml.jl")

include("./DynNets_models/DynNets_GasNetModelDirBin0Rec0.jl")

end



