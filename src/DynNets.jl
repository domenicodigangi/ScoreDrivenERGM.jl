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


abstract type GasNetModel end
abstract type GasNetModelBin <: GasNetModel end
abstract type GasNetModelDirBin0Rec0 <: GasNetModel end


#constants
targetErrValDynNets = 0.01

#Relations between Static and Dynamic Models]
identify(Model::GasNetModel,UnPar::Array{<:Real,1}, idType ) =
    StaticNets.identify(StaModType(Model),UnPar;idType = idType)

number_ergm_par(model::T where T <:GasNetModel) = length(model.indTvPar)


include("./DynNets_models/DynNets_SDModelsUtils.jl")

include("./DynNets_models/DynNets_GasNetModelDirBin1.jl")

include("./DynNets_models/DynNets_GasNetModelDirBin0Rec0.jl")

include("./DynNets_models/DynNets_GasNetModelDirBinERGM.jl")

end



