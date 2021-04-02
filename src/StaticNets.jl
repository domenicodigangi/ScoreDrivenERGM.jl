module StaticNets

using Distributions, StatsBase,Optim, LineSearches, StatsFuns,Roots,MLBase, Statistics, LinearAlgebra, Random

using ..AReg
using ..Utilities
using ..Scalings
using ..ErgmRcall



## STATIC NETWORK MODEL
abstract type NetModel end
abstract type NetModelDirBin <: NetModel end

#constants
targetErrValStaticNets = 1e-2
export targetErrValStaticNets

targetErrValStaticNetsW = 1e-5
export targetErrValStaticNetsW

bigConstVal = 10^6
export bigConstVal

maxLargeVal =  1e40# 1e-10 *sqrt(prevfloat(Inf))
export maxLargeVal

minSmallVal = 1e2*eps()
export minSmallVal

# region StaticNet interface (work in progress)

name(x::T where T <: NetModel) = error("to be defined")

type_of_obs(model::T where T <: NetModel) = error("to be defined")

obj_fun(model::T where T <: NetModel, obs_t, N, par) = error("to be defined")

grad_obj_fun(model::T where T <: NetModel, obs_t, N, par) = error("to be defined")

hessian_obj_fun(model::T where T <: NetModel, obs_t, N, par) = error("to be defined")

fisher_info_obj_fun(model::T where T <: NetModel, obs_t, N, par) = error("to be defined")

estimate(model::T where T <: NetModel, A) = error("to be defined")

estimate_sequence(model::T where T <: NetModel, obsT) = error("to be defined")


#endregion

include("./StaticNets_models/StaticNets_DirBin1.jl")
include("./StaticNets_models/StaticNets_DirBin0Rec0.jl")
include("./StaticNets_models/StaticNets_ErgmPML.jl")


end
