module StaticNets

using Distributions
using StatsBase
using Optim
using LineSearches
using StatsFuns
using Roots
using MLBase
using Statistics
using LinearAlgebra
using Random
using Distributed


using ..AReg
using ..Utilities
using ..Scalings
using ..ErgmRcall



## STATIC NETWORK MODEL
abstract type Ergm end
abstract type ErgmDirBin <: Ergm end

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

name(x::T where T <: Ergm) = error("to be defined")

type_of_obs(model::T where T <: Ergm) = error("to be defined")

obj_fun(model::T where T <: Ergm, obs_t, N, par) = error("to be defined")

grad_obj_fun(model::T where T <: Ergm, obs_t, N, par) = error("to be defined")

hessian_obj_fun(model::T where T <: Ergm, obs_t, N, par) = error("to be defined")

fisher_info_obj_fun(model::T where T <: Ergm, obs_t, N, par) = error("to be defined")

estimate(model::T where T <: Ergm, A) = error("to be defined")


estimate_sequence(model::T where T <: Ergm, obsT) = error("to be defined")


sample_ergm(model::T where T <: Ergm, N, parVec, nSample) = error("to be defined")  # return a 3D array N x N x nSample

"""
return a 4D array N x N x T x nSample 
"""
function sample_ergm_sequence(model::T where T <: Ergm, N, parVecSeq_T::Matrix, nSample)

    T = size(parVecSeq_T)[2]
    A_T_dgp = zeros(Int8, N, N, T, nSample)
    for t=1:T
        A_T_dgp[:, :, t, :] = sample_ergm(model, get_N_t(N,t), parVecSeq_T[:, t], nSample)
    end

    return A_T_dgp
end

#endregion

include("./StaticNets_models/StaticNets_DirBin1.jl")
include("./StaticNets_models/StaticNets_DirBin0Rec0.jl")
include("./StaticNets_models/StaticNets_ErgmPML.jl")
include("./StaticNets_models/StaticNets_conf_intervals.jl")


end
