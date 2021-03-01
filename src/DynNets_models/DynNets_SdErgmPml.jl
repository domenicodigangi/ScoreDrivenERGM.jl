import ..StaticNets:NetModeErgmPml, NetModelDirBin0Rec0

Base.@kwdef struct  SdErgmPml <: GasNetModel
    staticModel::NetModeErgmPml
    indTvPar :: BitArray{1} 
    scoreScalingType::String = "HESS" # String that specifies the rescaling of the score. For a list of possible choices see function scalingMatGas
end
export SdErgmPml

SdErgmPml(staticModel::NetModeErgmPml) = SdErgmPml(staticModel, trues(staticModel.nErgmPar))

SdErgmPml(ergmTermsString::String) = SdErgmPml(NetModeErgmPml(ergmTermsString))


function name(x::SdErgmPml) 
    if x.staticModel.ergmTermsString == ergm_term_string(NetModelDirBin0Rec0())
        return "GasNetModelDirBin0Rec0_pmle($(x.indTvPar), scal = $(x.scoreScalingType))"
    else
        return "SdErgmPML($(x.staticModel.ergmTermsString), $(x.indTvPar), scal = $(x.scoreScalingType))"
    end
end


statsFromMat(Model::SdErgmPml, A ::Matrix{<:Real}) = StaticNets.change_stats(StaticNets.NetModelDirBin0Rec0(), A)


function static_estimate(model::SdErgmPml, A_T)
    staticPars = ErgmRcall.get_static_mple(A_T, "edges +  mutual")
    return staticPars
end


function target_function_t(model::SdErgmPml, obs_t, par)
 
    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
 
    pll = StaticNets.pseudo_loglikelihood_strauss_ikeda( par, changeStat, response, weights)

    return pll
end

