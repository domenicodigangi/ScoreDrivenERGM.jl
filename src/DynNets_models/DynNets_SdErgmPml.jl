import ..StaticNets

import ..StaticNets:NetModeErgmPml, NetModelDirBin0Rec0

Base.@kwdef struct  SdErgmPml <: GasNetModel
    staticModel::NetModeErgmPml
    indTvPar :: BitArray{1} 
    scoreScalingType::String = "HESS_D" # String that specifies the rescaling of the score. For a list of possible choices see function scalingMatGas
    options::SortedDict{Any, Any} = SortedDict()
end
export SdErgmPml

SdErgmPml(staticModel::NetModeErgmPml) = SdErgmPml(staticModel=staticModel, indTvPar = trues(staticModel.nErgmPar))

SdErgmPml(ergmTermsString::String, isDirected::Bool) = SdErgmPml(NetModeErgmPml(ergmTermsString, isDirected))


function name(x::SdErgmPml) 
    if isempty(x.options)
        optString = ""
    else
        optString = ", " * reduce(*,["$k = $v, " for (k,v) in x.options])[1:end-2]
    end
    
    if x.staticModel.ergmTermsString == ergm_term_string(NetModelDirBin0Rec0())
        return  "GasNetModelDirBin0Rec0_pmle($(x.indTvPar), scal = $(x.scoreScalingType)$optString)"
    else
        return  "SdErgmPML($(x.indTvPar), scal = $(x.scoreScalingType)$optString)"
    end
end





statsFromMat(model::SdErgmPml, A ::Matrix{<:Real}) = StaticNets.change_stats(model.staticModel, A)


function static_estimate(model::SdErgmPml, A_T)
    staticPars = ErgmRcall.get_static_mple(A_T, model.staticModel.ergmTermsString)
    return staticPars
end


function target_function_t(model::SdErgmPml, obs_t, N, par)
 
    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
 
    pll = StaticNets.pseudo_loglikelihood_strauss_ikeda( par, changeStat, response, weights)

    return pll
end


function target_function_t_grad(model::SdErgmPml, obs_t, N, f_t) 

    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
    
    return StaticNets.grad_pseudo_loglikelihood_strauss_ikeda(f_t, changeStat, response, weights)
end


function target_function_t_hess(model::SdErgmPml, obs_t, N, f_t) 

    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
    
    return StaticNets.hessian_pseudo_loglikelihood_strauss_ikeda(f_t, changeStat, weights)

end

function target_function_t_fisher(model::SdErgmPml, obs_t, N,  f_t) 

    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
    
    return  StaticNets.fisher_info_pseudo_loglikelihood_strauss_ikeda(f_t, changeStat, weights)

end


function reference_model(model::SdErgmPml) 
    if model.staticModel.ergmTermsString == ergm_term_string(NetModelDirBin0Rec0())
        return GasNetModelDirBin0Rec0_mle(scoreScalingType=model.scoreScalingType, indTvPar = model.indTvPar)
    end
end