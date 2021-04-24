import ..StaticNets

import ..StaticNets:NetModeErgmPml, ErgmDirBin0Rec0


Base.@kwdef struct  SdErgmPml <: SdErgm
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
    
    if x.staticModel.ergmTermsString == ergm_term_string(ErgmDirBin0Rec0())
        return  "SdErgmDirBin0Rec0_pmle($(x.indTvPar), scal = $(x.scoreScalingType)$optString)"
    else
        return  "SdErgmPML($(x.indTvPar), scal = $(x.scoreScalingType)$optString)"
    end
end


stats_from_mat(model::SdErgmPml, A ::Matrix{<:Real}) = StaticNets.change_stats(model.staticModel, A)


function static_estimate(model::SdErgmPml, A_T)
    staticPars = ErgmRcall.get_one_mple(A_T, model.staticModel.ergmTermsString)
    return staticPars
end


function target_function_t(model::SdErgmPml, obs_t, N, par)
 
    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
 
    pll = StaticNets.obj_fun( par, changeStat, response, weights)

    return pll
end


function target_function_t_grad(model::SdErgmPml, obs_t, N, f_t) 

    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
    
    return StaticNets.grad_obj_fun(f_t, changeStat, response, weights)
end


function target_function_t_hess(model::SdErgmPml, obs_t, N, f_t) 

    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
    
    return StaticNets.hessian_obj_fun(f_t, changeStat, weights)

end


function target_function_t_fisher(model::SdErgmPml, obs_t, N,  f_t) 

    changeStat, response, weights = ErgmRcall.decomposeMPLEmatrix(obs_t)
    
    return  StaticNets.fisher_info_obj_fun(f_t, changeStat, weights)

end


function reference_model(model::SdErgmPml) 
    if model.staticModel.ergmTermsString == ergm_term_string(ErgmDirBin0Rec0())
        return SdErgmDirBin0Rec0_mle(scoreScalingType=model.scoreScalingType, indTvPar = model.indTvPar)
    end
end

#region list example models
model_edge_gwd(decay_par) = DynNets.SdErgmPml(staticModel = StaticNets.NetModeErgmPml("edges + gwidegree(decay = $decay_par, fixed = TRUE, cutoff=10) + gwodegree(decay = $decay_par, fixed = TRUE, cutoff=10)", true), indTvPar = [true, true, true], scoreScalingType="FISH_D")

model_edge_mutual = DynNets.SdErgmPml(staticModel = StaticNets.NetModeErgmPml("edges + mutual", true), indTvPar = [true, true], scoreScalingType="FISH_D")

model_edge_gwesp = DynNets.SdErgmPml(staticModel = StaticNets.NetModeErgmPml("edges + gwesp(decay = 0.25, fixed = TRUE, cutoff=10)", true), indTvPar = [true, true], scoreScalingType="FISH_D")

model_edge_mutual_gwesp = DynNets.SdErgmPml(staticModel = StaticNets.NetModeErgmPml("edges + mutual + gwesp(decay = 0.25, fixed = TRUE, cutoff=10)", true), indTvPar = [true, true, true], scoreScalingType="FISH_D")

model_edge_mutual_gwd(decay_par) = DynNets.SdErgmPml(staticModel = StaticNets.NetModeErgmPml("edges + mutual + gwidegree(decay = $decay_par, fixed = TRUE, cutoff=10) + gwodegree(decay = $decay_par, fixed = TRUE, cutoff=10)", true), indTvPar = [true, true, true, true], scoreScalingType="FISH_D")

model_rec_p_star = DynNets.SdErgmPml(staticModel = StaticNets.NetModeErgmPml("edges + mutual ", true), indTvPar = [true, true], scoreScalingType="FISH_D")
#endregion
