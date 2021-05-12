import ..StaticNets

import ..StaticNets:NetModeErgmPml, ErgmDirBin0Rec0


Base.@kwdef struct  SdErgmPml <: SdErgm
    staticModel::NetModeErgmPml
    indTvPar :: BitArray{1} 
    scoreScalingType::String = "FISH_D" # String that specifies the rescaling of the score. For a list of possible choices see function scalingMatGas
    options::SortedDict{Any, Any} = SortedDict()
end
export SdErgmPml


SdErgmPml(staticModel::NetModeErgmPml) = SdErgmPml(staticModel=staticModel, indTvPar = trues(staticModel.nErgmPar))


SdErgmPml(ergmTermsString::String, isDirected::Bool) = SdErgmPml(NetModeErgmPml(ergmTermsString, isDirected))


function Base.getproperty(x::SdErgmPml, p::Symbol)
    if p in fieldnames(typeof(x))
        return Base.getfield(x, p)
    else
        return Base.getproperty(x.staticModel, p)
    end
end



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
    else
        return model
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


function sample_time_var_par_from_dgp(model::SdErgmPml, dgpType, N, T; minVals = -Inf * ones(1), maxVals = Inf * ones(1), meanVals = nothing, sigma = [0.01], B = [0.95], A=[0.01], indTvPar=trues(number_ergm_par(model)), maxAttempts = 25000, plotFlag = false)

    
    @debug "[sample_time_var_par_from_dgp][init][$( (;model, dgpType, N, T, minVals, maxVals)) ]"

    nErgmPar = model.staticModel.nErgmPar

    if isnothing(meanVals)
        meanVals = ( maxVals .+ minVals )./2
    end

    parDgpT = zeros(Real,nErgmPar,T)

    if dgpType=="AR"
        length(sigma) == 1 ? sigmaVec = sigma[1].*ones(nErgmPar) : sigmaVec = sigma
        length(B) == 1 ? BVec = B[1].*ones(nErgmPar) : BVec = B

        okSampleFlag = false
        for n=1:maxAttempts        

            for p = 1:nErgmPar 
                parDgpT[p, :] = dgpAR(meanVals[p], BVec[p], sigmaVec[p], T)
            end

            if all(parDgpT .< maxVals) & all(parDgpT .> minVals)
                okSampleFlag = true
                break
            end
        end

        okSampleFlag ? () : error("could not sample a path respecting boundaries last sample = $parDgpT")

    end

    if plotFlag
     fig, ax = subplots(nErgmPar,1)
        for p in 1:nErgmPar
            ax[1,p].plot(parDgpT[p,:], "k")
        end
    end
    
     @debug "[sample_time_var_par_from_dgp][end]"
    return parDgpT
end


function list_example_dgp_settings(model::SdErgmPml; out="tuple", minVals = [-3.0, 0], maxVals = [-2.4, 1])

    
    dgpSetARlow = (type = "AR", opt = (B =[0.98], sigma = [0.01], minVals=minVals, maxVals = maxVals ))

    dgpSetARmed = (type = "AR", opt = (B =[0.98], sigma = [0.05], minVals=minVals, maxVals = maxVals))
    
    dgpSetARhigh = (type = "AR", opt = (B =[0.98], sigma = [0.1], minVals=minVals, maxVals = maxVals))

    dgpSetSIN = (type = "SIN", opt = ( nCycles=[1.5], minVals=minVals, maxVals = maxVals))

    dgpSetSDlow = (type = "SD", opt = (B =[0.98], A = [0.01], minVals=minVals, maxVals = maxVals))

    dgpSetSD = (type = "SD", opt = (B =[0.98], A = [0.3], minVals=minVals, maxVals = maxVals))
    
    dgpSetSDhigh = (type = "SD", opt = (B =[0.98], A = [3], minVals=minVals, maxVals = maxVals))

    tupleList =  (; dgpSetARlow, dgpSetARmed, dgpSetARhigh, dgpSetSIN, dgpSetSDlow, dgpSetSD, dgpSetSDhigh)

    if out == "tuple"
        return tupleList
    elseif out == "dict"
        d = Dict() 
        [d[string(dgp)[7:end]] = getfield(tupleList,dgp) for dgp in fieldnames(typeof(tupleList))]
        return d
    end
end
