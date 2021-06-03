

#region Uncertainties filtered parameters
# using FiniteDiff
# Base.eps(Real) = eps(Float64)

function A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model::SdErgm, N, obsT, vEstSdResParAll, ftot_0)

    T = length(obsT)
    nPar = length(vEstSdResParAll)
    gradT = zeros(nPar, T)
    hessT = zeros(nPar, nPar, T)

    vecUnParAll = unrestrict_all_par(model, vEstSdResParAll)    

    function obj_fun_t(xUn, t)        
        logLikeVecT = seq_loglike_sd_filter(model, N, obsT[1:t], xUn, ftot_0) 
        return - logLikeVecT[end]
    end

    for t = 1:T
    
        obj_fun(vecUnParAll) = obj_fun_t(vecUnParAll, t)

        gradT[:,t] = deepcopy(ForwardDiff.gradient(obj_fun, vecUnParAll))
        hessT[:,:,t] =  deepcopy(ForwardDiff.hessian(obj_fun, vecUnParAll))
        # gradT[:,t] = deepcopy(FiniteDiff.finite_difference_gradient(obj_fun, vecUnParAll))
        # hessT[:,:,t] =  deepcopy(FiniteDiff.finite_difference_hessian(obj_fun, vecUnParAll))
    end


    OPGradSum = sum([gt * gt' for gt in eachcol(gradT[:,2:end])] )
    HessSum = dropdims(sum(hessT[:,:,2:end], dims=3 ), dims=3)

    return OPGradSum, HessSum
end


function white_estimate_cov_mat_static_sd_par(model::SdErgm,  N,obsT, indTvPar, ftot_0, vEstSdResParAll; returnAllMats=false, enforcePosDef = true)

    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    errorFlag = false

    
    OPGradSum, HessSum = A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model, N, obsT, vEstSdResParAll, ftot_0)

    parCovHat = pinv(HessSum) * OPGradSum * pinv(HessSum)
    
    if enforcePosDef
        parCovHatPosDef, minEigenVal = make_pos_def(parCovHat)
    else
        parCovHatPosDef = parCovHat
        minEigenVal = 0 
    end

    if minEigenVal < 0 
        minEiegOPGrad = minimum(eigen(OPGradSum).values)
        minEiegHess = minimum(eigen(HessSum).values)
        Logging.@info("Outer Prod minimum eigenvalue $(minEiegOPGrad) , hessian minimum eigenvalue $(minEiegHess)")

        # if the negative eigenvalue of the cov mat is due to a negative eigenvalue of the hessian, do not use that estimate
        if minEiegHess < 0 
            errorFlag = true
        end
    end

    mvSDUnParEstCov = Symmetric(parCovHatPosDef)
    if returnAllMats
        return mvSDUnParEstCov, errorFlag, OPGradSum, HessSum
    else
        return mvSDUnParEstCov, errorFlag 
    end
end


function get_B_A_mats_for_TV(model::SdErgm, indTvPar, vEstSdResPar)
    # @debug "[get_B_A_mats_for_TV][begin]"
    nTvPar = sum(indTvPar)
    nErgmPar = number_ergm_par(model)
    B = zeros(nTvPar)
    A = zeros(nTvPar)
    
    lastIndRead = 0
    lastIndWrite = 0
    for e in 1:nErgmPar
        if indTvPar[e]
            bInd = lastIndRead+2
            aInd = lastIndRead+3
            B[lastIndWrite+1] = vEstSdResPar[bInd] 
            A[lastIndWrite+1] = vEstSdResPar[aInd] 
            lastIndRead += 3
            lastIndWrite += 1
        else
            lastIndRead += 1
        end

    end
    
    
    # @debug "[get_B_A_mats_for_TV][end]"
    return B, A
end



function distrib_filtered_par_from_sample_static_par(model::SdErgm, N, obsT, indTvPar, ftot_0, sampleUnParAll)
        
    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    nTvPar = sum(indTvPar)
    nSample = size(sampleUnParAll)[2]

    # any(.!indTvPar) ? error("assuming all parameters TV. otherwise need to set filter unc to zero for static ones") : ()    


    distribFilteredSD = zeros(nSample, nErgmPar,T)
    filtCovHatSample = zeros(nTvPar, T, nSample)
    flagIntegrated = is_integrated(model)
    for n=1:nSample
        vResPar = restrict_all_par(model, sampleUnParAll[:,n])

        vecSDParRe, vConstPar = divide_SD_par_from_const(model, vResPar)

        distribFilteredSD[n, :, :] , ~, ~, invScalMatT = DynNets.score_driven_filter( model, N, obsT, vecSDParRe, indTvPar;ftot_0=ftot_0, vConstPar=vConstPar)

        BMatSD, AMatSD = get_B_A_mats_for_TV(model, indTvPar, vResPar)

        constFiltUncCoeff = (BMatSD.^(-1)).*AMatSD
     
        if model.scoreScalingType[end] == 'D'
            for t in 1:T 
                filtCovHatSample[:,t,n] = constFiltUncCoeff./ invScalMatT[:,:,t][I(nErgmPar)][indTvPar]
            end
        else
            error()
        end

    end

    # remove samples that resulted in inconsistent P_t
    indBadSamples = [any(filtCovHatSample[:,:,n].<0) | any(.!isfinite.(filtCovHatSample[:,:,n])) for n in 1:nSample]

    
    if mean(indBadSamples) > 0.5 
        Logging.@error("Sampling from mvNormal resulted in $(mean(indBadSamples)*100) % of bad samples. Rejected") 
        errFlag = true
    else
        errFlag = false
    end

    filtCovHatSample = filtCovHatSample[:,:,.!indBadSamples]

    return distribFilteredSD, filtCovHatSample, errFlag
end


function distrib_static_par_from_mv_normal(model::SdErgm, N, obsT, indTvPar, ftot_0, vEstSdUnPar, mvSDUnParEstCov; nSample = 500)
        
    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    nTvPar = sum(indTvPar)
    # any(.!indTvPar) ? error("assuming all parameters TV. otherwise need to set filtering unc to zero for static ones") : ()    

    zeroMeanSample =  rand(MvNormal(mvSDUnParEstCov), nSample ) 
    sampleUnParAll = zeroMeanSample .+ vEstSdUnPar

    return sampleUnParAll
end


"""
Use parametric bootstrap to sample from the distribution of static parameters of the SD filter
"""
function par_bootstrap_distrib_filtered_par(model::SdErgm, N, obsT, indTvPar, ftot_0, vEstSdResPar; nSample = 250, logFlag = false)

    T = length(obsT)
    nErgmPar = number_ergm_par(model)
     
     
    nStaticSdPar = length(vEstSdResPar)
    vUnParEstDistrib = SharedArray(zeros(length(vEstSdResPar), nSample))
    distribFilteredSD = SharedArray(zeros(nSample, nErgmPar,T))
    filtCovHatSample = SharedArray(zeros(nErgmPar, nSample))
    errFlagVec = SharedArray{Bool}((nSample))
    flagIntegrated = is_integrated(model)

    Threads.@threads for n=1:nSample
        
        # try
            # sample SD dgp
            fVecTDgp, A_T_dgp, ~ = DynNets.score_driven_dgp( model, N, vEstSdResPar, indTvPar, (N,T))
            obsTNew = [stats_from_mat(model, A_T_dgp[:,:,t]) for t in 1:T ]

            # Estimate SD static params from SD DGP
            arrayAllParHat, conv_flag,UM, ftot_0 = estimate(model, N, obsTNew; indTvPar=indTvPar, indTargPar=falses(2), ftot_0=ftot_0)

            vResPar = DynNets.array_2_vec_all_par(model, arrayAllParHat, indTvPar)
    
            vUnPar = unrestrict_all_par(model, deepcopy(vResPar))

            vUnParEstDistrib[:,n] =  vUnPar

            vecSDParRe, vConstPar = divide_SD_par_from_const(model, vResPar)

            distribFilteredSD[n, :, :] , ~, ~ = DynNets.score_driven_filter( model, N, obsT, vecSDParRe, indTvPar; ftot_0=ftot_0, vConstPar=vConstPar)

            BMatSD, AMatSD = get_B_A_mats_for_TV(model, indTvPar, vResPar)

            filtCovHat = (BMatSD.^(-1)).*AMatSD
            filtCovHat[.!indTvPar] .= 0
            filtCovHatSample[:,n] = filtCovHat
            
        # catch
        #     errFlagVec[n] = true
        # end

     
    end

   return distribFilteredSD, filtCovHatSample, errFlagVec, vUnParEstDistrib
end


function non_par_bootstrap_distrib_filtered_par(model::SdErgm, N, obsT, indTvPar, ftot_0; nBootStrap = 100)

    vEstSdUnParBootDist = SharedArray(zeros(3*sum(model.indTvPar), nBootStrap))
    T = length(obsT)
    @time Threads.@threads for k=1:nBootStrap
        vEstSdUnParBootDist[:, k] = rand(1:T, T) |> (inds->(inds |> x-> DynNets.estimate(model, N, obsT; indTvPar=indTvPar, ftot_0 = ftot_0, shuffleObsInds = x) |> x-> getindex(x, 1) |> x -> DynNets.array_2_vec_all_par(model, x, model.indTvPar))) |> x -> DynNets.unrestrict_all_par(model, x)
    end

    return vEstSdUnParBootDist
end


function conf_bands_buccheri(model::SdErgm, N, obsT, indTvPar, fVecT_filt, distribFilteredSD, filtCovHatSample, quantilesVals::Vector{Vector{Float64}}, dropOutliers::Bool; nSample = 250, winsorProp=0.005)
    @debug "[conf_bands_buccheri][begin]"

    T = length(obsT)
    nErgmPar = number_ergm_par(model)

    parUncVarianceT = zeros(nErgmPar,T)
    filtUncVarianceT = zeros(nErgmPar,T)
 
    distribFilteredSD[isnan.(distribFilteredSD)] .= 0
    fVecT_filt[isnan.(fVecT_filt)] .= 0

    any(isnan.(filtCovHatSample)) ? error() : ()

    filtCovDiagHatMean = dropdims(mean(filtCovHatSample, dims=3), dims=3)
    @debug "[conf_bands_buccheri][size(filtCovDiagHatMean)=$(size(filtCovDiagHatMean))]"

    #for each time compute the variance of the filtered par under the normal distrib of static par
    indAmongTV = 0
    for k=1:nErgmPar
        indTvPar[k] ? indAmongTV += 1 : ()

        for t=1:T
            a_t_vec = distribFilteredSD[:,k,t]
            aHat_t = fVecT_filt[k,t] 
    
            # add filtering and parameter unc
            diff = a_t_vec[isfinite.(a_t_vec)] .- aHat_t
            extr = quantile(diff, [0.01, 0.99])
            if dropOutliers & (extr[1] != extr[2])
                diffNoOutl = filter(x->extr[1] < x < extr[2], diff)
            else    
                diffNoOutl = diff
            end

            # parUncVarianceT[k, t] = MinCovDet().fit(diffNoOutl).covariance_

            parUncVarianceT[k, t] = var(winsor(diffNoOutl, prop=winsorProp))
            
            isnan(parUncVarianceT[k, t]) ? (@show a_t_vec; @show aHat_t; @show diff; @show extr; @show t; @show k; error()) : ()

            if indTvPar[k]
                filtUncVarianceT[k, t] = filtCovDiagHatMean[indAmongTV,t]
            end
        end
    end


    nBands = length(quantilesVals)
    nQuant = 2*nBands
  
    confBandsPar = repeat(fVecT_filt, outer=(1,1,nBands, 2))
    confBandsParFilt = repeat(fVecT_filt, outer=(1,1,nBands, 2))

    for p = 1:number_ergm_par(model)
        for t=1:T
            sigmaPar = sqrt(parUncVarianceT[p,t])
            sigmaFiltPar = sqrt(parUncVarianceT[p,t] + filtUncVarianceT[p,t])
            for b=1:nBands
                if sigmaPar !== 0 
                    confBandsPar[p, t, b,  :] = quantile.(Normal(fVecT_filt[p, t], sigmaPar), quantilesVals[b])
                end
                # confBandsPar[p, t, b,  :] = fVecT_filt[p, t] .+  [1, -1] .* (1.96 * sigma)
                confBandsParFilt[p, t, b,  :] = quantile.(Normal(fVecT_filt[p, t], sigmaFiltPar), quantilesVals[b])
                # confBandsParFilt[p, t, b,  :] = fVecT_filt[p, t] .+  [1, -1] .* (1.96 * sigma)

                # catch e
                #     Logging.@error( (parUncVarianceT[p,t] , filtUncVarianceT[p,t], filtCovDiagHatMean))

                #     error(e)
                # end
            end
        end
    end
    return confBandsParFilt, confBandsPar
end


function conf_bands_par_uncertainty_blasques(model::SdErgm, obsT, fVecT_filt, distribFilteredSD, quantilesVals::Vector{Vector{Float64}}; nSample = 500)
    
    T = length(obsT)
    nErgmPar = number_ergm_par(model)

    parUncVarianceT = zeros(nErgmPar,T)

    nBands = length(quantilesVals)
    nQuant = 2*nBands   
    confBandsPar = repeat(fVecT_filt, outer=(1,1,nBands, 2))

    distribFilteredSD[isnan.(distribFilteredSD)] .= 0
    fVecT_filt[isnan.(fVecT_filt)] .= 0

    #for each time compute the variance of the filtered par under the normal distrib of static par
    
    for k=1:nErgmPar
        for t=1:T
            for b=1:nBands
                length(quantilesVals[b]) ==2 ? () : error()
             
                a_t_vec = distribFilteredSD[:,k,t]
                
                # add filtering and parameter unc
                confBandsPar[k, t, b, :] = Statistics.quantile(a_t_vec[.! isnan.(a_t_vec)], quantilesVals[b])
            end
        end
    end
   
    return confBandsPar
end


function conf_bands_coverage(parDgpTIn, confBandsIn; offset=0 )

    nErgmPar, T, nBands, nQuant  = size(confBandsIn)

    T = T-offset
    parDgpT = parDgpTIn[:, 1:end-offset]
    confBands = confBandsIn[:, 1+offset:end, :, :]

    nQuant ==2 ? () : error()
    nParDgp, TDgp = size(parDgpT)
    nParDgp == nErgmPar ? () : error()
    T == TDgp ? () : error()

    isCovered = falses(nErgmPar, T, nBands)
    for b in 1:nBands
        for p in 1:nErgmPar 
            for t in 1:T
                ub = confBands[p, t, b, 1] 
                lb = confBands[p, t, b, 2]
                ub<lb ? error("wrong bands ordering") : ()
                isCov = lb <= parDgpT[p, t] <= ub 
                isCovered[p, t, b] = isCov
            end
        end
    end
    return isCovered
end


function conf_bands_given_SD_estimates(model::SdErgm, N, obsT, vEstSdUnParAll, ftot_0, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE", dropOutliers = false, offset = 1,  nSample = 100 , mvSDUnParEstCov = Symmetric(zeros(3,3)), sampleStaticUnPar = zeros(3,3), winsorProp=0, xval=nothing)
    
    T = length(obsT)
    nStaticPar = length(indTvPar) + 2*sum(indTvPar)
    nErgmPar = number_ergm_par(model)

    vEstSdResParAll = restrict_all_par(model, vEstSdUnParAll)
    vEstSdResPar, vConstPar = divide_SD_par_from_const(model, vEstSdResParAll)

    fVecT_filt , target_fun_val_T, sVecT_filt = score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar; ftot_0 = ftot_0, vConstPar=vConstPar)

    isDistribFilteredSDAvail = false
    # are we providing a distribution of static parameters ?
    if sampleStaticUnPar != zeros(3,3)

        errFlagEstCov = false    
        errFlagSample = false

    else

        # if an external covariance matrix is given, use it to get a  static mv normal sample of stati parameters
        if mvSDUnParEstCov != Symmetric(zeros(3,3))
            errFlagEstCov = false    
            Logging.@info("[conf_bands_given_SD_estimates][using external gaussian covariance matrix]")

            sampleMvNormal = true

        elseif parUncMethod[1:2] == "PB"
            @debug "[conf_bands_given_SD_estimates][PB][vEstSdResParAll=$vEstSdResParAll]"
            # Parametric Bootstrap
            distribFilteredSD, filtCovHatSample, _,  sampleStaticUnPar = par_bootstrap_distrib_filtered_par(model, N, obsT, indTvPar, ftot_0, vEstSdResParAll)
            mean(errFlagVec) > 0.2 ? errFlagEstCov=true : errFlagEstCov = false

            if contains(parUncMethod, "SAMPLE")
                @debug "[conf_bands_given_SD_estimates][PB][SAMPLE]"
                
                    error("need to clean the sample of parametric bootstrapped paths for outliers(e.g. static estimates)")                
                isDistribFilteredSDAvail = true
                errFlagSample = false
                sampleMvNormal = false
            elseif contains(parUncMethod, "MVN")
                @debug "[conf_bands_given_SD_estimates][PB][MVN]"
                sampleMvNormal = true
                mvSDUnParEstCov = MinCovDet().fit(sampleStaticUnPar').covariance_
            end 

        elseif contains(parUncMethod, "WHITE-MLE")
            # Huber- White robust estimator
            try
                @debug "[conf_bands_given_SD_estimates][WHITE-MLE][vEstSdResParAll=$vEstSdResParAll]"
                mvSDUnParEstCov, errFlagEstCov = white_estimate_cov_mat_static_sd_par(model, N, obsT, indTvPar, ftot_0, vEstSdResParAll)
                    sampleMvNormal = true
            catch  
                Logging.@error("Error in computing white estimator")
                errFlagEstCov = true
            end
       
        elseif parUncMethod == "HESS"
                @debug "[conf_bands_given_SD_estimates][HESS][vEstSdResParAll=$vEstSdResParAll]"
            # Hessian of the objective function
            OPGradSum, HessSum = A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model, N, obsT, vEstSdResParAll, ftot_0)

            errFlagEstCov = false
            sampleMvNormal = true


            hess = HessSum
            mvSDUnParEstCov, minEigenVal = make_pos_def(Symmetric(pinv(hess)))
    
            
        elseif contains(parUncMethod, "NPB") 
            # non parametric bootstrap
            @debug "[conf_bands_given_SD_estimates][NPB][vEstSdResParAll=$vEstSdResParAll]"
            

            vEstSdUnParBootDist = non_par_bootstrap_distrib_filtered_par(model, N, obsT, indTvPar, ftot_0; nBootStrap = 100)
                
            if contains(parUncMethod, "SAMPLE")
                @debug "[conf_bands_given_SD_estimates][NPB][SAMPLE]"
                error("need to clean the sample of non parametric bootstrapped paths for outliers(e.g. static estimates)")                
                sampleMvNormal = false

            elseif contains(parUncMethod, "MVN")
                @debug "[conf_bands_given_SD_estimates][NPB][MVN]"
                sampleMvNormal = true
                mvSDUnParEstCov = MinCovDet().fit(sampleStaticUnPar').covariance_
            end 

        end
            
        if sampleMvNormal
            @debug "[conf_bands_given_SD_estimates][samplingMvNormal]"

            if errFlagEstCov
                @debug "[conf_bands_given_SD_estimates][error in estimating the covariance matrix]"
                mvSDUnParEstCov = Symmetric(zeros(nStaticPar, nStaticPar))
                distribFilteredSD = zeros(nSample, nErgmPar,T)
                filtCovHatSample = zeros(nErgmPar, T, nSample) 
                errFlagSample = true
            else        
                vEstSdUnPar = unrestrict_all_par(model, vEstSdResParAll)
                mvSDUnParEstCov = Symmetric(mvSDUnParEstCov)
                @debug "covMAt =$mvSDUnParEstCov"
                sampleStaticUnPar = distrib_static_par_from_mv_normal(model, N, obsT, indTvPar, ftot_0, vEstSdUnPar, mvSDUnParEstCov; nSample = nSample)
      
                errFlagSample = false
            end

        end
    end

    if !isDistribFilteredSDAvail 
        @debug "[conf_bands_given_SD_estimates][obtaining distribution of filtered latent]"
        distribFilteredSD, filtCovHatSample, errFlagSample = distrib_filtered_par_from_sample_static_par(model, N, obsT, indTvPar, ftot_0, sampleStaticUnPar)
    end

    # Compute confidence bands given distribution of filtered paths
    errFlagBands=false
    nBands = length(quantilesVals)
    confBandsPar = repeat(fVecT_filt, outer=(1,1,nBands, 2))
    confBandsFiltPar = repeat(fVecT_filt, outer=(1,1,nBands, 2))
    # try
        confBandsFiltPar,  confBandsPar = conf_bands_buccheri(model, N, obsT, indTvPar, fVecT_filt, distribFilteredSD, filtCovHatSample, quantilesVals, dropOutliers, winsorProp=winsorProp)
    # catch
        # errFlagBands = true
        # Logging.@error("Error in producing confidence bands") 
    # end
  
    if plotFlag
        @debug "[conf_bands_given_SD_estimates][plotting]"    
        fig, ax = plot_filtered_and_conf_bands(model, N, fVecT_filt, confBandsFiltPar ; parDgpTIn=parDgpT, nameConfBand1= "$parUncMethod - Filter+Par", nameConfBand2= "Par", confBands2In=confBandsPar, offset=offset, xval=xval)

    end

    errFlag = errFlagEstCov | errFlagSample | errFlagBands

    @debug "[conf_bands_given_SD_estimates][end]"    
    fVecT_filt::Array{<:Real,2}
    confBandsFiltPar::Array{<:Real,4}
    confBandsPar::Array{<:Real,4}
    errFlag::Bool
    mvSDUnParEstCov::LinearAlgebra.Symmetric{Float64,Array{Float64,2}}
    distribFilteredSD::Array{Float64,3}

    if plotFlag
        return (; fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD, fig, ax )
    else
        return (; fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD )
    end
end


function estimate_filter_and_conf_bands(model::SdErgm, A_T, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE",show_trace = false)
    
    N = size(A_T)[1]

    obsT = seq_of_obs_from_seq_of_mats(model, A_T)



    obsT, vEstSdResParAll, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0 = estimate_and_filter(model, N, obsT; indTvPar = indTvPar, show_trace = show_trace )
    Logging.@debug("[estimate_filter_and_conf_bands][vEstSdResParAll = $vEstSdResParAll] ")

    vEstSdUnPar = unrestrict_all_par(model, vEstSdResParAll)

    fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = conf_bands_given_SD_estimates(model, N, obsT, vEstSdUnPar, ftot_0, quantilesVals; indTvPar = indTvPar, parDgpT=parDgpT, plotFlag=plotFlag, parUncMethod = parUncMethod)

    return obsT, vEstSdResParAll, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag

end

