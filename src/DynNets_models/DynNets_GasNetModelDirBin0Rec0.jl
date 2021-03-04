import ..StaticNets:NetModelDirBin0Rec0, ergm_term_string

abstract type GasNetModelDirBin0Rec0 <: GasNetModel end


Base.@kwdef struct  GasNetModelDirBin0Rec0_mle <: GasNetModelDirBin0Rec0
    staticModel = NetModelDirBin0Rec0()
    indTvPar :: BitArray{1} = trues(2) #  what parameters are time varying   ?
    scoreScalingType::String = "HESS" # String that specifies the rescaling of the score. For a list of possible choices see function scalingMatGas
end
export GasNetModelDirBin0Rec0_mle


name(x::GasNetModelDirBin0Rec0_mle) = "GasNetModelDirBin0Rec0_mle($(x.indTvPar), scal = $(x.scoreScalingType))"
export name

Base.string(x::GasNetModelDirBin0Rec0) = name(x::GasNetModelDirBin0Rec0) 
export string


"""
Given the flag of constant parameters, a starting value for their unconditional means (their constant value, for those constant), return a starting point for the optimization
"""
function starting_point_optim(model::T where T <:GasNetModelDirBin0Rec0, indTvPar, UM; indTargPar =  falses(100))
    
    nTvPar = sum(indTvPar)
    NTargPar = sum(indTargPar)
    nErgmPar = length(indTvPar)
    
    # #set the starting points for the optimizations
    B0_Re  = 0.98; B0_Un = log(B0_Re ./ (1 .- B0_Re ))
    ARe_min =0.00001
    A0_Re  = 0.1 ; A0_Un = log(A0_Re  .-  ARe_min)
    
    # starting values for the vector of parameters that have to be optimized
    vParOptim_0 = zeros(nErgmPar + nTvPar*2 - NTargPar)
    last = 0
    for i=1:nErgmPar
        if indTvPar[i]
            if indTargPar[i]
                vParOptim_0[last+1:last+2] = [ B0_Un; A0_Un]
                last+=2
            else
                vParOptim_0[last+1:last+3] = [UM[i]*(1 .- B0_Re) ; B0_Un; A0_Un]
                last+=3
            end
        else
            vParOptim_0[last+1] = UM[i]
            last+=1
        end
    end
    return vParOptim_0, ARe_min
end


function scalingMatGas(model::T where T<: GasNetModelDirBin0Rec0,expMat::Array{<:Real,2},I_tm1::Array{<:Real,2})
    "Return the matrix required for the scaling of the score, given the expected
     matrix and the Scaling matrix at previous time. "
    if uppercase(model.scoreScalingType) == ""
        scalingMat = 1 #
    elseif uppercase(model.scoreScalingType) == "FISHER-EWMA"
        error()
        # λ = 0.5
        #
        # I = expMat.*(1-expMat)
        # diagI = sum(I,dims = 2)
        # [I[i,i] = diagI[i] for i=1:length(diagI) ]
        # I_t =  λ*I + (1-λ) *I_tm1
        # scalingMat = sqrt(I) ##
    elseif uppercase(model.scoreScalingType) == "FISHER-DIAG"
        error()
    end
    return scalingMat
end


function target_function_t(model::GasNetModelDirBin0Rec0_mle, obs_t, f_t)

    L, R, N = obs_t

    ll = StaticNets.logLikelihood( StaticNets.NetModelDirBin0Rec0(), L, R, N, f_t)

    return ll
end


function static_estimate(model::GasNetModelDirBin0Rec0_mle, statsT)
    L_mean  = mean([stat[1] for stat in statsT ])
    R_mean  = mean([stat[2] for stat in statsT ])
    N_mean  = mean([stat[3] for stat in statsT ])
    
    staticPars = StaticNets.estimate(model.staticModel, L_mean, R_mean, N_mean )
    return staticPars
end


GasNetModelDirBin0Rec0_pmle() = SdErgmPml(ergm_term_string(NetModelDirBin0Rec0()), true)
export GasNetModelDirBin0Rec0_pmle



import ..StaticNets:ergm_par_from_mean_vals

alpha_beta_to_theta_eta(α, β, N) = collect(ergm_par_from_mean_vals(NetModelDirBin0Rec0(), α*n_pox_dir_links(N), β*n_pox_dir_links(N), N))


theta_eta_to_alpha_beta(θ, η, N) =  collect(exp_val_stats(NetModelDirBin0Rec0(), θ, η, N))./n_pox_dir_links(N)



get_theta_eta_seq_from_alpha_beta(alpha_beta_seq, N) = reduce(hcat, [alpha_beta_to_theta_eta(ab[1], ab[2], N) for ab in eachcol(alpha_beta_seq)])


function beta_min_max_from_alpha_min(minAlpha, N;  minBeta = minAlpha/5 )
    # min beta val such that the minimum exp value rec links is not close zero   
     

    # min beta val such that the maximum exp value rec links is not close to the physical upper bound 
    
    physicalUpperBound = minAlpha/2
    maxBeta =  physicalUpperBound - minBeta
    
    minBeta >= maxBeta ? error((minBeta, maxBeta)) : ()

    return minBeta, maxBeta 
end


function sample_time_var_par_from_dgp(model::GasNetModelDirBin0Rec0, dgpType, N, T;  minAlpha = 0.25, maxAlpha = 0.3, nCycles = 2, phaseshift = 0.1, plotFlag=false, phaseAlpha = 0, sigma = 0.01, B = 0.95, A=0.001, maxAttempts = 5000, indTvPar=trues(number_ergm_par(model)))

    minBeta, maxBeta =  beta_min_max_from_alpha_min(minAlpha, N)


    Logging.@debug((;minBeta, maxBeta))
    Logging.@debug((;minAlpha, maxAlpha))
    
    minBetaSin, maxBetaSin = minBeta, maxBeta# .* [1,1.5]
    betaConst = minBeta#(maxBeta - minBeta)/2
    #phaseAlpha = rand()  * 2π
    phaseBeta = phaseAlpha + phaseshift * 2π

    α_β_parDgpT = zeros(2,T)
    
    for n=1:maxAttempts
        Nsteps1= 2
        if dgpType=="SIN"
            α_β_parDgpT[1,:] = dgpSin(minAlpha, maxAlpha, nCycles, T; phase = phaseAlpha)# -3# randSteps(0.05,0.5,2,T) #1.5#.00000000000000001
            α_β_parDgpT[2,:] .= dgpSin(minBetaSin, maxBetaSin, nCycles, T;phase= phaseBeta )# -3# randSteps(0.05,0.5,2,T) #1.5#.00000000000000001
        elseif dgpType=="steps"
            α_β_parDgpT[1,:] = randSteps(α_β_minMax[1], α_β_minMax[2], Nsteps1,T)
            α_β_parDgpT[2,:] = randSteps(η_0_minMax[1], η_0_minMax[2], Nsteps1,T)
        elseif dgpType=="AR"
            meanValAlpha = (minAlpha+maxAlpha)/2
            meanValBeta = (minBeta+maxBeta)/2

            θ_η_UM = alpha_beta_to_theta_eta(meanValAlpha, meanValBeta, N)

            θ_η_parDgpT = zeros(Real,2,T)

            θ_η_parDgpT[1,:] = dgpAR(θ_η_UM[1],B,sigma,T )
            θ_η_parDgpT[2,:] = dgpAR(θ_η_UM[2],B,sigma,T )

            break

        end

        if dgpType == "SD"
            meanValAlpha = (minAlpha+maxAlpha)/2
            meanValBeta = (minBeta+maxBeta)/2
    
            UM = alpha_beta_to_theta_eta(meanValAlpha, meanValBeta, N)

            model_mle = GasNetModelDirBin0Rec0_mle()
            Logging.@warn(" the score driven DGP used is the Maximum Likelihood one. PML is too slow")

            vUnPar, ~ = DynNets.starting_point_optim(model_mle, indTvPar, UM)
            vResParDgp = DynNets.restrict_all_par(model_mle, indTvPar, vUnPar)

            vResParDgp[2:3:end] .= B
            vResParDgp[3:3:end].= A
            vResParDgp = Real.(vResParDgp)

            fVecT, ~, ~ = DynNets.score_driven_filter_or_dgp( model_mle, N, vResParDgp, indTvPar; dgpNT = (N,T))

            global θ_η_parDgpT = fVecT

            break

        else
            try 
                global θ_η_parDgpT = get_theta_eta_seq_from_alpha_beta(α_β_parDgpT, N)
                break
            catch
                Logging.@warn("Dgp attempt failed")
            end
        end

    end

    if plotFlag
        fig, ax = subplots(2,2)
        ax[1,1].plot(α_β_parDgpT[1,:], "k")
        ax[2,1].plot(α_β_parDgpT[2,:], "k")
    end
    if plotFlag
        ax[1,2].plot(θ_η_parDgpT[1,:], "k")
        ax[2,2].plot(θ_η_parDgpT[2,:], "k")
    end
    return θ_η_parDgpT
end


function list_example_dgp_settings(model::GasNetModelDirBin0Rec0)

    dgpSetAR = (type = "AR", opt = (B =0.98, sigma = 0.002))

    dgpSetSIN = (type = "SIN", opt = ( nCycles=1.5))

    dgpSetSD = (type = "SD", opt = (B =0.98, A = 0.3))

    return (; dgpSetAR, dgpSetSIN, dgpSetSD)
end


function sample_mats_sequence(model::GasNetModelDirBin0Rec0, parDgpT::Matrix, N )
    T = size(parDgpT)[2]
    A_T_dgp = zeros(Int8, N, N, T)
    for t=1:T
        diadProb = StaticNets.diadProbFromPars(StaticNets.NetModelDirBin0Rec0(), parDgpT[:,t] )
        A_T_dgp[:,:,t] = StaticNets.samplSingMatCan(StaticNets.NetModelDirBin0Rec0(), diadProb, N)
    end
    return A_T_dgp
end


function sample_est_mle_pmle(model::GasNetModelDirBin0Rec0, parDgpT, N, Nsample; plotFlag = true, regimeString="")

    model_mle = GasNetModelDirBin0Rec0_mle()
    model_pmle = GasNetModelDirBin0Rec0_pmle()

    T = size(parDgpT)[2]
    indTvPar = trues(2)
   
    if plotFlag
        fig1, ax_mle = subplots(2,1)
        fig2, ax_pmle = subplots(2,1)
    end

    rmse(x::Matrix) = sqrt.(mean(x.^2, dims=2))

    rmse_mle = zeros(2,Nsample)
    rmse_pmle = zeros(2,Nsample)
    vEstSd_mle = zeros(8, Nsample)
    vEstSd_pmle = zeros(8, Nsample)

    for n=1:Nsample
        ## sample dgp
        A_T_dgp = sample_mats_sequence(model_mle, parDgpT, N)
        stats_T_dgp = [statsFromMat(model_mle, A_T_dgp[:,:,t]) for t in 1:T ]
        change_stats_T_dgp = change_stats(model_pmle, A_T_dgp)


        ## estimate SD
        estPar_pmle, conv_flag,UM_mple , ftot_0_mple = estimate(model_pmle, N,  change_stats_T_dgp; indTvPar=indTvPar,indTargPar=indTvPar)
        vResEstPar_pmle = DynNets.array2VecGasPar(model_pmle, estPar_pmle, indTvPar)
        fVecT_filt_p , logLikeVecT, sVecT_filt_p = score_driven_filter( model_pmle, N, obsT, vResEstPar_pmle, indTvPar; change_stats_T_dgp, ftot_0 = ftot_0_mple)
        vEstSd_pmle[:,n] = vcat(vResEstPar_pmle, ftot_0_mple)

        estPar_mle, conv_flag,UM_mle , ftot_0_mle = estimate(model_mle, N, stats_T_dgp; indTvPar=indTvPar, indTargPar=indTvPar)
        vResEstPar_mle = DynNets.array2VecGasPar(model_mle, estPar_mle, indTvPar)
        fVecT_filt , logLikeVecT, sVecT_filt = score_driven_filter_or_dgp( model_mle, N, vResEstPar_mle, indTvPar; obsT = stats_T_dgp, ftot_0=ftot_0_mle)
        vEstSd_mle[:,n] = vcat(vResEstPar_mle, ftot_0_mle)

        if plotFlag
            ax_mle[1].plot(fVecT_filt[1,:], "b", alpha =0.5)
            ax_mle[2].plot(fVecT_filt[2,:], "b", alpha =0.5)
            ax_pmle[1].plot(fVecT_filt_p[1,:], "r", alpha =0.5)
            ax_pmle[2].plot(fVecT_filt_p[2,:], "r", alpha =0.5)
        end

        rmse_mle[:,n] = rmse(fVecT_filt.- parDgpT)
        rmse_pmle[:,n] =rmse(fVecT_filt_p.- parDgpT)
    end
    
    avg_rmse_pmle = round.(mean(drop_nan_col(rmse_pmle), dims=2), digits=3)
    avg_rmse_mle = round.(mean(drop_nan_col(rmse_mle), dims=2), digits=3)

    if plotFlag
        ax_mle[1].plot(parDgpT[1,:], "k")
        ax_mle[2].plot(parDgpT[2,:], "k")
        ax_pmle[1].plot(parDgpT[1,:], "k")
        ax_pmle[2].plot(parDgpT[2,:], "k")

        ax_mle[1].set_title("MLE-SDERGM  N= $N , θ rmse = $(avg_rmse_mle[1]) " * regimeString)   
        ax_mle[2].set_title("MLE-SDERGM  N= $N , η rmse = $(avg_rmse_mle[2]) " * regimeString)   
        
        ax_pmle[1].set_title("PMLE-SDERGM  N= $N , θ rmse = $(avg_rmse_pmle[1])  " * regimeString)   
        ax_pmle[2].set_title("PMLE-SDERGM  N= $N , η rmse = $(avg_rmse_pmle[2])  " * regimeString)   
                
        fig1.tight_layout()
        fig2.tight_layout()
    end
    
    return (; vEstSd_mle, vEstSd_pmle, avg_rmse_mle, avg_rmse_pmle)
end

#endregion

#region Uncertainties filtered parameters


function A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model::GasNetModelDirBin0Rec0, N, obsT, vEstSdResPar, indTvPar, ftot_0)

    T = length(obsT)
    nPar = length(vEstSdResPar)
    gradT = zeros(nPar, T)
    hessT = zeros(nPar, nPar, T)

    vecUnParAll = unrestrict_all_par(model, indTvPar, vEstSdResPar)    


    for t = 1:T
        function obj_fun_t(xUn)

            xRe = restrict_all_par(model, indTvPar, xUn)

            vecSDParRe, vConstPar = divide_SD_par_from_const(model, indTvPar, xRe)

            oneInADterms  = (StaticNets.maxLargeVal + vecSDParRe[1])/StaticNets.maxLargeVal

            fVecT_filt, logLikeVecT, ~ = DynNets.score_driven_filter( model, N, obsT[1:t], vecSDParRe, indTvPar; vConstPar =  vConstPar, ftot_0 = ftot_0 .* oneInADterms)
        
            return - logLikeVecT[end]
        end


        obj_fun_t(vecUnParAll)

        gradT[:,t] = deepcopy(ForwardDiff.gradient(obj_fun_t, vecUnParAll))
        hessT[:,:,t] =  deepcopy(ForwardDiff.hessian(obj_fun_t, vecUnParAll))
    end

    # function obj_fun_T(xUn)

    #     vecSDParUn, vConstPar = DynNets.divide_SD_par_from_const(model, indTvPar, xUn)

    #     vecSDParRe = DynNets.restrict_SD_static_par(model, vecSDParUn)

    #     oneInADterms  = (StaticNets.maxLargeVal + vecSDParRe[1])/StaticNets.maxLargeVal

    #     fVecT_filt, target_fun_val_T, ~ = DynNets.score_driven_filter_or_dgp( model, N,  vecSDParRe, indTvPar; obsT = obsT, vConstPar =  vConstPar, ftot_0 = ftot_0 .* oneInADterms)

    #     return - target_fun_val_T
    # end
    # hessCumT =  ForwardDiff.hessian(obj_fun_T, vecUnParAll)
    # HessSum = hessCumT./(T-2)

    OPGradSum = sum([gt * gt' for gt in eachcol(gradT[:,2:end])] )
    HessSum = dropdims(sum(hessT[:,:,2:end], dims=3 ), dims=3)

    return OPGradSum, HessSum
end


function white_estimate_cov_mat_static_sd_par(model::GasNetModelDirBin0Rec0,  N,obsT, indTvPar, ftot_0, vEstSdResPar; returnAllMats=false)

    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    errorFlag = false

   
    OPGradSum, HessSum = A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model, N, obsT, vEstSdResPar, indTvPar, ftot_0)

    parCovHat = pinv(HessSum) * OPGradSum * pinv(HessSum)
    
    parCovHatPosDef, minEigenVal = make_pos_def(parCovHat)
   
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


function divide_in_B_A_mats_as_if_all_TV(model::GasNetModelDirBin0Rec0, indTvPar, vEstSdResPar)
  
    nTvPar = sum(indTvPar)
    nErgmPar = number_ergm_par(model)
    B = zeros(nTvPar)
    A = zeros(nTvPar)
    
    lastInd = 0
    for e in 1:nErgmPar
        if indTvPar[e]
            bInd = lastInd+2
            aInd = lastInd+3
            B[e] = vEstSdResPar[bInd] 
            A[e] = vEstSdResPar[aInd] 
            lastInd += 3
        else
            lastInd += 1
        end

    end
    return B, A
end


function distrib_filtered_par_from_mv_normal(model::GasNetModelDirBin0Rec0, N, obsT, indTvPar, ftot_0, vEstSdResPar, mvSDUnParEstCov; nSample = 1000)
        
    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    nTvPar = sum(indTvPar)
    any(.!indTvPar) ? error("assuming all parameters TV. otherwise need to set filter unc to zero for static ones") : ()    


    distribFilteredSD = zeros(nSample, nErgmPar,T)
    filtCovHatSample = zeros(nErgmPar, T, nSample)

    vEstSdUnPar = unrestrict_all_par(model, indTvPar, vEstSdResPar)
   
    zeroMeanSample =  rand(MvNormal(mvSDUnParEstCov), nSample ) 
    sampleUnParAll = zeroMeanSample .+ vEstSdUnPar

    for n=1:nSample
        vResPar = restrict_all_par(model, indTvPar, sampleUnParAll[:,n])

        vecSDParRe, vConstPar = divide_SD_par_from_const(model, indTvPar, vResPar)

        distribFilteredSD[n, :, :] , ~, ~, scalMatT = DynNets.score_driven_filter( model, N, obsT, vecSDParRe, indTvPar;ftot_0=ftot_0, vConstPar=vConstPar)

        BMatSD, AMatSD = divide_in_B_A_mats_as_if_all_TV(model, indTvPar, vResPar)

        constFiltUncCoeff = (BMatSD.^(-1)).*AMatSD
        
        for t in 1:T 
            filtCovHatSample[:,t,n] = constFiltUncCoeff./ scalMatT[:,:,t][I(nTvPar)]
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

    filtCovHatSampleGood = filtCovHatSample[:,:,.!indBadSamples]

    return distribFilteredSD, filtCovHatSampleGood, mvSDUnParEstCov, errFlag
end


"""
Use parametric bootstrap to sample from the distribution of static parameters of the SD filter
"""
function par_bootstrap_distrib_filtered_par(model::GasNetModelDirBin0Rec0, N, obsT, indTvPar, ftot_0, vEstSdResPar; nSample = 50)

    T = length(obsT)
    nErgmPar = number_ergm_par(model)
     
     
    nStaticSdPar = length(vEstSdResPar)
    vUnParEstDistrib = zeros(length(vEstSdResPar), nSample)
    distribFilteredSD = zeros(nSample, nErgmPar,T)
    filtCovHatSample = zeros(nErgmPar, nSample)
    errFlagVec = falses(nSample)

    for n=1:nSample
        
        # try
            # sample SD dgp
            fVecTDgp, A_T_dgp, ~ = DynNets.score_driven_filter_or_dgp( model, N, vEstSdResPar, indTvPar; dgpNT = (N,T))
            obsTNew = [statsFromMat(model, A_T_dgp[:,:,t]) for t in 1:T ]

            # Estimate SD static params from SD DGP
            arrayAllParHat, conv_flag,UM, ftot_0 = estimate(model, N, obsTNew; indTvPar=indTvPar, indTargPar=falses(2), ftot_0=ftot_0)

            vResPar = DynNets.array2VecGasPar(model, arrayAllParHat, indTvPar)
    
            vUnPar = unrestrict_all_par(model, indTvPar, deepcopy(vResPar))

            vUnParEstDistrib[:,n] =  vUnPar

            vecSDParRe, vConstPar = divide_SD_par_from_const(model, indTvPar, vResPar)

            distribFilteredSD[n, :, :] , ~, ~ = DynNets.score_driven_filter( model, N, obsT, vecSDParRe, indTvPar; ftot_0=ftot_0, vConstPar=vConstPar)

            BMatSD, AMatSD = divide_in_B_A_mats_as_if_all_TV(model, indTvPar, vResPar)

            filtCovHat = (BMatSD.^(-1)).*AMatSD
            filtCovHat[.!indTvPar] .= 0
            filtCovHatSample[:,n] = filtCovHat
            
        # catch
        #     errFlagVec[n] = true
        # end

     
    end

    mvSDUnParEstCov = cov(vUnParEstDistrib')

    return distribFilteredSD, filtCovHatSample, errFlagVec, mvSDUnParEstCov
end



function conf_bands_buccheri(model::GasNetModelDirBin0Rec0, obsT, indTvPar, fVecT_filt, distribFilteredSD, filtCovHatSample, quantilesVals::Vector{Vector{Float64}}; nSample = 500, )


    T = length(obsT)
    nErgmPar = number_ergm_par(model)

    parUncVarianceT = zeros(nErgmPar,T)
    filtUncVarianceT = zeros(nErgmPar,T)
 
    distribFilteredSD[isnan.(distribFilteredSD)] .= 0
    fVecT_filt[isnan.(fVecT_filt)] .= 0

    any(isnan.(filtCovHatSample)) ? error() : ()

    filtCovDiagHatMean = mean(filtCovHatSample, dims=3)

    #for each time compute the variance of the filtered par under the normal distrib of static par

    for k=1:nErgmPar
        if indTvPar[k]
            indAmongTv = sum(indTvPar[1:k-1]) +1 
            for t=1:T
                a_t_vec = distribFilteredSD[:,indAmongTv,t]
                aHat_t = fVecT_filt[k,t] 
        
                # add filtering and parameter unc
                diff = a_t_vec[isfinite.(a_t_vec)] .- aHat_t
                extr = quantile(diff, [0.01, 0.99])
                if extr[1] != extr[2]
                    diffNoOutl = filter(x->extr[1] < x < extr[2], diff)
                    parUncVarianceT[k, t] = var(diffNoOutl)
                else    
                    parUncVarianceT[k, t] = var(diff)
                end                
                isnan(parUncVarianceT[k, t]) ? (@show a_t_vec; @show aHat_t; @show diff; @show extr; @show t; @show k; error()) : ()

                filtUncVarianceT[k, t] = filtCovDiagHatMean[indAmongTv,t]
            end
        else         
            indAmongAll = sum(indTvPar[1:k-1])*3 +  sum(.!indTvPar[1:k-1]) + 1                        
            parUncVarianceT[k, :] .= mvSDUnParEstCov[indAmongAll,indAmongAll]
            filtUncVarianceT[k, t] = 0
        end
    end


    nBands = length(quantilesVals)
    nQuant = 2*nBands
  
    confBandsPar = repeat(fVecT_filt, outer=(1,1,nBands, 2))
    confBandsParFilt = repeat(fVecT_filt, outer=(1,1,nBands, 2))

    for p =1:number_ergm_par(model)
        for t=1:T
            for b=1:nBands
                length(quantilesVals[b]) ==2 ? () : error()
                confBandsPar[p, t, b,  :] = quantile.(Normal(fVecT_filt[p, t], sqrt(parUncVarianceT[p,t])), quantilesVals[b])
                try
                    confBandsParFilt[p, t, b,  :] = quantile.(Normal(fVecT_filt[p, t], sqrt(parUncVarianceT[p,t] + filtUncVarianceT[p,t])), quantilesVals[b])
                catch e
                    Logging.@error( (parUncVarianceT[p,t] , filtUncVarianceT[p,t], filtCovDiagHatMean))

                    error(e)
                end
            end
        end
    end
    return confBandsParFilt, confBandsPar
end


function conf_bands_par_uncertainty_blasques(model::GasNetModelDirBin0Rec0, obsT, fVecT_filt, distribFilteredSD, quantilesVals::Vector{Vector{Float64}}; nSample = 500)
    
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


function conf_bands_coverage(parDgpT, confBands )

    nErgmPar, T, nBands, nQuant  = size(confBands)
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


function plot_filtered_and_conf_bands(model::GasNetModelDirBin0Rec0, N, fVecT_filt, confBands1; confBands2 =zeros(2,2), parDgpT=zeros(2,2), nameConfBand1="1", nameConfBand2="2")

    T = size(fVecT_filt)[2]

    nBands = size(confBands1)[3]

    fig1, ax = subplots(2,1)
    for p in 1:2
        x = 1:T
        bottom = minimum(fVecT_filt[p,:])
        top = maximum(fVecT_filt[p,:])
        if parDgpT != zeros(2,2)
            ax[p].plot(x, parDgpT[p,:], "k", alpha =0.5)
            bottom = minimum(parDgpT[p,:])
            top = maximum(parDgpT[p,:])
        end
        delta = top-bottom
        margin = 0.5*delta
        ax[p].plot(x, fVecT_filt[p,:], "b", alpha =0.5)
        # ax[p].set_ylim([bottom - margin, top + margin])
        for b in 1:nBands
            if sum(confBands1) !=0
                ax[p].fill_between(x, confBands1[p, :, b,1], y2 =confBands1[p,:,b, 2],color =(0.9, 0.2 , 0.2, 0.1), alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
            end
            if confBands2 != zeros(2,2)
                ax[p].plot(x, confBands2[p, :, b, 1], "-g", alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
                ax[p].plot(x, confBands2[p,:, b, 2], "-g", alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
            end
        end
        ax[p].grid()
        
    end

    if sum(confBands1) !=0
        cov1 = round(mean(conf_bands_coverage(parDgpT, confBands1)), digits=2)
    else
        cov1 = 0
    end
    titleString = "$(name(model)), N = $N, T=$T, \n  $nameConfBand1 = $cov1"

    if confBands2 != zeros(2,2)
        cov2 = round(mean(conf_bands_coverage(parDgpT, confBands2)), digits=2)
        titleString = titleString * " $nameConfBand2 = $cov2"
    end

    ax[1].set_title(titleString)
end


function estimate_and_filter(model::GasNetModelDirBin0Rec0, A_T_dgp; indTvPar = model.indTvPar)

    N = size(A_T_dgp)[1]
    T = size(A_T_dgp)[3]
    obsT = [statsFromMat(model, A_T_dgp[:,:,t]) for t in 1:T ]

    estSdResPar, conv_flag, UM_mple, ftot_0 = estimate(model, N, obsT; indTvPar=indTvPar, indTargPar=falses(2))


    vEstSdResPar = array2VecGasPar(model, estSdResPar, indTvPar)

    fVecT_filt , target_fun_val_T, sVecT_filt = score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;ftot_0 = ftot_0)

    return obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0
end
    

function conf_bands_given_SD_estimates(model::GasNetModelDirBin0Rec0, obsT, N, vEstSdResPar, ftot_0, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE" )
    
    T = length(obsT)

    fVecT_filt , target_fun_val_T, sVecT_filt = score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar; ftot_0 = ftot_0)


    if contains(parUncMethod, "PAR-BOOTSTRAP")

        distribFilteredSD, filtCovHatSample, errFlagVec, mvSDUnParEstCov = par_bootstrap_distrib_filtered_par(model, N, obsT, indTvPar, ftot_0, vEstSdResPar)
        
        mean(errFlagVec) > 0.1 ? errFlagEstCov=true : errFlagEstCov = false

        if parUncMethod[14:end] == "SAMPLE"
            
        elseif parUncMethod[14:end] == "COV-MAT"

            distribFilteredSD, filtCovHatSample, mvSDUnParEstCov, errFlagMvNormSample = distrib_filtered_par_from_mv_normal(model, N, obsT, indTvPar, ftot_0, vEstSdResPar, mvSDUnParEstCov)

        end

    elseif parUncMethod == "WHITE-MLE"

         mvSDUnParEstCov, errFlagEstCov = white_estimate_cov_mat_static_sd_par(model, N, obsT, indTvPar, ftot_0, vEstSdResPar)
         
        distribFilteredSD, filtCovHatSample, mvSDUnParEstCov, errFlagMvNormSample = distrib_filtered_par_from_mv_normal(model, N, obsT, indTvPar, ftot_0, vEstSdResPar, mvSDUnParEstCov)
    end

    confBandsFiltPar,  confBandsPar = conf_bands_buccheri(model, obsT, indTvPar, fVecT_filt, distribFilteredSD, filtCovHatSample, quantilesVals)

  
    if plotFlag
        plot_filtered_and_conf_bands(model, N, fVecT_filt, confBandsFiltPar ; parDgpT=parDgpT, nameConfBand1= "$parUncMethod - Filter+Par", nameConfBand2= "Par", confBands2=confBandsPar)

    end

    errFlag = errFlagEstCov | errFlagMvNormSample

    return fVecT_filt::Array{<:Real,2}, confBandsFiltPar::Array{<:Real,4}, confBandsPar::Array{<:Real,4}, errFlag::Bool, mvSDUnParEstCov::LinearAlgebra.Symmetric{Float64,Array{Float64,2}}, distribFilteredSD::Array{Float64,3}
end


function estimate_filter_and_conf_bands(model::GasNetModelDirBin0Rec0, A_T_dgp, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE")
    
    N = size(A_T_dgp)[1]
    obsT, vEstSdResPar, fVecT_filt, ~, ~, conv_flag, ftot_0 = estimate_and_filter(model, A_T_dgp; indTvPar = indTvPar)
    
    fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = conf_bands_given_SD_estimates(model, obsT, N, vEstSdResPar, ftot_0, quantilesVals; indTvPar = indTvPar, parDgpT=parDgpT, plotFlag=plotFlag, parUncMethod = parUncMethod)

    return obsT, vEstSdResPar, fVecT_filt, confBandsFiltPar, confBandsPar, errFlag

end



function simulate_and_estimate_parallel(model::GasNetModelDirBin0Rec0, dgpSettings, T, N, nSample)

    count = SharedArray(ones(1))
    res = @sync @distributed vcat for k=1:nSample
        
        Logging.@info("Estimating N = $N , T=$T iter n $(count[1])")

        parDgpT = DynNets.dgp_misspecified(model, dgpSettings.type, N, T;  dgpSettings.opt...)

        A_T_dgp = DynNets.sample_dgp(model, parDgpT,N)

        obsT, vEstSdResPar, fVecT_filt, ~, ~, conv_flag, ftot_0 = DynNets.estimate_and_filter(model, A_T_dgp)
            
        count[1] += 1

        (;obsT, parDgpT, vEstSdResPar, fVecT_filt, conv_flag, ftot_0)
        
    end


    allObsT = [r.obsT for r in res]
    allParDgpT = reduce(((a,b) -> cat(a,b, dims=3)), [r.parDgpT for r in res])
    allvEstSdResPar = reduce(((a,b) -> cat(a,b, dims=2)), [r.vEstSdResPar for r in res])
    allfVecT_filt = reduce(((a,b) -> cat(a,b, dims=3)), [r.fVecT_filt for r in res])
    allConvFlag = [r.conv_flag for r in res]
    allftot_0 = reduce(((a,b) -> cat(a,b, dims=2)), [r.ftot_0 for r in res])

    Logging.@info("The fraction of estimates that resulted in errors is $(mean(.!allConvFlag)) ")

    return allObsT, allvEstSdResPar, allfVecT_filt, allParDgpT, allConvFlag, allftot_0
end




#endregion