import ..StaticNets:ErgmDirBin0Rec0, ergm_term_string

abstract type SdErgmDirBin0Rec0 <: SdErgm end


Base.@kwdef struct  SdErgmDirBin0Rec0_mle <: SdErgmDirBin0Rec0
    staticModel = ErgmDirBin0Rec0()
    indTvPar :: BitArray{1} = trues(2) #  what parameters are time varying   ?
    scoreScalingType::String = "FISH_D" # String that specifies the rescaling of the score. For a list of possible choices see function scalingMatGas
    options::SortedDict{Any, Any} = SortedDict("integrated"=>false, "initMeth"=>"estimateFirstObs")
end
export SdErgmDirBin0Rec0_mle

d = SortedDict(["first" => 31, "second" => "val"])


function name(x::SdErgmDirBin0Rec0_mle)  
    if isempty(x.options)
        optString = ""
    else
        optString = ", " * reduce(*,["$k = $v, " for (k,v) in x.options])[1:end-2]
    end

    "SdErgmDirBin0Rec0_mle($(x.indTvPar), scal = $(x.scoreScalingType)$optString)"
end
export name


Base.string(x::SdErgmDirBin0Rec0) = name(x::SdErgmDirBin0Rec0) 
export string


"""
Given the flag of constant parameters, a starting value for their unconditional means (their constant value, for those constant), return a starting point for the optimization
"""
function starting_point_optim(model::T where T <:SdErgmDirBin0Rec0, UM; indTargPar =  falses(100))
    
    nTvPar = sum(indTvPar)
    NTargPar = sum(indTargPar)
    nErgmPar = length(indTvPar)
    
    # #set the starting points for the optimizations
    B0_Re  = 0.98; B0_Un = log(B0_Re ./ (1 .- B0_Re ))
    ARe_min =1e-8
    if contains(model.scoreScalingType, "FISH")
        A0_Re  = 0.00001 ; 
    elseif contains(model.scoreScalingType, "HESS")
        A0_Re  = 0.1 ; 
    end    
    
    A0_Un = log(A0_Re  .-  ARe_min)
    
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


"""
Return the matrix required for the scaling of the score, given the expected
    matrix and the Scaling matrix at previous time. 
"""
function scalingMatGas(model::T where T<: SdErgmDirBin0Rec0,expMat::Array{<:Real,2},I_tm1::Array{<:Real,2})
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


function target_function_t(model::SdErgmDirBin0Rec0_mle, obs_t, N, f_t)

    L, R, N = obs_t

    ll = StaticNets.logLikelihood( StaticNets.ErgmDirBin0Rec0(), L, R, N, f_t)

    return ll
end


function target_function_t_grad(model::SdErgmDirBin0Rec0_mle, obs_t, N, f_t)
    
    L, R, N = obs_t
    
    θ, η = f_t
    x = exp(θ)

    x2y = exp(2θ + η)
    
    Z = (1 + 2x + x2y )
    g_θ = L - (N^2-N) * (x + x2y)/Z
    g_η = R - (N^2-N)/2* (x2y)/Z
    
    grad_tot_t = [g_θ, g_η]

    return grad_tot_t
end


function target_function_t_hess(model::SdErgmDirBin0Rec0_mle, obs_t, N, f_t)
    
    θ, η = f_t
    x = exp(θ)

    x2y = exp(2θ + η)
    xy = exp(θ + η)
    
    Z = (1 + 2x + x2y )
    h_θ_θ = - x*(1 + 2*xy + x2y)
    h_θ_η = - x2y *(1 + x)
    h_η_η = - x2y/2*(1 + 2*x)
    
    hess_tot_t = (N^2-N).*[h_θ_θ h_θ_η; h_θ_η h_η_η]./(Z^2)

    return hess_tot_t
end


function target_function_t_fisher(model::SdErgmDirBin0Rec0_mle, obs_t, N, f_t)
    # information equality holds and hessian does not depend on observations, hence taking the expectation does not change the result
    return  - target_function_t_hess(model, obs_t, N, f_t) 
end


function static_estimate(model::SdErgmDirBin0Rec0_mle, statsT)
    L_mean  = mean([stat[1] for stat in statsT ])
    R_mean  = mean([stat[2] for stat in statsT ])
    N_mean  = mean([stat[3] for stat in statsT ])
    
    staticPars = StaticNets.estimate(model.staticModel, L_mean, R_mean, N_mean )
    return staticPars
end



SdErgmDirBin0Rec0_pmle(;scoreScalingType="FISH_D", options=SortedDict()) = SdErgmPml(staticModel = NetModeErgmPml(ergm_term_string(ErgmDirBin0Rec0()), true), indTvPar = trues(2), scoreScalingType=scoreScalingType, options=options)
export SdErgmDirBin0Rec0_pmle



import ..StaticNets:ergm_par_from_mean_vals

alpha_beta_to_theta_eta(α, β, N) = collect(ergm_par_from_mean_vals(ErgmDirBin0Rec0(), α*n_pox_dir_links(N), β*n_pox_dir_links(N), N))


theta_eta_to_alpha_beta(θ, η, N) =  collect(exp_val_stats(ErgmDirBin0Rec0(), θ, η, N))./n_pox_dir_links(N)



get_theta_eta_seq_from_alpha_beta(alpha_beta_seq, N) = reduce(hcat, [alpha_beta_to_theta_eta(ab[1], ab[2], N) for ab in eachcol(alpha_beta_seq)])


function beta_min_max_from_alpha_min(minAlpha, N;  minBeta = minAlpha/5 )
    # min beta val such that the minimum exp value rec links is not close zero   
     

    # min beta val such that the maximum exp value rec links is not close to the physical upper bound 
    
    physicalUpperBound = minAlpha/2
    maxBeta =  physicalUpperBound - minBeta
    
    minBeta >= maxBeta ? error((minBeta, maxBeta)) : ()

    return minBeta, maxBeta 
end


function sample_time_var_par_from_dgp(model::SdErgmDirBin0Rec0, dgpType, N, T;  minAlpha = [0.25], maxAlpha = [0.3], nCycles = [2], phaseshift = [0.1], plotFlag=false, phaseAlpha = 0, sigma = [0.01], B = [0.95], A=[0.01], maxAttempts = 5000, indTvPar=trues(number_ergm_par(model)))

    minBeta, maxBeta =  beta_min_max_from_alpha_min(minAlpha[1], N)


    Logging.@debug((;minBeta, maxBeta))
    Logging.@debug((;minAlpha, maxAlpha))
    
    minBetaSin, maxBetaSin = minBeta, maxBeta# .* [1,1.5]
    betaConst = minBeta#(maxBeta - minBeta)/2
    #phaseAlpha = rand()  * 2π
    phaseBeta = phaseAlpha + phaseshift[1] * 2π

    α_β_parDgpT = zeros(2,T)
    
    for n=1:maxAttempts
        Nsteps1= 2
        if dgpType=="SIN"
            α_β_parDgpT[1,:] = dgpSin(minAlpha[1], maxAlpha[1], nCycles[1], T; phase = phaseAlpha)# -3# randSteps(0.05,0.5,2,T) #1.5#.00000000000000001
            α_β_parDgpT[2,:] .= dgpSin(minBetaSin, maxBetaSin, nCycles[1], T;phase= phaseBeta )# -3# randSteps(0.05,0.5,2,T) #1.5#.00000000000000001
        elseif dgpType=="steps"
            α_β_parDgpT[1,:] = randSteps(α_β_minMax[1], α_β_minMax[2], Nsteps1,T)
            α_β_parDgpT[2,:] = randSteps(η_0_minMax[1], η_0_minMax[2], Nsteps1,T)
        elseif dgpType=="AR"
            meanValAlpha = (minAlpha[1]+maxAlpha[1])/2
            meanValBeta = (minBeta+maxBeta)/2

            θ_η_UM = alpha_beta_to_theta_eta(meanValAlpha, meanValBeta, N)

            θ_η_parDgpT = zeros(Real,2,T)

            θ_η_parDgpT[1,:] = dgpAR(θ_η_UM[1],B[1],sigma[1],T )
            θ_η_parDgpT[2,:] = dgpAR(θ_η_UM[2],B[1],sigma[1],T )

            break

        end

        if dgpType == "SD"
            meanValAlpha = (minAlpha[1]+maxAlpha[1])/2
            meanValBeta = (minBeta+maxBeta)/2
    
            UM = alpha_beta_to_theta_eta(meanValAlpha, meanValBeta, N)

            Logging.@warn(" the score driven DGP used is the Maximum Likelihood one. PML is too slow")

            vUnPar, ~ = DynNets.starting_point_optim(model, UM)
            vResParDgp = DynNets.restrict_all_par(model, vUnPar)

            vResParDgp[2:3:end] .= B[1]
            vResParDgp[3:3:end].= A[1]
            vResParDgp = Real.(vResParDgp)

            fVecT, ~, ~ = DynNets.score_driven_filter_or_dgp( model, N, vResParDgp, indTvPar; dgpNT = (N,T))

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




function sample_est_mle_pmle(model::SdErgmDirBin0Rec0, parDgpT, N, Nsample; plotFlag = true, regimeString="")

    model_mle = SdErgmDirBin0Rec0_mle()
    model_pmle = SdErgmDirBin0Rec0_pmle()

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

    A_T_dgp_S = sample_ergm_sequence(model_mle, N, parDgpT, Nsample)

    for n=1:Nsample
        ## sample dgp
        stats_T_dgp = [stats_from_mat(model_mle, A_T_dgp_S[:,:,t, n]) for t in 1:T ]
        change_stats_T_dgp = change_stats(model_pmle, A_T_dgp_S)


        ## estimate SD
        estPar_pmle, conv_flag,UM_mple , ftot_0_mple = estimate(model_pmle, N,  change_stats_T_dgp; indTvPar=indTvPar,indTargPar=indTvPar)
        vResEstPar_pmle = DynNets.array_2_vec_all_par(model_pmle, estPar_pmle, indTvPar)
        fVecT_filt_p , logLikeVecT, sVecT_filt_p = score_driven_filter( model_pmle, N, obsT, vResEstPar_pmle, indTvPar; change_stats_T_dgp, ftot_0 = ftot_0_mple)
        vEstSd_pmle[:,n] = vcat(vResEstPar_pmle, ftot_0_mple)

        estPar_mle, conv_flag,UM_mle , ftot_0_mle = estimate(model_mle, N, stats_T_dgp; indTvPar=indTvPar, indTargPar=indTvPar)
        vResEstPar_mle = DynNets.array_2_vec_all_par(model_mle, estPar_mle, indTvPar)
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

