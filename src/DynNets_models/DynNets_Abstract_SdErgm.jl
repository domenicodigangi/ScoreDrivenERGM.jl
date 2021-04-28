
# using ScikitLearn
# @sk_import covariance: MinCovDet

import ..Utilities
abstract type SdErgm end

Base.string(x::SdErgm) = DynNets.name(x) 
export string


identify(model::SdErgm,UnPar::Array{<:Real,1}, idType ) = StaticNets.identify(model.staticModel, UnPar; idType = idType)


number_ergm_par(model::T where T <:SdErgm) = model.staticModel.nErgmPar


type_of_obs(model::SdErgm) =  StaticNets.type_of_obs(model.staticModel)


stats_from_mat(model::SdErgm, A ::Matrix{<:Real}) = StaticNets.stats_from_mat(model.staticModel, A ::Matrix{<:Real}) 

"""
Model to use for utilities and eventually fast sampling. The main use is to point pmle models to their mle counterpart, which can be used for sampling or for comparison purposes
"""
reference_model(model::SdErgm) = model


function seq_of_obs_from_seq_of_mats(model::T where T <:SdErgm, AT_in)

    AT = convert_to_array_of_mats(AT_in)
    T = length(AT)
    obsT = Array{type_of_obs(model), 1}(undef, T)
    for t in 1:T
        obsT[t] = DynNets.stats_from_mat(model, AT[t]) 
    end
    return obsT 
end


#region options and conversions of parameters for optimization
function setOptionsOptim(model::T where T<: SdErgm; show_trace = false)
    "Set the options for the optimization required in the estimation of the model.
    For the optimization use the Optim package."
    tol = eps()*10
    maxIter = 150
    opt = Optim.Options(  g_tol = 1e-8,
                     x_tol = tol,
                     x_abstol = tol,
                     x_reltol = tol,
                     f_tol = tol,
                     f_reltol = tol,
                     f_abstol = tol,
                     iterations = maxIter,
                     show_trace = show_trace ,#false,#
                     store_trace = true ,#false,#
                     show_every=5)

    algo = NewtonTrustRegion(; initial_delta = 0.1,
                    delta_hat = 0.2,
                    eta = 0.1,
                    rho_lower = 0.25,
                    rho_upper = 0.75)
    algo = Newton(; alphaguess = LineSearches.InitialHagerZhang(),
    linesearch = LineSearches.BackTracking())
      return opt, algo
end


function array2VecGasPar(model::SdErgm, ArrayGasPar, indTvPar :: BitArray{1})
    # optimizations routines need the parameters to be optimized to be in a vector

        NTvPar = sum(indTvPar)
        Npar = length(indTvPar)
        NgasPar = 2*NTvPar + Npar

        VecGasPar = zeros(NgasPar)
        last = 1
        for i=1:Npar
            VecGasPar[last:last+length(ArrayGasPar[i])-1] = ArrayGasPar[i]
            last = last + length(ArrayGasPar[i])
        end
        return VecGasPar
end


function vec2ArrayGasPar(model::SdErgm, VecGasPar::Array{<:Real,1}, indTvPar :: BitArray{1})
    
    Npar = length(indTvPar)
    ArrayGasPar = [zeros(Real, 1 + tv*2) for tv in indTvPar]#Array{Array{<:Real,1},1}(undef, Npar)
    last_i = 1

    for i=1:Npar
        nStatPerDyn = 1 + 2*indTvPar[i]
        ArrayGasPar[i][:] = VecGasPar[last_i: last_i+nStatPerDyn-1]
            last_i = last_i + nStatPerDyn

    end
    return ArrayGasPar
end


"""
Given vecAllPar divide it into a vector of Score Driven parameters and one of costant parameters
"""
function divide_SD_par_from_const(model::T where T <:SdErgm, indTvPar,  vecAllPar::Array{<:Real,1})

    nTvPar = sum(indTvPar)
    nErgmPar = length(indTvPar)

    vecSDParAll = zeros(Real,3nTvPar )
    vConstPar = zeros(Real,nErgmPar-nTvPar)

    lastInputInd = 0
    lastIndSD = 0
    lastConstInd = 0
    #extract the vector of gas parameters, addimng w from targeting when needed
    for i=1:nErgmPar
        if indTvPar[i]
            vecSDParAll[lastIndSD+1] = vecAllPar[lastInputInd + 1]
            vecSDParAll[lastIndSD+2] = vecAllPar[lastInputInd + 2]
            vecSDParAll[lastIndSD+3] = vecAllPar[lastInputInd + 3]
            lastInputInd +=3
            lastIndSD +=3
        else
            vConstPar[lastConstInd+1] = vecAllPar[lastInputInd  + 1]
            lastInputInd +=1
            lastConstInd +=1
        end
    end
    return vecSDParAll, vConstPar
end


function merge_SD_par_and_const(model::T where T <:SdErgm, indTvPar,  vecSDPar::Array{<:Real,1}, vConstPar)

    nTvPar = sum(indTvPar)
    nConstPar = sum(.!indTvPar)
    nConstPar == length(vConstPar) ? () : error()

    nErgmPar = length(indTvPar)
    nAllPar = 3*nTvPar + nConstPar

    vecAllPar = zeros(Real, nAllPar)

    lastInputInd = 0
    lastConstInd = 0
    lastIndAll = 0
    lastIndSD = 0
    for i=1:nErgmPar
        if indTvPar[i] 
            
            vecAllPar[lastIndAll+1] = vecSDPar[lastIndSD + 1]
            vecAllPar[lastIndAll+2] = vecSDPar[lastIndSD + 2]
            vecAllPar[lastIndAll+3] = vecSDPar[lastIndSD + 3]

            lastIndAll +=3
            lastIndSD +=3
        else
            vecAllPar[lastIndAll+1] = vConstPar[lastConstInd + 1]
                        
            lastIndAll +=1
            lastInputInd +=1
            lastConstInd +=1
        end
    end
    return vecAllPar
end


"""
Restrict the  Score Driven parameters  to appropriate link functions to ensure that they remain in the region where the SD dynamics is well specified (basically 0<=B<1  A>=0)
"""
function restrict_SD_static_par(model::T where T <:SdErgm, vecUnSDPar::Array{<:Real,1})

    nSDPar = length(vecUnSDPar)
    nTvPar, rem = divrem(nSDPar,3)

    rem == 0 ? () : error()

    arrayOfVecsReSd = [ [vecUnSDPar[i], link_R_in_0_1(vecUnSDPar[i+1]), link_R_in_R_pos(vecUnSDPar[i+2]) ] for i in 1:3:nSDPar]

    vecReSDPar = reduce(vcat, arrayOfVecsReSd)

    return vecReSDPar
end


"""
    inverse of restrict_SD_static_par
"""
function unrestrict_SD_static_par(model::T where T <:SdErgm, vecReSDPar::Array{<:Real,1})

    nSDPar = length(vecReSDPar)
    nTvPar, rem = divrem(nSDPar,3)

    rem == 0 ? () : error()

    arrayOfVecsUnSd = [ [vecReSDPar[i], inv_link_R_in_0_1(vecReSDPar[i+1]), inv_link_R_in_R_pos(vecReSDPar[i+2]) ] for i in 1:3:nSDPar]

    vecUnSDPar = reduce(vcat, arrayOfVecsUnSd)

    return vecUnSDPar
end


"""
    separate SD parameters from constant ones, unrestrict SD and merge them back 
"""
function unrestrict_all_par(model::T where T <:SdErgm, indTvPar, vAllPar)

    vSDRe, vConst = divide_SD_par_from_const(model, indTvPar, vAllPar)

    vSDUn = unrestrict_SD_static_par(model, vSDRe)

    merge_SD_par_and_const(model, indTvPar, vSDUn, vConst)
end

"""
    separate SD parameters from constant ones, restrict SD and merge them back 
"""
function restrict_all_par(model::T where T <:SdErgm, indTvPar, vAllPar)
   
    vSDUn, vConst = divide_SD_par_from_const(model, indTvPar, vAllPar)

    vSDRe = restrict_SD_static_par(model, vSDUn)

    merge_SD_par_and_const(model, indTvPar, vSDRe, vConst)
end


"""
Given the flag of constant parameters, a starting value for their unconditional means (their constant value, for those constant), return a starting point for the optimization
"""
function starting_point_optim(model::T where T <:SdErgm, indTvPar, UM; indTargPar =  falses(100))
    
    nTvPar = sum(indTvPar)
    NTargPar = sum(indTargPar)
    nErgmPar = length(indTvPar)
    
    # #set the starting points for the optimizations
    B0_Re  = 0.98; B0_Un = log(B0_Re ./ (1 .- B0_Re ))
    ARe_min =0.0000000001
    A0_Re  = 0.01 ; A0_Un = log(A0_Re  .-  ARe_min)
    
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
    @debug "[starting_point_optim][UM = $UM, vParOptim_0 = $vParOptim_0, ARe_min = $ARe_min]"
    return vParOptim_0, ARe_min
end


convert_to_array_of_mats(AT::Array{Array{<:Any, 2}, 1}) = AT
convert_to_array_of_mats(AT::Array{<:Any, 1}) = [m for m in AT]
convert_to_array_of_mats(AT::Array{<:Any, 3}) = [AT[:,:,t] for t in 1:size(AT)[3]]

#endregion


target_function_t(model::SdErgm, obs_t, N, f_t) = StaticNets.obj_fun(model.staticModel, obs_t, N, f_t)


function target_function_t_grad(model::T where T<: SdErgm, obs_t, N, f_t)

    target_fun_t(x) = target_function_t(model, obs_t, N, x)
    
    grad_tot_t = ForwardDiff.gradient(target_fun_t, f_t)

    return grad_tot_t
end


function target_function_t_hess(model::T where T<: SdErgm, obs_t, N, f_t)

    target_fun_t(x) = target_function_t(model, obs_t, N, x)
    
    hess_tot_t = ForwardDiff.hessian(target_fun_t, f_t)

    return hess_tot_t
end


function target_function_t_fisher(model::T where T<: SdErgm, obs_t, N, f_t)

    error("need to code the fisher manually, cannot be obtained by Automatic Differentiation in the general case, only when the fisher equality holds")
    return fisher_info_t
end


function updatedGasPar( model::T where T<: SdErgm, obs_t, N, ftot_t::Array{<:Real,1}, I_tm1::Array{<:Real,2}, indTvPar::BitArray{1}, Wgas::Array{<:Real,1}, Bgas::Array{<:Real,1}, Agas::Array{<:Real,1})
    
    
    #= likelihood and gradients depend on all the parameters (ftot_t), but
    only the time vaying ones (f_t) are to be updated=#
    
    target_fun_val_t = target_function_t(model, obs_t, N, ftot_t)
    
    grad_tot_t = target_function_t_grad(model, obs_t, N, ftot_t)
    
    I_reg = I(size(I_tm1)[1])



    if model.scoreScalingType =="HESS_D"
        hess_tot_t = target_function_t_hess(model, obs_t, N, ftot_t)
        I_updt = -hess_tot_t
        I_t = I_updt
        s_t = (grad_tot_t./I_t[I(size(I_t)[1])])[indTvPar]
    elseif model.scoreScalingType =="MAX_LINKS"
        hess_tot_t = target_function_t_hess(model, obs_t, N, ftot_t)
        I_t = ones(size(grad_tot_t)).*(N^2-N)
        s_t = grad_tot_t[indTvPar]./I_t
    elseif model.scoreScalingType =="HESS"
        hess_tot_t = target_function_t_hess(model, obs_t, N, ftot_t)
        I_updt = -hess_tot_t
        I_t = I_updt
        s_t = (I_t\grad_tot_t)[indTvPar]

    elseif model.scoreScalingType =="FISH_D"
        extremeScaledScore = 200
        fish_tot_t = target_function_t_fisher(model, obs_t, N, ftot_t)
        I_updt = fish_tot_t
        I_t = I_updt  
        s_t = clamp.( (grad_tot_t./sqrt.((I_t[I(size(I_t)[1])] )))[indTvPar], -extremeScaledScore, extremeScaledScore) 

    end


    f_t = ftot_t[indTvPar] #Time varying ergm parameters
    f_tp1 = Wgas .+ Bgas.* f_t .+ Agas.*s_t

    ftot_tp1 = copy(ftot_t)
    ftot_tp1[indTvPar] = f_tp1 #of all parameters udate dynamic ones with GAS
    
    #any(isnan.(ftot_tp1)) ? error() : ()

    # I_t is the inverse of the S_t matrix in typical notations

    return ftot_tp1, target_fun_val_t, I_t, grad_tot_t
end



function score_driven_filter( model::T where T<: SdErgm, N, obsT, vResGasPar::Array{<:Real,1}, indTvPar::BitArray{1}; vConstPar ::Array{<:Real,1} = zeros(Real,2), ftot_0::Array{<:Real,1} = zeros(Real,2))

    nErgmPar = number_ergm_par(model)
    NTvPar   = sum(indTvPar)
    T= length(obsT)

    # Organize parameters of the GAS update equation
    Wvec = vResGasPar[1:3:3*NTvPar]
    Bvec = vResGasPar[2:3:3*NTvPar]
    Avec = vResGasPar[3:3:3*NTvPar]

    #compute unconditional means according to SD dgp
    UMallPar = zeros(Real,nErgmPar)
    UMallPar[indTvPar] =  Wvec ./ (1 .- Bvec)
    if !all(indTvPar) # if not all parameters are time varying
        UMallPar[.!indTvPar] = vConstPar
    end

    sum(ftot_0)==0 ? ftot_0 = UMallPar : ()# identify(model,UMallNodesIO)

    fVecT = ones(Real,nErgmPar,T)
    sVecT = ones(Real,nErgmPar,T)
    invScalMatT = ones(Real,nErgmPar, nErgmPar,T)
    logLikeVecT = ones(Real,T)
   
    if NTvPar==0
        I_tm1 = ones(1,1)
    else
        I_tm1 = Float64.(Diagonal{Real}(I,NTvPar))
    end


    fVecT[:,1] = ftot_0
    if sum(.!indTvPar) >0
        fVecT[.!indTvPar,1] = vConstPar
    end

    ftot_tp1 = zeros(nErgmPar)
    for t=1:T

        N_t = Utilities.get_N_t(N, t)

        obs_t = obsT[t]

        #obj fun at time t is objFun(obs_t, f_t)

        # predictive step
        ftot_tp1, logLikeVecT[t], invScalMatT[:,:,t], sVecT[:,t] = updatedGasPar(model, obs_t, N_t, fVecT[:, t], I_tm1, indTvPar, Wvec, Bvec, Avec)

        I_tm1 = invScalMatT[:,:,t]

        # would the following be the update step instead ??
        #fVecT[:,t], loglike_t, I_tm1, grad_t = updatedGasPar(model, obs_t, N, fVecT[:,t-1], I_tm1, indTvPar, Wvec, Bvec, Avec)

        if t!=T
            fVecT[:,t+1] = ftot_tp1
        end
    end

    return fVecT::Array{<:Real, 2}, logLikeVecT::Vector{<:Real}, sVecT::Array{<:Real,2}, invScalMatT::Array{<:Real, 3}
end


function score_driven_dgp( model::T where T<: SdErgm, N, dgpNT, vResGasPar::Array{<:Real,1}, indTvPar::BitArray{1}; vConstPar ::Array{<:Real,1} = zeros(Real,2), ftot_0::Array{<:Real,1} = zeros(Real,2))

    nErgmPar = number_ergm_par(model)
    NTvPar   = sum(indTvPar)

    T = dgpNT[2]
    N_const = dgpNT[1]
    N_T = ones(Int, T) .* N_const 
    N = N_const

    # Organize parameters of the GAS update equation
    Wvec = vResGasPar[1:3:3*NTvPar]
    Bvec = vResGasPar[2:3:3*NTvPar]
    Avec = vResGasPar[3:3:3*NTvPar]

    # start values equal the unconditional mean,and  constant ones remain equal to the unconditional mean, hence initialize as:
    UMallPar = zeros(Real,nErgmPar)
    UMallPar[indTvPar] =  Wvec ./ (1 .- Bvec)
    if !all(indTvPar) # if not all parameters are time varying
        UMallPar[.!indTvPar] = vConstPar
    end

    fVecT = ones(Real,nErgmPar,T)
    sVecT = ones(Real,nErgmPar,T)
    invScalMatT = ones(Real,nErgmPar, nErgmPar,T)
    logLikeVecT = ones(Real,T)

    sum(ftot_0)==0 ? ftot_0 = UMallPar : ()# identify(model,UMallNodesIO)
    
    
    if NTvPar==0
        I_tm1 = ones(1,1)
    else
        I_tm1 = Float64.(Diagonal{Real}(I,NTvPar))
    end

    loglike = 0

    A_T = zeros(Int8, N, N, T)
  
    fVecT[:,1] = ftot_0

    ftot_tp1 = zeros(nErgmPar)
    for t=1:T
    #    println(t)
  
        diadProb = StaticNets.diadProbFromPars(StaticNets.ErgmDirBin0Rec0(), fVecT[:, t] )

        A_t = StaticNets.samplSingMatCan(StaticNets.ErgmDirBin0Rec0(), diadProb, N)
        
        A_T[:, : , t] = A_t
        
        obs_t = stats_from_mat(model, A_t)
  
        #obj fun at time t is objFun(obs_t, f_t)

        # predictive step
        ftot_tp1, logLikeVecT[t], invScalMatT[:,:,t], sVecT[:,t] = updatedGasPar(model, obs_t, N, fVecT[:, t], I_tm1, indTvPar, Wvec, Bvec, Avec)

        I_tm1 = invScalMatT[:,:,t]

        # would the following be the update step instead ??
        #fVecT[:,t], loglike_t, I_tm1, grad_t = updatedGasPar(model, obs_t,  N, fVecT[:,t-1], I_tm1, indTvPar, Wvec, Bvec, Avec)

        if t!=T
            fVecT[:,t+1] = ftot_tp1
        end
    end

    return fVecT::Array{Float64, 2}, A_T, sVecT, invScalMatT

end


estimate_single_snap_sequence(model::T where T<: SdErgm, obsT, aggregate=0) = StaticNets.estimate_sequence(model.staticModel, obsT)


"""
Estimate the GAS and static parameters
"""
function estimate(model::T where T<: SdErgm, N, obsT; indTvPar::BitArray{1}=model.indTvPar, indTargPar::BitArray{1} = falses(length(model.indTvPar)), UM:: Array{<:Real,1} = zeros(2), ftot_0 :: Array{<:Real,1} = zeros(2), vParOptim_0 =zeros(2), shuffleObsInds = zeros(Int, 2), show_trace = false )
    @debug "[estimate][start][indTvPar=$indTvPar, indTargPar=$indTargPar, UM=$UM, ftot_0=$ftot_0, vParOptim_0 = $vParOptim_0, shuffleObsInds=$shuffleObsInds, model.options = $(model.options)]"

    T = length(obsT);
    nErgmPar = number_ergm_par(model)
    NTvPar = sum(indTvPar)
    NTargPar = sum(indTargPar)
    Logging.@debug( "[estimate][Estimating N = $N , T=$T]")

  

    # UM is a vector with target values for dynamical parameters
    # if not given as input use the static estimates
    if !all( UM.== 0 )
        if any(indTargPar)
            error("targeting is not considered in the definition of the objective function. Before using it we need to update the latter")
        end
        staticPars = UM 
    else
        staticPars = static_estimate(model, obsT)
    end

    # ftot_0 is a vector with initial values (to be used in the SD iteration)
    # if not given as input estimate on first observations
    if prod(ftot_0.== 0 )&(!prod(.!indTvPar))
        ftot_0 =  static_estimate(model, obsT[1:5])
    end

    optims_opt, algo = setOptionsOptim(model; show_trace = show_trace )

    vParOptim_0_tmp, ARe_min = starting_point_optim(model, indTvPar, staticPars; indTargPar = indTargPar)

    if sum(vParOptim_0) == 0
        vParOptim_0 = vParOptim_0_tmp
    end

  

    #define the objective function for the optimization
  
    if !(sum(shuffleObsInds)  .== 0)
       # was observations shuffing required ?  (used for explorative analysis on non-parametric bootstrap)
        length(shuffleObsInds) == T ? () : error("wrong lenght of shuffling indices")       
        #shuffled obsrvations loglikelihood
        function objfunSdOptShuffled(vecUnPar::Array{<:Real,1})

            vecReSDPar,vecConstPar = divideCompleteRestrictPar(vecUnPar)

            oneInADterms  = (StaticNets.maxLargeVal + vecUnPar[1])/StaticNets.maxLargeVal

            foo, logLikeVecT, foo1 = score_driven_filter( model, N, obsT, vecReSDPar, indTvPar;  vConstPar =  vecConstPar, ftot_0 = ftot_0 .* oneInADterms)

                return - sum(logLikeVecT[shuffleObsInds])
        end
   
        ADobjfunSdOpt = TwiceDifferentiable(objfunSdOptShuffled, vParOptim_0; autodiff = :forward);
    else
        # standard log likelihood
        function objfunSdOpt(vecUnPar::Array{<:Real,1})

            vecUnSDPar, vecConstPar = divide_SD_par_from_const(model, indTvPar, vecUnPar)

            oneInADterms  = (StaticNets.maxLargeVal + vecUnPar[1])/StaticNets.maxLargeVal

            vecReSDPar = restrict_SD_static_par(model, vecUnSDPar)

            foo, logLikeVecT, foo1 = score_driven_filter( model, N, obsT, vecReSDPar, indTvPar;  vConstPar =  vecConstPar, ftot_0 = ftot_0 .* oneInADterms)

            return - sum(logLikeVecT)
        end
        ADobjfunSdOpt = TwiceDifferentiable(objfunSdOpt, vParOptim_0; autodiff = :forward);
        

        if haskey(model.options, "Firth")
            if model.options["Firth"]
                # log likelihood with firth additional term
                function objfunSdOptFirth(vecUnPar::Array{<:Real,1})
                    objfunSdOpt(vecUnPar) -  0.5 * LinearAlgebra.logabsdet( FiniteDiff.finite_difference_hessian(objfunSdOpt, vecUnPar))[1]
                end
                ADobjfunSdOpt = TwiceDifferentiable(objfunSdOptFirth, vParOptim_0; autodiff = :forward);
            end        
        end
        
    end

    integratedFlag = false
    if haskey(model.options, "integrated")
        # should we estimate  an integrated version of the SD filter?
        if model.options["integrated"]
            integratedFlag = true
            vParOptim_0 = ones(Real, NTvPar) * 0.1#vParOptim_0[3:3:end]
            function objfunSdOptInt(vecUnParIntegrated::Array{<:Real,1})  
                vecUnPar = zeros(Real, 3*NTvPar)
                vecUnPar[3:3:end] .= vecUnParIntegrated
                objfunSdOpt(vecUnPar)
            end

            ADobjfunSdOpt = TwiceDifferentiable(objfunSdOptInt, vParOptim_0; autodiff = :forward);
        end
    end
    
    
    
    #Run the optimization
   
    Logging.@debug("[estimate][Starting point for Optim $vParOptim_0]")
    Logging.@debug("[estimate][Starting point for Optim $(restrict_all_par(model, indTvPar, vParOptim_0))]")
    optim_out2  = optimize(ADobjfunSdOpt, vParOptim_0, algo, optims_opt)
    outParAllUn = Optim.minimizer(optim_out2)
    if integratedFlag
        outParAllUnFull = zeros(3*NTvPar)
        outParAllUnFull[1:3:end] .= 0
        outParAllUnFull[2:3:end] .= Inf
        outParAllUnFull[3:3:end] .= outParAllUn
        outParAllUn = deepcopy(outParAllUnFull)
    end
    vecAllParGasHat, vecAllParConstHat = divide_SD_par_from_const(model, indTvPar, restrict_all_par(model, indTvPar, outParAllUn))

    Logging.@debug(optim_out2)

    Logging.@debug("[estimate][Final paramters SD: $vecAllParGasHat , constant: $vecAllParConstHat ]")


    function reshape_results(vecAllParGasHat)
        arrayAllParHat = fill(Float64[],nErgmPar)
        lastGasInd = 0
        lastConstInd = 0
        for i=1:nErgmPar
            if indTvPar[i]
                arrayAllParHat[i] = vecAllParGasHat[lastGasInd+1:lastGasInd+3]
                lastGasInd += 3
            else
                arrayAllParHat[i] = vecAllParConstHat[lastConstInd+1]*ones(1)
                lastConstInd+=1
            end
        end
        return arrayAllParHat
    end

    arrayAllParHat = reshape_results(vecAllParGasHat)
    conv_flag =  Optim.converged(optim_out2)
   
     @debug "[estimate][end]"
    return  arrayAllParHat, conv_flag,UM , ftot_0
   
end
