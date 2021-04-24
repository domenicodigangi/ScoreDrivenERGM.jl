
# using ScikitLearn
# @sk_import covariance: MinCovDet

import ..Utilities
abstract type GasNetModel end

Base.string(x::GasNetModel) = DynNets.name(x) 
export string


identify(model::GasNetModel,UnPar::Array{<:Real,1}, idType ) = StaticNets.identify(model.staticModel, UnPar; idType = idType)


number_ergm_par(model::T where T <:GasNetModel) = model.staticModel.nErgmPar


type_of_obs(model::GasNetModel) =  StaticNets.type_of_obs(model.staticModel)


stats_from_mat(model::GasNetModel, A ::Matrix{<:Real}) = StaticNets.stats_from_mat(model.staticModel, A ::Matrix{<:Real}) 


function seq_of_obs_from_seq_of_mats(model::T where T <:GasNetModel, AT_in)

    AT = convert_to_array_of_mats(AT_in)
    T = length(AT)
    obsT = Array{type_of_obs(model), 1}(undef, T)
    for t in 1:T
        obsT[t] = DynNets.stats_from_mat(model, AT[t]) 
    end
    return obsT 
end


#region options and conversions of parameters for optimization
function setOptionsOptim(model::T where T<: GasNetModel; show_trace = false)
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


function array2VecGasPar(model::GasNetModel, ArrayGasPar, indTvPar :: BitArray{1})
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


function vec2ArrayGasPar(model::GasNetModel, VecGasPar::Array{<:Real,1}, indTvPar :: BitArray{1})
    
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
function divide_SD_par_from_const(model::T where T <:GasNetModel, indTvPar,  vecAllPar::Array{<:Real,1})

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


function merge_SD_par_and_const(model::T where T <:GasNetModel, indTvPar,  vecSDPar::Array{<:Real,1}, vConstPar)

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
function restrict_SD_static_par(model::T where T <:GasNetModel, vecUnSDPar::Array{<:Real,1})

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
function unrestrict_SD_static_par(model::T where T <:GasNetModel, vecReSDPar::Array{<:Real,1})

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
function unrestrict_all_par(model::T where T <:GasNetModel, indTvPar, vAllPar)

    vSDRe, vConst = divide_SD_par_from_const(model, indTvPar, vAllPar)

    vSDUn = unrestrict_SD_static_par(model, vSDRe)

    merge_SD_par_and_const(model, indTvPar, vSDUn, vConst)
end

"""
    separate SD parameters from constant ones, restrict SD and merge them back 
"""
function restrict_all_par(model::T where T <:GasNetModel, indTvPar, vAllPar)
   
    vSDUn, vConst = divide_SD_par_from_const(model, indTvPar, vAllPar)

    vSDRe = restrict_SD_static_par(model, vSDUn)

    merge_SD_par_and_const(model, indTvPar, vSDRe, vConst)
end


"""
Given the flag of constant parameters, a starting value for their unconditional means (their constant value, for those constant), return a starting point for the optimization
"""
function starting_point_optim(model::T where T <:GasNetModel, indTvPar, UM; indTargPar =  falses(100))
    
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


target_function_t(model::GasNetModel, obs_t, N, f_t) = StaticNets.obj_fun(model.staticModel, obs_t, N, f_t)


function target_function_t_grad(model::T where T<: GasNetModel, obs_t, N, f_t)

    target_fun_t(x) = target_function_t(model, obs_t, N, x)
    
    grad_tot_t = ForwardDiff.gradient(target_fun_t, f_t)

    return grad_tot_t
end


function target_function_t_hess(model::T where T<: GasNetModel, obs_t, N, f_t)

    target_fun_t(x) = target_function_t(model, obs_t, N, x)
    
    hess_tot_t = ForwardDiff.hessian(target_fun_t, f_t)

    return hess_tot_t
end


function target_function_t_fisher(model::T where T<: GasNetModel, obs_t, N, f_t)

    error("need to code the fisher manually, cannot be obtained by Automatic Differentiation in the general case, only when the fisher equality holds")
    return fisher_info_t
end


function updatedGasPar( model::T where T<: GasNetModel, obs_t, N, ftot_t::Array{<:Real,1}, I_tm1::Array{<:Real,2}, indTvPar::BitArray{1}, Wgas::Array{<:Real,1}, Bgas::Array{<:Real,1}, Agas::Array{<:Real,1})
    
    
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

"""
Barrier function to handle single N and sequence of Ns
"""
get_N_t(N_seq::AbstractArray, t::Int) = N_seq[t] 
get_N_t(N::Int, t) = N 

function score_driven_filter( model::T where T<: GasNetModel, N, obsT, vResGasPar::Array{<:Real,1}, indTvPar::BitArray{1}; vConstPar ::Array{<:Real,1} = zeros(Real,2), ftot_0::Array{<:Real,1} = zeros(Real,2))

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
    fVecT[.!indTvPar,1] = vConstPar

    ftot_tp1 = zeros(nErgmPar)
    for t=1:T

        N_t = get_N_t(N, t)

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


function score_driven_dgp( model::T where T<: GasNetModel, N, dgpNT, vResGasPar::Array{<:Real,1}, indTvPar::BitArray{1}; vConstPar ::Array{<:Real,1} = zeros(Real,2), ftot_0::Array{<:Real,1} = zeros(Real,2))

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
  
        diadProb = StaticNets.diadProbFromPars(StaticNets.NetModelDirBin0Rec0(), fVecT[:, t] )

        A_t = StaticNets.samplSingMatCan(StaticNets.NetModelDirBin0Rec0(), diadProb, N)
        
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


estimate_single_snap_sequence(model::T where T<: GasNetModel, obsT, aggregate=0) = StaticNets.estimate_sequence(model.staticModel, obsT)


"""
Estimate the GAS and static parameters
"""
function estimate(model::T where T<: GasNetModel, N, obsT; indTvPar::BitArray{1}=model.indTvPar, indTargPar::BitArray{1} = falses(length(model.indTvPar)), UM:: Array{<:Real,1} = zeros(2), ftot_0 :: Array{<:Real,1} = zeros(2), vParOptim_0 =zeros(2), shuffleObsInds = zeros(Int, 2), show_trace = false )
    @debug "[estimate][start][indTvPar=$indTvPar, indTargPar=$indTargPar, UM=$UM, ftot_0=$ftot_0, vParOptim_0 = $vParOptim_0, shuffleObsInds=$shuffleObsInds]"
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
    shuffleObsFlag = !(sum(shuffleObsInds)  .== 0)
    if shuffleObsFlag
        length(shuffleObsInds) == T ? () : error("wrong lenght of shuffling indices")
    end

    if shuffleObsFlag 

        function objfunGasShuffle(vecUnPar::Array{<:Real,1})# a function of the groups parameters

            vecReSDPar,vecConstPar = divideCompleteRestrictPar(vecUnPar)

            oneInADterms  = (StaticNets.maxLargeVal + vecUnPar[1])/StaticNets.maxLargeVal

            foo, logLikeVecT, foo1 = score_driven_filter( model, N, obsT, vecReSDPar, indTvPar;  vConstPar =  vecConstPar, ftot_0 = ftot_0 .* oneInADterms)

                return - sum(logLikeVecT[shuffleObsInds])
        end
   
        ADobjfunGas = TwiceDifferentiable(objfunGasShuffle, vParOptim_0; autodiff = :forward);
    else
        function objfunGas(vecUnPar::Array{<:Real,1})# a function of the groups parameters

            # vecReSDPar,vecConstPar = divideCompleteRestrictPar(vecUnPar)

            vecUnSDPar, vecConstPar = divide_SD_par_from_const(model, indTvPar, vecUnPar)

            oneInADterms  = (StaticNets.maxLargeVal + vecUnPar[1])/StaticNets.maxLargeVal

            vecReSDPar = restrict_SD_static_par(model, vecUnSDPar)

            foo, logLikeVecT, foo1 = score_driven_filter( model, N, obsT, vecReSDPar, indTvPar;  vConstPar =  vecConstPar, ftot_0 = ftot_0 .* oneInADterms)

            return - sum(logLikeVecT)
        end

        ADobjfunGas = TwiceDifferentiable(objfunGas, vParOptim_0; autodiff = :forward);


        if haskey(model.options, "Firth")
            if model.options["Firth"]

                @show FiniteDiff.finite_difference_hessian(objfunGas, vParOptim_0)
                function objfunGasFirth(vecUnPar::Array{<:Real,1})
                    objfunGas(vecUnPar) -  0.5 * LinearAlgebra.logabsdet( FiniteDiff.finite_difference_hessian(objfunGas, vecUnPar))[1]
                end

                @show objfunGasFirth(vParOptim_0)
        
                ADobjfunGas = TwiceDifferentiable(objfunGasFirth, vParOptim_0; autodiff = :forward);
            end
        
        end
    end
    



    #Run the optimization
   
    Logging.@debug("[estimate][Starting point for Optim $vParOptim_0]")
    Logging.@debug("[estimate][Starting point for Optim $(restrict_all_par(model, indTvPar, vParOptim_0))]")
    optim_out2  = optimize(ADobjfunGas, vParOptim_0, algo, optims_opt)
    outParAllUn = Optim.minimizer(optim_out2)
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


#region Uncertainties filtered parameters
# using FiniteDiff
# Base.eps(Real) = eps(Float64)

function A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model::GasNetModel, N, obsT, vEstSdResParAll, indTvPar, ftot_0)

    T = length(obsT)
    nPar = length(vEstSdResParAll)
    gradT = zeros(nPar, T)
    hessT = zeros(nPar, nPar, T)

    vecUnParAll = unrestrict_all_par(model, indTvPar, vEstSdResParAll)    

    function obj_fun_t(xUn, t)

            xRe = restrict_all_par(model, indTvPar, xUn)

            vecSDParRe, vConstPar = divide_SD_par_from_const(model, indTvPar, xRe)

            oneInADterms  = (StaticNets.maxLargeVal + vecSDParRe[1])/StaticNets.maxLargeVal

            fVecT_filt, logLikeVecT, ~ = DynNets.score_driven_filter( model, N, obsT[1:t], vecSDParRe, indTvPar; vConstPar =  vConstPar, ftot_0 = ftot_0 .* oneInADterms)
        
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


function white_estimate_cov_mat_static_sd_par(model::GasNetModel,  N,obsT, indTvPar, ftot_0, vEstSdResParAll; returnAllMats=false, enforcePosDef = true)

    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    errorFlag = false

    
    OPGradSum, HessSum = A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model, N, obsT, vEstSdResParAll, indTvPar, ftot_0)

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


function get_B_A_mats_for_TV(model::GasNetModel, indTvPar, vEstSdResPar)
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



function distrib_filtered_par_from_sample_static_par(model::GasNetModel, N, obsT, indTvPar, ftot_0, sampleUnParAll)
        
    T = length(obsT)
    nErgmPar = number_ergm_par(model)
    nTvPar = sum(indTvPar)
    nSample = size(sampleUnParAll)[2]

    # any(.!indTvPar) ? error("assuming all parameters TV. otherwise need to set filter unc to zero for static ones") : ()    


    distribFilteredSD = zeros(nSample, nErgmPar,T)
    filtCovHatSample = zeros(nTvPar, T, nSample)
    
    for n=1:nSample
        vResPar = restrict_all_par(model, indTvPar, sampleUnParAll[:,n])

        vecSDParRe, vConstPar = divide_SD_par_from_const(model, indTvPar, vResPar)

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


function distrib_static_par_from_mv_normal(model::GasNetModel, N, obsT, indTvPar, ftot_0, vEstSdUnPar, mvSDUnParEstCov; nSample = 500)
        
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
function par_bootstrap_distrib_filtered_par(model::GasNetModel, N, obsT, indTvPar, ftot_0, vEstSdResPar; nSample = 250, logFlag = false)

    T = length(obsT)
    nErgmPar = number_ergm_par(model)
     
     
    nStaticSdPar = length(vEstSdResPar)
    vUnParEstDistrib = SharedArray(zeros(length(vEstSdResPar), nSample))
    distribFilteredSD = SharedArray(zeros(nSample, nErgmPar,T))
    filtCovHatSample = SharedArray(zeros(nErgmPar, nSample))
    errFlagVec = SharedArray{Bool}((nSample))


    Threads.@threads for n=1:nSample
        
        # try
            # sample SD dgp
            fVecTDgp, A_T_dgp, ~ = DynNets.score_driven_dgp( model, N, vEstSdResPar, indTvPar, (N,T))
            obsTNew = [stats_from_mat(model, A_T_dgp[:,:,t]) for t in 1:T ]

            # Estimate SD static params from SD DGP
            arrayAllParHat, conv_flag,UM, ftot_0 = estimate(model, N, obsTNew; indTvPar=indTvPar, indTargPar=falses(2), ftot_0=ftot_0)

            vResPar = DynNets.array2VecGasPar(model, arrayAllParHat, indTvPar)
    
            vUnPar = unrestrict_all_par(model, indTvPar, deepcopy(vResPar))

            vUnParEstDistrib[:,n] =  vUnPar

            vecSDParRe, vConstPar = divide_SD_par_from_const(model, indTvPar, vResPar)

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


function non_par_bootstrap_distrib_filtered_par(model::GasNetModel, N, obsT, indTvPar, ftot_0; nBootStrap = 100)

    vEstSdUnParBootDist = SharedArray(zeros(3*sum(model.indTvPar), nBootStrap))
    T = length(obsT)
    @time Threads.@threads for k=1:nBootStrap
        vEstSdUnParBootDist[:, k] = rand(1:T, T) |> (inds->(inds |> x-> DynNets.estimate(model, N, obsT; indTvPar=indTvPar, ftot_0 = ftot_0, shuffleObsInds = x) |> x-> getindex(x, 1) |> x -> DynNets.array2VecGasPar(model, x, model.indTvPar))) |> x -> DynNets.unrestrict_all_par(model, model.indTvPar, x)
    end

    return vEstSdUnParBootDist
end


function conf_bands_buccheri(model::GasNetModel, N, obsT, indTvPar, fVecT_filt, distribFilteredSD, filtCovHatSample, quantilesVals::Vector{Vector{Float64}}, dropOutliers::Bool; nSample = 250, winsorProp=0.005)
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


function conf_bands_par_uncertainty_blasques(model::GasNetModel, obsT, fVecT_filt, distribFilteredSD, quantilesVals::Vector{Vector{Float64}}; nSample = 500)
    
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


function plot_filtered(model::GasNetModel, N, fVecT_filtIn; lineType = "-", lineColor = "b", parDgpTIn=zeros(2,2), offset = 0, fig = nothing, ax=nothing, gridFlag=true, xval=nothing)

    parDgpT = parDgpTIn[:, 1:end-offset]
    fVecT_filt = fVecT_filtIn[:, 1+offset:end, :, :]
    n_ergm_par= number_ergm_par(model)

    T = size(fVecT_filt)[2]

    if isnothing(fig)
        fig, ax = subplots(n_ergm_par,1)
    end

    if parDgpTIn != zeros(2,2)
        plot_filtered(model, N, parDgpTIn; lineType="-", lineColor="k", fig=fig, ax=ax, gridFlag=false, xval=xval)
    end
    for p in 1:n_ergm_par
        isnothing(xval) ? x = collect(1:T) : x = xval
        bottom = minimum(fVecT_filt[p,:])
        top = maximum(fVecT_filt[p,:])
       
        delta = top-bottom
        margin = 0.5*delta
        ax[p].plot(x, fVecT_filt[p,:], "$(lineType)$(lineColor)", alpha =0.5)
        # ax[p].set_ylim([bottom - margin, top + margin])
        gridFlag ? ax[p].grid() : ()
        
    end
 
    titleString = "$(name(model)), N = $(get_N_t(N, 1)), T=$(T + offset)"

    ax[1].set_title(titleString)

    return fig, ax
end


function plot_filtered_and_conf_bands(model::GasNetModel, N, fVecT_filtIn, confBands1In; lineType = "-", confBands2In =zeros(2,2,2,2), parDgpTIn=zeros(2,2), nameConfBand1="1", nameConfBand2="2", offset = 0, indBand = 1, xval=nothing)

    parDgpT = parDgpTIn[:, 1:end-offset]
    fVecT_filt = fVecT_filtIn[:, 1+offset:end, :, :]
    confBands1 = confBands1In[:, 1+offset:end, :, :]
    confBands2 = confBands2In[:, 1+offset:end, :, :]

    T = size(fVecT_filt)[2]

    nBands = size(confBands1)[3]

    fig, ax = plot_filtered(model::GasNetModel, N, fVecT_filtIn; lineType = lineType, parDgpTIn=parDgpTIn, offset = offset, xval=xval)

    for p in 1:number_ergm_par(model)
        isnothing(xval) ? x = collect(1:T) : x = xval
           for b = indBand
            if sum(confBands1) !=0
                ax[p].fill_between(x, confBands1[p, :, b,1], y2 =confBands1[p,:,b, 2],color =(0.9, 0.2 , 0.2, 0.1), alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
            end
            if confBands2 != zeros(2,2)
                ax[p].plot(x, confBands2[p, :, b, 1], "-g", alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
                ax[p].plot(x, confBands2[p,:, b, 2], "-g", alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
            end
        end
   
        
    end
    
    titleString = "$(name(model)), N = $N, T=$(T + offset), \n "
    titleString = "$(name(model)), \n "

    if parDgpTIn != zeros(2,2)
        if sum(confBands1) !=0
                cov1 = round(mean(conf_bands_coverage(parDgpTIn, confBands1In; offset=offset)[:,:,indBand]), digits=2)
                titleString = titleString * "$nameConfBand1 = $cov1"
        else
            cov1 = 0
        end

        if confBands2 != zeros(2,2)
            cov2 = round(mean(conf_bands_coverage(parDgpTIn, confBands2In; offset=offset)[:,:,indBand]), digits=2)
            titleString = titleString * " $nameConfBand2 = $cov2"
        end
    end

    ax[1].set_title(titleString)

    return fig, ax
end


function estimate_and_filter(model::GasNetModel, N, obsT; indTvPar = model.indTvPar, show_trace = false)

    T = length(obsT)   
    
    estSdResPar, conv_flag, UM_mple, ftot_0 = estimate(model, N, obsT; indTvPar=indTvPar, indTargPar=falses(length(indTvPar)), show_trace = show_trace)


    vEstSdResParAll = array2VecGasPar(model, estSdResPar, indTvPar)

    vEstSdResPar, vConstPar = divide_SD_par_from_const(model, indTvPar,vEstSdResParAll)

    fVecT_filt , target_fun_val_T, sVecT_filt = score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;ftot_0 = ftot_0, vConstPar=vConstPar)

    return (; obsT, vEstSdResParAll, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0)
end
    


function conf_bands_given_SD_estimates(model::GasNetModel, N, obsT, vEstSdUnParAll, ftot_0, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE", dropOutliers = false, offset = 1,  nSample = 500 , mvSDUnParEstCov = Symmetric(zeros(3,3)), sampleStaticUnPar = zeros(3,3), winsorProp=0, xval=nothing)
    
    T = length(obsT)
    nStaticPar = length(indTvPar) + 2*sum(indTvPar)
    nErgmPar = number_ergm_par(model)

    vEstSdResParAll = restrict_all_par(model, indTvPar, vEstSdUnParAll)
    vEstSdResPar, vConstPar = divide_SD_par_from_const(model, indTvPar,vEstSdResParAll)

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
            OPGradSum, HessSum = A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model, N, obsT, vEstSdResParAll, indTvPar, ftot_0)

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
                vEstSdUnPar = unrestrict_all_par(model, indTvPar, vEstSdResParAll)
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


function estimate_filter_and_conf_bands(model::GasNetModel, A_T, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE",show_trace = false)
    
    N = size(A_T)[1]

    obsT = seq_of_obs_from_seq_of_mats(model, A_T)



    obsT, vEstSdResParAll, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0 = estimate_and_filter(model, N, obsT; indTvPar = indTvPar, show_trace = show_trace )
    Logging.@debug("[estimate_filter_and_conf_bands][vEstSdResParAll = $vEstSdResParAll] ")

    vEstSdUnPar = unrestrict_all_par(model, indTvPar, vEstSdResParAll)

    fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = conf_bands_given_SD_estimates(model, N, obsT, vEstSdUnPar, ftot_0, quantilesVals; indTvPar = indTvPar, parDgpT=parDgpT, plotFlag=plotFlag, parUncMethod = parUncMethod)

    return obsT, vEstSdResParAll, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag

end



function simulate_and_estimate_parallel(model::GasNetModel, dgpSettings, T, N, nSample , singleSnap = false)

    counter = SharedArray(ones(1))
    res = @sync @distributed vcat for k=1:nSample
        
        Logging.@info("Estimating N = $N , T=$T iter n $(counter[1]), $(DynNets.name(model)), $(dgpSettings)")

        parDgpT = DynNets.sample_time_var_par_from_dgp(reference_model(model), dgpSettings.type, N, T;  dgpSettings.opt...)

        A_T_dgp = DynNets.sample_mats_sequence(reference_model(model), parDgpT,N)

        obsT = seq_of_obs_from_seq_of_mats(model, A_T_dgp)

        ~, vEstSdResPar, fVecT_filt, ~, ~, conv_flag, ftot_0 = estimate_and_filter(model, N, obsT; indTvPar = model.indTvPar)
        
        if singleSnap
            fVecT_filt_SS = zeros(DynNets.number_ergm_par(model), T)
        else
            fVecT_filt_SS =  DynNets.estimate_single_snap_sequence(model, A_T_dgp)
        end
        counter[1] += 1

        (;obsT, parDgpT, vEstSdResPar, fVecT_filt, conv_flag, ftot_0, fVecT_filt_SS)
 
    end


    allObsT = [r.obsT for r in res]
    allParDgpT = reduce(((a,b) -> cat(a,b, dims=3)), [r.parDgpT for r in res])
    allvEstSdResPar = reduce(((a,b) -> cat(a,b, dims=2)), [r.vEstSdResPar for r in res])
    allfVecT_filt = reduce(((a,b) -> cat(a,b, dims=3)), [r.fVecT_filt for r in res])
    allfVecT_filt_SS = reduce(((a,b) -> cat(a,b, dims=3)), [r.fVecT_filt_SS for r in res])
    allConvFlag = [r.conv_flag for r in res]
    allftot_0 = reduce(((a,b) -> cat(a,b, dims=2)), [r.ftot_0 for r in res])

    Logging.@info("The fraction of estimates that resulted in errors is $(mean(.!allConvFlag)) ")

    return allObsT, allvEstSdResPar, allfVecT_filt, allParDgpT, allConvFlag, allftot_0, allfVecT_filt_SS
end
