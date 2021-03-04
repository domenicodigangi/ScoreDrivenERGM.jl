
abstract type GasNetModel end


identify(model::GasNetModel,UnPar::Array{<:Real,1}, idType ) =
    StaticNets.identify(model.staticModel, UnPar; idType = idType)

number_ergm_par(model::T where T <:GasNetModel) = length(model.indTvPar)

type_of_obs(model::GasNetModel) =  StaticNets.type_of_obs(model.staticModel)

statsFromMat(model::GasNetModel, A ::Matrix{<:Real}) = StaticNets.statsFromMat(model.staticModel, A ::Matrix{<:Real}) 

# options and conversions of parameters for optimization
function setOptionsOptim(model::T where T<: GasNetModel)
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
                     show_trace = false,#false,#
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

    lastIndAll = 0
    lastIndSD = 0
    lastIndConst = 0
    for i=1:nErgmPar
        if indTvPar[i] 
            
            vecAllPar[lastIndAll+1] = vecSDPar[lastIndSD + 1]
            vecAllPar[lastIndAll+2] = vecSDPar[lastIndSD + 2]
            vecAllPar[lastIndAll+3] = vecSDPar[lastIndSD + 3]

            lastIndAll +=3
            lastIndSD +=3
        else
            vecAllPar[lastIndAll+1] = vConstPar[lastIndConst + 1]
                        
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
Restrict the  Score Driven parameters  to appropriate link functions to ensure that they remain in the region where the SD dynamics is well specified (basically 0<=B<1  A>=0)
"""
function unrestrict_SD_static_par(model::T where T <:GasNetModel, vecReSDPar::Array{<:Real,1})

    nSDPar = length(vecReSDPar)
    nTvPar, rem = divrem(nSDPar,3)

    rem == 0 ? () : error()

    arrayOfVecsUnSd = [ [vecReSDPar[i], inv_link_R_in_0_1(vecReSDPar[i+1]), inv_link_R_in_R_pos(vecReSDPar[i+2]) ] for i in 1:3:nSDPar]

    vecUnSDPar = reduce(vcat, arrayOfVecsUnSd)

    return vecUnSDPar
end


function unrestrict_all_par(model::T where T <:GasNetModel, indTvPar, vAllPar)
    vSDRe, vConst = divide_SD_par_from_const(model, indTvPar, vAllPar)

    vSDUn = unrestrict_SD_static_par(model, vSDRe)

    merge_SD_par_and_const(model, indTvPar, vSDUn, vConst)
end


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
    ARe_min =0.00000000001
    A0_Re  = 0.000005 ; A0_Un = log(A0_Re  .-  ARe_min)
    
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


function seq_of_obs_from_seq_of_mats(model::T where T <:GasNetModel, AT)

    T = size(AT)[3]
    obsT = Array{type_of_obs(model), 1}(undef, T)
    for t in 1:T
        obsT[t] = DynNets.statsFromMat(model, AT[:,:,t]) 
    end
    return obsT 
end


target_function_t(model::GasNetModel, obst_t, f_t) = error("Not yet defined")


function target_function_t_grad(model::T where T<: GasNetModel, obs_t, f_t)

    target_fun_t(x) = target_function_t(model, obs_t, x)
    
    grad_tot_t = ForwardDiff.gradient(target_fun_t, f_t)

    return grad_tot_t
end


function target_function_t_hess(model::T where T<: GasNetModel, obs_t, f_t)

    target_fun_t(x) = target_function_t(model, obs_t, x)
    
    hess_tot_t = ForwardDiff.hessian(target_fun_t, f_t)

    return hess_tot_t
end


function updatedGasPar( model::T where T<: GasNetModel, obs_t, ftot_t::Array{<:Real,1}, I_tm1::Array{<:Real,2}, indTvPar::BitArray{1}, Wgas::Array{<:Real,1}, Bgas::Array{<:Real,1}, Agas::Array{<:Real,1})
    
    
    #= likelihood and gradients depend on all the parameters (ftot_t), but
    only the time vaying ones (f_t) are to be updated=#
    
    target_fun_val_t = target_function_t(model, obs_t, ftot_t)
    
    grad_tot_t = target_function_t_grad(model, obs_t, ftot_t)
    
    hess_tot_t = target_function_t_hess(model, obs_t, ftot_t)

    if model.scoreScalingType =="HESS"
        I_updt = -hess_tot_t
    end
    ewmaWeight = 0.95
    I_t = ewmaWeight.*I_updt .+ (1-ewmaWeight).*I(size(I_updt)[1])

    f_t = ftot_t[indTvPar] #Time varying ergm parameters
    s_t = grad_tot_t[indTvPar]./I_t[I(size(I_t)[1])]

    f_tp1 = Wgas .+ Bgas.* f_t .+ Agas.*s_t

    ftot_tp1 = copy(ftot_t)
    ftot_tp1[indTvPar] = f_tp1 #of all parameters udate dynamic ones with GAS
    
    #any(isnan.(ftot_tp1)) ? error() : ()

    return ftot_tp1, target_fun_val_t, I_t, grad_tot_t
end


function score_driven_filter_or_dgp( model::T where T<: GasNetModel, N, vResGasPar::Array{<:Real,1}, indTvPar::BitArray{1}; vConstPar ::Array{<:Real,1} = zeros(Real,2), obsT=zeros(2,2), ftot_0::Array{<:Real,1} = zeros(Real,2), dgpNT = (0,0))

    sum(dgpNT) == 0 ? dgp = false : dgp = true

    nErgmPar = 2#
    NTvPar   = sum(indTvPar)

    if dgp
        T = dgpNT[2]
        N_const = dgpNT[1]
        N_T = ones(Int, T) .* N_const 
        N = N_const
    else
        T= length(obsT);
    end

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
    scalMatT = ones(Real,nErgmPar, nErgmPar,T)
    logLikeVecT = ones(Real,T)

    sum(ftot_0)==0 ? ftot_0 = UMallPar : ()# identify(model,UMallNodesIO)
    
    
    if NTvPar==0
        I_tm1 = ones(1,1)
    else
        I_tm1 = Float64.(Diagonal{Real}(I,NTvPar))
    end

    loglike = 0

    if dgp
        A_T = zeros(Int8, N, N, T)
    end

    fVecT[:,1] = ftot_0

    ftot_tp1 = zeros(nErgmPar)
    for t=1:T
    #    println(t)
        if dgp
            diadProb = StaticNets.diadProbFromPars(StaticNets.NetModelDirBin0Rec0(), fVecT[:, t] )

            A_t = StaticNets.samplSingMatCan(StaticNets.NetModelDirBin0Rec0(), diadProb, N)
            
            A_T[:, : , t] = A_t
            
            obs_t = statsFromMat(model, A_t)

        else   
            obs_t = obsT[t]
        end

        #obj fun at time t is objFun(obs_t, f_t)

        # predictive step
        ftot_tp1, logLikeVecT[t], scalMatT[:,:,t], sVecT[:,t] = updatedGasPar(model, obs_t, fVecT[:, t], I_tm1, indTvPar, Wvec, Bvec, Avec)

        I_tm1 = scalMatT[:,:,t]

        # would the following be the update step instead ??
        #fVecT[:,t], loglike_t, I_tm1, grad_t = updatedGasPar(model,N, obs_t, fVecT[:,t-1], I_tm1, indTvPar, Wvec, Bvec, Avec)

        if t!=T
            fVecT[:,t+1] = ftot_tp1
        end
    end

    #fVecT = hcat(fVecT[:, 2:end], ftot_tp1)

    if dgp
        return fVecT::Array{<:Real, 2}, A_T, sVecT, scalMatT
    else
        return fVecT::Array{<:Real, 2}, logLikeVecT::Vector{<:Real}, sVecT::Array{<:Real,2}, scalMatT::Array{<:Real, 3}
    end
end


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
    scalMatT = ones(Real,nErgmPar, nErgmPar,T)
    logLikeVecT = ones(Real,T)
   
    if NTvPar==0
        I_tm1 = ones(1,1)
    else
        I_tm1 = Float64.(Diagonal{Real}(I,NTvPar))
    end


    fVecT[:,1] = ftot_0

    ftot_tp1 = zeros(nErgmPar)
    for t=1:T

        obs_t = obsT[t]

        #obj fun at time t is objFun(obs_t, f_t)

        # predictive step
        ftot_tp1, logLikeVecT[t], scalMatT[:,:,t], sVecT[:,t] = updatedGasPar(model, obs_t, fVecT[:, t], I_tm1, indTvPar, Wvec, Bvec, Avec)

        I_tm1 = scalMatT[:,:,t]

        # would the following be the update step instead ??
        #fVecT[:,t], loglike_t, I_tm1, grad_t = updatedGasPar(model,N, obs_t, fVecT[:,t-1], I_tm1, indTvPar, Wvec, Bvec, Avec)

        if t!=T
            fVecT[:,t+1] = ftot_tp1
        end
    end

    return fVecT::Array{<:Real, 2}, logLikeVecT::Vector{<:Real}, sVecT::Array{<:Real,2}, scalMatT::Array{<:Real, 3}
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
    scalMatT = ones(Real,nErgmPar, nErgmPar,T)
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
        
        obs_t = statsFromMat(model, A_t)
  
        #obj fun at time t is objFun(obs_t, f_t)

        # predictive step
        ftot_tp1, logLikeVecT[t], scalMatT[:,:,t], sVecT[:,t] = updatedGasPar(model, obs_t, fVecT[:, t], I_tm1, indTvPar, Wvec, Bvec, Avec)

        I_tm1 = scalMatT[:,:,t]

        # would the following be the update step instead ??
        #fVecT[:,t], loglike_t, I_tm1, grad_t = updatedGasPar(model,N, obs_t, fVecT[:,t-1], I_tm1, indTvPar, Wvec, Bvec, Avec)

        if t!=T
            fVecT[:,t+1] = ftot_tp1
        end
    end

    return fVecT::Array{Float64, 2}, A_T, sVecT, scalMatT

end


"""
Estimate the GAS and static parameters
"""
function estimate(model::T where T<: GasNetModel, N, obsT; indTvPar::BitArray{1}=model.indTvPar, indTargPar::BitArray{1} = falses(length(model.indTvPar)), UM:: Array{<:Real,1} = zeros(2), ftot_0 :: Array{<:Real,1} = zeros(2), vParOptim_0 =zeros(2) )

    T = length(obsT);
    nErgmPar = 2 #
    NTvPar = sum(indTvPar)
    NTargPar = sum(indTargPar)
    Logging.@debug( "Estimating N = $N , T=$T")

    # UM is a vector with target values for dynamical ones. Parameters
    # if not given as input use the static estimates
    # single static estimate

    if prod(UM.== 0 )&(!prod(.!indTvPar))&(any(indTargPar))
        staticPars = static_estimate(model, obsT)
        UM = staticPars
    end

    # ftot_0 is a vector with initial values (to be used in the SD iteration)
    # if not given as input estimate on first 3 observations
    if prod(ftot_0.== 0 )&(!prod(.!indTvPar))
        ftot_0 =  static_estimate(model, obsT[1:5])
    end
    #UM = ftot_0

    optims_opt, algo = setOptionsOptim(model)


    vParOptim_0_tmp, ARe_min = starting_point_optim(model, indTvPar, UM; indTargPar = indTargPar)

    if sum(vParOptim_0) == 0
        vParOptim_0 = vParOptim_0_tmp
    end

  
    function divideCompleteRestrictPar(vecUnPar::Array{<:Real,1})

        # vecUnPar is a vector of unrestricted parameters that need to be optimized.
        # add some elements to take into account targeting, divide into GAs and
        # costant parameters, restrict the parameters to appropriate Utilitiesains
        vecReGasParAll = zeros(Real,3NTvPar )
        vecConstPar = zeros(Real,nErgmPar-NTvPar)
        # add w determined by B values to targeted parameters
        lastInputInd = 0
        lastGasInd = 0
        lastConstInd = 0
        #extract the vector of gas parameters, addimng w from targeting when needed
        for i=1:nErgmPar
            if indTvPar[i]
                if indTargPar[i]
                    B =  1 ./ (1 .+ exp.( .- vecUnPar[lastInputInd+1]))
                    vecReGasParAll[lastGasInd+1] = UM[i]*(1 .- B) # w
                    vecReGasParAll[lastGasInd+2] = B #B
                    vecReGasParAll[lastGasInd+3] =  ARe_min   .+  exp(vecUnPar[lastInputInd + 2]) # A
                    lastInputInd +=2
                    lastGasInd +=3
                else
                    vecReGasParAll[lastGasInd+1] = vecUnPar[lastInputInd  + 1]
                    vecReGasParAll[lastGasInd+2] =  1 ./ (1 .+ exp.( .- vecUnPar[lastInputInd + 2]))
                    vecReGasParAll[lastGasInd+3] = ARe_min   .+  exp(vecUnPar[lastInputInd + 3])
                    lastInputInd +=3
                    lastGasInd +=3
                end
            else
                vecConstPar[lastConstInd+1] = vecUnPar[lastInputInd  + 1]
                lastInputInd +=1
                lastConstInd +=1
            end
        end
    return vecReGasParAll,vecConstPar
    end
    # objective function for the optimization
    function objfunGas(vecUnPar::Array{<:Real,1})# a function of the groups parameters
        #vecUnGasPar,vecConstPar =  divideParVec(vecUnPar)

        vecReGasParAll,vecConstPar = divideCompleteRestrictPar(vecUnPar)

        oneInADterms  = (StaticNets.maxLargeVal + vecUnPar[1])/StaticNets.maxLargeVal

        foo, logLikeVecT, foo1 = score_driven_filter( model, N, obsT, vecReGasParAll, indTvPar;  vConstPar =  vecConstPar, ftot_0 = ftot_0 .* oneInADterms)

         return - sum(logLikeVecT)
    end
    #Run the optimization
    if uppercase(model.scoreScalingType) == "FISHER-EWMA"
        ADobjfunGas = objfunGas
    else
        ADobjfunGas = TwiceDifferentiable(objfunGas, vParOptim_0; autodiff = :forward);
    end

    Logging.@debug("Starting point for Optim $vParOptim_0")
    optim_out2  = optimize(ADobjfunGas,vParOptim_0 ,algo,optims_opt)
    outParAllUn = Optim.minimizer(optim_out2)
    vecAllParGasHat, vecAllParConstHat = divideCompleteRestrictPar(outParAllUn)

    Logging.@debug(optim_out2)

    Logging.@debug("Final paramters SD: $vecAllParGasHat , constant: $vecAllParConstHat ")


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
   
    return  arrayAllParHat, conv_flag,UM , ftot_0
   
end


#region Uncertainties filtered parameters


function A0_B0_est_for_white_cov_mat_obj_SD_filter_time_seq(model::GasNetModel, N, obsT, vEstSdResPar, indTvPar, ftot_0)

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


function white_estimate_cov_mat_static_sd_par(model::GasNetModel,  N,obsT, indTvPar, ftot_0, vEstSdResPar; returnAllMats=false)

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


function divide_in_B_A_mats_as_if_all_TV(model::GasNetModel, indTvPar, vEstSdResPar)
  
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


function distrib_filtered_par_from_mv_normal(model::GasNetModel, N, obsT, indTvPar, ftot_0, vEstSdResPar, mvSDUnParEstCov; nSample = 1000)
        
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
function par_bootstrap_distrib_filtered_par(model::GasNetModel, N, obsT, indTvPar, ftot_0, vEstSdResPar; nSample = 50)

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



function conf_bands_buccheri(model::GasNetModel, obsT, indTvPar, fVecT_filt, distribFilteredSD, filtCovHatSample, quantilesVals::Vector{Vector{Float64}}; nSample = 500, )


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


function plot_filtered_and_conf_bands(model::GasNetModel, N, fVecT_filt, confBands1; confBands2 =zeros(2,2), parDgpT=zeros(2,2), nameConfBand1="1", nameConfBand2="2")

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


function estimate_and_filter(model::GasNetModel, N, obsT; indTvPar = model.indTvPar)

    T = length(obsT)   
    
    estSdResPar, conv_flag, UM_mple, ftot_0 = estimate(model, N, obsT; indTvPar=indTvPar, indTargPar=falses(2))


    vEstSdResPar = array2VecGasPar(model, estSdResPar, indTvPar)

    fVecT_filt , target_fun_val_T, sVecT_filt = score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;ftot_0 = ftot_0)

    return obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0
end
    

function conf_bands_given_SD_estimates(model::GasNetModel, obsT, N, vEstSdResPar, ftot_0, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE" )
    
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


function estimate_filter_and_conf_bands(model::GasNetModel, A_T, quantilesVals::Vector{Vector{Float64}}; indTvPar = model.indTvPar, parDgpT=zeros(2,2), plotFlag=false, parUncMethod = "WHITE-MLE")
    
    N = size(A_T)[1]

    obsT = seq_of_obs_from_seq_of_mats(model, A_T)

    obsT, vEstSdResPar, fVecT_filt, ~, ~, conv_flag, ftot_0 = estimate_and_filter(model, N, obsT; indTvPar = indTvPar)
    
    fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = conf_bands_given_SD_estimates(model, obsT, N, vEstSdResPar, ftot_0, quantilesVals; indTvPar = indTvPar, parDgpT=parDgpT, plotFlag=plotFlag, parUncMethod = parUncMethod)

    return obsT, vEstSdResPar, fVecT_filt, confBandsFiltPar, confBandsPar, errFlag

end



function simulate_and_estimate_parallel(model::GasNetModel, dgpSettings, T, N, nSample)

    count = SharedArray(ones(1))
    res = @sync @distributed vcat for k=1:nSample
        
        Logging.@info("Estimating N = $N , T=$T iter n $(count[1])")

        parDgpT = DynNets.sample_time_var_par_from_dgp(model, dgpSettings.type, N, T;  dgpSettings.opt...)

        A_T_dgp = DynNets.sample_mats_sequence(model, parDgpT,N)

        obsT = seq_of_obs_from_seq_of_mats(model, A_T)

        ~, vEstSdResPar, fVecT_filt, ~, ~, conv_flag, ftot_0 = estimate_and_filter(model, N, obsT; indTvPar = indTvPar)
        
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