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


function setOptionsOptim(model::T where T<: GasNetModelDirBin0Rec0)
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


function static_estimate(model::GasNetModelDirBin0Rec0_mle, statsT)
    L_mean  = mean([stat[1] for stat in statsT ])
    R_mean  = mean([stat[2] for stat in statsT ])
    N_mean  = mean([stat[3] for stat in statsT ])
    
    staticPars = StaticNets.estimate(model.staticModel, L_mean, R_mean, N_mean )
    return staticPars
end


GasNetModelDirBin0Rec0_pmle() = SdErgmPml(ergm_term_string(NetModelDirBin0Rec0))
export GasNetModelDirBin0Rec0_pmle

