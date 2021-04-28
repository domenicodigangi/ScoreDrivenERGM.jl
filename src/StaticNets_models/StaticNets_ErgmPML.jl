"""
A generic ERGM for directed networks to be estimated using the pseudo likelihood 
"""
struct  NetModeErgmPml <: Ergm 
    ergmTermsString::String # needs to be compatible with R ergm package notation and names
    isDirected::Bool
    nErgmPar::Int
end
export NetModeErgmPml

NetModeErgmPml() = NetModeErgmPml("", false, 0)

NetModeErgmPml(ergmTermsString, isDirected)  = NetModeErgmPml(ergmTermsString, isDirected, 1 + count(i->(i=='+'), ergmTermsString))


name(x::NetModeErgmPml) = "NetModeErgmPml($(x.ergmTermsString))"


type_of_obs(model::NetModeErgmPml) =  Array{Float64, 2}


sample_ergm(model::NetModeErgmPml, N, parVec, nSample) = sample_ergm_RCall_sequence(model.ergmTermsString, N, parVec, nSample)[1][:,:,1,:]


sample_ergm_sequence(model::NetModeErgmPml, N, parVecSeq_T::Matrix, nSample) = sample_ergm_RCall_sequence(model.ergmTermsString, N, parVecSeq_T, nSample)


function change_stats(model::NetModeErgmPml, A::Matrix)
    ErgmRcall.get_change_stats(A,model.ergmTermsString)
end


function pseudo_loglikelihood_strauss_ikeda(par, changeStat, response, weights)
    # @debug "[pseudo_loglikelihood_strauss_ikeda][start][size par=$(size(par)), size changeStat = $(size(changeStat)), size response=$(size(response)), size weights=$(size(weights))]"
    
    H_vec = dropdims(sum(par.*changeStat', dims=1), dims=1)      

    logPTot = sum(- weights .* ( ((1 .-response) .* H_vec) .+ log.(1 .+ exp.(-H_vec))) )

    #old version should be equal to current one
    # logit_P = sum(par.*changeStat', dims=1)      

    # P = inv_logit.(logit_P)    

    # Pbar = 1 .- P

    # logPVec = log.([response[i] == zero(response[i]) ? Pbar[i] : P[i] for i=1:length(response) ])

    # logPTot = sum(logPVec.*weights)
    # @debug "[pseudo_loglikelihood_strauss_ikeda][end]"

    return  logPTot
end


function grad_pseudo_loglikelihood_strauss_ikeda(par, changeStat, response, weights)

    H_vec = dropdims(sum(par.*changeStat', dims=1), dims=1)      

    grad =  dropdims(sum((changeStat.*weights .* ( response .- (1 .+ exp.(-H_vec)).^(-1) ) ), dims=1), dims=1)


    return grad
end

function hessian_pseudo_loglikelihood_strauss_ikeda(par, changeStat, weights)

    H_vec = dropdims(sum(par.*changeStat', dims=1), dims=1)      

    P_vec = (1 .+ exp.(-H_vec)).^(-1) 

    nCSGroups, nErgmPar = size(changeStat) # number of groups of different change statistics
    prod = P_vec .* (1 .- P_vec).*weights
    hess = - [sum(prod.*changeStat[:,i].*changeStat[:,j]) for i = 1:nErgmPar, j=1:nErgmPar]

    return hess
end

function fisher_info_pseudo_loglikelihood_strauss_ikeda(par, changeStat,  weights)

    - hessian_pseudo_loglikelihood_strauss_ikeda(par, changeStat, weights)
end


obj_fun(par, changeStat, response, weights) =  pseudo_loglikelihood_strauss_ikeda(par, changeStat, response, weights)

grad_obj_fun(par, changeStat, response, weights) = grad_pseudo_loglikelihood_strauss_ikeda(par, changeStat, response, weights)

hessian_obj_fun(par, changeStat, weights) = hessian_pseudo_loglikelihood_strauss_ikeda(par, changeStat, weights)

fisher_info_obj_fun(par, changeStat,  weights) = fisher_info_pseudo_loglikelihood_strauss_ikeda(par, changeStat,  weights)

estimate_sequence(model::NetModeErgmPml, obsT::Array{Array{Float64, 2}}) = reduce(hcat, [ErgmRcall.get_one_mple(obst, model.ergmTermsString) for obst in obsT])

estimate_sequence(model::NetModeErgmPml, obsT::Array{T, 3} where T <: Real) = reduce(hcat, [ErgmRcall.get_one_mple(obsT[:,:,t], model.ergmTermsString) for t in 1:size(obsT)[3]])
