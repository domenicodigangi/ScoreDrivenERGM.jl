"""
A generic ERGM for directed networks to be estimated using the pseudo likelihood 
"""
struct  NetModeErgmPml <: NetModel 
    ergmTermsString::String # needs to be compatible with R ergm package notation and names
    isDirected::Bool
    nErgmPar::Int
end
export NetModeErgmPml

NetModeErgmPml() = NetModeErgmPml("", false, 0)

NetModeErgmPml(ergmTermsString, isDirected)  = NetModeErgmPml(ergmTermsString, isDirected, 1 + count(i->(i=='+'), ergmTermsString))

name(x::NetModeErgmPml) = "NetModeErgmPml($(x.ergmTermsString))"


type_of_obs(model::NetModeErgmPml) =  Array{Float64, 2}


function change_stats(model::NetModeErgmPml, A::Matrix)
    ErgmRcall.get_change_stats(A,model.ergmTermsString)
end


function pseudo_loglikelihood_strauss_ikeda(par, changeStat, response, weights)
    
    H_vec = dropdims(sum(par.*changeStat', dims=1), dims=1)      

    logPTot = sum(- weights .* ( ((1 .-response) .* H_vec) .+ log.(1 .+ exp.(-H_vec))) )

    #old version should be equal to current one
    # logit_P = sum(par.*changeStat', dims=1)      

    # P = inv_logit.(logit_P)    

    # Pbar = 1 .- P

    # logPVec = log.([response[i] == zero(response[i]) ? Pbar[i] : P[i] for i=1:length(response) ])

    # logPTot = sum(logPVec.*weights)

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


estimate(model::NetModeErgmPml, A::Matrix) = ErgmRcall.get_mple(model.ergmTermsString, A)


