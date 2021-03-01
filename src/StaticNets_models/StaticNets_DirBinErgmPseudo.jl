"""
A generic ERGM for directed networks to be estimated using the pseudo likelihood 
"""
struct  NetModeErgmPml <: NetModel 
    ergmTermsString::String # needs to be compatible with R ergm package notation and names
    isDirected::Bool
    nErgmPar::Int
end
export NetModeErgmPml

NetModeErgmPml(ergmTermsString, isDirected)  = NetModeErgmPml(ergmTermsString, isDirected, 1 + count(i->(i=='+'), ergmTermsString))

name(x::NetModeErgmPml) = "NetModeErgmPml($(x.ergmTermsString))"


type_of_obs(model::NetModeErgmPml) =  Array{Float64, 2}


function change_stats(model::NetModeErgmPml, A::Matrix)
    ErgmRcall.get_change_stats(A,model.ergmTermsString)
end


function pseudo_loglikelihood_strauss_ikeda(model::NetModeErgmPml, par, changeStat, response, weights)
    logit_P = sum(par.*changeStat', dims=1)      
    P = inv_logit.(logit_P)    
    logPVec = log.([response[i] == zero(response[i]) ? 1 - P[i] : P[i] for i=1:length(response) ])
    logPTot = sum(logPVec.*weights)
    return  logPTot
end


estimate(model::NetModeErgmPml, A::Matrix) = ErgmRcall.get_mple(model.ergmTermsString, A)


