"""
ERGM for directed networks with totan links and tortal reciprocated links as statistics 
"""
struct  NetModelDirBin0Rec0 <: NetModel end


ergm_term_string(x::NetModelDirBin0Rec0)  = "edges + mutual"


type_of_obs(model::NetModelDirBin0Rec0) =  Array{Float64, 1}


"""
given the 2 model's parameters compute the probabilities for each state of the diad
"""
function diadProbFromPars(Model::NetModelDirBin0Rec0, par)
    θ = par[1]
    η = par[2]
    eθ = exp(θ)
    e2θplusη = exp(2*θ + η)
    den = 1 + 2*eθ + e2θplusη
    p00 = 1/den
    p01 = eθ /den
    p10 = p01 
    p11 = e2θplusη/den
    diadProbsVec = [p00, p01, p10, p11]
    return diadProbsVec
end


"""
given the vector of diad states probabilities sample one random matrix from the corresponding pdf
"""
function samplSingMatCan(Model::NetModelDirBin0Rec0, diadProbsVec::Array{<:Real,1}, N)
    out = zeros(Int8,N,N)
    #display((maximum(expMat),minimum(expMat)))
    for c = 1:N
        for r=1:c-1
            diadState = rand(Categorical(diadProbsVec))
            if diadState==2
                out[r,c] = 1
            elseif diadState==3
                out[c,r] = 1
            elseif diadState==4
                out[r,c] = 1
                out[c,r] = 1
            end
        end
    end
    return out
end


function sample_ergm(model::NetModelDirBin0Rec0, N, θ_0, η_0, nSample)

    diadProb = diadProbFromPars(model, [θ_0, η_0])

    A_vec = [samplSingMatCan(model, diadProb, N) for i=1:nSample]

    return A_vec
end


function statsFromMat(Model::NetModelDirBin0Rec0, A ::Matrix{<:Real})
    L = sum(A)
    R = sum(A'.*A)/2
    N=size(A)[1]

    return [L, R, N]
end


function exp_val_stats(model::NetModelDirBin0Rec0, θ, η, N)
    x = exp(θ)
    x2y = exp(2*θ + η)
    z = 1 + 2*x + x2y
    Nlinks = n_pox_dir_links(N)
    α = (x + x2y)/z
    β = 0.5 * (x2y)/z
    L = Nlinks * α
    R = Nlinks * β
    return L, R
end


"""
return the ergm parameters that fix on average the input values for the ergm statistics:
    - L is  ∑_i>j A_ij + A_ji
    - R is  of ∑_i>j A_ij * A_ji /2
"""
function ergm_par_from_mean_vals(model::NetModelDirBin0Rec0, L, R, N)
    Nlinks = n_pox_dir_links(N)
    # average values per pair
    α = L/Nlinks
    β = R/Nlinks

    x = (2β-α)/(2α-2β-1)
    θ = log(x)

    y = 2(2*β^2  + β- 2*α*β )/((2β-α)^2)
    η =  log(y)

    return θ, η 
end


function estimate(model::NetModelDirBin0Rec0, L, R, N)   
    θ_est, η_est = ergm_par_from_mean_vals(model, L, R, N)
    vPar = [θ_est, η_est]
    return vPar
end


function estimate(Model::NetModelDirBin0Rec0, A)
    L, R, N = statsFromMat(Model, A) 
    return estimate(Model, L, R, N)
end


function logLikelihood(Model::NetModelDirBin0Rec0, L, R, N, par)
    θ, η = par
    z = 1 + 2*exp(θ) + exp(2*θ+η)
    return L * θ + R*η - (N*(N-1)/2)*log(z)
end


function logLikelihood(Model::NetModelDirBin0Rec0, A::Matrix, par)
    L, R, N = statsFromMat(Model, A)
    return logLikelihood(Model, L, R, N, par)
end

