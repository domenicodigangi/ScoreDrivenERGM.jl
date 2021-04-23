""
# functions that allow to sample sequences of ergms with different parameters' values from R package ergm
# and estimate the ergm

module ErgmRcall

using DataFrames
using RCall
using SparseArrays


function clean_start_RCall()
    # Clean R enviroment 
    R"rm(list = ls())
    sessionInfo()"
   R"set.seed(0)"
end
export clean_start_RCall


function install_and_load_R_package(pkgName::String)
    R"""
        if( !( " $pkgName " %in% rownames(installed.packages())) ){
            install.packages("$pkgName")
        }

        library("$pkgName")
    """

end

function init_R_settings()
    # R"""options(warn=-1) """

    R"""local({r <- getOption("repos")
        r["CRAN"] <- "http://cran.r-project.org" 
        options(repos=r)})
        """
end

function __init__()

    init_R_settings()    
    # install_and_load_R_package("sna")
    # install_and_load_R_package("coda")
    # install_and_load_R_package("network")
    # install_and_load_R_package("ergm")
    R"""library("network")"""
    R"""library("ergm")"""
    
    clean_start_RCall()
end


function sampleErgmRcall(parDgpT,N,Nsample,formula_ergm_str)
    """
    Function that samples from an ergm defined by formula_ergm_str (according to
    the notation of R package ergm)
    in order to work it needs RCall.jl insatalled and the packages listed at the
    beginning of the module installed in R
    """

    clean_start_RCall()
    T = size(parDgpT)[2]
    # For each t, sample the ERGM with parameters corresponding to the DGP at time t
    # and run a single snapeshot estimate
    @rput T; @rput parDgpT;@rput N;@rput Nsample

    reval("formula_ergm = net ~ "*formula_ergm_str)

     #create an empty network, the formula defining ergm, sample the ensemble and
     # store the sufficient statistics and change statistics in R
     R"
     net <- network.initialize(N)
       sampledMat_T_R =    array(0, dim=c(N,N,T,Nsample))
       changeStats_T_R = list()
       stats_T_R = list()
        for(t in 1:T){
            changeStats_t_R = list()
            stats_t_R = list()
            print(t)
            for(n in 1:Nsample){
                 net <- simulate(formula_ergm, nsim = 1, seed = sample(1:100000000,1), coef = parDgpT[,t],control = control.simulate.formula(MCMC.burnin = 100000))
                 sampledMat_T_R[,,t,n] <- as.matrix.network( net)
                 print(c(t,n,parDgpT[,t]))
                 chStat_t <- ergmMPLE(formula_ergm)
                 changeStats_t_R[[n]] <- cbind(chStat_t$response, chStat_t$predictor,chStat_t$weights)
                 stats_t_R[[n]] <- summary(formula_ergm)
                 }
                  changeStats_T_R[[t]] <-changeStats_t_R
                  stats_T_R[[t]] <- stats_t_R
             }"



    # import sampled networks in julia
    sampledMat_T = BitArray( @rget(sampledMat_T_R))
    changeStats_T = @rget changeStats_T_R;# tmp = Array{Array{Float64,2}}(T); for t=1:T tmp[t] =  changeStats_T[t];end;changeStats_T = tmp
    stats_T = @rget(stats_T_R); #tmp = zeros(Nterms,T); for t=1:T tmp[:,t] = stats_T[t];end;stats_T = tmp
    return sampledMat_T, changeStats_T, stats_T
end
export sampleErgmRcall


function get_edge_list(A::Matrix) 
    @debug "[get_edge_list][begin]"

    if size(A)[2] == 2 
        return A
    end

    size(A)[1] == size(A)[2] ? () : error("Adjacency matrix must be squared")
    spA = sparse(A)
    spA_lists = findnz(spA)
    @debug "[get_edge_list][end]"
    return hcat(spA_lists[1], spA_lists[2])
end


"""
Function that estimates a sequence of ergm defined by formula_ergm_str (according to the notation of R package ergm)
"""
function get_one_mle(A::Matrix{T} where T<:Integer, ergmTermsString::String)
    @debug "[estimate_mle_RCall][begin]"
    
    edge_list = get_edge_list(A)   
    @rput edge_list
    reval("formula_ergm = net ~ "*ergmTermsString)
    try
        R"
            net <- as.network.matrix(edge_list)

            tmp <- ergm(formula_ergm)#,estimate = 'MPLE')#)#

            estPar_R <- tmp[[1]]
        "
        estPar = @rget(estPar_R)
    catch
        @warn "[estimate_mle_RCall][mle failed. returning mple]"
        estPar = get_one_mple(A, ergmTermsString)
    end

    
    @debug "[estimate_mle_RCall][end]"
    return estPar
end
export get_one_mle




function get_change_stats(A::Matrix{T} where T<:Integer, ergmTermsString::String)
    # given a matrix returns the change statistics wrt to a given formula
    edge_list = get_edge_list(A)   
    @rput edge_list

    reval("formula_ergm = net ~ "* ergmTermsString)
    # store the sufficient statistics and change statistics in R
    R"""
        net <- network(edge_list)
        chStat_t <- ergmMPLE(formula_ergm)
        changeStats_t_R <- cbind(chStat_t$response, chStat_t$predictor,     chStat_t$weights)
            """

    changeStats = @rget changeStats_t_R;# tmp = Array{Array{Float64,2}}(T); for 
    return changeStats
end
export get_change_stats
 

""" Get a single pmle from adjacency matrix""" 
function get_one_mple(A::Matrix{T} where T<:Integer, ergmTermsString::String)
    edge_list = get_edge_list(A)   
    @rput edge_list
    reval("formula_ergm = net ~ "* ergmTermsString)
    # store the sufficient statistics and change statistics in R
    R"""
        net <- network(edge_list)
        mple_t_R <- ergmMPLE(formula_ergm, output="fit")$coef
        # print(mple_t_R)
        """

    mple_t = @rget mple_t_R;# tmp = Array{Array{Float64,2}}(T); for 
    return mple_t
end
export get_one_mple


""" Get a single pmle from change statistics for a single network""" 
function get_one_mple(changeStats_T::Array{T,2} where T<:Real, ergmTermsString::String)
    # and run a single snapeshot estimate
    @rput changeStats_T
    reval("formula_ergm = net ~ "* ergmTermsString)
    # store the sufficient statistics and change statistics in R
    R"""
        mple_static_R <- glm(changeStats_T[,1] ~ . -1, data = data.frame(changeStats_T[,c(2:(ncol(changeStats_T)-1))]), weights =changeStats_T[,ncol(changeStats_T)],family="binomial")$coefficients
        """

    mple_static = @rget mple_static_R;# tmp = Array{Array{Float64,2}}(T); for 

    return mple_static
end

""" Get a single pmle from change statistics for a sequence of networks""" 
function get_one_mple(changeStats_T::Array{Array{T,2},1} where T<:Real, ergmTermsString::String)
    return get_one_mple(reduce(vcat,changeStats_T), ergmTermsString)
end
export get_one_mple


function decomposeMPLEmatrix(ergmMPLEmatrix)  
    changeStat = ergmMPLEmatrix[:,2:3]
    response = ergmMPLEmatrix[:,1]
    weights = ergmMPLEmatrix[:,4]
    return changeStat, response, weights 
end
export decomposeMPLEmatrix

end
