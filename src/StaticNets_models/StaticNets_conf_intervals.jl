
"""
Given parEst, an estimate the ergm parameters, sample from that ergm nSample times and re-estimate the same ergm on the obtained sample of observations. 

Return a distribution of estimated parameter, aka parametric bootstrap following "Exponential Random Graph Models with Big Networks: Maximum Pseudolikelihood Estimation and the Parametric Bootstrap" of Christian S. Schmid, Bruce A. Desmarais. 
"""
function get_par_boot_ergm_distrib(model, parEst, N; nSample = 100)

    mat_sample = StaticNets.sample_ergm(model, N, parEst, nSample)

    estimates = zeros(length(parEst), nSample)

    Threads.@threads for i in 1:nSample
        estimates[:,i] = StaticNets.estimate(model,  mat_sample[:,:,i])  
    end
    return estimates
end
export get_par_boot_ergm_distrib



"""
Get the confidence intervals from parametric bootstrap, as described in "Exponential Random Graph Models with Big Networks: Maximum Pseudolikelihood Estimation and the Parametric Bootstrap" of Christian S. Schmid, Bruce A. Desmarais. 
"""
function get_conf_int_ergm_par_boot(model, N, parEst, quantilesVals; nSample = 100) 

    par_b_dist = get_par_boot_ergm_distrib(model, parEst, N; nSample = nSample)

    confInt = reduce(vcat, [quantile(par_b_dist[p,:], quantilesVals)' for p in 1:model.nErgmPar])
end
export get_conf_int_ergm_par_boot


is_between(x, interval) = interval[1] <= x <= interval[2]


function get_coverage_conf_int_par_boot_ergm(model, N, parDgp, quantilesVals; nSampleDgp=100, nSampleBands = 100)

    # sample from dgp
    matsSample = StaticNets.sample_ergm(model, N, parDgp, nSampleDgp)

    estimates = reduce(hcat,[StaticNets.estimate(model, A) for A in matsSample])

    coverFlags =falses(size(estimates))

    for n in 1:nSampleDgp
        confInt = get_conf_int_ergm_par_boot(model, N, estimates[:, n], quantilesVals; nSample=nSampleBands)
        
        for p in 1:model.nErgmPar  
            coverFlags[p,n] = is_between(parDgp[p], confInt[p,:]) 
        end
    end

    return coverFlags
end
export get_coverage_conf_int_par_boot_ergm
