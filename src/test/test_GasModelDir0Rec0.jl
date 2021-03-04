# test that the GasNetModel interface works for a concrete model<:GasNetModel 

using Pkg
Pkg.activate(".") 
Pkg.instantiate() 

using Test
using ScoreDrivenERGM

import ScoreDrivenERGM:StaticNets,DynNets


model_pmle_fish = DynNets.GasNetModelDirBin0Rec0_pmle("SQRT_FISH_D")
model_pmle_hess = DynNets.GasNetModelDirBin0Rec0_pmle("HESS_D")

model_mle = DynNets.GasNetModelDirBin0Rec0_mle()


# sample dgp
begin
N = 100
T = 200
quantileVals = [[0.975, 0.025]]
listDgpSettigns = DynNets.list_example_dgp_settings(model_mle)
dgpSet = listDgpSettigns.dgpSetAR
dgpSet.opt.sigma[1] = 0.1

parDgpT = DynNets.sample_time_var_par_from_dgp(model_mle, dgpSet.type, N, T;  dgpSet.opt...)

A_T = DynNets.sample_mats_sequence(model_mle, parDgpT,N)

obsT = DynNets.seq_of_obs_from_seq_of_mats(model_pmle_fish, A_T)


end

@time begin 
   obsT, vEstSdResPar, ftot_0, fVecT_filt, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(DynNets.GasNetModelDirBin0Rec0_pmle("HESS_D"), A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
    res = (;obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0)
end

# @time begin 
#     obsT, vEstSdResPar, ftot_0, fVecT_filt, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(DynNets.GasNetModelDirBin0Rec0_pmle("SQRT_FISH_D"), A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
#     resFish = (;obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0)
# end






fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model, N, res.obsT, res.vEstSdResPar, res.ftot_0, quantilesVals; indTvPar = model_pmle_fish.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "WHITE-MLE" )



using LinearAlgebra

@show mvSDUnParEstCov, errFlagEstCov = DynNets.white_estimate_cov_mat_static_sd_par(model_pmle_hess, N, res.obsT, model_pmle_hess.indTvPar, res.ftot_0, res.vEstSdResPar)[1][I(6)]






# correctly specified filter

# estimate






# store the sufficient statistics and change statistics in R




