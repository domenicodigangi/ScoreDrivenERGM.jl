"""
Load and plot simulations results on confidence bands' coverages
"""

#region import and models
using Pkg
Pkg.activate(".") 
Pkg.instantiate() 
using ScoreDrivenERGM
#endregion
using Profile
using ProfileView


begin
N=200
T=900

model_mle = DynNets.GasNetModelDirBin0Rec0_mle()
model_pmle = DynNets.GasNetModelDirBin0Rec0_pmle()
indTvPar = trues(2)

dgpSetAR, ~, dgpSetSD = ScoreDrivenERGM.DynNets.list_example_dgp_settings(model_mle)

dgpSetAR = (type = "AR", opt =( B = 0.98, sigma =0.01))
dgpSettings = dgpSetAR
model = model_mle
parDgpT = DynNets.sample_time_var_par_from_dgp(model, dgpSettings.type, N, T;  dgpSettings.opt...)

A_T_dgp = DynNets.sample_mats_sequence(model, parDgpT,N)

obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0 = DynNets.estimate_and_filter(model, N, obsT)

ProfileView.@profview obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0 = DynNets.estimate_and_filter(model, N, obsT)
        


ProfileView.@profview @time DynNets.conf_bands_given_SD_estimates(model, obsT, N, vEstSdResPar, ftot_0, quantilesVals;  parUncMethod = "WHITE-MLE")


@code_warntype DynNets.conf_bands_given_SD_estimates(res.model, obsT, N, vEstSdResPar, ftot_0, quantilesVals;  parUncMethod = "WHITE-MLE")

@code_warntype DynNets.score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;  ftot_0 = ftot_0)
