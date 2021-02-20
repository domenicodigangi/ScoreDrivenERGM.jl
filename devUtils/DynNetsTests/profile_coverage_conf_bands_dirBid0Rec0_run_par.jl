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

N=100
T=100

model_mle = DynNets.GasNetModelDirBin0Rec0_mle()
model_pmle = DynNets.GasNetModelDirBin0Rec0_pmle()
indTvPar = trues(2)

dgpSetAR, ~, dgpSetSD = ScoreDrivenERGM.DynNets.list_example_dgp_settings_for_paper(model_mle)

dgpSettings = dgpSetAR
model = model_mle
parDgpT = DynNets.dgp_misspecified(model, dgpSettings.type, N, T;  dgpSettings.opt...)

A_T_dgp = DynNets.sample_dgp(model, parDgpT,N)

obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0 = DynNets.estimate_and_filter(model, A_T_dgp)

ProfileView.@profview obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0 = DynNets.estimate_and_filter(model, A_T_dgp)
        
quantilesVals = [[0.975, 0.025]]


ProfileView.@profview @time DynNets.conf_bands_given_SD_estimates(model, obsT, N, vEstSdResPar, ftot_0, quantilesVals;  parUncMethod = "WHITE-MLE")


@code_warntype DynNets.conf_bands_given_SD_estimates(res.model, obsT, N, vEstSdResPar, ftot_0, quantilesVals;  parUncMethod = "WHITE-MLE")

@code_warntype DynNets.score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;  ftot_0 = ftot_0)
