# test that the GasNetModel interface works for a concrete model<:GasNetModel 

using Pkg
Pkg.activate(".") 
Pkg.instantiate() 

using Test
using ScoreDrivenERGM
import ScoreDrivenERGM:StaticNets,DynNets, Utilities
using LinearAlgebra
using SharedArrays
using PyPlot
using Statistics
using StatsBase
using DataStructures

# model = DynNets.GasNetModelDirBin0Rec0_pmle("HESS_D")
model = DynNets.GasNetModelDirBin0Rec0_pmle(scoreScalingType ="FISH_D")
model_mle = DynNets.GasNetModelDirBin0Rec0_mle(scoreScalingType="FISH_D")
model_2 = model_mle
# model_3 =  DynNets.GasNetModelDirBin0Rec0_pmle(scoreScalingType="FISH_D", options = SortedDict("Firth" => true))




# sample dgp
@time begin 
N = 50
T =300
quantileVals = [[0.975, 0.025]]
listDgpSettigns = DynNets.list_example_dgp_settings(model_mle)
# dgpSet = listDgpSettigns.dgpSetSD
# dgpSet.opt.A[1] = 1
dgpSet = listDgpSettigns.dgpSetARlow


parDgpT = DynNets.sample_time_var_par_from_dgp(model_mle, dgpSet.type, N, T;  dgpSet.opt...)

A_T = DynNets.sample_mats_sequence(model_mle, parDgpT,N)

    obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(model, A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
    res = (;  obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag)
end

@time begin
    obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(model_2, A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
    res_2 = 
    (;  obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag)
end

@time begin
    #     obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(model_3, A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
    #     res_3 = 
    #     (;  obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag)
end


# Single snapshot estimators
begin
fVecT_filt_SS =  DynNets.estimate_single_snap_sequence(model, A_T)

fVecT_filt_SS_2 =  DynNets.estimate_single_snap_sequence(model_2, A_T)

fig, ax =  DynNets.plot_filtered(model, N, fVecT_filt_SS; lineType = ".", lineColor="r", parDgpTIn=parDgpT)
DynNets.plot_filtered(model, N, fVecT_filt_SS_2; lineType = ".", lineColor="b", parDgpTIn=parDgpT, fig=fig, ax=ax)
DynNets.plot_filtered(model, N, fVecT_filt; lineType = "-", parDgpTIn=parDgpT, fig=fig, ax=ax)
DynNets.plot_filtered(model, N, mapslices(rolling_mean,fVecT_filt_SS, dims=2); lineType = "-", parDgpTIn=parDgpT, fig=fig, ax=ax)
 
end

@show var(parDgpT, dims=2)
@show var(fVecT_filt, dims=2)
@show var(fVecT_filt_SS, dims=2)


# Compare estimators of variances of static parameter's estimators
mvSDUnParEstCovWhite, errFlagEstCov = DynNets.white_estimate_cov_mat_static_sd_par(model_2, N, res_2.obsT, model_2.indTvPar, res_2.ftot_0, res_2.vEstSdResPar)

mvSDUnParEstCovWhite[I(6)]

distribFilteredSDParBoot, filtCovHatSample, errFlagVec , vEstSdUnParBootDist = DynNets.par_bootstrap_distrib_filtered_par(model_2, N, res_2.obsT, model_2.indTvPar, res_2.ftot_0, res_2.vEstSdResPar)


vEstSdUnParNPBootDist = DynNets.non_par_bootstrap_distrib_filtered_par(model_2, N, obsT, model_2.indTvPar, ftot_0; nBootStrap = 100)

figure()
hist(filtCovHatSample')
mapslices( x -> mean(winsor(x, prop=0.005)), filtCovHatSample, dims=2)

using ScikitLearn
@sk_import covariance: MinCovDet


DynNets.conf_bands_given_SD_estimates(model_2, N, res_2.obsT, vEstSdUnPar, res_2.ftot_0, [[0.975, 0.025]]; parDgpT=parDgpT, plotFlag=true, parUncMethod = "WHITE-MLE", offset = 0, mvSDUnParEstCov = 0.3 .*mvSDUnParEstCovWhite, winsorProp=0.05)

DynNets.conf_bands_given_SD_estimates(model_2, N, res_2.obsT, vEstSdUnPar, res_2.ftot_0, [[0.975, 0.025]]; parDgpT=parDgpT, plotFlag=true, parUncMethod = "NPB-MVN", offset = 0, mvSDUnParEstCov = MinCovDet().fit(Utilities.drop_bad_un_estimates(vEstSdUnParNPBootDist)').covariance_, winsorProp=0.05)

DynNets.conf_bands_given_SD_estimates(model_2, N, res_2.obsT, vEstSdUnPar, res_2.ftot_0, [[0.975, 0.025]]; parDgpT=parDgpT, plotFlag=true, parUncMethod = "PB-MVN", offset = 0, mvSDUnParEstCov = MinCovDet().fit(Utilities.drop_bad_un_estimates(vEstSdUnParBootDist)').covariance_, winsorProp=0.05)


DynNets.conf_bands_given_SD_estimates(model_2, N, res_2.obsT, vEstSdUnPar, res_2.ftot_0, [[0.975, 0.025]]; parDgpT=parDgpT, plotFlag=true, parUncMethod = "PB-SAMPLE", offset = 0, sampleStaticUnPar=Utilities.drop_bad_un_estimates(vEstSdUnParBootDist), winsorProp=0 )



if true
parNames = ["w_theta", "B_theta", "A_theta", "w_eta", "B_eta", "A_eta"]
sample1 = vEstSdUnParBootDist
sample1 = Utilities.drop_bad_un_estimates(sample1)

fig, ax = subplots(3,2)
redLines = DynNets.unrestrict_all_par(model_2, model_2.indTvPar, res_2.vEstSdResPar)
    if true
        sample1 = mapslices(x -> DynNets.restrict_all_par(model_2, model_2.indTvPar, x), sample1, dims=1)
        redLines = res_2.vEstSdResPar
    end

for i = 1:6
    ax[i].hist(sample1[i,:], range=quantile(sample1[i,:], [0.01, 0.99]), 20, alpha = 0.4, density=true)
    ax[i].set_title(parNames[i])
    ylim = ax[i].get_ylim()
    ax[i].vlines(redLines[i], ylim[1], ylim[2], color = "r" )

end
end

plt.figure()
plt.hist(mvSDUnParEstCovParBoot[3, :])

distribFilteredSD, filtCovHatSample,  _, errFlagSample = DynNets.distrib_filtered_par_from_mv_normal(model_2, N, res_2.obsT, model_2.indTvPar, res_2.ftot_0, res_2.vEstSdUnPar, mvSDUnParEstCov)




# nBootStrap = 150
# vEstSdUnParBootDist = SharedArray(zeros(3*sum(model.indTvPar), nBootStrap))

# @time Threads.@threads for k=1:nBootStrap
#     vEstSdUnParBootDist[:, k] = rand(1:T, T) |> (inds->(inds |> x-> DynNets.estimate(model, N, res.obsT; indTvPar=model.indTvPar, ftot_0 = res.ftot_0, shuffleObsInds = x) |> x-> getindex(x, 1) |> x -> DynNets.array2VecGasPar(model, x, model.indTvPar))) |> x -> DynNets.unrestrict_all_par(model, model.indTvPar, x)
# end


 
# @time begin 
#     obsT, vEstSdResPar, ftot_0, fVecT_filt, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(DynNets.GasNetModelDirBin0Rec0_pmle("FISH_D"), A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
#     resFish = (;obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0)
# end


# IL PROBLEMA € NELLA MATRICE DI VARIANZA COVARIANZA STIMATA CON WHITE. QUELLA NON HA LA GIUSTA MAGNITUDE, quando i parametri time varying variano molto





# @time fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model, N, res.obsT, res.vEstSdResPar, res.ftot_0, quantileVals; indTvPar = model.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "NPB-MVN")







# correctly specified filter

# estimate






# store the sufficient statistics and change statistics in R




