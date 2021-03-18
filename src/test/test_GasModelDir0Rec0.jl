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
model_3 =  DynNets.GasNetModelDirBin0Rec0_pmle(scoreScalingType="FISH_D", options = SortedDict("Firth" => true))




# sample dgp
@time begin 
N = 50
T =300
quantileVals = [[0.975, 0.025]]
listDgpSettigns = DynNets.list_example_dgp_settings(model_mle)
# dgpSet = listDgpSettigns.dgpSetSD
# dgpSet.opt.A[1] = 1
dgpSet = listDgpSettigns.dgpSetSDlow


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
    obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag = DynNets.estimate_filter_and_conf_bands(model_3, A_T, parDgpT=parDgpT, quantileVals, plotFlag=true);    
    res_3 = 
    (;  obsT, vEstSdResPar, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0, confBandsFiltPar, confBandsPar, errFlag)
end


res.vEstSdResPar 
res_2.vEstSdResPar
res_3.vEstSdResPar

mvSDUnParEstCov, errFlagEstCov = DynNets.white_estimate_cov_mat_static_sd_par(model_2, N, res_2.obsT, model_2.indTvPar, res_2.ftot_0, res_2.vEstSdResPar)

mvSDUnParEstCov[I(6)]

distribFilteredSD, filtCovHatSample,  _, errFlagSample = DynNets.distrib_filtered_par_from_mv_normal(model_2, N, res_2.obsT, model_2.indTvPar, res_2.ftot_0, res_2.vEstSdResPar, mvSDUnParEstCov)

plot(distribFilteredSD[:,2,:])



f_t = res_2.fVecT_filt[:,10]
obs_t = res_2.obsT[10]
target_fun_t(x) = DynNets.target_function_t(model_mle, obs_t, N, x)

using ForwardDiff
hess_tot_t = ForwardDiff.hessian(target_fun_t, f_t)
target_function_t_hess(model_mle, obs_t, N, f_t)


ForwardDiff.gradient(target_fun_t, f_t)

target_function_t_grad(model_mle, obs_t, N, f_t)


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


# @time fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCovBoot, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model, N, res.obsT, res.vEstSdResPar, res.ftot_0, quantileVals; indTvPar = model.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "NPB-SAMPLE")


BMatSD, AMatSD = DynNets.divide_in_B_A_mats_as_if_all_TV(model, model.indTvPar, res.vEstSdResPar)

constFiltUncCoeff = (BMatSD.^(-1)).*AMatSD
        
distribFilteredSD, filtCovHatSample, mvSDUnParEstCov, errFlagMvNormSample = DynNets.distrib_filtered_par_from_mv_normal(model, N, res.obsT, model.indTvPar, res.ftot_0, res.vEstSdResPar, mvSDUnParEstCov)

mean(filtCovHatSample, dims=3)



@time fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model, N, res.obsT, res.vEstSdResPar, res.ftot_0, quantileVals; indTvPar = model.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "WHITE-MLE", offset=1 )


@time fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model_mle, N, res_mle.obsT, res_mle.vEstSdResPar, res_mle.ftot_0, quantileVals; indTvPar = model.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "WHITE-MLE", offset=1 )

@time fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model_mle, N, res_mle.obsT, res_mle.vEstSdResPar, res_mle.ftot_0, quantileVals; indTvPar = model.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "NPB-COV-MAT", offset=1 )


# @time fVecT_filt, confBandsFiltPar, confBandsPar, errFlag, mvSDUnParEstCov, distribFilteredSD = DynNets.conf_bands_given_SD_estimates(model, N, res.obsT, res.vEstSdResPar, res.ftot_0, quantileVals; indTvPar = model.indTvPar, parDgpT=parDgpT, plotFlag=true, parUncMethod = "NPB-COV-MAT")




begin
nBootStrap = 50
vEstSdUnParBootDist = SharedArray(zeros(3*sum(model.indTvPar), nBootStrap))

w = ones(T)# dropdims( sqrt.(sum(res.sVecT_filt.^2, dims=1)) , dims=1)

weights = Weights(w./sum(w))
@time Threads.@threads for k=1:nBootStrap
    vEstSdUnParBootDist[:, k] = sample(1:T, weights, T) |> (inds->(inds |> x-> DynNets.estimate(model, N, res.obsT; indTvPar=model.indTvPar, ftot_0 = res.ftot_0, shuffleObsInds = x) |> x-> getindex(x, 1) |> x -> DynNets.array2VecGasPar(model, x, model.indTvPar))) |> x -> DynNets.unrestrict_all_par(model, model.indTvPar, x)
end
end

begin
mvSDUnParEstCov, errFlagEstCov = cov(drop_bad_un_estimates(vEstSdUnParBootDist)'), false
# mvSDUnParEstCov[I(6)] = [abs(diff(quantile(s, [1- (1-P)/2, (1-P)/2] ))[1]/2) for s in eachrow(vEstSdUnParBootDist)]
# mvSDUnParEstCov[I(6)] = cov(vEstSdUnParBootDist')[I(6)]
mvSDUnParEstCov, minEigenVal = Utilities.make_pos_def(mvSDUnParEstCov)
# mvSDUnParEstCov = Symmetric(mvSDUnParEstCov)

#  distribFilteredSD, filtCovHatSample, mvSDUnParEstCov, errFlagMvNormSample = DynNets.distrib_filtered_par_from_mv_normal(model, N, obsT, model.indTvPar, ftot_0, vEstSdResPar, mvSDUnParEstCov)

distribFilteredSD, filtCovHatSample, errFlagSample = DynNets.distrib_filtered_par_from_sample(model, N, obsT, model.indTvPar, ftot_0, vEstSdUnParBootDist)

confBandsFiltPar,  confBandsPar = DynNets.conf_bands_buccheri(model, N, obsT, model.indTvPar, res.fVecT_filt, distribFilteredSD, filtCovHatSample, quantileVals, false)


DynNets.plot_filtered_and_conf_bands(model, N, fVecT_filt, confBandsFiltPar ; parDgpTIn=parDgpT, nameConfBand1= " - Filter+Par", nameConfBand2= "Par", confBands2In=confBandsPar, offset = 1)


vEstSdResParBootDist = mapslices(x -> DynNets.restrict_all_par(model, model.indTvPar, x ), vEstSdUnParBootDist, dims=1)
parNames = ["w_theta", "B_theta", "A_theta", "w_eta", "B_eta", "A_eta"]
sample1 = vEstSdResParBootDist
sample1 = Utilities.drop_bad_un_estimates(vEstSdUnParBootDist)
fig, ax = subplots(3,2)
for i = 1:6
    ax[i].hist(sample1[i,:], range=quantile(sample1[i,:], [0.01, 0.99]), 20, alpha = 0.4)
    ax[i].set_title(parNames[i])
    ylim = ax[2,1].get_ylim()
    # ax[i].vlines(res.vEstSdResPar[i], ylim[1], ylim[2], color = "r" )

end
end





# correctly specified filter

# estimate






# store the sufficient statistics and change statistics in R




