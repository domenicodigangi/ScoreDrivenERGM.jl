# test that the GasNetModel interface works for a concrete model<:GasNetModel 

using Test
using ScoreDrivenERGM


model = ScoreDrivenERGM.DynNets.GasNetModelDirBin0Rec0_mle()

# sample dgp
listDgpSettigns = ScoreDrivenERGM.DynNets.list_example_dgp_settings(model)

N = 100
T = 300
parDgpT = ScoreDrivenERGM.DynNets.sample_time_var_par_from_dgp(model, listDgpSettigns.dgpSetSD.type, N, T;  listDgpSettigns.dgpSetSD.opt...)

A_T = ScoreDrivenERGM.DynNets.sample_mats_sequence(model, parDgpT,N)

obsT = seq_of_obs_from_seq_of_mats(model, A_T)

~ vEstSdResPar, fVecT_filt, ~, ~, conf_bands_coverage, ftot_0 = DynNets.estimate_and_filter(model, N, obsT)

# correctly specified filter

# estimate




