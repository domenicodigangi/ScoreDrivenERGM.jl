# test that the GasNetModel interface works for a concrete model<:GasNetModel 

using Test
using ScoreDrivenERGM

import ScoreDrivenERGM:StaticNets,DynNets


model_pmle = DynNets.GasNetModelDirBin0Rec0_pmle()
model_mle = DynNets.GasNetModelDirBin0Rec0_mle()

model = model_mle

# sample dgp
listDgpSettigns = DynNets.list_example_dgp_settings(model_mle)

N = 100
T = 200
parDgpT = DynNets.sample_time_var_par_from_dgp(model_mle, listDgpSettigns.dgpSetSD.type, N, T;  listDgpSettigns.dgpSetSD.opt...)

A_T = DynNets.sample_mats_sequence(model_mle, parDgpT,N)


model = model_pmle

@elapsed obsT = DynNets.seq_of_obs_from_seq_of_mats(model, A_T)

_, vEstSdResPar, fVecT_filt, _, _, conf_bands_coverage, ftot_0 = DynNets.estimate_and_filter(model, N, obsT)






# correctly specified filter

# estimate






# store the sufficient statistics and change statistics in R




