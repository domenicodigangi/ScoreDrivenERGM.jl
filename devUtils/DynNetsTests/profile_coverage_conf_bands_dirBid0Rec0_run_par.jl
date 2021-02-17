"""
Simulations to estimate coverage of confidence bands with Blasques and Buccheri's methods 
To Do : 
- profile R interface to see if simulations for PMLE can be faster
- 
"""


#region import and models

using Pkg
Pkg.activate(".") 
Pkg.instantiate() 
using DrWatson


begin
using Pkg
Pkg.activate(".") 
Pkg.instantiate() 
using ScoreDrivenERGM
import ScoreDrivenERGM:StaticNets, DynNets
import ScoreDrivenERGM.DynNets:GasNetModel,GasNetModelDirBin0Rec0, sample_dgp, statsFromMat, array2VecGasPar, unrestrict_all_par, number_ergm_par, estimate_filter_and_conf_bands, conf_bands_coverage_parallel, estimate, plot_filtered_and_conf_bands
using ScoreDrivenERGM.Utilities

model_mle = DynNets.GasNetModelDirBin0Rec0_mle()
model_pmle = DynNets.GasNetModelDirBin0Rec0_pmle()
indTvPar = trues(2)
end
#endregion


using Profile
using ProfileView

dgpSetAR, ~, dgpSetSD = ScoreDrivenERGM.DynNets.list_example_dgp_settings_for_paper(model_mle)

T = 100
N = 100


begin
Profile.clear()
@profile DynNets.sample_dgp_filter_and_estimate_coverage_once(model_pmle, dgpSetSD, T, N)                
ProfileView.view()
end



