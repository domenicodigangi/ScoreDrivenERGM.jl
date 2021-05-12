#########
#File: c:\Users\digan\Dropbox\Dynamic_Networks\repos\ScoreDrivenExponentialRandomGraphs\src\ScoreDrivenERGM.jl\src\test\test_new_dyn_net_model copy.jl
#Created Date: Monday May 10th 2021
#Author: Domenico Di Gangi,  <digangidomenico@gmail.com>
#-----
#Last Modified: Wednesday May 12th 2021 6:00:47 pm
#Modified By:  Domenico Di Gangi
#-----
#Description: Test SDERGM_pml for different combinations of statistics
#-----
########



using Test
using ScoreDrivenERGM
using StatsBase

import ScoreDrivenERGM:StaticNets,DynNets, Utilities, ErgmRcall


# sample dgp
N = 20
T = 30

ergmTermsString = "edges + mutual + gwesp(decay = 0.25, fixed = TRUE, cutoff=10)"
model = DynNets.SdErgmPml(ergmTermsString, true)
nErgmPar = model.nErgmPar
@test nErgmPar == 3
dgpVals0 = DynNets.static_estimate(model, Int8.(rand(Bool, N, N)))    
@test all(isfinite.(dgpVals0))


@testset " Mappings of static parameters beween restricted and unrestricted spaces
 " begin
    all_perm(xs, n) = vec(map(collect, Iterators.product(ntuple(_ -> xs, n)...)))

    listIndTvPar =  all_perm([true, false], nErgmPar)
    listIndTvPar = listIndTvPar[.!(all.( [.!indTvPar for indTvPar in listIndTvPar]))]


    for indTvPar in listIndTvPar 

        vUnPar, _ = DynNets.starting_point_optim(model, indTvPar, dgpVals0)
        @test all(isfinite.(vUnPar))

        vResPar =  DynNets.restrict_all_par(model, indTvPar, vUnPar)
        @test all(isfinite.(vResPar))

        @test all(isapprox.(DynNets.unrestrict_all_par(model, indTvPar, vResPar), vUnPar, atol = 1e-8))
        @test all(isapprox.(DynNets.restrict_all_par(model, indTvPar, vUnPar), vResPar, atol = 1e-8))

        for i=1:10
            vResParRand = rand(length(vResPar))
            vUnParRand =  DynNets.unrestrict_all_par(model, indTvPar, vResParRand)
            @test all(isfinite.(vUnParRand))
            @test all(isapprox.(DynNets.restrict_all_par(model, indTvPar, vUnParRand), vResParRand, atol = 1e-8))
        end

    end

 end


using Distributed

@testset " Sampling Change stats and estimates " begin
T =100
    dgpParT = hcat(dgpVals0.*ones(nErgmPar, round(Int,T/2)),2 *dgpVals0.* ones(nErgmPar, round(Int, T/2)))
    # test reasonable sampling
    A_T = DynNets.sample_ergm_sequence(model, N, dgpParT, 1) 
    eps = 0.01
    @test  eps < mean(A_T) < (1-eps)  # density
    @test !any(isnan.(A_T)) 
    @test all(isfinite.(A_T)) 

    obsT = [DynNets.stats_from_mat(model, A_T[:,:,t]) for t in 1:T ] 
    @test all([all(isfinite.(obs)) for obs in obsT])

end

ENV["JULIA_DEBUG"] = Main

model.options["integrated"] = true

estSdResPar, conv_flag, UM_mple, ftot_0 = DynNets.estimate(model, N, obsT; indTvPar=model.indTvPar, show_trace = true)


indTvPar = model.indTvPar
vEstSdResParAll = DynNets.array_2_vec_all_par(model, estSdResPar, indTvPar)

vEstSdResPar, vConstPar = DynNets.divide_SD_par_from_const(model, indTvPar,vEstSdResParAll)

fVecT_filt , target_fun_val_T, sVecT_filt = DynNets.score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;ftot_0 = ftot_0, vConstPar=vConstPar)


using PyPlot

# res_est = DynNets.estimate_and_filter(model, N, obsT; show_trace = true)
fig, ax = DynNets.plot_filtered(model, N, fVecT_filt)


est_SS = DynNets.estimate_single_snap_sequence(model, obsT)
fig, ax = DynNets.plot_filtered(model, N, est_SS, ax=ax, lineType = ".", lineColor="r")





# correctly specified filter

# estimate





