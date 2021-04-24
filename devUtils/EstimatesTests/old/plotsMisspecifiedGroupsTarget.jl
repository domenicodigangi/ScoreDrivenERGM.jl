
using JLD2, Distributions, Plots, StatPlots, AverageShiftedHistograms, DynNets
 plotly()



N_est = 900
Tvals =  [50 500 5000]
N_ind = length(Tvals)[1]


N = 10 ;
GBAdgp =1;
GBAest =1;   useStartVal = false

useStartVal? startValFlag = "T" : startValFlag = "F";# false #false #
# save
save_fold = "./data/estimatesTest/"
    file_name = "EstTest_N_$(N)_T_$(Tvals)_$(N_est)_GBAdgp_$(GBAdgp)_GBAest_$(GBAest)_$(useStartVal)_target.jld"
    save_path = save_fold*file_name#
estDataT = @load(save_path,groupsIndsEst,ObsAllT,ParTvDgpAllT,simPar_1,simPar_2,simPar_3,
        estPar_1,estPar_2,estPar_3,est_conv_flag,est_times)


##

RavgMSE(A,B) = sqrt.(mean((A .- B).^2,1))
AvgBias(A,B) = mean((A .- B),1)
tmpModelEst = DynNets.SdErgm1(zeros(10,10),[ones(N),ones(GBAest),ones(GBAest)],groupsIndsEst)
tmpModelDgp = DynNets.SdErgm1(zeros(10,10),[ones(N),ones(GBAdgp),ones(GBAdgp)],groupsIndsEst)

rmseTotGas = zeros(N_est,N,N_ind)
biasTotGas = zeros(N_est,N,N_ind)
rmseTotSnap = zeros(N_est,N,N_ind)
biasTotSnap = zeros(N_est,N,N_ind)
dgpTvParT = Array{Array{Float64,3},1}(N_ind)
filTvParT = Array{Array{Float64,3},1}(N_ind)
ParSnapT =  Array{Array{Float64,3},1}(N_ind)

#Remove Nans
if sum(isnan(estPar_3))/(prod(size(estPar_2)))<0.01
    println("NaN Fraction = $(sum(isnan(estPar_3))/(prod(size(estPar_2))))")
    for indT = 1:length(Tvals)
        #kind of bootstrap
        indNan  = find(isnan , estPar_3[indT,:,1])
        indReal = find(isfinite , estPar_3[indT,:,1])
        randInd = rand(indReal,length(indNan))
        estPar_1[indT,indNan,:] = estPar_1[indT,randInd,:]
        estPar_2[indT,indNan,1] = estPar_2[indT,randInd,1]
        estPar_3[indT,indNan,1] = estPar_3[indT,randInd,1]
    end
else
    #too many nans
    error()
end
# Compute MSE Bias and Snap estimates
@time for indT = 1:N_ind
#    indT=1
            T =Tvals[indT]
            tmpParDgp = zeros(T,N,N_est)
            tmpParFil = zeros(T,N,N_est)
            tmpParSnap = zeros(T,N,N_est)
        for n=1:N_est
#    n=100
    obsT = ObsAllT[indT][:,:,n]
    vecParEst =  [estPar_1[indT,n,:]; estPar_2[indT,n];  estPar_3[indT,n]]
    vecParDgp =  [simPar_1[indT,n,:]; simPar_2[indT,n];  simPar_3[indT,n]]

    GroupIndsEst  = groupsIndsEst ;
    tmpParFil[:,:,n] =   DynNets.score_driven_filter_or_dgpAndLikeliood(tmpModelEst,vecParEst,
                                                degsT = obsT,
                                                groupsInds = GroupIndsEst )[1]

    fitSnap = DynNets.estSnapSeq(DynNets.SdErgm1(obsT)) # sequence of snapshots estimate
    # remove the infinites
    for i=1:5
        infInd = find(.!isfinite.(fitSnap))

        replInd = infInd[infInd .> 2]
        fitSnap[replInd] = fitSnap[replInd.-1]
        infInd = find(.!isfinite.(fitSnap))
        replInd = infInd[infInd .< (Int(N*T) -2) ]
        fitSnap[replInd] = fitSnap[replInd.+1]
    end
    tmpParSnap[:,:,n] = fitSnap


    GroupIndsDgp =  groupsIndsEst
    tmpParDgp[:,:,n] = DynNets.score_driven_filter_or_dgpAndLikeliood(tmpModelDgp,vecParDgp,
                                                degsT = obsT,
                                                groupsInds = GroupIndsDgp )[1]


            rmseTotGas[n,:,indT] = RavgMSE(tmpParFil[:,:,n],tmpParDgp[:,:,n] )
            biasTotGas[n,:,indT] = AvgBias(tmpParFil[:,:,n],tmpParDgp[:,:,n] )
            rmseTotSnap[n,:,indT] = RavgMSE(fitSnap,tmpParDgp[:,:,n] )
            biasTotSnap[n,:,indT] = AvgBias(fitSnap,tmpParDgp[:,:,n] )
        end
    dgpTvParT[indT]  = tmpParDgp
    filTvParT[indT]  = tmpParFil
    ParSnapT[indT] = tmpParSnap
end

## Example single realization plots
Nlines=2
indPl = round(Int,linspace(1,N,Nlines))
C(g::ColorGradient) = RGB[g[z] for z=linspace(0,1,Nlines)]
lineColors = C(cgrad(:rainbow))

indT = 3
n=1
T = Tvals[indT]

fit_T = dgpTvParT[indT][:,:,n]
fil_fit_T =   filTvParT[indT][:,:,n]
fitSnap =   ParSnapT[indT][:,:,n]
 p=plot()
for i = 1:Nlines p = plot!(fit_T[:,indPl[i]],  color = lineColors[i]  ) end #lineColors
for i = 1:Nlines p = plot!(fil_fit_T[:,indPl[i]],linestyle=:dash,  color = lineColors[i]  ) end #lineColors
for i = 1:Nlines p= plot!(fitSnap[:,indPl[i]],linestyle=:dot,markershape=:circle,
    markersize=1,linewidth = 0, markerstrokecolor=  lineColors[i], color = lineColors[i]  ) end

p =plot(p,titlefont = font(16),xlims = (0,T) ,titleloc=:center,legend=:none,
    title = "Gas DGP,N $(N) T = $(T) B = $(simPar_2[indT,n,1]), A = $(simPar_3[indT,n,1])",size=(1350,600))


## Box Plots
yLim = (0,1)
    pRT1 = boxplot( rmseTotGas[:,:,1]  , ylims = yLim  , legend = false, title = "T = $(Tvals[1]), N = $(N)")
    pRT2 = boxplot( rmseTotGas[:,:,2]  , ylims = yLim  , legend = false, title = "T = $(Tvals[2])")
     pRT3 = boxplot( rmseTotGas[:,:,3]  , ylims = yLim , legend = false, title = "T = $(Tvals[3])")
     yLim = (-0.8,1.1)
    pBT1 = boxplot( biasTotGas[:,:,1] , ylims = yLim , legend = false, title = "T = $(Tvals[1])")
    pBT2 = boxplot( biasTotGas[:,:,2] , ylims = yLim , legend = false, title = "T = $(Tvals[2])")
    pBT3 = boxplot( biasTotGas[:,:,3] , ylims = yLim ,legend = false, title = "T = $(Tvals[3])")
    ptot = plot(pRT1,pRT2,pRT3,pBT1,pBT2,pBT3,titlefont = font(12) ,titleloc=:center,size=(1350,600) )
#
# yLim = (0,1.5)
#     pRT1 = boxplot( rmseTotSnap[:,:,1]  , ylims = yLim  , legend = false, title = "T = $(Tvals[1]), N = $(N), GWdgp =  $(GWdgp), Single Snapshot ")
#     pRT2 = boxplot( rmseTotSnap[:,:,2]  , ylims = yLim  , legend = false, title = "T = $(Tvals[2])")
#      pRT3 = boxplot( rmseTotSnap[:,:,3]  , ylims = yLim , legend = false, title = "T = $(Tvals[3])")
#      yLim = (-1.4,0.8)
#     pBT1 = boxplot( biasTotSnap[:,:,1] , ylims = yLim , legend = false, title = "T = $(Tvals[1])")
#     pBT2 = boxplot( biasTotSnap[:,:,2] , ylims = yLim , legend = false, title = "T = $(Tvals[2])")
#     pBT3 = boxplot( biasTotSnap[:,:,3] , ylims = yLim ,legend = false, title = "T = $(Tvals[3])")
#     ptot = plot(pRT1,pRT2,pRT3,pBT1,pBT2,pBT3,titlefont = font(12) ,titleloc=:center,size=(1350,600) )

#    title = ["($i)" for j = 1:1, i=1:11], titleloc = :right, titlefont = font(8)

## histograms of static parameters

function Bias(A::Array{<:Real,2},B::Array{<:Real,2})
    return  Bias =    (A.-B)./std(A .-B)
end
Bias(A::SharedArray{<:Real,2},B::SharedArray{<:Real,2}) = BiasRel( convert(Array{Real},A),convert(Array{Real},B))

for ind = 1:length(Tvals)

#ind=3
errA = Bias(simPar_3[ind,:,:],estPar_3[ind,:,:])
errB = Bias(simPar_2[ind,:,:],estPar_2[ind,:,:])
errW =  (simPar_1[ind,:,:].- estPar_1[ind,:,:])./(repmat(std(estPar_1[ind,:,:],1),N_est,1))
plot_tot =Vector{Plots.Plot{Plots.PlotlyBackend}}
rng_space = -5:.1:5

err= errW
tmpAsh = ash(err;rng =rng_space )
pW1 = plot(tmpAsh,
    title ="  T= $(Tvals[ind]),
    W    , mean(std) =  $(round(mean(std(err,1)),3)) ")
plot!(Normal(0,1),linewidth = 5)

err= errB
tmpAsh = ash(err;rng = rng_space)
pB1 = plot(tmpAsh,
title =    " $(N_est) Est   B   mu =  $(round(mean(err[:]),3)) ")
plot!(Normal(0,1),linewidth = 5)

err= errA

tmpAsh = ash(err;rng = rng_space)
pA1 = plot(tmpAsh,
title = startValFlag * "    $(GBAest) / $(N) A  mu =  $(round(mean(err[:]),3)), std =  $(round(std(err[:]),3)) ")
plot!(Normal(0,1),linewidth = 5)

exp = "pl_T_$(ind)=  plot(pW1,pB1,pA1,layout=(3,1) )"
eval(parse(exp))


end
std(err[:])

exp2 = " " ; for i = 1:ind exp2 *=  "pl_T_$(i) , " end

exp3 = "plot( " * exp2 * " layout = (1,ind),size=(1350,600),legend =:none)"
eval(parse(exp3))
#plot(pl_T_1 , pl_T_2,pl_T_3 , pl_T_3, layout = (1,4),size=(1350,600),legend =:none)

##

plot(errW)




##
