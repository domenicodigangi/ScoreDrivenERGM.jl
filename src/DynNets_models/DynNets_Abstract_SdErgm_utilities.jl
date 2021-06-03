


function plot_filtered(model::SdErgm, N, fVecT_filtIn; lineType = "-", lineColor = "b", parDgpTIn=zeros(2,2), offset = 0, fig = nothing, ax=nothing, gridFlag=true, xval=nothing)

    parDgpT = parDgpTIn[:, 1:end-offset]
    fVecT_filt = fVecT_filtIn[:, 1+offset:end, :, :]
    n_ergm_par= number_ergm_par(model)

    T = size(fVecT_filt)[2]

    if isnothing(fig)
        fig, ax = subplots(n_ergm_par,1)
    end

    if parDgpTIn != zeros(2,2)
        plot_filtered(model, N, parDgpTIn; lineType="-", lineColor="k", fig=fig, ax=ax, gridFlag=false, xval=xval)
    end
    for p in 1:n_ergm_par
        isnothing(xval) ? x = collect(1:T) : x = xval
        bottom = minimum(fVecT_filt[p,:])
        top = maximum(fVecT_filt[p,:])
       
        delta = top-bottom
        margin = 0.5*delta
        ax[p].plot(x, fVecT_filt[p,:], "$(lineType)$(lineColor)", alpha =0.5)
        # ax[p].set_ylim([bottom - margin, top + margin])
        gridFlag ? ax[p].grid() : ()
        
    end
 
    titleString = "$(name(model)), N = $(Utilities.get_N_t(N, 1)), T=$(T + offset)"

    ax[1].set_title(titleString)

    return fig, ax
end


function plot_filtered_and_conf_bands(model::SdErgm, N, fVecT_filtIn, confBands1In; lineType = "-", confBands2In =zeros(2,2,2,2), parDgpTIn=zeros(2,2), nameConfBand1="1", nameConfBand2="2", offset = 0, indBand = 1, xval=nothing)

    parDgpT = parDgpTIn[:, 1:end-offset]
    fVecT_filt = fVecT_filtIn[:, 1+offset:end, :, :]
    confBands1 = confBands1In[:, 1+offset:end, :, :]
    confBands2 = confBands2In[:, 1+offset:end, :, :]

    T = size(fVecT_filt)[2]

    nBands = size(confBands1)[3]

    fig, ax = plot_filtered(model::SdErgm, N, fVecT_filtIn; lineType = lineType, parDgpTIn=parDgpTIn, offset = offset, xval=xval)

    for p in 1:number_ergm_par(model)
        isnothing(xval) ? x = collect(1:T) : x = xval
           for b = indBand
           @show indBand
            if sum(confBands1) !=0
                ax[p].fill_between(x, confBands1[p, :, b,1], y2 =confBands1[p,:,b, 2],color =(0.9, 0.2 , 0.2, 0.1), alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
            end
            if sum(confBands2) != 0
           @show indBand
                ax[p].plot(x, confBands2[p, :, b, 1], "-g", alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
                ax[p].plot(x, confBands2[p,:, b, 2], "-g", alpha = 0.2*b/nBands  )#, color='b', alpha=.1)
            end
        end
   
        
    end
    
    titleString = "$(name(model)), N = $N, T=$(T + offset), \n "
    titleString = "$(name(model)), \n "

    if parDgpTIn != zeros(2,2)
        if sum(confBands1) !=0
                cov1 = round(mean(conf_bands_coverage(parDgpTIn, confBands1In; offset=offset)[:,:,indBand]), digits=2)
                titleString = titleString * "$nameConfBand1 = $cov1"
        else
            cov1 = 0
        end

        if sum(confBands2) !=0
            cov2 = round(mean(conf_bands_coverage(parDgpTIn, confBands2In; offset=offset)[:,:,indBand]), digits=2)
            titleString = titleString * " $nameConfBand2 = $cov2"
        end
    end

    ax[1].set_title(titleString)

    return fig, ax
end


function estimate_and_filter(model::SdErgm, N, obsT; indTvPar = model.indTvPar, show_trace = false)

    T = length(obsT)   
    
    estSdResPar, conv_flag, UM_mple, ftot_0 = estimate(model, N, obsT; indTvPar=indTvPar, indTargPar=falses(length(indTvPar)), show_trace = show_trace)


    vEstSdResParAll = array_2_vec_all_par(model, estSdResPar, indTvPar)

    vEstSdResPar, vConstPar = divide_SD_par_from_const(model, vEstSdResParAll)

    fVecT_filt , target_fun_val_T, sVecT_filt = score_driven_filter(model, N, obsT,  vEstSdResPar, indTvPar;ftot_0 = ftot_0, vConstPar=vConstPar)

    return (; obsT, vEstSdResParAll, fVecT_filt, target_fun_val_T, sVecT_filt, conv_flag, ftot_0)
end
    

function simulate_and_estimate_parallel(model::SdErgm, dgpSettings, T, N, nSample , singleSnap = false)

    counter = SharedArray(ones(1))
    res = @sync @distributed vcat for k=1:nSample
        
        Logging.@info("Estimating N = $N , T=$T iter n $(counter[1]), $(DynNets.name(model)), $(dgpSettings)")

        parDgpT = DynNets.sample_time_var_par_from_dgp(reference_model(model), dgpSettings.type, N, T;  dgpSettings.opt..., maxAttempts=100000)

        A_T_dgp = StaticNets.sample_ergm_sequence(reference_model(model).staticModel, N, parDgpT, 1)[:,:,:,1]

        obsT = seq_of_obs_from_seq_of_mats(model, A_T_dgp)

        ~, vEstSdResPar, fVecT_filt, ~, ~, conv_flag, ftot_0 = estimate_and_filter(model, N, obsT; indTvPar = model.indTvPar)
        
        if singleSnap
            fVecT_filt_SS = zeros(DynNets.number_ergm_par(model), T)
        else
            fVecT_filt_SS =  DynNets.estimate_single_snap_sequence(model, A_T_dgp)
        end
        counter[1] += 1

        (;obsT, parDgpT, vEstSdResPar, fVecT_filt, conv_flag, ftot_0, fVecT_filt_SS)
 
    end


    allObsT = [r.obsT for r in res]
    allParDgpT = reduce(((a,b) -> cat(a,b, dims=3)), [r.parDgpT for r in res])
    allvEstSdResPar = reduce(((a,b) -> cat(a,b, dims=2)), [r.vEstSdResPar for r in res])
    allfVecT_filt = reduce(((a,b) -> cat(a,b, dims=3)), [r.fVecT_filt for r in res])
    allfVecT_filt_SS = reduce(((a,b) -> cat(a,b, dims=3)), [r.fVecT_filt_SS for r in res])
    allConvFlag = [r.conv_flag for r in res]
    allftot_0 = reduce(((a,b) -> cat(a,b, dims=2)), [r.ftot_0 for r in res])

    Logging.@info("The fraction of estimates that resulted in errors is $(mean(.!allConvFlag)) ")

    return allObsT, allvEstSdResPar, allfVecT_filt, allParDgpT, allConvFlag, allftot_0, allfVecT_filt_SS
end

"""
Given the SD filter's parameter, compute the confidence bands and check their coverages
"""
function average_coverages(res, m; limitSample=nothing, quantilesVals = [ [0.975, 0.025]], winsorProp=0)
    
    N = res.N
    T = res.T
    nErgmPar = DynNets.number_ergm_par(res.model)
    nBands = length(quantilesVals)

    if isnothing(limitSample)
        nSample = res.nSample
    elseif res.nSample >= limitSample 
        nSample = limitSample
    else 
        error("sample size?")
    end     

    avgCover = SharedArray(zeros(nErgmPar, 2, nSample))
    allmvSDUnParEstCovWhite = SharedArray(zeros(3*nErgmPar, 3*nErgmPar,nSample))
    allConfBandsPar = SharedArray(zeros(nErgmPar, T, nBands, 2, nSample))
    allConfBandsFiltPar = SharedArray(zeros(nErgmPar, T, nBands, 2, nSample))
    constInds = SharedArray{Bool}((2, nSample))
    errInds = SharedArray{Bool}((2, nSample))

    errInds .= true

    count = SharedArray(ones(1))
    
    mvSDUnParEstCov = zeros(3,3)

    Threads.@threads for n=1:nSample
        
        Logging.@info("Estimating Conf Bands  N = $N , T=$T, $(DynNets.name(res.model)), $(res.dgpSettings) iter n $(count[1]), ")
        count[1] += 1

        vEstSdUnPar = unrestrict_all_par(res.model, res.allvEstSdResPar[:,n])
        errFlag = true
        try
            ~, allConfBandsFiltPar[:,:,:,:,n], allConfBandsPar[:,:,:,:,n], errFlag, allmvSDUnParEstCovWhite[:,:,n], distribFilteredSD = DynNets.conf_bands_given_SD_estimates(res.model, N, res.allObsT[n], vEstSdUnPar, res.allftot_0[:,n], quantilesVals;  parUncMethod = m, mvSDUnParEstCov=mvSDUnParEstCov, winsorProp=winsorProp )
        catch
        end

        coverFiltParUnc = DynNets.conf_bands_coverage(res.allParDgpT[:,:,n],   allConfBandsFiltPar[:,:,:,:,n])

        coverParUnc = DynNets.conf_bands_coverage(res.allParDgpT[:,:,n],   allConfBandsPar[:,:,:,:,n])

        constInds[:, n]  .= any(res.allvEstSdResPar[3:3:end,n].<=0.02, dims=1)
        for indPar in 1:2
            avgCover[indPar, 1, n] =  mean(coverFiltParUnc[indPar,:,1]) 

            avgCover[indPar, 2, n] =  mean(coverParUnc[indPar,:,1]) 
        end
            errInds[:, n]  .= errFlag

        # catch
        #     Logging.@warn("Error in estimating confidence bands $T, $N, $n")
        # end

    end
    return sdata(avgCover), sdata(constInds), sdata(errInds), sdata(allConfBandsFiltPar), sdata(allConfBandsPar), sdata(allmvSDUnParEstCovWhite)
end
export average_coverages





function list_example_dgp_settings(model::SdErgm; out="tuple", minVals = [-3.0, 0], maxVals = [-2.4, 1])

    
    dgpSetARlowlow = (type = "AR", opt = (B =[0.98], sigma = [0.005]))
    
    dgpSetARlowlowInt = (type = "AR", opt = (B =[1], sigma = [0.005]))
    
    dgpSetARlow = (type = "AR", opt = (B =[0.98], sigma = [0.01]))

    dgpSetARlowInt = (type = "AR", opt = (B =[1], sigma = [0.01]))

    dgpSetARmed = (type = "AR", opt = (B =[0.98], sigma = [0.05]))
    
    dgpSetARhigh = (type = "AR", opt = (B =[0.98], sigma = [0.1]))

    dgpSetSIN = (type = "SIN", opt = ( nCycles=[1.5]))

    dgpSetSDlow = (type = "SD", opt = (B =[0.98], A = [0.01]))

    dgpSetSD = (type = "SD", opt = (B =[0.98], A = [0.3]))
    
    dgpSetSDhigh = (type = "SD", opt = (B =[0.98], A = [3]))

    tupleList =  (; dgpSetARlowlow, dgpSetARlowlowInt, dgpSetARlow, dgpSetARlowInt, dgpSetARmed, dgpSetARhigh, dgpSetSIN, dgpSetSDlow, dgpSetSD, dgpSetSDhigh)
    
    if out == "tuple"
        return tupleList
    elseif out == "dict"
        d = Dict() 
        [d[string(dgp)[7:end]] = getfield(tupleList,dgp) for dgp in fieldnames(typeof(tupleList))]
        return d
    end
end
export list_example_dgp_settings