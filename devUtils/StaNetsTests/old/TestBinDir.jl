N = 10
fooErgmDirBin1
tmpDegsIO = [ Vector(1:N) Vector(1:N)]

Nnodes = N; deltaN = 3
tmpdegs  =   DegSeq2graphDegSeq(fooErgmDirBin1,tmpDegsIO)
tmpMod = ErgmDirBin1(tmpdegs)
out,it,estMod = estimate(tmpMod)

fit,degs = linSpacedFitnesses(fooErgmDirBin1,N)
par,it,estMod = estimate(ErgmDirBin1(degs))
parI = par[:,1];parO = par[:,2];parMat = parI.*parO'
expMat = putZeroDiag(parMat./(1 + parMat));expDeg = [squeeze(sum(expMat,2),2) squeeze(sum(expMat,1),1)]
relErrEst = maximum(abs.( (( expDeg - degs )./degs )[degs.!=0]))

##
sam = sampl(estMod,10)

meanMatSam = squeeze(mean(sam,3),3)
meanDeg = [squeeze(sum(meanMatSam,2),2) squeeze(sum(meanMatSam,1),1)]
relErrSam = maximum(abs.( (( meanDeg - degs )./degs )[degs.!=0]))
