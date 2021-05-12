# SDERGM
julia package for Score Driven Exponential Random Graphs Models, as proposed in the paper Score-Driven Exponential Random Graphs: A New Class of Time-Varying Parameter Models for Dynamical Networks https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3394593

### ERGM - Exponential Random Graph Models
A statistical model for graphs can be specified providing the probability distribution over the set of possible graphs, i.e. all possible adjacency matrices. If the distribution belongs to the exponential family, than the model is named ERGM.
### Score Driven (SD) Models, aka Generalized Autoregressive Score Models (GAS), aka Dynamically Conditional Score Models (DCS)
- GAS paper : Creal,  D.,  S.  J.  Koopman,  and  A.  Lucas  (2013).   Generalized  autoregressive  score  models  withapplications.Journal of Applied Econometrics  28(5), 777–795.
- DCS paper : Harvey, A. C. (2013).Dynamic  Models  for  Volatility  and  Heavy  Tails:  With  Applications  to  Fi-nancial  and  Economic  Time  Series.   Econometric  Society  Monographs.  Cambridge  UniversityPress.
- Interesting reference on Score Driven Smoothers and the relation between SD and State Space models https://arxiv.org/abs/1803.04874


## SDERGM depends on R - ergm
Some methods and scripts related with global stastistics (not the fitness models) require Julia to call multiple R Libraries. For this interface to work, a working installation of R and the ergm package is required. We aknowledge the importance of this library for our analysis. For further information, please refer to  
Statnet Development Team
(Pavel N. Krivitsky, Mark S. Handcock, David R. Hunter, Carter T. Butts, Chad Klumb, Steven M. Goodreau, and Martina Morris) (2003-2020).
statnet: Software tools for the Statistical Modeling of Network Data. 
URL http://statnet.org

## SDERGM is a Work in Progress
- This repository is an attempt at packaging the code from a research project. It is not a finished package by any means. The dependency on R-ergm, and the total lack of testing,  prevent the release 
  
- the fitness model has not been updated to the latest version of the codebase, hence is not currently working. For a working fitness model checkout commits from 2019. I would be glad to update it upon request. Also a python version of only the fitness model might be released upon request.
