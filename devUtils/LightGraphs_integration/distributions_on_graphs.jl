using Pkg
Pkg.activate(".") 
Pkg.instantiate() 

using ScoreDrivenERGM
using LightGraphs
using Distributions


# define test matrix
N = 100
A = rand(Bool,N,N)

g = SimpleDiGraph(A)



