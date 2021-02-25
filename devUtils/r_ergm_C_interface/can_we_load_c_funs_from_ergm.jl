using Pkg
Pkg.activate(".") 
Pkg.instantiate() 

using ScoreDrivenERGM
using LightGraphs


using Libdl
using RCall
R""" library("ergm") """

ergmC = Libdl.dlopen("C:\\Users\\digan\\Documents\\R\\win-library\\4.0\\ergm\\libs\\x64\\ergm")



# define test matrix
N = 100
A = rand(Bool,N,N)

g = SimpleDiGraph(A)

edgeList = collect(edges(g))
tails = getproperty.(edgeList, :src)
heads = getproperty.(edgeList, :dst)
dnedges = ne(g) 
dflag = Int(is_directed(g))


# try to understand what is wl input
@rput A
R" net <- network(A)" 
R"to_ergm_Cdouble(net)" 

# store the sufficient statistics and change statistics in R
R"""
    net <- network(A)
    chStat_t <- ergmMPLE(formula_ergm)
    changeStats_t_R <- cbind(chStat_t$response, chStat_t$predictor,     chStat_t$weights)
        """

# create julia variables (Refs ) that can be passed to MPLE_wrapper fun



# test MPLE_wrapper function
tailsIn = Ref(tails)
headsIn = Ref(heads)
ccall(:GetRNGstate, Int32)

# MPLE_wrapper(int *tails, int *heads, int *dnedges,
# 		  double *wl,
# 		  int *dn, int *dflag, int *bipartite, int *nterms, 
# 		  char **funnames, char **sonames, double *inputs,  
# 		  int *responsevec, double *covmat,
# 		  int *weightsvector,
# 		  int *maxDyads, int *maxDyadTypes)

# ABANDONING THE IDEA BECAUSE IT WOULD REQUIRE TOO MUCH WORK FOR THE MOMENT