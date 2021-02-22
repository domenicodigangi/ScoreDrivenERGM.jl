using Pkg
Pkg.activate(".") 
Pkg.instantiate() 

using ScoreDrivenERGM


using Libdl



ergmC = Libdl.dlopen("C:\\Users\\digan\\Documents\\R\\win-library\\4.0\\ergm\\libs\\x64\\ergm")

Libdl.dlsym(ergmC,:MPLE_wrapper)

