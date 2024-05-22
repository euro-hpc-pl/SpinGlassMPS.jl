using Test
using SpinGlassMPS
using SpinGlassNetworks
using MetaGraphs
using LinearAlgebra

my_tests = [
    "MPS_search.jl",
    "ising_MPS.jl",
]

for my_test in my_tests
    include(my_test)
end