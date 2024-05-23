
using SpinGlassMPS
using SpinGlassNetworks
using MetaGraphs
using LinearAlgebra

T = Float64
L = 20

FILE = "001.txt"
INSTANCE = "$(@__DIR__)/instances/square/$(L)/" * FILE

@time ig = ising_graph(INSTANCE)

max_states = 5 * 10^2
β = T(10)
dβ = T(β/10.0)

Dcut = 64 # 32
var_ϵ = 1E-8
max_sweeps = 4
schedule = fill(dβ, Int(ceil(β/dβ)))

@time igp = prune(ig)
@time ψ = MPS(igp, Dcut, var_ϵ, max_sweeps, schedule)

@time states, lprob, _ = solve(ψ, max_states)
en = energy(states[1:1], igp)
@show en
