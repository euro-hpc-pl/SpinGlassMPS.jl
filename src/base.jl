export is_left_normalized,
    is_right_normalized, MPS, AbstractMPS, verify_bonds, bond_dimension, tensor, proj
abstract type AbstractMPS{T<:Real} end


struct MPS{T<:Real} <: AbstractMPS{T}
    tensors::Vector{Array{T,3}}
end

# consturctors
MPS(::Type{T}, L::Int) where {T} = MPS(Vector{Array{T,3}}(undef, L))
MPS(L::Int) = MPS(Float64, L)
function MPS(states::Vector{Vector{T}}) where {T<:Number}
    state_arrays = [reshape(copy(v), (1, length(v), 1)) for v ∈ states]
    MPS(state_arrays)
end

@inline MPS(A::AbstractArray) = MPS(A, :right)

@inline function MPS(A::AbstractArray, s::Symbol, Dcut::Int = typemax(Int))
    @assert s ∈ (:left, :right)
    if s == :right
        ψ = _right_sweep(A)
        _left_sweep!(ψ, Dcut)
    else
        ψ = _left_sweep(A)
        _right_sweep!(ψ, Dcut)
    end
    ψ
end

@inline Base.setindex!(a::AbstractMPS, A::AbstractArray{<:Real,3}, i::Int) =
    a.tensors[i] = A
@inline bond_dimension(a::AbstractMPS) = maximum(size.(a.tensors, 3))
Base.hash(a::MPS, h::UInt) = hash(a.tensors, h)
@inline Base.:(==)(a::MPS, b::MPS) = a.tensors == b.tensors
@inline Base.:(≈)(a::MPS, b::MPS) = a.tensors ≈ b.tensors
Base.copy(a::MPS) = MPS(copy(a.tensors))
@inline physical_dim(ψ::AbstractMPS, i::Int) = size(ψ[i], 2)
@inline Base.eltype(::AbstractMPS{T}) where {T} = T

@inline Base.getindex(a::AbstractMPS, i) = getindex(a.tensors, i)
@inline Base.iterate(a::AbstractMPS) = iterate(a.tensors)
@inline Base.iterate(a::AbstractMPS, state) = iterate(a.tensors, state)
@inline Base.lastindex(a::AbstractMPS) = lastindex(a.tensors)
@inline Base.length(a::AbstractMPS) = length(a.tensors)
@inline Base.size(a::AbstractMPS) = (length(a.tensors),)
@inline Base.eachindex(a::AbstractMPS) = eachindex(a.tensors)
@inline LinearAlgebra.rank(ψ::AbstractMPS) = Tuple(size(A, 2) for A ∈ ψ)

local_basis(d::Int) = union(-1, 1:d-1)
local_basis(ψ::AbstractMPS, i::Int) = local_basis(physical_dim(ψ, i))

LinearAlgebra.I(ψ::AbstractMPS, i::Int) = I(size(ψ[i], 2))

function verify_bonds(ψ::AbstractMPS)
    L = length(ψ)

    @assert size(ψ[1], 1) == 1 "Incorrect size on the left boundary."
    @assert size(ψ[end], 3) == 1 "Incorrect size on the right boundary."

    for i ∈ 1:L-1
        @assert size(ψ[i], 3) == size(ψ[i+1], 1) "Incorrect link between $i and $(i+1)."
    end
end

is_left_normalized(ψ::MPS) = all(
    I(size(A, 3)) ≈ @tensor(Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y]; order = (α, σ))
    for A ∈ ψ
)

is_right_normalized(ϕ::MPS) = all(
    I(size(B, 1)) ≈ @tensor(Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]); order = (α, σ))
    for B in ϕ
)

function tensor(ψ::MPS, state::Union{Vector,NTuple})
    C = I
    for (A, σ) ∈ zip(ψ, state)
        C *= A[:, idx(σ), :]
    end
    tr(C)
end

function tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)

    for σ ∈ all_states(dims)
        Θ[idx.(σ)...] = tensor(ψ, σ)
    end
    Θ
end


function Base.show(io::IO, ψ::AbstractTensorNetwork)
    L = length(ψ)
    dims = [size(A) for A ∈ ψ]

    println(io, "Matrix product state on $L sites:")
    _show_sizes(io, dims)
    println(io, "   ")
end


function _show_sizes(io::IO, dims::Vector, sep::String = " x ", Lcut::Int = 8)
    L = length(dims)
    if L > Lcut
        for i ∈ 1:Lcut
            print(io, " ", dims[i], sep)
        end
        print(io, " ... × ", dims[end])
    else
        for i ∈ 1:(L-1)
            print(io, dims[i], sep)
        end
        println(io, dims[end])
    end
end

function proj(state, dims::T) where {T<:Union{Vector,NTuple}}
    P = Matrix{Float64}[]
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.0
        push!(P, v * v')
    end
    P
end

struct Solution
    energies::Vector{<:Real}
    states::Vector{Vector{Int}}
    probabilities::Vector{<:Real}
    degeneracy::Vector{Int}
    largest_discarded_probability::Real
end
