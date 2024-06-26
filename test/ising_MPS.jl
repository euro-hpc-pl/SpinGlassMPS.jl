@testset "MPS-based search produces correct results for small instances" begin
    for L ∈ [2, 3]
        N = L^2
        instance = "$(@__DIR__)/instances/basic/$(N)_001.txt"
        ig = ising_graph(instance)
        rank = Tuple(values(get_prop(ig, :rank)))

        β = 2.0
        dβ = β / 5.0
        schedule = fill(dβ, Int(ceil(β / dβ)))

        ϵ = 1E-8
        Dcut = 32
        var_ϵ = 1E-8
        max_sweeps = 4
        states = all_states(rank)

        @testset "Generating MPS for L=$N spins" begin
            ϱ = gibbs_tensor(ig, β)
            @testset "Sqrt of the Gibbs state (aka state tensor)" begin
                ψ = ones(rank...)
                for σ ∈ states
                    for i ∈ 1:N
                        h = get_prop(ig, i, :h)
                        nbrs = unique_neighbors(ig, i)
                        ψ[idx.(σ)...] *= exp(-0.5 * β * h * σ[i])
                        for j ∈ nbrs
                            J = get_prop(ig, i, j, :J)
                            ψ[idx.(σ)...] *= exp(-0.5 * β * σ[i] * J * σ[j])
                        end
                    end
                end
                ρ = abs.(ψ) .^ 2
                rψ = MPS(ψ)
                lψ = MPS(ψ, :left)

                @testset "produces correct Gibbs state" begin
                    @test ρ / sum(ρ) ≈ ϱ
                end
                @testset "MPS from the tensor" begin
                    @testset "can be right normalized" begin
                        @test dot(rψ, rψ) ≈ 1
                        @test_nowarn is_right_normalized(rψ)
                    end
                    @testset "can be left normalized" begin
                        @test dot(lψ, lψ) ≈ 1
                        @test_nowarn is_left_normalized(lψ)
                    end
                    @testset "both forms are the same (up to a phase factor)" begin
                        vlψ = vec(tensor(lψ))
                        vrψ = vec(tensor(rψ))
                        vψ = vec(ψ)
                        vψ /= norm(vψ)
                        @test abs(1 - abs(dot(vlψ, vrψ))) < ϵ
                        @test abs(1 - abs(dot(vlψ, vψ))) < ϵ
                    end
                end
                @testset "MPS from gates" begin
                    Gψ = MPS(ig, Dcut, var_ϵ, max_sweeps, schedule)
                    @testset "is built correctly" begin
                        @test abs(1 - abs(dot(Gψ, rψ))) < ϵ
                    end
                    @testset "is normalized" begin
                        @test dot(Gψ, Gψ) ≈ 1
                        @test_nowarn is_right_normalized(Gψ)
                    end
                    @testset "has correct links and non-trivial bond dimension" begin
                        @test bond_dimension(Gψ) > 1
                        @test_nowarn verify_bonds(Gψ)
                    end
                end
                @testset "Exact probabilities are calculated correctely" begin
                    for σ ∈ states
                        p, r = dot(rψ, σ), dot(rψ, proj(σ, rank), rψ)
                        @test p ≈ r
                        @test ϱ[idx.(σ)...] ≈ p
                    end
                end
                @testset "Results from MPS-based search agree with brute-force" begin
                    for max_states ∈ [1, N, 2 * N, 3 * N, N^2 - 3, N^2 - 2, N^2] # problem for N^2-1
                        states, _ = solve(rψ, max_states)
                        en = energy(states, ig)
                        sp = brute_force(ig, num_states = max_states)
                        @test sort(en) ≈ sp.energies
                    end
                end
            end
        end
    end
end
