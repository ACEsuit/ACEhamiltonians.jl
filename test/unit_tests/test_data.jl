using ACEhamiltonians, Test, LinearAlgebra
using JuLIP: Atoms
using StaticArrays: SVector
using ACE: CylindricalBondEnvelope

using ACEhamiltonians.MatrixManipulation:
    _blk_starts, _sblk_starts, _get_blocks!,
    _set_blocks!

using ACEhamiltonians.States:
    _guard_position, _locate_minimum_image, _locate_target_image


@testset "Matrix Manipulation" begin
    @testset "Constructors" begin
        @testset "atomic_blk_idxs" begin
            for z_s in ([1, 6, 1, 6], Atoms(;Z=[1, 6, 1, 6]))
                @test atomic_blk_idxs(1, 1, z_s) == [1 1 3 3; 1 3 1 3]
                @test atomic_blk_idxs(1, 6, z_s) == [1 1 3 3; 2 4 2 4]
                @test atomic_blk_idxs(6, 1, z_s) == [2 2 4 4; 1 3 1 3]
                @test atomic_blk_idxs(6, 6, z_s) == [2 2 4 4; 2 4 2 4]

                @test atomic_blk_idxs(1, 6, z_s; order_invariant=true) == [1 1 3 3 2 2 4 4; 2 4 2 4 1 3 1 3]
                @test atomic_blk_idxs(6, 1, z_s; order_invariant=true) == [2 2 4 4 1 1 3 3; 1 3 1 3 2 4 2 4]
            end
        end

        @testset "repeat_atomic_blk_idxs" begin
            blk_idxs_a = [
                1 2 3
                4 5 6]
            blk_idxs_b = [
                1 2 3 1 2 3 1 2 3
                4 5 6 4 5 6 4 5 6
                1 1 1 2 2 2 3 3 3]
            @test repeat_atomic_blk_idxs(blk_idxs_a, 3) == blk_idxs_b

            @test_throws AssertionError repeat_atomic_blk_idxs(blk_idxs_b, 3)
    
        end
    end

    @testset "Ancillary" begin

        @testset "_blk_starts" begin
            basis_def = Dict(1=>[0], 6=>[0,1], 7=>[0, 0, 1, 1, 2])
            atoms = Atoms(;Z=[1, 6, 7, 6])
            starts = [1, 2, 6, 19]

            # Check general functionality
            sub_blocks = [
                [1 1; 2 4], [2 4; 1 1], [1; 3;;], [3; 1;;], [2 2 4 4; 2 4 2 4],
                [2 4; 3 3], [3 3; 2 4]]

            for sblks in sub_blocks
                @test _blk_starts(sblks, atoms, basis_def) == starts[sblks]
            end

            # Ensure that cell indices are preserved and their presence does not effect the results
            @test _blk_starts([2 4; 3 3; 11 22], atoms, basis_def)[1:2, :] == _blk_starts([2 4; 3 3], atoms, basis_def)
            @test _blk_starts([2 4; 3 3; 11 12], atoms, basis_def)[3, :] == [11, 12]
            
        end

        @testset "_sblk_starts" begin
            basis_def = Dict(1=>[0], 6=>[0, 1], 7=>[0, 0, 1, 1, 2])
            starts = Dict(1=>[1], 6=>[1, 2], 7=>[1, 2, 3, 6, 9])
            z_s = collect(keys(basis_def))
            for z_1=z_s, z_2=z_s
                for s_i=1:length(basis_def[z_1]), s_j=length(basis_def[z_2])
                    @test _sblk_starts(z_1, z_2, s_i, s_j, basis_def) == [starts[z_1][s_i], starts[z_2][s_j]]
                end
            end
        end
    end

    @testset "Filters" begin
        @testset "filter_on_site_idxs" begin
            # Without cell index row
            @test filter_on_site_idxs([
                1 1 1 2 2 2 3 3 3;
                1 2 3 1 2 3 1 2 3
                ]) == [
                1 2 3;
                1 2 3
                 ]

            # With cell index row
            @test filter_on_site_idxs([
                1 1 2 2 1 1 2 2;
                1 2 1 2 1 2 1 2;
                1 1 1 1 2 2 2 2
                ]) == [
                1 2;
                1 2;
                1 1
                ]
            
        end
        
        @testset "filter_off_site_idxs" begin
            # Without cell index row
            @test filter_off_site_idxs([
                1 1 1 2 2 2 3 3 3;
                1 2 3 1 2 3 1 2 3
                ]) == [
                1 1 2 2 3 3;
                2 3 1 3 1 2
                ]

            # With cell index row
            @test filter_off_site_idxs([
                1 1 2 2 1 1 2 2;
                1 2 1 2 1 2 1 2;
                1 1 1 1 2 2 2 2
                ]) == [
                1 2 1 1 2 2;
                2 1 1 2 1 2;
                1 1 2 2 2 2
                ]
        end
        @testset "filter_upper_idxs" begin
            # Without cell index row
            @test filter_upper_idxs([
                1 1 1 2 2 2 3 3 3;
                1 2 3 1 2 3 1 2 3
                ]) == [
                1 1 1 2 2 3;
                1 2 3 2 3 3
                ]

            # With cell index row
            @test filter_upper_idxs([
                1 1 1 2 2 2 3 3 3;
                1 2 3 1 2 3 1 2 3;
                1 2 3 4 5 6 7 8 9
                ]) == [
                1 1 1 2 2 3;
                1 2 3 2 3 3
                1 2 3 5 6 9
                ]
        end

        @testset "filter_lower_idxs" begin
            # Without cell index row
            @test filter_lower_idxs([
                1 1 1 2 2 2 3 3 3;
                1 2 3 1 2 3 1 2 3
                ]) == [
                1 2 2 3 3 3;
                1 1 2 1 2 3
                ]

            # With cell index row
            @test filter_lower_idxs([
                1 1 1 2 2 2 3 3 3;
                1 2 3 1 2 3 1 2 3;
                1 2 3 4 5 6 7 8 9
                ]) == [
                1 2 2 3 3 3;
                1 1 2 1 2 3
                1 4 5 7 8 9
                ]
        end
    end
    @testset "Data Assignment" begin

        @testset "Private" begin


            mat_size = 1000
            n_cells = 10
            max_blk_size = 10
            max_idx = mat_size - max_blk_size - 1
            n_samples = 10
            n_blocks = 12

            # Gathering data from a matrix
            @testset "_get_blocks!" begin
                @testset "2D" begin
                    src = rand(mat_size, mat_size)
                    for _=1:n_samples
                        n, m = rand(1:max_blk_size), rand(1:max_blk_size)
                        starts = rand(1:max_idx, 2, n_blocks)
                        
                        ref = Array{valtype(src), 3}(undef, n, m, n_blocks)
                        for i=1:n_blocks
                            ref[:, :, i] = src[starts[1, i]:starts[1, i] + n - 1, starts[2, i]:starts[2, i] + m - 1]
                        end

                        target = Array{valtype(src), 3}(undef, n, m, n_blocks)
                        _get_blocks!(src, target, starts)

                        @test target == ref
                    end
                end

                @testset "3D" begin
                    src = rand(mat_size, mat_size, n_cells)
                    for _=1:n_samples
                        n, m = rand(1:max_blk_size), rand(1:max_blk_size)
                        starts = vcat(rand(1:max_idx, 2, n_blocks), rand(1:n_cells, 1, n_blocks))
                        
                        ref = Array{valtype(src), 3}(undef, n, m, n_blocks)
                        for i=1:n_blocks
                            ref[:, :, i] = src[starts[1, i]:starts[1, i] + n - 1, starts[2, i]:starts[2, i] + m - 1, starts[3, i]]
                        end

                        target = Array{valtype(src), 3}(undef, n, m, n_blocks)
                        _get_blocks!(src, target, starts)

                        @test target == ref
                    end
                end
            end

            # Placing data in the matrix
            @testset "_set_blocks!" begin

                # There is an issue here where the same block might get written to twice.
                @testset "2D" begin

                    target = rand(mat_size, mat_size)

                    for _=1:n_samples
                        n, m = rand(1:max_blk_size), rand(1:max_blk_size)
                        
                        # Starts must be generated so that they don't cause blocks to overlap 
                        starts = Matrix{Int}(undef, 2, n_blocks)
                        
                        for c=1:n_blocks
                            
                            i, j = rand(1:max_idx, 2)
    
                            safe_max = 0
                            while any(abs.(starts[1,1:c] .- i) .< n)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_blk_size"
                                i = rand(1:max_idx)
                            end

                            
                            safe_max = 0
                            while any(abs.(starts[2,1:c] .- j) .< m)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_blk_size"
                                j = rand(1:max_idx)
                            end

                            starts[1, c] = i
                            starts[2, c] = j 
                            
                        end
                        
                        src = rand(valtype(target), n, m, n_blocks)
                        _set_blocks!(src, target, starts)

                        ref = Array{valtype(src), 3}(undef, n, m, n_blocks)
                        for i=1:n_blocks
                            ref[:, :, i] = target[starts[1, i]:starts[1, i] + n - 1, starts[2, i]:starts[2, i] + m - 1]
                        end

                        @test src == ref
                    end
                end

                @testset "3D" begin

                    target = rand(mat_size, mat_size, n_cells)

                    for _=1:n_samples
                        n, m = rand(1:max_blk_size), rand(1:max_blk_size)
                        
                        # Starts must be generated so that they don't cause blocks to overlap 
                        starts = Matrix{Int}(undef, 3, n_blocks)
                        
                        for c=1:n_blocks
                            
                            i, j = rand(1:max_idx, 2)
    
                            safe_max = 0
                            while any(abs.(starts[1,1:c] .- i) .< n)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_blk_size"
                                i = rand(1:max_idx)
                            end

                            
                            safe_max = 0
                            while any(abs.(starts[2,1:c] .- j) .< m)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_blk_size"
                                j = rand(1:max_idx)
                            end

                            starts[1, c] = i
                            starts[2, c] = j
                            starts[3, c] = rand(1:n_cells)
                            
                        end
                        
                        src = rand(valtype(target), n, m, n_blocks)
                        _set_blocks!(src, target, starts)

                        ref = Array{valtype(src), 3}(undef, n, m, n_blocks)
                        for i=1:n_blocks
                            ref[:, :, i] = target[starts[1, i]:starts[1, i] + n - 1, starts[2, i]:starts[2, i] + m - 1, starts[3, i]]
                        end

                        @test src == ref
                    end
                end
            end
        end

        @testset "Public" begin
            # Again there is a fair bit of repetition going on here.

            basis_def = Dict(1=>[0], 6=>[0,1], 7=>[0, 0, 1, 1, 2])
            starts = Dict(1=>[1], 6=>[1, 2], 7=>[1, 2, 3, 6, 9])
            atom_sizes = Dict(k=>sum(2v .+ 1) for (k, v) in basis_def)

            atoms = Atoms(;Z=[1, 6, 1, 7, 7, 6])
            block_starts = [1, 2, 6, 7, 20, 33]

            species = convert(Vector{Int}, collect(keys(basis_def)))
            z_s = convert(Vector{Int}, atoms.Z)

            matrix_a = rand(36, 36)
            matrix_b = rand(36, 36, 10) 

            @testset "get_sub_blocks" begin

                @testset "2D" begin
                    for z_1=species, z_2=species
                        for s_i=1:length(basis_def[z_1]), s_j=length(basis_def[z_2])
                            rs, cs = 2basis_def[z_1][s_i] + 1, 2basis_def[z_2][s_j] + 1

                            blk_idxs = atomic_blk_idxs(z_1, z_2, z_s)
                            sub_blocks = get_sub_blocks(matrix_a, blk_idxs, s_i, s_j, atoms, basis_def)
                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(blk_idxs, 2)
                                r, c = block_starts[blk_idxs[:, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
                                sub_blocks_ref[:, :, i] = matrix_a[r:r+rs-1, c:c+cs-1]
                            end

                            @test sub_blocks == sub_blocks_ref
                            
                        end
                    end
                end

                @testset "3D" begin
                    for z_1=species, z_2=species
                        for s_i=1:length(basis_def[z_1]), s_j=length(basis_def[z_2])
                            rs, cs = 2basis_def[z_1][s_i] + 1, 2basis_def[z_2][s_j] + 1

                            blk_idxs = repeat_atomic_blk_idxs(atomic_blk_idxs(z_1, z_2, z_s), 10)
                            sub_blocks = get_sub_blocks(matrix_b, blk_idxs, s_i, s_j, atoms, basis_def)
                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(blk_idxs, 2)
                                r, c = block_starts[blk_idxs[1:2, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
                                
                                sub_blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, blk_idxs[3, i]]
                            end

                            @test sub_blocks == sub_blocks_ref
                            
                        end
                    end
                end

            end

            @testset "get_blocks" begin

                @testset "2D" begin
                    for z_1=species, z_2=species
                        rs, cs = atom_sizes[z_1], atom_sizes[z_2]

                        blk_idxs = atomic_blk_idxs(z_1, z_2, z_s)
                        blocks = get_blocks(matrix_a, blk_idxs, atoms, basis_def)
                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(blk_idxs, 2)
                            r, c = block_starts[blk_idxs[:, i]]
                            
                            blocks_ref[:, :, i] = matrix_a[r:r+rs-1, c:c+cs-1]
                        end

                        @test blocks == blocks_ref
                    end
                end

                @testset "3D" begin
                    for z_1=species, z_2=species
                        rs, cs = atom_sizes[z_1], atom_sizes[z_2]

                        blk_idxs = repeat_atomic_blk_idxs(atomic_blk_idxs(z_1, z_2, z_s), 10)
                        blocks = get_blocks(matrix_b, blk_idxs, atoms, basis_def)
                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(blk_idxs, 2)
                            r, c = block_starts[blk_idxs[1:2, i]]
                            
                            blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, blk_idxs[3, i]]
                        end

                        @test blocks == blocks_ref
                    end
                end
                
            end

            @testset "set_sub_blocks!" begin

                @testset "2D" begin
                    for z_1=species, z_2=species
                        for s_i=1:length(basis_def[z_1]), s_j=length(basis_def[z_2])
                            rs, cs = 2basis_def[z_1][s_i] + 1, 2basis_def[z_2][s_j] + 1

                            blk_idxs = atomic_blk_idxs(z_1, z_2, z_s)
                            sub_blocks = rand(rs, cs, size(blk_idxs, 2))
                            set_sub_blocks!(matrix_a, sub_blocks, blk_idxs, s_i, s_j, atoms, basis_def)

                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(blk_idxs, 2)
                                r, c = block_starts[blk_idxs[:, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
                                sub_blocks_ref[:, :, i] = matrix_a[r:r+rs-1, c:c+cs-1]
                            end

                            @test sub_blocks == sub_blocks_ref
                            
                        end
                    end
                end

                @testset "3D" begin
                    for z_1=species, z_2=species
                        for s_i=1:length(basis_def[z_1]), s_j=length(basis_def[z_2])
                            rs, cs = 2basis_def[z_1][s_i] + 1, 2basis_def[z_2][s_j] + 1

                            blk_idxs = repeat_atomic_blk_idxs(atomic_blk_idxs(z_1, z_2, z_s), 10)
                            sub_blocks = rand(rs, cs, size(blk_idxs, 2))
                            set_sub_blocks!(matrix_b, sub_blocks, blk_idxs, s_i, s_j, atoms, basis_def)

                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(blk_idxs, 2)
                                r, c = block_starts[blk_idxs[1:2, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
                                sub_blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, blk_idxs[3, i]]
                            end

                            @test sub_blocks == sub_blocks_ref
                            
                        end
                    end
                end                
            end

            @testset "set_blocks!" begin

                @testset "2D" begin
                    for z_1=species, z_2=species
                        rs, cs = atom_sizes[z_1], atom_sizes[z_2]

                        blk_idxs = atomic_blk_idxs(z_1, z_2, z_s)
                        blocks = rand(rs, cs, size(blk_idxs, 2))
                        set_blocks!(matrix_a, blocks, blk_idxs, atoms, basis_def)

                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(blk_idxs, 2)
                            r, c = block_starts[blk_idxs[1:2, i]]
                            blocks_ref[:, :, i] = matrix_a[r:r+rs-1, c:c+cs-1]
                        end

                        @test blocks == blocks_ref

                    end
                end

                @testset "3D" begin
                    for z_1=species, z_2=species
                        rs, cs = atom_sizes[z_1], atom_sizes[z_2]

                        blk_idxs = repeat_atomic_blk_idxs(atomic_blk_idxs(z_1, z_2, z_s), 10)
                        blocks = rand(rs, cs, size(blk_idxs, 2))
                        set_blocks!(matrix_b, blocks, blk_idxs, atoms, basis_def)

                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(blk_idxs, 2)
                            r, c = block_starts[blk_idxs[1:2, i]]
                            blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, blk_idxs[3, i]]
                        end

                        @test blocks == blocks_ref
                        
                    
                    end
                end

            end
        end

    end
    
end

function _error_free(expr::Expr)
    try
        eval(expr)
        return true
    catch
        return false
    end
end

@testset "States" begin
    t = SVector{3, Float64}

    # Just ensure that state entities can be instantiated manually without error
    @testset "Sanity Check" begin
        pos = t([0., 0., 0.])
        @test _error_free(:(AtomState($pos)))
        @test _error_free(:(BondState($pos, $pos, false)))
    end

    @testset "General Functionality" begin
        @testset "AtomStates" begin
            @test zero(AtomState{t}) == zero(AtomState{t}(rand(3))) == AtomState{t}(zero(t))

            @test AtomState(t([1, 1, 1])) == AtomState(t([1, 1, 1]))
            @test AtomState(t([1, 1, 1])) != AtomState(t([1, 1, 1.00000001]))

            @test AtomState(t([1, 1, 1])) ≈ AtomState(t([1, 1, 1.00000001]))

            @test AtomState(t([1, 1, 1])) == reflect(AtomState(t([1, 1, 1])))

            @test ison(AtomState(t([1, 1, 1])))

        end

        @testset "BondStates" begin
            a = BondState(ones(t), ones(t), false)
            b = BondState(t([1, 1, 1.00000001]), ones(t), false)
            c = BondState(ones(t), ones(t), true)
        
            @test zero(typeof(a)) == zero(a) == BondState{t, Bool}(zero(t), zero(t), false)
            
            @test a == BondState(ones(t), ones(t), false)
            @test a != b
            @test a != c

            @test a ≈ b
            @test !(a ≈ c)

            @test reflect(a) == BondState(ones(t), -ones(t), false)
            @test reflect(c) == BondState(-ones(t), -ones(t), true)

            @test !ison(a)
        end
    end

    @testset "Factory Helper Functions" begin
        # Note that no tests are run on the `_neighbours` method as is just calls functions
        # form JuLIP and NeighbourLists.

        @testset "_guard_position" begin
            rr0 = t([1., 1., 1.])
            @test _guard_position(ones(t), rr0, 1, 2) == ones(t)
            @test abs(norm(_guard_position(zero(t), rr0, 1, 2; cutoff=0.05)) - 0.05) < 1E-5
            @test abs(norm(_guard_position(t([0.001, 0.001, 0.001]), rr0, 1, 2; cutoff=0.05)) - 0.05) < 1E-5
        end

        @testset "_locate_minimum_image" begin
            idxs = [1, 2, 1, 2, 1, 2]
            vecs = [ones(3), ones(3), ones(3), ones(3) * 0.5, ones(3) * 0.4, ones(3)]
            @test _locate_minimum_image(2, idxs, vecs) == 4
            @test _locate_minimum_image(1, idxs, vecs) == 5
        end

        @testset "_locate_target_image" begin
            # _locate_target_image(j, idxs, images, image)
            idxs = [1, 2, 1, 2, 1, 2]
            images = [[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5]]

            @test _locate_target_image(1, idxs, images, [0,0,0]) == 1
            @test _locate_target_image(1, idxs, images, [0,0,2]) == 3
            @test _locate_target_image(2, idxs, images, [0,0,3]) == 4
            @test _locate_target_image(1, idxs, images, [0,0,100]) == 0

        end

    end

    # The following test-set is not fully comprehensive, however it should, together with
    # the previous tests, ensure that most serious errors are caught.
    @testset "get_state" begin
        cell = [10. 0 0; 0 10 0; 0 0 10]
        a = [1., 1, 1]
        b = normalize([1, 1, 1]) .+ 1
        @testset "atom states" begin
            atoms = Atoms(;Z=[1, 1], X=[a, b], cell=cell, pbc=true)
            @test length(get_state(1, atoms; r=10.)) == 4
            @test length(get_state(1, atoms; r=5.)) == 1
            @test length(get_state(1, atoms; r=.5)) == 0
            @test norm(get_state(1, atoms; r=5.)[1].rr) == 1.0
        end

        @testset "bond states" begin
            env = CylindricalBondEnvelope(10.0, 5.0, 5.0, floppy=false, λ=0.0)
            midpoint = 0.5(b - a) + a
            c = midpoint + normalize(rand(3)) * 0.3
            atoms = Atoms(;Z=[1, 1, 1], X=[a, b, c], cell=cell, pbc=true)
            bond_state, env_state = get_state(1, 2, atoms, env)

            # Ensure the rr0 and bond.rr values are correct
            @test bond_state.rr0 == env_state.rr0 == 2bond_state.rr

            # Check that environmental positions are relative to the bond's midpoint
            @test env_state.rr == c-midpoint

            # Ensure that the code can can locate a target cell when requested. 
            @test get_state(1, 2, atoms, env, [0, 0, 0]) == get_state(1, 2, atoms, env)
            
            # and can manually build it if it is not present in the neighbour list.
            @test length(get_state(1, 2, atoms, env, [0, 0, -1])) == 2
        
        end
    end
    
end
