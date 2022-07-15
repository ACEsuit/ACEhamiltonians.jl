using ACEhamiltonians, Test, Random
using JuLIP: Atoms


using ACEhamiltonians.MatrixManipulation: _block_starts, _sub_block_starts, _get_blocks!, _set_blocks!


@testset "Matrix Manipulation" begin
    @testset "Constructors" begin
        @testset "atomic_block_idxs" begin
            for z_s in ([1, 6, 1, 6], Atoms(;Z=[1, 6, 1, 6]))
                @test atomic_block_idxs(1, 1, z_s) == [1 1 3 3; 1 3 1 3]
                @test atomic_block_idxs(1, 6, z_s) == [1 1 3 3; 2 4 2 4]
                @test atomic_block_idxs(6, 1, z_s) == [2 2 4 4; 1 3 1 3]
                @test atomic_block_idxs(6, 6, z_s) == [2 2 4 4; 2 4 2 4]

                @test atomic_block_idxs(1, 6, z_s; order_invariant=true) == [1 1 3 3 2 2 4 4; 2 4 2 4 1 3 1 3]
                @test atomic_block_idxs(6, 1, z_s; order_invariant=true) == [2 2 4 4 1 1 3 3; 1 3 1 3 2 4 2 4]
            end
        end

        @testset "repeat_atomic_block_idxs" begin
            block_idxs_a = [
                1 2 3
                4 5 6]
            block_idxs_b = [
                1 2 3 1 2 3 1 2 3
                4 5 6 4 5 6 4 5 6
                1 1 1 2 2 2 3 3 3]
            @test repeat_atomic_block_idxs(block_idxs_a, 3) == block_idxs_b

            @test_throws AssertionError repeat_atomic_block_idxs(block_idxs_b, 3)
    
        end
    end

    @testset "Ancillary" begin

        @testset "_block_starts" begin
            basis_def = Dict(1=>[0], 6=>[0,1], 7=>[0, 0, 1, 1, 2])
            atoms = Atoms(;Z=[1, 6, 7, 6])
            starts = [1, 2, 6, 19]

            # Check general functionality
            sub_blocks = [
                [1 1; 2 4], [2 4; 1 1], [1; 3;;], [3; 1;;], [2 2 4 4; 2 4 2 4],
                [2 4; 3 3], [3 3; 2 4]]

            for sub_blocks in sub_blocks
                @test _block_starts(sub_blocks, atoms, basis_def) == starts[sub_blocks]
            end

            # Ensure that cell indices are preserved and their presence does not effect the results
            @test _block_starts([2 4; 3 3; 11 22], atoms, basis_def)[1:2, :] == _block_starts([2 4; 3 3], atoms, basis_def)
            @test _block_starts([2 4; 3 3; 11 12], atoms, basis_def)[3, :] == [11, 12]
            
        end

        @testset "_sub_block_starts" begin
            basis_def = Dict(1=>[0], 6=>[0, 1], 7=>[0, 0, 1, 1, 2])
            starts = Dict(1=>[1], 6=>[1, 2], 7=>[1, 2, 3, 6, 9])
            z_s = collect(keys(basis_def))
            for z_1=z_s, z_2=z_s
                for s_i=1:length(basis_def[z_1]), s_j=length(basis_def[z_2])
                    @test _sub_block_starts(z_1, z_2, s_i, s_j, basis_def) == [starts[z_1][s_i], starts[z_2][s_j]]
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
            max_block_size = 10
            max_idx = mat_size - max_block_size - 1
            n_samples = 10
            n_blocks = 12

            # Gathering data from a matrix
            @testset "_get_blocks!" begin
                @testset "2D" begin
                    src = rand(mat_size, mat_size)
                    for _=1:n_samples
                        n, m = rand(1:max_block_size), rand(1:max_block_size)
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
                        n, m = rand(1:max_block_size), rand(1:max_block_size)
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
                        n, m = rand(1:max_block_size), rand(1:max_block_size)
                        
                        # Starts must be generated so that they don't cause blocks to overlap 
                        starts = Matrix{Int}(undef, 2, n_blocks)
                        
                        for c=1:n_blocks
                            
                            i, j = rand(1:max_idx, 2)
    
                            safe_max = 0
                            while any(abs.(starts[1,1:c] .- i) .< n)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_block_size"
                                i = rand(1:max_idx)
                            end

                            
                            safe_max = 0
                            while any(abs.(starts[2,1:c] .- j) .< m)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_block_size"
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
                        n, m = rand(1:max_block_size), rand(1:max_block_size)
                        
                        # Starts must be generated so that they don't cause blocks to overlap 
                        starts = Matrix{Int}(undef, 3, n_blocks)
                        
                        for c=1:n_blocks
                            
                            i, j = rand(1:max_idx, 2)
    
                            safe_max = 0
                            while any(abs.(starts[1,1:c] .- i) .< n)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_block_size"
                                i = rand(1:max_idx)
                            end

                            
                            safe_max = 0
                            while any(abs.(starts[2,1:c] .- j) .< m)
                                safe_max += 1
                                @assert safe_max < 1000 "Increase mat_size or decrease max_block_size"
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

                            block_idxs = atomic_block_idxs(z_1, z_2, z_s)
                            sub_blocks = get_sub_blocks(matrix_a, block_idxs, s_i, s_j, atoms, basis_def)
                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(block_idxs, 2)
                                r, c = block_starts[block_idxs[:, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
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

                            block_idxs = repeat_atomic_block_idxs(atomic_block_idxs(z_1, z_2, z_s), 10)
                            sub_blocks = get_sub_blocks(matrix_b, block_idxs, s_i, s_j, atoms, basis_def)
                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(block_idxs, 2)
                                r, c = block_starts[block_idxs[1:2, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
                                
                                sub_blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, block_idxs[3, i]]
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

                        block_idxs = atomic_block_idxs(z_1, z_2, z_s)
                        blocks = get_blocks(matrix_a, block_idxs, atoms, basis_def)
                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(block_idxs, 2)
                            r, c = block_starts[block_idxs[:, i]]
                            
                            blocks_ref[:, :, i] = matrix_a[r:r+rs-1, c:c+cs-1]
                        end

                        @test blocks == blocks_ref
                    end
                end

                @testset "3D" begin
                    for z_1=species, z_2=species
                        rs, cs = atom_sizes[z_1], atom_sizes[z_2]

                        block_idxs = repeat_atomic_block_idxs(atomic_block_idxs(z_1, z_2, z_s), 10)
                        blocks = get_blocks(matrix_b, block_idxs, atoms, basis_def)
                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(block_idxs, 2)
                            r, c = block_starts[block_idxs[1:2, i]]
                            
                            blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, block_idxs[3, i]]
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

                            block_idxs = atomic_block_idxs(z_1, z_2, z_s)
                            sub_blocks = rand(rs, cs, size(block_idxs, 2))
                            set_sub_blocks!(matrix_a, sub_blocks, block_idxs, s_i, s_j, atoms, basis_def)

                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(block_idxs, 2)
                                r, c = block_starts[block_idxs[:, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
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

                            block_idxs = repeat_atomic_block_idxs(atomic_block_idxs(z_1, z_2, z_s), 10)
                            sub_blocks = rand(rs, cs, size(block_idxs, 2))
                            set_sub_blocks!(matrix_b, sub_blocks, block_idxs, s_i, s_j, atoms, basis_def)

                            sub_blocks_ref = zeros(size(sub_blocks)...)
                            for i=1:size(block_idxs, 2)
                                r, c = block_starts[block_idxs[1:2, i]] + [starts[z_1][s_i],  starts[z_2][s_j]] .- 1
                                sub_blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, block_idxs[3, i]]
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

                        block_idxs = atomic_block_idxs(z_1, z_2, z_s)
                        blocks = rand(rs, cs, size(block_idxs, 2))
                        set_blocks!(matrix_a, blocks, block_idxs, atoms, basis_def)

                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(block_idxs, 2)
                            r, c = block_starts[block_idxs[1:2, i]]
                            blocks_ref[:, :, i] = matrix_a[r:r+rs-1, c:c+cs-1]
                        end

                        @test blocks == blocks_ref

                    end
                end

                @testset "3D" begin
                    for z_1=species, z_2=species
                        rs, cs = atom_sizes[z_1], atom_sizes[z_2]

                        block_idxs = repeat_atomic_block_idxs(atomic_block_idxs(z_1, z_2, z_s), 10)
                        blocks = rand(rs, cs, size(block_idxs, 2))
                        set_blocks!(matrix_b, blocks, block_idxs, atoms, basis_def)

                        blocks_ref = zeros(size(blocks)...)
                        for i=1:size(block_idxs, 2)
                            r, c = block_starts[block_idxs[1:2, i]]
                            blocks_ref[:, :, i] = matrix_b[r:r+rs-1, c:c+cs-1, block_idxs[3, i]]
                        end

                        @test blocks == blocks_ref
                        
                    
                    end
                end

            end
        end

    end
    
end

