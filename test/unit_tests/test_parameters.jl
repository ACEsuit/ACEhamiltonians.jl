using Test
using ACEbase: write_dict, read_dict, load_json, save_json
using ACEhamiltonians
using ACEhamiltonians.Parameters: Label, NewParams


function check_keys(param::NewParams, args...)
    for (k, v) in args
        @testset "key: $k" begin
            @test param[k] == v
        end
    end
end


@testset "Parameters" begin

@testset "Params" begin
    
    @testset "GlobalParams" begin
        
        # Ensure that GlobalParams instances can be instantiated
        p₁ = GlobalParams(1.0)
        p₂ = GlobalParams(Label()=>1.0)

        # Check that the equality operator functions as intended
        @testset "Equality" begin
            @test p₁ == p₂
        end

        # Verify that GlobalParams entities can be indexed
        @testset "Indexable" begin
            check_keys(p₁, 0=>1.0, (0,)=>1.0, (0, 0, 0)=>1.0, (0, 0, 0, 0)=>1.0)
        end

        # Make sure that GlobalParams instances can be converted to & from dictionaries
        @testset "Dictionary Interconversion Stability" begin
            @test p₁ == read_dict(write_dict(p₁))
        end

        # Ensure that GlobalParams objects stored in JSON files are recoverable 
        @testset "JSON Interconversion Stability" begin
            f = tempname()
            save_json(f, write_dict(p₁))
            @test p₁ == read_dict(load_json(f))
        end
        
    end

       
    @testset "AtomicParams" begin
        
        # Ensure that AtomicParams instances can be instantiated
        p_on₁ = AtomicParams(1=>1.1, 6=>6.6)
        p_on₂ = AtomicParams((1,)=>1.1, (6,)=>6.6)
        p_on₃ = AtomicParams(Label(1)=>1.1, Label(6)=>6.6)

        p_off₁ = AtomicParams((1, 1)=>1.1, (1, 6)=>1.6, (6, 6)=>6.6)
        p_off₂ = AtomicParams((1, 1)=>1.1, (6, 1)=>1.6, (6, 6)=>6.6)
        p_off₃ = AtomicParams(Label(1,1)=>1.1, Label(1, 6)=>1.6, Label(6, 6)=>6.6)

        @testset "Equality" begin
            @test p_on₁ == p_on₂ == p_on₃
            @test p_off₁ == p_off₂ == p_off₃
        end

        @testset "Indexable" begin
            @testset "On-site" begin
                check_keys(
                    p_on₁, 1=>1.1, (1,)=>1.1, 6=>6.6, Label(6)=>6.6, (1, 1, 1)=>1.1,
                    (6, 999, 999)=>6.6)
            end

            @testset "Off-site" begin
                check_keys(
                    p_off₁, (1, 1)=>1.1, (6, 6)=>6.6, (1, 6)=>1.6, (6, 1)=>1.6,
                    (1,1,1,1)=>1.1, (6,1,999,1)=>1.6)
            end
        end

        @testset "Dictionary Interconversion Stability" begin
            @test p_on₁ == read_dict(write_dict(p_on₁))
            @test p_off₁ == read_dict(write_dict(p_off₁))
        end


        @testset "JSON Interconversion Stability" begin
            f_on = tempname()
            save_json(f_on, write_dict(p_on₁))
            @test p_on₁ == read_dict(load_json(f_on))

            f_off = tempname()
            save_json(f_off, write_dict(p_off₁))
            @test p_off₁ == read_dict(load_json(f_off))
        end
        
    end
    

    @testset "AzimuthalParams" begin
        b_def = Dict(1=>[0], 6=>[0, 0, 1])

        # Ensure that AzimuthalParams instance-s can be instantiated
        p_on₁ = AzimuthalParams(
            b_def, (1, 0, 0)=>1.00, (6, 0, 0)=>6.00, (6, 0, 1)=>6.01, (6, 1, 1)=>6.11)

        p_on₂ = AzimuthalParams(
            b_def, (1, 0, 0)=>1.00, (6, 0, 0)=>6.00, (6, 1, 0)=>6.01, (6, 1, 1)=>6.11)


        p_off₁ = AzimuthalParams(
            b_def, (1, 1, 0, 0)=>11.00, (1, 6, 0, 0)=>16.00, (1, 6, 0, 1)=>16.01,
            (6, 6, 0, 0)=>66.00, (6, 6, 0, 1)=>66.01, (6, 6, 1, 1)=>66.11)

        p_off₂ = AzimuthalParams(
            b_def, (1, 1, 0, 0)=>11.00, (6, 1, 0, 0)=>16.00, (6, 1, 1, 0)=>16.01,
            (6, 6, 0, 0)=>66.00, (6, 6, 1, 0)=>66.01, (6, 6, 1, 1)=>66.11)


        @testset "Equality" begin
            @test p_on₁ == p_on₂
            @test p_off₁ == p_off₂
        end

        @testset "Indexable" begin
            @testset "On-site" begin
                check_keys(
                    p_on₁, (1, 1, 1)=>1.00, (6, 1, 1)=>6.00, (6, 2, 2)=>6.00,
                    (6, 1, 2)=>6.00, (6, 2, 1)=>6.00, (6, 1, 3)=>6.01, (6, 3, 3)=>6.11)
            end

            @testset "Off-site" begin
                check_keys(
                    p_off₁, (1, 1, 1, 1)=>11.00, (6, 1, 2, 1)=>16.00, (1, 6, 1, 2)=>16.00,
                    (6, 6, 1, 2)=>66.00, (6, 6, 3, 2)=>66.01)
            end
        end

        @testset "Dictionary Interconversion Stability" begin
            @test p_on₁ == read_dict(write_dict(p_on₁))
            @test p_off₁ == read_dict(write_dict(p_off₁))
        end


        @testset "JSON Interconversion Stability" begin
            f_on = tempname()
            save_json(f_on, write_dict(p_on₁))
            @test p_on₁ == read_dict(load_json(f_on))

            f_off = tempname()
            save_json(f_off, write_dict(p_off₁))
            @test p_off₁ == read_dict(load_json(f_off))
        end
        
    end


    @testset "ShellParams" begin
        
        # Ensure that ShellParams instances can be instantiated
        p_on₁ = ShellParams(
            (1, 1, 1)=>1.11, (6, 1, 1)=>6.11, (6, 1, 2)=>6.12, (6, 2, 2)=>6.22)
        p_on₂ = ShellParams(
            Label(1, 1, 1)=>1.11, Label(6, 1, 1)=>6.11, Label(6, 1, 2)=>6.12,
            Label(6, 2, 2)=>6.22)

        p_on₃ = ShellParams(
                (1, 1, 1)=>1.11, (6, 1, 1)=>6.11, (6, 2, 1)=>6.12, (6, 2, 2)=>6.22)

        p_off₁ = ShellParams(
            (1, 1, 1, 1)=>11.11, (1, 6, 1, 1)=>16.11, (1, 6, 1, 2)=>16.12,
            (6, 6, 1, 1)=>66.11, (6, 6, 1, 2)=>66.12, (6, 6, 2, 2)=>66.11)

        p_off₂ = ShellParams(
            (1, 1, 1, 1)=>11.11, (6, 1, 1, 1)=>16.11, (6, 1, 2, 1)=>16.12,
            (6, 6, 1, 1)=>66.11, (6, 6, 2, 1)=>66.12, (6, 6, 2, 2)=>66.11)

        @testset "Equality" begin
            @test p_on₁ == p_on₂ == p_on₃
            @test p_off₁ == p_off₂
        end

        @testset "Indexable" begin
            @testset "On-site" begin
                check_keys(p_on₁, (1, 1, 1)=>1.11, (6, 1, 2)=>6.12, (6, 2, 1)=>6.12)
            end

            @testset "Off-site" begin
                check_keys(
                    p_off₁, (1, 1, 1, 1)=>11.11, (6, 1, 2, 1)=>16.12, (1, 6, 1, 2)=>16.12,
                    (6, 6, 1, 2)=>66.12)
            end
        end

        @testset "Dictionary Interconversion Stability" begin
            @test p_on₁ == read_dict(write_dict(p_on₁))
            @test p_off₁ == read_dict(write_dict(p_off₁))
        end


        @testset "JSON Interconversion Stability" begin
            f_on = tempname()
            save_json(f_on, write_dict(p_on₁))
            @test p_on₁ == read_dict(load_json(f_on))

            f_off = tempname()
            save_json(f_off, write_dict(p_off₁))
            @test p_off₁ == read_dict(load_json(f_off))
        end
        
    end

end

@testset "ParaSet" begin
    G = GlobalParams
    
    @testset "OnSiteParaSet" begin
        # Ensure that the ParaSet can be instantiated
        @testset "Instantiation" begin
            ps = OnSiteParaSet(G(2), G(4), G(12.0), G(0.5))
            @test @isdefined ps
        end

        # Make sure that only valid types are accepted
        @testset "Type Guarding" begin
            @test_throws TypeError OnSiteParaSet(G(1.0), G(1), G(1.0), G(0.1))
            @test_throws TypeError OnSiteParaSet(G(1), G(1.0), G(1.0), G(0.1))
            @test_throws TypeError OnSiteParaSet(G(1), G(1), G(1), G(0.1))
            @test_throws TypeError OnSiteParaSet(G(1), G(1), G(1.0), G(0))
        end

        ps = OnSiteParaSet(G(2), G(4), G(12.0), G(0.5))

        @testset "Equality" begin
            ps_a = OnSiteParaSet(G(2), G(4), G(12.0), G(0.5))
            ps_b = OnSiteParaSet(G(2), G(4), G(12.1), G(0.5))

            @test ps == ps_a
            @test ps ≠ ps_b
        end

        @testset "ison" begin
            @test ison(ps)
        end

        @testset "Dictionary Interconversion Stability" begin
            @test ps == read_dict(write_dict(ps))
        end

        @testset "JSON Interconversion Stability" begin
            fn = tempname()
            save_json(fn, write_dict(ps))
            @test ps == read_dict(load_json(fn))
        end



    end

    @testset "OffSiteParaSet" begin
        @testset "Instantiation" begin
            ps = OffSiteParaSet(G(2), G(4), G(12.0), G(12.0))
            @test @isdefined ps
        end  

        # Make sure that only valid types are accepted
        @testset "Type Guarding" begin
            @test_throws TypeError OffSiteParaSet(G(1.), G(1), G(1.0), G(1.0))
            @test_throws TypeError OffSiteParaSet(G(1), G(1.), G(1.0), G(1.0))
            @test_throws TypeError OffSiteParaSet(G(1), G(1), G(1), G(1.0))
            @test_throws TypeError OffSiteParaSet(G(1), G(1), G(1.0), G(1))
        end


        ps = OffSiteParaSet(G(2), G(4), G(12.0), G(12.0))

        @testset "Equality" begin
            ps_a = OffSiteParaSet(G(2), G(4), G(12.0), G(12.0))
            ps_b = OffSiteParaSet(G(2), G(4), G(12.1), G(12.0))

            @test ps == ps_a
            @test ps ≠ ps_b
        end

        @testset "ison" begin
            @test !ison(ps)
        end

        @testset "Dictionary Interconversion Stability" begin
            @test ps == read_dict(write_dict(ps))
        end

        @testset "JSON Interconversion Stability" begin
            fn = tempname()
            save_json(fn, write_dict(ps))
            @test ps == read_dict(load_json(fn))
        end


    end

end

end
