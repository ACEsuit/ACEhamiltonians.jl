using Documenter, ACEhamiltonians

makedocs(sitename="ACEhamiltonians.jl Documentation",
         pages = [
        "Home" => "index.md",
        "Getting Started" => "Getting_Started/Quick_Start.md",
        "Database Structure" => "Getting_Started/Data/Database_Structure.md",
        "Models" => ["Getting_Started/Model/Model_Construction.md", "Getting_Started/Model/Model_Fitting.md", "Getting_Started/Model/Model_Predicting.md"],
        "Bases" => ["Getting_Started/Bases/Basis_Construction.md", "Getting_Started/Bases/Basis_Fitting.md", "Getting_Started/Bases/Basis_Predicting.md"]
         ])

                 
deploydocs(
    repo = "github.com/ACEsuit/ACEhamiltonians.jl.git",
    devbranch = "main"
)
