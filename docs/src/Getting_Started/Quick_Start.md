


# Introduction
![alt text](https://avatars.githubusercontent.com/u/68508620?s=200&v=4)

The `ACEhamiltonians` package is a `Julia` package that provides tools for constructing, fitting, and predicting self-consistent Hamiltonian and overlap matrices in solid-state systems. It is based on the atomic cluster expansion (ACE) approach and the associated [ACEsuit package](https://github.com/ACEsuit/ACE.jl). The `ACEhamiltonians` package contains functions for generating on-site and off-site basis functions, fitting these bases to theoretical (DFT) data, and predicting the Hamiltonian and overlap matrices for any atomic configuration in real or reciprocal space. `ACEhamiltonians` provides a flexible and efficient way to model the electronic structure of materials and is a valuable tool for researchers in computational materials science. Please refer to the associated [article](https://www.nature.com/articles/s41524-022-00843-2) for a more in-depth description of the methodological underpinnings of this package.

# Getting Started
This quick-start guide provides an overview of the `ACEhamiltonians` framework, including the `Model` and `SubModel` structures.
The `Model` structure represents a model that includes on-site and off-site bases, their corresponding parameters, and the basis definition, while the `SubModel` structures within represents the interaction between atomic shells in a system.
The guide outlines the construction, fitting, and predicting processes for both structures, providing users with an efficient and flexible way to represent interactions in various systems.
Links to more detailed documents are also provided as necessary.

Installation and setup instructions can be found at the end of this document.
The `example_data.h5` dataset used throughout the examples and documentation has been supplied, and can be downloaded [here](https://github.com/ACEsuit/ACEhamiltonians_data).

## Models

The Model struct in the ACEhamiltonians framework represents a model that includes both on-site and off-site bases, their corresponding parameters, and the basis definition.
It also contains a label to identify the model and a dictionary for storing any meta-data associated with it.

The model constructor takes in the necessary information, such as the on-site and off-site parameters, and generates the associated on-site and off-site bases for all unique species pairs and their shells.
It handles cases with symmetrically equivalent interactions and allows for the use of dual-basis models when necessary (primarily for debugging).
The constructor ensures that the model is built with the appropriate bases and parameters, making it ready for fitting and predicting operations.

The Model structure in the ACEhamiltonians framework consists of several fields that play a crucial role in defining and utilizing the model. Here is a list of these fields and a brief description of their purpose:
1.  `on_site_bases`: A dictionary that stores the on-site basis functions for each unique species pair and their shells, indexed by a tuple of the form (z₁, n₁, n₂), where z₁ represents the atomic number and n₁ and n₂ are the shell indices.
    
2.  `off_site_bases`: A dictionary that stores the off-site basis functions for each unique species pair and their shells, indexed by a tuple of the form (z₁, z₂, n₁, n₂), where z₁ and z₂ represent the atomic numbers and n₁ and n₂ are the shell indices.
    
3.  `on_site_parameters`: An object of the `OnSiteParaSet` type that contains the on-site parameters for each unique species pair and their shells.
    
4.  `off_site_parameters`: An object of the `OffSiteParaSet` type that contains the off-site parameters for each unique species pair and their shells.
    
5.  `basis_definition`: A collection that describes the basis set for each species present in the model, including the atomic number and associated shell information.
    
6.  `label`: A string that serves as an identifier for the model, typically indicating the type of matrix being represented (e.g., "H" for Hamiltonian or "S" for overlap).
    
7.  `meta_data`: A dictionary that stores any additional meta-data associated with the model. This can include miscellaneous information or attributes that are not directly involved in the model construction, fitting, or prediction processes.


The [Model Construction](Model/Model_Construction.md) section outlines the process of creating an `ACEhamiltonians` model in Julia.
The model is initialised by providing parameters such as `model_type`, `basis_definition`, `on_site_parameters`, and `off_site_parameters`.
The `model_type` is set to "H" or "S" depending on whether the model will be fitted to Hamiltonian or overlap matrices.
`basis_definition` specifies the azimuthal quantum numbers, while `on_site_parameters` and `off_site_parameters` define the parameters required for constructing on-site and off-site bases respectively.
The file also details the various `Params` type structures (`GlobalParams`, `AtomicParams`, `AzimuthalParams`, and `ShellParams`) that provide different levels of specificity for parameter declarations.
The `OnSiteParaSet` and `OffSiteParaSet` structures are used to store all the required parameter definitions for on-site and off-site interactions.

The [Model Fitting](Model/Model_Fitting.md) section demonstrates how to use the various fitting subroutines within the `ACEhamiltonians` framework to fit a model.
With the model-level automated fitting procedure, you only need to specify the model to be fitted and the location(s) for gathering fitting data.
The `fit!` function performs the fitting operation and accepts several optional keyword arguments for customization.
You can also manually select fitting data by placing it into a dictionary keyed by basis id and providing it to the fitting function along with the model.
In this case, you only need to provide data for the bases you want to fit.

Finally, the [Model Predicting](Model/Model_Predicting.md) section illustrates how to make predictions using the `predict` methods. 
To create the real-space matrix for a system, call the `predict` method with a `model`, an `Atoms` entity representing the system, and the cell translation vectors.
You can then use the real-space matrix to construct the complex matrix at a specific k-point with the `real_to_complex` method.
The cell translation vectors determine which cells are included when constructing the real-space matrix.
You can use the `cell_translations` method to estimate them based on the distance cut-offs within the model.

 



## SubModels
Instead of constructing, fitting and preficting with the `Model` structure, whose input and output are both whole Hamiltonian, the `SubModel` structure gives the users 
more flexibility in terms of subblock fitting. 
It is nevertheless another key component of the `ACEHamiltonians` framework, representing the interaction between atomic shells in a system.
To use this structure, one start with constructibg on-site or off-site ACE basis and specifying a unique identifier (`id`) for the specific interaction.
Once fitted(with the help of data), the `SubModel` structure holds `coefficients` and `mean` values used for making predictions.
It is a flexible and efficient way to represent interactions in various systems, making it an essential part of the model fitting and prediction process.

Typically, creating isolated `SubModel` instances isn't commonly required, as the `Model` encompasses the essential functionality for constructing and applying bases. However, working with individual SubModel instances can be beneficial during hyperparameter optimization and are therefore discussed here.


The [`SubModel Construction`](Bases/Basis_Construction.md) section explains how to create individual on-site and off-site bases using the `on_site_ace_basis` and `off_site_ace_basis` functions, respectively. The process involves specifying necessary parameters such as quantum numbers, correlation order, maximum polynomial degree, and cutoff distances. After creating the core basis, it is wrapped in a `SubModel` instance, which associates the basis with an interaction and provides an identifier. The `SubModel` structure is utilised during the fitting process and when making predictions.


The [`SubModel Fitting`](Bases/Basis_Fitting.md) section explains how to fit individual `SubModel` instances by providing the basis and fitting data to the `fit!` function. The function requires a `DataSet` instance containing all necessary data for the fitting process. The `get_dataset` convenience function automatically collects all relevant data, simplifying the fitting operation. The fitting process involves loading the real-space Hamiltonian matrix, atoms object, cell translation vectors, and basis set definition, and then performing the fitting operation on the bases.


The [`SubModel Predicting`](Bases/Basis_Predicting.md) section describes how to make predictions for individual bases using the `predict` function. By providing a `SubModel` instance and a state-vector (a vector of `AbstractState` instances), predictions can be made for on-site and off-site interactions. The example demonstrates how to load an atoms object, obtain the on-site and off-site states representing the environment, and make predictions for these states. Predictions can also be made for multiple blocks simultaneously by providing a vector containing multiple state-vectors.


# Installation and Setup
The `ACEhamiltonians` package can be installed in either general user mode or development mode. User mode installs the package into the `.julia/packages/` directory, allowing it to be imported and used as any other `Julia` package. In contrast, development mode sets up `ACEhamiltonians` as a repository that can be modified as needed, making it the currently recommended option. To get started using `ACEhamiltonians` in user mode, simply start a `Julia` session and issue the relevant commands as provided in the following code-block. To install `ACEhamiltonians` for development purposes, follow the instructions provided in the following development sub-section.

```Julia
using Pkg
# Add MolSim registry
Pkg.Registry.add(RegistrySpec(url="https://github.com/JuliaMolSim/MolSim.git"))
# Readd the "General" registry to ensure stability
Pkg.Registry.add("General")
# Finally add the ACEhamiltonians package
Pkg.add(url="https://github.com/ACEsuit/ACEhamiltonians.jl.git", ref="Development")
```

Then verify the installation by starting a new `Julia` session and issue "`using ACEhamiltonians`" to import and build the package.

## Development

To download, install, and set up `ACEhamiltonians` for development purposes, follow these steps:

1. In the terminal, create a working directory and `cd` into it.
   Clone the repository and checkout the development branch.
   Then start a `julia` session using the newly cloned repository as the working directory.
   ```bash
   # Make a working directory (called ACEhamiltonians in this example) and cd into it 
   mkdir ACEhamiltonians && cd "$_"
   
   # Clone the ACEhamiltonians repository and checkout the Development branch
   git clone git@github.com:ACEsuit/ACEhamiltonians.jl.git --branch Development
   
   # Start a Julia session using the ACEhamiltonians package as the project/environment.
   julia --project=ACEhamiltonians.jl 
   ```

2. Within the `Julia` session, execute the following code.
   This will make the required registries available, instantiate the package, and mark it as a developmental package.
   ```Julia
   using Pkg
   
   # Add MolSim registry
   Pkg.Registry.add(RegistrySpec(url="https://github.com/JuliaMolSim/MolSim.git"))
   
   # Readd the "General" registry to ensure stability
   Pkg.Registry.add("General")
   
   # Instantiate the environment from the `Manifest.toml` and or `Project.toml` files.
   Pkg.instantiate()
   
   # Deactivate the current Julia environment (to permit the next command)
   Pkg.activate()
   
   # Make the ACEhamiltonians package available for development.
   Pkg.develop(path="./ACEhamiltonians.jl")
   ```

3. Finally, check that the `ACEhamiltonians` package is accessible by starting a new `Julia` session and issuing "`using ACEhamiltonians`" to import and build the package.


Some static package level development configuration options are made available in the `ACEhamiltonians.jl` source file.
These options are as follows:
 - `DUAL_BASIS_MODEL`: if set to `true` the off-site hetro-orbital interactions will be represented by a pair of bases; i.e. $sp$ and $ps$.
   Predictions will then be made by averaging the results of the two models like so $sp(AB) = 0.5\cdot(sp(AB) + ps(BA)^T)$.
   While this is somewhat redundant it does help aviate some current stability issues.
   This is set to `true` by default.
 - `BOND_ORIGIN_AT_MIDPOINT`: if set to `true` the point of the bond will be used as the origin for the off-site states objects.
   This is disabled by default as a when an environmental atom lies exactly at the midpoint spherical angles θ and ϕ become indeterminate and thus a direction cannot be determined. This results in violations of bond symmetry.
 - `SYMMETRY_FIX_ENABLED`: if set to `true` then a symmetry fix will be applied to homo-azimuthal interactions to force them to respect transposition symmetry. This is disabled by default as it is only valid when the bond origin lies at the midpoint.
