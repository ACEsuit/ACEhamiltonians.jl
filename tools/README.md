# Tools
This directory contains a selection of useful tools.
 - `rsm.j.`: command line interface for using `ACEhamiltonians` models.
 - `compile_model.jl`: used to pre-compile models.

# Command Line Interface
> Usage `julia rsm.jl <geometry-file> <H-model-file> <S-model-file>`

Arguments:
 - `<geometry-file>`: path an `ase` parsable file specifying the geometry of the system
   upon which the ACE-hamiltonians model is to be evaluated.
 - `<H-model-file>`: path to the JSON or binary file storing the Hamiltonian model to be
   used. Model files must end in an appropriate extension; i.e. *.json* for JSON formatted
   files and *.bin* for binary files.
 - `<S-model-file>`: path to the JSON or binary file storing the overlap model to be used.

The command line interface tool "`rsm.jl`" allows for `ACEhamiltonians` models to be used
from the command line. Following its execution `rsm.jl` will output four file:
 - `basis_indices.out`: a list specifying information about the effective basis functions
    used during the calculation.  
 - `H_rsm.out`: the resulting real space Hamiltonian matrix. 
 - `S_rsm.out`: the resulting real space overlap matrix.
 - `cell_indices.out`: the image indices for each cell slice in the real space Hamiltonian
   and overlap matrices.

The three-dimensional real space matrices are stored as a series of two-dimensional slices
with one slice for each cell. The image to which each cell corresponds is specified in the
associated cell indices file.

For stability purposes the `ACEhamiltonians` models are stored in JSON files. However,
this results in a non-trivial io overhead. Thus, it is recommended to use binary models
compile via the model compilation tool rather than use JSON files directly.


# Model Compiler
> Usage `julia compile_model.jl <input-file> <output-file>`

Arguments:
 - `<input-file>`: JSON file storing the model to be compiled.
 - `<output-file>`: file in which the compiled model should be placed.

This tool allows for JSON formatted model files to be pre-compiled to help reduce the
io overhead associated with loading the model.

For the purposes of stability and reproducibility, all ACE data-structures, including
`ACEhamiltonians` models, are stored in JSON formatted files. Unfortunately, the large
and recursive nature of the `ACEhamiltonians` model means that a non-trivial amount of
time is required to parse and construct each model. Thus, it is recommended to use the
inbuilt tools to parse the model into a binary file. This cuts the load time from around
a minute to a few fractions of a second. However, care must be taken as these binary files
are very sensitive to system changes and thus must be build anew for each compute system.

