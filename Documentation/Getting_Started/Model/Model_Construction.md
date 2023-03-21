# Model Construction

An `ACEhamiltonians` model can be created, as shown below, by constructing and passing the required parameters to the model factory.
```julia
# 1) Provide a label for the model (this should be H or S)
model_type = "H"

# 2) Definition of a s₃p₃d₁ silicon basis
basis_definition = BasisDef{Int64}(14=>[0, 0, 0, 1, 1, 1, 2])

# 3) On-site parameter deceleration
on_site_parameters = OnSiteParaSet(
    # Maximum correlation order
    GlobalParams(2),
    # Maximum polynomial degree
    GlobalParams(6),
    # Environmental cutoff radius
    GlobalParams(10.),
    # Scaling factor "r₀"
    GlobalParams(2.5)
)

# 4) Off-site parameter deceleration
off_site_parameters = OffSiteParaSet(
    # Maximum correlation order
    GlobalParams(1),
    # Maximum polynomial degree
    GlobalParams(6),
    # Bond cutoff radius
    GlobalParams(10.),
    # Environmental cutoff radius
    GlobalParams(5.),
)

# 5) Initialise the model
model = Model(basis_definition, on_site_parameters, off_site_parameters, model_type)
```
The model factory takes four arguments, namely;

- `model_type`:  a string used to inform the auto-fitting code of what data to load and use during the fitting operation. This is commonly set to either `"H"` or `"S"` depending on whether one intends to fit the model to Hamiltonian or overlap matrices.
- `basis_definition`: a dictionary specifying the azimuthal quantum numbers, ℓ, of each shell present on each of the species to be modelled. The dictionary is keyed by atomic numbers and valued by vectors of ℓ i.e. `Dict{atomic_number, [ℓ₁, ..., ℓᵢ]}`.  A minimal basis set for hydrocarbon systems would be `BasisDef{Int64}(1=>[0], 6=>[0, 0, 1])`. This declares hydrogen atoms as having only a single s-shell and carbon atoms as having two s-shells and one p-shell. In reality a regular `julia` dictionary can be used so long as it has the correct structure.
- `on_site_parameters`: is an `OnSiteParaSet` structure, which as name suggests, specifies all of the parameters required to construct the on-site bases. Specifically the correlation order, maximum polynomial degree, environmental cutoff radius, and the scaling parameter. Further information on the `OnSiteParaSet` and the `Params` structures used in its construction can be found [here](#on-site) and [here](#declarations) respectively.
- `off_site_parameters`: is an [`OffSiteParaSet`](#off-site) structure which similarly specifies the parameters for constructing the off-site bases. This provides the correlation order, maximum polynomial degree, bond cutoff distance, and environmental cutoff radius. Again, [`Params`](#declarations) structures are used in the construction of these instances.

Models can be saved to and loaded from JSON files using the commands `save_json("file_name.json", write_dict(model))` and `read_dict(load_json("file_name.json"))` respectively.
Although these files are stable and transportable they are also incredibly slow to work with.
As such it is strongly advised to use the [`Serialization`](https://docs.julialang.org/en/v1/stdlib/Serialization/) module to load and save models timidly manner when working locally.
However, such binary files can only be reliably loaded on the system that was used to create them.

## Parameters

### Declarations
Prior to constructing a model one must specify the parameters to be used in its construction.
This is done primarily through the use of dictionary-like `Params` structures, which are collected into `ParaSet` instances as required.
These are used exclusively to provide the parameters needed when constructing models within the `ACEhamiltonians` framework, or more specifically their underlying bases.
There are currently four `Params` type structures, namely `GlobalParams`, `AtomicParams`, `AzimuthalParams`, and `ShellParams`, each offering varying levels of specificity.

Each parameter, correlation order, maximum polynomial degree, environmental cutoff distance, etc. may be specified using any of the available `Params` based structures.
However, i) each `Params` instance may represent one, and only one, parameter, and ii) on-site and off-site parameters must not be mixed.

#### Global

A `GlobalParams` instance is used to indicate that a single value should be used for all relevant interactions.
Querying such instances will always return the value the specified constant value, irrespective of the key, so long as the query is valid.
```julia
julia> p = GlobalParams(2)
GlobalParams{Int64} with 1 entries:
  All => 2

julia> ν[1] # <- query parameter associated with H
2
julia> ν[(1, 6)] # <- query parameter associated with H-C interaction
2
julia> ν[(1, 6, 1, 2)] # <- interaction between 1ˢᵗ shell on H and 2ⁿᵈ shell on C
2
```
As can be seen, the specified value, `2`, will always be returned so long as the query is valid.
These instances are useful when specifying parameters that are constant across all bases, such as the correlation order, as it avoids having to repeatedly specify it for each and every interaction.

#### Species Resolved

These instances allow for parameters to be specified on a species by species basis.
This equates to one parameter per species for on-site interactions and one parameter per species pair for off-site interactions.
This will then result in all associated bases associated with a specific species/species-pair all using a common value, like so: 
```julia
julia> p_on  = AtomicParams(1=>9., 6=>11.)
AtomicParams{Float64} with 2 entries:
  6 => 11.0
  1 => 9.0

julia> p_off = AtomicParams((1, 1)=>9., (1, 6)=>10., (6, 6)=>11.)
AtomicParams{Float64} with 3 entries:
  (6, 6) => 11.0
  (1, 6) => 10.0
  (1, 1) => 9.0

# The value 11. is returned for all on-site C interaction queries 
julia> p_on[(6, 1, 1)] == p_on[(6, 1, 2)] == p_on[(6, 2, 2)] == 11.
true
# The value 10. is returned for all off-site H-C interaction queries 
julia> p_off[(1, 6, 1, 1)] == p_off[(6, 1, 2, 1)] == p_off[(6, 1, 2, 2)] == 10.
true
```
These instances are instantiated in a similar manner to dictionaries and offer a finer degree of control over the parameters than `GlobalParams` structures but are not as granular as their `AzimuthalParams` counterparts.

It is important to note that atom pair keys are permutationally invariant, i.e. the keys `(1, 6)` and `(6, 1)` are redundant and will overwrite one another like so:
```julia
julia> test = AtomicParams((1, 6)=>10., (6, 1)=>1000.)
AtomicParams{Float64} with 1 entries:
  (1, 6) => 1000.0

julia> test[(1, 6)] == test[(6, 1)] == 1000.0
true
```
Finally atomic numbers will be sorted so that the lowest atomic number comes first.
However, this is only a superficial visual change and queries will still be invariant to permutation.

#### Azimuthal Resolved

Parameters specified for each azimuthal quantum number of each species.
This allows for a finer degree of control and is a logical extension of the `AtomicParams` structure.
It is important to note that `AzimuthalParams` instances must be supplied with a basis definition.
This allows it to work out the azimuthal quantum number associated with each shell during lookup.
```julia
# Basis definition describing a H_1s C_2s1p basis set
julia> basis_def = Dict(1=>[0], 6=>[0, 0, 1])
julia> p_on = AzimuthalParams(
    basis_def, (1, 0, 0)=>1, (6, 0, 0)=>2, (6, 0, 1)=>3, (6, 1, 1)=>4)
AzimuthalParams{Int64} with 4 entries:
  (6, 0, 0) => 2
  (1, 0, 0) => 1
  (6, 1, 1) => 4
  (6, 0, 1) => 3

julia> p_off = AzimuthalParams(
    basis_def, (1, 1, 0, 0)=>1, (6, 6, 0, 0)=>2, (6, 6, 0, 1)=>3, (6, 6, 1, 1)=>4,
    (1, 6, 0, 0)=>6, (1, 6, 0, 1)=>6)

AzimuthalParams{Int64} with 6 entries:
    (1, 6, 0, 1) => 6
    (6, 6, 0, 1) => 3
    (1, 6, 0, 0) => 6
    (1, 1, 0, 0) => 1
    (6, 6, 1, 1) => 4
    (6, 6, 0, 0) => 2

# on-site interactions involving shells 1 % 2 will return 2 as they're both s-shells.
julia> p_on[(6, 1, 1)] == p_on[(6, 1, 2)] == p_on[(6, 2, 2)] == 2
true
```
While keys are agnostic to the ordering of the azimuthal numbers; the first atomic number `z₁` will always correspond to the first azimuthal number `ℓ₁`, i.e.:
 - `(z₁, ℓ₁, ℓ₂) == (z₁, ℓ₂, ℓ₁)`
 - `(z₁, z₂, ℓ₁, ℓ₂) == (z₂, z₁, ℓ₂, ℓ₁)`
 - `(z₁, z₂, ℓ₁, ℓ₂) ≠ (z₁, z₂ ℓ₂, ℓ₁)`
 - `(z₁, z₂, ℓ₁, ℓ₂) ≠ (z₂, z₁ ℓ₁, ℓ₂)`

#### Shell Resolved

`ShellParams` structures allow for individual values to be provided for each and everyunique interaction.
While this proved the finest degree of control it can quickly become untenable for systems with large basis sets or multiple species due the shear number of variable required.
```julia
# For H1s C2s1p basis set.
julia> p_on = ShellParams(
    (1, 1, 1)=>1, (6, 1, 1)=>2, (6, 1, 2)=>3, (6, 1, 3)=>4,
    (6, 2, 2)=>5, (6, 2, 3)=>6, (6, 3, 3)=>7)

ShellParams{Int64} with 7 entries:
  (6, 3, 3) => 7
  (1, 1, 1) => 1
  (6, 1, 3) => 4
  (6, 2, 2) => 5
  (6, 1, 1) => 2
  (6, 1, 2) => 3
  (6, 2, 3) => 6

julia> p_off = ShellParams(
    (1, 1, 1, 1)=>1, (1, 6, 1, 1)=>2, (1, 6, 1, 2)=>3, (1, 6, 1, 3)=>4,
    (6, 6, 1, 1)=>5, (6, 6, 1, 2)=>6, (6, 6, 1, 3)=>74, (6, 6, 2, 2)=>8,
    (6, 6, 2, 3)=>9, (6, 6, 3, 3)=>10)

ShellParams{Int64} with 10 entries:
  (6, 6, 2, 2) => 8
  (6, 6, 3, 3) => 10
  (6, 6, 1, 3) => 74
  (1, 1, 1, 1) => 1
  (1, 6, 1, 2) => 3
  (1, 6, 1, 1) => 2
  (1, 6, 1, 3) => 4
  (6, 6, 1, 1) => 5
  (6, 6, 1, 2) => 6
  (6, 6, 2, 3) => 9

```

### Collections
All of the parameter definitions required to specify a given interaction are collected into `ParaSet` structures.
Once instantiated, the `OnSiteParaSet` and `OffSiteParaSet` structures should contain all parameters required to construct all of the desired on/off-site bases.

#### On-site
The `OnSiteParaSet` structure holds all the `Params` instances required to construct the on-site bases.
Such structures are comprised of four parameter fields, each of which is discussed in turn below:

- `ν::Params{K, Int}`: correlation order, for on-site interactions the body order is one
  more than the correlation order.   
- `deg::Params{K, Int}`: maximum polynomial degree.
- `e_cut_out::Parameters{K, Float}`: environment's external cutoff distance, only atoms within the specified radius will contribute to the local environment. 
- `r0::Parameters{K, Float}`: scaling parameter (typically set to the nearest neighbour distances).

An example demonstrating the instantiateation of a simple `OffSiteParaSet` structure is now provided. 
```julia
# On-site parameter deceleration
on_site_parameters = OnSiteParaSet(
    # Maximum correlation order, set to 2 for all on-site bases
    GlobalParams(2),
    # Maximum polynomial degree, set to 6 for all on-site bases
    GlobalParams(6),
    # Environmental cutoff radius, set to 10.0 Å for all on-site bases
    GlobalParams(10.),
    # Scaling factor "r₀", set to 2.5 Å for all on-site bases
    GlobalParams(2.5)
)
```

#### Off-site

Likewise, the `OffSiteParaSet` structures store all parameters required to construct all of the off-site bases.

- `ν::Params{K, Int}`: correlation order, for off-site interactions the body order is two
  more than the correlation order.   
- `deg::Params{K, Int}`: maximum polynomial degree.
- `b_cut::Params{K, Float}`: cutoff distance for off-site interactions, only bonding interactions between atoms separated by a distance smaller or equal to this will be considered. 
- `e_cut_out::Params{K, Float}`: environment's external cutoff distance, this is also used as the radius for the cylindrical bond envelope.

The `OffSiteParaSet` can be instantiateated in a similar manner to their on-site counterparts like so:
```julia
# Off-site parameter deceleration
off_site_parameters = OffSiteParaSet(
    # Maximum correlation order, set to 1 for all off-site bases
    GlobalParams(1),
    # Maximum polynomial degree, set to 6 for all off-site bases
    GlobalParams(6),
    # Bond cutoff radius, set to 10.0 Å for all off-site interactions
    GlobalParams(10.),
    # Environmental cutoff radius, set to 5.0 Å for all off-site interactions
    GlobalParams(5.),
)
```

# TODO
- Add images showing the on-site and off-site environments.
- Discuss the individual parameters in more detail.