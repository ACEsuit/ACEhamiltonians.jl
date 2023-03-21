Individual on and off site bases may be constructed via calls to the `on_site_ace_basis` and `off_site_ace_basis` functions respectively. For on-site bases, one must specify six arguments:

 -  The azimuthal quantum numbers of the associated shells, `ℓ₁` and `ℓ₂`. For example, `ℓ₁, ℓ₂= 0, 1` would correspond to a s-p interaction.
 - The correlation order `ν`. For on-site interactions the body-order is equivalent to one greater than the correlation order; i.e. a correlation order of one would equate to a body-order of two. 
 - The maximum polynomial degree `deg` commonly a value between 2 and 20. This should be chosen via hyperparmeter optimisation.
 - The environmental cutoff distance `e_cut`. Only atoms within the specified cutoff radius will contribute to the local environment. A r_cut value of zero would result in predictions being treated as independent of their environment.
 - The scaling parameter `r0`; this is commonly set to the nearest neighbour distance.

```julia
ℓ₁, ℓ₂ = 0, 0
ν, deg = 2, 4
e_cut, r0 = 12.0, 2.5

on_site_core_basis = on_site_ace_basis(ℓ₁, ℓ₂, ν, deg, e_cut, r0)
```

The call to the off-site basis constructor is identical to its on-site counterpart. With the exception that a bond distance cutoff is provided in-place of `r0`. Which, as it name suggests, specifies the maximum distance up to which such interactions will be considered. That is to say only interactions between pairs of atoms separated by a distance less than or equal to `b_cut` will be evaluated. For off-site interactions the body-order is two greater than the correlation order. 
```julia
ℓ₁, ℓ₂ = 0, 1
ν, deg = 1, 4
b_cut, e_cut = 12.0, 5.0

off_site_core_basis = off_site_ace_basis(ℓ₁, ℓ₂, ν, deg, b_cut, e_cut)
```

In order to make use of the these bases one must wrap them in a `Basis` instance like so:
```julia
id_on = (6, 1, 1)
id_off = (6, 6, 1, 2)
on_site_ss_basis = Basis(on_site_core_basis, id_on)
off_site_sp_basis = Basis(off_site_core_basis, id_off)
```
The first argument is always the `SymmetricBasis` object returned by the on/off-site ACE basis constructor functions. The second argument is the basis `id`, this is used to identify what interaction the basis is associated with. An `id` of the form `(z, i, j)` indicates a basis represent the on-site interaction between the `i`ᵗʰ and `j`ᵗʰ shells of species `z`. Whereas an `id` of the form `(z₁, z₂, i, j)` indicates the basis represents an off-site interaction between the `i`ᵗʰ shell of species `z₁` and the `j`ᵗʰ shell of species `z₂`. Shell indices are used in place of azimuthal quantum numbers to avoid ambiguities arising from the use of non-minimal basis sets.

These `Basis` structures contain for members, `basis`, `id`, `coefficients` and `mean`. With the first two already having been discussed. The latter two `coefficients` and `mean` are set during the fitting process and are used only when making predictions.
