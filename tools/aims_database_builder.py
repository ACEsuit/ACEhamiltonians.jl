from typing import IO, Union
from io import TextIOWrapper
from os import remove
from os.path import join, exists, basename
import re
from h5py import File, Group, Dataset
import tarfile
import numpy as np
from ase import Atoms
from ase.io.aims import read_aims_output
from ase.io import read
from numpy import ndarray as Array
from glob import glob
from tqdm import tqdm

eV2Ha = 0.367493245336341E-01

_DNS = f'[ 0-9E.+\-]+'

# It is important to note that Python uses row-major order while Julia uses
# column-major order. This means that arrays written by one will be read as
# as the transpose of the other. For example and NxMxK tensor stored by Python
# in an HDF5 database will be read in as a KxMxN tensor by Julia and vise versa.
# Therefore, the python arrays must be transposed prior to writing them into
# the HDF5 database.


def read_hamiltonian_matrix(path):
    # Reads the gamma-point Hamiltonian matrix as produced by FHI-aims

    # Read in the file
    file = open(path, 'r').readlines()

    # Read in the matrix entries
    entries = [(int(row) - 1,int(column) - 1, float(value)) for
               line in file for column, row, value in [line.split()]]

    # Identify the number of columns/rows
    n = entries[-1][0] + 1

    # Construct the matrix
    H = np.zeros((n, n))

    # Fill in the matrix, (replace with faster method when time)
    for row, column, entry in entries:
        H[row][column] = entry


    # Aims only gives the lower triangle here so reflect across the diagonal
    # Lower & upper triangle indices with & without diagonal elements respectively.
    i_lower, i_upper,  = np.tril_indices(n, 0), np.triu_indices(n, 1)
    H[i_upper] = H.T[i_upper]

    return H


def read_overlap_matrix(path):
    # Reads the gamma-point overlap matrix as produced by FHI-aims
    # Open up the overlap matrix file
    file = open(path, 'r').readlines()

    # Check which file type this is, if it is omat.aims
    if 'omat.aims' in path:

        # Number of columns
        n = int(re.search('(?<=nsaos\=)[\d ]+(?=\n)', file[0]).group(0))

        # Number of characters per number
        c = int(re.search('(?<=format\()[\dd]+(?=\.)', file[0]).group(0).split('d')[1])

        # Remove non-data lines, concatenate, strip '\n' & replace exp term "D" with "E"
        chk = ['nsaos', '$', '#']  # Lines with these strings get removed
        data = ''.join([l for l in file if not any(s in l for s in chk)]
                       ).replace('\n', '').replace('D', 'E')

        # Split numbers by character width and convert to a numpy array of floats.
        data = np.array([float(data[i:i + c]) for i in range(0, len(data), c)])

        #Array to hold overlap matrix
        overlap = np.zeros((n, n))

        # Lower & upper triangle indices with & without diagonal elements respectively.
        i_lower, i_upper,  = np.tril_indices(n, 0), np.triu_indices(n, 1)

        # Set the lower triangle and diagonal elements of the overlap matrix
        overlap[i_lower] = data

        # Set the upper triangle of the overlap matrix equal to the lower triangle
        overlap[i_upper] = overlap.T[i_upper]

    # If the file type is overlap-matrix.out
    elif 'overlap-matrix.out' in path:
        # Read in each line, which should follow the format of row
        # column, value. This should be replaced by a faster, more elegant
        # solution.
        dat = [line.split() for line in file]
        indexs = np.array([[int(i[0]),int(i[1])] for i in dat]) - 1
        vals = np.array([float(i[2]) for i in dat])
        n = np.max(np.max(indexs, axis=0))+1
        overlap = np.zeros((n, n))
        for i, v in zip(indexs, vals):
            overlap[i[0],i[1]] = v

        overlap[np.tril_indices(n, -1)] = overlap.T[np.tril_indices(n, -1)]

    # Otherwise raise an error
    else:
        raise Exception('Unknown overlap matrix type')

    # Check that the overlap matrix obeys symmetry
    if not np.all(np.isclose(overlap, overlap.T)):
        raise Exception('Warning: Overlap matrix does not obey symmetry.')

    # Return the overlap matrix

    return overlap.T


def set_unit(target: Union[Group, Dataset], unit: str = 'n/a'):
    target.attrs['unit'] = unit


def structure(atoms: Atoms, target: Group):
    # Atomic numbers are stored in a vector named "atomic_numbers"
    target.create_dataset('atomic_numbers', data=atoms.get_atomic_numbers())

    # Positions are also stored in a similar manner. Note that the positions are
    # not explicitly transposed here as julia expects positions as a 3xN matrix;
    # the required transpose occurs naturally due to the row to column major
    # order difference between Python and Julia.
    positions = target.create_dataset('positions', data=atoms.get_positions())

    # Set the units to Angstroms
    set_unit(positions, 'Angstrom')

    # If periodic; store the lattice & periodicity of the system.
    if any(atoms.pbc):
        lattice = target.create_dataset('lattice', data=atoms.cell.array)
        set_unit(lattice, 'Angstrom')
        target.create_dataset('pbc', data=atoms.pbc)

def forces(atoms: Atoms, target: Group):
    # Store the forces into an appropriately named group and set the units.
    # Again no explicit transpose is required as it is occurs implicitly due
    # to the row->column major order change.
    set_unit(target.create_dataset('forces', data=atoms.get_forces()),
             'eV/Angstrom')

def total_energy(aims_out: IO, target: Group):
    # The read_aims_output function returns corrected energy, which is undesired,
    # and thus it must be extracted manually.
    val = float(re.search(fr'(?<=Total energy uncorrected {{6}}:){_DNS}',
                          aims_out.read()).group())
    energy = target.create_dataset('total_energy', data=val)
    set_unit(energy, 'eV')


def basis_set(atoms: Atoms, basis: IO, target: Group):
    # Read the basis set & strip out sub-shells as shells will always have a
    # fixed number of sub-shells with a fixed order. Thus storing such info is
    # unnecessary. Function number is also dropped as it is unimportant here.
    basis_text = np.array([*map(str.split, basis.readlines()[2:])])
    basis_text = basis_text[:, 1:-1][basis_text[:, -1] == '0']

    # Identify what species are present & get the index of its 1st occurrence
    unique = np.unique(atoms.get_atomic_numbers(), return_index=True)

    # Loop over the species present
    for z, idx in zip(*unique):
        # Extract the bases associated with this atom.
        bases = basis_text[basis_text[:, 1] == str(idx + 1)]
        # Extract/store the principle/azimuthal quantum numbers as integers.
        target.create_dataset(str(z), data=bases[:, -2:].astype(np.int_))


def density_of_states(dos: IO, aims_out: IO, target: Group):
    energy, value = np.fromstring(''.join(dos.readlines()[3:]), sep=' '
                                  ).reshape((-1, 2)).T
    broadening = float(list(filter(lambda l: l.startswith('  | points & broadening :'),
           aims_out.readlines()))[0].split()[-1])

    e_ds = target.create_dataset('energies', data=energy)
    v_ds = target.create_dataset('values', data=value)
    broadening = target.create_dataset('broadening', data=broadening)

    set_unit(e_ds, 'eV')
    set_unit(v_ds, '1/(eV.V_unit_cell)')
    set_unit(e_ds, 'eV')


def fermi_level(aims_out: IO, target: Group):
    value = float(re.search(fr'(?<=Chemical Potential {{26}}:){_DNS}',
                            aims_out.read()).group(0))
    fermi_level = target.create_dataset('fermi_level', data=value)
    set_unit(fermi_level, 'eV')


def k_points(aims_out: IO, target: Group):
    # Pull out the lines specifying the k-points and their weights
    lines = filter(lambda l: l.startswith('  | k-point: '),
                   aims_out.readlines())

    # Extract the numerical data
    k_points = np.array(list(map(str.split, lines))
                        )[:, [4, 5, 6, 9]].astype(float)

    target.create_dataset('k-points', data=k_points)


class Reconstructor:
    """

    Warnings:
        It is important to note that the structure of a real-space matrix will
        depend upon the code used to construct it. Thus its form may differ from
        what one might expect. The two most important considerations to be aware
        of when extracting real-space matrices from FHI-aims is that:
          - i ) the distance vector between a pair of atoms is calculated as
                `F(u,v) = u - v` rather than `F(u,v) = v - u`.
          - ii) atomic positions are mapped internally from the fractional domain
                of [0.0, 1.0) to [-0.5, 0.5).

        The first consideration means that the locations of atom blocks (i, j)
        and (j, i) may be exchanged with respect to that which one expects. The
        second condition will result in a more significant structural change.
        These two can be accounted for by i) taking the negative of the cell
        index matrix, i.e. `cell_indices = -cell_indices`, and ii) by wrapping
        the atomic coordinates to the fractional domain of [-0.5, 0.5).

        is a product of the code use to construct it; and thus may differ from
        what one might expect.

        may differ from what one might expect. Within FHI-aims the
        distance between two atoms is calculated as `F(u,v) = u - v` whereas
        in other codes it is calculated as `F(u,v) = v - u`.

    """

    def __init__(self, n_basis: int, n_cells_in_matrix: int,
                 column_index_matrix: Array, index_matrix: Array,
                 cell_indices: Array):

        self.n_basis = n_basis
        self.n_cells_in_matrix = n_cells_in_matrix
        self.column_index_matrix = column_index_matrix
        self.index_matrix = index_matrix
        self.cell_indices = cell_indices

    @classmethod
    def from_file(cls, path: Union[str, IO]) -> 'Reconstructor':
        """Initialise a `Reconstructor` instance from a file.

        This sets up and returns a `Reconstructor` instance which can then be
        used to regenerate the dense formatted matrices.

        Arguments:
            path: path to the file which contains all the indexing information
                required to reconstruct dense matrices from sparse formatted
                ones. This is typically named "rs_indices.out".

        Returns:
            reconstructor: a `Reconstructor` instance which can be used to
                regenerate the dense formatted matrices.

        Todo:
            - rewrite docstring to account for path possibly being an IO
              object.
        """
        # Load in the data
        if isinstance(path, str):
            config = open(path).readlines()
        else:
            config = path.readlines()

        # Collect size information
        n_matrix_size = int(config[0].split(':')[-1])
        n_cells_in_matrix = int(config[1].split(':')[-1])
        n_basis = int(config[2].split(':')[-1])

        # Gather the data blocks
        db1 = ''.join(config[4: 4 + n_cells_in_matrix])
        db2 = ''.join(config[5 + n_cells_in_matrix: 5 + 2 * n_cells_in_matrix])
        db3 = ''.join(config[6 + 2 * n_cells_in_matrix:
                             6 + 3 * n_cells_in_matrix])
        db4 = ''.join(config[7 + 3 * n_cells_in_matrix:
                             7 + 3 * n_cells_in_matrix + n_matrix_size])

        # Get the cell translation vectors (ignore the padding value at the end)
        cell_indices = np.fromstring(db1, sep=' ', dtype=int).reshape(-1, 3)[:-1]

        # Repeat for the matrix index data
        index_matrix = np.stack((
            np.fromstring(db2, sep=' ', dtype=int).reshape(n_cells_in_matrix, n_basis),
            np.fromstring(db3, sep=' ', dtype=int).reshape(n_cells_in_matrix, n_basis)
        ))

        # and for the matrix column data
        column_index_matrix = np.fromstring(db4, sep=' ', dtype=int)

        # Construct and return the instance
        return cls(n_basis, n_cells_in_matrix - 1, column_index_matrix,
                   index_matrix, cell_indices)

    def regenerate(self, path: IO) -> Array:
        """Reconstruct the real dense matrix from the sparse matrix data.

        This dense matrix reconstruction algorithm is implemented following the
        documentation provided in the FHI-aims "physics.f90" file.

        Arguments:
            path: a path to the sparse data file; such files are typically
             named "rs_hamiltonian.out" or "rs_overlap.out".

        Returns:
            matrix: the dense matrix.

        Warnings:
            The matrix returned by this method may not necessarily follow the
            structure that the user expects. This is because FHI-aims i) remaps
            the cell from fractional coordinates (0, 1] to (-0.5, 0.5] which
            inverts some atom positions, and ii) calculates distance via the
            form `F(x, y) = x - y` rather than `F(x, y) = y - x`. These two
            difference can result in specific atom blocks being located in
            different parts of the real space matrix than what one might
            normally expect.

        """
        # matrix_data = np.fromstring(open(path).read(), sep=' ', dtype=np.float64)
        matrix_data = np.fromstring(path.read(), sep=' ', dtype=np.float64)

        matrix = np.zeros((self.n_cells_in_matrix, self.n_basis, self.n_basis))

        # Subtract one to account for Fortran arrays starting at one rather than zero
        index_matrix = self.index_matrix - 1
        column_index_matrix = self.column_index_matrix - 1

        for i_cell_row in range(self.n_cells_in_matrix):
            for i_basis_row in range(self.n_basis):
                i_index_first = index_matrix[0, i_cell_row, i_basis_row]
                i_index_last = index_matrix[1, i_cell_row, i_basis_row]
                for i_index in range(i_index_first, i_index_last + 1):
                    i_basis_col = column_index_matrix[i_index]
                    matrix[i_cell_row, i_basis_row, i_basis_col] = matrix_data[i_index]

        # FHI-aims only fills the lower triangle of the Hamiltonian & overlap
        # matrices. Thus they must be symmetrised them.
        image_index = {tuple(k): i for i, k in enumerate(self.cell_indices)}
        triu_indicies = np.triu_indices(self.n_basis, 1)
        for cell, cell_i in image_index.items():
            mirror_cell_i = image_index[tuple(-i for i in cell)]
            matrix[(cell_i, *triu_indicies)] = matrix[mirror_cell_i].T[triu_indicies]

        return matrix


def parse_system(source: str, destination: Group):

    # The `aims.out` file will be used repeatedly
    aims_path = join(source, 'aims.out')

    # Load in the ASE.Atoms object from the 'aims.out` file. The `aims.out` file
    # is used rather than the `geometry.in` as the former ensures any "bad" atomic
    # positions are correctly folded back into the cell.
    atoms = read(aims_path)

    # Instantiate at `Reconstructor` instance which can be used to construct the
    # real space Hamiltonian and overlap matrices from the compressed row sparse
    # representation used by FHI-aims.
    matrix_builder = Reconstructor.from_file(open(join(source, 'rs_indices.out')))

    # Create "Data" and "Info" groups.
    data = destination.create_group('Data')
    info = destination.create_group('Info')

    # Add structural information
    # Save information about the system's geometry into a new "Structure" group.
    structure(atoms, destination.create_group('Structure'))

    # Append the basis set, k-points and cell translation vector information
    k_points(open(aims_path), info)
    basis_set(atoms, open(join(source, 'basis-indices.out')),
              info.create_group('Basis'))

    # The negative is taken of `cell_indices` to account for the difference in
    # convention in calculating distance vectors.
    info.create_dataset('Translations', data=-matrix_builder.cell_indices)

    # Add the results data
    forces(atoms, data)
    total_energy(open(aims_path), data)
    fermi_level(open(aims_path), data)
    density_of_states(open(join(source, 'KS_DOS_total.dat')), open(aims_path),
                      data.create_group('DoS'))

    H_matrix = matrix_builder.regenerate(open(join(source, 'rs_hamiltonian.out')))
    H_matrix = np.transpose(H_matrix, (0, 2, 1))
    H = data.create_dataset('H', data=H_matrix)
    set_unit(H, 'Ha')
    S_matrix = matrix_builder.regenerate(open(join(source, 'rs_overlap.out')))
    S_matrix = np.transpose(S_matrix, (0, 2, 1))
    data.create_dataset('S', data=S_matrix)

    H_gamma = data.create_dataset('H_gamma', data=read_hamiltonian_matrix(join(source, 'hamiltonian.out')).T)
    set_unit(H_gamma, 'Ha')
    data.create_dataset('S_gamma', data=read_overlap_matrix(join(source, 'overlap-matrix.out')).T)


# Note that this requires using the branch of FHI-aims that supports the writing
# out of the real-space matrices.
#   https://aims-git.rz-berlin.mpg.de/aims/FHIaims/-/tree/output_real_matrices

# Name of the database to create
database_name = ''

# Specify all of the FHI-aims results directories which should be parsed into
# a HDF5 database.
fhi_aims_results_directories = []

# Iterate over the directories and parse each of them into the newly created
# database `database_name`.
with File(database_name, 'w') as db:
    for directory in tqdm(fhi_aims_results_directories):
        try:
            parse_system(directory, db.create_group(basename(directory)))
        except Exception as e:
            print(directory)
            raise e from e