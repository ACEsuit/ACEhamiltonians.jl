import os.path
import re

import glob
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from typing import Optional, Union


class BandPaths:
    def __init__(self, band_paths):
        self.band_paths = band_paths

    @classmethod
    def build_path(cls, directory_path: str, symbols: list[tuple[str, str]],
                   file_format: Optional[str] = None):
        """Instantiates a `BandPaths` instance from a given directory.

        Arguments:
            directory_path: directory in which the requisite data files are
                located.
            symbols: a list of tuples of the form (i, j) where the i'th & j'th
                elements of the n'th tuple specify the labels for the start
                and end point of the n'th path segment. This is mostly used
                as a label.
            file_format: specifies how data should be loaded and from what
                files. Available options are:
                    - 'aims': load from FHI-aims "band1xxx.out" files.
                    - 'dftbp': load from DFTB+ "band.out" & "dftb_in.hsd" files.
                Omitted then an attempt will be made to detect the appropriate
                format.
        """
        # Resolve any user relative pathing
        directory_path = os.path.expanduser(directory_path)

        # Auto-detect the correct file-format if unspecified.
        if file_format is None:
            if os.path.exists(os.path.join(directory_path, 'dftb_in.hsd')):
                file_format = 'dftbp'
            elif os.path.exists(os.path.join(directory_path, 'control.in')):
                file_format = 'aims'
            else:
                file_format = 'dat'

        # Call the corresponding loader if it exists
        if file_format.lower() == 'aims':
            return cls._build_from_aims(directory_path, symbols)
        elif file_format.lower() == 'dftbp':
            return cls._build_from_dftbp(directory_path, symbols)
        elif file_format.lower() == 'dat':
            return cls._build_from_dat(directory_path, symbols)
        # Otherwise throw an error
        else:
            raise ValueError('Unknown format specified')

    @classmethod
    def _build_from_aims(cls, directory_path, symbols: list[tuple[str, str]]):
        # Locate all of the band1xxx.out files present in the specified
        # directory and order them correctly.
        band_path_files = sorted(
            glob.glob(os.path.join(directory_path, 'band1*.out')),
            key=lambda x: int(
                re.search(r'\d{4}', os.path.basename(x)).group(0)[1:4]))

        # Check that the number of paths indicated by `symbols` matches the
        # number of files found.
        assert len(band_path_files) == len(symbols)

        # Parse the files into `BandPath` objects
        return cls([BandPath.from_file(file, symbol, file_format='aims')
                    for file, symbol in zip(band_path_files, symbols)])

    @classmethod
    def _build_from_dftbp(cls, directory_path, symbols: list[tuple[str, str]]):
        return cls(BandPath.from_file(
            os.path.join(directory_path, 'band.out'), symbols,
            file_format='dftbp'))

    @classmethod
    def _build_from_dat(cls, directory_path, symbols: list[tuple[str, str]]):
        # Locate all of the band_x.out files present in the specified
        # directory and order them correctly.
        band_path_files = sorted(
            [f for f in glob.glob(os.path.join(directory_path, 'band_*.dat'))
             if re.search(r'band_\d+.dat', f)],
            key=lambda x: int(
                re.search(r'(?<=band_)\d+', os.path.basename(x)).group(0)))

        # Check that the number of paths indicated by `symbols` matches the
        # number of files found.
        assert len(band_path_files) == len(symbols)

        # Parse the files into `BandPath` objects
        return cls([BandPath.from_file(file, symbol, file_format='dat')
                    for file, symbol in zip(band_path_files, symbols)])

    def plot(
            self, energy_offset: float = 0.0,
            eigen_filter: Optional[slice] = None,
            set_ylim: Optional[list[float]] = None,
            energy_unit: str = 'eV', fmt='k-',
            figure: plt.Figure = None, axes: list[plt.Subplot] = None
        ) -> tuple[plt.Figure, list[plt.Subplot]]:
        """Plot band structure.

        Generate a band structure plot or add band structure data onto an
        already existing plot.

        Arguments:
            - energy_offset: if provided, then all energies are provided relative
                to this value. Commonly this is set to the fermi level.
            - eigen_filter: a slicer object may be supplied to filter out all but
                the desired energy levels from the band structure. For example the
                slicer `slicer(1,-1)` would omit the highest and lowest energy
                levels. This is of use when wanting to clear up the band structure
                plot.
            - set_ylim: allows for the y-axis limits to be set; i.e. the maximum
                and minimum energy values displayed on the plot. These values
                should be relative to the ``energy_offset`` if provided. Note that
                this is just a passthroughs for the `AxesSubplot.set_ylim` call.
            - energy_unit: a string specifying the energy unit used. This is only
                used when setting the y-axis label. Defaults to 'eV'.

        Keyword Arguments:
            - figure: `plt.Figure` entity into which the data is to be plotted.
            - axes: accompanying list of `plt.Subplot` entities.

        Returns:
            - figure: figure in which the plot is stored.
            - axes: sub-plots for each segment of the band-structure.

        Notes:
            If the `figure` and `axes` keyword arguments are supplied then the
            data will be plotted on to the figure provided. Otherwise a new
            plot will be generated.

            This currently does not support non-continuous band structures.
        """

        # If adding data onto an already existing plot then ensure that both
        # `figure` and `axes` are provided as they are mutually inclusive.
        if figure is None != axes is None:
            raise ValueError('The arguments `figure` & `axes` are mutually '
                             'inclusive; i.e. either both are specified or '
                             'neither are.')

        # Construct figure and sub-plots if none were provided.
        if figure is None or axes is None:
            # Construct a series of sub-plots; specifically one for each band-path.
            figure, axes = plt.subplots(
                1, len(self),
                # Make the widths of the sub-plots proportional to the lengths of the
                # band-paths the represent.
                gridspec_kw={
                    'width_ratios': np.array([p.path_length for p in self])},
                # Ensure a common y-axis is used throughout
                sharey=True,
                dpi=600)

        # Loop over the segments of the band-path
        for n, (axis, band_path) in enumerate(zip(axes, self), start=1):

            # Plot the data; filtering and offsetting the energy levels as needed
            axis.plot(band_path.segment_lengths,
                      band_path[eigen_filter].energies - energy_offset, fmt,
                      linewidth=0.8)

            # Remove the x-axial ticks but allow the symbols to remain
            axis.tick_params(axis='x', which='both', bottom=False, top=False,
                             labelbottom=True)

            # Remove the y-axial ticks on the sub-plots as they are redundant
            axis.tick_params(axis='y', which='both', left=False, right=False,
                             labelleft=False)

            # Remove internal x-axial padding so that the sub-plots join up to
            # look like a single plot.
            axis.set_xlim(0, 1)

            if set_ylim is not None:
                axis.set_ylim(set_ylim)

            # Label the start of each band-path with the appropriate symbol. The
            # end label is omitted for all by the last band-path in the series as
            # it is redundant to the start point of the path which follows it.
            if n != len(self):
                axis.set_xticks([0])
                axis.set_xticklabels([band_path.symbols[0]])
            else:
                # Need to set the end label for the last band-path
                axis.set_xticks([0, 1])
                axis.set_xticklabels([*band_path.symbols])

        # Remove external x-axial padding to help knit the sub-plots together
        plt.subplots_adjust(wspace=.0)

        # Pad left side to prevent the y-axis label from being cut off or
        # overlapping with the tick labels.
        plt.subplots_adjust(left=0.12)

        # Add ticks back to the y-axis of the first sub-plot
        axes[0].tick_params(axis='y', which='both', left=True, right=False,
                            labelleft=True)

        # Place the global x and y axis labels
        figure.supylabel(f'Energy ({energy_unit})')
        figure.supxlabel(r'k-points')

        # Finally return the plot
        return figure, axes

    def __getitem__(self, index) -> 'BandPath':
        """Select element an of `self.band_paths` via an inded.

        Allows for selection of a specific element of `self.band_paths` via
        indexing and permits iteration.
        """
        return self.band_paths[index]

    def __len__(self):
        """Number of band-path segments"""
        return len(self.band_paths)

    def __str__(self):
        points = [self.band_paths[0].symbols[0]]
        for path in self.band_paths:
            start, end = path.symbols
            if points[-1] != start:
                points.append('|')
                points.append(start)

            points.append('-')
            points.append(end)

        return ''.join(points)

    def __repr__(self):
        return str(self)


class BandPath:
    """

    Arguments:
        energies: an n×ε array specifying the ε eigen values associated with
            each of the n sample points taken along the band-path.
        k_points: an n×3 array specifying the k-points at which the band-path
            is sampled. Note that the order of the provided k-points must be
            sequential and free of any periodic boundary crossings.
        symbols: a tuple of strings used to label the start and end points of
            the band path. Typically the strings are just a pair of characters
            such as `('U', 'W')`.
    """
    def __init__(
            self, energies: np.ndarray, k_points: np.ndarray,
            symbols: tuple[str, str]):

        # Attribute Assignments
        # ---------------------
        self.energies = energies
        self.k_points = k_points
        self.symbols = symbols

        # Error Handling
        # --------------
        # Confirm that `energies` is of an expected shape
        if self.energies.ndim != 2:
            raise ValueError(
                'energies should be an n×ε array; aka number of k-points '
                'by number of eigen-values')

        # Verify `k_points` has an appropriate shape
        if self.k_points.ndim != 2 or self.k_points.shape[1] != 3:
            raise ValueError('k_points must be an n×3 array')

        # Ensure that the number of k-points supplied matches up with the
        # number of energy values.
        if self.k_points.shape[0] > self.energies.shape[0]:
            raise ValueError(
                'k-point count exceeds the number of energy values')
        if self.k_points.shape[0] < self.energies.shape[0]:
            raise ValueError(
                'energy value count exceeds the number of k-points')

        # Check that the k-points are sequential and do not cross PBC
        if not np.all(np.diff(
                np.sign(np.diff(self.k_points, axis=0)), axis=0) == 0):
            raise ValueError(
                'k_points must be sequential and free of any PBC crossings')

        # Check the symbols attribute is correct
        if len(self.symbols) != 2:
            raise ValueError('the symbols tuple should be of length 2')

    @property
    def k_start(self) -> float:
        """Start point of the band-path in k-space"""
        return self.k_points[0]

    @property
    def k_end(self) -> float:
        """End point of the band-path in k-space"""
        return self.k_points[-1]

    @property
    def path_length(self) -> float:
        """Length of the band path in fractional units"""
        return np.linalg.norm(self.k_start - self.k_end)

    @property
    def segment_lengths(self):
        """Fractional length of each point along the band path"""
        return np.array([
            np.linalg.norm(k-self.k_points[0]) for k in self.k_points]
        ) / self.path_length

    @property
    def n_k_points(self) -> int:
        """Number of points sampling the band-path"""
        return len(self.k_points)

    def __reversed__(self) -> 'BandPath':
        """Reverse the band-path

        Return a copy of the `BandPath` in which the direction of the band
        path has been reversed.

        """
        return self.__class__(
            self.energies[-1:0:-1, :],
            self.k_points[-1:0:-1, :],
            (self.symbols[1], self.symbols[0]))

    def __getitem__(self, index: Union[slice, None]) -> 'BandPath':
        """Filter out all but the specified eigen-values.

        Return a copy of the supplied `BandPath` with all but the selected
        eigen-values removed. This is useful when needing to remove low-lying
        core-states, or high-lying vacant states, to make the graphs scale
        more manageable.

        """
        if index is not None:
            return self.__class__(
                self.energies[:, index],
                self.k_points,
                self.symbols)
        else:
            return self

    def __str__(self):
        """Return a descriptive string"""
        return f'{self.__class__.__name__}({self.symbols[0]}->{self.symbols[1]})'

    def __repr__(self):
        """Enable meaningful representation"""
        return str(self)

    @classmethod
    def from_file(cls, path, symbols, file_format='aims'):
        """Parse band path data into a `BandPath` instance, or a list thereof.

        Arguments:
            path: path to the relevant band-path data file.
            symbols: for FHI-aims this should be a single tuple specifying
                the labels for the start and end point of the path; i.e.
                ('Γ', 'L'). For DFTB+ a list of such tuples should be
                provided.

        Notes:
             If loading a DFTB+ band-path then a list of `BandPath` entities
             will be returned, rather than a single entity.
        """

        if file_format == 'aims':
            data = np.loadtxt(path)
            return cls(data[:, 5::2], data[:, 1:4], symbols)
        elif file_format == 'dat':
            data = np.loadtxt(path)
            return cls(data[:, 3:],   data[:, 0:3], symbols)
        elif file_format == 'dftbp':
            return cls._from_dftbp(path, symbols)
        else:
            raise ValueError(f'Unknown file format supplied: "{file_format}"')

    @classmethod
    def _from_dftbp(cls, band_path, symbols):
        """Construct `BandPath` instances from a DFTB+ calculation.
        """

        dftb_input_file = os.path.join(
            os.path.dirname(band_path), 'dftb_in.hsd')

        # Read in the dftb_in.hsd file into a single string
        txt = open(dftb_input_file, 'r').read().lower()
        # Identify the text present in the `KPointsAndWeights` line and ensure
        # that the keyword `Klines` is specified. This text is used by 're' to
        # locate the start of the data block.
        start_txt = re.search('.*kpointsandweights.*klines.*', txt).group(0)
        # Slice out the contents of said data block
        block_txt = re.search(f'(?<={start_txt}).*?(?=}})', txt, re.DOTALL).group(0)
        # Parse the contents into a numpy array
        path_info = np.array(
            [list(map(float, i.split())) for i in
             re.findall(
                 r'[0-9]+[\s\d\.?\d+]+',
                 block_txt)])

        # Pull out the start and end points for each segment of the path, along
        # with the number of points in each segment.
        k_counts = path_info[:, 0].astype(int)
        k_points = path_info[:, 1:]

        # Read in the band structure file as a list of lines
        lines = open(band_path, 'r').readlines()
        # Locate where each k-point data block starts/ends
        splits = [
            n for n, line in enumerate(lines) if line.startswith(' KPT')
        ] + [len(lines)]

        # Pull out the lines associated with each k-point data block
        blocks = [lines[s+1:e-1] for s, e in zip(splits[:-1], splits[1:])]

        # Parse the values into a single array.
        data = np.array([
            [float(line.split()[1]) for line in block] for block in blocks])

        band_paths = []

        v_start, k_start = 0, None

        for n, (k_point, k_count) in enumerate(zip(k_points, k_counts)):
            if k_count != 1:
                band_paths.append(
                    cls(
                        data[v_start:v_start + k_count + 1],
                        np.linspace(k_start, k_point, k_count + 1),
                        symbols[n-1])
                )

                v_start += k_count

            k_start = k_point

        return band_paths


# Paths specifying the location of the directories in which data is stored
data_path = '/home/ajmhpc/Projects/ACEhamiltonians/Code/Working/Tools'


# Symbols specifying the labels of the start and end point of each path; one
# component for each path-segment.
symbols = [('W', 'X'), ('X', 'Γ'), ('Γ', 'L'),
           ('L', 'K'), ('K', 'Γ'), ('Γ', 'L')]

# Parse the band path data in to `BandPath`/`BandPaths` objects
band_path = BandPaths.build_path(data_path, symbols)

# Construct a plot showing one of the band-paths
figure, axes = band_path.plot(fmt='k-')

plt.show()