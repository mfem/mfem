"""File to load data from EFIT or GS file, which contains GS solver output data.
Contains interpolators which can be used to load e.g. in Firdrake, PETSc
functions."""
from os.path import splitext
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import rotate
from scipy.spatial import Delaunay
from scipy.linalg import lstsq
from scipy.interpolate import RegularGridInterpolator, interp1d, \
    LinearNDInterpolator, CloughTocher2DInterpolator, CubicHermiteSpline
from tokamak_utils import tokamak_data_handler

#pylint: disable=invalid-name

__all__ = ["Equilibrium_data_handler", "poloidal_mode_creator", 
           "polynomial_fit_2D", "get_wall_patch_conditionals"]

debug_separatrix = False
debug_skip_B_phi = False
debug_polynomial_multi_fit_regions = False
debug_polynomial_multi_fit_psi = False
debug_polynomial_multi_fit_B = False

class EquilibriumData:
    """Load equilibrium data from GS files, EFIT files, or from .geqdsk files.
    Note: not all setups will return the same arguments! See the internal
    methods to check which ones are provided dependning on the underlying
    files. The most extensive ones are the EFIT and .geqdsk loading methods,
    which return:
    Domain/grid geometry      nr, nz, dr, dz, min_r, min_z, max_r, max_z
    Field data                psi, fpol, pres, ffprim, pprime,
    M-axis, separatrix data   Maxis_r, Maxis_z, psi_Maxis, psi_Boundary.
    Inner wall data           fw_coords"""
    nr = 0
    nz = 0
    dr = 0.0
    dz = 0.0
    min_r = 0.0
    max_r = 0.0
    min_z = 0.0
    max_z = 0.0

    psi = np.array([])
    fpol = np.array([])   # g(\psix
    pres = np.array([])   # p(\psi)
    ffprim = np.array([]) # dg(\psi)/d\psi
    pprime = np.array([]) # dp(\psi)/d\psi

    def __init__(self,filename,option="EFIT"):
        if option == "EFIT":
            self.loadEFITfile(filename)
        elif option == "GSinput":
            self.loadGSinput(filename)
        elif option == 'G eqdsk':
            self.loadGeqdskfile(filename)
        else:
            raise ValueError("Unrecognized file format. Currently only EFIT, "\
                             "GSinput, and G eqdsk formats are supported.")

    def loadEFITfile(self, filename):
        """Load data from EFIT file"""
        f = open(filename,"rb")
        self.ids = np.fromfile(f,dtype='>S10',count=6)
        dummy_int = np.fromfile(f,dtype='>i',count=1)[0]
        self.nr = np.fromfile(f,dtype='>i',count=1)[0]
        self.nz = np.fromfile(f,dtype='>i',count=1)[0]
        r_range = np.fromfile(f,dtype='>d',count=1)[0]
        z_range = np.fromfile(f,dtype='>d',count=1)[0]
        self.R_toroidal_Maxis = np.fromfile(f,dtype='>d',count=1)[0]
        self.min_r = np.fromfile(f,dtype='>d',count=1)[0]
        z_center = np.fromfile(f,dtype='>d',count=1)[0]

        self.dr = r_range/(self.nr)
        self.dz = z_range/(self.nz)
        self.max_r = self.min_r + r_range
        self.min_z = z_center - z_range/2.0
        self.max_z = z_center + z_range/2.0

        self.Maxis_r = np.fromfile(f,dtype='>d',count=1)[0]
        self.Maxis_z = np.fromfile(f,dtype='>d',count=1)[0]
        self.psi_Maxis = np.fromfile(f,dtype='>d',count=1)[0]
        self.psi_Boundary = np.fromfile(f,dtype='>d',count=1)[0]
        self.B_toroidal_Maxis = np.fromfile(f,dtype='>d',count=1)[0]

        self.Ip = np.fromfile(f,dtype='>d',count=1)[0]
        simag = np.fromfile(f,dtype='>d',count=1)[0]
        dummy_double = np.fromfile(f,dtype='>d',count=1)[0]
        Maxis_r = np.fromfile(f,dtype='>d',count=1)[0]
        dummy_double = np.fromfile(f,dtype='>d',count=1)[0]

        Maxis_z = np.fromfile(f,dtype='>d',count=1)[0]
        dummy_double = np.fromfile(f,dtype='>d',count=1)[0]
        sibry = np.fromfile(f,dtype='>d',count=1)[0]
        dummy_double = np.fromfile(f,dtype='>d',count=1)[0]
        dummy_double = np.fromfile(f,dtype='>d',count=1)[0]

        self.fpol=np.fromfile(f,dtype='>d',count=self.nr)
        self.pres=np.fromfile(f,dtype='>d',count=self.nr)
        self.ffprim=np.fromfile(f,dtype='>d',count=self.nr)
        self.pprime=np.fromfile(f,dtype='>d',count=self.nr)
        nPoints = self.nr * self.nz
        self.psi=np.fromfile( \
            f,dtype='>d',count=nPoints).reshape((self.nr,self.nz),order='F')

        self.q=np.fromfile(f,dtype='>d',count=self.nr)
        try:
            num_fw_nodes = np.fromfile(f,dtype='>i',count=1)[0]
            num_lim_nodes = np.fromfile(f,dtype='>i',count=1)[0]
            self.fw_coords = np.fromfile(f,dtype='>d',count=num_fw_nodes*2)
            self.lim_coords = np.fromfile(f,dtype='>d',count=num_lim_nodes*2)
            num_vv_nodes = np.fromfile(f,dtype='>i',count=1)[0]
            self.vv_coords = np.fromfile(f,dtype='>d',count=num_vv_nodes*2)
        except IndexError:
            pass
        f.close()

    def loadGSinput(self, filename):
        """Load data from GS solver"""
        f = open(filename,"rb")
        self.nr = np.fromfile(f,dtype='>i',count=1)[0]
        self.nz = np.fromfile(f,dtype='>i',count=1)[0]
        self.min_r = np.fromfile(f,dtype='>d',count=1)[0]
        self.max_r = np.fromfile(f,dtype='>d',count=1)[0]
        self.min_z = np.fromfile(f,dtype='>d',count=1)[0]
        self.max_z = np.fromfile(f,dtype='>d',count=1)[0]

        self.dr = (self.max_r - self.min_r)/(self.nr -1)
        self.dz = (self.max_z - self.min_z)/(self.nz -1)
        nPoints = self.nr * self.nz

        self.psi=np.fromfile( \
            f,dtype='>d',count=nPoints).reshape((self.nr,self.nz),order='F')
        self.fpol=np.fromfile(f,dtype='>d',count=self.nr)
        self.pres=np.fromfile(f,dtype='>d',count=self.nr)
        self.ffprim=np.fromfile(f,dtype='>d',count=self.nr)
        self.pprime=np.fromfile(f,dtype='>d',count=self.nr)
        f.close()

    def loadGSoutput(self, filename, header):
        """Currently only working with binary files generated with GS solvers.
        In future GS solver should output in G-EQDSK file format."""

        file = open(header,"r")
        nr, nz = file.readline().split()
        self.nr = int(nr); self.nz = int(nz)
        min_r, max_r, min_z, max_z = file.readline().split()
        file.close()
        self.min_r = float(min_r)
        self.max_r = float(max_r)
        self.min_z = float(min_z)
        self.max_z = float(max_z)

        self.dr = (self.max_r - self.min_r)/(self.nr - 1)
        self.dz = (self.max_z - self.min_z)/(self.nz - 1)
        nPoints = self.nr * self.nz

        f = open(filename,"rb")
        f.seek(8) # Skip PETSC vector id
        self.psi = np.fromfile( \
            f, np.dtype(">d")).reshape((self.nr,self.nz),order='F')
        f.close()

    def loadGeqdskfile(self, filename):
        """Load data from G-eqdsk file."""
        geqdsk_data = read_geqdsk(filename)

        self.nr = geqdsk_data['nx']
        self.nz = geqdsk_data['ny']
        r_range = geqdsk_data['rdim']
        z_range = geqdsk_data['zdim']
        self.min_r = geqdsk_data['rgrid1']
        z_center = geqdsk_data['zmid']

        self.dr = r_range/(self.nr)
        self.dz = z_range/(self.nz)
        self.max_r = self.min_r + r_range
        self.min_z = z_center - z_range/2.0
        self.max_z = z_center + z_range/2.0

        self.Maxis_r = geqdsk_data['rmagx']
        self.Maxis_z = geqdsk_data['zmagx']
        self.psi_Maxis = geqdsk_data['simagx']
        self.psi_Boundary = geqdsk_data['sibdry']
        self.fpol = geqdsk_data['fpol']
        self.pres = geqdsk_data['pressure']
        self.ffprim = geqdsk_data['ffprim']
        self.pprime = geqdsk_data['pprime']
        self.psi = geqdsk_data['psi'].transpose()
        self.q = geqdsk_data['qpsi']

        self.sep_coords = np.array([j for i in zip(geqdsk_data['rbdry'],
                                                  geqdsk_data['zbdry'])
                                        for j in i])
        self.fw_coords = np.array([j for i in zip(geqdsk_data['xlim'],
                                                  geqdsk_data['ylim'])
                                       for j in i])


class Equilibrium_data_handler():
    """Class to load GS or EFIT data and build interpolators for the prognostic
    variables.
    :arg file_name: name of EFIT or GS file to load data from
    :arg option: option to pass to EquilibriumData class
    :arg rectangular_domain: switch on whether the outer wall is immersed
    within the EFIT data's rectangular computational domain, or if it is equal
    to the latter; defaults to False.
    :arg psi_interp_delaunay: can either use regular grid interpolator, which is
    fast, or Delaunay interpolator, which takes a little longer to build. The
    former may lead to interpolation errors near outer wall boundary as it uses
    0 psi values from outside outer wall. Arg defaults to True.
    :arg psiBgradp_linearnd: switch on using LinearND or CloughTocher2D inter-
    polators; defaults to LinearND. In practice, see small difference for some
    boundary values.
    :arg gp_linear: switch on using cubic Hermite interpolation for p and g,
    based on p and dp/dpsi (and g and dg/dpsi). Alternatively, use linear
    interpolation. Defaults to True (i.e. linear interpolation).
    :arg build_B, build_B_with_fit: build B on rz_coord grid inside outer wall,
    given psi and g values, Also use this B for an interpolator. Defaults to
    True. Optionally, can build B using polynomial fit from psi; defaults to
    False.
    :arg verbose: Include printed output at various execution steps; defaults
    to True.
    :arg skip_2D_interpolators: switch to skip building 2D interpolators; useful
    e.g. when only building B.
    :arg filler_value: Value used in interpolation for regions outside outer
    wall; defaults to 0
    :arg separatrix_data: 3-tuple to be passed to
    _compute_loop_separatrix_contour method
    :arg by_r_profile: provide simple (B_r, B_phi, B_z) = (0, 1/r, 0) profile
    instead of actual EFIT data; defaults to False.
    :args inner_wall_fac, out_wall_fac, wide_r_only: extend/shring inner/outer
    wall by a given factor (with respect to wall's centroid); default to 1.
    Optionally only extend in r-direction; defaults to False."""
    def __init__(self, file_name, option="EFIT", rectangular_domain=False,
                 psi_interp_delaunay=True, psiBgradp_linearnd=True, 
                 gp_linear=True, build_B=True, build_B_with_fit=False, 
                 verbose=True, skip_2D_interpolators=False, filler_value=0,
                 separatrix_data=None, by_r_profile=False, inner_wall_fac=1,
                 outer_wall_fac=1, wide_r_only=False):

        self.EFIT_dir = 'Temp_EFIT'
        npy_file_name = self.EFIT_dir + '/npy/' + file_name
        file_name = self.EFIT_dir + '/' + file_name
        self.file_name = file_name

        self.rect_domain = rectangular_domain
        self.psi_interp_delaunay = psi_interp_delaunay
        self.skip_2D_interpolators = skip_2D_interpolators
        self.separatrix_data = separatrix_data
        self.psiBgradp_linearnd = psiBgradp_linearnd
        self.build_B_with_fit = build_B_with_fit
        self.verbose = verbose
        self.filler_value = filler_value

        self.wall_from_data = False
        self.inner_wall_fac = inner_wall_fac
        self.outer_wall_fac = outer_wall_fac
        self.wide_r_only = wide_r_only
        self.return_coarse = 1

        self.built_tri = False
        self.tri = None
        self.built_grad_p_interpolators = False

        if skip_2D_interpolators:
            print("Building EFIT data handler without 2D interpolators; to "\
                   "include interpolators set skip_2D_interpolators=False.")
        self.psi_interp = None
        self.B_r_interp = None
        self.B_phi_interp = None
        self.B_z_interp = None
        self.B_array = None

        # Load EFIT data
        if verbose:
            print("Load EFIT data")
        self.EqData = EquilibriumData(file_name, option=option)
        self.maxis_coord = np.array((self.EqData.Maxis_r, self.EqData.Maxis_z))

        # Build coordinate array
        self.rz_coords, self.r_array, self.z_array, self.res_rz, \
            self.r_extend, self.z_extend = self._compute_rz_array_data()
        if np.linalg.norm(self.maxis_coord) < 1e-10:
            self.rz_center = np.array([np.average(self.r_extend),
                                       np.average(self.z_extend)])
        else:
            self.rz_center = self.maxis_coord

        # Build 1D interpolators for g, p
        if verbose:
            print("Build 1D interpolators")
        self.psi_min, self.psi_max = \
            self.EqData.psi_Maxis, self.EqData.psi_Boundary
        self.flux_grid = np.linspace(self.psi_min, self.psi_max, self.EqData.nr)

        if gp_linear:
            self.g_interp = interp1d(self.flux_grid, self.EqData.fpol)
            self.p_interp = interp1d(self.flux_grid, self.EqData.pres)
        elif not gp_linear:
            self.g_interp = CubicHermiteSpline(self.flux_grid, self.EqData.fpol,
                                               self.EqData.ffprim)
            self.p_interp = CubicHermiteSpline(self.flux_grid, self.EqData.pres,
                                               self.EqData.pprime)

        self.build_B_psi_default = [self.EqData.psi, separatrix_data,
                                    by_r_profile, build_B, npy_file_name]
        self.build_B_psi_interpolators(*self.build_B_psi_default,
                                       skip_2D_interpolators)

    def build_B_psi_interpolators(self, psi=None, separatrix_data=None,
                                  by_r_profile=False, build_B=True,
                                  npy_file_name=None,
                                  skip_2D_interpolators=False,
                                  skip_separatrix_computation=False,
                                  skip_B_phi=False):
        """Set up B and psi interpolators based on internal self.psi array,
        and also build B array. Arguments are as in init method, except for
        skip_separatrix_compution, which will skip the computation of the
        separatrix, and skip_B_phi (useful when done already, e.g. when using a
        different psi with same separatrix, such as perturbations to original
        psi)."""
        if psi is not None:
            self.set_psi(psi)

        # Build separatrix loop contour and corresponding polygon object
        if not skip_separatrix_computation:
            if self.verbose:
                print("Build separatrix loop")

            if separatrix_data is not None:
                self.cntr_separatrix = \
                    self._compute_loop_separatrix_contour(*separatrix_data)
            else:
                self.cntr_separatrix = self._compute_loop_separatrix_contour()
            self.polygon_separatrix = Polygon(self.cntr_separatrix)

        # Compute B field
        if by_r_profile:
            self.B_array = np.zeros((*self.psi.shape, 3))
            B_array_phi = np.ones_like(self.psi)
            r_array = np.reshape(self.rz_coords, (*self.psi.shape, 2))[:, :, 0]
            self.B_array[:, :, 1] = B_array_phi/r_array
        elif build_B:
            if self.verbose:
                print("Computing B field")

            if npy_file_name is None:
                npy_file_name = self.npy_file_name
            elif npy_file_name[:9] != self.EFIT_dir:
                npy_file_name = self.EFIT_dir + '/npy/' + npy_file_name
            B_arr_name = '{0}_B_array.npy'.format(splitext(npy_file_name)[0])
            try:
                self.B_array = np.load(B_arr_name)
            except FileNotFoundError:
                self.B_array = self._compute_B_rphiz(skip_B_phi=skip_B_phi,
                                                     verbose=self.verbose)
                # Save to avoid recomputing
                np.save(B_arr_name, self.B_array)

        # Build 2D interpolators for psi, B
        if not skip_2D_interpolators:
            self.build_2d_interpolators(build_B)

    def set_psi(self, psi_):
        """Set psi to specified array"""
        setattr(self, "psi", psi_)

    def build_2d_interpolators(self, build_B=True, grad_p=False):
        """Build 2D interpolators for psi and B. Optionally instead build
        2D interpolator for grad_p (not built by default since not always
        needed)."""
        # Build Delauny triangulation for psis within outer wall only
        psi_flat = self.psi.flatten()

        # Use EqData psi as outer wall indicator function
        psi_bool = (self.EqData.psi.flatten() != 0)

        psi_flat_inner = psi_flat[psi_bool]
        if not self.built_tri:
            rz_coords_inner = self.rz_coords[psi_bool]
            self.tri = Delaunay(rz_coords_inner)
            self.built_tri = True

        if grad_p:
            print("Build 2D interpolator for grad p")
            self.built_grad_p_interpolators = True
            if self.psi_interp_delaunay:
                p_grad_array = self.compute_p_grad(verbose_=self.verbose)
                dp_flat_dr = p_grad_array[:, :, 0].flatten()
                dp_flat_dz = p_grad_array[:, :, 2].flatten()

                dp_flat_dr_inner = dp_flat_dr[psi_bool]
                dp_flat_dz_inner = dp_flat_dz[psi_bool]

                if self.psiBgradp_linearnd:
                    self.dp_dr_interp = \
                        LinearNDInterpolator(self.tri, dp_flat_dr_inner)
                    self.dp_dz_interp = \
                        LinearNDInterpolator(self.tri, dp_flat_dz_inner)
                else:
                    self.dp_dr_interp = \
                        CloughTocher2DInterpolator(self.tri, dp_flat_dr_inner)
                    self.dp_dz_interp = \
                        CloughTocher2DInterpolator(self.tri, dp_flat_dz_inner)
            else:
                raise NotImplementedError('Regular Grid Interpolator not '\
                                          'implemented for grad p.')
            return None

        print("Build 2D interpolators for psi (and optionally B)")
        if self.psi_interp_delaunay:
            if build_B:
                B_flat_r = self.B_array[:, :, 0].flatten()
                B_flat_phi = self.B_array[:, :, 1].flatten()
                B_flat_z = self.B_array[:, :, 2].flatten()

                B_flat_r_inner = B_flat_r[psi_bool]
                B_flat_phi_inner = B_flat_phi[psi_bool]
                B_flat_z_inner = B_flat_z[psi_bool]

            if self.psiBgradp_linearnd:
                self.psi_interp = LinearNDInterpolator(self.tri, psi_flat_inner)
                if build_B:
                    self.B_r_interp = \
                        LinearNDInterpolator(self.tri, B_flat_r_inner)
                    self.B_phi_interp = \
                        LinearNDInterpolator(self.tri, B_flat_phi_inner)
                    self.B_z_interp = \
                        LinearNDInterpolator(self.tri, B_flat_z_inner)
            else:
                self.psi_interp = \
                    CloughTocher2DInterpolator(self.tri, psi_flat_inner)
                if build_B:
                    self.B_r_interp = \
                        CloughTocher2DInterpolator(self.tri, B_flat_r_inner)
                    self.B_phi_interp = \
                        CloughTocher2DInterpolator(self.tri, B_flat_phi_inner)
                    self.B_z_interp = \
                        CloughTocher2DInterpolator(self.tri, B_flat_z_inner)
        else:
            if build_B:
                raise NotImplementedError('Regular Grid Interpolator not '\
                                          'implemented for B, need to set '\
                                          'build_B = False.')
            self.psi_interp = \
                RegularGridInterpolator((self.r_array, self.z_array), self.psi)
        return None

    def _compute_rz_array_data(self):
        """Return rz array as well as max and min r and z extends"""
        min_r_, max_r_ = self.EqData.min_r, self.EqData.max_r
        min_z_, max_z_ = self.EqData.min_z, self.EqData.max_z
        nr_, nz_ = self.EqData.nr, self.EqData.nz

        # Psi values sit on cell centers
        dr_ = (max_r_ - min_r_)/nr_
        dz_ = (max_z_ - min_z_)/nz_

        r_array_ = np.linspace(min_r_ + dr_/2., max_r_ - dr_/2., nr_)
        z_array_ = np.linspace(min_z_ + dz_/2., max_z_ - dz_/2., nz_)
        rz_coords_ = np.array([(r_, z_) for r_ in r_array_ for z_ in z_array_])
        return rz_coords_, r_array_, z_array_, (dr_, dz_), \
            (min_r_, max_r_), (min_z_, max_z_)

    def compute_finite_difference(self, Out_, In_, d_, rows=True,
                                  verbose_=False):
        """Compute difference values along rows (corresponding to z derivative)
        or columns (corresponding to r derivative) for input array.
        Assumes outer wall is immersed in rectangular domain, with input array
        values outside domain (but inside rectangle) equal to 0. or nan;
        only computes difference values inside outer wall, and sets Out_
        outside outer wall to self.filler_value."""
        if not rows:
            Out_ = np.transpose(Out_)
            In_ = np.transpose(In_)

        for idx0, in_1d_arr in enumerate(In_):
            for idx1, in_val in enumerate(in_1d_arr):
                idx_ = idx0*In_.shape[1] + idx1
                if (verbose_ and idx_ % 100000 == 0):
                    print("Returning difference value for point nr "\
                          "{0} of {1}".format(idx_, In_.size))

                bdry_flag_r = 1 if idx1 == 0 else 0
                bdry_flag_l = 1 if idx1 == len(in_1d_arr) - 1 else 0

                if in_val == 0.:
                    Out_[idx0][idx1] = self.filler_value
                elif bdry_flag_r or in_1d_arr[idx1 - 1] == 0.:
                    # Right difference
                    Out_[idx0][idx1] = in_1d_arr[idx1 + 1] - in_1d_arr[idx1]
                elif bdry_flag_l or in_1d_arr[idx1 + 1] == 0.:
                    # Left difference
                    Out_[idx0][idx1] = in_1d_arr[idx1] - in_1d_arr[idx1 - 1]
                else:
                    # Centered difference
                    Out_[idx0][idx1] = \
                        (in_1d_arr[idx1 + 1] - in_1d_arr[idx1 - 1])/2.

                # Scale by grid delta
                if in_val != 0.:
                    Out_[idx0][idx1] /= d_

        if not rows:
            Out_ = np.transpose(Out_)
            In_ = np.transpose(In_)

    def compute_piecewise_polynomial_fit(self, dist_to_wall=0.02, poly_deg=5):
        """Compute a piecewise polynomial fit for psi within a region
        close to the wall up to dist_to_wall. Returns the fitted psi, the
        resulting analytically computed r and z components of B, together with a
        boolean array corresponding to the self.rz_coords coordinates at which
        the fitted psi and B are defined."""
        if self.rect_domain:
            raise NotImplementedError('Polynomial fit only implemented for ' \
                                      'tokamak shaped domain.')
        psi_array_ = self.psi

        # Use EqData psi as outer wall indicator function
        EqD_psi = self.EqData.psi

        # Find points where separatrix meets boundary
        coords_sep = self.rz_coords[np.abs(psi_array_.flatten()
                                    - self.EqData.psi_Boundary) < 0.02, :]

        # To do so build outer wall polygon, and look at distance of separatrix
        # to polygon
        outer_polygon = self.build_wall_polygon(wall=1)
        dist_sep_to_wall = np.zeros(coords_sep.shape[0])

        for idx, coord in enumerate(coords_sep):
            dist_sep_to_wall[idx] = \
                outer_polygon.exterior.distance(Point(coord))

        cond_close_to_wall = dist_sep_to_wall < np.min(dist_sep_to_wall) + 0.2
        close_to_wall = coords_sep[cond_close_to_wall]
        dist_sep_to_wall = dist_sep_to_wall[cond_close_to_wall]

        # Pick the two closest ones that are apart from each other
        # so as to find both separatrix ends meeting the boundary
        r_mid = np.average(close_to_wall[:, 0])
        close_to_wall0 = close_to_wall[close_to_wall[:, 0] < r_mid]
        dist_sep_to_wall0 = dist_sep_to_wall[close_to_wall[:, 0] < r_mid]
        close_to_wall1 = close_to_wall[close_to_wall[:, 0] > r_mid]
        dist_sep_to_wall1 = dist_sep_to_wall[close_to_wall[:, 0] > r_mid]

        # Separate coordinates into those close to and those away from boundary
        wall_flag_array = self.compute_grid_near_wall(dist_to_wall=dist_to_wall)
        outer_coords = self.rz_coords[wall_flag_array, :]
        # Also drop coordinates outside outer wall
        # (assuming psi = 0 outside outer wall)
        outer_coords = outer_coords[EqD_psi.flatten()[wall_flag_array] != 0]
        self.outer_coords = outer_coords
        self.wall_flag_array = wall_flag_array

        # Get polynomial fit patches
        ra, za = outer_coords[:, 0], outer_coords[:, 1]
        self.sep_at_wall0 = close_to_wall0[np.argmin(dist_sep_to_wall0)]
        self.sep_at_wall1 = close_to_wall1[np.argmin(dist_sep_to_wall1)]

        fit_bools = get_wall_patch_conditionals(ra, za,
                                                self.sep_at_wall0,
                                                self.sep_at_wall1,
                                                self.rz_center).values()

        if debug_polynomial_multi_fit_regions:
            s_ = 3
            outer_coords_reg = np.ones_like(outer_coords[:, 0])
            for idx, bools in enumerate(fit_bools):
                outer_coords_reg[bools] = idx
            sc = plt.scatter(outer_coords[:, 0], outer_coords[:, 1],
                             c=outer_coords_reg, s=s_, cmap=cm.rainbow)
            plt.colorbar(sc)
            plt.scatter(coords_sep[:, 0], coords_sep[:, 1], s=s_)
            plt.scatter(self.sep_at_wall0[0], self.sep_at_wall0[1], s=s_)
            plt.scatter(self.sep_at_wall1[0], self.sep_at_wall1[1], s=s_)
            plt.title('Polynomial fit regions for computation of\n$\Psi$, '\
                      '$B_r$, and $B_z$ near the boundary.')
            plt.show()
            if not debug_polynomial_multi_fit_psi:
                exit()

        # Compute actual fits
        psi_multi_fit = np.zeros(outer_coords.shape[0])
        psi_r_multi_fit = np.zeros(outer_coords.shape[0])
        psi_z_multi_fit = np.zeros(outer_coords.shape[0])

        psi_outer = psi_array_.flatten()[wall_flag_array]
        psi_outer = psi_outer[EqD_psi.flatten()[wall_flag_array] != 0]
        self.psi_outer = psi_outer

        for bool in fit_bools:
            coords_sub = outer_coords[bool]
            psi_outer_sub = psi_outer[bool]
            psi_fit = polynomial_fit_2D(coords_sub, psi_outer_sub)
            psi_r_fit = polynomial_fit_2D(coords_sub, psi_outer_sub,
                                          return_derivative='r')
            psi_z_fit = polynomial_fit_2D(coords_sub, psi_outer_sub,
                                          return_derivative='z')

            psi_multi_fit[bool] = psi_fit[:]
            psi_r_multi_fit[bool] = psi_r_fit[:]
            psi_z_multi_fit[bool] = psi_z_fit[:]

        if debug_polynomial_multi_fit_psi:
            psi_ratio = psi_outer/psi_multi_fit

            thrs = 0.005
            psi_thrs_vals = len(psi_ratio[np.logical_or(psi_ratio > 1 + thrs,
                                                        psi_ratio < 1 - thrs)])
            print("Number of points with relative fine grid - polynomial " \
                   "fit psi error bigger than {0}%: {1} out of {2}"\
                  .format(thrs, psi_thrs_vals, len(psi_ratio)))

            psi_ratio = np.clip(psi_ratio, 1 - thrs, 1 + thrs)

            sc = plt.scatter(outer_coords[:, 0], outer_coords[:, 1],
                             c=psi_ratio, cmap=cm.coolwarm, s=2)
            plt.colorbar(sc)
            plt.title('Ratio of EFIT grid $\Psi$ vs interpolated $\Psi$ '\
                      'near the boundary.')
            plt.show()
            if not debug_polynomial_multi_fit_B:
                exit()

        # Compute B_r and B_z from the psi derivatives
        B_r = psi_z_multi_fit/outer_coords[:, 0]
        B_z = -psi_r_multi_fit/outer_coords[:, 0]

        return psi_multi_fit, B_r, B_z, \
            np.logical_and(wall_flag_array, EqD_psi.flatten() != 0)

    def _compute_B_rphiz(self, skip_1byr=False, skip_B_phi=False,
                         verbose=False):
        """Get B_phi values from g, B_r and B_z values from Psi"""
        B_array_ = np.zeros((*self.psi.shape, 3))
        psi_array_ = self.psi

        # Optionally skip factor 1 over r to examine plasma induced part
        r_array = np.reshape(self.rz_coords, (*self.psi.shape, 2))[:, :, 0]
        if skip_1byr:
            print("Skipping 1/r coefficient for B computation.")
            r_array = np.ones_like(r_array)

        if debug_skip_B_phi or skip_B_phi:
            if debug_skip_B_phi:
                # May want to skip B phi computation to save computation time
                print("Skipping B phi computation")
        else:
            # Load values for B_phi
            psi_flat = psi_array_.flatten()
            B_array_phi = np.zeros_like(psi_flat)
            g_at_separatrix = self.g_interp(self.EqData.psi_Boundary)

            if verbose:
                print("Computing B_phi component")
            for idx, psi_ in enumerate(psi_flat):
                if verbose and idx % 100000 == 0:
                    print("Returning B_phi value for point nr {0} of {1}" \
                           .format(idx, psi_flat.shape[0]))
                if psi_ == 0.:
                    B_array_phi[idx] = self.filler_value
                elif self.polygon_separatrix.contains( \
                        Point(self.rz_coords[idx])):
                    psi_lim = self._limit_psi_val(psi_)
                    B_array_phi[idx] = self.g_interp(psi_lim)
                else:
                    B_array_phi[idx] = g_at_separatrix

            B_array_phi = np.reshape(B_array_phi, psi_array_.shape)
            B_array_[:, :, 1] = B_array_phi/r_array

        # Load values for B_r and B_z
        dpsi_dr = np.zeros_like(psi_array_)
        dpsi_dz = np.zeros_like(psi_array_)

        if verbose:
            print("Computing B_r component")
        self.compute_finite_difference(dpsi_dz, psi_array_, self.res_rz[1])
        B_array_[:, :, 0] = dpsi_dz/r_array

        if verbose:
            print("Computing B_z component")
        self.compute_finite_difference(dpsi_dr, psi_array_, self.res_rz[0],
                                       rows=False)
        B_array_[:, :, 2] = - dpsi_dr/r_array

        # Optionally overwrite the boundary values by a polynomial fit to avoid
        # errors associated with right/left differences
        if self.build_B_with_fit:
            psi_multi_fit, B_r_fit, B_z_fit, outer_bool = \
                self.compute_piecewise_polynomial_fit()

            B_array_fit = np.copy(np.reshape(B_array_,
                                             (*psi_array_.flatten().shape, 3)))
            B_array_fit[outer_bool, 0] = B_r_fit
            B_array_fit[outer_bool, 2] = B_z_fit
            B_array_fit = np.reshape(B_array_fit, (*psi_array_.shape, 3))

            if debug_polynomial_multi_fit_B:
                B_diff = np.abs(B_array_fit - B_array_)
                self.plot_contour_and_field(B_diff[:, :, 0].flatten(), \
                    title='Absolute difference between ' \
                    '$B_r$ as computed\nvia finite difference vs '\
                    'polynomial fit at boundary')
                self.plot_contour_and_field(B_diff[:, :, 2].flatten(),
                    title='Absolute difference between ' \
                    '$B_z$ as computed\nvia finite difference vs '\
                    'polynomial fit at boundary')
                exit()

            B_array_ = B_array_fit

        return B_array_

    def compute_B_div(self, B_array=None, r_array=None, verbose=False):
        """Compute divergence of B to check if it is zero."""
        if B_array is None:
            raise_flag = False
            if r_array is not None:
                raise_flag = True
            B_array = self.B_array
            r_array = np.reshape(self.rz_coords, (*self.psi.shape, 2))[:, :, 0]
        if (B_array is not None and r_array is None) or raise_flag:
            raise RuntimeError('Need to either pass both B and r array in '\
                               'computation of div B or neither.')

        if B_array.shape[:2] != r_array.shape:
            raise RuntimeError('B and r arrays must have same shape (up to ' \
                               'third component of B')

        drBr_dr = np.zeros(B_array.shape[:2])
        dBz_dz = np.zeros(B_array.shape[:2])
        Br, Bz = B_array[:, :, 0], B_array[:, :, 2]

        self.compute_finite_difference(dBz_dz, Bz, self.res_rz[1],
                                       verbose_=verbose)
        self.compute_finite_difference(drBr_dr, r_array*Br, self.res_rz[0],
                                       rows=False, verbose_=verbose)

        return drBr_dr/r_array + dBz_dz

    def compute_j(self, B_array=None, r_array=None, verbose=False):
        """Compute density current field j = curl(B)"""
        if B_array is None:
            raise_flag = False
            if r_array is not None:
                raise_flag = True
            B_array = self.B_array
            r_array = np.reshape(self.rz_coords, (*self.psi.shape, 2))[:, :, 0]
        if (B_array is not None and r_array is None) or raise_flag:
            raise RuntimeError('Need to either pass both B and r array in '\
                               'computation of div B or neither.')

        if B_array.shape[:2] != r_array.shape:
            raise RuntimeError('B and r arrays must have same shape (up to ' \
                               'third component of B')

        # Compute J
        dBphi_dz = np.zeros(B_array.shape[:2])
        dBr_dz = np.zeros(B_array.shape[:2])
        dBz_dr = np.zeros(B_array.shape[:2])
        drBphi_dr = np.zeros(B_array.shape[:2])
        Br, Bphi, Bz = B_array[:, :, 0], B_array[:, :, 1], B_array[:, :, 2]

        self.compute_finite_difference(dBphi_dz, Bphi, self.res_rz[1],
                                       verbose_=verbose)
        self.compute_finite_difference(dBr_dz, Br, self.res_rz[1],
                                       verbose_=verbose)
        self.compute_finite_difference(dBz_dr, Bz, self.res_rz[0],
                                       rows=False, verbose_=verbose)
        self.compute_finite_difference(drBphi_dr, r_array*Bphi, self.res_rz[0],
                                       rows=False, verbose_=verbose)

        j_array = np.zeros_like(B_array)
        j_array[:, :, 0] = - dBphi_dz
        j_array[:, :, 1] = dBr_dz - dBz_dr
        j_array[:, :, 2] = drBphi_dr/r_array
        return j_array

    def compute_j_cross_B(self, B_array=None, r_array=None, verbose=False):
        """Compute j cross B to check if it is zero for Taylor states."""
        j_array = self.compute_j(B_array, r_array, verbose)
        return np.cross(j_array, B_array)

    def compute_p_grad(self, verbose_=False):
        """Compute gradient of p via p(psi(r, z))."""

        # Load values for p
        psi_flat = self.psi.flatten()
        p_array = np.zeros_like(psi_flat)
        p_at_separatrix = self.p_interp(self.EqData.psi_Boundary)

        if verbose_:
            print("Setting p array for grad p computation.")
        for idx, psi_ in enumerate(psi_flat):
            if verbose_ and idx % 100000 == 0:
                print("Returning p value for point nr {0} of {1}" \
                       .format(idx, psi_flat.shape[0]))
            if psi_ == 0.:
                p_array[idx] = self.filler_value
            elif self.polygon_separatrix.contains( \
                    Point(self.rz_coords[idx])):
                psi_lim = self._limit_psi_val(psi_)
                p_array[idx] = self.p_interp(psi_lim)
            else:
                p_array[idx] = p_at_separatrix

        p_array = np.reshape(p_array, self.psi.shape)

        # Compute dp/dr, dp/dz
        dp_dr = np.zeros_like(p_array)
        dp_dz = np.zeros_like(p_array)

        self.compute_finite_difference(dp_dr, p_array, self.res_rz[0],
                                       rows=False, verbose_=verbose_)
        self.compute_finite_difference(dp_dz, p_array, self.res_rz[1],
                                       verbose_=verbose_)

        p_grad_array = np.zeros((*self.psi.shape, 3))
        p_grad_array[:, :, 0] = dp_dr[:,:]
        p_grad_array[:, :, 2] = dp_dz[:,:]

        return p_grad_array

    def set_flagged_psi(self, load, psi_array_file_name, save_psi=True):
        """Method to set flagged_psi_array attribute. If load, this is
        done by loading from a saved numpy array, otherwise it is built using
        compute_flagged_psi method. If the array is built, it can optionally
        also be saved."""
        if load:
            try:
                psi_flagged_array = np.load(psi_array_file_name)
                # Check if array has correct dimension
                if psi_flagged_array.shape != self.rz_coords.shape:
                    raise RuntimeError('Flagged psi array must have same '\
                        'dimension as rz coordiante array; may have loaded '\
                        'psi array corresponding to different EFIT file?')

                setattr(self, 'psi_flagged_array', psi_flagged_array)
                return None
            except FileNotFoundError:
                # If file does not exist move on to building array
                pass

        psi_flagged_array = self.compute_flagged_psi()
        setattr(self, 'psi_flagged_array', psi_flagged_array)

        if save_psi:
            np.save(self.EFIT_dir + '/npy/' + psi_array_file_name,
                    psi_flagged_array)
        return None

    def scale_arr_to_phys(self, cntr_arr, reverse=False):
        """Scale pixel contour array to physical array"""
        min_r, max_r = self.r_extend
        min_z, max_z = self.z_extend
        r_pix_range, z_pix_range = self.psi.shape
        r_phys_range, z_phys_range = max_r - min_r, max_z - min_z

        if reverse:
            cntr_arr[:, 0] = (cntr_arr[:, 0] - min_r)*r_pix_range/r_phys_range
            cntr_arr[:, 1] = -(cntr_arr[:, 1] - max_z)*z_pix_range/z_phys_range
        else:
            cntr_arr[:, 0] = cntr_arr[:, 0]*r_phys_range/r_pix_range + min_r
            cntr_arr[:, 1] = -(cntr_arr[:, 1]*z_phys_range/z_pix_range - max_z)


    def _get_pgq_info(self, name):
        """Return information relevant to p, g, q plotting routines."""
        if name == 'p':
            return self.EqData.pres, 'Pressure $p$', 'nt / m$^2$'
        if name == 'g':
             return self.EqData.fpol, 'Poloidal current $g$', 'm-T'
        if name == 'q':
            return self.EqData.q, 'Safety factor $q$', 'Factor'
        raise ValueError('Can only plot g, q, or p as a function of psi.')

    def plot_from_psi(self, name):
        """Plot p, g, or q as functions of psi."""
        eqd, t_, l_ = self._get_pgq_info(name)
        plt.plot(self.flux_grid, eqd)
        plt.title('{0} as a function of $\Psi$'.format(t_))
        plt.xlabel('$\Psi$')
        plt.ylabel('{0}'.format(l_))
        plt.show()

    def plot_contour_from_psi(self, name, val, show_plot=True):
        """Plot a fixed value of p, g, or q as 
        a psi contour in RZ coordinates."""
        eqd, t_, _ = self._get_pgq_info(name)

        if val > max(eqd) or val < min(eqd):
            raise ValueError('Requested {0} value {1} not in data range.' \
                             .format(name, val))

        # Find psi values corresponding to p/g/q value
        psi_vals = []
        for idx_d in range(len(eqd)-1):
            a_, b_ = val - eqd[idx_d], eqd[idx_d+1] - val
            if (a_ >= 0 and b_ > 0) or (a_ <= 0 and b_ < 0):
                psi_vals.append( \
                    self.flux_grid[idx_d if abs(a_) < abs(b_) else idx_d+1])

        # Compute psi contours and drop ones outside separatrix
        plt.figure()
        get_cntrs = plt.contour(np.flip(self.psi.transpose(), axis=0), psi_vals)
        cntr_arrays = get_cntrs.allsegs[0]
        plt.close()

        cntrs_to_plot = []
        for cntr_array in cntr_arrays:
            self.scale_arr_to_phys(cntr_array)
            if self.polygon_separatrix.contains(Point(cntr_array[0, :])):
                cntrs_to_plot.append(cntr_array)

        # Plot psi contours
        for idx_c, cntr in enumerate(cntrs_to_plot):
            b_ = (max(np.absolute(eqd)) - abs(val)) \
                  /(max(np.absolute(eqd)) - min(np.absolute(eqd)))
            plt.plot(cntr[:, 0], cntr[:, 1], color=(b_, b_, 0),
                     label='{0} factor {1}'.format(name, val),
                     linestyle=(0, (5, 5)))

        # Optionally do plot with psi field
        if show_plot:
            self.plot_contour_and_field(self.psi.flatten())
            plt.show()

    def plot_contour_and_field(self, field_to_plot, coords=None, step=10,
                               skip_field_step=False, title=None,
                               cmap=cm.viridis, s_=2, skip_separatrix=False,
                               save_plot_name=None):
        """Plot contour for loop part of separatrix and given array values.
        Only include every step^th entry in scatter plot for ease on memory.
        If skip_field_step is set True, then the step setup does not apply for
        the field array (useful when field array itself has been computed with
        a step setup)."""
        if skip_field_step:
            field_step = 1
        else:
            field_step = step
        if coords is None:
            coords = self.rz_coords[::step, :]

        sc = plt.scatter(coords[:, 0], coords[:, 1],
                         c=field_to_plot[::field_step], cmap=cmap, s=s_)
        plt.colorbar(sc)
        if not skip_separatrix:
            plt.plot(self.cntr_separatrix[:, 0], self.cntr_separatrix[:, 1],
                     color='red', label='Separatrix loop',
                     linestyle=(0, (5, 5)))
            plt.legend()
        if title is not None:
            plt.title(title)

        if save_plot_name is not None:
            plt.savefig(save_plot_name)
            plt.clf()
        else:
            plt.show()

    def _compute_loop_separatrix_contour(self, psi_init_diff=None,
                                         psi_step=None, psi_step_fac=None,
                                         get_sep_from_eqdata=True):
        """Compute contour of separatrix part that forms a loop. Do so by
        picking contour value slightly smaller than psi value at separatrix;
        see algorithm. May optionally also return separatrix coordinates from
        EqData object, if it has them; defaults to True."""
        if get_sep_from_eqdata and hasattr(self.EqData, 'sep_coords'):
            if self.verbose:
                print('Getting separatrix contour from data.')
            return np.reshape(np.array(self.EqData.sep_coords),
                                 (int(len(self.EqData.sep_coords)/2.), 2))

        if psi_init_diff is None:
            psi_init_diff = 0
        if psi_step is None:
            psi_step = 0.0001
        if psi_step_fac is None:
            psi_step_fac = 0.1

        # Compute contour line of separatrix using plt.contour
        # plt.contour will either find 1 contour just outside separatrix or
        # 2 contours, i.e. one just inside separatrix and 1 between divertors
        # Do loop to find the 2 contour scenario and only keep the one just
        # inside separatrix
        psi_array_ = self.psi
        psi_separatrix = self.EqData.psi_Boundary
        c_ = 0
        while True:
            psi_cnt = (psi_separatrix - psi_init_diff
                       - psi_step*(1 + c_*psi_step_fac))
            plt.figure()
            get_cntrs = plt.contour(np.flip(psi_array_.transpose(), axis=0),
                                    [psi_cnt, ])

            # Get contours via allsegs
            cntr_arrays = get_cntrs.allsegs[0]
            plt.close()

            if len(cntr_arrays) != 1:
                # Can visually inspect contours if algorithm fails (to e.g. set
                # better psi_step, psi_step_fac)
                if debug_separatrix:
                    plt.close()
                    plt.figure()
                    plt.imshow(np.flip(psi_array_.transpose(), axis=0))
                    psi_cntrs = [psi_cnt, ]
                    plt.contour(np.flip(psi_array_.transpose(), axis=0),
                                psi_cntrs, colors='orange', linewidths=0.75)
                    image_center = np.array(psi_array_.shape)/2.
                    plt.scatter(image_center[0], image_center[1])
                    plt.contour(np.flip(psi_array_.transpose(), axis=0), 100,
                                colors='red', linewidths=0.75)
                    plt.show()
                    exit()
                break
            c_ += 1

        # Scale contour arrays to physical space and find contour for just
        # inside separatrix via distance to M axis
        # If M axis is not given (i.e. 0, 0), pick domain center instead
        dist_cntr = []
        for cntr_arr in cntr_arrays:
            self.scale_arr_to_phys(cntr_arr)
            dist_cntr.append(np.linalg.norm(cntr_arr[0, :] - self.rz_center))

        return cntr_arrays[dist_cntr.index(min(dist_cntr))]

    def compute_flagged_psi(self):
        """Compute flags for whether psi is inside loop part of separatrix or
        not, done by computing a contour corresponding to the loop part. Returns
        nx2 array where n is the number of coordinates. The first column
        contains psi values, the second one flags (0 or 1). The order of the
        array is equal to rz_coords_ from method compute_rz_array_data."""

        # This takes a short while, so include print output on progress
        if self.verbose:
            print("Identifying points inside separatrix")
        interior_flag = np.zeros(self.rz_coords.shape[0])
        for idx, row in enumerate(self.rz_coords):
            if idx % 50000 == 0 and self.verbose:
                print("Checking point nr {0} of {1}" \
                       .format(idx, self.rz_coords.shape[0]))
            interior_flag[idx] = self.polygon_separatrix.contains(Point(row))
        if self.verbose:
            print("Number of points inside separatrix: {0} of {1}" \
                   .format(int(np.sum(interior_flag)), self.rz_coords.shape[0]))

        # Build array containing psi values and flags; has same dimension
        # Array has same dimension as rz_coords, so given rz_coords row,
        # can look into flagged_psi_array for according psi value and flag
        # for if the point is inside separatrix loop.
        psi_flagged_array = np.vstack((self.psi.flatten(),
                                       interior_flag)).transpose()

        return psi_flagged_array

    def _limit_psi_val(self, value):
        """Limit given psi value with respect to maximum and minimum psi inside
        separatrix loop to avoid interpolation over/undershoot issues"""
        m1_fac = 1 if self.psi_min < self.psi_max else -1
        if m1_fac*value < m1_fac*self.psi_min:
            return self.psi_min
        if m1_fac*value > m1_fac*self.psi_max:
            return self.psi_max
        return value

    def build_wall_polygon(self, from_tokamak_data_handler=None, wall=0,
                           move_inside=0):
        """Build Polygon object corresponding to wall. 0 for metal wall, 1 for
        inner VV wall, 2 for outer VV wall. The coordinates are loaded from the
        poiloidal_tokamak_mesh_creator object."""
        if from_tokamak_data_handler is None:
            from_tokamak_data_handler = not self.wall_from_data

        if from_tokamak_data_handler:
            if wall == 0:
                coord_handler = \
                    tokamak_data_handler(return_coarse=self.return_coarse, \
                        wide_inner_wall_fac=self.inner_wall_fac,
                        wide_r_only=self.wide_r_only)
                r_coords, z_coords = \
                    coord_handler.VacuumVesselMetalWallCoordinates()
            elif wall in (1, 2):
                coord_handler = \
                    tokamak_data_handler(return_coarse=self.return_coarse, \
                        wide_outer_wall_fac=self.outer_wall_fac,
                        wide_r_only=self.wide_r_only)
                if wall == 1:
                    r_coords, z_coords = \
                        coord_handler.VacuumVesselFirstWallCoordinates()
                else:
                    r_coords, z_coords = \
                        coord_handler.VacuumVesselSecondWallCoordinates()
        else:
            if wall == 0:
                coord_list = self.EqData.fw_coords
                fac = self.inner_wall_fac
            elif wall == 1 and hasattr(self.EqData, 'vv_coords'):
                coord_list = self.EqData.vv_coords
                fac = self.outer_wall_fac
            else:
                print('EFIT does not have outer wall coordinates, using ' \
                       'built-in coordinates from tokamak_data_handler class.')
                return self.build_wall_polygon(True, wall, move_inside)

            coords_ = np.reshape(np.array(coord_list),
                                 (int(len(coord_list)/2.), 2))

            r, z = coords_[:, 0], coords_[:, 1]
            r_coords, z_coords = \
                tokamak_data_handler.scale_vessel_coords(r, z, fac, \
                    scale_r_only=self.wide_r_only)

        coords_ = np.vstack((r_coords, z_coords)).transpose()

        if move_inside != 0:
            for coord in coords_:
                coord += move_inside*(self.maxis_coord - coord)

        # Build polygon object
        return Polygon(coords_)

    def compute_grid_near_wall(self, coords=None, dist_to_wall=0.01,
                               verbose=None, verbose_out_freq=50000):
        """Compute array of flags indicating which of the given coordinates are
        near outer wall."""
        if self.verbose and verbose is None:
            verbose = True

        wall_arr_name = self.EFIT_dir + '/npy/{0}_wall_grid_array.npy' \
            .format(splitext(self.file_name)[0])
        try:
            return np.load(wall_arr_name)
        except FileNotFoundError:
            pass

        if coords is None:
            coords = self.rz_coords

        outer_polygon = self.build_wall_polygon(wall=1,
                                                move_inside=dist_to_wall)
        wall_flag_array = np.zeros(coords.shape[0]).astype(bool)
        if self.verbose:
            print("Finding coordinates close to boundary for " \
                   "boundary polynomial")
        for idx, coord in enumerate(coords):
            if (self.verbose and idx % verbose_out_freq == 0):
                print("Checking coordinate nr "\
                       "{0} of {1}".format(idx, coords.shape[0]))
            wall_flag_array[idx] = not outer_polygon.contains(Point(coord))

        np.save(wall_arr_name, wall_flag_array)
        return wall_flag_array

    def evaluate_wall_at_coords(self, coords, verbose=None, 
                                verbose_out_freq=50000, wall=0):
        """Return indicator function on whether we are inside wall.
        0 for metal wall, 1 for inner VV wall, 2 for outer VV wall."""
        if self.verbose and verbose is None:
            verbose = True

        # Set up indicator function on data grid
        polygon_ = self.build_wall_polygon(wall=wall)

        val_array = np.zeros((coords.shape[0],))

        for idx, coord in enumerate(coords):
            if verbose and idx % verbose_out_freq == 0:
                print("Computing wall flag value for point nr "
                       "{0} of {1}".format(idx, coords.shape[0]))
    
            if polygon_.contains(Point(coord)):
                val_array[idx] = 1

        return val_array

    def evaluate_separatrix_at_coords(self, coords, verbose=None, 
                                      verbose_out_freq=50000):
        """Return indicator function on whether we are inside separatrix."""
        if self.verbose and verbose is None:
            verbose = True

        val_array = np.zeros((coords.shape[0],))

        for idx, coord in enumerate(coords):
            if verbose and idx % verbose_out_freq == 0:
                print("Computing separatrix flag value for point nr "
                       "{0} of {1}".format(idx, coords.shape[0]))
    
            if self.polygon_separatrix.contains(Point(coord)):
                val_array[idx] = 1

        return val_array

    def evaluate_r_th_at_coords(self, coords, verbose=None,
                                verbose_out_freq=50000, filler_val=1):
        """Return coordinate function for normalized distance and angle relative
        to magnetic axis and separatrix."""
        if self.verbose and verbose is None:
            verbose = True

        val_array = np.zeros((coords.shape[0], 2))

        for idx, coord in enumerate(coords):
            if verbose and idx % verbose_out_freq == 0:
                print("Computing r coordinate value for point nr "
                       "{0} of {1}".format(idx, coords.shape[0]))

            # r coordinate
            coord_vec = coord - self.maxis_coord

            if np.linalg.norm(coord_vec) < 1e-12:
                val_array[idx, 0] = 0.
            elif self.polygon_separatrix.contains(Point(coord)):
                # Draw line through coord and magnetic axis, find intersection
                # with separatrix, scale between 0 and 1 accordingly
                d_coord_vec = np.linalg.norm(coord_vec)
                out_point = coord + 10*(coord_vec)/d_coord_vec
                line = LineString([self.maxis_coord, out_point])
                sep_point = \
                    list(self.polygon_separatrix.intersection(line).coords)[1]

                val_array[idx, 0] = \
                    d_coord_vec/np.linalg.norm(sep_point - self.maxis_coord)
            else:
                val_array[idx, 0] = filler_val

            # theta coordinate
            val_array[idx, 1] = np.arctan2(coord_vec[1], coord_vec[0])

        return val_array

    def evaluate_at_coords(self, coords, fname, verbose=None,
                           verbose_out_freq=50000, transl_fac=1e-4):
        """Evaluate field at a specified coordinate array of
        shape nx2, for n coordinates."""
        if self.verbose and verbose is None:
            verbose = True

        # Include link to inner wall evaluator
        if fname == 'separatrix':
            return self.evaluate_separatrix_at_coords( \
                coords, verbose, verbose_out_freq)
        if fname == 'r_theta':
            return self.evaluate_r_th_at_coords( \
                coords, verbose, verbose_out_freq)
        if fname == 'eta' or fname == 'metal wall':
            return self.evaluate_wall_at_coords( \
                coords, verbose, verbose_out_freq, wall=0)
        if fname == 'inner VV wall':
            return self.evaluate_wall_at_coords( \
                coords, verbose, verbose_out_freq, wall=1)
        if fname == 'outer VV wall':
            return self.evaluate_wall_at_coords( \
                coords, verbose, verbose_out_freq, wall=2)
        if fname == 'p':
            interp = self.p_interp
            val_array = np.zeros((coords.shape[0],))
        elif fname == 'g':
            interp = self.g_interp
            val_array = np.ones((coords.shape[0],))
            val_array *= interp(self.EqData.psi_Boundary)
        elif fname == 'psi':
            val_array = np.zeros((coords.shape[0],))
        elif fname == 'B_r':
            interp = self.B_r_interp
            val_array = np.zeros((coords.shape[0],))
        elif fname == 'B_phi':
            interp = self.B_phi_interp
            val_array = np.zeros((coords.shape[0],))
        elif fname == 'B_z':
            interp = self.B_z_interp
            val_array = np.zeros((coords.shape[0],))
        elif fname in ('dp_dr', 'dp_dz'):
            self.build_2d_interpolators(grad_p=True)
            if fname == 'dp_dr':
                interp = self.dp_dr_interp
                val_array = np.zeros((coords.shape[0],))
            elif fname == 'dp_dz':
                interp = self.dp_dz_interp
                val_array = np.zeros((coords.shape[0],))
        else:
            raise RuntimeError('Field to be evaluated must be separatrix, ' \
                               'r_theta, eta, inner wall, outer wall,  p, g, '\
                               'psi, B_r, B_phi, B_z, dp_dr, or dp_dz')

        for idx, coord in enumerate(coords):
            if verbose and idx % verbose_out_freq == 0:
                print("Computing {0} value for point nr {1} of {2}" \
                       .format(fname, idx, coords.shape[0]))
            coord_ = np.copy(coord)

            # If (Delaunay based) interpolator returns Nan then coordinate is
            # outside triangulation's convex hull. In this case, nudge coord
            # slightly towards magnetic axis until it is inside convex hull
            if fname in ('p', 'g', 'psi'):
                psi_val = self.psi_interp(coord_)
                while np.isnan(psi_val):
                    if self.rect_domain:
                        psi_val = self.filler_value
                        break
                    coord_ += transl_fac*(self.maxis_coord - coord_)
                    psi_val = self.psi_interp(coord_)

                if fname == 'psi':
                    val_array[idx] = psi_val
                elif self.polygon_separatrix.contains(Point(coord_)):
                    gp_val = self._limit_psi_val(psi_val)
                    val_array[idx] = interp(gp_val)
            else:
                # Also do nudging for vector type values
                vec_val = interp(coord_)
                while np.isnan(vec_val):
                    if self.rect_domain:
                        vec_val = self.filler_value
                        break
                    coord_ += transl_fac*(self.maxis_coord - coord_)
                    vec_val = interp(coord_)
                val_array[idx] = vec_val

        return val_array

    def evaluate_r_theta_at_coords(self, coords, verbose=None,
                                   verbose_out_freq=50000, transl_fac=None):
        """Evaluate inner local coordiantes at a specified coordinate array of
        shape nx2, for n coordinates. Arg transl_fac has no effect and is kept
        for compatibility reasons."""
        return self.evaluate_at_coords(coords, 'r_theta', verbose,
                                       verbose_out_freq)

    def evaluate_eta_at_coords(self, coords, verbose=None,
                               verbose_out_freq=50000, transl_fac=None):
        """Evaluate resistivity at a specified coordinate array of shape nx2,
        for n coordinates. Arg transl_fac has no effect and is kept for
        compatibility reasons."""
        return self.evaluate_at_coords(coords, 'eta', verbose,
                                       verbose_out_freq)

    def evaluate_metal_wall_at_coords(self, coords, verbose=None,
                                      verbose_out_freq=50000, transl_fac=None):
        """Evaluate inner VV wall at a specified coordinate array of shape nx2,
        for n coordinates. Arg transl_fac has no effect and is kept for
        compatibility reasons."""
        return self.evaluate_at_coords(coords, 'metal wall', verbose,
                                       verbose_out_freq)

    def evaluate_inner_VV_wall_at_coords(self, coords, verbose=None,
                                      verbose_out_freq=50000, transl_fac=None):
        """Evaluate inner VV wall at a specified coordinate array of shape nx2,
        for n coordinates. Arg transl_fac has no effect and is kept for
        compatibility reasons."""
        return self.evaluate_at_coords(coords, 'inner VV wall', verbose,
                                       verbose_out_freq)

    def evaluate_outer_VV_wall_at_coords(self, coords, verbose=None,
                                      verbose_out_freq=50000, transl_fac=None):
        """Evaluate outer VV wall at a specified coordinate array of shape nx2,
        for n coordinates. Arg transl_fac has no effect and is kept for
        compatibility reasons."""
        return self.evaluate_at_coords(coords, 'outer VV wall', verbose,
                                       verbose_out_freq)

    def evaluate_p_at_coords(self, coords, verbose=None,
                             verbose_out_freq=50000, transl_fac=1e-4):
        """Evaluate pressure at a specified coordinate array of shape nx2,
        for n coordinates."""
        return self.evaluate_at_coords(coords, 'p', verbose,
                                       verbose_out_freq, transl_fac)

    def evaluate_grad_p_at_coords(self, coords, verbose=None,
                                  verbose_out_freq=50000, transl_fac=1e-4):
        """Evaluate gradient of pressure at a specified coordinate array of
        shape nx2, for n coordinates."""
        p_r_ =  self.evaluate_at_coords(coords, 'dp_dr', verbose,
                                        verbose_out_freq, transl_fac)
        p_z_ =  self.evaluate_at_coords(coords, 'dp_dz', verbose,
                                        verbose_out_freq, transl_fac)
        return np.dstack((p_r_, np.zeros_like(p_r_), p_z_))

    def evaluate_g_at_coords(self, coords, verbose=None,
                             verbose_out_freq=50000, transl_fac=1e-4):
        """Evaluate g field at a specified coordinate array of shape nx2,
        for n coordinates."""
        return self.evaluate_at_coords(coords, 'g', verbose,
                                       verbose_out_freq, transl_fac)

    def evaluate_psi_at_coords(self, coords, verbose=None,
                             verbose_out_freq=50000, transl_fac=1e-4):
        """Evaluate psi field at a specified coordinate array of shape nx2,
        for n coordinates."""
        return self.evaluate_at_coords(coords, 'psi', verbose,
                                       verbose_out_freq, transl_fac)

    def evaluate_B_at_coords(self, coords, verbose=None,
                             verbose_out_freq=50000, transl_fac=1e-4):
        """Evaluate B field at a specified coordinate array of shape nx2,
        for n coordinates."""
        B_r_ = self.evaluate_at_coords(coords, 'B_r', verbose,
                                       verbose_out_freq, transl_fac)
        B_phi_ = self.evaluate_at_coords(coords, 'B_phi', verbose,
                                         verbose_out_freq, transl_fac)
        B_z_ = self.evaluate_at_coords(coords, 'B_z', verbose,
                                       verbose_out_freq, transl_fac)
        return np.dstack((B_r_, B_phi_, B_z_))


class poloidal_mode_creator():
    """Set up higher order poloidal modes on EFIT grid.
    :arg handler: class `Equilibrium_data_handler` type object.
    :args res_r, res_theta: resolution of (r, theta) coordinates used
    in poloidal plane to compute modes; default to 100, 200.
    :arg r_from_psi_contours: switch to choose r coordiantes either as following
    the EFIT psi contours, or as equispaced between the separatrix and the
    magnetic axis; defaults to True.
    :arg verbose: Include verbose output when computing (r-theta) coordinates;
    defaults to False."""
    def __init__(self, handler, res_r=100, res_theta=200,
                 r_from_psi_contours=True, verbose=False):
        self.verbose = verbose

        # Work in (r, theta) coordinates to avoid difficulties with Jacobian
        # transformation in integrals. Should have at least twice as high
        # res_theta compared to number of modes to make sin/cos Fourier
        # decomposition work!
        self.handler = handler
        self.res_r, self.res_theta = res_r, res_theta
        self.r_from_psi_contours = r_from_psi_contours
        self.rth_to_RZ = self._compute_r_theta_coords()

        self.built_tri = False
        self.tri = None
        self.to_rth_interpolators = {}
        self.interp_count = 0

    def _compute_r_theta_coords(self):
        """Compute array that contains (R, Z) locations for a set of r-theta
        coordinates as specified by a separatrix, magnetic axis, rule
        for r coordinates (defaults to according to psi contours), and
        equal split of 2*pi for theta coordinates."""
        handler = self.handler
        res_r, res_theta = self.res_r, self.res_theta
        # Along each fixed theta, find (R, Z) locations
        # of r = k/res_r, k = 1, ..., res_r
        res_r_ntrp = max(res_r, 50)
        m_c = handler.maxis_coord

        rth_to_RZ = np.zeros((res_r, res_theta, 2))
        for th_idx in range(res_theta):
            # Set up equidistanced points along theta lines in (R, Z) space
            s_line = LineString([m_c, (m_c[0] + 100, m_c[1])])
            line = rotate(s_line, 2*np.pi*th_idx/res_theta, origin=Point(m_c),
                          use_radians=True)
            sep_point = \
                list(handler.polygon_separatrix.intersection(line).coords)[1]
            r_points = np.array([m_c + k/res_r_ntrp*(sep_point - m_c)
                                     for k in range(res_r_ntrp+1)])

            # Find r values at either at equidistanced points or according to
            # psi contours, and use these to build interpolator
            if self.r_from_psi_contours:
                r_vals = handler.evaluate_psi_at_coords( \
                    r_points, verbose=self.verbose)
                r_vals = (r_vals - r_vals[0])/(r_vals[-1] - r_vals[0])
            else:
                r_vals = handler.evaluate_r_theta_at_coords( \
                    r_points, verbose=self.verbose)[:, 0]
            r_interp = interp1d( \
                r_vals, [k/res_r_ntrp for k in range(res_r_ntrp+1)])

            # Use interpolator to find (R, Z) space distances for equidistanced
            # points with respect to r coordinate
            r_dists = r_interp([k/res_r for k in range(res_r)])

            # Use distances to obtain corresponding points in (R, Z) space
            rth_to_RZ[:, th_idx, :] = \
                np.array([m_c + r_d*(sep_point - m_c) for r_d in r_dists])
        return rth_to_RZ

    def _interpolate_to_RZ(self, rth_array, fname=None, fill_value=0):
        """Interpolate given r-theta array to EFIT RZ coordinates."""
        if rth_array.shape != (self.res_r, self.res_theta):
            raise ValueError('Array to be interpolated into RZ coords has ' \
                             'to be in r-theta coords.')

        if not self.built_tri:
            self.tri = Delaunay( \
                self.rth_to_RZ.reshape(self.res_r*self.res_theta, 2))
            self.built_tri = True

        # Use filler value outside separatrix
        # Also may have coordiantes between separatrix and
        # interpolator hull; use filler value there too
        to_RZ_= LinearNDInterpolator(self.tri, rth_array.flatten(),
                                     fill_value=fill_value)

        return to_RZ_(self.handler.rz_coords) \
            .reshape(self.handler.EqData.psi.shape)

    @staticmethod
    def get_r_profile(pos_=0.5, ext_=0.5):
        """Return an cosine shaped r profile type lambda function mapping to
        [0, 1], valued 0 at r = 0, 1, and with profile's center and extend
        determined by pos_ and ext_, which take values between 0 and 1."""
        a_, b_ = 1/(1 - pos_) - 1, 1/ext_ - 1
        return lambda r_: ((1 - np.cos(2*np.pi*r_**a_))/2.)**b_

    def get_poloidal_perturbation(self, c_0=0, c_m=[], d_m=[], out_rz=True):
        """Compute poloidal Fourier series with cofficients c0, and c_m for cos,
        d_m for sin. The coefficients themselves should be lambda functions
        taking values r between 0 and 1 (and c_m, d_m should be lists thereof).
        Alternatively, they can be floats/ints and a standard r profile 1 - 
        cos(pi*r) is applied. Returns a (res_r, res_theta) shaped array for
        values in r-theta space, or optionally in R-Z space."""
        if (not isinstance(c_m, (list, tuple))
                or not isinstance(d_m, (list, tuple))):
            raise ValueError('c_m, d_m Fourier coefficients must be '\
                             'a list or tuple.')
        if not len(c_m) == len(d_m):
            raise ValueError('Fourier coefficient lists c_m, d_m must be of '\
                             'equal length.')
        if self.res_theta < 2*len(c_m):
            raise RuntimeError('Poloidal mode resolution is too small for '\
                               'requested number of modes.')
        res_r, res_theta = self.res_r, self.res_theta

        # Check input coefficients for floats/ints
        if isinstance(c_0, (int, float)) or c_0 is None:
            c_0 = lambda r_, c0=c_0: c0*self.get_r_profile()(r_)
        for lst in c_m, d_m:
            for idx_c, coeff_ in enumerate(lst):
                if isinstance(coeff_, (int, float)) or coeff_ is None:
                    lst[idx_c] = lambda r_, coeff=coeff_: \
                        coeff*self.get_r_profile()(r_)

        # Set up modes given coefficients
        nr_modes = len(c_m)
        mode_0 = np.zeros((res_r, res_theta))
        modes_m = np.zeros((nr_modes, res_r, res_theta))

        for idx_r in range(res_r):
            mode_0[idx_r] = c_0(idx_r/res_r)

        th_vals = np.array([2*np.pi*k/res_theta for k in range(res_theta)])

        for idx_m, mode in enumerate(modes_m):
            for idx_r in range(res_r):
                mode[idx_r] += c_m[idx_m](idx_r/res_r)*np.cos((idx_m+1)*th_vals)
                mode[idx_r] += d_m[idx_m](idx_r/res_r)*np.sin((idx_m+1)*th_vals)

        # Add up modes to obtain perturbation
        pert_ = mode_0
        for mode in modes_m:
            pert_ += mode
        if out_rz:
            return self._interpolate_to_RZ(pert_)
        return pert_

    def get_poloidal_mode(self, mode_nr, sine=False, r_func=None, out_rz=True):
        """Compute single poloidal Fourier series mode, given mode number,
        whether or not to return sine or cosine contribution, and an optional
        r-profile. By default, the r-profile is given by 1 - cos(pi*r). """
        if mode_nr == 0:
            return self.get_poloidal_perturbation(r_func, out_rz=out_rz)

        c_l, d_l = [0 for _ in range(mode_nr)], [0 for _ in range(mode_nr)]
        if sine:
            d_l[-1] = r_func
        else:
            c_l[-1] = r_func

        return self.get_poloidal_perturbation(c_m=c_l, d_m=d_l, out_rz=out_rz)

    def get_Fourier_decomposition(self, nr_modes, array_=None, Barray_=None,
                                  fname=None, return_f_vals=False):
        """Get poloidal Fourier decomposition either for given array on EFIT
        grid (together with B field to compute magnetic axis and separatrix).
        Alternatively, get decomp. of a given field in the EFIT grid,
        implemented in class `Equilibrium_data_handlerz `evaluate_at_coords`
        method."""
        if array_ is None and fname is None:
            raise ValueError('Need to pass either a field array or a field ' \
                             'name for Fourier decomposition.')

        # Get field values at (r, theta coordinates)
        res_r, res_theta = self.res_r, self.res_theta
        f_vals = np.zeros((res_r, res_theta))

        if array_ is not None:
            if Barray_ is None:
                logger.info('Computing poloidal decomposition assuming the ' \
                            'initial EFIT file separatrix and m-axis.')
                # Build inteprolator
                if fname not in self.self.to_rth_interpolators:
                    self.self.to_rth_interpolators[fname] = \
                        LinearNDInterpolator(self.handler.tri, array_.flatten())
                for idx, const_th_coords in enumerate(self.rth_to_RZ):
                    f_vals[idx, :] = \
                        self.self.to_rth_interpolators[fname](const_th_coords)
            else:
                raise NotImplementedError('To compute Fourier decomposition ' \
                    'for a given array and a given B field, need to ' \
                    'first compute separatrix and magnetic axis for the B ' \
                    'field array.')
        else:
            for idx, const_th_coords in enumerate(self.rth_to_RZ):
                f_vals[idx, :] = \
                    self.handler.evaluate_at_coords(const_th_coords, fname)

        # Set up mode coefficients
        coeff_0 = np.zeros((res_r, ))
        mode_0 = np.zeros((res_r, res_theta))

        # Compute coefficients, replacing integrals 
        # by averaged sums in theta space
        for idx_r, f_th_vals in enumerate(f_vals):
            coeff_0[idx_r] = np.sum(f_th_vals)/res_theta

        # Finally, set up modes given coefficients
        for idx_r in range(res_r):
            mode_0[idx_r] = coeff_0[idx_r]

        if nr_modes == 0:
            if return_f_vals:
                return mode_0, f_vals
            return mode_0

        coeffs = np.zeros((nr_modes, res_r, 2))
        modes_m = np.zeros((nr_modes, res_r, res_theta))

        th_vals = np.array([2*np.pi*k/res_theta for k in range(res_theta)])
        for idx_c, cffs in enumerate(coeffs):
            for idx_r, f_th_vals in enumerate(f_vals):
                cffs[idx_r][0] = \
                    2*np.sum(f_th_vals*np.cos((idx_c+1)*th_vals))/res_theta
                cffs[idx_r][1] = \
                    2*np.sum(f_th_vals*np.sin((idx_c+1)*th_vals))/res_theta

        for idx_m, mode in enumerate(modes_m):
            for idx_r in range(res_r):
                mode[idx_r] += coeffs[idx_m][idx_r][0]*np.cos((idx_m+1)*th_vals)
                mode[idx_r] += coeffs[idx_m][idx_r][1]*np.sin((idx_m+1)*th_vals)

        modes = np.zeros((nr_modes+1, res_r, res_theta))
        modes[0, :, :] = mode_0
        modes[1:, :, :] = modes_m
        if return_f_vals:
            return modes, f_vals
        return modes

    def plot_r_theta_field(self, r_th_fld, title=None,
                           cmap=cm.viridis, s_=2, skip_separatrix=False):
        """Plot field that is given in r-theta coordinates in (R, Z) poloidal
        plane."""
        RZ_array = np.vstack((self.rth_to_RZ[:, :, 0].flatten(),
                              self.rth_to_RZ[:, :, 1].flatten())).transpose()

        sc = plt.scatter(RZ_array[:, 0], RZ_array[:, 1], 
                         c=r_th_fld, cmap=cmap, s=s_)
        plt.colorbar(sc)
        if not skip_separatrix:
            plt.plot(self.handler.cntr_separatrix[:, 0],
                     self.handler.cntr_separatrix[:, 1],
                     color='red', label='Separatrix loop',
                     linestyle=(0, (5, 5)))
            plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()


def get_wall_patch_conditionals(ra, za, sep_at_wall0, sep_at_wall1, rz_center,
                                dist_=0.5, z_w=2/3., z_neg=-2, z_pos=2,
                                min_z=-6, max_z=6, symbolic=False,
                                right_triple=True):
    """Return dictionary of conditionals specifying patches of a poloidal plane
    for which to do e.g. polynomial fits. There are 9 patches: 1 for each
    separatrix/boundary meeting point, 1 between the meeting points, 2 for the
    left edge, 3 for the right edge, one for the area opposite the meeting
    points. dist_ specifies the size of the patches at the meeting points,
    z_w the relative z-extend of the opposite patch, and z_neg, z_pos the
    z-extend of the middle patch on the right edge. Optionally use only one
    patch for the right edge (defaults to False)."""
    fit_bools = {}

    r_sep_min = np.min((sep_at_wall0[0], sep_at_wall1[0]))
    r_sep_max = np.max((sep_at_wall0[0], sep_at_wall1[0]))
    z_sep_max = np.max((sep_at_wall0[1], sep_at_wall1[1]))

    if symbolic:
        and_, not_ = and_sym, not_sym
    else:
        and_, not_ = np.logical_and, np.logical_not

    # Build two small regions near where the separatrix meets the boundary
    def _get_bool_near_point(p_, dst):
        """Return sub-array condition of coordinate array outer_coords only
        containing those coordinates within square with center (pr, pz),
        side length 2*dst."""
        rp, zp = p_
        return and_(and_(ra <= rp + dst, ra >= rp - dst),
                    and_(za <= zp + dst, za >= zp - dst))


    cond_sep0 = _get_bool_near_point(sep_at_wall0, dist_)
    cond_sep1 = _get_bool_near_point(sep_at_wall1, dist_)

    fit_bools['separatrix patch 1'] = cond_sep0
    fit_bools['separatrix patch 2'] = cond_sep1

    # Cut remaining coordinates into 4 areas (left right top bottom)
    cond_not_at_sep = and_(not_(cond_sep0), not_(cond_sep1))

    if sep_at_wall0[1] < 0:
        z_cond_bew_sep = za < z_sep_max
        # Need to write out not condition for ge etc. because
        # it is not implemented for that on the symbolic side
        cond_opp_sep = za > (rz_center[1]*(1 - z_w) + max_z*z_w)
        not_cond_opp_sep = za <= (rz_center[1]*(1 - z_w) + max_z*z_w)
    else:
        z_cond_bew_sep = za > z_sep_max
        cond_opp_sep = za < (rz_center[1]*(1 - z_w) + min_z*z_w)
        not_cond_opp_sep = za >= (rz_center[1]*(1 - z_w) + min_z*z_w)

    # Fit petween separatrix/boundary meeting points
    cond_betw_sep = and_(and_(ra > r_sep_min, ra < r_sep_max),
                         z_cond_bew_sep)
    cond_between_sep = and_(cond_not_at_sep, cond_betw_sep)
    fit_bools['between separatrix'] = cond_between_sep

    # Fit opposite of separatrix/boundary meeting points
    cond_away_sep = and_(cond_not_at_sep, not_(cond_between_sep))
    fit_bools['opposite separatrix'] = and_(cond_opp_sep, cond_away_sep)

    # Fits for left and right sides of the boundary
    cond_at_lr = and_(cond_away_sep, not_cond_opp_sep)
    cond_r, not_cond_r = ra > rz_center[0], ra <= rz_center[0]
    if right_triple:
        cond_z1, not_cond_z1 = za > z_pos, za <= z_pos
        cond_z2, not_cond_z2 = za < z_neg, za >= z_neg
        fit_bools['right edge top'] = and_(and_(cond_r, cond_z1), cond_at_lr)
        fit_bools['right edge bottom'] = and_(and_(cond_r, cond_z2), cond_at_lr)
        fit_bools['right edge center'] = \
            and_(and_(cond_r, and_(not_cond_z1, not_cond_z2)), cond_at_lr)
    else:
        fit_bools['right edge'] = and_(cond_r, cond_at_lr)

    cond_bz, not_cond_bz = za < 0, za >= 0
    fit_bools['left edge bottom'] = \
        and_(cond_bz, and_(not_cond_r, cond_at_lr))
    fit_bools['left edge top'] = \
        and_(not_cond_bz, and_(not_cond_r, cond_at_lr))

    return fit_bools

def polynomial_fit_2D(coords, arr_, xy=None, deg=3, return_derivative=None):
    """Compute polynomial fit given 2D coordinates, a scalar value array, and
    the desired polynomial degree (up to x**deg*y**deg). Returns the
    polynomial as an expression for the given coordinate xy. If
    return_derivative is set to 'r' or 'z', instead return the polynomial's 'r'
    or 'z' derivative, respectively."""
    expnts = lambda x_, y_: [x_*0 + 1] + [x_**i*y_**j for i in range(deg + 1)
                                          for j in range(deg + 1)][1:]

    x_, y_ = coords[:, 0], coords[:, 1]

    Poly_A = np.array(expnts(x_, y_)).T
    coeff = lstsq(Poly_A, arr_)[0]
    nr_coeffs = len(coeff)

    def _get_exp_term(x_, y_, a, b):
        """Replace negative exponents by zero"""
        if a == -1:
            return x_*0
        if b == -1:
            return y_*0
        return x_**a*y_**b

    if xy is not None:
        x_, y_ = xy[0], xy[1]

    if return_derivative is None:
        expnts_ = expnts(x_, y_)
    elif return_derivative == 'r':
        expnts_ = [x_*0, ] + [i*_get_exp_term(x_, y_, i-1, j)
                              for i in range(deg + 1)
                                  for j in range(deg + 1)][1:]
    elif return_derivative == 'z':
        expnts_ = [x_*0, ] + [j*_get_exp_term(x_, y_, i, j-1)
                              for i in range(deg + 1)
                                  for j in range(deg + 1)][1:]
    else:
        raise ValueError('return_derivative must be either None, "r", or "z".')

    poly_ = coeff[0]*expnts_[0]

    for i in range(1, nr_coeffs):
        poly_ += coeff[i]*expnts_[i]

    return poly_


def and_sym(c1, c2):
    """Logical and in case the input is either a ufl.condition object of type
    Condition, GT, GE, LT, or LE"""
    from firedrake import conditional
    from ufl.conditional import GT, GE, LT, LE, Conditional
    if (not isinstance(c1, (GT, GE, LT, LE, Conditional))
            or not isinstance(c2, (GT, GE, LT, LE, Conditional))):
        raise ValueError('Symbolic and requires ufl Conditional, ' \
                         'GT, GE, LT, or LE.')

    if isinstance(c1, Conditional):
        c1 = c1 > 0
    if isinstance(c2, Conditional):
        c2 = c2 > 0

    return conditional(c1, conditional(c2, 1, 0), 0)


def not_sym(c1):
    """Logical not in case the input is a ufl condition"""
    from firedrake import conditional
    from ufl.conditional import Conditional
    if not isinstance(c1, Conditional):
        raise ValueError('Symbolic nor only implemented for ufl Conditional.')

    if isinstance(c1, Conditional):
        return conditional(c1 > 0, 0, 1)

def read_geqdsk(f):
    """Reads a G-EQDSK file. By Ben Dudson, from tokamak utilites repo
    https://github.com/bendudson/pyTokamak"""

    if isinstance(f, str):
        # If the input is a string, treat as file name
        with open(f) as fh: # Ensure file is closed
            return read_geqdsk(fh) # Call again with file object

    # Read the first line, which should contain the mesh sizes
    desc = f.readline()
    if not desc:
        raise IOError("Cannot read from input file")

    s = desc.split() # Split by whitespace
    if len(s) < 3:
        raise IOError("First line must contain at least 3 numbers")

    idum = int(s[-3])
    nxefit = int(s[-2])
    nyefit = int(s[-1])

    # Use a generator to read numbers
    token = file_numbers(f)

    try:
        xdim   = float(token.next())
        zdim   = float(token.next())
        rcentr = float(token.next())
        rgrid1 = float(token.next())
        zmid   = float(token.next())

        rmagx  = float(token.next())
        zmagx  = float(token.next())
        simagx = float(token.next())
        sibdry = float(token.next())
        bcentr = float(token.next())

        cpasma = float(token.next())
        simagx = float(token.next())
        xdum   = float(token.next())
        rmagx  = float(token.next())
        xdum   = float(token.next())

        zmagx  = float(token.next())
        xdum   = float(token.next())
        sibdry = float(token.next())
        xdum   = float(token.next())
        xdum   = float(token.next())
    except:
        xdim = float(next(token))
        zdim = float(next(token))
        rcentr = float(next(token))
        rgrid1 = float(next(token))
        zmid = float(next(token))

        rmagx = float(next(token))
        zmagx = float(next(token))
        simagx = float(next(token))
        sibdry = float(next(token))
        bcentr = float(next(token))

        cpasma = float(next(token))
        simagx = float(next(token))
        xdum = float(next(token))
        rmagx = float(next(token))
        xdum = float(next(token))

        zmagx = float(next(token))
        xdum = float(next(token))
        sibdry = float(next(token))
        xdum = float(next(token))
        xdum = float(next(token))

    # Read arrays
    def read_array(n, name="Unknown"):
        data = np.zeros([n])
        try:
            for i in np.arange(n):
                try:
                    data[i] = float(token.next())
                except:
                    data[i] = float(next(token))
        except:
            raise IOError("Failed reading array '"+name+"' of size ", n)
        return data

    def read_2d(nx, ny, name="Unknown"):
        data = np.zeros([nx, ny])
        for i in np.arange(nx):
            data[i,:] = read_array(ny, name+"["+str(i)+"]")
        return data

    fpol   = read_array(nxefit, "fpol")
    pres   = read_array(nxefit, "pres")
    workk1 = read_array(nxefit, "workk1")
    workk2 = read_array(nxefit, "workk2")
    psi    = read_2d(nxefit, nyefit, "psi")
    qpsi   = read_array(nxefit, "qpsi")

    # Some files may have nx and ny switched around for psi
    psi_tol = 0.5
    if nxefit != nyefit and abs(psi[0, nyefit-2] - psi[0, nyefit-1]) > psi_tol:
        print("Switching nx, ny for psi array")
        psi = psi.flatten().reshape((nyefit, nxefit))

    # Read boundary and limiters, if present
    try:
        nbdry = int(token.next())
        nlim  = int(token.next())
    except:
        nbdry = int(next(token))
        nlim = int(next(token))

    if nbdry > 0:
        rbdry = np.zeros([nbdry])
        zbdry = np.zeros([nbdry])
        for i in range(nbdry):
            try:
                rbdry[i] = float(token.next())
                zbdry[i] = float(token.next())
            except:
                rbdry[i] = float(next(token))
                zbdry[i] = float(next(token))
    else:
        rbdry = [0]
        zbdry = [0]

    if nlim > 0:
        xlim = np.zeros([nlim])
        ylim = np.zeros([nlim])
        for i in range(nlim):
            try:
                xlim[i] = float(token.next())
                ylim[i] = float(token.next())
            except:
                xlim[i] = float(next(token))
                ylim[i] = float(next(token))
    else:
        xlim = [0]
        ylim = [0]

    # Construct R-Z mesh
    r = np.zeros([nxefit, nyefit])
    z = r.copy()
    for i in range(nxefit):
        r[i,:] = rgrid1 + xdim*i/float(nxefit-1)
    for j in range(nyefit):
        z[:,j] = (zmid-0.5*zdim) + zdim*j/float(nyefit-1)

    # Create dictionary of values to return
    return  {# Number of horizontal and vertical points
             'nx': nxefit, 'ny':nyefit,
             # Location of the grid-points
             'r': r, 'z': z,
             # Size of the domain in meters
             'rdim': xdim, 'zdim': zdim,
             # Reference vacuum toroidal field (m, T)
             'rcentr': rcentr, 'bcentr': bcentr,
             # R of left side of domain
             'rgrid1': rgrid1,
             # Z at the middle of the domain
             'zmid': zmid,
             # Location of magnetic axis
             'rmagx': rmagx, 'zmagx': zmagx,
             # Poloidal flux at the axis (Weber / rad)
             'simagx': simagx,
             # Poloidal flux at plasma boundary (Weber / rad)
             'sibdry': sibdry,
             # ???
             'cpasma': cpasma,
             # Poloidal flux in Weber/rad on grid points
             'psi': psi,
             # Poloidal current function on uniform flux grid
             'fpol': fpol,
             # Plasma pressure in nt/m^2 on uniform flux grid
             'pressure': pres,
             # Derivatives wrt psi of fpol and pres
             'ffprim': workk1, 'pprime': workk2,
             # q values on uniform flux grid
             'qpsi': qpsi,
             # Plasma boundary
             'nbdry': nbdry, 'rbdry': rbdry, 'zbdry': zbdry,
             # Wall boundary
             'nlim': nlim, 'xlim': xlim, 'ylim': ylim}


def file_numbers(fp):
    """Generator to get numbers from a text file"""
    toklist = []
    while True:
        line = fp.readline()
        if not line: break
        # Match numbers in the line using regular expression
        pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?'
        toklist = re.findall(pattern, line)
        for tok in toklist:
            yield tok
