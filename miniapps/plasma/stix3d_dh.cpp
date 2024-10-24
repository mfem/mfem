// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//   -----------------------------------------------------------------------
//   Stix3D_DH Miniapp: Cold Plasma Electromagnetic Simulation Code
//   -----------------------------------------------------------------------
//
//   Assumes that all sources and boundary conditions oscillate with the same
//   frequency although not necessarily in phase with one another.  This
//   assumption implies that we can factor out the time dependence which we
//   take to be of the form exp(-i omega t).  With these assumptions we can
//   write the Maxwell equations for D and H in the form:
//
//   -i omega D    = Curl H - J
//    i omega mu H = Curl epsilon^{-1} D
//
//   Which combine to yield:
//
//   Curl epsilon^{-1} Curl H - omega^2 mu H = Curl epsilon^{-1} J
//
//   In a cold plasma the dielectric tensor, epsilon, is complex-valued and
//   anisotropic.  The anisotropy aligns with the external magnetic field and
//   the values depend on the properties of the plasma including the masses and
//   charges of its constituent ion species.
//
//   For a magnetic field aligned with the z-axis the dielectric tensor has
//   the form:
//              | S  -iD 0 |
//    epsilon = |iD   S  0 |
//              | 0   0  P |
//
//   Where:
//      S = 1 - Sum_species omega_p^2 / (omega^2 - omega_c^2)
//      D = Sum_species omega_p^2 omega_c / (omega^2 - omega_c^2)
//      P = 1 - Sum_species omega_p^2 / omega^2
//
//   and:
//      omega_p is the plasma frequency
//      omega_c is the cyclotron frequency
//      omega   is the driving frequency
//
//   The plasma and cyclotron frequencies depend primarily on the properties
//   of the ion species.  We also include a complex-valued mass correction
//   which depends on the plasma temperature.
//
//   We discretize this equation with H(Curl) a.k.a Nedelec basis
//   functions.  The curl curl operator must be handled with
//   integration by parts which yields a surface integral:
//
//   (W, Curl epsilon^{-1} Curl H) = (Curl W, epsilon^{-1} Curl H)
//               + (W, n x (epsilon^{-1} Curl H))_{\Gamma}
//
//   or
//
//   (W, Curl epsilon^{-1} Curl H) = (Curl W, epsilon^{-1} Curl H)
//               - i omega (W, n x E)_{\Gamma}
//
//   Assuming J = 0 on the boundary
//
//   The non-linear sheath boundary condition can be used to set the
//   components of E that are tangent to the boundary. The governing
//   equations are:
//
//      E_t = - Grad Phi_{RF} (where Phi_{RF} is the sheath potential)
//      Phi_{RF} = i omega z_{sh} D_n
//
//   Where D_n is the normal component of D = epsilon E and z_{sh}
//   is the sheath impedance. The impedance z_{sh} is a function of the
//   plasma density, plasma temperature, ion charges, ion masses,
//   magnetic field strength, and the sheath potential itself. Clearly the
//   dependence on the potential is the source of the non-linearity.
//
//   The sheath boundary condition can be easily incorporated into the
//   weak form of the curl-curl operator:
//
//   (W, Curl epsilon^{-1} Curl H) = (Curl W, epsilon^{-1} Curl H)
//               + i omega (W, n x Grad Phi_{RF})_{\Gamma}
//
//   To compute Phi_{RF} we augment the Maxwell equations with the
//   relation between D_n and Phi_{RF}:
//
//      - i omega z_{sh} D_n + Phi_{RF} = 0
//
//   or
//
//      z_{sh} Curl H + Phi_{RF} = 0
//
// Compile with: make stix3d_dh
//
// Sample runs:
//
//   By default the sources and fields are all zero
//     mpirun -np 4 stix3d_dh
//
// EXAMPLE COMMAND LINES HERE

#include "cold_plasma_dielectric_coefs.hpp"
#include "cold_plasma_dielectric_dh_solver.hpp"
#include "../common/mesh_extras.hpp"
#include "plasma.hpp"
#include "g_eqdsk_data.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <complex>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::plasma;

class MeshTransformCoefficient : public VectorCoefficient
{
private:
   double hphi_rad_;

   mutable Vector uvw_;

public:
   MeshTransformCoefficient(double hphi_deg)
      : VectorCoefficient(3), hphi_rad_(hphi_deg * M_PI / 180.0),
        uvw_(3)
   {}

   void Eval(Vector &xyz, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      T.Transform(ip, uvw_);

      const double r   = uvw_[0];
      const double phi = hphi_rad_ * (1.0 - uvw_[2]);
      const double z   = uvw_[1];

      xyz[0] = r * cos(phi);
      xyz[1] = r * sin(phi);
      xyz[2] = z;
   }
};

class VectorConstantCylCoefficient : public VectorCoefficient
{
private:
   bool cyl;
   Vector vec;
   mutable Vector x;
public:
   /** The constant vector v is defined in either cartesian or cylindrical
       coordinates.

       If cyl == true
          v = (v_r, v_phi, v_z)
       Else
          v = (v_x, v_y, v_z)
   */
   VectorConstantCylCoefficient(bool cyl_, const Vector &v)
      : VectorCoefficient(3), cyl(cyl_), vec(v), x(3) {}
   using VectorCoefficient::Eval;

   ///  Evaluate the vector coefficient at @a ip.
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      if (cyl)
      {
         V.SetSize(3);

         T.Transform(ip, x);

         double r = sqrt(x[0] * x[0] + x[1] * x[1]);
         double cosphi = x[0] / r;
         double sinphi = x[1] / r;

         V[0] = vec[0] * cosphi - vec[1] * sinphi / r;
         V[1] = vec[0] * sinphi + vec[1] * cosphi / r;
         V[2] = vec[2];
      }
      else
      {
         V = vec;
      }
   }

   /// Return a reference to the constant vector in this class.
   const Vector& GetVec() { return vec; }
};

// Admittance for Absorbing Boundary Condition
Coefficient * SetupImpedanceCoefficient(const Mesh & mesh,
                                        const Array<int> & abcs);

// Storage for user-supplied, real-valued impedance
static Vector pw_eta_(0);      // Piecewise impedance values
static Vector pw_bdr_eta_(0);  // Piecewise impedance values (by bdr attr)

// Storage for user-supplied, complex-valued impedance
//static Vector pw_eta_re_(0);      // Piecewise real impedance
//static Vector pw_eta_inv_re_(0);  // Piecewise inverse real impedance
//static Vector pw_eta_im_(0);      // Piecewise imaginary impedance
//static Vector pw_eta_inv_im_(0);  // Piecewise inverse imaginary impedance

// Current Density Function
static bool j_cyl_ = false;
static Vector rod_params_
(0); // Amplitude of x, y, z current source, position in 2D, and radius
static Vector slab_params_
(0); // Amplitude of x, y, z current source, position in 2D, and size in 2D
static Vector curve_params_
(0); // Text here
static int slab_profile_;
static int vol_profile_ = 0;

void rod_current_source_r(const Vector &x, Vector &j);
void rod_current_source_i(const Vector &x, Vector &j);
void slab_current_source_r(const Vector &x, Vector &j);
void slab_current_source_i(const Vector &x, Vector &j);
void curve_current_source_r(const Vector &x, Vector &j);
void curve_current_source_i(const Vector &x, Vector &j);
void j_src_r(const Vector &x, Vector &j)
{
   if (rod_params_.Size() > 0)
   {
      rod_current_source_r(x, j);
   }
   else if (slab_params_.Size() > 0)
   {
      slab_current_source_r(x, j);
   }
   else if (curve_params_.Size() > 0)
   {
      curve_current_source_r(x, j);
   }
}
void j_src_i(const Vector &x, Vector &j)
{
   if (rod_params_.Size() > 0)
   {
      rod_current_source_i(x, j);
   }
   else if (slab_params_.Size() > 0)
   {
      slab_current_source_i(x, j);
   }
   else if (curve_params_.Size() > 0)
   {
      curve_current_source_i(x, j);
   }
}

// Electric Field Boundary Condition: The following function returns zero but
// any function could be used.
void e_bc_r(const Vector &x, Vector &E);
void e_bc_i(const Vector &x, Vector &E);
class ColdPlasmaPlaneWaveH: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWaveH(char type,
                        double omega,
                        const Vector & B,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        double res_lim,
                        bool realPart);

   void SetCurrentSlab(double Jy, double xJ, double delta, double Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }
   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   int nuprof_;
   double res_lim_;
   double omega_;
   double Bmag_;
   double Jy_;
   double xJ_;
   double dx_;
   double Lx_;
   complex<double> kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector h_r_;
   Vector h_i_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   // const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<double> S_;
   complex<double> D_;
   complex<double> P_;
};

class ColdPlasmaPlaneWaveE: public VectorCoefficient
{
public:
   ColdPlasmaPlaneWaveE(char type,
                        double omega,
                        const Vector & B,
                        const Vector & number,
                        const Vector & charge,
                        const Vector & mass,
                        const Vector & temp,
                        int nuprof,
                        double res_lim,
                        bool realPart);

   void SetCurrentSlab(double Jy, double xJ, double delta, double Lx)
   { Jy_ = Jy; xJ_ = xJ; dx_ = delta, Lx_ = Lx; }

   void SetPhaseShift(const Vector & beta)
   { beta_r_ = beta; beta_i_ = 0.0; }
   void SetPhaseShift(const Vector & beta_r,
                      const Vector & beta_i)
   { beta_r_ = beta_r; beta_i_ = beta_i; }

   void GetWaveVector(Vector & k_r, Vector & k_i) const
   { k_r = k_r_; k_i = k_i_; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip);

private:
   char type_;
   bool realPart_;
   int nuprof_;
   double res_lim_;
   double omega_;
   double Bmag_;
   double Jy_;
   double xJ_;
   double dx_;
   double Lx_;
   complex<double> kappa_;
   Vector b_;   // Normalized vector in direction of B
   Vector bc_;  // Normalized vector perpendicular to b_, (by-bz,bz-bx,bx-by)
   Vector bcc_; // Normalized vector perpendicular to b_ and bc_
   Vector e_r_;
   Vector e_i_;
   Vector k_r_;
   Vector k_i_;
   Vector beta_r_;
   Vector beta_i_;

   // const Vector & B_;
   const Vector & numbers_;
   const Vector & charges_;
   const Vector & masses_;
   const Vector & temps_;

   complex<double> S_;
   complex<double> D_;
   complex<double> P_;
};
void AdaptInitialMesh(ParMesh &pmesh,
                      ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace & H1FESpace,
                      ParFiniteElementSpace & H1VFESpace,
                      ParFiniteElementSpace & HCurlFESpace,
                      ParFiniteElementSpace & HDivFESpace,
                      ParFiniteElementSpace & L2FESpace,
                      VectorCoefficient & BCoef,
                      VectorCoefficient & kReCoef,
                      Coefficient & rhoCoef,
                      Coefficient & TeCoef,
                      Coefficient & TiCoef,
                      Coefficient & nueCoef,
                      Coefficient & nuiCoef,
                      int & size_h1,
                      int & size_l2,
                      Array<int> & density_offsets,
                      Array<int> & temperature_offsets,
                      BlockVector & density,
                      BlockVector & temperature,
                      ParGridFunction & BField,
                      ParGridFunction & k_gf,
                      ParGridFunction & density_gf,
                      ParGridFunction & temperature_gf,
                      ParGridFunction & iontemp_gf,
                      ParGridFunction & nue_gf,
                      ParGridFunction & nui_gf,
                      Coefficient &ReCoef,
                      Coefficient &ImCoef,
                      int p, double tol, int max_its, int max_dofs,
                      bool visualization);

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & H1VFESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            VectorCoefficient & BCoef,
            VectorCoefficient & kReCoef,
            Coefficient & rhoCoef,
            Coefficient & TeCoef,
            Coefficient & TiCoef,
            Coefficient & nueCoef,
            Coefficient & nuiCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & BField,
            ParGridFunction & k_gf,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf,
            ParGridFunction & iontemp_gf,
            ParGridFunction & nue_gf,
            ParGridFunction & nui_gf);

//static double freq_ = 1.0e9;

// Mesh Size
//static Vector mesh_dim_(0); // x, y, z dimensions of mesh

// Prints the program's logo to the given output stream
void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);

   if ( Mpi::Root() ) { display_banner(cout); }

   int logging = 1;

   // Parse command-line options.
   const char *mesh_file = "ellipse_origin_h0pt0625_o3.mesh";
   int mesh_order = -1;
   double init_amr_tol = 1e-5;
   int init_amr_max_its = 10;
   int init_amr_max_dofs = 100000;
   int ser_ref_levels = 0;
   int order = 1;
   int maxit = 100;
   int sol = 4;
   int prec = 1;
   // int nspecies = 2;
   bool amr_stix = false;
   int amr_coef = 0;
   bool herm_conv = false;
   bool vis_u = false;
   bool visualization = true;
   bool visit = true;

   double freq = 1.0e6;
   const char * wave_type = " ";

   Vector BVec(3);
   BVec = 0.0; BVec(0) = 0.1;

   bool phase_shift = false;
   Vector kVec;
   Vector kReVec;
   Vector kImVec;

   double hz = -1.0; // Extruded mesh thickness in meters
   double hphi = -1.0; // Cylindrically extruded mesh thickness in degrees

   Vector numbers;
   Vector charges;
   Vector masses;
   Vector temps;
   Vector minority;
   Vector temp_charges;
   double nue = 0;
   double nui = 0;
   double Ti = 0;

   PlasmaProfile::Type dpt_def = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type dpt_vac = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type dpt_sol = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type dpt_cor = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt_def = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt_vac = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt_sol = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tpt_cor = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type nept = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type nipt = PlasmaProfile::CONSTANT;
   PlasmaProfile::Type tipt = PlasmaProfile::CONSTANT;
   BFieldProfile::Type bpt = BFieldProfile::CONSTANT;
   Array<int> dpa_vac;
   Array<int> dpa_sol;
   Array<int> dpa_cor;
   Array<int> tpa_vac;
   Array<int> tpa_sol;
   Array<int> tpa_cor;
   Vector dpp_def;
   Vector dpp_vac;
   Vector dpp_sol;
   Vector dpp_cor;
   Vector tpp_def;
   Vector tpp_vac;
   Vector tpp_sol;
   Vector tpp_cor;
   Vector bpp;
   Vector nepp;
   Vector nipp;
   Vector tipp;
   int nuprof = 0;
   double res_lim = 0.01;

   Array<int> abcs; // Absorbing BC attributes
   Array<int> sbca; // Sheath BC attributes
   Array<int> peca; // Perfect Electric Conductor BC attributes
   Array<int> dbca1; // Dirichlet BC attributes
   Array<int> dbca2; // Dirichlet BC attributes
   Array<int> dbcas; // Dirichlet BC attributes for multi-strap antenna source
   Array<int> dbcaw; // Dirichlet BC attributes for plane wave source
   Array<int> nbca1; // Neumann BC attributes
   Array<int> nbca2; // Neumann BC attributes
   Array<int> nbcaw; // Neumann BC attributes for plane wave source
   Vector dbcv1; // Dirichlet BC values
   Vector dbcv2; // Dirichlet BC values
   Vector nbcv1; // Neumann BC values
   Vector nbcv2; // Neumann BC values

   int msa_n = 0;
   Vector msa_p(0);
   Vector msa_c(0);

   int num_elements = 10;

   SolverOptions solOpts;
   solOpts.maxIter = 1000;
   solOpts.kDim = 50;
   solOpts.printLvl = 1;
   solOpts.relTol = 1e-4;
   solOpts.absTol = 1e-6;
   solOpts.euLvl = 1;

   bool logo = false;
   bool cyl = false;
   bool per_y = false;
   bool check_eps_inv = false;
   bool pa = false;
   const char *device_config = "cpu";
   const char *eqdsk_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&logo, "-logo", "--print-logo", "-no-logo",
                  "--no-print-logo", "Print logo and exit.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_order, "-mo", "--mesh-order",
                  "Geometry order for cylindrically symmetric mesh.");
   args.AddOption(&cyl, "-cyl", "--cylindrical-coords", "-cart",
                  "--cartesian-coords",
                  "Cartesian (x, y, z) coordinates or "
                  "Cylindrical (z, rho, phi).");
   args.AddOption(&per_y, "-per-y", "--periodic-in-y", "-no-per-y",
                  "--not-periodic-in-y",
                  "The input mesh is periodic in the y-direction.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&amr_stix, "-amr-stix", "--init-amr-stix", "-no-amr-stix",
                  "--no-init-amr-stix",
                  "Initial AMR to capture Stix S or P coefficient.");
   args.AddOption(&amr_coef, "-amr-coef", "--init-amr-coef",
                  "Choose which Stix coef to refine, 0 - S, 1 - P.");
   args.AddOption(&init_amr_tol, "-iatol", "--init-amr-tol",
                  "Initial AMR tolerance.");
   args.AddOption(&init_amr_max_its, "-iamit", "--init-amr-max-its",
                  "Initial AMR Maximum Number of Iterations.");
   args.AddOption(&init_amr_max_dofs, "-iamdof", "--init-amr-max-dofs",
                  "Initial AMR Maximum Number of DoFs.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   // args.AddOption(&nspecies, "-ns", "--num-species",
   //               "Number of ion species.");
   args.AddOption(&freq, "-f", "--frequency",
                  "Frequency in Hertz (of course...)");
   args.AddOption(&hz, "-mh", "--mesh-height",
                  "Thickness of extruded mesh in meters.");
   args.AddOption(&hphi, "-mhc", "--mesh-height-cyl",
                  "Thickness of cylindrically extruded mesh in degrees.");
   args.AddOption((int*)&dpt_def, "-dp", "--density-profile",
                  "Density Profile Type (for ions): \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&dpp_def, "-dpp", "--density-profile-params",
                  "Density Profile Parameters:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&dpa_vac, "-dpa-vac", "-vacuum-density-profile-attr",
                  "Density Profile Vacuum Attributes");
   args.AddOption((int*)&dpt_vac, "-dp-vac", "--vacuum-density-profile",
                  "Density Profile Type (for ions) in Vacuum: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&dpp_vac, "-dpp-vac", "--vacuum-density-profile-params",
                  "Density Profile Parameters in Vacuum:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&dpa_sol, "-dpa-sol", "-sol-density-profile-attr",
                  "Density Profile Scrape-off Layer Attributes");
   args.AddOption((int*)&dpt_sol, "-dp-sol", "--sol-density-profile",
                  "Density Profile Type (for ions) in Scrape-off Layer: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&dpp_sol, "-dpp-sol", "--sol-density-profile-params",
                  "Density Profile Parameters in Scrape-off Layer:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&dpa_cor, "-dpa-core", "-core-density-profile-attr",
                  "Density Profile Core Attributes");
   args.AddOption((int*)&dpt_cor, "-dp-core", "--core-density-profile",
                  "Density Profile Type (for ions) in Core region: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&dpp_cor, "-dpp-core", "--core-density-profile-params",
                  "Density Profile Parameters in Core region:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&bpt, "-bp", "--Bfield-profile",
                  "BField Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&bpp, "-bpp", "--Bfield-profile-params",
                  "BField Profile Parameters:\n"
                  "  B_P: value at -1, value at 1, "
                  "radius in x, radius in y, location of center, Bz, placeholder.");
   args.AddOption((int*)&tpt_def, "-tp", "--temperature-profile",
                  "Temperature Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tpp_def, "-tpp", "--temperature-profile-params",
                  "Temperature Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&tpa_vac, "-tpa-vac", "-vac-temp-profile-attr",
                  "Temperature Profile (for ions) in Vacuum");
   args.AddOption((int*)&tpt_vac, "-tp-vac", "--vac-temp-profile",
                  "Temperature Profile Type (for ions) in Vacuum: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tpp_vac, "-tpp-vac", "--vac-temp-profile-params",
                  "Temperature Profile Parameters in Vacuum:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&tpa_sol, "-tpa-sol", "-sol-temp-profile-attr",
                  "Temperature Profile Scrape-off Layer Attributes");
   args.AddOption((int*)&tpt_sol, "-tp-sol", "--sol-temp-profile",
                  "Temperature Profile Type (for ions) in Scrape-off Layer: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tpp_sol, "-tpp-sol", "--sol-temp-profile-params",
                  "Temperature Profile Parameters in Scrape-off Layer:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&tpa_cor, "-tpa-core", "-core-temp-profile-attr",
                  "Temperature Profile Core Attributes");
   args.AddOption((int*)&tpt_cor, "-tp-core", "--core-temp-profile",
                  "Temperature Profile Type (for ions) in Core region: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyprebolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tpp_cor, "-tpp-core", "--core-temp-profile-params",
                  "Temperature Profile Parameters in Core region:\n"
                  "   CONSTANT: density value\n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&nept, "-nep", "--electron-collision-profile",
                  "Electron Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&nepp, "-nepp", "--electron-collisions-profile-params",
                  "Electron Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&nipt, "-nip", "--ion-collision-profile",
                  "Ion Collisions Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&nipp, "-nipp", "--ion-collisions-profile-params",
                  "Ion Collisions Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption((int*)&tipt, "-tip", "--min-ion-temp-profile",
                  "Minority Ion Temperature Profile Type: \n"
                  "0 - Constant, 1 - Constant Gradient, "
                  "2 - Hyperbolic Tangent, 3 - Elliptic Cosine.");
   args.AddOption(&tipp, "-tipp", "--min-ion-temp-profile-params",
                  "Minority Ion Temperature Profile Parameters: \n"
                  "   CONSTANT: temperature value \n"
                  "   GRADIENT: value, location, gradient (7 params)\n"
                  "   TANH:     value at 0, value at 1, skin depth, "
                  "location of 0 point, unit vector along gradient, "
                  "   ELLIPTIC_COS: value at -1, value at 1, "
                  "radius in x, radius in y, location of center.");
   args.AddOption(&nuprof, "-nuprof", "--collisional-profile",
                  "Temperature Profile Type: \n"
                  "0 - Standard e-i Collision Freq, 1 - Custom Freq.");
   args.AddOption(&res_lim, "-res-lim", "--resonance-limiter",
                  "Resonance limit factor [0,1).");
   args.AddOption(&wave_type, "-w", "--wave-type",
                  "Wave type: 'R' - Right Circularly Polarized, "
                  "'L' - Left Circularly Polarized, "
                  "'O' - Ordinary, 'X' - Extraordinary, "
                  "'J' - Current Slab (in conjunction with -slab), "
                  "'Z' - Zero");
   // args.AddOption(&BVec, "-B", "--magnetic-flux",
   //                "Background magnetic flux vector");
   args.AddOption(&kVec, "-k-vec", "--phase-vector",
                  "Phase shift vector across periodic directions."
                  " For complex phase shifts input 3 real phase shifts "
                  "followed by 3 imaginary phase shifts");
   args.AddOption(&msa_n, "-ns", "--num-straps","");
   args.AddOption(&msa_p, "-sp", "--strap-params","");
   args.AddOption(&msa_c, "-sc", "--strap-currents","");
   //args.AddOption(&numbers, "-num", "--number-densites",
   //               "Number densities of the various species");
   args.AddOption(&charges, "-q", "--charges",
                  "Charges of the various species "
                  "(in units of electron charge)");
   //args.AddOption(&masses, "-mass", "--masses",
   //               "Masses of the various species (in amu)");
   args.AddOption(&minority, "-min", "--minority",
                  "Minority Ion Species: charge, mass (amu), concentration."
                  " Concentration being: n_min/n_e");
   args.AddOption(&prec, "-pc", "--precond",
                  "Preconditioner: 1 - Diagonal Scaling, 2 - ParaSails, "
                  "3 - Euclid, 4 - AMS");
   args.AddOption(&sol, "-s", "--solver",
                  "Solver: 1 - GMRES, 2 - FGMRES, 3 - MINRES"
#ifdef MFEM_USE_SUPERLU
                  ", 4 - SuperLU"
#endif
#ifdef MFEM_USE_STRUMPACK
                  ", 5 - STRUMPACK"
#endif
                 );
   args.AddOption(&solOpts.maxIter, "-sol-it", "--solver-iterations",
                  "Maximum number of solver iterations.");
   args.AddOption(&solOpts.kDim, "-sol-k-dim", "--solver-krylov-dimension",
                  "Krylov space dimension for GMRES and FGMRES.");
   args.AddOption(&solOpts.relTol, "-sol-tol", "--solver-tolerance",
                  "Relative tolerance for GMRES or FGMRES.");
   args.AddOption(&solOpts.absTol, "-sol-abs-tol",
                  "--solver-absolute-tolerance",
                  "Absolute tolerance for GMRES or FGMRES.");
   args.AddOption(&solOpts.printLvl, "-sol-prnt-lvl", "--solver-print-level",
                  "Logging level for solvers.");
   args.AddOption(&solOpts.euLvl, "-eu-lvl", "--euclid-level",
                  "Euclid factorization level for ILU(k).");
   args.AddOption(&pw_eta_, "-pwz", "--piecewise-eta",
                  "Piecewise values of Impedance (one value per abc surface)");
   /*
   args.AddOption(&pw_eta_re_, "-pwz-r", "--piecewise-eta-r",
                  "Piecewise values of Real part of Complex Impedance "
                  "(one value per abc surface)");
   args.AddOption(&pw_eta_im_, "-pwz-i", "--piecewise-eta-i",
                  "Piecewise values of Imaginary part of Complex Impedance "
                  "(one value per abc surface)");
   */
   args.AddOption(&rod_params_, "-rod", "--rod_params",
                  "3D Vector Amplitude (Real x,y,z, Imag x,y,z), "
                  "2D Position, Radius");
   args.AddOption(&slab_params_, "-slab", "--slab_params",
                  "3D Vector Amplitude (Real x,y,z, Imag x,y,z), "
                  "2D Position, 2D Size");
   args.AddOption(&curve_params_, "-curve", "--curve_params",
                  "2D Vector Amplitude (Real theta,phi, theta,phi)");
   args.AddOption(&vol_profile_, "-vol-prof", "--vol_profile",
                  "0 (Constant) or 1 (Sin Function)");
   args.AddOption(&slab_profile_, "-slab-prof", "--slab_profile",
                  "0 (Constant) or 1 (Sin Function)");
   args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
                  "Absorbing Boundary Condition Surfaces");
   args.AddOption(&sbca, "-sbcs", "--sheath-bc-surf",
                  "Sheath Boundary Condition Surfaces");
   args.AddOption(&peca, "-pecs", "--pec-bc-surf",
                  "Perfect Electrical Conductor Boundary Condition Surfaces");
   args.AddOption(&dbcaw, "-dbcs-pw", "--dirichlet-bc-pw-surf",
                  "Dirichlet Boundary Condition Surfaces Using Plane Wave");
   args.AddOption(&dbca1, "-dbcs1", "--dirichlet-bc-1-surf",
                  "Dirichlet Boundary Condition Surfaces Using Value 1");
   args.AddOption(&dbca2, "-dbcs2", "--dirichlet-bc-2-surf",
                  "Dirichlet Boundary Condition Surfaces Using Value 2");
   args.AddOption(&dbcv1, "-dbcv1", "--dirichlet-bc-1-vals",
                  "Dirichlet Boundary Condition Value 1 (v_x v_y v_z)"
                  " or (Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&dbcv2, "-dbcv2", "--dirichlet-bc-2-vals",
                  "Dirichlet Boundary Condition Value 2 (v_x v_y v_z)"
                  " or (Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&dbcas, "-dbcs-msa", "--dirichlet-bc-straps",
                  "Dirichlet Boundary Condition Surfaces Using "
                  "Multi-Strap Antenna");
   args.AddOption(&nbca1, "-nbcs1", "--neumann-bc-1-surf",
                  "Neumann Boundary Condition Surfaces Using Value 1");
   args.AddOption(&nbca2, "-nbcs2", "--neumann-bc-2-surf",
                  "Neumann Boundary Condition Surfaces Using Value 2");
   args.AddOption(&nbcaw, "-nbcs-pw", "--neumann-bc-pw-surf",
                  "Neumann Boundary Condition Surfaces Using Plane Wave");
   args.AddOption(&nbcv1, "-nbcv1", "--neumann-bc-1-vals",
                  "Neuamnn Boundary Condition (surface current) "
                  "Value 1 (v_x v_y v_z) or "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&nbcv2, "-nbcv2", "--neumann-bc-2-vals",
                  "Neumann Boundary Condition (surface current) "
                  "Value 2 (v_x v_y v_z) or "
                  "(Re(v_x) Re(v_y) Re(v_z) Im(v_x) Im(v_y) Im(v_z))");
   args.AddOption(&num_elements, "-nume", "--num-elements",
                "The number of mesh elements in extruded direction");
   args.AddOption(&maxit, "-maxit", "--max-amr-iterations",
                  "Max number of iterations in the main AMR loop.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&vis_u, "-vis-u", "--visualize-energy", "-no-vis-u",
                  "--no-visualize-energy",
                  "Enable or disable visualization of energy density.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&eqdsk_file, "-eqdsk", "--eqdsk-file",
                  "G EQDSK input file.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (logo)
   {
      return 1;
   }
   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }

   if (dpp_def.Size() == 0)
   {
      dpp_def.SetSize(1);
      dpp_def[0] = 1.0e19;
   }

   if (nepp.Size() == 0)
   {
      nepp.SetSize(1);
      nepp[0] = 0;
   }
   if (nipp.Size() == 0)
   {
      nipp.SetSize(1);
      nipp[0] = 0;
   }
   if (tipp.Size() == 0)
   {
      tipp.SetSize(1);
      tipp[0] = 0;
   }

   if (bpp.Size() == 0)
   {
      bpt = BFieldProfile::CONSTANT;
      bpp.SetSize(3);
      bpp[0] = 0.0; bpp[1] = 0.1; bpp[2] = 0.0;
   }
   if (bpt == BFieldProfile::CONSTANT)
   {
      BVec = bpp;
   }

   if (charges.Size() == 0)
   {
      if (minority.Size() == 0)
      {
         charges.SetSize(2);
         charges[0] = -1.0;
         charges[1] =  1.0;

         masses.SetSize(2);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
      }
      else
      {
         charges.SetSize(3);
         charges[0] = -1.0;
         charges[1] =  1.0;
         charges[2] = minority[0];

         masses.SetSize(3);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = minority[1];
      }
   }
   if (minority.Size() == 0)
   {
      if (charges.Size() == 2)
      {
         numbers.SetSize(2);
         masses.SetSize(2);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         switch (dpt_def)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp_def[0];
               numbers[1] = dpp_def[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp_def[0];
               numbers[1] = dpp_def[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp_def[1];
               numbers[1] = dpp_def[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp_def[1];
               numbers[1] = dpp_def[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp_def[1];
               numbers[1] = dpp_def[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 1.0e19;
               break;
         }
      }
      else
      {
         numbers.SetSize(3);
         masses.SetSize(3);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = 3.01604928;
         switch (dpt_def)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp_def[0];
               numbers[1] = 0.5*dpp_def[0];
               numbers[2] = 0.5*dpp_def[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp_def[0];
               numbers[1] = 0.5*dpp_def[0];
               numbers[2] = 0.5*dpp_def[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp_def[1];
               numbers[1] = 0.5*dpp_def[1];
               numbers[2] = 0.5*dpp_def[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp_def[1];
               numbers[1] = 0.5*dpp_def[1];
               numbers[2] = 0.5*dpp_def[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp_def[1];
               numbers[1] = 0.5*dpp_def[1];
               numbers[2] = 0.5*dpp_def[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 0.5*1.0e19;
               numbers[2] = 0.5*1.0e19;
               break;
         }
      }
   }
   if (minority.Size() > 0)
   {
      if (charges.Size() == 2)
      {
         temp_charges.SetSize(3);
         temp_charges[0] = charges[0];
         temp_charges[1] = charges[1];
         temp_charges[2] = minority[0];
         charges.SetSize(3);
         charges = temp_charges;

         numbers.SetSize(3);
         masses.SetSize(3);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = minority[1];
         switch (dpt_def)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp_def[0];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp_def[0];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))
                            *dpp_def[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp_def[0];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp_def[0];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))
                            *dpp_def[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp_def[1];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp_def[1];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))
                            *dpp_def[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp_def[1];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp_def[1];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))
                            *dpp_def[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp_def[1];
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*dpp_def[1];
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))
                            *dpp_def[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = (1.0/(1.0+minority[0]*minority[2]))*1.0e19;
               numbers[2] = ((minority[0]*minority[2])/(1.0+minority[0]*minority[2]))*1.0e19;
               break;
         }
      }
      else
      {
         temp_charges.SetSize(4);
         temp_charges[0] = charges[0];
         temp_charges[1] = charges[1];
         temp_charges[2] = charges[2];
         temp_charges[3] = minority[0];
         charges.SetSize(4);
         charges = temp_charges;

         numbers.SetSize(4);
         masses.SetSize(4);
         masses[0] = me_u_;
         masses[1] = 2.01410178;
         masses[2] = 3.01604928;
         masses[3] = minority[1];
         switch (dpt_def)
         {
            case PlasmaProfile::CONSTANT:
               numbers[0] = dpp_def[0];
               numbers[1] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[0];
               numbers[2] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[0];
               numbers[3] = minority[2]*dpp_def[0];
               break;
            case PlasmaProfile::GRADIENT:
               numbers[0] = dpp_def[0];
               numbers[1] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[0];
               numbers[2] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[0];
               numbers[3] = minority[2]*dpp_def[0];
               break;
            case PlasmaProfile::TANH:
               numbers[0] = dpp_def[1];
               numbers[1] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[1];
               numbers[2] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[1];
               numbers[3] = minority[2]*dpp_def[1];
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               numbers[0] = dpp_def[1];
               numbers[1] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[1];
               numbers[2] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[1];
               numbers[3] = minority[2]*dpp_def[1];
               break;
            case PlasmaProfile::PARABOLIC:
               numbers[0] = dpp_def[1];
               numbers[1] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[1];
               numbers[2] = 0.5*(1.0-minority[0]*minority[2])*dpp_def[1];
               numbers[3] = minority[2]*dpp_def[1];
               break;
            default:
               numbers[0] = 1.0e19;
               numbers[1] = 0.5*(1.0-minority[0]*minority[2])*1.0e19;
               numbers[2] = 0.5*(1.0-minority[0]*minority[2])*1.0e19;
               numbers[3] = minority[2]*1.0e19;
               break;
         }
      }
   }
   if (temps.Size() == 0)
   {
      temps.SetSize(numbers.Size());
      if (tpp_def.Size() == 0)
      {
         tpp_def.SetSize(1);
         tpp_def[0] = 1.0e3;
         for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp_def[0];}
      }
      else
      {
         switch (tpt_def)
         {
            case PlasmaProfile::CONSTANT:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp_def[0];}
               break;
            case PlasmaProfile::GRADIENT:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp_def[0];}
               break;
            case PlasmaProfile::TANH:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp_def[1];}
               break;
            case PlasmaProfile::ELLIPTIC_COS:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp_def[1];}
               break;
            case PlasmaProfile::PARABOLIC:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = tpp_def[1];}
               break;
            default:
               for (int i=0; i<numbers.Size(); i++) {temps[i] = 1e3;}
               break;
         }
      }
   }
   if (num_elements <= 0)
   {
      num_elements = 10;
   }
   if (hz < 0.0 && !cyl)
   {
      hz = 0.1;
   }
   if (cyl)
   {
      if (mesh_order <= 0)
      {
         mesh_order = 1;
      }
      if (hphi < 0.0)
      {
         hphi = 3;
      }
      hz = 1.0;

      j_cyl_ = cyl;
   }
   double omega = 2.0 * M_PI * freq;
   if (kVec.Size() != 0)
   {
      phase_shift = true;
   }

   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   if (Mpi::Root())
   {
      double lam0 = c0_ / freq;
      double Bmag = 5.4; //BVec.Norml2();
      double kvecmag = kVec.Norml2();
      double Rval = 0.0;
      double Lval = 0.0;

      std::complex<double> S = S_cold_plasma(omega, kvecmag, Bmag, nue, nui, numbers,
                                             charges, masses, temps, Ti, nuprof,
                                             Rval,Lval);
      std::complex<double> P = P_cold_plasma(omega, kvecmag, nue, numbers,
                                             charges, masses, temps, Ti, nuprof);
      std::complex<double> D = D_cold_plasma(omega, kvecmag, Bmag, nue, nui, numbers,
                                             charges, masses, temps, Ti, nuprof,
                                             Rval,Lval);
      std::complex<double> R = R_cold_plasma(omega, Bmag, nue, nui, numbers,
                                             charges, masses, temps, Ti, nuprof);
      std::complex<double> L = L_cold_plasma(omega, Bmag, nue, nui, numbers,
                                             charges, masses, temps, Ti, nuprof);

      cout << "\nConvenient Terms:\n";
      cout << "R = " << R << ",\tL = " << L << endl;
      cout << "S = " << S << ",\tD = " << D << ",\tP = " << P << endl;

      cout << "\nSpecies Properties (number, charge, mass):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << numbers[i] << '\t' << charges[i] << '\t' << masses[i] << '\n';
      }
      cout << "\nPlasma and Cyclotron Frequencies by Species (GHz):\n";
      for (int i=0; i<numbers.Size(); i++)
      {
         cout << omega_p(numbers[i], charges[i], masses[i]) / (2.0e9 * M_PI)
              << '\t'
              << omega_c(Bmag, charges[i], masses[i]) / (2.0e9 * M_PI) << '\n';
      }

      cout << "\nWavelengths (meters):\n";
      cout << "   Free Space Wavelength: " << lam0 << '\n';
      /*
      if (S < D)
      {
         cout << "   Decaying L mode:       " << lam0 / sqrt(D-S) << '\n';
      }
      else
      {
         cout << "   Oscillating L mode:    " << lam0 / sqrt(S-D) << '\n';
      }
      if (S < - D)
      {
         cout << "   Decaying R mode:       " << lam0 / sqrt(-S-D) << '\n';
      }
      else
      {
         cout << "   Oscillating R mode:    " << lam0 / sqrt(S+D) << '\n';
      }
      if (P < 0)
      {
         cout << "   Decaying O mode:       " << lam0 / sqrt(-P) << '\n';
      }
      else
      {
         cout << "   Oscillating O mode:    " << lam0 / sqrt(P) << '\n';
      }
      if ((S * S - D * D) / S < 0)
      {
         cout << "   Decaying X mode:       " << lam0 * sqrt(-S/(S*S-D*D))
              << '\n';
      }
      else
      {
         cout << "   Oscillating X mode:    " << lam0 * sqrt(S/(S*S-D*D))
              << '\n';
      }
      */
      cout << endl;
   }

   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   if ( Mpi::Root() && logging > 0 ) { cout << "Building Mesh ..." << endl; }

   tic_toc.Clear();
   tic_toc.Start();

   // Mesh * mesh = new Mesh(num_elements, 3, 3, Element::HEXAHEDRON, 1,
   //                      mesh_dim_(0), mesh_dim_(1), mesh_dim_(2));
   Mesh * mesh2d = new Mesh(mesh_file, 1, 1);
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh2d->UniformRefinement();
   }
   Mesh * mesh = Extrude2D(mesh2d, num_elements, hz);
   delete mesh2d;
   if (cyl)
   {
      mesh->SetCurvature(mesh_order);

      MeshTransformCoefficient mtc(hphi);
      mesh->Transform(mtc);
   }
   /*
   {
      Array<int> v2v(mesh->GetNV());
      for (int i=0; i<v2v.Size(); i++) { v2v[i] = i; }
      for (int i=0; i<mesh->GetNV() / 4; i++) { v2v[4 * i + 3] = 4 * i; }

      Mesh * per_mesh = miniapps::MakePeriodicMesh(mesh, v2v);
      delete mesh;
      mesh = per_mesh;
   }
   */
   tic_toc.Stop();

   if (Mpi::Root() && logging > 0 )
   {
      cout << " done in " << tic_toc.RealTime() << " seconds." << endl;
   }
   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }

   mesh->EnsureNCMesh();
   // Ensure that quad and hex meshes are treated as non-conforming.
   /*
   if (maxit > 1 || amr_stix)
   {
      mesh->EnsureNCMesh();
   }
   */

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   if ( Mpi::Root() && logging > 0 )
   { cout << "Building Parallel Mesh ..." << endl; }
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   if (Mpi::Root())
   {
      cout << "Starting initialization." << endl;
   }
   /*
   {
     for (int i=0; i<pmesh.GetNBE(); i++)
       {
    cout << i << '\t' << pmesh.GetBdrElementBaseGeometry(i)
         << '\t' << pmesh.GetBdrAttribute(i) << endl;
       }
   }
   */
   // If values for Voltage BCs were not set issue a warning and exit
   /*
   if ( ( vbcs.Size() > 0 && kbcs.Size() == 0 ) ||
        ( kbcs.Size() > 0 && vbcs.Size() == 0 ) ||
        ( vbcv.Size() < vbcs.Size() ) )
   {
      if ( Mpi::Root() )
      {
         cout << "The surface current (K) boundary condition requires "
              << "surface current boundary condition surfaces (with -kbcs), "
              << "voltage boundary condition surface (with -vbcs), "
              << "and voltage boundary condition values (with -vbcv)."
              << endl;
      }
      return 3;
   }
   */

   // VectorConstantCoefficient BCoef(BVec);

   H1_ParFESpace H1FESpace(&pmesh, order, pmesh.Dimension());
   H1_ParFESpace H1VFESpace(&pmesh, order, pmesh.Dimension(),
                            BasisType::GaussLobatto, 3);
   ND_ParFESpace HCurlFESpace(&pmesh, order, pmesh.Dimension());
   RT_ParFESpace HDivFESpace(&pmesh, order, pmesh.Dimension());
   L2_ParFESpace L2FESpace(&pmesh, order, pmesh.Dimension());

   ParGridFunction BField(&HDivFESpace);
   ParGridFunction temperature_gf;
   ParGridFunction density_gf;
   ParGridFunction nue_gf(&H1FESpace);
   ParGridFunction nui_gf(&H1FESpace);
   ParGridFunction iontemp_gf(&H1FESpace);

   G_EQDSK_Data *eqdsk = NULL;
   {
      named_ifgzstream ieqdsk(eqdsk_file);
      if (ieqdsk)
      {
         eqdsk = new G_EQDSK_Data(ieqdsk);
         if (Mpi::Root())
         {
            eqdsk->PrintInfo();
            if (logging > 0)
            {
               eqdsk->DumpGnuPlotData("stix2d_dh_eqdsk");
            }
         }
      }
   }

   BFieldProfile::CoordSystem b_coord_sys =
      cyl ? BFieldProfile::POLOIDAL : BFieldProfile::CARTESIAN_3D;
   BFieldProfile BCoef(bpt, bpp, false, b_coord_sys, eqdsk);
   BFieldProfile BUnitCoef(bpt, bpp, true, b_coord_sys, eqdsk);

   BField.ProjectCoefficient(BCoef);

   PlasmaProfile::CoordSystem coord_sys =
      cyl ? PlasmaProfile::POLOIDAL : PlasmaProfile::CARTESIAN_3D;
   PlasmaProfile nueCoef(nept, nepp, coord_sys, eqdsk);
   nue_gf.ProjectCoefficient(nueCoef);
   PlasmaProfile nuiCoef(nipt, nipp, coord_sys, eqdsk);
   nui_gf.ProjectCoefficient(nuiCoef);
   PlasmaProfile TiCoef(tipt, tipp, coord_sys, eqdsk);
   iontemp_gf.ProjectCoefficient(TiCoef);

   int size_h1 = H1FESpace.GetVSize();
   int size_l2 = L2FESpace.GetVSize();

   Array<int> density_offsets(numbers.Size() + 1);
   Array<int> temperature_offsets(numbers.Size() + 2);

   density_offsets[0] = 0;
   temperature_offsets[0] = 0;
   temperature_offsets[1] = size_h1;

   for (int i=1; i<=numbers.Size(); i++)
   {
      density_offsets[i]     = density_offsets[i - 1] + size_l2;
      temperature_offsets[i + 1] = temperature_offsets[i] + size_h1;
   }

   BlockVector density(density_offsets);
   BlockVector temperature(temperature_offsets);

   if (Mpi::Root())
   {
      cout << "Creating plasma profile." << endl;
   }

   /*
   if (Mpi::Root())
   {
      cout << "   Setting default temperature profile type " << tpt_def
           << " with parameters \"";
      tpp_def.Print(cout);
   }
   */
   PlasmaProfile TeCoef(tpt_def, tpp_def, coord_sys, eqdsk);
   if (tpa_vac.Size() > 0)
   {
      /*
      if (Mpi::Root())
      {

         cout << "   Setting vacuum layer temperature profile type " << tpt_sol
              << " with parameters \"";
         tpp_vac.Print(cout);
         cout << "\" on attributes \"" << tpa_vac << "\".";
      }
      */
      TeCoef.SetParams(tpa_vac, tpt_vac, tpp_vac);
   }
   if (tpa_sol.Size() > 0)
   {
      /*
      if (Mpi::Root())
      {

         cout << "   Setting scrape-off layer temperature profile type " << tpt_sol
              << " with parameters \"";
         tpp_sol.Print(cout);
         cout << "\" on attributes \"" << tpa_sol << "\".";
      }
      */
      TeCoef.SetParams(tpa_sol, tpt_sol, tpp_sol);
   }
   if (tpa_cor.Size() > 0)
   {
      /*
      if (Mpi::Root())
      {
         cout << "   Setting core temperature profile type " << tpt_cor
              << " with parameters \"";
         tpp_cor.Print(cout);
         cout << "\" on attributes \"" << tpa_cor << "\".";
      }
      */
      TeCoef.SetParams(tpa_cor, tpt_cor, tpp_cor);
   }
   /*
   if (Mpi::Root())
   {
      cout << "   Setting default density profile type " << dpt_def
           << " with parameters \"";
      dpp_def.Print(cout);
   }
   */
   PlasmaProfile rhoCoef(dpt_def, dpp_def, coord_sys, eqdsk);
   if (dpa_vac.Size() > 0)
   {
      /*
      if (Mpi::Root())
      {
         cout << "   Setting vacuum density profile type " << dpt_vac
              << " with parameters \"";
         dpp_vac.Print(cout);
         cout << "\" on attributes \"" << dpa_vac << "\".";
      }
      */
      rhoCoef.SetParams(dpa_vac, dpt_vac, dpp_vac);
   }
   if (dpa_sol.Size() > 0)
   {
      /*
      if (Mpi::Root())
      {
         cout << "   Setting scrape-off layer density profile type " << dpt_sol
              << " with parameters \"";
         dpp_sol.Print(cout);
         cout << "\" on attributes \"" << dpa_sol << "\".";
      }
      */
      rhoCoef.SetParams(dpa_sol, dpt_sol, dpp_sol);
   }
   if (dpa_cor.Size() > 0)
   {
      /*
      if (Mpi::Root())
      {
         cout << "   Setting core density profile type " << dpt_cor
              << " with parameters \"";
         dpp_cor.Print(cout);
         cout << "\" on attributes \"" << dpa_cor << "\".";
      }
      */
      rhoCoef.SetParams(dpa_cor, dpt_cor, dpp_cor);
   }

   for (int i=0; i<=numbers.Size(); i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i).GetMemory());
      temperature_gf.ProjectCoefficient(TeCoef);
   }
   for (int i=0; i<charges.Size(); i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i).GetMemory());
      density_gf.ProjectCoefficient(rhoCoef);
      density_gf *= numbers[i]/numbers[0];
   }

   ColdPlasmaPlaneWaveH HReCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps,
                                nuprof, res_lim, true);
   ColdPlasmaPlaneWaveH HImCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps,
                                nuprof, res_lim, false);

   ColdPlasmaPlaneWaveE EReCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps,
                                nuprof, res_lim, true);
   ColdPlasmaPlaneWaveE EImCoef(wave_type[0], omega, BVec,
                                numbers, charges, masses, temps,
                                nuprof, res_lim, false);

   if (wave_type[0] != ' ')
   {
      Vector kr(3), ki(3);
      HReCoef.GetWaveVector(kr, ki);

      mfem::out << "Plane wave propagation vector: ("
                << complex<double>(kr(0),ki(0)) << ","
                << complex<double>(kr(1),ki(1)) << ","
                << complex<double>(kr(2),ki(2)) << ")" << endl;

      if (!phase_shift)
      {
         kVec.SetSize(6);
         kVec = 0.0;

         if (per_y)
         {
            kVec[1] = kr[1];
            kVec[4] = ki[1];
         }

         kVec[2] = kr[2];
         kVec[5] = ki[2];

         phase_shift = true;
      }
      else if (kVec.Size() < 6)
      {
         kr = kVec;
         ki = 0.0;
         kVec.SetSize(6);
         for (int i=0; i<3; i++) { kVec[i] = kr[i]; }
         for (int i=0; i<3; i++) { kVec[3+i] = ki[i]; }
      }

      kReVec.SetDataAndSize(&kVec[0], 3);
      kImVec.SetDataAndSize(&kVec[3], 3);

      HReCoef.SetPhaseShift(kReVec, kImVec);
      HImCoef.SetPhaseShift(kReVec, kImVec);
      EReCoef.SetPhaseShift(kReVec, kImVec);
      EImCoef.SetPhaseShift(kReVec, kImVec);
   }
   else
   {
      if (phase_shift)
      {
         if (kVec.Size() >= 3)
         {
            kReVec.SetDataAndSize(&kVec[0], 3);
         }
         else
         {
            kReVec.SetSize(3);
            kReVec = 0.0;
         }
         if (kVec.Size() >= 6)
         {
            kImVec.SetDataAndSize(&kVec[3], 3);
         }
         else
         {
            kImVec.SetSize(3);
            kImVec = 0.0;
         }
      }
   }

   if ( Mpi::Root() && logging > 0 )
   {
      mfem::out << "Setting phase shift of ("
         << complex<double>(kReVec[0],kImVec[0]) << ","
         << complex<double>(kReVec[1],kImVec[1]) << ","
         << complex<double>(kReVec[2],kImVec[2]) << ")" << endl;
   }

   VectorConstantCylCoefficient kReCoef(cyl, kReVec);
   VectorConstantCylCoefficient kImCoef(cyl, kImVec);

   ParComplexGridFunction kcomplex_gf(&H1VFESpace);
   kcomplex_gf.ProjectCoefficient(kReCoef, kImCoef);

   ParGridFunction k_gf(&H1VFESpace);
   k_gf.ProjectCoefficient(kReCoef);

   if (amr_stix)
   {
      if (amr_coef == 1)
      {
         if (Mpi::Root())
         {
            cout << "Adapting mesh to Stix 'P' coefficient." << endl;
         }

         StixPCoef RePCoef(BField, k_gf, nue_gf, nui_gf, density, temperature,
                           iontemp_gf, L2FESpace, H1FESpace,
                           omega, charges, masses, nuprof, res_lim,
                           true);
         StixPCoef ImPCoef(BField, k_gf, nue_gf, nui_gf, density, temperature,
                           iontemp_gf, L2FESpace, H1FESpace,
                           omega, charges, masses, nuprof, res_lim,
                           false);

         L2_ParFESpace err_fes(&pmesh, 0, pmesh.Dimension());

         AdaptInitialMesh(pmesh, err_fes,
                          H1FESpace, H1VFESpace, HCurlFESpace, HDivFESpace, L2FESpace,
                          BCoef, kReCoef, rhoCoef, TeCoef, TiCoef,
                          nueCoef, nuiCoef,
                          size_h1, size_l2,
                          density_offsets, temperature_offsets,
                          density, temperature,
                          BField, k_gf, density_gf, temperature_gf, iontemp_gf,
                          nue_gf, nui_gf,
                          RePCoef, ImPCoef,
                          order,
                          init_amr_tol, init_amr_max_its, init_amr_max_dofs,
                          visualization);
      }
      else
      {
         if (Mpi::Root())
         {
            cout << "Adapting mesh to Stix 'S' coefficient." << endl;
         }

         StixSCoef ReSCoef(BField, k_gf, nue_gf, nui_gf, density, temperature,
                           iontemp_gf, L2FESpace, H1FESpace,
                           omega, charges, masses, nuprof, res_lim,
                           true);
         StixSCoef ImSCoef(BField, k_gf, nue_gf, nui_gf, density, temperature,
                           iontemp_gf, L2FESpace, H1FESpace,
                           omega, charges, masses, nuprof, res_lim,
                           false);

         L2_ParFESpace err_fes(&pmesh, 0, pmesh.Dimension());

         AdaptInitialMesh(pmesh, err_fes,
                          H1FESpace, H1VFESpace, HCurlFESpace, HDivFESpace, L2FESpace,
                          BCoef, kReCoef, rhoCoef, TeCoef, TiCoef,
                          nueCoef, nuiCoef,
                          size_h1, size_l2,
                          density_offsets, temperature_offsets,
                          density, temperature,
                          BField, k_gf, density_gf, temperature_gf, iontemp_gf,
                          nue_gf, nui_gf,
                          ReSCoef, ImSCoef,
                          order,
                          init_amr_tol, init_amr_max_its, init_amr_max_dofs,
                          visualization);
      }
   }


   if (Mpi::Root())
   {
      cout << "Creating coefficients for Maxwell equations." << endl;
   }

   // Create a coefficient describing the magnetic permeability
   ConstantCoefficient muCoef(mu0_);

   // Create a coefficient describing the surface admittance
   Coefficient * etaCoef = SetupImpedanceCoefficient(pmesh, abcs);

   // Create tensor coefficients describing the dielectric permittivity
   InverseDielectricTensor epsilonInv_real(BField, k_gf, nue_gf, nui_gf, density,
                                           temperature, iontemp_gf,
                                           L2FESpace, H1FESpace,
                                           omega, charges, masses, nuprof,
                                           res_lim, true);
   InverseDielectricTensor epsilonInv_imag(BField, k_gf, nue_gf, nui_gf, density,
                                           temperature, iontemp_gf,
                                           L2FESpace, H1FESpace,
                                           omega, charges, masses, nuprof,
                                           res_lim, false);
   SusceptibilityTensor suscept_real(BField, k_gf, nue_gf, nui_gf, density,
                                           temperature, iontemp_gf,
                                           L2FESpace, H1FESpace,
                                           omega, charges, masses, nuprof,
                                           res_lim, true);
   SusceptibilityTensor suscept_imag(BField, k_gf, nue_gf, nui_gf, density,
                                           temperature, iontemp_gf,
                                           L2FESpace, H1FESpace,
                                           omega, charges, masses, nuprof,
                                           res_lim, false);
   TransposeMatrixCoefficient suscept_realT(suscept_real);
   TransposeMatrixCoefficient suscept_imagT(suscept_imag);
   MatrixSumCoefficient suscept_real_diss(suscept_real, suscept_realT,1.0,-1.0);
   MatrixSumCoefficient suscept_imag_diss(suscept_imag, suscept_imagT);
   SPDDielectricTensor epsilon_abs(BField, k_gf, nue_gf, nui_gf, density,
                                   temperature,
                                   iontemp_gf, L2FESpace, H1FESpace,
                                   omega, charges, masses, nuprof, res_lim);
   SheathImpedance z_r(BField, density, temperature,
                       L2FESpace, H1FESpace,
                       omega, charges, masses, true);
   SheathImpedance z_i(BField, density, temperature,
                       L2FESpace, H1FESpace,
                       omega, charges, masses, false);

   if (check_eps_inv)
   {
      DielectricTensor epsilon_real(BField, k_gf, nue_gf, nui_gf,
                                    density, temperature, iontemp_gf,
                                    L2FESpace, H1FESpace,
                                    omega, charges, masses, nuprof, res_lim,
                                    true);
      DielectricTensor epsilon_imag(BField, k_gf, nue_gf, nui_gf,
                                    density, temperature, iontemp_gf,
                                    L2FESpace, H1FESpace,
                                    omega, charges, masses, nuprof, res_lim,
                                    false);
      DenseMatrix epsInvRe(3,3);
      DenseMatrix epsInvIm(3,3);
      DenseMatrix epsRe(3,3);
      DenseMatrix epsIm(3,3);

      DenseMatrix IRe(3,3);
      DenseMatrix IIm(3,3);

      for (int i=0; i<pmesh.GetNE(); i++)
      {
         ElementTransformation *T = pmesh.GetElementTransformation(i);
         Geometry::Type g = pmesh.GetElementBaseGeometry(i);
         const IntegrationPoint &ip = Geometries.GetCenter(g);

         epsilonInv_real.Eval(epsInvRe, *T, ip);
         epsilonInv_imag.Eval(epsInvIm, *T, ip);

         epsilon_real.Eval(epsRe, *T, ip);
         epsilon_imag.Eval(epsIm, *T, ip);

         Mult(epsInvRe, epsRe, IRe);
         AddMult_a(-1.0, epsInvIm, epsIm, IRe);

         Mult(epsInvRe, epsIm, IIm);
         AddMult(epsInvIm, epsRe, IIm);

         IRe(0,0) -= 1.0;
         IRe(1,1) -= 1.0;
         IRe(2,2) -= 1.0;

         double nrmRe = IRe.MaxMaxNorm();
         double nrmIm = IIm.MaxMaxNorm();

         if (nrmRe + nrmIm > 1e-13)
         {
            cout << "element " << i << " on processor "
                 << Mpi::WorldRank() << endl;
            IRe.Print(cout);
            IIm.Print(cout);
            cout << endl;
         }
      }
   }

   if (visualization && wave_type[0] != ' ')
   {
      if (Mpi::Root())
      {
         cout << "Visualize input fields." << endl;
      }
      ParComplexGridFunction HField(&HCurlFESpace);
      HField.ProjectCoefficient(HReCoef, HImCoef);
      ParComplexGridFunction EField(&HCurlFESpace);
      EField.ProjectCoefficient(EReCoef, EImCoef);

      Vector zeroVec(3); zeroVec = 0.0;
      VectorConstantCoefficient zeroCoef(zeroVec);
      double max_Hr = HField.real().ComputeMaxError(zeroCoef);
      double max_Hi = HField.imag().ComputeMaxError(zeroCoef);
      double max_Er = EField.real().ComputeMaxError(zeroCoef);
      double max_Ei = EField.imag().ComputeMaxError(zeroCoef);

      char vishost[] = "localhost";
      int  visport   = 19916;

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      socketstream sock_Hr, sock_Hi, sock_Er, sock_Ei, sock_B;
      sock_Hr.precision(8);
      sock_Hi.precision(8);
      sock_Er.precision(8);
      sock_Ei.precision(8);
      sock_B.precision(8);

      ostringstream hr_keys, hi_keys;
      hr_keys << "aaAcPPPPvvv valuerange 0.0 " << max_Hr;
      hi_keys << "aaAcPPPPvvv valuerange 0.0 " << max_Hi;

      ostringstream er_keys, ei_keys;
      er_keys << "aaAcpppppvvv valuerange 0.0 " << max_Er;
      ei_keys << "aaAcpppppvvv valuerange 0.0 " << max_Ei;

      Wy += offy;
      VisualizeField(sock_Hr, vishost, visport,
                     HField.real(), "Exact Magnetic Field, Re(H)",
                     Wx, Wy, Ww, Wh, hr_keys.str().c_str());

      Wx += offx;
      VisualizeField(sock_Hi, vishost, visport,
                     HField.imag(), "Exact Magnetic Field, Im(H)",
                     Wx, Wy, Ww, Wh, hi_keys.str().c_str());

      Wx += offx;
      VisualizeField(sock_Er, vishost, visport,
                     EField.real(), "Exact Electric Field, Re(E)",
                     Wx, Wy, Ww, Wh, er_keys.str().c_str());
      Wx += offx;
      VisualizeField(sock_Ei, vishost, visport,
                     EField.imag(), "Exact Electric Field, Im(E)",
                     Wx, Wy, Ww, Wh, ei_keys.str().c_str());

      Wx -= offx;
      Wy += offy;

      VisualizeField(sock_B, vishost, visport,
                     BField, "Background Magnetic Field",
                     Wx, Wy, Ww, Wh);

      for (int i=0; i<charges.Size(); i++)
      {
         Wx += offx;

         socketstream sock;
         sock.precision(8);

         stringstream oss;
         oss << "Density Species " << i;
         density_gf.MakeRef(&L2FESpace, density.GetBlock(i).GetMemory());
         VisualizeField(sock, vishost, visport,
                        density_gf, oss.str().c_str(),
                        Wx, Wy, Ww, Wh);
      }
   }

   if (Mpi::Root())
   {
      cout << "Setup boundary conditions." << endl;
   }

   // Setup coefficients for Dirichlet BC
   int dbcsSize = (peca.Size() > 0) + (dbca1.Size() > 0) + (dbca2.Size() > 0) +
                  (dbcas.Size() > 0) + (dbcaw.Size() > 0);

   Array<ComplexVectorCoefficientByAttr*> dbcs(dbcsSize);

   Vector zeroVec(3); zeroVec = 0.0;
   Vector dbc1ReVec;
   Vector dbc1ImVec;
   Vector dbc2ReVec;
   Vector dbc2ImVec;

   if (dbcv1.Size() >= 3)
   {
      dbc1ReVec.SetDataAndSize(&dbcv1[0], 3);
   }
   else
   {
      dbc1ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv1.Size() >= 6)
   {
      dbc1ImVec.SetDataAndSize(&dbcv1[3], 3);
   }
   else
   {
      dbc1ImVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv2.Size() >= 3)
   {
      dbc2ReVec.SetDataAndSize(&dbcv2[0], 3);
   }
   else
   {
      dbc2ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (dbcv2.Size() >= 6)
   {
      dbc2ImVec.SetDataAndSize(&dbcv2[3], 3);
   }
   else
   {
      dbc2ImVec.SetDataAndSize(&zeroVec[0], 3);
   }

   VectorConstantCoefficient zeroCoef(zeroVec);
   VectorConstantCoefficient dbc1ReCoef(dbc1ReVec);
   VectorConstantCoefficient dbc1ImCoef(dbc1ImVec);
   VectorConstantCoefficient dbc2ReCoef(dbc2ReVec);
   VectorConstantCoefficient dbc2ImCoef(dbc2ImVec);

   if (dbcsSize > 0)
   {
      int c = 0;
      if (peca.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = peca;
         dbcs[c]->real = &zeroCoef;
         dbcs[c]->imag = &zeroCoef;
         c++;
      }
      if (dbca1.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbca1;
         dbcs[c]->real = &dbc1ReCoef;
         dbcs[c]->imag = &dbc1ImCoef;
         c++;
      }
      if (dbca2.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbca2;
         dbcs[c]->real = &dbc2ReCoef;
         dbcs[c]->imag = &dbc2ImCoef;
         c++;
      }
      if (dbcas.Size() > 0)
      {

         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbcas;
         dbcs[c]->real = &zeroCoef; //&HReStrapCoef;
         dbcs[c]->imag = &zeroCoef; //&HImStrapCoef;
         c++;
      }
      if (dbcaw.Size() > 0)
      {
         dbcs[c] = new ComplexVectorCoefficientByAttr;
         dbcs[c]->attr = dbcaw;
         dbcs[c]->real = &zeroCoef; //&HReCoef;
         dbcs[c]->imag = &zeroCoef; //&HImCoef;
         c++;
      }
   }

   int nbcsSize = (nbca1.Size() > 0) + (nbca2.Size() > 0) + (nbcaw.Size() > 0);

   Array<ComplexVectorCoefficientByAttr*> nbcs(nbcsSize);

   Vector nbc1ReVec;
   Vector nbc1ImVec;
   Vector nbc2ReVec;
   Vector nbc2ImVec;

   if (nbcv1.Size() >= 3)
   {
      nbc1ReVec.SetDataAndSize(&nbcv1[0], 3);
   }
   else
   {
      nbc1ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv1.Size() >= 6)
   {
      nbc1ImVec.SetDataAndSize(&nbcv1[3], 3);
   }
   else
   {
      nbc1ImVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv2.Size() >= 3)
   {
      nbc2ReVec.SetDataAndSize(&nbcv2[0], 3);
   }
   else
   {
      nbc2ReVec.SetDataAndSize(&zeroVec[0], 3);
   }
   if (nbcv2.Size() >= 6)
   {
      nbc2ImVec.SetDataAndSize(&nbcv2[3], 3);
   }
   else
   {
      nbc2ImVec.SetDataAndSize(&zeroVec[0], 3);
   }

   VectorConstantCoefficient nbc1ReCoef(nbc1ReVec);
   VectorConstantCoefficient nbc1ImCoef(nbc1ImVec);
   VectorConstantCoefficient nbc2ReCoef(nbc2ReVec);
   VectorConstantCoefficient nbc2ImCoef(nbc2ImVec);

   if (nbcsSize > 0)
   {
      int c = 0;
      if (nbca1.Size() > 0)
      {
         nbcs[c] = new ComplexVectorCoefficientByAttr;
         nbcs[c]->attr = nbca1;
         nbcs[c]->real = &nbc1ReCoef;
         nbcs[c]->imag = &nbc1ImCoef;
         c++;
      }
      if (nbca2.Size() > 0)
      {
         nbcs[c] = new ComplexVectorCoefficientByAttr;
         nbcs[c]->attr = nbca2;
         nbcs[c]->real = &nbc2ReCoef;
         nbcs[c]->imag = &nbc2ImCoef;
         c++;
      }
      if (nbcaw.Size() > 0)
      {
         nbcs[c] = new ComplexVectorCoefficientByAttr;
         nbcs[c]->attr = nbcaw;
         nbcs[c]->real = &EReCoef;
         nbcs[c]->imag = &EImCoef;
         c++;
      }
   }

   Array<ComplexCoefficientByAttr*> sbcs((sbca.Size() > 0)? 1 : 0);
   if (sbca.Size() > 0)
   {
      sbcs[0] = new ComplexCoefficientByAttr;
      sbcs[0]->real = &z_r;
      sbcs[0]->imag = &z_i;
      sbcs[0]->attr = sbca;
      AttrToMarker(pmesh.bdr_attributes.Max(), sbcs[0]->attr,
                   sbcs[0]->attr_marker);
   }

   if (Mpi::Root())
   {
      cout << "Creating Cold Plasma Dielectric solver." << endl;
   }

   // Create the cold plasma EM solver
   CPDSolverDH CPD(pmesh, order, omega,
                   (CPDSolverDH::SolverType)sol, solOpts,
                   (CPDSolverDH::PrecondType)prec,
                   conv, BUnitCoef,
                   epsilonInv_real, epsilonInv_imag, 
                   suscept_real_diss, suscept_imag_diss,
                   epsilon_abs,
                   muCoef, etaCoef,
                   (phase_shift) ? &kReCoef : NULL,
                   (phase_shift) ? &kImCoef : NULL,
                   abcs,
                   dbcs, nbcs, sbcs,
                   // e_bc_r, e_bc_i,
                   // EReCoef, EImCoef,
                   (rod_params_.Size() > 0 ||slab_params_.Size() > 0 ||curve_params_.Size() > 0) ?
                   j_src_r : NULL,
                   (rod_params_.Size() > 0 ||slab_params_.Size() > 0 ||curve_params_.Size() > 0) ?
                   j_src_i : NULL, vis_u, pa);

   // Initialize GLVis visualization
   if (visualization)
   {
      CPD.InitializeGLVis();
   }

   // Initialize VisIt visualization
   VisItDataCollection visit_dc("STIX3D-AMR-Parallel", &pmesh);

   Array<ParComplexGridFunction*> auxFields;

   if ( visit )
   {
      CPD.RegisterVisItFields(visit_dc);

      auxFields.SetSize(2);
      auxFields[0] = new ParComplexGridFunction(&HCurlFESpace);
      auxFields[1] = new ParComplexGridFunction(&HCurlFESpace);

      auxFields[0]->ProjectCoefficient(HReCoef, HImCoef);
      auxFields[1]->ProjectCoefficient(EReCoef, EImCoef);

      visit_dc.RegisterField("Re_H_Exact", &auxFields[0]->real());
      visit_dc.RegisterField("Im_H_Exact", &auxFields[0]->imag());

      visit_dc.RegisterField("Re_E_Exact", &auxFields[1]->real());
      visit_dc.RegisterField("Im_E_Exact", &auxFields[1]->imag());

      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(0).GetMemory());
      visit_dc.RegisterField("Electron_Temp", &temperature_gf);

      density_gf.MakeRef(&L2FESpace, density.GetBlock(0).GetMemory());
      visit_dc.RegisterField("Electron_Density", &density_gf);

      //nue_gf *= 1/omega;
      visit_dc.RegisterField("Electron_Collisional_Profile", &nue_gf);
      visit_dc.RegisterField("Ion_Collisional_Profile", &nui_gf);

      visit_dc.RegisterField("Min_Ion_Temp", &iontemp_gf);

      visit_dc.RegisterField("B_background", &BField);

      visit_dc.RegisterField("Re_Phase_Shift", &kcomplex_gf.real());
      visit_dc.RegisterField("Im_Phase_Shift", &kcomplex_gf.imag());

      CPD.WriteVisItFields(0);
   }
   if (Mpi::Root()) { cout << "Initialization done." << endl; }

   // The main AMR loop. In each iteration we solve the problem on the current
   // mesh, visualize the solution, estimate the error on all elements, refine
   // the worst elements and update all objects to work with the new mesh. We
   // refine until the maximum number of dofs in the Nedelec finite element
   // space reaches 10 million.
   const int max_dofs = 10000000;
   for (int it = 1; it <= maxit; it++)
   {
      if (Mpi::Root())
      {
         cout << "\nAMR Iteration " << it << endl;
      }

      // Display the current number of DoFs in each finite element space
      CPD.PrintSizes();

      // Assemble all forms
      CPD.Assemble();

      // Solve the system and compute any auxiliary fields
      CPD.Solve();

      /*
      // Compute error
      double glb_error = CPD.GetError(EReCoef, EImCoef);
      if (Mpi::Root())
      {
         cout << "Global L2 Error " << glb_error << endl;
      }
      */

      // Determine the current size of the linear system
      int prob_size = CPD.GetProblemSize();

      // Write fields to disk for VisIt
      if ( visit )
      {
         CPD.WriteVisItFields(it);
      }

      // Send the solution by socket to a GLVis server.
      if (visualization)
      {
         CPD.DisplayToGLVis();
      }

      if (Mpi::Root())
      {
         cout << "AMR iteration " << it << " complete." << endl;
      }

      // Check stopping criteria
      if (prob_size > max_dofs)
      {
         if (Mpi::Root())
         {
            cout << "Reached maximum number of dofs, exiting..." << endl;
         }
         break;
      }
      if ( it == maxit )
      {
         break;
      }

      // Wait for user input. Ask every 10th iteration.
      char c = 'c';
      if (Mpi::Root() && (it % 10 == 0))
      {
         cout << "press (q)uit or (c)ontinue --> " << flush;
         cin >> c;
      }
      MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

      if (c != 'c')
      {
         break;
      }

      // Estimate element errors using the Zienkiewicz-Zhu error estimator.
      Vector errors(pmesh.GetNE());
      CPD.GetErrorEstimates(errors);

      double local_max_err = errors.Max();
      double global_max_err;
      MPI_Allreduce(&local_max_err, &global_max_err, 1,
                    MPI_DOUBLE, MPI_MAX, pmesh.GetComm());

      // Refine the elements whose error is larger than a fraction of the
      // maximum element error.
      const double frac = 0.5;
      double threshold = frac * global_max_err;
      if (Mpi::Root()) { cout << "Refining ..." << endl; }
      {
         pmesh.RefineByError(errors, threshold);
         /*
              Array<Refinement> refs;
              for (int i=0; i<pmesh.GetNE(); i++)
              {
                 if (errors[i] > threshold)
                 {
                    refs.Append(Refinement(i, 3));
                 }
              }
              if (refs.Size() > 0)
              {
                 pmesh.GeneralRefinement(refs);
              }
         */
      }

      // Update the magnetostatic solver to reflect the new state of the mesh.
      Update(H1FESpace, H1VFESpace, HCurlFESpace, HDivFESpace, L2FESpace, BCoef,
             kReCoef, rhoCoef, TeCoef, TiCoef,
             nueCoef, nuiCoef,
             size_h1, size_l2,
             density_offsets, temperature_offsets,
             density, temperature,
             BField, k_gf, density_gf, temperature_gf, iontemp_gf,
             nue_gf, nui_gf);
      CPD.Update();

      if (pmesh.Nonconforming() && Mpi::WorldSize() > 1 && false)
      {
         if (Mpi::Root()) { cout << "Rebalancing ..." << endl; }
         pmesh.Rebalance();

         // Update again after rebalancing
         Update(H1FESpace, H1VFESpace, HCurlFESpace, HDivFESpace, L2FESpace, BCoef,
                kReCoef, rhoCoef, TeCoef, TiCoef,
                nueCoef, nuiCoef,
                size_h1, size_l2,
                density_offsets, temperature_offsets,
                density, temperature,
                BField, k_gf, density_gf, temperature_gf, iontemp_gf,
                nue_gf, nui_gf);
         CPD.Update();
      }
   }

   // Send the solution by socket to a GLVis server.
   if (visualization)
   {
      CPD.DisplayAnimationToGLVis();
   }
   if (Mpi::Root() && auxFields.Size() > 0)
   {
      cout << "Deleting auxiliary field vectors" << endl;
   }

   for (int i=0; i<auxFields.Size(); i++)
   {
      delete auxFields[i];
   }

   if (Mpi::Root())
   {
      cout << "Deleting remaining objects" << endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   return 0;
}

void AdaptInitialMesh(ParMesh &pmesh, ParFiniteElementSpace &err_fespace,
                      ParFiniteElementSpace & H1FESpace,
                      ParFiniteElementSpace & H1VFESpace,
                      ParFiniteElementSpace & HCurlFESpace,
                      ParFiniteElementSpace & HDivFESpace,
                      ParFiniteElementSpace & L2FESpace,
                      VectorCoefficient & BCoef,
                      VectorCoefficient & kReCoef,
                      Coefficient & rhoCoef,
                      Coefficient & TeCoef,
                      Coefficient & TiCoef,
                      Coefficient & nueCoef,
                      Coefficient & nuiCoef,
                      int & size_h1,
                      int & size_l2,
                      Array<int> & density_offsets,
                      Array<int> & temperature_offsets,
                      BlockVector & density,
                      BlockVector & temperature,
                      ParGridFunction & BField,
                      ParGridFunction & k_gf,
                      ParGridFunction & density_gf,
                      ParGridFunction & temperature_gf,
                      ParGridFunction & iontemp_gf,
                      ParGridFunction & nue_gf,
                      ParGridFunction & nui_gf,
                      Coefficient &ReCoef,
                      Coefficient &ImCoef,
                      int p, double tol, int max_its, int max_dofs,
                      bool visualization)
{
   ConstantCoefficient zeroCoef(0.0);

   ParComplexGridFunction gf(&L2FESpace);

   ComplexLpErrorEstimator estimator(p, ReCoef, ImCoef, gf);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0);
   refiner.SetTotalErrorNormP(p);
   refiner.SetLocalErrorGoal(tol);

   vector<socketstream> sout(2);
   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 275, Wh = 250; // window size
   int offx = Ww + 3;

   for (int it = 0; it < max_its; it++)
   {
      HYPRE_Int global_dofs = L2FESpace.GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of L2 unknowns: " << global_dofs << endl;
      }

      gf.ProjectCoefficient(ReCoef, ImCoef);

      double l2_nrm = gf.ComputeL2Error(zeroCoef, zeroCoef);
      double l2_err = gf.ComputeL2Error(ReCoef, ImCoef);
      if (Mpi::Root())
      {
         if (l2_nrm > 0.0)
         {
            cout << "Relative L2 Error: " << l2_err << " / " << l2_nrm
                 << " = " << l2_err / l2_nrm
                 << endl;
         }
         else
         {
            cout << "L2 Error: " << l2_err << endl;
         }
      }

      // 19. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         VisualizeField(sout[0], vishost, visport, gf.real(),
                        "Stix S Real",
                        Wx, Wy, Ww, Wh);
         Wx += offx;

         VisualizeField(sout[1], vishost, visport, gf.imag(),
                        "Stix S Imaginary",
                        Wx, Wy, Ww, Wh);
      }

      if (global_dofs > max_dofs)
      {
         if (Mpi::Root())
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 20. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(pmesh);
      if (refiner.Stop())
      {
         if (Mpi::Root())
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 21. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      Update(H1FESpace, H1VFESpace, HCurlFESpace, HDivFESpace, L2FESpace,
             BCoef, kReCoef, rhoCoef, TeCoef, TiCoef,
             nueCoef, nuiCoef,
             size_h1, size_l2,
             density_offsets, temperature_offsets,
             density, temperature,
             BField, k_gf, density_gf, temperature_gf, iontemp_gf,
             nue_gf, nui_gf);

      err_fespace.Update();
      gf.Update();

      // 22. Load balance the mesh, and update the space and solution. Currently
      //     available only for nonconforming meshes.
      if (pmesh.Nonconforming())
      {
         pmesh.Rebalance();

         // Update the space and the GridFunction. This time the update matrix
         // redistributes the GridFunction among the processors.
         Update(H1FESpace, H1VFESpace, HCurlFESpace, HDivFESpace, L2FESpace,
                BCoef, kReCoef, rhoCoef, TeCoef, TiCoef,
                nueCoef, nuiCoef,
                size_h1, size_l2,
                density_offsets, temperature_offsets,
                density, temperature,
                BField, k_gf, density_gf, temperature_gf, iontemp_gf,
                nue_gf, nui_gf);

         err_fespace.Update();
         gf.Update();
      }

   }
   if (Mpi::Root())
   {
      cout << endl;
   }
}

void Update(ParFiniteElementSpace & H1FESpace,
            ParFiniteElementSpace & H1VFESpace,
            ParFiniteElementSpace & HCurlFESpace,
            ParFiniteElementSpace & HDivFESpace,
            ParFiniteElementSpace & L2FESpace,
            VectorCoefficient & BCoef,
            VectorCoefficient & kReCoef,
            Coefficient & rhoCoef,
            Coefficient & TeCoef,
            Coefficient & TiCoef,
            Coefficient & nueCoef,
            Coefficient & nuiCoef,
            int & size_h1,
            int & size_l2,
            Array<int> & density_offsets,
            Array<int> & temperature_offsets,
            BlockVector & density,
            BlockVector & temperature,
            ParGridFunction & BField,
            ParGridFunction & k_gf,
            ParGridFunction & density_gf,
            ParGridFunction & temperature_gf,
            ParGridFunction & iontemp_gf,
            ParGridFunction & nue_gf,
            ParGridFunction & nui_gf)
{
   H1FESpace.Update();
   H1VFESpace.Update();
   HCurlFESpace.Update();
   HDivFESpace.Update();
   L2FESpace.Update();

   BField.Update();
   BField.ProjectCoefficient(BCoef);

   k_gf.Update();
   k_gf.ProjectCoefficient(kReCoef);

   nue_gf.Update();
   nue_gf.ProjectCoefficient(nueCoef);
   nui_gf.Update();
   nui_gf.ProjectCoefficient(nuiCoef);

   iontemp_gf.Update();
   iontemp_gf.ProjectCoefficient(TiCoef);

   size_l2 = L2FESpace.GetVSize();
   for (int i=1; i<density_offsets.Size(); i++)
   {
      density_offsets[i] = density_offsets[i - 1] + size_l2;
   }
   density.Update(density_offsets);
   for (int i=0; i<density_offsets.Size()-1; i++)
   {
      density_gf.MakeRef(&L2FESpace, density.GetBlock(i).GetMemory());
      density_gf.ProjectCoefficient(rhoCoef);
   }

   size_h1 = H1FESpace.GetVSize();
   for (int i=1; i<temperature_offsets.Size(); i++)
   {
      temperature_offsets[i] = temperature_offsets[i - 1] + size_h1;
   }
   temperature.Update(temperature_offsets);
   for (int i=0; i<temperature_offsets.Size()-1; i++)
   {
      temperature_gf.MakeRef(&H1FESpace, temperature.GetBlock(i).GetMemory());
      temperature_gf.ProjectCoefficient(TeCoef);
   }
}

const char * banner[6] =
{
   R"(  _________ __   __        ________       ___________    ___ ___  )",
   R"( /   _____//  |_|__|__  ___\_____  \   __| _/\______ \  /   |   \ )",
   R"( \_____  \\   __\  \  \/  /  _(__  <  / __ |  |    |  \/    ~    \)",
   R"( /        \|  | |  |>    <  /       \/ /_/ |  |    `   \    Y    /)",
   R"(/_______  /|__| |__/__/\_ \/______  /\____ | /_______  /\___|_  / )",
   R"(        \/               \/       \/      \/         \/       \/  )"
};

// Print the stix3d ascii logo to the given ostream
void display_banner(ostream & os)
{
   for (int i=0; i<6; i++)
   {
      os << banner[i] << endl;
   }
   os << endl
      << "* Thomas H. Stix was a pioneer in the use of radio frequency"
      << " waves to heat" << endl
      << "  terrestrial plasmas to solar temperatures. He made important"
      << " contributions" << endl
      << "  to experimental and theoretic plasma physics. In the Stix"
      << " application, the" << endl
      << "  plasma dielectric for the wave equation is formulated using"
      << " the \"Stix\"" << endl
      << "  notation, \"S, D, P\"." << endl<< endl << flush;
}

void record_cmd_line(int argc, char *argv[])
{
   ofstream ofs("stix3d_dh_cmd.txt");

   for (int i=0; i<argc; i++)
   {
      ofs << argv[i] << " ";
      if (strcmp(argv[i], "-bm"     ) == 0 ||
          strcmp(argv[i], "-cm"     ) == 0 ||
          strcmp(argv[i], "-sm"     ) == 0 ||
          strcmp(argv[i], "-bpp"    ) == 0 ||
          strcmp(argv[i], "-dpp"    ) == 0 ||
          strcmp(argv[i], "-dpp-vac") == 0 ||
          strcmp(argv[i], "-dpp-sol") == 0 ||
          strcmp(argv[i], "-dpp-cor") == 0 ||
          strcmp(argv[i], "-dpa-vac") == 0 ||
          strcmp(argv[i], "-dpa-sol") == 0 ||
          strcmp(argv[i], "-dpa-cor") == 0 ||
          strcmp(argv[i], "-tpp"    ) == 0 ||
          strcmp(argv[i], "-tpp-vac") == 0 ||
          strcmp(argv[i], "-tpp-sol") == 0 ||
          strcmp(argv[i], "-tpp-cor") == 0 ||
          strcmp(argv[i], "-tpa-sol") == 0 ||
          strcmp(argv[i], "-tpa-cor") == 0 ||
          strcmp(argv[i], "-nepp"   ) == 0 ||
          strcmp(argv[i], "-nipp"   ) == 0 ||
          strcmp(argv[i], "-B"      ) == 0 ||
          strcmp(argv[i], "-k-vec"  ) == 0 ||
          strcmp(argv[i], "-q"      ) == 0 ||
          strcmp(argv[i], "-min"    ) == 0 ||
          strcmp(argv[i], "-jsrc"   ) == 0 ||
          strcmp(argv[i], "-rod"    ) == 0 ||
          strcmp(argv[i], "-slab"   ) == 0 ||
          strcmp(argv[i], "-curve"  ) == 0 ||
          strcmp(argv[i], "-sp"     ) == 0 ||
          strcmp(argv[i], "-abcs"   ) == 0 ||
          strcmp(argv[i], "-sbcs"   ) == 0 ||
          strcmp(argv[i], "-pecs"   ) == 0 ||
          strcmp(argv[i], "-dbcs-msa") == 0 ||
          strcmp(argv[i], "-dbcs-pw") == 0 ||
          strcmp(argv[i], "-dbcs1"  ) == 0 ||
          strcmp(argv[i], "-dbcs2"  ) == 0 ||
          strcmp(argv[i], "-dbcv1"  ) == 0 ||
          strcmp(argv[i], "-dbcv2"  ) == 0 ||
          strcmp(argv[i], "-nbcs"   ) == 0 ||
          strcmp(argv[i], "-nbcs1"  ) == 0 ||
          strcmp(argv[i], "-nbcs2"  ) == 0 ||
          strcmp(argv[i], "-nbcv"   ) == 0 ||
          strcmp(argv[i], "-nbcv1"  ) == 0 ||
          strcmp(argv[i], "-nbcv2"  ) == 0)
      {
         ofs << "'" << argv[++i] << "' ";
      }
   }
   ofs << endl << flush;
   ofs.close();
}

// The Impedance is an optional coefficient defined on boundary surfaces which
// can be used in conjunction with absorbing boundary conditions.
Coefficient *
SetupImpedanceCoefficient(const Mesh & mesh, const Array<int> & abcs)
{
   Coefficient * coef = NULL;

   if ( pw_eta_.Size() > 0 )
   {
      MFEM_VERIFY(pw_eta_.Size() == abcs.Size(),
                  "Each impedance value must be associated with exactly one "
                  "absorbing boundary surface.");

      pw_bdr_eta_.SetSize(mesh.bdr_attributes.Size());

      if ( abcs[0] == -1 )
      {
         pw_bdr_eta_ = pw_eta_[0];
      }
      else
      {
         pw_bdr_eta_ = 0.0;

         for (int i=0; i<pw_eta_.Size(); i++)
         {
            pw_bdr_eta_[abcs[i]-1] = pw_eta_[i];
         }
      }
      coef = new PWConstCoefficient(pw_bdr_eta_);
   }

   return coef;
}

void rod_current_source_r(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = rod_params_.Size() == 9;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = rod_params_(o+0);
   double y0 = rod_params_(o+1);
   double radius = rod_params_(o+2);

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double r2 = (r - x0) * (r - x0) + (z - y0) * (z - y0);

   if (r2 <= radius * radius)
   {
      if (!j_cyl_)
      {
         j(0) = rod_params_(0);
         j(1) = rod_params_(1);
         j(2) = rod_params_(2);
      }
      else
      {
         double cosphi = x[0] / r;
         double sinphi = x[1] / r;

         double j_r   = rod_params_(0);
         double j_phi = rod_params_(1);
         double j_z   = rod_params_(2);

         j(0) = j_r * cosphi - j_phi * sinphi;
         j(1) = j_r * sinphi + j_phi * cosphi;
         j(2) = j_z;
      }
   }
   // j *= height;
}

void rod_current_source_i(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = rod_params_.Size() == 9;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = rod_params_(o+0);
   double y0 = rod_params_(o+1);
   double radius = rod_params_(o+2);

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double r2 = (r - x0) * (r - x0) + (z - y0) * (z - y0);

   if (r2 <= radius * radius)
   {
      if (cmplx)
      {
         if (!j_cyl_)
         {
            j(0) = rod_params_(3);
            j(1) = rod_params_(4);
            j(2) = rod_params_(5);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = rod_params_(3);
            double j_phi = rod_params_(4);
            double j_z   = rod_params_(5);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
      }
   }
   // j *= height;
}

void slab_current_source_r(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = slab_params_.Size() == 12;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = slab_params_(o+0);
   double y0 = slab_params_(o+1);
   double z0 = slab_params_(o+2);
   double dx = slab_params_(o+3);
   double dy = slab_params_(o+4);
   double dz = slab_params_(o+5);

   double xr = x[0];
   double y = x[1];
   double z = x[2];

   if (xr >= x0-0.5*dx && xr <= x0+0.5*dx &&
       y >= y0-0.5*dy && y <= y0+0.5*dy &&
       z >= z0-0.5*dz && z <= z0+0.5*dz )
   {

      j(0) = slab_params_(0);
      j(1) = slab_params_(1);
      j(2) = slab_params_(2);

      if (slab_profile_ == 1)
      { j *= 0.5 * (1.0 + sin(M_PI*((2.0 * (y - y0) + dy)/dy - 0.5))); }
   }
}

void slab_current_source_i(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   bool cmplx = slab_params_.Size() == 12;

   int o = 3 + (cmplx ? 3 : 0);

   double x0 = slab_params_(o+0);
   double y0 = slab_params_(o+1);
   double z0 = slab_params_(o+2);
   double dx = slab_params_(o+3);
   double dy = slab_params_(o+4);
   double dz = slab_params_(o+5);

   double xr = x[0];
   double y = x[1];
   double z = x[2];

   if (xr >= x0-0.5*dx && xr <= x0+0.5*dx &&
       y >= y0-0.5*dy && y <= y0+0.5*dy &&
       z >= z0-0.5*dz && z <= z0+0.5*dz )
   {
      if (cmplx)
      {

         j(0) = slab_params_(3);
         j(1) = slab_params_(4);
         j(2) = slab_params_(5);

         if (slab_profile_ == 1)
         { j *= 0.5 * (1.0 + sin(M_PI*((2.0 * (y - y0) + dy)/dy - 0.5))); }
      }
   }
}

void curve_current_source_v0_r(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   // 1 antenna with real amplitude
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");
   MFEM_VERIFY(curve_params_.Size() > 2, "data missing from curve_params_");

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double theta = atan2(z, r);

   double rmin = 2.415 + 0.035;
   double rmax = rmin + 0.02;
   double length = 0.325;

   double rthetamax = (3.8*M_PI)/180.0;
   double rthetamin = (-1.0*3.8*M_PI)/180.0;
   double theta_ext = rthetamax - rthetamin;

   /*
   double length = 0.325; 

   double rmin = 0.25;
   double rmax = 0.255;

   double rthetamax = 0.2/rmin;
   double rthetamin = -0.2/rmin;
   double theta_ext = rthetamax - rthetamin;
   */

   if (theta >= rthetamin && theta <= rthetamax &&
       r >= rmin && r <= rmax)
   {
      if (!j_cyl_)
      {
         j(0) = -1.0*curve_params_(1)*sin(theta);
         j(1) = curve_params_(1)*cos(theta);
         j(2) = curve_params_(2);
      }
      else
      {
         double cosphi = x[0] / r;
         double sinphi = x[1] / r;

         double j_r   = -curve_params_(1)*sin(theta);
         double j_phi = curve_params_(2);
         double j_z   = curve_params_(1)*cos(theta);

         j(0) = j_r * cosphi - j_phi * sinphi;
         j(1) = j_r * sinphi + j_phi * cosphi;
         j(2) = j_z;
      }
      if (vol_profile_ == 1)
      {
         double arc_len = rmin*fabs(theta);
         j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + length)/length - 0.5)));
      }
   }
}

void curve_current_source_v0_i(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   // 1 antenna with complex amplitude
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   if (curve_params_.Size() < 7)
   {
      return;
   }

   else
   {
      double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
      double z = (j_cyl_) ? x[2] : x[1];

      double theta = atan2(z, r);

      double rmin = 2.415 + 0.035;
      double rmax = rmin + 0.02;
      double length = 0.325;
      double rthetamax = (3.8*M_PI)/180.0;
      double rthetamin = (-1.0*3.8*M_PI)/180.0;
      double theta_ext = rthetamax - rthetamin;
      /*
      double length = 0.325; 

      double rmin = 0.25;
      double rmax = 0.255;

      double rthetamax = 0.2/rmin;
      double rthetamin = -0.2/rmin;
      double theta_ext = rthetamax - rthetamin;
      */

      if (theta >= rthetamin && theta <= rthetamax &&
          r >= rmin && r <= rmax)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(5)*sin(theta);
            j(1) = curve_params_(5)*cos(theta);
            j(2) = curve_params_(6);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(5)*sin(theta);
            double j_phi = curve_params_(6);
            double j_z   = curve_params_(5)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + length)/length - 0.5)));
         }
      }
   }
}

void curve_current_source_v1_r(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   // 1 or 2 antenna with real amplitude(s)
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");
   MFEM_VERIFY(curve_params_.Size() > 0, "data missing from curve_params_");
   if (curve_params_(0) == 1)
   {
      MFEM_VERIFY(curve_params_.Size() > 2, "data missing from curve_params_");
   }
   else
   {
      MFEM_VERIFY(curve_params_.Size() > 4, "data missing from curve_params_");
   }

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double theta = atan2(z, r);

   double thetamax1 = 8.71;
   double thetamin1 = 1.11;
   double thetamax2 = -1.25;
   double thetamin2 = -8.85;

   double theta_ext = thetamax1 - thetamin1;
   double rmin = (2.415 + 0.035);

   double xmin = rmin*cos(theta);
   double xmax = xmin + 0.04;

   double zmin1 = rmin * sin((M_PI * thetamin1) / 180.);
   double zmax1 = rmin * sin((M_PI * thetamax1) / 180.);

   if (curve_params_(0) == 1)
   {
      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(1)*sin(theta);
            j(1) = curve_params_(1)*cos(theta);
            j(2) = curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(1)*sin(theta);
            double j_phi = curve_params_(2);
            double j_z   = curve_params_(1)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta) - rmin*(theta_ext/2.0 + thetamin1)*
                             (M_PI/180.);
            double dlant = rmin*((theta_ext*M_PI)/180.);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + dlant)/dlant - 0.5)));
         }
      }
   }
   else
   {
      double zmin2 = rmin * sin((M_PI * thetamin2) / 180.);
      double zmax2 = rmin * sin((M_PI * thetamax2) / 180.);

      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(1)*sin(theta);
            j(1) = curve_params_(1)*cos(theta);
            j(2) = curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(1)*sin(theta);
            double j_phi = curve_params_(2);
            double j_z   = curve_params_(1)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta) - rmin*(theta_ext/2.0 + thetamin1)*
                             (M_PI/180.);
            double dlant = rmin*((theta_ext*M_PI)/180.);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + dlant)/dlant - 0.5)));
         }
      }
      else if (r >= xmin && r <= xmax &&
               z >= zmin2 && z <= zmax2)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(3)*sin(theta);
            j(1) = curve_params_(3)*cos(theta);
            j(2) = curve_params_(4);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(3)*sin(theta);
            double j_phi = curve_params_(4);
            double j_z   = curve_params_(3)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta) - rmin*(theta_ext/2.0 + fabs(thetamax2))*
                             (M_PI/180.);
            double dlant = rmin*((theta_ext*M_PI)/180.);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + dlant)/dlant - 0.5)));
         }
      }
   }
}

void curve_current_source_v1_i(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   // 1 or 2 antennas with complex amplitude(s)
   MFEM_ASSERT(x.Size() == 3,"current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double theta = atan2(z, r);

   double thetamax1 = 8.71;
   double thetamin1 = 1.11;
   double thetamax2 = -1.25;
   double thetamin2 = -8.85;

   double theta_ext = thetamax1 - thetamin1;
   double rmin = (2.415 + 0.035);

   double xmin = rmin*cos(theta);
   double xmax = xmin + 0.04;

   double zmin1 = rmin * sin((M_PI * thetamin1) / 180.);
   double zmax1 = rmin * sin((M_PI * thetamax1) / 180.);

   if (curve_params_.Size() < 7)
   {
      return;
   }

   else if (curve_params_(0) == 1 && curve_params_.Size() < 8)
   {
      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(5)*sin(theta);
            j(1) = curve_params_(5)*cos(theta);
            j(2) = curve_params_(6);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(5)*sin(theta);
            double j_phi = curve_params_(6);
            double j_z   = curve_params_(5)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta) - rmin*(theta_ext/2.0 + thetamin1)*
                             (M_PI/180.);
            double dlant = rmin*((theta_ext*M_PI)/180.);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + dlant)/dlant - 0.5)));
         }
      }
   }
   else
   {
      double zmin2 = rmin * sin((M_PI * thetamin2) / 180.);
      double zmax2 = rmin * sin((M_PI * thetamax2) / 180.);

      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(5)*sin(theta);
            j(1) = curve_params_(5)*cos(theta);
            j(2) = curve_params_(6);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(5)*sin(theta);
            double j_phi = curve_params_(6);
            double j_z   = curve_params_(5)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta) - rmin*(theta_ext/2.0 + thetamin1)*
                             (M_PI/180.);
            double dlant = rmin*((theta_ext*M_PI)/180.);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + dlant)/dlant - 0.5)));
         }
      }
      else if (r >= xmin && r <= xmax &&
               z >= zmin2 && z <= zmax2)
      {
         if (!j_cyl_)
         {
            j(0) = -1.0*curve_params_(7)*sin(theta);
            j(1) = curve_params_(7)*cos(theta);
            j(2) = curve_params_(8);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = -curve_params_(7)*sin(theta);
            double j_phi = curve_params_(8);
            double j_z   = curve_params_(7)*cos(theta);

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double arc_len = rmin*fabs(theta) - rmin*(theta_ext/2.0 + fabs(thetamax2))*
                             (M_PI/180.);
            double dlant = rmin*((theta_ext*M_PI)/180.);
            j *= 0.5 * (1.0 + sin(M_PI*((2.0 * arc_len + dlant)/dlant - 0.5)));
         }
      }
   }
}

void curve_current_source_v2_r(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");
   MFEM_VERIFY(curve_params_.Size() > 0, "data missing from curve_params_");
   if (curve_params_(0) == 1)
   {
      MFEM_VERIFY(curve_params_.Size() > 2, "data missing from curve_params_");
   }
   else
   {
      MFEM_VERIFY(curve_params_.Size() > 4, "data missing from curve_params_");
   }

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double zmin1 = 0.0466;
   double zmax1 = 0.3655;

   double xmin = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195;
   double xmax = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195 + 0.02;

   double b = 0.415;
   double c = 0.15;

   if (curve_params_(0) == 1)
   {
      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,
                                                                                       4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(1)/mag;
            j(2) = curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(2);
            double j_z   = curve_params_(1)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.328835;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0
                             - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - 0.0466232;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }

      }
   }
   else
   {
      double zmin2 = -0.3709;
      double zmax2 = -0.0523;

      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,
                                                                                       4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(1)/mag;
            j(2) = curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(2);
            double j_z   = curve_params_(1)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.328835;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0
                             - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - 0.0466232;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
      else if (r >= xmin && r <= xmax &&
               z >= zmin2 && z <= zmax2)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,
                                                                                       4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(3)/mag;
            j(2) = curve_params_(4);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(4);
            double j_z   = curve_params_(3)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.328835;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0
                                   - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - 0.0523328;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   }
}

void curve_current_source_v2_i(const Vector &x, Vector &j)
{
   // NOT UP TO DATE
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double zmin1 = 0.0466;
   double zmax1 = 0.3655;

   double xmin = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195;
   double xmax = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195 + 0.02;

   double b = 0.415;
   double c = 0.15;

   if (curve_params_.Size() < 7)
   {
      return;
   }

   if (curve_params_(0) != 1)
   {
      MFEM_VERIFY(curve_params_.Size() > 8, "data missing from curve_params_");
   }

   else if (curve_params_(0) == 1 && curve_params_.Size() < 8)
   {
      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,
                                                                                       4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(5)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(6)/mag;
            j(2) = curve_params_(5);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(5)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(5);
            double j_z   = curve_params_(6)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.328835;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0
                             - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - 0.0466232;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }

      }
   }
   else
   {
      double zmin2 = -0.3709;
      double zmax2 = -0.0523;

      if (r >= xmin && r <= xmax &&
          z >= zmin1 && z <= zmax1)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,
                                                                                       4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(5)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(6)/mag;
            j(2) = curve_params_(5);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(5)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(5);
            double j_z   = curve_params_(6)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.328835;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0
                             - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - 0.0466232;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
      else if (r >= xmin && r <= xmax &&
               z >= zmin2 && z <= zmax2)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,
                                                                                       4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(7)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(7)/mag;
            j(2) = curve_params_(8);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(7)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(8);
            double j_z   = curve_params_(7)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.328835;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0
                                   - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - 0.0523328;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   }
}

void curve_current_source_v3_r(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double xmin = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195;
   double xmax = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195 + 0.02;

   double b = 0.415;
   double c = 0.15;

   // Configurations:
   double A = curve_params_(5); // Standard dipole: 0/pi/0/pi
   //double B = curve_params_(6); // Modified dipole: 0/pi/pi/0

   // Top Strap 1:
   double zmin1t = 0.11792;
   double zmax1t = 0.4325;
   double phi1r = 0.0;
   double phi1l = 0.08;
   if (r >= xmin && r <= xmax &&
      z >= zmin1t && z <= zmax1t &&
      x[1] >= phi1r && x[1] <= phi1l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(1)/mag;
            j(2) = curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(2);
            double j_z   = curve_params_(1)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin1t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 1:
   double zmin1b = -0.30925;
   double zmax1b = 0.01211;
   if (r >= xmin && r <= xmax &&
       z >= zmin1b && z <= zmax1b &&
       x[1] >= phi1r && x[1] <= phi1l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(3)/mag;
            j(2) = curve_params_(4);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(4);
            double j_z   = curve_params_(3)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax1b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Top Strap 2:
   double zmin2t = 0.074;
   double zmax2t = 0.39137;
   double phi2r = 0.181;
   double phi2l = 0.08+0.181;
   if (r >= xmin && r <= xmax &&
      z >= zmin2t && z <= zmax2t &&
      x[1] >= phi2r && x[1] <= phi2l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(1)/mag;
            j(2) = A*curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = A*curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = A*curve_params_(2);
            double j_z   = A*curve_params_(1)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin2t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 2:
   double zmin2b = -0.34802;
   double zmax2b = -0.0283;
   if (r >= xmin && r <= xmax &&
       z >= zmin2b && z <= zmax2b &&
       x[1] >= phi2r && x[1] <= phi2l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(3)/mag;
            j(2) = A*curve_params_(4);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = A*curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = A*curve_params_(4);
            double j_z   = A*curve_params_(3)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax2b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Top Strap 3:
   double zmin3t = 0.01818;
   double zmax3t = 0.33835;
   double phi3r = 0.181*2;
   double phi3l = 0.08+0.181*2;
   if (r >= xmin && r <= xmax &&
      z >= zmin3t && z <= zmax3t &&
      x[1] >= phi3r && x[1] <= phi3l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(1)/mag;
            j(2) = curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(2);
            double j_z   = curve_params_(1)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin3t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 3:
   double zmin3b = -0.39616;
   double zmax3b = -0.0791;
   if (r >= xmin && r <= xmax &&
       z >= zmin3b && z <= zmax3b &&
       x[1] >= phi3r && x[1] <= phi3l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(3)/mag;
            j(2) = curve_params_(4);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(4);
            double j_z   = curve_params_(3)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax3b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Top Strap 4:
   double zmin4t = -0.02218;
   double zmax4t = 0.29953;
   double phi4r = 0.181*3;
   double phi4l = 0.08+0.181*3;
   if (r >= xmin && r <= xmax &&
      z >= zmin4t && z <= zmax4t &&
      x[1] >= phi4r && x[1] <= phi4l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(1)/mag;
            j(2) = A*curve_params_(2);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = A*curve_params_(1)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = A*curve_params_(2);
            double j_z   = A*curve_params_(1)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin4t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 4:
   double zmin4b = -0.42963;
   double zmax4b = -0.11485;
   if (r >= xmin && r <= xmax &&
       z >= zmin4b && z <= zmax4b &&
       x[1] >= phi4r && x[1] <= phi4l)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(3)/mag;
            j(2) = curve_params_(4);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(3)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(4);
            double j_z   = curve_params_(3)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax4b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   }

void curve_current_source_v3_i(const Vector &x, Vector &j)
{
   MFEM_ASSERT(x.Size() == 3, "current source requires 3D space.");

   j.SetSize(x.Size());
   j = 0.0;

   double r = (j_cyl_) ? sqrt(x[0] * x[0] + x[1] * x[1]) : x[0];
   double z = (j_cyl_) ? x[2] : x[1];

   double xmin = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195;
   double xmax = 2.44-0.415*pow(z,2.0)-0.150*pow(z,4.0)+0.0195 + 0.02;

   double b = 0.415;
   double c = 0.15;

   // Configurations:
   double A = curve_params_(5); // Standard dipole: 0/pi/0/pi

   // Top Strap 1:
   double zmin1t = 0.11792;
   double zmax1t = 0.4325;
   double phi1r = 0.0;
   double phi1l = 0.08;
   if (r >= xmin && r <= xmax &&
      z >= zmin1t && z <= zmax1t &&
      x[2] >= phi1l && x[2] <= phi1r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(6)/mag;
            j(2) = curve_params_(7);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(7);
            double j_z   = curve_params_(6)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin1t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 1:
   double zmin1b = -0.30925;
   double zmax1b = 0.01211;
   if (r >= xmin && r <= xmax &&
            z >= zmin1b && z <= zmax1b &&
            x[2] >= phi1l && x[2] <= phi1r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(8)/mag;
            j(2) = curve_params_(9);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(9);
            double j_z   = curve_params_(8)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax1b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Top Strap 2:
   double zmin2t = 0.074;
   double zmax2t = 0.39137;
   double phi2r = 0.181;
   double phi2l = 0.08+0.181;
   if (r >= xmin && r <= xmax &&
      z >= zmin2t && z <= zmax2t &&
      x[2] >= phi2l && x[2] <= phi2r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(6)/mag;
            j(2) = A*curve_params_(7);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = A*curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = A*curve_params_(6);
            double j_z   = A*curve_params_(7)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin2t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 2:
   double zmin2b = -0.34802;
   double zmax2b = -0.0283;
   if (r >= xmin && r <= xmax &&
            z >= zmin2b && z <= zmax2b &&
            x[2] >= phi2l && x[2] <= phi2r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(8)/mag;
            j(2) = A*curve_params_(9);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(9);
            double j_z   = curve_params_(8)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax2b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Top Strap 3:
   double zmin3t = 0.01818;
   double zmax3t = 0.33835;
   double phi3r = 0.181*2;
   double phi3l = 0.08+0.181*2;
   if (r >= xmin && r <= xmax &&
      z >= zmin3t && z <= zmax3t &&
      x[2] >= phi3l && x[2] <= phi3r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(6)/mag;
            j(2) = curve_params_(7);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(7);
            double j_z   = curve_params_(6)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin3t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 3:
   double zmin3b = -0.39616;
   double zmax3b = -0.0791;
   if (r >= xmin && r <= xmax &&
            z >= zmin3b && z <= zmax3b &&
            x[2] >= phi3l && x[2] <= phi3r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = curve_params_(8)/mag;
            j(2) = curve_params_(9);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = curve_params_(9);
            double j_z   = curve_params_(8)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax3b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Top Strap 4:
   double zmin4t = -0.02218;
   double zmax4t = 0.29953;
   double phi4r = 0.181*3;
   double phi4l = 0.08+0.181*3;
   if (r >= xmin && r <= xmax &&
      z >= zmin4t && z <= zmax4t &&
      x[2] >= phi4l && x[2] <= phi4r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(6)/mag;
            j(2) = A*curve_params_(7);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = A*curve_params_(6)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = A*curve_params_(7);
            double j_z   = A*curve_params_(6)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }

         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0 - zmin4t;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   // Bottom Strap 4:
   double zmin4b = -0.42963;
   double zmax4b = -0.11485;
   if (r >= xmin && r <= xmax &&
            z >= zmin4b && z <= zmax4b &&
            x[2] >= phi4l && x[2] <= phi4r)
      {
         double mag = sqrt(4*pow(b,2.0)*pow(z,2.0)+16*pow(c,2.0)*pow(z,6.0)-16*b*c*pow(z,4.0) + 1);
         if (!j_cyl_)
         {
            j(0) = A*curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            j(1) = A*curve_params_(8)/mag;
            j(2) = A*curve_params_(9);
         }
         else
         {
            double cosphi = x[0] / r;
            double sinphi = x[1] / r;

            double j_r   = A*curve_params_(8)*(-2*b*z - 4*c*pow(z,3.0))/mag;
            double j_phi = A*curve_params_(9);
            double j_z   = A*curve_params_(8)/mag;

            j(0) = j_r * cosphi - j_phi * sinphi;
            j(1) = j_r * sinphi + j_phi * cosphi;
            j(2) = j_z;
         }
         if (vol_profile_ == 1)
         {
            double dlant = 0.325;
            double arc_len = -1.0*(z + (4*pow(b,2.0)*pow(z,3.0))/3.0 
               - (16.0/5)*b*c*pow(z,5.0) + (16*pow(c,2.0)*pow(z,7.0))/7.0) - zmax4b;
            j *= pow(cos((M_PI/dlant)*((arc_len+dlant) - dlant/2)),2.0);
         }
      }
   }


void curve_current_source_r(const Vector &x, Vector &j)
{
   curve_current_source_v3_r(x, j);
}

void curve_current_source_i(const Vector &x, Vector &j)
{
   curve_current_source_v3_i(x, j);
}

void e_bc_r(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;

}

void e_bc_i(const Vector &x, Vector &E)
{
   E.SetSize(3);
   E = 0.0;
}
ColdPlasmaPlaneWaveH::ColdPlasmaPlaneWaveH(char type,
                                           double omega,
                                           const Vector & B,
                                           const Vector & number,
                                           const Vector & charge,
                                           const Vector & mass,
                                           const Vector & temp,
                                           int nuprof,
                                           double res_lim,
                                           bool realPart)
   : VectorCoefficient(3),
     type_(type),
     realPart_(realPart),
     nuprof_(nuprof),
     res_lim_(res_lim),
     omega_(omega),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     dx_(0.05),
     Lx_(1.0),
     kappa_(0.0),
     b_(B),
     bc_(3),
     bcc_(3),
     h_r_(3),
     h_i_(3),
     k_r_(3),
     k_i_(3),
     beta_r_(3),
     beta_i_(3),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   b_ *= 1.0 / Bmag_;

   {
      double bx = b_(0);
      double by = b_(1);
      double bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }

   beta_r_ = 0.0;
   beta_i_ = 0.0;

   double nue_ = 0;
   double nui_ = 0;
   double Ti_ = 0;
   double k_ = 18;
   double Rval_ = 0.0;
   double Lval_ = 0.0;

   S_ = S_cold_plasma(omega_, k_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_, Ti_,
                      nuprof_, Rval_, Lval_);
   D_ = D_cold_plasma(omega_, k_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_, Ti_,
                      nuprof_, Rval_, Lval_);
   P_ = P_cold_plasma(omega_, k_, nue_, numbers_, charges_, masses_,
                      temps_, Ti_, nuprof_);

   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         complex<double> h = sqrt((S_ - D_) * (epsilon0_ / mu0_));

         h_r_.Set(-M_SQRT1_2 * h.real(), bcc_);
         h_r_.Add(-M_SQRT1_2 * h.imag(), bc_);
         h_i_.Set( M_SQRT1_2 * h.real(), bc_);
         h_i_.Add(-M_SQRT1_2 * h.imag(), bcc_);
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         complex<double> h = sqrt((S_ + D_) * (epsilon0_ / mu0_));

         h_r_.Set(-M_SQRT1_2 * h.real(), bcc_);
         h_r_.Add( M_SQRT1_2 * h.imag(), bc_);
         h_i_.Set(-M_SQRT1_2 * h.real(), bc_);
         h_i_.Add(-M_SQRT1_2 * h.imag(), bcc_);
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<double> h = sqrt(P_ * (epsilon0_ / mu0_));

         h_r_.Set(h.real(), bcc_);
         h_i_.Set(h.imag(), bcc_);
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<double> h = S_ * sqrt(S_ - D_ * D_ / S_);
         h *= sqrt((epsilon0_ / mu0_) / (S_ * S_ + D_ * D_));

         h_r_.Set(-h.real(), b_);
         h_i_.Set(-h.imag(), b_);
      }
      break;
      case 'J':
         // MFEM_VERIFY(fabs(B_[2]) == Bmag_,
         //             "Current slab require a magnetic field in the z-direction.");
         h_r_ = 0.0;
         h_i_ = 0.0;
         k_r_ = 0.0;
         k_i_ = 0.0;
         break;
      default:
         h_r_ = 0.0;
         h_i_ = 0.0;
         k_r_ = 0.0;
         k_i_ = 0.0;
   }
}

void ColdPlasmaPlaneWaveH::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   double x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<double> i = complex<double>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      case 'R': // Right Circularly Polarized, propagating along B
      case 'O': // Ordinary wave propagating perpendicular to B
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<double> kx = 0.0;
         for (int d=0; d<3; d++)
         {
            kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
         }
         complex<double> phase = exp(i * kx);
         double phase_r = phase.real();
         double phase_i = phase.imag();

         if (realPart_)
         {
            for (int d=0; d<3; d++)
            {
               V[d] = h_r_[d] * phase_r - h_i_[d] * phase_i;
            }
         }
         else
         {
            for (int d=0; d<3; d++)
            {
               V[d] = h_r_[d] * phase_i + h_i_[d] * phase_r;
            }
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         if (true/* k_.Size() == 0 */)
         {
            complex<double> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

            complex<double> skL = sin(kE * Lx_);
            complex<double> E0 = i * Jy_ /
                                 (omega_ * epsilon0_ * skL *
                                  (S_ * S_ - D_ * D_));

            complex<double> Ex = i * D_ * E0;
            complex<double> Ey = S_ * E0;

            if (x[0] <= xJ_ - 0.5 * dx_)
            {
               complex<double> skLJ = sin(kE * (Lx_ - xJ_));
               complex<double> skd  = sin(kE * 0.5 * dx_);
               complex<double> skx  = sin(kE * x[0]);

               Ex *= -2.0 * skLJ * skd * skx;
               Ey *= -2.0 * skLJ * skd * skx;
            }
            else if (x[0] <= xJ_ + 0.5 * dx_)
            {
               complex<double> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
               complex<double> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
               complex<double> skx  = sin(kE * x[0]);
               complex<double> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= skL - ck1 * skx - ck2 * skLx;
               Ey *= skL - ck1 * skx - ck2 * skLx;
            }
            else
            {
               complex<double> skJ  = sin(kE * xJ_);
               complex<double> skd  = sin(kE * 0.5 * dx_);
               complex<double> skLx = sin(kE * (Lx_ - x[0]));

               Ex *= -2.0 * skJ * skd * skLx;
               Ey *= -2.0 * skJ * skd * skLx;
            }

            if (realPart_)
            {
               V[0] = Ex.real();
               V[1] = Ey.real();
               V[2] = 0.0;
            }
            else
            {
               V[0] = Ex.imag();
               V[1] = Ey.imag();
               V[2] = 0.0;
            }
         }
         else
         {
            // General phase shift
            V = 0.0; // For now...
         }
      }
      break;
      case 'Z':
         V = 0.0;
         break;
   }
}

ColdPlasmaPlaneWaveE::ColdPlasmaPlaneWaveE(char type,
                                           double omega,
                                           const Vector & B,
                                           const Vector & number,
                                           const Vector & charge,
                                           const Vector & mass,
                                           const Vector & temp,
                                           int nuprof,
                                           double res_lim,
                                           bool realPart)
   : VectorCoefficient(3),
     type_(type),
     realPart_(realPart),
     nuprof_(nuprof),
     res_lim_(res_lim),
     omega_(omega),
     Bmag_(B.Norml2()),
     Jy_(0.0),
     xJ_(0.5),
     dx_(0.05),
     Lx_(1.0),
     kappa_(0.0),
     b_(B),
     bc_(3),
     bcc_(3),
     e_r_(3),
     e_i_(3),
     k_r_(3),
     k_i_(3),
     beta_r_(3),
     beta_i_(3),
     numbers_(number),
     charges_(charge),
     masses_(mass),
     temps_(temp)
{
   b_ *= 1.0 / Bmag_;

   {
      double bx = b_(0);
      double by = b_(1);
      double bz = b_(2);

      bc_(0) = by - bz;
      bc_(1) = bz - bx;
      bc_(2) = bx - by;

      bcc_(0) = by*by + bz*bz - bx*(by + bz);
      bcc_(1) = bz*bz + bx*bx - by*(bz + bx);
      bcc_(2) = bx*bx + by*by - bz*(bx + by);

      bc_  *= 1.0 / bc_.Norml2();
      bcc_ *= 1.0 / bcc_.Norml2();
   }

   beta_r_ = 0.0;
   beta_i_ = 0.0;

   double nue_ = 0;
   double nui_ = 0;
   double Ti_ = 0;
   double k_ = 18;
   double Rval_ = 0.0;
   double Lval_ = 0.0;

   S_ = S_cold_plasma(omega_, k_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_, Ti_,
                      nuprof_, Rval_, Lval_);
   D_ = D_cold_plasma(omega_, k_, Bmag_, nue_, nui_, numbers_, charges_, masses_,
                      temps_, Ti_,
                      nuprof_, Rval_, Lval_);
   P_ = P_cold_plasma(omega_, k_, nue_, numbers_, charges_, masses_,
                      temps_, Ti_, nuprof_);

   switch (type_)
   {
      case 'L':
      {
         kappa_ = omega_ * sqrt(S_ - D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         e_r_.Set(M_SQRT1_2, bc_);
         e_i_.Set(M_SQRT1_2, bcc_);
      }
      break;
      case 'R':
      {
         kappa_ = omega_ * sqrt(S_ + D_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), b_);
         k_i_.Set(kappa_.imag(), b_);

         e_r_.Set( M_SQRT1_2, bc_);
         e_i_.Set(-M_SQRT1_2, bcc_);
      }
      break;
      case 'O':
      {
         kappa_ = omega_ * sqrt(P_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         e_r_.Set(1.0, b_);
         e_i_ = 0.0;
      }
      break;
      case 'X':
      {
         kappa_ = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;
         if (kappa_.imag() < 0.0) { kappa_ *= -1.0; }

         k_r_.Set(kappa_.real(), bc_);
         k_i_.Set(kappa_.imag(), bc_);

         complex<double> den = sqrt(S_ * S_ + D_ * D_);
         complex<double> ec  = D_ / den;
         complex<double> ecc = S_ / den;

         e_r_.Set(ecc.real(), bcc_);
         e_r_.Add(ec.imag(), bc_);
         e_i_.Set(-ec.real(), bc_);
         e_i_.Add(ecc.imag(), bcc_);
      }
      break;
      case 'J':
         // MFEM_VERIFY(fabs(B_[2]) == Bmag_,
         //           "Current slab require a magnetic field in the z-direction.");
         e_r_ = 0.0;
         e_i_ = 0.0;
         k_r_ = 0.0;
         k_i_ = 0.0;
         break;
      default:
         e_r_ = 0.0;
         e_i_ = 0.0;
         k_r_ = 0.0;
         k_i_ = 0.0;
   }
}

void ColdPlasmaPlaneWaveE::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(3);

   double x_data[3];
   Vector x(x_data, 3);
   T.Transform(ip, x);

   complex<double> i = complex<double>(0.0,1.0);

   switch (type_)
   {
      case 'L': // Left Circularly Polarized, propagating along B
      case 'R': // Right Circularly Polarized, propagating along B
      case 'O': // Ordinary wave propagating perpendicular to B
      case 'X': // eXtraordinary wave propagating perpendicular to B
      {
         complex<double> kx = 0.0;
         for (int d=0; d<3; d++)
         {
            kx += (k_r_[d] - beta_r_[d] + i * (k_i_[d] - beta_i_[d])) * x[d];
         }
         complex<double> phase = exp(i * kx);
         double phase_r = phase.real();
         double phase_i = phase.imag();

         if (realPart_)
         {
            for (int d=0; d<3; d++)
            {
               V[d] = e_r_[d] * phase_r - e_i_[d] * phase_i;
            }
         }
         else
         {
            for (int d=0; d<3; d++)
            {
               V[d] = e_r_[d] * phase_i + e_i_[d] * phase_r;
            }
         }
      }
      break;
      case 'J':  // Slab of current density perpendicular to propagation
      {
         // if (k_r_.Size() == 0 && k_i_.Size() == 0)
         //     {
         complex<double> kE = omega_ * sqrt(S_ - D_ * D_ / S_) / c0_;

         complex<double> skL = sin(kE * Lx_);
         complex<double> E0 = -1.0 * i * Jy_ /
                              (omega_ * epsilon0_ * skL *
                               (S_ * S_ - D_ * D_));

         complex<double> Ex = i * D_ * E0;
         complex<double> Ey = S_ * E0;

         if (x[0] <= xJ_ - 0.5 * dx_)
         {
            complex<double> skLJ = sin(kE * (Lx_ - xJ_));
            complex<double> skd  = sin(kE * 0.5 * dx_);
            complex<double> skx  = sin(kE * x[0]);

            Ex *= -2.0 * skLJ * skd * skx;
            Ey *= -2.0 * skLJ * skd * skx;
         }
         else if (x[0] <= xJ_ + 0.5 * dx_)
         {
            complex<double> ck1  = cos(kE * (Lx_ - xJ_ - 0.5 * dx_));
            complex<double> ck2  = cos(kE * (xJ_ - 0.5 * dx_));
            complex<double> skx  = sin(kE * x[0]);
            complex<double> skLx = sin(kE * (Lx_ - x[0]));

            Ex *= skL - ck1 * skx - ck2 * skLx;
            Ey *= skL - ck1 * skx - ck2 * skLx;
         }
         else
         {
            complex<double> skJ  = sin(kE * xJ_);
            complex<double> skd  = sin(kE * 0.5 * dx_);
            complex<double> skLx = sin(kE * (Lx_ - x[0]));

            Ex *= -2.0 * skJ * skd * skLx;
            Ey *= -2.0 * skJ * skd * skLx;
         }

         if (realPart_)
         {
            V[0] = Ex.real();
            V[1] = Ey.real();
            V[2] = 0.0;
         }
         else
         {
            V[0] = Ex.imag();
            V[1] = Ey.imag();
            V[2] = 0.0;
         }
         // }
         /*
         else
         {
         // General phase shift
         V = 0.0; // For now...
         }
         */
      }
      break;
      case 'Z':
         V = 0.0;
         break;
   }
}