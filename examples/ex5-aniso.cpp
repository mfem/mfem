//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/star.mesh -pa
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/beam-hex.mesh -pa
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex5 -m ../data/star.mesh -pa -d cuda
//               ex5 -m ../data/star.mesh -pa -d raja-cuda
//               ex5 -m ../data/star.mesh -pa -d raja-omp
//               ex5 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D asymptotic heat diffusion
//               problem in the mixed formulation corresponding to the system
//
//                                 k^-1.q +         grad T =  g
//                                  div q + div(T*c) + a T = -f
//
//               with natural boundary condition q.n = 0, where n is the outer
//               normal. The tensor k represents the heat conductivity, where its
//               symmetric and antisymmetric parts can be adjusted. The scalar a
//               is then the heat capacity, which can be zero, changing the problem
//               to steady-state, indefinite, saddle-point. The r.h.s. is f = 0 and
//               g = -a * <initial temperature> for the definite problem and
//               g = -<initial temperature> for the indefinite one. These problems
//               are offered:
//               1) sine diffusion - with the asymptotic (a -> infinity) reference
//                                   solution with the first order correction
//               2) MFEM text conv-diff - random Gaussian blobs of conductivity
//                                        and circular velocity with ASCII art
//                                        of MFEM text as IC
//               3) diffusion ring - arc segment IC diffused along circle
//               4) diffusion ring Gauss - Gaussian blobs IC diffused along circle
//               5) diffusion ring sine - sine profile in radial and angular
//                                        direction is diffused along circle,
//                                        analytic solution for asymptotic
//                                        diffusion with zero radial diffusion
//               6) boundary layer - exponentially decaying boundary layer problem
//               7) steady peak - a peak profile with a constant conductivity and
//                                a manufactured steady-state solution
//               8) steady varying angle - a concave radial profile diffused
//                                         along the circle with a manufactured
//                                         steady-state solution
//               9) Sovinec problem - a sine profile with diffusion perpendicular
//                                    to gradient of potential with a manufactured
//                                    steady-state solution
//               10) Umansky problem - a transition profile with with diffusion
//                                     along the interface, where the width is
//                                     measured automatically
//               We discretize with Raviart-Thomas finite elements (heat flux q)
//               and piecewise discontinuous polynomials (temperature T). Alternatively,
//               the piecewise discontinuous polynomials are used for both quantities.
//
//               The example demonstrates the use of the DarcyForm class, as
//               well as hybridization of mixed systems and the collective saving
//               of several grid functions in VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "darcyop.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

#define DIFFUSION_STRONG

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef std::function<void(const Vector &, DenseMatrix &)> MatFunc;

enum Problem
{
   SteadyDiffusion = 1,
   MFEMLogo,
   DiffusionRing,
   DiffusionRingGauss,
   DiffusionRingSine,
   BoundaryLayer,
   SteadyPeak,
   SteadyVaryingAngle,
   Sovinec,
   Umansky,
};

constexpr real_t epsilon = numeric_limits<real_t>::epsilon();

struct ProblemParams
{
   Problem prob;
   int nx, ny;
   real_t x0, y0, sx, sy;
   int order;
   real_t k, ks, ka;
   real_t t_0;
   real_t a;
   real_t c;
};

class RotationInterpolator : public DiscreteInterpolator
{
   VectorCoefficient &b_vcoeff;

   DenseMatrix b_vals;

public:

   RotationInterpolator(VectorCoefficient &b) : b_vcoeff(b) { }

   void AssembleElementMatrix(const FiniteElement &fe,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   {
      MFEM_ASSERT(&dom_fe == &ran_fe, "");
      AssembleElementMatrix(dom_fe, Trans, elmat);
   }
};

class VectorRotatedWeakDivergenceIntegrator : public BilinearFormIntegrator
{
   VectorCoefficient &b_vcoeff;

   DenseMatrix te_dshape, te_dshapext, te_curlshape;
   Vector tr_shape, te_bgshape;

public:
   VectorRotatedWeakDivergenceIntegrator(VectorCoefficient &b)
      : b_vcoeff(b) { }

   const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& Trans) const override
   {
      const int order = Trans.OrderGrad(&test_fe) + trial_fe.GetOrder() +
                        Trans.OrderJ();// + test_fe.GetOrder() + 1;
      return &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

class VectorRotatedDivergenceIntegrator : public BilinearFormIntegrator
{
   const GridFunction &b_gf;

   DenseMatrix tr_dshape, tr_dshapext, b_vals;
   Vector te_shape, tr_divshape;

public:
   VectorRotatedDivergenceIntegrator(const GridFunction &b)
      : b_gf(b) { }

   const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& Trans) const override
   {
      const int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder() +
                        Trans.OrderJ();// + trial_fe.GetOrder() + 1;
      return &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

class DGRotatedWeakNormalTraceIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient &b_vcoeff;
   Coefficient *rho;
   VectorCoefficient *u;
   real_t alpha, beta;

private:
   Vector tr_shape1, te_shape1, tr_shape2, te_shape2;

public:
   /// Construct integrator with $\rho = 1$, $\beta = 0$.
   DGRotatedWeakNormalTraceIntegrator(VectorCoefficient &bvc, real_t a = 1.)
      : b_vcoeff(bvc) { rho = NULL; u = NULL; alpha = a; beta = 0.; }

   /// Construct integrator with $\rho = 1$, $\beta = \alpha/2$.
   DGRotatedWeakNormalTraceIntegrator(VectorCoefficient &bvc,
                                      VectorCoefficient &u_,
                                      real_t a = 1.)
      : b_vcoeff(bvc) { rho = NULL; u = &u_; alpha = a; beta = 0.5*a; }

   /// Construct integrator with $\rho = 1$.
   DGRotatedWeakNormalTraceIntegrator(VectorCoefficient &bvc,
                                      VectorCoefficient &u_,
                                      real_t a, real_t b)
      : b_vcoeff(bvc) { rho = NULL; u = &u_; alpha = a; beta = b; }

   DGRotatedWeakNormalTraceIntegrator(VectorCoefficient &bvc, Coefficient &rho_,
                                      VectorCoefficient &u_,
                                      real_t a, real_t b)
      : b_vcoeff(bvc) { rho = &rho_; u = &u_; alpha = a; beta = b; }

   void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                           const FiniteElement &test_fe1,
                           const FiniteElement &trial_fe2,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

class DGRotatedNormalTraceIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient &b_vcoeff;
   Coefficient *rho;
   VectorCoefficient *u;
   real_t alpha, beta;

private:
   Vector tr_shape1, te_shape1, tr_shape2, te_shape2;

public:
   /// Construct integrator with $\rho = 1$, $\beta = 0$.
   DGRotatedNormalTraceIntegrator(VectorCoefficient &bvc, real_t a = 1.)
      : b_vcoeff(bvc) { rho = NULL; u = NULL; alpha = a; beta = 0.; }

   /// Construct integrator with $\rho = 1$, $\beta = \alpha/2$.
   DGRotatedNormalTraceIntegrator(VectorCoefficient &bvc, VectorCoefficient &u_,
                                  real_t a = 1.)
      : b_vcoeff(bvc) { rho = NULL; u = &u_; alpha = a; beta = 0.5*a; }

   /// Construct integrator with $\rho = 1$.
   DGRotatedNormalTraceIntegrator(VectorCoefficient &bvc, VectorCoefficient &u_,
                                  real_t a, real_t b)
      : b_vcoeff(bvc) { rho = NULL; u = &u_; alpha = a; beta = b; }

   DGRotatedNormalTraceIntegrator(VectorCoefficient &bvc, Coefficient &rho_,
                                  VectorCoefficient &u_,
                                  real_t a, real_t b)
      : b_vcoeff(bvc) { rho = &rho_; u = &u_; alpha = a; beta = b; }

   void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                           const FiniteElement &test_fe1,
                           const FiniteElement &trial_fe2,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

class RotatedNormalTraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient &b_vcoeff;
   Vector face_shape, normal, shape1_n, shape2_n;
   DenseMatrix shape1, shape2;
   real_t sign;

public:
   RotatedNormalTraceJumpIntegrator(VectorCoefficient &bvc, real_t sign_ = 1.)
      : b_vcoeff(bvc), sign(sign_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

MatFunc GetKFun(const ProblemParams &params);
VecFunc GetBFun(const ProblemParams &params);
TFunc GetTFun(const ProblemParams &params);
VecTFunc GetQFun(const ProblemParams &params);
VecFunc GetCFun(const ProblemParams &params);
TFunc GetFFun(const ProblemParams &params);
FluxFunction* GetFluxFun(const ProblemParams &params,
                         VectorCoefficient &ccoeff);
MixedFluxFunction* GetHeatFluxFun(const ProblemParams &params, int dim);

real_t UmanskyTestWidth(const GridFunction &u);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "";
   int ref_levels = -1;
   real_t dr = 0.;
   bool dg = false;
   bool brt = false;
   bool upwinded = false;
   int iproblem = Problem::SteadyDiffusion;
   real_t tf = 1.;
   int nt = 0;
   int ode = 1;
   ProblemParams pars;
   pars.nx = 0;
   pars.ny = 0;
   pars.x0 = 0.;
   pars.y0 = 0.;
   pars.sx = 1.;
   pars.sy = 1.;
   pars.order = 1;
   const int &order = pars.order;
   pars.k = 1.;
   pars.ks = 1.;
   pars.ka = 0.;
   pars.a = 0.;
   pars.c = 1.;
   real_t td = 0.5;
   bool bc_neumann = false;
   bool reduction = false;
   bool hybridization = false;
   bool nonlinear = false;
   bool nonlinear_conv = false;
   bool nonlinear_diff = false;
   int hdg_scheme = 1;
   int solver_type = (int)DarcyOperator::SolverType::LBFGS;
   int isol_ctrl = (int)DarcyOperator::SolutionController::Type::Native;
   bool pa = false;
   const char *device_config = "cpu";
   bool reconstruct = false;
   bool rotated = false;
   bool mfem = false;
   bool visit = false;
   bool paraview = false;
   bool visualization = true;
   int vis_iters = -1;
   bool analytic = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&dr, "-dr", "--delta-random",
                  "Relative random displacement of the mesh nodes.");
   args.AddOption(&pars.nx, "-nx", "--ncells-x",
                  "Number of cells in x.");
   args.AddOption(&pars.ny, "-ny", "--ncells-y",
                  "Number of cells in y.");
   args.AddOption(&pars.sx, "-sx", "--size-x",
                  "Size along x axis.");
   args.AddOption(&pars.sy, "-sy", "--size-y",
                  "Size along y axis.");
   args.AddOption(&pars.order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&dg, "-dg", "--discontinuous", "-no-dg",
                  "--no-discontinuous", "Enable DG elements for fluxes.");
   args.AddOption(&brt, "-brt", "--broken-RT", "-no-brt",
                  "--no-broken-RT", "Enable broken RT elements for fluxes.");
   args.AddOption(&upwinded, "-up", "--upwinded", "-ce", "--centered",
                  "Switches between upwinded (1) and centered (0=default) stabilization.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=sine diffusion\n\t\t"
                  "2=MFEM logo\n\t\t"
                  "3=diffusion ring\n\t\t"
                  "4=diffusion ring - Gauss source\n\t\t"
                  "5=diffusion ring - sine source\n\t\t"
                  "6=boundary layer\n\t\t"
                  "7=steady peak\n\t\t"
                  "8=steady varying angle\n\t\t"
                  "9=Sovinec\n\t\t"
                  "10=Umansky\n\t\t");
   args.AddOption(&tf, "-tf", "--time-final",
                  "Final time.");
   args.AddOption(&nt, "-nt", "--ntimesteps",
                  "Number of time steps.");
   args.AddOption(&ode, "-ode", "--ode-solver",
                  "ODE time solver (1=Bacward Euler, 2=RK23L, 3=RK23A, 4=RK34).");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.ka, "-ka", "--kappa_anti",
                  "Antisymmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.a, "-a", "--heat_capacity",
                  "Heat capacity coefficient (0=indefinite problem)");
   args.AddOption(&pars.c, "-c", "--velocity",
                  "Convection velocity");
   args.AddOption(&td, "-td", "--stab_diff",
                  "Diffusion stabilization factor (1/2=default)");
   args.AddOption(&bc_neumann, "-bcn", "--bc-neumann", "-no-bcn",
                  "--no-bc-neumann", "Enable Neumann outflow boundary condition.");
   args.AddOption(&reduction, "-rd", "--reduction", "-no-rd",
                  "--no-reduction", "Enable reduction.");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.AddOption(&nonlinear, "-nl", "--nonlinear", "-no-nl",
                  "--no-nonlinear", "Enable non-linear regime.");
   args.AddOption(&nonlinear_conv, "-nlc", "--nonlinear-convection", "-no-nlc",
                  "--no-nonlinear-convection", "Enable non-linear convection regime.");
   args.AddOption(&nonlinear_diff, "-nld", "--nonlinear-diffusion", "-no-nld",
                  "--no-nonlinear-diffusion", "Enable non-linear diffusion regime.");
   args.AddOption(&hdg_scheme, "-hdg", "--hdg_scheme",
                  "HDG scheme (1=HDG-I, 2=HDG-II, 3=Rusanov, 4=Godunov).");
   args.AddOption(&solver_type, "-nls", "--nonlinear-solver",
                  "Nonlinear solver type (1=LBFGS, 2=LBB, 3=Newton).");
   args.AddOption(&isol_ctrl, "-sn", "--solution-norm",
                  "Solution norm (0=native, 1=flux, 2=potential).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&reconstruct, "-rec", "--reconstruct", "-no-rec",
                  "--no-reconstruct",
                  "Enable or disable quantities reconstruction.");
   args.AddOption(&rotated, "-rot", "--rotated", "-no-rot",
                  "--no-rotated",
                  "Enable or disable fluxes rotation.");
   args.AddOption(&mfem, "-mfem", "--mfem", "-no-mfem",
                  "--no-mfem",
                  "Enable or disable MFEM output.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visit",
                  "Enable or disable Visit output.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView output.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_iters, "-vis-its", "--visualization-iters",
                  "Set step for GLVis visualization of the solver iterations (<0=off).");
   args.AddOption(&analytic, "-anal", "--analytic", "-no-anal",
                  "--no-analytic",
                  "Enable or disable analytic solution.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Set the problem options
   pars.prob = (Problem)iproblem;
   const Problem &problem = pars.prob;
   bool bconv = false, bnlconv = false, bnldiff = nonlinear_diff, btime = false;
   switch (problem)
   {
      case Problem::SteadyDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::Umansky:
         break;
      case Problem::MFEMLogo:
         bconv = true;
         break;
      default:
         cerr << "Unknown problem" << endl;
         return 1;
   }

   if (bnldiff && reduction)
   {
      cerr << "Reduction is not possible with non-linear diffusion" << endl;
      return 1;
   }

   if (!bconv && !bnlconv && upwinded)
   {
      cerr << "Upwinded scheme cannot work without advection" << endl;
      return 1;
   }

   if (bnlconv && !nonlinear)
   {
      cerr << "Nonlinear convection can only work in the nonlinear regime" << endl;
      return 1;
   }

   if (nonlinear && !hybridization)
   {
      cerr << "Warning: A linear solver is used" << endl;
   }

   if (btime && nt <= 0)
   {
      cerr << "You must specify the number of time steps for time evolving problems"
           << endl;
      return 1;
   }

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   if (pars.ny <= 0)
   {
      pars.ny = pars.nx;
   }

   Mesh *mesh = NULL;
   if (strlen(mesh_file) > 0)
   {
      mesh = new Mesh(mesh_file, 1, 1);

      Vector x_min(2), x_max(2);
      mesh->GetBoundingBox(x_min, x_max);
      pars.x0 = x_min(0);
      pars.y0 = x_min(1);
      pars.sx = x_max(0) - x_min(0);
      pars.sy = x_max(1) - x_min(1);
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(pars.nx, pars.ny,
                                            Element::QUADRILATERAL, false,
                                            pars.sx, pars.sy));
   }

   int dim = mesh->Dimension();

   // Mark boundary conditions
   Array<int> bdr_is_dirichlet(mesh->bdr_attributes.Max());
   Array<int> bdr_is_neumann(mesh->bdr_attributes.Max());
   bdr_is_dirichlet = 0;
   bdr_is_neumann = 0;

   switch (problem)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyPeak:
      case Problem::Sovinec:
         //free (zero Dirichlet)
         if (bc_neumann)
         {
            bdr_is_neumann[1] = -1;//outflow
            bdr_is_neumann[2] = -1;//outflow
         }
         break;
      case Problem::BoundaryLayer:
         bdr_is_dirichlet[0] = -1;
         bdr_is_dirichlet[2] = -1;
         break;
      case Problem::SteadyVaryingAngle:
         bdr_is_dirichlet = -1;
         break;
      case Problem::Umansky:
         bdr_is_dirichlet[0] = -1;
         bdr_is_dirichlet[1] = -1;
         break;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   if (strlen(mesh_file) > 0)
   {
      if (ref_levels < 0)
      {
         ref_levels = (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      }
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   if (dr > 0.) { RandomizeMesh(*mesh, dr); }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *V_coll, *V_coll_dg = NULL;
   if (dg)
   {
      // In the case of LDG formulation, we chose a closed basis as it
      // is customary for HDG to match trace DOFs, but an open basis can
      // be used instead.
      V_coll = new L2_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else if (brt)
   {
      V_coll = new BrokenRT_FECollection(order, dim);
      V_coll_dg = new L2_FECollection(order+1, dim);
   }
   else
   {
      V_coll = new RT_FECollection(order, dim);
   }
   FiniteElementCollection *W_coll = new L2_FECollection(order, dim,
                                                         BasisType::GaussLobatto);

   FiniteElementSpace *V_space = new FiniteElementSpace(mesh, V_coll,
                                                        (dg)?(dim):(1));
   FiniteElementSpace *V_space_dg = (V_coll_dg)?(new FiniteElementSpace(
                                                    mesh, V_coll_dg, dim)):(NULL);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, W_coll);

   DarcyForm *darcy = new DarcyForm(V_space, W_space);

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   pars.t_0 = 1.; //base temperature

   ConstantCoefficient acoeff(pars.a); //heat capacity

   constexpr unsigned int seed = 0;
   srand(seed);// init random number generator

   //H1_FECollection B_coll(order+1, dim);
   //FiniteElementSpace B_space(mesh, &B_coll, dim);
   GridFunction b_gf(V_space);
   auto bFun = GetBFun(pars);
   VectorFunctionCoefficient bcoeff(dim, bFun);
   b_gf.ProjectCoefficient(bcoeff);
   VectorGridFunctionCoefficient bhcoeff(&b_gf);

   DenseMatrix ikmat;
   ikmat.Diag(1. / (pars.k * pars.ks), dim);
   ikmat(0,0) = 1. / pars.k;
   MatrixConstantCoefficient ikcoeff_rot(ikmat);
   auto kFun = GetKFun(pars);
   MatrixFunctionCoefficient kcoeff_norm(dim, kFun); //tensor conductivity
   InverseMatrixCoefficient ikcoeff_norm(
      kcoeff_norm); //inverse tensor conductivity
   MatrixCoefficient &kcoeff = kcoeff_norm;
   MatrixCoefficient &ikcoeff = (rotated)?((MatrixCoefficient&)ikcoeff_rot)
                                :((MatrixCoefficient&)ikcoeff_norm);

   auto cFun = GetCFun(pars);
   VectorFunctionCoefficient ccoeff(dim, cFun); //velocity

   auto tFun = GetTFun(pars);
   FunctionCoefficient tcoeff(tFun); //temperature
   SumCoefficient gcoeff(0., tcoeff, 1., -1.); //boundary heat flux rhs

   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); //temperature rhs

   auto qFun = GetQFun(pars);
   VectorFunctionCoefficient qcoeff(dim, qFun); //heat flux
   ConstantCoefficient one;
   VectorSumCoefficient qtcoeff_(ccoeff, qcoeff, tcoeff, one);//total flux
   VectorCoefficient &qtcoeff = (bconv)?((VectorCoefficient&)qtcoeff_)
                                :((VectorCoefficient&)qcoeff);//<--velocity is undefined

   // 7. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   q_h, v_h \in V_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   q_h \in V_h, w_h \in W_h
   BilinearForm *Mq =(!nonlinear && !bnldiff)?(darcy->GetFluxMassForm()):(NULL);
   NonlinearForm *Mqnl = (nonlinear && !bnldiff)?
                         (darcy->GetFluxMassNonlinearForm()):(NULL);
   BlockNonlinearForm *Mnl = (bnldiff)?(darcy->GetBlockNonlinearForm()):(NULL);
   MixedBilinearForm *B = darcy->GetFluxDivForm();
   BilinearForm *Mt = (!nonlinear && ((dg && td > 0.) || bconv || btime ||
                                      pars.a > 0.))?
                      (darcy->GetPotentialMassForm()):(NULL);
   NonlinearForm *Mtnl = (nonlinear && ((dg && td > 0.) || bconv || bnlconv ||
                                        pars.a > 0. || btime))?
                         (darcy->GetPotentialMassNonlinearForm()):(NULL);
   FluxFunction *FluxFun = NULL;
   NumericalFlux *FluxSolver = NULL;
   MixedFluxFunction *HeatFluxFun = NULL;

   //diffusion

   if (!bnldiff)
   {
      //linear diffusion
      if (dg)
      {
         if (Mq)
         {
            Mq->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
         }
         if (Mqnl)
         {
            Mqnl->AddDomainIntegrator(new VectorMassIntegrator(ikcoeff));
         }
      }
      else
      {
         if (Mq)
         {
            Mq->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
         }
         if (Mqnl)
         {
            Mqnl->AddDomainIntegrator(new VectorFEMassIntegrator(ikcoeff));
         }
      }
   }
   else
   {
      //nonlinear diffusion
      HeatFluxFun = GetHeatFluxFun(pars, dim);
      if (dg)
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
      }
      else
      {
         Mnl->AddDomainIntegrator(new MixedConductionNLFIntegrator(*HeatFluxFun));
      }
   }

   //diffusion stabilization
   if (dg)
   {
      if (bnldiff)
      {
         cerr << "Warning: Using linear stabilization for non-linear diffusion" << endl;
      }

      if (upwinded && td > 0. && hybridization)
      {
         if (Mt)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
            Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td),
                                     bdr_is_neumann);
         }
         if (Mtnl)
         {
            Mtnl->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td));
            Mtnl->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(ccoeff, kcoeff, td),
                                       bdr_is_neumann);
         }
      }
      else if (!upwinded && td > 0.)
      {
         if (Mt)
         {
            Mt->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
#ifndef DIFFUSION_STRONG
            if (rotated)
            {
               Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            }
            else
#endif //DIFFUSION_STRONG
            {
               Mt->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                        bdr_is_neumann);
            }
         }
         if (Mtnl)
         {
            Mtnl->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td));
            Mtnl->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(kcoeff, td),
                                       bdr_is_neumann);
         }
      }
   }

   //divergence/weak gradient

   if (dg)
   {
      if (rotated)
      {
#ifndef DIFFUSION_STRONG
         B->AddDomainIntegrator(new VectorRotatedWeakDivergenceIntegrator(bhcoeff));
#else //DIFFUSION_STRONG
         B->AddDomainIntegrator(new VectorRotatedDivergenceIntegrator(b_gf));
#endif //DIFFUSION_STRONG
      }
      else
      {
         B->AddDomainIntegrator(new VectorDivergenceIntegrator());
      }
   }
   else
   {
      B->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
   }

   if (dg || brt)
   {
      if (upwinded)
      {
         B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                         new DGNormalTraceIntegrator(ccoeff, -1.)));
         B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                            ccoeff, -1.)), bdr_is_neumann);
      }
      else
      {
         if (rotated)
         {
#ifndef DIFFUSION_STRONG
            B->AddInteriorFaceIntegrator(new DGRotatedWeakNormalTraceIntegrator(bhcoeff));
            B->AddBdrFaceIntegrator(new DGRotatedWeakNormalTraceIntegrator(bhcoeff));
#else //DIFFUSION_STRONG
            B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                            new DGRotatedNormalTraceIntegrator(bhcoeff, -1.)));
            B->AddBdrFaceIntegrator(new TransposeIntegrator(
                                       new DGRotatedNormalTraceIntegrator(bhcoeff, -1.)), bdr_is_neumann);
#endif //DIFFUSION_STRONG
         }
         else
         {
            B->AddInteriorFaceIntegrator(new TransposeIntegrator(
                                            new DGNormalTraceIntegrator(-1.)));
            B->AddBdrFaceIntegrator(new TransposeIntegrator(new DGNormalTraceIntegrator(
                                                               -1.)), bdr_is_neumann);
         }
      }
   }

   //linear convection in the linear regime

   if (bconv && Mt)
   {
      Mt->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
      if (upwinded)
      {
         Mt->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
         Mt->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
      }
      else
      {
         Mt->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         if (hybridization)
         {
            //centered scheme does not work with Dirichlet when hybridized,
            //giving an diverging system, we use the full BC flux here
            Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                     bdr_is_neumann);
         }
         else
         {
            Mt->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         }
      }
   }

   //linear convection in the nonlinear regime

   if (bconv && Mtnl)
   {
      Mtnl->AddDomainIntegrator(new ConservativeConvectionIntegrator(ccoeff));
      if (upwinded)
      {
         Mtnl->AddInteriorFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
         Mtnl->AddBdrFaceIntegrator(new HDGConvectionUpwindedIntegrator(ccoeff));
      }
      else
      {
         Mtnl->AddInteriorFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         if (hybridization)
         {
            //centered scheme does not work with Dirichlet when hybridized,
            //giving an diverging system, we use the full BC flux here
            Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff),
                                       bdr_is_neumann);
         }
         else
         {
            Mtnl->AddBdrFaceIntegrator(new HDGConvectionCenteredIntegrator(ccoeff));
         }
      }
   }

   //nonlinear convection in the nonlinear regime

   if (bnlconv && Mtnl)
   {
      FluxFun = GetFluxFun(pars, ccoeff);
      switch (hdg_scheme)
      {
         case 1: FluxSolver = new HDGFlux(*FluxFun, HDGFlux::HDGScheme::HDG_1); break;
         case 2: FluxSolver = new HDGFlux(*FluxFun, HDGFlux::HDGScheme::HDG_2); break;
         case 3: FluxSolver = new RusanovFlux(*FluxFun); break;
         case 4: FluxSolver = new ComponentwiseUpwindFlux(*FluxFun); break;
         default:
            cerr << "Unknown HDG scheme" << endl;
            exit(1);
      }
      Mtnl->AddDomainIntegrator(new HyperbolicFormIntegrator(*FluxSolver, 0, -1.));
      Mtnl->AddInteriorFaceIntegrator(new HyperbolicFormIntegrator(
                                         *FluxSolver, 0, -1.));
      Mtnl->AddBdrFaceIntegrator(new HyperbolicFormIntegrator(
                                    *FluxSolver, 0, -1.));
   }

   //inertial term

   if (pars.a > 0.)
   {
      if (Mt)
      {
         Mt->AddDomainIntegrator(new MassIntegrator(acoeff));
      }
      else
      {
         Mtnl->AddDomainIntegrator(new MassIntegrator(acoeff));
      }
   }

   //set hybridization / assembly level

   Array<int> ess_flux_tdofs_list;
   if (!dg && !brt)
   {
      V_space->GetEssentialTrueDofs(bdr_is_neumann, ess_flux_tdofs_list);
   }

   FiniteElementCollection *trace_coll = NULL;
   FiniteElementSpace *trace_space = NULL;


   if (hybridization)
   {
      chrono.Clear();
      chrono.Start();

      trace_coll = new DG_Interface_FECollection(order, dim);
      trace_space = new FiniteElementSpace(mesh, trace_coll);
      if (rotated)
         darcy->EnableHybridization(trace_space,
                                    new RotatedNormalTraceJumpIntegrator(bhcoeff),
                                    ess_flux_tdofs_list);
      else
         darcy->EnableHybridization(trace_space,
                                    new NormalTraceJumpIntegrator(),
                                    ess_flux_tdofs_list);

      chrono.Stop();
      std::cout << "Hybridization init took " << chrono.RealTime() << "s.\n";
   }
   else if (reduction)
   {
      chrono.Clear();
      chrono.Start();

      if (dg || brt)
      {
         darcy->EnableFluxReduction();
      }
      else if (!bconv && !bnlconv)
      {
         darcy->EnablePotentialReduction(ess_flux_tdofs_list);
      }
      else
      {
         std::cerr << "No possible reduction!" << std::endl;
         return 1;
      }

      chrono.Stop();
      std::cout << "Reduction init took " << chrono.RealTime() << "s.\n";
   }

   if (pa) { darcy->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

   // 8. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   const Array<int> block_offsets(DarcyOperator::ConstructOffsets(*darcy));

   std::cout << "***********************************************************\n";
   if (!reduction || (reduction && !dg && !brt))
   {
      std::cout << "dim(V) = " << block_offsets[1] - block_offsets[0] << "\n";
   }
   if (!reduction || (reduction && (dg || brt)))
   {
      std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   }
   if (!reduction)
   {
      if (hybridization)
      {
         std::cout << "dim(M) = " << block_offsets[3] - block_offsets[2] << "\n";
         std::cout << "dim(V+W+M) = " << block_offsets.Last() << "\n";
      }
      else
      {
         std::cout << "dim(V+W) = " << block_offsets.Last() << "\n";
      }
   }
   std::cout << "***********************************************************\n";

   // 9. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction q,t for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (q,t) and the linear forms (fform, gform).
   MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   x = 0.;
   GridFunction q_h, t_h, qt_h, q_hs, t_hs, tr_hs;
   q_h.MakeRef(V_space, x.GetBlock(0), 0);
   t_h.MakeRef(W_space, x.GetBlock(1), 0);

   if (btime)
   {
      t_h.ProjectCoefficient(tcoeff); //initial condition
   }

   if (!dg && !brt)
   {
      q_h.ProjectBdrCoefficientNormal(qcoeff,
                                      bdr_is_neumann);   //essential Neumann BC
   }

   LinearForm *gform(new LinearForm);
   gform->Update(V_space, rhs.GetBlock(0), 0);
   if (dg)
   {
      gform->AddBdrFaceIntegrator(new VectorBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_is_dirichlet);
   }
   else if (brt)
   {
      gform->AddBdrFaceIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                  bdr_is_dirichlet);
   }
   else
   {
      gform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(gcoeff),
                                   bdr_is_dirichlet);
   }

   LinearForm *fform(new LinearForm);
   fform->Update(W_space, rhs.GetBlock(1), 0);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   if (!hybridization)
   {
      if (upwinded)
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qtcoeff, +1.),
                                     bdr_is_neumann);
      else
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, qtcoeff, +1., 0.),
                                     bdr_is_neumann);
   }
   if (bconv)
   {
      if (upwinded)
         fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1.),
                                     bdr_is_dirichlet);
      else
      {
         if (hybridization)
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +2., 0.),
                                        bdr_is_dirichlet);//<-- full BC flux, see above
         else
            fform->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(tcoeff, ccoeff, +1., 0.),
                                        bdr_is_dirichlet);
      }
   }

   //prepare (reduced) solution and rhs vectors

   LinearForm *hform = NULL;

   //Neumann BC for the hybridized system

   if (hybridization)
   {
      hform = new LinearForm();
      hform->Update(trace_space, rhs.GetBlock(2), 0);
      //note that Neumann BC must be applied only for the heat flux
      //and not the total flux for stability reasons
      hform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(qcoeff, 2),
                                   bdr_is_neumann);
   }

   //construct the operator

   Array<Coefficient*> coeffs({(Coefficient*)&gcoeff,
                               (Coefficient*)&fcoeff,
                               (Coefficient*)&qtcoeff});

   DarcyOperator op(ess_flux_tdofs_list, darcy, gform, fform, hform, coeffs,
                    (DarcyOperator::SolverType) solver_type, false, btime);

   op.EnableSolutionController(
      (DarcyOperator::SolutionController::Type) isol_ctrl);

   if (vis_iters >= 0)
   {
      op.EnableIterationsVisualization(vis_iters);
   }

   //construct the time solver

   ODESolver *ode_solver;

   switch (ode)
   {
      case 1: ode_solver = new BackwardEulerSolver(); break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK23Solver(); break;
      case 4: ode_solver = new SDIRK34Solver(); break;
      default:
         MFEM_ABORT("Unknown solver");
         return 1;
   }

   ode_solver->Init(op);

   //iterate in time

   if (!btime) { nt = 1; }

   const real_t dt = tf / nt; //time step

   for (int ti = 0; ti < nt; ti++)
   {
      //set current time

      real_t t = tf * ti / nt;

      //perform time step

      real_t dt_ = dt;//<---ignore time step changes
      ode_solver->Step(x, t, dt_);

      // 12. Compute the L2 error norms.

      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      real_t err_q  = q_h.ComputeL2Error(qcoeff, irs);
      real_t norm_q = ComputeLpNorm(2., qcoeff, *mesh, irs);
      real_t err_t  = t_h.ComputeL2Error(tcoeff, irs);
      real_t norm_t = ComputeLpNorm(2., tcoeff, *mesh, irs);

      if (btime)
      {
         cout << "iter:\t" << ti
              << "\ttime:\t" << t
              << "\tq_err:\t" << err_q / norm_q
              << "\tt_err:\t" << err_t / norm_t
              << endl;
      }
      else
      {
         if (problem == Problem::Umansky)
         {
            const real_t w = UmanskyTestWidth(t_h);
            cout << "Umansky width: " << w << "\n";
         }
         cout << "|| q_h - q_ex || / || q_ex || = " << err_q / norm_q << "\n";
         cout << "|| t_h - t_ex || / || t_ex || = " << err_t / norm_t << "\n";
      }

      if (reconstruct)
      {
         darcy->Reconstruct(x, x.GetBlock(2), qt_h, q_hs, t_hs, tr_hs);
         real_t err_qt = qt_h.ComputeL2Error(qtcoeff, irs);
         real_t norm_qt = ComputeLpNorm(2., qtcoeff, *mesh, irs);
         cout << "|| qt_h - qt_ex || / || qt_ex || = " << err_qt / norm_qt << "\n";
         real_t err_qs = q_hs.ComputeL2Error(qcoeff, irs);
         cout << "|| q_hs - q_ex || / || q_ex || = " << err_qs / norm_q << "\n";
         real_t err_ts = t_hs.ComputeL2Error(tcoeff, irs);
         cout << "|| t_hs - t_ex || / || t_ex || = " << err_ts / norm_t << "\n";
      }

      // Project the fluxes

      GridFunction q_vh;

      if (V_space_dg)
      {
         VectorGridFunctionCoefficient coeff(&q_h);
         q_vh.SetSpace(V_space_dg);
         q_vh.ProjectCoefficient(coeff);
      }
      else
      {
         if (rotated)
         {
            DiscreteLinearOperator rot(V_space, V_space);
            rot.AddDomainInterpolator(new RotationInterpolator(bhcoeff));
            rot.Assemble();
            rot.Finalize();
            q_vh.SetSpace(V_space);
            rot.Mult(q_h, q_vh);
         }
         else
         {
            q_vh.MakeRef(V_space, q_h, 0);
         }
      }

      // Project the analytic solution

      static GridFunction q_a, qt_a, t_a, c_gf;

      q_a.SetSpace((V_space_dg)?(V_space_dg):(V_space));
      q_a.ProjectCoefficient(qcoeff);

      qt_a.SetSpace((V_space_dg)?(V_space_dg):(V_space));
      qt_a.ProjectCoefficient(qtcoeff);

      t_a.SetSpace(W_space);
      t_a.ProjectCoefficient(tcoeff);

      if (bconv)
      {
         c_gf.SetSpace((V_space_dg)?(V_space_dg):(V_space));
         c_gf.ProjectCoefficient(ccoeff);
      }

      // 13. Save the mesh and the solution. This output can be viewed later using
      //     GLVis: "glvis -m ex5.mesh -g sol_q.gf" or "glvis -m ex5.mesh -g
      //     sol_t.gf".
      if (mfem)
      {
         stringstream ss;
         ss.str("");
         ss << "ex5";
         if (btime) { ss << "_" << ti; }
         ss << ".mesh";
         ofstream mesh_ofs(ss.str());
         mesh_ofs.precision(8);
         mesh->Print(mesh_ofs);

         ss.str("");
         ss << "sol_q";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream q_ofs(ss.str());
         q_ofs.precision(8);
         q_vh.Save(q_ofs);

         ss.str("");
         ss << "sol_t";
         if (btime) { ss << "_" << ti; }
         ss << ".gf";
         ofstream t_ofs(ss.str());
         t_ofs.precision(8);
         t_h.Save(t_ofs);
      }

      // 14. Save data in the VisIt format
      if (visit)
      {
         static VisItDataCollection visit_dc("Example5", mesh);
         if (ti == 0)
         {
            visit_dc.RegisterField("heat flux", &q_vh);
            visit_dc.RegisterField("temperature", &t_h);
            if (analytic)
            {
               visit_dc.RegisterField("heat flux analytic", &q_a);
               visit_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         visit_dc.SetCycle(ti);
         visit_dc.SetTime(t); // set the time
         visit_dc.Save();
      }

      // 15. Save data in the ParaView format
      if (paraview)
      {
         static ParaViewDataCollection paraview_dc("Example5", mesh);
         if (ti == 0)
         {
            paraview_dc.SetPrefixPath("ParaView");
            paraview_dc.SetLevelsOfDetail(order);
            paraview_dc.SetDataFormat(VTKFormat::BINARY);
            paraview_dc.SetHighOrderOutput(true);
            paraview_dc.RegisterField("heat flux",&q_vh);
            paraview_dc.RegisterField("temperature",&t_h);
            if (analytic)
            {
               paraview_dc.RegisterField("heat flux analytic", &q_a);
               paraview_dc.RegisterField("temperature analytic", &t_a);
            }
         }
         paraview_dc.SetCycle(ti);
         paraview_dc.SetTime(t); // set the time
         paraview_dc.Save();
      }

      // 16. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         const char vishost[] = "localhost";
         const int  visport   = 19916;
         static socketstream q_sock(vishost, visport);
         q_sock.precision(8);
         q_sock << "solution\n" << *mesh << q_vh << endl;
         if (ti == 0)
         {
            q_sock << "window_title 'Heat flux'" << endl;
            q_sock << "keys Rljvvvvvmmc" << endl;
         }
         static socketstream t_sock(vishost, visport);
         t_sock.precision(8);
         t_sock << "solution\n" << *mesh << t_h << endl;
         if (ti == 0)
         {
            t_sock << "window_title 'Temperature'" << endl;
            t_sock << "keys Rljmmc" << endl;
         }
         if (reconstruct)
         {
            static socketstream qt_sock(vishost, visport);
            qt_sock.precision(8);
            qt_sock << "solution\n" << *mesh << qt_h << endl;
            if (ti == 0)
            {
               qt_sock << "window_title 'Total flux'" << endl;
               qt_sock << "keys Rljvvvvvmmc" << endl;
            }
            static socketstream qs_sock(vishost, visport);
            qs_sock.precision(8);
            qs_sock << "solution\n" << *mesh << q_hs << endl;
            if (ti == 0)
            {
               qs_sock << "window_title 'Recon. flux'" << endl;
               qs_sock << "keys Rljvvvvvmmc" << endl;
            }
            static socketstream ts_sock(vishost, visport);
            ts_sock.precision(8);
            ts_sock << "solution\n" << *mesh << t_hs << endl;
            if (ti == 0)
            {
               ts_sock << "window_title 'Recon. temperature'" << endl;
               ts_sock << "keys Rljmmc" << endl;
            }
         }
         if (analytic)
         {
            static socketstream qa_sock(vishost, visport);
            qa_sock.precision(8);
            qa_sock << "solution\n" << *mesh << q_a << endl;
            if (ti == 0)
            {
               qa_sock << "window_title 'Heat flux analytic'" << endl;
               qa_sock << "keys Rljvvvvvmmc" << endl;
            }
            if (bconv || bnlconv)
            {
               static socketstream qta_sock(vishost, visport);
               qta_sock.precision(8);
               qta_sock << "solution\n" << *mesh << qt_a << endl;
               if (ti == 0)
               {
                  qta_sock << "window_title 'Total flux analytic'" << endl;
                  qta_sock << "keys Rljvvvvvmmc" << endl;
               }
            }
            static socketstream ta_sock(vishost, visport);
            ta_sock.precision(8);
            ta_sock << "solution\n" << *mesh << t_a << endl;
            if (ti == 0)
            {
               ta_sock << "window_title 'Temperature analytic'" << endl;
               ta_sock << "keys Rljmmc" << endl;
            }
            if (bconv)
            {
               static socketstream c_sock(vishost, visport);
               c_sock.precision(8);
               c_sock << "solution\n" << *mesh << c_gf << endl;
               if (ti == 0)
               {
                  c_sock << "window_title 'Velocity'" << endl;
                  c_sock << "keys Rljvvvvvmmc" << endl;
               }
            }
         }
      }
   }

   // 17. Free the used memory.

   delete ode_solver;
   delete HeatFluxFun;
   delete FluxFun;
   delete FluxSolver;
   delete fform;
   delete gform;
   delete hform;
   delete darcy;
   delete W_space;
   delete V_space;
   delete V_space_dg;
   delete trace_space;
   delete W_coll;
   delete V_coll;
   delete V_coll_dg;
   delete trace_coll;
   delete mesh;

   return 0;
}

MatFunc GetKFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   const real_t &ka = params.ka;

   auto bFun = GetBFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            kappa.Diag(k, ndim);
            kappa(0,0) *= ks;
            kappa(0,1) = +ka * k;
            kappa(1,0) = -ka * k;
            if (ndim > 2)
            {
               kappa(0,2) = +ka * k;
               kappa(2,0) = -ka * k;
            }
         };
      case Problem::MFEMLogo:
      {
         constexpr int n = 80;
         constexpr real_t xmax = 1.;
         constexpr real_t ymax = 1.;
         constexpr real_t wmax = .05;
         constexpr real_t kmax = .8;
         DenseMatrix bubbles(5, n);
         for (int i = 0; i < n; i++)
         {
            bubbles(0, i) = rand_real() * xmax;
            bubbles(1, i) = rand_real() * ymax;
            bubbles(2, i) = rand_real() * wmax;
            bubbles(3, i) = rand_real() * k * kmax;
            bubbles(4, i) = rand_real() * ks;
            //bubbles(5, i) = rand_real() * ka;
         }

         return [=](const Vector &x, DenseMatrix &kappa)
         {
            real_t kap = 0.;
            real_t kap_s = 0.;
            real_t kap_a = 0.;
            for (int i = 0; i < bubbles.Width(); i++)
            {
               const real_t dx = x(0) - bubbles(0,i);
               const real_t dy = x(1) - bubbles(1,i);
               const real_t w = bubbles(2,i);
               const real_t k = bubbles(3,i) * exp(-(dx*dx+dy*dy)/(w*w));
               kap += k;
               kap_s += k * bubbles(4,i);
               //kap_a += k * bubbles(5, i);
            }
            const int ndim = x.Size();
            const real_t kmin = (1. - kmax) * k;
            kappa.Diag(kmin + kap, ndim);
            kappa(0,0) = kmin + kap_s;
            kappa(0,1) = +kap_a * k;
            kappa(1,0) = -kap_a * k;
            if (ndim > 2)
            {
               kappa(0,2) = +kap_a * k;
               kappa(2,0) = -kap_a * k;
            }
         };
      }
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::Umansky:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            bFun(x, b);

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }
   return MatFunc();
}

VecFunc GetBFun(const ProblemParams &params)
{
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
         return [=](const Vector &x, Vector &b)
         {
            const int ndim = x.Size();
            b.SetSize(ndim);
            b = 0.;
            b(1) = 1.;
         };
      case Problem::MFEMLogo:
         MFEM_ABORT("Not implemented");
         break;
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, Vector &b)
         {
            const int ndim = x.Size();
            b.SetSize(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            b(0) = (r>0.)?(-dx(1) / r):(1.);
            b(1) = (r>0.)?(+dx(0) / r):(0.);
         };
      case Problem::Sovinec:
         return [=](const Vector &x, Vector &b)
         {
            const int ndim = x.Size();
            b.SetSize(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            //const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_x = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_y = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
            const real_t psi_norm = hypot(psi_x, psi_y);
            if (psi_norm > 0.)
            {
               b(0) = -psi_y / psi_norm;
               b(1) = +psi_x / psi_norm;
            }
            else
            {
               b = 0.;
            }
         };
      case Problem::Umansky:
         return [=](const Vector &x, Vector &b)
         {
            const int ndim = x.Size();
            b.SetSize(ndim);
            const real_t s = hypot(sx, sy);
            b(0) = +sx / s;
            b(1) = +sy / s;
         };
   }
   return VecFunc();
}

TFunc GetTFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;
   const real_t hx = sx / params.nx;
   const real_t hy = sy / params.ny;
   const real_t &order = params.order;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            if (a <= 0.) { return t0; }

            Vector ddT((ndim<=2)?(2):(4));
            ddT(0) = -t_0 * M_PI*M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1));//xx,yy
            ddT(1) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * cos(M_PI*x(1));//xy
            if (ndim > 2)
            {
               ddT(0) *= sin(M_PI*x(2));//xx,yy,zz
               ddT(1) *= sin(M_PI*x(2));//xy
               //xz
               ddT(2) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
               //yz
               ddT(3) = +t_0 * M_PI*M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1)) * cos(M_PI*x(2));

            }

            DenseMatrix kappa;
            kFun(x, kappa);

            real_t div = -(kappa(0,0) + kappa(1,1)) * ddT(0) - (kappa(0,1) + kappa(1,0)) * ddT(1);
            if (ndim > 2)
            {
               div += -kappa(2,2) * ddT(0) - (kappa(0,2) + kappa(2,0)) * ddT(2)
               - (kappa(1,2) + kappa(2,1)) * ddT(3);
            }
            return t0 - div / a * t;
         };
      case Problem::MFEMLogo:
         return [=](const Vector &x, real_t t) -> real_t
         {
#if 1
            //Banner
            constexpr int iw = 38;
            constexpr int ih = 7;
            static const unsigned char logo[ih][iw] = {
               "##     ## ######## ######## ##     ##",
               "###   ### ##       ##       ###   ###",
               "#### #### ##       ##       #### ####",
               "## ### ## ######   ######   ## ### ##",
               "##     ## ##       ##       ##     ##",
               "##     ## ##       ##       ##     ##",
               "##     ## ##       ######## ##     ##",
            };
#else
            //Collosal
            constexpr int iw = 50;
            constexpr int ih = 8;
            static const unsigned char logo[ih][iw] = {
               "888b     d888 8888888888 8888888888 888b     d888",
               "8888b   d8888 888        888        8888b   d8888",
               "88888b.d88888 888        888        88888b.d88888",
               "888Y88888P888 8888888    8888888    888Y88888P888",
               "888 Y888P 888 888        888        888 Y888P 888",
               "888  Y8P  888 888        888        888  Y8P  888",
               "888   8   888 888        888        888   8   888",
               "888       888 888        8888888888 888       888",
            };
#endif

            constexpr real_t w = 0.8;
            constexpr real_t h = (w * ih) / iw;
            constexpr real_t xo = 0.5;
            constexpr real_t yo = 0.5;
            const real_t dx = x(0) - xo;
            const real_t dy = x(1) - yo;

            const int ix = (dx/w + 0.5) * iw;
            const int iy = (dy/h + 0.5) * ih;

            if (ix < 0 || ix >= iw || iy < 0 || iy >= ih)
            {
               return 0.;
            }

            const real_t T = (logo[ih-1-iy][ix] != ' ')?(t_0):(0.);
            return T;
         };
      case Problem::DiffusionRing:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.25;
            constexpr real_t r1 = 0.35;
            constexpr real_t dr01 = 0.025;
            constexpr real_t theta0 = 11./12. * M_PI;
            constexpr real_t dtheta0 = 1./48. * M_PI;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t theta = fabs(atan2(dx(1), dx(0)));

            if (r < r0 - dr01 || r > r1 + dr01 || theta < theta0 - dtheta0)
            {
               return 0.;
            }

            const real_t dr = min(r - r0 + dr01, r1 + dr01 - r) / dr01;
            const real_t dth = (theta - theta0 + dtheta0) / dtheta0;
            return min(1., dr) * min(1., dth) * t_0;
         };
      case Problem::DiffusionRingGauss:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.025;
            constexpr real_t x_c = 0.15;

            const real_t dx_l = x(0) - (x0       + x_c  * sx);
            const real_t dx_r = x(0) - (x0 + (1. - x_c) * sx);
            const real_t dy = x(1) - (y0 + 0.5*sy);
            const real_t r_l = hypot(dx_l, dy);
            const real_t r_r = hypot(dx_r, dy);

            return - exp(- r_l*r_l/(r0*r0)) + exp(- r_r*r_r/(r0*r0));
         };
      case Problem::DiffusionRingSine:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { return 0.; }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            return 1. / (1. + t * k * C*C / a) * cos(w0*th) * sin(M_PI * r/r0);
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t k_para = M_PI*M_PI * k * ks;
            const real_t k_perp = k;
            const real_t k_frac = sqrt(k_para/k_perp);
            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            return - (e_down + e_up) / denom * sin(M_PI * x(0));
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATIONMETHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            return x(0)*x(1) * pow(arg, s);
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATIONMETHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            return 1. - r*r*r;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t &kappa_perp = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return psi / kappa_perp;
         };
      case Problem::Umansky:
         // M. V. Umansky, M. S. Day and T. D. Rognlien, On Numerical Solution
         // of Strongly Anisotropic Diffusion Equation on Misaligned Grids,
         // Numerical Heat Transfer, Part B: Fundamentals, 47(6), pp. 533-554
         // (2005).
         // Adopted from plasma-dev:miniapps/plasma/transport2d.cpp
         return [=](const Vector &x, real_t t) -> real_t
         {
            if (x(0) < hx && x(1) < hy)
            {
               return 0.5 * (1.0 -
                             std::pow(1.0 - x(0) / hx, order) +
                             std::pow(1.0 - x(1) / hy, order));
            }
            else if (x(0) > sx - hx && x(1) > sy - hy)
            {
               return 0.5 * (1.0 +
                             std::pow(1.0 - (sx - x(0)) / hx, order) -
                             std::pow(1.0 - (sy - x(1)) / hy, order));
            }
            // else if (x_[0] > Lx_ - hx_ || x_[1] < hy_)
            else if (hx * (x(1) + hy) < hy * x(0))
            {
               return 1.0;
            }
            // else if (x_[0] < hx_ || x_[1] > Ly_ - hy_)
            else if (hx * x(1) > hy * (x(0) + hx))
            {
               return 0.0;
            }
            else
            {
               return 0.5 * (1.0 + std::tanh(M_LN10 * (x(0) / hx - x(1) / hy)));
            }
         };
   }
   return TFunc();
}

VecTFunc GetQFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector gT(vdim);
            gT = 0.;
            gT(0) = t_0 * M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1));
            gT(1) = t_0 * M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1));
            if (vdim > 2)
            {
               gT(0) *= sin(M_PI*x(2));
               gT(1) *= sin(M_PI*x(2));
               gT(2) = t_0 * M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
            }

            DenseMatrix kappa;
            kFun(x, kappa);

            if (vdim <= 2)
            {
               v(0) = -kappa(0,0) * gT(0) -kappa(0,1) * gT(1);
               v(1) = -kappa(1,0) * gT(0) -kappa(1,1) * gT(1);
            }
            else
            {
               kappa.Mult(gT, v);
               v.Neg();
            }
         };
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::Umansky:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
         };
      case Problem::DiffusionRingSine:
         return [=](const Vector &x, real_t t, Vector &v)
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { v = 0.; return;  }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            const real_t T_r = -C / (1. + t * k * C*C / a) * sin(w0*th)
                               * sin(M_PI * r/r0);
            v(0) = + k * T_r * sin(th);
            v(1) = - k * T_r * cos(th);
         };
      case Problem::BoundaryLayer:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            const real_t k_para = M_PI*M_PI * kappa(0,0);
            const real_t k_perp = kappa(1,1);
            const real_t k_frac = sqrt(k_para/k_perp);

            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            const real_t T_x = - (e_down + e_up) / denom * M_PI * cos(M_PI * x(0));
            const real_t T_y = k_frac * (e_down - e_up) / denom * sin(M_PI * x(0));
            v(0) = -kappa(0,0) * T_x;
            v(1) = -kappa(1,1) * T_y;
         };
      case Problem::SteadyPeak:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_x = x(1) * pow(arg, s-1) * (arg + x(0) * s * arg_x);
            const real_t T_y = x(0) * pow(arg, s-1) * (arg + x(1) * s * arg_y);
            v(0) = -kappa(0,0) * T_x;
            v(1) = -kappa(1,1) * T_y;
         };
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            const real_t kappa_r = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_r = - 3. * r;
            v(0) = -kappa_r * T_r * dx(0);
            v(1) = -kappa_r * T_r * dx(1);
         };
      case Problem::Sovinec:
         return [=](const Vector &x, real_t, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            v(0) = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            v(1) = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
         };
   }
   return VecTFunc();
}

VecFunc GetCFun(const ProblemParams &params)
{
   const real_t &c = params.c;

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::Umansky:
         // null
         break;
      case Problem::MFEMLogo:
      {
         constexpr int n = 80;
         constexpr real_t xmax = 1.;
         constexpr real_t ymax = 1.;
         constexpr real_t wmax = .05;
         DenseMatrix bubbles(4, n);
         for (int i = 0; i < n; i++)
         {
            bubbles(0, i) = rand_real() * xmax;
            bubbles(1, i) = rand_real() * ymax;
            bubbles(2, i) = rand_real() * wmax;
            bubbles(3, i) = (rand_real() * 2. - 1.) * c;
         }

         return [=](const Vector &x, Vector &v)
         {
            const int vdim = x.Size();
            v.SetSize(vdim);
            v = 0.;
            for (int i = 0; i < bubbles.Width(); i++)
            {
               const real_t dx = x(0) - bubbles(0,i);
               const real_t dy = x(1) - bubbles(1,i);
               const real_t w = bubbles(2,i);
               const real_t c = bubbles(3,i) * exp(-(dx*dx+dy*dy)/(w*w));
               v(0) += +c * dy;
               v(1) += -c * dx;
            }
         };
      }
   }
   return VecFunc();
}

TFunc GetFFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto TFun = GetTFun(params);
   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t T = TFun(x, 0);
            return -((a > 0.)?(a):(1.)) * T;
         };
      case Problem::BoundaryLayer:
      case Problem::Umansky:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::SteadyPeak:
         return [=](const Vector &x, real_t) -> real_t
         {
            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_xx = x(1) * pow(arg, s-2) * (2.*s * arg_x * arg + x(0) * s * ((s-1) * arg_x*arg_x - M_PI*M_PI * arg*arg));
            const real_t T_yy = x(0) * pow(arg, s-2) * (2.*s * arg_y * arg + x(1) * s * ((s-1) * arg_y*arg_y - M_PI*M_PI * arg*arg));
            return kappa(0,0) * T_xx + kappa(1,1) * T_yy;
         };
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t kappa_r = ks * k;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_rr = - 9. * r;
            return kappa_r * T_rr;
         };
      case Problem::Sovinec:
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return -2.*M_PI*M_PI * psi;
         };
   }
   return TFunc();
}

FluxFunction* GetFluxFun(const ProblemParams &params, VectorCoefficient &ccoef)
{
   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::Umansky:
         //null
         break;
   }

   return NULL;
}

MixedFluxFunction* GetHeatFluxFun(const ProblemParams &params, int dim)
{
   auto KFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
      case Problem::SteadyVaryingAngle:
      case Problem::Sovinec:
      case Problem::Umansky:
         static MatrixFunctionCoefficient kappa(dim, KFun);
         static InverseMatrixCoefficient ikappa(kappa);
         return new LinearDiffusionFlux(ikappa);
   }

   return NULL;
}

real_t FindYVal(const GridFunction &u, real_t u_target, real_t x,
                real_t y0, real_t y1)
{
   // Adopted from plasma-dev:miniapps/plasma/transport2d.cpp
   Mesh *mesh = u.FESpace()->GetMesh();

   Array<int> elem;
   Array<IntegrationPoint> ips;
   DenseMatrix point_mat(2, 1);
   point_mat(0,0) = x;

   real_t tol = 1e-5;

   real_t a = y0;
   real_t b = y1;

   real_t ua = 1.0 - u_target;
   //real_t ub = 0.0 - u_target;
   real_t uc = 0.5 - u_target;

   const int nmax = 20;
   int n = 0;
   while (n < nmax)
   {
      real_t c = 0.5 * (a + b);

      point_mat(1,0) = c;
      int nfound = mesh->FindPoints(point_mat, elem, ips);

      if (nfound != 1)
      {
         MFEM_ABORT("Point (" << x << ", " << c << ") not found");
      }

      if (elem[0] >= 0)
      {
         uc = u.GetValue(elem[0], ips[0]) - u_target;
      }
      else
      {
         uc = -infinity();
      }

      if (std::abs(uc) < tol || 0.5 * (b - a) < tol)
      {
         return c;
      }

      if (ua * uc < 0.0)
      {
         b = c;
         //ub = uc;
      }
      else
      {
         a = c;
         ua = uc;
      }

      n++;
   }
   return -1.0;
}

real_t UmanskyTestWidth(const GridFunction &u)
{
   // Adopted from plasma-dev:miniapps/plasma/transport2d.cpp
   Mesh *mesh = u.FESpace()->GetMesh();

   Vector min, max;
   mesh->GetBoundingBox(min,max);

   double xMid = 0.5 * (max[0] + min[0]);
   double y0 = min[1];
   double y1 = max[1];

   double y25 = FindYVal(u, 0.25, xMid, y0, y1);
   double y75 = FindYVal(u, 0.75, xMid, y0, y1);

   return y25 - y75;
}

void RotationInterpolator::AssembleElementMatrix(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &elmat)
{
   const int vdim = Trans.GetSpaceDim();
   MFEM_ASSERT(vdim == 2, "Not implemented");
   const IntegrationRule &nodes = fe.GetNodes();
   const int ndof = fe.GetDof();

   Vector b;
   b_vcoeff.Eval(b_vals, Trans, nodes);

   elmat.SetSize(ndof * vdim);

   for (int i = 0; i < nodes.Size(); i++)
   {
      const IntegrationPoint &ip = nodes[i];
      b_vals.GetColumnReference(i, b);
      Trans.SetIntPoint(&ip);

      for (int d = 0; d < vdim; d++)
      {
         elmat(i + d * ndof, i) = b(d);
      }

      //2D
      elmat(i + 0 * ndof, i + ndof) = -b(1);
      elmat(i + 1 * ndof, i + ndof) = +b(0);
   }
}

void VectorRotatedWeakDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   const int tr_dof = trial_fe.GetDof();
   const int te_dof = test_fe.GetDof();
   const int dim = test_fe.GetDim();
   const int vdim = Trans.GetSpaceDim();
   MFEM_ASSERT(vdim == 2, "Not implemented");

   te_dshape.SetSize(te_dof, dim);
   te_dshapext.SetSize(te_dof, vdim);
   te_bgshape.SetSize(te_dof);
   te_curlshape.SetSize(te_dof * vdim, 1);
   tr_shape.SetSize(tr_dof);

   Vector b_q(vdim);

   elmat.SetSize(te_dof, tr_dof * vdim);
   elmat = 0.;

   const IntegrationRule &ir = *GetIntegrationRule(trial_fe, test_fe, Trans);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Trans.SetIntPoint(&ip);
      trial_fe.CalcShape(ip, tr_shape);
      test_fe.CalcDShape(ip, te_dshape);
      mfem::Mult(te_dshape, Trans.AdjugateJacobian(), te_dshapext);

      b_vcoeff.Eval(b_q, Trans, ip);

      const real_t w = -ip.weight;

      //b-grad
      te_dshapext.Mult(b_q, te_bgshape);

      for (int j = 0; j < tr_dof; j++)
         for (int i = 0; i < te_dof; i++)
         {
            elmat(i,j) += w * tr_shape(j) * te_bgshape(i);
         }

      //b-curl
      for (int j = 0; j < tr_dof; j++)
         for (int i = 0; i < te_dof; i++)
         {
            const real_t curl = -te_dshapext(i, 0) * b_q(1) + te_dshapext(i, 1) * b_q(0);
            elmat(i,j + tr_dof) += w * tr_shape(j) * curl;
         }
   }
}

void VectorRotatedDivergenceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   const int tr_dof = trial_fe.GetDof();
   const int te_dof = test_fe.GetDof();
   const int dim = trial_fe.GetDim();
   const int vdim = Trans.GetSpaceDim();
   MFEM_ASSERT(vdim == 2, "Not implemented");

   tr_dshape.SetSize(tr_dof, dim);
   tr_dshapext.SetSize(tr_dof, vdim);
   tr_divshape.SetSize(tr_dof*vdim);
   te_shape.SetSize(te_dof);

   Vector b_q(vdim);

   elmat.SetSize(te_dof, tr_dof * vdim);
   elmat = 0.;

   b_vals.SetSize(tr_dof, vdim);
   Vector b_valsv(b_vals.GetData(), tr_dof*vdim);
   b_gf.GetElementDofValues(Trans.ElementNo, b_valsv);

   const IntegrationRule &ir = *GetIntegrationRule(trial_fe, test_fe, Trans);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Trans.SetIntPoint(&ip);
      trial_fe.CalcDShape(ip, tr_dshape);
      test_fe.CalcShape(ip, te_shape);
      mfem::Mult(tr_dshape, Trans.AdjugateJacobian(), tr_dshapext);

      const real_t w = ip.weight;

      for (int i = 0; i < tr_dof; i++)
      {
         tr_divshape(i         ) = +b_vals(i,0) * tr_dshapext(i, 0) + b_vals(i,
                                                                             1) * tr_dshapext(i, 1);
         tr_divshape(i + tr_dof) = -b_vals(i,1) * tr_dshapext(i, 0) + b_vals(i,
                                                                             0) * tr_dshapext(i, 1);
      }

      AddMult_a_VWt(w, te_shape, tr_divshape, elmat);
   }
}

void DGRotatedWeakNormalTraceIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_fe1, const FiniteElement &test_fe1,
   const FiniteElement &trial_fe2, const FiniteElement &test_fe2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2;

   double un, a, b, w;

   const int dim = test_fe1.GetDim();
   tr_ndof1 = trial_fe1.GetDof();
   te_ndof1 = test_fe1.GetDof();
   Vector vu(dim), nor(dim), bvec(dim), bnor1(dim), bnor2(dim);

   if (Trans.Elem2No >= 0)
   {
      tr_ndof2 = trial_fe2.GetDof();
      te_ndof2 = test_fe2.GetDof();
   }
   else
   {
      tr_ndof2 = 0;
      te_ndof2 = 0;
   }

   tr_shape1.SetSize(tr_ndof1);
   te_shape1.SetSize(te_ndof1);
   tr_shape2.SetSize(tr_ndof2);
   te_shape2.SetSize(te_ndof2);
   elmat.SetSize(te_ndof1 + te_ndof2, (tr_ndof1 + tr_ndof2)*dim);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      Trans.Elem1->SetIntPoint(&eip1);
      if (tr_ndof2 && te_ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
         Trans.Elem2->SetIntPoint(&eip2);
      }
      trial_fe1.CalcPhysShape(*Trans.Elem1, tr_shape1);

      Trans.Face->SetIntPoint(&ip);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      b_vcoeff.Eval(bvec, *Trans.Elem1, eip1);
      bnor1(0) = nor * bvec;
      bnor1(1) = -bvec(1) * nor(0) + bvec(0) * nor(1);

      if (tr_ndof2)
      {
         Trans.Elem2->SetIntPoint(&eip2);
         b_vcoeff.Eval(bvec, *Trans.Elem2, eip2);
         bnor2(0) = nor * bvec;
         bnor2(1) = -bvec(1) * nor(0) + bvec(0) * nor(1);
      }

      test_fe1.CalcPhysShape(*Trans.Elem1, te_shape1);

      a = ((tr_ndof2)?(0.5):(1.)) * alpha;
      if (beta != 0.)
      {
         u->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
         b = (un != 0.)?(beta * un / fabs(un)):(0.);
      }
      else
      {
         b = 0.;
      }

      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (rho)
      {
         double rho_p;
         if (un >= 0.0 && tr_ndof2 && te_ndof2)
         {
            Trans.Elem2->SetIntPoint(&eip2);
            rho_p = rho->Eval(*Trans.Elem2, eip2);
         }
         else
         {
            rho_p = rho->Eval(*Trans.Elem1, eip1);
         }
         a *= rho_p;
         b *= rho_p;
      }

      w = ip.weight * (a+b);
      if (w != 0.0)
      {
         for (int d = 0; d < dim; d++)
            for (int i = 0; i < te_ndof1; i++)
               for (int j = 0; j < tr_ndof1; j++)
               {
                  elmat(i, j + d*tr_ndof1) += w * te_shape1(i) * tr_shape1(j) * bnor1(d);
               }
      }

      if (tr_ndof2 && te_ndof2)
      {
         trial_fe2.CalcPhysShape(*Trans.Elem2, tr_shape2);
         test_fe2.CalcPhysShape(*Trans.Elem2, te_shape2);

         if (w != 0.0)
         {
            for (int d = 0; d < dim; d++)
               for (int i = 0; i < te_ndof2; i++)
                  for (int j = 0; j < tr_ndof1; j++)
                  {
                     elmat(te_ndof1 + i, j + d*tr_ndof1) -=
                        w * te_shape2(i) * tr_shape1(j) * bnor1(d);
                  }
         }

         w = ip.weight * (b-a);
         if (w != 0.0)
         {
            for (int d = 0; d < dim; d++)
               for (int i = 0; i < te_ndof2; i++)
                  for (int j = 0; j < tr_ndof2; j++)
                  {
                     elmat(te_ndof1 + i, tr_ndof1*dim + j + d*tr_ndof2) +=
                        w * te_shape2(i) * tr_shape2(j) * bnor2(d);
                  }

            for (int d = 0; d < dim; d++)
               for (int i = 0; i < te_ndof1; i++)
                  for (int j = 0; j < tr_ndof2; j++)
                  {
                     elmat(i, tr_ndof1*dim + j + d*tr_ndof2) -=
                        w * te_shape1(i) * tr_shape2(j) * bnor2(d);
                  }
         }
      }
   }
}

void DGRotatedNormalTraceIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_fe1, const FiniteElement &test_fe1,
   const FiniteElement &trial_fe2, const FiniteElement &test_fe2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2;

   double un, a, b, w;

   const int dim = test_fe1.GetDim();
   tr_ndof1 = trial_fe1.GetDof();
   te_ndof1 = test_fe1.GetDof();
   Vector vu(dim), nor(dim), bvec(dim), bnor1(dim), bnor2(dim);

   if (Trans.Elem2No >= 0)
   {
      tr_ndof2 = trial_fe2.GetDof();
      te_ndof2 = test_fe2.GetDof();
   }
   else
   {
      tr_ndof2 = 0;
      te_ndof2 = 0;
   }

   tr_shape1.SetSize(tr_ndof1);
   te_shape1.SetSize(te_ndof1);
   tr_shape2.SetSize(tr_ndof2);
   te_shape2.SetSize(te_ndof2);
   elmat.SetSize((te_ndof1 + te_ndof2)*dim, tr_ndof1 + tr_ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      Trans.Elem1->SetIntPoint(&eip1);
      if (tr_ndof2 && te_ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
         Trans.Elem2->SetIntPoint(&eip2);
      }
      trial_fe1.CalcPhysShape(*Trans.Elem1, tr_shape1);

      Trans.Face->SetIntPoint(&ip);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      b_vcoeff.Eval(bvec, *Trans.Elem1, eip1);
      bnor1(0) = nor * bvec;
      bnor1(1) = -bvec(1) * nor(0) + bvec(0) * nor(1);

      if (tr_ndof2)
      {
         Trans.Elem2->SetIntPoint(&eip2);
         b_vcoeff.Eval(bvec, *Trans.Elem2, eip2);
         bnor2(0) = nor * bvec;
         bnor2(1) = -bvec(1) * nor(0) + bvec(0) * nor(1);
      }

      test_fe1.CalcPhysShape(*Trans.Elem1, te_shape1);

      a = 0.5 * alpha;
      if (beta != 0.)
      {
         u->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
         b = (un != 0.)?(beta * un / fabs(un)):(0.);
      }
      else
      {
         b = 0.;
      }

      // note: if |alpha/2|==|beta| then |a|==|b|, i.e. (a==b) or (a==-b)
      //       and therefore two blocks in the element matrix contribution
      //       (from the current quadrature point) are 0

      if (rho)
      {
         double rho_p;
         if (un >= 0.0 && tr_ndof2 && te_ndof2)
         {
            Trans.Elem2->SetIntPoint(&eip2);
            rho_p = rho->Eval(*Trans.Elem2, eip2);
         }
         else
         {
            rho_p = rho->Eval(*Trans.Elem1, eip1);
         }
         a *= rho_p;
         b *= rho_p;
      }

      w = ip.weight * (a+b);
      if (w != 0.0)
      {
         for (int d = 0; d < dim; d++)
            for (int i = 0; i < te_ndof1; i++)
               for (int j = 0; j < tr_ndof1; j++)
               {
                  elmat(i + d*te_ndof1, j) += w * te_shape1(i) * tr_shape1(j) * bnor1(d);
               }
      }

      if (tr_ndof2 && te_ndof2)
      {
         trial_fe2.CalcPhysShape(*Trans.Elem2, tr_shape2);
         test_fe2.CalcPhysShape(*Trans.Elem2, te_shape2);

         if (w != 0.0)
         {
            for (int d = 0; d < dim; d++)
               for (int i = 0; i < te_ndof2; i++)
                  for (int j = 0; j < tr_ndof1; j++)
                  {
                     elmat(te_ndof1*dim + i + d*te_ndof2, j) -=
                        w * te_shape2(i) * tr_shape1(j) * bnor2(d);
                  }
         }

         w = ip.weight * (b-a);
         if (w != 0.0)
         {
            for (int d = 0; d < dim; d++)
               for (int i = 0; i < te_ndof2; i++)
                  for (int j = 0; j < tr_ndof2; j++)
                  {
                     elmat(te_ndof1*dim + i + d*te_ndof2, tr_ndof1 + j) +=
                        w * te_shape2(i) * tr_shape2(j) * bnor2(d);
                  }

            for (int d = 0; d < dim; d++)
               for (int i = 0; i < te_ndof1; i++)
                  for (int j = 0; j < tr_ndof2; j++)
                  {
                     elmat(i + d*te_ndof1, tr_ndof1 + j) -=
                        w * te_shape1(i) * tr_shape2(j) * bnor1(d);
                  }
         }
      }
   }
}

void RotatedNormalTraceJumpIntegrator::AssembleFaceMatrix(
   const FiniteElement &trial_face_fe, const FiniteElement &test_fe1,
   const FiniteElement &test_fe2, FaceElementTransformations &Trans,
   DenseMatrix &elmat)
{
   int i, j, face_ndof, ndof1, ndof2, dim;
   int order;

   MFEM_VERIFY(trial_face_fe.GetMapType() == FiniteElement::VALUE, "");

   face_ndof = trial_face_fe.GetDof();
   ndof1 = test_fe1.GetDof();
   dim = test_fe1.GetDim();

   face_shape.SetSize(face_ndof);
   normal.SetSize(dim);
   shape1.SetSize(ndof1,dim);
   shape1_n.SetSize(ndof1);

   if (Trans.Elem2No >= 0)
   {
      ndof2 = test_fe2.GetDof();
      shape2.SetSize(ndof2,dim);
      shape2_n.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }

   Vector bvec(dim), bnor(dim);

   elmat.SetSize((ndof1 + ndof2) * dim, face_ndof);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      if (Trans.Elem2No >= 0)
      {
         order = max(test_fe1.GetOrder(), test_fe2.GetOrder());
      }
      else
      {
         order = test_fe1.GetOrder();
      }
      order += trial_face_fe.GetOrder() + Trans.OrderW();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      // Trace finite element shape function
      trial_face_fe.CalcShape(ip, face_shape);
      Trans.SetIntPoint(&ip);
      CalcOrtho(Trans.Jacobian(), normal);

      b_vcoeff.Eval(bvec, *Trans.Elem1, Trans.GetElement1IntPoint());
      bnor(0) = normal * bvec;
      bnor(1) = -bvec(1) * normal(0) + bvec(0) * normal(1);

      // Side 1 finite element shape function
      test_fe1.CalcPhysShape(*Trans.Elem1, shape1_n);
      face_shape *= ip.weight * sign;
      for (int d = 0; d < dim; d++)
         for (i = 0; i < ndof1; i++)
            for (j = 0; j < face_ndof; j++)
            {
               elmat(i+d*ndof1, j) += shape1_n(i) * face_shape(j) * bnor(d);
            }
      if (ndof2)
      {
         b_vcoeff.Eval(bvec, *Trans.Elem2, Trans.GetElement2IntPoint());
         bnor(0) = normal * bvec;
         bnor(1) = -bvec(1) * normal(0) + bvec(0) * normal(1);

         // Side 2 finite element shape function
         test_fe2.CalcPhysShape(*Trans.Elem2, shape2_n);
         // Subtract contribution from side 2
         for (int d = 0; d < dim; d++)
            for (i = 0; i < ndof2; i++)
               for (j = 0; j < face_ndof; j++)
               {
                  elmat(ndof1*dim+i+d*ndof2, j) -= shape2_n(i) * face_shape(j) * bnor(d);
               }
      }
   }
}
