// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
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
//            -----------------------------------------------------
//                     Fourier Miniapp:  Thermal Diffusion
//            -----------------------------------------------------
//
// This miniapp solves a time dependent heat equation.
//

#include "adv_diff_v1_solver.hpp"
#include <cassert>
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::thermal;

void display_banner(ostream & os);

//static int prob_ = 1;
//static int gamma_ = 10;
static double alpha_   = NAN;
static double chiPara_ = 1.0;
static double chiPerp_ = 1.0;

static double nu_      = NAN;
static double vMag_    = 0.0;

static double TInf_    = 1.0;
static double TMax_    = 5.0;
static double TWPara_  = 0.125;
static double TWPerp_  = 0.125;

static vector<Vector> aVec_(2);
/*
double QFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
      {
         return 2.0 * M_PI * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
      }
      case 2:
      case 4:
      {
         double a = 0.4;
         double b = 0.8;

         double r = pow(x[0] / a, 2) + pow(x[1] / b, 2);
         double e = exp(-0.25 * t * M_PI * M_PI / (a * b) );

         if ( r == 0.0 )
            return 0.25 * M_PI * M_PI *
                   ( (1.0 - e) * ( pow(a, -2) + pow(b, -2) ) + e / (a * b));

         return ( M_PI / r ) *
                ( 0.25 * M_PI * pow(a * b, -4) *
                  ( pow(b * b * x[0],2) + pow(a * a * x[1], 2) +
                    (a - b) * (b * pow(b * x[0], 2) - a * pow(a*x[1],2)) * e) *
                  cos(0.5 * M_PI * sqrt(r)) +
                  0.5 * pow(a * b, -2) * (x * x) * (1.0 - e) *
                  sin(0.5 * M_PI * sqrt(r)) / sqrt(r)
                );

      }
      case 3:
      {
         double cx = cos(M_PI * (x[0]-0.5));
         double cy = cos(M_PI * (x[1]-0.5));
         double c2x = cos(2.0 * M_PI * (x[0]-0.5));
         double s2x = sin(2.0 * M_PI * (x[0]-0.5));
         double c2y = cos(2.0 * M_PI * (x[1]-0.5));
         double s2y = sin(2.0 * M_PI * (x[1]-0.5));
         double c2a = cos(2.0 * alpha_);
         double s2a = sin(2.0 * alpha_);
         double ccg = 0.5 * M_PI * M_PI * gamma_ * pow(cx * cy, gamma_ - 2);
         double perp = 1.0 * gamma_ * (c2x * c2y - 1.0) + c2x + c2y + 2.0;
         double para = 0.5 * (gamma_ * (c2x * c2y - s2a * s2x * s2y - 1.0) +
                              (gamma_ - 1.0) * c2a * (c2x - c2y) +
                              c2x + c2y + 2.0);
         return ccg * (1.0 * perp + (chi_max_ratio_ - 1.0) * para);
      }

   }
}
*/
//static double chi_ratio_ = 1.0;

double T0Func(const Vector &x, double t)
{
   double ca = cos(alpha_);
   double sa = sin(alpha_);

   double r2Para = pow(ca * x[0] + sa * x[1], 2);
   double r2Perp = pow(ca * x[1] - sa * x[0], 2);

   double dPara = 4.0 * chiPara_ * t + pow(TWPara_, 2) / M_LN2;
   double dPerp = 4.0 * chiPerp_ * t + pow(TWPerp_, 2) / M_LN2;

   double e = exp(-r2Para / dPara - r2Perp / dPerp);

   double sPara = sqrt(1.0 + 4.0 * M_LN2 * chiPara_ * t / pow(TWPara_, 2));
   double sPerp = sqrt(1.0 + 4.0 * M_LN2 * chiPerp_ * t / pow(TWPerp_, 2));

   return (TMax_ - TInf_) * e / (sPara * sPerp);
}

double TFunc(const Vector &x, double t)
{
   double ct[2];
   Vector ctVec(ct, 2);
   ctVec[0] = -vMag_ * cos(nu_) * t;
   ctVec[1] = -vMag_ * sin(nu_) * t;

   double tol = 1e-12;

   double dPara = 4.0 * chiPara_ * t + pow(TWPara_, 2) / M_LN2;
   double spread = sqrt(fabs(dPara * log(tol)));

   double xt[2];
   Vector xtVec(xt, 2);

   double T = TInf_;

   int si = (int)ceil(0.5 * spread);
   for (int i=-si; i<=si; i++)
   {
      for (int j=-si; j<=si; j++)
      {
         xtVec = x;
         xtVec.Add(1.0, ctVec);
         xtVec.Add(i, aVec_[0]);
         xtVec.Add(j, aVec_[1]);
         T += T0Func(xtVec, t);
      }
   }

   return T;
}

void dTFunc(const Vector &x, double t, Vector &dT)
{
   dT.SetSize(x.Size());
   dT = 0.0;

   double ca = cos(alpha_);
   double sa = sin(alpha_);

   double xt = x[0] - vMag_ * cos(nu_) * t;
   double yt = x[1] - vMag_ * sin(nu_) * t;

   double r2Para = pow(ca * xt + sa * yt, 2);
   double r2Perp = pow(ca * yt - sa * xt, 2);

   double dPara = 4.0 * chiPara_ * t + pow(TWPara_, 2) / M_LN2;
   double dPerp = 4.0 * chiPerp_ * t + pow(TWPerp_, 2) / M_LN2;

   double e = exp(-r2Para / dPara - r2Perp / dPerp);

   double sPara = sqrt(1.0 + 4.0 * M_LN2 * chiPara_ * t / pow(TWPara_, 2));
   double sPerp = sqrt(1.0 + 4.0 * M_LN2 * chiPerp_ * t / pow(TWPerp_, 2));

   dT[0] = (ca * (ca * xt + sa * yt) / dPara -
            sa * (ca * yt - sa * xt) / dPerp);
   dT[1] = (sa * (ca * xt + sa * yt) / dPara +
            ca * (ca * yt - sa * xt) / dPerp);
   dT *= -2.0 * (TMax_ - TInf_) * e / (sPara * sPerp);
}

void ChiFunc(const Vector &x, DenseMatrix &M)
{
   M.SetSize(2);

   double ca = cos(alpha_);
   double sa = sin(alpha_);
   double ca2 = ca * ca;
   double sa2 = sa * sa;
   double saca = sa * ca;

   M(0,0) = chiPerp_ * sa2 + chiPara_ * ca2;
   M(1,1) = chiPerp_ * ca2 + chiPara_ * sa2;

   M(0,1) = (chiPara_ - chiPerp_) * saca;
   M(1,0) = M(0,1);
}

void bbTFunc(const Vector &x, DenseMatrix &M)
{
   M.SetSize(2);

   double ca = cos(alpha_);
   double sa = sin(alpha_);
   double ca2 = ca * ca;
   double sa2 = sa * sa;
   double saca = sa * ca;

   M(0,0) = ca2;
   M(1,1) = sa2;

   M(0,1) = saca;
   M(1,0) = M(0,1);
}

class ChiGridFuncCoef : public MatrixCoefficient
{
private:
   GridFunction * T_;

public:
   ChiGridFuncCoef(GridFunction & T) : MatrixCoefficient(2), T_(&T) {}

   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip);
};

void qFunc(const Vector &x, double t, Vector &q)
{
   DenseMatrix Chi(x.Size());
   Vector dT(x.Size());

   dTFunc(x, t, dT);
   ChiFunc(x, Chi);

   Chi.Mult(dT, q);
   q *= -1.0;
}
/*
long int factorial(unsigned int n)
{
   long int fact = 1;
   for (unsigned int i=2; i<=n; i++)
   {
      fact *= i;
   }
   return fact;
}

// Returns the Gamma(n) function for a positive integer n
long int gamma(unsigned int n)
{
   assert(n > 0);
   return factorial(n-1);
}

// Returns Gamma(n+1/2) for a positive integer n
double gamma1_2(unsigned int n)
{
   return sqrt(M_PI) * factorial(2*n) / (pow(4, n) * factorial(n));
}

double TNorm()
{
   switch (prob_)
   {
      case 1:
         return 0.5;
      case 2:
         return (gamma1_2((unsigned int)gamma_) /
                 gamma((unsigned int)gamma_+1)) / sqrt(M_PI);
   }
}

double qPerpNorm()
{
   switch (prob_)
   {
      case 1:
         return M_PI * M_SQRT1_2 * chiPara_ / chiPerp_;
      case 3:
         return sqrt(M_PI * gamma_) * M_SQRT1_2 *
                sqrt(gamma1_2((unsigned int)gamma_-1) *
                     gamma1_2((unsigned int)gamma_)) /
                sqrt(gamma((unsigned int)gamma_) * gamma((unsigned int)gamma_+1));
   }
}

double qParaNorm()
{
   switch (prob_)
   {
      case 1:
         return 0.0;
      case 3:
   return (chiPara_ / chiPerp_) * qPerpNorm();
   }
}
*/

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   // print the cool banner
   if (mpi.Root()) { display_banner(cout); }

   // 2. Parse command-line options.
   int n = -1;
   int order = 1;
   int irOrder = -1;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int ode_split_solver_type = 1;
   int ode_exp_solver_type = -1;
   int ode_imp_solver_type = -1;
   int vis_steps = 1;
   double dt = 0.5;
   double t_final = 5.0;
   double tol = 1e-4;
   const char *basename = "Adv_Diff_v1";
   const char *mesh_file = "../../data/periodic-square.mesh";
   bool zero_start = false;
   bool static_cond = false;
   bool gfprint = true;
   bool visit = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&irOrder, "-iro", "--int-rule-order",
                  "Integration Rule Order.");
   args.AddOption(&alpha_, "-alpha", "--constant-angle",
                  "Angle for constant B field (in degrees)");
   args.AddOption(&chiPara_, "-chi-para", "--chi-parallel",
                  "Chi parallel to B.");
   args.AddOption(&chiPerp_, "-chi-perp", "--chi-perpendicular",
                  "Chi perpendicular to B.");
   args.AddOption(&nu_, "-nu", "--velocity-angle",
                  "Advection direction (in degrees)");
   args.AddOption(&vMag_, "-v", "--velocity",
                  "Advection speed.");
   args.AddOption(&TInf_, "-T-inf", "--T-at-infinity",
                  "T(0) far from center.");
   args.AddOption(&TMax_, "-T-max", "--T-maximum",
                  "T(0) at origin.");
   args.AddOption(&TWPara_, "-T-w-para", "--T-width-parallel",
                  "Width of T(0) parallel to B.");
   args.AddOption(&TWPerp_, "-T-w-perp", "--T-width-perpendicular",
                  "Width of T(0) perpendicular to B.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&t_final, "-tf", "--final-time",
                  "Final Time.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Tolerance used to determine convergence to steady state.");
   args.AddOption(&ode_split_solver_type, "-ss", "--ode-split-solver",
                  "ODE Split solver:\n"
                  "            1 - First Order Fractional Step,\n"
                  "            2 - Strang Splitting (2nd Order).");
   args.AddOption(&ode_exp_solver_type, "-se", "--ode-exp-solver",
                  "ODE Explicit solver:\n"
                  "            1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&ode_imp_solver_type, "-si", "--ode-imp-solver",
                  "ODE Implicit solver: L-stable methods\n\t"
                  "            1 - Backward Euler,\n\t"
                  "            2 - SDIRK23, 3 - SDIRK33,\n\t"
                  "            A-stable methods (not L-stable)\n\t"
                  "            22 - ImplicitMidPointSolver,\n\t"
                  "            23 - SDIRK23, 34 - SDIRK34.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&gfprint, "-print", "--print","-no-print","--no-print",
                  "Print results (grid functions) to disk.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (ode_exp_solver_type < 0)
   {
      ode_exp_solver_type = ode_split_solver_type;
   }
   if (ode_imp_solver_type < 0)
   {
      ode_imp_solver_type = ode_split_solver_type;
   }

   if (irOrder < 0)
   {
      irOrder = std::max(4, 2 * order - 2);
   }

   if (isnan(alpha_))
   {
      alpha_ = 0.0;
   }
   else
   {
      alpha_ *= M_PI / 180.0;
   }

   if (isnan(nu_))
   {
      nu_ = 0.0;
   }
   else
   {
      nu_ *= M_PI / 180.0;
   }

   if (strstr(mesh_file, "square") != NULL)
   {
      aVec_[0].SetSize(2);
      aVec_[1].SetSize(2);
      aVec_[0][0] = 2.0; aVec_[0][1] = 0.0;
      aVec_[1][0] = 0.0; aVec_[1][1] = 2.0;
   }
   else if (strstr(mesh_file, "hexagon") != NULL)
   {
      aVec_[0].SetSize(2);
      aVec_[1].SetSize(2);
      aVec_[0][0] = 1.5; aVec_[0][1] = 0.5 * sqrt(3.0);
      aVec_[1][0] = 0.0; aVec_[1][1] = sqrt(3.0);
   }

   // 3. Construct a (serial) mesh of the given size on all processors.  We
   //    can handle triangular and quadrilateral surface meshes with the
   //    same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 4. This step is no longer needed

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list(0);
   Array<int> ess_bdr(0);
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
   }

   // The following is required for mesh refinement
   // mesh->EnsureNCMesh();

   // 6. Define the ODE solver used for time integration. Several implicit
   //    methods are available, including singly diagonal implicit Runge-Kutta
   //    (SDIRK).
   ODESolver *ode_exp_solver = NULL;
   ODESolver *ode_imp_solver = NULL;
   switch (ode_exp_solver_type)
   {
      case 1: ode_exp_solver = new ForwardEulerSolver; break;
      case 2: ode_exp_solver = new RK2Solver(1.0); break;
      case 3: ode_exp_solver = new RK3SSPSolver; break;
      case 4: ode_exp_solver = new RK4Solver; break;
      case 6: ode_exp_solver = new RK6Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown Explicit ODE solver type: "
                 << ode_exp_solver_type << '\n';
         }
         return 3;
   }
   switch (ode_imp_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_imp_solver = new BackwardEulerSolver; break;
      case 2:  ode_imp_solver = new SDIRK23Solver(2); break;
      case 3:  ode_imp_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_imp_solver = new ImplicitMidpointSolver; break;
      case 23: ode_imp_solver = new SDIRK23Solver; break;
      case 34: ode_imp_solver = new SDIRK34Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown Implicit ODE solver type: "
                 << ode_imp_solver_type << '\n';
         }
         return 3;
   }

   // 12. Define the parallel finite element spaces. We use:
   //
   //     H(curl) for electric field,
   //     H(div) for magnetic flux,
   //     H(div) for thermal flux,
   //     H(grad)/H1 for electrostatic potential,
   //     L2 for temperature

   // L2 contains discontinuous "cell-center" finite elements, type 2 is
   // "positive"
   L2_FECollection L2FEC0(0, dim);
   L2_FECollection L2FEC(order-1, dim);

   // RT contains Raviart-Thomas "face-centered" vector finite elements with
   // continuous normal component.
   RT_FECollection HDivFEC(order-1, dim);

   // ND contains Nedelec "edge-centered" vector finite elements with
   // continuous tangential component.
   ND_FECollection HCurlFEC(order, dim);

   // H1 contains continuous "node-centered" Lagrange finite elements.
   H1_FECollection HGradFEC(order, dim);

   ParFiniteElementSpace   L2FESpace0(pmesh, &L2FEC0);
   ParFiniteElementSpace    L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace  HDivFESpace(pmesh, &HDivFEC);
   ParFiniteElementSpace HCurlFESpace(pmesh, &HCurlFEC);
   ParFiniteElementSpace HGradFESpace(pmesh, &HGradFEC);

   // The terminology is TrueVSize is the unique (non-redundant) number of dofs
   // HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   // HYPRE_Int glob_size_rt = HDivFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_h1 = HGradFESpace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of Temperature unknowns:       " << glob_size_h1 << endl;
   }

   // int Vsize_l2 = L2FESpace.GetVSize();
   // int Vsize_rt = HDivFESpace.GetVSize();
   // int Vsize_h1 = HGradFESpace.GetVSize();

   // grid functions E, B, T, F, P, and w which is the Joule heating
   ParGridFunction q(&HCurlFESpace);
   ParGridFunction qPara(&HCurlFESpace);
   ParGridFunction qPerp(&HCurlFESpace);
   ParGridFunction Q(&L2FESpace);
   ParGridFunction T1(&HGradFESpace);
   ParGridFunction T0(&HGradFESpace);
   ParGridFunction dT(&HGradFESpace);
   ParGridFunction errorq(&L2FESpace0);
   ParGridFunction errorqPara(&L2FESpace0);
   ParGridFunction errorqPerp(&L2FESpace0);
   ParGridFunction errorT(&L2FESpace0);
   T0 = 0.0;
   T1 = 0.0;
   dT = 1.0;

   // 13. Get the boundary conditions, set up the exact solution grid functions
   //     These VectorCoefficients have an Eval function.  Note that e_exact and
   //     b_exact in this case are exact analytical solutions, taking a 3-vector
   //     point as input and returning a 3-vector field
   FunctionCoefficient TCoef(TFunc);
   VectorFunctionCoefficient qCoef(2, qFunc);

   Vector velVec(2);
   velVec[0] = vMag_ * cos(nu_); velVec[1] = vMag_ * sin(nu_);
   VectorConstantCoefficient velCoef(velVec);

   Vector zeroVec(dim); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   ConstantCoefficient oneCoef(1.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   IdentityMatrixCoefficient ICoef(2);
   MatrixFunctionCoefficient bbTCoef(2, bbTFunc);
   MatrixSumCoefficient ImbbTCoef(bbTCoef, ICoef, -1.0);

   MatVecCoefficient qParaCoef(bbTCoef, qCoef);
   MatVecCoefficient qPerpCoef(ImbbTCoef, qCoef);

   ConstantCoefficient SpecificHeatCoef(1.0);
   MatrixFunctionCoefficient ConductionCoef(2, ChiFunc);
   ConstantCoefficient HeatSourceCoef(0.0);

   Q.ProjectCoefficient(HeatSourceCoef);

   if (!zero_start)
   {
      T1.ProjectCoefficient(TCoef);
      q.ProjectCoefficient(qCoef);
   }

   T1.GridFunction::ComputeElementL2Errors(TCoef, errorT);
   q.GridFunction::ComputeElementL2Errors(qCoef, errorq);
   qPara.GridFunction::ComputeElementL2Errors(qParaCoef, errorqPara);
   qPerp.GridFunction::ComputeElementL2Errors(qPerpCoef, errorqPerp);

   ParBilinearForm m1(&HCurlFESpace);
   m1.AddDomainIntegrator(new VectorFEMassIntegrator);
   m1.Assemble();

   ParMixedBilinearForm gPara(&HGradFESpace, &HCurlFESpace);
   gPara.AddDomainIntegrator(new MixedVectorGradientIntegrator(bbTCoef));
   gPara.Assemble();

   ParMixedBilinearForm gPerp(&HGradFESpace, &HCurlFESpace);
   gPerp.AddDomainIntegrator(new MixedVectorGradientIntegrator(ImbbTCoef));
   gPerp.Assemble();

   HypreParMatrix M1C;
   Vector RHS1(HCurlFESpace.GetTrueVSize()), X1(HCurlFESpace.GetTrueVSize());

   Array<int> ess_tdof_list_q(0);
   // Array<int> ess_bdr_q;
   // HCurlFESpace.GetEssentialTrueDofs(ess_bdr_q, ess_tdof_list_q);

   m1.FormSystemMatrix(ess_tdof_list_q, M1C);

   HypreDiagScale Precond(M1C);
   HyprePCG M1Inv(M1C);
   M1Inv.SetTol(1e-12);
   M1Inv.SetMaxIter(200);
   M1Inv.SetPrintLevel(0);
   M1Inv.SetPreconditioner(Precond);

   // 14. Initialize the Diffusion operator, the GLVis visualization and print
   //     the initial energies.
   DiffusionTDO imp_oper(HGradFESpace,
                         zeroCoef, ess_bdr,
                         SpecificHeatCoef, false,
                         ConductionCoef, false,
                         HeatSourceCoef, false);

   AdvectionTDO exp_oper(HGradFESpace,
                         velCoef);

   // This function initializes all the fields to zero or some provided IC
   // oper.Init(F);

   socketstream vis_q, vis_errq;
   socketstream vis_qPara, vis_errqPara;
   socketstream vis_qPerp, vis_errqPerp;
   socketstream vis_T, vis_errT;
   char vishost[] = "localhost";
   int  visport   = 19916;
   int Wx = 0, Wy = 0; // window position
   int Ww = 400, Wh = 270; // window size
   int offx = Ww+3, offy = Wh+25; // window offsets
   const char * h1_keys = "mmaaAc";
   const char * l2_keys = "aaAc";

   if (visualization)
   {
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_T.precision(8);
      vis_errT.precision(8);
      vis_q.precision(8);
      vis_errq.precision(8);
      vis_qPara.precision(8);
      vis_errqPara.precision(8);
      vis_qPerp.precision(8);
      vis_errqPerp.precision(8);

      VisualizeField(vis_T, vishost, visport,
                               T1, "Temperature", Wx, Wy, Ww, Wh, h1_keys);

      Wy += offy;
      VisualizeField(vis_errT, vishost, visport,
                               errorT, "Error in T", Wx, Wy, Ww, Wh, l2_keys);

      Wx += offx;
      Wy -= offy;
      VisualizeField(vis_q, vishost, visport,
                               q, "Heat Flux", Wx, Wy, Ww, Wh, NULL, true);

      Wy += offy;
      VisualizeField(vis_errq, vishost, visport,
                               errorq, "Error in q", Wx, Wy, Ww, Wh, l2_keys);

      Wx += offx;
      Wy -= offy;
      VisualizeField(vis_qPara, vishost, visport,
                               qPara, "Parallel Heat Flux",
                               Wx, Wy, Ww, Wh);

      Wy += offy;
      VisualizeField(vis_errqPara, vishost, visport,
                               errorqPara, "Error in q para",
                               Wx, Wy, Ww, Wh, l2_keys);

      Wx += offx;
      Wy -= offy;
      VisualizeField(vis_qPerp, vishost, visport,
                               qPerp, "Perpendicular Heat Flux",
                               Wx, Wy, Ww, Wh);

      Wy += offy;
      VisualizeField(vis_errqPerp, vishost, visport,
                               errorqPerp, "Error in q perp",
                               Wx, Wy, Ww, Wh, l2_keys);
   }
   // VisIt visualization
   VisItDataCollection visit_dc(basename, pmesh);
   if ( visit )
   {
      visit_dc.RegisterField("q", &q);
      visit_dc.RegisterField("qPara", &qPara);
      visit_dc.RegisterField("qPerp", &qPerp);
      visit_dc.RegisterField("T", &T1);

      visit_dc.RegisterField("L2 Error T", &errorT);
      visit_dc.RegisterField("L2 Error q", &errorq);
      visit_dc.RegisterField("L2 Error q para", &errorqPara);
      visit_dc.RegisterField("L2 Error q perp", &errorqPerp);

      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // 15. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt). The object oper is the MagneticDiffusionOperator which
   //     has a Mult() method and an ImplicitSolve() method which are used by
   //     the time integrators.
   ode_imp_solver->Init(imp_oper);
   ode_exp_solver->Init(exp_oper);

   ofstream ofs_err("adv_diff_v1.err");

   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         if (myid == 0)
         {
            cout << "Final Time Reached" << endl;
         }
         last_step = true;
      }

      // F is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      T0 = T1;
      // ode_imp_solver->Step(T1, t, dt);
      // t -= dt;
      ode_exp_solver->Step(T1, t, dt);
      if (ti % 10 == 0)
      {
         double t_imp = t - 10.0 * dt;
         double dt_imp = 10.0 * dt;
         ode_imp_solver->Step(T1, t_imp, dt_imp);
      }

      add(1.0, T1, -1.0, T0, dT);

      double maxT    = T1.ComputeMaxError(zeroCoef);
      double maxDiff = dT.ComputeMaxError(zeroCoef);

      if ( !last_step )
      {
         if ( maxT == 0.0 )
         {
            last_step = (maxDiff < tol) ? true:false;
         }
         else if ( maxDiff/maxT < tol )
         {
            last_step = true;
         }
         if (last_step && myid == 0)
         {
            cout << "Converged to Steady State" << endl;
         }
      }
      /*
      if (debug == 1)
      {
         oper.Debug(basename,t);
      }
      */
      gPara.Mult(T1, qPara);
      gPerp.Mult(T1, qPerp);

      qPara.ParallelAssemble(RHS1);
      X1 = 0.0;
      M1Inv.Mult(RHS1, X1);
      qPara.Distribute(X1);
      qPara *= -chiPara_;

      qPerp.ParallelAssemble(RHS1);
      X1 = 0.0;
      M1Inv.Mult(RHS1, X1);
      qPerp.Distribute(X1);
      qPerp *= -chiPerp_;

      q = qPara;
      q += qPerp;

      if (gfprint)
      {
         ostringstream q_name, T_name, mesh_name;
         q_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "q." << setfill('0') << setw(6) << myid;
         T_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "T." << setfill('0') << setw(6) << myid;
         mesh_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                   << "mesh." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);
         mesh_ofs.close();

         ofstream q_ofs(q_name.str().c_str());
         q_ofs.precision(8);
         q.Save(q_ofs);
         q_ofs.close();

         ofstream T_ofs(T_name.str().c_str());
         T_ofs.precision(8);
         T1.Save(T_ofs);
         T_ofs.close();
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization | visit)
         {
            TCoef.SetTime(t);
            double nrm1 = T1.ComputeL2Error(oneCoef);
            double err1 = T1.ComputeL2Error(TCoef);
            double qnrm1 = q.ComputeL2Error(zeroVecCoef);
            double qerr1 = q.ComputeL2Error(qCoef);
            if (myid == 0)
            {
               cout << t << " L2 Relative Error of Solution: " << err1 / nrm1
                    << "\t" << qerr1 / qnrm1 << endl;
               ofs_err << t << "\t" << err1 / nrm1 << "\t"
                       << qerr1 / qnrm1 << endl;
            }
            T1.GridFunction::ComputeElementL2Errors(TCoef, errorT);
            q.GridFunction::ComputeElementL2Errors(qCoef, errorq);
            qPara.GridFunction::ComputeElementL2Errors(qParaCoef, errorqPara);
            qPerp.GridFunction::ComputeElementL2Errors(qPerpCoef, errorqPerp);
         }
         if (visualization)
         {
            VisualizeField(vis_q, vishost, visport,
                                     q, "Heat Flux", Wx, Wy, Ww, Wh);

            VisualizeField(vis_qPara, vishost, visport,
                                     qPara, "Parallel Heat Flux",
                                     Wx, Wy, Ww, Wh);

            VisualizeField(vis_qPerp, vishost, visport,
                                     qPerp, "Perpendicular Heat Flux",
                                     Wx, Wy, Ww, Wh);

            // Wx += offx;
            // VisualizeField(vis_U, vishost, visport,
            //                       U1, "Energy", Wx, Wy, Ww, Wh);

            // Wx -= offx;
            // Wy += offy;
            VisualizeField(vis_T, vishost, visport,
                                     T1, "Temperature",
                                     Wx, Wy, Ww, Wh, h1_keys);

            // Wx += offx;
            VisualizeField(vis_errT, vishost, visport,
                                     errorT, "Error in T",
                                     Wx, Wy, Ww, Wh, l2_keys);

            // Wx += offx;
            VisualizeField(vis_errq, vishost, visport,
                                     errorq, "Error in q",
                                     Wx, Wy, Ww, Wh, l2_keys);

            VisualizeField(vis_errqPara, vishost, visport,
                                     errorqPara, "Error in q para",
                                     Wx, Wy, Ww, Wh, l2_keys);

            VisualizeField(vis_errqPerp, vishost, visport,
                                     errorqPerp, "Error in q perp",
                                     Wx, Wy, Ww, Wh, l2_keys);
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }
   ofs_err.close();

   if (visualization)
   {
      vis_q.close();
      vis_T.close();
      vis_errT.close();
      vis_errq.close();
      vis_errqPara.close();
      vis_errqPerp.close();
   }

   double loc_T_max = T1.Normlinf();
   double T_max = -1.0;
   MPI_Allreduce(&loc_T_max, &T_max, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
   double err1 = T1.ComputeL2Error(TCoef);
   if (myid == 0)
   {
      cout << "L2 Error of Solution: " << err1 << endl;
      cout << "Maximum Temperature: " << T_max << endl;
      cout << "| chi_eff - 1 | = " << fabs(1.0/T_max - 1) << endl;
   }

   // 16. Free the used memory.
   delete ode_imp_solver;
   delete ode_exp_solver;
   delete pmesh;

   return 0;
}

void display_banner(ostream & os)
{
   os << "___________                 .__              " << endl
      << "\\_   _____/___  __ _________|__| ___________ " << endl
      << " |    __)/  _ \\|  |  \\_  __ \\  |/ __ \\_  __ \\" << endl
      << " |    | (  <_> )  |  /|  | \\/  \\  ___/|  | \\/" << endl
      << " \\__  |  \\____/|____/ |__|  |__|\\___  >__|   " << endl
      << "    \\/                              \\/       , not..." << endl
      << flush;
}
