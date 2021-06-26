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

#include "fourier_hybrid_solver.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::thermal;

void display_banner(ostream & os);

static int    prob_          = 1;
static int    unit_vec_type_ = 1;
static bool   non_linear_    = false;
static double alpha_         = NAN;
static double theta_         = NAN;
static double gamma_         = 10.0;
static double chi_perp_      = 1.0;
static double chi_para_      = 1.0;
static double a_             = 0.15;
static double b_             = 0.85;
static double xc_            = 0.0;
static double yc_            = 0.0;

double TFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
         return x[0] * x[1] * pow(sin(M_PI * x[0]) * sin(M_PI * x[1]), gamma_);
      case 2:
         return 1.0 - pow(pow(x[0] - xc_, 2) + pow(x[1] - yc_, 2), 1.5);
      case 3:
         return 1.0 + (a_ * x[0] + b_ * x[1]) * pow(x[0] * x[0] + x[1] * x[1], 1.5);
      case 4:
         return 1.0 - pow(a_ * pow(x[0] * cos(theta_) + x[1] * sin(theta_), 2) +
                          b_ * pow(x[0] * sin(theta_) - x[1] * cos(theta_), 2), 1.5);
      default:
         return 0.0;
   }
}

void qFunc(const Vector &x, Vector &q)
{
   q.SetSize(2);

   switch (prob_)
   {
      case 1:
      {
         double ssg = pow(sin(M_PI * x[0]) * sin(M_PI * x[1]), gamma_ - 1.0);
         double ca = cos(alpha_);
         double sa = sin(alpha_);
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);
         double xcx = sx + M_PI * gamma_ * x[0] * cx;
         double ycy = sy + M_PI * gamma_ * x[1] * cy;
         double cd = chi_para_ - chi_perp_;
         double cdca = cd * ca * ca + chi_perp_;
         double cdsa = cd * sa * sa + chi_perp_;
         q[0] = - x[0] * cd * ca * sa * sx * ycy - x[1] * cdca * sy * xcx;
         q[1] = - x[1] * cd * ca * sa * sy * xcx - x[0] * cdsa * sx * ycy;
         q *= ssg;
      }
      break;
      default:
         q = 0.0;
   }
}

void qParaFunc(const Vector &x, Vector &q)
{
   q.SetSize(2);

   switch (prob_)
   {
      case 1:
      {
         double ssg = pow(sin(M_PI * x[0]) * sin(M_PI * x[1]), gamma_ - 1.0);
         double ca = cos(alpha_);
         double sa = sin(alpha_);
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);
         double xcx = sx + M_PI * gamma_ * x[0] * cx;
         double ycy = sy + M_PI * gamma_ * x[1] * cy;
         double cd = chi_para_;
         double cdca = cd * ca * ca;
         double cdsa = cd * sa * sa;
         q[0] = - x[0] * cd * ca * sa * sx * ycy - x[1] * cdca * sy * xcx;
         q[1] = - x[1] * cd * ca * sa * sy * xcx - x[0] * cdsa * sx * ycy;
         q *= ssg;
      }
      break;
      default:
         q = 0.0;
   }
}

void qPerpFunc(const Vector &x, Vector &q)
{
   q.SetSize(2);

   switch (prob_)
   {
      case 1:
      {
         double ssg = pow(sin(M_PI * x[0]) * sin(M_PI * x[1]), gamma_ - 1.0);
         double ca = cos(alpha_);
         double sa = sin(alpha_);
         double cx = cos(M_PI * x[0]);
         double cy = cos(M_PI * x[1]);
         double sx = sin(M_PI * x[0]);
         double sy = sin(M_PI * x[1]);
         double xcx = sx + M_PI * gamma_ * x[0] * cx;
         double ycy = sy + M_PI * gamma_ * x[1] * cy;
         double cd = - chi_perp_;
         double cdca = cd * ca * ca + chi_perp_;
         double cdsa = cd * sa * sa + chi_perp_;
         q[0] = - x[0] * cd * ca * sa * sx * ycy - x[1] * cdca * sy * xcx;
         q[1] = - x[1] * cd * ca * sa * sy * xcx - x[0] * cdsa * sx * ycy;
         q *= ssg;
      }
      break;
      default:
         q = 0.0;
   }
}

void UnitBFunc(const Vector &x, Vector &b)
{
   switch (unit_vec_type_)
   {
      case 2:
      {
         b[0] = -x[1] + yc_;
         b[1] = x[0] - xc_;
      }
      break;
      case 3:
      {
         b[0] = -3.0 * a_ * x[0] * x[1] -
                b_ * (x[0] * x[0] + 4.0 * x[1] * x[1]);
         b[1] = a_ * (4.0 * x[0] * x[0] + x[1] * x[1]) + 3.0 * b_ * x[0] * x[1];
      }
      break;
      case 4:
      {
         double ct = cos(theta_);
         double st = sin(theta_);
         double ctst = 0.5 * sin(2.0 * theta_);
         b[0] = x[1] * (a_ * st * st + b_ * ct * ct) + (a_ - b_) * x[0] * ctst;
         b[1] = -x[0] * (a_ * ct * ct + b_ * st * st) - (a_ - b_) * x[1] * ctst;
      }
      break;
      default:
         b[0] = cos(alpha_);
         b[1] = sin(alpha_);
   }
   double nrm = b.Norml2();
   if ( nrm > 0.0 ) { b /= nrm; }
}

double QFunc(const Vector &x, double t)
{
   switch (prob_)
   {
      case 1:
      {
         double cx  = cos(M_PI * x[0]);
         double sx  = sin(M_PI * x[0]);
         double s2x = sin(2.0 * M_PI * x[0]);
         double cy  = cos(M_PI * x[1]);
         double sy  = sin(M_PI * x[1]);
         double s2y = sin(2.0 * M_PI * x[1]);
         double ca  = cos(alpha_);
         double sa  = sin(alpha_);
         double s2a = sin(2.0 * alpha_);
         double chi_sc = chi_perp_ * sa * sa + chi_para_ * ca * ca;
         double chi_cs = chi_perp_ * ca * ca + chi_para_ * sa * sa;
         double chi_s2 = (chi_para_ - chi_perp_) * s2a;
         double s2gcx = s2x + M_PI * x[0] * (gamma_ * cx * cx - 1.0);
         double s2gcy = s2y + M_PI * x[1] * (gamma_ * cy * cy - 1.0);
         double sgcx = sx + M_PI * x[0] * gamma_ * cx;
         double sgcy = sy + M_PI * x[1] * gamma_ * cy;
         return -1.0 * (M_PI * gamma_ * x[0] * chi_cs * s2gcy * sx * sx +
                        M_PI * gamma_ * x[1] * chi_sc * s2gcx * sy * sy +
                        chi_s2 * sgcx * sgcy * sx * sy) *
                pow(sx * sy, gamma_ - 2.0);
      }
      case 2:
      {
         return 9.0 * chi_perp_ * sqrt(pow(x[0] - xc_, 2) + pow(x[1] - yc_, 2));
      }
      default:
         return 0.0;
   }
}

void shiftUnitSquare(const Vector &x, Vector &p)
{
   p[0] = x[0] - 0.5;
   p[1] = x[1] - 0.5;
}

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
   int el_type = Element::QUADRILATERAL;
   int ode_solver_type = 1;
   int coef_type = 0;
   int vis_steps = 1;
   double dt = 0.5;
   double t_final = 5.0;
   double tol = 1e-4;
   const char *basename = "FourierHybrid";
   const char *mesh_file = "";
   bool zero_start = true;
   bool static_cond = false;
   bool gfprint = true;
   bool visit = true;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&n, "-n", "--num-elems-1d",
                  "Number of elements in x and y directions.  "
                  "Total number of elements is n^2.");
   args.AddOption(&prob_, "-p", "--problem",
                  "Specify problem type: 1 - Square, 2 - Ellipse.");
   args.AddOption(&coef_type, "-c", "--coef",
                  "Specify diffusion coefficient type: "
                  "0 - Constant, 1 - Linearized, 2 - Non-Linear.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&irOrder, "-iro", "--int-rule-order",
                  "Integration Rule Order.");
   args.AddOption(&alpha_, "-alpha", "--constant-angle",
                  "Angle for constant B field (in degrees)");
   args.AddOption(&theta_, "-theta", "--tilt-angle",
                  "Angle for orientation of ellipse (in degrees)");
   args.AddOption(&a_, "-a", "--ellipse-a",
                  "First size parameter for ellipse");
   args.AddOption(&b_, "-b", "--ellipse-b",
                  "Second size parameter for ellipse");
   args.AddOption(&xc_, "-xc", "--x-center",
                  "x coordinate of field center");
   args.AddOption(&yc_, "-yc", "--y-center",
                  "y coordinate of field center");
   args.AddOption(&chi_perp_, "-chi-perp", "--chi-perpendicular",
                  "Chi perpendicular to field lines.");
   args.AddOption(&chi_para_, "-chi-para", "--chi-parallel",
                  "Chi along field lines.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&t_final, "-tf", "--final-time",
                  "Final Time.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Tolerance used to determine convergence to steady state.");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 2-Triangle, 3-Quadrilateral.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3\n\t."
                  "\t   22 - Mid-Point, 23 - SDIRK23, 34 - SDIRK34.");
   args.AddOption(&zero_start, "-z", "--zero-start", "-no-z",
                  "--no-zero-start",
                  "Initial guess of zero or exact solution.");
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

   unit_vec_type_ = prob_;
   non_linear_ = coef_type > 0;

   // 3. Construct a (serial) mesh of the given size on all processors.  We
   //    can handle triangular and quadrilateral surface meshes with the
   //    same code.
   Mesh *mesh = (n > 0) ?
                new Mesh(n, n, (Element::Type)el_type, true) :
                new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (prob_ > 1) { mesh->Transform(shiftUnitSquare); }

   // 4. This step is no longer needed

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr(0);
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
   }

   // The following is required for mesh refinement
   // mesh->EnsureNCMesh();

   // 6. Define the ODE solver used for time integration. Several implicit
   //    methods are available, including singly diagonal implicit Runge-Kutta
   //    (SDIRK).
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 34: ode_solver = new SDIRK34Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
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
   ND_FECollection HCurlFEC(order, dim);

   // H1 contains continuous "node-centered" Lagrange finite elements.
   H1_FECollection HGradFEC(order, dim);

   ParFiniteElementSpace   L2FESpace0(pmesh, &L2FEC0);
   ParFiniteElementSpace    L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace  HDivFESpace(pmesh, &HDivFEC);
   ParFiniteElementSpace  HCurlFESpace(pmesh, &HCurlFEC);
   ParFiniteElementSpace  HGradFESpace(pmesh, &HGradFEC);

   // The terminology is TrueVSize is the unique (non-redundant) number of dofs
   // HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   // HYPRE_Int glob_size_rt = HDivFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_h1 = HGradFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_rt = HDivFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of Temperature unknowns:     " << glob_size_h1 << endl;
      cout << "Number of Heat Flux unknowns:       " << glob_size_rt << endl;
      cout << "Number of Thermal Energy unknowns:  " << glob_size_l2 << endl;
   }

   // int Vsize_l2 = L2FESpace.GetVSize();
   // int Vsize_rt = HDivFESpace.GetVSize();
   // int Vsize_h1 = HGradFESpace.GetVSize();

   // grid functions E, B, T, F, P, and w which is the Joule heating
   ParGridFunction  T_gf(&HGradFESpace);
   ParGridFunction  q_gf(&HDivFESpace);
   ParGridFunction  qPerpT_gf(&HDivFESpace);
   ParGridFunction  qParaT_gf(&HDivFESpace);
   ParGridFunction  qPerp_gf(&HDivFESpace);
   ParGridFunction  qPara_gf(&HDivFESpace);
   ParGridFunction  b_gf(&HDivFESpace);
   ParGridFunction dT_gf(&HGradFESpace);
   ParGridFunction Qs_gf(&HGradFESpace);
   ParGridFunction errorT(&L2FESpace0);
   ParGridFunction errorq(&L2FESpace0);
   ParGridFunction errorqPerp(&L2FESpace0);
   ParGridFunction errorqPara(&L2FESpace0);
   ParGridFunction errorqPerpT(&L2FESpace0);
   ParGridFunction errorqParaT(&L2FESpace0);
   T_gf  = 0.0;
   q_gf  = 0.0;
   dT_gf = 1.0;

   // 13. Get the boundary conditions, set up the exact solution grid functions
   //     These VectorCoefficients have an Eval function.  Note that e_exact and
   //     b_exact in this case are exact analytical solutions, taking a 3-vector
   //     point as input and returning a 3-vector field
   FunctionCoefficient TCoef(TFunc);
   VectorFunctionCoefficient qCoef(2, qFunc);
   VectorFunctionCoefficient qParaCoef(2, qParaFunc);
   VectorFunctionCoefficient qPerpCoef(2, qPerpFunc);

   Vector zeroVec(2); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);
   ConstantCoefficient SpecificHeatCoef(1.0);
   // MatrixFunctionCoefficient ConductionCoef(2, ChiFunc);
   FunctionCoefficient HeatSourceCoef(QFunc);

   VectorFunctionCoefficient UnitBCoef(2, UnitBFunc);

   b_gf.ProjectCoefficient(UnitBCoef);
   Qs_gf.ProjectCoefficient(HeatSourceCoef);

   T_gf.ProjectCoefficient(TCoef);
   q_gf.ProjectCoefficient(qCoef);
   qPara_gf.ProjectCoefficient(qParaCoef);
   qPerp_gf.ProjectCoefficient(qPerpCoef);

   double T_nrm = T_gf.ComputeL2Error(zeroCoef);
   double q_nrm = q_gf.ComputeL2Error(zeroVecCoef);
   double qPara_nrm = qPara_gf.ComputeL2Error(zeroVecCoef);
   double qPerp_nrm = qPerp_gf.ComputeL2Error(zeroVecCoef);

   T_gf.ProjectBdrCoefficient(TCoef, ess_bdr);
   q_gf.ProjectBdrCoefficientNormal(qCoef, ess_bdr);
   T_gf.GridFunction::ComputeElementL2Errors(TCoef, errorT);
   q_gf.GridFunction::ComputeElementL2Errors(qCoef, errorq);

   // 14. Initialize the Diffusion operator, the GLVis visualization and print
   //     the initial energies.
   cout << "Building TDO" << endl;
   HybridThermalDiffusionTDO oper(HGradFESpace,
                                  HCurlFESpace,
                                  HDivFESpace,
                                  L2FESpace,
                                  zeroVecCoef,
                                  zeroCoef, ess_bdr,
                                  chi_perp_,
                                  chi_para_,
                                  prob_,
                                  coef_type,
                                  UnitBCoef,
                                  SpecificHeatCoef, false,
                                  // ConductionCoef, false,
                                  HeatSourceCoef, false);

   // This function initializes all the fields to zero or some provided IC
   // oper.Init(F);

   socketstream vis_T, vis_q, vis_b, vis_Q, vis_errT, vis_errq;
   char vishost[] = "localhost";
   int  visport   = 19916;
   if (visualization)
   {
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_T.precision(8);
      vis_Q.precision(8);
      vis_q.precision(8);
      vis_b.precision(8);
      vis_errT.precision(8);
      vis_errq.precision(8);

      int Wx = 0, Wy = 0; // window position
      int Ww = 350, Wh = 350; // window size
      int offx = Ww+10, offy = Wh+45; // window offsets

      VisualizeField(vis_Q, vishost, visport,
                               Qs_gf, "Heat Source", Wx, Wy, Ww, Wh);

      Wy += offy;
      VisualizeField(vis_b, vishost, visport,
                               b_gf, "Unit B Field",
                               Wx, Wy, Ww, Wh, NULL, true);

      Wx += offx; Wy -= offy;
      VisualizeField(vis_T, vishost, visport,
                               T_gf, "Temperature", Wx, Wy, Ww, Wh);

      Wy += offy;
      VisualizeField(vis_errT, vishost, visport,
                               errorT, "Error in T", Wx, Wy, Ww, Wh);

      Wx += offx; Wy -= offy;
      VisualizeField(vis_q, vishost, visport,
                               q_gf, "Heat Flux",
                               Wx, Wy, Ww, Wh, NULL, true);

      Wy += offy;
      VisualizeField(vis_errq, vishost, visport,
                               errorq, "Error in q", Wx, Wy, Ww, Wh);
   }
   // VisIt visualization
   VisItDataCollection visit_dc(basename, pmesh);
   if ( visit )
   {
      visit_dc.RegisterField("T", &T_gf);
      visit_dc.RegisterField("Qs", &Qs_gf);
      visit_dc.RegisterField("q", &q_gf);
      visit_dc.RegisterField("qPerp", &qPerp_gf);
      visit_dc.RegisterField("qPara", &qPara_gf);
      visit_dc.RegisterField("qPerpT", &qPerpT_gf);
      visit_dc.RegisterField("qParaT", &qParaT_gf);
      visit_dc.RegisterField("b", &b_gf);
      visit_dc.RegisterField("L2 Error T", &errorT);
      visit_dc.RegisterField("L2 Error q", &errorq);
      visit_dc.RegisterField("L2 Error qPerp", &errorqPerp);
      visit_dc.RegisterField("L2 Error qPara", &errorqPara);
      visit_dc.RegisterField("L2 Error qPerpT", &errorqPerpT);
      visit_dc.RegisterField("L2 Error qParaT", &errorqParaT);

      oper.SetVisItDC(visit_dc);

      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   ostringstream oss_errs;
   oss_errs << "fourier_hybrid_errs"
            << "_p" << prob_ << "_c" << coef_type
            << "_e" << (int)floor(log10(chi_para_/chi_perp_));
   if (n > 0) { oss_errs << "_n" << n; }
   oss_errs << "_o" << order << ".dat";
   ofstream ofs_errs;
   if (myid == 0) { ofs_errs.open(oss_errs.str().c_str()); }

   // 15. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt). The object oper is the MagneticDiffusionOperator which
   //     has a Mult() method and an ImplicitSolve() method which are used by
   //     the time integrators.
   ode_solver->Init(oper);
   double t = 0.0;

   int tsize = HGradFESpace.GetTrueVSize();
   int qsize = HDivFESpace.GetTrueVSize();
   Vector X0(tsize+qsize), X1(tsize+qsize), dX(tsize+qsize);
   Vector T1(X1.GetData(), tsize);
   Vector q1(&(X1.GetData())[tsize], qsize);
   X0 = 0.0; X1 = 0.0; dX = 0.0;
   T_gf.ParallelProject(T1);

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
      X0 = X1;
      ode_solver->Step(X1, t, dt);

      T_gf.Distribute(T1);
      q_gf.Distribute(q1);

      TCoef.SetTime(t);

      oper.GetParaFluxFromTemp(T_gf, qParaT_gf);
      oper.GetPerpFluxFromTemp(T_gf, qPerpT_gf);

      oper.GetParaFluxFromFlux(q_gf, qPara_gf);
      oper.GetPerpFluxFromFlux(q_gf, qPerp_gf);

      T_gf.GridFunction::ComputeElementL2Errors(TCoef, errorT);
      q_gf.GridFunction::ComputeElementL2Errors(qCoef, errorq);
      qPerp_gf.GridFunction::ComputeElementL2Errors(qPerpCoef, errorqPerp);
      qPara_gf.GridFunction::ComputeElementL2Errors(qParaCoef, errorqPara);
      qPerpT_gf.GridFunction::ComputeElementL2Errors(qPerpCoef, errorqPerpT);
      qParaT_gf.GridFunction::ComputeElementL2Errors(qParaCoef, errorqParaT);
      double l2_error_T = T_gf.ComputeL2Error(TCoef);
      double l2_error_q = q_gf.ComputeL2Error(qCoef);

      if ( myid == 0 )
      {
         ofs_errs << t << '\t' << l2_error_T << '\t' << l2_error_q << endl;
         cout << t << '\t' << l2_error_T << '\t' << l2_error_q << endl;
      }

      add(1.0, X1, -1.0, X0, dX);

      Vector dT(dX.GetData(), tsize);
      dT_gf.Distribute(dT);

      double maxT    = T_gf.ComputeMaxError(zeroCoef);
      double maxDiff = dT_gf.ComputeMaxError(zeroCoef);

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
      if (gfprint)
      {
         ostringstream T_name, q_name, mesh_name;
         T_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "T." << setfill('0') << setw(6) << myid;
         q_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "q." << setfill('0') << setw(6) << myid;
         mesh_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                   << "mesh." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);
         mesh_ofs.close();

         ofstream T_ofs(T_name.str().c_str());
         T_ofs.precision(8);
         T_gf.Save(T_ofs);
         T_ofs.close();

         ofstream q_ofs(q_name.str().c_str());
         q_ofs.precision(8);
         q_gf.Save(q_ofs);
         q_ofs.close();
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size

            VisualizeField(vis_T, vishost, visport,
                                     T_gf, "Temperature", Wx, Wy, Ww, Wh);

            VisualizeField(vis_q, vishost, visport,
                                     q_gf, "Heat Flux", Wx, Wy, Ww, Wh);

            VisualizeField(vis_errT, vishost, visport,
                                     errorT, "Error in T", Wx, Wy, Ww, Wh);

            VisualizeField(vis_errq, vishost, visport,
                                     errorq, "Error in q", Wx, Wy, Ww, Wh);
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }

   // oper.GetParaFluxFromTemp(T_gf, qParaT_gf);
   // oper.GetPerpFluxFromTemp(T_gf, qPerpT_gf);

   if (visualization)
   {
      vis_T.close();
      vis_q.close();
      vis_errT.close();
      vis_errq.close();
   }
   if (myid == 0) { ofs_errs.close(); }

   /*
   double loc_T_max = T1.Normlinf();
   double T_max = -1.0;
   MPI_Allreduce(&loc_T_max, &T_max, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
   */
   double err1 = T_gf.ComputeL2Error(TCoef);
   double errq = q_gf.ComputeL2Error(qCoef);
   double errqParaT = qParaT_gf.ComputeL2Error(qParaCoef);
   double errqPerpT = qPerpT_gf.ComputeL2Error(qPerpCoef);
   double errqPara = qPara_gf.ComputeL2Error(qParaCoef);
   double errqPerp = qPerp_gf.ComputeL2Error(qPerpCoef);
   double T_max = T_gf.ComputeMaxError(zeroCoef);
   double q_max = q_gf.ComputeMaxError(zeroVecCoef);
   // double qParaT_max = qParaT_gf.ComputeMaxError(zeroVecCoef);
   // double qPerpT_max = qPerpT_gf.ComputeMaxError(zeroVecCoef);
   // double qPara_max = qPara_gf.ComputeMaxError(zeroVecCoef);
   // double qPerp_max = qPerp_gf.ComputeMaxError(zeroVecCoef);
   if (myid == 0)
   {
      cout << "Maximum Temperature:     " << T_max << endl;
      cout << "Maximum Flux Magnitude:  " << q_max << endl;
      cout << "L2 Error of Temperature: " << err1
           << ", (relative " << err1 / T_nrm << ")"
           << endl;
      cout << "L2 Error of Flux: " << errq
           << ", (relative " << errq / q_nrm << ")"
           << endl;
      cout << "L2 Error of Para Flux: " << errqPara
           << ", (relative " << errqPara / qPara_nrm << ")"
           << endl;
      cout << "L2 Error of Perp Flux: " << errqPerp
           << ", (relative " << errqPerp / qPerp_nrm << ")"
           << endl;
      cout << "L2 Error of Para Flux T: " << errqParaT
           << ", (relative " << errqParaT / qPara_nrm << ")"
           << endl;
      cout << "L2 Error of Perp Flux T: " << errqPerpT
           << ", (relative " << errqPerpT / qPerp_nrm << ")"
           << endl;
      cout << "| chi_eff - 1 | = " << fabs(1.0/T_max - 1) << endl;
   }

   // 16. Free the used memory.
   delete ode_solver;
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
      << "    \\/                              \\/       " << endl
      << flush;
}
