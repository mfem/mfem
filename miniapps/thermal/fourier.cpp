// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//            -----------------------------------------------------
//                     Fourier Miniapp:  Thermal Diffusion
//            -----------------------------------------------------
//
// This miniapp solves a time dependent heat equation.
//

#include "fourier_solver.hpp"
#include <cassert>
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::common;
using namespace mfem::thermal;

void display_banner(ostream & os);

static double TInf_    = 1.0;
static double TMin_    = 10.0;
static double TMax_    = 100.0;
static double kappa_ = 1.0;
static double c_rho_ = 1.0;

static vector<Vector> aVec_;

double TFunc(const Vector &x, double t)
{
   double f = 0.5 * (TMax_ - TMin_) * cos(M_PI * x[0]) * cos(M_PI * x[1]);
   double g = exp(-2.0 * M_PI * M_PI * kappa_ * t / c_rho_);
   return 0.5 * (TMax_ + TMin_) + f * g;
}

class ConstantStateVariableCoef : public StateVariableCoef
{
private:
   double val_;

public:
   ConstantStateVariableCoef(double val) : val_(val) {}

   virtual ConstantStateVariableCoef * Clone() const
   {
      return new ConstantStateVariableCoef(val_);
   }

   virtual bool NonTrivialValue(FieldType deriv) const { return false; }

   double Eval_Func(ElementTransformation &T, const IntegrationPoint &ip)
   { return val_; }
};

class DiffusionCoef : public StateVariableCoef
{
private:
   Coefficient * T_;
   double Tau_;
   double kappa_;
   int p_;

public:
   DiffusionCoef(Coefficient &T, double Tau, double kappa, int p)
      : T_(&T), Tau_(Tau), kappa_(kappa), p_(p)
   {
      MFEM_VERIFY(kappa_ > 0.0, "Diffusion coefficient must be positive");
      MFEM_VERIFY(Tau_ > 0.0, "Temperature scale must be positive");
      MFEM_VERIFY(p_ >= 0, "Nonlinear power must be non-negative");
   }

   virtual DiffusionCoef * Clone() const
   {
      return new DiffusionCoef(*T_, Tau_, kappa_, p_);
   }

   virtual bool NonTrivialValue(FieldType deriv) const
   {
      return deriv == FieldType::TEMPERATURE;
   }

   double Eval_Func(ElementTransformation &T, const IntegrationPoint &ip)
   {
      if (p_ == 0)
      {
         return kappa_;
      }
      else
      {
         // double temp = T_gf_->GetValue(T);
         double temp = T_->Eval(T, ip);
         MFEM_VERIFY(temp > 0.0, "Temperature must be positive");
         return kappa_ * pow(sqrt(temp/Tau_), p_);
      }
   }

   double Eval_dT(ElementTransformation &T, const IntegrationPoint &ip)
   {
      if (p_ == 0)
      {
         return 0.0;
      }
      else
      {
         // double temp = T_gf_->GetValue(T);
         double temp = T_->Eval(T, ip);
         MFEM_VERIFY(temp > 0.0, "Temperature must be positive");
         return 0.5 * kappa_ * p_ * (1.0 / Tau_) * pow(sqrt(temp/Tau_), p_ - 2);
      }
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   // print the cool banner
   if (mpi.Root()) { display_banner(cout); }

   // 2. Parse command-line options.
   int logging = 0;
   int order = 1;
   int irOrder = -1;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;

   DGParams dg_params;
   dg_params.sigma = -1.0;
   dg_params.kappa = -1.0;

   // Diffusion cofficient is kappa_ * pow(T/Tau_, 0.5 * p_)
   int    p     = 0;
   // double kappa = 1.0;
   double Tau   = 10.0;

   // Heat capacity per unit volume c_p * rho
   // double c_rho = 1.0;

   int ode_solver_type = 1;
   int ode_msr_type = 1;
   int ode_acc_type = 3;
   int ode_rej_type = 2;
   int ode_lim_type = 2;

   int vis_steps = 1;
   double dt = -1.0;
   double t_init = 0.0;
   double t_final = 5.0;

   const char *basename = "Fourier";
   const char *mesh_file = "../../data/periodic-annulus-sector.msh";
   const char *bc_file = "";
   bool zero_start = false;
   bool static_cond = false;
   bool gfprint = true;
   bool visit = true;
   bool visualization = true;

   double tol = -1.0;
   double rho = 1.2;

   double etaScl = 1.0;
   Vector etaVec;

   double diff_eta = 1.0;

   double gamma_acc = 0.9;
   double kI_acc = 1.0 / 15.0;
   double kP_acc = 0.13;
   double kD_acc = 0.2;
   double c_acc = 1.05;

   double gamma_rej = 0.9;
   double kI_rej = 0.2;
   double kP_rej = 0.0;
   double kD_rej = 0.2;
   double c_rej = 0.95;

   double lim_lo  = 1.0;
   double lim_hi  = 1.2;
   double lim_max = 2.0;

   bool ode_epus = true;

   int term_flag = 31;
   int vis_flag = 31;

   bool gnuplot = true;

   OptionsParser args(argc, argv);
   args.AddOption(&logging, "-l", "--logging",
                  "Set the logging level.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&bc_file, "-bc", "--bc-file",
                  "Boundary condition input file.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&dg_params.sigma, "-dg-s", "--dg-sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&dg_params.kappa, "-dg-k", "--dg-kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&irOrder, "-iro", "--int-rule-order",
                  "Integration Rule Order.");
   args.AddOption(&c_rho_, "-c-rho", "--heat-capacity",
                  "Heat capacity per unit volume (c_p * rho)");
   args.AddOption(&kappa_, "-kappa", "--diff-coef",
                  "Linear diffusion coefficient k in (k*sqrt(T/Tau)^p)");
   args.AddOption(&Tau, "-Tau", "--nl-temp-scale",
                  "Nonlinear diffusion scale factor (k*sqrt(T/Tau)^p)");
   args.AddOption(&p, "-p", "--nl-temp-pow",
                  "Nonlinear diffusion power (k*sqrt(T/Tau)^p)");
   args.AddOption(&TInf_, "-T-inf", "--T-at-infinity",
                  "T(0) far from center.");
   args.AddOption(&TMin_, "-T-min", "--T-minimum",
                  "Min(T(0))");
   args.AddOption(&TMax_, "-T-max", "--T-maximum",
                  "Max(T(0))");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&t_final, "-tf", "--final-time",
                  "Final Time.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Tolerance used to determine convergence to steady state.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - SDIRK 212, 2 - SDIRK 534.");
   args.AddOption(&ode_msr_type, "-err", "--error-measure",
                  "Error measure:\n"
                  "\t   1 - Absolute/Relative Error with Infinity-Norm\n"
                  "\t   2 - Absolute/Relative Error with 2-Norm\n");
   args.AddOption(&ode_acc_type, "-acc", "--accept-factor",
                  "Adjustment factor after accepted steps:\n"
                  "\t   0 - Constant adjustment factor (default 1.05)\n"
                  "\t   1 - Standard error (integrated and scaled)\n"
                  "\t   2 - Integrated error\n"
                  "\t   3 - Proportional and Integrated errors\n"
                  "\t   4 - Proportional, Integrated, and Derivative errors\n");
   args.AddOption(&ode_rej_type, "-rej", "--reject-factor",
                  "Adjustment factor after rejected steps:\n"
                  "\t   0 - Constant adjustment factor (default 0.95)\n"
                  "\t   1 - Standard error (integrated and scaled)\n"
                  "\t   2 - Integrated error\n"
                  "\t   3 - Proportional and Integrated errors\n"
                  "\t   4 - Proportional, Integrated, and Derivative errors\n");
   args.AddOption(&ode_lim_type, "-lim", "--limiter",
                  "Adjustment limiter:\n"
                  "\t   1 - Dead zone limiter\n"
                  "\t   2 - Maximum limiter");
   args.AddOption(&etaScl, "-eta", "--eta-scalar",
                  "Constant for denominator of relative error measure.");
   args.AddOption(&etaVec, "-eta-vec", "--eta-vector",
                  "Vector for denominator of relative error measure.");
   args.AddOption(&kP_acc, "-kPa", "--k-P-acc",
                  "Proportional gain for accepted steps.");
   args.AddOption(&kI_acc, "-kIa", "--k-I-acc",
                  "Integral gain for accepted steps.");
   args.AddOption(&kD_acc, "-kDa", "--k-D-acc",
                  "Derivative gain for accepted steps.");
   args.AddOption(&c_acc, "-cfa", "--const-factor-acc",
                  "Constant adjustment factor, must be > 1 "
                  "(only used with -acc 0).");
   args.AddOption(&kP_rej, "-kPr", "--k-P-rej",
                  "Proportional gain for rejected steps.");
   args.AddOption(&kI_rej, "-kIr", "--k-I-rej",
                  "Integral gain for rejected steps.");
   args.AddOption(&kD_rej, "-kDr", "--k-D-rej",
                  "Derivative gain for rejected steps.");
   args.AddOption(&c_rej, "-cfr", "--const-factor-rej",
                  "Constant adjustment factor, must be < 1 "
                  "(only used with -rej 0).");
   args.AddOption(&lim_lo, "-lo", "--lower-limit",
                  "Lower limit of dead zone.");
   args.AddOption(&lim_hi, "-hi", "--upper-limit",
                  "Upper limit of dead zone.");
   args.AddOption(&lim_max, "-max", "--max-limit",
                  "Limiter maximum.");
   args.AddOption(&rho, "-rho", "--rejection",
                  "Rejection tolerance.");
   args.AddOption(&diff_eta, "-eta", "--error-scaling",
                  "Error is |r/(|y|+eta)|.");
   args.AddOption(&ode_epus, "-epus", "--error-per-unit-step",
                  "-eps", "--error-per-step",
                  "Select Error per step or error per unit step.");
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
   args.AddOption(&gnuplot, "-gp", "--gnuplot", "-no-gp", "--no-gnuplot",
                  "Enable or disable GnuPlot visualization.");
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
   if (dg_params.kappa < 0.0)
   {
      dg_params.kappa = (double)(order+1)*(order+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (irOrder < 0)
   {
      irOrder = std::max(4, 2 * order - 2);
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

   aVec_.resize(dim);
   if (strstr(mesh_file, "centered-segment") != NULL)
   {
      aVec_[0].SetSize(1);
      aVec_[0][0] = 2.0;
   }
   else if (strstr(mesh_file, "square") != NULL)
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
   else if (strstr(mesh_file, "cube") != NULL)
   {
      aVec_[0].SetSize(3);
      aVec_[1].SetSize(3);
      aVec_[2].SetSize(3);
      aVec_[0][0] = 2.0; aVec_[0][1] = 0.0; aVec_[0][2] = 0.0;
      aVec_[1][0] = 0.0; aVec_[1][1] = 2.0; aVec_[1][2] = 0.0;
      aVec_[2][0] = 0.0; aVec_[2][1] = 0.0; aVec_[2][2] = 2.0;
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
   ODEController ode_controller;

   ODEEmbeddedSolver        * ode_solver   = NULL;
   ODERelativeErrorMeasure  * ode_err_msr  = NULL;
   ODEStepAdjustmentFactor  * ode_step_acc = NULL;
   ODEStepAdjustmentFactor  * ode_step_rej = NULL;
   ODEStepAdjustmentLimiter * ode_step_lim = NULL;

   switch (ode_solver_type)
   {
      case 1: ode_solver = new SDIRK212Solver; break;
      case 2: ode_solver = new SDIRK534Solver; break;
   }
   switch (ode_msr_type)
   {
      case 1:
         ode_err_msr = (etaVec.Size() == 0) ?
                       new ParMaxAbsRelDiffMeasure(MPI_COMM_WORLD, etaScl) :
                       new ParMaxAbsRelDiffMeasure(MPI_COMM_WORLD, etaVec);
         break;
      case 2:
         ode_err_msr = (etaVec.Size() == 0) ?
                       new ParL2AbsRelDiffMeasure(MPI_COMM_WORLD, etaScl) :
                       new ParL2AbsRelDiffMeasure(MPI_COMM_WORLD, etaVec);
         break;
      default:
         cout << "Unknown difference measure type: " << ode_msr_type << '\n';
         return 3;
   }
   switch (ode_acc_type)
   {
      case 0:
         ode_step_acc = new ConstantAcceptFactor(c_acc);
         break;
      case 1:
         ode_step_acc = new StdAdjFactor(gamma_acc, kI_acc);
         break;
      case 2:
         ode_step_acc = new IAdjFactor(kI_acc);
         break;
      case 3:
         ode_step_acc = new PIAdjFactor(kP_acc, kI_acc);
         break;
      case 4:
         ode_step_acc = new PIDAdjFactor(kP_acc, kI_acc, kD_acc);
         break;
      default:
         cout << "Unknown adjustment factor type: " << ode_acc_type << '\n';
         return 3;
   }
   switch (ode_rej_type)
   {
      case 0:
         ode_step_rej = new ConstantRejectFactor(c_rej);
         break;
      case 1:
         ode_step_rej = new StdAdjFactor(gamma_rej, kI_rej);
         break;
      case 2:
         ode_step_rej = new IAdjFactor(kI_rej);
         break;
      case 3:
         ode_step_rej = new PIAdjFactor(kP_rej, kI_rej);
         break;
      case 4:
         ode_step_rej = new PIDAdjFactor(kP_rej, kI_rej, kD_rej);
         break;
      default:
         cout << "Unknown adjustment factor type: " << ode_rej_type << '\n';
         return 3;
   }
   switch (ode_lim_type)
   {
      case 1:
         ode_step_lim = new DeadZoneLimiter(lim_lo, lim_hi, lim_max);
         break;
      case 2:
         ode_step_lim = new MaxLimiter(lim_max);
         break;
      default:
         cout << "Unknown adjustment limiter type: " << ode_lim_type << '\n';
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

   // H1 contains continuous "node-centered" Lagrange finite elements.
   H1_FECollection HGrad_FEC(order, dim);
   DG_FECollection DG_FEC(order, dim);

   ParFiniteElementSpace L2FESpace0(pmesh, &L2FEC0);
   ParFiniteElementSpace  L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace    fespace(pmesh, &DG_FEC);

   // The terminology is TrueVSize is the unique (non-redundant) number of dofs
   // HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   // HYPRE_Int glob_size_rt = HDivFESpace.GlobalTrueVSize();
   HYPRE_Int glob_size = fespace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of Temperature unknowns:       " << glob_size << endl;
   }

   ParGridFunction Q(&L2FESpace);
   ParGridFunction T1(&fespace);
   ParGridFunction ExactT(&fespace);
   ParGridFunction errorT(&L2FESpace0);
   T1 = 0.0;

   ParGridFunction yGF(&fespace, (double*)NULL);
   ParGridFunction kGF(&fespace, (double*)NULL);

   GridFunctionCoefficient yCoef(&yGF);
   GridFunctionCoefficient kCoef(&kGF);
   SumCoefficient ykCoef(yCoef, kCoef, 1.0, dt);

   // 13. Get the boundary conditions, set up the exact solution grid functions
   //     These VectorCoefficients have an Eval function.  Note that e_exact and
   //     b_exact in this case are exact analytical solutions, taking a 3-vector
   //     point as input and returning a 3-vector field
   FunctionCoefficient TCoef(TFunc);
   TCoef.SetTime(0.0);

   Vector zeroVec(dim); zeroVec = 0.0;
   ConstantCoefficient zeroCoef(0.0);
   ConstantCoefficient oneCoef(1.0);
   VectorConstantCoefficient zeroVecCoef(zeroVec);

   ConstantStateVariableCoef HeatCapacityCoef(c_rho_);
   DiffusionCoef             ThermalConductivityCoef(ykCoef, Tau, kappa_, p);
   ConstantStateVariableCoef HeatSourceCoef(0.0);

   Q.ProjectCoefficient(HeatSourceCoef);

   if (!zero_start)
   {
      T1.ProjectCoefficient(TCoef);
   }

   T1.GridFunction::ComputeElementL2Errors(TCoef, errorT);
   ExactT.ProjectCoefficient(TCoef);

   if (mpi.Root())
   {
      cout << "Configuring boundary conditions" << endl;
   }
   CoefFactory coefFact;
   AdvectionDiffusionBC bcs(pmesh->bdr_attributes);
   if (strncmp(bc_file,"",1) != 0)
   {
      if (mpi.Root())
      {
         cout << "Reading boundary conditions from " << bc_file << endl;
      }
      ifstream bcfs(bc_file);
      bcs.LoadBCs(coefFact, bcfs);
   }

   DGAdvectionDiffusionTDO oper(mpi, dg_params, fespace,
                                yGF, kGF, ykCoef, bcs,
                                term_flag, vis_flag, false, logging);

   oper.SetHeatCapacityCoef(HeatCapacityCoef);
   oper.SetConductivityCoef(ThermalConductivityCoef);
   oper.SetHeatSourceCoef(HeatSourceCoef);

   Visualizer vis_T;
   Visualizer vis_ExactT;
   Visualizer vis_ErrorT;

   int Wx = 0, Wy = 0; // window position
   int Ww = 400, Wh = 270; // window size

   int offx = Ww+3/*, offy = Wh+25*/; // window offsets
   const char * h1_keys = "mmaaAc";
   const char * l2_keys = "aaAc";

   if (visualization)
   {
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_T.SetTitle("Temperature (t = 0.0)");
      vis_T.SetGeometry(Wx, Wy, Ww, Wh);
      vis_T.SetKeys(h1_keys);
      vis_T.SetPrecision(8);
      vis_T.VisField(T1);

      Wx += offx;
      vis_ExactT.SetTitle("Exact Temperature (t = 0.0)");
      vis_ExactT.SetGeometry(Wx, Wy, Ww, Wh);
      vis_ExactT.SetKeys(h1_keys);
      vis_ExactT.SetPrecision(8);
      vis_ExactT.VisField(ExactT);

      Wx += offx;
      vis_ErrorT.SetTitle("Error in Temperature (t = 0.0)");
      vis_ErrorT.SetGeometry(Wx, Wy, Ww, Wh);
      vis_ErrorT.SetKeys(l2_keys);
      vis_ErrorT.SetPrecision(8);
      vis_ErrorT.VisField(errorT);
   }
   // VisIt visualization
   VisItDataCollection visit_dc(basename, pmesh);
   if ( visit )
   {
      visit_dc.RegisterField("T", &T1);
      visit_dc.RegisterField("Exact T", &ExactT);
      visit_dc.RegisterField("L2 Error T", &errorT);

      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   {
      double h_min, h_max, kappa_min, kappa_max;
      pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      double dt_cfl = h_min * h_min / kappa_;

      if (dt < 0.0)
      {
         dt = dt_cfl;
      }

      cout << "dt = " << dt << ", dt_cfl = " << dt_cfl << endl;
   }

   // 15. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt). The object oper is the MagneticDiffusionOperator which
   //     has a Mult() method and an ImplicitSolve() method which are used by
   //     the time integrators.
   ofstream ofs_err("fourier_pid.err");

   double t = 0.0;

   double nrm0 = T1.ComputeL2Error(zeroCoef);
   double err0 = T1.ComputeL2Error(TCoef);

   if (tol < 0.0)
   {
      tol = err0 / nrm0;
   }

   if (myid == 0)
   {
      cout << t << " L2 Relative Error of Initial Condition: "
           << err0 / nrm0
           << '\t' << err0 << '\t' << nrm0
           << endl;
      ofs_err << t << "\t" << err0 / nrm0 << "\t"
              << endl;
   }

   ode_solver->Init(oper);

   ode_controller.Init(*ode_solver, *ode_err_msr,
                       *ode_step_acc, *ode_step_rej, *ode_step_lim);

   ode_controller.SetOutputFrequency(vis_steps);
   ode_controller.SetTimeStep(dt);
   ode_controller.SetTolerance(tol);
   ode_controller.SetRejectionLimit(rho);
   if (ode_epus) { ode_controller.SetErrorPerUnitStep(); }

   ofstream ofs_gp;

   if (gnuplot)
   {
      ofs_gp.open("fourier_pid.dat");
      ode_controller.SetOutput(ofs_gp);
   }

   t = t_init;
   while (t < t_final)
   {
      ode_controller.Run(T1, t, t_final);

      T1.ExchangeFaceNbrData();
      TCoef.SetTime(t);
      double nrm1 = T1.ComputeL2Error(oneCoef);
      double err1 = T1.ComputeL2Error(TCoef);

      if (myid == 0)
      {
         cout << t << " L2 Relative Error of Solution: " << err1 / nrm1
              << endl;
         ofs_err << t << "\t" << err1 / nrm1 << "\t"
                 << endl;
      }
      if (visualization)
      {
         T1.GridFunction::ComputeElementL2Errors(TCoef, errorT);
         ExactT.ProjectCoefficient(TCoef);
         ExactT.ExchangeFaceNbrData();

         ostringstream ossT, ossX, ossE;
         ossT << "Temperature (t = " << t << ")";
         ossX << "Exact Temperature (t = " << t << ")";
         ossE << "Error in T (t = " << t << ")";

         vis_T.SetTitle(ossT.str());
         vis_ExactT.SetTitle(ossX.str());
         vis_ErrorT.SetTitle(ossE.str());

         vis_T.VisField(T1);
         vis_ExactT.VisField(ExactT);
         vis_ErrorT.VisField(errorT);
      }
   }

   ofs_err.close();
   ofs_gp.close();

   if (visualization)
   {
      vis_T.Close();
      vis_ExactT.Close();
      vis_ErrorT.Close();
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
      << "    \\/                              \\/       , not..." << endl
      << flush;
}
