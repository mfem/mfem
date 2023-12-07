//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make pacoustics
//
// sample runs

// mpirun -np 4 pacoustics-primal -o 3 -m ../../data/star.mesh -sref 1 -pref 2 -rnum 1.9 -sc -prob 0
// mpirun -np 4 pacoustics-primal -o 3 -m ../../data/inline-quad.mesh -sref 1 -pref 2  -rnum 5.2 -sc -prob 1
// mpirun -np 4 pacoustics-primal -o 4 -m ../../data/inline-tri.mesh -sref 1 -pref 2  -rnum 7.1 -sc -prob 1
// mpirun -np 4 pacoustics-primal -o 2 -m ../../data/inline-hex.mesh -sref 0 -pref 1 -rnum 1.9 -sc -prob 0

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Helmholtz problem

//     - Δ p - ω² p = f ,   in Ω
//                p = p₀, on ∂Ω

// It solves the following kinds of problems
//    a) f̃ = 0 and p₀ is a plane wave
//    b) A manufactured solution problem where p_exact is a gaussian beam

// The DPG Primal deals with the Second Order Equation
//     - Δ p - ω² p = f , in Ω    (1)
//                p = p₀, on ∂Ω

// The primal-DPG formulation is obtained by integration by parts of (1)
// and the introduction of a trace unknown on the mesh skeleton

// p ∈ H¹(Ω),
// p̂ ∈ H^-1/2(Ω)
// (∇p,∇q) + ω²(p,q) + <p̂,q> = 0,      ∀ q ∈ H^1(Ω)
//                         p = p₀      on ∂Ω

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "util/pml.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

complex<double> acoustics_solution(const Vector & X);
void acoustics_solution_grad(const Vector & X,vector<complex<double>> &dp);
complex<double> acoustics_solution_laplacian(const Vector & X);

double p_exact_r(const Vector &x);
double p_exact_i(const Vector &x);
double rhs_func_r(const Vector &x);
double rhs_func_i(const Vector &x);
void gradp_exact_r(const Vector &x, Vector &gradu);
void gradp_exact_i(const Vector &x, Vector &gradu);
double d2_exact_r(const Vector &x);
double d2_exact_i(const Vector &x);

int dim;
double omega;

enum prob_type
{
   plane_wave,
   gaussian_beam
};

static const char *enum_str[] =
{
   "plane_wave",
   "gaussian_beam"
};

prob_type prob;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   bool static_cond = false;
   int iprob = 0;
   int sr = 0;
   int pr = 0;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: plane wave, 1: Gaussian beam");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sr, "-sref", "--serial-ref",
                  "Number of parallel refinements.");
   args.AddOption(&pr, "-pref", "--parallel-ref",
                  "Number of parallel refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   if (iprob > 1) { iprob = 0; }
   prob = (prob_type)iprob;
   omega = 2.*M_PI*rnum;

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   enum TrialSpace
   {
      p_space     = 0,
      hatp_space  = 1,
   };
   enum TestSpace
   {
      q_space = 0,
   };

   // H1 space for p
   FiniteElementCollection *p_fec = new H1_FECollection(order,dim);
   ParFiniteElementSpace *p_fes = new ParFiniteElementSpace(&pmesh,p_fec);

   // H^-1/2 space for p̂
   FiniteElementCollection * hatp_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatp_fes = new ParFiniteElementSpace(&pmesh,hatp_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * q_fec = new H1_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(p_fes);
   trial_fes.Append(hatp_fes);
   test_fec.Append(q_fec);

   // Bilinear form Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negomeg2(-omega*omega);

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(); // needed for AMR

   // Trial itegrators
   // (∇ p,∇ q)
   a->AddTrialIntegrator(new DiffusionIntegrator(one),nullptr,
                         TrialSpace::p_space,TestSpace::q_space);
   //  ω² (p,q)
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(negomeg2), nullptr,
                         TrialSpace::p_space, TestSpace::q_space);

   // < p̂,q >
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,
                         TrialSpace::hatp_space,TestSpace::q_space);

   // test integrators
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,
                        TestSpace::q_space, TestSpace::q_space);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),nullptr,
                        TestSpace::q_space, TestSpace::q_space);

   // RHS
   FunctionCoefficient f_rhs_r(rhs_func_r);
   FunctionCoefficient f_rhs_i(rhs_func_i);
   if (prob == prob_type::gaussian_beam)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs_r),
                               new DomainLFIntegrator(f_rhs_i),
                               TestSpace::q_space);
   }

   socketstream p_out_r;
   socketstream p_out_i;
   if (myid == 0)
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "    ω    |" ;
      std::cout  << "  H¹ Error  |"
                 << "  Rate  |" ;
      std::cout << "  Residual  |"
                << "  Rate  |"
                << " PCG it |" << endl;
      std::cout << std::string(82,'-')
                << endl;
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0 = 0;

   ParGridFunction p_r, p_i;
   FunctionCoefficient pex_r(p_exact_r);
   FunctionCoefficient pex_i(p_exact_i);
   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(enum_str[prob], &pmesh);
      paraview_dc->SetPrefixPath("ParaViewPrimal/Acoustics");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("p_r",&p_r);
      paraview_dc->RegisterField("p_i",&p_i);
   }

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=pr; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         p_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = p_fes->GetVSize();
      offsets[2] = hatp_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      ParGridFunction p_gf_r(p_fes, x, offsets[0]);
      ParGridFunction p_gf_i(p_fes, x, offsets.Last()+ offsets[0]);
      p_gf_r.ProjectBdrCoefficient(pex_r, ess_bdr);
      p_gf_i.ProjectBdrCoefficient(pex_i, ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();
      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);

      tdof_offsets[0] = 0;
      for (int i=0; i<num_blocks; i++)
      {
         int h = BlockA_r->GetBlock(i,i).Height();
         tdof_offsets[i+1] = h;
         tdof_offsets[num_blocks+i+1] = h;
      }
      tdof_offsets.PartialSum();

      BlockOperator blockA(tdof_offsets);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            blockA.SetBlock(i,j,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
            blockA.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
         }
      }

      X = 0.;
      BlockDiagonalPreconditioner M(tdof_offsets);
      M.owns_blocks=0;

      HypreBoomerAMG * solver_p = new HypreBoomerAMG((HypreParMatrix &)
                                                     BlockA_r->GetBlock(0,0));
      solver_p->SetPrintLevel(0);
      M.SetDiagonalBlock(0,solver_p);
      M.SetDiagonalBlock(num_blocks,solver_p);

      HypreSolver * solver_hatp = nullptr;
      if (dim == 2)
      {
         // AMS preconditioner for 2D H(div) (trace) space
         solver_hatp = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(1,1),
                                    hatp_fes);
         dynamic_cast<HypreAMS*>(solver_hatp)->SetPrintLevel(0);
      }
      else
      {
         // ADS preconditioner for 3D H(div) (trace) space
         solver_hatp = new HypreADS((HypreParMatrix &)BlockA_r->GetBlock(1,1),
                                    hatp_fes);
         dynamic_cast<HypreADS*>(solver_hatp)->SetPrintLevel(0);
      }

      M.SetDiagonalBlock(1,solver_hatp);
      M.SetDiagonalBlock(num_blocks+1,solver_hatp);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-6);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);

      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
      }

      int num_iter = cg.GetNumIterations();

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();
      double maxresidual = residuals.Max();
      double globalresidual = residual * residual;
      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      p_r.MakeRef(p_fes, x, 0);
      p_i.MakeRef(p_fes, x, offsets.Last());

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      double H1Error = 0.0;
      double rate_err = 0.0;
      VectorFunctionCoefficient pex_grad_r(dim,gradp_exact_r);
      VectorFunctionCoefficient pex_grad_i(dim,gradp_exact_i);
      double p_err_r = p_r.ComputeH1Error(&pex_r,&pex_grad_r);
      double p_err_i = p_i.ComputeH1Error(&pex_i,&pex_grad_i);

      H1Error = sqrt(p_err_r*p_err_r + p_err_i*p_err_i);

      rate_err = (it) ? dim*log(err0/H1Error)/log((double)dof0/dofs) : 0.0;
      err0 = H1Error;

      double rate_res = (it) ? dim*log(res0/globalresidual)/log((
                                                                   double)dof0/dofs) : 0.0;

      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(1) << std::fixed
                   << std::setw(4) <<  2*rnum << " π  | ";
         std::cout << std::setprecision(3) << std::setw(10)
                   << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | " ;
         std::cout << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << num_iter << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeField(p_out_r,vishost, visport, p_r,
                        "Numerical presure (real part)", 0, 0, 500, 500, keys);
         VisualizeField(p_out_i,vishost, visport, p_i,
                        "Numerical presure (imaginary part)", 501, 0, 500, 500, keys);
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((double)it);
         paraview_dc->Save();
      }

      if (it == pr)
      {
         break;
      }

      pmesh.UniformRefinement();

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete q_fec;
   delete hatp_fes;
   delete hatp_fec;
   delete p_fec;
   delete p_fes;

   return 0;
}

double p_exact_r(const Vector &x)
{
   return acoustics_solution(x).real();
}

double p_exact_i(const Vector &x)
{
   return acoustics_solution(x).imag();
}

void gradp_exact_r(const Vector &x, Vector &grad_r)
{
   grad_r.SetSize(x.Size());
   vector<complex<double>> grad;
   acoustics_solution_grad(x,grad);
   for (unsigned i = 0; i < grad.size(); i++)
   {
      grad_r[i] = grad[i].real();
   }
}

void gradp_exact_i(const Vector &x, Vector &grad_i)
{
   grad_i.SetSize(x.Size());
   vector<complex<double>> grad;
   acoustics_solution_grad(x,grad);
   for (unsigned i = 0; i < grad.size(); i++)
   {
      grad_i[i] = grad[i].imag();
   }
}

double d2_exact_r(const Vector &x)
{
   return acoustics_solution_laplacian(x).real();
}

double d2_exact_i(const Vector &x)
{
   return acoustics_solution_laplacian(x).imag();
}


// f = -Δ p - ω² p
double rhs_func_r(const Vector &x)
{
   return -d2_exact_r(x) - omega * omega * p_exact_r(x);
}

// f = -Δ p - ω² p
double rhs_func_i(const Vector &x)
{
   return -d2_exact_i(x) - omega * omega * p_exact_i(x);
}

complex<double> acoustics_solution(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         return exp(alpha);
      }
      break;
      case gaussian_beam:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double shift = 0.1;
         double xprim=X(0) + shift;
         double yprim=X(1) + shift;

         double x = xprim*sina - yprim*cosa;
         double y = xprim*cosa + yprim*sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);

         double phi0 = atan(fact*y);

         double r = y + 1./y/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;
         double pf = pow(2.0/M_PI/(w*w),0.25);

         return pf*exp(ze);
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         return 1;
         break;
   }
}

void acoustics_solution_grad(const Vector & X, vector<complex<double>> & dp)
{
   dp.resize(X.Size());
   complex<double> zi = complex<double>(0., 1.);
   // initialize
   for (int i = 0; i<X.Size(); i++) { dp[i] = 0.0; }
   switch (prob)
   {
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         complex<double> p = exp(alpha);
         for (int i = 0; i<X.Size(); i++)
         {
            dp[i] = zi * beta * p;
         }
      }
      break;
      case gaussian_beam:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double shift = 0.1;
         double xprim=X(0) + shift;
         double yprim=X(1) + shift;

         double x = xprim*sina - yprim*cosa;
         double y = xprim*cosa + yprim*sina;
         double dxdxprim = sina, dxdyprim = -cosa;
         double dydxprim = cosa, dydyprim =  sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);
         double dwdy = w0*fact*fact*y/sqrt(aux);

         double phi0 = atan(fact*y);
         double dphi0dy = cos(phi0)*cos(phi0)*fact;

         double r = y + 1./y/(fact*fact);
         double drdy = 1. - 1./(y*y)/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
         complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/
                                 (r*r)*drdy + zi*dphi0dy/2.;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;

         complex<double> zp = pf*exp(ze);
         complex<double> zdpdx = zp*zdedx;
         complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;

         dp[0] = (zdpdx*dxdxprim + zdpdy*dydxprim);
         dp[1] = (zdpdx*dxdyprim + zdpdy*dydyprim);
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         break;
   }
}

complex<double> acoustics_solution_laplacian(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
      case plane_wave:
      {
         double beta = omega/std::sqrt((double)X.Size());
         complex<double> alpha = beta * zi * X.Sum();
         return dim * beta * beta * exp(alpha);
      }
      break;
      case gaussian_beam:
      {
         double rk = omega;
         double degrees = 45;
         double alpha = (180+degrees) * M_PI/180.;
         double sina = sin(alpha);
         double cosa = cos(alpha);
         // shift the origin
         double shift = 0.1;
         double xprim=X(0) + shift;
         double yprim=X(1) + shift;

         double x = xprim*sina - yprim*cosa;
         double y = xprim*cosa + yprim*sina;
         double dxdxprim = sina, dxdyprim = -cosa;
         double dydxprim = cosa, dydyprim =  sina;
         //wavelength
         double rl = 2.*M_PI/rk;

         // beam waist radius
         double w0 = 0.05;

         // function w
         double fact = rl/M_PI/(w0*w0);
         double aux = 1. + (fact*y)*(fact*y);

         double w = w0*sqrt(aux);
         double dwdy = w0*fact*fact*y/sqrt(aux);
         double d2wdydy = w0*fact*fact*(1. - (fact*y)*(fact*y)/aux)/sqrt(aux);

         double phi0 = atan(fact*y);
         double dphi0dy = cos(phi0)*cos(phi0)*fact;
         double d2phi0dydy = -2.*cos(phi0)*sin(phi0)*fact*dphi0dy;

         double r = y + 1./y/(fact*fact);
         double drdy = 1. - 1./(y*y)/(fact*fact);
         double d2rdydy = 2./(y*y*y)/(fact*fact);

         // pressure
         complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r +
                              zi*phi0/2.;

         complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
         complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/
                                 (r*r)*drdy + zi*dphi0dy/2.;
         complex<double> zd2edxdx = -2./(w*w) - 2.*zi*M_PI/rl/r;
         complex<double> zd2edxdy = 4.*x/(w*w*w)*dwdy + 2.*zi*M_PI*x/rl/(r*r)*drdy;
         complex<double> zd2edydx = zd2edxdy;
         complex<double> zd2edydy = -6.*x*x/(w*w*w*w)*dwdy*dwdy + 2.*x*x/
                                    (w*w*w)*d2wdydy - 2.*zi*M_PI*x*x/rl/(r*r*r)*drdy*drdy
                                    + zi*M_PI*x*x/rl/(r*r)*d2rdydy + zi/2.*d2phi0dydy;

         double pf = pow(2.0/M_PI/(w*w),0.25);
         double dpfdy = -pow(2./M_PI/(w*w),-0.75)/M_PI/(w*w*w)*dwdy;
         double d2pfdydy = -1./M_PI*pow(2./M_PI,-0.75)*(-1.5*pow(w,-2.5)
                                                        *dwdy*dwdy + pow(w,-1.5)*d2wdydy);


         complex<double> zp = pf*exp(ze);
         complex<double> zdpdx = zp*zdedx;
         complex<double> zdpdy = dpfdy*exp(ze)+zp*zdedy;
         complex<double> zd2pdxdx = zdpdx*zdedx + zp*zd2edxdx;
         complex<double> zd2pdxdy = zdpdy*zdedx + zp*zd2edxdy;
         complex<double> zd2pdydx = dpfdy*exp(ze)*zdedx + zdpdx*zdedy + zp*zd2edydx;
         complex<double> zd2pdydy = d2pfdydy*exp(ze) + dpfdy*exp(
                                       ze)*zdedy + zdpdy*zdedy + zp*zd2edydy;

         return (zd2pdxdx*dxdxprim + zd2pdydx*dydxprim)*dxdxprim
                + (zd2pdxdy*dxdxprim + zd2pdydy*dydxprim)*dydxprim
                + (zd2pdxdx*dxdyprim + zd2pdydx*dydyprim)*dxdyprim
                + (zd2pdxdy*dxdyprim + zd2pdydy*dydyprim)*dydyprim;
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         return 1;
         break;
   }
}
