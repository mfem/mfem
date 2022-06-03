//
// Compile with: make shell1d
//
// Description: it solves a 1D equation  
//                      u_tt = -u_xxxx
//              using an auxiliary variable V=-u_xx and the Newmark-Beta integrator
//  MMS
//  u = cos(t)*sin(x) with x in [0, 2pi]

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//1D MMS
double InitialSolution(const Vector &pt)
{
   double x = pt(0);
   return sin(x);
}

double InitialRate(const Vector &pt)
{
   return 0.;
}

double ExactSolution(const Vector &pt, const double t)
{
   double x = pt(0);
   return cos(t)*sin(x);
}



class ShellOperator : public SecondOrderTimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; 

   BilinearForm *M, *K;
   SparseMatrix Mmat, Mmat0, Kmat, *S, *Sc;
   Solver *invM, *invS;
   double current_dt, fac0old;
   Array<int> block_trueOffsets;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   GMRESSolver solver;  // the block system is not symmetric 

   BlockOperator *BlockSystem;
   BlockDiagonalPreconditioner *prec;
   BlockVector *vx, *rhs;

   mutable Vector z, V; // auxiliary vector

public:
   ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr);

   using SecondOrderTimeDependentOperator::Mult;
   virtual void Mult(const Vector &u, const Vector &du_dt,
                     Vector &d2udt2) const;
   
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &u, const Vector &dudt, Vector &d2udt2);

   virtual ~ShellOperator();
};


ShellOperator::ShellOperator(FiniteElementSpace &f, Array<int> &ess_bdr)
   : SecondOrderTimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL),
     K(NULL), current_dt(0.0), block_trueOffsets(3), z(height), V(height) 
{
   const double rel_tol = 1e-8;
   ConstantCoefficient one(1);

   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(one));
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   fac0old=0.;

   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = height;
   block_trueOffsets[2] = height;
   block_trueOffsets.PartialSum();

   BlockSystem = new BlockOperator(block_trueOffsets);
   BlockSystem->SetBlock(0,0, &Mmat);
   BlockSystem->SetBlock(0,1, &Kmat);
   BlockSystem->SetBlock(1,0, &Kmat, -1.);

   // Define a block diagonal preconditioner
   //                 P = [ diag(M)            0             ]
   //                     [  0       M/fac0 + K diag(M)^-1 K ]
   prec = new BlockDiagonalPreconditioner(block_trueOffsets);

   Vector Md(height);
   Mmat.GetDiag(Md);
   SparseMatrix tmp = Kmat;
   for (int i = 0; i < Md.Size(); i++)
   {
      tmp.ScaleRow(i, 1./Md(i));
   }
   Sc = mfem::Mult(Kmat, tmp);

   invM = new DSmoother(Mmat);
   invM->iterative_mode = false;

   prec->SetDiagonalBlock(0, invM);

   rhs = new BlockVector(block_trueOffsets);
   vx = new BlockVector(block_trueOffsets);
   invS=NULL;
}

void ShellOperator::Mult(const Vector &u, const Vector &du_dt,
                        Vector &d2udt2)  const
{
   // Compute:
   //    V = M^{-1}*K(v)
   //    d2udt2 = M^{-1}*[-K V]
   Kmat.Mult(u, z);
   M_solver.Mult(z, V);

   Kmat.Mult(V, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, d2udt2);
}

void ShellOperator::ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Solve the equation:
   //    d2udt2 = M^{-1}*[-K V]
   // where
   //         V = M^{-1}*[K(u+fac0*d2udt2)]
   // 
   // Need to solve the 2x2 system:
   // | M   K      ||a| = | 0      |
   // | -K  M/fac0 ||V|   | Ku/fac0|
   
   if (fac0old<1e-14)
   {
      fac0old=fac0;
      Mmat0=Mmat;
      Mmat0*=(1./fac0);
      BlockSystem->SetBlock(1,1, &Mmat0);

      S = Add(1.0, Mmat0, 1., *Sc);
#ifndef MFEM_USE_SUITESPARSE
      invS = new GSSmoother(*S);
#else
      invS = new UMFPackSolver(*S);
#endif
      invS->iterative_mode = false;
      prec->SetDiagonalBlock(1, invS);

      solver.iterative_mode = false;
      solver.SetAbsTol(0.0);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(100);
      solver.SetOperator(*BlockSystem);
      solver.SetPreconditioner(*prec);
      solver.SetPrintLevel(0);
   }
   else if(fac0!=fac0old)
   {
      cout << " fac0 changes..."<<endl;
      fac0old=fac0;
      Mmat0=Mmat;
      Mmat0*=(1./fac0);
      BlockSystem->SetBlock(1,1, &Mmat0);

      delete invS;
      delete S;
      S = Add(1.0, Mmat0, 1., *Sc);
#ifndef MFEM_USE_SUITESPARSE
      invS = new GSSmoother(*S);
#else
      invS = new UMFPackSolver(*S);
#endif
      invS->iterative_mode = false;
      prec->SetDiagonalBlock(1, invS);

      solver.SetOperator(*BlockSystem);
      solver.SetPreconditioner(*prec);
   }

   Kmat.Mult(u, z);
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      z[ess_tdof_list[i]] = 0.0;
   }
   z*=(1./fac0);

   *rhs=0.;
   rhs->GetBlock(1)=z;

   solver.Mult(*rhs, *vx);
   d2udt2=vx->GetBlock(0);
}

ShellOperator::~ShellOperator()
{
   delete M;
   delete K;
   delete BlockSystem;
   delete prec;
   delete invM;
   delete invS;
   delete S;
   delete Sc;
   delete vx;
   delete rhs;
}

int main(int argc, char *argv[])
{
   int order = 2;
   double t_final = 2*M_PI;
   double dt = 1e-2;
   bool visualization = true;
   int vis_steps = 2;
   int ref_levels = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(8, 2*M_PI);
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(&mesh, &fe_coll);
   
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
   }

   SecondOrderODESolver *ode_solver = new NewmarkSolver();

   //    Set the initial conditions for u.  
   GridFunction u_gf(&fespace);
   GridFunction dudt_gf(&fespace);

   FunctionCoefficient u_0(InitialSolution);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   FunctionCoefficient dudt_0(InitialRate);
   dudt_gf.ProjectCoefficient(dudt_0);
   Vector dudt;
   dudt_gf.GetTrueDofs(dudt);

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(8);
         sout << "solution\n" << mesh << u_gf;
         sout << "window_size 800 800\n";
         sout << "valuerange -1 1\n";
         sout << "keys cmma\n"; 
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }         

   ShellOperator oper(fespace, ess_bdr);
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {

      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, dudt, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         dudt_gf.SetFromTrueDofs(dudt);
         if (visualization)
         {
            sout << "solution\n" << mesh << u_gf << flush;
         }
      }
   }

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   FunctionCoefficient ucoeff (ExactSolution);
   ucoeff.SetTime(t);
   double err_u  = u_gf.ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeLpNorm(2., ucoeff, mesh, irs);

   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";

   delete ode_solver;
   return 0;
}
