//                 MFEM Primal DPG parallel example for diffusion
//
// Compile with: make pdiffusion-primal
//
// Sample runs
// mpirun -np 4 pdiffusion-primal -m ../../data/inline-quad.mesh -o 3 -sref 1 -pref 2

//       - Δ u = f,   in Ω
//           u = u₀, on ∂Ω



// --------------------------------------
// |   |     u     |      σ̂    |  RHS   |
// --------------------------------------
// | v |  (∇u,∇v)  |  -(σ̂ₙ,v)  |  (f,v) |
//
// u ∈ H¹(Ω),   σ̂ₙ ∈ H^-1/2(Τ)


#include "mfem.hpp"
#include "util/pweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

void exact_u(const Vector & X, Vector & u);
void exact_gradu(const Vector & X, DenseMatrix &du);
void exact_laplacian_u(const Vector & X, Vector & d2u);
void f_exact(const Vector & X, Vector & f);


int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int sref = 0; // initial uniform mesh refinements
   int pref = 0; // parallel mesh refinements for AMR
   bool static_cond = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sref, "-sref", "--num-serial-refinements",
                  "Number of initial serial uniform refinements");
   args.AddOption(&pref, "-pref", "--num-parallel-refinements",
                  "Number of AMR refinements");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
      return 1;
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int i = 0; i<sref; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();


   // H1 space for u
   FiniteElementCollection *u_fec = new H1_FECollection(order,dim);
   ParFiniteElementSpace *u_fes = new ParFiniteElementSpace(&pmesh,u_fec,dim);


   // H^-1/2 space for σ̂
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatsigma_fes = new ParFiniteElementSpace(&pmesh,
                                                                   hatsigma_fec,dim);

   ParFiniteElementSpace *hatsigma1_fes = new ParFiniteElementSpace(&pmesh,
                                                                    hatsigma_fec);

   int test_order = order+delta_order;
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;

   trial_fes.Append(u_fes);
   trial_fes.Append(hatsigma_fes);
   test_fec.Append(v_fec);


   ConstantCoefficient one(1.0);
   VectorFunctionCoefficient f(dim,
                               f_exact); // rhs for the manufactured solution problem
   VectorFunctionCoefficient uex(dim,exact_u);
   MatrixFunctionCoefficient graduex(dim,exact_gradu);

   ParDPGWeakForm * a = new ParDPGWeakForm(trial_fes,test_fec);
   a->SetTestFECollVdim(0,dim);
   a->StoreMatrices(true); // this is needed for estimation of residual

   // (∇u,∇v)
   a->AddTrialIntegrator(new VectorDiffusionIntegrator(one),0,0);

   // // -<σ̂,v> (sign is included in σ̂)
   a->AddTrialIntegrator(new VectorTraceIntegrator,1,0);

   // // (∇v,∇δv)
   a->AddTestIntegrator(new VectorDiffusionIntegrator(one),0,0);
   // // (v,δv)
   a->AddTestIntegrator(new VectorMassIntegrator(one),0,0);

   a->AddDomainLFIntegrator(new VectorDomainLFIntegrator(f),0);


   if (myid == 0)
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "  H1 Error  |"
                << "  Rate  |"
                << "  Residual  |"
                << "  Rate  |"
                << " PCG it |" << endl;
      std::cout << std::string(72,'-') << endl;
   }


   socketstream u_out, ex_out;

   double err0 = 0.;
   int dof0=0.;
   double res0=0.0;

   ParGridFunction u_gf(u_fes);
   ParGridFunction exact_gf(u_fes);
   u_gf = 0.0;

   if (static_cond) { a->EnableStaticCondensation(); }

   for (int it = 0; it<=pref; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         u_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = hatsigma_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      u_gf.MakeRef(u_fes,x.GetBlock(0),0);
      u_gf.ProjectBdrCoefficient(uex,ess_bdr);
      exact_gf.ProjectCoefficient(uex);

      Vector X,B;
      OperatorPtr Ah;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockOperator * A = Ah.As<BlockOperator>();



      // BlockDiagonalPreconditioner M(A->RowOffsets());
      BlockDiagonalPreconditioner M(A->RowOffsets());
      M.owns_blocks = 1;

      // HypreBoomerAMG * amg0 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(0,0));
      // amg0->SetSystemsOptions(dim);
      MUMPSSolver * amg0 = new MUMPSSolver(MPI_COMM_WORLD);
      amg0->SetOperator((HypreParMatrix &)A->GetBlock(0,0));


      MUMPSSolver * prec = new MUMPSSolver(MPI_COMM_WORLD);
      prec->SetOperator((HypreParMatrix&)A->GetBlock(1,1));
      M.SetDiagonalBlock(1,prec);

      // Array<int> toffsets(4);
      // toffsets[0] = 0;
      // toffsets[1] = A->RowOffsets()[1];
      // toffsets[2] = (A->RowOffsets()[2]-A->RowOffsets()[1])/dim;
      // toffsets[3] = (A->RowOffsets()[2]-A->RowOffsets()[1])/dim;
      // toffsets.PartialSum();

      // BlockDiagonalPreconditioner M(toffsets);
      // M.owns_blocks = 1;

      // HypreBoomerAMG * amg0 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(0,0));
      // amg0->SetSystemsOptions(dim);

      // HypreParMatrix & A11 = (HypreParMatrix &)A->GetBlock(1,1);
      // Array<int> tdofs1(A11.Height()/2);
      // Array<int> tdofs2(A11.Height()/2);
      // for (int i = 0; i<tdofs1.Size(); i++)
      // {
      //    tdofs1[i] = i;
      //    tdofs2[i] = tdofs1.Size() + i;
      // }

      // HypreParMatrix * S1 = GetSubHypreParMatrix(tdofs1,A11);
      // HypreParMatrix * S2 = GetSubHypreParMatrix(tdofs2,A11);
      // MUMPSSolver * prec1 = new MUMPSSolver(MPI_COMM_WORLD);
      // prec1->SetOperator(*S1);
      // MUMPSSolver * prec2 = new MUMPSSolver(MPI_COMM_WORLD);
      // prec2->SetOperator(*S2);
      // HypreSolver * prec1;
      // HypreSolver * prec2;
      // if (dim == 2)
      // {
      //    // AMS preconditioner for 2D H(div) (trace) space
      //    prec1 = new HypreAMS(*S1, hatsigma1_fes);
      //    prec2 = new HypreAMS(*S2, hatsigma1_fes);

      // }
      // else
      // {
      //    // ADS preconditioner for 3D H(div) (trace) space
      //    prec1 = new HypreADS(*S1, hatsigma1_fes);
      //    prec2 = new HypreADS(*S2, hatsigma1_fes);
      // }
      // M.SetDiagonalBlock(1,prec1);
      // M.SetDiagonalBlock(2,prec2);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(*A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);
      // Vector residuals;

      double residual = residuals.Norml2();

      double maxresidual = residuals.Max();
      double globalresidual = residual * residual;

      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      u_gf.MakeRef(u_fes,x.GetBlock(0),0);

      int dofs = u_fes->GlobalTrueVSize() + hatsigma_fes->GlobalTrueVSize();

      double u_err = u_gf.ComputeL2Error(uex);

      double rate_err = (it) ? dim*log(err0/u_err)/log((double)dof0/dofs) : 0.0;
      double rate_res = (it) ? dim*log(res0/globalresidual)/log((
                                                                   double)dof0/dofs) : 0.0;
      err0 = u_err;
      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | "
                   << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << cg.GetNumIterations() << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcm\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;

         VisualizeField(u_out,vishost,visport,u_gf,
                        "Numerical u", 0,0,500,500,keys);

         VisualizeField(ex_out,vishost,visport,exact_gf,
                        "Exact u", 0,0,500,500,keys);
      }

      if (it == pref) { break; }

      pmesh.UniformRefinement();

      for (int i=0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      hatsigma1_fes->Update(false);
      a->Update();
      exact_gf.Update();
   }

   delete a;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete u_fec;
   delete u_fes;

   return 0;

}



void exact_u(const Vector & X, Vector & u)
{
   double alpha = M_PI * (X.Sum());
   int size = X.Size();
   for (int i = 0; i<size; i++)
   {
      u(i) = sin(M_PI*X(i));
   }

}

void exact_gradu(const Vector & X, DenseMatrix & du)
{
   du.SetSize(X.Size());
   for (int i = 0; i<du.Size(); i++)
   {
      du(i,i) = M_PI * cos(M_PI * X(i));
   }
}

void exact_laplacian_u(const Vector & X, Vector & du)
{
   int size = X.Size();
   du.SetSize(size);
   for (int i = 0; i<size; i++)
   {
      du(i) = -M_PI*M_PI * sin(M_PI*X(i));
   }
}

void f_exact(const Vector & X, Vector & f)
{
   f.SetSize(X.Size());
   exact_laplacian_u(X,f);
   f.Neg();
}
