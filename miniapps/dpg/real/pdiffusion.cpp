//                 MFEM UW DPG parallel example
//
// Compile with: make poisson_fosls
//
//     - Δ u = f, in Ω
//         u = 0, on ∂Ω

// First Order System

//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = 0, in ∂Ω

// UW-DPG:
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, σ̂ ∈ H^-1/2  
// -(u , ∇⋅τ) + < û, τ⋅n> - (σ , τ) = 0,      ∀ τ ∈ H(div,Ω)      
//  (σ , ∇ v) - < σ̂, v  >           = (f,v)   ∀ v ∈ H^1(Ω)
//            û = 0        on ∂Ω 

// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | τ | -(u,∇⋅τ)  |  -(σ,τ)   | < û, τ⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | v |           |  (σ,∇ v)  |           | -<σ̂,v>  |  (f,v)  |  

// where (τ,v) ∈  H(div,Ω) × H^1(Ω) 

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob_type
{
   lshape,   
   general  
};

prob_type prob;

double exact(const Vector & X)
{
   double x = X[0];
   double y = X[1];

   double r = sqrt(x*x + y*y);
   double alpha = 2./3.;
   double theta = atan2(y,x);
   if (theta < 0) theta += 2*M_PI;

   return pow(r,alpha) * sin(alpha * theta);
}

void gradexact(const Vector & X, Vector & grad)
{
   grad.SetSize(2);
   double x = X[0];
   double y = X[1];

   double r = sqrt(x*x + y*y);
   double alpha = 2./3.;
   double theta = atan2(y,x);
   if (theta < 0) theta += 2*M_PI;

   double r_x = x/r;
   double r_y = y/r;
   double theta_x = - y / (r*r);
   double theta_y = x / (r*r);
   double beta = alpha * pow(r,alpha - 1.);
   grad[0] = beta*(r_x * sin(alpha*theta) + r * theta_x * cos(alpha*theta));
   grad[1] = beta*(r_y * sin(alpha*theta) + r * theta_y * cos(alpha*theta));
}


int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int ref = 1;
   bool adjoint_graph_norm = false;
   bool visualization = true;
   int iprob = 0;
   bool static_cond = false;
   double theta = 0.7;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&ref, "-ref", "--num_refinements",
                  "Number of uniform refinements");    
   args.AddOption(&theta, "-theta", "--theta_factor",
                  "Refinement factor");                              
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");   
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: lshape, 1: General");       
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
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   if (prob == prob_type::lshape)
   {
      mesh_file = "lshape2.mesh";
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.UniformRefinement();

   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *u_fes = new ParFiniteElementSpace(&pmesh,u_fec);

   // Vector L2 space for σ 
   FiniteElementCollection *sigma_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *sigma_fes = new ParFiniteElementSpace(&pmesh,sigma_fec, dim); 

   // H^1/2 space for û 
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatu_fes = new ParFiniteElementSpace(&pmesh,hatu_fec);

   // H^-1/2 space for σ̂ 
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);   
   ParFiniteElementSpace *hatsigma_fes = new ParFiniteElementSpace(&pmesh,hatsigma_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);

   // Normal equation weak formulation
   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatsigma_fes);

   test_fec.Append(tau_fec);
   test_fec.Append(v_fec);

   ParDPGWeakForm * a = new ParDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(true);

   //  -(u,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),0,0);

   // -(σ,τ) 
   TransposeIntegrator * mass = new TransposeIntegrator(new VectorFEMassIntegrator(negone));
   a->AddTrialIntegrator(mass,1,0);

   // (σ,∇ v)
   TransposeIntegrator * grad = new TransposeIntegrator(new GradientIntegrator(one));
   a->AddTrialIntegrator(grad,1,1);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,0);

   // -<σ̂,v> (sign is included in σ̂)
   a->AddTrialIntegrator(new TraceIntegrator,3,1);

   // test integrators (space-induced norm for H(div) × H1)
   // (∇⋅τ,∇⋅δτ)
   a->AddTestIntegrator(new DivDivIntegrator(one),0,0);
   // (τ,δτ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   // (∇v,∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(one),1,1);
   // (v,δv)
   a->AddTestIntegrator(new MassIntegrator(one),1,1);

   // additional terms for adjoint graph norm
   if (adjoint_graph_norm)
   {
      // -(∇v,δτ) 
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(negone),1,0);
      // -(τ,∇δv) 
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(one),0,1);
      // (τ,δτ)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),0,0);
   }
   // RHS
   if (prob == prob_type::general)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(one),1);
   }

   FunctionCoefficient uex(exact);
   Array<int> elements_to_refine;
   ParGridFunction hatu_gf;


   socketstream u_out;
   socketstream sigma_out;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      u_out.open(vishost, visport);
      sigma_out.open(vishost, visport);
   }

   ConvergenceStudy rates_u;


   for (int iref = 0; iref<ref; iref++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = sigma_fes->GetVSize();
      offsets[3] = hatu_fes->GetVSize();
      offsets[4] = hatsigma_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      if (prob == prob_type::lshape)
      {
         hatu_gf.MakeRef(hatu_fes,x.GetBlock(2));
         hatu_gf.ProjectBdrCoefficient(uex,ess_bdr);
      }

      Vector X,B;
      OperatorPtr Ah;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockOperator * A = Ah.As<BlockOperator>();

      BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(A->RowOffsets());
      M->owns_blocks = 1;
      int skip = 0;
      if (!static_cond)
      {
         HypreBoomerAMG * amg0 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(0,0));
         HypreBoomerAMG * amg1 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(1,1));
         amg0->SetPrintLevel(0);
         amg1->SetPrintLevel(0);
         M->SetDiagonalBlock(0,amg0);
         M->SetDiagonalBlock(1,amg1);
         skip=2;
      }
      HypreBoomerAMG * amg2 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(skip,skip));
      amg2->SetPrintLevel(0);
      M->SetDiagonalBlock(skip,amg2);
      HypreSolver * prec;
      if (dim == 2)
      {
         prec = new HypreAMS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatsigma_fes);
      }
      else
      {
         prec = new HypreADS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatsigma_fes);
      }
      M->SetDiagonalBlock(skip+1,prec);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(3);
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete M;

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();

      double maxresidual = residuals.Max(); 
      double globalresidual = residual * residual; 

      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      if (myid == 0)
      {
         cout << "Global Residual = " << globalresidual << endl;
      }

      elements_to_refine.SetSize(0);
      for (int iel = 0; iel<pmesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * maxresidual)
         {
            elements_to_refine.Append(iel);
         }
      }

      ParGridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(0));

      rates_u.AddL2GridFunction(&u_gf,&uex);

      ParGridFunction sigma_gf;
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      if (visualization)
      {
         u_out << "parallel " << num_procs << " " << myid << "\n";
         u_out.precision(8);
         u_out << "solution\n" << pmesh << u_gf <<
                  "window_title 'Numerical u' "
                  << flush;

         sigma_out << "parallel " << num_procs << " " << myid << "\n";
         sigma_out.precision(8);
         sigma_out << "solution\n" << pmesh << sigma_gf <<
                  "window_title 'Numerical flux' "
                  << flush;
      }


      if (iref == ref-1)
      {
         break;
      }

      pmesh.GeneralRefinement(elements_to_refine);

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   rates_u.Print();

   delete a;
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete sigma_fes;
   delete u_fec;
   delete u_fes;

   return 0;
}