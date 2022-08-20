//                                MFEM Ultraweak DPG example
//
// Compile with: make  uw_dpgp
//
// sample runs 
// mpirun -np 6 ./pconvection-diffusion  -o 3 -ref 0 -prob 0 -eps 1e-2 -beta '2 3'
// mpirun -np 6 ./pconvection-diffusion  -o 2 -ref 3 -prob 0 -eps 1e-1 -beta '4 2' -theta 0.0
// mpirun -np 6 ./pconvection-diffusion  -o 3 -ref 3 -prob 1 -eps 1e-3 -beta '1 0' -theta 0.0 -test-norm 2

// AMR runs
// mpirun -np 6 ./pconvection-diffusion  -o 3 -ref 15 -prob 2 -eps 1e-3 -beta '1 0' -theta 0.7 -test-norm 2 -sc
// mpirun -np 6 ./pconvection-diffusion  -o 3 -ref 20 -prob 3 -eps 5e-3 -theta 0.7 -test-norm 2 -sc
// mpirun -np 6 ./pconvection-diffusion  -o 2 -ref 20 -prob 4 -eps 1e-2 -beta '1 2' -theta 0.7 -test-norm 2 -sc

//     - εΔu + ∇⋅(βu) = f,   in Ω
//                  u = u_0, on ∂Ω

// First Order System

//     - ∇⋅σ + ∇⋅(βu) = f,   in Ω
//        1/ε σ - ∇u  = 0,   in Ω
//                  u = u_0, on ∂Ω

// UW-DPG:
// 
// u ∈ L^2(Ω), σ ∈ (L^2(Ω))^dim 
// û ∈ H^1/2, f̂ ∈ H^-1/2  
// -(βu , ∇v)  + (σ , ∇v)     + < f̂ ,  v  > = (f,v),   ∀ v ∈ H^1(Ω)      
//   (u , ∇⋅τ) + 1/ε (σ , τ)  + < û , τ⋅n > = 0,       ∀ τ ∈ H(div,Ω)
//                                        û = u_0  on ∂Ω 

// Note: 
// f̂ := βu - σ
// û := -u

// -------------------------------------------------------------
// |   |     u     |     σ     |   û       |     f̂    |  RHS    |
// -------------------------------------------------------------
// | v |-(βu , ∇v) | (σ , ∇v)  |           | < f̂ ,v > |  (f,v)  |
// |   |           |           |           |          |         |
// | τ | (u ,∇⋅τ)  | 1/ε(σ , τ)|  <û,τ⋅n>  |          |    0    |  

// where (v,τ) ∈  H^1(Ω_h) × H(div,Ω_h)  

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

enum prob_type
{
   polynomial,
   sinusoidal, 
   EJ, // from https://www.sciencedirect.com/science/article/pii/S0898122113003751
   curved_streamlines, // form https://par.nsf.gov/servlets/purl/10160205
   bdr_layer // from https://web.pdx.edu/~gjay/pub/dpg2.pdf
};

enum test_norm_type
{
   standard,   
   adjoint_graph,
   robust  
};

prob_type prob;
test_norm_type test_norm;
Vector beta;
double epsilon;
// Function returns the solution u, and gradient du and the Laplacian d2u
void solution(const Vector & x, double & u, Vector & du, double & d2u);
double exact_u(const Vector & X);
void exact_sigma(const Vector & X, Vector & sigma);
double exact_hatu(const Vector & X);
void exact_hatf(const Vector & X, Vector & hatf);
double f_exact(const Vector & X);
double bdr_data(const Vector &X);
void beta_function(const Vector & X, Vector & beta_val);

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int ref = 1;
   bool visualization = true;
   int iprob = 0;
   int itest_norm = 0;
   double theta = 0.7;
   bool static_cond = false;
   epsilon = 1e0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&epsilon, "-eps", "--epsilon",
                  "Epsilon coefficient");               
   args.AddOption(&ref, "-ref", "--num_refinements",
                  "Number of uniform refinements");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                           
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: lshape, 1: General");                         
   args.AddOption(&itest_norm, "-test-norm", "--test-norm", "Choice of test norm"
                  " 0: Standard, 1: Adjoint Graph, 2: Robust");     
   args.AddOption(&beta, "-beta", "--beta",
                  "Vector Coefficient beta");          
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

   if (iprob > 4) { iprob = 4; }
   prob = (prob_type)iprob;

   test_norm = (test_norm_type)itest_norm;

   if (prob == prob_type::EJ || prob == prob_type::curved_streamlines || 
       prob == prob_type::bdr_layer)
   {
      mesh_file = "../../../data/inline-quad.mesh";
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   bool exact_known = true;

   switch (prob)
   {
   case polynomial:
   case sinusoidal:
   case EJ:
   {
      if (beta.Size() == 0)
      {
         beta.SetSize(dim);
         beta[0] = 1.;
         beta[1] = 0.;
      }
      break;
   }
   case bdr_layer:
   {
      beta.SetSize(dim);
      beta[0] = 1.;
      beta[1] = 2.;
      exact_known = false;
   }
   break;
   default:
      // do nothing; beta is defined as a FunctionCoefficient
      break;
   }

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
   FiniteElementCollection * hatf_fec = new RT_Trace_FECollection(order-1,dim);   
   ParFiniteElementSpace *hatf_fes = new ParFiniteElementSpace(&pmesh,hatf_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient eps(epsilon);
   ConstantCoefficient eps1(1./epsilon);
   ConstantCoefficient negeps1(-1./epsilon);
   ConstantCoefficient eps2(1/(epsilon*epsilon));
   ConstantCoefficient negeps(-epsilon);


   VectorFunctionCoefficient betacoeff(dim,beta_function);
   ScalarVectorProductCoefficient negbetacoeff(-1.0,betacoeff);
   OuterProductCoefficient bbtcoeff(betacoeff,betacoeff);

   // Normal equation weak formulation
   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatf_fes);
   test_fec.Append(v_fec);
   test_fec.Append(tau_fec);

   ParDPGWeakForm * a = new ParDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(true);

   //-(βu , ∇v)
   a->AddTrialIntegrator(new MixedScalarWeakDivergenceIntegrator(betacoeff),0,0);

   // (σ,∇ v)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),1,0);

   // (u ,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(negone),0,1);

   // 1/ε (σ,τ)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(eps1)),1,1);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,1);

   // <f̂ ,v> 
   a->AddTrialIntegrator(new TraceIntegrator,3,0);


   FiniteElementCollection *coeff_fec = new L2_FECollection(0,dim);
   ParFiniteElementSpace *coeff_fes = new ParFiniteElementSpace(&pmesh,coeff_fec);
   ParGridFunction c1_gf, c2_gf;
   GridFunctionCoefficient c1_coeff(&c1_gf);
   GridFunctionCoefficient c2_coeff(&c2_gf);

   switch (test_norm)
   {
      case standard:
      {   
         if (myid == 0)
         {
            mfem::out << "\n Test norm: Standard" << endl;
         }
         // (∇v,∇δv)
         a->AddTestIntegrator(new DiffusionIntegrator(one),0,0);
         // (v,δv)
         a->AddTestIntegrator(new MassIntegrator(one),0,0);
         // (∇⋅τ,∇⋅δτ)
         a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
         // (τ,δτ)
         a->AddTestIntegrator(new VectorFEMassIntegrator(one),1,1);
      }
      break;
      case adjoint_graph:
      {
         if (myid == 0)
         {
            mfem::out << "\n Test norm: Adjoint Graph" << endl;
         }
         // (∇v,∇δv)
         a->AddTestIntegrator(new DiffusionIntegrator(one),0,0);
         // (β⋅∇v, β⋅∇δv)   
         a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff), 0,0);
         // (v,δv)
         a->AddTestIntegrator(new MassIntegrator(one),0,0);
         // (∇⋅τ,∇⋅δτ)
         a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
         // (τ,δτ)
         a->AddTestIntegrator(new VectorFEMassIntegrator(one),1,1);
         // 1/ε^2 (τ,δτ)
         a->AddTestIntegrator(new VectorFEMassIntegrator(eps2),1,1);
         // 1/ε (∇v, δτ)
         a->AddTestIntegrator(new MixedVectorGradientIntegrator(eps1),0,1);
         // - (β ⋅ ∇v,∇⋅δτ)  
         a->AddTestIntegrator(new MixedGradDivIntegrator(betacoeff),0,1);
         // 1/ε (τ,∇δv) 
         a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(negeps1),1,0);
         // -(β ∇⋅τ ,∇⋅δv)  
         a->AddTestIntegrator(new MixedDivGradIntegrator(betacoeff),1,0);
      }
      break;
      default:
      {
         if (myid == 0)
         {
            mfem::out << "\n Test norm: Robust" << endl;
         }
         c1_gf.SetSpace(coeff_fes);
         c2_gf.SetSpace(coeff_fes);
         Array<int> dofs;
         for (int i = 0; i < pmesh.GetNE(); i++)
         {
            double volume = pmesh.GetElementVolume(i);
            double c1 = min(epsilon/volume, 1.);
            double c2 = min(1./epsilon, 1./volume);
            coeff_fes->GetElementDofs(i,dofs);
            c1_gf.SetSubVector(dofs,c1);
            c2_gf.SetSubVector(dofs,c2);
         }
         // c1 (v,δv)
         a->AddTestIntegrator(new MassIntegrator(c1_coeff),0,0);
         // ε (∇v,∇δv)
         a->AddTestIntegrator(new DiffusionIntegrator(eps),0,0);
         // (β⋅∇v, β⋅∇δv)   
         a->AddTestIntegrator(new DiffusionIntegrator(bbtcoeff), 0,0);
         // c2 (τ,δτ)
         a->AddTestIntegrator(new VectorFEMassIntegrator(c2_coeff),1,1);
         // (∇⋅τ,∇⋅δτ)
         a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
      }
      break;
   }


   FunctionCoefficient f(f_exact);
   if (prob == prob_type::sinusoidal || 
       prob == prob_type::polynomial || 
       prob == prob_type::curved_streamlines)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f),0);
   }

   FunctionCoefficient hatuex(exact_hatu);
   VectorFunctionCoefficient hatfex(dim,exact_hatf);
   Array<int> elements_to_refine;
   FunctionCoefficient uex(exact_u);
   VectorFunctionCoefficient sigmaex(dim,exact_sigma);

   ParGridFunction hatu_gf;
   ParGridFunction hatf_gf;

   socketstream uex_out;
   socketstream u_out;
   socketstream sigma_out;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      u_out.open(vishost, visport);
      if (exact_known) 
      {
         uex_out.open(vishost, visport);
      }   
      sigma_out.open(vishost, visport);
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0;
   if (myid == 0)
   {
      mfem::out << " Refinement |" 
               << "    Dofs    |" ;
      if (exact_known)
      {
         mfem::out << "  L2 Error  |" 
                  << "  Rate  |"; 
      }         
      mfem::out << "  Residual  |" 
                << "  Rate  |" 
                << " CG it  |" << endl;
      mfem::out << " -------------------"      
                <<  "-------------------" ;
      if (exact_known)
      {
         mfem::out << "----------------------";
      }
        mfem::out << "------------------" << endl;   
   }


   for (int iref = 0; iref<=ref; iref++)
   {
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      Array<int> ess_tdof_list_uhat;
      Array<int> ess_tdof_list_fhat;
      Array<int> ess_bdr_uhat;
      Array<int> ess_bdr_fhat;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr_uhat.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr_fhat.SetSize(pmesh.bdr_attributes.Max());

         if (prob == prob_type::EJ)
         {
            ess_bdr_uhat = 0;
            ess_bdr_fhat = 1;
            ess_bdr_uhat[1] = 1;
            ess_bdr_fhat[1] = 0;
         }
         else
         {
            ess_bdr_uhat = 1;
            ess_bdr_fhat = 0;
         }

         hatu_fes->GetEssentialTrueDofs(ess_bdr_uhat, ess_tdof_list_uhat);
         hatf_fes->GetEssentialTrueDofs(ess_bdr_fhat, ess_tdof_list_fhat);
      }

      // shift the ess_tdofs
      int n = ess_tdof_list_uhat.Size();
      int m = ess_tdof_list_fhat.Size();
      Array<int> ess_tdof_list(n+m);
      for (int j = 0; j < n; j++)
      {
         ess_tdof_list[j] = ess_tdof_list_uhat[j] 
                          + u_fes->GetTrueVSize() 
                          + sigma_fes->GetTrueVSize();
      }
      for (int j = 0; j < m; j++)
      {
         ess_tdof_list[j+n] = ess_tdof_list_fhat[j] 
                            + u_fes->GetTrueVSize() 
                            + sigma_fes->GetTrueVSize()
                            + hatu_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = sigma_fes->GetVSize();
      offsets[3] = hatu_fes->GetVSize();
      offsets[4] = hatf_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      hatu_gf.MakeRef(hatu_fes,x.GetBlock(2));
      FunctionCoefficient bdr_cf(bdr_data);
      hatu_gf.ProjectBdrCoefficient(bdr_cf,ess_bdr_uhat);

      hatf_gf.MakeRef(hatf_fes,x.GetBlock(3));
      hatf_gf.ProjectBdrCoefficientNormal(hatfex,ess_bdr_fhat);

      OperatorPtr Ah;
      Vector X,B;
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
         skip = 2;
      }
      HypreBoomerAMG * amg2 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(skip,skip));
      amg2->SetPrintLevel(0);
      M->SetDiagonalBlock(skip,amg2);

      HypreSolver * prec;
      if (dim == 2)
      {
         prec = new HypreAMS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatf_fes);
      }
      else
      {
         prec = new HypreADS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatf_fes);
      }
      M->SetDiagonalBlock(skip+1,prec);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(200000);
      cg.SetPrintLevel(-1);
      cg.SetPreconditioner(*M);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      int num_iter = cg.GetNumIterations();
      delete M;

      a->RecoverFEMSolution(X,x);
      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();
      double maxresidual = residuals.Max(); 

      double gresidual = residual * residual; 

      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&gresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      gresidual = sqrt(gresidual);

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

      ParGridFunction sigma_gf;
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1));

      int dofs = u_fes->GlobalTrueVSize()
               + sigma_fes->GlobalTrueVSize()
               + hatu_fes->GlobalTrueVSize()
               + hatf_fes->GlobalTrueVSize();


      double rate_res = (iref) ? dim*log(res0/gresidual)/log((double)dof0/dofs) : 0.0;
      res0 = gresidual;

      if (exact_known)
      {
         double u_err = u_gf.ComputeL2Error(uex);
         double sigma_err = sigma_gf.ComputeL2Error(sigmaex);
         double L2Error = sqrt(u_err*u_err + sigma_err*sigma_err);

         double rate_err = (iref) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
         dof0 = dofs;

         err0 = L2Error;

         if (myid == 0)
         {
            mfem::out << std::right << std::setw(11) << iref << " | " 
                     << std::setw(10) <<  dof0 << " | " 
                     << std::setprecision(3) 
                     << std::setw(10) << std::scientific <<  err0 << " | " 
                     << std::setprecision(2) 
                     << std::setw(6) << std::fixed << rate_err << " | " 
                     << std::setprecision(3) 
                     << std::setw(10) << std::scientific <<  res0 << " | " 
                     << std::setprecision(2) 
                     << std::setw(6) << std::fixed << rate_res << " | " 
                     << std::setw(6) << std::fixed << num_iter << " | " 
                     << std::resetiosflags(std::ios::showbase)
                     << std::endl;
         }
      }
      else
      {
         dof0 = dofs;
         if (myid == 0)
         {
            mfem::out << std::right << std::setw(11) << iref << " | " 
                     << std::setw(10) <<  dof0 << " | " 
                     << std::setprecision(3) 
                     << std::setw(10) << std::scientific <<  res0 << " | " 
                     << std::setprecision(2) 
                     << std::setw(6) << std::fixed << rate_res << " | " 
                     << std::setw(6) << std::fixed << num_iter << " | " 
                     << std::resetiosflags(std::ios::showbase)
                     << std::endl;
         }
      }   



      if (visualization)
      {
         if (exact_known)
         {
            ParGridFunction uex_gf(u_fes);
            uex_gf.ProjectCoefficient(uex);
            uex_out << "parallel " << num_procs << " " << myid << "\n";
            uex_out.precision(8);
            uex_out << "solution\n" << pmesh << uex_gf;
            if (iref == 0) uex_out  << "keys cgRjmlk \n";
            uex_out << "window_title 'Exact u' "
                    << flush;
         }
         u_out << "parallel " << num_procs << " " << myid << "\n";
         u_out.precision(8);
         u_out << "solution\n" << pmesh << u_gf;
         if (iref == 0) u_out  << "keys cgRjmlk \n";
         u_out << "window_title 'Numerical u' "
               << flush;

         sigma_out << "parallel " << num_procs << " " << myid << "\n";
         sigma_out.precision(8);
         sigma_out << "solution\n" << pmesh << sigma_gf;
         if (iref == 0) sigma_out  << "keys cgRjmlk \n";
         sigma_out << "window_title 'Numerical flux' "
                   << flush;
      }

      if (iref == ref)
         break;

      pmesh.GeneralRefinement(elements_to_refine,1,1);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();

      if (test_norm == test_norm_type::robust)
      {
         coeff_fes->Update();
         c1_gf.Update();
         c2_gf.Update();
         Array<int> edofs;
         for (int i = 0; i < pmesh.GetNE(); i++)
         {
            double volume = pmesh.GetElementVolume(i);
            double c1 = min(epsilon/volume, 1.);
            double c2 = min(1./epsilon, 1./volume);
            coeff_fes->GetElementDofs(i,edofs);
            c1_gf.SetSubVector(edofs,c1);
            c2_gf.SetSubVector(edofs,c2);
         }
      }
   }

   delete coeff_fes;
   delete coeff_fec;
   delete a;
   delete tau_fec;
   delete v_fec;
   delete hatf_fes;
   delete hatf_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete sigma_fes;
   delete u_fec;
   delete u_fes;

   return 0;
}

void solution(const Vector & X, double & u, Vector & du, double & d2u)
{
   double x = X[0];
   double y = X[1];
   double z = 0.;
   if (X.Size() == 3) z = X[2];
   du.SetSize(X.Size());
   du = 0.;
   d2u = 0.;

   switch(prob)
   {
      case polynomial:
      {
         int n=2;
         int m=2;
         u = pow(x,n)*pow(y,m);
         du[0] = n * pow(x,n-1) * pow(y,m);
         du[1] = m * pow(x,n) * pow(y,m-1);
         d2u = n * (n-1) * pow(x,n-2) * pow(y,m)
             + m * (m-1) * pow(x,n) * pow(y,m-2);
      }
      break;
      case sinusoidal:
      {
         double alpha = M_PI * (x + y + z);
         u = sin(alpha);
         du.SetSize(X.Size());
         for (int i = 0; i<du.Size(); i++)
         {
            du[i] = M_PI * cos(alpha);
         }
         d2u = - M_PI*M_PI * u * du.Size();
      }
      break;
      case EJ:
      {
         double alpha = sqrt(1. + 4. * epsilon * epsilon * M_PI * M_PI);
         double r1 = (1. + alpha) / (2.*epsilon);
         double r2 = (1. - alpha) / (2.*epsilon);
         double denom = exp(-r2) - exp(-r1);


         double g1 = exp(r2*(x-1.));
         double g1_x = r2*g1;
         double g1_xx = r2*g1_x;
         double g2 = exp(r1*(x-1.));
         double g2_x = r1*g2;
         double g2_xx = r1*g2_x;
         double g = g1-g2;
         double g_x = g1_x - g2_x;
         double g_xx = g1_xx - g2_xx;


         u = g * cos(M_PI * y)/denom;
         double u_x = g_x * cos(M_PI * y)/denom;
         double u_xx = g_xx * cos(M_PI * y)/denom;
         double u_y = -M_PI * g * sin(M_PI*y)/denom;
         double u_yy = -M_PI * M_PI * u;
         du[0] = u_x;
         du[1] = u_y;
         d2u = u_xx + u_yy;
      }
      break;
      case curved_streamlines:
      {
         double r = sqrt(x*x+y*y);
         u = atan((1.0-r)/epsilon);
         double alpha = -2.0*r + r*r + epsilon*epsilon + 1;
         double denom = r*alpha;
         du[0] = - x* epsilon / denom;
         du[1] = - y* epsilon / denom;
         d2u = epsilon * (r*r - epsilon*epsilon - 1.0) / (r*alpha*alpha);
      }
      break;
      default:
      MFEM_ABORT("Wrong code path");
      break;
   }
}


void beta_function(const Vector & X, Vector & beta_val)
{
   beta_val.SetSize(2);
   if (prob == prob_type::curved_streamlines)
   {
      double x = X(0);
      double y = X(1);
      beta_val(0) = exp(x)*sin(y);
      beta_val(1) = exp(x)*cos(y);
   }
   else
   {
      beta_val = beta;
   }
}



double exact_u(const Vector & X)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   return u;
}

void exact_sigma(const Vector & X, Vector & sigma)
{
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   // σ = ε ∇ u
   sigma = du;
   sigma *= epsilon;
}

double exact_hatu(const Vector & X)
{
   return -exact_u(X);
}

void exact_hatf(const Vector & X, Vector & hatf)
{
   Vector sigma;
   exact_sigma(X,sigma);
   Vector beta_val;
   beta_function(X,beta_val);
   exact_sigma(X,sigma);
   double u = exact_u(X);
   hatf.SetSize(X.Size());
   for (int i = 0; i<hatf.Size(); i++)
   {
      hatf[i] = beta_val[i] * u - sigma[i];
   }
}

double f_exact(const Vector & X)
{
   // f = - εΔu + ∇⋅(βu)
   double u, d2u;
   Vector du;
   solution(X,u,du,d2u);
   Vector beta_val;
   beta_function(X,beta_val);

   double s = 0;
   for (int i = 0; i<du.Size(); i++)
   {
      s += beta_val[i] * du[i];
   }
   return -epsilon * d2u + s;
}

double bdr_data(const Vector &X)
{
   if (prob == prob_type::bdr_layer)
   {  
      double x = X(0);
      double y = X(1);

      if (y==0.0)
      {
         return -(1.0-x);
      }
      else if (x == 0.0)
      {
         return -(1.0-y);
      }
      else
      {
         return 0.0;
      }
   }
   else
   {
      return exact_hatu(X);
   }   
}