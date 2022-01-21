//                   MFEM Ultraweak DPG MPI acoustics (Helmholtz) example
//
// Compile with: make uw_dpgp
//
//     - Δ p ± ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω
//
// First Order System

//   ∇ p - ω u = 0, in Ω
// - ∇⋅u ± ω p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/ω 
// 
// UW-DPG:
// 
// p ∈ L^2(Ω), u ∈ (L^2(Ω))^dim 
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)  
// -(p,  ∇⋅v) - ω (u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)      
//  (u , ∇ q) ± ω (p , q) + < û, q >  = (f,q)   ∀ q ∈ H^1(Ω)
//                                  û = u_0     on ∂Ω 

// Note: 
// p̂ := p on Γ_h (skeleton)
// û := -u on Γ_h  

// -------------------------------------------------------------
// |   |     p     |     u     |    p̂      |    û    |  RHS    |
// -------------------------------------------------------------
// | v | -(p, ∇⋅v) | - ω (u,v) | < p̂, v⋅n> |         |         |
// |   |           |           |           |         |         |
// | q | ± ω (p,q) | (u , ∇ q) |           | < û,q > |  (f,q)  |  

// where (q,v) ∈  H^1(Ω) × H(div,Ω) 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// #define DEFINITE

double p_exact(const Vector &x);
void u_exact(const Vector &x, Vector & u);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
double divu_exact(const Vector &x);
double d2_exact(const Vector &x);
double hatp_exact(const Vector & X);
void hatu_exact(const Vector & X, Vector & hatu);

int dim;
double omega;

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;
   double theta = 0.0;
   bool adjoint_graph_norm = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");      
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                    
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");                                
   args.AddOption(&ref, "-ref", "--serial_ref",
                  "Number of serial refinements.");  
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

   omega = 2.0 * M_PI * rnum;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();

   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   // L2 space for p
   FiniteElementCollection *p_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *p_fes = new ParFiniteElementSpace(&pmesh,p_fec);

   // Vector L2 space for u 
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *u_fes = new ParFiniteElementSpace(&pmesh,u_fec, dim); 

   // H^1/2 space for p̂  
   FiniteElementCollection * hatp_fec = new H1_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatp_fes = new ParFiniteElementSpace(&pmesh,hatp_fec);

   // H^-1/2 space for û  
   FiniteElementCollection * hatu_fec = new RT_Trace_FECollection(order-1,dim);   
   ParFiniteElementSpace *hatu_fes = new ParFiniteElementSpace(&pmesh,hatu_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * q_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);


   Array<ParFiniteElementSpace * > trial_fes; 
   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);

   Array<FiniteElementCollection * > test_fec; 
   test_fec.Append(q_fec);
   test_fec.Append(v_fec);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector vec0(dim); vec0 = 0.;
   VectorConstantCoefficient vzero(vec0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   ConstantCoefficient negomeg(-omega);

   ParNormalEquations * a = new ParNormalEquations(trial_fes,test_fec);
   a->StoreMatrices(true);


   // Integrators

   // ± ω (p,q)
#ifdef DEFINITE
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(omeg),0,0);
#else
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(negomeg),0,0);
#endif   

// (u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),1,0);

// -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),0,1);

// - ω (u,v)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(negomeg)),1,1);

// < p̂, v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,2,1);

// < û,q >
   a->AddTrialIntegrator(new TraceIntegrator,3,0);


   // test integrators 

   //space-induced norm for H(div) × H1
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),0,0);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),0,0);
   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),1,1);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),1,1);

   // additional integrators for the adjoint graph norm
   if (adjoint_graph_norm)
   {   
      // -ω (∇q,δv)
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(negomeg),0,1);
      // -ω (v,δq)
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg),1,0);
      // ω^2 (v,δv)
      a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2),1,1);

#ifdef DEFINITE
      // - (∇⋅v,δq)   
      a->AddTestIntegrator(new VectorFEDivergenceIntegrator(negone),1,0);
      // - (q,∇⋅v)   
      a->AddTestIntegrator(new MixedScalarWeakGradientIntegrator(one),0,1);
#else
      // (∇⋅v,δq)   
      a->AddTestIntegrator(new VectorFEDivergenceIntegrator(one),1,0);
      // (q,∇⋅v)   
      a->AddTestIntegrator(new MixedScalarWeakGradientIntegrator(negone),0,1);
#endif
      // (q,δq)
      a->AddTestIntegrator(new MassIntegrator(one),0,0);
   }   

   // RHS
   FunctionCoefficient f_rhs(rhs_func);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs),0);


   FunctionCoefficient hatpex(hatp_exact);
   FunctionCoefficient pex(p_exact);
   VectorFunctionCoefficient uex(dim,u_exact);
   Array<int> elements_to_refine;
   ParGridFunction hatp_gf;




   socketstream p_out;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      p_out.open(vishost, visport);
   }
   double res0 = 0.;
   double err0 = 0.;
   int dof0;
   if (myid == 0)
   {
      mfem::out << " Refinement |" 
               << "    Dofs    |" 
               << "  L2 Error  |" 
               << " Relative % |" 
               << "  Rate  |" 
               << "  Residual  |" 
               << "  Rate  |" << endl;
      mfem::out << " --------------------"      
               <<  "-------------------"    
               <<  "-------------------"    
               <<  "-------------------" << endl;   
   }


   for (int i = 0; i<ref; i++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatp_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_tdof_list[i] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = p_fes->GetVSize();
      offsets[2] = u_fes->GetVSize();
      offsets[3] = hatp_fes->GetVSize();
      offsets[4] = hatu_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      hatp_gf.MakeRef(hatp_fes,x.GetBlock(2));
      hatp_gf.ProjectBdrCoefficient(hatpex,ess_bdr);

      Vector X,B;
      OperatorPtr Ah;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockOperator * A = Ah.As<BlockOperator>();

      BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(A->RowOffsets());
      M->owns_blocks = 1;

      HypreBoomerAMG * amg0 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(0,0));
      HypreBoomerAMG * amg1 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(1,1));
      HypreBoomerAMG * amg2 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(2,2));
      amg0->SetPrintLevel(0);
      amg1->SetPrintLevel(0);
      amg2->SetPrintLevel(0);
      amg0->SetRelaxType(16);
      amg1->SetRelaxType(16);
      amg2->SetRelaxType(16);

      M->SetDiagonalBlock(0,amg0);
      M->SetDiagonalBlock(1,amg1);
      M->SetDiagonalBlock(2,amg2);

      HypreSolver * prec;
      if (dim == 2)
      {
         prec = new HypreAMS((HypreParMatrix &)A->GetBlock(3,3), hatu_fes);
      }
      else
      {
         prec = new HypreADS((HypreParMatrix &)A->GetBlock(3,3), hatu_fes);
      }
      M->SetDiagonalBlock(3,prec);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-8);
      cg.SetMaxIter(20000);
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


      elements_to_refine.SetSize(0);
      for (int iel = 0; iel<pmesh.GetNE(); iel++)
      {
         if (residuals[iel] > theta * maxresidual)
         {
            elements_to_refine.Append(iel);
         }
      }


      ParGridFunction p_gf;
      p_gf.MakeRef(p_fes,x.GetBlock(0));

      ParGridFunction u_gf;
      u_gf.MakeRef(u_fes,x.GetBlock(1));


      ParGridFunction pex_gf(p_fes);
      ParGridFunction uex_gf(u_fes);
      pex_gf.ProjectCoefficient(pex);
      uex_gf.ProjectCoefficient(uex);

      int dofs = p_fes->GlobalTrueVSize()
               + u_fes->GlobalTrueVSize()
               + hatp_fes->GlobalTrueVSize()
               + hatu_fes->GlobalTrueVSize();

      double p_err = p_gf.ComputeL2Error(pex);
      double p_norm = pex_gf.ComputeL2Error(zero);
      double u_err = u_gf.ComputeL2Error(uex);
      double u_norm = uex_gf.ComputeL2Error(vzero);

      double L2Error = sqrt(p_err*p_err + u_err*u_err);
      double L2norm = sqrt(p_norm * p_norm + u_norm * u_norm);

      double rel_error = L2Error/L2norm;

      double rate_err = (i) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
      double rate_res = (i) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;

      err0 = L2Error;
      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         mfem::out << std::right << std::setw(11) << i << " | " 
                   << std::setw(10) <<  dof0 << " | " 
                   << std::setprecision(3) 
                   << std::setw(10) << std::scientific <<  err0 << " | " 
                   << std::setprecision(3) 
                   << std::setw(10) << std::fixed <<  rel_error * 100. << " | " 
                   << std::setprecision(2) 
                   << std::setw(6) << std::fixed << rate_err << " | " 
                   << std::setprecision(3) 
                   << std::setw(10) << std::scientific <<  res0 << " | " 
                   << std::setprecision(2) 
                   << std::setw(6) << std::fixed << rate_res << " | " 
                   << std::resetiosflags(std::ios::showbase)
                   << std::endl;
      }   
      if (visualization)
      {
         p_out << "parallel " << num_procs << " " << myid << "\n";
         p_out.precision(8);
         p_out << "solution\n" << pmesh << p_gf <<
                  "window_title 'Numerical pressure' "
                  << flush;
      }

      if (i == ref)
         break;

      pmesh.GeneralRefinement(elements_to_refine,1,1);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();

   }

   delete a;
   delete q_fec;
   delete v_fec;
   delete hatp_fes;
   delete hatp_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete u_fec;
   delete p_fec;
   delete u_fes;
   delete p_fes;

   return 0;
}

double rhs_func(const Vector &x)
{
   double p = p_exact(x);
   double divu = divu_exact(x);
   // f = - ∇⋅u ± ω p, 
#ifdef DEFINITE   
   return -divu + omega * p;
#else
   return -divu - omega * p;
#endif   
}

double p_exact(const Vector &x)
{
   return sin(omega*x.Sum());
}

void gradp_exact(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   grad = omega * cos(omega * x.Sum());
}

void u_exact(const Vector &x, Vector & u)
{
   gradp_exact(x,u);
   u *= 1./omega;
}

double divu_exact(const Vector &x)
{
   return d2_exact(x)/omega;
}

double d2_exact(const Vector &x)
{
   return -dim * omega * omega * sin(omega*x.Sum());
}

double hatp_exact(const Vector & X)
{
   return p_exact(X);
}

void hatu_exact(const Vector & X, Vector & hatu)
{
   u_exact(X,hatu);
   hatu *= -1.;
}