//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make uw_dpg
//

//     - Δ p - ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω

// First Order System

//  ∇ p + i ω u = 0, in Ω
//  ∇⋅u + i ω p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/(i ω) 

// UW-DPG:
// 
// p ∈ L^2(Ω), u ∈ (L^2(Ω))^dim 
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)  
// -(p,  ∇⋅v) + i ω (u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)      
// -(u , ∇ q) + i ω (p , q) + < û, q >  = (f,q)   ∀ q ∈ H^1(Ω)
//                                  p̂  = p_0     on ∂Ω 

// Note: 
// p̂ := p on Γ_h (skeleton)
// û := u on Γ_h  

// -------------------------------------------------------------
// |   |     p     |     u     |    p̂      |    û    |  RHS    |
// -------------------------------------------------------------
// | v | -(p, ∇⋅v) | i ω (u,v) | < p̂, v⋅n> |         |         |
// |   |           |           |           |         |         |
// | q | i ω (p,q) |-(u , ∇ q) |           | < û,q > |  (f,q)  |  

// where (q,v) ∈  H^1(Ω) × H(div,Ω) 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void acoustics_solution(const Vector & X, complex<double> & p, 
                        vector<complex<double>> &dp, complex<double> & d2p);

void acoustics_solution_r(const Vector & X, double & p, 
                          Vector &dp, double & d2p);      

void acoustics_solution_i(const Vector & X, double & p, 
                          Vector &dp, double & d2p);                                                

double p_exact_r(const Vector &x);
double p_exact_i(const Vector &x);
void u_exact_r(const Vector &x, Vector & u);
void u_exact_i(const Vector &x, Vector & u);
double rhs_func_r(const Vector &x);
double rhs_func_i(const Vector &x);
void gradp_exact_r(const Vector &x, Vector &gradu);
void gradp_exact_i(const Vector &x, Vector &gradu);
double divu_exact_r(const Vector &x);
double divu_exact_i(const Vector &x);
double d2_exact_r(const Vector &x);
double d2_exact_i(const Vector &x);
double hatp_exact_r(const Vector & X);
double hatp_exact_i(const Vector & X);
void hatu_exact(const Vector & X, Vector & hatu);
void hatu_exact_r(const Vector & X, Vector & hatu);
void hatu_exact_i(const Vector & X, Vector & hatu);

int dim;
double omega;

enum prob_type
{
   plane_wave,
   gaussian_beam  
};

prob_type prob;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;
   double theta = 0.0;
   bool adjoint_graph_norm = false;
   bool static_cond = false;
   int iprob = 0;

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
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: plane wave, 1: Gaussian beam");                    
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                    
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");                                
   args.AddOption(&ref, "-ref", "--serial_ref",
                  "Number of serial refinements.");       
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");                                            
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


   if (iprob > 1) { iprob = 0; }
   prob = (prob_type)iprob;

   omega = 2.*M_PI*rnum;


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


   if (myid == 0)
   {
      mfem::out << "p_fes space true dofs = " << p_fes->GetTrueVSize() << endl;
      mfem::out << "u_fes space true dofs = " << u_fes->GetTrueVSize() << endl;
      mfem::out << "hatp_fes space true dofs = " << hatp_fes->GetTrueVSize() << endl;
      mfem::out << "hatu_fes space true dofs = " << hatu_fes->GetTrueVSize() << endl;
   }

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector vec0(dim); vec0 = 0.;
   VectorConstantCoefficient vzero(vec0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   ConstantCoefficient negomeg(-omega);

   // Normal equation weak formulation
   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);

   test_fec.Append(q_fec);
   test_fec.Append(v_fec);

   ComplexParNormalEquations * a = new ComplexParNormalEquations(trial_fes,test_fec);
   a->StoreMatrices();
   // i ω (p,q)
   a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(omeg),0,0);

// -(u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(negone)),nullptr,1,0);

// -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),nullptr,0,1);

//  i ω (u,v)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(omeg)),1,1);

// < p̂, v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,nullptr,2,1);

// < û,q >
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,3,0);

// test integrators 
   //space-induced norm for H(div) × H1
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,0,0);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),nullptr,0,0);
   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),nullptr,1,1);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

   // additional integrators for the adjoint graph norm
   if (adjoint_graph_norm)
   {   
      // -i ω (∇q,δv)
      a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negomeg),0,1);
      // i ω (v,∇ δq)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakDivergenceIntegrator(negomeg),1,0);
      // ω^2 (v,δv)
      a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2),nullptr,1,1);

      // - i ω (∇⋅v,δq)   
      a->AddTestIntegrator(nullptr,new VectorFEDivergenceIntegrator(negomeg),1,0);
      // i ω (q,∇⋅v)   
      a->AddTestIntegrator(nullptr,new MixedScalarWeakGradientIntegrator(negomeg),0,1);
      // ω^2 (q,δq)
      a->AddTestIntegrator(new MassIntegrator(omeg2),nullptr,0,0);
   }

   // RHS
   FunctionCoefficient f_rhs_r(rhs_func_r);
   FunctionCoefficient f_rhs_i(rhs_func_i);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs_r),new DomainLFIntegrator(f_rhs_i),0);
   
   

   FunctionCoefficient hatpex_r(hatp_exact_r);
   FunctionCoefficient hatpex_i(hatp_exact_i);
   Array<int> elements_to_refine;

   socketstream p_out_r;
   socketstream p_out_i;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      p_out_r.open(vishost, visport);
      p_out_i.open(vishost, visport);
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
      if (static_cond) { a->EnableStaticCondensation(); }
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         // ess_bdr[1] = 0;
         // ess_bdr[2] = 0;
         hatp_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = p_fes->GetVSize();
      offsets[2] = u_fes->GetVSize();
      offsets[3] = hatp_fes->GetVSize();
      offsets[4] = hatu_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;
      double * xdata = x.GetData();

      ParComplexGridFunction hatp_gf(hatp_fes);
      hatp_gf.real().MakeRef(hatp_fes,&xdata[offsets[2]]);
      hatp_gf.imag().MakeRef(hatp_fes,&xdata[offsets.Last()+ offsets[2]]);
      hatp_gf.ProjectBdrCoefficient(hatpex_r,hatpex_i, ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();
      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());



      MFEM_VERIFY(static_cond, "preconditioner not implemented for the non-static condensation case");



      Array<int> tdof_offsets(5);
      tdof_offsets[0] = 0;
      tdof_offsets[1] = hatp_fes->GetTrueVSize();
      tdof_offsets[2] = hatu_fes->GetTrueVSize();
      tdof_offsets[3] = hatp_fes->GetTrueVSize();
      tdof_offsets[4] = hatu_fes->GetTrueVSize();

      tdof_offsets.PartialSum();

      BlockOperator blockA(tdof_offsets);
      blockA.SetBlock(0,0, &BlockA_r->GetBlock(0,0));
      blockA.SetBlock(0,1, &BlockA_r->GetBlock(0,1));
      blockA.SetBlock(1,0, &BlockA_r->GetBlock(1,0));
      blockA.SetBlock(1,1, &BlockA_r->GetBlock(1,1));

      blockA.SetBlock(0,2, &BlockA_i->GetBlock(0,0),-1);
      blockA.SetBlock(0,3, &BlockA_i->GetBlock(0,1),-1);
      blockA.SetBlock(1,2, &BlockA_i->GetBlock(1,0),-1);
      blockA.SetBlock(1,3, &BlockA_i->GetBlock(1,1),-1);


      blockA.SetBlock(2,2, &BlockA_r->GetBlock(0,0));
      blockA.SetBlock(2,3, &BlockA_r->GetBlock(0,1));
      blockA.SetBlock(3,2, &BlockA_r->GetBlock(1,0));
      blockA.SetBlock(3,3, &BlockA_r->GetBlock(1,1));


      blockA.SetBlock(2,0, &BlockA_i->GetBlock(0,0));
      blockA.SetBlock(2,1, &BlockA_i->GetBlock(0,1));
      blockA.SetBlock(3,0, &BlockA_i->GetBlock(1,0));
      blockA.SetBlock(3,1, &BlockA_i->GetBlock(1,1));







      int numblocks = BlockA_r->NumRowBlocks();
      Array2D<HypreParMatrix *> Ab_r(numblocks,numblocks);
      Array2D<HypreParMatrix *> Ab_i(numblocks,numblocks);

      for (int ii = 0; ii<numblocks; ii++)
      {
         for (int jj = 0; jj<numblocks; jj++)
         {
            Ab_r(ii,jj) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(ii,jj));
            Ab_i(ii,jj) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(ii,jj));
         }
      }
      HypreParMatrix * A_r = HypreParMatrixFromBlocks(Ab_r);
      HypreParMatrix * A_i = HypreParMatrixFromBlocks(Ab_i);

      ComplexHypreParMatrix Ac(A_r,A_i,true,true);

      HypreParMatrix * A = Ac.GetSystemMatrix();

      if (myid == 0)
      {   
         mfem::out << "Size of the linear system: " << A->Height() << std::endl;
      }
      X = 0.;



      BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(tdof_offsets);
      HypreBoomerAMG * amg = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(0,0));
      amg->SetPrintLevel(0);
      HypreAMS * ams = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(1,1), hatu_fes);
      ams->SetPrintLevel(0);

      M->SetDiagonalBlock(0,amg);
      M->SetDiagonalBlock(1,ams);
      M->SetDiagonalBlock(2,amg);
      M->SetDiagonalBlock(3,ams);

      
      MINRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetRelTol(1e-6);
      gmres.SetMaxIter(2000);
      gmres.SetPrintLevel(3);
      gmres.SetPreconditioner(*M); 
      // gmres.SetOperator(*A);
      gmres.SetOperator(blockA);
      gmres.Mult(B, X);
      // delete prec;

      // MUMPSSolver mumps;
      // mumps.SetOperator(*A);
      // mumps.Mult(B,X);

      delete A;
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

      ParComplexGridFunction p(p_fes);
      p.real().MakeRef(p_fes,x.GetData());
      p.imag().MakeRef(p_fes,&x.GetData()[offsets.Last()]);

      ParComplexGridFunction pgf_ex(p_fes);
      FunctionCoefficient p_ex_r(p_exact_r);
      FunctionCoefficient p_ex_i(p_exact_i);
      pgf_ex.ProjectCoefficient(p_ex_r, p_ex_i);

      int dofs = p_fes->GlobalTrueVSize()
               + u_fes->GlobalTrueVSize()
               + hatp_fes->GlobalTrueVSize()
               + hatu_fes->GlobalTrueVSize();
      dofs/=2;         

      double p_err_r = p.real().ComputeL2Error(p_ex_r);
      double p_err_i = p.imag().ComputeL2Error(p_ex_i);

      double L2Error = sqrt(p_err_r*p_err_r + p_err_i*p_err_i);

      double rate_err = (i) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
      double rate_res = (i) ? dim*log(res0/residual)/log((double)dof0/dofs) : 0.0;

      err0 = L2Error;
      res0 = residual;
      dof0 = dofs;

      if (myid == 0)
      {
         mfem::out << std::right << std::setw(11) << i << " | " 
            << std::setw(10) <<  dof0 << " | " 
            << std::setprecision(3) 
            << std::setw(10) << std::scientific <<  err0 << " | " 
            << std::setprecision(3) 
            << std::setw(10) << std::fixed <<  0.0 << " | " 
            << std::setprecision(2) 
            << std::setw(6) << std::fixed << rate_err << " | " 
            << std::setprecision(3) 
            << std::setw(10) << std::scientific <<  res0 << " | " 
            << std::setprecision(2) 
            << std::setw(6) << std::fixed << rate_res << " | " 
            << std::resetiosflags(std::ios::showbase)
            << std::setw(10) << std::scientific 
            << std::endl;
      }

      if (visualization)
      {
         p_out_r << "parallel " << num_procs << " " << myid << "\n";
         p_out_r.precision(8);
         p_out_r << "solution\n" << pmesh << p.real() <<
                  "window_title 'Real Numerical presure' "
                  << flush;

         p_out_i << "parallel " << num_procs << " " << myid << "\n";
         p_out_i.precision(8);
         p_out_i << "solution\n" << pmesh << p.imag() <<
                  "window_title 'Imag Numerical presure' "
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

double p_exact_r(const Vector &x)
{
   double p,d2p;
   Vector dp;
   acoustics_solution_r(x,p,dp,d2p);
   return p;
}

double p_exact_i(const Vector &x)
{
   double p,d2p;
   Vector dp;
   acoustics_solution_i(x,p,dp,d2p);
   return p;
}

double hatp_exact_r(const Vector & X)
{
   return p_exact_r(X);
}

double hatp_exact_i(const Vector & X)
{
   return p_exact_i(X);
}

void gradp_exact_r(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   double p,d2p;
   acoustics_solution_r(x,p,grad,d2p);
}

void gradp_exact_i(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   double p,d2p;
   acoustics_solution_i(x,p,grad,d2p);
}

double d2_exact_r(const Vector &x)
{
   double p,d2p;
   Vector dp;
   acoustics_solution_r(x,p,dp,d2p);
   return d2p;
}

double d2_exact_i(const Vector &x)
{
   double p,d2p;
   Vector dp;
   acoustics_solution_i(x,p,dp,d2p);
   return d2p;
}

//  u = - ∇ p / (i ω )
//    = i (∇ p_r + i * ∇ p_i)  / ω 
//    = - ∇ p_i / ω + i ∇ p_r / ω 
void u_exact_r(const Vector &x, Vector & u)
{
   gradp_exact_i(x,u);
   u *= -1./omega;
}

void u_exact_i(const Vector &x, Vector & u)
{
   gradp_exact_r(x,u);
   u *= 1./omega;
}

void hatu_exact_r(const Vector & X, Vector & hatu)
{
   u_exact_r(X,hatu);
}
void hatu_exact_i(const Vector & X, Vector & hatu)
{
   u_exact_i(X,hatu);
}

//  ∇⋅u = i Δ p / ω
//      = i (Δ p_r + i * Δ p_i)  / ω 
//      = - Δ p_i / ω + i Δ p_r / ω 

double divu_exact_r(const Vector &x)
{
   return -d2_exact_i(x)/omega;
}

double divu_exact_i(const Vector &x)
{
   return d2_exact_r(x)/omega;
}

// f = ∇⋅u + i ω p 
// f_r = ∇⋅u_r - ω p_i  
double rhs_func_r(const Vector &x)
{
   double p = p_exact_i(x);
   double divu = divu_exact_r(x);
   return divu - omega * p;
}

// f_i = ∇⋅u_i + ω p_r
double rhs_func_i(const Vector &x)
{
   double p = p_exact_r(x);
   double divu = divu_exact_i(x);
   return divu + omega * p;
}


void acoustics_solution_r(const Vector & X, double & p, 
                          Vector &dp, double & d2p)
{
   complex<double> zp, d2zp;
   vector<complex<double>> dzp;
   acoustics_solution(X,zp,dzp,d2zp);
   p = zp.real();
   d2p = d2zp.real();
   dp.SetSize(X.Size());
   for (int i = 0; i<X.Size(); i++)
   {
      dp[i] = dzp[i].real();
   }
}                           

void acoustics_solution_i(const Vector & X, double & p, 
                          Vector &dp, double & d2p)
{
   complex<double> zp, d2zp;
   vector<complex<double>> dzp;
   acoustics_solution(X,zp,dzp,d2zp);
   p = zp.imag();
   d2p = d2zp.imag();
   dp.SetSize(X.Size());
   for (int i = 0; i<X.Size(); i++)
   {
      dp[i] = dzp[i].imag();
   }
}


void acoustics_solution(const Vector & X, complex<double> & p, vector<complex<double>> & dp, 
         complex<double> & d2p)
{
   dp.resize(X.Size());
   complex<double> zi = complex<double>(0., 1.);
   switch (prob)
   {
   case plane_wave:
   {
      double beta = omega/std::sqrt((double)X.Size());
      complex<double> alpha = beta * zi * X.Sum();
      p = exp(-alpha);
      d2p = - dim * beta * beta * p;
      for (int i = 0; i<X.Size(); i++)
      {
         dp[i] = - zi * beta * p;
      }
   }
      break;
   default:
   {
      double rk = omega;
      double alpha = 45 * M_PI/180.;
      double sina = sin(alpha); 
      double cosa = cos(alpha);
      // shift the origin
      double xprim=X(0) + 0.1; 
      double yprim=X(1) + 0.1;

      double  x = xprim*sina - yprim*cosa;
      double  y = xprim*cosa + yprim*sina;
      double  dxdxprim = sina, dxdyprim = -cosa;
      double  dydxprim = cosa, dydyprim =  sina;
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
      complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r + zi*phi0/2.;

      complex<double> zdedx = -2.*x/(w*w) - 2.*zi*M_PI*x/rl/r;
      complex<double> zdedy = 2.*x*x/(w*w*w)*dwdy - zi*rk + zi*M_PI*x*x/rl/(r*r)*drdy + zi*dphi0dy/2.;
      complex<double> zd2edxdx = -2./(w*w) - 2.*zi*M_PI/rl/r;
      complex<double> zd2edxdy = 4.*x/(w*w*w)*dwdy + 2.*zi*M_PI*x/rl/(r*r)*drdy;
      complex<double> zd2edydx = zd2edxdy;
      complex<double> zd2edydy = -6.*x*x/(w*w*w*w)*dwdy*dwdy + 2.*x*x/(w*w*w)*d2wdydy - 2.*zi*M_PI*x*x/rl/(r*r*r)*drdy*drdy
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
      complex<double> zd2pdydy = d2pfdydy*exp(ze) + dpfdy*exp(ze)*zdedy + zdpdy*zdedy + zp*zd2edydy;

      p = zp;
      dp[0] = (zdpdx*dxdxprim + zdpdy*dydxprim);
      dp[1] = (zdpdx*dxdyprim + zdpdy*dydyprim);

      d2p = (zd2pdxdx*dxdxprim + zd2pdydx*dydxprim)*dxdxprim + (zd2pdxdy*dxdxprim + zd2pdydy*dydxprim)*dydxprim
          + (zd2pdxdx*dxdyprim + zd2pdydx*dydyprim)*dxdyprim + (zd2pdxdy*dxdyprim + zd2pdydy*dydyprim)*dydyprim;
   }
   break;
   }

}
