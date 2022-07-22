//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make pcomplex_uw_dpg
//
// sample runs
// ./pcomplex_uw_dpg -o 3 -m ../../../data/inline-quad.mesh -sref 2 -pref 3 -rnum 4.1 -prob 0 -sc -graph-norm

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
   double theta = 0.0;
   bool adjoint_graph_norm = false;
   bool static_cond = false;
   int iprob = 0;
   int sr = 0;
   int pr = 1;

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
   args.AddOption(&sr, "-sref", "--serial_ref",
                  "Number of parallel refinements.");                                              
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");  
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
   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
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


   // if (myid == 0)
   // {
   //    mfem::out << "p_fes space true dofs = " << p_fes->GetTrueVSize() << endl;
   //    mfem::out << "u_fes space true dofs = " << u_fes->GetTrueVSize() << endl;
   //    mfem::out << "hatp_fes space true dofs = " << hatp_fes->GetTrueVSize() << endl;
   //    mfem::out << "hatu_fes space true dofs = " << hatu_fes->GetTrueVSize() << endl;
   // }

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

   VectorFunctionCoefficient hatuex_r(dim,hatu_exact_r);
   VectorFunctionCoefficient hatuex_i(dim,hatu_exact_i);
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
      mfem::out << "\n  Ref |" 
                << "       Mesh       |"
                << "    Dofs    |" 
                << "   ω   |" 
                << "  L2 Error  |" 
                << " Relative % |" 
                << "  Rate  |" 
                << "  Residual  |" 
                << "  Rate  |" 
                << " PCG it |"
                << " PCG time |"  << endl;
      mfem::out << " --------------------"      
                <<  "---------------------"    
                <<  "---------------------"    
                <<  "---------------------"    
                <<  "---------------------"    
                <<  "-------------------" << endl;      
   }


   for (int it = 0; it<pr; it++)
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
         // hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
                           //   + hatp_fes->GetTrueVSize(); 
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
      // ParComplexGridFunction hatu_gf(hatu_fes);
      // hatu_gf.real().MakeRef(hatu_fes,&xdata[offsets[3]]);
      // hatu_gf.imag().MakeRef(hatu_fes,&xdata[offsets.Last()+ offsets[3]]);
      // hatu_gf.ProjectCoefficientNormal(hatuex_r,hatuex_i, ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();
      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);

      tdof_offsets[0] = 0;
      int skip = (static_cond) ? 0 : 2;
      int k = (static_cond) ? 2 : 0;
      for (int i=0; i<num_blocks;i++)
      {
         tdof_offsets[i+1] = trial_fes[i+k]->GetTrueVSize(); 
         tdof_offsets[num_blocks+i+1] = trial_fes[i+k]->GetTrueVSize(); 
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

      BlockDiagonalPreconditioner * M = new BlockDiagonalPreconditioner(tdof_offsets);


      if (!static_cond)
      {
         HypreBoomerAMG * solver_p = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(0,0));
         solver_p->SetPrintLevel(0);
         solver_p->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_u = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(1,1));
         solver_u->SetPrintLevel(0);
         solver_u->SetSystemsOptions(dim);
         M->SetDiagonalBlock(0,solver_p);
         M->SetDiagonalBlock(1,solver_u);
         M->SetDiagonalBlock(num_blocks,solver_p);
         M->SetDiagonalBlock(num_blocks+1,solver_u);
      }


      HypreBoomerAMG * solver_hatp = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(skip,skip));
      // amg->SetCycleNumSweeps(5, 5);
      solver_hatp->SetPrintLevel(0);
      
      HypreSolver * solver_hatu = nullptr;
      if (dim == 2)
      {
         solver_hatu = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1),hatu_fes);
         dynamic_cast<HypreAMS*>(solver_hatu)->SetPrintLevel(0);
      }
      else
      {
         solver_hatu = new HypreADS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1), hatu_fes);
         dynamic_cast<HypreAMS*>(solver_hatu)->SetPrintLevel(0);
      }
      

      M->SetDiagonalBlock(skip,solver_hatp);
      M->SetDiagonalBlock(skip+1,solver_hatu);
      M->SetDiagonalBlock(skip+num_blocks,solver_hatp);
      M->SetDiagonalBlock(skip+num_blocks+1,solver_hatu);

      StopWatch chrono;
      
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-7);
      cg.SetAbsTol(1e-7);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(*M); 
      cg.SetOperator(blockA);
      chrono.Clear();
      chrono.Start();
      cg.Mult(B, X);
      chrono.Stop();
      delete M;

      int ne = pmesh.GetNE();
      MPI_Allreduce(MPI_IN_PLACE,&ne,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      int ne_x = (dim == 2) ? (int)sqrt(ne) : (int)cbrt(ne);
      ostringstream oss;
      double pcg_time = chrono.RealTime();
      if (myid == 0)
      {
         if (dim == 2)
         {
            oss << ne_x << " x " << ne_x ;
         }
         else
         {
            oss << ne_x << " x " << ne_x << " x " << ne_x ;
         }
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

      ParComplexGridFunction u(u_fes);
      u.real().MakeRef(u_fes,&x.GetData()[offsets[1]]);
      u.imag().MakeRef(u_fes,&x.GetData()[offsets.Last()+offsets[1]]);


      // Error in pressure 
      ParComplexGridFunction pgf_ex(p_fes);
      FunctionCoefficient p_ex_r(p_exact_r);
      FunctionCoefficient p_ex_i(p_exact_i);
      pgf_ex.ProjectCoefficient(p_ex_r, p_ex_i);

      double p_err_r = p.real().ComputeL2Error(p_ex_r);
      double p_err_i = p.imag().ComputeL2Error(p_ex_i);
      double p_error = sqrt(p_err_r*p_err_r + p_err_i*p_err_i);
      double p_norm_r = pgf_ex.real().ComputeL2Error(zero);
      double p_norm_i = pgf_ex.imag().ComputeL2Error(zero);
      double p_norm = sqrt(p_norm_r*p_norm_r + p_norm_i*p_norm_i);

      // Error in velocity
      ParComplexGridFunction ugf_ex(u_fes);
      VectorFunctionCoefficient u_ex_r(dim,u_exact_r);
      VectorFunctionCoefficient u_ex_i(dim,u_exact_i);

      double u_err_r = u.real().ComputeL2Error(u_ex_r);
      double u_err_i = u.imag().ComputeL2Error(u_ex_i);
      double u_error = sqrt(u_err_r*u_err_r + u_err_i*u_err_i);
      double u_norm_r = pgf_ex.real().ComputeL2Error(vzero);
      double u_norm_i = pgf_ex.imag().ComputeL2Error(vzero);
      double u_norm = sqrt(u_norm_r*u_norm_r + u_norm_i*u_norm_i);


      double L2Error = sqrt(p_error*p_error + u_error*u_error);
      double L2norm = sqrt(p_norm*p_norm + u_norm*u_norm);

      double rel_err = L2Error/L2norm;

      int dofs = p_fes->GlobalTrueVSize()
               + u_fes->GlobalTrueVSize()
               + hatp_fes->GlobalTrueVSize()
               + hatu_fes->GlobalTrueVSize();


      double rate_err = (it) ? dim*log(err0/rel_err)/log((double)dof0/dofs) : 0.0;
      double rate_res = (it) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;

      err0 = rel_err;
      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         mfem::out << std::right << std::setw(5) << it << " | " 
                  << std::setw(16) << oss.str() << " | " 
                  << std::setw(10) <<  dof0 << " | " 
                  << std::setprecision(0) << std::fixed
                  << std::setw(2) <<  2*rnum << " π  | " 
                  << std::setprecision(3) 
                  << std::setw(10) << std::scientific <<  err0 << " | " 
                  << std::setprecision(3) 
                  << std::setw(10) << std::fixed <<  rel_err * 100. << " | " 
                  << std::setprecision(2) 
                  << std::setw(6) << std::fixed << rate_err << " | " 
                  << std::setprecision(3) 
                  << std::setw(10) << std::scientific <<  res0 << " | " 
                  << std::setprecision(2) 
                  << std::setw(6) << std::fixed << rate_res << " | " 
                  << std::setw(6) << std::fixed << num_iter << " | " 
                  << std::setprecision(5) 
                  << std::setw(8) << std::fixed << pcg_time << " | " 
                  << std::scientific 
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

      if (it == pr)
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
