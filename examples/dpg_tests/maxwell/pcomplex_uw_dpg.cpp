//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make pcomplex_uw_dpg
//
// sample run 
// ./pcomplex_uw_dpg -o 3 -m ../../../data/inline-quad.mesh -sref 2 -pref 3 -rnum 4.1 -prob 1 -sc -graph-norm

//      ∇×(1/μ ∇×E) - ω^2 ϵ E = Ĵ ,   in Ω
//                E×n = E_0, on ∂Ω

// First Order System

//  i ω μ H + ∇ × E = 0,   in Ω
// -i ω ϵ E + ∇ × H = J,   in Ω
//            E × n = E_0, on ∂Ω

// note: Ĵ = -iωJ
// in 2D 
// E is vector valued and H is scalar. 
//    (∇ × E, F) = (E, ∇ × F) + < n × E , F>
// or (∇ ⋅ AE , F) = (AE, ∇ F) + < AE ⋅ n, F>
// where A = A = [0 1; -1 0];

// UW-DPG:
// 
// in 3D 
// E,H ∈ (L^2(Ω))^3 
// Ê ∈ H_0^1/2(Ω)(curl, Γ_h), Ĥ ∈ H^-1/2(curl, Γ_h)  
//  i ω μ (H,F) + (E,∇ × F) + < Ê, F × n > = 0,      ∀ F ∈ H(curl,Ω)      
// -i ω ϵ (E,G) + (H,∇ × G) + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                   Ê × n = E_0     on ∂Ω 
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) | < n × Ê, F > |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < n × Ĥ, G > |  (J,G)  |  
// where (F,G) ∈  H(curl,Ω) × H(curl,Ω)

// in 2D 
// E ∈ L^2(Ω)^2, H ∈ L^2(Ω)
// Ê ∈ H^-1/2(Ω)(Γ_h), Ĥ ∈ H^1/2(Γ_h)  
//  i ω μ (H,F) + (E, ∇ × F) + < AÊ, F > = 0,      ∀ F ∈ H^1      
// -i ω ϵ (E,G) + (H,∇ × G)  + < Ĥ, G × n > = (J,G)   ∀ G ∈ H(curl,Ω)      
//                                    Ê = E_0     on ∂Ω 
// -------------------------------------------------------------------------
// |   |       E      |      H      |      Ê       |       Ĥ      |  RHS    |
// -------------------------------------------------------------------------
// | F |  (E,∇ × F)   | i ω μ (H,F) |   < Ê, F >   |              |         |
// |   |              |             |              |              |         |
// | G | -i ω ϵ (E,G) |  (H,∇ × G)  |              | < Ĥ, G × n > |  (J,G)  |  

// where (F,G) ∈  H^1 × H(curl,Ω)


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);

void H_exact_r(const Vector &x, Vector & H_r);
void H_exact_i(const Vector &x, Vector & H_i);


void  rhs_func_r(const Vector &x, Vector & J_r);
void  rhs_func_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);
void curlH_exact_r(const Vector &x,Vector &curlH_r);
void curlH_exact_i(const Vector &x,Vector &curlH_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);

void hatE_exact_r(const Vector & X, Vector & hatE_r);
void hatE_exact_i(const Vector & X, Vector & hatE_i);

void hatH_exact_r(const Vector & X, Vector & hatH_r);
void hatH_exact_i(const Vector & X, Vector & hatH_i);

double hatH_exact_scalar_r(const Vector & X);
double hatH_exact_scalar_i(const Vector & X);

void maxwell_solution(const Vector & X, 
                      std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE);

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                      Vector &curlE_r, 
                      Vector &curlcurlE_r);

void maxwell_solution_i(const Vector & X, Vector &E_i, 
                      Vector &curlE_i, 
                      Vector &curlcurlE_i);     

int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;

enum prob_type
{
   polynomial,
   plane_wave,
   fichera_oven  
};

prob_type prob;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../../data/inline-hex.mesh";
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
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");                  
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: polynomial, 1: plane wave, 2: Gaussian beam");                     
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


   if (iprob > 2) { iprob = 0; }
   prob = (prob_type)iprob;
   omega = 2.*M_PI*rnum;

   if (prob == 2)
   {
      mesh_file = "../../../data/fichera-oven.mesh";
      omega = 5.0;
   }


   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   dimc = (dim == 3) ? 3 : 1;
   mesh.EnsureNCMesh();

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   int test_order = order+delta_order;

   // Define spaces
   // L2 space for E
   FiniteElementCollection *E_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec,dim);

   // Vector L2 space for H 
   FiniteElementCollection *H_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *H_fes = new ParFiniteElementSpace(&pmesh,H_fec, dimc); 

   // H^-1/2 (curl) space for Ê   
   FiniteElementCollection * hatE_fec = nullptr;
   FiniteElementCollection * hatH_fec = nullptr; 
   FiniteElementCollection * F_fec = nullptr;
   if (dim == 3)
   {
      hatE_fec = new ND_Trace_FECollection(order,dim);
      hatH_fec = new ND_Trace_FECollection(order,dim);   
      F_fec = new ND_FECollection(test_order, dim);
   }
   else
   {
      hatE_fec = new RT_Trace_FECollection(order-1,dim);
      hatH_fec = new H1_Trace_FECollection(order,dim);   
      F_fec = new H1_FECollection(test_order, dim);
   } 
   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);
   ParFiniteElementSpace *hatH_fes = new ParFiniteElementSpace(&pmesh,hatH_fec);

   FiniteElementCollection * G_fec = new ND_FECollection(test_order, dim);

   // Coefficients
   Vector dim_zero(dim); dim_zero = 0.0;
   Vector dimc_zero(dimc); dimc_zero = 0.0;
   VectorConstantCoefficient E_zero(dim_zero);
   VectorConstantCoefficient H_zero(dimc_zero);


   ConstantCoefficient one(1.0);
   ConstantCoefficient eps2omeg2(epsilon*epsilon*omega*omega);
   ConstantCoefficient mu2omeg2(mu*mu*omega*omega);
   ConstantCoefficient muomeg(mu*omega);
   ConstantCoefficient negepsomeg(-epsilon*omega);
   ConstantCoefficient epsomeg(epsilon*omega);
   ConstantCoefficient negmuomeg(-mu*omega);

   DenseMatrix rot_mat(2);
   rot_mat(0,0) = 0.; rot_mat(0,1) = 1.;
   rot_mat(1,0) = -1.; rot_mat(1,1) = 0.;
   MatrixConstantCoefficient rot(rot_mat);
   ScalarMatrixProductCoefficient epsrot(epsomeg,rot);
   ScalarMatrixProductCoefficient negepsrot(negepsomeg,rot);

   // Normal equation weak formulation
   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(E_fes);
   trial_fes.Append(H_fes);
   trial_fes.Append(hatE_fes);
   trial_fes.Append(hatH_fes);

   test_fec.Append(F_fec);
   test_fec.Append(G_fec);

   ComplexParNormalEquations * a = new ComplexParNormalEquations(trial_fes,test_fec);
   a->StoreMatrices();

   // (E,∇ × F)
   a->AddTrialIntegrator(new TransposeIntegrator(new CurlIntegrator(one)),nullptr,0,0);

   // -i ω ϵ (E , G)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(negepsomeg)),0,1);

   // i ω μ (H, F)
   if (dim == 3)
   {
      a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(muomeg)),1,0);
   }
   else
   {
      a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(muomeg),1,0);
   }
   //  (H,∇ × G) 
   a->AddTrialIntegrator(new TransposeIntegrator(new CurlIntegrator(one)),nullptr,1,1);

   // < n×Ê,F>
   if (dim == 3)
   {
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,2,0);
   }
   else
   {
      a->AddTrialIntegrator(new TraceIntegrator,nullptr,2,0);
   }

   // < n×Ĥ ,G>
   a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,3,1);


   // test integrators 
   //space-induced norm for H(curl) × H(curl)
   if (dim == 3)
   {
      // (∇×F,∇×δF)
      a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,0,0);
      // (F,δF)
      a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,0,0);
   }
   else
   {
      // (∇F,∇δF)
      a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,0,0);
      // (F,δF)
      a->AddTestIntegrator(new MassIntegrator(one),nullptr,0,0);
   }

   // (∇×G ,∇× δG)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,1,1);
   // (G,δG)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

   // additional integrators for the adjoint graph norm
   if (adjoint_graph_norm)
   {   
      if(dim == 3)
      {
         // μ^2 ω^2 (F,δF)
         a->AddTestIntegrator(new VectorFEMassIntegrator(mu2omeg2),nullptr,0,0);
         // -i ω μ (F,∇ × δG) = (F, ω μ ∇ × δ G)
         a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(negmuomeg),0,1);
         // -i ω ϵ (∇ × F, δG)
         a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(negepsomeg),0,1);
         // i ω μ (∇ × G,δF)
         a->AddTestIntegrator(nullptr,new MixedVectorCurlIntegrator(epsomeg),1,0);
         // i ω ϵ (G, ∇ × δF )
         a->AddTestIntegrator(nullptr,new MixedVectorWeakCurlIntegrator(muomeg),1,0);
         // ϵ^2 ω^2 (G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2),nullptr,1,1);
      }
      else
      {
         // μ^2 ω^2 (F,δF)
         a->AddTestIntegrator(new MassIntegrator(mu2omeg2),nullptr,0,0);

         // -i ω μ (F,∇ × δG) = i (F, -ω μ ∇ × δ G)
         a->AddTestIntegrator(nullptr,
            new TransposeIntegrator(new CurlIntegrator(negmuomeg)),0,1);

         // -i ω ϵ (∇ × F, δG) = i (- ω ϵ A ∇ F,δG), A = [0 1; -1; 0]
         a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negepsrot),0,1);   

         // i ω μ (∇ × G,δF) = i (ω μ ∇ × G, δF )
         a->AddTestIntegrator(nullptr,new CurlIntegrator(muomeg),1,0);

         // i ω ϵ (G, ∇ × δF ) =  i (ω ϵ G, A ∇ δF) = i ( G , ω ϵ A ∇ δF) 
         a->AddTestIntegrator(nullptr,
                   new TransposeIntegrator(new MixedVectorGradientIntegrator(epsrot)),1,0);

         // or    i ( ω ϵ A^t G, ∇ δF) = i (- ω ϵ A G, ∇ δF)
         // a->AddTestIntegrator(nullptr,
                  //  new MixedVectorWeakDivergenceIntegrator(epsrot),1,0);
         // ϵ^2 ω^2 (G,δG)
         a->AddTestIntegrator(new VectorFEMassIntegrator(eps2omeg2),nullptr,1,1);            
      }
   }

   // RHS
   
   VectorFunctionCoefficient f_rhs_r(dim,rhs_func_r);
   VectorFunctionCoefficient f_rhs_i(dim,rhs_func_i);
   if (prob != 2)
   {
      a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_r),
                               new VectorFEDomainLFIntegrator(f_rhs_i),1);
   }
   
   VectorFunctionCoefficient hatEex_r(dim,hatE_exact_r);
   VectorFunctionCoefficient hatEex_i(dim,hatE_exact_i);

   VectorFunctionCoefficient hatHex_r(dimc,hatH_exact_r);
   VectorFunctionCoefficient hatHex_i(dimc,hatH_exact_i);

   FunctionCoefficient hatH_2D_ex_r(hatH_exact_scalar_r);
   FunctionCoefficient hatH_2D_ex_i(hatH_exact_scalar_i);


   Array<int> elements_to_refine;

   socketstream E_out_r;
   socketstream Eex_out_r;
   // socketstream E_out_i;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      E_out_r.open(vishost, visport);
      Eex_out_r.open(vishost, visport);
      // E_out_i.open(vishost, visport);
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0;
   if (myid == 0)
   {
      if (prob != 2)
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
         hatE_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         // hatH_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int j = 0; j < ess_tdof_list.Size(); j++)
      {
         ess_tdof_list[j] += E_fes->GetTrueVSize() + H_fes->GetTrueVSize();
                           // + hatE_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = H_fes->GetVSize();
      offsets[3] = hatE_fes->GetVSize();
      offsets[4] = hatH_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;
      double * xdata = x.GetData();

      ParComplexGridFunction hatE_gf(hatE_fes);
      hatE_gf.real().MakeRef(hatE_fes,&xdata[offsets[2]]);
      hatE_gf.imag().MakeRef(hatE_fes,&xdata[offsets.Last()+ offsets[2]]);

      ParComplexGridFunction hatH_gf(hatH_fes);
      hatH_gf.real().MakeRef(hatH_fes,&xdata[offsets[3]]);
      hatH_gf.imag().MakeRef(hatH_fes,&xdata[offsets.Last()+ offsets[3]]);

      if (dim == 3)
      {
         hatE_gf.ProjectBdrCoefficientTangent(hatEex_r,hatEex_i, ess_bdr);
         // hatH_gf.ProjectBdrCoefficientTangent(hatHex_r,hatHex_i, ess_bdr);
      }
      else
      {
         hatE_gf.ProjectBdrCoefficientNormal(hatEex_r,hatEex_i, ess_bdr);
         // hatH_gf.ProjectBdrCoefficient(hatH_2D_ex_r,hatH_2D_ex_i, ess_bdr);

      }
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
         HypreBoomerAMG * solver_E = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(0,0));
         solver_E->SetPrintLevel(0);
         solver_E->SetSystemsOptions(dim);
         HypreBoomerAMG * solver_H = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(1,1));
         solver_H->SetPrintLevel(0);
         solver_H->SetSystemsOptions(dim);
         M->SetDiagonalBlock(0,solver_E);
         M->SetDiagonalBlock(1,solver_H);
         M->SetDiagonalBlock(num_blocks,solver_E);
         M->SetDiagonalBlock(num_blocks+1,solver_H);
      }

      HypreSolver * solver_hatH = nullptr;
      HypreAMS * solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip,skip), 
                               hatE_fes);
      solver_hatE->SetPrintLevel(0);  
      if (dim == 2)
      {
         solver_hatH = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1));
         dynamic_cast<HypreBoomerAMG*>(solver_hatH)->SetPrintLevel(0);
      }
      else
      {
         solver_hatH = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(skip+1,skip+1), hatH_fes);
         dynamic_cast<HypreAMS*>(solver_hatH)->SetPrintLevel(0);
      }

      M->SetDiagonalBlock(skip,solver_hatE);
      M->SetDiagonalBlock(skip+1,solver_hatH);
      M->SetDiagonalBlock(skip+num_blocks,solver_hatE);
      M->SetDiagonalBlock(skip+num_blocks+1,solver_hatH);


      StopWatch chrono;

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-7);
      cg.SetAbsTol(1e-7);
      cg.SetMaxIter(100000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(*M); 
      cg.SetOperator(blockA);
      chrono.Clear();
      chrono.Start();
      cg.Mult(B, X);
      chrono.Stop();

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
         // mfem::out << "Mesh: " << oss.str() << std::endl;
         // mfem::out << "PCG time = " << pcg_time << std::endl;
      }

      int num_iter = cg.GetNumIterations();
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

      ParComplexGridFunction E(E_fes);
      E.real().MakeRef(E_fes,x.GetData());
      E.imag().MakeRef(E_fes,&x.GetData()[offsets.Last()]);

      VectorFunctionCoefficient E_ex_r(dim,E_exact_r);
      VectorFunctionCoefficient E_ex_i(dim,E_exact_i);

      ParComplexGridFunction H(H_fes);
      H.real().MakeRef(H_fes,&x.GetData()[offsets[1]]);
      H.imag().MakeRef(H_fes,&x.GetData()[offsets.Last()+offsets[1]]);

      VectorFunctionCoefficient H_ex_r(dimc,H_exact_r);
      VectorFunctionCoefficient H_ex_i(dimc,H_exact_i);
      
      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      double E_err_r = E.real().ComputeL2Error(E_ex_r);
      double E_err_i = E.imag().ComputeL2Error(E_ex_i);
      double H_err_r = H.real().ComputeL2Error(H_ex_r);
      double H_err_i = H.imag().ComputeL2Error(H_ex_i);

      double L2Error = sqrt(  E_err_r*E_err_r + E_err_i*E_err_i 
                            + H_err_r*H_err_r + H_err_i*H_err_i );

      ParComplexGridFunction Egf_ex(E_fes);
      ParComplexGridFunction Hgf_ex(H_fes);
      Egf_ex.ProjectCoefficient(E_ex_r, E_ex_i);
      Hgf_ex.ProjectCoefficient(H_ex_r, H_ex_i);

      double E_norm_r = Egf_ex.real().ComputeL2Error(E_zero);
      double E_norm_i = Egf_ex.imag().ComputeL2Error(E_zero);
      double H_norm_r = Hgf_ex.real().ComputeL2Error(H_zero);
      double H_norm_i = Hgf_ex.imag().ComputeL2Error(H_zero);

      double L2norm = sqrt(  E_norm_r*E_norm_r + E_norm_i*E_norm_i 
                           + H_norm_r*H_norm_r + H_norm_i*H_norm_i );

      double rel_err = L2Error/L2norm;

      double rate_err = (it) ? dim*log(err0/rel_err)/log((double)dof0/dofs) : 0.0;
      double rate_res = (it) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;

      err0 = rel_err;
      res0 = globalresidual;
      dof0 = dofs;
      if (myid == 0)
      {
         // mfem::out << "dof0     = " << dof0 << endl;
         // mfem::out << "residual = " << globalresidual << endl;
         // mfem::out << "num_iter = " << num_iter << endl;

         if (prob != 2)
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
      }

      if (visualization)
      {
         E_out_r << "parallel " << num_procs << " " << myid << "\n";
         E_out_r.precision(8);
         E_out_r << "solution\n" << pmesh << E.real() <<
                  "window_title 'Real Numerical Electric field' "
                  << flush;

         // E_out_i.precision(8);
         // E_out_i << "solution\n" << pmesh << E.imag() <<
         //          "window_title 'Imag Numerical Electric field' "
         //          << flush;         


         Eex_out_r.precision(8);
         Eex_out_r << "parallel " << num_procs << " " << myid << "\n";
         Eex_out_r << "solution\n" << pmesh << Egf_ex.real()  
                  << "window_title 'Real Exact Electric field' " 
                  << flush;
         // socketstream E_i_sock(vishost, visport);
         // E_i_sock.precision(8);
         // E_i_sock << "solution\n" << pmesh << Egf_ex.imag()  
         //          << "window_title 'Imag Exact Electric field' " 
         //          << flush;


      }

      if (it == pr-1)
         break;

      pmesh.GeneralRefinement(elements_to_refine,1,1);
      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   delete a;
   delete F_fec;
   delete G_fec;
   delete hatH_fes;
   delete hatH_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete H_fec;
   delete E_fec;
   delete H_fes;
   delete E_fes;

   return 0;
}                                       

void E_exact_r(const Vector &x, Vector & E_r)
{
   Vector curlE_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   Vector curlE_i;
   Vector curlcurlE_i;

   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   Vector E_r;
   Vector curlcurlE_r;

   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   Vector E_i;
   Vector curlcurlE_i;

   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{
   Vector E_r;
   Vector curlE_r;
   maxwell_solution_r(x,E_r,curlE_r,curlcurlE_r);
}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{
   Vector E_i;
   Vector curlE_i;
   maxwell_solution_i(x,E_i,curlE_i,curlcurlE_i);
}


void H_exact_r(const Vector &x, Vector & H_r)
{
   // H = i ∇ × E / ω μ  
   // H_r = - ∇ × E_i / ω μ  
   Vector curlE_i;
   curlE_exact_i(x,curlE_i);
   H_r.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_r(i) = - curlE_i(i) / (omega * mu);
   }
}

void H_exact_i(const Vector &x, Vector & H_i)
{
   // H = i ∇ × E / ω μ  
   // H_i =  ∇ × E_r / ω μ  
   Vector curlE_r;
   curlE_exact_r(x,curlE_r);
   H_i.SetSize(dimc);
   for (int i = 0; i<dimc; i++)
   {
      H_i(i) = curlE_r(i) / (omega * mu);
   }
}

void curlH_exact_r(const Vector &x,Vector &curlH_r)
{
   // ∇ × H_r = - ∇ × ∇ × E_i / ω μ  
   Vector curlcurlE_i;
   curlcurlE_exact_i(x,curlcurlE_i);
   curlH_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_r(i) = -curlcurlE_i(i) / (omega * mu);
   }
}

void curlH_exact_i(const Vector &x,Vector &curlH_i)
{
   // ∇ × H_i = ∇ × ∇ × E_r / ω μ  
   Vector curlcurlE_r;
   curlcurlE_exact_r(x,curlcurlE_r);
   curlH_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      curlH_i(i) = curlcurlE_r(i) / (omega * mu);
   }
}

void hatE_exact_r(const Vector & x, Vector & hatE_r)
{
   if (dim == 3)
   {
      E_exact_r(x,hatE_r);
   }
   else
   {
      Vector E_r;
      E_exact_r(x,E_r);
      hatE_r.SetSize(hatE_r.Size());
      // rotate E_hat
      hatE_r[0] = E_r[1];
      hatE_r[1] = -E_r[0];
   }
}

void hatE_exact_i(const Vector & x, Vector & hatE_i)
{
   if (dim == 3)
   {
      E_exact_i(x,hatE_i);
   }
   else
   {
      Vector E_i;
      E_exact_i(x,E_i);
      hatE_i.SetSize(hatE_i.Size());
      // rotate E_hat
      hatE_i[0] = E_i[1];
      hatE_i[1] = -E_i[0];
   }
}

void hatH_exact_r(const Vector & x, Vector & hatH_r)
{
   H_exact_r(x,hatH_r);
}

void hatH_exact_i(const Vector & x, Vector & hatH_i)
{
   H_exact_i(x,hatH_i);
}

double hatH_exact_scalar_r(const Vector & x)
{
   Vector hatH_r;
   H_exact_r(x,hatH_r);
   return hatH_r[0];
}

double hatH_exact_scalar_i(const Vector & x)
{
   Vector hatH_i;
   H_exact_i(x,hatH_i);
   return hatH_i[0];
}

// J = -i ω ϵ E + ∇ × H 
// J_r + iJ_i = -i ω ϵ (E_r + i E_i) + ∇ × (H_r + i H_i) 
void  rhs_func_r(const Vector &x, Vector & J_r)
{
   // J_r = ω ϵ E_i + ∇ × H_r
   Vector E_i, curlH_r;
   E_exact_i(x,E_i);
   curlH_exact_r(x,curlH_r);
   J_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_r(i) = omega * epsilon * E_i(i) + curlH_r(i);
   }
}

void  rhs_func_i(const Vector &x, Vector & J_i)
{
   // J_i = - ω ϵ E_r + ∇ × H_i
   Vector E_r, curlH_i;
   E_exact_r(x,E_r);
   curlH_exact_i(x,curlH_i);
   J_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_i(i) = -omega * epsilon * E_r(i) + curlH_i(i);
   }
}

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E, 
                      std::vector<complex<double>> &curlE, 
                      std::vector<complex<double>> &curlcurlE)
{
   double x = X(0);
   double y = X(1);
   double z;
   if (dim == 3) z = X(2);

   E.resize(dim);
   curlE.resize(dimc);
   curlcurlE.resize(dim);
   switch (prob)
   {
   case prob_type::polynomial:
   {
      if (dim == 3)
      {
         E[0] = y * z * (1.0 - y) * (1.0 - z);
         E[1] = x * y * z * (1.0 - x) * (1.0 - z);
         E[2] = x * y * (1.0 - x) * (1.0 - y);
         curlE[0] = (1.0 - x) * x * (y*(2.0*z-3.0)+1.0);
         curlE[1] = 2.0*(1.0 - y)*y*(x-z);
         curlE[2] = (z-1)*z*(1.0+y*(2.0*x-3.0));
         curlcurlE[0] = 2.0 * y * (1.0 - y) - (2.0 * x - 3.0) * z * (1 - z);
         curlcurlE[1] = 2.0 * y * (x * (1.0 - x) + (1.0 - z) * z);
         curlcurlE[2] = 2.0 * y * (1.0 - y) + x * (3.0 - 2.0 * z) * (1.0 - x);
      }
      else
      {
         E[0] = y * (1.0 - y);
         E[1] = x * y * (1.0 - x);
         curlE[0] = y*(3.0 - 2*x) - 1.0;
         curlcurlE[0] = 3.0 - 2*x;
         curlcurlE[1] = 2.0*y;
      }
   }
      break;

   case prob_type::plane_wave:
   {
      std::complex<double> zi(0,1);
      std::complex<double> pw = exp(-zi * omega * (X.Sum()));
      E[0] = pw;
      E[1] = 0.0;
      if (dim == 3)
      {
         E[2] = 0.0;
         curlE[0] = 0.0;
         curlE[1] = -zi * omega * pw;
         curlE[2] =  zi * omega * pw;

         curlcurlE[0] = 2.0 * omega * omega * pw;
         curlcurlE[1] = - omega * omega * pw;
         curlcurlE[2] = - omega * omega * pw;
      }
      else
      {
         curlE[0] = zi * omega * pw;
         curlcurlE[0] =   omega * omega * pw;
         curlcurlE[1] = - omega * omega * pw ;
      }
   }
      break;

   default:
      MFEM_VERIFY(dim == 3, "Fichera 'oven' problem only for dim = 3");
      if (abs(z -3.0) < 1e-10)
      {
         E[0] = sin(M_PI*y);
      }

      break;
   }
   
}

void maxwell_solution_r(const Vector & X, Vector &E_r, 
                        Vector &curlE_r, 
                        Vector &curlcurlE_r)
{
   E_r.SetSize(dim);
   curlE_r.SetSize(dimc);
   curlcurlE_r.SetSize(dim);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<dim ; i++)
   {
      E_r(i) = E[i].real();
      curlcurlE_r(i) = curlcurlE[i].real();
   }
   for (int i = 0; i<dimc; i++)
   {
      curlE_r(i) = curlE[i].real();
   }
}

void maxwell_solution_i(const Vector & X, Vector &E_i, 
                      Vector &curlE_i, 
                      Vector &curlcurlE_i)
{
   E_i.SetSize(dim);
   curlE_i.SetSize(dimc);
   curlcurlE_i.SetSize(dim);

   std::vector<complex<double>> E;
   std::vector<complex<double>> curlE;
   std::vector<complex<double>> curlcurlE;

   maxwell_solution(X,E,curlE,curlcurlE);
   for (int i = 0; i<dim; i++)
   {
      E_i(i) = E[i].imag();
      curlcurlE_i(i) = curlcurlE[i].imag();
   }
   for (int i = 0; i<dimc; i++)
   {
      curlE_i(i) = curlE[i].imag();
   }
}  
