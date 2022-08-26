//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make pacoustics
//
// sample runs
// mpirun -np 4 ./pacoustics -o 3 -m ../../../data/inline-quad.mesh -sref 2 -pref 3 -rnum 4.1 -prob 0 -sc -graph-norm

//     - Δ p - ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω

// PML formulation
// - ∇⋅(|J| J^-1 J^-T ∇ p) - ω^2  |J| p = f

// First Order System

//  ∇ p + i ω α u = 0, in Ω
//  ∇⋅u + i ω β p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/(i ω), α:= J^T J / |J|, β:= |J| 

// UW-DPG:
// 
// p ∈ L^2(Ω), u ∈ (L^2(Ω))^dim 
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)  
// -(p,  ∇⋅v) + i ω (α u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)      
// -(u , ∇ q) + i ω (β p , q) + < û, q >  = (f,q)   ∀ q ∈ H^1(Ω)
//                                  p̂  = p_0     on ∂Ω 

// Note: 
// p̂ := p on Γ_h (skeleton)
// û := u on Γ_h  

// ----------------------------------------------------------------
// |   |     p       |     u       |    p̂      |    û    |  RHS    |
// ----------------------------------------------------------------
// | v | -(p, ∇⋅v)   | i ω (α u,v) | < p̂, v⋅n> |         |         |
// |   |             |             |           |         |         |
// | q | i ω (β p,q) |-(u , ∇ q)   |           | < û,q > |  (f,q)  |  

// where (q,v) ∈  H^1(Ω) × H(div,Ω) 

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "util/pml.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

complex<double> acoustics_solution(const Vector & X);
double p_bdr_data_r(const Vector &x);
double p_bdr_data_i(const Vector &x);

void alpha_function_re(const Vector & x, CartesianPML *, DenseMatrix & K);
void alpha_function_im(const Vector & x, CartesianPML *, DenseMatrix & K);
void alpha_function_abs2(const Vector & x, CartesianPML *, DenseMatrix & K);
double beta_function_re(const Vector & x, CartesianPML *);
double beta_function_im(const Vector & x, CartesianPML *);
double beta_function_abs2(const Vector & x, CartesianPML *);

int dim;
double omega;

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

   omega = 2.*M_PI*rnum;


   Mesh mesh(mesh_file, 1, 1);
   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }
   dim = mesh.Dimension();
   mesh.EnsureNCMesh();

   Array2D<double> length(dim, 2); length = 0.0;
   length[0][0] = 0.125;
   length[0][1] = 0.125;
   length[1][0] = 0.125;
   length[1][1] = 0.125;
   CartesianPML * pml = new CartesianPML(&mesh,length);
   pml->SetOmega(omega);

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   pml->SetAttributes(&pmesh);

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

   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector vec0(dim); vec0 = 0.;
   VectorConstantCoefficient vzero(vec0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   ConstantCoefficient negomeg(-omega);


   FunctionCoefficient bdr_data_re(p_bdr_data_r);
   FunctionCoefficient bdr_data_im(p_bdr_data_i);

   Array<ParFiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);

   test_fec.Append(q_fec);
   test_fec.Append(v_fec);


   Array<int> attr;
   Array<int> attrPML;
   if (pmesh.attributes.Size())
   {
      attr.SetSize(pmesh.attributes.Max());
      attrPML.SetSize(pmesh.attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (pmesh.attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }
   // Non-PML coefficients
   RestrictedCoefficient omeg_restr(omeg,attr);
   RestrictedCoefficient negomeg_restr(negomeg,attr);
   RestrictedCoefficient omeg2_restr(omeg2,attr);


   // PML coefficients
   // α 
   PmlMatrixCoefficient alpha_re(dim,alpha_function_re,pml);
   PmlMatrixCoefficient alpha_im(dim,alpha_function_im,pml);
   PmlMatrixCoefficient alpha2(dim,alpha_function_abs2,pml);
   // β 
   PmlCoefficient beta_re(beta_function_re,pml);
   PmlCoefficient beta_im(beta_function_im,pml);
   PmlCoefficient beta_abs2(beta_function_abs2,pml);

   ProductCoefficient omeg_beta_re(omeg,beta_re);
   ProductCoefficient omeg_beta_im(omeg,beta_im);
   ProductCoefficient negomeg_beta_re(negomeg,beta_re);
   ProductCoefficient negomeg_beta_im(negomeg,beta_im);
   ProductCoefficient omeg2_beta2(omeg2,beta_abs2);

   RestrictedCoefficient omeg_beta_re_restr(omeg_beta_re,attrPML);
   RestrictedCoefficient omeg_beta_im_restr(omeg_beta_im,attrPML);
   RestrictedCoefficient negomeg_beta_re_restr(negomeg_beta_re,attrPML);
   RestrictedCoefficient negomeg_beta_im_restr(negomeg_beta_im,attrPML);
   RestrictedCoefficient omeg2_beta2_restr(omeg2_beta2,attrPML);

   ScalarMatrixProductCoefficient omeg_alpha_re(omeg,alpha_re); 
   ScalarMatrixProductCoefficient omeg_alpha_im(omeg,alpha_im);
   ScalarMatrixProductCoefficient negomeg_alpha_re(negomeg,alpha_re);
   ScalarMatrixProductCoefficient negomeg_alpha_im(negomeg,alpha_im); 
   ScalarMatrixProductCoefficient omeg2_alpha2(omeg2,alpha2);

   MatrixRestrictedCoefficient omeg_alpha_re_restr(omeg_alpha_re,attrPML);
   MatrixRestrictedCoefficient omeg_alpha_im_restr(omeg_alpha_im,attrPML);
   MatrixRestrictedCoefficient negomeg_alpha_re_restr(negomeg_alpha_re,attrPML);
   MatrixRestrictedCoefficient negomeg_alpha_im_restr(negomeg_alpha_im,attrPML);
   MatrixRestrictedCoefficient omeg2_alpha2_restr(omeg2_alpha2,attrPML);

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices();
   // Not in PML
   // i ω (p,q)
   a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(omeg_restr),0,0);

   // In PML
   // i ω (p,q) = i ω ( (β_r p,q) + i (β_i p,q) )
   //           = (- ω b_i p ) + i (ω β_r p,q)      
   a->AddTrialIntegrator(new MixedScalarMassIntegrator(negomeg_beta_im_restr),
                         new MixedScalarMassIntegrator(omeg_beta_re_restr),0,0);
   // -(u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(negone)),nullptr,1,0);
   // -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),nullptr,0,1);

   //  i ω (α u,v)
   // Not in PML
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(omeg_restr)),1,1);
   // In PML
   // i ω (α u,v) =  i ω ( (α_re u,v) + i (α_im u,v) )
   //             = (-ω a_im u,v) + i (ω a_re u, v)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(negomeg_alpha_im_restr)),
                         new TransposeIntegrator(new VectorFEMassIntegrator(omeg_alpha_re_restr)),1,1);
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
      // Not in PML
      // -i ω (∇q,δv)
      a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negomeg_restr),0,1);

      // i ω (v,∇ δq)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakDivergenceIntegrator(negomeg_restr),1,0);

      // ω^2 (v,δv)
      a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2_restr),nullptr,1,1);

      // - i ω (∇⋅v,δq)   
      a->AddTestIntegrator(nullptr,new VectorFEDivergenceIntegrator(negomeg_restr),1,0);

      // i ω (q,∇⋅v)   
      a->AddTestIntegrator(nullptr,new MixedScalarWeakGradientIntegrator(negomeg_restr),0,1);

      // ω^2 (q,δq)
      a->AddTestIntegrator(new MassIntegrator(omeg2_restr),nullptr,0,0);

      // In PML
      // -i ω (α ∇q,δv) = -i ω ( (α_r ∇q,δv) + i (α_i ∇q,δv) )
      //                = (ω α_i ∇q,δv) + i (-ω α_r ∇q,δv) 
      a->AddTestIntegrator(new MixedVectorGradientIntegrator(omeg_alpha_im_restr),
                           new MixedVectorGradientIntegrator(negomeg_alpha_re_restr),0,1);

      // i ω (α^* v,∇ δq)  = i ω (ᾱ v,∇ δq) (since α is diagonal)
      //                   = i ω ( (α_r v,∇ δq) - i (α_i v,∇ δq)  
      //                   = (ω α_i v, ∇ δq) + i (ω α_r v,∇ δq )    
      a->AddTestIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg_alpha_im_restr),
                           new MixedVectorWeakDivergenceIntegrator(omeg_alpha_re_restr),1,0);

      // ω^2 (|α|^2 v,δv) α α^* = |α|^2 since α is diagonal   
      a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2_alpha2_restr),nullptr,1,1);
      
      // - i ω (β ∇⋅v,δq) = - i ω ( (β_re ∇⋅v,δq) + i (β_im ∇⋅v,δq) ) 
      //                  = (ω β_im ∇⋅v,δq) + i (-ω β_re ∇⋅v,δq )
      a->AddTestIntegrator(new VectorFEDivergenceIntegrator(omeg_beta_im_restr),
                           new VectorFEDivergenceIntegrator(negomeg_beta_re_restr),1,0);
      
      // i ω (β̄ q,∇⋅v) =  i ω ( (β_re ∇⋅v,δq) - i (β_im ∇⋅v,δq) ) 
      //               =  (ω β_im ∇⋅v,δq) + i (ω β_re ∇⋅v,δq )
      a->AddTestIntegrator(new MixedScalarWeakGradientIntegrator(omeg_beta_im_restr),
                           new MixedScalarWeakGradientIntegrator(omeg_beta_re_restr),0,1);
      
      // ω^2 (β̄ β q,δq) = (ω^2 |β|^2 )
      a->AddTestIntegrator(new MassIntegrator(omeg2_beta2_restr),nullptr,0,0);
   }


   Array<int> elements_to_refine;

   socketstream p_out_r;
   socketstream p_out_i;
   ParComplexGridFunction *pt = nullptr;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      p_out_r.open(vishost, visport);
      p_out_i.open(vishost, visport);
   }


   double res0 = 0.;
   int dof0;
   if (myid == 0)
   {
      mfem::out << "\n  Ref |" 
                << "       Mesh       |"
                << "    Dofs    |" 
                << "   ω   |" 
                << "  Residual  |" 
                << "  Rate  |" 
                << " PCG it |"
                << " PCG time |"  << endl;
      mfem::out << " --------------------"      
                <<  "---------------------"    
                <<  "---------------------"    
                <<  "--------------------------" << endl;      
   }

   for (int it = 0; it<=pr; it++)
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

      Array<int> int_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         int_bdr.SetSize(pmesh.bdr_attributes.Max());
         int_bdr = 0;
         int_bdr[1] = 1;
      }


      ParComplexGridFunction hatp_gf(hatp_fes);
      hatp_gf.real().MakeRef(hatp_fes,&xdata[offsets[2]]);
      hatp_gf.imag().MakeRef(hatp_fes,&xdata[offsets.Last()+ offsets[2]]);
      hatp_gf.ProjectBdrCoefficient(bdr_data_re,bdr_data_im, int_bdr);

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
      // solver_hatp->SetCycleNumSweeps(5, 5);
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
      cg.SetRelTol(1e-6);
      cg.SetAbsTol(0.0);
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


      int dofs = p_fes->GlobalTrueVSize()
               + u_fes->GlobalTrueVSize()
               + hatp_fes->GlobalTrueVSize()
               + hatu_fes->GlobalTrueVSize();

      double rate_res = (it) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;

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
      {
         if (visualization)
         {
            pt = new ParComplexGridFunction(p_fes);
            pt->real() = p.real();
            pt->imag() = p.imag();
         }
         break;
      }

      pmesh.GeneralRefinement(elements_to_refine,1,1);
      pml->SetAttributes(&pmesh);

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();   
   }
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      ParGridFunction x_t(p_fes);
      x_t = pt->real();

      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << num_procs << " " << myid << "\n"
               << "solution\n" << pmesh << x_t << "autoscale off\n"
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n" << flush;

      if (myid == 0)
      {
         cout << "GLVis visualization paused."
               << " Press space (in the GLVis window) to resume it.\n";
      }

      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         double t = (double)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos(2.0*M_PI*t), pt->real(), sin(2.0*M_PI*t), pt->imag(), x_t);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << x_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   delete pt;
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

double p_bdr_data_r(const Vector &x)
{
   return acoustics_solution(x).real();
}

double p_bdr_data_i(const Vector &x)
{
   return acoustics_solution(x).imag();
}

complex<double> acoustics_solution(const Vector & X)
{
   complex<double> zi = complex<double>(0., 1.);
   // double rk = omega;
   // double alpha = 45.0 * M_PI/180.;
   // double sina = sin(alpha); 
   // double cosa = cos(alpha);
   // // shift the origin
   // // double shift = -0.5;
   // double shift = 0.1;
   // double xprim=X(0) + shift; 
   // double yprim=X(1) + shift;
   // double  x = xprim*sina - yprim*cosa;
   // double  y = xprim*cosa + yprim*sina;
   // //wavelength
   // double rl = 2.*M_PI/rk;
   // // beam waist radius
   // double w0 = 0.05;
   // // function w
   // double fact = rl/M_PI/(w0*w0);
   // double aux = 1. + (fact*y)*(fact*y);
   // double w = w0*sqrt(aux);
   // double phi0 = atan(fact*y);
   // double r = y + 1./y/(fact*fact);
   // // pressure
   // complex<double> ze = - x*x/(w*w) - zi*rk*y - zi * M_PI * x * x/rl/r + zi*phi0/2.;
   // double pf = pow(2.0/M_PI/(w*w),0.25);
   // complex<double> zp = pf*exp(ze);
   // return zp;

   // plane wave
   double beta = omega/std::sqrt((double)X.Size());
   complex<double> alpha = beta * zi * X.Sum();
   return -exp(-alpha);
   // double x = X(0)-0.5;
   // double y = X(1)-0.5;
   // double r = sqrt(x*x + y*y);
   // double beta = omega * r;
   // complex<double> Ho = jn(0, beta) + zi * yn(0, beta);
   // return 0.25*zi*Ho;
}

// α:= J^T J / |J|
void alpha_function_re(const Vector & x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs,omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      K(i,i) = (pow(dxs[i], 2)/det).real();
   }
}
void alpha_function_im(const Vector & x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs,omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      K(i,i) = (pow(dxs[i], 2)/det).imag();
   }
}
void alpha_function_abs2(const Vector & x, CartesianPML * pml, DenseMatrix & K)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs,omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   K = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      complex<double> a = pow(dxs[i], 2)/det;
      K(i,i) = a.imag() * a.imag() + a.real() * a.real();
   }
}
double beta_function_re(const Vector & x, CartesianPML * pml)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs,omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   return det.real();
}
double beta_function_im(const Vector & x, CartesianPML * pml)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs,omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   return det.imag();
}
double beta_function_abs2(const Vector & x, CartesianPML * pml)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs,omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }
   return det.imag()*det.imag() + det.real()*det.real();
}