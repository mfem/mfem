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

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;
   double theta = 0.0;
   bool adjoint_graph_norm = false;
   bool static_cond = false;

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
   dim = mesh.Dimension();
   mesh.EnsureNCMesh();
   mesh.UniformRefinement();
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
   
   
   if (static_cond) { a->EnableStaticCondensation(); }
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


   Vector x(2*offsets.Last());
   x = 0.;
   double * xdata = x.GetData();
   FunctionCoefficient hatpex_r(hatp_exact_r);
   FunctionCoefficient hatpex_i(hatp_exact_i);
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
   int numblocks = BlockA_r->NumRowBlocks();
   Array2D<HypreParMatrix *> Ab_r(numblocks,numblocks);
   Array2D<HypreParMatrix *> Ab_i(numblocks,numblocks);

   for (int i = 0; i<numblocks; i++)
   {
      for (int j = 0; j<numblocks; j++)
      {
         Ab_r(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_r->GetBlock(i,j));
         Ab_i(i,j) = dynamic_cast<HypreParMatrix*>(&BlockA_i->GetBlock(i,j));
      }
   }
   HypreParMatrix * A_r = HypreParMatrixFromBlocks(Ab_r);
   HypreParMatrix * A_i = HypreParMatrixFromBlocks(Ab_i);

   ComplexHypreParMatrix Ac(A_r,A_i,false,false);

   HypreParMatrix * A = Ac.GetSystemMatrix();

   if (myid == 0)
   {   
      mfem::out << "Size of the linear system: " << A->Height() << std::endl;
   }

   MUMPSSolver mumps;
   mumps.SetOperator(*A);
   mumps.Mult(B,X);

   a->RecoverFEMSolution(X,x);

   ParComplexGridFunction p(p_fes);
   p.real().MakeRef(p_fes,x.GetData());
   p.imag().MakeRef(p_fes,&x.GetData()[offsets.Last()]);

   ParComplexGridFunction pgf_ex(p_fes);
   FunctionCoefficient p_ex_r(p_exact_r);
   FunctionCoefficient p_ex_i(p_exact_i);
   pgf_ex.ProjectCoefficient(p_ex_r, p_ex_i);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock_r(vishost, visport);
   sol_sock_r << "parallel " << num_procs << " " << myid << "\n";
   sol_sock_r.precision(8);
   sol_sock_r << "solution\n" << pmesh << p.real() << flush;

   socketstream sol_sock_i(vishost, visport);
   sol_sock_i << "parallel " << num_procs << " " << myid << "\n";
   sol_sock_i.precision(8);
   sol_sock_i << "solution\n" << pmesh << p.imag() << flush;


   socketstream ex_sock_r(vishost, visport);
   ex_sock_r << "parallel " << num_procs << " " << myid << "\n";
   ex_sock_r.precision(8);
   ex_sock_r << "solution\n" << pmesh << pgf_ex.real() << flush;

   socketstream ex_sock_i(vishost, visport);
   ex_sock_i << "parallel " << num_procs << " " << myid << "\n";
   ex_sock_i.precision(8);
   ex_sock_i << "solution\n" << pmesh << pgf_ex.imag() << flush;


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
   return cos(omega*x.Sum()/std::sqrt((double)x.Size()));
}

double p_exact_i(const Vector &x)
{
   return -sin(omega*x.Sum()/std::sqrt((double)x.Size()));
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
   grad = -omega/std::sqrt((double)x.Size()) * sin(omega * x.Sum()/std::sqrt((double)x.Size()));
}

void gradp_exact_i(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   grad = -omega/std::sqrt((double)x.Size()) * cos(omega * x.Sum()/std::sqrt((double)x.Size()));
}

double d2_exact_r(const Vector &x)
{
   return -omega * omega * cos(omega*x.Sum()/std::sqrt((double)x.Size()));
}

double d2_exact_i(const Vector &x)
{
   return  omega * omega * sin(omega*x.Sum()/std::sqrt((double)x.Size()));
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







