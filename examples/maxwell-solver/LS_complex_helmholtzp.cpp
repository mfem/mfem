

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "FOSLS.hpp"
#include "lor.hpp"

using namespace std;
using namespace mfem;

int dim;
double omega;
int exact = 0;

void helmholtz_solution(const Vector &x, complex<double> & sol, 
                        std::vector<complex<double>> & grad,
                        complex<double> & grad2);

double p_exact_re(const Vector &x);
void u_exact_re(const Vector &x, Vector &u);
double p_exact_im(const Vector &x);
void u_exact_im(const Vector &x, Vector &u);
void gradp_exact_re(const Vector &x, Vector &gradu);
double divu_exact_re(const Vector &x);
void gradp_exact_im(const Vector &x, Vector &gradu);
double divu_exact_im(const Vector &x);


void f_exact_re(const Vector &x, Vector &f);
double g_exact_re(const Vector &x);
void f_exact_im(const Vector &x, Vector &f);
double g_exact_im(const Vector &x);
void plotfield(socketstream &,ParMesh * pmesh,const ParGridFunction & , string &);

// ----------------------------------------------------------------------
// |   |            p             |             u           |    RHS    | 
// ----------------------------------------------------------------------
// | q | (grad p,grad q)+w^2(p,q) |-iw(div u,q)+iw(u,grad q)| -iw(f,q)  |
// |   |                          |                         |           |
// | v | iw(p,div v)-iw(grad p,v) | (div u,div v)+w^2(u,v)  | (f,div v) |
// ----------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   int sr = 1;
   int pr = 1;
   double rnum=1.0;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&sr, "-sr", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pr", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");                  
   args.AddOption(&exact, "-solution", "--exact_solution",
                  "Exact solution : 0-polynomial, 1-plane wave");                  
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
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   omega = 2.0 * M_PI * rnum;
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   dim = mesh->Dimension();

   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int i = 0; i < pr; i++ )
   {
      pmesh->UniformRefinement();
   }

   int btype = BasisType::GaussLobatto;
   ParMesh pmesh_lor(pmesh, order, btype);

   unique_ptr<FiniteElementCollection> H1fec_ho, H1fec_lor;
   unique_ptr<FiniteElementCollection> RTfec_ho, RTfec_lor;

   H1fec_ho.reset(new H1_FECollection(order, dim));
   H1fec_lor.reset(new H1_FECollection(1, dim));
   RTfec_ho.reset(new RT_FECollection(order-1, dim, BasisType::GaussLobatto, BasisType::Integrated));
   RTfec_lor.reset(new RT_FECollection(0, dim, BasisType::GaussLobatto, BasisType::Integrated));
   
   ParFiniteElementSpace H1fes_ho(pmesh, H1fec_ho.get());
   ParFiniteElementSpace H1fes_lor(&pmesh_lor, H1fec_lor.get());
   ParFiniteElementSpace RTfes_ho(pmesh, RTfec_ho.get());
   ParFiniteElementSpace RTfes_lor(&pmesh_lor, RTfec_lor.get());

   HYPRE_Int H1size = H1fes_ho.GlobalTrueVSize();
   HYPRE_Int RTsize = RTfes_ho.GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of H1 True Dofs = " << H1size << endl;
      cout << "Number of RT True Dofs = " << RTsize << endl;
   }

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      H1fes_ho.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   FunctionCoefficient p_ex_re(p_exact_re);
   VectorFunctionCoefficient u_ex_re(dim,u_exact_re);
   FunctionCoefficient p_ex_im(p_exact_im);
   VectorFunctionCoefficient u_ex_im(dim,u_exact_im);

   VectorFunctionCoefficient f_ex_re(dim,f_exact_re);
   FunctionCoefficient g_ex_re(g_exact_re);
   VectorFunctionCoefficient f_ex_im(dim,f_exact_im);
   FunctionCoefficient g_ex_im(g_exact_im);

   int n0 = H1fes_ho.GetVSize();
   int N0 = H1fes_ho.GetTrueVSize();
   int n1 = RTfes_ho.GetVSize();
   int N1 = RTfes_ho.GetTrueVSize();
   Array<int> block_offsets(5);
   block_offsets[0] = 0;
   block_offsets[1] = n0;
   block_offsets[2] = n1;
   block_offsets[3] = n0;
   block_offsets[4] = n1;
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(5);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = N0;
   block_trueOffsets[2] = N1;
   block_trueOffsets[3] = N0;
   block_trueOffsets[4] = N1;
   block_trueOffsets.PartialSum();

   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector X(block_trueOffsets), Rhs(block_trueOffsets);
   x = 0.0;  rhs = 0.0; X = 0.0;  Rhs = 0.0;

   ParGridFunction p_gf_re, p_gf_im, u_gf_re, u_gf_im;

   p_gf_re.MakeRef(&H1fes_ho,x.GetBlock(0)); p_gf_re = 0.0;
   u_gf_re.MakeRef(&RTfes_ho,x.GetBlock(1)); u_gf_re = 0.0;
   p_gf_im.MakeRef(&H1fes_ho,x.GetBlock(2)); p_gf_im = 0.0;
   u_gf_im.MakeRef(&RTfes_ho,x.GetBlock(3)); u_gf_im = 0.0;

   // E_gf_re.ProjectBdrCoefficientTangent(E_ex_re,ess_bdr);
   // E_gf_im.ProjectBdrCoefficientTangent(E_ex_im,ess_bdr);
   p_gf_re.ProjectCoefficient(p_ex_re);
   p_gf_im.ProjectCoefficient(p_ex_im);

// ----------------------------------------------------------------------
// |   |            p             |             u           |    RHS    | 
// ----------------------------------------------------------------------
// | q | (grad p,grad q)+w^2(p,q) |-iw(div u,q)+iw(u,grad q)| -iw(g,q)  |
// |   |                          |                         |           |
// | v | iw(p,div v)-iw(grad p,v) | (div u,div v)+w^2(u,v)  | (g,div v) |
// ----------------------------------------------------------------------

   // for convinience we convert the above 2 x 2 blocks to 4 x 4 in order
   // to accomodate complex valued operators

   // A = (grad p,grad q)+w^2(p,q)
   // B = (div u,div v)+w^2(u,v)
   // C = -w(div u,q) + w(u,grad q)
   // D = w(p,div v)-w(grad p,v)
   // b0 = w(g_im,q)
   // b1 = (g_re,div v)
   // b2 = -w(g_re,q)
   // b3 = (g_im,div v)

   // | A   0   0  -C |  | p_re |     | b0 |
   // | 0   B  -D   0 |  | u_re |  =  | b1 |
   // | 0   C   A   0 |  | p_im |     | b2 |
   // | D   0   0   B |  | u_im |     | b3 |

   ConstantCoefficient one(1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient negomeg(-omega);
   ConstantCoefficient omeg2(omega * omega);

   ProductCoefficient wgi(omeg,g_ex_im);
   ProductCoefficient negwgr(negomeg,g_ex_re);

   ParLinearForm b0, b1, b2, b3;
   b0.Update(&H1fes_ho,rhs.GetBlock(0),0);
   b1.Update(&RTfes_ho,rhs.GetBlock(1),0);
   b2.Update(&H1fes_ho,rhs.GetBlock(2),0);
   b3.Update(&RTfes_ho,rhs.GetBlock(3),0);


   b0.AddDomainIntegrator(new DomainLFIntegrator(wgi));
   b1.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(g_ex_re));
   b2.AddDomainIntegrator(new DomainLFIntegrator(negwgr));
   b3.AddDomainIntegrator(new VectorFEDomainLFDivIntegrator(g_ex_im));
   
   b0.Assemble();
   b1.Assemble();
   b2.Assemble();
   b3.Assemble();

   Array2D<HypreParMatrix *> Ah(4,4); 
   for (int i = 0; i<4; i++)
   {
      for (int j = 0; j<4; j++)
      {
         Ah[i][j] = nullptr;
      }
   }

   // A00 = (grad p,grad q)+w^2(p,q)
   ParBilinearForm a00(&H1fes_ho);
   a00.AddDomainIntegrator(new DiffusionIntegrator(one));
   a00.AddDomainIntegrator(new MassIntegrator(omeg2));
   a00.Assemble();
   a00.EliminateEssentialBC(ess_bdr,x.GetBlock(0),rhs.GetBlock(0),mfem::Operator::DIAG_ONE);
   a00.Finalize();
   Ah[0][0] = a00.ParallelAssemble();

   // -C = w(div u,q) - w(u,grad q)
   ParMixedBilinearForm a03(&RTfes_ho,&H1fes_ho);
   // w(divu,q)
   a03.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
   // -w(u, gradq)
   a03.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   a03.Assemble();
   a03.EliminateTestDofs(ess_bdr);
   a03.Finalize();
   Ah[0][3] = a03.ParallelAssemble();

   // A11 = (div u,div v)+w^2(u,v)
   ParBilinearForm a11(&RTfes_ho);
   a11.AddDomainIntegrator(new DivDivIntegrator(one));
   a11.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a11.Assemble();
   a11.Finalize();
   Ah[1][1] = a11.ParallelAssemble();

   // A12 = -w(p,div v)+w(grad p,v)
   ParMixedBilinearForm a12(&H1fes_ho,&RTfes_ho);
   // -w(p,divv)
   a12.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(omeg));
   // w(grad p,v)
   a12.AddDomainIntegrator(new MixedVectorGradientIntegrator(omeg));
   a12.Assemble();
   a12.EliminateTrialDofs(ess_bdr,x.GetBlock(2),rhs.GetBlock(1));
   a12.Finalize();
   Ah[1][2] = a12.ParallelAssemble();

   // A21 = -w(div u,q) + w(u,grad q)
   ParMixedBilinearForm a21(&RTfes_ho,&H1fes_ho);
   // -w(div u,q) 
   a21.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
   // w(u,grad q)
   a21.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(negomeg));
   a21.Assemble();
   a21.EliminateTestDofs(ess_bdr);
   a21.Finalize();
   Ah[2][1] = a21.ParallelAssemble();

   // A22 = (grad p,grad q)+w^2(p,q)
   ParBilinearForm a22(&H1fes_ho);
   a22.AddDomainIntegrator(new DiffusionIntegrator(one));
   a22.AddDomainIntegrator(new MassIntegrator(omeg2));
   a22.Assemble();
   a22.EliminateEssentialBC(ess_bdr,x.GetBlock(2),rhs.GetBlock(2),mfem::Operator::DIAG_ONE);
   a22.Finalize();
   Ah[2][2] = a22.ParallelAssemble();


   // A30 = w(p,div v)-w(grad p,v)
   ParMixedBilinearForm a30(&H1fes_ho,&RTfes_ho);
   // w(p,div v)
   a30.AddDomainIntegrator(new MixedScalarWeakGradientIntegrator(negomeg));
   // -w(grad p,v)
   a30.AddDomainIntegrator(new MixedVectorGradientIntegrator(negomeg));
   a30.Assemble();
   a30.EliminateTrialDofs(ess_bdr,x.GetBlock(0),rhs.GetBlock(3));
   a30.Finalize();
   Ah[3][0] = a30.ParallelAssemble();

   ParBilinearForm a33(&RTfes_ho);
   a33.AddDomainIntegrator(new DivDivIntegrator(one));
   a33.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a33.Assemble();
   a33.Finalize();
   Ah[3][3] = a33.ParallelAssemble();

   for (int i = 0; i<2; i++)
   {
      H1fes_ho.GetRestrictionMatrix()->Mult(x.GetBlock(2*i), X.GetBlock(2*i));
      H1fes_ho.GetProlongationMatrix()->MultTranspose(rhs.GetBlock(2*i),Rhs.GetBlock(2*i));
      RTfes_ho.GetRestrictionMatrix()->Mult(x.GetBlock(2*i+1), X.GetBlock(2*i+1));
      RTfes_ho.GetProlongationMatrix()->MultTranspose(rhs.GetBlock(2*i+1),Rhs.GetBlock(2*i+1));
   }

   HypreParMatrix * A = HypreParMatrixFromBlocks(Ah);


// -----------------------------------------------------
//        L O R    P R E C O N D I T I O N E R 
// -----------------------------------------------------
   Array2D<HypreParMatrix *> Ah_lor(4,4); 
   for (int i = 0; i<4; i++)
   {
      for (int j = 0; j<4; j++)
      {
         Ah_lor[i][j] = nullptr;
      }
   }

   ParBilinearForm a00_lor(&H1fes_lor);
   a00_lor.AddDomainIntegrator(new DiffusionIntegrator(one));
   a00_lor.AddDomainIntegrator(new MassIntegrator(omeg2));
   a00_lor.Assemble();
   a00_lor.EliminateEssentialBC(ess_bdr,mfem::Operator::DIAG_ONE);
   a00_lor.Finalize();
   Ah_lor[0][0] = a00_lor.ParallelAssemble();

   ParMixedBilinearForm a03_lor(&RTfes_lor,&H1fes_lor);
   a03_lor.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(omeg));
   a03_lor.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(omeg));
   a03_lor.Assemble();
   a03_lor.EliminateTestDofs(ess_bdr);
   a03_lor.Finalize();
   Ah_lor[0][3] = a03_lor.ParallelAssemble();
   Ah_lor[3][0] = Ah_lor[0][3]->Transpose();

   ParBilinearForm a11_lor(&RTfes_lor);
   a11_lor.AddDomainIntegrator(new DivDivIntegrator(one));
   a11_lor.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a11_lor.Assemble();
   a11_lor.Finalize();
   Ah_lor[1][1] = a11_lor.ParallelAssemble();


   ParMixedBilinearForm a21_lor(&RTfes_lor,&H1fes_lor);
   a21_lor.AddDomainIntegrator(new MixedScalarDivergenceIntegrator(negomeg));
   a21_lor.AddDomainIntegrator(new MixedVectorWeakDivergenceIntegrator(negomeg));
   a21_lor.Assemble();
   a21_lor.EliminateTestDofs(ess_bdr);
   a21_lor.Finalize();
   Ah_lor[2][1] = a21_lor.ParallelAssemble();
   Ah_lor[1][2] = Ah_lor[2][1]->Transpose();

   ParBilinearForm a22_lor(&H1fes_lor);
   a22_lor.AddDomainIntegrator(new DiffusionIntegrator(one));
   a22_lor.AddDomainIntegrator(new MassIntegrator(omeg2));
   a22_lor.Assemble();
   a22_lor.EliminateEssentialBC(ess_bdr,mfem::Operator::DIAG_ONE);
   a22_lor.Finalize();
   Ah_lor[2][2] = a22_lor.ParallelAssemble();

   ParBilinearForm a33_lor(&RTfes_lor);
   a33_lor.AddDomainIntegrator(new DivDivIntegrator(one));
   a33_lor.AddDomainIntegrator(new VectorFEMassIntegrator(omeg2));
   a33_lor.Assemble();
   a33_lor.Finalize();
   Ah_lor[3][3] = a33_lor.ParallelAssemble();

   HypreParMatrix * A_lor = HypreParMatrixFromBlocks(Ah_lor);


// -----------------------------------------------------
// -----------------------------------------------------

   FiniteElement::MapType t = FiniteElement::H_DIV;
   Array<int> perm = ComputeVectorFE_LORPermutation(RTfes_ho, RTfes_lor, t);

   HypreBoomerAMG * amg_p0 = new HypreBoomerAMG(*Ah[0][0]);
   amg_p0->SetPrintLevel(0);
   HypreBoomerAMG * amg_lor_p0 = new HypreBoomerAMG(*Ah_lor[0][0]);
   amg_lor_p0->SetPrintLevel(0);
   HypreBoomerAMG * amg_p2 = new HypreBoomerAMG(*Ah[2][2]);
   amg_p2->SetPrintLevel(0);
   HypreBoomerAMG * amg_lor_p2 = new HypreBoomerAMG(*Ah_lor[2][2]);
   amg_lor_p2->SetPrintLevel(0);

   Solver *prec1 = nullptr;
   Solver *prec3 = nullptr;
   Solver *prec1_lor = nullptr;
   Solver *prec3_lor = nullptr;
   if (dim == 2) 
   {
      prec1 = new HypreAMS(*Ah[1][1],&RTfes_ho);
      dynamic_cast<HypreAMS *>(prec1)->SetPrintLevel(0);
      prec3 = new HypreAMS(*Ah[3][3],&RTfes_ho);
      dynamic_cast<HypreAMS *>(prec3)->SetPrintLevel(0);
      prec1_lor = new HypreAMS(*Ah_lor[1][1],&RTfes_lor);
      dynamic_cast<HypreAMS *>(prec1_lor)->SetPrintLevel(0);
      prec3_lor = new HypreAMS(*Ah_lor[3][3],&RTfes_lor);
      dynamic_cast<HypreAMS *>(prec3_lor)->SetPrintLevel(0);
   }
   else
   {
      prec1 = new HypreADS(*Ah[1][1],&RTfes_ho);
      dynamic_cast<HypreADS *>(prec1)->SetPrintLevel(0);
      prec3 = new HypreADS(*Ah[3][3],&RTfes_ho);
      dynamic_cast<HypreADS *>(prec3)->SetPrintLevel(0);
      prec1_lor = new HypreADS(*Ah_lor[1][1],&RTfes_lor);
      dynamic_cast<HypreADS *>(prec1_lor)->SetPrintLevel(0);
      prec3_lor = new HypreADS(*Ah_lor[3][3],&RTfes_lor);
      dynamic_cast<HypreADS *>(prec3_lor)->SetPrintLevel(0);
   }
   // 1st preconditioner: Exact LOR with direct solver
   ComplexLORSolver M_lor_exact(*A_lor, perm);

   // 2nd preconditioner: AMG/AMS on the high order system
   BlockDiagonalPreconditioner M(block_trueOffsets);
   M.SetDiagonalBlock(0,amg_p0);
   M.SetDiagonalBlock(1,prec1);
   M.SetDiagonalBlock(2,amg_p2);
   M.SetDiagonalBlock(3,prec3);

   // 3rd preconditioner: AMG/AMS on the LOR system
   BlockDiagonalPreconditioner M_lor2(block_trueOffsets);
   M_lor2.SetDiagonalBlock(0,amg_lor_p0);
   M_lor2.SetDiagonalBlock(1,prec1_lor);
   M_lor2.SetDiagonalBlock(2,amg_lor_p2);
   M_lor2.SetDiagonalBlock(3,prec3_lor);

   ComplexLORSolver M_lor(*A_lor, perm,false,&M_lor2);

   Vector Y(X), Z(X);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-6);
   cg.SetMaxIter(5000);
   cg.SetPrintLevel(3);
   cg.SetOperator(*A);
   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M_lor_exact);
   cg.Mult(Rhs, X);
   chrono.Stop();
   cout << "PCG Exact LOR time = " << chrono.RealTime() << endl;

   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M_lor);
   cg.Mult(Rhs, Y);
   chrono.Stop();
   cout << "PCG AMG/AMS LOR time = " << chrono.RealTime() << endl;

   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M);
   cg.Mult(Rhs, Z);
   chrono.Stop();
   cout << "PCG AMG/AMS HO time = " << chrono.RealTime() << endl;
   {
      MUMPSSolver mumps;
      mumps.SetPrintLevel(0);
      mumps.SetOperator(*A);
      mumps.Mult(Rhs,X);
   }


   p_gf_re = 0.0;
   p_gf_im = 0.0;
   u_gf_re = 0.0;
   u_gf_im = 0.0;

   p_gf_re.Distribute(&(X.GetBlock(0)));
   u_gf_re.Distribute(&(X.GetBlock(1)));
   p_gf_im.Distribute(&(X.GetBlock(2)));
   u_gf_im.Distribute(&(X.GetBlock(3)));

   ConvergenceStudy ratesH1;
   ConvergenceStudy ratesRT;

   VectorFunctionCoefficient gradp_ex(dim,gradp_exact_re);
   FunctionCoefficient divu_ex(divu_exact_re);

   ratesH1.AddH1GridFunction(&p_gf_re,&p_ex_re,&gradp_ex);
   ratesRT.AddHdivGridFunction(&u_gf_re,&u_ex_re,&divu_ex);

   ratesH1.Print(true);
   ratesRT.Print(true);

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << p_gf_re <<
               "window_title 'Numerical Pressure (real part)' "
               << flush;
      socketstream sol_sockex(vishost, visport);
      ParGridFunction p_ex(&H1fes_ho);
      p_ex.ProjectCoefficient(p_ex_re);
      sol_sockex << "parallel " << num_procs << " " << myid << "\n";
      sol_sockex.precision(8);
      sol_sockex << "solution\n" << *pmesh << p_ex <<
               "window_title 'Exact Pressure (real part)' "
               << flush;         
   }

   MPI_Finalize();

   return 0;
}



void helmholtz_solution(const Vector &X, complex<double> &sol, 
                                         std::vector<complex<double>> &grad,
                                         complex<double> &grad2)
{
   double x = X(0), y = X(1);
   double z;
   if (dim == 3 ) z = X(2);

   complex<double> zi(0,1);
   if (exact == 0)
   {
      if (dim == 2)
      {
         sol = x*(1.0-x) * y*(1.0-y);

         grad[0] = (1.0 - 2*x) * y*(1.0 - y);
         grad[1] = (1.0 - 2*y) * x*(1.0 - x);
         grad2 = -2 * y*(1.0 - y) - 2 * x*(1.0 - x);
      }
      else
      {
         sol = x*(1.0-x) * y*(1.0-y) * z*(1.0-z);
         grad[0] = (1.0 - 2*x) * y*(1.0 - y) * z*(1.0-z);
         grad[1] = (1.0 - 2*y) * x*(1.0 - x) * z*(1.0-z);
         grad[2] = (1.0 - 2*z) * x*(1.0 - x) * y*(1.0-y);
         grad2 = -2 * y*(1.0 - y) * z*(1.0-z)  
                 -2 * x*(1.0 - x) * z*(1.0-z) 
                 -2 * x*(1.0 - x) * y*(1.0-y);
      }
   }
   else
   {
      complex<double> alpha;
      if (dim == 2)
      {
         alpha = zi * omega / sqrt(2);
         sol = exp(alpha*(x+y));
         grad[0] = alpha * sol;
         grad[1] = alpha * sol;
         grad2 = 2.0*alpha*alpha*sol;
      }
      else
      {
         alpha = zi * omega / sqrt(3);
         sol = exp(alpha*(x+y+z));
         grad[0] = alpha * sol;
         grad[1] = alpha * sol;
         grad[2] = alpha * sol;
         grad2 = 3.0*alpha*alpha*sol;
      }
   }
}

double p_exact_re(const Vector &x)
{
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   return sol.real();
}

double p_exact_im(const Vector &x)
{
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   return sol.imag();
}

void gradp_exact_re(const Vector &x, Vector &gradp)
{
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   for (int i=0; i<dim; i++)
   {
      gradp[i] = grad[i].real();
   }
}
void gradp_exact_im(const Vector &x, Vector &gradp)
{
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   for (int i=0; i<dim; i++)
   {
      gradp[i] = grad[i].real();
   }
}


void u_exact_re(const Vector &x, Vector &u)
{
   complex<double> zi(0,1);
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   // u = i grad p / w 
   for (int i=0; i<dim; i++)
   {
      u[i] = (zi * grad[i]/omega).real();
   }
}

void u_exact_im(const Vector &x, Vector &u)
{
   complex<double> zi(0,1);
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   // u = i grad p / w 
   for (int i=0; i<dim; i++)
   {
      u[i] = (zi * grad[i]/omega).imag();
   }
}

double divu_exact_re(const Vector &x)
{
   complex<double> zi(0,1);
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   return (zi/omega * grad2).real();

}
double divu_exact_im(const Vector &x)
{
   complex<double> zi(0,1);
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);
   return (zi/omega * grad2).imag();
}

void f_exact_re(const Vector &x, Vector &f)
{
   f = 0.0;
}

void f_exact_im(const Vector &x, Vector &f)
{
   f = 0.0;
}

double g_exact_re(const Vector &x)
{
   // f = i omega p + div u  
   // f = i / omega *( omega * omega p + grad2) 
   complex<double> zi(0,1);
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);

   return (zi / omega *(omega * omega * sol + grad2)).real();
}

double g_exact_im(const Vector &x)
{
   // f = i omega p + div u  
   // f = i / omega *( omega * omega p + grad2) 
   complex<double> zi(0,1);
   complex<double>sol;
   std::vector<complex<double>>grad(dim);
   complex<double>grad2;
   helmholtz_solution(x,sol,grad,grad2);

   return (zi / omega *(omega * omega * sol + grad2)).imag();
}

void plotfield(socketstream & socket, ParMesh * pmesh, const ParGridFunction & pgf, string & title )
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   ostringstream oss;
   oss << title;
   socket << "parallel " << num_procs << " " << myid << "\n";
   socket.precision(8);
   socket << "solution\n" << *pmesh << pgf 
          << "window_title '" << oss.str() << "'" << flush;
}
