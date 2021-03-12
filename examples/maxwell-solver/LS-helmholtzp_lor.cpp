
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "FOSLS.hpp"
#include "lor.hpp"
using namespace std;
using namespace mfem;

// #define DEFINITE



double p_exact(const Vector &x);
void u_exact(const Vector &x, Vector & u);
double rhs_func(const Vector &x);
void gradp_exact(const Vector &x, Vector &gradu);
double divu_exact(const Vector &x);
double d2_exact(const Vector &x);

#ifdef DEFINITE   
   bool definite = true;
#else
   bool definite = false;
#endif
int dim;
double omega;
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

   // 4. Refine the serial mesh on all processors to increase the resolution.
   for (int i = 0; i < sr; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

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

   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = H1fes_ho.TrueVSize();
   block_trueOffsets[2] = RTfes_ho.TrueVSize();
   block_trueOffsets.PartialSum();

   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
   trueX = 0.0;  trueRhs = 0.0;

   FunctionCoefficient p_ex(p_exact);
   FunctionCoefficient f_rhs(rhs_func);
   VectorFunctionCoefficient gradp_ex(dim,gradp_exact);
   VectorFunctionCoefficient u_ex(dim,u_exact);
   FunctionCoefficient divu_ex(divu_exact);

   Vector trueY(trueX);
   Vector trueZ(trueX);

   Array<ParFiniteElementSpace *> fes_ho(2);
   fes_ho[0] = &H1fes_ho;
   fes_ho[1] = &RTfes_ho;

   HelmholtzFOSLS ho_system(fes_ho,definite);

   ho_system.SetOmega(omega);
   Array<FunctionCoefficient * > F_rhs(1);
   F_rhs[0] = &f_rhs;
   ho_system.SetLoadData(F_rhs);

   Array<FunctionCoefficient * > P_ex(1);
   P_ex[0] = &p_ex;
   ho_system.SetEssentialData(P_ex);

   Array<ParFiniteElementSpace *> fes_lor(2);
   fes_lor[0] = &H1fes_lor;
   fes_lor[1] = &RTfes_lor;
   HelmholtzFOSLS lor_system(fes_lor,definite);
   lor_system.SetOmega(omega);

   Array2D<HypreParMatrix *> Ah_ho(2,2);
   ho_system.GetFOSLSLinearSystem(Ah_ho,trueX,trueRhs);

   Array2D<HypreParMatrix *> Ah_lor(2,2);
   lor_system.GetFOSLSMatrix(Ah_lor);
   HypreParMatrix * A_ho = HypreParMatrixFromBlocks(Ah_ho);
   HypreParMatrix * A_lor = HypreParMatrixFromBlocks(Ah_lor);

   HypreBoomerAMG * amg_p = new HypreBoomerAMG(*Ah_ho[0][0]);
   amg_p->SetPrintLevel(0);
   HypreBoomerAMG * amg_lor_p = new HypreBoomerAMG(*Ah_lor[0][0]);
   amg_lor_p->SetPrintLevel(0);

   Solver *prec = nullptr;
   Solver *prec_lor = nullptr;
   if (dim == 2) 
   {
      prec = new HypreAMS(*Ah_ho[1][1],&RTfes_ho);
      dynamic_cast<HypreAMS *>(prec)->SetPrintLevel(0);
      prec_lor = new HypreAMS(*Ah_lor[1][1],&RTfes_lor);
      dynamic_cast<HypreAMS *>(prec_lor)->SetPrintLevel(0);
   }
   else
   {
      prec = new HypreADS(*Ah_ho[1][1],&RTfes_ho);
      dynamic_cast<HypreADS *>(prec)->SetPrintLevel(0);
      prec_lor = new HypreADS(*Ah_lor[1][1],&RTfes_lor);
      dynamic_cast<HypreADS *>(prec_lor)->SetPrintLevel(0);
   }

   BlockDiagonalPreconditioner M(block_trueOffsets);
   BlockDiagonalPreconditioner M_lor2(block_trueOffsets);

   FiniteElement::MapType t = FiniteElement::H_DIV;
   Array<int> perm = ComputeVectorFE_LORPermutation(RTfes_ho, RTfes_lor, t);
   
   LORSolver M_lor(*A_lor, perm);
   M.SetDiagonalBlock(0,amg_p);
   ScaledOperator S(prec,1.0);
   M.SetDiagonalBlock(1,&S);

   M_lor2.SetDiagonalBlock(0,amg_lor_p);
   ScaledOperator S_lor(prec_lor,1.0);
   M_lor2.SetDiagonalBlock(1,&S_lor);

   LORSolver M_lor_inexact(*A_lor, perm, false, &M_lor2);

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   // GMRESSolver cg(MPI_COMM_WORLD);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-6);
   // cg.SetAbsTol(1e-6);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(3);
   cg.SetOperator(*A_ho);
   // cg.SetPreconditioner(M);
   cg.SetPreconditioner(M_lor);
   cg.Mult(trueRhs, trueX);
   chrono.Stop();
   cout << "LOR exact - PCG time " << chrono.RealTime() << endl;

   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M_lor_inexact);
   cg.Mult(trueRhs, trueY);

   chrono.Stop();
   cout << "LOR inexact PCG time " << chrono.RealTime() << endl;

   chrono.Clear();
   chrono.Start();
   cg.SetPreconditioner(M);
   cg.Mult(trueRhs, trueZ);
   

   chrono.Stop();
   cout << "AMG/AMS PCG time " << chrono.RealTime() << endl;

   for (int i = 0; i<2; i++)
   {
      for (int j = 0; j<2; j++)
      {
         delete Ah_ho[i][j];
         delete Ah_lor[i][j];
      }
   }

   ParGridFunction p_gf(&H1fes_ho);
   ParGridFunction u_gf(&RTfes_ho);
   ParGridFunction p_zero(&H1fes_ho);
   ParGridFunction u_zero(&RTfes_ho);
   p_gf = 0.0; p_zero = 0.0;
   u_gf = 0.0; u_zero = 0.0;
   p_gf.Distribute(&(trueX.GetBlock(0)));
   u_gf.Distribute(&(trueX.GetBlock(1)));

   double H1_error = p_gf.ComputeH1Error(&p_ex,&gradp_ex);
   double H1_norm  = p_zero.ComputeH1Error(&p_ex,&gradp_ex);
   double Hdiv_error = u_gf.ComputeHDivError(&u_ex,&divu_ex);
   double Hdiv_norm = u_zero.ComputeHDivError(&u_ex,&divu_ex);


   if (myid == 0)
   {
      cout << "H1 rel error     = " << H1_error/H1_norm << endl;
      cout << "H(div) rel error = " << Hdiv_error/Hdiv_norm << endl;
   }

   // 10. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << p_gf <<
               "window_title 'Numerical Pressure (real part)' "
               << flush;
   }

   // // 11. Free the used memory.
   delete amg_lor_p;
   delete amg_p;
   delete prec;
   delete prec_lor;
   delete pmesh;
   MPI_Finalize();
   return 0;
}

double rhs_func(const Vector &x)
{
   double p = p_exact(x);
   double divu = divu_exact(x);
#ifdef DEFINITE   
   return -divu + omega * p;
#else
   return divu + omega * p;
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