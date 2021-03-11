

//                                MFEM Example multigrid-grid Cycle

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "FOSLS.hpp"

using namespace std;
using namespace mfem;

void maxwell_solution(const Vector &x, std::vector<complex<double>> & sol, 
                                       std::vector<complex<double>> & curl,
                                       std::vector<complex<double>> & curl2);

void E_exact_re(const Vector &x, Vector &E);
void H_exact_re(const Vector &x, Vector &H);
void E_exact_im(const Vector &x, Vector &E);
void H_exact_im(const Vector &x, Vector &H);

void f_exact_re(const Vector &x, Vector &f);
void g_exact_re(const Vector &x, Vector &g);
void f_exact_im(const Vector &x, Vector &f);
void g_exact_im(const Vector &x, Vector &g);
void plotfield(socketstream &,ParMesh * pmesh,const ParGridFunction & , string &);

int dim;
double omega;
int exact = 0;

   // ----------------------------------------------------------------------
   // |   |            E             |             H          |     RHS    | 
   // ----------------------------------------------------------------------
   // | F | (curlE,curlF)+w^2(E,F)   | iw(curlH,F)+iw(H,curF) | -iw(J,F)   |
   // |   |                          |                        |            |
   // | G |-iw(E,curlG)-iw(curlE,G)  | (curlH,curlG)+w^2(H,G) | -(J,curlG) |

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/inline-hex.mesh";
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
   // omega = rnum;
   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   dim = mesh->Dimension();

   MFEM_VERIFY(dim == 3, "only 3D problems supported by this formulation");

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

   FiniteElementCollection *fec = new ND_FECollection(order,dim); 
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of True Dofs = " << size << endl;
   }

   VectorFunctionCoefficient E_ex_re(dim,E_exact_re);
   VectorFunctionCoefficient H_ex_re(dim,H_exact_re);
   VectorFunctionCoefficient E_ex_im(dim,E_exact_im);
   VectorFunctionCoefficient H_ex_im(dim,H_exact_im);
   VectorFunctionCoefficient f_ex_re(dim,f_exact_re);
   VectorFunctionCoefficient g_ex_re(dim,g_exact_re);
   VectorFunctionCoefficient f_ex_im(dim,f_exact_im);
   VectorFunctionCoefficient g_ex_im(dim,g_exact_im);

   int n = fespace->GetVSize();
   int N = fespace->GetTrueVSize();
   Array<int> block_offsets(5);
   block_offsets = n; 
   block_offsets[0] = 0;
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(5);
   block_trueOffsets = N;
   block_trueOffsets[0] = 0;
   block_trueOffsets.PartialSum();

   BlockVector X(block_trueOffsets), Rhs(block_trueOffsets);
   X = 0.0;  Rhs = 0.0;

   ComplexMaxwellFOSLS fosls(fespace);
   fosls.SetOmega(omega);
   Array<VectorFunctionCoefficient * > ess_data(4);
   ess_data[0] = &E_ex_re;
   ess_data[1] = &H_ex_re;
   ess_data[2] = &E_ex_im;
   ess_data[3] = &H_ex_im;
   fosls.SetEssentialData(ess_data);
   Array<VectorFunctionCoefficient * > loads(4);
   loads[0] = &f_ex_re;
   loads[1] = &g_ex_re;
   loads[2] = &f_ex_im;
   loads[3] = &g_ex_im;
   fosls.SetLoadData(loads);

   Array2D<HypreParMatrix *> Ah;
   fosls.GetFOSLSLinearSystem(Ah,X,Rhs);

   HypreParMatrix * A = HypreParMatrixFromBlocks(Ah);
   
   HypreAMS ams0(*Ah[0][0],fespace);
   HypreAMS ams1(*Ah[1][1],fespace);

   BlockDiagonalPreconditioner prec(block_trueOffsets);
   prec.SetDiagonalBlock(0,&ams0);
   prec.SetDiagonalBlock(1,&ams1);
   prec.SetDiagonalBlock(2,&ams0);
   prec.SetDiagonalBlock(3,&ams1);

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-6);
   // cg.SetAbsTol(1e-6);
   cg.SetMaxIter(400);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(prec);
   cg.Mult(Rhs, X);
   chrono.Stop();
   double t1 = chrono.RealTime();
   if (myid == 0)
   {
      cout << "PCG time = " << t1 << endl;
   }
   
   // {
   //    MUMPSSolver mumps;
   //    mumps.SetPrintLevel(0);
   //    mumps.SetOperator(*A);
   //    mumps.Mult(Rhs,X);
   // }
   ParGridFunction E_gf_re(fespace);
   ParGridFunction H_gf_re(fespace);
   ParGridFunction E_gf_im(fespace);
   ParGridFunction H_gf_im(fespace);
   E_gf_re = 0.0;
   E_gf_im = 0.0;
   H_gf_re = 0.0;
   H_gf_im = 0.0;

   E_gf_re.Distribute(&(X.GetBlock(0)));
   H_gf_re.Distribute(&(X.GetBlock(1)));
   E_gf_im.Distribute(&(X.GetBlock(2)));
   H_gf_im.Distribute(&(X.GetBlock(3)));

   double E_re_L2_Error = E_gf_re.ComputeL2Error(E_ex_re);
   double E_im_L2_Error = E_gf_im.ComputeL2Error(E_ex_im);
   double H_re_L2_Error = H_gf_re.ComputeL2Error(H_ex_re);
   double H_im_L2_Error = H_gf_im.ComputeL2Error(H_ex_im);

   ParGridFunction zero(fespace);
   zero = 0.0;
   double E_re_L2_norm = zero.ComputeL2Error(E_ex_re);
   double E_im_L2_norm = zero.ComputeL2Error(E_ex_im);
   double H_re_L2_norm = zero.ComputeL2Error(H_ex_re);
   double H_im_L2_norm = zero.ComputeL2Error(H_ex_im);
   if (myid == 0)
   {
      cout << "E_re L2 Error = " << E_re_L2_Error/E_re_L2_norm << endl;
      cout << "E_im L2 Error = " << E_im_L2_Error/E_im_L2_norm << endl;
      cout << "H_re L2 Error = " << H_re_L2_Error/H_re_L2_norm << endl;
      cout << "H_im L2 Error = " << H_im_L2_Error/H_im_L2_norm << endl;
   }   


   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock0(vishost, visport);
      socketstream sol_sock1(vishost, visport);
      socketstream sol_sock2(vishost, visport);
      socketstream sol_sock3(vishost, visport);
      string str0 = "E_re";
      plotfield(sol_sock0,pmesh, E_gf_re,str0);
      string str1 = "E_im";
      plotfield(sol_sock1,pmesh,E_gf_im,str1);
      string str2 = "H_re";
      plotfield(sol_sock2,pmesh,H_gf_re,str2);
      string str3 = "H_im";
      plotfield(sol_sock3,pmesh,H_gf_im,str3);

      ParGridFunction E_exact_re(fespace);
      ParGridFunction E_exact_im(fespace);
      ParGridFunction H_exact_re(fespace);
      ParGridFunction H_exact_im(fespace);
      E_exact_re.ProjectCoefficient(E_ex_re);
      E_exact_im.ProjectCoefficient(E_ex_im);
      H_exact_re.ProjectCoefficient(H_ex_re);
      H_exact_im.ProjectCoefficient(H_ex_im);

      socketstream sol_sock_ex0(vishost, visport);
      socketstream sol_sock_ex1(vishost, visport);
      socketstream sol_sock_ex2(vishost, visport);
      socketstream sol_sock_ex3(vishost, visport);
      str0 = "E_exact_re";
      plotfield(sol_sock_ex0,pmesh,E_exact_re,str0);
      str1 = "E_exact_im";
      plotfield(sol_sock_ex1,pmesh,E_exact_im,str1);
      str2 = "H_exact_re";
      plotfield(sol_sock_ex2,pmesh,H_exact_re,str2);
      str3 = "H_exact_im";
      plotfield(sol_sock_ex3,pmesh,H_exact_im,str3);
   }

   MPI_Finalize();
   return 0;
}


void maxwell_solution(const Vector &X, std::vector<complex<double>> &sol, 
                                       std::vector<complex<double>> &curl,
                                       std::vector<complex<double>> &curl2)
{
   double x = X(0), y = X(1), z = X(2);

   complex<double> zi(0,1);
   if (exact == 0)
   {
      sol[0] = y*(1.0-y)*z*(1.0-z) + zi * 2.0;
      sol[1] = y*x*(1.0-x)*z*(1.0-z)+ zi * 2.0;
      sol[2] = x*(1.0-x)*y*(1.0-y) + zi * 2.0;

      curl[0] = (1.0-x)*x*(y*(2.0*z-3.0)+1.0);
      curl[1] = 2.0*(1.0-y)*y*(x-z);
      curl[2] = (z-1.0)*z*(y*(2*x-3)+1.0);

      curl2[0] = (2.0*x-3.0)*(z-1.0)*z-2.0*y*y+2*y;
      curl2[1] = -2.0*y*(x*x-x+(z-1.0)*z);
      curl2[2] = 2*(x*(1.5-z)+x*x*(z-1.5)-y*y+y);
   }
   else
   {
      complex<double> alpha = zi * omega / sqrt(3);
      sol[0] = exp(alpha*(x+y+z));
      sol[1] = 0.0;
      sol[2] = 0.0;

      curl[0] = 0.0;
      curl[1] = alpha * sol[0];
      curl[2] = -alpha * sol[0];

      curl2[0] = -2.0 * alpha * alpha * sol[0];
      curl2[1] = alpha * alpha * sol[0];
      curl2[2] = curl2[1];
   }


   // sol[0] = 1.0 + 2.0*zi;
   // sol[1] = 1.0 + 2.0*zi;
   // sol[2] = 1.0 + 2.0*zi;
   // curl[0] = 0.0;
   // curl[1] =0.0;
   // curl[2] =0.0;
   // curl2[0] =0.0;
   // curl2[1] =0.0;
   // curl2[2] =0.0;  

}

void E_exact_re(const Vector &x, Vector &E)
{
   std::vector<complex<double>>sol(3);
   std::vector<complex<double>>curl(3);
   std::vector<complex<double>>curl2(3);
   maxwell_solution(x,sol,curl,curl2);
   for (int i=0; i<dim; i++)
   {
      E(i) = sol[i].real();
   }
}
void H_exact_re(const Vector &x, Vector &H)
{
   complex<double> zi(0,1);
   std::vector<complex<double>>sol(3);
   std::vector<complex<double>>curl(3);
   std::vector<complex<double>>curl2(3);
   // H = i curlE / w 
   maxwell_solution(x,sol,curl,curl2);
   for (int i=0; i<dim; i++)
   {
      H[i] = (zi * curl[i]/omega).real();
   }
}
void E_exact_im(const Vector &x, Vector &E)
{
   std::vector<complex<double>>sol(3);
   std::vector<complex<double>>curl(3);
   std::vector<complex<double>>curl2(3);
   maxwell_solution(x,sol,curl,curl2);
   for (int i=0; i<dim; i++)
   {
      E(i) = sol[i].imag();
   }
}
void H_exact_im(const Vector &x, Vector &H)
{
   complex<double> zi(0,1);
   std::vector<complex<double>>sol(3);
   std::vector<complex<double>>curl(3);
   std::vector<complex<double>>curl2(3);
   // H = i curlE / w 
   maxwell_solution(x,sol,curl,curl2);
   for (int i=0; i<dim; i++)
   {
      H[i] = (zi * curl[i]/omega).imag();
   }
}

void f_exact_re(const Vector &x, Vector &f)
{
   f = 0.0;
}
void g_exact_re(const Vector &x, Vector &g)
{
   // J = i omega E - curl H 
   // J = - i / omega (curl curl E - omega * omega E)
   complex<double> zi(0,1);
   std::vector<complex<double>>sol(3);
   std::vector<complex<double>>curl(3);
   std::vector<complex<double>>curl2(3);
   maxwell_solution(x,sol,curl,curl2);
   for (int i=0; i<dim; i++)
   {
      g(i) = (-zi / omega *(curl2[i] - omega * omega * sol[i])).real();
   }
}
void f_exact_im(const Vector &x, Vector &f)
{
   f = 0.0;
}
void g_exact_im(const Vector &x, Vector &g)
{
   // J = i omega E - curl H 
   // J = - i / omega (curl curl E - omega * omega E)
   complex<double> zi(0,1);
   std::vector<complex<double>>sol(3);
   std::vector<complex<double>>curl(3);
   std::vector<complex<double>>curl2(3);
   maxwell_solution(x,sol,curl,curl2);
   for (int i=0; i<dim; i++)
   {
      g(i) = (-zi / omega *(curl2[i] - omega * omega * sol[i])).imag();
   }
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
