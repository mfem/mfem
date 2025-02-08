#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static int A_flag_ = 1;

void A_exact(const Vector &pt, Vector &A)
{
   A.SetSize(3);
   A = 0.0;

   real_t x = pt[0];
   real_t y = pt[1];
   real_t z = pt[2];

   if (A_flag_ == 1)
   {
      real_t d = 9.0 / 16.0 + 9.0 / 4.0;

      if (x > 0.0)
      {
         A[0] = -8.0 * x * y * y * z / (9.0 * (x * x + y * y));
         A[0] += x * log((x * x + y * y) / d) / 24.0;
      }
      if (y > 0.0)
      {
         A[1] = -8.0 * y * y * y * z / (9.0 * (x * x + y * y));
         A[1] += -29.0 * y * log((x * x + y * y) / d) / 72.0;
      }
   }
   else
   {
      A[1] = 1.0 - z;
   }
}

void B_exact(const Vector &pt, Vector &B)
{
   B.SetSize(3);

   real_t x = pt[0];
   real_t y = pt[1];
   real_t z = pt[2];

   if (A_flag_ == 1)
   {
      B[0] = y * y;
      B[1] = -x * y;
      B[2] = x * (2.0 * z  - 1.0);

      B *= 8.0 * y / (9.0 * (x * x + y * y));
   }
   else
   {
      B[0] = 1.0;
   }
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/prism-hex-nc-trans.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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

   ConstantCoefficient ZCoef(0.0);
   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient ZeroCoef(zeroVec);
   VectorFunctionCoefficient ACoef(3, A_exact);
   VectorFunctionCoefficient BCoef(3, B_exact);

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   // int sdim = mesh.SpaceDimension();

   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   ND_FECollection nd_fec(order, dim);
   RT_FECollection rt_fec(order - 1, dim);
   L2_FECollection l2_fec(order - 1, dim);

   ParFiniteElementSpace nd_fes(&pmesh, &nd_fec);
   ParFiniteElementSpace rt_fes(&pmesh, &rt_fec);
   ParFiniteElementSpace l2_fes(&pmesh, &l2_fec);

   ParGridFunction nd_gf(&nd_fes);
   ParGridFunction dnd_gf(&rt_fes);
   ParGridFunction rt_gf(&rt_fes);
   ParGridFunction ddnd_gf(&l2_fes);
   ParGridFunction drt_gf(&l2_fes);

   ParDiscreteLinearOperator curl(&nd_fes, &rt_fes);
   curl.AddDomainInterpolator(new CurlInterpolator);
   curl.Assemble();

   ParDiscreteLinearOperator div(&rt_fes, &l2_fes);
   div.AddDomainInterpolator(new DivergenceInterpolator);
   div.Assemble();

   nd_gf.ProjectCoefficient(ACoef);
   real_t nrm_a = nd_gf.ComputeL2Error(ZeroCoef);

   real_t err_a = nd_gf.ComputeL2Error(ACoef);
   std::cout << "Error in A: " << err_a/nrm_a << endl;

   rt_gf.ProjectCoefficient(BCoef);
   real_t nrm_b = rt_gf.ComputeL2Error(ZeroCoef);

   curl.Mult(nd_gf, dnd_gf);

   real_t err_curla = dnd_gf.ComputeL2Error(BCoef);
   std::cout << "Error in CurlA: " << err_curla/nrm_b << endl;

   real_t err_b = rt_gf.ComputeL2Error(BCoef);
   std::cout << "Error in B: " << err_b/nrm_b << endl;

   div.Mult(dnd_gf, ddnd_gf);
   div.Mult(rt_gf, drt_gf);

   real_t err_divcurla = ddnd_gf.ComputeL2Error(ZCoef);
   std::cout << "Error in DivCurlA: " << err_divcurla << endl;

   real_t err_divb = drt_gf.ComputeL2Error(ZCoef);
   std::cout << "Error in DivB: " << err_divb << endl;

   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << nd_gf
               << "window_title 'Projected A'\n" << flush;
   }
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << dnd_gf
               << "window_title 'B = Curl A'\n" << flush;
   }
   if (visualization && false)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << rt_gf
               << "window_title 'Projected B'\n" << flush;
   }
}
