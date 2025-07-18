//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "remap.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   // int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command line options.
   // 1. Parse command line options.
   int dim = 2;
   int order = 3;
   int qorder = 6;
   int ref_levels = 0;
   int nrTest = 1000;
   if (myid) { out.Disable(); }

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&qorder, "-qo", "--quad-order", "Quadrature order");
   args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&ref_levels, "-r", "--refine", "Mesh refinement levels");
   args.AddOption(&nrTest, "-n", "--test", "Number of tests to run");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh ser_mesh("../../data/mobius-strip.mesh", 1, 1);
   for (int i=0; i<ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   ser_mesh.Clear();

   auto f = [](const Vector &x)->real_t
   {
      return std::exp(x[0]*M_PI)*std::exp(x[1]*M_PI)+1;
   };

   auto df = [](const Vector &x, Vector &y) -> void
   {
      y.SetSize(2);
      y[0] = M_PI * std::exp(x[0]*M_PI) * std::sin(x[1]*M_PI);
      y[1] = M_PI * std::sin(x[0]*M_PI) * std::exp(x[1]*M_PI);
   };

   QuadratureSpace qspace(&mesh, qorder);
   H1_FECollection fec(order, mesh.Dimension(), BasisType::Positive);
   ParFiniteElementSpace fespace(&mesh, &fec);
   std::vector<FiniteElementSpace*> fes({&fespace});

   Array<int> offsets({0, qspace.GetSize(), fespace.GetTrueVSize()});
   offsets.PartialSum();
   BlockVector x(offsets);
   QuadratureFunction qf(&qspace, x.GetBlock(0).GetData());
   ParGridFunction gf(&fespace);
   gf.MakeTRef(&fespace, x.GetBlock(1).GetData());

   FunctionCoefficient coord0([](const Vector &x) { return x[0]; });
   FunctionCoefficient coord1([](const Vector &x) { return x[1]; });
   coord0.Project(qf);
   gf.ProjectCoefficient(coord1);
   gf.SetTrueVector();

   Array<int> space_idx({-1, 0});
   ComposedFunctional func(f, df, qspace, fes, space_idx);

   Vector y(1);
   BlockVector g(offsets);

   func.Mult(x, y);
   func.GetGradient().Mult(x, g);

   QuadratureFunction gf_qf(qspace);
   gf_qf.ProjectGridFunction(gf);
   std::vector<QuadratureFunction> qf_out({QuadratureFunction(qspace), QuadratureFunction(qspace)});
   FunctionCoefficient result_cf(f);
   VectorArrayCoefficient qf_gf_cf(2);
   qf_gf_cf.Set(0, new QuadratureFunctionCoefficient(qf));
   qf_gf_cf.Set(1, new GridFunctionCoefficient(&gf));
   QuadratureFunction qf_all(qspace, 2);
   qf_gf_cf.Project(qf_all);
   Vector all_point(2);
   Vector grad_point(2);
   real_t result = 0.0;
   for (int i=0; i<qspace.GetSize(); i++)
   {
      all_point.MakeRef(qf_all, i*2);
      result += f(all_point)*qspace.GetWeights()[i];
      df(all_point, grad_point);
      all_point = grad_point;
      qf_out[0][i] = grad_point[0];
      qf_out[1][i] = grad_point[1];
   }
   MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);

   out << "Functional value: " << y[0] << endl;
   out << "Expected value: " << result << endl;
   out << "Difference: " << std::abs(result - y[0]) << endl;

   qf_out[0] *= qspace.GetWeights();
   out << "Gradient diff [0]: " << g.GetBlock(0).DistanceTo(qf_out[0]) << std::endl;
   ParLinearForm lf(&fespace);
   QuadratureFunctionCoefficient qf_out1_cf(qf_out[1]);
   Vector lf_vec(fespace.GetTrueVSize());
   lf.AddDomainIntegrator(new QuadratureDomainLFIntegrator(qf_out[1], fespace));
   lf.Assemble();
   lf.ParallelAssemble(lf_vec);
   out << "Gradient diff [1]: " << g.GetBlock(1).DistanceTo(lf_vec) << std::endl;
   SparseMatrix A(qf);
}
