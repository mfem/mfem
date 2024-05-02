#include "dfem/dfem_refactor.hpp"
#include "fem/bilininteg.hpp"
#include "fem/coefficient.hpp"
#include "linalg/auxiliary.hpp"
#include "linalg/hypre.hpp"

using namespace mfem;
using mfem::internal::tensor;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../data/ref-square.mesh";
   int polynomial_order = 1;
   int ir_order = 2;
   int refinements = 1;

   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   L2_FECollection fec(polynomial_order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);

   const IntegrationRule &ir = IntRules.Get(fes.GetFE(0)->GetGeomType(),
                                            ir_order * fec.GetOrder());

   ParGridFunction u(&fes);

   // -\nabla \cdot (\nabla u + p * I) -> (\nabla u + p * I, \nabla v)
   auto advection_kernel = [](const tensor<double, 2> &dudxi,
                              const tensor<double, 2, 2> &J,
                              const double &w)
   {
      constexpr tensor<double, 2> b{1.0, 1.0};
      return std::tuple{dot(b, dudxi * inv(J)) * det(J) * w};
   };

   std::tuple argument_operators_0{Gradient{"quantity"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_operator_0{Value{"quantity"}};
   ElementOperator op_0{advection_kernel, argument_operators_0, output_operator_0};

   std::array solutions{FieldDescriptor{&fes, "quantity"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator advection_op{solutions, parameters, std::tuple{op_0}, mesh, ir};

   auto adv_du = advection_op.template GetDerivativeWrt<0>({&u}, {mesh_nodes});
   HypreParMatrix A;
   adv_du->Assemble(A);

   std::ofstream mmatofs("dfem_mat.dat");
   A.PrintMatlab(mmatofs);
   mmatofs.close();

   auto vector_func = [](const Vector &, Vector &u)
   {
      u = 1.0;
   };

   VectorFunctionCoefficient vel_coeff(dim, vector_func);

   ParBilinearForm adv_form(&fes);
   constexpr double alpha = 1.0;
   auto integ = new ConvectionIntegrator(vel_coeff, alpha);
   integ->SetIntRule(&ir);
   adv_form.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(vel_coeff, alpha));
   // adv_form.AddDomainIntegrator(integ);
   adv_form.Assemble();
   adv_form.Finalize();

   auto K = adv_form.ParallelAssemble();
   std::ofstream kmatofs("mfem_mat.dat");
   K->PrintMatlab(kmatofs);
   kmatofs.close();

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u << std::flush;

   return 0;
}
