#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <int dim = 2>
class VectorDiffusionQFunction
{
public:
   VectorDiffusionQFunction() = default;

   MFEM_HOST_DEVICE inline
   auto operator() (const tensor<real_t, dim, dim>& dudxi,
                    const tensor<real_t, dim, dim>& J,
                    const real_t& w) const
   {
      auto invJ = inv(J);
      return mfem::tuple{dudxi * invJ * det(J) * w * transpose(invJ)};
   }
};

int test_vector_diffusion(std::string mesh_file,
                          int refinements,
                          int polynomial_order)
{
   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   const int dim = mesh.Dimension();
   const int vdim = dim;
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   Array<int> ess_tdof;
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof);

   const IntegrationRule &ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(), 2 * h1fec.GetOrder() + 1);

   ParGridFunction u(&h1fes);

   auto f1 = [](const Vector& coords, Vector &u)
   {
      const real_t x = coords(0);
      const real_t y = coords(1);
      u(0) = 2.345 + 0.25 * x * x * y + y * y * x;
      u(1) = 2.345 - 0.25 * x * y * y + y * x * x;
   };

   VectorFunctionCoefficient u_c(dim, f1);
   u.ProjectCoefficient(u_c);

   constexpr int Potential = 0;
   constexpr int Coordinates = 1;

   std::vector solutions{FieldDescriptor{Potential, &h1fes}};
   std::vector parameters{FieldDescriptor{Coordinates, &mesh_fes}};
   DifferentiableOperator dop{solutions, parameters, mesh};

   VectorDiffusionQFunction vector_diffusion_kernel;
   mfem::tuple input_operators{Gradient<Potential>{}, Gradient<Coordinates>{}, Weight{}};
   mfem::tuple output_operator{Gradient<Potential>{}};
   auto derivatives = std::integer_sequence<size_t, Potential> {};
   dop.AddDomainIntegrator(vector_diffusion_kernel, input_operators,
                           output_operator, ir, derivatives);

   Vector x(u), y1(h1fes.GetTrueVSize()), y2(h1fes.GetTrueVSize());

   ParBilinearForm A_form(&h1fes);
   auto A_integ = new VectorDiffusionIntegrator(vdim);
   A_integ->SetIntegrationRule(ir);
   A_form.AddDomainIntegrator(A_integ);
   A_form.Assemble();
   A_form.Finalize();

   HypreParMatrix *A_mfem = A_form.ParallelAssemble();
   A_mfem->PrintMatlab(out);
   out << "\n";

   dop.SetParameters({mesh_nodes});
   dop.Mult(x, y1);
   y1.HostRead();

   HypreParMatrix A_dfem;
   dop.GetDerivative(Potential, {&u}, {mesh_nodes})->Assemble(A_dfem);
   A_dfem.PrintMatlab(out);

   A_form.Mult(x, y2);
   y2.HostRead();

   Vector diff(y2);
   diff -= y1;
   if (diff.Norml2() > 1e-10)
   {
      out << "||F(u) - ex||_l2 = " << diff.Norml2() << "\n";
      print_vector(diff);
      print_vector(y1);
      print_vector(y2);
      return 1;
   }

   return 0;
}

DFEM_TEST_MAIN(test_vector_diffusion);
