#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <typename stress_t>
class ElasticityOperator : public Operator
{
public:
   ElasticityOperator(stress_t &stress, Array<int> &ess_tdofs) :
      stress(stress),
      ess_tdofs(ess_tdofs)
   {

   }

   stress_t stress;
   Array<int> ess_tdofs;
};

int test_nonlinear_elasticity_3d(std::string mesh_file,
                                 int refinements,
                                 int polynomial_order)
{
   constexpr int dim = 3;
   constexpr int vdim = dim;

   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(polynomial_order);
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection h1fec(polynomial_order, dim);
   ParFiniteElementSpace h1fes(&mesh, &h1fec, vdim);

   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(
                      0)->GetDim() - 1);

   out << "#qp: " << ir.GetNPoints() << "\n";
   out << "#dof_el: " << h1fes.GetRestrictionMatrix()->Height() / mesh.GetNE() <<
       "\n";

   ParGridFunction u(&h1fes);

   ConstantCoefficient l_coeff(0.5), m_coeff(0.25);

   auto elasticity_kernel = [](const tensor<real_t, dim, dim> &dudxi,
                               const tensor<real_t, dim, dim> &J,
                               const double &w)
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      real_t D1 = 100.0;
      real_t C1 = 50.0;
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      real_t detF = det(I + dudx);
      real_t p = -2.0 * D1 * detF * (detF - 1);
      auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      auto sigma = -(p / detF) * I + 2.0 * (C1 / pow(detF, 5.0 / 3.0)) * devB;
      return mfem::tuple{sigma * det(J) * w * transpose(invJ)};
   };

   mfem::tuple argument_operators{Gradient{"displacement"}, Gradient{"coordinates"}, Weight{}};
   mfem::tuple output_operator{Gradient{"displacement"}};

   ElementOperator op{elasticity_kernel, argument_operators, output_operator};

   std::array solutions{FieldDescriptor{&h1fes, "displacement"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop{solutions, parameters, mfem::tuple{op}, mesh, ir};

   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm b(&h1fes);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   b.Assemble();
   auto B = b.ParallelAssemble();

   Vector X = u.GetTrueVector();

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetRelTol(1e-8);
   gmres.SetMaxIter(5000);
   gmres.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(gmres);
   newton.SetOperator(dop);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(100);
   newton.SetPrintLevel(1);

   dop.SetParameters({mesh_nodes});

   Vector zero;
   newton.Mult(zero, X);

   return 0;
}

DFEM_TEST_MAIN(test_nonlinear_elasticity_3d);
