#include "dfem/dfem_refactor.hpp"
#include "linalg/hypre.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <typename diffusion_t, typename force_t>
class DiffusionOperator : public Operator
{
   template <typename diffusion_du_t>
   class DiffusionJacobianOperator : public Operator
   {
   public:
      DiffusionJacobianOperator(const DiffusionOperator *diffusion,
                                std::shared_ptr<diffusion_du_t> diff_du) :
         Operator(diffusion->Height()), s(diffusion)
      {
         diff_du->Assemble(A);
         A.EliminateBC(s->ess_tdofs, Operator::DiagonalPolicy::DIAG_ONE);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         A.Mult(x, y);
      }

      const DiffusionOperator *s;
      HypreParMatrix A;
   };

public:
   DiffusionOperator(diffusion_t &diffusion, force_t &force,
                     Array<int> &ess_tdofs) :
      Operator(diffusion.Height()), diffusion(diffusion),
      force(force), ess_tdofs(ess_tdofs), f(force.Height()) {}

   void SetParameters(ParGridFunction &mesh_nodes)
   {
      diffusion.SetParameters({&mesh_nodes});
      force.SetParameters({&mesh_nodes});

      Vector zero;

      this->mesh_nodes.SetSpace(mesh_nodes.ParFESpace());
      this->mesh_nodes = mesh_nodes;
   }

   void Mult(const Vector &x, Vector &r) const override
   {
      diffusion.Mult(x, r);
      force.Mult(x, f);
      r -= f;
      r.SetSubVector(ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      ParGridFunction u(const_cast<ParFiniteElementSpace *>
                        (*std::get_if<const ParFiniteElementSpace *>
                         (&diffusion.solutions[0].data)));
      u.SetFromTrueDofs(x);

      auto dfdu = diffusion.template GetDerivativeWrt<0>({&u}, {&mesh_nodes});
      dfdu->Assemble(A);
      A.EliminateBC(ess_tdofs, DiagonalPolicy::DIAG_ONE);
      return A;
      // delete jacobian_operator;
      // jacobian_operator = new
      // DiffusionJacobianOperator<typename std::remove_pointer<decltype(dfdu.get())>::type>
      // (this, dfdu);
      // return *jacobian_operator;
   }

   diffusion_t &diffusion;
   force_t &force;

   const Array<int> ess_tdofs;
   mutable Vector f;

   mutable ParGridFunction mesh_nodes;

   mutable Operator *jacobian_operator = nullptr;
   mutable HypreParMatrix A;
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../data/ref-square.mesh";
   int polynomial_order = 2;
   int ir_order = 2;
   int refinements = 4;

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

   H1_FECollection potential_fec(polynomial_order, dim);
   ParFiniteElementSpace potential_fes(&mesh, &potential_fec);

   const IntegrationRule &potential_ir =
      IntRules.Get(potential_fes.GetFE(0)->GetGeomType(),
                   ir_order * potential_fec.GetOrder());

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   Array<int> ess_tdofs;
   potential_fes.GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdofs);

   ParGridFunction u(&potential_fes);
   u = 0.0;

   auto diffusion_kernel = [](const double &u,
                              const tensor<double, 2> &dudxi,
                              const tensor<double, 2, 2> &J,
                              const double &w)
   {
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      return std::tuple{(1.0 + u * u) * dudx * det(J) * w * transpose(invJ)};
   };

   std::tuple argument_operators_0{Value{"potential"}, Gradient{"potential"}, Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_operator_0{Gradient{"potential"}};
   ElementOperator op_0{diffusion_kernel, argument_operators_0, output_operator_0};

   auto force_kernel = [](const tensor<double, 2, 2> &J,
                          const double &w)
   {
      return std::tuple{1.0 * det(J) * w};
   };
   std::tuple argument_operators_1{Gradient{"coordinates"}, Weight{"integration_weights"}};
   std::tuple output_operator_1{Value{"potential"}};
   ElementOperator op_1{force_kernel, argument_operators_1, output_operator_1};

   std::array solutions{FieldDescriptor{&potential_fes, "potential"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator diffusion_op{solutions, parameters, std::tuple{op_0}, mesh, potential_ir};
   DifferentiableOperator force_op{solutions, parameters, std::tuple{op_1}, mesh, potential_ir};

   DiffusionOperator diffusion(diffusion_op, force_op, ess_tdofs);

   diffusion.SetParameters({*mesh_nodes});

   HypreBoomerAMG amg;
   amg.SetPrintLevel(0);

   CGSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(1e-12);
   solver.SetRelTol(1e-12);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(2);
   solver.SetPreconditioner(amg);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(diffusion);
   newton.SetSolver(solver);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);

   Vector zero;
   Vector x(potential_fes.GetTrueVSize());
   u.ParallelProject(x);
   newton.Mult(zero, x);

   u.SetFromTrueDofs(x);

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u << std::flush;

   return 0;
}
