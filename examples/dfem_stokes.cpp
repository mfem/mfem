#include "dfem/dfem.hpp"

using namespace mfem;
using mfem::internal::tensor;

template <typename momentum_t, typename mass_conservation_t>
class NavierStokesOperator : public Operator
{
   template <typename momentum_du_t, typename momentum_dp_t>
   class NavierStokesJacobianOperator : public Operator
   {
   public:
      NavierStokesJacobianOperator(const NavierStokesOperator *ns,
                                   std::shared_ptr<momentum_du_t> mom_du,
                                   std::shared_ptr<momentum_dp_t> mom_dp) :
         Operator(ns->Height()), ns(ns), block_op(ns->block_offsets)
      {
         mom_du->Assemble(A);
         A.EliminateBC(ns->vel_ess_tdofs, Operator::DiagonalPolicy::DIAG_ONE);

         mom_dp->Assemble(D);
         D.EliminateRows(ns->vel_ess_tdofs);

         Dt = new TransposeOperator(D);

         block_op.SetBlock(0, 0, &A);
         block_op.SetBlock(0, 1, &D);
         block_op.SetBlock(1, 0, Dt);
         // std::ofstream amatofs("dfem_mat.dat");
         // block_op.PrintMatlab(amatofs);
         // amatofs.close();
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         block_op.Mult(x, y);
      }

      ~NavierStokesJacobianOperator()
      {
         delete Dt;
      }

      const NavierStokesOperator *ns = nullptr;
      HypreParMatrix A, D;
      TransposeOperator *Dt = nullptr;
      BlockOperator block_op;
   };

public:
   NavierStokesOperator(momentum_t &momentum,
                        mass_conservation_t &mass_conservation,
                        Array<int> &offsets, Array<int> &vel_ess_tdofs) :
      Operator(offsets.Last()), momentum(momentum),
      mass_conservation(mass_conservation),
      block_offsets(offsets), vel_ess_tdofs(vel_ess_tdofs) {}

   void SetParameters(ParGridFunction &mesh_nodes)
   {
      momentum.SetParameters({&mesh_nodes});
      mass_conservation.SetParameters({&mesh_nodes});
      this->mesh_nodes.SetSpace(mesh_nodes.ParFESpace());
      this->mesh_nodes = mesh_nodes;
   }

   void Mult(const Vector &x, Vector &r) const override
   {
      Vector ru(r.ReadWrite() + block_offsets[0],
                block_offsets[1] - block_offsets[0]);
      Vector rp(r.ReadWrite() + block_offsets[1],
                block_offsets[2] - block_offsets[1]);

      momentum.Mult(x, ru);

      mass_conservation.Mult(x, rp);

      ru.SetSubVector(vel_ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      xtmp = x;
      BlockVector xb(xtmp.ReadWrite(), block_offsets);

      ParGridFunction u(const_cast<ParFiniteElementSpace *>
                        (*std::get_if<const ParFiniteElementSpace *>
                         (&momentum.solutions[0].data)));
      ParGridFunction p(const_cast<ParFiniteElementSpace *>
                        (*std::get_if<const ParFiniteElementSpace *>
                         (&momentum.solutions[1].data)));
      u.SetFromTrueDofs(xb.GetBlock(0));
      p.SetFromTrueDofs(xb.GetBlock(1));
      auto mom_du = momentum.template GetDerivativeWrt<0>({&u, &p}, {&mesh_nodes});
      auto mom_dp = momentum.template GetDerivativeWrt<1>({&u, &p}, {&mesh_nodes});
      delete jacobian_operator;
      jacobian_operator = new NavierStokesJacobianOperator<
      typename std::remove_pointer<decltype(mom_du.get())>::type,
      typename std::remove_pointer<decltype(mom_dp.get())>::type>(this, mom_du,
                                                                  mom_dp);
      return *jacobian_operator;
   }

   momentum_t &momentum;
   mass_conservation_t &mass_conservation;

   const Array<int> block_offsets;
   const Array<int> vel_ess_tdofs;
   mutable Vector xtmp;

   mutable ParGridFunction mesh_nodes;

   mutable Operator *jacobian_operator = nullptr;
};

double reynolds = 10.0;

int main(int argc, char *argv[])
{
   constexpr int dim = 3;
   constexpr int vdim = dim;

   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../data/ref-cube.mesh";
   int polynomial_order = 2;
   int ir_order = 2;
   int refinements = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&refinements, "-r", "--refinements", "");
   args.AddOption(&reynolds, "-rey", "--reynolds", "");
   args.ParseCheck();

   Mesh mesh_serial = Mesh(mesh_file);
   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.SetCurvature(1);
   mesh_serial.Clear();

   ParGridFunction* mesh_nodes = static_cast<ParGridFunction *>(mesh.GetNodes());
   ParFiniteElementSpace &mesh_fes = *mesh_nodes->ParFESpace();

   H1_FECollection velocity_fec(polynomial_order, dim);
   ParFiniteElementSpace velocity_fes(&mesh, &velocity_fec, dim);

   H1_FECollection pressure_fec(polynomial_order - 1, dim);
   ParFiniteElementSpace pressure_fes(&mesh, &pressure_fec);

   const IntegrationRule &velocity_ir =
      IntRules.Get(velocity_fes.GetFE(0)->GetGeomType(),
                   ir_order * velocity_fec.GetOrder());

   const IntegrationRule &pressure_ir =
      IntRules.Get(pressure_fes.GetFE(0)->GetGeomType(),
                   ir_order * pressure_fec.GetOrder());

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;
   Array<int> vel_ess_tdofs;
   velocity_fes.GetEssentialTrueDofs(bdr_attr_is_ess, vel_ess_tdofs);

   ParGridFunction u(&velocity_fes);
   ParGridFunction p(&pressure_fes);

   auto u_f = [](const Vector &coords, Vector &u)
   {
      const double x = coords(0);
      const double z = coords(2);
      if (z >= 1.0)
      {
         u(0) = 1.0;
      }
      else
      {
         u(0) = 0.0;
      }
      u(1) = 0.0;
      u(2) = 0.0;
   };
   auto u_coef = VectorFunctionCoefficient(dim, u_f);

   u.ProjectCoefficient(u_coef);
   p = 0.0;

   // -\nabla \cdot (\nabla u + p * I) -> (\nabla u + p * I, \nabla v)
   auto momentum_kernel = [](const tensor<double, dim> &u,
                             const tensor<double, dim, dim> &dudxi,
                             const double &p,
                             const tensor<double, dim, dim> &J,
                             const double &w)
   {
      static constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      double Re = reynolds;
      return mfem::tuple{(outer(u, u) - 1.0 / Re * dudx + p * I) * det(J) * w * transpose(invJ)};
   };

   mfem::tuple argument_operators_0{Value{"velocity"}, Gradient{"velocity"}, Value{"pressure"}, Gradient{"coordinates"}, Weight{}};
   mfem::tuple output_operator_0{Gradient{"velocity"}};
   ElementOperator op_0{momentum_kernel, argument_operators_0, output_operator_0};

   // (\nabla \cdot u, q)
   auto mass_conservation_kernel = [](const tensor<double, dim, dim> &dudxi,
                                      const tensor<double, dim, dim> &J,
                                      const double &w)
   {
      return mfem::tuple{tr(dudxi * inv(J)) * det(J) * w};
   };

   mfem::tuple argument_operators_1{Gradient{"velocity"}, Gradient{"coordinates"}, Weight{}};
   mfem::tuple output_operator_1{Value{"pressure"}};
   ElementOperator op_1{mass_conservation_kernel, argument_operators_1, output_operator_1};

   std::array solutions{FieldDescriptor{&velocity_fes, "velocity"}, FieldDescriptor{&pressure_fes, "pressure"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator momentum_op{solutions, parameters, mfem::tuple{op_0}, mesh, velocity_ir};
   DifferentiableOperator mass_conservation_op{solutions, parameters, mfem::tuple{op_1}, mesh, pressure_ir};

   // Preconditioner form
   auto pressure_mass_kernel = [](const double &p,
                                  const tensor<double, dim, dim> &J,
                                  const double &w)
   {
      return mfem::tuple{p * det(J) * w};
   };

   mfem::tuple pms_args{Value{"pressure"}, Gradient{"coordinates"}, Weight{}};
   mfem::tuple pms_outs{Value{"pressure"}};
   ElementOperator pressure_mass{pressure_mass_kernel, pms_args, pms_outs};
   std::array pms_sols{FieldDescriptor{&pressure_fes, "pressure"}};
   std::array pms_params{FieldDescriptor{&mesh_fes, "coordinates"}};
   DifferentiableOperator pressure_mass_op{pms_sols, pms_params, mfem::tuple{pressure_mass}, mesh, pressure_ir};

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = velocity_fes.GetTrueVSize();
   block_offsets[2] = pressure_fes.GetTrueVSize();
   block_offsets.PartialSum();

   NavierStokesOperator navierstokes(momentum_op, mass_conservation_op,
                                     block_offsets,
                                     vel_ess_tdofs);

   BlockVector x(block_offsets), y(block_offsets);
   u.ParallelProject(x.GetBlock(0));
   // p.ParallelProject(x.GetBlock(1));
   navierstokes.SetParameters(*mesh_nodes);

   HypreParMatrix A;
   momentum_op.template GetDerivativeWrt<0>({&u, &p}, {mesh_nodes})->Assemble(A);
   A.EliminateBC(vel_ess_tdofs, Operator::DiagonalPolicy::DIAG_ONE);
   HypreBoomerAMG amg(A);
   amg.SetMaxLevels(50);
   amg.SetPrintLevel(0);

   HypreParMatrix Mp;
   pressure_mass_op.template GetDerivativeWrt<0>({&p}, {mesh_nodes})->Assemble(Mp);

   HypreDiagScale Mp_inv(Mp);

   BlockDiagonalPreconditioner prec(block_offsets);
   prec.SetDiagonalBlock(0, &amg);
   prec.SetDiagonalBlock(1, &Mp_inv);

   GMRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-8);
   solver.SetKDim(100);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(2);
   solver.SetPreconditioner(prec);

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(navierstokes);
   newton.SetSolver(solver);
   newton.SetRelTol(1e-8);
   newton.SetMaxIter(500);
   newton.SetPrintLevel(1);

   Vector zero;
   newton.Mult(zero, x);

   u.SetFromTrueDofs(x.GetBlock(0));

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u << std::flush;

   return 0;
}
