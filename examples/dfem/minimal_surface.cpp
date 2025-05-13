//                       MFEM Example - Minimal Surface
//
// Compile with: make minimal_surface
//
// Sample runs:  mpirun -np 4 minimal_surface -der 0
//               mpirun -np 4 minimal_surface -der 0 -o 2
//               mpirun -np 4 minimal_surface -der 0 -r 2
//               mpirun -np 4 minimal_surface -der 1
//               mpirun -np 4 minimal_surface -der 2
//
// Description:  This example code demonstrates the use of MFEM to solve the minimal
//              surface problem in 2D:
//
//              min \int sqrt(1 + |\nabla u|^2) dx
//
//              with Dirichlet boundary conditions. The nonlinear problem is solved
//              using Newton's method, where the necessary derivatives are computed
//              in one of three ways (controlled by -der command line parameter):
//              0) Automatic differentiation using Enzyme or dual type (default)
//              1) Hand-coded derivatives
//              2) Finite differences
//
//              The example demonstrates the use of MFEM's nonlinear solvers,
//              automatic differentiation capabilities, and GLVis/ParaView visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

enum DerivativeType
{
   AUTODIFF,
   HANDCODED,
   FD
};

template <typename dscalar_t, int dim = 2>
class MinimalSurface : public Operator
{
private:
   static constexpr int SOLUTION_U = 1;
   static constexpr int MESH_NODES = 2;

   template <typename T>
   MFEM_HOST_DEVICE inline
   static auto coeff(const tensor<T, dim> &a)
   {
      return real_t(1.0) / sqrt(real_t(1.0) + sqnorm(a));
   }

   struct MFApply
   {
      MFEM_HOST_DEVICE inline
      auto operator()(
         const tensor<dscalar_t, dim> &dudxi,
         const tensor<real_t, dim, dim> &J,
         const real_t &w) const
      {
         const auto invJ = inv(J);
         const auto dudx = dudxi * invJ;
         return tuple{coeff(dudx) * dudx * transpose(invJ) * det(J) * w};
      }
   };

   struct ManualDerivativeApply
   {
      MFEM_HOST_DEVICE inline
      auto operator()(
         const tensor<real_t, dim> &ddelta_udxi,
         const tensor<real_t, dim> &dudxi,
         const tensor<real_t, dim, dim> &J,
         const real_t &w) const
      {
         const auto invJ = inv(J);
         const auto dudx = dudxi * invJ;
         const auto ddelta_udx = ddelta_udxi * invJ;

         const auto c = coeff(dudx);
         const auto term1 = c * ddelta_udx;
         const auto term2 = c * c * c * dot(dudx, ddelta_udx) * dudx;

         return tuple{(term1 - term2) * transpose(invJ) * det(J) * w};
      }
   };

   class MinimalSurfaceJacobian : public Operator
   {
   public:
      MinimalSurfaceJacobian(const MinimalSurface *minsurface,
                             const Vector &x) :
         Operator(minsurface->Height()),
         minsurface(minsurface),
         z(minsurface->Height())
      {
         minsurface->u.SetFromTrueDofs(x);
         auto mesh_nodes = static_cast<ParGridFunction*>
                           (minsurface->H1.GetParMesh()->GetNodes());
         dres_du = minsurface->res->GetDerivative(SOLUTION_U, {&minsurface->u}, {mesh_nodes});
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(minsurface->ess_tdofs, 0.0);

         dres_du->Mult(z, y);

         for (int i = 0; i < minsurface->ess_tdofs.Size(); i++)
         {
            y[minsurface->ess_tdofs[i]] = x[minsurface->ess_tdofs[i]];
         }
      }

      const MinimalSurface *minsurface = nullptr;
      std::shared_ptr<DerivativeOperator> dres_du;
      mutable Vector z;
   };

   class MinimalSurfaceHandcodedJacobian : public Operator
   {
      static constexpr int DIRECTION_U = 3;

   public:
      MinimalSurfaceHandcodedJacobian(const MinimalSurface *minsurface,
                                      const Vector &x) :
         Operator(minsurface->Height()),
         minsurface(minsurface),
         z(minsurface->Height())
      {
         Array<int> all_domain_attr;
         all_domain_attr.SetSize(minsurface->H1.GetMesh()->attributes.Max());
         all_domain_attr = 1;

         auto &mesh_nodes = *static_cast<ParGridFunction *>
                            (minsurface->H1.GetParMesh()->GetNodes());
         auto &mesh_nodes_fes = *mesh_nodes.ParFESpace();

         std::vector<FieldDescriptor> solutions = {{DIRECTION_U, &minsurface->H1}};
         std::vector<FieldDescriptor> parameters =
         {
            {SOLUTION_U, &minsurface->H1},
            {MESH_NODES, &mesh_nodes_fes}
         };

         dres_du = std::make_shared<DifferentiableOperator>(
                      solutions, parameters, *minsurface->H1.GetParMesh());

         auto input_operators = tuple
         {
            Gradient<DIRECTION_U>{},
            Gradient<SOLUTION_U>{},
            Gradient<MESH_NODES>{},
            Weight{}
         };

         auto output_operators = tuple
         {
            Gradient<SOLUTION_U>{}
         };

         ManualDerivativeApply manual_derivative_apply;
         dres_du->AddDomainIntegrator(manual_derivative_apply, input_operators,
                                      output_operators, minsurface->ir,
                                      all_domain_attr);

         minsurface->u.SetFromTrueDofs(x);
         dres_du->SetParameters({&minsurface->u, &mesh_nodes});
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(minsurface->ess_tdofs, 0.0);

         dres_du->Mult(z, y);

         for (int i = 0; i < minsurface->ess_tdofs.Size(); i++)
         {
            y[minsurface->ess_tdofs[i]] = x[minsurface->ess_tdofs[i]];
         }
      }

      const MinimalSurface *minsurface = nullptr;
      std::shared_ptr<DifferentiableOperator> dres_du;
      mutable Vector z;
   };


public:
   MinimalSurface(ParFiniteElementSpace &H1,
                  const IntegrationRule &ir,
                  int deriv_type = AUTODIFF) :
      Operator(H1.GetTrueVSize(), H1.GetTrueVSize()),
      H1(H1),
      ir(ir),
      u(&H1),
      derivative_type(deriv_type)
   {
      Array<int> all_domain_attr;
      all_domain_attr.SetSize(H1.GetMesh()->attributes.Max());
      all_domain_attr = 1;

      auto &mesh_nodes = *static_cast<ParGridFunction *>(H1.GetParMesh()->GetNodes());
      auto &mesh_nodes_fes = *mesh_nodes.ParFESpace();

      {
         std::vector<FieldDescriptor> solutions = {{SOLUTION_U, &H1}};
         std::vector<FieldDescriptor> parameters = {{MESH_NODES, &mesh_nodes_fes}};

         res = std::make_shared<DifferentiableOperator>(
                  solutions, parameters, *H1.GetParMesh());

         auto input_operators = tuple
         {
            Gradient<SOLUTION_U>{},
            Gradient<MESH_NODES>{},
            Weight{}
         };

         auto output_operators = tuple
         {
            Gradient<SOLUTION_U>{}
         };

         MFApply mf_apply_qf;
         auto derivatives = std::integer_sequence<size_t, SOLUTION_U> {};
         res->AddDomainIntegrator(mf_apply_qf, input_operators, output_operators, ir,
                                  all_domain_attr, derivatives);

         res->SetParameters({&mesh_nodes});
      }


      Array<int> ess_bdr(H1.GetParMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      H1.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      res->Mult(x, y);
      y.SetSubVector(ess_tdofs, 0.0);
   }

   Operator& GetGradient(const Vector &x) const override
   {
      switch (derivative_type)
      {
         case FD:
            fd_jac = std::make_shared<FDJacobian>(*this, x);
            return *fd_jac;

         case HANDCODED:
         {
            man_dres_du = std::make_shared<MinimalSurfaceHandcodedJacobian>(this, x);
            return *man_dres_du;
         }

         case AUTODIFF:
         default:
            dres_du = std::make_shared<MinimalSurfaceJacobian>(this, x);
            return *dres_du;
      }
   }

private:
   ParFiniteElementSpace &H1;
   const IntegrationRule &ir;

   mutable ParGridFunction u;

   Array<int> ess_tdofs;

   std::shared_ptr<DifferentiableOperator> res;
   mutable std::shared_ptr<MinimalSurfaceJacobian> dres_du;
   mutable std::shared_ptr<MinimalSurfaceHandcodedJacobian> man_dres_du;
   mutable std::shared_ptr<FDJacobian> fd_jac;
   int derivative_type;
};


real_t boundary_func(const Vector &coords)
{
   const real_t x = coords(0);
   const real_t y = coords(1);
   if (coords.Size() == 3)
   {
      MFEM_ABORT("internal error");
   }
   const real_t a = 1.0e-2;
   return log(cos(a * x) / cos(a * y)) / a;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE
   Mpi::Init();
   Hypre::Init();

   // 2. Parse command-line options
   int order = 1;
   const char *device_config = "cpu";
   bool visualization = true;
   int refinements = 0;
   int derivative_type = AUTODIFF;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&refinements, "-r", "--refinements", "");
   args.AddOption(&derivative_type, "-der", "--derivative-type",
                  "Derivative computation type: 0=AutomaticDifferentiation, 1=HandCoded, 2=FiniteDifference");

   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as CUDA
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 4. Create a 2D mesh on the square domain [-π/2,π/2]^2
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL);
   mesh.SetCurvature(order);

   auto transform_mesh = [](const Vector &cold, Vector &cnew)
   {
      cnew = cold;
      cnew -= 0.5;
      cnew *= M_PI;
   };

   mesh.Transform(transform_mesh);

   // 5. Refine the mesh to increase the resolution
   for (int i = 0; i < refinements; i++)
   {
      mesh.UniformRefinement();
   }

   int dim = mesh.Dimension();

   // 6. Define a parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 7. Define a parallel finite element space on the parallel mesh
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace H1(&pmesh, &fec);

   // 8. Set up the integration rule
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(),
                                  2 * order + 1);

   ParGridFunction u(&H1);
   Vector X(H1.GetTrueVSize());

   // 9. Create the nonlinear operator for the minimal surface equation
   std::unique_ptr<Operator> minsurface;
#ifdef MFEM_USE_ENZYME
   // When Enzyme is available, use it for automatic differentiation
   minsurface = std::make_unique<MinimalSurface<real_t>>(H1, *ir,
                                                         derivative_type);
#else
   // When Enzyme is not available, use the dual type for automatic differentiation
   using mfem::future::dual;
   using dual_t = dual<real_t, real_t>;
   minsurface = std::make_unique<MinimalSurface<dual_t>>(H1, *ir,
                                                         derivative_type);
#endif

   // 10. Set up and apply the boundary conditions
   Array<int> ess_bdr(H1.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;

   // 11. Set up the essential boundary conditions and initial condition
   FunctionCoefficient boundary_coeff(boundary_func);
   u.ProjectCoefficient(boundary_coeff);
   u *= 1e-2;
   u.ProjectBdrCoefficient(boundary_coeff, ess_bdr);

   // 12. Set up the linear solver to be used within Newton's method
   CGSolver krylov(MPI_COMM_WORLD);
   krylov.SetAbsTol(0.0);
   krylov.SetRelTol(1e-4);
   krylov.SetMaxIter(500);
   krylov.SetPrintLevel(2);

   // 13. Set up the nonlinear solver (Newton) for the minimal surface equation
   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetOperator(*minsurface);
   newton.SetAbsTol(0.0);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(10);
   newton.SetSolver(krylov);
   newton.SetPrintLevel(1);

   // 14. Solve the nonlinear system using Newton's method
   H1.GetRestrictionMatrix()->Mult(u, X);
   Vector zero;
   newton.Mult(zero, X);
   H1.GetProlongationMatrix()->Mult(X, u);

   // 15. Save the solution in parallel using ParaView format
   ParaViewDataCollection dc("minimal_surface", &pmesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("solution", &u);
   dc.SetCycle(0);
   dc.Save();

   return 0;
}
