// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                   -------------------------------------
//                   Minimal Surface 2D Problem with dFEM
//                   -------------------------------------
//
// Compile with: make dfem-minimal-surface
//
// Sample runs:  mpirun -np 4 dfem-minimal-surface -der 0
//               mpirun -np 4 dfem-minimal-surface -der 0 -o 2
//               mpirun -np 4 dfem-minimal-surface -der 0 -r 1
//               mpirun -np 4 dfem-minimal-surface -der 1
//               mpirun -np 4 dfem-minimal-surface -der 2
//
// Device sample runs:
//               mpirun -np 4 dfem-minimal-surface -der 0 -r 1 -o 2 -d cuda
//               mpirun -np 4 dfem-minimal-surface -der 1 -r 1 -o 2 -d cuda
//             * mpirun -np 4 dfem-minimal-surface -der 0 -r 1 -o 2 -d hip
//             * mpirun -np 4 dfem-minimal-surface -der 1 -r 1 -o 2 -d hip
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               minimal surface problem in 2D:
//
//               $ \min \left( -\nabla \cdot (1 / \sqrt(1 + |\nabla u|^2) \nabla u) \right) $
//
//               with Dirichlet boundary conditions. The nonlinear problem is
//               solved using Newton's method, where the necessary derivatives
//               are computed in one of three ways (controlled by -der command
//               line parameter):
//
//               -der 0 = Automatic differentiation using Enzyme or dual type
//                        (default)
//               -der 1 = Hand-coded derivatives
//               -der 2 = Finite differences
//
//               The example demonstrates the use of MFEM's nonlinear solvers,
//               automatic differentiation capabilities, and GLVis/ParaView
//               visualization.

#include "mfem.hpp"

using namespace mfem;

// This example code demonstrates the use of new features in MFEM that are in
// development but exposed through the mfem::future namespace. All features
// under this namespace might change their interface or behavior in upcoming
// releases until they have stabilized.
using namespace mfem::future;
using mfem::future::tensor;

// Derivative type enum
// This enum is used to specify the type of derivative computation.
// Possibilities are:
// - AUTODIFF, which uses automatic differentiation (Enzyme or dual type),
// - HANDCODED, which uses a manually implemented derivative, and
// - FD, finite difference.
enum DerivativeType
{
   AUTODIFF,
   HANDCODED,
   FD
};

// Minimal surface operator.
//
// This class implements the minimal surface equation, which is a nonlinear
// operator that provides the residual.
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
      return 1_r / sqrt(1_r + sqnorm(a));
   }

   // The 'nvcc' compiler needs MFApply and ManualDerivativeApply to be public.
public:
   // Matrix-Free version of the pointwise residual form for the minimal
   // surface equation.
   struct MFApply
   {
      // Using DifferentiableOperator, we can define the residual form as a
      // matrix-free operation. This allows us to compute the residual without
      // explicitly forming any matrices or other large, temporary data
      // structures.
      //
      // The inputs are the gradient of the solution in *reference coordinates*,
      // the Jacobian of the coordinates, and the integration rule weights.
      //
      // The output is the residual in *physical coordinates* which also
      // includes the necessary transformation from reference to physical
      // coordinates for the gradient of the test function.
      //
      // Due to the description of how this pointwise operation is used in
      // DifferentiableOperator, we know it is applied to the gradient of the
      // test function in reference coordinates e.g.
      // $ \int coeff(\nabla_x u) (\nabla_x u) J^{-T} \det(J) w
      //        (\nabla_{\xi} v) d\xi $
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

   // This is the derivative of the residual form with respect to the
   // solution $u$.
   //
   // The inputs and outputs follow the same rules as the MFApply operator.
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

private:
   // This class implements the Jacobian of the minimal surface operator. It
   // mostly acts as a wrapper to retrieve the Jacobian and apply essential
   // boundary conditions appropriately.
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

         // One can retrieve the derivative of a DifferentiableOperator wrt a
         // field variable if the derivative has been requested during the
         // DifferentiableOperator::AddDomainIntegrator call.
         dres_du = minsurface->res->GetDerivative(
                      SOLUTION_U, {&minsurface->u}, {mesh_nodes});
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(minsurface->ess_tdofs, 0.0);

         dres_du->Mult(z, y);

         auto d_y = y.HostReadWrite();
         const auto d_x = x.HostRead();
         for (int i = 0; i < minsurface->ess_tdofs.Size(); i++)
         {
            d_y[minsurface->ess_tdofs[i]] = d_x[minsurface->ess_tdofs[i]];
         }
      }

      // Pointer to the wrapped MinimalSurface operator
      const MinimalSurface *minsurface = nullptr;

      // Pointer to the DifferentiableOperator that computes the Jacobian
      std::shared_ptr<DerivativeOperator> dres_du;

      // Temporary vector
      mutable Vector z;
   };

   // This class implements the Jacobian of the minimal surface operator using
   // manually computed derivatives.
   class MinimalSurfaceHandcodedJacobian : public Operator
   {
      // For the Jacobian action we need another field ID for the direction
      // of u, called du, in dR/du = J * du.
      static constexpr int DIRECTION_U = 3;

   public:
      MinimalSurfaceHandcodedJacobian(const MinimalSurface *minsurface,
                                      const Vector &x) :
         Operator(minsurface->Height()),
         minsurface(minsurface),
         z(minsurface->Height())
      {
         Array<int> all_domain_attr(minsurface->H1.GetMesh()->attributes.Max());
         all_domain_attr = 1;

         auto &mesh_nodes = *static_cast<ParGridFunction *>
                            (minsurface->H1.GetParMesh()->GetNodes());
         auto &mesh_nodes_fes = *mesh_nodes.ParFESpace();

         std::vector<FieldDescriptor> solutions =
         {
            {DIRECTION_U, &minsurface->H1}
         };
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

         auto d_y = y.HostReadWrite();
         const auto d_x = x.HostRead();
         for (int i = 0; i < minsurface->ess_tdofs.Size(); i++)
         {
            d_y[minsurface->ess_tdofs[i]] = d_x[minsurface->ess_tdofs[i]];
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
      Array<int> all_domain_attr(H1.GetMesh()->attributes.Max());
      all_domain_attr = 1;

      auto &mesh_nodes =
         *static_cast<ParGridFunction *>(H1.GetParMesh()->GetNodes());
      auto &mesh_nodes_fes = *mesh_nodes.ParFESpace();

      // The following section is the heart of this example. It shows how to
      // create and interact with the DifferentialOperator class.

      // The constructor of DifferentiableOperator takes two vectors of
      // FieldDescriptors. A FieldDescriptor can be viewed as a a pair of an
      // identifier (the field ID) and it's accompanying space.
      std::vector<FieldDescriptor> solutions;
      solutions.push_back(FieldDescriptor(SOLUTION_U, &H1));
      std::vector<FieldDescriptor> parameters;
      parameters.push_back(FieldDescriptor(MESH_NODES, &mesh_nodes_fes));

      // Create the DifferentiableOperator on the desired mesh.
      res = std::make_shared<DifferentiableOperator>(
               solutions, parameters, *H1.GetParMesh());

      // DifferentiableOperator::AddIntegrator consists mainly of multiple
      // components. The input and output operators and the pointwise
      // "quadrature function" form a description of how the inputs and outputs
      // to the pointwise function have to be treated.

      // The input operators tuple consists of derived FieldOperator types.
      // Here, we use Gradient<FIELD_ID> to signal that we request the gradient
      // on the reference coordinates of the FIELD_ID field to be interpolated
      // and translated to the pointwise function as the first and second input.
      // Other choices are possible, e.g. Value<FIELD_ID> to interpolate the
      // pointwise funciton. `Weight` is a special field that translates the
      // integration rule weights to the input of the pointwise function.
      auto input_operators = tuple
      {
         Gradient<SOLUTION_U>{},
         Gradient<MESH_NODES>{},
         Weight{}
      };

      // The output operators tuple also consists of derived FieldOperator
      // types. Currently, only _one_ output operator is allowed. One should
      // think of this as an operator on the output of a pointwise function. For
      // example with the above input operators and the output operator below we
      // create the following operator sequence:
      //
      // $ B^T D(B u, B x, w) $
      //
      // where B is the gradient interpolation operator, D is the pointwise
      // function and u and x are solution and coordinate functions,
      // respectively. The output operator is the gradient of the basis of the
      // solution, which completes the "diffusion" like weak form.
      auto output_operators = tuple
      {
         Gradient<SOLUTION_U>{}
      };

      // The pointwise function is defined as a lambda function. Here we just
      // instantiate an object for it which is passed to
      // DifferentiableOperator::AddDomainIntegrator.
      MFApply mf_apply_qf;

      // The integeger sequence is used to specify which derivatives of the
      // formed integrator should be formed. This is necessary to specify at
      // compile time in order to instantiate the correct functions.
      auto derivatives = std::integer_sequence<size_t, SOLUTION_U> {};
      res->AddDomainIntegrator(mf_apply_qf, input_operators, output_operators,
                               ir, all_domain_attr, derivatives);

      // Before we are able to use DifferentiableOperator::Mult, we need to call
      // DifferentiableOperator::SetParameters to set the parameters of the
      // operator. Here, only the mesh node function is required. We do this
      // here once, because we know that the nodes won't change. If they do,
      // we'd have to call SetParameters before each call to Mult. This is done
      // to be mathematically consistent with fixing paramaters.
      res->SetParameters({&mesh_nodes});

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
            man_dres_du = std::make_shared<MinimalSurfaceHandcodedJacobian>(
                             this, x);
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

// Boundary function for the minimal surface problem described by the Scherk
// surface.
// See https://en.wikipedia.org/wiki/Scherk_surface for more details.
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
                  "Derivative computation type: 0=AutomaticDifferentiation,"
                  " 1=HandCoded, 2=FiniteDifference");

   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 4. Create a 2D mesh on the square domain [-π/2,π/2]^2
   Mesh mesh = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL);
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
   // When Enzyme is not available, use the dual type for automatic
   // differentiation
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

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel "
               << Mpi::WorldSize() << " " <<  Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u << std::flush;
   }

   // 16. Save the solution in parallel using ParaView format
   ParaViewDataCollection dc("dfem-minimal-surface-output", &pmesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("solution", &u);
   dc.SetCycle(0);
   dc.Save();

   return 0;
}
