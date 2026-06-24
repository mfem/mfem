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
//                --------------------------------------------
//                Hyperelasticity with dFEM Stress Residuals
//                ------------------------------------------
//
// Compile with: make dfem-hyperelasticity-stress
//
// Sample runs:  mpirun -np 4 dfem-hyperelasticity-stress -o 1 -rs 0 -no-vis
//
// Description:  This miniapp solves a quasistatic solid mechanics problem on
//               the 3D beam used by the Hooke miniapp. The material response is
//               specified through a stress-based formulation, by providing the
//               first Piola-Kirchhoff stress P(F).
//               The jacobian is obtained with DifferentiableOperator::GetDerivative
//               and provided to Newton.

#include "mfem.hpp"
#include "../../fem/dfem/doperator.hpp"
#include "../../fem/dfem/backends/local_qf/prelude.hpp"

// Utils for output folder handling
#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#elif __cplusplus >= 201402L
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "C++14 or later is required for filesystem support."
#endif

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

#ifdef MFEM_USE_ENZYME

constexpr int dim = 3;
constexpr int Displacement = 0;
constexpr int Coords = 1;

enum class MaterialType
{
   NeoHookean
};

enum class PreconditionerType
{
   None,
   Diagonal
};

MaterialType ParseMaterial(const char *material)
{
   const std::string name(material);
   if (name == "neo-hookean" || name == "neohookean")
   {
      return MaterialType::NeoHookean;
   }
   MFEM_ABORT("Unknown material '" << name
              << "'. Available materials: neo-hookean");
   return MaterialType::NeoHookean;
}

template <typename Material, typename dscalar_t>
struct HyperelasticStressQFunction
{
   MFEM_HOST_DEVICE inline
   void operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   tensor<dscalar_t, dim, dim> &dvdxi) const
   {
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;
      const auto P = material().PK1(F, dudx);
      dvdxi = P * transpose(invJ) * det(J) * w;
   }

private:
   MFEM_HOST_DEVICE inline
   const Material& material() const
   {
      return static_cast<const Material&>(*this);
   }
};

template <typename dscalar_t>
struct NeoHookeanStress :
   HyperelasticStressQFunction<NeoHookeanStress<dscalar_t>, dscalar_t>
{
   real_t lambda = 100.0;
   real_t mu = 50.0;

   // P(F) = ∂Ψ/∂F = μ (F - F^{-T}) + λ log(J) F^{-T}
   MFEM_HOST_DEVICE inline
   auto PK1(const tensor<dscalar_t, dim, dim> &F,
            const tensor<dscalar_t, dim, dim> & /* dudx */) const
   {
      const auto J = det(F);
      const auto logJ = log(J);
      return mu * F + (lambda * logJ - mu) * inv(transpose(F));
   }
};

class HyperelasticOperator : public Operator
{
   // Matrix-free Jacobian-vector product used by Newton's method. This wraps the
   // nonlinear residual operator and requests the first variation with respect to the displacement field (Jacobian = dR/dU).
   class JacobianOperator : public Operator
   {
   public:
      JacobianOperator(const HyperelasticOperator &oper, const Vector &state) :
         Operator(oper.Height()),
         oper(oper),
         state(state),
         z(oper.Height())
      {
         MultiVector X{state, oper.mesh_nodes_tdofs};
         jacobian = oper.residual_dop->GetDerivative(Displacement, X);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         // Essential directions are removed before applying the Jacobian, then
         // restored as identity rows so constrained dofs stay fixed in Newton.
         z = x;
         z.SetSubVector(oper.ess_tdofs, 0.0);

         MultiVector Y{y};
         jacobian->Mult(z, Y);

         auto d_y = y.ReadWrite();
         const auto d_x = x.Read();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_y[d_dofs[i]] = d_x[d_dofs[i]];
         });
      }

      void AssembleDiagonal(Vector &diag) const override
      {
         jacobian->AssembleDiagonal(diag);

         auto d_diag = diag.ReadWrite();
         const auto d_dofs = oper.ess_tdofs.Read();
         mfem::forall(oper.ess_tdofs.Size(), [=] MFEM_HOST_DEVICE (int i)
         {
            d_diag[d_dofs[i]] = 1.0;
         });
      }

   private:
      const HyperelasticOperator &oper;
      Vector state;
      mutable Vector z;
      std::shared_ptr<DerivativeOperator> jacobian;
   };

public:
   // The nonlinear mechanics operator is defined from the first
   // Piola-Kirchhoff stress P(F), which defines the nonlinear residual R(u).
   // DifferentiableOperator::GetDerivative provides the first variation of the
   // residual with respect to the displacement field, i.e. the Jacobian dR/dU.
   HyperelasticOperator(ParFiniteElementSpace &fes,
                        const IntegrationRule &ir,
                        MaterialType material) :
      Operator(fes.GetTrueVSize()),
      fes(fes),
      ir(ir),
      material(material)
   {
      auto &mesh_nodes =
         *static_cast<ParGridFunction *>(fes.GetParMesh()->GetNodes());
      mesh_nodes_fes = mesh_nodes.ParFESpace();
      mesh_nodes.GetTrueDofs(mesh_nodes_tdofs);

      const std::vector<FieldDescriptor> inputs =
      {
         {Displacement, &fes},
         {Coords, mesh_nodes_fes}
      };
      const std::vector<FieldDescriptor> outputs =
      {
         {Displacement, &fes}
      };

      residual_dop = std::make_shared<DifferentiableOperator>(
                        inputs, outputs, *fes.GetParMesh());

      Array<int> all_domain_attr;
      if (fes.GetMesh()->attributes.Size() > 0)
      {
         all_domain_attr.SetSize(fes.GetMesh()->attributes.Max());
         all_domain_attr = 1;
      }

      auto derivatives = std::integer_sequence<size_t, Displacement> {};
      switch (material)
      {
         case MaterialType::NeoHookean:
         {
            // Default finite-strain material. This is registered as a standard dFEM integrator, not a functional.
            // The first variation will represent the Jacobian-vector product for the nonlinear residual.
            NeoHookeanStress<dscalar_t> stress;
            residual_dop->AddDomainIntegrator<LocalQFBackend>(
               stress,
               Inputs<Gradient<Displacement>, Gradient<Coords>, Weight> {},
               Outputs<Gradient<Displacement>> {},
               ir, all_domain_attr, derivatives);
            break;
         }
      }
   }

   void SetEssentialAttributes(const Array<int> &ess_bdr)
   {
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }

   void SetPrescribedDisplacementAttributes(const Array<int> &disp_bdr)
   {
      fes.GetEssentialTrueDofs(disp_bdr, prescribed_tdofs);
   }

   const Array<int>& GetPrescribedDisplacementTDofs() const
   {
      return prescribed_tdofs;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Stress formulation residual R(u), assembled directly from P(F).
      MultiVector X{x, mesh_nodes_tdofs};
      MultiVector Y{y};
      residual_dop->Mult(X, Y);
      y.SetSubVector(ess_tdofs, 0.0);
   }

   Operator& GetGradient(const Vector &x) const override
   {
      // Newton asks for the gradient of the nonlinear residual: dR/dU.
      jacobian = std::make_shared<JacobianOperator>(*this, x);
      return *jacobian;
   }

private:
   ParFiniteElementSpace &fes;
   ParFiniteElementSpace *mesh_nodes_fes = nullptr;
   const IntegrationRule &ir;
   MaterialType material;
   Vector mesh_nodes_tdofs;
   Array<int> ess_tdofs;
   Array<int> prescribed_tdofs;

   // Variational notation used by this mechanics operator:
   //   residual_dop: R(u) = P(F(u)) : dF/du
   //   jacobian   first variation of the residual: dR/du = dP/dF : dF/du
   std::shared_ptr<DifferentiableOperator> residual_dop;
   mutable std::shared_ptr<JacobianOperator> jacobian;
};


#endif // MFEM_USE_ENZYME

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   const int num_procs = Mpi::WorldSize();
   const int myid = Mpi::WorldRank();
   Hypre::Init();

#ifndef MFEM_USE_ENZYME
   if (Mpi::Root())
   {
      mfem::out << "This miniapp requires MFEM_USE_ENZYME=YES because it uses "
                << "dFEM automatic differentiation.\n";
   }
   return 0;
#else
   int order = 1;
   const char *device_config = "cpu";
   int serial_refinement_levels = 0;
   const char *material_name = "neo-hookean";
   int prec_type = static_cast<int>(PreconditionerType::None);
   bool visualization = true;
   bool paraview = false;
   int visport = 19916;
   const char *outfolder = "./Output";

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&serial_refinement_levels, "-rs", "--ref-serial",
                  "Number of uniform refinements on the serial mesh.");
   args.AddOption(&material_name, "-mat", "--material",
                  "Material: neo-hookean.");
   args.AddOption(&prec_type, "-pc", "--preconditioner",
                  "Preconditioner: 0=none, 1=diagonal.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview",
                  "Enable or disable ParaView DataCollection output.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.AddOption(&outfolder, "-of", "--output-folder",
                  "Output folder for ParaView DataCollection files.");
   args.ParseCheck();

   const MaterialType material = ParseMaterial(material_name);

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh =
      Mesh::MakeCartesian3D(8, 2, 2, Element::HEXAHEDRON, 8.0, 1.0, 1.0);
   if (mesh.Dimension() != dim)
   {
      MFEM_ABORT("This example only works in 3D.");
   }
   mesh.EnsureNodes();

   for (int l = 0; l < serial_refinement_levels; l++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.EnsureNodes();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec, dim, Ordering::byNODES);

   const IntegrationRule &ir =
      IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * order + 1);

   HyperelasticOperator elasticity_op(fes, ir, material);

   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_attr(pmesh.bdr_attributes.Max());
      ess_attr = 0;
      ess_attr[4] = 1;
      ess_attr[2] = 1;
      elasticity_op.SetEssentialAttributes(ess_attr);

      Array<int> displaced_attr(pmesh.bdr_attributes.Max());
      displaced_attr = 0;
      displaced_attr[2] = 1;
      elasticity_op.SetPrescribedDisplacementAttributes(displaced_attr);
   }

   ParGridFunction U_gf(&fes);
   U_gf = 0.0;

   Vector U;
   U_gf.GetTrueDofs(U);
   U.SetSubVector(elasticity_op.GetPrescribedDisplacementTDofs(), 1.0e-2);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-4);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(3);

   std::unique_ptr<Solver> pc;
   switch (static_cast<PreconditionerType>(prec_type))
   {
      case PreconditionerType::None:
         break;
      case PreconditionerType::Diagonal:
         pc = std::make_unique<OperatorJacobiSmoother>();
         cg.SetPreconditioner(*pc);
         break;
      default:
         MFEM_ABORT("Unknown preconditioner type: " << prec_type);
   }

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(cg);
   newton.SetOperator(elasticity_op);
   newton.SetAbsTol(0.0);
#ifdef MFEM_USE_SINGLE
   newton.SetRelTol(1e-4);
#else
   newton.SetRelTol(1e-6);
#endif
   newton.SetMaxIter(10);
   newton.SetPrintLevel(1);

   Vector zero;
   newton.Mult(zero, U);

   U_gf.Distribute(U);

   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << U_gf << std::flush;
   }

   if (paraview)
   {
      if (Mpi::Root())
      {
         fs::create_directories(outfolder);
      }

      // Create a ParaView data collection
      ParaViewDataCollection pd("dfem-hyperelasticity", &pmesh);
      pd.SetPrefixPath(outfolder);
      pd.RegisterField("displacement", &U_gf);
      pd.SetDataFormat(VTKFormat::BINARY);
      if (order > 1)
      {
         pd.SetHighOrderOutput(true);
         pd.SetLevelsOfDetail(order);
      }
      pd.SetCycle(0);
      pd.SetTime(0.0);
      pd.Save();
   }

   return 0;
#endif // MFEM_USE_ENZYME
}
