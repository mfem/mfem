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
//                Hyperelasticity with dFEM Second Derivatives
//                --------------------------------------------
//
// Compile with: make dfem-hyperelasticity
//
// Sample runs:  mpirun -np 4 dfem-hyperelasticity -o 1 -rs 0 -no-vis
//               mpirun -np 4 dfem-hyperelasticity -mat linear-elastic -no-vis
//               mpirun -np 4 dfem-hyperelasticity -mat mooney-rivlin -no-vis
//
// Description:  This miniapp solves a quasistatic solid mechanics problem on
//               the 3D beam used by the Hooke miniapp. The material response is
//               specified through a strain-energy density. The nonlinear
//               residual is obtained with DifferentiableOperator::GetDerivative
//               and Newton's Hessian-vector products are obtained with the new
//               DifferentiableOperator::GetSecondDerivative functionality.

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
constexpr int Energy = 2;

enum class MaterialType
{
   NeoHookean,
   LinearElastic,
   MooneyRivlin
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
   if (name == "linear-elastic" || name == "linear")
   {
      return MaterialType::LinearElastic;
   }
   if (name == "mooney-rivlin" || name == "mooney" || name == "rivlin")
   {
      return MaterialType::MooneyRivlin;
   }
   MFEM_ABORT("Unknown material '" << name
              << "'. Available materials: neo-hookean, linear-elastic, "
              << "mooney-rivlin.");
   return MaterialType::NeoHookean;
}

template <typename Material, typename dscalar_t>
struct HyperelasticEnergyQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<dscalar_t, dim, dim> &dudxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w,
                   dscalar_t &energy) const
   {
      // The material supplies only psi, the strain-energy density at a point.
      // The q-function adapter supplies the finite element measure: it maps
      // reference gradients to physical gradients and multiplies psi by
      // det(J) * w, i.e. the quadrature form of dx in Pi_int = int psi dx.
      const auto invJ = inv(J);
      const auto dudx = dudxi * invJ;
      const auto F = IdentityMatrix<dim>() + dudx;
      energy = material().psi(F, dudx) * det(J) * w;
   }

private:
   MFEM_HOST_DEVICE inline
   const Material& material() const
   {
      return static_cast<const Material&>(*this);
   }
};

template <typename dscalar_t>
struct NeoHookeanEnergy :
   HyperelasticEnergyQFunction<NeoHookeanEnergy<dscalar_t>, dscalar_t>
{
   real_t D1 = 100.0;
   real_t C1 = 50.0;

   // Hooke miniapp Neo-Hookean model:
   // Ψ(F) = D1 (J - 1)^2 + C1 (J^(-2/3) I1 - dim),
   //        J = det(F), I1 = tr(FᵀF).
   // This gives σ = 2 D1 (J - 1) I + 2 C1 J^(-5/3) dev(F Fᵀ)
   // consistent with the Hooke miniapp.
   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> &F,
                 const tensor<dscalar_t, dim, dim> & /* dudx */) const
   {
      const auto C = transpose(F) * F;
      const auto J = det(F);
      const auto I1_bar = pow(J, -2.0_r / 3.0_r) * tr(C);
      return D1 * (J - 1.0_r) * (J - 1.0_r)
             + C1 * (I1_bar - real_t(dim));
   }
};

template <typename dscalar_t>
struct LinearElasticEnergy :
   HyperelasticEnergyQFunction<LinearElasticEnergy<dscalar_t>, dscalar_t>
{
   real_t lambda = 100.0;
   real_t mu = 50.0;

   // Ψ(ε) = λ/2 tr(ε)^2 + μ ε:ε,  ε = sym(∇u)
   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> & /* F */,
                 const tensor<dscalar_t, dim, dim> &dudx) const
   {
      const auto strain = sym(dudx);
      const auto tr_strain = tr(strain);
      return 0.5_r * lambda * tr_strain * tr_strain
             + mu * ddot(strain, strain);
   }
};

template <typename dscalar_t>
struct MooneyRivlinEnergy :
   HyperelasticEnergyQFunction<MooneyRivlinEnergy<dscalar_t>, dscalar_t>
{
   real_t c1 = 25.0;
   real_t c2 = 25.0;
   real_t kappa = 100.0;

   // Ψ(F) = c₁(Ī₁ - dim) + c₂(Ī₂ - dim) + κ/2 log(J)^2,
   //        J = det F, Ī₁ = J^(-2/3) I₁, Ī₂ = J^(-4/3) I₂
   MFEM_HOST_DEVICE inline
   dscalar_t psi(const tensor<dscalar_t, dim, dim> &F,
                 const tensor<dscalar_t, dim, dim> & /* dudx */) const
   {
      const auto C = transpose(F) * F;
      const auto J = det(F);
      const auto Jm23 = pow(J, -2.0_r / 3.0_r);
      const auto I1 = tr(C);
      const auto I2 = 0.5_r * (I1 * I1 - tr(C * C));
      const auto I1_bar = Jm23 * I1;
      const auto I2_bar = Jm23 * Jm23 * I2;
      const auto log_J = log(J);
      return c1 * (I1_bar - real_t(dim))
             + c2 * (I2_bar - real_t(dim))
             + 0.5_r * kappa * log_J * log_J;
   }
};

class HyperelasticOperator : public Operator
{
   // Matrix-free Hessian-vector product used by Newton's method. This wraps the
   // functional second-derivative interface and applies the same essential-dof
   // treatment as the Hooke elasticity Jacobian operator.
   // The wrapped DerivativeOperator returned by GetSecondDerivative computes
   // the unconstrained second variation. This class adapts it to the Newton
   // solve by zeroing constrained directions and restoring identity rows on
   // essential true dofs.
   class HessianOperator : public Operator
   {
   public:
      HessianOperator(const HyperelasticOperator &oper, const Vector &state) :
         Operator(oper.Height()),
         oper(oper),
         state(state),
         z(oper.Height())
      {
         MultiVector X{state, oper.mesh_nodes_tdofs};
         hessian = oper.internal_energy_dop->GetSecondDerivative(Displacement, X);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         // Essential directions are removed before applying the Hessian, then
         // restored as identity rows so constrained dofs stay fixed in Newton.
         z = x;
         z.SetSubVector(oper.ess_tdofs, 0.0);

         MultiVector Y{y};
         hessian->Mult(z, Y);

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
         hessian->AssembleDiagonal(diag);

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
      std::shared_ptr<DerivativeOperator> hessian;
   };

public:
   // The nonlinear mechanics operator is defined from a scalar strain-energy
   // density. Registering the q-function as a dFEM functional makes the first
   // derivative available as the residual and the second derivative available
   // as a matrix-free Hessian-vector product.
   HyperelasticOperator(ParFiniteElementSpace &fes,
                        const IntegrationRule &ir,
                        MaterialType material) :
      Operator(fes.GetTrueVSize()),
      fes(fes),
      ir(ir),
      qspace(*fes.GetParMesh(), ir),
      qspace_vec(qspace, 1),
      q(qspace_vec),
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
         {Energy, &qspace_vec}
      };

      internal_energy_dop = std::make_shared<DifferentiableOperator>(
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
            // Default finite-strain material. The functional registration tells
            // dFEM to build the first and second variations from this energy.
            // NOTE: the registration for functional will likely change in the future.
            NeoHookeanEnergy<dscalar_t> energy;
            internal_energy_dop->AddDomainIntegrator<LocalQFBackend, true>(
               energy,
               Inputs<Gradient<Displacement>, Gradient<Coords>, Weight> {},
               Outputs<Identity<Energy>> {},
               ir, all_domain_attr, derivatives);
            break;
         }
         case MaterialType::LinearElastic:
         {
            // Linear elastic material, also expressed as an energy so it uses
            // exactly the same GetDerivative/GetSecondDerivative machinery.
            LinearElasticEnergy<dscalar_t> energy;
            internal_energy_dop->AddDomainIntegrator<LocalQFBackend, true>(
               energy,
               Inputs<Gradient<Displacement>, Gradient<Coords>, Weight> {},
               Outputs<Identity<Energy>> {},
               ir, all_domain_attr, derivatives);
            break;
         }
         case MaterialType::MooneyRivlin:
         {
            // Compressible Mooney-Rivlin material with an isochoric invariant
            // split and logarithmic volumetric penalty.
            MooneyRivlinEnergy<dscalar_t> energy;
            internal_energy_dop->AddDomainIntegrator<LocalQFBackend, true>(
               energy,
               Inputs<Gradient<Displacement>, Gradient<Coords>, Weight> {},
               Outputs<Identity<Energy>> {},
               ir, all_domain_attr, derivatives);
            break;
         }
      }

      // The first variation of a functional is exposed as a stateless
      // derivative operator: it does not capture a linearization state here.
      // Instead, the current displacement is passed to gradient->Mult(X, Y),
      // so this wrapper can be cached and reused for every residual evaluation.
      gradient = internal_energy_dop->GetDerivative(Displacement);
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
      // Residual R(u) = dE/du. For functional integrators, GetDerivative
      // returns the gradient action directly, including the pointwise reverse
      // seed for the summed energy.
      MultiVector X{x, mesh_nodes_tdofs};
      MultiVector Y{y};
      gradient->Mult(X, Y);
      y.SetSubVector(ess_tdofs, 0.0);
   }

   Operator& GetGradient(const Vector &x) const override
   {
      // Newton asks for the gradient of the nonlinear residual. Since the
      // residual is the energy gradient, this is the Hessian of the energy.
      hessian = std::make_shared<HessianOperator>(*this, x);
      return *hessian;
   }

private:
   ParFiniteElementSpace &fes;
   ParFiniteElementSpace *mesh_nodes_fes = nullptr;
   const IntegrationRule &ir;
   QuadratureSpace qspace;
   VectorQuadratureSpace qspace_vec;
   QuadratureFunction q;
   MaterialType material;
   Vector mesh_nodes_tdofs;
   Array<int> ess_tdofs;
   Array<int> prescribed_tdofs;

   // Variational notation used by this mechanics operator:
   //   internal_energy_dop: Pi_int(u) = int_Omega psi(u) dx, where psi is the strain-energy density
   //   gradient:            first variation, d Pi_int / du
   //   hessian:             second variation at the current Newton state
   std::shared_ptr<DifferentiableOperator> internal_energy_dop;
   std::shared_ptr<DerivativeOperator> gradient;
   mutable std::shared_ptr<HessianOperator> hessian;
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
                << "dFEM functional second derivatives.\n";
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
                  "Material: neo-hookean, linear-elastic, or mooney-rivlin.");
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

   // Problem size
   const int size = fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      mfem::out << "#dofs: " << size << std::endl;
   }


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
   cg.SetRelTol(1e-1);
   cg.SetMaxIter(10000);
   cg.SetPrintLevel(2);

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
#ifdef MFEM_USE_SINGLE
   newton.SetRelTol(1e-4);
#elif defined MFEM_USE_DOUBLE
   newton.SetRelTol(1e-6);
#else
   MFEM_ABORT("Floating point type undefined");
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
