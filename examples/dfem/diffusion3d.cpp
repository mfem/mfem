#include <mfem.hpp>

#include "fem/qinterp/det.cpp"
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep

#include <fem/dfem/doperator.hpp>
#include <general/forall.hpp>
#include <linalg/kernels.hpp>
#include <linalg/tensor.hpp>
#include <fem/kernel_dispatch.hpp>

#include "kernels_pa.hpp"

using namespace mfem;
using mfem::internal::tensor;

#undef NVTX_COLOR
#define NVTX_COLOR nvtx::kAquamarine
#include "general/nvtx.hpp"


///////////////////////////////////////////////////////////////////////////////
struct StiffnessIntegrator : public BilinearFormIntegrator
{
   const FiniteElementSpace *fes;
   const real_t *B, *G, *DX;
   int ne, d1d, q1d;
   Vector J0, dx;

public:
   StiffnessIntegrator()
   {
      dbg();
      StiffnessKernels::Specialization<2, 3>::Add();
      StiffnessKernels::Specialization<3, 5>::Add();
      StiffnessKernels::Specialization<4, 8>::Add();
   }

   void AssemblePA(const FiniteElementSpace &fespace) override
   {
      fes = &fespace;
      auto *mesh = fes->GetMesh();
      const int DIM = mesh->Dimension();
      ne = mesh->GetNE();
      const auto p = fes->GetFE(0)->GetOrder();
      const auto q = 2 * p + mesh->GetElementTransformation(0)->OrderW();
      const auto type = mesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);
      const int NQPT = ir.GetNPoints();
      d1d = fes->GetFE(0)->GetOrder() + 1;
      q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      MFEM_VERIFY(NQPT == q1d * q1d * q1d, "");
      const DofToQuad *maps = &fes->GetFE(0)->GetDofToQuad(ir, DofToQuad::TENSOR);
      const GridFunction *nodes = (mesh->EnsureNodes(), mesh->GetNodes());
      const FiniteElementSpace *nfes = nodes->FESpace();
      const int nVDIM = nfes->GetVDim();
      dx.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      J0.SetSize(nVDIM * DIM * NQPT * ne, Device::GetDeviceMemoryType());
      dx.UseDevice(true), J0.UseDevice(true);
      B = maps->B.Read(), G = maps->G.Read(), DX = dx.Read();

      const Operator *NR = nfes->GetElementRestriction(
                              ElementDofOrdering::LEXICOGRAPHIC);
      const QuadratureInterpolator *nqi = nfes->GetQuadratureInterpolator(ir);
      nqi->SetOutputLayout(QVectorLayout::byVDIM);
      const int nd = nfes->GetFE(0)->GetDof();
      Vector xe(nVDIM * nd * ne, Device::GetDeviceMemoryType());
      NR->Mult(*nodes, (xe.UseDevice(true), xe));
      nqi->Derivatives(xe, J0);

      const auto w_r = ir.GetWeights().Read();
      const auto W = Reshape(w_r, q1d, q1d, q1d);
      const auto J = Reshape(J0.Read(), 3, 3, q1d, q1d, q1d, ne);
      auto DX_w = Reshape(dx.Write(), 3, 3, q1d, q1d, q1d, ne);
      mfem::forall_3D(ne, q1d, q1d, q1d, [=] MFEM_HOST_DEVICE(int e)
      {
         mfem::foreach_z_thread(q1d, [&](int qz)
         {
            mfem::foreach_y_thread(q1d, [&](int qy)
            {
               mfem::foreach_x_thread(q1d, [&](int qx)
               {
                  const real_t w = W(qx, qy, qz);
                  const real_t *Jtr = &J(0, 0, qx, qy, qz, e);
                  const real_t detJ = kernels::Det<3>(Jtr);
                  const real_t wd = w * detJ;
                  real_t Jrt[9], A[9], D[9] =
                  {
                     wd, 0.0, 0.0,
                     0.0, wd, 0.0,
                     0.0, 0.0, wd
                  };
                  kernels::CalcInverse<3>(Jtr, Jrt);
                  kernels::MultABt(3, 3, 3, D, Jrt, A);
                  kernels::Mult(3, 3, 3, A, Jrt, &DX_w(0, 0, qx, qy, qz, e));
               });
            });
         });
         MFEM_SYNC_THREAD;
      });
   }

   template <int T_D1D = 0, int T_Q1D = 0>
   static void StiffnessMult(const int NE,
                             const real_t *b, const real_t *g,
                             const real_t *dx,
                             const real_t *xe, real_t *ye,
                             const int d1d, const int q1d)
   {
      // dbg();
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_VERIFY(D1D <= Q1D, "");

      constexpr int DIM = 3, VDIM = 1;
      const auto XE = Reshape(xe, D1D, D1D, D1D, VDIM, NE);
      const auto DX = Reshape(dx, 3, 3, Q1D, Q1D, Q1D, NE);
      auto YE = Reshape(ye, D1D, D1D, D1D, VDIM, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         constexpr int MQ1 = SetMaxOf(T_Q1D ? T_Q1D : 32);
         constexpr int MD1 = SetMaxOf(T_D1D ? T_D1D : 32);

         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         regs5d_t<VDIM, DIM, MQ1> r0, r1;

         LoadMatrix(D1D, Q1D, b, sB);
         LoadMatrix(D1D, Q1D, g, sG);

         LoadDofs3d(e, D1D, XE, r0);
         Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

         for (int qz = 0; qz < Q1D; qz++)
         {
            mfem::foreach_y_thread(Q1D, [&](int qy)
            {
               mfem::foreach_x_thread(Q1D, [&](int qx)
               {
                  real_t v[3], u[3] = { r1[0][0][qz][qy][qx],
                                        r1[0][1][qz][qy][qx],
                                        r1[0][2][qz][qy][qx]
                                      };
                  const real_t *dx = &DX(0, 0, qx, qy, qz, e);
                  kernels::Mult(3, 3, dx, u, v);
                  r0[0][0][qz][qy][qx] = v[0];
                  r0[0][1][qz][qy][qx] = v[1];
                  r0[0][2][qz][qy][qx] = v[2];
               });
            });
         }
         GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1);
         WriteDofs3d(e, D1D, r1, YE);
      });
   }

   using StiffnessKernelType = decltype(&StiffnessMult<>);
   MFEM_REGISTER_KERNELS(StiffnessKernels, StiffnessKernelType, (int, int));

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      StiffnessKernels::Run(d1d, q1d, ne, B, G, DX, x.Read(), y.ReadWrite(),
                            d1d, q1d);
   }
};

template <int D1D, int Q1D>
StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Kernel() { return StiffnessMult<D1D, Q1D>; }

StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Fallback(int d1d, int q1d)
{
   dbg("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
   return StiffnessMult<>;
}

///////////////////////////////////////////////////////////////////////////////
struct ∂DiffusionIntegrator : public BilinearFormIntegrator
{
   ParMesh *pmesh;
   ParGridFunction *nodes;
   const ParFiniteElementSpace *pfes, *mesh_pfes;
   int P1d, Q1d;

   static constexpr int U = 0, Ξ = 1; // potential, coordinates

public:
   ∂DiffusionIntegrator() = default;

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      dbg();

      const auto p = pfes->GetFE(0)->GetOrder();
      // const auto q = 2 * p + pmesh->GetElementTransformation(0)->OrderW();
      const auto q = 2 * p + 3;
      const auto type = pmesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);

      assert(pfes);
      assert(mesh_pfes);

      static auto solutions = std::vector{FieldDescriptor{U, pfes}};
      static auto parameters = std::vector{FieldDescriptor{Ξ, mesh_pfes}};

      static DifferentiableOperator dop(solutions, parameters, *pmesh);

      static auto diffusion_mf_kernel =
         [] MFEM_HOST_DEVICE (
            const tensor<real_t, 3>& dudxi,
            const tensor<real_t, 3, 3>& J,
            const real_t& w)
      {
         auto invJ = inv(J);
         return mfem::tuple{((dudxi * invJ)) * transpose(invJ) * det(J) * w};
      };

      static bool setup = true;
      if (setup)
      {
         dbg("\x1b[33mSetup");
         dop.AddDomainIntegrator(diffusion_mf_kernel,
                                 mfem::tuple{ Gradient<U>{}, Gradient<Ξ>{}, Weight{} },
                                 mfem::tuple{ Gradient<U>{} },
                                 ir);
         dop.SetParameters({ nodes });
         setup = false;
      }

      dop.Mult(x, y);
      // assert(false);
   }

   ////////////////////////////////////////////////////////////////////////////
   void AssemblePA(const FiniteElementSpace &fes) override
   {
      dbg();
      pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes);
      assert(pfes);

      pmesh = pfes->GetParMesh();
      nodes = static_cast<ParGridFunction *>(pmesh->GetNodes());
      assert(nodes);

      mesh_pfes = nodes->ParFESpace();
      assert(mesh_pfes);

      const auto p = pfes->GetFE(0)->GetOrder();
      // const auto q = 2 * p + pmesh->GetElementTransformation(0)->OrderW();
      const auto q = 2 * p + 3;
      dbg("p:{} q:{}", p, q);
      const auto type = pmesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);

      P1d = p + 1;
      Q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      dbg("P1d:{} Q1d:{} ", P1d, Q1d);
      // assert(false);

      // constexpr int DIM = 3;
      // const int spatial_dim = DIM,
      //           local_size = DIM * DIM,
      //           element_size = ir.GetNPoints(),
      //           total_size = DIM * DIM * ir.GetNPoints() * pmesh->GetNE();
      // ParametricSpace qdata_space(spatial_dim, local_size, element_size, total_size);
      // ParametricFunction qdata(qdata_space);
      // dbg("qdata size:{} ", qdata.Size());

#if 0
      constexpr int Ξ = 1, Δ = 2;

      DifferentiableOperator ∂_op(
      { {}},                                     // solutions
      { { Ξ, mesh_pfes }, { Δ, &qdata.space } }, // parameters
      *pmesh);

      auto qSetup =
         [] MFEM_HOST_DEVICE(const tensor<real_t, DIM, DIM> &J,
                             const real_t &w)
      {
         const auto invJ = inv(J);
         return mfem::tuple{ invJ * transpose(invJ) * det(J) * w };
      };
      ∂_op.AddDomainIntegrator(qSetup,                                   // kernel
                                 mfem::tuple{ Gradient<Ξ>{}, Weight{} }, // inputs
                                 mfem::tuple{ None<Δ>{} },               // outputs
                                 ir);
      ∂_op.SetParameters({ nodes, &qdata });

      Vector x(pfes->GetTrueVSize());
      x = 0.0;
      ∂_op.Mult(x, qdata);
      qdata.HostRead();
      assert(false);
#elif 0
      constexpr int U = 0, Ξ = 1, Δ = 2;
      DifferentiableOperator ∂_op(
      { {U, pfes}},                           // solutions
      { { Ξ, mesh_pfes } }, // parameters
      // { { Ξ, mesh_pfes }, { Δ, &qdata.space } }, // parameters
      *pmesh);
      auto qMass = [](const real_t &u,
                      const tensor<real_t, DIM, DIM> &J,
                      const real_t &w)
      {
         return mfem::tuple{ u * w * det(J) };
      };
      ∂_op.AddDomainIntegrator(qMass,
                                 mfem::tuple{ Value<U>{}, Gradient<Ξ>{}, Weight{} }, // inputs
                                 mfem::tuple{ Value<U>{} },                          // outputs
                                 ir);
      ∂_op.SetParameters({ nodes });
#else
      // nothing to do in setup
#endif
   }
};

///////////////////////////////////////////////////////////////////////////////
void AddKernelSpecializations()
{
   dbg();
   using Det = QuadratureInterpolator::DetKernels;
   Det::Specialization<3, 3, 2, 2>::Add();
   Det::Specialization<3, 3, 4, 4>::Add();

   using Grad = QuadratureInterpolator::GradKernels;
   Grad::Specialization<3, QVectorLayout::byVDIM, false, 3, 2, 3>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM, false, 3, 3, 5>::Add();
   Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 4, 5>::Add();
   Grad::Specialization<3, QVectorLayout::byVDIM, false, 3, 4, 8>::Add();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
   constexpr int DIM = 3;

   Mpi::Init();
   AddKernelSpecializations();

   const char* device_config = "cpu";
   int version = 0;
   int order = 1;
   int refinements = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&version, "-v", "--version", "");
   args.AddOption(&order, "-o", "--order", "");
   args.AddOption(&refinements, "-r", "--refinements", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0)
   {
      device.Print();
   }

   out << std::setprecision(8);

   Mesh smesh = Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON);
   smesh.EnsureNodes();
   MFEM_ASSERT(smesh.Dimension() == DIM, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      smesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.SetCurvature(order);
   smesh.Clear();

   out << "#el: " << pmesh.GetNE() << "\n";

   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   ParFiniteElementSpace& mesh_fes = *nodes->ParFESpace();

   H1_FECollection h1fec(order, DIM);
   ParFiniteElementSpace h1fes(&pmesh, &h1fec);

   out << "#dofs " << h1fes.GetTrueVSize() << "\n";

   const IntegrationRule& ir =
      IntRules.Get(h1fes.GetFE(0)->GetGeomType(),
                   h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(0)->GetOrder() + h1fes.GetFE(
                      0)->GetDim() - 1);

   printf("#ndof per el = %d\n", h1fes.GetFE(0)->GetDof());
   printf("#nqp = %d\n", ir.GetNPoints());
   printf("#q1d = %d\n", (int)floor(pow(ir.GetNPoints(), 1.0/DIM) + 0.5));

   ParGridFunction x(&h1fes), y(&h1fes);


   Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   h1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ConstantCoefficient one(1.0);

   ParLinearForm b(&h1fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.UseFastAssembly(true);
   b.Assemble();

   std::unique_ptr<ParBilinearForm> a;
   std::unique_ptr<DifferentiableOperator> dop;

   const int NE = pmesh.GetNE();
   const int NQPT = ir.GetNPoints();
   const int d1d = h1fes.GetFE(0)->GetOrder() + 1;
   const int q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   dbg("NQPT:{} d1d:{} q1d:{}", NQPT, d1d, q1d);
   MFEM_VERIFY(NQPT == q1d * q1d * q1d, "");

   const int spatial_dim = DIM;
   const int local_size = DIM * DIM;
   const int element_size = local_size * NQPT;
   const int total_size = element_size * NE;
   dbg("spatial_dim: {}, local_size: {}, element_size: {}, total_size: {}",
       spatial_dim, local_size, element_size, total_size);
   ParametricSpace qdata_space(spatial_dim, local_size, element_size, total_size,
                               d1d, q1d);
   ParametricFunction qd(qdata_space);
   MFEM_VERIFY(qd.Size() == 3 * 3 * q1d * q1d * q1d * NE, "");
   MFEM_VERIFY(qd.Size() == total_size, "");

   if (version < 2)
   {
      a = std::make_unique<ParBilinearForm>(&h1fes);
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a->AddDomainIntegrator(new DiffusionIntegrator());
      a->Assemble();
   }
   else if (version == 2) // MF ∂fem
   {
      auto diffusion_mf_kernel = [] MFEM_HOST_DEVICE (
                                    const tensor<real_t, DIM>& ∇u,
                                    const tensor<real_t, DIM, DIM>& J,
                                    const real_t& w)
      {
         auto invJ = inv(J);
         return mfem::tuple{((∇u * invJ)) * transpose(invJ) * det(J) * w};
      };
      constexpr int U = 0, Ξ = 1;
      auto solutions = std::vector{FieldDescriptor{U, &h1fes}};
      auto parameters = std::vector{FieldDescriptor{Ξ, &mesh_fes}};
      dop = std::make_unique<DifferentiableOperator>(solutions, parameters, pmesh);
      auto inputs = mfem::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}};
      auto output = mfem::tuple{Gradient<U>{}};
      dop->AddDomainIntegrator(diffusion_mf_kernel, inputs, output, ir);
      dop->SetParameters({nodes});
   }
   else if (version == 3) // PA ∂fem
   {
      {
         auto diffusion_pa_setup =
            [] MFEM_HOST_DEVICE(const real_t &u,
                                const tensor<real_t, DIM, DIM> &J,
                                const real_t &w)
         {
            const auto invJ = inv(J);
            tensor<real_t, DIM, DIM> C {{{0.0}}};
            C(0, 0) = 1.0, C(1, 1) = 1.0;
            if (DIM == 3) { C(2, 2) = 1.0; }
            const auto res = C * invJ * transpose(invJ) * det(J) * w;
            static_assert(sizeof(res) == DIM*DIM*sizeof(real_t));
            return mfem::tuple{res};
         };

         constexpr int U = 0, Ξ = 1, Q = 2;
         auto solutions = std::vector{FieldDescriptor{U, &h1fes}};
         auto parameters = std::vector{FieldDescriptor{Ξ, &mesh_fes},
                                       FieldDescriptor{Q, &qd.space}};
         DifferentiableOperator ∂Setup(solutions, parameters, pmesh);
         auto inputs = mfem::tuple{Value<U>(), Gradient<Ξ>{}, Weight{}};
         auto output = mfem::tuple{ None<Q>{} };
         ∂Setup.AddDomainIntegrator(diffusion_pa_setup, inputs, output, ir);
         ∂Setup.SetParameters({nodes, &qd});
         ParGridFunction x(&h1fes);
         ∂Setup.Mult(x, qd);
         // assert(false && "🔥🔥🔥");
      }

      {
         constexpr int U = 0, Q = 1;
         auto diffusion_pa_apply =
            [] MFEM_HOST_DEVICE(const tensor<real_t, DIM> &∇u,
                                const tensor<real_t, DIM, DIM> &Q)
         {
            return mfem::tuple{ Q * ∇u };
         };

         auto solutions = std::vector{FieldDescriptor{U, &h1fes}};
         auto parameters = std::vector{FieldDescriptor{Q, &qd.space}};
         dop = std::make_unique<DifferentiableOperator>(solutions, parameters, pmesh);
         auto inputs = mfem::tuple{Gradient<U>{}, None<Q>{}};
         auto output = mfem::tuple{Gradient<U>{}};
         dop->AddDomainIntegrator(diffusion_pa_apply, inputs, output, ir);
         dop->SetParameters({ &qd });
      }
   }
   else
   {
      MFEM_ABORT("Invalid version");
   }

   OperatorHandle A;
   Vector B, X;
   if (version >= 2)
   {
      Operator *A_ptr;
      dop->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
      A.Reset(A_ptr);
   }
   else
   {
      a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   }

   const real_t rtol = 0.0;
   const int max_it = 32, print_lvl = -1;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetOperator(*A);
   cg.iterative_mode = false;
   if constexpr (true) // check
   {
      cg.SetPrintLevel(1);
      cg.SetMaxIter(100);
      cg.SetRelTol(1e-8);
      cg.SetAbsTol(0.0);
      cg.Mult(B, X);
      MFEM_VERIFY(cg.GetConverged(), "CG solver did not converge.");
      MFEM_DEVICE_SYNC;
      mfem::out << "✅" << std::endl;
   }
   cg.SetAbsTol(0.0);
   cg.SetRelTol(rtol);
   cg.SetMaxIter(max_it);
   cg.SetPrintLevel(print_lvl);

   if (visualization)
   {
      if (version >= 2)
      {
         dop->RecoverFEMSolution(X, b, x);
      }
      else
      {
         a->RecoverFEMSolution(X, b, x);
      }
      int  visport   = 19916;
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock << "solution\n" << pmesh << x << std::flush;
   }

   return EXIT_SUCCESS;
}