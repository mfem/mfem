#include <mfem.hpp>

#include "fem/qinterp/det.cpp"
#include "fem/qinterp/grad.hpp" // IWYU pragma: keep

#include <fem/dfem/doperator.hpp>
#include <fem/dfem/tuple.hpp>
#include <fem/dfem/util.hpp>

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

static int D1D = 0, Q1D = 0;

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
      StiffnessKernels::Specialization<5, 10>::Add();
      StiffnessKernels::Specialization<7, 15>::Add();
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
      d1d = p + 1;
      q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      MFEM_VERIFY(d1d == D1D, "D1D mismatch: " << d1d << " != " << D1D);
      MFEM_VERIFY(q1d == Q1D, "Q1D mismatch: " << q1d << " != " << Q1D);
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

   template <int MD1, int MQ1, int T_D1D = 0, int T_Q1D = 0>
   static void StiffnessMult(const int NE,
                             const real_t *b, const real_t *g,
                             const real_t *dx,
                             const real_t *xe, real_t *ye,
                             const int d1d, const int q1d)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int DIM = 3, VDIM = 1;
      const auto XE = Reshape(xe, D1D, D1D, D1D, VDIM, NE);
      const auto DX = Reshape(dx, 3, 3, Q1D, Q1D, Q1D, NE);
      auto YE = Reshape(ye, D1D, D1D, D1D, VDIM, NE);

      mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE(int e)
      {
         MFEM_SHARED real_t smem[MQ1][MQ1];
         MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1];
         regs5d_t<VDIM, DIM, MQ1> r0, r1;

         LoadMatrix(D1D, Q1D, b, sB);
         LoadMatrix(D1D, Q1D, g, sG);

         LoadDofs3d(e, D1D, XE, r0);
         regs_Grad3d(D1D, Q1D, smem, sB, sG, r0, r1);

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
         regs_GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1);
         WriteDofs3d(e, D1D, r1, YE);
      });
   }

   using StiffnessKernelType = decltype(&StiffnessMult<1,1>);
   MFEM_REGISTER_KERNELS(StiffnessKernels, StiffnessKernelType, (int, int));

   void AddMultPA(const Vector &x, Vector &y) const override
   {
      StiffnessKernels::Run(d1d, q1d, ne, B, G, DX, x.Read(), y.ReadWrite(),
                            d1d, q1d);
   }
};

template <int D1D, int Q1D>
StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Kernel()
{
   return StiffnessMult<SetMaxOf(D1D), SetMaxOf(Q1D), D1D, Q1D>;
}

StiffnessIntegrator::StiffnessKernelType
StiffnessIntegrator::StiffnessKernels::Fallback(int d1d, int q1d)
{
   dbg("\x1b[33mFallback d1d:{} q1d:{}", d1d, q1d);
   return StiffnessMult<DofQuadLimits::MAX_D1D, DofQuadLimits::MAX_Q1D>;
}

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
   ParFiniteElementSpace& mfes = *nodes->ParFESpace();

   H1_FECollection fec(order, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   const auto p = fes.GetFE(0)->GetOrder();
   const auto q = 2 * p + pmesh.GetElementTransformation(0)->OrderW();
   const auto type = pmesh.GetElementBaseGeometry(0);
   const IntegrationRule &ir = IntRules.Get(type, q);
   D1D = p + 1;
   Q1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
   dbg("D1D: {}, Q1D: {}", D1D, Q1D);

   const int NE = pmesh.GetNE();
   const int NQPT = ir.GetNPoints();

   ParGridFunction x(&fes), y(&fes);

   Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 1;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   ConstantCoefficient one(1.0);

   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.UseFastAssembly(true);
   b.Assemble();

   std::unique_ptr<ParBilinearForm> a;
   std::unique_ptr<DifferentiableOperator> ∂op;


   const int elem_size = DIM * DIM * NQPT;
   const int total_size = elem_size * NE;
   dbg("DIM: {}, local_size: {}, elem_size: {}, total_size: {}",
       DIM, DIM * DIM, elem_size, total_size);
   ParametricSpace qdata_space(DIM, DIM * DIM, elem_size, total_size, D1D, Q1D);
   ParametricFunction qd(qdata_space);

   if (version < 2)
   {
      a = std::make_unique<ParBilinearForm>(&fes);
      a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      if (version == 0) { a->AddDomainIntegrator(new DiffusionIntegrator(&ir)); }
      if (version == 1) { a->AddDomainIntegrator(new StiffnessIntegrator()); }
      a->Assemble();
      if (version == 0)
      {
         BilinearFormIntegrator *bfi = a->GetDBFI()->operator[](0);
         auto *di = dynamic_cast<DiffusionIntegrator*>(bfi);
         assert(di);
         const int d1d = di->dofs1D, q1d = di->quad1D;
         dbg("\x1b[33md1d: {} q1d: {}", d1d, q1d);
         MFEM_VERIFY(d1d == D1D, "D1D mismatch: " << d1d << " != " << D1D);
         MFEM_VERIFY(q1d == Q1D, "Q1D mismatch: " << q1d << " != " << Q1D);
      }
   }
   else if (version == 2) // MF ∂fem
   {
      constexpr int U = 0, Ξ = 1;
      auto solutions = std::vector{FieldDescriptor{U, &fes}};
      auto parameters = std::vector{FieldDescriptor{Ξ, &mfes}};
      auto diffusion_mf_kernel =
         [] MFEM_HOST_DEVICE (const tensor<real_t, DIM>& ∇u,
                              const tensor<real_t, DIM, DIM>& J,
                              const real_t& w)
      {
         auto invJ = inv(J);
         return mfem::tuple{((∇u * invJ)) * transpose(invJ) * det(J) * w};
      };
      ∂op = std::make_unique<DifferentiableOperator>(solutions, parameters, pmesh);
      ∂op->SetParameters({nodes});
      ∂op->AddDomainIntegrator(diffusion_mf_kernel,
                                 mfem::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}},
                                 mfem::tuple{Gradient<U>{}},
                                 ir);
   }
   else if (version == 3) // PA ∂fem
   {
      constexpr int U = 0, Ξ = 1, Q = 2;
      FieldDescriptor u_fd{U, &fes}, Ξ_fd{Ξ, &mfes}, q_fd{Q, &qd.space};
      auto w = Weight{};
      auto q = None<Q> {};
      auto u = None<U> {};
      auto ∇u = Gradient<U> {};
      auto ∇Ξ = Gradient<Ξ> {};
      auto u_sol = std::vector{u_fd},
           q_param = std::vector{q_fd},
           Ξ_q_params = std::vector{Ξ_fd, q_fd};
      mfem::tuple u_J_w = {u, ∇Ξ, w};
      mfem::tuple ∇u_q = {∇u, q};

      auto setup =
         [] MFEM_HOST_DEVICE(const real_t &u,
                             const tensor<real_t, DIM, DIM> &J,
                             const real_t &w)
      {
         return mfem::tuple{inv(J) * transpose(inv(J)) * det(J) * w};
      };
      DifferentiableOperator ∂Setup(u_sol, Ξ_q_params, pmesh);
      ∂Setup.SetParameters({nodes, &qd});
      ∂Setup.AddDomainIntegrator(setup, u_J_w, mfem::tuple{q}, ir);
      ∂Setup.Mult(Vector{fes.GetTrueVSize()}, qd);

      auto apply =
         [] MFEM_HOST_DEVICE(const tensor<real_t, DIM> &∇u,
                             const tensor<real_t, DIM, DIM> &q)
      {
         return mfem::tuple{q * ∇u};
      };
      ∂op = std::make_unique<DifferentiableOperator>(u_sol, q_param, pmesh);
      ∂op->SetParameters({ &qd });
      ∂op->AddDomainIntegrator(apply, ∇u_q, mfem::tuple{∇u}, ir);
   }
   else { MFEM_ABORT("Invalid version"); }

   OperatorHandle A;
   Vector B, X;
   if (version >= 2)
   {
      Operator *A_ptr;
      ∂op->FormLinearSystem(ess_tdof_list, x, b, A_ptr, X, B);
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
         ∂op->RecoverFEMSolution(X, b, x);
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