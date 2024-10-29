#include "dfem/dfem.hpp"
#include "dfem/dfem_test_macro.hpp"
#include <fstream>

using namespace mfem;
using mfem::internal::tensor;
using mfem::internal::dual;

class FDJacobian : public Operator
{
public:
   FDJacobian(const Operator &op, const Vector &x) :
      Operator(op.Height()),
      op(op),
      x(x)
   {
      f.SetSize(Height());
      xpev.SetSize(Height());
      op.Mult(x, f);
      xnorm = x.Norml2();
   }

   void Mult(const Vector &v, Vector &y) const override
   {
      x.HostRead();

      // See [1] for choice of eps.
      //
      // [1] Woodward, C.S., Gardner, D.J. and Evans, K.J., 2015. On the use of
      // finite difference matrix-vector products in Newton-Krylov solvers for
      // implicit climate dynamics with spectral elements. Procedia Computer
      // Science, 51, pp.2036-2045.
      real_t eps = lambda * (lambda + xnorm / v.Norml2());

      for (int i = 0; i < x.Size(); i++)
      {
         xpev(i) = x(i) + eps * v(i);
      }

      // y = f(x + eps * v)
      op.Mult(xpev, y);

      // y = (f(x + eps * v) - f(x)) / eps
      for (int i = 0; i < x.Size(); i++)
      {
         y(i) = (y(i) - f(i)) / eps;
      }
   }

   virtual MemoryClass GetMemoryClass() const override
   {
      return Device::GetDeviceMemoryClass();
   }

private:
   const Operator &op;
   Vector x, f;
   mutable Vector xpev;
   real_t lambda = 1.0e-6;
   real_t xnorm;
};

template <typename elasticity_t>
class ElasticityOperator : public Operator
{
   template <typename elasticity_du_t>
   class ElasticityJacobianOperator : public Operator
   {
   public:
      ElasticityJacobianOperator(const ElasticityOperator *elasticity,
                                 std::shared_ptr<elasticity_du_t> dRdu) :
         Operator(elasticity->Height()),
         elasticity(elasticity),
         dRdu(dRdu),
         x_ess(dRdu->Height())
      {
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         x_ess = x;
         x_ess.SetSubVector(elasticity->ess_tdofs, 0.0);

         dRdu->Mult(x_ess, y);

         for (int i = 0; i < elasticity->ess_tdofs.Size(); i++)
         {
            y[elasticity->ess_tdofs[i]] = x[elasticity->ess_tdofs[i]];
         }
      }

      const ElasticityOperator *elasticity = nullptr;
      std::shared_ptr<elasticity_du_t> dRdu;
      mutable Vector x_ess;
   };

public:
   ElasticityOperator(ParFiniteElementSpace &fes, elasticity_t &elasticity,
                      Array<int> &ess_tdofs) :
      Operator(fes.GetTrueVSize()),
      fes(fes),
      elasticity(elasticity),
      ess_tdofs(ess_tdofs) {}

   void Mult(const Vector &x, Vector &r) const override
   {
      elasticity.Mult(x, r);
      r.SetSubVector(ess_tdofs, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      ParGridFunction u(const_cast<ParFiniteElementSpace *>
                        (*std::get_if<const ParFiniteElementSpace *>
                         (&elasticity.solutions[0].data)));

      u.SetFromTrueDofs(x);
      auto dRdu = elasticity.template GetDerivativeWrt<0>({&u}, {mesh_nodes});

      jacobian.reset(
         new ElasticityJacobianOperator<
         typename std::remove_pointer<decltype(dRdu.get())>::type> (this, dRdu));

      // jacobian.reset(new FDJacobian(*this, x));

      return *jacobian;
   }

   void SetParameters(ParGridFunction &mesh_nodes)
   {
      elasticity.SetParameters({&mesh_nodes});
      this->mesh_nodes = &mesh_nodes;
   }

   ParFiniteElementSpace &fes;
   elasticity_t &elasticity;
   Array<int> ess_tdofs;
   mutable ParGridFunction *mesh_nodes = nullptr;
   mutable std::shared_ptr<Operator> jacobian;
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
   out << "#dof: " << h1fes.GetNDofs() << "\n";

   ParGridFunction u(&h1fes);

   auto elasticity_kernel = [] MFEM_HOST_DEVICE
                            (const tensor<dual<real_t, real_t>, dim, dim> &dudxi,
                             const tensor<real_t, dim, dim> &J,
                             const real_t &w)
   {
      // shear modulus
      real_t D1{0.1e6};
      // bulk modulus
      real_t C1{1.0e6};
      constexpr auto I = mfem::internal::IsotropicIdentity<dim>();
      auto invJ = inv(J);
      auto dudx = dudxi * invJ;
      auto F = det(I + dudx);
      auto p = -2.0 * D1 * F * (F - 1);
      auto devB = dev(dudx + transpose(dudx) + dot(dudx, transpose(dudx)));
      auto sigma = -(p / F) * I + 2.0 * (C1 / pow(F, 5.0 / 3.0)) * devB;

      return mfem::tuple{sigma * det(J) * w * transpose(invJ)};
   };

   mfem::tuple argument_operators{Gradient{"displacement"}, Gradient{"coordinates"}, Weight{}};
   mfem::tuple output_operator{Gradient{"displacement"}};

   ElementOperator op(elasticity_kernel, argument_operators, output_operator, ir);

   std::array solutions{FieldDescriptor{&h1fes, "displacement"}};
   std::array parameters{FieldDescriptor{&mesh_fes, "coordinates"}};

   DifferentiableOperator dop(solutions, parameters, mfem::tuple{op}, mesh,
                              AutoDiff::NativeDualNumber{});

   ElasticityOperator elasticity(h1fes, dop, ess_tdof_list);

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
   b.UseFastAssembly(true);
   b.Assemble();
   auto B = b.ParallelAssemble();

   Vector X = u.GetTrueVector();

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-8);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetSolver(cg);
   newton.SetOperator(elasticity);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(100);
   // newton.SetAdaptiveLinRtol();
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());

   elasticity.SetParameters(*mesh_nodes);

   // Vector zero;
   newton.Mult(*B, X);

   u.SetFromTrueDofs(X);

   ParaViewDataCollection paraview_dc("dfem", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(polynomial_order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("displacement", &u);
   paraview_dc.Save();

   return 0;
}

DFEM_TEST_MAIN(test_nonlinear_elasticity_3d);
