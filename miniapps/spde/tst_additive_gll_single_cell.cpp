#include "frac_noise.hpp"
#include "periodic_fraclap_coefficients.hpp"

using namespace std;
using namespace mfem;

class AdditiveGLLMultilevelGenerator : public Solver
{
public:
   AdditiveGLLMultilevelGenerator(ParMesh &pmesh_,
                                  int par_ref_levels_,
                                  int order_,
                                  real_t sigma_,
                                  real_t s_,
                                  int smoother_applications_)
      : Solver(0),
        pmesh(pmesh_),
        sigma(sigma_),
        s(s_),
        smoother_applications(smoother_applications_),
        gll_rules(0, Quadrature1D::GaussLobatto)
   {
      MFEM_VERIFY(order_ >= 1, "Expected finite element order >= 1.");
      MFEM_VERIFY(smoother_applications >= 1,
                  "Expected at least one smoother application.");

      fec.reset(new H1_FECollection(order_, pmesh.Dimension(),
                                    BasisType::GaussLobatto));
      ParFiniteElementSpace *coarse_fes =
         new ParFiniteElementSpace(&pmesh, fec.get());

      fespaces.reset(new ParFiniteElementSpaceHierarchy(&pmesh, coarse_fes,
                                                        false, true));

      for (int l = 0; l < par_ref_levels_; l++)
      {
         fespaces->AddUniformlyRefinedLevel(1, Ordering::byVDIM);
      }

      const int nlevels = fespaces->GetNumLevels();
      prolongations.SetSize(nlevels - 1);
      for (int level = 0; level < nlevels - 1; level++)
      {
         prolongations[level] = fespaces->GetProlongationAtLevel(level);
      }

      operators.SetSize(nlevels);
      smoothers.SetSize(nlevels);

      for (int level = 0; level < nlevels; level++)
      {
         ParFiniteElementSpace &fes = fespaces->GetFESpaceAtLevel(level);
         unique_ptr<ParBilinearForm> bf(new ParBilinearForm(&fes));
         ConstantCoefficient one(1.0);
         ConstantCoefficient mass_coef(sigma*sigma);
         const IntegrationRule &mass_ir = GetMassRule(fes);

         bf->AddDomainIntegrator(new MassIntegrator(mass_coef, &mass_ir));
         bf->AddDomainIntegrator(new DiffusionIntegrator(one));
         bf->Assemble();
         bf->Finalize();

         HypreParMatrix *mat = bf->ParallelAssemble();
         operators[level] = mat;

         Vector diag(fes.GetTrueVSize());
         mat->GetDiag(diag);
         for (int i = 0; i < diag.Size(); i++)
         {
            diag[i] = 1.0/sqrt(diag[i]);
         }
         Vector tmp(fes.GetTrueVSize());
         mat->Mult(diag, tmp);
         real_t omega = InnerProduct(fes.GetComm(), diag, tmp);
         mat->GetDiag(diag);
         diag *= omega;

         Array<int> ess_tdofs;
         OperatorJacobiSmoother *jac =
            new OperatorJacobiSmoother(diag, ess_tdofs, 1.0);
         jac->iterative_mode = false;
         jac->Setup(diag);
         smoothers[level] = jac;
      }

      height = width = fespaces->GetFinestFESpace().GetTrueVSize();
   }

   ~AdditiveGLLMultilevelGenerator() override
   {
      for (int i = 0; i < operators.Size(); i++) { delete operators[i]; }
      for (int i = 0; i < smoothers.Size(); i++) { delete smoothers[i]; }
   }

   ParFiniteElementSpace &GetFinestFESpace() const
   {
      return fespaces->GetFinestFESpace();
   }

   const IntegrationRule &GetMassRule(ParFiniteElementSpace &fes) const
   {
      const FiniteElement *fe = fes.GetTypicalFE();
      return gll_rules.Get(fe->GetGeomType(), 2*fe->GetOrder() - 1);
   }

   void PrintMassDiagonalDiagnostics() const
   {
      for (int level = 0; level < fespaces->GetNumLevels(); level++)
      {
         ParFiniteElementSpace &fes = fespaces->GetFESpaceAtLevel(level);
         unique_ptr<ParBilinearForm> mass(new ParBilinearForm(&fes));
         ConstantCoefficient one(1.0);
         const IntegrationRule &mass_ir = GetMassRule(fes);
         mass->AddDomainIntegrator(new MassIntegrator(one, &mass_ir));
         mass->Assemble();
         mass->Finalize();
         unique_ptr<HypreParMatrix> M(mass->ParallelAssemble());

         Vector diag(fes.GetTrueVSize());
         Vector rowsum(fes.GetTrueVSize());
         Vector ones(fes.GetTrueVSize());
         ones = 1.0;
         M->GetDiag(diag);
         M->AbsMult(ones, rowsum);
         rowsum -= diag;

         real_t local_max = 0.0;
         for (int i = 0; i < rowsum.Size(); i++)
         {
            local_max = std::max(local_max, std::abs(rowsum[i]));
         }

         real_t global_max = 0.0;
         MPI_Allreduce(&local_max, &global_max, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, fes.GetComm());

         if (Mpi::Root())
         {
            cout << "Level " << level
                 << " true size=" << fes.GlobalTrueVSize()
                 << " local true size=" << fes.GetTrueVSize()
                 << " mass offdiag row-sum max=" << global_max
                 << " mass quadrature points=" << mass_ir.GetNPoints()
                 << endl;
         }
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      const int nlevels = fespaces->GetNumLevels();
      MFEM_VERIFY(x.Size() == fespaces->GetFinestFESpace().GetTrueVSize(),
                  "Input vector size does not match finest FE space.");
      y.SetSize(x.Size());

      vector<Vector*> u;
      vector<Vector*> v;
      u.reserve(nlevels);
      v.reserve(nlevels);
      for (int level = 0; level < nlevels; level++)
      {
         u.push_back(new Vector(fespaces->GetFESpaceAtLevel(level)
                                .GetTrueVSize()));
         v.push_back(new Vector(fespaces->GetFESpaceAtLevel(level)
                                .GetTrueVSize()));
      }

      SymmetrizedSmoother ms(smoothers[nlevels - 1], operators[nlevels - 1]);
      ApplyRepeated(ms, x, *u[nlevels - 1]);

      Vector rhs(x);
      Vector tmp1, tmp2, tmp3;

      for (int level = nlevels - 2; level >= 0; level--)
      {
         ApplyRepeated(*smoothers[level + 1], rhs, tmp1);
         tmp2.SetSize(tmp1.Size());
         operators[level + 1]->Mult(tmp1, tmp2);
         tmp3.SetSize(rhs.Size());
         add(1.0, rhs, -1.0, tmp2, tmp3);
         rhs.SetSize(prolongations[level]->Width());
         prolongations[level]->MultTranspose(tmp3, rhs);

         ms.SetSmoother(*smoothers[level]);
         ms.SetOperator(*operators[level]);
         ApplyRepeated(ms, rhs, *u[level]);
      }

      Array<real_t> h(nlevels);
      for (int level = 0; level < nlevels; level++)
      {
         real_t h_min, h_max, kappa_min, kappa_max;
         fespaces->GetFESpaceAtLevel(level).GetParMesh()->GetCharacteristics(
            h_min, h_max, kappa_min, kappa_max);
         h[level] = h_max;
      }

      real_t mu = pow(sigma*sigma + 1.0/(h[0]*h[0]), s);
      v[0]->Set(mu, *u[0]);

      for (int level = 1; level < nlevels; level++)
      {
         mu = pow(sigma*sigma + 1.0/(h[level]*h[level]), s);
         tmp1.SetSize(v[level]->Size());
         prolongations[level - 1]->Mult(*v[level - 1], tmp1);
         tmp2.SetSize(tmp1.Size());
         operators[level]->Mult(tmp1, tmp2);
         ApplyTransposeRepeated(*smoothers[level], tmp2, tmp3);
         tmp1.Add(-1.0, tmp3);

         v[level]->Set(mu, *u[level]);
         v[level]->Add(1.0, tmp1);
      }

      mu = pow(sigma*sigma + 1.0/(h[0]*h[0]), s);
      y.Set(1.0/mu, *v[nlevels - 1]);

      for (int level = 0; level < nlevels; level++)
      {
         delete u[level];
         delete v[level];
      }
   }

   void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("SetOperator is not supported.");
   }

private:
   void ApplyRepeated(const Solver &solver, const Vector &x, Vector &y) const
   {
      y.SetSize(x.Size());
      Vector src(x);
      Vector dst(x.Size());
      for (int i = 0; i < smoother_applications; i++)
      {
         solver.Mult(src, dst);
         if (i + 1 < smoother_applications) { src = dst; }
      }
      y = dst;
   }

   void ApplyTransposeRepeated(const Solver &solver,
                               const Vector &x,
                               Vector &y) const
   {
      y.SetSize(x.Size());
      Vector src(x);
      Vector dst(x.Size());
      for (int i = 0; i < smoother_applications; i++)
      {
         solver.MultTranspose(src, dst);
         if (i + 1 < smoother_applications) { src = dst; }
      }
      y = dst;
   }

   ParMesh &pmesh;
   real_t sigma;
   real_t s;
   int smoother_applications;

   mutable IntegrationRules gll_rules;
   unique_ptr<FiniteElementCollection> fec;
   unique_ptr<ParFiniteElementSpaceHierarchy> fespaces;
   Array<Operator*> prolongations;
   Array<Operator*> operators;
   Array<Solver*> smoothers;
};

static Mesh MakeSingleCellMesh(int dim)
{
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL,
                                   false, 1.0, 1.0, false);
   }
   return Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON,
                                1.0, 1.0, 1.0, false);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   Hypre::Init();

   const char *device_config = "cpu";
   int dim = 2;
   int order = 4;
   int par_ref_levels = 4;
   int smoother_applications = 1;
   bool paraview = true;
   real_t sigma = 0.0;
   real_t s = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&dim, "-dim", "--dimension",
                  "Dimension. The Neumann MMS test currently supports 2D.");
   args.AddOption(&order, "-o", "--order",
                  "GLL H1 finite element order.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel uniform refinements from one cell.");
   args.AddOption(&smoother_applications, "-ns",
                  "--num-smoother-applications",
                  "Number of smoother applications on every level.");
   args.AddOption(&sigma, "-sigma", "--sigma", "Mass scale sigma.");
   args.AddOption(&s, "-s", "--s", "Fractional exponent parameter.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();

   MFEM_VERIFY(dim == 2,
               "The Neumann MMS in periodic_fraclap_coefficients.hpp is 2D.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh mesh = MakeSingleCellMesh(dim);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   AdditiveGLLMultilevelGenerator generator(pmesh, par_ref_levels, order,
                                            sigma, s, smoother_applications);
   ParFiniteElementSpace &fes = generator.GetFinestFESpace();

   if (Mpi::Root())
   {
      cout << "Additive GLL multilevel test from one non-periodic cell" << endl;
      cout << "  dim=" << dim
           << " order=" << order
           << " par_ref_levels=" << par_ref_levels
           << " finest true size=" << fes.GlobalTrueVSize()
           << " sigma=" << sigma
           << " s=" << s
           << " neumann alpha=" << 2.0*(1.0 - s)
           << " smoother_applications=" << smoother_applications << endl;
   }

   generator.PrintMassDiagonalDiagnostics();

   const real_t alpha = 2.0*(1.0 - s);
   FunctionCoefficient exact_coeff(periodic_fraclap::UExact);
   FunctionCoefficient rhs_coeff(
      [alpha](const Vector &X) { return periodic_fraclap::RHS(X, alpha); });

   ParGridFunction exact(&fes);
   exact.ProjectCoefficient(exact_coeff);

   ParGridFunction rhs_gf(&fes);
   rhs_gf.ProjectCoefficient(rhs_coeff);

   ParLinearForm rhs(&fes);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(rhs_coeff));
   rhs.Assemble();
   unique_ptr<HypreParVector> rhs_true(rhs.ParallelAssemble());
   rhs_true->UseDevice(true);

   Vector sol_true(fes.GetTrueVSize());
   generator.Mult(*rhs_true, sol_true);

   ParGridFunction sol(&fes);
   sol.SetFromTrueDofs(sol_true);

   ParGridFunction err(&fes);
   err = sol;
   err -= exact;

   if (Mpi::Root())
   {
      ConstantCoefficient zero(0.0);
      const real_t exact_l2 = exact.ComputeL2Error(zero);
      const real_t err_l2 = err.ComputeL2Error(zero);
      cout << "  exact L2 norm=" << exact_l2 << endl;
      cout << "  additive GLL absolute L2 error=" << err_l2 << endl;
      cout << "  additive GLL relative L2 error=" << err_l2/exact_l2 << endl;
   }

   if (paraview)
   {
      ParaViewDataCollection paraview_dc("additive_gll_single_cell",
                                         fes.GetParMesh());
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("exact", &exact);
      paraview_dc.RegisterField("rhs", &rhs_gf);
      paraview_dc.RegisterField("solution", &sol);
      paraview_dc.RegisterField("error", &err);
      paraview_dc.Save();
   }

   return EXIT_SUCCESS;
}
