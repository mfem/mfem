#include "dfem/dfem_refactor.hpp"
#include "fem/pbilinearform.hpp"

using namespace mfem;
using mfem::internal::tensor;

constexpr int DIMENSION = 2;

template <int dim = 2>
struct TemperatureMassQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const real_t &T,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      return mfem::tuple{T * det(J) * w};
   }
};

template <int dim = 2>
struct TemperatureDiffusionQFunction
{
   MFEM_HOST_DEVICE inline
   auto operator()(const tensor<real_t, 2> &dTdxi,
                   const tensor<real_t, dim, dim> &J,
                   const real_t &w) const
   {
      auto invJ = inv(J);
      auto dTdx = dTdxi * invJ;
      auto JxW = det(J) * w * transpose(invJ);
      return mfem::tuple{dTdx * JxW};
   }
};

class HeatOperator : public TimeDependentOperator
{
   static constexpr int Position = 0;
   static constexpr int Temperature = 1;

   class HeatResidual : public Operator
   {
   public:
      HeatResidual(
         HeatOperator &op,
         const real_t &gamma,
         const Vector &T,
         const Vector &prevT,
         const Vector &source,
         const Vector &prev_source) :
         Operator(op.Height()),
         op(op),
         gamma(gamma),
         prevT(prevT),
         source(source),
         prev_source(prev_source),
         z(T.Size()),
         H1tsize(op.H1fes.GetTrueVSize()) {}

      void Mult(const Vector &T, Vector &R) const override
      {
         auto x_gf = static_cast<ParGridFunction*>(op.H1fes.GetParMesh()->GetNodes());

         R = 0.0;

         op.mass->SetParameters({x_gf});
         op.mass->Mult(T, R);

         // Current F(T)
         op.diffusion->SetParameters({x_gf});
         op.diffusion->AddMult(T, R, gamma);
         op.mass->AddMult(source, R, -gamma);

         // Previous F(T)
         op.diffusion->SetParameters({x_gf});
         op.diffusion->AddMult(prevT, R, gamma);
         op.mass->AddMult(prev_source, R, -gamma);

         // Previous time stepping terms
         op.mass->SetParameters({x_gf});
         op.mass->AddMult(prevT, R, -1.0);

         R.SetSubVector(op.temperature_ess_tdof, 0.0);
      }

      Operator& GetGradient(const Vector &u) const override
      {
         fd_jacobian.reset(new FDJacobian(*this, u));
         // std::ofstream fd_jac_out("fd_jac.dat");
         // fd_jacobian->PrintMatlab(fd_jac_out);
         // fd_jac_out.close();
         return *fd_jacobian;
      }

      HeatOperator &op;
      const real_t gamma;

      const int H1tsize;

      Vector prevT, source, prev_source;
      mutable Vector z;

      mutable std::shared_ptr<FDJacobian> fd_jacobian;
   };

public:
   HeatOperator(
      ParFiniteElementSpace &H1fes,
      Array<int> &temperature_ess_attr,
      const IntegrationRule &ir,
      Coefficient &temperature_exact_coeff,
      Coefficient &source_coeff) :
      TimeDependentOperator(H1fes.GetTrueVSize()),
      H1fes(H1fes),
      H1tsize(H1fes.GetTrueVSize()),
      temperature_ess_attr(temperature_ess_attr),
      ir(ir),
      temperature_exact_coeff(temperature_exact_coeff),
      T_gf(&H1fes),
      source_coeff(source_coeff),
      source_gf(&H1fes),
      source_tdof(H1tsize),
      prev_source_tdof(H1tsize),
      prevT(H1tsize)
   {
      auto mesh = H1fes.GetParMesh();
      x_gf = static_cast<ParGridFunction*>(mesh->GetNodes());
      ParFiniteElementSpace& mesh_fes = *x_gf->ParFESpace();

      H1fes.GetEssentialTrueDofs(temperature_ess_attr, temperature_ess_tdof);

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Temperature, &H1fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         mass =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         mfem::tuple inputs{Value<Temperature>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Value<Temperature>{}};

         auto mass_qf = TemperatureMassQFunction<DIMENSION> {};
         mass->AddDomainIntegrator(mass_qf, inputs, outputs, ir);
      }

      {
         auto solutions = std::vector
         {
            FieldDescriptor{Temperature, &H1fes},
         };

         auto parameters = std::vector
         {
            FieldDescriptor{Position, &mesh_fes}
         };

         diffusion =
            std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

         mfem::tuple inputs{Gradient<Temperature>{}, Gradient<Position>{}, Weight{}};
         mfem::tuple outputs{Gradient<Temperature>{}};

         auto diffusion_qf = TemperatureDiffusionQFunction<DIMENSION> {};
         auto derivatives = std::integer_sequence<size_t, Temperature> {};
         diffusion->AddDomainIntegrator(
            diffusion_qf, inputs, outputs, ir, derivatives);
      }
   }

   void SetTime(const real_t t)
   {
      TimeDependentOperator::SetTime(t);
      temperature_exact_coeff.SetTime(t);
      source_coeff.SetTime(t);
   }

   void Step(Vector &T, real_t &t, const real_t &dt)
   {
      this->SetTime(t);
      prevT = T;
      source_gf.ProjectCoefficient(source_coeff);
      source_gf.GetTrueDofs(prev_source_tdof);

      this->SetTime(t + dt);
      T_gf.SetFromTrueDofs(T);
      T_gf.ProjectBdrCoefficient(temperature_exact_coeff, temperature_ess_attr);
      T_gf.GetTrueDofs(T);

      source_gf.ProjectCoefficient(source_coeff);
      source_gf.GetTrueDofs(source_tdof);

      // Implicit midpoint
      HeatResidual residual(*this, 0.5*dt, T, prevT, source_tdof, prev_source_tdof);

      GMRESSolver krylov(MPI_COMM_WORLD);
      krylov.SetRelTol(1e-4);
      krylov.SetMaxIter(1000);
      krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

      NewtonSolver newton(MPI_COMM_WORLD);
      newton.SetOperator(residual);
      newton.SetSolver(krylov);
      newton.SetRelTol(1e-8);
      newton.SetMaxIter(10);
      newton.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

      Vector zero;
      newton.Mult(zero, T);

      t += dt;
   }

   std::shared_ptr<DifferentiableOperator> mass;
   std::shared_ptr<DifferentiableOperator> diffusion;

   ParGridFunction *x_gf, source_gf, T_gf;

   const Array<int> temperature_ess_attr;
   Array<int> temperature_ess_tdof;

   ParFiniteElementSpace &H1fes;
   const int H1tsize;

   Vector source_tdof, prev_source_tdof, prevT;

   IntegrationRule ir;
   Coefficient &temperature_exact_coeff;
   Coefficient &source_coeff;
};

int main(int argc, char* argv[])
{
   constexpr int dim = 2;

   Mpi::Init();

   const char* device_config = "cpu";
   const char* mesh_file = "";
   int polynomial_order_temperature = 2;
   int refinements = 0;
   int problem_type = 0;
   real_t t_final = 0.0;
   real_t dt = 1e-3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&polynomial_order_temperature, "-ot", "--order-temperature", "");
   args.AddOption(&refinements, "-r", "--r", "");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&t_final, "-tf", "--tf", "");
   args.AddOption(&dt, "-dt", "--dt", "");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root() == 0)
   {
      device.Print();
   }

   out << std::setprecision(8);

   Mesh mesh_serial = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   MFEM_ASSERT(mesh_serial.Dimension() == dim, "incorrect mesh dimension");

   for (int i = 0; i < refinements; i++)
   {
      mesh_serial.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

   mesh.EnsureNodes();
   mesh_serial.Clear();

   out << "#el: " << mesh.GetNE() << "\n";

   H1_FECollection temperature_fec(polynomial_order_temperature);

   ParFiniteElementSpace H1fes(&mesh, &temperature_fec);

   HYPRE_BigInt global_size_temperature = H1fes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      out << "Number of temperature unknowns: " << global_size_temperature << "\n";
   }

   const IntegrationRule &integration_rule =
      IntRules.Get(H1fes.GetFE(0)->GetGeomType(),
                   2 * H1fes.GetFE(0)->GetOrder() + 1);

   Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
   bdr_attr_is_ess = 1;

   Vector T(H1fes.GetTrueVSize());

   ParGridFunction T_gf(&H1fes);

   auto temperature_exact = [](const Vector &coords, real_t t)
   {
      const real_t x = coords(0);
      const real_t y = coords(1);

      return (pow(cos(y),2) + pow(sin(x),2))/exp(2.*t);
   };

   FunctionCoefficient temperature_exact_coeff(temperature_exact);

   T_gf.ProjectCoefficient(temperature_exact_coeff);
   T_gf.GetTrueDofs(T);

   auto source_term = [](const Vector &coords, real_t t)
   {
      const real_t x = coords(0);
      const real_t y = coords(1);

      return (-2*pow(cos(x),2))/exp(2.*t) + (2*pow(cos(y),
                                                   2))/exp(2.*t) + (2*pow(sin(x),2))/exp(2.*t) - (2.*(pow(cos(y),2) + pow(sin(x),
                                                         2)))/exp(2.*t) - (2*pow(sin(y),2))/exp(2.*t);
   };

   FunctionCoefficient source_term_coeff(source_term);

   HeatOperator heat(H1fes, bdr_attr_is_ess, integration_rule,
                     temperature_exact_coeff, source_term_coeff);

   real_t t = 0.0;
   out << "time step: " << dt << "\n";
   real_t t_old;
   bool last_step = false;

   T_gf.SetFromTrueDofs(T);

   ParGridFunction Terr_gf(&H1fes), Tex_gf(&H1fes);
   Terr_gf = 0.0;
   Tex_gf.ProjectCoefficient(temperature_exact_coeff);

   ParaViewDataCollection dc("dfem_heat", &mesh);
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(polynomial_order_temperature);
   dc.RegisterField("temperature", &T_gf);
   dc.RegisterField("temperature_exact", &Tex_gf);
   dc.RegisterField("temperature_error", &Terr_gf);
   dc.SetCycle(0);
   dc.SetTime(0);
   dc.Save();

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (Mpi::Root())
      {
         out << "step " << std::setw(5) << ti
             << ",\tt = " << std::setw(5) << std::setprecision(4) << t
             << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt;
         out << std::endl;
      }

      heat.Step(T, t, dt);

      T_gf.SetFromTrueDofs(T);
      temperature_exact_coeff.SetTime(t);
      real_t T_l2err = T_gf.ComputeL2Error(temperature_exact_coeff);
      if (Mpi::Root())
      {
         out << "|T - T_exact|_L2 = " << T_l2err << std::endl;
      }

      Tex_gf.ProjectCoefficient(temperature_exact_coeff);
      for (int i = 0; i < Terr_gf.Size(); i++)
      {
         Terr_gf(i) = abs(Tex_gf(i) - T_gf(i));
      }

      if (ti % 1 == 0)
      {
         dc.SetCycle(ti);
         dc.SetTime(t);
         dc.Save();
      }

      if (Mpi::Root())
      {
         out << "\n" << std::endl;
      }
   }

   return 0;
}

// class HeatOperator : public Operator
// {
//    static constexpr int Position = 0;
//    static constexpr int Temperature = 1;

// public:
//    HeatOperator(
//       ParFiniteElementSpace &H1fes,
//       Array<int> &temperature_ess_attr,
//       const IntegrationRule &ir,
//       Coefficient &temperature_exact_coeff,
//       Coefficient &source_coeff) :
//       Operator(H1fes.GetTrueVSize()),
//       H1fes(H1fes),
//       H1tsize(H1fes.GetTrueVSize()),
//       temperature_ess_attr(temperature_ess_attr),
//       ir(ir),
//       temperature_exact_coeff(temperature_exact_coeff),
//       T_gf(&H1fes),
//       source_coeff(source_coeff),
//       source_gf(&H1fes),
//       source_tdof(H1tsize)
//    {
//       auto mesh = H1fes.GetParMesh();
//       x_gf = static_cast<ParGridFunction*>(mesh->GetNodes());
//       ParFiniteElementSpace& mesh_fes = *x_gf->ParFESpace();

//       H1fes.GetEssentialTrueDofs(temperature_ess_attr, temperature_ess_tdof);

//       {
//          {
//             auto solutions = std::vector
//             {
//                FieldDescriptor{Temperature, &H1fes},
//             };

//             auto parameters = std::vector
//             {
//                FieldDescriptor{Position, &mesh_fes}
//             };

//             diffusion =
//                std::make_shared<DifferentiableOperator>(solutions, parameters, *mesh);

//             mfem::tuple inputs{Gradient<Temperature>{}, Gradient<Position>{}, Weight{}};
//             mfem::tuple outputs{Gradient<Temperature>{}};

//             auto diffusion_qf = TemperatureDiffusionQFunction<DIMENSION> {};
//             auto derivatives = std::integer_sequence<size_t, Temperature> {};
//             diffusion->AddDomainIntegrator(
//                diffusion_qf, inputs, outputs, ir, derivatives);
//          }
//       }

//       {
//          ParBilinearForm diffusion(&H1fes);
//          auto integ = new DiffusionIntegrator();
//          integ->SetIntegrationRule(ir);
//          diffusion.AddDomainIntegrator(integ);
//          diffusion.Assemble();
//          diffusion.Finalize();
//          K.reset(diffusion.ParallelAssemble());
//       }

//       {
//          ParLinearForm source_lf(&H1fes);
//          source_lf.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
//          source_lf.Assemble();
//          source_tdof = *source_lf.ParallelAssemble();
//       }
//    }

//    void Mult(const Vector &T, Vector &R) const override
//    {
//       // K->Mult(T, R);
//       diffusion->SetParameters({x_gf});
//       diffusion->Mult(T, R);
//       R -= source_tdof;
//       R.SetSubVector(temperature_ess_tdof, 0.0);
//    }

//    Operator &GetGradient(const Vector &T) const override
//    {
//       fd_jacobian.reset(new FDJacobian(*this, T));
//       return *fd_jacobian;
//    }

//    ParGridFunction *x_gf, source_gf, T_gf;

//    std::shared_ptr<HypreParMatrix> K;
//    mutable std::shared_ptr<FDJacobian> fd_jacobian;
//    std::shared_ptr<DifferentiableOperator> diffusion;

//    const Array<int> temperature_ess_attr;
//    Array<int> temperature_ess_tdof;

//    ParFiniteElementSpace &H1fes;
//    const int H1tsize;

//    Vector source_tdof;

//    IntegrationRule ir;
//    Coefficient &temperature_exact_coeff;
//    Coefficient &source_coeff;
// };

// int main(int argc, char* argv[])
// {
//    constexpr int dim = 2;

//    Mpi::Init();

//    const char* device_config = "cpu";
//    const char* mesh_file = "";
//    int polynomial_order_temperature = 2;
//    int refinements = 0;
//    int problem_type = 0;
//    real_t t_final = 0.0;
//    real_t dt = 1e-3;

//    OptionsParser args(argc, argv);
//    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
//    args.AddOption(&polynomial_order_temperature, "-ot", "--order-temperature", "");
//    args.AddOption(&refinements, "-r", "--r", "");
//    args.AddOption(&device_config, "-d", "--device",
//                   "Device configuration string, see Device::Configure().");
//    args.AddOption(&t_final, "-tf", "--tf", "");
//    args.AddOption(&dt, "-dt", "--dt", "");
//    args.ParseCheck();

//    Device device(device_config);
//    if (Mpi::Root() == 0)
//    {
//       device.Print();
//    }

//    out << std::setprecision(8);

//    Mesh mesh_serial = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
//    MFEM_ASSERT(mesh_serial.Dimension() == dim, "incorrect mesh dimension");

//    for (int i = 0; i < refinements; i++)
//    {
//       mesh_serial.UniformRefinement();
//    }
//    ParMesh mesh(MPI_COMM_WORLD, mesh_serial);

//    mesh.EnsureNodes();
//    mesh_serial.Clear();

//    out << "#el: " << mesh.GetNE() << "\n";

//    H1_FECollection temperature_fec(polynomial_order_temperature);

//    ParFiniteElementSpace H1fes(&mesh, &temperature_fec);

//    HYPRE_BigInt global_size_temperature = H1fes.GlobalTrueVSize();
//    if (Mpi::Root())
//    {
//       out << "Number of temperature unknowns: " << global_size_temperature << "\n";
//    }

//    const IntegrationRule &integration_rule =
//       IntRules.Get(H1fes.GetFE(0)->GetGeomType(),
//                    2 * H1fes.GetFE(0)->GetOrder() + 1);

//    Array<int> bdr_attr_is_ess(mesh.bdr_attributes.Max());
//    bdr_attr_is_ess = 1;

//    Vector T(H1fes.GetTrueVSize());

//    ParGridFunction T_gf(&H1fes);

//    auto temperature_exact = [](const Vector &coords, real_t t)
//    {
//       const real_t x = coords(0);
//       const real_t y = coords(1);

//       return pow(cos(y),2) + pow(sin(x),2);
//    };

//    FunctionCoefficient temperature_exact_coeff(temperature_exact);

//    T_gf.ProjectCoefficient(temperature_exact_coeff);
//    T_gf.GetTrueDofs(T);

//    auto source_term = [](const Vector &coords, real_t t)
//    {
//       const real_t x = coords(0);
//       const real_t y = coords(1);

//       return -2*pow(cos(x),2) + 2*pow(cos(y),2) + 2*pow(sin(x),2) - 2*pow(sin(y),2);
//    };

//    FunctionCoefficient source_term_coeff(source_term);

//    HeatOperator heat(H1fes, bdr_attr_is_ess, integration_rule,
//                      temperature_exact_coeff, source_term_coeff);


//    CGSolver krylov(MPI_COMM_WORLD);
//    krylov.SetRelTol(1e-4);
//    krylov.SetMaxIter(1000);
//    krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

//    NewtonSolver newton(MPI_COMM_WORLD);
//    newton.SetOperator(heat);
//    newton.SetSolver(krylov);
//    newton.SetRelTol(1e-8);
//    newton.SetMaxIter(50);
//    newton.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

//    Vector zero;
//    T_gf.ProjectBdrCoefficient(temperature_exact_coeff, bdr_attr_is_ess);
//    T_gf.GetTrueDofs(T);

//    newton.Mult(zero, T);

//    T_gf.SetFromTrueDofs(T);

//    ParGridFunction Terr_gf(&H1fes), Tex_gf(&H1fes);
//    Terr_gf = 0.0;
//    Tex_gf.ProjectCoefficient(temperature_exact_coeff);

//    real_t T_l2err = T_gf.ComputeL2Error(temperature_exact_coeff);
//    if (Mpi::Root())
//    {
//       out << "|T - T_exact|_L2 = " << T_l2err << std::endl;
//    }

//    Tex_gf.ProjectCoefficient(temperature_exact_coeff);
//    for (int i = 0; i < Terr_gf.Size(); i++)
//    {
//       Terr_gf(i) = abs(Tex_gf(i) - T_gf(i));
//    }

//    ParaViewDataCollection dc("dfem_heat", &mesh);
//    dc.SetHighOrderOutput(true);
//    dc.SetLevelsOfDetail(polynomial_order_temperature);
//    dc.RegisterField("temperature", &T_gf);
//    dc.RegisterField("temperature_exact", &Tex_gf);
//    dc.RegisterField("temperature_error", &Terr_gf);
//    dc.SetCycle(0);
//    dc.SetTime(0);
//    dc.Save();
// }
