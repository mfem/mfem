#include "linear_elasticity.hpp"

#include "../chpt/dynamic_checkpointing.hpp"
#include "../chpt/fixed_slot_checkpoint_storage.hpp"

#include <cmath>
#include <iomanip>
#include <cstring>

using namespace mfem;
using namespace std;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/examples/dyn_hex2d_tri.msh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/examples/dyn_hex2d_quad.msh";

struct State
{
   mfem::real_t time = 0.0; //time of the state
   mfem::real_t dt=0.0;
   mfem::real_t obj  = 0.0; //accumulated objective 
   mfem::BlockVector v; //state of the system
};

// Snapshot = *view* (non-owning) used only during Store() packing and Read() callback
struct StateSnapshotView
{
   mfem::real_t time = 0.0;
   mfem::real_t dt=0.0;
   mfem::real_t obj  = 0.0;

   // Points to n*sizeof(real_t) bytes:
   // - during Store(): points to current State::v data (host)
   // - during Read(): points into the storage slot bytes (valid only during callback)
   const unsigned char *v_bytes = nullptr;
};

class StateSnapshotViewPacker
{
public:
   explicit StateSnapshotViewPacker(int n) : n_(n)
   {
      MFEM_VERIFY(n_ > 0, "StateSnapshotViewPacker: n must be > 0.");
   }

   std::size_t SlotBytes() const
   {
      return (std::size_t)(3 + n_) * sizeof(mfem::real_t);
   }

   void Pack(const StateSnapshotView &s, void *dst) const
   {
      MFEM_VERIFY(dst != nullptr, "Pack: dst is null.");
      MFEM_VERIFY(s.v_bytes != nullptr, "Pack: snapshot v_bytes is null.");

      unsigned char *b = static_cast<unsigned char*>(dst);

      std::memcpy(b + 0*sizeof(mfem::real_t), &s.time, sizeof(mfem::real_t));
      std::memcpy(b + 1*sizeof(mfem::real_t), &s.dt,   sizeof(mfem::real_t));
      std::memcpy(b + 2*sizeof(mfem::real_t), &s.obj,  sizeof(mfem::real_t));

      std::memcpy(b + 3*sizeof(mfem::real_t),
                  s.v_bytes,
                  (std::size_t)n_ * sizeof(mfem::real_t));
   }

   // Important: Unpack returns a *view* pointing into src bytes (no allocation).
   void Unpack(const void *src, StateSnapshotView &out) const
   {
      MFEM_VERIFY(src != nullptr, "Unpack: src is null.");

      const unsigned char *b = static_cast<const unsigned char*>(src);

      std::memcpy(&out.time, b + 0*sizeof(mfem::real_t), sizeof(mfem::real_t));
      std::memcpy(&out.dt,   b + 1*sizeof(mfem::real_t), sizeof(mfem::real_t));
      std::memcpy(&out.obj,  b + 2*sizeof(mfem::real_t), sizeof(mfem::real_t));

      out.v_bytes = b + 3*sizeof(mfem::real_t);
   }

   int VectorSize() const { return n_; }

private:
   int n_ = 0;
};

struct AdjState
{
   mfem::real_t time;
   mfem::real_t obj;
   mfem::BlockVector adj;
};


///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoElasticyLambdaCoeff class converts E modulus of elasticity
/// and Poisson's ratio to Lame's lambda coefficient
class IsoElasticyLambdaCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticyLambdaCoeff(mfem::Coefficient *E,
                          mfem::Coefficient *nu):
      E(E), nu(nu) { }

   /// Evaluates the Lame's lambda coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Lambda = [](const real_t E, const real_t nu)
      {
         return E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
      };
      return Lambda(EE, nn);
   }
};



///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoElasticySchearCoeff class converts E modulus of elasticity
/// and Poisson's ratio to Shear coefficient
///
class IsoElasticySchearCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticySchearCoeff(mfem::Coefficient *E_, mfem::Coefficient *nu_):
      E(E_), nu(nu_) { }

   /// Evaluates the shear coefficient coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Schear = [](const real_t E, const real_t nu)
      {
         return E / (2.0 * (1.0 + nu));
      };
      return Schear(EE, nn);
   }
};



int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = MESH_QUAD;
   const char *device_config = "cpu";
   int order = 3;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 1;
   bool paraview = true;
   bool visualization = true;
   int ode_solver_type = 4;
   real_t Tfinal = 0.07;
   real_t dt = 0.005;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&mesh_tri, "-tri", "--triangular", "-no-tri",
                  "--no-triangular", "Enable or not triangular mesh.");
   args.AddOption(&mesh_quad, "-quad", "--quadrilateral", "-no-quad",
                  "--no-quadrilateral", "Enable or not quadrilateral mesh.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of parallel mesh refinement levels.");
   args.AddOption(&paraview, "-pa", "--paraview", "-no-pa",
                  "--no-paraview", "Enable or not Paraview output.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or not visualization.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());

   args.AddOption(&Tfinal, "-T",   
                    "--tfinal",   "Terminate when accumulated time reaches Tfinal.");
   args.AddOption(&dt,    "-dt", "--dt",  "Time step.");   

   args.ParseCheck();

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }
   
   // Read the (serial) mesh from the given mesh file on all processors.  We
   // can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   // and volume meshes with the same code.
   Mesh mesh(mesh_tri ? MESH_TRI : mesh_quad ? MESH_QUAD : mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   
   // Refine the serial mesh on all processors to increase the resolution. In
   // this example we do 'ref_levels' of uniform refinement. We choose
   // 'ref_levels' to be the largest number that gives a final mesh with no
   // more than 1000 elements.
   {
      const int ref_levels =
         (int)floor(log(1000. / mesh.GetNE()) / log(2.) / dim);
      for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   }
   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh.GetNE() << std::endl;
   }

   // Define a parallel mesh by a partitioning of the serial mesh. Refine
   // this mesh further in parallel to increase the resolution. Once the
   // parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < par_ref_levels; l++) { pmesh.UniformRefinement(); }   

   // Allocate the time dependent linear elasticity operator
   LinearElasticityTimeDependentOperator lin_elasticity_op(pmesh, order);

   // Set the material coefficients
   ConstantCoefficient rho_coef(0.5); // density coefficient for topology optimization
   
   // Set elasticity coefficients for material 1 and 2
   ConstantCoefficient E1(0.1);
   ConstantCoefficient E2(1.0);
   ConstantCoefficient nu1(0.3);
   ConstantCoefficient nu2(0.3);

   //Lame coefficients
   IsoElasticyLambdaCoeff lambda1(&E1, &nu1);
   IsoElasticySchearCoeff mu1(&E1, &nu1);
   IsoElasticyLambdaCoeff lambda2(&E2, &nu2);
   IsoElasticySchearCoeff mu2(&E2, &nu2);

   // Set density coefficients for material 1 and 2
   ConstantCoefficient dens1_coef(0.5);
   ConstantCoefficient dens2_coef(1.0);

   // Set damping coefficients
   ProductCoefficient cm1_coef(0.02, dens1_coef);
   ProductCoefficient cm2_coef(0.02, dens2_coef);

   ProductCoefficient cl1_coef(0.01, lambda1);
   ProductCoefficient cmu1_coef(0.01, mu1);
   ProductCoefficient cl2_coef(0.01, lambda2);
   ProductCoefficient cmu2_coef(0.01, mu2);

   lin_elasticity_op.SetElasticityCoefficients(lambda1, mu1, lambda2, mu2);
   
   lin_elasticity_op.SetDensityMaterialCoefficients(dens1_coef, dens2_coef);
   
   lin_elasticity_op.SetDampingMaterialCoefficients(cm1_coef, cm2_coef);
   lin_elasticity_op.SetDampingMaterialCoefficients(cl1_coef, cmu1_coef,
                                                    cl2_coef, cmu2_coef);

   lin_elasticity_op.SetDensity(rho_coef);
   
   //set bottom bdr to zero (both the velocities and the displacements)
   lin_elasticity_op.SetZeroBdr(1);

   lin_elasticity_op.SetVolForce(1.0 /*period*/, 1.0 /*amplitude*/, 0.2 /*radius*/,
                                 0.0 /*x center*/ , 0.0 /*y center*/, 0.0 /*z center*/,
                                 5.0 /* train length*/, 2.5 /*center of the train*/, 2.0 /*power*/);

   lin_elasticity_op.AssembleExplicit();

   // test mult explicit
   {
      BlockVector tst; tst.Update(lin_elasticity_op.GetTrueBlockOffsets()); 
      tst=0.0; //tst.Randomize();
      tst.UseDevice(true); tst.Read();

      BlockVector grd; grd.Update(lin_elasticity_op.GetTrueBlockOffsets());
      grd=0.0;
      lin_elasticity_op.Mult(tst,grd);
      lin_elasticity_op.GetVelocity().SetFromTrueDofs(grd.GetBlock(1));
   }   

   std::shared_ptr<ExampleObjectiveIntegrand> eobj=
      std::make_shared<ExampleObjectiveIntegrand>(lin_elasticity_op.GetFESpace());
   //set the objective for the integration process                                                   

   ParaViewDataCollection paraview_dc("isoel", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("disp", &(lin_elasticity_op.GetDisplacement()));
   paraview_dc.RegisterField("velo", &(lin_elasticity_op.GetVelocity()));

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   lin_elasticity_op.SetTime(0.0);
   ode_solver->Init(lin_elasticity_op);

   //Check the Jacobian of the time dependent operator
   //⟨J(x) v,w⟩=⟨v,J(x)^T w⟩
   {
      auto CheckJacobianMultTranspose = [&](int ntrials = 5)
      {
         BlockVector x, v, w, xp, xm, fp, fm, jv_fd, jtw;
         for (BlockVector *bv : {&x, &v, &w, &xp, &xm, &fp, &fm, &jv_fd, &jtw})
         {
            bv->Update(lin_elasticity_op.GetTrueBlockOffsets());
            *bv = 0.0;
         }

         const real_t t0  = 0.0137;
         const real_t eps = 1.0;

         double worst_rel_err = 0.0;

         for (int k = 0; k < ntrials; k++)
         {
            x.Randomize();
            v.Randomize();
            w.Randomize();

            xp = x;
            xm = x;
            xp.Add(+eps, v);
            xm.Add(-eps, v);

            lin_elasticity_op.SetTime(t0);
            lin_elasticity_op.Mult(xp, fp);

            lin_elasticity_op.SetTime(t0);
            lin_elasticity_op.Mult(xm, fm);

            jv_fd = fp;
            jv_fd -= fm;
            jv_fd *= (0.5 / eps);   // central-difference J(x) v

            lin_elasticity_op.SetTime(t0);
            lin_elasticity_op.JacobianMultTranspose(x, w, jtw);

            //const double lhs = GlobalDot(jv_fd, w); // <J(x) v, w>
            //const double rhs = GlobalDot(v, jtw);   // <v, J(x)^T w>

            const double lhs = mfem::InnerProduct(pmesh.GetComm(), jv_fd, w); // <J(x) v, w>
            const double rhs = mfem::InnerProduct(pmesh.GetComm(), v, jtw); // <v, J(x)^T w>

            double scale = 1.0;
            if (std::abs(lhs) > scale) { scale = std::abs(lhs); }
            if (std::abs(rhs) > scale) { scale = std::abs(rhs); }

            const double rel_err = std::abs(lhs - rhs) / scale;
            if (rel_err > worst_rel_err) { worst_rel_err = rel_err; }

            if (Mpi::Root())
            {
               mfem::out << "JacobianMultTranspose trial " << k
                   << ": lhs=" << std::setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel_err << '\n';
            }
         }

         if (Mpi::Root())
         {
            mfem::out << "Worst relative error = " << worst_rel_err << '\n';
         }

         MFEM_VERIFY(worst_rel_err < 1e-6,
               "JacobianMultTranspose check failed.");
      };

      // CheckJacobianMultTranspose();

   }

   //check the ode solver adjoint
   //⟨DFh​(xn​) v,w⟩=⟨v,DFh​(xn​)^T w⟩.
   {
      auto CheckStepAdjointStep = [&](int ntrials = 5)
      {
         auto probe_solver = ODESolver::Select(ode_solver_type);
         MFEM_VERIFY(probe_solver->SupportsAdjoint(ODESolver::AdjointMode::Discrete),
                     "Selected ODE solver does not support discrete adjoint stepping.");

         const real_t t0  = 0.0137;
         const real_t h   = dt;
         const real_t eps = 1.0;

         BlockVector x0, v, w, xp, xm, x_plus, x_minus, jv_fd, lam0;
         for (BlockVector *bv : {&x0, &v, &w, &xp, &xm,
                              &x_plus, &x_minus, &jv_fd, &lam0})
         {
            bv->Update(lin_elasticity_op.GetTrueBlockOffsets());
            *bv = 0.0;
         }

         double worst_rel_err = 0.0;

         for (int k = 0; k < ntrials; k++)
         {
            x0.Randomize();
            v.Randomize();
            w.Randomize();

            // Finite-difference action of the one-step map Phi_h
            xp = x0;
            xm = x0;
            xp.Add(+eps, v);
            xm.Add(-eps, v);

            x_plus  = xp;
            x_minus = xm;

            auto step_plus = ODESolver::Select(ode_solver_type);
            real_t tp = t0;
            real_t hp = h;
            step_plus->Init(lin_elasticity_op);
            step_plus->Step(x_plus, tp, hp);

            auto step_minus = ODESolver::Select(ode_solver_type);
            real_t tm = t0;
            real_t hm = h;
            step_minus->Init(lin_elasticity_op);
            step_minus->Step(x_minus, tm, hm);

            MFEM_VERIFY(std::abs(hp - hm) < 1e-14,
                  "Forward plus/minus steps returned different dt.");

            jv_fd = x_plus;
            jv_fd -= x_minus;
            jv_fd *= (0.5 / eps);   // centered FD: D Phi_h(x0) v

            // Discrete adjoint action of the same one-step map
            auto adj_solver = ODESolver::Select(ode_solver_type);
            adj_solver->Init(lin_elasticity_op);
            adj_solver->EnableAdjoint(ODESolver::AdjointMode::Discrete);
            adj_solver->SetSolution(x0, t0);

            lam0 = w;
            real_t ta = t0 + hp;
            real_t ha = hp;
            adj_solver->AdjointStep(lam0, ta, ha);

            //const double lhs = GlobalDot(jv_fd, w); // <D Phi_h(x0) v, w>
            //const double rhs = GlobalDot(v, lam0);  // <v, D Phi_h(x0)^T w>

            const double lhs = mfem::InnerProduct(pmesh.GetComm(), jv_fd, w); // <D Phi_h(x0) v, w>
            const double rhs = mfem::InnerProduct(pmesh.GetComm(), v, lam0);  // <v, D Phi_h(x0)^T w>


            double scale = 1.0;
            if (std::abs(lhs) > scale) { scale = std::abs(lhs); }
            if (std::abs(rhs) > scale) { scale = std::abs(rhs); }

            const double rel_err = std::abs(lhs - rhs) / scale;
            worst_rel_err = std::max(worst_rel_err, rel_err);

            if (Mpi::Root())
            {
               mfem::out << "Step/AdjointStep trial " << k
                   << ": lhs=" << std::setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel_err << '\n';
            }
         }

         if (Mpi::Root())
         {
            mfem::out << "Worst relative error for Step/AdjointStep = "
                << worst_rel_err << '\n';
         }

         MFEM_VERIFY(worst_rel_err < 1e-6,
               "Step/AdjointStep adjoint-identity check failed.");
      };

      // CheckStepAdjointStep();

   }

   /// Multiple time steps
   {
      auto GlobalDot = [&](const Vector &a, const Vector &b)
      {
         double local  = static_cast<double>(a * b);
         double global = 0.0;
         MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
         return global;
      };

      auto MakeBlockVector = [&]()
      {
         BlockVector x;
         x.Update(lin_elasticity_op.GetTrueBlockOffsets());
         x = 0.0;
         return x;
      };

      auto RolloutNSteps = [&](const BlockVector &x_init,
                         int nsteps,
                         real_t t_init,
                         real_t h_desired,
                         std::vector<BlockVector> *states_out,
                         std::vector<real_t> *times_out)
      {
         auto solver = ODESolver::Select(ode_solver_type);
         solver->Init(lin_elasticity_op);

         BlockVector x = MakeBlockVector();
         x = x_init;
         real_t t = t_init;

         if (states_out)
         {
            states_out->resize(nsteps + 1);
            for (int i = 0; i <= nsteps; i++)
            {
               (*states_out)[i].Update(lin_elasticity_op.GetTrueBlockOffsets());
               (*states_out)[i] = 0.0;
            }
            (*states_out)[0] = x;
         }

         if (times_out)
         {
            times_out->assign(nsteps + 1, 0.0);
            (*times_out)[0] = t;
         }

         for (int i = 0; i < nsteps; i++)
         {
            real_t h = h_desired;
            solver->Step(x, t, h);

            if (states_out) { (*states_out)[i + 1] = x; }
            if (times_out)  { (*times_out)[i + 1] = t; }
         }

         return x;
      };

      auto CheckNStepMapAdjoint = [&](int nsteps, int ntrials = 5)
      {
         auto probe_solver = ODESolver::Select(ode_solver_type);
         MFEM_VERIFY(probe_solver->SupportsAdjoint(ODESolver::AdjointMode::Discrete),
                  "Selected ODE solver does not support discrete adjoint stepping.");

         const real_t t0  = 0.0137;
         const real_t h   = dt;
         const real_t eps = 1.0;
         const double time_tol = 1e-13;

         double worst_rel_err = 0.0;

         for (int k = 0; k < ntrials; k++)
         {
            BlockVector x0      = MakeBlockVector();
            BlockVector v       = MakeBlockVector();
            BlockVector w       = MakeBlockVector();
            BlockVector x_plus  = MakeBlockVector();
            BlockVector x_minus = MakeBlockVector();
            BlockVector jv_fd   = MakeBlockVector();
            BlockVector lambda  = MakeBlockVector();

            x0.Randomize();
            v.Randomize();
            w.Randomize();

            std::vector<BlockVector> states;
            std::vector<real_t> times, times_plus, times_minus;

            // Base trajectory: save every primal state x_i and time t_i
            RolloutNSteps(x0, nsteps, t0, h, &states, &times);

            // Centered finite-difference action of the n-step map Phi_n
            BlockVector xp = MakeBlockVector();
            BlockVector xm = MakeBlockVector();
            xp = x0;
            xm = x0;
            xp.Add(+eps, v);
            xm.Add(-eps, v);

            x_plus  = RolloutNSteps(xp, nsteps, t0, h, nullptr, &times_plus);
            x_minus = RolloutNSteps(xm, nsteps, t0, h, nullptr, &times_minus);

            // Guard against adaptive / perturbed step schedules.
            MFEM_VERIFY((int)times_plus.size()  == nsteps + 1, "Unexpected times_plus size.");
            MFEM_VERIFY((int)times_minus.size() == nsteps + 1, "Unexpected times_minus size.");
            for (int i = 0; i <= nsteps; i++)
            {
               MFEM_VERIFY(std::abs(times_plus[i]  - times[i]) < time_tol,
                     "Perturbed + trajectory changed the step times.");
               MFEM_VERIFY(std::abs(times_minus[i] - times[i]) < time_tol,
                     "Perturbed - trajectory changed the step times.");
            }

            jv_fd = x_plus;
            jv_fd -= x_minus;
            jv_fd *= (0.5 / eps);  // centered FD for D Phi_n(x0) v

            // Reverse discrete adjoint over the saved primal states
            auto adj_solver = ODESolver::Select(ode_solver_type);
            adj_solver->Init(lin_elasticity_op);
            adj_solver->EnableAdjoint(ODESolver::AdjointMode::Discrete);

            lambda = w;
            for (int i = nsteps - 1; i >= 0; i--)
            {
               real_t ti = times[i + 1];
               real_t hi = times[i + 1] - times[i];

               adj_solver->SetSolution(states[i], times[i]);
               adj_solver->AdjointStep(lambda, ti, hi);
            }

            const double lhs = GlobalDot(jv_fd, w);   // <D Phi_n(x0) v, w>
            const double rhs = GlobalDot(v, lambda);  // <v, D Phi_n(x0)^T w>

            double scale = 1.0;
            if (std::abs(lhs) > scale) { scale = std::abs(lhs); }
            if (std::abs(rhs) > scale) { scale = std::abs(rhs); }

            const double rel_err = std::abs(lhs - rhs) / scale;
            worst_rel_err = std::max(worst_rel_err, rel_err);

            if (Mpi::Root())
            {
               mfem::out << "n-step Step/AdjointStep trial " << k
                   << " (nsteps=" << nsteps << ")"
                   << ": lhs=" << std::setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel_err << '\n';
            }
         }

         if (Mpi::Root())
         {
            mfem::out << "Worst relative error for n-step Step/AdjointStep = "
                << worst_rel_err << '\n';
         }

         MFEM_VERIFY(worst_rel_err < 1e-6,
                  "n-step Step/AdjointStep adjoint-identity check failed.");
      };

      // CheckNStepMapAdjoint(10);

   }

   // using checkpointing
   {
      auto GlobalDot = [&](const Vector &a, const Vector &b)
      {
         double local  = static_cast<double>(a * b);
         double global = 0.0;
         MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());
         return global;
      };

      auto MakeBlockVector = [&]()
      {
         BlockVector x;
         x.Update(lin_elasticity_op.GetTrueBlockOffsets());
         x = 0.0;
         return x;
      };

      auto RolloutNSteps = [&](const BlockVector &x_init,
                         int nsteps,
                         real_t t_init,
                         real_t h_desired,
                         std::vector<real_t> *times_out)
      {
         auto solver = ODESolver::Select(ode_solver_type);
         solver->Init(lin_elasticity_op);

         BlockVector x = MakeBlockVector();
         x = x_init;
         real_t t = t_init;

         if (times_out)
         {
            times_out->assign(nsteps + 1, 0.0);
            (*times_out)[0] = t;
         }

         for (int i = 0; i < nsteps; i++)
         {
            real_t h = h_desired;
            solver->Step(x, t, h);
            if (times_out) { (*times_out)[i + 1] = t; }
         }

         return x;
      };

      auto CheckNStepMapAdjointWithCheckpointing =
         [&](int nsteps, int ncheckpoints, int ntrials = 5)
      {
         MFEM_VERIFY(nsteps > 0, "nsteps must be > 0.");
         MFEM_VERIFY(ncheckpoints > 0, "ncheckpoints must be > 0.");

         auto probe_solver = ODESolver::Select(ode_solver_type);
         MFEM_VERIFY(
            probe_solver->SupportsAdjoint(ODESolver::AdjointMode::Discrete),
               "Selected ODE solver does not support discrete adjoint stepping.");

         const real_t t0  = 0.0137;
         const real_t h   = dt;
         const real_t eps = 1.0;
         const double time_tol = 1e-13;

         const int n = lin_elasticity_op.GetState().Size();
         StateSnapshotViewPacker packer(n);

         using Storage =
            mfem::FixedSlotMemoryCheckpointStorage<StateSnapshotView,
                                             StateSnapshotViewPacker>;
         using Checkpointing = mfem::DynamicCheckpointing<StateSnapshotView, Storage>;
         using Step = typename Checkpointing::Step;

         double worst_rel_err = 0.0;

         for (int k = 0; k < ntrials; k++)
         {
            BlockVector x0      = MakeBlockVector();
            BlockVector v       = MakeBlockVector();
            BlockVector w       = MakeBlockVector();
            BlockVector x_plus  = MakeBlockVector();
            BlockVector x_minus = MakeBlockVector();
            BlockVector jv_fd   = MakeBlockVector();

            x0.Randomize();
            v.Randomize();
            w.Randomize();

            Storage storage(ncheckpoints, packer);
            Checkpointing ckpt(ncheckpoints, storage);

            auto primal_solver = ODESolver::Select(ode_solver_type);
            auto adj_solver    = ODESolver::Select(ode_solver_type);
            adj_solver->Init(lin_elasticity_op);
            adj_solver->EnableAdjoint(ODESolver::AdjointMode::Discrete);

            bool need_reinit_primal = true;
            std::vector<real_t> times(nsteps + 1, 0.0);
            std::vector<real_t> step_span(nsteps, 0.0);

            auto make_snapshot = [&](const State &u) -> StateSnapshotView
            {
               MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed.");

               StateSnapshotView snap;
               snap.time = u.time;
               snap.dt   = u.dt;
               snap.obj  = u.obj;
               snap.v_bytes = reinterpret_cast<const unsigned char *>(u.v.HostRead());
               return snap;
            };

            auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
            {
               u_out.time = snap.time;
               u_out.dt   = snap.dt;
               u_out.obj  = snap.obj;

               MFEM_VERIFY(u_out.v.Size() == n,
                        "restore_snapshot: BlockVector size mismatch.");

               mfem::real_t *vh = u_out.v.HostWrite();
               std::memcpy(vh, snap.v_bytes, (std::size_t)n * sizeof(mfem::real_t));
               u_out.v.Read(true);

               // We are about to restart a stepping sequence from a restored state.
               need_reinit_primal = true;
            };

            auto primal_step = [&](State &u_st, Step i)
            {
               if (need_reinit_primal)
               {
                  primal_solver->Init(lin_elasticity_op);
                  need_reinit_primal = false;
               }

               real_t t_step = u_st.time;
               real_t h_step = h;
               primal_solver->Step(u_st.v, t_step, h_step);

               u_st.time = t_step;
               u_st.dt   = h_step;
               u_st.obj  = 0.0;
            };

            auto adjoint_step = [&](AdjState &adj_st, const State &u_st, Step i)
            {
               const real_t ti = u_st.time;
               const real_t hi = step_span[static_cast<int>(i)];

               real_t t_adj = ti + hi;
               real_t h_adj = hi;
               adj_solver->SetSolution(u_st.v, ti);
               adj_solver->AdjointStep(adj_st.adj, t_adj, h_adj);

               adj_st.time = ti;
               adj_st.obj  = 0.0;
            };

            State u;
            u.v.Update(lin_elasticity_op.GetTrueBlockOffsets());
            u.v = x0;
            u.time = t0;
            u.dt   = h;
            u.obj  = 0.0;

            times[0] = t0;
            for (int i = 0; i < nsteps; i++)
            {
               ckpt.ForwardStep(static_cast<Step>(i), u,
                           primal_step, make_snapshot);
               times[i + 1]   = u.time;
               step_span[i]   = times[i + 1] - times[i];
            }

            BlockVector xp = MakeBlockVector();
            BlockVector xm = MakeBlockVector();
            xp = x0;
            xm = x0;
            xp.Add(+eps, v);
            xm.Add(-eps, v);

            std::vector<real_t> times_plus, times_minus;
            x_plus  = RolloutNSteps(xp, nsteps, t0, h, &times_plus);
            x_minus = RolloutNSteps(xm, nsteps, t0, h, &times_minus);

            MFEM_VERIFY((int)times_plus.size()  == nsteps + 1,
                  "Unexpected times_plus size.");
            MFEM_VERIFY((int)times_minus.size() == nsteps + 1,
                  "Unexpected times_minus size.");
            for (int i = 0; i <= nsteps; i++)
            {
               MFEM_VERIFY(std::abs(times_plus[i]  - times[i]) < time_tol,
                     "Perturbed + trajectory changed the step times.");
               MFEM_VERIFY(std::abs(times_minus[i] - times[i]) < time_tol,
                     "Perturbed - trajectory changed the step times.");
            }

            jv_fd = x_plus;
            jv_fd -= x_minus;
            jv_fd *= (0.5 / eps);

            AdjState adj_st;
            adj_st.adj.Update(lin_elasticity_op.GetTrueBlockOffsets());
            adj_st.adj = w;
            adj_st.time = times[nsteps];
            adj_st.obj  = 0.0;

            State u_wrk;
            u_wrk.v.Update(lin_elasticity_op.GetTrueBlockOffsets());
            u_wrk.v = 0.0;
            u_wrk.time = 0.0;
            u_wrk.dt   = 0.0;
            u_wrk.obj  = 0.0;

            for (Step i = static_cast<Step>(nsteps) - 1; ; --i)
            {
               ckpt.BackwardStep(i, adj_st, u_wrk,
                           primal_step, adjoint_step,
                           make_snapshot, restore_snapshot);
               if (i == 0) { break; }
            }

            const double lhs = GlobalDot(jv_fd, w);      // <D Phi_n(x0) v, w>
            const double rhs = GlobalDot(v, adj_st.adj); // <v, D Phi_n(x0)^T w>

            double scale = 1.0;
            if (std::abs(lhs) > scale) { scale = std::abs(lhs); }
            if (std::abs(rhs) > scale) { scale = std::abs(rhs); }

            const double rel_err = std::abs(lhs - rhs) / scale;
            worst_rel_err = std::max(worst_rel_err, rel_err);

            if (Mpi::Root())
            {
               mfem::out << "Checkpointed n-step adjoint test, trial " << k
                   << " (nsteps=" << nsteps
                   << ", ncheckpoints=" << ncheckpoints << ")"
                   << ": lhs=" << std::setprecision(16) << lhs
                   << ", rhs=" << rhs
                   << ", rel_err=" << rel_err << '\n';
            }
         }

         if (Mpi::Root())
         {
            mfem::out << "Worst relative error for checkpointed n-step adjoint test = "
                << worst_rel_err << '\n';
         }

         MFEM_VERIFY(worst_rel_err < 1e-6,
               "Checkpointed n-step adjoint-identity check failed.");
      };

      //test gradients of an objective
      auto CheckNStepObjAdjointWithCheckpointing =
         [&](int nsteps, int ncheckpoints, int ntrials = 5)
      {
         MFEM_VERIFY(nsteps > 0, "nsteps must be > 0.");
         MFEM_VERIFY(ncheckpoints > 0, "ncheckpoints must be > 0.");

         auto probe_solver = ODESolver::Select(ode_solver_type);
         MFEM_VERIFY(
            probe_solver->SupportsAdjoint(ODESolver::AdjointMode::Discrete),
               "Selected ODE solver does not support discrete adjoint stepping.");

         const real_t t0  = 0.0;
         const real_t h   = dt;
         const real_t eps = 1.0;
         const double time_tol = 1e-13;

         const int n = lin_elasticity_op.GetState().Size();
         StateSnapshotViewPacker packer(n);

         using Storage =
            mfem::FixedSlotMemoryCheckpointStorage<StateSnapshotView,
                                             StateSnapshotViewPacker>;
         using Checkpointing = mfem::DynamicCheckpointing<StateSnapshotView, Storage>;
         using Step = typename Checkpointing::Step;

         Storage storage(ncheckpoints, packer);
         Checkpointing ckpt(ncheckpoints, storage);

         auto primal_solver = ODESolver::Select(ode_solver_type);
         auto adj_solver    = ODESolver::Select(ode_solver_type);
         adj_solver->Init(lin_elasticity_op);
         adj_solver->EnableAdjoint(ODESolver::AdjointMode::Discrete);         

         bool need_reinit_primal = true;//true if the primal ode solver must be reinitialized 
         std::vector<real_t> times(nsteps + 1, 0.0); //stores the times instances for the checkpoints
         std::vector<real_t> step_span(nsteps, 0.0); //stores the time steps

         auto make_snapshot = [&](const State &u) -> StateSnapshotView
         {
            MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed.");

            StateSnapshotView snap;
            snap.time = u.time;
            snap.dt   = u.dt;
            snap.obj  = u.obj;
            snap.v_bytes = reinterpret_cast<const unsigned char *>(u.v.HostRead());
            return snap;
         };

         auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
         {
            u_out.time = snap.time;
            u_out.dt   = snap.dt;
            u_out.obj  = snap.obj;

            MFEM_VERIFY(u_out.v.Size() == n,
                     "restore_snapshot: BlockVector size mismatch.");

            mfem::real_t *vh = u_out.v.HostWrite();
            std::memcpy(vh, snap.v_bytes, (std::size_t)n * sizeof(mfem::real_t));
            u_out.v.Read(true);

            // We are about to restart a stepping sequence from a restored state.
            need_reinit_primal = true;
         };

         auto primal_step = [&](State &u_st, Step i)
         {
            if (need_reinit_primal)
            {
               primal_solver->Init(lin_elasticity_op);
               need_reinit_primal = false;
            }

            real_t t_step = u_st.time;
            real_t h_step = h;
            primal_solver->Step(u_st.v, t_step, h_step);

            u_st.time = t_step;
            u_st.dt   = h_step;
            u_st.obj  = eobj->EvalScalar(u_st.v);
         };

         auto adjoint_step = [&](AdjState &adj_st, const State &u_st, Step i)
         {
            const real_t ti = u_st.time;
            const real_t hi = step_span[static_cast<int>(i)];
            const real_t obj = u_st.obj;

            real_t t_adj = ti + hi;
            real_t h_adj = hi;
            adj_solver->SetSolution(u_st.v, ti);
            adj_solver->AdjointStep(adj_st.adj, t_adj, h_adj);

            adj_st.time = ti;
            adj_st.obj  = obj;
         };

         //forward computations
         State u;
         u.v.Update(lin_elasticity_op.GetTrueBlockOffsets());
         u.v = 0.0;
         u.time = t0;
         u.dt   = h;
         u.obj  = 0.0;

         times[0] = t0;
         for (int i = 0; i < nsteps; i++)
         {
            ckpt.ForwardStep(static_cast<Step>(i), u,
                           primal_step, make_snapshot);
            times[i + 1]   = u.time;
            step_span[i]   = times[i + 1] - times[i];
         }

         AdjState adj_st;
         adj_st.adj.Update(lin_elasticity_op.GetTrueBlockOffsets());
         adj_st.adj = 0.0;
         eobj->EvalGradient(u.v,adj_st.adj);
         adj_st.time = times[nsteps];
         adj_st.obj  = u.obj;

         State u_wrk;
         u_wrk.v.Update(lin_elasticity_op.GetTrueBlockOffsets());
         u_wrk.v = 0.0;
         u_wrk.time = 0.0;
         u_wrk.dt   = 0.0;
         u_wrk.obj  = 0.0;

         for (Step i = static_cast<Step>(nsteps) - 1; ; --i)
         {
            ckpt.BackwardStep(i, adj_st, u_wrk,
                           primal_step, adjoint_step,
                           make_snapshot, restore_snapshot);
            if (i == 0) { break; }
         }

         real_t sca=1.0;
         BlockVector x_plus  = MakeBlockVector();
         BlockVector x_minus = MakeBlockVector();
         BlockVector xp = MakeBlockVector();
         BlockVector xm = MakeBlockVector();
         BlockVector pp = MakeBlockVector(); pp.Randomize();
         const real_t pgrad = mfem::InnerProduct(pmesh.GetComm(), pp, adj_st.adj);
         for(int sc=0;sc<ntrials; sc++)
         {
            xp=0.0; xp.Add(+sca,pp);
            xm=0.0; xm.Add(-sca,pp);
            std::vector<real_t> times_plus, times_minus;
            x_plus  = RolloutNSteps(xp, nsteps, t0, h, &times_plus);
            x_minus = RolloutNSteps(xm, nsteps, t0, h, &times_minus);

            real_t objp=eobj->EvalScalar(x_plus);
            real_t objm=eobj->EvalScalar(x_minus);


            const real_t fd_grad = (objp-objm)/(2*sca);
            const real_t abs_err = std::abs(fd_grad-pgrad);
            if (Mpi::Root())
            {
               mfem::out << "Checkpointed n-step obj test, trial " << sc
                   << " (nsteps=" << nsteps
                   << ", ncheckpoints=" << ncheckpoints << ")"
                   << ", scale="<< sca << " "
                   << ", finite difference ="<< fd_grad <<" "
                   << ", projected gradient ="<< pgrad << " "
                   << ", abs_err=" << abs_err << '\n';
            }

            sca=sca/10;

         }
      };


      //CheckNStepMapAdjointWithCheckpointing(30, 8);
      CheckNStepObjAdjointWithCheckpointing(50, 8);

   }

   //checkpointing with ParaView recording of the forward and the adjoint
   {

      BlockVector x; x.Update(lin_elasticity_op.GetTrueBlockOffsets());  x=0.0;
      BlockVector g; g.Update(lin_elasticity_op.GetTrueBlockOffsets());  g=0.0;

      auto AdjointWithCheckpointing =
         [&](int nsteps, int ncheckpoints)
      {
         MFEM_VERIFY(nsteps > 0, "nsteps must be > 0.");
         MFEM_VERIFY(ncheckpoints > 0, "ncheckpoints must be > 0.");

         auto probe_solver = ODESolver::Select(ode_solver_type);
         MFEM_VERIFY(
            probe_solver->SupportsAdjoint(ODESolver::AdjointMode::Discrete),
               "Selected ODE solver does not support discrete adjoint stepping.");

         const real_t t0  = 0.0;
         const real_t h   = dt;
         const real_t eps = 1.0;
         const double time_tol = 1e-13;

         const int n = lin_elasticity_op.GetState().Size();
         StateSnapshotViewPacker packer(n);

         using Storage =
            mfem::FixedSlotMemoryCheckpointStorage<StateSnapshotView,
                                             StateSnapshotViewPacker>;
         using Checkpointing = mfem::DynamicCheckpointing<StateSnapshotView, Storage>;
         using Step = typename Checkpointing::Step;

         Storage storage(ncheckpoints, packer);
         Checkpointing ckpt(ncheckpoints, storage);

         auto primal_solver = ODESolver::Select(ode_solver_type);
         auto adj_solver    = ODESolver::Select(ode_solver_type);
         adj_solver->Init(lin_elasticity_op);
         adj_solver->EnableAdjoint(ODESolver::AdjointMode::Discrete);         

         bool need_reinit_primal = true;//true if the primal ode solver must be reinitialized 
         std::vector<real_t> times(nsteps + 1, 0.0); //stores the times instances for the checkpoints
         std::vector<real_t> step_span(nsteps, 0.0); //stores the time steps

         auto make_snapshot = [&](const State &u) -> StateSnapshotView
         {
            MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed.");

            StateSnapshotView snap;
            snap.time = u.time;
            snap.dt   = u.dt;
            snap.obj  = u.obj;
            snap.v_bytes = reinterpret_cast<const unsigned char *>(u.v.HostRead());
            return snap;
         };

         auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
         {
            u_out.time = snap.time;
            u_out.dt   = snap.dt;
            u_out.obj  = snap.obj;

            MFEM_VERIFY(u_out.v.Size() == n,
                     "restore_snapshot: BlockVector size mismatch.");

            mfem::real_t *vh = u_out.v.HostWrite();
            std::memcpy(vh, snap.v_bytes, (std::size_t)n * sizeof(mfem::real_t));
            u_out.v.Read(true);

            // We are about to restart a stepping sequence from a restored state.
            need_reinit_primal = true;
         };

         auto primal_step = [&](State &u_st, Step i)
         {
            if (need_reinit_primal)
            {
               primal_solver->Init(lin_elasticity_op);
               need_reinit_primal = false;
            }

            real_t t_step = u_st.time;
            real_t h_step = h;
            primal_solver->Step(u_st.v, t_step, h_step);

            u_st.time = t_step;
            u_st.dt   = h_step;
            u_st.obj  = eobj->EvalScalar(u_st.v);
         };

         auto adjoint_step = [&](AdjState &adj_st, const State &u_st, Step i)
         {
            const real_t ti = u_st.time;
            const real_t hi = step_span[static_cast<int>(i)];
            const real_t obj = u_st.obj;

            real_t t_adj = ti + hi;
            real_t h_adj = hi;
            adj_solver->SetSolution(u_st.v, ti);
            adj_solver->AdjointStep(adj_st.adj, t_adj, h_adj);

            adj_st.time = ti;
            adj_st.obj  = obj;
         };

         ParaViewDataCollection paraview_dc("frw", &pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.RegisterField("disp", &(lin_elasticity_op.GetDisplacement()));
         paraview_dc.RegisterField("velo", &(lin_elasticity_op.GetVelocity()));

         //forward computations
         State u;
         u.v.Update(lin_elasticity_op.GetTrueBlockOffsets());
         u.v = x;
         u.time = t0;
         u.dt   = h;
         u.obj  = 0.0;

         times[0] = t0;
         for (int i = 0; i < nsteps; i++)
         {
            if((i%5)==0){
               paraview_dc.SetCycle(i);
               paraview_dc.SetTime(times[i]);
               lin_elasticity_op.GetVelocity().SetFromTrueDofs(u.v.GetBlock(1));
               lin_elasticity_op.GetDisplacement().SetFromTrueDofs(u.v.GetBlock(0));
               paraview_dc.Save();

               if(Mpi::Root()){
                  // mfem::out<<"FWD Time="<<times[i]<<" dt="<< h <<'\n';
               }
            }

            

            ckpt.ForwardStep(static_cast<Step>(i), u,
                           primal_step, make_snapshot);
            times[i + 1]   = u.time;
            step_span[i]   = times[i + 1] - times[i];
         }

         paraview_dc.SetCycle(nsteps);
         paraview_dc.SetTime(times[nsteps]);
         lin_elasticity_op.GetVelocity().SetFromTrueDofs(u.v.GetBlock(1));
         lin_elasticity_op.GetDisplacement().SetFromTrueDofs(u.v.GetBlock(0));
         paraview_dc.Save();

         ParGridFunction adisp(lin_elasticity_op.GetDisplacement());
         ParGridFunction avelo(lin_elasticity_op.GetVelocity());
         ParaViewDataCollection paraview_ac("adj", &pmesh);
         paraview_ac.SetPrefixPath("ParaView");
         paraview_ac.SetLevelsOfDetail(order);
         paraview_ac.SetDataFormat(VTKFormat::BINARY);
         paraview_ac.SetHighOrderOutput(true);
         paraview_ac.RegisterField("adisp", &(adisp));
         paraview_ac.RegisterField("avelo", &(avelo));

         real_t o=eobj->EvalScalar(u.v);
         if(Mpi::Root()){
            mfem::out<<" obj="<<o<<'\n';
         }

         AdjState adj_st;
         adj_st.adj.Update(lin_elasticity_op.GetTrueBlockOffsets());
         adj_st.adj = 0.0;
         eobj->EvalGradient(u.v,adj_st.adj);
         adj_st.time = times[nsteps];
         adj_st.obj  = u.obj;

         State u_wrk;
         u_wrk.v.Update(lin_elasticity_op.GetTrueBlockOffsets());
         u_wrk.v = 0.0;
         u_wrk.time = 0.0;
         u_wrk.dt   = 0.0;
         u_wrk.obj  = 0.0;

         Vector tmpv(u_wrk.v.GetBlock(0)); tmpv=0.0;

         paraview_ac.SetCycle(nsteps);
         paraview_ac.SetTime(times[nsteps]);
         lin_elasticity_op.MultInvMass(adj_st.adj.GetBlock(0),tmpv);
         adisp.SetFromTrueDofs(tmpv);
         lin_elasticity_op.MultInvMass(adj_st.adj.GetBlock(1),tmpv);
         avelo.SetFromTrueDofs(tmpv);
         paraview_ac.Save();


         for (Step i = static_cast<Step>(nsteps) - 1; ; --i)
         {
            ckpt.BackwardStep(i, adj_st, u_wrk,
                           primal_step, adjoint_step,
                           make_snapshot, restore_snapshot);
            if((i%5)==0){ 
               paraview_ac.SetCycle(i);
               paraview_ac.SetTime(times[i]);
               lin_elasticity_op.MultInvMass(adj_st.adj.GetBlock(0),tmpv);
               adisp.SetFromTrueDofs(tmpv);
               lin_elasticity_op.MultInvMass(adj_st.adj.GetBlock(1),tmpv);
               avelo.SetFromTrueDofs(tmpv);
               paraview_ac.Save();
               
               if(Mpi::Root()){
                  //mfem::out<<"ADJ Time="<<times[i]<<" dt="<<step_span[i]<<'\n';
               }
            }

            if (i == 0) { break; }
         }

         lin_elasticity_op.MultInvMass(adj_st.adj.GetBlock(0),tmpv);
         g.GetBlock(0)=tmpv;
         lin_elasticity_op.MultInvMass(adj_st.adj.GetBlock(1),tmpv);
         g.GetBlock(1)=tmpv;
      };  

      for(int bi=0;bi<10;bi++){

         AdjointWithCheckpointing(1000,100);
         x.Add(-0.01,g);
      }

   }

   return EXIT_SUCCESS;
}