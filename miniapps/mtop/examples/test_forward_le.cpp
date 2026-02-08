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
   mfem::real_t time = 0.0;
   mfem::real_t obj  = 0.0;
   mfem::Vector v;
};

// Snapshot = *view* (non-owning) used only during Store() packing and Read() callback
struct StateSnapshotView
{
   mfem::real_t time = 0.0;
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
      return (std::size_t)(2 + n_) * sizeof(mfem::real_t);
   }

   void Pack(const StateSnapshotView &s, void *dst) const
   {
      MFEM_VERIFY(dst != nullptr, "Pack: dst is null.");
      MFEM_VERIFY(s.v_bytes != nullptr, "Pack: snapshot v_bytes is null.");

      unsigned char *b = static_cast<unsigned char*>(dst);

      std::memcpy(b, &s.time, sizeof(mfem::real_t));
      std::memcpy(b + sizeof(mfem::real_t), &s.obj, sizeof(mfem::real_t));

      std::memcpy(b + 2*sizeof(mfem::real_t),
                  s.v_bytes,
                  (std::size_t)n_ * sizeof(mfem::real_t));
   }

   // Important: Unpack returns a *view* pointing into src bytes (no allocation).
   void Unpack(const void *src, StateSnapshotView &out) const
   {
      MFEM_VERIFY(src != nullptr, "Unpack: src is null.");

      const unsigned char *b = static_cast<const unsigned char*>(src);

      std::memcpy(&out.time, b, sizeof(mfem::real_t));
      std::memcpy(&out.obj,  b + sizeof(mfem::real_t), sizeof(mfem::real_t));

      out.v_bytes = b + 2*sizeof(mfem::real_t);
   }

   int Size() const { return n_; }

private:
   int n_ = 0;
};

struct AdjState
{
   mfem::real_t time;
   mfem::real_t obj;
   mfem::Vector adj;
   mfem::Vector grd;
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

   ParaViewDataCollection paraview_dc("isoel", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("disp", &(lin_elasticity_op.GetDisplacement()));
   paraview_dc.RegisterField("velo", &(lin_elasticity_op.GetVelocity()));


   /*
   {
      int s=10;

      StateSnapshotViewPacker packer(lin_elasticity_op.GetState().Size());

      // storage stores StateSnapshotView snapshots using fixed-size slots
      using Storage=mfem::FixedSlotMemoryCheckpointStorage<StateSnapshotView, StateSnapshotViewPacker>;
      Storage storage(s, packer);

      auto make_snapshot = [&](const State &u) -> StateSnapshotView
      {
         MFEM_VERIFY(u.v.Size() == n, "make_snapshot: State.v size changed!");

         // Ensure host access if MFEM device is in use:
         const mfem::real_t *vh = u.v.HostRead();

         StateSnapshotView snap;
         snap.time = u.time;
         snap.obj  = u.obj;
         snap.v_bytes = reinterpret_cast<const unsigned char*>(vh);
         return snap;
      };

      auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
      {
         u_out.time = snap.time;
         u_out.obj  = snap.obj;

         if (u_out.v.Size() != n) { u_out.v.SetSize(n); }

         mfem::real_t *vh = u_out.v.HostWrite();
         std::memcpy(vh,
                     snap.v_bytes,
                     (std::size_t)n * sizeof(mfem::real_t));
      };


      using Step = mfem::DynamicCheckpointing<StateSnapshotView, Storage>::Step;
      State u;
      u.v.SetSize(lin_elasticity_op.GetState().Size());




      auto primal_step = [&](StateCheckPoint &u, Step i)
      {
         if (Mpi::Root()){
            std::cout<<"Primal step:  time= "<<u.time<<" obj= "<<u.obj;
         }
         const double dt = 0.01;
         u.obj=dt*i;
         u.time=dt*i;
         u.v=(mfem::real_t)i;

         if (Mpi::Root()){
            std::cout<<" out Step: "<<i<<" time="<<u.time<<" obj="<<u.obj<<std::endl;
         }
      };

      auto adjoint_step = [&](AdjState &lambda, const StateCheckPoint &u_i, Step i)
      {
         const double dt = 0.01;
         MFEM_ASSERT(lambda.adj.Size() == u_i.state.Size(), "lambda and u_i size mismatch.");
         if (Mpi::Root()){
            std::cout<<"Adj step: time= "<<u_i.time<<" obj= "<<u_i.obj;
            std::cout<<" adj time= "<<lambda.time<<" adj obj="<<lambda.obj<<std::endl;
         }

         lambda.obj=-u_i.obj;
         lambda.time=lambda.time-dt;

      };


      // Initial condition 
      StateCheckPoint spt; spt.obj=-1.0; spt.time=-1.0; spt.state=(lin_elasticity_op.GetState());

      Step i=0;
      mfem::real_t t=0.0;
      mfem::real_t dt=0.01;
      while(t<0.2)
      {
         ckpt.ForwardStep(i, spt, primal_step, make_snapshot);
         t=t+dt;
         ++i;
      }

      const Step m = i;
      if (Mpi::Root()){
         std::cout<<" Total number of steps="<<m<<std::endl;
      }

      AdjState ast; ast.obj=1.0; ast.time=spt.time; 
      ast.adj=(lin_elasticity_op.GetState());
      ast.grd=(lin_elasticity_op.GetState());

      for (Step j = m - 1; j >= 0; --j)
      {
         if (Mpi::Root()){
            std::cout<<" Outer steps="<<j<<std::endl;
         }

         ckpt.BackwardStep(j, ast, spt, primal_step, adjoint_step, make_snapshot, restore_snapshot);
         
         if (j == 0) { break; }
      }
      


   }
      */




   //test time integration
   {
      real_t t = 0.0;

      BlockVector tsol; tsol.Update(lin_elasticity_op.GetTrueBlockOffsets());
      //set initial conditions at time t
      tsol=0.0;

      // 4. Define the ODE solver used for time integration. Several explicit
      //    Runge-Kutta methods are available.
      unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

      lin_elasticity_op.SetTime(t);
      ode_solver->Init(lin_elasticity_op);

      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      lin_elasticity_op.GetVelocity().SetFromTrueDofs(tsol.GetBlock(1));
      lin_elasticity_op.GetDisplacement().SetFromTrueDofs(tsol.GetBlock(0));
      paraview_dc.Save();

      real_t dt_real = 0.005;
      //ode_solver->Run(tsol, t, dt_real, 1.0);

      for(int i=0;i<6000; i++){
         ode_solver->Step(tsol, t, dt_real);

         if (Mpi::Root())
         {
            std::cout << "t: " << t << std::endl; 
         }

         if((i%5)==0){
            paraview_dc.SetCycle(i+1);
            paraview_dc.SetTime(t);
            lin_elasticity_op.GetVelocity().SetFromTrueDofs(tsol.GetBlock(1));
            lin_elasticity_op.GetDisplacement().SetFromTrueDofs(tsol.GetBlock(0));
            paraview_dc.Save();
         }
      }

   }

   /*
   if (paraview)
   {
      ParaViewDataCollection paraview_dc("isoel", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      //paraview_dc.RegisterField("disp", &sol);
      paraview_dc.RegisterField("disp", &(lin_elasticity_op.GetDisplacement()));
      paraview_dc.RegisterField("velo", &(lin_elasticity_op.GetVelocity()));
      paraview_dc.Save();
   }
   */

   return EXIT_SUCCESS;
}