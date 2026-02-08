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
   mfem::Vector v; //state of the system
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
   real_t Tfinal = 1.0;
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


   //Forward computations
   {
      int s=10; //number of snapshots to be stored by the checkpointing process

      //define the packer object
      int n=lin_elasticity_op.GetState().Size();
      StateSnapshotViewPacker packer(n);

      // storage stores StateSnapshotView snapshots using fixed-size slots
      using Storage=mfem::FixedSlotMemoryCheckpointStorage<StateSnapshotView, StateSnapshotViewPacker>;
      Storage storage(s, packer);

      // Snapshot type is StateSnapshotView
      using Checkpointing = mfem::DynamicCheckpointing<StateSnapshotView, Storage>;
      Checkpointing ckpt(s, storage);

      // Returns view of the State and avoids data transfer
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

       //Transfers data from the snaphot view to the State u_out.
      auto restore_snapshot = [&](const StateSnapshotView &snap, State &u_out)
      {
         u_out.time = snap.time;
         u_out.obj  = snap.obj;

         if (u_out.v.Size() != n) { u_out.v.SetSize(n); }

         mfem::real_t *vh = u_out.v.HostWrite();
         std::memcpy(vh,
                     snap.v_bytes,
                     (std::size_t)n * sizeof(mfem::real_t));

         //make sure that the date is on the device            
         u_out.v.Read(true);
      };

      using Step = mfem::DynamicCheckpointing<StateSnapshotView, Storage>::Step;


      //execute one integration step
      auto primal_step = [&](State &u_st, Step i)
      {
         //begin with curent state u_st
         real_t t=u_st.time;
         real_t ldt=dt;
         real_t obj=u_st.obj;

         //make sure the integration does not overjump Tfinal
         if((t+ldt)>Tfinal){
            ldt=Tfinal-t;
         }

         //advance u_st
         ode_solver->Step(u_st.v,t,ldt);
         //TO-DO update objective 

         //return updated u_st
         u_st.dt=t-u_st.time;
         u_st.time=t;
         u_st.obj=obj;

         if (Mpi::Root())
         {
            mfem::out<<"t: "<<u_st.time<<" dt="<<u_st.dt<<"\n";
         }
      };

      State u;
      u.v.SetSize(lin_elasticity_op.GetState().Size());
      u.obj=0.0;
      u.time=0.0;
      u.dt=0.0;

      //set initial state to 0
      u.v=0.0;

      // Forward sweep (unknown number of steps) 
      real_t t = 0.0;
      Step i = 0;

      while(t<Tfinal)
      {
         ckpt.ForwardStep(i,u, primal_step, make_snapshot);
         t=u.time;
         ++i;
      }

   }



   return EXIT_SUCCESS;
}