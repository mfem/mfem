#include "mfem.hpp"


class InterpCoeff: public mfem::Coefficient
{
public:
    InterpCoeff(std::shared_ptr<mfem::Coefficient> c1_,
                std::shared_ptr<mfem::Coefficient> c2_,
                std::shared_ptr<mfem::Coefficient> rho_)
                : c1(c1_), c2(c2_), rho(rho_)
                {

                }

    virtual mfem::real_t 	Eval (mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {
        mfem::real_t v1=c1->Eval(T,ip);
        mfem::real_t v2=c2->Eval(T,ip);
        mfem::real_t r=rho->Eval(T,ip);

        return v1*r+v2*(1.0-r);
    }

private:
    std::shared_ptr<mfem::Coefficient> c1;
    std::shared_ptr<mfem::Coefficient> c2;
    std::shared_ptr<mfem::Coefficient> rho;
};


namespace mfem{

class PAExplicitDiffusionOperator : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &pfes;
   Array<int> ess_tdof_list;

   // Partially assembled bilinear forms (no assembled sparse matrices)
   ParBilinearForm k_form; // diffusion stiffness

   // Minv(i) = 1 / diag(M)(i) on TRUE dofs
   Vector Minv;
   mutable Vector tmpv;

public:
   /// ess_bdr: boundary attribute marker (size = pmesh->bdr_attributes.Max()), 1 -> essential (Dirichlet)
   /// kappa: diffusivity coefficient
   PAExplicitDiffusionOperator(ParFiniteElementSpace &pfes_,
                               const Array<int> &ess_tdof_list_,
                               Coefficient &kappa)
      : TimeDependentOperator(pfes_.GetTrueVSize(), 0.0, TimeDependentOperator::EXPLICIT),
        pfes(pfes_),
        k_form(&pfes_),
        Minv(height),
        tmpv(height)
   {
      // Essential TRUE dofs
      // pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      ess_tdof_list=ess_tdof_list_;

      // --- Mass operator (PA) ---
      auto mifi=new MassIntegrator();
      int order=pfes.GetOrder(0);
      IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
      const IntegrationRule &ir_ni = gll_rules.Get(pfes.GetParMesh()->GetTypicalElementGeometry(),
                                                        2 * order - 1);
      mifi->SetIntRule(&ir_ni);

      ParBilinearForm m_form(&pfes); // mass
      m_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m_form.AddDomainIntegrator(mifi);
      m_form.Assemble();

      // Diagonal on TRUE dofs (works with PA in ParBilinearForm)
      Vector Mdiag(height);
      m_form.AssembleDiagonal(Mdiag);

      Minv = Mdiag;
      Minv.Reciprocal();

      // --- Diffusion operator (PA) ---
      k_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k_form.AddDomainIntegrator(new DiffusionIntegrator(kappa));
      k_form.Assemble();
   }

   /// Compute du_dt = -Minv .* (K u)
   void Mult(const Vector &u, Vector &du_dt) const override
   {
      du_dt = 0.0;

      // du_dt += -1 * (P^T K_local P) u  (true-dof action; PA supported)
      k_form.TrueAddMult(u, du_dt, -1.0);

      // Apply diagonal inverse mass
      du_dt *= Minv;

      // Strongly enforce Dirichlet: derivative is zero on essential tdofs
      if (ess_tdof_list.Size())
      {
         du_dt.SetSubVector(ess_tdof_list, 0.0);
      }
   }

   const Array<int> &GetEssentialTrueDofs() const { return ess_tdof_list; }

   /// y = (df/dx(x,t))^T * w
   void JacobianMultTranspose(const Vector &x,
                                      const Vector &w,
                                      Vector &y) const override
   {
      y=0.0;
      tmpv.Set(1.0,w);
      if (ess_tdof_list.Size()){
        tmpv.SetSubVector(ess_tdof_list, 0.0);}
      // Apply diagonal inverse mass
      tmpv*=Minv;
      // y += -1 * (P^T K_local P) u  (true-dof action; PA supported)
      k_form.TrueAddMult(tmpv, y, -1.0); 
   }
};    

};

class DiffusionTDOP:public mfem::TimeDependentOperator
{
public:
    DiffusionTDOP(mfem::ParMesh* mesh_, int vorder =1, int dorder=1):order(vorder)
    {
        mesh=mesh_;
        int dim=mesh->Dimension();
        fec.reset(new mfem::H1_FECollection(vorder,dim));
        fes.reset(new mfem::ParFiniteElementSpace(mesh,fec.get()));

        dfec.reset(new mfem::H1_FECollection(dorder,dim));
        dfes.reset(new mfem::ParFiniteElementSpace(mesh,dfec.get()));

        gfdens.SetSpace(dfes.get());
        

        //the state vector of the TDOP consists of [solution, density, obj]
        siz_u=fes->GetTrueVSize();  //solution
        siz_d=dfes->GetTrueVSize(); //design
        siz_f=1; //objective;

        block_true_offsets.SetSize(4);
        block_true_offsets[0] = 0;
        block_true_offsets[1] = siz_u;
        block_true_offsets[2] = siz_d;
        block_true_offsets[3] = siz_f;
        block_true_offsets.PartialSum();

        //set the width and the height of the operator
        this->width=  block_true_offsets[3];
        this->height= block_true_offsets[3];
    }

    const mfem::ParFiniteElementSpace* GetStateFES(){ return fes.get(); }
    const mfem::ParFiniteElementSpace* GetDesignFEM(){ return dfes.get(); }

    virtual 
    ~DiffusionTDOP()
    {

    }

    void SetLoad(int ind, std::shared_ptr<mfem::Coefficient> l)
    {
        load[ind]=l;
    }

    void SetZeroBC(int ind)
    {
        zero_bc.insert(ind);
    }

    // free the allocated bilinear forms, matrices and solvers
    void Reset()
    {

    }

    /// Asembles the load, the mass and the stiffness matrices for
    /// a give true vector tvdens with densities varying from 0 to 1
    /// and a given time t.
    void Assemble(const mfem::Vector& tvdens, mfem::real_t t)
    {
        
        gfdens.SetFromTrueDofs(tvdens);
        gfc.reset(new mfem::GridFunctionCoefficient(&gfdens));

        if(nullptr==kbf.get())
        {
            // set constrained dofs
            SetEssTDofs(ess_tdofv);
            //allocate the system matrices and solvers
            //allocate diagonal mass matrix and its inverse
            {
                mfem::IntegrationRules gll_rules(0, mfem::Quadrature1D::GaussLobatto);
                const mfem::IntegrationRule &ir_ni = gll_rules.Get(mesh->GetTypicalElementGeometry(),
                                                        2 * order - 1);

                mfem::ParBilinearForm mform(fes.get());
                mform.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
                cs.reset(new InterpCoeff(cs1,cs2,gfc));
                auto *mblfi = new mfem::MassIntegrator(*cs);
                mblfi->SetIntRule(&ir_ni);
                mform.AddDomainIntegrator(mblfi);

                mfem::Vector diag(fes->GetTrueVSize());
                mform.AssembleDiagonal(diag);
                minv=diag;
                minv.Reciprocal();
                // Ensure essential dofs stay fixed: zero inverse there (extra safety)
                if (ess_tdofv.Size())
                {
                    minv.SetSubVector(ess_tdofv, 0.0);
                }
            }

            //allocate the stiffness matrix
            {
                kap.reset(new InterpCoeff(kap1,kap2,gfc));
                kbf.reset(new mfem::ParBilinearForm(fes.get()));
                kbf->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
                kbf->AddDomainIntegrator(new mfem::DiffusionIntegrator(*kap));
                kbf->Assemble();
            }
        }

        //allocate the RHS


    }

    virtual
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override
    {
        mfem::BlockVector bx(const_cast<mfem::Vector&>(x), block_true_offsets);
        mfem::BlockVector by(y, block_true_offsets);
        //compute the time derivatives
        //compute the time derivative of the objective 
    }


    void SetEssTDofs(mfem::Array<int>& ess_dofs)
    {
        ess_dofs.DeleteAll();

        mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr=0;
        for (auto it=zero_bc.begin(); it!=zero_bc.end(); ++it)
        {
            int attr = *it;
            ess_bdr[attr-1] = 1;
        }

        fes->GetEssentialTrueDofs(ess_bdr,ess_dofs);
    }

    /// Sets Dirichlet dofs to zero
    void SetEssTDofs(mfem::Vector& v) const
    {
        mfem::Array<int> loc_tdofs;
        mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr=0;
        for (auto it=zero_bc.begin(); it!=zero_bc.end(); ++it)
        {
            int attr = *it;
            ess_bdr[attr-1] = 1;
        }

        fes->GetEssentialTrueDofs(ess_bdr, loc_tdofs);

        for (int j=0; j<loc_tdofs.Size(); j++)
        {
            v[loc_tdofs[j]]=0.0;
        }
    }

private:

int order;
mfem::ParMesh* mesh;
std::unique_ptr<mfem::FiniteElementCollection> fec;
std::unique_ptr<mfem::ParFiniteElementSpace> fes;

std::unique_ptr<mfem::FiniteElementCollection> dfec;
std::unique_ptr<mfem::ParFiniteElementSpace> dfes;

std::unique_ptr<mfem::HypreBoomerAMG> prec;
std::unique_ptr<mfem::CGSolver> ls;

std::shared_ptr<mfem::Coefficient> kap1, kap2;
std::shared_ptr<mfem::Coefficient> cs1, cs2;
std::shared_ptr<InterpCoeff> cs,kap;
std::shared_ptr<mfem::GridFunctionCoefficient> gfc;
mfem::ParGridFunction gfdens;

std::map<int, std::shared_ptr<mfem::Coefficient>> load;
std::set<int> zero_bc;
// holds the constrained DOFs
mfem::Array<int> ess_tdofv;

mfem::Array<int> block_true_offsets;
int siz_u;
int siz_d;
int siz_f;

std::unique_ptr<mfem::ParBilinearForm> kbf;
mfem::Vector minv;

};

using namespace std;
using namespace mfem;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_tri.mesh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";

int main(int argc, char *argv[])
{
    // Initialize MPI and HYPRE.
   Mpi::Init();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = MESH_QUAD;
   const char *device_config = "cpu";
   int order = 2;
   bool pa = false;
   bool dfem = false;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 1;
   bool paraview = false;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&dfem, "-dfem", "--dFEM", "-no-dfem", "--no-dFEM",
                  "Enable or not dFEM.");
   args.AddOption(&mesh_tri, "-tri", "--triangular", "-no-tri",
                  "--no-triangular", "Enable or not triangular mesh.");
   args.AddOption(&mesh_quad, "-quad", "--quadrilateral", "-no-quad",
                  "--no-quadrilateral", "Enable or not quadrilateral mesh.");
   args.AddOption(&par_ref_levels, "-prl", "--par-ref-levels",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or not Paraview visualization");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();
   MFEM_VERIFY(!(pa && dfem), "pa and dfem cannot be both set");

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

   mfem::H1_FECollection fec(order, pmesh.Dimension());
   mfem::ParFiniteElementSpace pfes(&pmesh, &fec);

   mfem::Array<int> ess_tdof_list;
   {
       // Dirichlet boundary marker (example: all boundary attributes)
       mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
       ess_bdr = 0;
       ess_bdr[9]=1.0;
       pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }


   mfem::ConstantCoefficient kappa(1.0);

   mfem::PAExplicitDiffusionOperator oper(pfes, ess_tdof_list, kappa);

   // Initial condition as a ParGridFunction
   mfem::ParGridFunction u_gf(&pfes);
   u_gf = 0.0; 
   // replace with projection
   {
        mfem::FunctionCoefficient fc([](const Vector &x) -> real_t
        {   return std::sin(4.0*x[0]*M_PI)*std::sin(4.0*x[1]*M_PI); });
        //project
        u_gf.ProjectCoefficient(fc);
   }


   // True dof vector state
   mfem::Vector u;
   u_gf.GetTrueDofs(u);

   // Enforce homogeneous Dirichlet initially
   u.SetSubVector(oper.GetEssentialTrueDofs(), 0.0);
   u_gf.SetFromTrueDofs(u);

   // set paraview output
   ParaViewDataCollection paraview_dc("tdiff", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("temp", &u_gf);

   // Pick an explicit solver
   mfem::RK4Solver ode;
   ode.Init(oper);

   double t = 0.0;
   double dt = 1e-5; // diffusion CFL ~ O(h^2); must be small

   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(t);
   paraview_dc.Save();


   /*
   for (int ti = 0; ti < 10; ti++)
   {
      ode.Step(u, t, dt);

      // Keep Dirichlet dofs pinned (avoid drift)
      u.SetSubVector(oper.GetEssentialTrueDofs(), 0.0);

      u_gf.SetFromTrueDofs(u);
      paraview_dc.SetCycle(ti+1);
      paraview_dc.SetTime(t);
      paraview_dc.Save();
   }
   */


   //test the RK4 adjoint
   {
        Vector x;
        {
            mfem::FunctionCoefficient fc([](const Vector &x) -> real_t
            {   return std::sin(4.0*x[0]*M_PI)*std::sin(4.0*x[1]*M_PI); });
            //project
            u_gf.ProjectCoefficient(fc);
        }
        u_gf.GetTrueDofs(x);
        // Enforce homogeneous Dirichlet initially
        x.SetSubVector(oper.GetEssentialTrueDofs(), 0.0);
        u_gf.SetFromTrueDofs(x);
        u.Set(1.0,x);

        Vector vone(x); vone=1.0;

        //compute one step
        t=0.0;
        ode.Step(x, t, dt);
        //compute objective 
        real_t obj=mfem::InnerProduct(pmesh.GetComm(),x,vone);

        //compute the adjoint
        mfem::Vector lam(u.Size()); lam=0.0;
        {
            ode.EnableAdjoint(mfem::ODESolver::AdjointMode::Discrete);
            ode.SetSolution(u,0.0); //set the solution at t=0.0;
            lam.Set(1.0,vone);
            ode.AdjointStep(lam,t,dt); //on exit lam is the adjoint at t-dt
        }

        mfem::Vector rnd(u.Size()); rnd.Randomize();
        rnd.SetSubVector(oper.GetEssentialTrueDofs(), 0.0);

        real_t iprod=mfem::InnerProduct(pmesh.GetComm(),rnd,lam);

        real_t sca=1.0;
        for(int i=0;i<20;i++){
            x.Set(sca,rnd);
            x.Add(1.0,u);
            t=0.0;
            ode.Step(x, t, dt);
            real_t pobj=mfem::InnerProduct(pmesh.GetComm(),x,vone);

            x.Set(-sca,rnd);
            x.Add(1.0,u);
            t=0.0;
            ode.Step(x, t, dt);
            real_t mobj=mfem::InnerProduct(pmesh.GetComm(),x,vone);

            if(mfem::Mpi::Root())
            {
                std::cout<<" scale="<<sca<<" "<<" o="<<obj<<" p="<<pobj
                            <<" do="<<(pobj-mobj)/(2.0*sca)
                            <<" oo="<<(obj-mobj)/sca
                            <<" to="<<iprod<<std::endl;
            }

            sca=sca/10.0;
        }

   }



   return EXIT_SUCCESS;

}