#include "anderson_fp_solver.hpp"

#include "mtop_solvers.hpp"

using namespace std;
using namespace mfem;

constexpr auto MESH_TRI = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_tri.mesh";
constexpr auto MESH_QUAD = MFEM_SOURCE_DIR "/miniapps/mtop/sq_2D_9_quad.mesh";


/// Fixed-point operator for testing Anderson acceleration
class FPOperator : public mfem::Operator
{
public:
   FPOperator(const HypreParMatrix& A_, const Vector& rhs_) :
                        mfem::Operator(A_.Height()), A(&A_), rhs(rhs_)  
   {
       Vector diag(A_.Width());
       tmp.SetSize(A_.Width()); 
       tmp=1.0;
       //A_.AbsMult(tmp, diag);
       A->GetDiag(diag);
       smoother = std::make_unique<OperatorJacobiSmoother>(diag, Array<int>(), 1.0);

   }

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      A->Mult(x, tmp);
      tmp.Neg();
      tmp.Add(1.0, rhs);
      smoother->Mult(tmp, y); y.Add(1.0, x);
   }

private:
   const HypreParMatrix *A;
   const Vector& rhs;
   mutable Vector tmp;
   std::unique_ptr<OperatorJacobiSmoother> smoother;

};


class DensCoeff : public mfem::Coefficient
{
private:
    real_t l;
public:
    DensCoeff(real_t d=1.0) : l(d) {}

    virtual real_t Eval(mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip)
    {
        Vector x;
        T.Transform(ip, x);
        real_t r = x.Norml2();
        r=sin(M_PI*r/l);
        if(r>0.5)
            r=1.0;
        else
            r=0.0;
        return r;
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
   int order = 2;
   bool pa = false;
   bool dfem = false;
   bool mesh_tri = false;
   bool mesh_quad = false;
   int par_ref_levels = 1;
   bool paraview = false;
   bool visualization = true;
   int m=1;
   real_t beta=1.0;

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
    args.AddOption(&m, "-ma", "--m-accel",
                  "Anderson acceleration parameter m.");
    args.AddOption(&beta, "-beta", "--beta",
                  "Anderson acceleration relaxation parameter beta.");
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

    // Define a finite element space on the mesh. Here we use continuous
    // Lagrange finite elements of the specified order.
    H1_FECollection fec(order,dim);
    ParFiniteElementSpace fespace(&pmesh, &fec, 1);

    // Define the solution vector x as a finite element grid function
    ParGridFunction x(&fespace); x=0.0;

    std::unique_ptr<HypreParMatrix> A;
    Vector rhs; rhs.SetSize(fespace.GetTrueVSize());
    {
        // Set up the linear system Ax=b
        ParBilinearForm a(&fespace);
        ConstantCoefficient diff_coeff(0.01);
        a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
        a.AddDomainIntegrator(new MassIntegrator());
        a.Assemble();
        a.Finalize();   

        A.reset(a.ParallelAssemble());

        ParLinearForm b(&fespace);
        DensCoeff dens_coeff(1.0);
        b.AddDomainIntegrator(new DomainLFIntegrator(dens_coeff));
        b.Assemble();
        b.ParallelAssemble(rhs);
    }


    FPOperator G(*A, rhs);

    Vector res;
    Vector x_new;
    x_new.SetSize(fespace.GetTrueVSize());
    res.SetSize(fespace.GetTrueVSize());

    for(int i=0; i<50; i++)
    {
        G.Mult(x.GetTrueVector(), x_new);
        add(-1.0, x_new, 1.0, x.GetTrueVector(), res);
        real_t res_norm = InnerProduct(pmesh.GetComm(), res, res);
        real_t x_norm = InnerProduct(pmesh.GetComm(), x_new, x_new);
        if(0==pmesh.GetMyRank()){    
            std::cout << "Iter " << i << " : Residual norm = " << res_norm << "  Solution norm = " << x_norm << std::endl;    
        }
        //x.SetFromTrueDofs(x_new); x.SetTrueVector();
        x.GetTrueVector()=x_new;
    }

    AndersonFixedPointSolverParGramDeviceIP aa(pmesh.GetComm(),/*m=*/m);
    aa.SetOperator(G);

    // Typical knobs:
    aa.SetMaxIter(200);
    aa.SetRelTol(1e-10);
    aa.SetAbsTol(0.0);
    aa.SetBeta(beta);

    // Default recommendation knobs:
    aa.SetRegularizationRel(1e-12); // try 1e-10 .. 1e-6 if coefficients blow up
    aa.SetRcond(1e-12);             // try 1e-10 if history gets near-dependent

    aa.SetPrintLevel(1);

    x.GetTrueVector()=0.0;
    aa.Mult(x.GetTrueVector(), x_new);
    real_t x_norm = InnerProduct(pmesh.GetComm(), x_new, x_new);
    if(0==pmesh.GetMyRank()){    
            std::cout  <<  "  Solution norm = " << x_norm << std::endl;    
    }

    x.SetFromTrueDofs(x_new); x.SetTrueVector();

    if (paraview)
    {
        ParaViewDataCollection paraview_dc("anderson", &pmesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.RegisterField("filt", &x);
        paraview_dc.Save();
    }

   return EXIT_SUCCESS;

}