#include "mtop_solvers.hpp"

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
   ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   //set mass and diffusion coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient diff_coef(0.5*0.5);

   // Define a parallel finite element space hierarchy on the parallel mesh.
   // Here we use continuous Lagrange finite elements. We start with order 1
   // on the coarse level and geometrically refine the spaces by the specified
   // amount. Afterwards, we increase the order of the finite elements by a
   // factor of 2 for each additional level.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *coarse_fespace = new ParFiniteElementSpace(pmesh, fec);

   Array<FiniteElementCollection*> collections;
   collections.Append(fec);
   ParFiniteElementSpaceHierarchy* fespaces = new ParFiniteElementSpaceHierarchy(
      pmesh, coarse_fespace, true, true); // take ownership of mesh and coarse space
   for (int l = 0; l < par_ref_levels; l++)
   {
      fespaces->AddUniformlyRefinedLevel(1, Ordering::byVDIM);
   }

   // set the prolongation operators for the multigrid
   mfem::Array<mfem::Operator *> prolongations;
   const int nlevels = fespaces->GetNumLevels();
   prolongations.SetSize(nlevels - 1);
   for (int level = 0; level < nlevels - 1; ++level)
   {
      prolongations[level] = fespaces->GetProlongationAtLevel(level);
   }

   //set the operators and the smoothers for the multigrid
   mfem::Array<mfem::Operator *> operators;
   mfem::Array<Solver *> smoothers;
   operators.SetSize(nlevels);
   smoothers.SetSize(nlevels);
   for (int level = 0; level < nlevels; ++level)
   {
      // mass matrix on each level
      ParFiniteElementSpace &fespace = fespaces->GetFESpaceAtLevel(level);
      unique_ptr<ParBilinearForm> bf(new ParBilinearForm(&fespace));
      bf->AddDomainIntegrator(new MassIntegrator(one));
      bf->AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
      bf->Assemble();
      bf->Finalize();
      HypreParMatrix *mat= bf->ParallelAssemble();
      operators[level] = mat;

      // Jacobi smoother on each level
      Vector diag(fespace.GetTrueVSize());
      mat->GetDiag(diag);
      OperatorJacobiSmoother *jac =
         new OperatorJacobiSmoother(1.0 /*weight*/);
      jac->iterative_mode = false;
      jac->Setup(diag);
      smoothers[level]=jac;
   }


   HYPRE_BigInt size = fespaces->GetFinestFESpace().GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   int myrank;
   MPI_Comm_rank(pmesh->GetComm(), &myrank);

   //Set up the parallel random linear form
   unique_ptr<ParLinearForm> b(new ParLinearForm(&fespaces->GetFinestFESpace()));
   LinearFormIntegrator *rint = 
      new WhiteGaussianNoiseDomainLFIntegrator(pmesh->GetComm(), 7497+17*myrank);
   b->AddDomainIntegrator(rint);
   b->Assemble();
   unique_ptr<HypreParVector> r(b->ParallelAssemble());
   r->UseDevice(true);


   // Solver the standard problem with unit mass and diffusion matrices
   unique_ptr<ParBilinearForm> bf(new ParBilinearForm(&fespaces->GetFinestFESpace()));

   bf->AddDomainIntegrator(new MassIntegrator(one));
   bf->AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
   bf->Assemble();   
   bf->Finalize();
   unique_ptr<HypreParMatrix> A(bf->ParallelAssemble());

   // Solve the linear system A X = B
   HypreBoomerAMG amg;
   amg.SetPrintLevel(0);
   amg.SetOperator(*A);
   unique_ptr<CGSolver> solver(new CGSolver(pmesh->GetComm()));
   solver->SetPrintLevel(0);
   solver->SetOperator(*A);   
   solver->SetRelTol(1e-12);
   solver->SetMaxIter(500);
   solver->SetAbsTol(1e-14);
   solver->SetPreconditioner(amg);

   ParGridFunction x(&fespaces->GetFinestFESpace());
   x = 0.0; 
   x.UseDevice(true);
   x.SetTrueVector(); x.GetTrueVector().UseDevice(true);
   solver->Mult(*r, x.GetTrueVector());
   x.SetFromTrueVector();

   //Save the solution
   {
      ParaViewDataCollection paraview_dc("rnd",fespaces->GetFinestFESpace().GetParMesh());
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("rnd", &x);
      paraview_dc.Save();
   }




   //for(int i = 0; i < prolongations.Size(); i++){delete prolongations[i];}
   for(int i = 0; i < operators.Size(); i++){delete operators[i];}
   for(int i = 0; i < smoothers.Size(); i++){delete smoothers[i];}

   delete fespaces;
   for(int i = 0; i < collections.Size(); i++){delete collections[i];}
   
   return EXIT_SUCCESS;
}