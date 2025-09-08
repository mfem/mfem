#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ScaledSolver: public Solver
{
private:
    real_t sca;
    Solver* sol;
    bool own_solver;
public:
    ScaledSolver(real_t a, Solver* sol_, bool own_=true):Solver(*sol_)
    {
        own_solver=own_;
        sol=sol_;
        sca=a;
    }

    virtual
    ~ScaledSolver(){
        if(own_solver){
            delete sol;
        }
    }

    void Mult (const Vector &x, Vector &y) const override
    {
        sol->Mult(x,y);
        y*=sca;
    }

    void MultTranspose(const Vector &x, Vector &y) const override
       { sol->MultTranspose(x, y); y *=sca; }


    virtual void SetOperator(const Operator &op) override
    {
        sol->SetOperator(op);
    }

};

class FractionalMultigrid : public GeometricMultigrid
{
private:
    ConstantCoefficient coeff;
    ConstantCoefficient one;
    HypreBoomerAMG* amg;
    real_t s=0.0;
    real_t sigma=1.0;

public:

    FractionalMultigrid(ParFiniteElementSpaceHierarchy& fespaces,
                       Array<int>& ess_bdr)
       : GeometricMultigrid(fespaces, ess_bdr), coeff(1.0), one(1.0)
    {
       ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0));

       for (int level = 1; level < fespaces.GetNumLevels(); ++level)
       {
          ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), level);
       }
    }


    virtual ~FractionalMultigrid() override
    {
        delete amg;
    }
private:
    void ConstructBilinearForm(ParFiniteElementSpace& fespace,
                               bool partial_assembly)
    {
        ParBilinearForm* form = new ParBilinearForm(&fespace);
        if (partial_assembly)
        {
           form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
        }
        form->AddDomainIntegrator(new DiffusionIntegrator(coeff));
        form->AddDomainIntegrator(new MassIntegrator(one));
        form->Assemble();
        bfs.Append(form);
    }

    void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace& coarse_fespace)
    {
        real_t hmin,hmax,kmin,kmax;
        coarse_fespace.GetParMesh()->GetCharacteristics(hmin,hmax,kmin,kmax);
        real_t mu=hmin*hmin+sigma*sigma;


        ConstructBilinearForm(coarse_fespace, false);

        HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
        bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], *hypreCoarseMat);

        amg = new HypreBoomerAMG(*hypreCoarseMat);
        amg->SetPrintLevel(-1);

        CGSolver* pcg = new CGSolver(MPI_COMM_WORLD);
        pcg->SetPrintLevel(-1);
        pcg->SetMaxIter(10);
        pcg->SetRelTol(sqrt(1e-4));
        pcg->SetAbsTol(0.0);
        pcg->SetOperator(*hypreCoarseMat);
        pcg->SetPreconditioner(*amg);

        Solver* sop=new ScaledSolver(std::pow(mu,s), pcg);

        AddLevel(hypreCoarseMat, sop, true, true);

    }

    void ConstructOperatorAndSmoother(ParFiniteElementSpace& fespace, int level)
    {
        real_t hmin,hmax,kmin,kmax;
        fespace.GetParMesh()->GetCharacteristics(hmin,hmax,kmin,kmax);
        real_t mu=hmin*hmin+sigma*sigma;

        const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
        ConstructBilinearForm(fespace, true);

        OperatorPtr opr;
        opr.SetType(Operator::ANY_TYPE);
        bfs.Last()->FormSystemMatrix(ess_tdof_list, opr);
        opr.SetOperatorOwner(false);

        Vector diag(fespace.GetTrueVSize());
        bfs.Last()->AssembleDiagonal(diag);

        Solver* smoother = new OperatorChebyshevSmoother(
           *opr, diag, ess_tdof_list, 2, fespace.GetParMesh()->GetComm());


        Solver* sop=new ScaledSolver(std::pow(mu,s),  smoother);

        AddLevel(opr.Ptr(), sop, true, true);
    }


};



int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int geometric_refinements = 4;
   int order_refinements = 0;
   const char *device_config = "cpu";
   bool visualization = true;
   int order=1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy has order 2^{or}.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space hierarchy on the parallel mesh.
   //    Here we use continuous Lagrange finite elements. We start with order 1
   //    on the coarse level and geometrically refine the spaces by the specified
   //    amount. Afterwards, we increase the order of the finite elements by a
   //    factor of 2 for each additional level.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   ParFiniteElementSpace *coarse_fespace = new ParFiniteElementSpace(pmesh, fec);

   ParFiniteElementSpaceHierarchy* fespaces = new ParFiniteElementSpaceHierarchy(
      pmesh, coarse_fespace, true, true);

   for (int level = 0; level < geometric_refinements; ++level)
   {
      fespaces->AddUniformlyRefinedLevel();
   }

   HYPRE_BigInt size = fespaces->GetFinestFESpace().GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr = 1;
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(&fespaces->GetFinestFESpace());
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(&fespaces->GetFinestFESpace());
   x = 0.0;


   FractionalMultigrid* M= new FractionalMultigrid(*fespaces, ess_bdr);
   M->SetCycleType(Multigrid::CycleType::VCYCLE,1,1);

   OperatorPtr A;
   Vector X, B;
   M->FormFineLinearSystem(x, *b, A, X, B);

   // 11. Solve the linear system A X = B.
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*M);
   cg.Mult(B, X);

   // 12. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   M->RecoverFineFEMSolution(X, *b, x);


   {
       ParaViewDataCollection paraview_dc("flow", fespaces->GetFinestFESpace().GetParMesh());
       paraview_dc.SetPrefixPath("ParaView");
       paraview_dc.SetLevelsOfDetail(order);
       paraview_dc.SetDataFormat(VTKFormat::BINARY);
       paraview_dc.SetHighOrderOutput(true);
       paraview_dc.SetCycle(0);
       paraview_dc.SetTime(0.0);
       paraview_dc.RegisterField("sol",&x);
       paraview_dc.Save();
   }


   delete M;
   delete b;
   delete fespaces;
   delete fec;

   return 0;
}
