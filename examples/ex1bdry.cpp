#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// for boundary gradient integral from Laplacian operator
class BoundaryGradIntegrator : public BilinearFormIntegrator
{
private:
  Vector shape1, dshape_dn, nor;
  DenseMatrix dshape, dshapedxt, invdfdx;

public:
  void AssembleFaceMatrix(const FiniteElement &el1,
                          const FiniteElement &el2,
                          FaceElementTransformations &Trans,
                          DenseMatrix &elmat) override
  {
   int i, j;
   const int dim   = el1.GetDim();
   const int ndof1 = el1.GetDof();

   const IntegrationRule *ir = IntRule;

   nor.SetSize(dim);
   shape1.SetSize(ndof1);
   dshape_dn.SetSize(ndof1);
   dshape.SetSize(ndof1, dim);
   dshapedxt.SetSize(ndof1, dim);
   invdfdx.SetSize(dim);

   elmat.SetSize(ndof1);
   elmat = 0.0;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1;
      Trans.Loc1.Transform(ip, eip1);
      el1.CalcShape(eip1, shape1);
      //d of shape function, evaluated at eip1
      el1.CalcDShape(eip1, dshape);

      Trans.Elem1->SetIntPoint(&eip1);

      CalcInverse(Trans.Elem1->Jacobian(), invdfdx);  //inverse Jacobian
      //invdfdx.Transpose();
      Mult(dshape, invdfdx, dshapedxt);  // dshapedxt = grad phi* J^-1

      //get normal vector
      Trans.Face->SetIntPoint(&ip);
      const DenseMatrix &J = Trans.Face->Jacobian(); 
      if (dim == 1)
      {
         nor(0) = 2 * eip1.x - 1.0;
      }
      else if (dim == 2)
      {
         nor(0) = J(1, 0);
         nor(1) = -J(0, 0);
      }
      else if (dim == 3)
      {
         nor(0) = J(1, 0) * J(2, 1) - J(2, 0) * J(1, 1);
         nor(1) = J(2, 0) * J(0, 1) - J(0, 0) * J(2, 1);
         nor(2) = J(0, 0) * J(1, 1) - J(1, 0) * J(0, 1);
      }

      // multiply weight into normal, make answer negative
      // (boundary integral is subtracted)
      nor *= -ip.weight;

      dshapedxt.Mult(nor, dshape_dn);

      for (i = 0; i < ndof1; i++)
      {
         for (j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i) * dshape_dn(j);
         }
      }
   }
  }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
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
      MPI_Finalize();
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
   //    more than 10,000 elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
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

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
   }
   else if (pmesh->GetNodes())
   {
      fec = pmesh->GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator);

   //Set up boundary term
   BoundaryGradIntegrator *bdrg = new BoundaryGradIntegrator;
   int bdr_geom = Geometry::SEGMENT; 
   if(dim==3) bdr_geom = Geometry::SQUARE;
   int pBF_order =  2 * order + 1;
   const IntegrationRule *ir_pBF  = &IntRules.Get(bdr_geom, pBF_order);
   bdrg->SetIntRule(ir_pBF);

   a->AddBdrFaceIntegrator(bdrg);
   a->Assemble();
   a->Finalize();

   Vector x_vec(a->Height()); x_vec=1.0;
   Vector y_vec(a->Height());

   a->Mult(x_vec,y_vec);
   

   delete a;
   delete fespace;
   if (order > 0) { delete fec; }
   delete pmesh;

   MPI_Finalize();

   return 0;
}
