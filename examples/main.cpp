//                                MFEM MAGMA example
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   const char *device_config = "hip";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();


   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new DG_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;


   // 5. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   Vector rhs(x.Size());
   rhs.Randomize();

   Vector x_ref = rhs;
   Vector x_magma = x_ref;

   // 6. Set up the bilinear form a(.,.) on the finite element space
   const int NE = mesh.GetNE();
   const int ndofs = x_ref.Size() / NE;

   MassIntegrator mass_int;
   Vector VecMassMats(ndofs * ndofs * NE);
   mass_int.AssembleEA(fespace, VecMassMats, false);

   //Batch solver...
   DenseTensor LUMassMats(ndofs, ndofs, NE);
   DenseTensor MassMats(ndofs, ndofs, NE);
   Array<int> P(rhs.Size());

   for (int i=0; i<ndofs * ndofs * NE; ++i)
   {
      LUMassMats.HostWrite()[i] = VecMassMats.HostRead()[i];
      MassMats.HostWrite()[i] = VecMassMats.HostRead()[i];
   }

   //
   //Compute reference solution
   //
   mfem::BatchLUFactor(LUMassMats, P);
   mfem::BatchLUSolve(LUMassMats, P, x_ref);


   //
   //Compute Magma solution
   //

   //Compute magma version with variable batch solver
   magma_int_t magma_device = 0;
   magma_queue_t magma_queue;

   magma_setdevice(magma_device);
   magma_queue_create(magma_device, &magma_queue);


   //Number of rows and columns of each matrix
   Array<magma_int_t> num_of_rows(NE);
   Array<magma_int_t> num_of_cols(NE);

   //Pointers to mass matrices
   Array<double *> magma_LUMassMats(NE);
   for (int i=0; i<NE; ++i)
   {
      num_of_rows[i] = ndofs;
      num_of_cols[i] = ndofs;
      magma_LUMassMats.HostWrite()[i] = &MassMats.ReadWrite()[i*ndofs*ndofs];
   }

   Array<magma_int_t> ldda = num_of_cols;
   Array<magma_int_t *> dipiv_array(NE);
   Array<magma_int_t> info_array(NE);

   //Pointers to pivot vectors
   P.HostReadWrite();
   P = 0.0;
   for (int i=0; i<NE; ++i)
   {
      dipiv_array.HostWrite()[i] = &P.ReadWrite()[i*ndofs];
   }

   //Perform variable batch size factorization
   magma_dgetrf_vbatched(num_of_rows.ReadWrite(), num_of_cols.ReadWrite(),
                         magma_LUMassMats.ReadWrite(), ldda.ReadWrite(),
                         dipiv_array.ReadWrite(), info_array.ReadWrite(),
                         NE, magma_queue);


   //
   //Question: How do we perform the batch LU/Solve ?
   //


   //rhs only has 1 column
   Array<int> rhs_num_of_cols(NE);
   rhs_num_of_cols = 1;

   //Question: Do we need to perform the pivoting ourselfs?

   //x_magma_ptrs to x_magma
   Array<double *> x_magma_ptrs(NE);
   for (int i=0; i<NE; ++i)
   {
      x_magma_ptrs.HostWrite()[i] = &x_magma.ReadWrite()[i*ndofs];
   }

   Array<magma_int_t> trSolve_ldda(NE + 1);
   Array<magma_int_t> trSolve_lddb(NE + 1);
   for (int i=0; i<NE+1; ++i)
   {
      trSolve_ldda[i] = ndofs; //??
      trSolve_lddb[i] = ndofs; //??
   }

   //
   //Current error:
   // hip error code: 'hipErrorInvalidValue':1 at /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/rocBLAS/library/src/rocblas_auxiliary.cpp:633
   //


   //L(Ux) = b //lower solve
   magmablas_dtrsm_vbatched(MagmaLeft, MagmaLower, MagmaNoTrans,
                            MagmaNonUnit,
                            num_of_rows.ReadWrite(),
                            rhs_num_of_cols.ReadWrite(),
                            1.0,
                            magma_LUMassMats.ReadWrite(), trSolve_ldda.ReadWrite(),
                            x_magma_ptrs.ReadWrite(), trSolve_lddb.ReadWrite(),
                            NE, magma_queue);

   //Ux = b //upper solve
   magmablas_dtrsm_vbatched(MagmaLeft, MagmaUpper, MagmaNoTrans,
                            MagmaUnit,
                            num_of_rows.ReadWrite(),
                            rhs_num_of_cols.ReadWrite(),
                            1.0,
                            magma_LUMassMats.ReadWrite(), trSolve_ldda.ReadWrite(),
                            x_magma_ptrs.ReadWrite(), trSolve_lddb.ReadWrite(),
                            NE, magma_queue);


   //Compute error
   x_magma -= x_ref;
   double error = x_magma.Norml2();
   std::cout<<"error = "<<error<<std::endl;


   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
