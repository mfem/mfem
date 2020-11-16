//                       MFEM Example - Parallel Version
//
// Compile with: make ex_norm
//
// Sample runs:
//        mpirun -n 4 ./ex_surf_int_p -m star-int.mesh
//
// Device sample runs:
//
// Description: This is a prototype example showing how one might
//         compute the flux of a vector field through a surface.  The
//         vector field is represented using an H(Div),
//         a.k.a. Raviart-Thomas, basis.  The surface integral is
//         computed using a ParLinearForm object with a specialized
//         integrator, SurfaceFluxLFIntegrator.  This integrator
//         requires a VectorCoefficient which specifies the
//         orientation of the surface integral.  The magnitude of the
//         VectorCoefficient is unimportant (assuming it's of order
//         1), however, it must pierce the surface of interest in a
//         consistent direction.  This direction determines the sign
//         of the flux integral.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//void ProjectNormals(ParGridFunction &n);
void FFunc(const Vector &x, Vector &f)
{
  int dim = x.Size();
  f.SetSize(dim);
  for (int d=0; d<dim; d++) f(d) = 1.0 + d;
}

class SurfaceFluxLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   SurfaceFluxLFIntegrator(VectorCoefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   
   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tet.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume, as well as periodic meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement (2 by default, or
   //    specified on the command line with -rs).
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution (1 time by
   //    default, or specified on the command line with -rp). Once the parallel
   //    mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }
   pmesh.ReorientTetMesh();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Raviart-Thomas finite elements of the specified order.
   RT_FECollection fec(order-1, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   if (mpi.Root())
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 10. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary faces will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   VectorFunctionCoefficient FCoef(dim, FFunc);
   ParGridFunction x(&fespace);

   x.ProjectCoefficient(FCoef);

   Vector xVec(dim); xVec = 0.0; xVec(0) = 1.0;
   Vector yVec(dim); yVec = 0.0; if (dim > 1) { yVec(1) = 1.0; }
   Vector zVec(dim); zVec = 0.0; if (dim > 2) { zVec(2) = 1.0; }
   Vector nxVec(dim); nxVec = 0.0; nxVec(0) = -1.0;
   Vector nyVec(dim); nyVec = 0.0; if (dim > 1) { nyVec(1) = -1.0; }
   Vector nzVec(dim); nzVec = 0.0; if (dim > 2) { nzVec(2) = -1.0; }
   VectorConstantCoefficient xCoef(xVec);
   VectorConstantCoefficient yCoef(yVec);
   VectorConstantCoefficient zCoef(zVec);
   VectorConstantCoefficient nxCoef(nxVec);
   VectorConstantCoefficient nyCoef(nyVec);
   VectorConstantCoefficient nzCoef(nzVec);


   Array<int> bdr(pmesh.bdr_attributes.Max());
   for (int i=1; i<=pmesh.bdr_attributes.Max(); i++)
   {
     bdr = 0; bdr[i-1] = 1;

     double fluxX = 0.0;
     double fluxY = 0.0;
     double fluxZ = 0.0;
     
     ParLinearForm fluxIntX(&fespace);
     fluxIntX.AddBoundaryIntegrator(new SurfaceFluxLFIntegrator(xCoef), bdr);
     fluxIntX.Assemble();
     fluxX = fluxIntX(x);

     if (dim > 1)
     {
       ParLinearForm fluxIntY(&fespace);
       fluxIntY.AddBoundaryIntegrator(new SurfaceFluxLFIntegrator(yCoef), bdr);
       fluxIntY.Assemble();
       fluxY = fluxIntY(x);
     }
     if (dim > 2)
     {
       ParLinearForm fluxIntZ(&fespace);
       fluxIntZ.AddBoundaryIntegrator(new SurfaceFluxLFIntegrator(zCoef), bdr);
       fluxIntZ.Assemble();
       fluxZ = fluxIntZ(x);
     }

     if (mpi.Root()) { cout << "Fluxes " << i << ": "
			    << fluxX << ' ' << fluxY << ' ' << fluxZ << endl; }
   }
   
   // 16. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << mpi.WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << mpi.WorldSize() << " "
	       << mpi.WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   return 0;
}

void SurfaceFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      Q.Eval(Qvec, Tr, ip);

      el.CalcPhysShape(Tr, shape);

      double area = nor.Norml2();
      double prod = Qvec*nor;
      double signedArea = area * ((fabs(prod) < 1e-4 * fabs(area)) ? 0.0 :
				  copysign(1.0, prod));
      
      elvect.Add(ip.weight*signedArea, shape);
   }
}
