#include "mfem.hpp"
#include "../common/mesh_extras.hpp"
#include "../common/pmesh_extras.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;

double rhoFunc(const Vector &x)
{
   // Off-center Gaussian
   double s_data[3];
   Vector s(s_data, x.Size());
   s = 0.0; s[0] = -1.0; s += x;
   return exp(-(s * s));
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/ball-nurbs.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int ir_order = 2;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&ir_order, "-o", "--order",
                  "Integration rule order.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement where 'ref_levels' is user defined.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   ConstantCoefficient massDensity(1.0);
   // FunctionCoefficient massDensity(rhoFunc);

   L2_FECollection fec(0, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);
   ParFiniteElementSpace *vfes = new ParFiniteElementSpace(pmesh, &fec, sdim);
   ParGridFunction elemMass(fes);
   ParGridFunction elemCent(vfes);

   /*
   {
   double vol = 0.0;
   double surf = 0.0;
   double mass = 0.0;

   for (int i=0; i<mesh->GetNE(); i++)
   {
      ElementTransformation *T = mesh->GetElementTransformation(i);
      Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);

         double w = T->Weight() * ip.weight;

         vol += w;
         mass += w * massDensity.Eval(*T, ip);
      }
   }

   for (int i=0; i<mesh->GetNBE(); i++)
   {
      ElementTransformation *T = mesh->GetBdrElementTransformation(i);
      Geometry::Type geom = mesh->GetBdrElementBaseGeometry(i);
      const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);

         double w = T->Weight() * ip.weight;

         surf += w;
      }
   }

   cout << "Volume:       " << vol << endl;
   cout << "Surface Area: " << surf << endl;
   cout << "Mass:         " << mass << endl;
   }
   */
   double vol = ComputeVolume(*pmesh, ir_order);
   double area = ComputeSurfaceArea(*pmesh, ir_order);

   Vector mom1(sdim);
   Vector cent(sdim); cent = 0.0;
   DenseMatrix mom2(sdim);

   double mass  = ComputeZerothMoment(*pmesh, massDensity, ir_order);
   double mass1 = ComputeFirstMoment(*pmesh, massDensity, ir_order, mom1);
   double mass2 = ComputeSecondMoment(*pmesh, massDensity, cent,
                                      ir_order, mom2);

   if (myid == 0)
   {
      cout << "Volume:       " << vol << endl;
      cout << "Surface Area: " << area << endl;
      cout << "Mass:         " << mass << endl;
      cout << "Mass:         " << mass1 << endl;
      cout << "Mass:         " << mass2 << endl;

      cout << "First Moment:   "; mom1.Print(cout);
      mom1 /= mass1;
      cout << "Center of Mass: "; mom1.Print(cout);
      cout << "Second Moment (moment of inertia):\n";
      mom2.Print(cout);
   }

   ComputeElementZerothMoments(*pmesh, massDensity, ir_order, elemMass);
   ComputeElementCentersOfMass(*pmesh, massDensity, ir_order, elemCent);

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream m_sock(vishost, visport);
      m_sock << "parallel " << num_procs << " " << myid << "\n";
      m_sock.precision(8);
      m_sock << "solution\n" << *pmesh << elemMass
             << "window_title 'Element Masses'" << flush;

      socketstream c_sock(vishost, visport);
      c_sock << "parallel " << num_procs << " " << myid << "\n";
      c_sock.precision(8);
      c_sock << "solution\n" << *pmesh << elemCent
             << "window_title 'Element Centers'"
             << "keys vvv" << flush;
   }

   // 16. Free the used memory.
   delete fes;
   delete vfes;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
