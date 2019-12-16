#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double func(const Vector &x) { return x[0] + 2.0 * x[1]; }

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
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int log = 0;
   bool dg = false;
   bool di = true;  // Domain Integration
   bool bi = true;  // Boundary Integration
   bool fi = true;  // Interior Face Integration
   bool bfi = true; // Boundary Face Integration
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&log, "-l", "--log",
                  "Adjust level of screen output.");
   args.AddOption(&dg, "-dg", "--discontinuous-galerkin", "-h1",
                  "--continuous", "Select H1 or DG space.");
   args.AddOption(&di, "-di", "--domain-integration", "-no-di",
                  "--no-domain-integration",
                  "Enable or disable domain integration test.");
   args.AddOption(&bi, "-bi", "--boundary-integration", "-no-bi",
                  "--no-boundary-integration",
                  "Enable or disable boundary integration test.");
   args.AddOption(&fi, "-fi", "--face-integration", "-no-fi",
                  "--no-face-integration",
                  "Enable or disable interior face integration test.");
   args.AddOption(&bfi, "-bfi", "--bdr-face-integration", "-no-bfi",
                  "--no-bdr-face-integration",
                  "Enable or disable boundary face integration test.");
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
      for (int l = 0; l < ser_ref_levels; l++)
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
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *h1_fec;
   FiniteElementCollection *dg_fec;
   h1_fec = new H1_FECollection(order, dim);
   dg_fec = new DG_FECollection(order, dim);

   ParFiniteElementSpace *h1_fespace = new ParFiniteElementSpace(pmesh, h1_fec);
   ParFiniteElementSpace *dg_fespace = new ParFiniteElementSpace(pmesh, dg_fec);

   ParFiniteElementSpace *fespace = dg ? dg_fespace : h1_fespace;
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x with initial guess of zero,
   //     which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   FunctionCoefficient coef(func);
   x.ProjectCoefficient(coef);

   GridFunctionCoefficient xCoef(&x);

   double tol = 1e-6;
   int npts = 0;

   if (di)
   {
      // Testing Coefficient::Eval as it occurs in Bilinear- and LinearForm
      // Domain Integrators and GridFunction::ProjectCoefficient.
      cout << "Checking " << pmesh->GetNE()
           << " elements in a non-DG context" << endl;
      for (int i=0; i<pmesh->GetNE(); i++)
      {
         ElementTransformation *T = h1_fespace->GetElementTransformation(i);
         const FiniteElement   *fe = h1_fespace->GetFE(i);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                  2*order + 2);

         double tip_data[3];
         Vector tip(tip_data, 3);
         for (int j=0; j<ir.GetNPoints(); j++)
         {
            npts++;
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            T->Transform(ip, tip);

            double f_val = func(tip);
            double gf_val = xCoef.Eval(*T, ip);

            if (fabs(f_val - gf_val) > tol)
            {
               cout << f_val << " " << gf_val << endl;
            }
         }
      }
      cout << "Checked " << npts << " points within elements" << endl;

      // Testing Coefficient::Eval as it occurs in Bilinear- and LinearForm
      // Domain Integrators and GridFunction::ProjectCoefficient.
      npts = 0;
      cout << "Checking " << pmesh->GetNE()
           << " elements in a DG context" << endl;
      for (int i=0; i<pmesh->GetNE(); i++)
      {
         ElementTransformation *T = dg_fespace->GetElementTransformation(i);
         const FiniteElement   *fe = dg_fespace->GetFE(i);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                  2*order + 2);

         double tip_data[3];
         Vector tip(tip_data, 3);
         for (int j=0; j<ir.GetNPoints(); j++)
         {
            npts++;
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            T->Transform(ip, tip);

            double f_val = func(tip);
            double gf_val = xCoef.Eval(*T, ip);

            if (fabs(f_val - gf_val) > tol)
            {
               cout << f_val << " " << gf_val << endl;
            }
         }
      }
      cout << "Checked " << npts << " points within elements" << endl;
   }

   if (bi)
   {
      // Testing Coefficient::Eval as it appears in Bilinear- and LinearForm
      // Boundary Integrators and GridFunction::ProjectBdrCoefficient* methods.
      npts = 0;
      cout << "Checking " << pmesh->GetNBE()
           << " boundary elements in a non-DG context" << endl;
      for (int i=0; i<pmesh->GetNBE(); i++)
      {
         ElementTransformation *T = h1_fespace->GetBdrElementTransformation(i);
         const FiniteElement   *fe = h1_fespace->GetBE(i);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                  2*order + 2);
         double tip_data[3];
         Vector tip(tip_data, 3);
         for (int j=0; j<ir.GetNPoints(); j++)
         {
            npts++;
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            T->Transform(ip, tip);

            double f_val = func(tip);
            double gf_val = xCoef.Eval(*T, ip);

            if (fabs(f_val - gf_val) > tol)
            {
               cout << f_val << " " << gf_val << endl;
            }
         }
      }
      cout << "Checked " << npts << " points within boundary elements" << endl;
   }

   if (fi)
   {
      // Testing Coefficient::Eval as it occurs in Bilinear- and LinearForm
      // Face Integrators
      npts = 0;
      cout << "Checking " << pmesh->GetNumFaces()
           << " faces in a DG context" << endl;
      for (int i=0; i<pmesh->GetNumFaces(); i++)
      {
         if (log > 0) { cout << "Getting trans for face " << i << endl; }
         FaceElementTransformations *T =
            pmesh->GetInteriorFaceTransformations(i);
         if (T != NULL)
         {
            const IntegrationRule &ir = IntRules.Get(T->FaceGeom, 2*order + 2);

            if (log > 0)
            {
               cout << i << " " << T->Elem1No
                    << " " << T->Elem2No << endl;
            }

            double tip_data[3];
            double tip1_data[3];
            double tip2_data[3];
            Vector tip(tip_data, 3);
            Vector tip1(tip1_data, 3);
            Vector tip2(tip2_data, 3);
            for (int j=0; j<ir.GetNPoints(); j++)
            {
               npts++;
               const IntegrationPoint &ip = ir.IntPoint(j);
               IntegrationPoint eip1, eip2;

               T->Loc1.Transform(ip, eip1);
               T->Loc2.Transform(ip, eip2);

               double gf_val1 = NAN;
               double gf_val2 = NAN;

               if (T->Elem1)
               {
                  T->Elem1->SetIntPoint(&eip1);
                  T->Elem1->Transform(eip1, tip1);
                  gf_val1 = xCoef.Eval(*T->Elem1, eip1);
                  if (log > 0)
                  {
                     cout << "Elem1 (" << tip1[0] << "," << tip1[1] << ","
                          << tip1[2] << ") -> " << gf_val1 << endl;
                  }
               }
               if (T->Elem2)
               {
                  T->Elem2->SetIntPoint(&eip2);
                  T->Elem2->Transform(eip2, tip2);
                  gf_val2 = xCoef.Eval(*T->Elem2, eip2);
                  if (log > 0)
                  {
                     cout << "Elem2 (" << tip2[0] << "," << tip2[1] << ","
                          << tip2[2] << ") -> " << gf_val2 << endl;
                  }
               }

               if (T->Face)
               {
                  T->Face->SetIntPoint(&ip);
                  T->Face->Transform(ip, tip);
               }

               double f_val = func(tip);
               // double gf_val = (T->Face) ? xCoef.Eval(*T->Face, ip) : NAN;
               double gf_val = xCoef.Eval(*T, ip);
               if (log > 0)
               {
                  cout << "Face  (" << tip[0] << "," << tip[1] << ","
                       << tip[2] << ") -> " << gf_val << endl;
               }

               if (fabs(f_val - gf_val) > tol)
               {
                  cout << i << " " << f_val << " " << gf_val << endl;
               }
            }
         }
      }
      cout << "Checked " << npts << " points within faces" << endl;
   }

   if (bfi)
   {
      // Testing Coefficient::Eval as it occurs in Bilinear- and LinearForm
      // Boundary Face Integrators
      npts = 0;
      cout << "Checking " << pmesh->GetNBE()
           << " boundary faces in a DG contextx" << endl;
      for (int i=0; i<pmesh->GetNBE(); i++)
      {
         if (log > 0)
         {
            cout << "Getting trans for boundary face " << i << endl;
         }
         FaceElementTransformations *T = pmesh->GetBdrFaceTransformations(i);
         if (T != NULL)
         {
            const IntegrationRule &ir = IntRules.Get(T->FaceGeom, 2*order + 2);

            if (log > 0)
            {
               cout << i << " " << T->Elem1No << " " << T->Elem2No << endl;
            }

            double tip_data[3];
            double tip1_data[3];
            Vector tip(tip_data, 3);
            Vector tip1(tip1_data, 3);
            for (int j=0; j<ir.GetNPoints(); j++)
            {
               npts++;
               const IntegrationPoint &ip = ir.IntPoint(j);
               IntegrationPoint eip1;

               T->Loc1.Transform(ip, eip1);

               double gf_val1 = NAN;

               if (T->Elem1)
               {
                  T->Elem1->SetIntPoint(&eip1);
                  T->Elem1->Transform(eip1, tip1);
                  gf_val1 = xCoef.Eval(*T->Elem1, eip1);
                  if (log > 0)
                  {
                     cout << "Elem1 (" << tip1[0] << "," << tip1[1] << ","
                          << tip1[2] << ") -> " << gf_val1 << endl;
                  }
               }

               T->Face->SetIntPoint(&ip);
               T->Face->Transform(ip, tip);

               double f_val = func(tip);
               // double gf_val = xCoef.Eval(*T->Face, ip);
               double gf_val = xCoef.Eval(*T, ip);

               if (log > 0)
               {
                  cout << "Face  (" << tip[0] << "," << tip[1] << ","
                       << tip[2] << ") -> " << gf_val << endl;
               }

               if (fabs(f_val - gf_val) > tol)
               {
                  cout << i << " " << f_val << " " << gf_val << endl;
               }
            }
         }
      }
      cout << "Checked " << npts << " points within boundary faces" << endl;
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 17. Free the used memory.
   delete dg_fespace;
   delete h1_fespace;
   delete dg_fec;
   delete h1_fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}
