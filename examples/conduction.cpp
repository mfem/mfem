//                                MFEM Example 0
//
// mpirun -np 4 conduction

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double tempinit(const Vector &x)
{
   double val = 0.0;
   double sf = 100.0;
   val = 1.0 - 1.0*(0.5*(1+std::tanh(sf*(x(1)-2.1))) - 0.5*(1+std::tanh(sf*(x(
                                                                               1)-4.1)))+0.0);
   return val;
}

void ModifyBoundaryAttributesForNodeMovement(Mesh *mesh)
{
   const int dim = mesh->Dimension();
   GridFunction x = *(mesh->GetNodes());
   Vector vert_coord;
   mesh->GetVertices(vert_coord);
   int ntotv = vert_coord.Size()/dim;
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      if (mesh->GetBdrAttribute(i) == 2) { continue; }
      // attribute 1 is at x = 0 and x = 12
      // we specify:
      // attribute 1 at x = 0 in y = [0, 2], [4,6]
      // attribute 3 at x = 0 in y = [2, 4]
      // attribute 4 at x = 12
      //      int f = mesh->GetBdrElementEdgeIndex(i);
      const Element *be =  mesh->GetBdrElement(i);
      //      const Element *fe = mesh->GetFace(f);
      const int *bv = be->GetVertices();
      //      const int *fv = fe->GetVertices();
      int nv = be->GetNVertices();
      double yavg = 0.0;
      double xavg = 0.0;
      for (int vi = 0; vi < nv; ++vi)
      {
         int vertex_index = bv[vi];
         double xc = vert_coord(vertex_index);
         double yc = vert_coord(vertex_index+ntotv);
         xavg += xc;
         yavg += yc;
      }
      xavg /= nv;
      yavg /= nv;
      if (xavg == 12.0)
      {
         mesh->SetBdrAttribute(i, 4);
      }
      else if (yavg < 2.0)
      {
         mesh->SetBdrAttribute(i, 1);
      }
      else if (yavg > 2.0 && yavg < 4.0)
      {
         mesh->SetBdrAttribute(i, 3);
      }
      else if (yavg > 4.0 && yavg < 6.0)
      {
         mesh->SetBdrAttribute(i, 1);
      }
      else
      {
         MFEM_ABORT("check the mesh\n");
      }
   }
   mesh->SetAttributes();
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   //    int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command line options.
   const char *mesh_file = "../miniapps/meshing/optimized.mesh";
   int order = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
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

   Mesh *mesh2 = new Mesh(Mesh::MakeQuadTo4TriMesh(12, 6, 12.0, 6.0));
   {
      ofstream mesh_ofs("splitmesh.mesh");
      mesh_ofs.precision(14);
      mesh2->Print(mesh_ofs);
   }

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   Mesh mesh(mesh_file);
   ModifyBoundaryAttributesForNodeMovement(&mesh);
   {
      ofstream mesh_ofs("bdrattrfixed.mesh");
      mesh_ofs.precision(14);
      mesh.Print(mesh_ofs);
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   const int dim = pmesh.Dimension();

   // 3. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec);

   Array<int> dbc_bdr(pmesh.bdr_attributes.Max());
   dbc_bdr = 0;
   dbc_bdr[0] = 1;
   dbc_bdr[1] = 1;
   dbc_bdr[2] = 1;

   Array<int> nbc_bdr(pmesh.bdr_attributes.Max());
   nbc_bdr = 0;
   nbc_bdr[3] = 1;

   Array<int> ess_tdof_list(0);
   fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);

   FunctionCoefficient dbcCoef(tempinit);
   ConstantCoefficient nbcCoef(0.0);

   ParGridFunction u(&fespace);
   u = 0.0;

   // Thermal conductivity
   L2_FECollection fecl2(0, dim);
   ParFiniteElementSpace l2fespace(&pmesh, &fecl2);
   ParGridFunction kappa(&l2fespace);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      kappa(e) = pmesh.GetAttribute(e) == 1 ? 100000 : 1e-4;
   }
   GridFunctionCoefficient cKappa(&kappa);

   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(cKappa));
   a.AddDomainIntegrator(new MassIntegrator());
   a.Assemble();

   u.ProjectCoefficient(dbcCoef);

   ParLinearForm b(&fespace);
   b = 0.0;

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

   HypreSolver *amg = new HypreBoomerAMG;
   if (false)
   {
      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(*amg);
      pcg.SetOperator(*A);
      pcg.Mult(B, X);
   }
   else
   {
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(200);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetPreconditioner(*amg);
      gmres.SetOperator(*A);
      gmres.Mult(B, X);
   }
   delete amg;

   // 13. Recover the parallel grid function corresponding to U. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, u);

   if (true)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize()
               << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u
               << "window_title '" << " Solution'"
               << " keys 'mmc'" << flush;
   }

   // Get Mean Element Temperature
   u.GetElementAverages(kappa);
   if (true)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize()
               << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << kappa
               << "window_title '" << " Mean element temperature'"
               << " keys 'mmc'" << flush;
   }

   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      if (pmesh.GetAttribute(e) == 2) { continue; }
      if (kappa(e) < 0.5)
      {
         pmesh.SetAttribute(e, 3);
      }
   }
   pmesh.SetAttributes();
   {
      ostringstream mesh_name;
      mesh_name << "final.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      //pmesh->PrintAsOne(mesh_ofs);
      pmesh.PrintAsSerial(mesh_ofs);
   }


   return 0;
}
