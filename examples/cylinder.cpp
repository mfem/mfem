#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void mobius_trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   const char *new_mesh_file = "cylinder.mesh";
   int nx = 6;
   int ny = 1;
   int order = 1;
   int close_strip = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&new_mesh_file, "-m", "--mesh-out-file",
                  "Output Mesh file to write.");
   args.AddOption(&nx, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&order, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh *mesh;
   // The mesh could use quads (default) or triangles
   Element::Type el_type = Element::QUADRILATERAL;
   // Element::Type el_type = Element::TRIANGLE;
   mesh = new Mesh(nx, ny, el_type, 1, 2*M_PI, 1.0);

   mesh->SetCurvature(order, false, 3, Ordering::byVDIM);

   if (close_strip)
   {
      Array<int> v2v(mesh->GetNV());
      for (int i = 0; i < v2v.Size(); i++)
      {
         v2v[i] = i;
      }
      // identify vertices on vertical lines (with a flip)
      for (int j = 0; j <= ny; j++)
      {
         int v_old = nx + j * (nx + 1);
         int v_new = ((close_strip == 1) ? j : (ny - j)) * (nx + 1);
         v2v[v_old] = v_new;
      }
      // renumber elements
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         Element *el = mesh->GetElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      // renumber boundary elements
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         Element *el = mesh->GetBdrElement(i);
         int *v = el->GetVertices();
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            v[j] = v2v[v[j]];
         }
      }
      mesh->RemoveUnusedVertices();
      mesh->RemoveInternalBoundaries();
   }

   mesh->Transform(mobius_trans);
   mesh->SetCurvature(order, false, 3, Ordering::byVDIM);

   GridFunction &nodes = *mesh->GetNodes();
   for (int i = 0; i < nodes.Size(); i++)
   {
      if (std::abs(nodes(i)) < 1e-12)
      {
         nodes(i) = 0.0;
      }
   }

   ofstream ofs(new_mesh_file);
   ofs.precision(8);
   mesh->Print(ofs);
   ofs.close();

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   delete mesh;
   return 0;
}

void mobius_trans(const Vector &x, Vector &p)
{
   double R = 1.8;
   double H = 2.0;

   p.SetSize(3);
   p[0] = R*cos(x[0]);
   p[1] = R*sin(x[0]);
   p[2] = H*x[1];
}
