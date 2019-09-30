#include "mfem.hpp"
#include "../common/mesh_extras.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

int main(int argc, char ** argv)
{
   int mf, mb, na, nb, nt;
   double af, ab, ba, bb, bt;
   bool per_y = false;
   bool visualization = true;

   mf = mb = na = nb = nt = -1;
   af = ab = ba = bb = bt = -1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mf, "-mf", "--num-front",
                  "Number of elements in front of antenna (>= 1).");
   args.AddOption(&mb, "-mb", "--num-back",
                  "Number of elements behind antenna (>= 1).");
   args.AddOption(&nb, "-nb", "--num-bottom",
                  "Number of elements below antenna (>= 1).");
   args.AddOption(&nt, "-nt", "--num-top",
                  "Number of elements above antenna (>= 1).");
   args.AddOption(&na, "-na", "--num-across",
                  "Number of elements across antenna (>= 2).");
   args.AddOption(&af, "-af", "--size-front",
                  "Distance in front of antenna (> 0).");
   args.AddOption(&ab, "-ab", "--size-back",
                  "Distance behind antenna (> 0).");
   args.AddOption(&bb, "-bb", "--size-bottom",
                  "Distance below antenna (> 0).");
   args.AddOption(&bt, "-bt", "--size-top",
                  "Distance above antenna (> 0).");
   args.AddOption(&ba, "-ba", "--size-across",
                  "Distance across antenna (> 0).");
   args.AddOption(&per_y, "-per-y", "--periodic-y", "-no-per-y",
                  "--no-periodic-y",
                  "Make the mesh periodic in the y direction.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   if (mf < 0) { mf = 1; }
   if (mb < 0) { mb = 1; }
   if (na < 0) { na = 2; }
   if (nb < 0) { nb = 1; }
   if (nt < 0) { nt = 1; }

   if (af < 0) { af = 0.75; }
   if (ab < 0) { ab = 0.25; }
   if (ba < 0) { ba = 0.5; }
   if (bb < 0) { bb = 0.25; }
   if (bt < 0) { bt = 0.25; }

   args.PrintOptions(cout);

   MFEM_VERIFY(na >= 2,
               "There must be at least two elements across "
               "the face of the antenna");
   MFEM_VERIFY(mf > 0 && na > 0 && nb > 0 && nt > 0,
               "Numbers of elements must be greater than zero.");
   MFEM_VERIFY(af > 0.0 && ab > 0.0 && ba > 0.0 && bb > 0.0 && bt > 0.0,
               "Distances must be greater than zero.");

   int mx = mf + mb;
   int ny = nb + na + nt;

   // double ax = af + ab;
   double by = bb + ba + bt;

   int nelem = mx * ny;
   int nnode = (mx + 1) * (ny + 1) + na - 1;
   int nbdr  = 2 * mx + 2 * ny + 2 * na;

   Mesh *mesh = new Mesh(2, nnode, nelem, nbdr);

   // Create vertices
   double c[2];
   for (int j=0; j<=ny; j++)
   {
      double y0 = by * j / ny;
      double ya = (j<=nb) ? (bb * j / nb) :
                  ((j<=nb+na)? (bb + ba * (j - nb) / na) :
                   (bb + ba + bt * (j - nb - na) / nt));
      for (int i=0; i<=mx; i++)
      {
         c[0] = (i <= mf) ? ((af * i) / mf) : (af + ab * (i - mf) / mb);

         if (i <= mf)
         {
            c[1] = y0 * (mf - i) / mf + ya * i / mf;
         }
         else
         {
            c[1] = y0 * (i - mf) / mb + ya * (mx - i) / mb;
         }

         mesh->AddVertex(c);
      }
   }
   for (int j=1; j < na; j++)
   {
      c[0] = (1.0 + 1.0e-4) * af;
      c[1] = bb + ba * j / na;
      mesh->AddVertex(c);
   }

   // Create elements
   int v[4];
   for (int j=0; j<nb; j++)
   {
      for (int i=0; i<mx; i++)
      {
         v[0] = j * (mx + 1) + i;
         v[1] = j * (mx + 1) + i + 1;
         v[2] = (j + 1) * (mx + 1) + i + 1;
         v[3] = (j + 1) * (mx + 1) + i;

         mesh->AddQuad(v);
      }
   }
   for (int j=nb; j<nb + na; j++)
   {
      for (int i=0; i<mf; i++)
      {
         v[0] = j * (mx + 1) + i;
         v[1] = j * (mx + 1) + i + 1;
         v[2] = (j + 1) * (mx + 1) + i + 1;
         v[3] = (j + 1) * (mx + 1) + i;

         mesh->AddQuad(v);
      }
      for (int i=mf; i<mx; i++)
      {
         if (i == mf)
         {
            if (j == nb)
            {
               v[0] = j * (mx + 1) + i;
               v[1] = j * (mx + 1) + i + 1;
               v[2] = (j + 1) * (mx + 1) + i + 1;
               v[3] = (mx + 1) * (ny + 1);
               // v[3] = (j + 1) * (mx + 1) + i;
            }
            else if (j == nb + na -1)
            {
               v[0] = nnode - 1;
               v[1] = j * (mx + 1) + i + 1;
               v[2] = (j + 1) * (mx + 1) + i + 1;
               v[3] = (j + 1) * (mx + 1) + i;
            }
            else
            {
               // v[0] = j * (mx + 1) + i;
               v[0] = nnode - na + j - nb;
               v[1] = j * (mx + 1) + i + 1;
               v[2] = (j + 1) * (mx + 1) + i + 1;
               v[3] = nnode - na + j - nb + 1;
               // v[3] = (j + 1) * (mx + 1) + i;
            }
         }
         else
         {
            v[0] = j * (mx + 1) + i;
            v[1] = j * (mx + 1) + i + 1;
            v[2] = (j + 1) * (mx + 1) + i + 1;
            v[3] = (j + 1) * (mx + 1) + i;
         }

         mesh->AddQuad(v);
      }
   }
   for (int j=nb + na; j<ny; j++)
   {
      for (int i=0; i<mx; i++)
      {
         v[0] = j * (mx + 1) + i;
         v[1] = j * (mx + 1) + i + 1;
         v[2] = (j + 1) * (mx + 1) + i + 1;
         v[3] = (j + 1) * (mx + 1) + i;

         mesh->AddQuad(v);
      }
   }

   // Create boundary elements
   for (int i=0; i<mx; i++)
   {
      v[0] = i;
      v[1] = i + 1;
      mesh->AddBdrSegment(v, 1);
   }
   for (int j=0; j<ny; j++)
   {
      v[0] = (mx + 1) * j + mx;
      v[1] = (mx + 1) * (j + 1) + mx;
      mesh->AddBdrSegment(v, 2);
   }
   for (int i=mx; i>0; i--)
   {
      v[0] = (mx + 1) * ny + i;
      v[1] = (mx + 1) * ny + i - 1;
      mesh->AddBdrSegment(v, 3);
   }
   for (int j=ny; j>0; j--)
   {
      v[0] = j * (mx + 1);
      v[1] = (j - 1) * (mx + 1);
      mesh->AddBdrSegment(v, 4);
   }
   for (int j=nb; j<na + nb; j++)
   {
      v[0] = (mx + 1) * j + mf;
      v[1] = (mx + 1) * (j + 1) + mf;
      mesh->AddBdrSegment(v, 5);
   }
   for (int j=nb+na; j>nb; j--)
   {
      if (j == nb + na)
      {
         v[0] = (mx + 1) * j + mf;
      }
      else
      {
         v[0] = nnode - (nb + na - j);
      }
      if (j == nb + 1)
      {
         v[1] = (mx + 1) * (j - 1) + mf;
      }
      else
      {
         v[1] = nnode - (nb + na - j + 1);
      }
      mesh->AddBdrSegment(v, 6);
   }
   mesh->FinalizeTopology();

   if (per_y)
   {
      Array<int> v2v(mesh->GetNV());
      for (int i=0; i<v2v.Size(); i++) { v2v[i] = i; }

      for (int i=0; i<=mx; i++)
      {
         v2v[(mx + 1) * ny + i] = i;
      }

      Mesh * per_mesh = miniapps::MakePeriodicMesh(mesh, v2v);
      delete mesh;
      mesh = per_mesh;
   }

   ofstream mesh_ofs("simple_antenna.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);

   // Output the resulting mesh to GLVis
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

   // Clean up and exit
   delete mesh;
}
