
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
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

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(4,4, Element::QUADRILATERAL, true, 1.0, 1.0, false);

   int dim = mesh->Dimension();

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << 
       "window_title 'Original Mesh' " << flush;
   }

   // Extend the mesh by n layers
   // This is assuming uniform quad/hex mesh (for now)
   
   

   // extrute on one dimension
   // d = 1 +x, -1 -x, 2 +y, -2 +y , 3 +z, -3, -z

   // copy the original mesh;
   Mesh * mesh_orig = new Mesh(*mesh);
   Mesh * mesh_ext = nullptr;

   Array<int> directions(6);
   directions[0] = 1;
   directions[1] = -1;
   directions[2] = 2;
   directions[3] = -2;
   directions[4] = 2;
   directions[5] = -1;

   for (int j=0; j<directions.Size(); j++)
   {
      int d = directions[j];
      int nrelem = mesh_orig->GetNE();


      Vector pmin;
      Vector pmax;
      mesh_orig->GetBoundingBox(pmin,pmax);

      DenseMatrix J(dim);
      double hmin, hmax;
      hmin = infinity();
      hmax = -infinity();
      Vector attr(nrelem);
      // element size

      for (int iel=0; iel<nrelem; ++iel)
      {
         int geom = mesh_orig->GetElementBaseGeometry(iel);
         ElementTransformation *T = mesh_orig->GetElementTransformation(iel);
         T->SetIntPoint(&Geometries.GetCenter(geom));
         Geometries.JacToPerfJac(geom, T->Jacobian(), J);
         attr(iel) = J.Det();
         if (attr(iel) < 0.0)
         {
            attr(iel) = -pow(-attr(iel), 1.0/double(dim));
         }
         else
         {
            attr(iel) = pow(attr(iel), 1.0/double(dim));
         }
         hmin = min(hmin, attr(iel));
         hmax = max(hmax, attr(iel));
      }
      MFEM_VERIFY(hmin==hmax, "Case not supported yet")

      double val;
      // find the vertices on the specific boundary
      switch (d)
      {
      case 1:
         val = pmax[0];
         break;
      case -1:
         val = pmin[0];
         hmax = -hmax;
         break;    
      case 2:
         val = pmax[1];
         break;
      case -2:
         val = pmin[1];
         hmax = -hmax;
         break;   
      case 3:
         val = pmax[2];
         break;
      case -3:
         val = pmin[2];
         hmax = -hmax;
         break;      
      }
      int k = 0;
      for (int i = 0; i<mesh_orig->GetNV(); ++i)
      {
         double * coords = mesh_orig->GetVertex(i);
         switch (abs(d))
         {
         case 1:
            if (coords[0] == val) k++;
            break;
         case 2:
            if (coords[1] == val) k++;
            break;   
         case 3:
            if (coords[2] == val) k++;
            break;      
         }
      }
      int nrvertices = mesh_orig->GetNV() + k;
      int nrelements = mesh_orig->GetNE() + pow(pow(k,1.0/(dim-1))-1.0,dim-1);

      mesh_ext = new Mesh(dim, nrvertices, nrelements);

      // Add existing vertices
      Array<int> vmap(mesh_orig->GetNV()); vmap = 0;
      k = mesh_orig->GetNV();
      for (int i=0; i<mesh_orig->GetNV(); ++i)
      {
         double * vert = mesh_orig->GetVertex(i);
         mesh_ext->AddVertex(vert);
         switch (abs(d))
         {
         case 1:
            if (vert[0] == val) 
            {
               vmap[i] = k;
               k++;
            }
            break;
         case 2:
            if (vert[1] == val)
            {
               vmap[i] = k;
               k++;
            }
            break;   
         case 3:
            if (vert[2] == val)
            {
               vmap[i] = k;
               k++;
            }
            break;      
         }
      } 
      // Add existing elements
      for (int i=0; i<mesh_orig->GetNE(); ++i)
      {
         Array<int>ind;
         mesh_orig->GetElementVertices(i,ind);
         if (dim == 2)
         {
            mesh_ext->AddQuad(ind);
         }
         else if (dim == 3)
         {
            mesh_ext->AddHex(ind);
         }
      } 
      // Add new vertices
      k = mesh_orig->GetNV();
      for (int i=0; i<mesh_orig->GetNV(); ++i)
      {
         double * vert = mesh_orig->GetVertex(i);
         switch (abs(d))
         {
         case 1:
            if (vert[0] == val) 
            {
               double coords[dim];
               coords[0] = vert[0] + hmax;
               coords[1] = vert[1];
               if (dim == 3) coords[2] = vert[2];
               mesh_ext->AddVertex(coords);
            }
            break;
         case 2:
            if (vert[1] == val)
            {
               double coords[dim];
               coords[0] = vert[0];
               coords[1] = vert[1] + hmax;
               if (dim == 3) coords[2] = vert[2];
               mesh_ext->AddVertex(coords);
            }
            break;   
         case 3:
            if (vert[2] == val)
            {
               double coords[dim];
               coords[0] = vert[0];
               coords[1] = vert[1];
               coords[2] = vert[2] + hmax;
               mesh_ext->AddVertex(coords);
            }
            break;      
         }
      }    
      // loop through boundary elements and extend in the given direction
      for (int i=0; i<mesh_orig->GetNBE(); ++i) 
      {
         Array<int> vertices;
         mesh_orig->GetBdrElementVertices(i,vertices);
         if (dim == 2)
         {
            int ind[4];
            if (vmap[vertices[0]] && vmap[vertices[1]])
            {
               ind[0] = vmap[vertices[0]];
               ind[1] = vmap[vertices[1]];
               ind[2] = vertices[1];
               ind[3] = vertices[0];
               mesh_ext->AddQuad(ind);
            }
         }
         else if (dim == 3)
         {
            int ind[8];
            if (vmap[vertices[0]] && vmap[vertices[1]] && vmap[vertices[2]] && vmap[vertices[3]])
            {
               ind[0] = vmap[vertices[0]];
               ind[1] = vmap[vertices[1]];
               ind[2] = vmap[vertices[2]];
               ind[3] = vmap[vertices[3]];
               ind[4] = vertices[0];
               ind[5] = vertices[1];
               ind[6] = vertices[2];
               ind[7] = vertices[3];
               mesh_ext->AddHex(ind);
            }
         }   
      }
      mesh_ext->FinalizeTopology();

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream mesh_sock(vishost, visport);
         mesh_sock.precision(8);
         mesh_sock << "mesh\n" << *mesh_ext <<
         "window_title 'New Mesh' " << flush;
      }

      if (j<directions.Size())
      {
         delete mesh_orig;
         mesh_orig = mesh_ext;
      }
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      mesh_sock << "mesh\n" << *mesh_ext <<
      "window_title 'New Mesh' " << flush;
   }

   // 15. Free the used memory.
   delete mesh;
   return 0;
}
