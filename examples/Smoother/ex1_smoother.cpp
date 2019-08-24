//                                MFEM Example 1
//
// Compile with: make ex1
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


void print(std::vector<int> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input[i] << ' ';
	}
   std::cout << endl;
}

struct patch_nod_info 
{
   int nrpatch;
   int ref_levels;
   Mesh mesh;
   // constructor
   patch_nod_info(Mesh * mesh_, int ref_levels_);
private:

};
// constructor
patch_nod_info::patch_nod_info(Mesh * mesh_, int ref_levels_) 
               : mesh(*mesh_), ref_levels(ref_levels_) 
{
   /* The patches are define by all the "active" vertices of the coarse mesh
      We define a low order H1 fespace and perform refinents so that we can get
      the H1 prolongation operator recursively. This way we can easily find  
      all the patches that the fine mesh vertices contribute to. After the vertices 
      are done we the edges, faces and elements that can be found easily because they
      contribute to the same patches as their vertices. */

      // Numver of patches
      nrpatch = mesh.GetNV();
      int dim = mesh.Dimension();
      FiniteElementCollection *fec = new H1_FECollection(1, dim);
      FiniteElementSpace *fespace = new FiniteElementSpace(&mesh, fec);

      SparseMatrix * Pr = nullptr;
// 4. Refine the mesh 
   for (int i = 0; i < ref_levels; i++)
   {
      const FiniteElementSpace cfespace(*fespace);
      mesh.UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::MFEM_SPARSEMAT);
      fespace->GetTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      SparseMatrix * P;
      Tr.Get(P);
      if (!Pr)
      {
         Pr = P;
      }
      else
      {
         Pr = Mult(*P,*Pr);
      }
   }
   Pr->Threshold(0.0);
   int nvert = mesh.GetNV();
   vector<Array<int>> vertex_contr;
   vertex_contr.resize(nvert);
   for (int iv = 0; iv< nvert; iv++)
   {
      int nz = Pr->RowSize(iv);
      vertex_contr[iv].SetSize(nz);
      int *col = Pr->GetRowColumns(iv);
      for (int i = 0; i<nz; i++)
      {
         vertex_contr[iv][i] = col[i];   
      }
      cout << "Fine vertex no: " << iv << " contributes to patches: " ;
      vertex_contr[iv].Print();
   }

   vector<Array<int>> edge_contr;
   Array<int> edge_vertices;
   int nedge = mesh.GetNE();
   edge_contr.resize(nedge);
   for (int ie=0; ie< nedge; ie++ )
   {
      mesh.GetEdgeVertices(ie,edge_vertices);
      int nv = edge_vertices.Size(); // always 2 but ok
      cout << "Fine edge no: " << ie << " vertices: " ; edge_vertices.Print()  ;

      // The edge will contribute to the same patches as its vertices
      for(int iv=0; iv< nv; iv++)
      {
         int ivert = edge_vertices[iv];
         edge_contr[ie].Append(vertex_contr[ivert]);
      }
      cout << "Fine edge no: " << ie << " contributes to patches: " ;
      edge_contr[ie].Sort(); edge_contr[ie].Unique();
      edge_contr[ie].Print();
   }

   vector<Array<int>> face_contr;
   Array<int> face_vertices;
   int nface = mesh.GetNFaces();
   face_contr.resize(nface);
   for (int ifc=0; ifc< nface; ifc++ )
   {
      mesh.GetFaceVertices(ifc,face_vertices);
      int nv = face_vertices.Size(); 
      cout << "Fine face no: " << ifc << " vertices: " ; face_vertices.Print()  ;

      // The face will contribute to the same patches as its vertices
      for(int iv=0; iv< nv; iv++)
      {
         int ivert = face_vertices[iv];
         face_contr[ifc].Append(vertex_contr[ivert]);
      }
      cout << "Fine face no: " << ifc << " contributes to patches: " ;
      face_contr[ifc].Sort(); face_contr[ifc].Unique();
      face_contr[ifc].Print();
   }

   vector<Array<int>> elem_contr;
   Array<int> elem_vertices;
   int nelem = mesh.GetNE();
   elem_contr.resize(nelem);
   for (int iel=0; iel< nelem; iel++ )
   {
      mesh.GetElementVertices(iel,elem_vertices);
      int nv = elem_vertices.Size(); 

      // The edge will contribute to the same patches as its vertices
      for(int iv=0; iv< nv; iv++)
      {
         int ivert = elem_vertices[iv];
         elem_contr[iel].Append(vertex_contr[ivert]);
      }
      cout << "Fine elem no: " << iel << " contributes to patches: " ;
      elem_contr[iel].Sort(); elem_contr[iel].Unique();
      elem_contr[iel].Print();
   }


}




int dim;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/one-hex.mesh";
   int order = 1;
   int sdim = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int ref_levels = 1;
   int initref    = 1;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
   args.AddOption(&ref_levels, "-sr", "--serial-refinements", "Number of mesh refinements");
   args.AddOption(&initref, "-iref", "--init-refinements", "Number of initial mesh refinements");
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
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. 
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);

   Mesh * mesh; 
   // Define a simple square or cubic mesh
   if (sdim == 2)
   {
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true,1.0, 1.0,false);
      // mesh = new Mesh(1, 1, Element::TRIANGLE, true,1.0, 1.0,false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true,1.0, 1.0,1.0, false);
   }
   dim = mesh->Dimension();
   for (int i=0; i<initref; i++) {mesh->UniformRefinement();}


   Mesh * cmesh = new Mesh(*mesh);

   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }


   // Mesh * mesh_copy = new Mesh(*cmesh); 

   // patch_nod_info * P = new patch_nod_info(mesh_copy,ref_levels);
   patch_nod_info * P = new patch_nod_info(cmesh,ref_levels);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *cmesh << flush;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }


   // 15. Free the used memory.
   delete mesh;

   return 0;
}
