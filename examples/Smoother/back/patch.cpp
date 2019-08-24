//                                MFEM Example 1
//
// Compile with: make AddScwarz
//
// Example run: ./AddSchwarz -sr 4 -o 2 -d 2

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>    // std::sort
using namespace std;
using namespace mfem;

vector <int> getIntersection(vector < vector <int> > &sets);

void print(std::vector<int> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input[i] << ' ';
	}
   std::cout << endl;
}

struct elem_contr
{
   vector<int> patches;
};


// struct that store the vertex patch contributions for each element in the current mesh
struct patch_info 
{
   int nrpatch;
   int nrmeshes;
   vector<elem_contr> patch_elem_list;
   std::vector<Mesh *> meshes;
   std::vector<Array<Embedding> > embeddings;
   // constructor
   patch_info(std::vector<Mesh *> meshes, std::vector<Array<Embedding> > embeddings);
private:
   CoarseFineTransformations cf_tr ; 
   int get_coarse_parent(int iel);

};
// constructor
patch_info::patch_info(std::vector<Mesh *> meshes_,std::vector<Array<Embedding> > embeddings_) 
          : meshes(meshes_),  embeddings(embeddings_)
{
   // The patch unique id is the vertex number which remains unchanged after refinement
   
   nrmeshes = meshes.size();

   Mesh * cmesh = meshes[0];
   Mesh * mesh = meshes[nrmeshes-1];
   nrpatch = cmesh->GetNV();

   // Construct the a patch list for the fine grid elements
   int nrelem = mesh->GetNE();
   patch_elem_list.resize(nrelem);
   for (int i=0; i<nrelem ; i++)
   {
      int parent = get_coarse_parent(i);
      /* Find the patches that the parent belongs to and 
         copy them to the patch list of the child */  
      Array<int>parent_vert;
      cmesh->GetElementVertices(parent,parent_vert);
      int nv = parent_vert.Size();
      patch_elem_list[i].patches.resize(nv);
      for (int j=0; j<nv; j++)
      {
         patch_elem_list[i].patches[j] = parent_vert[j];
      }
      // std::sort(patch_elem_list[i].patches.begin(),patch_elem_list[i].patches.end());
      // cout << "fine element no: " << i << " belongs to patches: ";
      // print(patch_elem_list[i].patches);
   }

   // Now we can take care of the vertices
   // int nrvert = mesh->GetNV();
   // // loop through all the vertices that are not defining their own patch
   // Table * Vert2elem = mesh->GetVertexToElementTable();
   // Table * Vert2edge = mesh->GetEdgeVertexTable();

   // Vert2edge->Print(cout);
   // for (int iv=nrpatch-1; iv< nrvert; iv++)
   // {
   //    //Find their "face elements"
   // }

//   std::cout << "myvector contains:";
//   for (std::vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
//     std::cout << ' ' << *it;
//   std::cout << '\n';
   // Now we are ready to determine the patches of the vertices


};

int patch_info:: get_coarse_parent(int iel)
{
   // First find how many meshes involved

   int parent = iel;
   int nrmeshes = meshes.size();
   // need to check if this is going to work for amr
   for (int i = nrmeshes-2; i>=0; i--)
   {
      parent = embeddings[i][parent].parent;
   }
   return parent;
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

   // Mesh * cmesh = new Mesh(*mesh);

   std::vector<Mesh *> meshes(ref_levels+1);
   std::vector<Array<Embedding>> embeddings(ref_levels);

   // 4. Refine the mesh 
   for (int l = 0; l < ref_levels; l++)
   {
      meshes[l]=new Mesh(*mesh);
      mesh->UniformRefinement();
      embeddings[l] = mesh->GetRefinementTransforms().embeddings;
   }
   meshes[ref_levels]=new Mesh(*mesh);


   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);


   patch_info * test = new patch_info(meshes, embeddings);

   Array<int> vert;
   for (int i=0; i<meshes[0]->GetNE(); i++)
   {
      meshes[0]->GetEdgeVertices(i,vert);
      cout << "Edge no: " << i << " vertices: "; vert.Print();
   }

   // int nrelesC = cfespace->GetMesh()->GetNE();
   // for (int i=0; i<nrelesC ; i++)
   // {
   //    Array<int> elem_dofs;   
   //    cfespace->GetElementDofs(i,elem_dofs);
   //    // cout << "Element " << i << ": " ; elem_dofs.Print(cout);
   // }


   // int nreles = fespace->GetMesh()->GetNE();
   // for (int i=0; i<nreles ; i++)
   // {
   //    Array<int> elem_dofs;   
   //    fespace->GetElementDofs(i,elem_dofs);
   //    // cout << "Element " << i << ": " ; elem_dofs.Print(cout);
   // }


   // Plan: 
   // 1) from coarse2fine transformation make a list of interior elements in the patch.
         // done
   
   // 2) identify if a vertex is in the patch by checking that all the elements it belongs to are in the interior of the patch.
      /*
      a) for each patch 
         b) for each element in the patch
            c) for each vertex in the element
               d) find the neighboring elements of the vertex and 
                  check if they are all in the patch
                  if yes then the vertex is in the patch 
      */
     // or A vertex can contribute to the same patches that its elements contribute (no other)
     /* So...
        a) for each (non dirichlet) vertex find its elements
           b) find the patches lists that this elements contribute 
              c) the vertex will contribute to the intersection of these lists
            // note: This will imply non dirichlet bc on the boundary of the patch in case the problem has 
            
     */

    // for each vertex find its "face" elements
    // find where the faces belong to by checking there neighboring elements. 
    // the face will belong to the intersection of the lists of patches. 
    // the vertex will belong to the intercection of the intercections. 
    // if the interesection is empty then the vertex is in its own patch
    // for an edge 




   // 3) Edges and faces can then be checked if they are interior or not by checking their vertices. 
   // 4) It seems that the vertex numbering is not changing after a refinement. 




   // 5. Define a finite element space on the mesh. 

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   // Array<int> ess_tdof_list;
   // if (mesh->bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(mesh->bdr_attributes.Max());
   //    ess_bdr = 1;
   //    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   // // 7. Set up the linear form b(.) 
   // LinearForm *b = new LinearForm(fespace);
   // ConstantCoefficient one(1.0);
   // b->AddDomainIntegrator(new DomainLFIntegrator(one));
   // b->Assemble();

   // // 8. Define the solution vector x as a finite element grid function
   // GridFunction x(fespace);
   // x = 0.0;

   // // 9. Set up the bilinear form a(.,.) 
   // BilinearForm *a = new BilinearForm(fespace);
   // a->AddDomainIntegrator(new DiffusionIntegrator(one));
   // a->Assemble();

   // SparseMatrix A;
   // Vector B, X;
   // a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   // cout << "Size of linear system: " << A.Height() << endl;

   // patch *P = new patch(cfespace,fespace);


   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *meshes[0] << flush;
   }

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *fespace->GetMesh() << flush;
   }

   // delete fespace;
   // delete mesh;
   return 0;
}








// The main function that receives a set of sets as parameter and 
// returns a set containing intersection of all sets 
vector <int> getIntersection(vector < vector <int> > &sets) 
{ 
    vector <int> result;  // To store the reaultant set 
    int smallSetInd = 0;  // Initialize index of smallest set 
    int minSize = sets[0].size(); // Initialize size of smallest set 
  
    // sort all the sets, and also find the smallest set 
    for (int i = 1 ; i < sets.size() ; i++) 
    { 
        // sort this set 
        sort(sets[i].begin(), sets[i].end()); 
  
        // update minSize, if needed 
        if (minSize > sets[i].size()) 
        { 
            minSize = sets[i].size(); 
            smallSetInd = i; 
        } 
    } 
  
    map<int,int> elementsMap; 
  
    // Add all the elements of smallest set to a map, if already present, 
    // update the frequency 
    for (int i = 0; i < sets[smallSetInd].size(); i++) 
    { 
        if (elementsMap.find( sets[smallSetInd][i] ) == elementsMap.end()) 
            elementsMap[ sets[smallSetInd][i] ] = 1; 
        else
            elementsMap[ sets[smallSetInd][i] ]++; 
    } 
  
    // iterate through the map elements to see if they are present in 
    // remaining sets 
    map<int,int>::iterator it; 
    for (it = elementsMap.begin(); it != elementsMap.end(); ++it) 
    { 
        int elem = it->first; 
        int freq = it->second; 
  
        bool bFound = true; 
  
        // Iterate through all sets 
        for (int j = 0 ; j < sets.size() ; j++) 
        { 
            // If this set is not the smallest set, then do binary search in it 
            if (j != smallSetInd) 
            { 
                // If the element is found in this set, then find its frequency 
                if (binary_search( sets[j].begin(), sets[j].end(), elem )) 
                { 
                   int lInd = lower_bound(sets[j].begin(), sets[j].end(), elem) 
                                                            - sets[j].begin(); 
                   int rInd = upper_bound(sets[j].begin(), sets[j].end(), elem) 
                                                            - sets[j].begin(); 
  
                   // Update the minimum frequency, if needed 
                   if ((rInd - lInd) < freq) 
                       freq = rInd - lInd; 
                } 
                // If the element is not present in any set, then no need  
                // to proceed for this element. 
                else
                { 
                    bFound = false; 
                    break; 
                } 
            } 
        } 
  
        // If element was found in all sets, then add it to result 'freq' times 
        if (bFound) 
        { 
            for (int k = 0; k < freq; k++) 
                result.push_back(elem); 
        } 
    } 
    return result; 
} 