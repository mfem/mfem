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

   std::vector<Mesh *> meshes(ref_levels+1);
   std::vector<Array<Embedding>> embeddings(ref_levels);

   FiniteElementCollection *fec = new H1_FECollection(1, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

   std::vector<SparseMatrix*>  P(ref_levels);

   // 4. Refine the mesh 
   for (int i = 0; i < ref_levels; i++)
   {
      meshes[i] = new Mesh(*mesh); 
      const FiniteElementSpace cfespace(*fespace);
      mesh->UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::MFEM_SPARSEMAT);
      fespace->GetTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      Tr.Get(P[i]);
   }
      meshes[ref_levels] = new Mesh(*mesh); 

    
    SparseMatrix * AP  = Mult(*P[1], *P[0]);

    // Array<
    // Transpose(*P[0])->;
    AP->Threshold(1e-12);
    Transpose(*AP)-> Print(cout);
    
    int nz = Transpose(*AP)->RowSize(0);
    int *col = Transpose(*AP)->GetRowColumns(0);

    cout << "dofs in the patch : " << endl;
    for (int i = 0; i<nz-1; i++)
    {
        cout << col[i] << ", " ;
    }
        cout << col[nz-1] << endl ;



    nz = AP->RowSize(21);
    col = AP->GetRowColumns(21);

    cout << "Vertex 0 contributes to : " << endl;
    for (int i = 0; i<nz-1; i++)
    {
        cout << col[i] << ", " ;
    }
        cout << col[nz-1] << endl ;


    FiniteElementCollection *ndfec = new  ND_FECollection(order,dim);
    FiniteElementSpace *ND_fespace = new FiniteElementSpace(mesh, ndfec);

    Array<int> vertex_dofs;
    ND_fespace->GetVertexDofs(0,vertex_dofs);
    vertex_dofs.Print(cout);
    cout<< "vertex_dofs_size = "<< vertex_dofs.Size() << endl;

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