//                                MFEM Example 1
//
// Compile with: make AddScwarz
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
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

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);

   Mesh * mesh; 
   // Define a simple square mesh
   if (sdim == 2)
   {
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true,1.0, 1.0,false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true,1.0, 1.0,1.0, false);
   }


   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      // int ref_levels =
      //    (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
       int ref_levels = 1;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // mesh->PrintInfo(cout);

   int nrelem = mesh->GetNE();
   int nrvert = mesh->GetNV();
   int nredge = mesh->GetNEdges();
   int nrface = mesh->GetNFaces();

   // 5. Define a finite element space on the mesh. 
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);


   // cout << "Element to dof table " << endl;
   // fespace->GetElementToDofTable().Print();

   // Array<int>edge_dofs;
   // Array<int>edge_vert;
   // for (int i=0; i< nredge; i++ )
   // {
   //    // fespace->GetEdgeDofs(i,edge_dofs);
   //    mesh->GetEdgeVertices(i,edge_vert);
   //    cout << "edge no " << i << " vertices :" ; edge_vert.Print();
   // }
   // Array<int>face_dofs;
   // Array<int>face_vert;
   // for (int i=0; i< nrface; i++ )
   // {
   //    // fespace->GetFaceDofs(i,face_dofs);
   //    mesh->GetFaceVertices(i,face_vert);
   //    // cout << " face no " << i << " dofs :" ; face_dofs.Print();
   //    cout << "face no " << i << " vertices :" ; face_vert.Print(); 
   // }
   // Array<int>elem_vert;
   // for (int i=0; i< nrelem; i++ )
   // {
   //    // fespace->GetFaceDofs(i,face_dofs);
   //    mesh->GetElementVertices(i,elem_vert);
   //    // cout << " face no " << i << " dofs :" ; face_dofs.Print();
   //    cout << "elem no " << i << " vertices :" ; elem_vert.Print();
   // }

   // construct a list of indices for each patch/vertex (that is not essential)
   // // Get essential
   // Array<int> ess_tdof_list;
   // if (mesh->bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(mesh->bdr_attributes.Max());
   //    ess_bdr = 1;
   //    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }
   // cout<< "essential boundary dofs: " ; ess_tdof_list.Print(); 



   Array<Array<int>> patch(nrvert);
   // Initialize each patch by an array consisting of the vertex its self
   // Numbering of vertices starts from 0
   for (int i=0; i<nrvert; i++)
   {
      Array<int> vert(1);
      vert=i;
      patch[i] = vert;
   }
   // Loop through all the edges and find the the vertices they contribute to
   Array<int>edge_vert;
   Array<int>edge_int_dofs;
   for (int i=0; i< nredge; i++ )
   {
      mesh->GetEdgeVertices(i,edge_vert);
      int nv = edge_vert.Size();
      fespace->GetEdgeInteriorDofs(i,edge_int_dofs);
      for (int j=0; j<nv ; j++)
      {
         int k = edge_vert[j];
         patch[k].Append(edge_int_dofs);
      }
   }
// Loop through all the faces and find the the vertices they contribute to
   Array<int>face_vert;
   Array<int>face_int_dofs;
   for (int i=0; i< nrface; i++ )
   {
      mesh->GetFaceVertices(i,face_vert); 
      int nv = face_vert.Size();
      fespace->GetFaceInteriorDofs(i,face_int_dofs);
      for (int j=0; j<nv ; j++)
      {
         int k = face_vert[j];
         patch[k].Append(face_int_dofs);
      }
   }
// Loop through all the elements and find the the vertices they contribute to
   Array<int>elem_vert;
   Array<int>elem_int_dofs;
   for (int i=0; i< nrelem; i++ )
   {
      mesh->GetElementVertices(i,elem_vert); 
      int nv = elem_vert.Size();
      fespace->GetElementInteriorDofs(i,elem_int_dofs);
      for (int j=0; j<nv ; j++)
      {
         int k = elem_vert[j];
         patch[k].Append(elem_int_dofs);
      }
   }
   for (int i=0; i<nrvert; i++)
   {
      cout << "Patch no: " << i << " dofs " ;  
      patch[i].Print();   
   }

   // Build a sparse matrix out of this map to extract the patch submatrix
   Array<SparseMatrix *> Pid(nrvert); 
   Array<int> dofoffset(nrvert);
   dofoffset = 0;
   for (int i=0; i<nrvert; i++)
   {
      int height = fespace->GetVSize();
      int width  = patch[i].Size();
      Pid[i] = new SparseMatrix(height,width);
      Pid[i]->Set(i,dofoffset[i],1.0); // Fill in the vertex dof (1 column for each vertex)
      dofoffset[i]++;
   }

   // Fill the sparse matrix with the edge dof indices (1 column for each dof)
   for (int i=0; i< nredge; i++ )
   {
      mesh->GetEdgeVertices(i,edge_vert);
      int nv = edge_vert.Size();
      fespace->GetEdgeInteriorDofs(i,edge_int_dofs);
      int ne = edge_int_dofs.Size();
      for (int j=0; j<nv ; j++)
      {
         int k = edge_vert[j];
         for (int l=0; l < ne; l++)
         {
            int m = edge_int_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }
   // Fill the sparse matrix with the face dof indices (1 column for each dof)
   for (int i=0; i< nrface; i++ )
   {
      mesh->GetEdgeVertices(i,face_vert);
      int nv = face_vert.Size();
      fespace->GetFaceInteriorDofs(i,face_int_dofs);
      int nf = face_int_dofs.Size();
      for (int j=0; j<nv ; j++)
      {
         int k = face_vert[j];
         for (int l=0; l < nf; l++)
         {
            int m = face_int_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }

   // Fill the sparse matrix with the element (middle) dof indices (1 column for each dof)
   for (int i=0; i< nrelem; i++ )
   {
      mesh->GetElementVertices(i,elem_vert);
      int nv = elem_vert.Size();
      fespace->GetElementInteriorDofs(i,elem_int_dofs);
      int nel = elem_int_dofs.Size();
      for (int j=0; j<nv ; j++)
      {
         int k = elem_vert[j];
         for (int l=0; l < nel; l++)
         {
            int m = elem_int_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }

   Pid[0]->Finalize();
   Pid[0]->PrintMatlab(cout);


   // SparseMatrix * S = new SparseMatrix(5,5);
   // // S->PrintMatlab(std::cout);
   // S->Set(1,1,1.0);
   // S->Finalize();
   // // S->Print(cout);
   // S->PrintMatlab(cout);


























   // cout << "Edges to vertex table " << endl;
   // mesh->GetEdgeVertexTable()->Print();


   // if (sdim == 3)
   // {
   //    cout << "Faces to edges table " << endl;
   //    mesh->GetFaceEdgeTable()->Print(); // is this really Face to Vertex table?
   // }


   // Custom vertex patch partitioning partitioning
   // Array<int> vertex_dofs;
   // for (int i=0; i< nrelems; i++ )
   // {
   //    fespace->GetElementVertices(i,vertex_dofs);
   //    std::cout << "Element "<< i+1 << " Vertex dofs: " ; vertex_dofs.Print();
   // }

   // This is local numbering of nodes
   // for (int i=0; i< nrelems; i++ )
   // {
   //    cout << "Element " << i+1 << " Number of vertices: " << 
   //                          mesh->GetElement(i)->GetNVertices() << endl;

   //    const int ne =  mesh->GetElement(i)->GetNEdges();
   //    for (int j=0; j< ne; j++ )
   //    {
   //       const int *ev = mesh->GetElement(i)->GetEdgeVertices(j);
   //       cout << "Edge " << j << " vertices " <<  ev[0] << ", "<< ev[1] << endl;
   //    }
   // }

   // Array<int> vertex_dofs;
   // fespace->GetElementToDofTable().Print();
   // for (int i=0; i< nrelems; i++ )
   // {
   //    std::cout << "Vertex dofs " << endl;
   //    fespace->GetElementVertices(i,vertex_dofs);
   //    vertex_dofs.Print();
   // }

 

   // Array<int> vertex_dofs;
   // Array<int> interior_dofs;
   // for (int i=0; i< nrelems; i++ )
   // {
   //    std::cout << "Element " << i+1 << endl;
   //    fespace->GetElementVertices(i,vertex_dofs);
   //    std::cout << "Vertex dofs " << endl;
   //    vertex_dofs.Print();
   //    fespace->GetElementInteriorDofs(i,interior_dofs);
   //    std::cout << "Interior dofs " << endl;
   //    interior_dofs.Print();


   // }

   // std::cout << "Number of global unknowns: " << fespace->GetVSize() << endl;
   // std::cout << "Number of vertex dofs " << fespace->GetNVDofs() << endl;
   // std::cout << "Number of edge   dofs " << fespace->GetNEDofs() << endl;
   // std::cout << "Number of face   dofs " << fespace->GetNFDofs() << endl;
   // std::cout << "Number of total  dofs " << fespace->GetNDofs() << endl;


//    cout << "Number of finite element unknowns: "
//         << fespace->GetTrueVSize() << endl;

//    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
//    //    In this example, the boundary conditions are defined by marking all
//    //    the boundary attributes from the mesh as essential (Dirichlet) and
//    //    converting them to a list of true dofs.
   // Array<int> ess_tdof_list;
   // if (mesh->bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(mesh->bdr_attributes.Max());
   //    ess_bdr = 1;
   //    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   // LinearForm *b = new LinearForm(fespace);
   // ConstantCoefficient one(1.0);
   // b->AddDomainIntegrator(new DomainLFIntegrator(one));
   // b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   // GridFunction x(fespace);
   // x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   // BilinearForm *a = new BilinearForm(fespace);
   // if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   // a->AddDomainIntegrator(new DiffusionIntegrator(one));
   // a->Assemble();

   // OperatorPtr A;
   // Vector B, X;
   // a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // cout << "Size of linear system: " << A->Height() << endl;


   // A->PrintMatlab(cout);
// #ifndef MFEM_USE_SUITESPARSE
//       // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
//       GSSmoother M((SparseMatrix&)(*A));
//       PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
// #else
//       // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
//       UMFPackSolver umf_solver;
//       umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//       umf_solver.SetOperator(*A);
//       umf_solver.Mult(B, X);
// #endif

//    // 12. Recover the solution as a finite element grid function.
//    a->RecoverFEMSolution(X, *b, x);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "mesh\n" << *mesh << flush;
   }

//    // 15. Free the used memory.
//    delete a;
//    delete b;
//    delete fespace;
//    if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
