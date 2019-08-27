//                                MFEM Example 1
//
// Compile with: make ex1
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

struct patch_nod_info 
{
   int nrpatch;
   vector<Array<int>> vertex_contr;
   vector<Array<int>> edge_contr;
   vector<Array<int>> face_contr;
   vector<Array<int>> elem_contr;

   // constructor
   patch_nod_info(Mesh * mesh_, int ref_levels_);
private:
   int ref_levels=0;;
   Mesh *mesh=nullptr;
};
// constructor
patch_nod_info::patch_nod_info(Mesh * mesh_, int ref_levels_) 
               : mesh(mesh_), ref_levels(ref_levels_) 
{
   /* The patches are define by all the "active" vertices of the coarse mesh
      We define a low order H1 fespace and perform refinents so that we can get
      the H1 prolongation operator recursively. This way we can easily find  
      all the patches that the fine mesh vertices contribute to. After the vertices 
      are done the edges, faces and elements can be found easily because they
      contribute to the same patches as their vertices. */

      // Numver of patches
      nrpatch = mesh->GetNV();
      int dim = mesh->Dimension();
      FiniteElementCollection *fec = new H1_FECollection(1, dim);
      FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

      SparseMatrix * Pr = nullptr;
// 4. Refine the mesh 
   for (int i = 0; i < ref_levels; i++)
   {
      const FiniteElementSpace cfespace(*fespace);
      mesh->UniformRefinement();
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
   int nvert = mesh->GetNV();
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
      // cout << "Fine vertex no: " << iv << " contributes to patches: " ;
      // vertex_contr[iv].Print();
   }

   Array<int> edge_vertices;
   int nedge = mesh->GetNEdges();
   edge_contr.resize(nedge);
   for (int ie=0; ie< nedge; ie++ )
   {
      mesh->GetEdgeVertices(ie,edge_vertices);
      int nv = edge_vertices.Size(); // always 2 but ok
      // The edge will contribute to the same patches as its vertices
      for(int iv=0; iv< nv; iv++)
      {
         int ivert = edge_vertices[iv];
         edge_contr[ie].Append(vertex_contr[ivert]);
      }
      edge_contr[ie].Sort(); edge_contr[ie].Unique();
      // cout << "Fine edge no: " << ie << " contributes to patches: " ;
      // edge_contr[ie].Print();
   }

   Array<int> face_vertices;
   int nface = mesh->GetNFaces();
   face_contr.resize(nface);
   for (int ifc=0; ifc< nface; ifc++ )
   {
      mesh->GetFaceVertices(ifc,face_vertices);
      int nv = face_vertices.Size(); 
      // The face will contribute to the same patches as its vertices
      for(int iv=0; iv< nv; iv++)
      {
         int ivert = face_vertices[iv];
         face_contr[ifc].Append(vertex_contr[ivert]);
      }
      face_contr[ifc].Sort(); face_contr[ifc].Unique();
      // cout << "Fine face no: " << ifc << " contributes to patches: " ;
      // face_contr[ifc].Print();
   }


   Array<int> elem_vertices;
   int nelem = mesh->GetNE();
   elem_contr.resize(nelem);
   for (int iel=0; iel< nelem; iel++ )
   {
      mesh->GetElementVertices(iel,elem_vertices);
      int nv = elem_vertices.Size(); 
      // The element will contribute to the same patches as its vertices
      for(int iv=0; iv< nv; iv++)
      {
         int ivert = elem_vertices[iv];
         elem_contr[iel].Append(vertex_contr[ivert]);
      }
      elem_contr[iel].Sort(); elem_contr[iel].Unique();
      // cout << "Fine elem no: " << iel << " contributes to patches: " ;
      // elem_contr[iel].Print();
   }
   delete fespace;
   delete fec;
}


struct patch_assembly 
{
   int nrpatch;
   int ref_levels;
   Mesh cmesh;
   Array<SparseMatrix *> Pid; 
   // constructor
   patch_assembly(Mesh * cmesh_, int ref_levels_,FiniteElementSpace *fespace);
private:
};

patch_assembly::patch_assembly(Mesh * cmesh_, int ref_levels_,FiniteElementSpace *fespace) 
               : cmesh(*cmesh_), ref_levels(ref_levels_) 
{

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   patch_nod_info *patches = new patch_nod_info(&cmesh,ref_levels);
   chrono.Stop();
   cout << "Vertex patch info time " << chrono.RealTime() << "s. \n";


   nrpatch = patches->nrpatch;
   Pid.SetSize(nrpatch);
   // Build a sparse matrix out of this map to extract the patch submatrix
   Array<int> dofoffset(nrpatch); dofoffset = 0;
   int height = fespace->GetVSize();
   // allocation of sparse matrices.
   for (int i=0; i<nrpatch; i++)
   {
      Pid[i] = new SparseMatrix(height);
   }
   // Now the filling of the matrices with vertex,edge,face,interior dofs
   Mesh * mesh = fespace->GetMesh();
   int nrvert = mesh->GetNV();
   int nredge = mesh->GetNEdges();
   int nrface = mesh->GetNFaces();
   int nrelem = mesh->GetNE();
   // First the vertices
   chrono.Clear();
   chrono.Start();
   for (int i=0; i< nrvert; i++ )
   {
      int np = patches->vertex_contr[i].Size();
      Array<int> vertex_dofs;
      fespace->GetVertexDofs(i,vertex_dofs);
      int nv = vertex_dofs.Size();

      for (int j=0; j<np ; j++)
      {
         int k = patches->vertex_contr[i][j];
         for (int l=0; l < nv; l++)
         {
            int m = vertex_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }
   chrono.Stop();
   cout << "Vertex dofs time " << chrono.RealTime() << "s. \n";
   
   chrono.Clear();
   chrono.Start();
    // Edges
   for (int i=0; i< nredge; i++ )
   {
      int np = patches->edge_contr[i].Size();
      Array<int> edge_dofs;
      fespace->GetEdgeInteriorDofs(i,edge_dofs);
      int ne = edge_dofs.Size();
      for (int j=0; j<np ; j++)
      {
         // cout << "j = " << j << endl;
         int k = patches->edge_contr[i][j];
         for (int l=0; l < ne; l++)
         {
            int m = edge_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }
   chrono.Stop();
   cout << "Edge dofs time " << chrono.RealTime() << "s. \n";
   
   chrono.Clear();
   chrono.Start();   
   
   // Faces
   for (int i=0; i< nrface; i++ )
   {
      int np = patches->face_contr[i].Size();
      Array<int> face_dofs;
      fespace->GetFaceInteriorDofs(i,face_dofs);
      int nfc = face_dofs.Size();
      for (int j=0; j<np ; j++)
      {
         // cout << "j = " << j << endl;
         int k = patches->face_contr[i][j];
         for (int l=0; l < nfc; l++)
         {
            int m = face_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }

   chrono.Stop();
   cout << "Face dofs time " << chrono.RealTime() << "s. \n";

   chrono.Clear();
   chrono.Start();   

   // The following can be skipped in case of static condensation
   // Elements
   for (int i=0; i< nrelem; i++ )
   {
      int np = patches->elem_contr[i].Size();
      Array<int> elem_dofs;
      fespace->GetElementInteriorDofs(i,elem_dofs);
      int nel = elem_dofs.Size();
      for (int j=0; j<np ; j++)
      {
         // cout << "j = " << j << endl;
         int k = patches->elem_contr[i][j];
         for (int l=0; l < nel; l++)
         {
            int m = elem_dofs[l];
            Pid[k]->Set(m,dofoffset[k],1.0);
            dofoffset[k]++;
         }
      }
   }
   chrono.Stop();
   cout << "Elem dofs time " << chrono.RealTime() << "s. \n";


   chrono.Clear();
   chrono.Start();   

   for (int i=0; i< nrpatch; i++ )
   {
      Pid[i]->SetWidth(dofoffset[i]);
      Pid[i]->Finalize();
   }   

   chrono.Stop();
   cout << "Sparse matrix finalize time " << chrono.RealTime() << "s. \n";
}

class AddSchwarzSmoother : public Solver {
private:
   int nrpatch;
   /// The linear system matrix
   SparseMatrix * A;
   patch_assembly * P;
   Array<SparseMatrix  *> A_local;
   Array<UMFPackSolver *> invA_local;
   Array<int>vert_dofs;


public:
   AddSchwarzSmoother(Mesh * cmesh_, int ref_levels_, FiniteElementSpace *fespace,SparseMatrix *A_);

   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~AddSchwarzSmoother() {}
};
// constructor
AddSchwarzSmoother::AddSchwarzSmoother(Mesh * cmesh_, int ref_levels_, FiniteElementSpace *fespace_,SparseMatrix *A_)
   : Solver(A_->Height(), A_->Width()), A(A_) 
{

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();   
   P = new patch_assembly(cmesh_,ref_levels_, fespace_);
   chrono.Stop();
   cout << "Total patch dofs info time " << chrono.RealTime() << "s. \n";


   nrpatch = P->nrpatch;
   A_local.SetSize(nrpatch);
   invA_local.SetSize(nrpatch);

   chrono.Clear();
   chrono.Start();   

   for (int i=0; i< nrpatch; i++ )
   {
      SparseMatrix * Pr = P->Pid[i];
      // construct the local problems. Factor the patch matrices
      A_local[i] = RAP(*Pr,*A,*Pr);
      invA_local[i] = new UMFPackSolver;
      invA_local[i]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      invA_local[i]->SetOperator(*A_local[i]);   
   }

   chrono.Stop();
   cout << "Matrix extraction and setting up UMFPACK time " << chrono.RealTime() << "s. \n";


}

void AddSchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   // Apply the smoother patch on the restriction of the residual
   Array<Vector> res_local(nrpatch);
   Array<Vector> sol_local(nrpatch);
   Array<Vector> zaux(nrpatch);
   z = 0.0;
   for (int i=0; i< nrpatch; i++)
   {
      SparseMatrix * Pr = P->Pid[i];
      res_local[i].SetSize(Pr->NumCols()); 
      sol_local[i].SetSize(Pr->NumCols()); 
      Pr->MultTranspose(r,res_local[i]);
      invA_local[i]->Mult(res_local[i],sol_local[i]);
      zaux[i].SetSize(r.Size()); zaux[i]=0.0; 
      Pr->Mult(sol_local[i],zaux[i]); zaux[i] *= 0.5;
      z += zaux[i];
   }
}

void get_solution(const Vector &x, double & u, double & d2u);
double u_exact(const Vector &x);
double f_exact(const Vector &x);

int isol=0;
int dim;
double omega;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/one-hex.mesh";
   int order = 1;
   int sdim = 1;
   bool static_cond = false;
   const char *device_config = "cpu";
   bool visualization = true;
   int ref_levels = 1;
   int initref    = 1;
   // number of wavelengths
   double k = 0.5;
   StopWatch chrono;


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
   args.AddOption(&ref_levels, "-sr", "--serial-refinements", 
                  "Number of mesh refinements");
   args.AddOption(&initref, "-iref", "--init-refinements", 
                  "Number of initial mesh refinements");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&isol, "-sol", "--solution", 
                  "Exact Solution: 0) Polynomial, 1) Sinusoidal.");               
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // Angular frequency
   omega = 2.0 * M_PI * k;

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


   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);



   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) 
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient f(f_exact);
   b->AddDomainIntegrator(new DomainLFIntegrator(f));
   // b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   GridFunction x(fespace);
   x = 0.0;
   FunctionCoefficient u_ex(u_exact);
   x.ProjectCoefficient(u_ex);

   // 9. Set up the bilinear form a(.,.) 
   ConstantCoefficient sigma(-pow(omega, 2));

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->AddDomainIntegrator(new MassIntegrator(sigma));
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "Size of linear system: " << A.Height() << endl;

   FiniteElementSpace *prec_fespace = (a->StaticCondensationIsEnabled() ? a->SCFESpace() : fespace);
   Array<int>elem_dofs;
   prec_fespace->GetElementInteriorDofs(0,elem_dofs);
   elem_dofs.Print();
   chrono.Clear();
   chrono.Start();
   AddSchwarzSmoother * prec = new AddSchwarzSmoother(cmesh,ref_levels, prec_fespace, &A);
   chrono.Stop();
   // Need to invasticate the time scalings. TODO
   cout << "Smoother construction time " << chrono.RealTime() << "s. \n";
   
   int maxit(100);
   double rtol(0.0);
   double atol(1.e-10);
   // CGSolver solver;
   GMRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxit);
   solver.SetOperator(A);
   solver.SetPreconditioner(*prec);
   solver.SetPrintLevel(1);
   solver.Mult(B,X);



   a->RecoverFEMSolution(X, *b, x);


   GridFunction ugf(fespace);
   ugf.ProjectCoefficient(u_ex);


   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double L2error = x.ComputeL2Error(u_ex);
      
   cout << " || u_h - u ||_{L^2} = " << L2error <<  endl;



   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      if (dim == 2) 
      {
         sol_sock <<  "solution\n" << *mesh << x  << "keys rRljc\n" << flush;
      }
      else
      {
         sol_sock <<  "solution\n" << *mesh << x  << "keys lc\n" << flush;
      }
   }




   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "mesh\n" << *cmesh << flush;
   // }

   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "mesh\n" << *mesh << flush;
   // }


   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   delete mesh;
   return 0;
}


void get_solution(const Vector &x, double & u, double & d2u)
{

   if (dim == 2)
   {
      if (isol == 0)
      {
         u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
         d2u = -2.0* ( x[1]*(1.0 - x[1]) + x[0]*(1.0 - x[0])); 
      }
      else
      {
         double alpha = omega / sqrt(2.0);
         u = cos(alpha * (x[0] + x[1]));
         d2u = -2.0* alpha * alpha * u;
      }
   }
   else
   {
      if (isol == 0)
      {
         u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
         d2u = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1] 
            -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2] 
            -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
      }
      else
      {
         double alpha = omega / sqrt(3.0);
         u = cos(alpha * (x[0] + x[1] + x[2]));
         d2u = -3.0* alpha * alpha * u;
      }
   }
}

double u_exact(const Vector &x)
{
   double u, d2u;
   get_solution(x, u, d2u);
   return u;
}

double f_exact(const Vector &x)
{
   double u, d2u;
   get_solution(x, u, d2u);
   // return -d2u;
   return -d2u - omega*omega * u;
}