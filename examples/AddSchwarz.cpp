//                                MFEM Example 1
//
// Compile with: make AddScwarz
//
// Example run: ./AddSchwarz -sr 4 -o 2 -d 2

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class AddSchwarzSmoother : public Solver {
private:
   /// The linear system matrix
   SparseMatrix * A;
   Array<SparseMatrix *> Pid; 
   Array<SparseMatrix *> A_local;
   Array<UMFPackSolver *> invA_local;
   int nrvert;
   Array<int>vert_dofs;

public:
   AddSchwarzSmoother(SparseMatrix *A_, FiniteElementSpace *fespace);

   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~AddSchwarzSmoother() {}
};
// constructor
AddSchwarzSmoother::AddSchwarzSmoother(SparseMatrix * A_, FiniteElementSpace *fespace)
   : Solver(A_->Height(), A_->Width()), A(A_) 
{

   Array<int> ess_tdof_list;
   if (fespace->GetMesh()->bdr_attributes.Size())
   {
      Array<int> ess_bdr(fespace->GetMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   nrvert = fespace->GetMesh()->GetNV();      
   int nredge = fespace->GetMesh()->GetNEdges();
   int nrface = fespace->GetMesh()->GetNFaces();
   int nrelem = fespace->GetMesh()->GetNE();
   Array<int>edge_vert;
   Array<int>edge_int_dofs;
   Array<int>face_vert;
   Array<int>face_int_dofs;
   Array<int>elem_vert;
   Array<int>elem_int_dofs;

   vert_dofs.SetSize(0);
   for (int i=0; i<nrvert; i++)
   {
      int j = ess_tdof_list.FindSorted(i);
      if (j == -1 ) 
         vert_dofs.Append(i);
   }

   // Build a sparse matrix out of this map to extract the patch submatrix
   Pid.SetSize(nrvert);
   Array<int> dofoffset(nrvert);
   dofoffset = 0;
   for (int i=0; i<nrvert; i++)
   {
      int height = fespace->GetVSize();
      Pid[i] = new SparseMatrix(height);
      // skip if its a dirichlet vertex
      // int m = vert_dofs[i];
      Pid[i]->Set(i,dofoffset[i],1.0); // Fill in the vertex dof (1 column for each vertex)
      dofoffset[i]++;
   }

   // Fill the sparse matrix with the edge dof indices (1 column for each dof)
   for (int i=0; i< nredge; i++ )
   {
      fespace->GetMesh()->GetEdgeVertices(i,edge_vert);
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
      fespace->GetMesh()->GetEdgeVertices(i,face_vert);
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
      fespace->GetMesh()->GetElementVertices(i,elem_vert);
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

   A_local.SetSize(nrvert);
   invA_local.SetSize(nrvert);
   for (int i=0; i< nrvert; i++ )
   {
      Pid[i]->SetWidth(dofoffset[i]);
      Pid[i]->Finalize();

      // construct the local problems. Factor the patch matrices
      A_local[i] = RAP(*Pid[i],*A,*Pid[i]);
      invA_local[i] = new UMFPackSolver;
      invA_local[i]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      invA_local[i]->SetOperator(*A_local[i]);   
   }
}


void AddSchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   // Apply the smoother patch on the restriction of the residual
   Array<Vector> res_local(nrvert);
   Array<Vector> sol_local(nrvert);
   Array<Vector> zaux(nrvert);
   z = 0.0;
   for (int i=0; i< nrvert; i++)
   // for (int ii=0; ii< vert_dofs.Size(); ii++)  // loop through only the patches associate with a non-dirichlet vertex
   {
      // int i = vert_dofs[ii];
      res_local[i].SetSize(Pid[i]->NumCols()); 
      sol_local[i].SetSize(Pid[i]->NumCols()); 
      Pid[i]->MultTranspose(r,res_local[i]);
      invA_local[i]->Mult(res_local[i],sol_local[i]);
      zaux[i].SetSize(r.Size()); zaux[i]=0.0; 
      Pid[i]->Mult(sol_local[i],zaux[i]); zaux[i] *= 0.5;
      z += zaux[i];
   }
}

void get_solution(const Vector &x, double & u, double & d2u);
double u_exact(const Vector &x);
double f_exact(const Vector &x);
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


   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sdim, "-d", "--dimension", "Dimension");
   args.AddOption(&ref_levels, "-sr", "--serial-refinements", "Number of mesh refinements");
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
   // Define a simple square mesh
   if (sdim == 2)
   {
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true,1.0, 1.0,false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true,1.0, 1.0,1.0, false);
   }
   dim = mesh->Dimension();

   // 4. Refine the mesh 
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. 
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
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   GridFunction x(fespace);
   x = 0.0;
   FunctionCoefficient u_ex(u_exact);
   x.ProjectCoefficient(u_ex);

   // 9. Set up the bilinear form a(.,.) 
   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "Size of linear system: " << A.Height() << endl;


   AddSchwarzSmoother *prec = new AddSchwarzSmoother(&A,fespace);
   // DSmoother *prec = new DSmoother(A);
   // GSSmoother *prec = new GSSmoother(A);

   int maxit(100);
   double rtol(1.e-6);
   double atol(0.0);
   CGSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxit);
   solver.SetOperator(A);
   solver.SetPreconditioner(*prec);
   solver.SetPrintLevel(1);
   solver.Mult(B,X);

   // 12. Recover the solution as a finite element grid function.
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



   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      // sol_sock << "mesh\n" << *mesh << flush;
      sol_sock << "solution\n" << *mesh << x << flush;
   }

//    // 15. Free the used memory.
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
      u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]);
      d2u = -2.0* ( x[1]*(1.0 - x[1]) + x[0]*(1.0 - x[0])); 
   }
   else
   {
      u = x[0]*(1.0 - x[0]) * x[1]*(1.0 - x[1]) * x[2]*(1.0 - x[2]);
      d2u = -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[1]) * x[1] 
         -2.0*(-1.0 + x[0]) * x[0] * (-1.0 + x[2]) * x[2] 
         -2.0*(-1.0 + x[1]) * x[1] * (-1.0 + x[2]) * x[2];
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
   return -d2u;
}