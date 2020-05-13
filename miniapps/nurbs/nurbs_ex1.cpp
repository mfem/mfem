//                          MFEM Example 1 - NURBS Version
//
// Compile with: make nurbs_ex1
//
// Sample runs:  nurbs_ex1 -m square-nurbs.mesh -o 2 -no-ibp
//               nurbs_ex1 -m cube-nurbs.mesh -o 2 -no-ibp
//               nurbs_ex1 -m pipe-nurbs-2d.mesh -o 2 -no-ibp
//               nurbs_ex1 -m ../../data/square-disc-nurbs.mesh -o -1
//               nurbs_ex1 -m ../../data/disc-nurbs.mesh -o -1
//               nurbs_ex1 -m ../../data/pipe-nurbs.mesh -o -1
//               nurbs_ex1 -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


/** Class for integrating the bilinear form a(u,v) := (Q Laplace u, v) where Q
    can be a scalar coefficient. */
class Diffusion2Integrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   Vector shape,laplace;
#endif
   Coefficient *Q;

public:
   /// Construct a diffusion integrator with coefficient Q = 1
   Diffusion2Integrator() { Q = NULL; }

   /// Construct a diffusion integrator with a scalar coefficient q
   Diffusion2Integrator (Coefficient &q) : Q(&q) { }

   /** Given a particular Finite Element
       computes the element stiffness matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat)
   {
      int nd = el.GetDof();
      int dim = el.GetDim();
      double w;

#ifdef MFEM_THREAD_SAFE
      Vector shape(nd);
      Vector laplace(nd);
#else
      shape.SetSize(nd);
      laplace.SetSize(nd);
#endif
      elmat.SetSize(nd);

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order;
         if (el.Space() == FunctionSpace::Pk)
         {
            order = 2*el.GetOrder() - 2;
         }
         else
         {
            order = 2*el.GetOrder() + dim - 1;
         }

         if (el.Space() == FunctionSpace::rQk)
         {
            ir = &RefinedIntRules.Get(el.GetGeomType(),order);
         }
         else
         {
            ir = &IntRules.Get(el.GetGeomType(),order);
         }
      }

      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Trans.SetIntPoint(&ip);
         w = -ip.weight * Trans.Weight();

         el.CalcShape(ip, shape);
         el.CalcPhysLaplacian(Trans, laplace);

         if (Q)
         {
            w *= Q->Eval(Trans, ip);
         }

         for (int j = 0; j < nd; j++)
         {
            for (int i = 0; i < nd; i++)
            {
               elmat(i, j) += w*shape(i)*laplace(j);
            }
         }
      }
   }

};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   const char *per_file  = "none";
   Array<int> master(0);
   Array<int> slave(0);
   bool static_cond = false;
   bool visualization = 1;
   bool ibp = 1;
   Array<int> order(1);
   order[0] = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&per_file, "-p", "--per",
                  "Periodic BCS file.");
   args.AddOption(&master, "-pm", "--master",
                  "Master boundaries for periodic BCs");
   args.AddOption(&slave, "-ps", "--slave",
                  "Slave boundaries for periodic BCs");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ibp, "-ibp", "--ibp", "-no-ibp",
                  "--no-ibp",
                  "Selects the standard weak form (IBP) or the nonstandard (NO-IBP).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   NURBSExtension *NURBSext = NULL;
   int own_fec = 0;

   if (mesh->NURBSext)
   {
      fec = new NURBSFECollection(order[0]);
      own_fec = 1;

      int nkv = mesh->NURBSext->GetNKV();
      if (order.Size() == 1)
      {
         int tmp = order[0];
         order.SetSize(nkv);
         order = tmp;
      }

      if (order.Size() != nkv ) { mfem_error("Wrong number of orders set."); }
      NURBSext = new NURBSExtension(mesh->NURBSext, order);

      // Read periodic BCs from file
      std::ifstream in;
      in.open(per_file, std::ifstream::in);
      if (in.is_open())
      {
         int psize;
         in >> psize;
         master.SetSize(psize);
         slave.SetSize(psize);
         master.Load(in, psize);
         slave.Load(in, psize);
         in.close();
      }
      master.Print();
      slave.Print();
      NURBSext->ConnectBoundaries(master,slave);
   }
   else if (order[0] == -1) // Isoparametric
   {
      if (mesh->GetNodes())
      {
         fec = mesh->GetNodes()->OwnFEC();
         own_fec = 0;
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
      else
      {
         cout <<"Mesh does not have FEs --> Assume order 1.\n";
         fec = new H1_FECollection(1, dim);
         own_fec = 1;
      }
   }
   else
   {
      if (order.Size() > 1) { cout <<"Wrong number of orders set, needs one.\n"; }
      fec = new H1_FECollection(abs(order[0]), dim);
      own_fec = 1;
   }

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, NURBSext, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   if (!ibp)
   {
      if (!mesh->NURBSext)
      {
         cout << "No integration by parts requires a NURBS mesh."<< endl;
         return 2;
      }
      if (mesh->NURBSext->GetNP()>1)
      {
         cout << "No integration by parts requires a NURBS mesh, with only 1 patch."<<
              endl;
         cout << "A C_1 discretisation is required."<< endl;
         cout << "Currently only C_0 multipatch coupling implemented."<< endl;
         return 3;
      }
      if (order[0]<2)
      {
         cout << "No integration by parts requires at least quadratic NURBS."<< endl;
         cout << "A C_1 discretisation is required."<< endl;
         return 4;
      }
   }



   // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      // Remove periodic BCs
      for (int i = 0; i < master.Size(); i++)
      {
         ess_bdr[master[i]-1] = 0;
         ess_bdr[slave[i]-1] = 0;
      }
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   if (ibp)
   {
      a->AddDomainIntegrator(new DiffusionIntegrator(one));
   }
   else
   {
      a->AddDomainIntegrator(new Diffusion2Integrator(one));
   }

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
   // 10. Define a simple Jacobi preconditioner and use it to
   //     solve the system A X = B with PCG.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
   // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
   UMFPackSolver umf_solver;
   umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   umf_solver.SetOperator(A);
   umf_solver.Mult(B, X);
#endif

   // 11. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 12. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 14. Save data in the VisIt format
   VisItDataCollection visit_dc("Example1", mesh);
   visit_dc.RegisterField("solution", &x);
   visit_dc.Save();

   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (own_fec) { delete fec; }
   delete mesh;

   return 0;
}
