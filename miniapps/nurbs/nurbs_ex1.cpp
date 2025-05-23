//                          MFEM Example 1 - NURBS Version
//
// Compile with: make nurbs_ex1
//
// Sample runs:  nurbs_ex1 -m ../../data/square-nurbs.mesh -o 2 -no-ibp
//               nurbs_ex1 -m ../../data/square-nurbs.mesh -o 2 --weak-bc
//               nurbs_ex1 -m ../../data/cube-nurbs.mesh -o 2 -no-ibp
//               nurbs_ex1 -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp
//               nurbs_ex1 -m ../../data/pipe-nurbs-2d.mesh -o 2 -r 2 --neu "3"
//               nurbs_ex1 -m ../../data/square-disc-nurbs.mesh -o -1
//               nurbs_ex1 -m ../../data/disc-nurbs.mesh -o -1
//               nurbs_ex1 -m ../../data/pipe-nurbs.mesh -o -1
//               nurbs_ex1 -m ../../data/beam-hex-nurbs.mesh -pm 1 -ps 2
//               nurbs_ex1 -m meshes/two-squares-nurbs.mesh -o 1 -rf meshes/two-squares.ref
//               nurbs_ex1 -m meshes/two-squares-nurbs-rot.mesh -o 1 -rf meshes/two-squares.ref
//               nurbs_ex1 -m meshes/two-squares-nurbs-autoedge.mesh -o 1 -rf meshes/two-squares.ref
//               nurbs_ex1 -m meshes/two-cubes-nurbs.mesh -o 1 -r 3 -rf meshes/two-cubes.ref
//               nurbs_ex1 -m meshes/two-cubes-nurbs-rot.mesh -o 1 -r 3 -rf meshes/two-cubes.ref
//               nurbs_ex1 -m meshes/two-cubes-nurbs-autoedge.mesh -o 1 -r 3 -rf meshes/two-cubes.ref
//               nurbs_ex1 -m ../../data/segment-nurbs.mesh -r 2 -o 2 -lod 3
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               The boundary conditions can be enforced either strongly or weakly.
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
#include <list>

using namespace std;
using namespace mfem;

class Data
{
public:
   real_t x,val;
   Data(real_t x_, real_t val_) {x=x_; val=val_;};
};

inline bool operator==(const Data& d1,const Data& d2) { return (d1.x == d2.x); }
inline bool operator <(const Data& d1,const Data& d2) { return (d1.x  < d2.x); }

/** Class for integrating the bilinear form a(u,v) := (Q Laplace u, v) where Q
    can be a scalar coefficient. */
class Diffusion2Integrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   Vector shape, laplace;
#endif
   Coefficient *Q;

public:
   /// Construct a diffusion integrator with coefficient Q = 1
   Diffusion2Integrator() { Q = NULL; }

   /// Construct a diffusion integrator with a scalar coefficient q
   Diffusion2Integrator (Coefficient &q) : Q(&q) { }

   /** Given a particular Finite Element
       computes the element stiffness matrix elmat. */
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override
   {
      int nd = el.GetDof();
      int dim = el.GetDim();
      real_t w;

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

         for (int jj = 0; jj < nd; jj++)
         {
            for (int ii = 0; ii < nd; ii++)
            {
               elmat(ii, jj) += w*shape(ii)*laplace(jj);
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
   const char *ref_file  = "";
   int ref_levels = -1;
   Array<int> master(0);
   Array<int> slave(0);
   Array<int> neu(0);
   bool static_cond = false;
   bool visualization = 1;
   int lod = 0;
   bool ibp = 1;
   bool strongBC = 1;
   real_t kappa = -1;
   Array<int> order(1);
   int visport = 19916;
   order[0] = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&per_file, "-p", "--per",
                  "Periodic BCS file.");
   args.AddOption(&ref_file, "-rf", "--ref-file",
                  "File with refinement data");
   args.AddOption(&master, "-pm", "--master",
                  "Master boundaries for periodic BCs");
   args.AddOption(&slave, "-ps", "--slave",
                  "Slave boundaries for periodic BCs");
   args.AddOption(&neu, "-n", "--neu",
                  "Boundaries with Neumann BCs");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&ibp, "-ibp", "--ibp", "-no-ibp",
                  "--no-ibp",
                  "Selects the standard weak form (IBP) or the nonstandard (NO-IBP).");
   args.AddOption(&strongBC, "-sbc", "--strong-bc", "-wbc",
                  "--weak-bc",
                  "Selects strong or weak enforcement of Dirichlet BCs.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Sets the SIPG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&lod, "-lod", "--level-of-detail",
                  "Refinement level for 1D solution output (0 means no output).");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (!strongBC & (kappa < 0))
   {
      kappa = 4*(order.Max()+1)*(order.Max()+1);
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement and knot insertion of knots defined
   //    in a refinement file. We choose 'ref_levels' to be the largest number
   //    that gives a final mesh with no more than 50,000 elements.
   {
      // Mesh refinement as defined in refinement file
      if (mesh->NURBSext && (strlen(ref_file) != 0))
      {
         mesh->RefineNURBSFromFile(ref_file);
      }

      if (ref_levels < 0)
      {
         ref_levels =
            (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
      mesh->PrintInfo();
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
   Array<int> ess_bdr(0);
   Array<int> neu_bdr(0);
   Array<int> per_bdr(0);
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      neu_bdr.SetSize(mesh->bdr_attributes.Max());
      per_bdr.SetSize(mesh->bdr_attributes.Max());

      ess_bdr = 1;
      neu_bdr = 0;
      per_bdr = 0;

      // Apply Neumann BCs
      for (int i = 0; i < neu.Size(); i++)
      {
         if ( neu[i]-1 >= 0 &&
              neu[i]-1 < mesh->bdr_attributes.Max())
         {
            ess_bdr[neu[i]-1] = 0;
            neu_bdr[neu[i]-1] = 1;
         }
         else
         {
            cout <<"Neumann boundary "<<neu[i]<<" out of range -- discarded"<< endl;
         }
      }

      // Correct for periodic BCs
      for (int i = 0; i < master.Size(); i++)
      {
         if ( master[i]-1 >= 0 &&
              master[i]-1 < mesh->bdr_attributes.Max())
         {
            ess_bdr[master[i]-1] = 0;
            neu_bdr[master[i]-1] = 0;
            per_bdr[master[i]-1] = 1;
         }
         else
         {
            cout <<"Master boundary "<<master[i]<<" out of range -- discarded"<< endl;
         }
      }
      for (int i = 0; i < slave.Size(); i++)
      {
         if ( slave[i]-1 >= 0 &&
              slave[i]-1 < mesh->bdr_attributes.Max())
         {
            ess_bdr[slave[i]-1] = 0;
            neu_bdr[slave[i]-1] = 0;
            per_bdr[slave[i]-1] = 1;
         }
         else
         {
            cout <<"Slave boundary "<<slave[i]<<" out of range -- discarded"<< endl;
         }
      }
   }
   cout <<"Boundary conditions:"<< endl;
   cout <<" - Periodic  : "; per_bdr.Print();
   cout <<" - Essential : "; ess_bdr.Print();
   cout <<" - Neumann   : "; neu_bdr.Print();


   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   ConstantCoefficient one(1.0);
   ConstantCoefficient mone(-1.0);
   ConstantCoefficient zero(0.0);

   LinearForm *b = new LinearForm(fespace);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->AddBoundaryIntegrator( new BoundaryLFIntegrator(one),neu_bdr);
   if (!strongBC)
      b->AddBdrFaceIntegrator(
         new DGDirichletLFIntegrator(zero, one, -1.0, kappa), ess_bdr);

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
      a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(mone, 0.0, 0.0), neu_bdr);
   }

   if (!strongBC)
   {
      a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, -1.0, kappa), ess_bdr);
   }

   // 9. Assemble the bilinear form and the corresponding linear system,
   //    applying any necessary transformations such as: eliminating boundary
   //    conditions, applying conforming constraints for non-conforming AMR,
   //    static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   Array<int> ess_tdof_list(0);
   if (strongBC)
   {
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
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
   {
      ofstream mesh_ofs("refined.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
      sol_ofs.close();
   }

   // 13. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   if (mesh->Dimension() == 1 && lod > 0)
   {
      std::list<Data> sol;

      Vector      vals,coords;
      GridFunction *nodes = mesh->GetNodes();
      if (!nodes)
      {
         nodes = new GridFunction(fespace);
         mesh->GetNodes(*nodes);
      }

      for (int i = 0; i <  mesh->GetNE(); i++)
      {
         int geom       = mesh->GetElementBaseGeometry(i);
         RefinedGeometry *refined_geo = GlobGeometryRefiner.Refine(( Geometry::Type)geom,
                                                                   lod, 1);

         x.GetValues(i, refined_geo->RefPts, vals);
         nodes->GetValues(i, refined_geo->RefPts, coords);

         for (int j = 0; j < vals.Size(); j++)
         {
            sol.push_back(Data(coords[j],vals[j]));
         }
      }
      sol.sort();
      sol.unique();
      ofstream sol_ofs("solution.dat");
      for (std::list<Data>::iterator d = sol.begin(); d != sol.end(); ++d)
      {
         sol_ofs<<d->x <<"\t"<<d->val<<endl;
      }

      sol_ofs.close();
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
