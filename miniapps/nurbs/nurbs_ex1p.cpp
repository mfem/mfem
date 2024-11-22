//                     MFEM Example 1 - Parallel NURBS Version
//
// Compile with: make nurbs_ex1p
//
// Sample runs:  mpirun -np 4 nurbs_ex1p -m ../../data/square-disc.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/star.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/escher.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/fichera.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 nurbs_ex1p -m ../../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 nurbs_ex1p -m ../../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex1p -m ../../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex1p -m ../../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex1p -m ../../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 nurbs_ex1p -m ../../data/star-surf.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/square-disc-surf.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/inline-segment.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/amr-quad.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/amr-hex.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/mobius-strip.mesh
//               mpirun -np 4 nurbs_ex1p -m ../../data/mobius-strip.mesh -o -1 -sc
//               mpirun -np 4 nurbs_ex1p -m ../../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex1p -m ../../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex1p -m ../../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 nurbs_ex1p -m ../../data/square-nurbs.mesh -o 2 -no-ibp
//               mpirun -np 4 nurbs_ex1p -m ../../data/cube-nurbs.mesh -o 2 -no-ibp
//               mpirun -np 4 nurbs_ex1p -m ../../data/pipe-nurbs-2d.mesh -o 2 -no-ibp
//               mpirun -np 4 nurbs_ex1p -m meshes/square-nurbs.mesh -r 4 -pm "1" -ps "2"
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
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = -1;
   Array<int> order(1);
   order[0] = 1;
   bool static_cond = false;
   bool visualization = 1;
   bool ibp = 1;
   bool strongBC = 1;
   real_t kappa = -1;
   Array<int> master(0);
   int visport = 19916;
   Array<int> slave(0);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
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
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (!strongBC & (kappa < 0))
   {
      kappa = 4*(order.Max()+1)*(order.Max()+1);
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 10,000 elements.
   {
      if (ref_levels < 0)
      {
         ref_levels =
            (int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      }

      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
      if (myid == 0)
      {
         mesh->PrintInfo();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   if (!pmesh->NURBSext)
   {
      int par_ref_levels = 2;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order. If
   //    order < 1, we instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   NURBSExtension *NURBSext = NULL;
   int own_fec = 0;

   if (order[0] == -1) // Isoparametric
   {
      if (pmesh->GetNodes())
      {
         fec = pmesh->GetNodes()->OwnFEC();
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
   else if (pmesh->NURBSext && (order[0] > 0) )  // Subparametric NURBS
   {
      fec = new NURBSFECollection(order[0]);
      own_fec = 1;
      int nkv = pmesh->NURBSext->GetNKV();

      if (order.Size() == 1)
      {
         int tmp = order[0];
         order.SetSize(nkv);
         order = tmp;
      }
      if (order.Size() != nkv ) { mfem_error("Wrong number of orders set."); }
      NURBSext = new NURBSExtension(pmesh->NURBSext, order);

      // Enforce periodic BC's
      if (master.Size() > 0)
      {
         if (myid == 0)
         {
            cout<<"Connecting boundaries"<<endl;
            cout<<" - master : "; master.Print();
            cout<<" - slave  : "; slave.Print();
         }

         NURBSext->ConnectBoundaries(master,slave);
      }
   }
   else
   {
      if (order.Size() > 1) { cout <<"Wrong number of orders set, needs one.\n"; }
      fec = new H1_FECollection(abs(order[0]), dim);
      own_fec = 1;
   }
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh,NURBSext,fec);
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   if (!ibp)
   {
      if (!pmesh->NURBSext)
      {
         cout << "No integration by parts requires a NURBS mesh."<< endl;
         return 2;
      }
      if (pmesh->NURBSext->GetNP()>1)
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

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      if (strongBC)
      {
         ess_bdr = 1;
      }
      else
      {
         ess_bdr = 0;
      }

      // Remove periodic BCs from essential boundary list
      for (int i = 0; i < master.Size(); i++)
      {
         ess_bdr[master[i]-1] = 0;
         ess_bdr[slave[i]-1] = 0;
      }

      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));

   if (!strongBC)
      b->AddBdrFaceIntegrator(
         new DGDirichletLFIntegrator(zero, one, -1.0, kappa));
   b->Assemble();

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;

   // 10. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(fespace);
   if (ibp)
   {
      a->AddDomainIntegrator(new DiffusionIntegrator(one));
   }
   else
   {
      a->AddDomainIntegrator(new Diffusion2Integrator(one));
   }
   if (!strongBC)
   {
      a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, -1.0, kappa));
   }

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
   //     preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(A);
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(200);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 14. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 16. Save data in the VisIt format
   VisItDataCollection visit_dc("Example1-Parallel", pmesh);
   visit_dc.RegisterField("solution", &x);
   visit_dc.Save();

   // 17. Free the used memory.
   delete pcg;
   delete amg;
   delete a;
   delete b;
   delete fespace;
   if (own_fec) { delete fec; }
   delete pmesh;

   return 0;
}
