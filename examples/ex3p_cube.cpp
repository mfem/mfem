//                       MFEM Example 3 - Parallel Version
//
// Compile with: make ex3p
//
// Sample runs:  mpirun -np 4 ex3p -m ../data/star.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh
//               mpirun -np 4 ex3p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/fichera.mesh
//               mpirun -np 4 ex3p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex3p -m ../data/fichera-q3.mesh
//               mpirun -np 4 ex3p -m ../data/square-disc-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/beam-hex-nurbs.mesh
//               mpirun -np 4 ex3p -m ../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex3p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex3p -m ../data/mobius-strip.mesh -o 2 -f 0.1
//               mpirun -np 4 ex3p -m ../data/klein-bottle.mesh -o 2 -f 0.1
//
// Description:  This example code solves a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example demonstrates the use of H(curl) finite element
//               spaces with the curl-curl and the (vector finite element) mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Static condensation is
//               also illustrated.
//
//               We recommend viewing examples 1-2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
void test_vec(const Vector &x, Vector &v)
{ v.SetSize(3); v(0) = 1.2; v(1) = 1.1; v(2) = 1.0; v /= v.Norml2(); }

static double alpha_ = NAN;
static double beta_ = NAN;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   int rs = 0;
   int rp = 0;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&rs, "-rs", "--refine-serial",
                  "");
   args.AddOption(&rp, "-rp", "--refine-parallel",
                  "");
   args.AddOption(&alpha_, "-a", "--alpha", "");
   args.AddOption(&beta_, "-b", "--beta", "");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   alpha_ = (isnan(alpha_)) ? 0.0 : alpha_ * M_PI / 180.0;
   beta_  = (isnan(beta_))  ? 0.5 * M_PI : beta_ * M_PI / 180.0;

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels = rs;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
   //    meshes need to be reoriented before we can define high-order Nedelec
   //    spaces on them.
   Array<int> part(mesh->GetNE());
   for (int i=0; i<mesh->GetNE(); i++) { part[i] = i % num_procs; }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, part);
   delete mesh;
   {
      int par_ref_levels = rp;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   // pmesh->ReorientTetMesh();
   pmesh->ExchangeFaceNbrData();

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   FiniteElementCollection *d_fec = new L2_FECollection(order, dim, 1);
   ParFiniteElementSpace *d_fespace =
      new ParFiniteElementSpace(pmesh, d_fec, 3);

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorFunctionCoefficient tvCoef(sdim, test_vec);
   VectorFunctionCoefficient fCoef(sdim, f_exact);
   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fCoef));
   cout << myid << ": Assembling b" << endl;
   b->Assemble();
   cout << myid << ": Done assembling b" << endl;

   // 9. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x by projecting the exact
   //    solution. Note that only values from the boundary edges will be used
   //    when eliminating the non-homogeneous boundary condition to modify the
   //    r.h.s. vector b.
   ParGridFunction x(fespace);
   ParGridFunction d_x(d_fespace);
   ParGridFunction d_f(d_fespace);
   VectorFunctionCoefficient E(sdim, E_exact);

   /*
   x.ProjectCoefficient(tvCoef);
   {
   ofstream ofsx("x_tv.vec"); x.Print(ofsx,1); ofsx.close();

   double err = x.ComputeL2Error(tvCoef);
   if (myid == 0)
     {
       cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
     }

    if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x
          << "window_title 'test'"
               << "keys vvv"
          << flush;
   }

   }
   // exit(0);
   */
   {
      // const Operator &P = *fespace->GetProlongationMatrix();
      // P.Print("P.mat");

      ostringstream ossPM; ossPM << "pmesh." << setfill('0') << setw(6) << myid;
      ofstream ofsPM(ossPM.str());
      pmesh->ParPrint(ofsPM);
      ofsPM.close();

      ostringstream ossF; ossF << "face_info." << myid;
      ofstream ofsF(ossF.str());

      for (int i=0; i<pmesh->GetNumFaces(); i++)
      {
         int e1, e2;
         int i1, i2;
         pmesh->GetFaceElements(i, &e1, &e2);
         pmesh->GetFaceInfos(i, &i1, &i2);
         ofsF << i << " (" << e1 << " " << i1/64 << " " << i1%64 << ") ("
              << e2 << " " << i2/64 << " " << i2%64 << ")" << endl;
      }

      ofsF.close();

      ParGridFunction f(fespace);
      f.ProjectCoefficient(tvCoef);

      {
         double errf = f.ComputeL2Error(tvCoef);
         if (myid == 0)
         {
            cout << endl;
            cout << "|| f_h - f ||_{L^2} = " << errf << endl;
         }

         cout << myid << ": printing f" << endl;
         ostringstream ossf; ossf << "f.gf." << myid;
         ofstream ofsf(ossf.str().c_str());
         f.Print(ofsf, 1);

         cout << myid << ": printing F" << endl;
         Vector F(fespace->TrueVSize());
         f.ParallelProject(F);

         ostringstream ossF; ossF << "F.vec." << myid;
         ofstream ofsF(ossF.str().c_str());
         F.Print(ofsF, 1);

         ParGridFunction f2(fespace);
         cout << myid << ": printing f again" << endl;
         f2.Distribute(F);

         ostringstream ossf2; ossf2 << "f2.gf." << myid;
         ofstream ofsf2(ossf2.str().c_str());
         f2.Print(ofsf2, 1);
      }

      ParLinearForm f_lf(fespace);
      f_lf.AddDomainIntegrator(new VectorFEDomainLFIntegrator(tvCoef));
      f_lf.Assemble();

      ParBilinearForm m(fespace);
      m.AddDomainIntegrator(new VectorFEMassIntegrator);
      cout << myid << ": Assembling m" << endl;
      m.Assemble();
      cout << myid << ": Done assembling m" << endl;

      cout << myid << ": Projecting f onto x" << endl;
      x.ProjectCoefficient(tvCoef);
      cout << myid << ": Done projecting f onto x" << endl;

      double f2 = f_lf(x);
      if (myid == 0)
      {
         cout << endl << "f(f_h) = " << f2 << endl << endl;
      }

      HypreParMatrix M;
      Vector B, X;
      m.FormLinearSystem(ess_tdof_list, x, f_lf, M, X, B);

      {
         ostringstream oss; oss << "B.vec." << myid;
         ofstream ofsB(oss.str().c_str());
         B.Print(ofsB, 1);

         set<int> bs;
         for (int i=0; i<B.Size(); i++)
         {
            bs.insert((int)fabs(1.0e8 * B[i]));
         }
         cout << "bs.size(): " << bs.size() << endl;

         set<int>::iterator sit;
         ofsB << "\nbs\n";
         for (sit=bs.begin(); sit !=bs.end(); sit++)
         {
            ofsB << ((double)*sit)*1e-8 << endl;
         }
      }
      if (false)
      {
	M.Print("M.mat");

	HypreParMatrix * Mt = M.Transpose();
	Mt->Print("Mt.mat");

	HypreParMatrix * MA = Add(1.0, M, -1.0, *Mt);
	MA->Print("MA.mat");

	delete Mt;
	delete MA;
      }
      
      HypreDiagScale diag(M);
      HyprePCG pcg(M);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(diag);
      pcg.Mult(B, X);
      m.RecoverFEMSolution(X, f_lf, x);

      f.ProjectVectorFieldOn(d_f);
      x.ProjectVectorFieldOn(d_x);
      {
         double errf = f.ComputeL2Error(tvCoef);
         double d_errf = d_f.ComputeL2Error(tvCoef);
         double err = x.ComputeL2Error(tvCoef);
         double d_err = d_x.ComputeL2Error(tvCoef);
         if (myid == 0)
         {
            cout << endl;
            cout << "|| f_h - f        ||_{L^2} = " << errf << endl;
            cout << "|| disc f_h - f   ||_{L^2} = " << d_errf << endl;
            cout << "|| sol_h - f      ||_{L^2} = " << err << endl;
            cout << "|| disc sol_h - f ||_{L^2} = " << d_err << endl;
         }
      }

      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         socketstream f_sock(vishost, visport);
         f_sock << "parallel " << num_procs << " " << myid << "\n";
         f_sock.precision(8);
         f_sock << "solution\n" << *pmesh << f
                << "window_title 'f, order = " << order << "'"
                << "keys vvv"
                << flush;
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << *pmesh << x
                  << "window_title 'f test, order = " << order << "'"
                  << "keys vvv"
                  << flush;
      }
   }
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
   /*
   VisItDataCollection visit_dc("Example3-Parallel-Cube", pmesh);
   visit_dc.RegisterField("sol", &x);
   visit_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
   visit_dc.Save();
   */
   MPI_Finalize();
   exit(0);
   cout << "Projecting onto x" << endl;
   x.ProjectCoefficient(E);
   cout << "Done projecting onto x" << endl;

   // 10. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(1.0);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 11. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   cout << "Assembling a" << endl;
   a->Assemble();
   cout << "Done assembling a" << endl;

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }
   {
      A.Print("A.mat");
      ofstream ofsB("B.vec"); B.Print(ofsB,1); ofsB.close();
      ofstream ofsX("X.vec"); X.Print(ofsX,1); ofsX.close();
      ofstream ofsb("b_lf.vec"); b->Print(ofsb,1); ofsb.close();
      ofstream ofsx("x_gf.vec"); x.Print(ofsx,1); ofsx.close();

      double nrmB = B.Norml2();
      double nrmX = X.Norml2();
      double nrmb = b->Norml2();
      double nrmx = x.Norml2();

      if (myid == 0)
      {
         cout << "Norm of B: " << nrmB << endl;
         cout << "Norm of X: " << nrmX << endl;
         cout << "Norm of b: " << nrmb << endl;
         cout << "Norm of x: " << nrmx << endl;
      }

   }

   // 12. Define and apply a parallel PCG solver for AX=B with the AMS
   //     preconditioner from hypre.
   if (true)
   {
      ParFiniteElementSpace *prec_fespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
      HypreSolver *ams = new HypreAMS(A, prec_fespace);
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-12);
      pcg->SetMaxIter(500);
      pcg->SetPrintLevel(2);
      pcg->SetPreconditioner(*ams);
      pcg->Mult(B, X);
      delete pcg;
      delete ams;
   }
   else
   {
      HypreGMRES gmres(A);
      gmres.SetTol(1e-8);
      gmres.SetMaxIter(5000);
      gmres.SetPrintLevel(1);
      gmres.Mult(B, X);
   }

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 14. Compute and print the L^2 norm of the error.
   {
      double err = x.ComputeL2Error(E);
      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
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

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x
               << "keys vvv" << flush;
   }

   // 17. Free the used memory.
   // delete pcg;
   // delete ams;
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();

   return 0;
}


void E_exact(const Vector &x, Vector &E)
{
   int dim = x.Size();
   if (dim == 3)
   {
      E(0) = cos(alpha_) * sin(beta_);
      E(1) = sin(alpha_) * sin(beta_);
      E[2] = cos(beta_);
      E *= sin(M_PI * x[0]) * sin(M_PI * x[1]) * sin(M_PI * x[2]);
   }
   else
   {
      E(0) = cos(alpha_);
      E(1) = sin(alpha_);
      E *= sin(M_PI * x[0]) * sin(M_PI * x[1]);
   }
}

void f_exact(const Vector &x, Vector &f)
{
   int dim = x.Size();

   double pi2 = M_PI * M_PI;
   double ca = cos(alpha_);
   double sa = sin(alpha_);
   double cx = cos(M_PI * x[0]);
   double sx = sin(M_PI * x[0]);
   double cy = cos(M_PI * x[1]);
   double sy = sin(M_PI * x[1]);

   if (dim == 3)
   {
      double cb = cos(beta_);
      double sb = sin(beta_);
      double cz = cos(M_PI * x[2]);
      double sz = sin(M_PI * x[2]);

      double s3 = (1.0 + 2.0 * pi2) * sx * sy * sz;

      f(0) = ca * sb * s3 + pi2 * cx * (cb * sy * cz + sa * sb * cy * sz);
      f(1) = sa * sb * s3 + pi2 * cy * (cb * sx * cz + ca * sb * cx * sz);
      f(2) = cb * s3 + pi2 * cz * sb * (ca * cx * sy + sa * sx * cy);
   }
   else
   {
      f(0) = ca * (1.0 + pi2) * sx * sy + pi2 * sa * cx * cy;
      f(1) = sa * (1.0 + pi2) * sx * sy + pi2 * ca * cx * cy;
   }
}
