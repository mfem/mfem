//                           hp-Refinement Demo
//
// Compile with: make hptest
//
// Sample runs: TODO
//
// Description: This is a demo of the hp-refinement capability of MFEM.
//              One of the benchmark problems with a known exact solution is
//              solved on a sequence of meshes where both the size (h) and the
//              polynomial order (p) of elements is adapted.
//

#include "mfem.hpp"
#include <fstream>

#include "exact.hpp"
#include "util.hpp"
#include "error.hpp"

using namespace std;
using namespace mfem;


struct HPError
{
   double err, err_h, err_p;
   int dof, dof_h, dof_p;
};


void MakeConforming(GridFunction &sol)
{
    FiniteElementSpace* fes = sol.FESpace();
    const SparseMatrix* P = fes->GetConformingProlongation();
    const SparseMatrix* Q = fes->GetConformingRestrictionInterpolation();
    if (P)
    {
        Vector X;
        X.SetSize(Q->Height());
        Q->Mult(sol, X);
        P->Mult(X, sol);
    }
}

void EstimateHPErrors(FiniteElementSpace* fes,
                      Coefficient &exsol,
                      VectorCoefficient &exgrad,
                      Array<HPError> &elem_hp_error, socketstream &ex_sock)
{
    Mesh* mesh = fes->GetMesh();

    // h-refined mesh and space with halved order
    Mesh* mesh_h = new Mesh(*mesh);
    FiniteElementSpace* fes_h = new FiniteElementSpace(*fes, mesh_h); // FIXME: copy variable orders

    for (int i = 0; i < mesh->GetNE(); i++)
    {
        int p = fes->GetElementOrder(i);
        fes_h->SetElementOrder(i, (p+2)/2);
    }
    fes_h->Update(false);
    mesh_h->UniformRefinement();
    fes_h->Update(false);


    // space with orders increased by 1
    FiniteElementSpace* fes_p = new FiniteElementSpace(*fes, mesh);
    for (int i = 0; i < mesh->GetNE(); i++)
    {
        int p = fes->GetElementOrder(i);
        fes_p->SetElementOrder(i, p+1);
    }
    fes_p->Update(false);

    GridFunction sol(fes);
    sol.ProjectCoefficient(exsol);
    MakeConforming(sol);

    GridFunction sol_h(fes_h);
    sol_h.ProjectCoefficient(exsol);
    MakeConforming(sol_h);

    GridFunction sol_p(fes_p);
    sol_p.ProjectCoefficient(exsol);
    MakeConforming(sol_p);

    if (1)
    {
        GridFunction *vis_sol = ProlongToMaxOrder(&sol);
        ex_sock.precision(8);
        ex_sock << "solution\n" << *mesh << *vis_sol << flush;
        //ex_sock << "pause\n" << flush;
        delete vis_sol;
    }

    Array<double> elemError;
    Array<double> elemError_h;
    Array<double> elemError_p;
    Array<int> elemRef;
    CalculateH10Error2(&sol, &exgrad, &elemError, &elemRef, 10);
    CalculateH10Error2(&sol_h, &exgrad, &elemError_h, &elemRef, 10);
    CalculateH10Error2(&sol_p, &exgrad, &elemError_p, &elemRef, 10);

    elem_hp_error.SetSize(mesh->GetNE());
    for (int j = 0; j < mesh->GetNE(); j++)
    {
        // Initialize elem_hp_error

        // Original element
        int p = fes->GetElementOrder(j);
        int dof = p*p;
        elem_hp_error[j].err =elemError[j];
        elem_hp_error[j].dof = dof;

        // For p-refinement compute error and dofs per element
        int q = fes_p->GetElementOrder(j);
        int pdof = q*q;
        elem_hp_error[j].err_p = elemError_p[j];
        elem_hp_error[j].dof_p = pdof;

        // h-refinement
        elem_hp_error[j].err_h = 0;
        elem_hp_error[j].dof_h = 0;

    }

    // For h-refinement sum up errors and dofs for each coarse element
    // and compute error decrease and dof increase per coarse element
    const CoarseFineTransformations tr = mesh_h->GetRefinementTransforms();
    for (int i = 0; i < mesh_h->GetNE(); i++)
    {
        int j = tr.embeddings[i].parent;
        int p = fes_h->GetElementOrder(i);
        int hdof = p*p;
        elem_hp_error[j].err_h += elemError_h[i];
        elem_hp_error[j].dof_h += hdof;
    }

    delete mesh_h;
    delete fes_h;
    delete fes_p;
}


int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "layer-quad.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = false;
   double ref_threshold = 0.7;
   bool aniso = false;
   bool hp = true;
   int max_order = 6;
   int int_order = 10;
   bool relaxed_hp = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Initial mesh finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&relaxed_hp, "-x", "--relaxed-hp", "-no-x",
                  "--no-relaxed-hp", "Set relaxed hp conformity.");
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

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Load and adjust the Mesh
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }
   mesh.EnsureNCMesh(true);

   // We don't support mixed meshes at the moment
   MFEM_VERIFY(mesh.GetNumGeometries(dim) == 1, "Mixed meshes not supported.");
   Geometry::Type geom = mesh.GetElementGeometry(0);

   // Prepare exact solution Coefficients
   FunctionCoefficient exsol(layer2_exsol);
      /*prob_type ? ((dim == 3) ? layer3_exsol  : layer2_exsol)
                : ((dim == 3) ? fichera_exsol : lshape_exsol) );*/

   VectorFunctionCoefficient exgrad(dim, layer2_exgrad);
      /*prob_type ? ((dim == 3) ? layer3_exgrad  : layer2_exgrad)
                : ((dim == 3) ? fichera_exgrad : lshape_exgrad) );*/

   FunctionCoefficient rhs(layer2_laplace);
      /*prob_type ? ((dim == 3) ? layer3_laplace  : layer2_laplace)
                : ((dim == 3) ? fichera_laplace : lshape_laplace) );*/

   // Define a finite element space on the mesh. Initially the polynomial
   // order is constant everywhere.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   fespace.SetRelaxedHpConformity(relaxed_hp);

   GridFunction x(&fespace);
   x = 0.0;

   // All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   socketstream ex_sock;
   socketstream ord_sock;

   if (visualization)
   {
      sol_sock.open(vishost, visport);
      ex_sock.open(vishost, visport);
      ord_sock.open(vishost, visport);
   }


   std::ofstream conv("hp.err");

   // The main AMR loop. In each iteration we solve the problem on the
   // current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 30000;
   //const int max_dofs = 200;
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // Assemble the linear form. The right hand side is manufactured
      // so that the solution is the analytic solution.
      LinearForm lf(&fespace);
      DomainLFIntegrator *dlfi = new DomainLFIntegrator(rhs);
      dlfi->SetIntRule(&IntRules.Get(geom, int_order));
      lf.AddDomainIntegrator(dlfi);
      lf.Assemble();

      // Assemble bilinear form.
      BilinearForm bf(&fespace);
      if (pa) { bf.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      bf.AddDomainIntegrator(new DiffusionIntegrator());
      bf.Assemble();

       // Set Dirichlet boundary values in the GridFunction x.
      // Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x = 0; // FIXME
      x.ProjectBdrCoefficient(exsol, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 16. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      bf.FormLinearSystem(ess_tdof_list, x, lf, A, X, B, copy_interior);

      // 17. Solve the linear system A X = B.
      if (!pa)
      {
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
      }
      else // No preconditioning for now in partial assembly mode.
      {
         CG(*A, B, X, 3, 2000, 1e-12, 0.0);
      }

      // 18. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      bf.RecoverFEMSolution(X, lf, x);

      // 19. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         GridFunction *vis_x = ProlongToMaxOrder(&x);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << *vis_x << flush;
         sol_sock << "pause\n" << flush;
         delete vis_x;

         L2_FECollection l2fec(0, dim);
         FiniteElementSpace l2fes(&mesh, &l2fec);
         GridFunction orders(&l2fes);
         for (int i = 0; i < orders.Size(); i++)
         {
            orders(i) = fespace.GetElementOrder(i);
         }

         ord_sock.precision(8);
         ord_sock << "solution\n" << mesh << orders
                 // << "keys Rjlmc\n"
                  << flush;
         //ord_sock << "pause\n" << flush;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // Calculate the H^1_0 errors of elements as well as the total error.
      Array<double> elem_error;
      Array<int> ref_type;
      double error;
      {
         error = CalculateH10Error2(&x, &exgrad, &elem_error, &ref_type, int_order);
         error = std::sqrt(error);
      }

      // Save dofs and error for convergence plot
      conv << cdofs << " " << error << endl;

      // Project the exact solution to h-refined and p-refined versions of the
      // mesh and determine whether to refine elements in 'h' or in 'p'.
      Array<int> hp_ref(mesh.GetNE());
      if (hp)
      {
         Array<HPError> elem_hp_error;
         EstimateHPErrors(&fespace, exsol, exgrad, elem_hp_error, ex_sock);

         for (int i = 0; i < mesh.GetNE(); i++)
         {
             double err_max = sqrt(elem_error.Max());
             if (sqrt(elem_error[i]) > ref_threshold * err_max)
             {
                 cout << " Element:  " << i << endl;
                 cout << " Error: " << sqrt(elem_error[i]) << endl;
                 cout << " dofs : " << elem_hp_error[i].dof << " error: " << elem_hp_error[i].err << endl;
                 cout << " dofs h: " << elem_hp_error[i].dof_h << " error h: " << elem_hp_error[i].err_h << endl;
                 cout << " dofs p: " << elem_hp_error[i].dof_p << " error p: " << elem_hp_error[i].err_p << endl;

                 double division_h = (sqrt(elem_hp_error[i].err) - sqrt(elem_hp_error[i].err_h))/(elem_hp_error[i].dof_h - elem_hp_error[i].dof);
                 double division_p = (sqrt(elem_hp_error[i].err) - sqrt(elem_hp_error[i].err_p))/(elem_hp_error[i].dof_p - elem_hp_error[i].dof);

                 cout << " h: " << division_h << " p: " << division_p << endl;
                 if (division_h < 0 || division_h > division_p || elem_hp_error[i].dof_p > max_order*max_order)
                 {
                     hp_ref[i] = 0;
                 }
                 else
                 {
                     hp_ref[i] = 1;
                 }
             }
         }
      }

      int h_refined = 0;
      int p_refined = 0;
      // p-refine elements
      if (hp)
      {
          double err_max = sqrt(elem_error.Max());
          for (int i = 0; i < mesh.GetNE(); i++)
          {
             if (sqrt(elem_error[i]) > ref_threshold * err_max)
             {
                 if (hp_ref[i] == 1)
                 {
                     int p = fespace.GetElementOrder(i);
                     fespace.SetElementOrder(i, min(max_order,p+1));
                     p_refined++;
                 }
                 if (hp_ref[i] == 0)
                 {
                     int p = fespace.GetElementOrder(i);
                     fespace.SetElementOrder(i, (p+2)/2);
                     h_refined++;
                 }
             }
          }

         // Update the space, interpolate the solution. FIXME
         fespace.Update(false);
         //x.Update(); // NOT IMPLEMENTED YET
      }

      cout << " h-refined = " << h_refined << ", p-refined = " << p_refined << endl;

      // h-refine elements
      Array<Refinement> refinements;
      {
         double err_max = sqrt(elem_error.Max());
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            if (sqrt(elem_error[i]) > ref_threshold * err_max)
            {
               if (!hp || hp_ref[i] == 0) // h-refinements only (if doing hp)
               {
                  int type = aniso ? ref_type[i] : 7;
                  refinements.Append(Refinement(i, type));
               }
            }
         }
      }
      mesh.GeneralRefinement(refinements,-1,2);

      // Update the space, interpolate the solution.
      fespace.Update(false);
      x.Update();


      // Inform also the bilinear and linear forms that the space has changed.
      bf.Update();
      lf.Update();
   }

   return 0;
}



