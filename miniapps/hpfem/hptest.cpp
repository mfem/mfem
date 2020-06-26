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

const char vishost[] = "localhost";
const int  visport   = 19916;

const char* keys = "Rjlmc*******";


void VisualizeField(socketstream &sock, GridFunction &gf, const char *title,
                    const char * keys = NULL, int w = 400, int h = 400,
                    int x = 0, int y = 0, bool vec = false)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (!sock.is_open() || !sock)
      {
         sock.open(vishost, visport);
         sock.precision(8);
         newly_opened = true;
      }
      sock << "solution\n";

      mesh.Print(sock);
      gf.Save(sock);

      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";

         if (keys) { sock << "keys " << keys << "\n"; }
         else { sock << "keys mAc\n"; }

         if (vec) { sock << "vvv"; }
         sock << endl;
      }

      connection_failed = !sock && !newly_opened;
   }
   while (connection_failed);
}


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

int HalfOrder(int p)
{
   return p/2 + 1;
   //return (p + 1) / 2;
}

#define CONTINUOUS 0

void EstimateHPErrors(FiniteElementSpace* fes,
                      Coefficient &exsol,
                      VectorCoefficient &exgrad, int int_order,
                      Array<HPError> &elem_hp_error, socketstream dbg_sock[])
{
    Mesh* mesh = fes->GetMesh();

    // h-refined mesh and space with halved order
    Mesh* mesh_h = new Mesh(*mesh);
#if CONTINUOUS
    FiniteElementSpace* fes_h = new FiniteElementSpace(*fes, mesh_h); // FIXME: copy variable orders
#else
    L2_FECollection fec(1, mesh->Dimension());
    FiniteElementSpace* fes_h = new FiniteElementSpace(mesh_h, &fec);
#endif

    for (int i = 0; i < mesh->GetNE(); i++)
    {
        int p = fes->GetElementOrder(i);
        fes_h->SetElementOrder(i, HalfOrder(p));
    }
    fes_h->Update(false);
    mesh_h->UniformRefinement();
    fes_h->Update(false);


    // space with orders increased by 1
#if CONTINUOUS
    FiniteElementSpace* fes_p = new FiniteElementSpace(*fes, mesh);
#else
    FiniteElementSpace* fes_p = new FiniteElementSpace(mesh, &fec);
#endif
    for (int i = 0; i < mesh->GetNE(); i++)
    {
        int p = fes->GetElementOrder(i);
        fes_p->SetElementOrder(i, p+1);
    }
    fes_p->Update(false);

    GridFunction sol(fes);
    sol.ProjectCoefficient(exsol);
#if CONTINUOUS
    MakeConforming(sol);
#endif

    GridFunction sol_h(fes_h);
    sol_h.ProjectCoefficient(exsol);
#if CONTINUOUS
    MakeConforming(sol_h);
#endif

    GridFunction sol_p(fes_p);
    sol_p.ProjectCoefficient(exsol);
#if CONTINUOUS
    MakeConforming(sol_p);
#endif

    if (0)
    {
       GridFunction *vis_sol = ProlongToMaxOrder(&sol);
       VisualizeField(dbg_sock[0], *vis_sol, "Projected exsol",
                      keys, 600, 500, 610, 70);
       delete vis_sol;

       vis_sol = ProlongToMaxOrder(&sol_h);
       VisualizeField(dbg_sock[1], *vis_sol, "Projected h-refined exsol",
                      keys, 600, 500, 1220, 70);
       delete vis_sol;

       vis_sol = ProlongToMaxOrder(&sol_p);
       VisualizeField(dbg_sock[2], *vis_sol, "Projected p-refined exsol",
                      keys, 600, 500, 1220, 620);
       delete vis_sol;
    }

    Array<double> elemError;
    Array<double> elemError_h;
    Array<double> elemError_p;
    Array<int> elemRef;
    CalculateH10Error2(&sol, &exgrad, &elemError, &elemRef, int_order);
    CalculateH10Error2(&sol_h, &exgrad, &elemError_h, &elemRef, int_order);
    CalculateH10Error2(&sol_p, &exgrad, &elemError_p, &elemRef, int_order);

    elem_hp_error.SetSize(mesh->GetNE());
    for (int j = 0; j < mesh->GetNE(); j++)
    {
        // Initialize elem_hp_error

        // Original element
        int p = fes->GetElementOrder(j);
        int dof = p*p;
        elem_hp_error[j].err = elemError[j];
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

    // take a square root of all the partial errors
    /*for (int i = 0; i < mesh->GetNE(); i++)
    {
       HPError err = elem_hp_error[i];
       err.err = sqrt(err.err);
       err.err_h = sqrt(err.err_h);
       err.err_p = sqrt(err.err_p);
    }*/

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
   bool visualization = true;
   double ref_threshold = 0.7;
   bool aniso = false;
   bool hp = true;
   int max_order = 6;
   int int_order = 30;
   bool relaxed_hp = false;
   bool pause = false;
   int nc_limit = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Initial mesh finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&hp, "-hp", "--hp", "-no-hp", "--no-hp",
                  "Enable hp refinement.");
   args.AddOption(&relaxed_hp, "-x", "--relaxed-hp", "-no-x", "--no-relaxed-hp",
                  "Set relaxed hp conformity.");
   args.AddOption(&ref_threshold, "-rt", "--ref-threshold",
                  "Refine elements with error larger than threshold * max_error.");
   args.AddOption(&nc_limit, "-nc", "--nc-limit",
                  "Set maximum difference of refinement levels of adjacent elements.");
   args.AddOption(&pause, "-p", "--pause", "-no-p", "--no-pause",
                  "Wait for user input after each iteration.");
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
   socketstream sol_sock, ord_sock, dbg_sock[3];

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
         PCG(*A, M, B, X, 3, 500, 1e-13, 0.0);
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
      if (visualization)
      {
         GridFunction *vis_x = ProlongToMaxOrder(&x);
         VisualizeField(sol_sock, *vis_x, "Solution", keys, 600, 500, 0, 70);
         delete vis_x;

         L2_FECollection l2fec(0, dim);
         FiniteElementSpace l2fes(&mesh, &l2fec);
         GridFunction orders(&l2fes);
         for (int i = 0; i < orders.Size(); i++)
         {
            orders(i) = fespace.GetElementOrder(i);
         }
         VisualizeField(ord_sock, orders, "Orders", keys, 600, 500, 0, 620);
         ord_sock << "valuerange 1 5\n" << flush;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // Calculate the H^1_0 errors of elements as well as the total error.
      Array<double> elem_error;
      Array<int> ref_type;
      double error = sqrt(CalculateH10Error2(&x, &exgrad, &elem_error,
                                             &ref_type, int_order));

      // Save dofs and error for convergence plot
      conv << cdofs << " " << error << endl;

      // Project the exact solution to h-refined and p-refined versions of the
      // mesh and determine whether to refine elements in 'h' or in 'p'.
      if (hp)
      {
         Array<HPError> elem_hp_error;
         EstimateHPErrors(&fespace, exsol, exgrad, int_order, elem_hp_error, dbg_sock);

         /*Array<int> elem_ref(mesh.GetNE());
         Array<double> elem_rate(mesh.GetNE());

         // calculate error decrease rate and select h- or p- option for each element
         double rate_max = 0;
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            HPError &err = elem_hp_error[i];

            double rate_h = (err.err - err.err_h) / (err.dof_h - err.dof);
            double rate_p = (err.err - err.err_p) / (err.dof_p - err.dof);

            cout << "Element " << i
                 << ", order " << fespace.GetElementOrder(i) << "\n";
            cout << "  real error: " << sqrt(elem_error[i]) << endl;
            cout << "  dofs: " << err.dof << ", err: " << err.err << endl;
            cout << "  dofs_h: " << err.dof_h << ", err_h: " << err.err_h
                 << " -> rate_h: " << rate_h << endl;
            cout << "  dofs_p: " << err.dof_p << ", err_p: " << err.err_p
                 << " -> rate_p: " << rate_p << endl;

            if (rate_h > rate_p || err.dof_p > max_order*max_order || rate_h < 0.0)
            {
               elem_ref[i] = 0;
               elem_rate[i] = rate_h;
               cout << "  -- selected h ref\n" << endl;
            }
            else
            {
               elem_ref[i] = 1;
               elem_rate[i] = rate_p;
               cout << "  -- selected p ref\n" << endl;
            }

            rate_max = std::max(elem_rate[i], rate_max);
         }*/

         //const double rate_threshold = 0.9;

         // p-refine elements
         int h_refined = 0, p_refined = 0;
         double err_max = sqrt(elem_error.Max());
         Array<Refinement> refinements;
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            /*if (elem_rate[i] >= rate_threshold * rate_max ||
                elem_rate[i] < 0.0)*/
            if (sqrt(elem_error[i]) > ref_threshold * err_max)
            {
               const HPError &err = elem_hp_error[i];
               cout << "Element " << i << ": err_h = "
                    << err.err_h << ", err_p = " << err.err_p << endl;

               int p = fespace.GetElementOrder(i);
               if (err.err_p < err.err_h && p < max_order)
               //if (elem_ref[i] == 1) // p ref
               {
                  fespace.SetElementOrder(i, p+1);
                  p_refined++;
               }
               else
               {
                  int p = fespace.GetElementOrder(i);
                  fespace.SetElementOrder(i, HalfOrder(p));
                  refinements.Append(Refinement(i));
                  h_refined++;
               }
            }
         }

         if (pause)
         {
            cout << "Press ENTER to continue...";
            cin.get();
         }

         // Update the space, interpolate the solution. FIXME
         fespace.Update(false);
         //x.Update(); // NOT IMPLEMENTED YET

         // h-refine elements
         mesh.GeneralRefinement(refinements, -1, nc_limit);

         cout << " h-refined = " << h_refined
              << ", p-refined = " << p_refined << endl;
      }
      else // !hp
      {
         if (pause)
         {
            cout << "Press ENTER to continue...";
            cin.get();
         }

         Array<Refinement> refinements;
         double err_max = sqrt(elem_error.Max());
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            if (sqrt(elem_error[i]) > ref_threshold * err_max)
            {
               int type = aniso ? ref_type[i] : 7;
               refinements.Append(Refinement(i, type));
            }
         }
         mesh.GeneralRefinement(refinements, -1, nc_limit);
      }

      // Update the space, interpolate the solution.
      fespace.Update(false);
      x.Update();

      // Inform also the bilinear and linear forms that the space has changed.
      bf.Update();
      lf.Update();
   }

   return 0;
}



