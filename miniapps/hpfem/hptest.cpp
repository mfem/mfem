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

const char* keys = "Rjlmc*******";

struct HPRefinement : public Refinement
{
   int orders[4];

   HPRefinement() = default;

   HPRefinement(int index, int type = 7)
      : Refinement(index, type)
   {
      orders[0] = orders[1] = orders[2] = orders[3] = 0;
   }
};

struct HPCandidate
{
   double err;
   int dof;
   int orders[4];
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
   //return p/2 + 1;
   return (p + 1) / 2;
   //return max(1, p/2);

}

bool ContainsVertex(Mesh *mesh, int elem, const Vertex& vert)
{
   Array<int> v;
   mesh->GetElementVertices(elem, v);
   for (int j = 0; j < v.Size(); j++)
   {
      double* vertex = mesh->GetVertex(v[j]);
      double dist = 0.0;
      for (int l = 0; l < 2; l++)
      {
         double d = vert(l) - vertex[l];
         dist += d*d;
      }
      if (dist == 0) { return true; }
   }
   return false;
}

void Solve(FiniteElementSpace *fespace, GridFunction *sln,
           Coefficient *exsol, Coefficient *rhs,
           bool pa, int int_order)
{
   DomainLFIntegrator *dlfi = new DomainLFIntegrator(*rhs);
   Geometry::Type geom = fespace->GetMesh()->GetElementGeometry(0);
   dlfi->SetIntRule(&IntRules.Get(geom, int_order));

   // Assemble the linear form. The right hand side is manufactured
   // so that the solution is the analytic solution.
   LinearForm lf(fespace);
   lf.AddDomainIntegrator(dlfi);
   lf.Assemble();

   // Assemble the bilinear form.
   BilinearForm bf(fespace);
   if (pa) { bf.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   bf.AddDomainIntegrator(new DiffusionIntegrator());
   bf.Assemble();

   // Set Dirichlet boundary values in the GridFunction x.
   // Determine the list of Dirichlet true DOFs in the linear system.
   Array<int> ess_bdr(fespace->GetMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   *sln = 0; // FIXME
   sln->ProjectBdrCoefficient(*exsol, ess_bdr);
   Array<int> ess_tdof_list;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 16. Create the linear system: eliminate boundary conditions, constrain
   //     hanging nodes and possibly apply other transformations. The system
   //     will be solved for true (unconstrained) DOFs only.
   OperatorPtr A;
   Vector B, X;

   const int copy_interior = 1;
   bf.FormLinearSystem(ess_tdof_list, *sln, lf, A, X, B, copy_interior);

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
   bf.RecoverFEMSolution(X, lf, *sln);
}


void FindHPRef(int elem, FiniteElementSpace *fes, FiniteElementSpace *fes_p,
               FiniteElementSpace *fes_h, FiniteElementSpace *fes_h2,
               Array<double> elemError, Array<double> elemError_p,
               Array<double> elemError_h, Array<double> elemError_h2,
               int max_order, std::map<int, HPRefinement> *hp_refs)
{
   Mesh* mesh_h = fes_h->GetMesh();

   // Find sons of elem
   int sons[4];
   const CoarseFineTransformations tr = mesh_h->GetRefinementTransforms();
   int l = 0;
   for (int i = 0; i < mesh_h->GetNE(); i++)
   {
      int j = tr.embeddings[i].parent;
      if (j == elem)
      {
         sons[l] = i;
         l++;
      }
   }

   int o = fes->GetElementOrder(elem);
   int op = fes_p->GetElementOrder(elem);

   double s_err = sqrt(elemError[elem]);
   cout << "Element " << elem << " (order " << o << "): err = " << s_err << ", dof = " << o*o << "\n  ";


   // initialize candidates
   int n_cand = 17;
   HPCandidate candidate[n_cand];
   for (int id = 0; id < n_cand; id++)
   {
      candidate[id].err = 0;
      candidate[id].dof = 0;
   }

   // p-candidate:
   candidate[0].err = elemError_p[elem];
   candidate[0].dof = op*op;
   candidate[0].orders[0] = op;

   // hp-candidates
   int cand_id = 1;
   int k[4];
   for (k[0] = 0; k[0] < 2; k[0]++)
   {
      for (k[1] = 0; k[1] < 2; k[1]++)
      {
         for (k[2] = 0; k[2] < 2; k[2]++)
         {
            for (k[3] = 0; k[3] < 2; k[3]++)
            {
               for (int son = 0; son < 4; son++)
               {
                  if (k[son] == 0)
                  {
                     int oh = fes_h->GetElementOrder(sons[son]);
                     candidate[cand_id].err += elemError_h[sons[son]];
                     candidate[cand_id].dof += oh*oh;
                     candidate[cand_id].orders[son] = oh;
                  }
                  else
                  {
                     int oh2 = fes_h2->GetElementOrder(sons[son]);
                     candidate[cand_id].err += elemError_h2[sons[son]];
                     candidate[cand_id].dof += oh2*oh2;
                     candidate[cand_id].orders[son] = oh2;
                  }
               }
               cand_id++;
            }
         }
      }
   }


   double max_rate = -1000.0;
   int best_id;

   for (int id = 0; id < n_cand; id++)
   {
      if (id == 0)
      {
         cout << "Candidate: " << id << ", err = " << sqrt(candidate[id].err) << ", dof = " << candidate[id].dof
              <<  ", orders = " << candidate[id].orders[0] << "\n  ";
      }
      else
      {
         cout << "Candidate: " << id << ", err = " << sqrt(candidate[id].err) << ", dof = " << candidate[id].dof
              <<  ", orders = " << candidate[id].orders[0] << " " << candidate[id].orders[1]
              << " " << candidate[id].orders[2] << " " << candidate[id].orders[3] << "\n  ";
      }

      double rate = (s_err - sqrt(candidate[id].err)) / (candidate[id].dof - o*o + 1);
      if (rate > max_rate && candidate[id].orders[0] <= max_order)
      {
         max_rate = rate;
         best_id = id;
      }
   }

   if (best_id == 0)
   {
      cout << "Best candidate: " << best_id << ", err = " << sqrt(candidate[best_id].err) << ", dof = "
           << candidate[best_id].dof <<  ", orders = " << candidate[best_id].orders[0] << "\n  ";
   }
   else
   {
      cout << "Best candidate: " << best_id << ", err = " << sqrt(candidate[best_id].err) << ", dof = "
           << candidate[best_id].dof <<  ", orders = " << candidate[best_id].orders[0] << " " << candidate[best_id].orders[1]
           << " " << candidate[best_id].orders[2] << " " << candidate[best_id].orders[3] << "\n  ";
   }

   // Put the best candidate into hp_refinements
   (*hp_refs)[elem].index = elem;
   if (best_id > 0) { (*hp_refs)[elem].ref_type = 7; }
   else { (*hp_refs)[elem].ref_type = 0; }
   for (int i = 0; i < 4; i++)
   {
      (*hp_refs)[elem].orders[i] = candidate[best_id].orders[i];
   }

}


int main(int argc, char *argv[])
{
   // Parse command-line options.
   int dim = 2;
   int problem = 1;
   int order = 1;
   double ref_threshold = 0.7;
   bool aniso = false;
   bool hp = true;
   int max_order = 10;
   int int_order = 30;
   bool relaxed_hp = false;
   const char *conv_file = "conv.err";
   bool wait = false;
   int nc_limit = 2;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type: 0 = L-shaped, 1 = inner layer.");
   args.AddOption(&dim, "-dim", "--dimension", "Dimension (2 or 3).");
   args.AddOption(&order, "-o", "--order",
                  "Initial mesh finite element order (polynomial degree).");
   args.AddOption(&hp, "-hp", "--hp", "-no-hp", "--no-hp",
                  "Enable hp refinement.");
   args.AddOption(&relaxed_hp, "-x", "--relaxed-hp", "-no-x", "--no-relaxed-hp",
                  "Set relaxed hp conformity.");
   args.AddOption(&conv_file, "-f", "--file",
                  "Convergence file to use.");
   args.AddOption(&ref_threshold, "-rt", "--ref-threshold",
                  "Refine elements with error larger than threshold * max_error.");
   args.AddOption(&nc_limit, "-nc", "--nc-limit",
                  "Set maximum difference of refinement levels of adjacent elements.");
   args.AddOption(&wait, "-w", "--wait", "-no-w", "--no-wait",
                  "Wait for user input after each iteration.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-dev", "--device",
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

   MFEM_VERIFY(dim >= 2 && dim <= 3, "Invalid dimension.");
   MFEM_VERIFY(problem >= 0 && problem <= 1, "Invalid problem type.");

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // Load and adjust the Mesh
   const char *mesh_file =
      problem ? ((dim == 3) ? "layer-hex.mesh" : "layer-quad.mesh")
              : ((dim == 3) ? "fichera-hex.mesh" : "lshape-quad.mesh");

   Mesh mesh(mesh_file, 1, 1);

   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }
   mesh.EnsureNCMesh(true);
   //mesh.UniformRefinement();

   // We don't support mixed meshes at the moment
   MFEM_VERIFY(mesh.GetNumGeometries(dim) == 1, "Mixed meshes not supported.");
   //Geometry::Type geom = mesh.GetElementGeometry(0);

   // Prepare exact solution Coefficients
   FunctionCoefficient exsol(
      problem ? ((dim == 3) ? layer3_exsol  : layer2_exsol)
              : ((dim == 3) ? fichera_exsol : lshape_exsol) );

   VectorFunctionCoefficient exgrad(dim,
      problem ? ((dim == 3) ? layer3_exgrad  : layer2_exgrad)
              : ((dim == 3) ? fichera_exgrad : lshape_exgrad) );

   FunctionCoefficient rhs(
      problem ? ((dim == 3) ? layer3_laplace  : layer2_laplace)
              : ((dim == 3) ? fichera_laplace : lshape_laplace) );

   // Define a finite element space on the mesh. Initially the polynomial
   // order is constant everywhere.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   fespace.SetRelaxedHpConformity(relaxed_hp);

   // All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");

   // Connect to GLVis.
   socketstream sol_sock, ord_sock, dbg_sock[3], err_sock;

   std::ofstream conv(conv_file);

   // The main AMR loop. In each iteration we solve the problem on the
   // current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 100000;
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // Solve for the current mesh: (h, p)
      GridFunction sol(&fespace);
      Solve(&fespace, &sol, &exsol, &rhs, pa, int_order);


      // 19. Send solution by socket to the GLVis server.
      if (visualization)
      {
         GridFunction *vis_x = ProlongToMaxOrder(&sol);
         VisualizeField(sol_sock, *vis_x, "Solution", keys, 600, 500, 0, 70);


         GridFunction projsol(&fespace);
         Vector tmp = sol;
         projsol.ProjectCoefficient(exsol);
         MakeConforming(projsol);
         projsol -= tmp;
         vis_x = ProlongToMaxOrder(&projsol);
         VisualizeField(err_sock, *vis_x, "Projected Solution", keys, 600, 500, 0, 70);

         delete vis_x;

         L2_FECollection l2fec(0, dim);
         FiniteElementSpace l2fes(&mesh, &l2fec);
         GridFunction orders(&l2fes);
         for (int i = 0; i < orders.Size(); i++)
         {
            orders(i) = fespace.GetElementOrder(i);
         }
         VisualizeField(ord_sock, orders, "Orders", keys, 600, 500, 0, 620);
         //ord_sock << "valuerange 1 5\n" << flush;
      }

      if (cdofs > max_dofs)
      {
         cout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // Calculate the H^1_0 errors of elements as well as the total error.
      Array<double> elemError;
      Array<int> ref_type;
      double error = sqrt(CalculateH10Error2(&sol, &exgrad, &elemError,
                                             &ref_type, int_order));

      // Save dofs and error for convergence plot
      conv << cdofs << " " << error << endl;
      cout << cdofs << " " << error << endl;   

      // Project the exact solution to h-refined and p-refined versions of the
      // mesh and determine whether to refine elements in 'h' or in 'p'.
      if (hp)
      {         
//         // Prepare refined mesh and enriched spaces
//         Mesh mesh_h(mesh);
//         FiniteElementSpace fes_h(fespace, &mesh_h); // FIXME: copy variable orders
//         FiniteElementSpace fes_h2(fespace, &mesh_h); // FIXME: copy variable orders
//         FiniteElementSpace fes_p(fespace, &mesh);

//         for (int i = 0; i < mesh.GetNE(); i++)
//         {
//             int p = fespace.GetElementOrder(i);
//             fes_h.SetElementOrder(i, HalfOrder(p));
//             fes_h2.SetElementOrder(i, HalfOrder(p)+1);
//             fes_p.SetElementOrder(i, p+1);
//         }
//         fes_h.Update(false);
//         fes_h2.Update(false);
//         fes_p.Update(false);

//         mesh_h.UniformRefinement();
//         fes_h.Update(false);
//         fes_h2.Update(false);

//         // Solve for h-refined mesh (h/2, HalfOrder)
//         GridFunction sol_h(&fes_h);
//         Solve(&fes_h, &sol_h, &exsol, &rhs, pa, int_order);

//         // Solve for h-refined mesh (h/2, HalfOrder+1)
//         GridFunction sol_h2(&fes_h2);
//         Solve(&fes_h2, &sol_h2, &exsol, &rhs, pa, int_order);

//         // Solve for p-refined mesh (h, p+1)
//         GridFunction sol_p(&fes_p);
//         Solve(&fes_p, &sol_p, &exsol, &rhs, pa, int_order);



//         Array<double> elemError_h;
//         Array<double> elemError_h2;
//         Array<double> elemError_p;

//         CalculateH10Error2(&sol_h, &exgrad, &elemError_h, &ref_type, int_order);
//         CalculateH10Error2(&sol_h2, &exgrad, &elemError_h2, &ref_type, int_order);
//         CalculateH10Error2(&sol_p, &exgrad, &elemError_p, &ref_type, int_order);

         int h_refined = 0, p_refined = 0;
         double err_max = sqrt(elemError.Max());

         Array<Refinement> refinements;
         std::map<int, HPRefinement> hp_refs;

         for (int i = 0; i < mesh.GetNE(); i++)
         {
            if (sqrt(elemError[i]) > ref_threshold * err_max)
            {

//               FindHPRef(i, &fespace, &fes_p, &fes_h, &fes_h2, elemError, elemError_p, elemError_h, elemError_h2, max_order, &hp_refs);

               // hp-strategy based on known singularity
               int p = fespace.GetElementOrder(i);
               Vertex center(0.0, 0.0);
               if (ContainsVertex(&mesh, i, center))
               {
                  hp_refs[i].ref_type = 7;
                  hp_refs[i].orders[0] = hp_refs[i].orders[1] = hp_refs[i].orders[2] = hp_refs[i].orders[3] = p;
               }
               else if (p == max_order)
               {
                  hp_refs[i].ref_type = 7;
                  hp_refs[i].orders[0] = hp_refs[i].orders[1] = hp_refs[i].orders[2] = hp_refs[i].orders[3] = HalfOrder(p);
               }
               else
               {
                  hp_refs[i].ref_type = 0;
                  hp_refs[i].orders[0] = p+1;
               }


               if (hp_refs[i].ref_type > 0)
               {
                  refinements.Append(Refinement(i));
                  h_refined++;
                  cout << "=> h-refined" << endl;
               }
               else
               {
                  int p = fespace.GetElementOrder(i);
                  fespace.SetElementOrder(i, p+1);
                  p_refined++;
                  cout << "=> p-refined" << endl;
               }
            }
         }

         // Update the space, interpolate the solution. FIXME
         fespace.Update(false);
         //x.Update(); // NOT IMPLEMENTED YET

         // h-refine elements
         mesh.GeneralRefinement(refinements, -1, nc_limit);
         fespace.Update(false);


         if (refinements.Size())
         {
            // Assign sons to all parents
            std::map<int, std::vector<int>> ref_sons;
            const CoarseFineTransformations tr = mesh.GetRefinementTransforms();
            for (int i = 0; i < mesh.GetNE(); i++)
            {
               int j = tr.embeddings[i].parent;
               ref_sons[j].push_back(i);
            }

            // set orders for h-refined elements
            for (int i = 0; i < mesh.GetNE(); i++)
            {
               int j = tr.embeddings[i].parent;
               if (sqrt(elemError[j]) > ref_threshold * err_max)
               {
                  if (hp_refs[j].ref_type > 0)
                  {
                     for (int k = 0; k < 4; k++)
                     {
                        fespace.SetElementOrder(ref_sons[j][k], hp_refs[j].orders[k]);
                     }
                  }
               }
            }
         }
         cout << "\nh-refined = " << h_refined
              << ", p-refined = " << p_refined << endl;


      }
      else // !hp
      {
         if (wait)
         {
            cout << "Press ENTER to continue...";
            cin.get();
         }

         Array<Refinement> refinements;
         double err_max = sqrt(elemError.Max());
         for (int i = 0; i < mesh.GetNE(); i++)
         {
            if (sqrt(elemError[i]) > ref_threshold * err_max)
            {
               int type = aniso ? ref_type[i] : 7;
               refinements.Append(Refinement(i, type));
            }
         }
         mesh.GeneralRefinement(refinements, -1, nc_limit);
      }

      // Update the space, interpolate the solution.
      fespace.Update(false);
      sol.Update();

      if (wait)
      {
         cout << "Press ENTER to continue...";
         cin.get();
      }

   }

   return 0;
}



