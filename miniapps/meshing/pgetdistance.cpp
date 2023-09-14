
#include "mesh-fitting.hpp"

using namespace mfem;
using namespace std;

// Fischer-Tropsch like geometry
double reactor_no_rects(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.2);

   double r1 = 0.2;
   double r2 = 1.0;
   double in_trapezium_val = in_trapezium(x, 0.05, 0.1, r2-r1);

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.4;
   double k = 2;
   double t = 0.2;
   double in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   //   double in_rectangle_val = in_rectangle(x, 0.99, 0.0, 0.12, 0.35);
   //   return_val = max(return_val, in_rectangle_val);

   //   double in_rectangle_val2 = in_rectangle(x, 0.99, 0.5, 0.12, 0.28);
   //   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}

double reactor2(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.2);

   double r1 = 0.2;
   double r2 = 1.0;
   double in_trapezium_val = in_trapezium(x, 0.05, 0.1, r2-r1);

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.4;
   double k = 2;
   double t = 0.15;
   double in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   double in_rectangle_val = in_rectangle(x, 0.99, 0.0, 0.12, 0.35);
   return_val = max(return_val, in_rectangle_val);

   double in_rectangle_val2 = in_rectangle(x, 0.99, 0.5, 0.12, 0.28);
   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}

// make pgetdistance -j && mpirun -np 6 pgetdistance -sls 11 -amriter 5 -o 3 -dx 0.01 -jid 1 -ds 0.05
int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Set the method's default parameters.
   const char *mesh_file = "square01.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 1;
   int rp_levels         = 0;
   bool visualization    = true;
   int amr_iters         = 1;
   int surf_ls_type      = 1;
   double dxv = 0;
   int jobid = 0;
   double ds = 0.2;
   bool comp_dist = true;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&amr_iters, "-amriter", "--amr-iter",
                  "Number of amr iterations on background mesh");
   args.AddOption(&surf_ls_type, "-sls", "--sls",
                  "Choice of level set function.");
   args.AddOption(&dxv, "-dx", "--dx",
                  "Shift of level set function.");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");
   args.AddOption(&ds, "-ds", "--ds",
                  "Domain stretching beyond current mesh.");
   args.AddOption(&comp_dist, "-dist", "--dist", "-no-dist",
                  "--no-dist",
                  "Enable or disable distance calculation.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   const int dim = mesh->Dimension();

   // Define level-set coefficient
   FunctionCoefficient *ls_coeff = NULL;
   if (surf_ls_type == 1) // reactor
   {
      ls_coeff = new FunctionCoefficient(circle_level_set);
   }
   else if (surf_ls_type == 3) // reactor
   {
      ls_coeff = new FunctionCoefficient(squircle_level_set);
   }
   else if (surf_ls_type == 11) // reactor
   {
      ls_coeff = new FunctionCoefficient(reactor2);
   }
   else if (surf_ls_type == 12) // reactor
   {
      ls_coeff = new FunctionCoefficient(reactor_no_rects);
   }
   else
   {
      MFEM_ABORT("Surface fitting level set type not implemented yet.")
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // 4. Setup background mesh for surface fitting
   ParMesh *pmesh_surf_fit_bg = NULL;
   Mesh *mesh_surf_fit_bg = NULL;
   if (dim == 2)
   {
      mesh_surf_fit_bg =
         new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true));
   }
   else if (dim == 3)
   {
      mesh_surf_fit_bg =
         new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON, true));
   }
   mesh_surf_fit_bg->EnsureNCMesh();
   pmesh_surf_fit_bg = new ParMesh(MPI_COMM_WORLD, *mesh_surf_fit_bg);
   delete mesh_surf_fit_bg;

   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   ParFiniteElementSpace *pfespace =
      new ParFiniteElementSpace(pmesh, fec, dim, 0);
   pmesh->SetNodalFESpace(pfespace);

   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);
   x.SetTrueVector();

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   L2_FECollection mat_coll(0, dim);
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   ParFiniteElementSpace surf_fit_fes(pmesh, &surf_fit_fec);

   // Background mesh FECollection, FESpace, and GridFunction
   H1_FECollection *surf_fit_bg_fec = NULL;
   ParFiniteElementSpace *surf_fit_bg_fes = NULL;
   ParGridFunction *surf_fit_bg_gf0 = NULL;

   pmesh_surf_fit_bg->SetCurvature(1);

   Vector p_min(dim), p_max(dim);
   pmesh->GetBoundingBox(p_min, p_max);
   GridFunction &x_bg = *pmesh_surf_fit_bg->GetNodes();
   const int num_nodes = x_bg.Size() / dim;
   for (int i = 0; i < num_nodes; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         double length_d = p_max(d) - p_min(d),
                extra_d = ds * length_d;
         x_bg(i + d*num_nodes) = p_min(d) - extra_d +
                                 x_bg(i + d*num_nodes) * (length_d + 2*extra_d);
      }
   }
   surf_fit_bg_fec = new H1_FECollection(mesh_poly_deg, dim);
   surf_fit_bg_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg, surf_fit_bg_fec);
   surf_fit_bg_gf0 = new ParGridFunction(surf_fit_bg_fes);

   OptimizeMeshWithAMRAroundZeroLevelSet(*pmesh_surf_fit_bg, *ls_coeff,
                                         amr_iters, *surf_fit_bg_gf0);
   pmesh_surf_fit_bg->Rebalance();
   surf_fit_bg_fes->Update();
   surf_fit_bg_gf0->Update();
   surf_fit_bg_gf0->ProjectCoefficient(*ls_coeff);

   if (comp_dist)
   {
      ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, *ls_coeff,
                                        *surf_fit_bg_gf0, 2);
      if (dxv != 0)
      {
         *surf_fit_bg_gf0 += -dxv;
         ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, *ls_coeff,
                                           *surf_fit_bg_gf0);
         *surf_fit_bg_gf0 += dxv;
      }
   }


   if (visualization)
   {
      x0 -= x;
      socketstream vis;
      common::VisualizeField(vis, "localhost", 19916, *surf_fit_bg_gf0,
                             "Distance on bg", 900, 400, 300, 300, "jRmclA");
   }

   {
      DataCollection *dc = NULL;
      dc = new VisItDataCollection("DistBG_"+std::to_string(jobid),
                                   pmesh_surf_fit_bg);
      dc->RegisterField("Level-set", surf_fit_bg_gf0);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
      delete dc;
   }
   {
      ostringstream mesh_name;
      mesh_name << "BGMesh"+std::to_string(jobid)+".mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh_surf_fit_bg->PrintAsOne(mesh_ofs);
   }
   {
      ostringstream gf_name;
      gf_name << "BGMesh"+std::to_string(jobid)+".gf";
      ofstream gf_ofs(gf_name.str().c_str());
      gf_ofs.precision(8);
      surf_fit_bg_gf0->SaveAsOne(gf_ofs);
   }

   delete surf_fit_bg_gf0;
   delete surf_fit_bg_fes;
   delete surf_fit_bg_fec;
   delete pfespace;
   delete fec;
   delete pmesh_surf_fit_bg;
   delete pmesh;

   return 0;
}
