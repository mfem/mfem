// Extract level-set from a 2 material mesh
#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include <iostream>
#include <fstream>
#include "tmop-fitting.hpp"

using namespace mfem;
using namespace std;

void ExtendRefinementListToNeighbors(ParMesh &pmesh, Array<int> &intel)
{
   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);
   const int quad_order = 4;

   el_to_refine = 0.0;

   for (int i = 0; i < intel.Size(); i++)
   {
      el_to_refine(intel[i]) = 1.0;
   }

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   el_to_refine.ExchangeFaceNbrData();
   GridFunctionCoefficient field_in_dg(&el_to_refine);
   lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      Array<int> dofs;
      Vector x_vals;
      lhfespace.GetElementDofs(e, dofs);
      const IntegrationRule &ir =
         irRules.Get(pmesh.GetElementGeometry(e), quad_order);
      lhx.GetValues(e, ir, x_vals);
      double max_val = x_vals.Max();
      if (max_val > 0)
      {
         intel.Append(e);
      }
   }

   intel.Sort();
   intel.Unique();
}

void GetMaterialInterfaceElements(ParMesh *pmesh, ParGridFunction &mat,
                                  Array<int> &intel)
{
   intel.SetSize(0);
   mat.ExchangeFaceNbrData();
   const int NElem = pmesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");
   for (int f = 0; f < pmesh->GetNumFaces(); f++ )
   {
      Array<int> nbrs;
      pmesh->GetFaceAdjacentElements(f,nbrs);
      Vector matvals;
      Array<int> vdofs;
      Vector vec;
      Array<int> els;
      //if there is more than 1 element across the face.
      if (nbrs.Size() > 1)
      {
         matvals.SetSize(nbrs.Size());
         for (int j = 0; j < nbrs.Size(); j++)
         {
            if (nbrs[j] < NElem)
            {
               matvals(j) = mat(nbrs[j]);
               els.Append(nbrs[j]);
            }
            else
            {
               const int Elem2NbrNo = nbrs[j] - NElem;
               mat.ParFESpace()->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs);
               mat.FaceNbrData().GetSubVector(vdofs, vec);
               matvals(j) = vec(0);
            }
         }
         if (matvals(0) != matvals(1))
         {
            intel.Append(els);
         }
      }
   }
}

void OptimizeMeshWithAMRAroundZeroLevelSetOfGF(ParMesh &pmesh,
                                               ParGridFunction &ls_coeff_in,
                                               int amr_iter, int neighbors,
                                               ParGridFunction &distance_s,
                                               const int quad_order = 5,
                                               Array<ParGridFunction *> *pgf_to_update = NULL)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*ls_coeff_in.ParFESpace()->GetMesh());
   finder.SetL2AvgType(FindPointsGSLIB::AvgType::NONE);

   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   finder.Interpolate(*pmesh.GetNodes(),
                      ls_coeff_in,
                      x,
                      pmesh.GetNodalFESpace()->GetOrdering());
   //   remap.ComputeAtNewPosition(*pmesh.GetNodes(),
   //                              x,
   //                              pmesh.GetNodalFESpace()->GetOrdering());
   x.ExchangeFaceNbrData();

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int iter = 0; iter < amr_iter; iter++)
   {
      el_to_refine = 0.0;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         DenseMatrix x_grad;
         h1fespace.GetElementDofs(e, dofs);
         const IntegrationRule &ir = irRules.Get(pmesh.GetElementGeometry(e),
                                                 quad_order);
         x.GetValues(e, ir, x_vals);
         double min_val = x_vals.Min();
         double max_val = x_vals.Max();
         // If the zero level set cuts the elements, mark it for refinement
         if (min_val < 0 && max_val >= 0)
         {
            el_to_refine(e) = 1.0;
         }
      }

      // Refine an element if its neighbor will be refined
      for (int inner_iter = 0; inner_iter < neighbors; inner_iter++)
      {
         el_to_refine.ExchangeFaceNbrData();
         GridFunctionCoefficient field_in_dg(&el_to_refine);
         lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            Array<int> dofs;
            Vector x_vals;
            lhfespace.GetElementDofs(e, dofs);
            const IntegrationRule &ir =
               irRules.Get(pmesh.GetElementGeometry(e), quad_order);
            lhx.GetValues(e, ir, x_vals);
            double max_val = x_vals.Max();
            if (max_val > 0)
            {
               el_to_refine(e) = 1.0;
            }
         }
      }

      // Make the list of elements to be refined
      Array<int> el_to_refine_list;
      for (int e = 0; e < el_to_refine.Size(); e++)
      {
         if (el_to_refine(e) > 0.0)
         {
            el_to_refine_list.Append(e);
         }
      }

      int loc_count = el_to_refine_list.Size();
      int glob_count = loc_count;
      MPI_Allreduce(&loc_count, &glob_count, 1, MPI_INT, MPI_SUM,
                    pmesh.GetComm());
      MPI_Barrier(pmesh.GetComm());
      if (glob_count > 0)
      {
         pmesh.GeneralRefinement(el_to_refine_list, 1);
      }

      // Update
      h1fespace.Update();
      x.Update();
      finder.Interpolate(*pmesh.GetNodes(),
                         ls_coeff_in,
                         x,
                         pmesh.GetNodalFESpace()->GetOrdering());
      //      remap.ComputeAtNewPosition(*pmesh.GetNodes(),
      //                                 x,
      //                                 pmesh.GetNodalFESpace()->GetOrdering());

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();

      distance_s.ParFESpace()->Update();
      distance_s.Update();
      distance_s = x;
      //      remap.ComputeAtNewPosition(pmesh.GetNodes(),
      //                                 distance_s,
      //                                 pmesh.GetNodalFESpace()->GetOrdering());

      if (pgf_to_update != NULL)
      {
         for (int i = 0; i < pgf_to_update->Size(); i++)
         {
            (*pgf_to_update)[i]->ParFESpace()->Update();
            (*pgf_to_update)[i]->Update();
         }
      }
      //      std::cout << iter << " k10doneiamriter\n";
   }
}

// make tmop-fitting-ls_from_mat_mesh -j && mpirun -np 1 tmop-fitting-ls_from_mat_mesh -m greshodeformed.mesh  -rs 0 -amriter 8 -refint 1 -o 1 -jid 1

int main (int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   const int num_procs = Mpi::WorldSize();
   Hypre::Init();

   // 1. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int order     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   int quad_order        = 8;
   int amr_iters         = 0;
   int ref_int_neighbors   = 0;
   bool visualization    = true;
   int jobid  = 0;

   // 2. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&amr_iters, "-amriter", "--amr-iter",
                  "Number of amr iterations on background mesh");
   args.AddOption(&ref_int_neighbors, "-refint", "--refint",
                  "Layers of neighbors to refine");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);

   const int dim = mesh->Dimension();

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   if (myid == 0)
   {
      std::cout << "ParMesh setup\n";
   }

   for (int lev = 0; lev < rp_levels; lev++)
   {
      pmesh->UniformRefinement();
   }
   int neglob = pmesh->GetGlobalNE();
   int neglob_preamr = neglob;
   if (myid == 0)
   {
      std::cout << "k10-Number of elements in input mesh: " << neglob << std::endl;
   }

   if (pmesh->GetNodes() == NULL)
   {
      std::cout << "Setting mesh curvature to 1\n";
      pmesh->SetCurvature(1, 0, -1, 0);
   }


   L2_FECollection mat_coll(0, dim);
   ParFiniteElementSpace mat_fes(pmesh, &mat_coll);
   ParGridFunction mat(&mat_fes);
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      int attr = pmesh->GetAttribute(e);
      MFEM_VERIFY(attr==1 || attr == 2,"Only 2 materials supported right now.");
      mat(e) = attr == 1 ? -1.0 : 1.0; //should now be -1 or 1
   }

   HRefUpdater HRUpdater = HRefUpdater();

   // Setup background mesh for surface fitting
   // Define relevant spaces and gridfunctions
   ParMesh *pmesh_surf_fit_bg = NULL;
   Mesh *mesh_surf_fit_bg = NULL;

   if (dim == 2)
   {
      mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL,
                                                        true));
   }
   else if (dim == 3)
   {
      mesh_surf_fit_bg = new Mesh(Mesh::MakeCartesian3D(4, 4, 4, Element::HEXAHEDRON,
                                                        true));
   }
   mesh_surf_fit_bg->EnsureNCMesh(true);
   for (int lev = 0; lev < rs_levels; lev++)
   {
      mesh_surf_fit_bg->UniformRefinement(0);
   }
   pmesh_surf_fit_bg = new ParMesh(MPI_COMM_WORLD, *mesh_surf_fit_bg);
   delete mesh_surf_fit_bg;

   pmesh_surf_fit_bg->SetCurvature(order, 0, -1, 0);
   Vector p_min(dim), p_max(dim);
   pmesh->GetBoundingBox(p_min, p_max);
   GridFunction &x_bg = *pmesh_surf_fit_bg->GetNodes();
   const int num_nodes = x_bg.Size() / dim;
   for (int i = 0; i < num_nodes; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         double length_d = p_max(d) - p_min(d),
                extra_d = 0.0 * length_d;
         x_bg(i + d*num_nodes) = p_min(d) - extra_d +
                                 x_bg(i + d*num_nodes) * (length_d + 2*extra_d);
      }
   }

   FiniteElementCollection *surf_fit_bg_fec = NULL;
   ParFiniteElementSpace *surf_fit_bg_fes = NULL;
   ParGridFunction *surf_fit_bg_gf0 = NULL;
   surf_fit_bg_fec = new H1_FECollection(order, dim);
   surf_fit_bg_fes = new ParFiniteElementSpace(pmesh_surf_fit_bg, surf_fit_bg_fec);
   surf_fit_bg_gf0 = new ParGridFunction(surf_fit_bg_fes);


   if (visualization)
   {
      socketstream vis1, vis2, vis3, vis4;

      common::VisualizeField(vis1, "localhost", 19916, *surf_fit_bg_gf0,
                             "Surface dof", 400, 000, 400, 400);
   }

   OptimizeMeshWithAMRAroundZeroLevelSetOfGF(*pmesh_surf_fit_bg,
                                             mat, amr_iters,
                                             ref_int_neighbors,
                                             *surf_fit_bg_gf0,
                                             quad_order);
   neglob = pmesh_surf_fit_bg->GetGlobalNE();
   pmesh_surf_fit_bg->Rebalance();
   surf_fit_bg_gf0->ParFESpace()->Update();
   surf_fit_bg_gf0->Update();


   if (visualization)
   {
      socketstream vis1, vis2, vis3, vis4;

      common::VisualizeField(vis1, "localhost", 19916, *surf_fit_bg_gf0,
                             "Remap 1", 400, 400, 400, 400);
   }

   ParGridFunction diff(*surf_fit_bg_gf0);
   if (myid == 0)
   {
      std::cout << neglob << " k10donemanualinterpolation and now compute distance\n";
   }

   if (visualization)
   {
      socketstream vis1, vis2, vis3, vis4;
      common::VisualizeField(vis1, "localhost", 19916, *surf_fit_bg_gf0,
                             "Remap 2", 800, 400, 400, 400);
   }



   {
      DataCollection *dc = NULL;
      dc = new VisItDataCollection("ls_input_"+std::to_string(jobid), pmesh);
      dc->RegisterField("level-set", &mat);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
      delete dc;
   }

   //    std::cout << myid << " " << pmesh_surf_fit_bg->GetNE() << " k10ne\n";
   ParGridFunction gftemp(*surf_fit_bg_gf0);
   GridFunctionCoefficient gfc(&gftemp);
   ComputeScalarDistanceFromLevelSet(*pmesh_surf_fit_bg, gfc,
                                     *surf_fit_bg_gf0, 6, 0, 50);

   if (visualization)
   {
      socketstream vis1, vis2, vis3, vis4;
      common::VisualizeField(vis1, "localhost", 19916, *surf_fit_bg_gf0,
                             "Remap 3", 000, 400, 400, 400);
   }

   {
      DataCollection *dc = NULL;
      dc = new VisItDataCollection("ls_from_mat_"+std::to_string(jobid),
                                   pmesh_surf_fit_bg);
      dc->RegisterField("level-set", surf_fit_bg_gf0);
      dc->RegisterField("level-set-exact", &gftemp);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
      delete dc;
   }

   {
      ostringstream mesh_name;
      mesh_name << "ls_from_mat_" + std::to_string(jobid) + ".mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh_surf_fit_bg->PrintAsOne(mesh_ofs);
   }
   {
      ostringstream mesh_name;
      mesh_name << "ls_from_mat_" + std::to_string(jobid) + ".gf";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      surf_fit_bg_gf0->SaveAsOne(mesh_ofs);
   }
   if (myid == 0)
   {
      std::cout << neglob << " k10don\n";
   }



   return 0;
}
