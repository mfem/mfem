// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// -----------------------------------------------------------------------------
// Find Points Miniapp: Evaluate grid function in physical space - Parallel Ver.
// -----------------------------------------------------------------------------
//
// This miniapp demonstrates the interpolation of a high-order grid function on
// a set of points in physical-space. The miniapp is based on GSLIB-FindPoints,
// which provides two key functionalities. First, for a given set of points in
// the physical-space, it determines the computational coordinates (element
// number, reference-space coordinates inside the element, and processor number
// [in parallel]) for each point. Second, based on computational coordinates, it
// interpolates a grid function in the given points. Inside GSLIB, computation
// of the coordinates requires use of a Hash Table to identify the candidate
// processor and element for each point, followed by the Newton's method to
// determine the reference-space coordinates inside the candidate element.
//
// Compile with: make pfindpts
//
// Sample runs:
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-q3.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-p4-tri.mesh -o 4
//    mpirun -np 2 pfindpts -m ../../data/inline-tri.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-tet.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-hex.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-wedge.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/amr-quad.mesh -o 2


#include "mfem.hpp"

using namespace mfem;
using namespace std;

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += x(d) * x(d); }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

int main (int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Set the method's default parameters.
   const char *mesh_file = "../../data/rt-2d-q3.mesh";
   int mesh_poly_deg     = 3;
   int rs_levels         = 0;
   int rp_levels         = 0;
   bool visualization    = true;
   int fieldtype         = 0;
   int ncomp             = 1;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Field type: 0 - H1, 1 - L2, 2 - H(div).");
   args.AddOption(&ncomp, "-nc", "--ncomp",
                  "VDim for GridFunction");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   if (myid == 0)
   {
      cout << "Mesh curvature of the original mesh: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }

   // Mesh bounding box (for the full serial mesh).
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be positive.");
   mesh->GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   if (myid == 0)
   {
      cout << "--- Generating equidistant point for:\n"
           << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n"
           << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
      if (dim == 3)
      {
         cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
      }
   }

   // Distribute the mesh.
   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }

   // Curve the mesh based on the chosen polynomial degree.
   H1_FECollection fec(mesh_poly_deg, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fec, dim);
   pmesh.SetNodalFESpace(&pfespace);
   if (myid == 0)
   {
      cout << "Mesh curvature of the curved mesh: " << fec.Name() << endl;
   }

   MFEM_ASSERT(ncomp > 0, " Invalid input for ncomp.");
   int ncfinal = ncomp;
   GridFunction field_vals;
   H1_FECollection fech(mesh_poly_deg, dim);
   L2_FECollection fecl(mesh_poly_deg, dim);
   ND_FECollection fechdiv(mesh_poly_deg, dim);
   ParFiniteElementSpace *sc_fes = NULL;
   if (fieldtype == 0)
   {
      sc_fes = new ParFiniteElementSpace(&pmesh, &fech, ncomp);
      if (myid == 0) { std::cout << "H1-GridFunction\n"; }
   }
   else if (fieldtype == 1)
   {
      sc_fes = new ParFiniteElementSpace(&pmesh, &fecl, ncomp);
      if (myid == 0) { std::cout << "L2-GridFunction\n"; }
   }
   else if (fieldtype == 2)
   {
      sc_fes = new ParFiniteElementSpace(&pmesh, &fechdiv);
      ncfinal = 2;
      if (myid == 0) { std::cout << "H(div)-GridFunction\n"; }
   }
   field_vals.SetSpace(sc_fes);

   // Define a scalar function on the mesh.
   VectorFunctionCoefficient F(ncfinal, F_exact);
   field_vals.ProjectCoefficient(F);

   // Display the mesh and the field through glvis.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sout;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(8);
         sout << "solution\n" << pmesh << field_vals;
         if (dim == 2) { sout << "keys RmjA*****\n"; }
         if (dim == 3) { sout << "keys mA\n"; }
         sout << flush;
      }
   }

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box. Note
   // also that all tasks search the same points (not mandatory).
   const int pts_cnt_1D = 5;
   const int pts_cnt = pow(pts_cnt_1D, dim);
   Vector vxyz(pts_cnt * dim);
   if (dim == 2)
   {
      L2_QuadrilateralElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         vxyz(i)           = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i) = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
      }
   }
   else
   {
      L2_HexahedronElement el(pts_cnt_1D - 1, BasisType::ClosedUniform);
      const IntegrationRule &ir = el.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         vxyz(i)             = pos_min(0) + ip.x * (pos_max(0)-pos_min(0));
         vxyz(pts_cnt + i)   = pos_min(1) + ip.y * (pos_max(1)-pos_min(1));
         vxyz(2*pts_cnt + i) = pos_min(2) + ip.z * (pos_max(2)-pos_min(2));
      }
   }

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt*ncfinal);
   // FindPoints using GSLIB and interpolate
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Interpolate(pmesh, vxyz, field_vals, interp_vals);
   Array<unsigned int> code_out    = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();
   Vector dist_p_out = finder.GetDist();

   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   int npt = 0;
   for (int j = 0; j < ncfinal; j++)
   {
      for (int i = 0; i < pts_cnt; i++)
      {
         if (j == 1)
         {
            (task_id_out[i] == (unsigned)myid) ? found_loc++ : found_away++;
         }

         if (code_out[i] < 2)
         {
            for (int d = 0; d < dim; d++) { pos(d) = vxyz(d * pts_cnt + i); }
            Vector exact_val(ncfinal);
            F_exact(pos, exact_val);
            max_err  = std::max(max_err, fabs(exact_val(j) - interp_vals[npt]));
            max_dist = std::max(max_dist, dist_p_out(i));
            if (code_out[i] == 1 && j == 1) { face_pts++; }
         }
         else { if (j == 1) { not_found++; } }
         npt++;
      }
   }

   int pts_cnt_glob = pts_cnt,
       found_loc_glob = found_loc,
       found_away_glob = found_away,
       not_found_glob  = not_found,
       face_pts_glob   = face_pts;
   double max_err_glob = max_err,
          max_dist_glob = max_dist;

   MPI_Allreduce(&pts_cnt,    &pts_cnt_glob,    1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&found_loc,  &found_loc_glob,  1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&found_away, &found_away_glob, 1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&not_found,  &not_found_glob,  1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&face_pts,   &face_pts_glob,   1, MPI_INT, MPI_SUM,
                 MPI_COMM_WORLD);

   MPI_Allreduce(&max_err,  &max_err_glob,  1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&max_dist, &max_dist_glob, 1, MPI_DOUBLE, MPI_MAX,
                 MPI_COMM_WORLD);

   // We print only the task 0 result (other tasks should be identical except
   // the number of points found locally).
   if (myid == 0)
   {
      cout << setprecision(16)
           << "Searched points:      "   << pts_cnt_glob
           << "\nFound on local mesh:  " << found_loc_glob
           << "\nFound on other tasks: " << found_away_glob
           << "\nMax interp error:     " << max_err_glob
           << "\nMax dist (of found):  " << max_dist_glob
           << "\nPoints not found:     " << not_found_glob
           << "\nPoints on faces:      " << face_pts_glob << endl;
   }

   // Free the internal gslib data.
   finder.FreeData();
   MPI_Finalize();
   return 0;
}
