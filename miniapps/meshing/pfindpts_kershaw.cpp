// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
//    ----------------------------------------------------------------------
//    Parallel Find Points Miniapp: Evaluate grid function in physical space
//    ----------------------------------------------------------------------
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
//    mpirun -np 2 pfindpts_kershaw -d 2 -o 3 -mo 2 -hr
//    mpirun -np 2 pfindpts -m ../../data/inline-tri.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -po 1
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -po 1 -gfo 1 -nc 2
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -o 3 -hr
//    mpirun -np 2 pfindpts -m ../../data/inline-tet.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-hex.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/inline-wedge.mesh -o 3
//    mpirun -np 2 pfindpts -m ../../data/amr-quad.mesh -o 2
//    mpirun -np 2 pfindpts -m ../../data/rt-2d-q3.mesh -o 3 -mo 4 -ft 2
//    mpirun -np 2 pfindpts -m ../../data/inline-quad.mesh -ft 1 -no-vis -sr0
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 2
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 2 -hr
//    mpirun -np 2 pfindpts -m ../../data/square-mixed.mesh -o 2 -mo 3 -ft 2
//    mpirun -np 2 pfindpts -m ../../data/fichera-mixed.mesh -o 3 -mo 2
//    mpirun -np 2 pfindpts -m ../../data/inline-pyramid.mesh -o 1 -mo 1
//    mpirun -np 2 pfindpts -m ../../data/tinyzoo-3d.mesh -o 1 -mo 1

#include "mfem.hpp"
#include "../common/mfem-common.hpp"
#include "tmop-fitting.hpp"

using namespace mfem;
using namespace std;

double func_order = -1.0;

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += std::pow(x(d), func_order); }
   return res;
}

void grad_field_func(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   for (int d = 0; d < x.Size(); d++)
   {
      grad(d) = func_order*std::pow(x(d), func_order-1.0);
   }
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

double GetMinDet(ParMesh *pmesh, ParFiniteElementSpace *pfespace,
                 IntegrationRules *irules, int quad_order)
{
   double tauval = infinity();
   const int NE = pmesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(pfespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = pmesh->GetElementTransformation(i);

      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }

      const IntegrationRule &ir2 = pfespace->GetFE(i)->GetNodes();
      for (int j = 0; j < ir2.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir2.IntPoint(j));
         //         const IntegrationPoint ip = ir2.IntPoint(j);
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   double minJ0;
   MPI_Allreduce(&tauval, &minJ0, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   tauval = minJ0;
   return tauval;
}

// smooth = 1, fo = 2 so o=2
// make pfindpts_kershaw -j && mpirun -np 4 pfindpts_kershaw -d 2 -o 2 -mo 3 -ft 0 -rs 1 -npt 1000 -et 0 -smooth 1 -fo 2 -jid 1
// smooth = 2, fo = 2 so o=3*2 = 6
// make pfindpts_kershaw -j && mpirun -np 4 pfindpts_kershaw -d 2 -o 6 -mo 3 -ft 0 -rs 1 -npt 1000 -et 0 -smooth 2 -fo 2 -jid 1
int main (int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Set the method's default parameters.
   int dim               = 3;
   int order             = 3;
   int etype             = 0;
   int mesh_poly_deg     = 3;
   int npt               = 100;
   int rs_levels         = 0;
   int rp_levels         = 0;
   bool visualization    = true;
   int fieldtype         = 0;
   int ncomp             = 1;
   bool search_on_rank_0 = false;
   bool hrefinement      = false;
   int point_ordering    = 0;
   int gf_ordering       = 0;
   bool proj             = false; //L2 or H1 projection
   int smooth            = 2; //kershaw transformation parameter
   int jobid  = 0;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-d", "--dim",
                  "2D or 3D");
   args.AddOption(&func_order, "-fo", "--func-order",
                  "function order for exact.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&etype, "-et", "--element type",
                  "0-quad/hex, 1-tri/tet");
   args.AddOption(&npt, "-npt", "--npt",
                  "# points per proc");
   args.AddOption(&smooth, "-smooth", "--smooth",
                  "smooth parameter of kershaw");
   args.AddOption(&mesh_poly_deg, "-mo", "--mesh-order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&fieldtype, "-ft", "--field-type",
                  "Field type: 0 - H1, 1 - L2, 2 - H(div), 3 - H(curl).");
   //   args.AddOption(&ncomp, "-nc", "--ncomp",
   //                  "Number of components for H1 or L2 GridFunctions");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&proj, "-proj", "--proj", "-no-proj",
                  "--no-proj",
                  "Enable or disable GLVis visualization.");
   //   args.AddOption(&search_on_rank_0, "-sr0", "--search-on-r0", "-no-sr0",
   //                  "--no-search-on-r0",
   //                  "Enable search only on rank 0 (disable to search points on all tasks).");
   args.AddOption(&hrefinement, "-hr", "--h-refinement", "-no-hr",
                  "--no-h-refinement",
                  "Do random h refinements to mesh (does not work for pyramids).");
   //   args.AddOption(&point_ordering, "-po", "--point-ordering",
   //                  "Ordering of points to be found."
   //                  "0 (default): byNodes, 1: byVDIM");
   //   args.AddOption(&gf_ordering, "-gfo", "--gridfunc-ordering",
   //                  "Ordering of fespace that will be used for gridfunction to be interpolated."
   //                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&jobid, "-jid", "--jid",
                  "job id used for visit  save files");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   func_order = func_order <= 0 ? std::min(order, 2) : func_order;

   // Initialize and refine the starting mesh.
   Mesh *mesh = NULL;
   if (dim == 2)
   {
      int nx = 6*(rs_levels+1);
      mesh = new Mesh(Mesh::MakeCartesian2D(nx, nx, etype == 0 ?
                                            Element::QUADRILATERAL:
                                            Element::TRIANGLE));
   }
   else if (dim == 3)
   {
      int nx = 6*(rs_levels+1);
      mesh = new Mesh(Mesh::MakeCartesian3D(nx, nx, nx, etype == 0 ?
                                            Element::HEXAHEDRON :
                                            Element::TETRAHEDRON));
   }
   else
   {
      MFEM_ABORT("invalid -dim");
   }
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
   if (hrefinement) { mesh->EnsureNCMesh(true); }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh.UniformRefinement(); }


   // Curve the mesh based on the chosen polynomial degree.
   H1_FECollection fecm(mesh_poly_deg, dim);
   ParFiniteElementSpace pfespace(&pmesh, &fecm, dim);
   pmesh.SetNodalFESpace(&pfespace);
   if (myid == 0)
   {
      cout << "Mesh curvature of the curved mesh: " << fecm.Name() << endl;
   }
   ParGridFunction nodes(&pfespace);
   pmesh.SetNodalGridFunction(&nodes);

   // Kershaw transformation
   if (smooth > 0)
   {
      // 1 leads to a linear transformation, 2 cubic, and 3 5th order.
      common::KershawTransformation kershawT(pmesh.Dimension(), 0.3, 0.3, smooth);
      pmesh.Transform(kershawT);
   }

   IntegrationRules *irules = &IntRulesLo;
   double tauval = GetMinDet(&pmesh, &pfespace, irules, 2*mesh_poly_deg+3);
   if (myid == 0)
   {
      cout << "Minimum det of the mesh: " << tauval << endl;
   }
   MFEM_VERIFY(tauval > 0,"Negative det found");

   // Random h-refinements to mesh
   if (hrefinement) { pmesh.RandomRefinement(0.5); }
   pfespace.Update();
   nodes.Update();

   int nelemglob = pmesh.GetGlobalNE();

   MFEM_VERIFY(ncomp > 0, "Invalid number of components.");
   int vec_dim = ncomp;
   FiniteElementCollection *fec = NULL;
   if (fieldtype == 0)
   {
      fec = new H1_FECollection(order, dim);
      if (myid == 0) { cout << "H1-GridFunction\n"; }
   }
   else if (fieldtype == 1)
   {
      fec = new L2_FECollection(order, dim);
      if (myid == 0) { cout << "L2-GridFunction\n"; }
   }
   else
   {
      if (myid == 0) { MFEM_ABORT("Invalid FECollection type."); }
   }
   ParFiniteElementSpace sc_fes(&pmesh, fec, ncomp, gf_ordering);
   ParGridFunction field_vals(&sc_fes);

   // Project the GridFunction using VectorFunctionCoefficient.
   //   VectorFunctionCoefficient F(vec_dim, F_exact);
   FunctionCoefficient F(field_func);
   VectorFunctionCoefficient gradF(dim, grad_field_func);
   field_vals.ProjectCoefficient(F);

   // Do L2 projection
   if (proj)
   {
      Array<int> ess_tdof_list;
      if (pmesh.bdr_attributes.Size())
      {
         Array<int> ess_bdr(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         sc_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      ConstantCoefficient one(1.0);
      ParBilinearForm a(&sc_fes);
      ParLinearForm b(&sc_fes);
      b.AddDomainIntegrator(new DomainLFIntegrator(F));
      a.AddDomainIntegrator(new MassIntegrator(one));
      if (fieldtype == 0)
      {
         b.AddDomainIntegrator(new DomainLFGradIntegrator(gradF));
         a.AddDomainIntegrator(new DiffusionIntegrator(one));
      }
      b.Assemble();
      a.Assemble();
      //       a.Finalize();
      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, field_vals, b, A, X, B);
      Solver *prec = new HypreBoomerAMG;
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(*prec);
      cg.SetOperator(*A);
      cg.Mult(B, X);
      delete prec;
      a.RecoverFEMSolution(X, b, field_vals);
   }

   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(irules->Get(i, 2*mesh_poly_deg+5));
   }

   double l2err = field_vals.ComputeL2Error(F, irs);
   double h1err = field_vals.ComputeH1Error(&F, &gradF);
   if (myid == 0)
   {
      std::cout << "Global L2 and H1 error: " << l2err << " " << h1err << std::endl;
   }

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
   //   const int pts_cnt_1D = 25;
   //   int pts_cnt = pow(pts_cnt_1D, dim);
   int pts_cnt = npt;
   Vector vxyz(pts_cnt * dim);
   vxyz.Randomize(myid+1);
   // The implied ordering is byNodes i.e. x0,x1,x2..y0,y1,y2..z0,z1,z2
   // For the first 10% points, set the x coordinates such that they are on
   // element boundary
   int nxlayers = 6*(rs_levels+1);
   int npt_on_faces = int(0.1*pts_cnt);
   for (int i = 0; i < npt_on_faces; i++)
   {
      vxyz(i) = (1.0/nxlayers)*(i % nxlayers);
   }
   //   std::cout << npt_on_faces << " k10nptonface\n";
   //      vxyz.Print();

   if ( (myid != 0) && (search_on_rank_0) )
   {
      pts_cnt = 0;
      vxyz.Destroy();
   }

   StopWatch FindPointsSW;
   FindPointsSW.Clear();

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt*vec_dim);
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   FindPointsSW.Start();
   finder.Setup(pmesh);
   finder.Interpolate(vxyz, field_vals, interp_vals, point_ordering);
   FindPointsSW.Stop();
   Array<unsigned int> code_out    = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();
   Vector dist_p_out = finder.GetDist();
   Vector rst = finder.GetReferencePosition();
   //   std::cout << " print rst\n";
   //   rst.Print();
   //   for (int i = 0; i < pts_cnt; i++)
   //   {
   //       std::cout << i << " " << code_out[i] << " k101code\n";
   //   }

   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double err = 0.0, max_err = 0.0, max_dist = 0.0;

   Vector pos(dim);
   for (int j = 0; j < vec_dim; j++)
   {
      for (int i = 0; i < pts_cnt; i++)
      {
         if (j == 0)
         {
            (task_id_out[i] == (unsigned)myid) ? found_loc++ : found_away++;
         }

         if (code_out[i] < 2)
         {
            for (int d = 0; d < dim; d++)
            {
               pos(d) = point_ordering == Ordering::byNODES ?
                        vxyz(d*pts_cnt + i) :
                        vxyz(i*dim + d);
            }
            Vector exact_val(vec_dim);
            F_exact(pos, exact_val);
            err = gf_ordering == Ordering::byNODES ?
                  fabs(exact_val(j) - interp_vals[i + j*pts_cnt]) :
                  fabs(exact_val(j) - interp_vals[i*vec_dim + j]);
            max_err  = std::max(max_err, err);
            max_dist = std::max(max_dist, dist_p_out(i));
            if (code_out[i] == 1 && j == 0) { face_pts++; }
         }
         else { if (j == 0) { not_found++; } }
      }
   }

   MPI_Allreduce(MPI_IN_PLACE, &found_loc, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &found_away, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &face_pts, 1, MPI_INT, MPI_SUM, pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &not_found, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &max_dist, 1, MPI_DOUBLE, MPI_MAX,
                 pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, pfespace.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, &npt_on_faces, 1, MPI_INT, MPI_SUM,
                 pfespace.GetComm());

   if (myid == 0)
   {
      cout << "Minimum det of the mesh: " << tauval << endl;
      cout << setprecision(16)
           << "Total number of elements: " << nelemglob
           << "\nTotal number of procs: " << num_procs
           << "\nSearched total points: " << pts_cnt*num_procs
           << "\nFound locally on ranks:  " << found_loc
           << "\nFound on other tasks: " << found_away
           << "\nPoints not found:     " << not_found
           << "\nPoints on faces:      " << face_pts
           << "\nPoints put on faces:  " << npt_on_faces
           << "\nMax interp error:     " << max_err
           << "\nMax dist (of found):  " << max_dist
           << "\nTotal Time:  " << FindPointsSW.RealTime()
           << endl;
   }

   // Print timing info
   if (myid == 0)
   {
      cout << "FindPointsGSLIB-Timing-info " <<
           "jobid,mindet,ne,np,npts,foundloc,foundaway,notfound,foundface,totface,maxerr,maxdist,"<<
           "totaltime,setup_split,setup_nodalmapping,setup_setup,findpts_findpts,findpts_mapelemrst,"
           <<
           "interpolate_h1,interpolate_general,interpolate_l2_pass2 " <<
           jobid << "," <<
           tauval << "," <<
           num_procs << "," <<
           pts_cnt*num_procs << "," <<
           found_loc << "," <<
           found_away << "," <<
           not_found << "," <<
           face_pts << "," <<
           npt_on_faces << "," <<
           max_err << "," <<
           max_dist << "," <<
           FindPointsSW.RealTime()  << "," <<
           finder.setup_split_time << "," <<
           finder.setup_nodalmapping_time << "," <<
           finder.setup_findpts_setup_time << "," <<
           finder.findpts_findpts_time << "," <<
           finder.findpts_mapelemrst_time << "," <<
           finder.interpolate_h1_time << "," <<
           finder.interpolate_general_time << "," <<
           finder.interpolate_l2_pass2_time << "," <<
           std::endl;
   }

   {
      ostringstream mesh_name;
      mesh_name << "kershawint.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      if (hrefinement)
      {
         pmesh.PrintAsOne(mesh_ofs);
      }
      else
      {
         pmesh.PrintAsSerial(mesh_ofs);
      }
   }

   // Free the internal gslib data.
   finder.FreeData();

   delete fec;

   return 0;
}
