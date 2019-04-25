// mpirun -np 2 ./exp -m RT2D.mesh -o 3

#include "mfem.hpp"
#include "fem/gslib.hpp" // TODO move to mfem.hpp (double declaration bug ??)

#include <fstream>
#include <ctime>

using namespace mfem;
using namespace std;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

// Initial condition
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += x(d) * x(d); }
   return res;
}

int main (int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
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
      cout << "Mesh curvature: ";
      if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
      else { cout << "(NONE)"; }
      cout << endl;
   }

   // Mesh bounding box.
   Vector pos_min, pos_max;
   MFEM_VERIFY(mesh_poly_deg > 0, "The order of the mesh must be a positive.");
   mesh->GetBoundingBox(pos_min, pos_max, mesh_poly_deg);
   if (myid == 0)
   {
      std::cout << "x in [" << pos_min(0) << ", " << pos_max(0) << "]\n";
      std::cout << "y in [" << pos_min(1) << ", " << pos_max(1) << "]\n";
      if (dim == 3)
      {
         std::cout << "z in [" << pos_min(2) << ", " << pos_max(2) << "]\n";
      }
   }

   // Distribute the mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   // Define a scalar function on the mesh.
   ParFiniteElementSpace sc_fes(pmesh, fec, 1);
   GridFunction field_vals(&sc_fes);
   FunctionCoefficient fc(field_func);
   field_vals.ProjectCoefficient(fc);

   /*
   socketstream sout;
   char vishost[] = "localhost";
   int  visport   = 19916;
   sout.open(vishost, visport);
   sout.precision(1e-6);
   sout << "solution\n" << *pmesh << field_vals;
   sout << "pause\n";
   sout << flush;
   */

   findpts_gslib *gsfl = new findpts_gslib(MPI_COMM_WORLD);
   const double rel_bbox_el = 0.05;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;
   gsfl->gslib_findpts_setup(*pmesh, rel_bbox_el, newton_tol, npts_at_once);

   // Generate equidistant points in physical coordinates over the whole mesh.
   // Note that some points might be outside, if the mesh is not a box.
   // Note that all tasks search the same points.
   const int pts_cnt_1D = 5;
   const int pts_cnt = std::pow(pts_cnt_1D, dim);
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

   Array<uint> el_id_out(pts_cnt), code_out(pts_cnt), task_id_out(pts_cnt);
   Vector pos_r_out(pts_cnt * dim), dist_p_out(pts_cnt);
   MPI_Barrier(MPI_COMM_WORLD);

   // Finds points stored in vxyz.
   gsfl->gslib_findpts(vxyz, code_out, task_id_out,
                       el_id_out, pos_r_out, dist_p_out);

   // FINDPTS_EVAL
   Vector fout(pts_cnt);
   MPI_Barrier(MPI_COMM_WORLD);
   // Returns function values in fout.
   gsfl->gslib_findpts_eval(code_out, task_id_out, el_id_out,
                            pos_r_out, field_vals, fout);

   gsfl->gslib_findpts_free();

   int face_pts = 0, not_found = 0, found_loc = 0, found_away = 0;
   double maxv = -100.0;
   Vector pos(dim);
   for (int i = 0; i < pts_cnt; i++)
   {
      (task_id_out[i] == myid) ? found_loc++ : found_away++;

      if (code_out[i] < 2)
      {
         for (int d = 0; d < dim; d++) { pos(d) = vxyz(d * pts_cnt + i); }
         double exact_val = field_func(pos);
         double delv = abs(exact_val-fout(i));
         if (delv > maxv) { maxv = delv; }
         if (code_out[i] == 1) { face_pts++; }
      }
      else { not_found++; }
   }

   std:cout << "---\n--- Task " << myid
            << "\nFound on local mesh:  " << found_loc
            << "\nFound on other tasks: " << found_away << std::endl;

   double glob_maxerr;
   MPI_Allreduce(&maxv, &glob_maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   double max_dist = dist_p_out.Max(), glob_md;
   MPI_Allreduce(&max_dist, &glob_md, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   int glob_nf;
   MPI_Allreduce(&not_found, &glob_nf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nbp;
   MPI_Allreduce(&face_pts, &glob_nbp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      cout << setprecision(16) << "---\n--- Total statistics:"
           << "\nMax interp error: " << glob_maxerr
           << "\nMax distance:     " << glob_md
           << "\nPoints not found: " << glob_nf
           << "\nPoints on faces:  " << glob_nbp << std::endl;
   }

   delete pfespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();

   return 0;
}
