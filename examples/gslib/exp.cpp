// mpirun -np 2 ./exp -m RT2D.mesh -qo 8 -o 3

#include "mfem.hpp"
#include "fem/gslib.hpp" // TODO move to mfem.hpp (double declaration bug ??)

#include <fstream>
#include <ctime>

using namespace mfem;
using namespace std;

IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

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
   int quad_type         = 1;
   int quad_order        = 8;

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
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
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

   // 9. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = pfespace->GetFE(0)->GetGeomType();
   if (quad_order > 4)
   {
      if (quad_order % 2 == 0) {quad_order = 2*quad_order - 4;}
      else {quad_order = 2*quad_order - 3;}
   }
   switch (quad_type)
   {
      case 1: ir = &IntRulesLo.Get(geom_type, quad_order);
   }
   if (myid==0) {cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;}

   const int NE = pfespace->GetMesh()->GetNE(), nsp = ir->GetNPoints();
   
   ParGridFunction nodes(pfespace);
   pmesh->GetNodes(nodes);

   int NR = sqrt(nsp);
   if (dim==3) {NR = cbrt(nsp);}

   int sz1 = pow(NR,dim);
   Vector fx(dim*NE*sz1);
   Vector dumfield(NE*sz1);

   int pt_id = 0;
   int tnp = NE*nsp;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fx[pt_id] = nodes.GetValue(i, ip, 1);
         fx[tnp+pt_id] = nodes.GetValue(i, ip, 2);
         dumfield[pt_id] = pow(fx[pt_id],2)+pow(fx[tnp+pt_id],2);
         if (dim==3)
         {
            fx[2*tnp+pt_id] =nodes.GetValue(i, ip, 3);
            dumfield[pt_id] += pow(fx[2*tnp+pt_id],2);
         }
         pt_id++;
      }
   }

   findpts_gslib *gsfl = new findpts_gslib(MPI_COMM_WORLD);
   const double rel_bbox_el = 0.05;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;
   gsfl->gslib_findpts_setup(*pmesh, rel_bbox_el, newton_tol, npts_at_once);

   // Generate random points in reference coordinates.
   const int pts_per_el = 10;
   const int pts_cnt    = NE * pts_per_el;
   Vector rrxa(pts_cnt), rrya(pts_cnt), rrza(pts_cnt);
   Vector vxyz(pts_cnt * dim);

   pt_id = 0;
   IntegrationPoint ipt;
   Vector pos;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < pts_per_el; j++)
      {
         Geometries.GetRandomPoint(pfespace->GetFE(i)->GetGeomType(), ipt);

         rrxa[pt_id] = ipt.x;
         rrya[pt_id] = ipt.y;
         if (dim == 3) { rrza[pt_id] = ipt.z; }

         nodes.GetVectorValue(i, ipt, pos);
         for (int d = 0; d < dim; d++) { vxyz(pts_cnt*d + pt_id) = pos(d); }
         pt_id++;
      }
   }

   cout << "Task id: " << myid << " \n"
        << "-- Points to find: " << pts_cnt << " \n"
        << "-- Number of elem: " << NE << endl;

   Array<uint> el_id_out(pts_cnt), code_out(pts_cnt), task_id_out(pts_cnt);
   Vector pos_r_out(pts_cnt * dim), dist_p_out(pts_cnt);
   MPI_Barrier(MPI_COMM_WORLD);

   // Finds points stored in vxyz.
   gsfl->gslib_findpts(vxyz, code_out, task_id_out,
                       el_id_out, pos_r_out, dist_p_out);

   // FINDPTS_EVAL
   Vector fout(pts_cnt);
   MPI_Barrier(MPI_COMM_WORLD);
   // Corresponds to the same points as above (vxyz) -- uses that output.
   // Returns function values in fout.
   // Note that the points where dumfield is defined are the ones given to _setup.
   gsfl->gslib_findpts_eval(&fout, &code_out, &task_id_out, &el_id_out, &pos_r_out, &dumfield, pts_cnt);
   gsfl->gslib_findpts_free();

   int nbp = 0, nnpt = 0, nerrh = 0;
   double maxv = -100.0 ,maxvr = -100.0;
   for (int it = 0; it < pts_cnt; it++)
   {
      if (code_out[it] < 2)
      {
         // TODO evaluate the FE function at vxyz.
         double val = pow(vxyz[it],2)+pow(vxyz[it+pts_cnt],2);
         if (dim==3) { val += pow(vxyz[it+2*pts_cnt],2); }
         double delv = abs(val-fout[it]);
         double rxe = abs(rrxa[it] - 0.5*pos_r_out[it*dim+0]-0.5);
         double rye = abs(rrya[it] - 0.5*pos_r_out[it*dim+1]-0.5);
         double rze = abs(rrza[it] - 0.5*pos_r_out[it*dim+2]-0.5);
         double delvr =  ( rxe < rye ) ? rye : rxe;
         if (dim==3) { delvr = ( ( delvr < rze ) ? rze : delvr ); }
         if (delv > maxv) {maxv = delv;}
         if (delvr > maxvr) {maxvr = delvr;}
         if (code_out[it] == 1) {nbp += 1;}
         if (delvr > 1.e-10) {nerrh += 1;}
      }
      else { nnpt++; }
   }

   double glob_maxerr;
   MPI_Allreduce(&maxv, &glob_maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   double glob_maxrerr;
   MPI_Allreduce(&maxvr, &glob_maxrerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   int glob_nnpt;
   MPI_Allreduce(&nnpt, &glob_nnpt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nbp;
   MPI_Allreduce(&nbp, &glob_nbp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nerrh;
   MPI_Allreduce(&nerrh, &glob_nerrh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      cout << setprecision(16);
      cout << "maximum error: " << glob_maxerr << " \n";
      cout << "maximum rst error: " << glob_maxrerr << " \n";
      cout << "points not found: " << glob_nnpt << " \n";
      cout << "points on element border: " << glob_nbp << " \n";
      cout << "points with error > 1.e-10: " << glob_nerrh << " \n";
   }

   delete pfespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();

   return 0;
}
