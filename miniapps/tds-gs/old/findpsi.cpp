//
// Compile with: make findpsi
//
// Sample runs:
//  ./findpsi 
//  ./findpsi -m ./meshes/RegPoloidalQuadMeshNonAligned_true.mesh -g ./interpolated.gf

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double func_order;

// Scalar function to project
double field_func(const Vector &x)
{
   const int dim = x.Size();
   double res = 0.0;
   for (int d = 0; d < dim; d++) { res += std::pow(x(d), func_order); }
   return res;
}

void F_exact(const Vector &p, Vector &F)
{
   F(0) = field_func(p);
   for (int i = 1; i < F.Size(); i++) { F(i) = (i+1)*F(0); }
}

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "./meshes/mesh_refine.mesh";
   const char *sltn_file = "./gf/final_model2_pc5_cyc1_it5.gf";
   int order             = 3;
   int mesh_poly_deg     = 1;
   int point_ordering    = 0;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sltn_file, "-g", "--gf",
                  "GrifFunction file to use.");
   args.AddOption(&point_ordering, "-po", "--point-ordering",
                  "Ordering of points to be found."
                  "0 (default): byNodes, 1: byVDIM");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   func_order = std::min(order, 2);

   // Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1, false);
   const int dim = mesh.Dimension();
   cout << "Mesh curvature of the original mesh: ";
   if (mesh.GetNodes()) { cout << mesh.GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // Curve the mesh based on the chosen polynomial degree (this is necessary step).
   H1_FECollection fecm(mesh_poly_deg, dim);
   FiniteElementSpace fespace(&mesh, &fecm, dim);
   mesh.SetNodalFESpace(&fespace);
   GridFunction Nodes(&fespace);
   mesh.SetNodalGridFunction(&Nodes);
   cout << "Mesh curvature of the curved mesh: " << fecm.Name() << endl;

   FiniteElementCollection *fec = NULL;
   fec = new H1_FECollection(order, dim);
   
   FiniteElementSpace sc_fes(&mesh, fec);
   GridFunction field_vals(&sc_fes);

   VectorFunctionCoefficient F(1, F_exact);
   field_vals.ProjectCoefficient(F);

   ifstream mat_stream(sltn_file);
   GridFunction func(&mesh, mat_stream);

   int pts_cnt = 2;
   Vector vxyz(pts_cnt * dim);
   double xin, yin;
   for (int i = 0; i < pts_cnt; i++)
   {
      if (i==0){
          xin = 6.266344; 
          yin = 6.493299e-1;
      }
      else{
          xin = 4.955016; 
          yin = -3.567333;
      }
      if (point_ordering == Ordering::byNODES)
      {
         vxyz(i)           = xin;
         vxyz(pts_cnt + i) = yin;
      }
      else
      {
         vxyz(i*dim + 0) = xin;
         vxyz(i*dim + 1) = yin;
      }
   }

   // Find and Interpolate FE function values on the desired points.
   Vector interp_vals(pts_cnt);
   FindPointsGSLIB finder;
   finder.Setup(mesh);
   finder.SetL2AvgType(FindPointsGSLIB::NONE);
   finder.Interpolate(vxyz, field_vals, interp_vals, point_ordering);
   Array<unsigned int> code_out    = finder.GetCode();
   Vector dist_p_out = finder.GetDist();

   // check interpolation using an exact solution
   int face_pts = 0, not_found = 0, found = 0;
   double error = 0.0, max_err = 0.0, max_dist = 0.0;
   Vector pos(dim);
   for (int i = 0; i < pts_cnt; i++)
   {
      if (code_out[i] < 2)
      {
         found++;
         for (int d = 0; d < dim; d++)
         {
            pos(d) = point_ordering == Ordering::byNODES ?
                     vxyz(d*pts_cnt + i) :
                     vxyz(i*dim + d);
         }
         Vector exact_val(1);
         F_exact(pos, exact_val);
         error = fabs(exact_val(0) - interp_vals[i]);
         max_err  = std::max(max_err, error);
         max_dist = std::max(max_dist, dist_p_out(i));
         if (code_out[i] == 1) { face_pts++; }
      }
      else { not_found++; }
   }

   // interpolate psi GridFunction
   finder.Interpolate(vxyz, func, interp_vals, point_ordering);
   for (int i = 0; i < pts_cnt; i++)
   {
      if (code_out[i] < 2)
      {
         found++;
         for (int d = 0; d < dim; d++)
         {
            pos(d) = point_ordering == Ordering::byNODES ?
                     vxyz(d*pts_cnt + i) :
                     vxyz(i*dim + d);
         }
         printf("i = %i interp_val = %10.6e, r = %10.6e, z = %10.6e\n",i,interp_vals[i],pos(0),pos(1));
         if (code_out[i] == 1) { face_pts++; }
      }
      else { not_found++; }
   }

   cout << setprecision(16)
        << "Searched points:     "   << pts_cnt
        << "\nFound points:        " << found
        << "\nMax interp error:    " << max_err
        << "\nMax dist (of found): " << max_dist
        << "\nPoints not found:    " << not_found
        << "\nPoints on faces:     " << face_pts << endl;

   // Free the internal gslib data.
   finder.FreeData();

   delete fec;

   return 0;
}
