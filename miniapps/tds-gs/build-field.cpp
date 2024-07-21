//
// Compile with: make findpsi
//
// Sample runs:
//  ./findpsi 
//  ./findpsi -m ./meshes/RegPoloidalQuadMeshNonAligned_true.mesh -g ./interpolated.gf

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double  alpha = 1.44525792e-01,
        r_x = 4.91157885e+00,
        z_x = -3.61688204e+00,
        psi_x = 1.28863812e+00;

// transform matrix:
// [v_X] = [cos -sin][v_R  ]
// [v_Y] = [sin  cos][v_phi]
//
// Safty factor:
// q = q0+q2 r^2 with r^2 = (R-R0)^2/a^2 + z^2/Z0^2
//
double q0 = 1.2, q2 = 2.8, a_i=2.7832, R0=6.2, Z0=5.1944, B0=1.0;
void B_exact(const Vector &x, Vector &B)
{
   const double R = sqrt(x(0)*x(0)+x(1)*x(1)), Z = x(2);
   const double q = q0 + q2*((R-R0)*(R-R0)+Z*Z)/a_i/a_i;
   double B_R, B_Z, B_phi, cosphi, sinphi;

   B_R = -Z/q/R*B0;
   B_Z = (R-R0)/q/R*B0;
   B_phi = R0/R*B0;

   cosphi = x(0)/R;
   sinphi = x(1)/R;

   B(0) = B_R*cosphi-B_phi*sinphi;
   B(1) = B_R*sinphi+B_phi*cosphi;
   B(2) = B_Z;
};

class GridFunction2DCoefficient : public VectorCoefficient
{
private:
   GridFunction *gf;
   Mesh         *mesh;

   FindPointsGSLIB finder;
 
public:
   GridFunction2DCoefficient(int dim, GridFunction *gf_, Mesh *mesh_)
      : VectorCoefficient(dim), gf(gf_), mesh(mesh_){
      finder.Setup(*mesh);
      finder.SetL2AvgType(FindPointsGSLIB::NONE); 
   };
 
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip){
      double f, R, Z, cosphi, sinphi;
      double x[3];
      Vector interp_vals(1), vxy(2), transip(x, 3);
      T.Transform(ip, transip);
      V.SetSize(vdim);

      R = sqrt(transip(0)*transip(0)+transip(1)*transip(1));
      Z = transip(2);

      vxy(0) = R;
      vxy(1) = Z;

      if (Z>4.12587 || Z<z_x || R<4.046 || R>8.37937){
        //f = f_x;
        f = psi_x;
      }
      else{
        finder.Interpolate(vxy, *gf, interp_vals, 0/*point_ordering*/);
        Array<unsigned int> code_out    = finder.GetCode();
        if (code_out[0] < 2){
           //f = std::min(f_x, f_x + alpha*(interp_vals[0]-psi_x));
           //f = std::min(psi_x, interp_vals[0]);
           f = interp_vals[0];
        }
        else {MFEM_ABORT("Cannot find the point");}
        cout<<R<<" "<<Z<<" interp="<<f<<endl;
      }

      cosphi = transip(0)/R;
      sinphi = transip(1)/R;

      V(0) = -f*sinphi;
      V(1) =  f*cosphi;
      V(2) = 0.0;
   };

   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir){
      cout<<"debug"<<endl;
      M.SetSize(vdim, ir.GetNPoints());

      int    pts_cnt=0;
      double xsave[4][ir.GetNPoints()];     //save R, z, x, y
      int    isave[ir.GetNPoints()];        //index for searching
      Vector fout(ir.GetNPoints());
      fout = 1e10;

      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         T.SetIntPoint(&ip);

         double R, Z, x[3];
         Vector transip(x, 3);
         T.Transform(ip, transip);

         R = sqrt(transip(0)*transip(0)+transip(1)*transip(1));
         Z = transip(2);
         xsave[0][i]=R;
         xsave[1][i]=Z;
         xsave[2][i]=transip(0);
         xsave[3][i]=transip(1);

         if (Z>4.12587 || Z<z_x || R<4.046 || R>8.37937){
            //f = f_x;
            fout(i) = psi_x;
            isave[i] = 0;
         }
         else{
            pts_cnt++;
            isave[i] = 1;
         }
      }

      if (pts_cnt>0){
        Vector interp_vals(pts_cnt), vxy(pts_cnt * 2);
        int j=0;
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            if (isave[i]==1)
            {
                vxy(j)           = xsave[0][i];
                vxy(pts_cnt + j) = xsave[1][i];
                j++;
            }
        }
        finder.Interpolate(vxy, *gf, interp_vals, 0);
        Array<unsigned int> code_out    = finder.GetCode();
        j=0;
        for (int i = 0; i < ir.GetNPoints(); i++)
        {
            if (isave[i]==1)
            {
                if (fout(i)<1e10) MFEM_ABORT("Error in fout!");
                if (code_out[j] < 2) MFEM_ABORT("Cannot find the point");
                    
                fout(i) =  std::min(psi_x, interp_vals[j]);
                j++;
            }
        }
      }

      Vector Mi;
      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         double cosphi, sinphi;
         cosphi = xsave[2][i]/xsave[0][i];
         sinphi = xsave[3][i]/xsave[0][i];

         if (fout(i)>9.9e10) MFEM_ABORT("Error in computing fout!");
         M.GetColumnReference(i, Mi);
         Mi(0) = -fout(i)*sinphi;
         Mi(1) =  fout(i)*cosphi;
         Mi(2) = 0.0;
      }
   };
 
   virtual ~GridFunction2DCoefficient() 
   {finder.FreeData();};
};

int main (int argc, char *argv[])
{
   // Set the method's default parameters.
   const char *mesh_file = "./solution/mesh_taylor1_trimmer.mesh";
   const char *mesh_file3D = "./meshes/RegPoloidalQuadMeshNonAligned_true_extrude.mesh";
   const char *sltn_file = "./solution/psi_taylor1_trimmer.gf";
   int mesh_poly_deg     = 1;
   int point_ordering    = 0;
   int order             = 2;
   bool visualization    = false;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&sltn_file, "-g", "--gf",
                  "GrifFunction file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Vector finite element order (polynomial degree).");
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

   // Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1, false);
   const int dim = mesh.Dimension();
   cout << "Mesh curvature of the original mesh: ";
   if (mesh.GetNodes()) { cout << mesh.GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // Curve the mesh based on the chosen polynomial degree (this is a necessary step).
   H1_FECollection fecm(mesh_poly_deg, dim);
   FiniteElementSpace fespace(&mesh, &fecm, dim);
   mesh.SetNodalFESpace(&fespace);
   GridFunction Nodes(&fespace);
   mesh.SetNodalGridFunction(&Nodes);
   cout << "Mesh curvature of the curved mesh: " << fecm.Name() << endl;

   // Read the solution psi
   ifstream mat_stream(sltn_file);
   GridFunction psi(&mesh, mat_stream);

   // Initialize 3D mesh and space
   Mesh mesh3D(mesh_file3D, 1, 1);
   int dim3D = mesh3D.Dimension();
   int sdim3D = mesh3D.SpaceDimension();
   if (dim3D!=3 || sdim3D!=3){
      cout << "wrong dimensions in mesh!"<<endl;
      return 1;
   }
   FiniteElementCollection *Bfec=new ND_FECollection(order, dim3D);
   FiniteElementSpace *Bfespace=new FiniteElementSpace(&mesh3D, Bfec);
   cout << "Number of elements: "<<mesh.GetNE()<<endl;

   GridFunction Bvec(Bfespace);
   //VectorFunctionCoefficient VecCoeff(sdim3D, B_exact);
   //Bvec.ProjectCoefficient(VecCoeff);
   GridFunction2DCoefficient PsiCoeff(sdim3D, &psi, &mesh);
   Bvec.ProjectCoefficient(PsiCoeff);

   HYPRE_Int size = Bfespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << size << endl;

   if (visualization){
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh3D << Bvec << flush;
   }

   if (false){
      ofstream mesh_ofs("findpsi.mesh");
      mesh_ofs.precision(8);
      mesh3D.Print(mesh_ofs);
      ofstream sol_ofs("bvec.gf");
      sol_ofs.precision(8);
      Bvec.Save(sol_ofs);
   }

   /*
   int pts_cnt = 2;
   Vector vxyz(pts_cnt * dim);
   double xin, yin;
   for (int i = 0; i < pts_cnt; i++)
   {
      if (i==0){
          xin = 6.26634357;
          yin = 6.49329884e-01; 
      }
      else{
          xin = 4.95501596; 
          yin = -3.56733283;
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

   // check interpolation using an exact solution
   int face_pts = 0, not_found = 0, found = 0;
   Vector pos(dim);

   // interpolate psi GridFunction
   finder.Interpolate(vxyz, psi, interp_vals, point_ordering);
   Array<unsigned int> code_out    = finder.GetCode();
   Vector dist_p_out = finder.GetDist();
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
         printf("i = %i interp_val = %10.8e, r = %10.8e, z = %10.8e\n",i,interp_vals[i],pos(0),pos(1));
         if (code_out[i] == 1) { face_pts++; }
      }
      else { not_found++; }
   }

   cout << setprecision(16)
        << "Searched points:     "   << pts_cnt
        << "\nFound points:        " << found
        << "\nPoints not found:    " << not_found
        << "\nPoints on faces:     " << face_pts << endl;

   // Free the internal gslib data.
   finder.FreeData();
   */

   delete Bfespace;
   delete Bfec;

   return 0;
}
