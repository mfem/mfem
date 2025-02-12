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
        psi_x = 1.28863812e+00,
        f_x = -32.86000000;
//note minimal f = 1.44525792e-01*(-5.64560650e+00-1.28863812)-32.86
//               = -33.8621771956
//
// Paraview B_R = Bvec_X*coordsX/sqrt(coordsX^2+coordsY^2) + Bvec_Y*coordsY/sqrt(coordsX^2+coordsY^2)
// The maximum of B_R is 1e-4, which is good

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

// this gives a vector coefficient of Psi/R e_φ 
class PsiOverRCoefficient : public VectorCoefficient
{
private:
   GridFunction *gf;
   Mesh         *mesh;
   FindPointsGSLIB finder;
 
public:
   PsiOverRCoefficient(int dim, GridFunction *gf_, Mesh *mesh_)
      : VectorCoefficient(dim), gf(gf_), mesh(mesh_)
   { finder.Setup(*mesh); finder.SetL2AvgType(FindPointsGSLIB::NONE); };
 
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip){
      double psi, R, Z, cosphi, sinphi;
      double x[3];
      Vector interp_vals(1), vxy(2), transip(x, 3);
      T.Transform(ip, transip);
      V.SetSize(vdim);

      R = sqrt(transip(0)*transip(0)+transip(1)*transip(1));
      Z = transip(2);
      vxy(0) = R;
      vxy(1) = Z;

      finder.Interpolate(vxy, *gf, interp_vals, 0/*point_ordering*/);
      Array<unsigned int> code_out = finder.GetCode();
      if (code_out[0] < 2){
         psi = interp_vals[0];
         //cout<<R<<" "<<Z<<" interp="<<psi<<endl;
      }
      else {MFEM_ABORT("Cannot find the point");}

      cosphi = transip(0)/R;
      sinphi = transip(1)/R;
      V(0) = -psi/R*sinphi;
      V(1) =  psi/R*cosphi;
      V(2) = 0.0;
   };

   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir){
      M.SetSize(vdim, ir.GetNPoints());

      int    dof=ir.GetNPoints();
      double xsave[4][dof];     //save R, z, x, y
      Vector interp_vals(dof), vxy(dof*2);

      // collect points for search
      for (int i = 0; i < dof; i++){
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

         vxy(i)       = R;
         vxy(dof + i) = Z;
      }

      // search an array of vxy
      finder.Interpolate(vxy, *gf, interp_vals, 0/*point_ordering*/);
      Array<unsigned int> code_out = finder.GetCode();
      Vector Mi;
      for (int i = 0; i < dof; i++){
         double cosphi, sinphi, psi;
         if (code_out[i] >= 2){
             cout<<dof<<endl;
             cout<<i<<" "<<vxy(i)<<" "<<vxy(dof + i)<<endl;
             MFEM_ABORT("Cannot find the point");
         }
         else
             psi = interp_vals[i];
         cosphi = xsave[2][i]/xsave[0][i];
         sinphi = xsave[3][i]/xsave[0][i];
         M.GetColumnReference(i, Mi);
         Mi(0) = -psi/xsave[0][i]*sinphi;
         Mi(1) =  psi/xsave[0][i]*cosphi;
         Mi(2) = 0.0;
      }
   };
 
   virtual ~PsiOverRCoefficient() {finder.FreeData();};
};

// this function compute the toroidal field from the given psi and f
class GridFunction2DCoefficient : public VectorCoefficient
{
private:
   GridFunction *gf;
   Mesh         *mesh;
   FindPointsGSLIB finder;
   double fFun(const double psi){return f_x + alpha*(psi-psi_x);};
 
public:
   GridFunction2DCoefficient(int dim, GridFunction *gf_, Mesh *mesh_)
      : VectorCoefficient(dim), gf(gf_), mesh(mesh_){
      finder.Setup(*mesh);
      finder.SetL2AvgType(FindPointsGSLIB::NONE); 
   };
 
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip){
      MFEM_ABORT("this is for debug only, we should use Eval with DenseMatrix");
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

   // this works but needs to turn on VectorFiniteElement::Project_ND
   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationRule &ir){
      M.SetSize(vdim, ir.GetNPoints());

      int    pts_cnt=0, dof=ir.GetNPoints();
      double xsave[4][dof];     //save R, z, x, y
      int    isave[dof];        //index for searching
      Vector fout(dof);
      fout = 1e10;

      // collect points for search
      for (int i = 0; i < dof; i++){
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
            fout(i) = f_x;
            // finding points within the domain
            isave[i] = 0;
         }
         else{
            pts_cnt++;
            // finding points within the domain
            isave[i] = 1;
         }
      }

      // search an array of vxy
      if (pts_cnt>0){
         Vector interp_vals(pts_cnt), vxy(pts_cnt * 2);
         int j=0;
         for (int i = 0; i < dof; i++){
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
         for (int i = 0; i < dof; i++){
             if (isave[i]==1)
             {
                 if (fout(i)<1e10) MFEM_ABORT("Error as fout is already set!");
                 if (code_out[j] >= 2) 
                 {
                     cout<<pts_cnt<<endl;
                     cout<<j<<" "<<vxy(j)<<" "<<vxy(pts_cnt + j)<<endl;
                     MFEM_ABORT("Cannot find the point");
                 }
                 fout(i) =  std::min(f_x, fFun(interp_vals[j]));
                 j++;
             }
         }
      }

      // Set B_φ = f(psi)/R
      Vector Mi;
      for (int i = 0; i < dof; i++){
         double cosphi, sinphi;
         cosphi = xsave[2][i]/xsave[0][i];
         sinphi = xsave[3][i]/xsave[0][i];
         if (fout(i)>9.9e10) MFEM_ABORT("Error in computing fout!");
         M.GetColumnReference(i, Mi);
         Mi(0) = -fout(i)/xsave[0][i]*sinphi;
         Mi(1) =  fout(i)/xsave[0][i]*cosphi;
         Mi(2) = 0.0;
      }
   };
 
   virtual ~GridFunction2DCoefficient() {finder.FreeData();};
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
   int size = Bfespace->GetVSize();     //expect VSize and TrueVSize are the same
   cout << "Number of elements: "<<mesh.GetNE()<<endl;
   cout << "Number of finite element unknowns: " << size << endl;

   GridFunction Bperp(Bfespace), Btor(Bfespace);

   // Project B toroidal
   // VectorFunctionCoefficient VecCoeff(sdim3D, B_exact);
   // Bvec.ProjectCoefficient(VecCoeff);
   GridFunction2DCoefficient PsiCoeff(sdim3D, &psi, &mesh);
   Btor.ProjectCoefficient(PsiCoeff);

   // Solve (f, Bperp) = (curl f, psi/R e_φ) + <f, n x psi/R e_φ>
   // where f is a test function in H(curl)
   ConstantCoefficient one(1.0);
   BilinearForm mass(Bfespace);
   mass.AddDomainIntegrator(new VectorFEMassIntegrator(one));
   mass.Assemble();
   mass.Finalize();

   CGSolver M_solver;
   DSmoother M_prec(mass.SpMat()); 
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-8);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(1);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(mass.SpMat());

   cout << "Assemble RHS" <<endl;
   LinearForm rhs(Bfespace);
   PsiOverRCoefficient PsiOverR(sdim3D, &psi, &mesh);
   rhs.AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(PsiOverR));
   rhs.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(PsiOverR));
   rhs.Assemble();

   Vector x(size);
   x = 0.0;
   M_solver.Mult(rhs, x);

   Bperp.SetFromTrueDofs(x);

   if (visualization){
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh3D << Bperp << flush;
   }

   if (true){
      ofstream mesh_ofs("build-field.mesh");
      mesh_ofs.precision(8);
      mesh3D.Print(mesh_ofs);
      ofstream sol_ofs("Bperp.gf");
      sol_ofs.precision(8);
      Bperp.Save(sol_ofs);
      ofstream sol_ofs1("Btor.gf");
      sol_ofs1.precision(8);
      Btor.Save(sol_ofs1);

      ParaViewDataCollection paraview_dc("build-field", &mesh3D);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("Bperp",&Bperp);
      paraview_dc.RegisterField("Btor",&Btor);
      paraview_dc.Save();
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