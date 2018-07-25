// ./s1wrapper -m RT2D.mesh -qo 8 -o 3
//
#include "mfem.hpp"
extern "C" {
# include "gslib/src/cpp/findpts_h.h"
}

#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>
#include <string>

using namespace mfem;
using namespace std;

#include "fpt_wrapper.hpp"

int main (int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int quad_type         = 1;
   int quad_order        = 8;
   bool visualization    = true;
   int verbosity_level   = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   cout << "Mesh curvature: ";
   if (mesh->GetNodes()) { cout << mesh->GetNodes()->OwnFEC()->Name(); }
   else { cout << "(NONE)"; }
   cout << endl;

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   // 3. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    number of components in the vector finite element space is specified by
   //    the last parameter of the FiniteElementSpace constructor.
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else { fec = new H1_FECollection(mesh_poly_deg, dim); }
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 4. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   GridFunction *x = mesh->GetNodes();

   // 8. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = *x;

   // 9. Setup the quadrature rule for the non-linear form integrator.
   const IntegrationRule *ir = NULL;
   const int geom_type = fespace->GetFE(0)->GetGeomType();
   int quad_eval = quad_order;
   int myid = 0;
//   cout << quad_order << " the quad_order" << endl;
   if (quad_order > 4) 
   {
    if (quad_order % 2 == 0) {quad_order = 2*quad_order - 4;}
    else {quad_order = 2*quad_order - 3;}
   }
   switch (quad_type)
   {
      case 1: ir = &IntRulesGLL.Get(geom_type, quad_order);
   }
   if (myid==0) {cout << "Quadrature points per cell: " << ir->GetNPoints() << endl;}

   // 10. write out all dofs
   const int NE = fespace->GetMesh()->GetNE(),
   dof = fespace->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
   
   GridFunction nodes(fespace);
   mesh->GetNodes(nodes);


   int NR = sqrt(nsp);
   if (dim==3) {NR = cbrt(nsp);}

   int sz1 = pow(NR,dim);
   Vector fx(dim*NE*sz1);
   Vector dumfield(NE*sz1);
   int np;

   np = 0;
   int tnp = NE*nsp;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < nsp; j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
        fx[np] = nodes.GetValue(i, ip, 1);
        fx[tnp+np] =nodes.GetValue(i, ip, 2);
        dumfield[np] = pow(fx[np],2)+pow(fx[tnp+np],2);
        if (dim==3) {fx[2*tnp+np] =nodes.GetValue(i, ip, 3);
                    dumfield[np] += pow(fx[2*tnp+np],2);}
        np = np+1;
      }
   }

   findpts_gslib *gsfl=NULL;
   gsfl = new findpts_gslib();
   gsfl->gslib_findpts_setup(fespace,mesh,quad_order);

// generate random points by r,s,t
   int llim = 100;
   int nlim = NE*llim;
   double xmn = 0,xmx=1,ymn=0,ymx=1,zmn=0,zmx=1; //Domain extent
   double mnv,mxv,dlv;
   Vector rrxa(nlim),rrya(nlim),rrza(nlim);
   Vector vxyz(nlim*dim);

   np = 0;
   IntegrationPoint ipt;
   for (int i = 0; i < NE; i++)
   {
      for (int j = 0; j < llim; j++)
      {
        Geometries.GetRandomPoint(fespace->GetFE(i)->GetGeomType(),ipt);
        rrxa[np] = ipt.x;
        rrya[np] = ipt.y;
        if (dim==3) {rrza[np] = ipt.z;}
        vxyz[np] = nodes.GetValue(i, ipt, 1);
        vxyz[np+nlim] = nodes.GetValue(i, ipt, 2);
        if (dim==3) {vxyz[np+2*nlim] = nodes.GetValue(i, ipt, 3);}
        np = np+1;
      }
   }

   int nxyz = nlim;

  if (myid==0) {cout << "Total Points to be found: " << nxyz << " \n";}

   Array<uint> pel(nxyz);
   Array<uint> pcode(nxyz);
   Array<uint> pproc(nxyz);
   Vector pr(nxyz*dim);
   Vector pd(nxyz);
   int start_s=clock();
//   gsfl->gslib_findpts(&pcode,&pproc,&pel,&pr,&pd,&vrxa,&vrya,&vrza,nxyz);
   gsfl->gslib_findpts(&pcode,&pproc,&pel,&pr,&pd,&vxyz,nxyz);
   int stop_s=clock();
   if (myid==0) {cout << "findpts order: " << NR << " \n";}
   if (myid==0) {cout << "findpts time (sec): " << (stop_s-start_s)/1000000. << endl;}

// FINDPTS_EVAL
   Vector fout(nxyz);
   start_s=clock();
   gsfl->gslib_findpts_eval(&fout,&pcode,&pproc,&pel,&pr,&dumfield,nxyz);
   stop_s=clock();
   if (myid==0) {cout << "findpts_eval time (sec): " << (stop_s-start_s)/1000000. << endl;}
   gsfl->gslib_findpts_free();

   int nbp = 0;
   int nnpt = 0;
   int nerrh = 0;
   double maxv = -100.;
   double maxvr = -100.;
   int it;
   for (it = 0; it < nxyz; it++)
   {
    if (pcode[it] < 2) {
    double val = pow(vxyz[it],2)+pow(vxyz[it+nlim],2);
    if (dim==3) val += pow(vxyz[it+2*nlim],2);
    double delv = abs(val-fout[it]);
    double rxe = abs(rrxa[it] - 0.5*pr[it*dim+0]-0.5);
    double rye = abs(rrya[it] - 0.5*pr[it*dim+1]-0.5);
    double rze = abs(rrza[it] - 0.5*pr[it*dim+2]-0.5);
    double delvr =  ( rxe < rye ) ? rye : rxe;
    if (dim==3) {delvr = ( ( delvr < rze ) ? rze : delvr );}
    if (delv > maxv) {maxv = delv;}
    if (delvr > maxvr) {maxvr = delvr;}
    if (pcode[it] == 1) {nbp += 1;}
    if (delvr > 1.e-10) {nerrh += 1;}
    if (delvr > 1.e-10) {
    cout << rrxa[it] << " " << pr[it*dim] << " " << rrya[it] << " " << pr[it*dim+1] <<  " " << delvr << " k10r\n";
      }
    if (delv > 1.e-10) {
    cout <<  val << " " << fout[it] << " k10s\n";
     }
    if (delvr > 1.e-10) {
//    cout << rrxa[it] << " " << rrya[it] << " " << vrxa[it] << " " << vrya[it] << " k10e\n";
      }
   }
   else
   {
    nnpt += 1;
   }
   }
  double glob_maxerr=maxv;
  int glob_nnpt=nnpt;
  int glob_nbp=nbp;
  int glob_nerrh=nerrh;
  cout << setprecision(16);
  if (myid==0) {cout << "maximum error: " << glob_maxerr << " \n";}
  if (myid==0) {cout << "points not found: " << glob_nnpt << " \n";}
  if (myid==0) {cout << "points on element border: " << glob_nbp << " \n";}
  if (myid==0) {cout << "points with error > 1.e-10: " << glob_nerrh << " \n";}

   delete fespace;
   delete fec;
   delete mesh;

   return 0;
}
