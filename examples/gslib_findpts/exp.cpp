// r-adapt shape+size:
// ./cfp -m square.mesh -qo 4
//mpirun -np 2 p1wrapper -m RT2D.mesh -qo 14 -o 3
// TO DO: Add checks inside wrapper for array sizes etc...
//
#include "mfem.hpp"
#include <fstream>
#include <ctime>

extern "C" {
# include "3rd_party/gslib/src/cpp/findpts_h.h"
}


using namespace mfem;
using namespace std;
  

#include "fpt_wrapper.hpp"


IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

int main (int argc, char *argv[])
{
   // 0. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   int rp_levels         = 0;
   int quad_type         = 1;
   int quad_order        = 8;
   bool visualization    = true;

   // 1. Parse command-line options.
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
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0) {args.PrintOptions(cout);}

   // 3. Initialize and refine the starting mesh.
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
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // 4. Define a finite element space on the mesh. Here we use vector finite
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
   ParFiniteElementSpace *pfespace = new ParFiniteElementSpace(pmesh, fec, dim);

   // 5. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   pmesh->SetNodalFESpace(pfespace);

   // 6. Set up an empty right-hand side vector b, which is equivalent to b=0.
   Vector b(0);

   // 7. Get the mesh nodes (vertices and other degrees of freedom in the finite
   //    element space) as a finite element grid function in fespace. Note that
   //    changing x automatically changes the shapes of the mesh elements.
   ParGridFunction x(pfespace);
   pmesh->SetNodalGridFunction(&x);

   x.SetTrueVector();
   x.SetFromTrueVector();

   // 10. Save the starting (prior to the optimization) mesh to a file. This
   //     output can be viewed later using GLVis: "glvis -m perturbed -np
   //     num_mpi_tasks".
   {
      ostringstream mesh_name;
      mesh_name << "perturbed." << setfill('0') << setw(6) << myid;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
   }

   // 11. Store the starting (prior to the optimization) positions.
   ParGridFunction x0(pfespace);
   x0 = x;

   // 12. Setup the quadrature rule for the non-linear form integrator.
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

   const int NE = pfespace->GetMesh()->GetNE(),
   dof = pfespace->GetFE(0)->GetDof(), nsp = ir->GetNPoints();
   
   ParGridFunction nodes(pfespace);
   pmesh->GetNodes(nodes);

   int NR = sqrt(nsp);
   if (dim==3) {NR = cbrt(nsp);}

   int sz1 = pow(NR,dim);
   double fx[dim*NE*sz1];
   double dumfield[NE*sz1];
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
   gsfl = new findpts_gslib(pfespace,pmesh,quad_order);
   gsfl->gslib_findpts_setup(0.05,1.e-12,256);

// random vector in domain 
   int nlim = 50000;
   double xmn = 0,xmx=0.5,ymn=-1,ymx=1,zmn=0,zmx=0.5; //Domain extent
   double mnv,mxv,dlv;
   Vector vrxv(nlim),vryv(nlim),vrzv(nlim);
   vrxv.Randomize();
   vryv.Randomize();
   vrzv.Randomize();
   mnv = vrxv.Min();
   vrxv -= mnv;
   mxv = vrxv.Max();
   vrxv *= 1./mxv;
   dlv = xmx-xmn;
   vrxv *= dlv;
   vrxv -= -xmn;

   mnv = vryv.Min();
   vryv -= mnv;
   mxv = vryv.Max();
   vryv *= 1./mxv;
   dlv = ymx-ymn;
   vryv *= dlv;
   vryv -= -ymn;

   mnv = vrzv.Min();
   vrzv -= mnv;
   mxv = vrzv.Max();
   vrzv *= 1./mxv;
   dlv = zmx-zmn;
   vrzv *= dlv;
   vrzv -= -zmn;

   double *vrx;
   double *vry;
   double *vrz;
   int nxyz;
   vrx = vrxv.GetData();
   vry = vryv.GetData();
   vrz = vrzv.GetData();
   nxyz = vrxv.Size();

   if (myid==0) {cout << "Num procs: " << num_procs << " \n";}
   if (myid==0) {cout << "Points per proc: " << nxyz << " \n";}
   if (myid==0) {cout << "Total Points to be found: " << nxyz*num_procs << " \n";}

   uint pcode[nxyz];
   uint pproc[nxyz];
   uint pel[nxyz];
   double pr[nxyz*dim];
   double pd[nxyz];
   double fout[nxyz];
   int start_s=clock();
   gsfl->gslib_findpts(pcode,pproc,pel,pr,pd,vrx,vry,vrz,nxyz);
   MPI_Barrier(MPI_COMM_WORLD);
   int stop_s=clock();
   if (myid==0) {cout << "findpts order: " << NR << " \n";}
   if (myid==0) {cout << "findpts time (sec): " << (stop_s-start_s)/1000000. << endl;}
 
// FINDPTS_EVAL
   MPI_Barrier(MPI_COMM_WORLD);
   start_s=clock();
   gsfl->gslib_findpts_eval(fout,pcode,pproc,pel,pr,dumfield,nxyz);
   stop_s=clock();
   if (myid==0) {cout << "findpts_eval time (sec): " << (stop_s-start_s)/1000000. << endl;}
   gsfl->gslib_findpts_free();

   int nbp = 0;
   int nnpt = 0;
   int nerrh = 0;
   double maxv = -100.;
   int it;
   for (it = 0; it < nxyz; it++)
   {
    if (pcode[it] < 2) {
    double val = pow(vrx[it],2)+pow(vry[it],2);
    if (dim==3) val += pow(vrz[it],2);
    double delv = abs(val-fout[it]);
    if (delv > maxv) {maxv = delv;}
    if (pcode[it] == 1) {nbp += 1;}
    if (delv > 1.e-10) {nerrh += 1;}
   }
   else
   {
    nnpt += 1;
   }
   }
   MPI_Barrier(MPI_COMM_WORLD);
   double glob_maxerr;
   MPI_Allreduce(&maxv, &glob_maxerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   int glob_nnpt;
   MPI_Allreduce(&nnpt, &glob_nnpt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nbp;
   MPI_Allreduce(&nbp, &glob_nbp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   int glob_nerrh;
   MPI_Allreduce(&nerrh, &glob_nerrh, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   cout << setprecision(16);
   if (myid==0) {cout << "maximum error: " << glob_maxerr << " \n";}
   if (myid==0) {cout << "points not found: " << glob_nnpt << " \n";}
   if (myid==0) {cout << "points on element border: " << glob_nbp << " \n";}
   if (myid==0) {cout << "points with error > 1.e-10: " << glob_nerrh << " \n";}
 
   delete pfespace;
   delete fec;
   delete pmesh;
   MPI_Finalize();

   return 0;
}
