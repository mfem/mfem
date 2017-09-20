#include "mfem.hpp"
#include "../common/bravais.hpp"
#include <fstream>
#include <iostream>
#include <cerrno>      // errno

#include "meta_material_solver.hpp"

#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

using namespace std;
using namespace mfem;
using namespace mfem::miniapps;
using namespace mfem::bravais;

// Volume Fraction Coefficient
static int prob_ = -1;
double vol_frac_coef(const Vector &);

int CreateDirectory(const string &dir_name, MPI_Comm & comm, int myid);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // 2. Parse command-line options.
   int bl_type = 1;
   string lattice_label = "";
   int order = 1;
   // int sr = 0, pr = 2;
   int logging = 0;

   bool visualization = 1;
   bool visit = true;
   bool densityCalc = true;
   bool stiffnessCalc = false;
   bool bandGapCalc = false;

   double a = -1.0, b = -1.0, c = -1.0;
   double alpha = -1.0, beta = -1.0, gamma = -1.0;
   double alpha_deg = -1.0, beta_deg = -1.0, gamma_deg = -1.0;
   double lcf = 0.3;
   double density_tol = 0.05;
   double stiffness_tol = 0.05;
   double band_gap_tol = 0.05;
   int    band_gap_max_ref = 2;
   // double lambda = 2.07748e+9;
   // double mu = 0.729927e+9;
   // Gallium Arsenide at T=300K
   // double lambda = 5.34e+11;
   // double mu = 3.285e+11;
   // double rho0 = 0.0;

   // Acrylonitrile Butadiene Styrene (ABS)
   double    rho = 1110.0; // Mass density 1.075 kg/m^3
   double      E = 2.0e9;  // Young's Modulus 2GPa
   double     nu = 0.4064; // Poisson's Ratio
   double epsRel = 10.0;   // Relative Dielectric Permittivity
   double  muRel = 1.0;    // Relative Magnetic Permeability

   OptionsParser args(argc, argv);
   args.AddOption(&bl_type, "-bl", "--bravais-lattice",
                  "Bravais Lattice Type: \n"
                  "  1 - Primitive Cubic (a),\n"
                  "  2 - Face-Centered Cubic (a),\n"
                  "  3 - Body-Centered Cubic (a),\n"
                  "  4 - Tetragonal (a, c),\n"
                  "  5 - Body-Centered Tetragonal (a, c),\n"
                  "  6 - Orthorhombic (a < b < c),\n"
                  "  7 - Face-Centered Orthorhombic (a < b < c),\n"
                  "  8 - Body-Centered Orthorhombic (a < b < c),\n"
                  "  9 - C-Centered Orthorhombic (a < b, c),\n"
                  " 10 - Hexagonal Prism (a, c),\n"
                  " 11 - Rhombohedral (a, 0 < alpha < pi),\n"
                  " 12 - Monoclinic (a, b <= c, 0 < alpha < pi/2),\n"
                  " 13 - C-Centered Monoclinic (a, b <= c, 0 < alpha < pi/2),\n"
                  " 14 - Triclinic (0 < alpha, beta, gamma < pi)\n"
                 );
   args.AddOption(&a, "-a", "--lattice-a",
                  "Lattice spacing a");
   args.AddOption(&b, "-b", "--lattice-b",
                  "Lattice spacing b");
   args.AddOption(&c, "-c", "--lattice-c",
                  "Lattice spacing c");
   args.AddOption(&alpha, "-alpha", "--lattice-alpha",
                  "Lattice angle alpha");
   args.AddOption(&beta, "-beta", "--lattice-beta",
                  "Lattice angle beta");
   args.AddOption(&gamma, "-gamma", "--lattice-gamma",
                  "Lattice angle gamma");
   args.AddOption(&alpha_deg, "-alpha-deg", "--lattice-alpha-degrees",
                  "Lattice angle alpha in degrees");
   args.AddOption(&beta_deg, "-beta-deg", "--lattice-beta-degrees",
                  "Lattice angle beta in degrees");
   args.AddOption(&gamma_deg, "-gamma-deg", "--lattice-gamma-degrees",
                  "Lattice angle gamma in degrees");
   args.AddOption(&lcf, "-lcf", "--lattice-coef-frac",
                  "Fraction of inscribed circle radius for rods");
   args.AddOption(&density_tol, "-rtol", "--density-tolerance",
                  "Stopping tolerance specified as a relative difference "
                  "in computed density");
   args.AddOption(&stiffness_tol, "-ctol", "--stiffness-tolerance",
                  "Stopping tolerance specified as a relative difference "
                  "in the 2-norm of the computed stiffness tensor");
   args.AddOption(&band_gap_tol, "-bgtol", "--band-gap-tolerance",
                  "Stopping tolerance specified as a relative difference "
                  "in the computed band gap");
   args.AddOption(&band_gap_max_ref, "-bgmr", "--band-gap-max-ref",
                  "Maximum number of uniform mesh refinements to perform "
                  "when computing the band gap");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   // args.AddOption(&sr, "-sr", "--serial-refinement",
   //               "Number of serial refinement levels.");
   // args.AddOption(&pr, "-pr", "--parallel-refinement",
   //               "Number of parallel refinement levels.");
   // args.AddOption(&prob_, "-p", "--problem-type",
   //                "Problem Geometry.");
   // args.AddOption(&lambda, "-l", "--lambda",
   //               "Lambda");
   // args.AddOption(&mu, "-m", "--mu",
   //               "Mu");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit",
                  "--no-visualization",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&densityCalc, "-rho", "--density", "-no-rho", "--no-density",
                  "Enable or disable density calculation.");
   args.AddOption(&stiffnessCalc, "-C", "--stiffness",
                  "-no-C", "--no-stiffness",
                  "Enable or disable stiffness tensor calculation.");
   // args.AddOption(&dispersionPlot, "-disp", "--dispersion",
   //             "-no-disp", "--no-dispersion",
   //             "Enable or disable dispersion plot calculation.");
   args.AddOption(&bandGapCalc, "-bg", "--band-gap",
                  "-no-bg", "--no-band-gap",
                  "Enable or disable band gap calculation.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   if ( alpha_deg > 0.0 ) { alpha = alpha_deg * M_PI / 180.0; }
   if (  beta_deg > 0.0 ) { beta =  beta_deg * M_PI / 180.0; }
   if ( gamma_deg > 0.0 ) { gamma = gamma_deg * M_PI / 180.0; }

   BRAVAIS_LATTICE_TYPE lattice_type = (BRAVAIS_LATTICE_TYPE)(bl_type + 5);
   BravaisLattice * bravais = BravaisLatticeFactory(lattice_type,
                                                    a, b, c,
                                                    alpha, beta, gamma,
                                                    logging);
   BravaisLattice3D * bravais3d = dynamic_cast<BravaisLattice3D*>(bravais);

   lattice_label = bravais->GetLatticeTypeLabel();

   bravais3d->GetAxialLengths(a, b, c);
   bravais3d->GetInteraxialAngles(alpha, beta, gamma);

   ostringstream oss_prefix;
   oss_prefix << "Meta-Material-" << lattice_label
              << "-" << (int)round(100.0*a)
              << "-" << (int)round(100.0*b)
              << "-" << (int)round(100.0*c)
              << "-" << (int)round(180.0*alpha / M_PI)
              << "-" << (int)round(180.0*beta  / M_PI)
              << "-" << (int)round(180.0*gamma / M_PI);
   // << "-r" << sr + pr;

   CreateDirectory(oss_prefix.str(),comm,myid);

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   /*
   Mesh * mesh = bravais->GetPeriodicWignerSeitzMesh();

   int euler = mesh->EulerNumber();
   if ( myid == 0 ) { cout << "Initial Euler Number:  " << euler << endl; }
   mesh->CheckElementOrientation(false);
   mesh->CheckBdrElementOrientation(false);

   // 4. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement.
   {
      int ref_levels = sr;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
         int euler = mesh->EulerNumber();
         if ( myid == 0 )
         {
            cout << l+1 << ", Refined Euler Number:  " << euler << endl;
         }
         mesh->CheckElementOrientation(false);
         mesh->CheckBdrElementOrientation(false);
      }
   }
   */
   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   /*
   delete mesh;
   {
      int par_ref_levels = pr;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }
   */
   // L2_ParFESpace * L2FESpace    = new L2_ParFESpace(pmesh, 0,
   //                                               pmesh->Dimension());
   /*
   // int nElems = L2FESpace->GetVSize();
   int nElems = mesh->GetNE();
   cout << myid << ": nElems = " << nElems << endl;
   delete mesh;
   */
   // LatticeCoefficient latCoef(*bravais, lcf);

   // ParGridFunction * vf0 = new ParGridFunction(L2FESpace);
   // ParGridFunction * vf1 = new ParGridFunction(L2FESpace);

   // FunctionCoefficient vfFunc(vol_frac_coef);
   // vf0->ProjectCoefficient(latCoef);
   /*
   vf1->ProjectCoefficient(vfFunc);

   double vf13 = (*vf0)[13];
   double dvf = 0.01 * (0.5 - vf13);
   (*vf1)[13] += dvf;
   */
   VisData vd("localhost", 19916, 1440, 900, 238, 238, 10, 45);
   /*
   if (visualization)
   {
      socketstream vf_sock;
      VisualizeField(vf_sock, *vf0, "Volume Fraction 0", vd);
      vd.IncrementWindow();
      // socketstream vf1_sock;
      // VisualizeField(vf1_sock, *vf1, "Volume Fraction 1", vd);
      // vd.IncrementWindow();
   }
   */
   /*
   meta_material::Density density(*pmesh, bravais->GetUnitCellVolume(),
                                  0.0, rho);

   density.SetVolumeFraction(*vf0);
   */
   if (densityCalc)
   {
      LatticeCoefficient rhoCoef(*bravais, lcf, 0.0, rho);

      // The density computation does not require a periodic mesh
      Mesh * mesh_rho = bravais->GetWignerSeitzMesh();
      mesh_rho->EnsureNCMesh();
      ParMesh *pmesh_rho = new ParMesh(MPI_COMM_WORLD, *mesh_rho);
      delete mesh_rho;

      meta_material::Density density(*pmesh_rho, rho,
                                     bravais->GetUnitCellVolume(),
                                     rhoCoef, density_tol);

      vector<double> effective_rho;
      density.GetHomogenizedProperties(effective_rho);
      if ( myid == 0 )
      {
         ostringstream oss;
         oss << oss_prefix.str() << "/density.dat";
         ofstream ofs(oss.str().c_str());

         cout << "Effective Density:  ";
         for (unsigned int i=0; i<effective_rho.size(); i++)
         {
            cout << effective_rho[i]; ofs << effective_rho[i];
            if ( i < effective_rho.size()-1 ) { cout << ", "; ofs << "\t"; }
         }
         cout << endl; ofs << endl;
         ofs.close();
      }

      if (visualization)
      {
         density.InitializeGLVis(vd);
         density.DisplayToGLVis();
      }
      if ( visit )
      {
         density.WriteVisItFields(oss_prefix.str(), "Density");
      }
      delete pmesh_rho;
   }

   if (stiffnessCalc)
   {
      double lambda = E * nu / ( (1.0 + nu) * (1.0 - 2.0 * nu) );
      double mu = 0.5 * E / (1.0 + nu);
      double mat_scale = 1.0e-6;

      LatticeCoefficient lambdaCoef(*bravais, lcf, lambda * mat_scale, lambda);
      LatticeCoefficient muCoef(*bravais, lcf, mu * mat_scale, mu);
      /*
      meta_material::StiffnessTensor elasticity(*pmesh,
                                                bravais->GetUnitCellVolume(),
                                                lambda * mat_scale, mu * mat_scale,
                                                lambda, mu);

      elasticity.SetVolumeFraction(*vf0);
      */
      Mesh * mesh_C = bravais->GetPeriodicWignerSeitzMesh();
      // mesh_C->UniformRefinement();
      // mesh_C->UniformRefinement();
      mesh_C->EnsureNCMesh();
      ParMesh *pmesh_C = new ParMesh(MPI_COMM_WORLD, *mesh_C);
      delete mesh_C;

      meta_material::StiffnessTensor elasticity(*pmesh_C,
                                                bravais->GetUnitCellVolume(),
                                                lambdaCoef, muCoef,
                                                stiffness_tol);

      vector<double> elas;
      elasticity.GetHomogenizedProperties(elas);
      if ( myid == 0 )
      {
         ostringstream oss;
         oss << oss_prefix.str() << "/stiffness_tensor.dat";
         ofstream ofs(oss.str().c_str());

         cout << "Effective Elasticity Tensor:  " << endl;
         int k = 0;
         for (unsigned int i=0; i<6; i++)
         {
            for (unsigned int j=0; j<i; j++)
            {
               cout << " -----------";
               ofs << elas[(11 - j) * j / 2 + i] << "\t";
            }
            for (unsigned int j=i; j<6; j++)
            {
               cout << " " << elas[k]; ofs << elas[k];
               if ( k < 20 ) { ofs << "\t"; }
               k++;
            }
            cout << endl; ofs << endl;
         }
         cout << endl;
         ofs.close();
      }
      if (visualization)
      {
         elasticity.InitializeGLVis(vd);
         elasticity.DisplayToGLVis();
      }
      if ( visit )
      {
         elasticity.WriteVisItFields(oss_prefix.str(), "StiffnessTensor");
      }
      delete pmesh_C;
   }

   if (bandGapCalc)
   {
      LatticeCoefficient epsCoef(*bravais, lcf, 1.0, epsRel);
      LatticeCoefficient  muCoef(*bravais, lcf, 1.0,  muRel);

      Mesh * mesh_bg = bravais->GetPeriodicWignerSeitzMesh();
      ParMesh *pmesh_bg = new ParMesh(MPI_COMM_WORLD, *mesh_bg);
      delete mesh_bg;

      meta_material::MaxwellBandGap maxwell_bg(*pmesh_bg, *bravais,
                                               epsCoef, muCoef,
                                               band_gap_max_ref, band_gap_tol);

      vector<double> bg;
      maxwell_bg.GetHomogenizedProperties(bg);

      if (visualization)
      {
         maxwell_bg.InitializeGLVis(vd);
         maxwell_bg.DisplayToGLVis();
      }
      if ( visit )
      {
         maxwell_bg.WriteVisItFields(oss_prefix.str(), "MaxwellBandGap");
      }
      delete pmesh_bg;
   }

   // delete vf0;
   // delete vf1;
   // delete L2FESpace;
   // delete pmesh;
   delete bravais;

   MPI_Finalize();

   if ( myid == 0 )
   {
      cout << "Exiting Main" << endl;
   }

   return 0;
}

int CreateDirectory(const string &dir_name, MPI_Comm & comm, int myid)
{
   int err;
#ifndef MFEM_USE_MPI
   err = mkdir(dir_name.c_str(), 0775);
   err = (err && (errno != EEXIST)) ? 1 : 0;
#else
   if (myid == 0)
   {
      err = mkdir(dir_name.c_str(), 0775);
      err = (err && (errno != EEXIST)) ? 1 : 0;
      MPI_Bcast(&err, 1, MPI_INT, 0, comm);
   }
   else
   {
      // Wait for rank 0 to create the directory
      MPI_Bcast(&err, 1, MPI_INT, 0, comm);
   }
#endif
   return err;
}

double
distToLine(double ox, double oy, double oz,
           double tx, double ty, double tz, const Vector & x)
{
   double xo_data[3];
   double xt_data[3];
   Vector xo(xo_data, 3);
   Vector xt(xt_data, 3);

   // xo = x - {ox,oy,oz}
   xo[0] = x[0] - ox;
   xo[1] = x[1] - oy;
   xo[2] = x[2] - oz;

   // xt = cross_product({tx,ty,tz}, xo)
   xt[0] = ty * xo[2] - tz * xo[1];
   xt[1] = tz * xo[0] - tx * xo[2];
   xt[2] = tx * xo[1] - ty * xo[0];

   return xt.Norml2();
}
double
vol_frac_coef(const Vector & x)
{
   switch ( prob_ )
   {
      case -1:
         // Uniform
         return 1.0;
         break;
      case 0:
         // Slab
         if ( fabs(x(0)) <= 0.25 ) { return 1.0; }
         break;
      case 1:
         // Cylinder
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= 0.5 ) { return 1.0; }
         break;
      case 2:
         // Sphere
         if ( x.Norml2() <= 0.5 ) { return 1.0; }
         break;
      case 3:
         // Sphere and 3 Rods
      {
         double r1 = 0.14, r2 = 0.36, r3 = 0.105;
         if ( x.Norml2() <= r1 ) { return 0.0; }
         if ( x.Norml2() <= r2 ) { return 1.0; }
         if ( sqrt(x(1)*x(1)+x(2)*x(2)) <= r3 ) { return 1.0; }
         if ( sqrt(x(2)*x(2)+x(0)*x(0)) <= r3 ) { return 1.0; }
         if ( sqrt(x(0)*x(0)+x(1)*x(1)) <= r3 ) { return 1.0; }
      }
      break;
      case 4:
         // Sphere and 4 Rods
      {
         double r1 = 0.14, r2 = 0.28, r3 = 0.1;
         if ( x.Norml2() <= r1 ) { return 0.0; }
         if ( x.Norml2() <= r2 ) { return 1.0; }

         Vector y = x;
         y[0] -= 0.5; y[1] -= 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] -= 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] += 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] += 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] -= 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] -= 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] += 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] += 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         double a = r3;
         double b = 1.0/sqrt(3.0);
         if ( distToLine(0.0, 0.0, 0.0, b, b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0,-b, b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0,-b,-b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b,-b, b, x) <= a ) { return 1.0; }
      }
      break;
      case 5:
         // Two spheres in a BCC configuration
         if ( x.Norml2() <= 0.3 )
         {
            return 1.0;
         }
         else
         {
            for (int i=0; i<8; i++)
            {
               int i1 = i%2;
               int i2 = (i/2)%2;
               int i4 = i/4;

               Vector u = x;
               u(0) -= i1?-0.5:0.5;
               u(1) -= i2?-0.5:0.5;
               u(2) -= i4?-0.5:0.5;

               if ( u.Norml2() <= 0.2 ) { return 1.0; }
            }
         }
         break;
      case 6:
         // Sphere and 6 Rods
      {
         double r1 = 0.12, r2 = 0.19, r3 = 0.08;
         if ( x.Norml2() <= r1 ) { return 0.0; }
         if ( x.Norml2() <= r2 ) { return 1.0; }

         Vector y = x;
         y[0] -= 0.5; y[1] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] -= 0.5; y[1] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[0] += 0.5; y[1] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         y = x; y[1] -= 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[1] -= 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[1] += 0.5; y[2] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[1] += 0.5; y[2] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         y = x; y[2] -= 0.5; y[0] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[2] -= 0.5; y[0] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[2] += 0.5; y[0] -= 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }
         y = x; y[2] += 0.5; y[0] += 0.5;
         if ( y.Norml2() <= r1 ) { return 0.0; }
         if ( y.Norml2() <= r2 ) { return 1.0; }

         double a = r3;
         double b = 1.0/sqrt(2.0);
         if ( distToLine(0.0, 0.0, 0.0, b, b, 0, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b,-b, 0, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b, 0, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, b, 0,-b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, 0, b, b, x) <= a ) { return 1.0; }
         if ( distToLine(0.0, 0.0, 0.0, 0, b,-b, x) <= a ) { return 1.0; }

      }
      break;
      case 7:
         if ( fabs(x(1)) + fabs(x(2)) < 0.25 ||
              fabs(x(0)) + fabs(x(2)) < 0.25 ||
              fabs(x(0)) + fabs(x(1) - 0.5) < 0.25 )
         {
            return 1.0;
         }
         break;
   }
   return 0.0;
}
