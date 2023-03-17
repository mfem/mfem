#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include "marking.hpp"

#ifdef MFEM_USE_ALGOIM
#include "shape_grad.hpp"
#endif

using namespace mfem;
using namespace std;


const double radius = 0.4;

double sphere_ls(const Vector &x)
{
   const int dim = x.Size();
   const double xc = x(0) - 0.5;
   const double yc = (dim > 1) ? x(1) - 0.5 : 0.0;
   const double zc = (dim > 2) ? x(2) - 0.5 : 0.0;
   const double r = sqrt(xc*xc + yc*yc + zc*zc);

   return (r >= radius) ? -1.0 : 1.0;
}

double exact_dist_sphere(const Vector &x)
{
   const int dim = x.Size();
   const double xc = x(0) - 0.5;
   const double yc = (dim > 1) ? x(1) - 0.5 : 0.0;
   const double zc = (dim > 2) ? x(2) - 0.5 : 0.0;
   const double r = sqrt(xc*xc + yc*yc + zc*zc);

   //return 3*(radius*radius*radius-r*r*r);
   return 3*(radius*radius*radius-r*r*r);
}

double Gyroid(const Vector &xx)
{
   const double period = 2.0 * M_PI;
   double x = xx[0]*period;
   double y = xx[1]*period;
   double z = (xx.Size()==3) ? xx[2]*period : 0.0;

   return std::sin(x)*std::cos(y) +
          std::sin(y)*std::cos(z) +
          std::sin(z)*std::cos(x) + x*x+y*y;
}


int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();
   Hypre::Init();

   // Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int solver_type = 0;
   int problem = 1;
   int rs_levels = 2;
   int order = 2;
   double t_param = 1.0;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&solver_type, "-s", "--solver",
                  "Solver type:\n\t"
                  "0: Heat\n\t"
                  "1: P-Laplacian\n\t"
                  "2: Rvachev scaling");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type:\n\t"
                  "0: Point source\n\t"
                  "1: Circle / sphere level set in 2D / 3D\n\t"
                  "2: 2D sine-looking level set\n\t"
                  "3: Gyroid level set in 2D or 3D\n\t"
                  "4: Combo of a doughnut and swiss cheese shapes in 3D.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&t_param, "-t", "--t-param",
                  "Diffusion time step (scaled internally scaled by dx*dx).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myrank == 0) { args.PrintOptions(cout); }


   // Refine the mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }

   // MPI distribution.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   Coefficient *ls_coeff = nullptr;
   ls_coeff = new FunctionCoefficient(exact_dist_sphere);

   H1_FECollection hfec(order, dim);
   L2_FECollection lfec(order, dim);
   ParFiniteElementSpace pfes_s(&pmesh, &hfec), pfes_v(&pmesh, &lfec, dim);
   ParGridFunction distance_s(&pfes_s), distance_v(&pfes_v);

   distance_s.ProjectCoefficient(*ls_coeff);


   ElementMarker marker(pmesh,false, true,2);
   marker.SetLevelSetFunction(distance_s);

   Array<int> el_marks;
   marker.MarkElements(el_marks);
   for(int i=0;i<pmesh.GetNE();i++)
   {
       pmesh.SetAttribute(i,el_marks[i]);
   }

   {
       GradientGridFunctionCoefficient gco(&distance_s);
       distance_v.ProjectCoefficient(gco,(int)ElementMarker::SBElementType::CUT);
   }


   ParNonlinearForm* nf=new ParNonlinearForm(&pfes_s);
   ParNonlinearForm* sf=new ParNonlinearForm(&pfes_s);

   ConstantCoefficient one(1.0);
   VolShapeIntegrator* itg=new VolShapeIntegrator(one,el_marks);
   nf->AddDomainIntegrator(itg);


   ParGridFunction gyro_s(&pfes_s), gyro_v(&pfes_v);
   {
       FunctionCoefficient gfc(Gyroid);
       gyro_s.ProjectCoefficient(gfc);
   }
   GridFunctionCoefficient gfc(&gyro_s);
   GradientGridFunctionCoefficient gco(&gyro_s);

   SurfShapeIntegrator* its=new SurfShapeIntegrator(gfc, gco, el_marks);
   sf->AddDomainIntegrator(its);

   Vector lsf(pfes_s.GetTrueVSize());
   Vector lgr(pfes_s.GetTrueVSize());
   Vector sgr(pfes_s.GetTrueVSize());
   distance_s.GetTrueDofs(lsf);
   double vol=nf->GetEnergy(lsf);
   double surf=sf->GetEnergy(lsf);

   nf->Mult(lsf,lgr);
   ParGridFunction grad_s(&pfes_s);
   grad_s.SetFromTrueDofs(lgr);

   sf->Mult(lsf,sgr);

   if(myrank==0){
       std::cout<<"Vol="<<vol<<" surf="<<surf<<std::endl;
   }


   //check gradients by FD
   {
       mfem::Vector prtv;
       mfem::Vector tmpv;

       prtv.SetSize(lsf.Size());
       tmpv.SetSize(lsf.Size());

       prtv.Randomize();

       double nd=mfem::InnerProduct(pmesh.GetComm(),prtv,prtv);
       double td=mfem::InnerProduct(pmesh.GetComm(),prtv,lgr);
       double sd=mfem::InnerProduct(pmesh.GetComm(),prtv,sgr);

       td=td/nd;
       sd=sd/nd;
       double lsc=1.0;
       double lqoi;
       double sqoi;

       for(int l=0; l<8;l++){
           lsc/=10.0;
           prtv/=10.0;
           add(prtv,lsf,tmpv);
           distance_s.SetFromTrueDofs(tmpv);
           marker.SetLevelSetFunction(distance_s);
           marker.MarkElements(el_marks);

           lqoi=nf->GetEnergy(tmpv);
           sqoi=sf->GetEnergy(tmpv);

           double ld=(lqoi-vol)/lsc;
           double ls=(sqoi-surf)/lsc;
           if(myrank==0){
               std::cout<<" vol="<<vol<<" lvo="<< lqoi<<" dx="<<lsc<<" FD app="<< ld/nd<<" gr="<< td <<" err="<< std::fabs(ld/nd-td) <<std::endl;
               std::cout<<" sur="<<surf<<" lso="<< sqoi<<" dx="<<lsc<<" FD app="<< ls/nd<<" gr="<< sd<<" err="<< std::fabs(ls/nd-sd) <<std::endl;
           }
       }



   }

   delete sf;
   delete nf;


   // ParaView output.
   ParaViewDataCollection dacol("ParaViewDistance", &pmesh);
   dacol.SetLevelsOfDetail(order);
   dacol.RegisterField("distance", &distance_s);
   dacol.RegisterField("grad", &grad_s);
   dacol.SetTime(1.0);
   dacol.SetCycle(1);
   dacol.Save();



   delete ls_coeff;

   return 0;
}
