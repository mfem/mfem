//                                MFEM Example 3 -- modified for NURBS FE
//
// Compile with: make waveguide2Dmodeadjoint
//
// Sample runs:  mpirun -np 1 waveguide2Dmodeadjoint
//


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <cmath>
#include "IGAopt.hpp"

using namespace std;
using namespace mfem;

std::vector<real_t> data_Ex_real;
std::vector<real_t> data_Ex_imag;
std::vector<real_t> data_Ey_real;
std::vector<real_t> data_Ey_imag;
std::vector<real_t> data_Ez_real;
std::vector<real_t> data_Ez_imag;

void get_source_E(real_t x, real_t y, real_t z, 
                  real_t &Exr, real_t &Exi, real_t &Eyr, real_t &Eyi, real_t &Ezr, real_t &Ezi);

void source_r(const Vector &x, Vector & f);
void source_i(const Vector &x, Vector & f);

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   std::ifstream file_Ex_real("./mode_2000_x_x400_y800_4000/E_x_real.txt");
   std::ifstream file_Ex_imag("./mode_2000_x_x400_y800_4000/E_x_imag.txt");
   std::ifstream file_Ey_real("./mode_2000_x_x400_y800_4000/E_y_real.txt");
   std::ifstream file_Ey_imag("./mode_2000_x_x400_y800_4000/E_y_imag.txt");
   std::ifstream file_Ez_real("./mode_2000_x_x400_y800_4000/E_z_real.txt");
   std::ifstream file_Ez_imag("./mode_2000_x_x400_y800_4000/E_z_imag.txt");

   real_t temp_Ex_real;
   real_t temp_Ex_imag;
   real_t temp_Ey_real;
   real_t temp_Ey_imag;
   real_t temp_Ez_real;
   real_t temp_Ez_imag;

   // 检查文件是否成功打开
   if (!file_Ex_real.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
   if (!file_Ex_imag.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
   if (!file_Ey_real.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
   if (!file_Ey_imag.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
   if (!file_Ez_real.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
   if (!file_Ez_imag.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return;
    }
 
    // 读取文件中的数字并存入vector
   while (file_Ex_real >> temp_Ex_real) {
        data_Ex_real.push_back(temp_Ex_real);
    }
   while (file_Ex_imag >> temp_Ex_imag) {
        data_Ex_imag.push_back(temp_Ex_imag);
    }
   while (file_Ey_real >> temp_Ey_real) {
        data_Ey_real.push_back(temp_Ey_real);
    }
   while (file_Ey_imag >> temp_Ey_imag) {
        data_Ey_imag.push_back(temp_Ey_imag);
    }
   while (file_Ez_real >> temp_Ez_real) {
        data_Ez_real.push_back(temp_Ez_real);
    }
   while (file_Ez_imag >> temp_Ez_imag) {
        data_Ez_imag.push_back(temp_Ez_imag);
    }

   file_Ex_real.close();
   file_Ex_imag.close();
   file_Ey_real.close();
   file_Ey_imag.close();
   file_Ez_real.close();
   file_Ez_imag.close();

   // Parse command-line options.
   const char *mesh_file = "./meshes/cubes-nurbs.mesh";
   int order = 1;
   const char *device_config = "cpu";
   //real_t freq = 5.0;

   Device device(device_config);
   device.Print();

   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Refine the mesh to increase the resolution.
   for (int l = 0; l < 4; l++)
   {
      mesh->UniformRefinement();
   }
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

   FiniteElementCollection *fec = nullptr;
   NURBSExtension *NURBSext = nullptr;
   fec = new NURBS_HCurlFECollection(order,dim);
   NURBSext  = new NURBSExtension(pmesh->NURBSext, order);                                                                              
   mfem::out<<"ID "<<myid<<" "<<getpid()<<" Create NURBS fec and ext"<<std::endl;

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, NURBSext, fec);
   mfem::out << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;

   Array<int> ess_tdof_list;

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mfem::out << "Number of knowns in essential BCs: "
        << ess_tdof_list.Size() << endl;

   ComplexOperator::Convention conv = ComplexOperator::HERMITIAN;

   VectorFunctionCoefficient port_source_r(dim, source_r);
   VectorFunctionCoefficient port_source_i(dim, source_i);
   ConstantCoefficient one(1.0);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(port_source_r), new VectorFEDomainLFIntegrator(port_source_i));
   b.Vector::operator=(0.0);
   b.Assemble();
   int Size_b = b.Vector::Size();
   mfem::out<<"Size_b ********************* "<<Size_b<<" "<<fespace->GetNDofs()<<std::endl;
   // for(int i = 0;i < Size_b; i ++)
   // {
   //    mfem::out<<b.Vector::GetData()[i]<<" ";
   // }
   // mfem::out<<std::endl;

   ParComplexGridFunction x(fespace);
   x = 0.0;

   // mfem::out<<"ID "<<myid<<" "<<getpid()<<std::endl;
   // int k = 1;
   // while(k)
   // {
   //  sleep(5);
   // }

   // // Integrators inside the computational domain (excluding the PML region)
   ParSesquilinearForm a(fespace, conv);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   NURBSHCurl_VectorMassIntegrator *di_NURBSVectorMassIntegrator = new NURBSHCurl_VectorMassIntegrator(one);
   a.AddDomainIntegrator(di_NURBSVectorMassIntegrator,di_NURBSVectorMassIntegrator);

   a.Assemble(0);
   OperatorPtr A;
   Vector B, X;

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   ParBilinearForm prec(fespace);
   prec.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   prec.AddDomainIntegrator(di_NURBSVectorMassIntegrator);
   prec.Assemble();

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fespace->GetTrueVSize();
   offsets[2] = fespace->GetTrueVSize();
   offsets.PartialSum();
   std::unique_ptr<Operator> pc_r;
   std::unique_ptr<Operator> pc_i;

   real_t s = (conv == ComplexOperator::HERMITIAN) ? -1_r : 1_r;
   
   pc_r.reset(new OperatorJacobiSmoother(prec, ess_tdof_list));
   pc_i.reset(new ScaledOperator(pc_r.get(), s));
   
   BlockDiagonalPreconditioner BlockDP(offsets);
   BlockDP.SetDiagonalBlock(0, pc_r.get());
   BlockDP.SetDiagonalBlock(1, pc_i.get());
   GMRESSolver gmres(MPI_COMM_WORLD);

   gmres.SetPrintLevel(1);
   gmres.SetKDim(50);
   gmres.SetMaxIter(100000);
   gmres.SetRelTol(1e-5);
   gmres.SetAbsTol(0.001);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(BlockDP);
   mfem::out<<"ID "<<myid<<" "<<getpid()<<std::endl;

   gmres.Mult(B, X);
   a.RecoverFEMSolution(X, b, x);

   ofstream mesh_ofs("pndnurbs_add.mesh");
   mesh_ofs.precision(8);
   
   pmesh->Print(mesh_ofs);
   
   ofstream sol_r_ofs("pndnurbs-adusol_r.gf");
   ofstream sol_i_ofs("pndnurbs-adusol_i.gf");
   sol_r_ofs.precision(8);
   sol_i_ofs.precision(8);
   x.real().Save(sol_r_ofs);
   x.imag().Save(sol_i_ofs);
   


   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection("nd_nurbs", pmesh);
   pd->SetPrefixPath("./ParaViewaduMode");
   pd->RegisterField("solution_real", &(x.real()));
   pd->RegisterField("solution_imag", &(x.imag()));
   pd->SetLevelsOfDetail(order);
   pd->SetDataFormat(VTKFormat::BINARY);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
   delete pd;
   // delete mesh;

   // Free the used memory.
   // delete fespace;
   // delete fec;
   // delete pmesh;
   //delete NURBSext;// NURBSext have been destoryed when construct the ParNURBSext!!!
   if(myid == 0)
   {
      mfem::out<<"****** FINISH ******"<<std::endl;
   }
   return 0;
   
}

void source_r(const Vector &x, Vector & f)
{
   real_t a,b,c;
   a = 0;
   b = 0;
   c = 0;
   real_t zz = x[2] + 1000;
   real_t yy = x[1] - 1000-1000;
   if(yy < 4000 && yy > 0)
   {
      get_source_E(yy,zz,x[0], f[1], a, f[2], b, f[0], c);
   }
   else{
      f = 0.0;
   }
   //mfem::out<<f[0]<<" "<<f[1]<<" "<<f[2]<<" ";
}

void source_i(const Vector &x, Vector & f)
{
   real_t a,b,c;
   a = 0;
   b = 0;
   c = 0;
   real_t zz = x[2] + 1000;
   real_t yy = x[1] - 1000-1000;
   if(yy < 4000 && yy > 0)
   {
      get_source_E(yy,zz,x[0], a, f[1], b, f[2], c, f[0]);
   }
   else{
      f = 0.0;
   }
}

void get_source_E(real_t x, real_t y, real_t z, 
                  real_t &Exr, real_t &Exi, real_t &Eyr, real_t &Eyi, real_t &Ezr, real_t &Ezi)
{
    real_t dx = 20;
    real_t dy = 20;
    int I = 200;//J = 200;

    int Ix = (int)(x/dx);
    int Iy = (int)(y/dy);

    real_t addx = 1.0*(x - Ix*dx)/dx;
    real_t addy = 1.0*(y - Iy*dy)/dy;

    real_t Ex_real = (1.0-addy)*((1.0-addx)*data_Ex_real[I*Iy + Ix] + addx*data_Ex_real[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ex_real[I*(Iy+1) + Ix] + addx*data_Ex_real[I*(Iy+1) + Ix+1]);
    real_t Ex_imag = (1.0-addy)*((1.0-addx)*data_Ex_imag[I*Iy + Ix] + addx*data_Ex_imag[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ex_imag[I*(Iy+1) + Ix] + addx*data_Ex_imag[I*(Iy+1) + Ix+1]);
    real_t Ey_real = (1.0-addy)*((1.0-addx)*data_Ey_real[I*Iy + Ix] + addx*data_Ey_real[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ey_real[I*(Iy+1) + Ix] + addx*data_Ey_real[I*(Iy+1) + Ix+1]);
    real_t Ey_imag = (1.0-addy)*((1.0-addx)*data_Ey_imag[I*Iy + Ix] + addx*data_Ey_imag[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ey_imag[I*(Iy+1) + Ix] + addx*data_Ey_imag[I*(Iy+1) + Ix+1]);
    real_t Ez_real = (1.0-addy)*((1.0-addx)*data_Ez_real[I*Iy + Ix] + addx*data_Ez_real[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ez_real[I*(Iy+1) + Ix] + addx*data_Ez_real[I*(Iy+1) + Ix+1]);
    real_t Ez_imag = (1.0-addy)*((1.0-addx)*data_Ez_imag[I*Iy + Ix] + addx*data_Ez_imag[I*Iy + Ix+1])+
                     addy*((1.0-addx)*data_Ez_imag[I*(Iy+1) + Ix] + addx*data_Ez_imag[I*(Iy+1) + Ix+1]);
    
    real_t phai = - z*0.0071918126554328;
    real_t real_phase = cos(phai);
    real_t imag_phase = -sin(phai);

    Exr =  real_phase * Ex_real - imag_phase * Ex_imag;
    Exi =  imag_phase * Ex_real + real_phase * Ex_imag;

    Eyr =  real_phase * Ey_real - imag_phase * Ey_imag;
    Eyi =  imag_phase * Ey_real + real_phase * Ey_imag;

    Ezr =  real_phase * Ez_real - imag_phase * Ez_imag;
    Ezi =  imag_phase * Ez_real + real_phase * Ez_imag;
    //std::cout<<Ex_real<<" "<<data_Ex_real[I*Iy + Ix]<<" "<<data_Ex_real[I*Iy + Ix+1]<<" "<<data_Ex_real[I*(Iy+1) + Ix]<<" "<<data_Ex_real[I*(Iy+1) + Ix+1]<<" "<<addx<<" "<<addy<<std::endl;
    
    return;
}

