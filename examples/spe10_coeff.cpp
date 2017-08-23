/*
 * spe10_coeff.cpp
 *
 *  Created on: Aug 23, 2017
 *      Author: neumueller
 */

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class InversePermeabilityFunction
{
public:

   enum SliceOrientation {NONE, XY, XZ, YZ};

   static void SetNumberCells(int Nx_, int Ny_, int Nz_);
   static void SetMeshSizes(double hx, double hy, double hz);
   static void Set2DSlice(SliceOrientation o, int npos );

   static void ReadPermeabilityFile(const std::string fileName);
#ifdef MFEM_USE_MPI
   static void ReadPermeabilityFile(const std::string fileName, MPI_Comm comm);
#endif
   static void SetConstantInversePermeability(double ipx, double ipy, double ipz);

   template<class F>
   static void Transform(const F & f)
   {
      for (int i = 0; i < 3*Nx*Ny*Nz; ++i)
         inversePermeability[i] = f(inversePermeability[i]);
   }

   static void InversePermeability(const Vector & x, Vector & val);
   static double PermeabilityXY(Vector &x);
   static void NegativeInversePermeability(const Vector & x, Vector & val);
   static void Permeability(const Vector & x, Vector & val);

   static double Norm2Permeability(const Vector & x);

   static double Norm2InversePermeability(const Vector & x);
   static double Norm1InversePermeability(const Vector & x);
   static double NormInfInversePermeability(const Vector & x);

   static double InvNorm2(const Vector & x);
   static double InvNorm1(const Vector & x);
   static double InvNormInf(const Vector & x);


   static void ClearMemory();

private:
   static int Nx;
   static int Ny;
   static int Nz;
   static double hx;
   static double hy;
   static double hz;
   static double * inversePermeability;

   static SliceOrientation orientation;
   static int npos;
};


void InversePermeabilityFunction::SetNumberCells(int Nx_, int Ny_, int Nz_)
{
   Nx = Nx_;
   Ny = Ny_;
   Nz = Nz_;
}

void InversePermeabilityFunction::SetMeshSizes(double hx_, double hy_,
                                               double hz_)
{
   hx = hx_;
   hy = hy_;
   hz = hz_;
}

void InversePermeabilityFunction::Set2DSlice(SliceOrientation o, int npos_ )
{
   orientation = o;
   npos = npos_;
}

void InversePermeabilityFunction::SetConstantInversePermeability(double ipx,
                                                                 double ipy, double ipz)
{
   int compSize = Nx*Ny*Nz;
   int size = 3*compSize;
   inversePermeability = new double [size];
   double *ip = inversePermeability;

   for (int i(0); i < compSize; ++i)
   {
      ip[i] = ipx;
      ip[i+compSize] = ipy;
      ip[i+2*compSize] = ipz;
   }

}

#ifdef MFEM_USE_MPI
void InversePermeabilityFunction::ReadPermeabilityFile(const std::string
                                                       fileName, MPI_Comm comm)
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   StopWatch chrono;

   chrono.Start();
   if (myid == 0)
      ReadPermeabilityFile(fileName);
   else
      inversePermeability = new double [3*Nx*Ny*Nz];
   chrono.Stop();

   if (myid==0)
      std::cout<<"Permeability file read in " << chrono.RealTime() << ".s \n";

   chrono.Clear();

   chrono.Start();
   MPI_Bcast(inversePermeability, 3*Nx*Ny*Nz, MPI_DOUBLE, 0, comm);
   chrono.Stop();

   if (myid==0)
      std::cout<<"Permeability field distributed in " << chrono.RealTime() << ".s \n";

}
#endif

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string
                                                       fileName)
{
   std::ifstream permfile(fileName.c_str());

   if (!permfile.is_open())
   {
      std::cout << "Error in opening file " << fileName << "\n";
      mfem_error("File do not exists");
   }

   inversePermeability = new double [3*Nx*Ny*Nz];
   double *ip = inversePermeability;
   double tmp;
   for (int l = 0; l < 3; l++)
   {
      for (int k = 0; k < Nz; k++)
      {
         for (int j = 0; j < Ny; j++)
         {
            for (int i = 0; i < Nx; i++)
            {
               permfile >> *ip;
               *ip = 1./(*ip);
               ip++;
            }
            for (int i = 0; i < 60-Nx; i++)
               permfile >> tmp; // skip unneeded part
         }
         for (int j = 0; j < 220-Ny; j++)
            for (int i = 0; i < 60; i++)
               permfile >> tmp;  // skip unneeded part
      }

      if (l < 2) // if not processing Kz, skip unneeded part
         for (int k = 0; k < 85-Nz; k++)
            for (int j = 0; j < 220; j++)
               for (int i = 0; i < 60; i++)
                  permfile >> tmp;
   }

}

void InversePermeabilityFunction::InversePermeability(const Vector & x,
                                                      Vector & val)
{
   val.SetSize(3);

   unsigned int i,j,k;

   switch (orientation)
   {
      case NONE:
         i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
         j = (int)floor(x[1]/hy/(1.+3e-16));
         k = Nz-1-(int)floor(x[2]/hz/(1.+3e-16));
         break;
      case XY:
         i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
         j = (int)floor(x[1]/hy/(1.+3e-16));
         k = npos;
         break;
      case XZ:
         i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
         j = npos;
         k = Nz-1-(int)floor(x[2]/hz/(1.+3e-16));
         break;
      case YZ:
         i = npos;
         j = (int)floor(x[1]/hy/(1.+3e-16));
         k = Nz-1-(int)floor(x[2]/hz/(1.+3e-16));
         break;
      default:
      {
         mfem_error("InversePermeabilityFunction::InversePermeability");
      }
   }


   int NMax = 3*Nx*Ny*Nz-1;
   if(Ny*Nx*k + Nx*j + i>NMax || Ny*Nx*k + Nx*j + i + Nx*Ny*Nz>NMax || Ny*Nx*k + Nx*j + i + 2*Nx*Ny*Nz>NMax)
   {
	   cout << " the indicies are wrong!" << endl;
	   cout << i << " " << j << " " << k << endl;
   }

   val[0] = inversePermeability[Ny*Nx*k + Nx*j + i];
   val[1] = inversePermeability[Ny*Nx*k + Nx*j + i + Nx*Ny*Nz];

   if (orientation == NONE)
      val[2] = inversePermeability[Ny*Nx*k + Nx*j + i + 2*Nx*Ny*Nz];

}

double InversePermeabilityFunction::PermeabilityXY(Vector &x)
{
   unsigned int i,j,k;

   i = Nx-1-(int)floor(x[0]/hx/(1.+3e-16));
   j = (int)floor(x[1]/hy/(1.+3e-16));
   k = npos;

   return 1./inversePermeability[Ny*Nx*k + Nx*j + i];
}

void InversePermeabilityFunction::NegativeInversePermeability(const Vector & x,
                                                              Vector & val)
{
   InversePermeability(x,val);
   val *= -1.;
}


void InversePermeabilityFunction::Permeability(const Vector & x, Vector & val)
{
   InversePermeability(x,val);

   for (double * it = val.GetData(), *end = val.GetData()+val.Size(); it != end;
        ++it )
      (*it) = 1./ (*it);
}

double InversePermeabilityFunction::Norm2Permeability(const Vector & x)
{
   Vector val(3);
   Permeability(x,val);
   return val.Norml2();
}


double InversePermeabilityFunction::Norm2InversePermeability(const Vector & x)
{
   Vector val(3);
   InversePermeability(x,val);
   return val.Norml2();
}

double InversePermeabilityFunction::Norm1InversePermeability(const Vector & x)
{
   Vector val(3);
   InversePermeability(x,val);
   return val.Norml1();
}

double InversePermeabilityFunction::NormInfInversePermeability(const Vector & x)
{
   Vector val(3);
   InversePermeability(x,val);
   return val.Normlinf();
}

double InversePermeabilityFunction::InvNorm2(const Vector & x)
{
   Vector val(3);
   InversePermeability(x,val);
   return 1./val.Norml2();
}

double InversePermeabilityFunction::InvNorm1(const Vector & x)
{
   Vector val(3);
   InversePermeability(x,val);
   return 1./val.Norml1();
}

double InversePermeabilityFunction::InvNormInf(const Vector & x)
{
   Vector val(3);
   InversePermeability(x,val);
   return 1./val.Normlinf();
}


void InversePermeabilityFunction::ClearMemory()
{
   delete[] inversePermeability;
}

int InversePermeabilityFunction::Nx(60);
int InversePermeabilityFunction::Ny(220);
int InversePermeabilityFunction::Nz(85);
double InversePermeabilityFunction::hx(20);
double InversePermeabilityFunction::hy(10);
double InversePermeabilityFunction::hz(2);
double * InversePermeabilityFunction::inversePermeability(NULL);
InversePermeabilityFunction::SliceOrientation
InversePermeabilityFunction::orientation( InversePermeabilityFunction::NONE );
int InversePermeabilityFunction::npos(-1);



