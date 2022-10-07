// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NAVIER_3D_BRINK_WORKFLOW
#define MFEM_NAVIER_3D_BRINK_WORKFLOW

#include "mfem.hpp"
#include "navier_solver.hpp"
#include <fstream>

namespace mfem
{
namespace navier
{

struct s_NavierContext
{
   int order = 2;
   //double kin_vis = 1.0 / (2.64e-3*1.0/ (1.516e-5))  ;//0.055;//0.04231;//0.0456 ;  //0.055
   double kin_vis = 1.0 / ( 0.1 / (1e-3))  ;//0.055;//0.04231;//0.0456 ;  //0.055
   double t_final = 0.2;
   double dt = 1.0e-4;
};

class DensityCoeff:public mfem::Coefficient
{
 public:

    enum ProjectionType {zero_one, continuous}; 

    enum PatternType {Circle};

private:


    double cx;
    double cy;
    double eta;//threshold
    ProjectionType prtype; 
    PatternType    pttype;

public:

    DensityCoeff()
    {
        cx=0.5;
        cy=0.5;
        cz=1.0;
        eta=std::sqrt(2.0)/4;//0.1;
        prtype=zero_one;
        pttype=Ball;

    }

    void SetBallCoord(double xx, double yy, double zz)
    {
        cx=xx;
        cy=yy;
        cz=zz;
    }

    void SetBallR(double RR)
    {
        eta=RR;
    }

    void SetThreshold(double eta_)
    {
       eta=eta_;
    }
    
    void SetProjectionType(ProjectionType prtype_)
    {
       prtype=prtype_;
    }


    void SetPatternType(PatternType pttype_)
    {
      pttype=pttype_;
    }


    virtual
    ~DensityCoeff()
    {

    }

    virtual
    double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {
	if(pttype==Circle){	
            double x[3];
            Vector transip(x, 3);
            T.Transform(ip,transip);
            double rr=(x[0]-cx)*(x[0]-cx);
            rr=rr+(x[1]-cy)*(x[1]-cy);
            if(T.GetDimension()==3)
            {
                rr=rr+(x[2]-cz)*(x[2]-cz);
            }
            rr=std::sqrt(rr);
            if(prtype==continuous){return rr;}

            if(rr>eta){return 0.0;}
            return 1.0;  
        }else
        if(pttype==FCC){

        }else{ //pttype=SchwarzP
            const double period = 2.0 * M_PI;
            double xv[3];
            mfem::Vector xx(xv,3);
            T.Transform(ip,xx);
            double x=xv[0]*period;
            double y=xv[1]*period;
            double z=xv[2]*period;

            double vv=std::cos(x)+std::cos(y)+std::cos(z);
	    if(prtype==continuous){return vv;}

            if(fabs(vv)>-eta && fabs(vv)<eta){ return 1.0;}
            else{return 0.0;}
        }

    }
};

class BrinkPenalAccel:public mfem::VectorCoefficient
{
public:
    BrinkPenalAccel(int dim):mfem::VectorCoefficient(dim)
    {

    }

    virtual
    ~BrinkPenalAccel()
    {

    }

    void SetVel(mfem::GridFunction* gfvel)
    {
        vel=gfvel;
    }

    void SetDensity(mfem::Coefficient* coeff)
    {
        dcoeff=coeff;
    }

    void SetBrinkmannPenalization( double BrinkammPen )
    {
        BrinkammPen_ = BrinkammPen;
    }

    void SetParams(        
       double anx, 
       double any,
       double anz,
       double aa)
       {
            mnx  = anx;
            mny  = any;
            mnz  = anz;
            ma   = aa;
       };

    virtual void Eval (mfem::Vector &V, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip)
    {
        V.SetSize(GetVDim());

        if(vel==nullptr)
        {
            V=0.0;
        }
        else if(true)
        {
            double dens=dcoeff->Eval(T,ip); 

            if(true)
            {
                if(dens<1e-8)
                {
                        V(0)=mnx * ma;
                        V(1)=mny * ma;
                        V(2)=mnz * ma;
                }else
                {
                    vel->GetVectorValue(T,ip,V);
                    V*=-BrinkammPen_;
                }
            }
            else
            {
                if(std::abs(dens)<0.25)
                {
                    vel->GetVectorValue(T,ip,V);
                    V*=-BrinkammPen_;

                }
                else if(dens>0.0)
                {
                    V(0)=mnx * ma;
                    V(1)=mny * ma;
                    V(2)=mnz * ma;
                }
                else
                {
                    V(0)=-1.0*mnx * ma;
                    V(1)=-1.0*mny * ma;
                    V(2)=-1.0*mnz * ma;
                }
            }
        }
        else
        {
            if( 2 == T.Attribute )
            {
                    V(0)=mnx * ma;
                    V(1)=mny * ma;
                    V(2)=mnz * ma;
            }
            else if(1 == T.Attribute )
            {
                vel->GetVectorValue(T,ip,V);
                V*=-1000.0;
            }
            else
            {
                std::cout<<" unknown element attribute: " <<T.Attribute << std::endl;
                mfem_error();
            }
        }
    };
    


private:
    mfem::GridFunction* vel = nullptr;
    mfem::Coefficient* dcoeff = nullptr;

    double mnx  = 0.0;
    double mny  = 0.0;
    double mnz  = 0.0;
    double ma   = 0.0;

    double BrinkammPen_ =10000.0;

};


class Navier3dBrinkWorkflow 
{


private:

   const MPI_Session & mMPI;

   ParMesh * mPMesh;

   const struct s_NavierContext & mCtk;

   DensityCoeff * mDensCoeff = nullptr;

   NavierSolver * mFlowsolver = nullptr;

   ParaViewDataCollection * mPvdc = nullptr;

   BrinkPenalAccel * mBp = nullptr;

   bool mVisualization = true;
   
public:

   Navier3dBrinkWorkflow( 
       const MPI_Session & aMPI,
       ParMesh * pmesh,
       const struct s_NavierContext & aCtk );

       double mnx  = 0.0;
       double mny  = 0.0;
       double mnz  = 0.0;
       double ma   = 0.0;
       double meta = 0.0;

   ~Navier3dBrinkWorkflow();

   void SetParams(        
       double anx, 
       double any,
       double anz,
       double aa,
       double aeta)
       {
            mnx  = anx;
            mny  = any;
            mnz  = anz;
            ma   = aa;
            meta = aeta;
       };

   void SetDensityCoeff(    
        enum DensityCoeff::PatternType aGeometry,
        enum DensityCoeff::ProjectionType aProjectionType);

   void SetupFlowSolver();

   void SetInitialConditions( std::function<void(const Vector &, double, Vector &)> TDF, bool LoadSolVecFromFile, double BrinmannPen = 1.0e4 );

   void SetupOutput( );

   void Perform( );

   void Postprocess( const int & runID );
};

} // namespace navier

} // namespace mfem

#endif
