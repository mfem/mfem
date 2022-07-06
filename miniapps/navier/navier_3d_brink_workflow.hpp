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
   double kin_vis = 4.9e-3;//0.04231;//0.0456 ;  //0.055
   double t_final = 0.6;
   double dt = 1.0e-4;
};

class DensityCoeff:public mfem::Coefficient
{
 public:

    enum ProjectionType {zero_one, continuous}; 

    enum PatternType {Ball, Gyroid, SchwarzP, SchwarzD,FCC, BCC, Octet};

private:


    double cx;
    double cy;
    double cz;
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
	if(pttype==Ball){	
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

            std::vector< std::vector<double>> tCenterVal(14);
            tCenterVal[0].push_back( 0.0); tCenterVal[0].push_back( 0.0); tCenterVal[0].push_back( 0.0);
            tCenterVal[1].push_back( 1.0); tCenterVal[1].push_back( 0.0); tCenterVal[1].push_back( 0.0);
            tCenterVal[2].push_back( 0.0); tCenterVal[2].push_back( 1.0); tCenterVal[2].push_back( 0.0);
            tCenterVal[3].push_back( 0.0); tCenterVal[3].push_back( 0.0); tCenterVal[3].push_back( 1.0);
            tCenterVal[4].push_back( 1.0); tCenterVal[4].push_back( 1.0); tCenterVal[4].push_back( 0.0);
            tCenterVal[5].push_back( 1.0); tCenterVal[5].push_back( 0.0); tCenterVal[5].push_back( 1.0);
            tCenterVal[6].push_back( 0.0); tCenterVal[6].push_back( 1.0); tCenterVal[6].push_back( 1.0);
            tCenterVal[7].push_back( 1.0); tCenterVal[7].push_back( 1.0); tCenterVal[7].push_back( 1.0);
            tCenterVal[8].push_back( 0.0); tCenterVal[8].push_back( 0.5); tCenterVal[8].push_back( 0.5);
            tCenterVal[9].push_back( 1.0); tCenterVal[9].push_back( 0.5); tCenterVal[9].push_back( 0.5);
            tCenterVal[10].push_back( 0.5); tCenterVal[10].push_back( 0.0); tCenterVal[10].push_back( 0.5);
            tCenterVal[11].push_back( 0.5); tCenterVal[11].push_back( 1.0); tCenterVal[11].push_back( 0.5);
            tCenterVal[12].push_back( 0.5); tCenterVal[12].push_back( 0.5); tCenterVal[12].push_back( 0.0);
            tCenterVal[13].push_back( 0.5); tCenterVal[13].push_back( 0.5); tCenterVal[13].push_back( 1.0);

            double x[3];
            Vector transip(x, 3);
            T.Transform(ip,transip);

            double vv = DBL_MAX;
            // loop over corners
            for( int Ik =0; Ik <14; Ik++)
            {
                double rr=(x[0]-tCenterVal[Ik][0])*(x[0]-tCenterVal[Ik][0]);
                       rr=rr+(x[1]-tCenterVal[Ik][1])*(x[1]-tCenterVal[Ik][1]);
                if(T.GetDimension()==3)
                {
                   rr=rr+(x[2]-tCenterVal[Ik][2])*(x[2]-tCenterVal[Ik][2]);
                }
                rr=std::sqrt(rr);

                vv = std::min( vv, rr);           
            }
            if(prtype==continuous){return vv;}

            if(fabs(vv)>eta){ return 0.0;}
            else{return 1.0;}
        }else
        if(pttype==BCC){

            std::vector< std::vector<double>> tCenterVal(9);
            tCenterVal[0].push_back( 0.0); tCenterVal[0].push_back( 0.0); tCenterVal[0].push_back( 0.0);
            tCenterVal[1].push_back( 1.0); tCenterVal[1].push_back( 0.0); tCenterVal[1].push_back( 0.0);
            tCenterVal[2].push_back( 0.0); tCenterVal[2].push_back( 1.0); tCenterVal[2].push_back( 0.0);
            tCenterVal[3].push_back( 0.0); tCenterVal[3].push_back( 0.0); tCenterVal[3].push_back( 1.0);
            tCenterVal[4].push_back( 1.0); tCenterVal[4].push_back( 1.0); tCenterVal[4].push_back( 0.0);
            tCenterVal[5].push_back( 1.0); tCenterVal[5].push_back( 0.0); tCenterVal[5].push_back( 1.0);
            tCenterVal[6].push_back( 0.0); tCenterVal[6].push_back( 1.0); tCenterVal[6].push_back( 1.0);
            tCenterVal[7].push_back( 1.0); tCenterVal[7].push_back( 1.0); tCenterVal[7].push_back( 1.0);
            tCenterVal[8].push_back( 0.5); tCenterVal[8].push_back( 0.5); tCenterVal[8].push_back( 0.5);


            double x[3];
            Vector transip(x, 3);
            T.Transform(ip,transip);

            double vv = DBL_MAX;
            // loop over corners
            for( int Ik =0; Ik <9; Ik++)
            {
                double rr=(x[0]-tCenterVal[Ik][0])*(x[0]-tCenterVal[Ik][0]);
                       rr=rr+(x[1]-tCenterVal[Ik][1])*(x[1]-tCenterVal[Ik][1]);
                if(T.GetDimension()==3)
                {
                   rr=rr+(x[2]-tCenterVal[Ik][2])*(x[2]-tCenterVal[Ik][2]);
                }
                rr=std::sqrt(rr);

                vv = std::min( vv, rr);           
            }
            if(prtype==continuous){return vv;}

            if(fabs(vv)>eta){ return 0.0;}
            else{return 1.0;}
        }else
        if(pttype==Octet){
            std::vector< double > Val(24);

            double x[3];
            Vector transip(x, 3);
            T.Transform(ip,transip);

            Val[0] =   std::abs( x[1] - x[0] ) + x[2] ;
            Val[1] =   std::abs( x[1] + x[0] - 1.0 ) + x[2];
            Val[2] =   std::abs( x[1] - x[0] ) + std::abs( x[2] - 1.0 );
            Val[3] =   std::abs( x[1] + x[0] - 1.0 ) + std::abs( x[2] - 1.0 );

            Val[4] =   std::abs( x[1] - x[2] ) + x[0] ;
            Val[5] =   std::abs( x[1] + x[2] - 1.0 ) + x[0];
            Val[6] =   std::abs( x[1] - x[2] ) + std::abs( x[0] - 1.0 );
            Val[7] =   std::abs( x[1] + x[2] - 1.0 ) + std::abs( x[0] - 1.0 );

            Val[8] =   std::abs( x[2] - x[0] ) + x[1] ;
            Val[9] =   std::abs( x[2] + x[0] - 1.0 ) + x[1];
            Val[10] =  std::abs( x[2] - x[0] ) + std::abs( x[1] - 1.0 );
            Val[11] =  std::abs( x[2] + x[0] - 1.0 ) + std::abs( x[1] - 1.0 );

            Val[12] =  std::abs( x[1] - x[0] + 0.5 ) + std::abs( x[2] - 0.5);
            Val[13] =  std::abs( x[1] + x[0] - 1.5 ) + std::abs( x[2] - 0.5);
            Val[14] =  std::abs( x[1] - x[0] - 0.5 ) + std::abs( x[2] - 0.5);
            Val[15] =  std::abs( x[1] + x[0] - 0.5 ) + std::abs( x[2] - 0.5);

            Val[16] =  std::abs( x[1] - x[2] + 0.5 ) + std::abs( x[0] - 0.5);
            Val[17] =  std::abs( x[1] + x[2] - 1.5 ) + std::abs( x[0] - 0.5);
            Val[18] =  std::abs( x[1] - x[2] - 0.5 ) + std::abs( x[0] - 0.5);
            Val[19] =  std::abs( x[1] + x[2] - 0.5 ) + std::abs( x[0] - 0.5);

            Val[20] =  std::abs( x[2] - x[0] + 0.5 ) + std::abs( x[1] - 0.5);
            Val[21] =  std::abs( x[2] + x[0] - 1.5 ) + std::abs( x[1] - 0.5);
            Val[22] =  std::abs( x[2] - x[0] - 0.5 ) + std::abs( x[1] - 0.5);
            Val[23] =  std::abs( x[2] + x[0] - 0.5 ) + std::abs( x[1] - 0.5);

            double vv = DBL_MAX;
            // loop over corners
            for( int Ik =0; Ik <24; Ik++)
            {
                vv = std::min( vv, Val[Ik]);           
            }
            if(prtype==continuous){return vv;}

            if(fabs(vv)>eta){ return 0.0;}
            else{return 1.0;}
        }else
        if(pttype==Gyroid){
            const double period = 2.0 * M_PI;
            double xv[3];
            mfem::Vector xx(xv,3);
            T.Transform(ip,xx);
            double x=xv[0]*period;
            double y=xv[1]*period;
            double z=xv[2]*period;
   
            double vv=std::sin(x)*std::cos(y) +
                      std::sin(y)*std::cos(z) +
                      std::sin(z)*std::cos(x);
            if(prtype==continuous)
            {
                double val;
                if( vv > 0 )
                {
                    return -1.0*( vv - eta );
                }
                else
                {
                    return vv + eta;
                }
                
            }

            if(fabs(vv)>-eta && fabs(vv)<eta){ return 1.0;}
            else{return 0.0;}
            //if(fabs(vv)>eta){ return 0.0;}
            //else{return 1.0;}
        }else
        if(pttype==SchwarzD){
            const double period = 2.0 * M_PI;
            double xv[3];
            mfem::Vector xx(xv,3);
            T.Transform(ip,xx);
            double x=xv[0]*period;
            double y=xv[1]*period;
            double z=xv[2]*period;
            
            double vv=sin(x)*sin(y)*sin(z) +
                      sin(x)*cos(y)*cos(z) +
                      cos(x)*sin(y)*cos(z) +
                      cos(x)*cos(y)*sin(z);

            if(prtype==continuous){return vv;}
            
            if(fabs(vv)>eta){ return 0.0;}
            else{return 1.0;}		
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
        double dens=dcoeff->Eval(T,ip);
        V.SetSize(GetVDim());
        if(vel==nullptr)
        {
            V=0.0;
        }
        else
        {
            if(dens<1e-8)
            {
                V(0)=mnx * ma;
                V(1)=mny * ma;
                V(2)=mnz * ma;
            }else{
                vel->GetVectorValue(T,ip,V);
                V*=-10000.0;
            }
        }
    }

private:
    mfem::GridFunction* vel = nullptr;
    mfem::Coefficient* dcoeff = nullptr;

    double mnx  = 0.0;
    double mny  = 0.0;
    double mnz  = 0.0;
    double ma   = 0.0;

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

   void SetDensityCoeff();

   void SetupFlowSolver();

   void SetInitialConditions( std::function<void(const Vector &, double, Vector &)> TDF );

   void SetupOutput( );

   void Perform( );

   void Postprocess( const int & runID );
};

} // namespace navier

} // namespace mfem

#endif
