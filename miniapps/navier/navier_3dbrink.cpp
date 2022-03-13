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

// 3D flow over a cylinder benchmark example

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 2;
   double kin_vis = 0.1;
   double t_final = 1.0;
   double dt = 1e-4;
} ctx;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);
   double zi = x(2);

   u(0) = 0.0;
   u(1) = 0.0;
   if(zi<=1e-8){
       if(t<1.0){ u(2) = t;}
       else{ u(2)=1.0; }
   }else{
       u(2)=0.0;
   }
   u(2)=0.0;
}

class DensityCoeff:public mfem::Coefficient
{
public:
    DensityCoeff()
    {
        cx=0.5;
        cy=0.5;
        cz=1.0;
        eta=0.1;
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
    
    enum ProjectionType {zero_one, continuous}; 

    void SetProjectionType(ProjectionType ptype_)
    {
       prtype=ptype_;
    }

    enum PatternType {Ball, Gyroid, SchwarzP, SchwarzD};

    void SetPatternType(PatternType pttype_)
    {
      pttype=pttype_;
    }


    virtual
    ~DensityCoeff()
    {

    }

    virtual
    double Eval(ElementTransformation &T, const IntegrationPoint &ip)
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
            if(prtype==continuous){return vv;}

            if(fabs(vv)>eta){ return 0.0;}
            else{return 1.0;}
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

            if(fabs(vv)>eta){ return 0.0;}
            else{return 1.0;}
        }

    }

private:
    double cx;
    double cy;
    double cz;
    double eta;//threshold
    ProjectionType prtype; 
    PatternType    pttype;


};


class BrinkPenalAccel:public mfem::VectorCoefficient
{
public:
    BrinkPenalAccel(int dim):mfem::VectorCoefficient(dim)
    {
        dcoeff=nullptr;
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

    virtual void Eval (Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
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
                V(0)=0.0;
                V(1)=0.0;
                V(2)=10.0;
            }else{
                vel->GetVectorValue(T,ip,V);
                V*=-10000.0;
            }
        }
    }

private:
    mfem::GridFunction* vel;
    mfem::Coefficient* dcoeff;

};


int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   int serial_refinements = 0;

   Mesh *mesh = new Mesh("bar3d.msh");
   //mesh->EnsureNCMesh(true);

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   if (mpi.Root())
   {
      std::cout << "Mesh of elements: " << mesh->GetNE() << std::endl;
   }

   DensityCoeff dens;
   //dens.SetBallCoord(0.5,0.5,1.0);
   //dens.SetBallR(0.1);
   dens.SetThreshold(0.15);
   dens.SetPatternType(DensityCoeff::PatternType::Gyroid);
   dens.SetPatternType(DensityCoeff::PatternType::SchwarzP);
   dens.SetPatternType(DensityCoeff::PatternType::SchwarzD);
   dens.SetProjectionType(DensityCoeff::ProjectionType::zero_one);


   //Refine the mesh
   if(0)
   {
       int nclimit=1;
       for (int iter = 0; iter<3; iter++)
       {
           Array<Refinement> refs;
           for (int i = 0; i < pmesh->GetNE(); i++)
           {
              bool refine = false;
              Geometry::Type geom = pmesh->GetElementBaseGeometry(i);
              ElementTransformation *T = pmesh->GetElementTransformation(i);
              RefinedGeometry *RefG = mfem::GlobGeometryRefiner.Refine(geom, 2, 1);
              IntegrationRule &ir = RefG->RefPts;

              // Refine any element where different materials are detected. A more
              // sophisticated logic can be implemented here -- e.g. don't refine
              // the interfaces between certain materials.
              Array<int> mat(ir.GetNPoints());
              double matsum = 0.0;
              for (int j = 0; j < ir.GetNPoints(); j++)
              {
                 //T->Transform(ir.IntPoint(j), pt);
                 //int m = material(pt, xmin, xmax);
                 int m = dens.Eval(*T,ir.IntPoint(j));
                 mat[j] = m;
                 matsum += m;
                 if ((int)matsum != m*(j+1))
                 {
                    refine = true;
                 }
              }

              // Mark the element for refinement
              if (refine)
              {
                  refs.Append(Refinement(i));
              }

           }

           //pmesh->GeneralRefinement(refs, -1, nclimit);
           pmesh->GeneralRefinement(refs, 0, nclimit);
           //pmesh->GeneralRefinement(refs);
       }

       //pmesh->Rebalance();
   }



   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);
   flowsolver.EnablePA(true);
   flowsolver.EnableNI(true);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   u_ic->ProjectCoefficient(u_excoeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   //Array<int> attr(pmesh->bdr_attributes.Max());
   // Inlet is attribute 1.
   //attr[0] = 1;
   // Walls is attribute 2.
   //attr[1] = 1;
   //flowsolver.AddVelDirichletBC(vel, attr);

   Array<int> domain_attr(pmesh->attributes.Max());
   domain_attr = 1;
   BrinkPenalAccel bp(pmesh->Dimension());
   bp.SetDensity(&dens);
   bp.SetVel(flowsolver.GetCurrentVelocity());
   flowsolver.AddAccelTerm(&bp,domain_attr);


   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();
   ParGridFunction *d_gf = new ParGridFunction(*p_gf);
   dens.SetProjectionType(DensityCoeff::ProjectionType::continuous);
   d_gf->ProjectCoefficient(dens);
   dens.SetProjectionType(DensityCoeff::ProjectionType::zero_one);

   ParaViewDataCollection pvdc("3dfoc1", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("density",  d_gf);
   pvdc.Save();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);
      bp.SetVel(flowsolver.GetCurrentVelocity());
      //bp.SetVel(flowsolver.GetProvisionalVelocity());
      if (step % 3000 == 0)
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      if (mpi.Root())
      {
         printf("%11s %11s\n", "Time", "dt");
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();
   delete d_gf;
   delete pmesh;

   return 0;
}
