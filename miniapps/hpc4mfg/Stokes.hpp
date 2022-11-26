#ifndef STOKES_HPP
#define STOKES_HPP

#include "hpc4solvers.hpp"
#include "mfem.hpp"

namespace mfem{
    namespace stokes{

        class DensityCoeff:public mfem::Coefficient
{
 public:

    enum ProjectionType {zero_one, continuous}; 

    enum PatternType {Ball, Gyroid, SchwarzP, SchwarzD,FCC, BCC, Octet, TRUSS,Ellipse};

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
        if(pttype==Ellipse){	
            mfem::Vector transip(3);
            mfem::Vector rotatedCoords(3);
            T.Transform(ip,transip);
            transip -= 0.5;

            double Angele = M_PI /6.0;

            mfem::DenseMatrix tRot(3); tRot = 0.0;
            tRot(0,0) = std::cos(Angele);
            tRot(1,0) = -std::sin(Angele);
            tRot(0,1) = std::sin(Angele);
            tRot(1,1) = std::cos(Angele);

            tRot.Mult(transip,rotatedCoords);

            rotatedCoords +=0.5;

            double rr=(rotatedCoords[0]-cx)*(rotatedCoords[0]-cx);
            rr=rr+(rotatedCoords[1]-cy)*(rotatedCoords[1]-cy)/0.25;
            if(T.GetDimension()==3)
            {
                mfem_error("not implemented");
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
        if(pttype==TRUSS){
            std::vector< double > Val(3);

            double x[3];
            Vector transip(x, 3);
            T.Transform(ip,transip);

            Val[0] =   std::sqrt( std::pow( x[0] - 0.5,2 ) + std::pow( x[1] - 0.5,2 ) )  ;
            Val[1] =   std::sqrt( std::pow( x[1] - 0.5,2 ) + std::pow( x[2] - 0.5,2 ) ) ;
            Val[2] =   std::sqrt( std::pow( x[0] - 0.5,2 ) + std::pow( x[2] - 0.5,2 ) ) ;


            double vv = DBL_MAX;
            // loop over corners
            for( int Ik =0; Ik <3; Ik++)
            {
                vv = std::min( vv, Val[Ik]);           
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

            if(true)
            {
            Val[0]  = std::sqrt( std::pow(x[0] - 1.0 + x[1], 2 ) + std::pow(x[0] - 1.0 + x[1], 2 ) + std::pow(x[2], 2 ) );
            Val[1]  = std::sqrt( std::pow(x[1] - 1.0 + x[2], 2 ) + std::pow(x[2] - 1.0 + x[1], 2 ) + std::pow(x[0], 2 ) );
            Val[2]  = std::sqrt( std::pow(x[0] - 1.0 + x[2], 2 ) + std::pow(x[0] - 1.0 + x[2], 2 ) + std::pow(x[1], 2 ) );

            Val[3]  = std::sqrt( std::pow(x[0] - 1.0 + x[1], 2 ) + std::pow(x[0] - 1.0 + x[1], 2 ) + std::pow(x[2] - 1.0, 2 ) );
            Val[4]  = std::sqrt( std::pow(x[1] - 1.0 + x[2], 2 ) + std::pow(x[2] - 1.0 + x[1], 2 ) + std::pow(x[0] - 1.0, 2 ) );
            Val[5]  = std::sqrt( std::pow(x[0] - 1.0 + x[2], 2 ) + std::pow(x[0] - 1.0 + x[2], 2 ) + std::pow(x[1] - 1.0, 2 ) );

            Val[6]  = std::sqrt( std::pow(x[0] - x[1], 2 ) + std::pow(x[1] - x[0], 2 ) + std::pow(x[2], 2 ) );
            Val[7]  = std::sqrt( std::pow(x[1] - x[2], 2 ) + std::pow(x[2] - x[1], 2 ) + std::pow(x[0], 2 ) );
            Val[8]  = std::sqrt( std::pow(x[0] - x[2], 2 ) + std::pow(x[2] - x[0], 2 ) + std::pow(x[1], 2 ) );

            Val[9]  = std::sqrt( std::pow(x[0] - x[1], 2 ) + std::pow(x[1] - x[0], 2 ) + std::pow(x[2] - 1.0, 2 ) );
            Val[10] = std::sqrt( std::pow(x[1] - x[2], 2 ) + std::pow(x[2] - x[1], 2 ) + std::pow(x[0] - 1.0, 2 ) );
            Val[11] = std::sqrt( std::pow(x[0] - x[2], 2 ) + std::pow(x[2] - x[0], 2 ) + std::pow(x[1] - 1.0, 2 ) );

            Val[12] = std::sqrt( std::pow(x[0] - 0.5 + x[1], 2 ) + std::pow(x[1] - 0.5 + x[0], 2 ) + std::pow(x[2] - 0.5, 2 ) );
            Val[13] = std::sqrt( std::pow(x[0] - 0.5 - x[1], 2 ) + std::pow(x[1] + 0.5 - x[0], 2 ) + std::pow(x[2] - 0.5, 2 ) );

            Val[14] = std::sqrt( std::pow(x[1] - 0.5 + x[2], 2 ) + std::pow(x[2] - 0.5 + x[1], 2 ) + std::pow(x[0] - 0.5, 2 ) );
            Val[15] = std::sqrt( std::pow(x[1] - 0.5 - x[2], 2 ) + std::pow(x[2] + 0.5 - x[1], 2 ) + std::pow(x[0] - 0.5, 2 ) );

            Val[16] = std::sqrt( std::pow(x[0] - 0.5 + x[2], 2 ) + std::pow(x[2] - 0.5 + x[0], 2 ) + std::pow(x[1] - 0.5, 2 ) );
            Val[17] = std::sqrt( std::pow(x[0] - 0.5 - x[2], 2 ) + std::pow(x[2] + 0.5 - x[0], 2 ) + std::pow(x[1] - 0.5, 2 ) );

            Val[18] = std::sqrt( std::pow(x[0] + 0.5 - x[1], 2 ) + std::pow(x[1] - 0.5 - x[0], 2 ) + std::pow(x[2] - 0.5, 2 ) );
            Val[19] = std::sqrt( std::pow(x[0] - 1.5 + x[1], 2 ) + std::pow(x[1] - 1.5 + x[0], 2 ) + std::pow(x[2] - 0.5, 2 ) );

            Val[20] = std::sqrt( std::pow(x[1] + 0.5 - x[2], 2 ) + std::pow(x[2] - 0.5 - x[1], 2 ) + std::pow(x[0] - 0.5, 2 ) );
            Val[21] = std::sqrt( std::pow(x[1] - 1.5 + x[2], 2 ) + std::pow(x[2] - 1.5 + x[1], 2 ) + std::pow(x[0] - 0.5, 2 ) );

            Val[22] = std::sqrt( std::pow(x[0] + 0.5 - x[2], 2 ) + std::pow(x[2] - 0.5 - x[0], 2 ) + std::pow(x[1] - 0.5, 2 ) );
            Val[23] = std::sqrt( std::pow(x[0] - 1.5 + x[2], 2 ) + std::pow(x[2] - 1.5 + x[0], 2 ) + std::pow(x[1] - 0.5, 2 ) );
            }
            else
            {
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
            }

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
            const double period = 2.0 * M_PI*4.0;
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
            // if(prtype==continuous)
            // {
            //     double val;
            //     if( vv > 0 )
            //     {
            //         return -1.0*( vv - eta );
            //     }
            //     else
            //     {
            //         return vv + eta;
            //     }      
            // }

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

        if(dcoeff == nullptr)
        {
            mfem_error("dcoeff is nullptr!");
        }

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
                        V(0)= -1.0*mnx * ma;
                        V(1)= -1.0*mny * ma;
                    if(T.GetDimension()==3)
                    {
                        V(2)=-1.0* mnz * ma;
                    }
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
                    if(T.GetDimension()==3)
                    {
                        V(2)=mnz * ma;
                    }
                }
                else
                {
                    V(0)=-1.0*mnx * ma;
                    V(1)=-1.0*mny * ma;
                    if(T.GetDimension()==3)
                    {
                        V(2)=-1.0*mnz * ma;
                    }
                }
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

    double BrinkammPen_ =10000.0;

    };


class Stokes
{


public:
    Stokes(mfem::ParMesh* mesh_, int order_=2)
    {
        pmesh=mesh_;
        int dim=pmesh->Dimension();

        fec_u = new H1_FECollection(order_,dim);
        fes_u = new ParFiniteElementSpace(pmesh,fec_u,dim);   

        fec_p = new H1_FECollection(order_-1,dim);
        fes_p = new ParFiniteElementSpace(pmesh,fec_p);

        fes_u_vol = new ParFiniteElementSpace(pmesh,fec_u,1); 
  
        sol.SetSize(fes_u->GetTrueVSize()); sol=0.0;
        rhs.SetSize(fes_u->GetTrueVSize()); rhs=0.0;

        solgf.SetSpace(fes_u); solgf =  0.0;

        SetLinearSolver();
    }

    ~Stokes(){
        delete ls;
        delete prec;

        delete fes_p;
        delete fec_p;
        delete fes_u;
        delete fec_u;

        delete fes_u_vol;

        delete b;

        delete a_u;
        delete a_p;
        delete a_up;
        delete a_pu;


        for(size_t i=0;i<materials.size();i++){
            delete materials[i];
        }
    }

    /// Set the Linear Solver
    void SetLinearSolver(double rtol=1e-8, double atol=1e-12, int miter=1)
    {
        linear_rtol=rtol;
        linear_atol=atol;
        linear_iter=miter;
    }

    /// Solves the forward problem.
    void FSolve();

    /// Adds Dirichlet BC
    void AddDirichletBC(int id, double val)
    {
        bc[id]=mfem::ConstantCoefficient(val);
        AddDirichletBC(id,bc[id]);
    }

    /// Adds Dirichlet BC
    void AddDirichletBC(int id, mfem::Coefficient& val)
    {
        bcc[id]=&val;
    }

    /// Adds Neumann BC
    void AddNeumannBC(int id, double val)
    {
        nc[id]=mfem::ConstantCoefficient(val);
        AddNeumannBC(id,nc[id]);
    }

    /// Adds Neumann BC
    void AddNeumannBC(int id, mfem::Coefficient& val)
    {
        ncc[id]=&val;
    }

    /// Returns the solution
    mfem::ParGridFunction& GetSolution(){return solgf;}

    /// Add material to the solver. The pointer is owned by the solver.
    void AddMaterial(MatrixCoefficient* nmat)
    {
        materials.push_back(nmat);
    }

    void SetVelocity( mfem::VectorCoefficient* vel )
    {
        vel_ = vel;
    }

    void SetGradTempMean( mfem::VectorCoefficient* avgGradTemp )
    {
        avgGradTemp_ = avgGradTemp;
    }

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

    /// Returns the solution vector.
    mfem::Vector& GetSol(){return sol;}

    void GetSol(ParGridFunction& sgf){
        sgf.SetSpace(fes_u); sgf.SetFromTrueDofs(sol);}

    void Postprocess();

private:
    mfem::ParMesh* pmesh;

    std::vector<MatrixCoefficient*> materials;

    ParBilinearForm *a_u = nullptr;
    ParBilinearForm *a_p = nullptr;
    ParMixedBilinearForm *a_up = nullptr;
    ParMixedBilinearForm *a_pu = nullptr;
    ParLinearForm *b = nullptr;

    //solution true vector
    mfem::Vector sol;
    mfem::Vector rhs;
    mfem::ParGridFunction solgf;

    mfem::VectorCoefficient* vel_ = nullptr;
    mfem::VectorCoefficient* avgGradTemp_ = nullptr;

    mfem::FiniteElementCollection *fec_p;
    mfem::FiniteElementCollection *fec_u;
    mfem::ParFiniteElementSpace	  *fes_p;
    mfem::ParFiniteElementSpace	  *fes_u;
    mfem::ParFiniteElementSpace	  *fes_u_vol;

    // OperatorHandle L_uu;
    // OperatorHandle D_pu;
    // OperatorHandle D_up;

    DensityCoeff * mDensCoeff = nullptr;
    double mnx  = 0.0;
    double mny  = 0.0;
    double mnz  = 0.0;
    double ma   = 0.0;
    double meta = 0.0;

    //Linear solver parameters
    double linear_rtol;
    double linear_atol;
    int linear_iter;

    int print_level = 1;

    mfem::HypreBoomerAMG *prec = nullptr; //preconditioner
    mfem::CGSolver *ls = nullptr;  //linear solver

    // holds DBC in coefficient form
    std::map<int, mfem::Coefficient*> bcc;

    // holds internal DBC
    std::map<int, mfem::ConstantCoefficient> bc;

    // holds NBC in coefficient form
    std::map<int, mfem::Coefficient*> ncc;

    // holds internal NBC
    std::map<int, mfem::ConstantCoefficient> nc;

    mfem::Array<int> ess_tdofv;

    ParaViewDataCollection * mPvdc = nullptr;
};


    }

}

#endif
