#ifndef MMS_COEFFICIENTS_HPP
#define MMS_COEFFICIENTS_HPP

#include "mfem.hpp"

namespace mfem{


template<typename fp_type, typename gp_type,
        template<typename> class tfunctor>
class PotentialGradients
{
public:

    fp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z){
        tfunctor<fp_type> tf;
        return tf(t,x,y,z);
    }

    gp_type Grad_x(fp_type t,fp_type x,fp_type y, fp_type z)
    {
        mfem::future::dual<fp_type,gp_type> td; td.value=t; td.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> xd; xd.value=x; xd.gradient=(1.0);
        mfem::future::dual<fp_type,gp_type> yd; yd.value=y; yd.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> zd; zd.value=z; zd.gradient=(0.0);

        tfunctor< mfem::future::dual<fp_type,gp_type> > tf;
        mfem::future::dual<fp_type,gp_type> rez=tf(td,xd,yd,zd);
        return rez.gradient;
    }

    gp_type Grad_y(fp_type t,fp_type x,fp_type y, fp_type z)
    {
        mfem::future::dual<fp_type,gp_type> td; td.value=t; td.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> xd; xd.value=x; xd.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> yd; yd.value=y; yd.gradient=(1.0);
        mfem::future::dual<fp_type,gp_type> zd; zd.value=z; zd.gradient=(0.0);

        tfunctor< mfem::future::dual<fp_type,gp_type> > tf;
        mfem::future::dual<fp_type,gp_type> rez=tf(td,xd,yd,zd);
        return rez.gradient;
    }

    gp_type Grad_z(fp_type t,fp_type x,fp_type y, fp_type z)
    {
        mfem::future::dual<fp_type,gp_type> td; td.value=t; td.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> xd; xd.value=x; xd.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> yd; yd.value=y; yd.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> zd; zd.value=z; zd.gradient=(1.0);

        tfunctor< mfem::future::dual<fp_type,gp_type> > tf;
        mfem::future::dual<fp_type,gp_type> rez=tf(td,xd,yd,zd);
        return rez.gradient;
    }

    gp_type Grad_t(fp_type t,fp_type x,fp_type y, fp_type z)
    {
        mfem::future::dual<fp_type,gp_type> td; td.value=t; td.gradient=(1.0);
        mfem::future::dual<fp_type,gp_type> xd; xd.value=x; xd.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> yd; yd.value=y; yd.gradient=(0.0);
        mfem::future::dual<fp_type,gp_type> zd; zd.value=z; zd.gradient=(0.0);

        tfunctor< mfem::future::dual<fp_type,gp_type> > tf;
        mfem::future::dual<fp_type,gp_type> rez=tf(td,xd,yd,zd);
        return rez.gradient;
    }
};

template<typename fp_type, typename gp_type,
          template<typename> typename V>
class ScalarLaplacian
{
public:
    gp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
    {
        typedef mfem::future::dual<fp_type,gp_type> fd;
        mfem::future::dual<fd,fd> dt; dt=0.0;
        mfem::future::dual<fd,fd> dx; dx=0.0;
        mfem::future::dual<fd,fd> dy; dy=0.0;
        mfem::future::dual<fd,fd> dz; dz=0.0;
        mfem::future::dual<fd,fd> rezxx,rezyy,rezzz;


        dt.value.value=t;
        dx.value.value=x;
        dy.value.value=y;
        dz.value.value=z;

        V<mfem::future::dual<fd,fd>> ff;

        dx.value.gradient=1.0; dx.gradient.value=1.0;
        rezxx=ff(dt,dx,dy,dz);
        dx.value.gradient=0.0; dx.gradient.value=0.0;
        dy.value.gradient=1.0; dy.gradient.value=1.0;
        rezyy=ff(dt,dx,dy,dz);
        dy.value.gradient=0.0; dy.gradient.value=0.0;
        dz.value.gradient=1.0; dz.gradient.value=1.0;
        rezzz=ff(dt,dx,dy,dz);
        dz.value.gradient=0.0; dz.gradient.value=0.0;


        return rezxx.gradient.gradient+rezyy.gradient.gradient+rezzz.gradient.gradient;
    }

};

///example potentials
template <typename fp_type>
class ExPotentialZ{
public:
    fp_type operator()(fp_type t,fp_type x,fp_type y,fp_type z)
    {
        return t*cos(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
    }
};

template <typename fp_type>
class ExPotentialY{
public:
    fp_type  operator()(fp_type t,fp_type x,fp_type y,fp_type z)
    {
        return t*sin(M_PI*x)*sin(M_PI*y)*cos(M_PI*z);
    }
};

template <typename fp_type>
class ExPotentialX{
public:
    fp_type operator()(fp_type t,fp_type x,fp_type y,fp_type z)
    {
        return t*sin(M_PI*x)*cos(M_PI*y)*sin(M_PI*z);
    }
};



template< template<typename> typename PotentialX=ExPotentialX,
          template<typename> typename PotentialY=ExPotentialY,
          template<typename> typename PotentialZ=ExPotentialZ>
class ADDivFree3DVelocity:public VectorCoefficient
{
public:
    ADDivFree3DVelocity():VectorCoefficient(3){
        vlc.reset(new VectorLaplacianCoeff());
        tdv.reset(new TimeDerivativeCoeff());
    }

    virtual void Eval(Vector &V, ElementTransformation &T,
                      const IntegrationPoint &ip) override
    {
        Vector x;
        T.Transform(ip, x);
        real_t t=VectorCoefficient::GetTime();

        V[0]=vx(t,x[0],x[1],x[2]);
        V[1]=vy(t,x[0],x[1],x[2]);
        V[2]=vz(t,x[0],x[1],x[2]);
    }


    VectorCoefficient* VectorLaplacian(){return vlc.get();};
    VectorCoefficient* TimeDerivative() {return tdv.get();};

private:

    /// Ax,Ay,Az are template functors imlementing the vector potential
    template<typename fp_type, typename gp_type,
             template<typename> class Ax,
             template<typename> class Ay,
             template<typename> class Az>
    class VelocityX
    {
    public:
        gp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
        {
            PotentialGradients<fp_type,gp_type,Ay> poty;
            PotentialGradients<fp_type,gp_type,Az> potz;
            return potz.Grad_y(t,x,y,z)-poty.Grad_z(t,x,y,z);
        }
    };

    template<typename fp_type, typename gp_type,
             template<typename> class Ax,
             template<typename> class Ay,
             template<typename> class Az>
    class VelocityY
    {
    public:
        gp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
        {
            PotentialGradients<fp_type,gp_type,Ax> potx;
            PotentialGradients<fp_type,gp_type,Az> potz;
            return potx.Grad_z(t,x,y,z)-potz.Grad_x(t,x,y,z);
        }
    };

    template<typename fp_type, typename gp_type,
             template<typename> class Ax,
             template<typename> class Ay,
             template<typename> class Az>
    class VelocityZ
    {
    public:
        gp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
        {
            PotentialGradients<fp_type,gp_type,Ax> potx;
            PotentialGradients<fp_type,gp_type,Ay> poty;
            return poty.Grad_x(t,x,y,z)-potx.Grad_y(t,x,y,z);
        }

    };

    //templated Velocities
    template <typename fp_type>
    class TVelocityX{
    public:
        VelocityX<fp_type,fp_type,PotentialX,PotentialY,PotentialZ> ff;

        fp_type  operator()(fp_type t,fp_type x,fp_type y,fp_type z)
        {
            return ff(t,x,y,z);
        }
    };

    template <typename fp_type>
    class TVelocityY{
    public:
        VelocityY<fp_type,fp_type,PotentialX,PotentialY,PotentialZ> ff;

        fp_type  operator()(fp_type t,fp_type x,fp_type y,fp_type z)
        {
            return ff(t,x,y,z);
        }
    };

    template <typename fp_type>
    class TVelocityZ{
    public:
        VelocityZ<fp_type,fp_type,PotentialX,PotentialY,PotentialZ> ff;

        fp_type  operator()(fp_type t,fp_type x,fp_type y,fp_type z)
        {
            return ff(t,x,y,z);
        }
    };

    TVelocityX<real_t> vx;
    TVelocityY<real_t> vy;
    TVelocityZ<real_t> vz;

    class TimeDerivativeCoeff:public VectorCoefficient
    {
    public:
        TimeDerivativeCoeff():VectorCoefficient(3)
        {

        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip) override
        {
            Vector x;
            T.Transform(ip, x);
            real_t t=VectorCoefficient::GetTime();

            V[0]=vlx(t,x[0],x[1],x[2]);
            V[1]=vly(t,x[0],x[1],x[2]);
            V[2]=vlz(t,x[0],x[1],x[2]);
        }

        PotentialGradients<real_t,real_t,TVelocityX> vlx;
        PotentialGradients<real_t,real_t,TVelocityY> vly;
        PotentialGradients<real_t,real_t,TVelocityZ> vlz;

    };

    class VectorLaplacianCoeff:public VectorCoefficient
    {
    public:
        VectorLaplacianCoeff():VectorCoefficient(3)
        {

        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip) override
        {
            Vector x;
            T.Transform(ip, x);
            real_t t=VectorCoefficient::GetTime();

            V[0]=vlx(t,x[0],x[1],x[2]);
            V[1]=vly(t,x[0],x[1],x[2]);
            V[2]=vlz(t,x[0],x[1],x[2]);
        }

        ScalarLaplacian<real_t,real_t,TVelocityX> vlx;
        ScalarLaplacian<real_t,real_t,TVelocityY> vly;
        ScalarLaplacian<real_t,real_t,TVelocityZ> vlz;
    };

    std::unique_ptr<VectorLaplacianCoeff> vlc;
    std::unique_ptr<TimeDerivativeCoeff> tdv;
};

template <typename fp_type>
class ExPotential{
public:
    fp_type operator()(fp_type t,fp_type x,fp_type y)
    {
        return t*sin(M_PI*x)*sin(M_PI*y);
    }
};



template<template<typename> typename Potential=ExPotential>
class ADDivFree2DVelocity:public VectorCoefficient
{
public:
    ADDivFree2DVelocity():VectorCoefficient(2){
        vlc.reset(new VectorLaplacianCoeff());
        tdv.reset(new TimeDerivativeCoeff());
    }

    virtual void Eval(Vector &V, ElementTransformation &T,
                      const IntegrationPoint &ip) override
    {
        V.SetSize(2);
        Vector x(2); x=0.0;
        T.Transform(ip, x);
        real_t t=VectorCoefficient::GetTime();
        V[0]=vx(t,x[0],x[1],0.0);
        V[1]=vy(t,x[0],x[1],0.0);
    }


    VectorCoefficient* VectorLaplacian(){return vlc.get();};
    VectorCoefficient* TimeDerivative() {return tdv.get();};

private:
    template <typename fp_type>
    class PotentialZ{
    public:
        fp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
        {
            return pot(t,x,y);
        }

        Potential<fp_type> pot;
    };

    template<typename fp_type, typename gp_type,
             template<typename> class Az>
    class VelocityX
    {
    public:
        gp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
        {
            PotentialGradients<fp_type,gp_type,Az> potz;
            return potz.Grad_y(t,x,y,z);
        }
    };

    template<typename fp_type, typename gp_type,
             template<typename> class Az>
    class VelocityY
    {
    public:
        gp_type operator()(fp_type t,fp_type x,fp_type y, fp_type z)
        {
            PotentialGradients<fp_type,gp_type,Az> potz;
            return -potz.Grad_x(t,x,y,z);
        }
    };


    //templated Velocities
    template <typename fp_type>
    class TVelocityX{
    public:
        VelocityX<fp_type,fp_type,PotentialZ> ff;

        fp_type  operator()(fp_type t,fp_type x,fp_type y,fp_type z)
        {
            return ff(t,x,y,z);
        }
    };

    template <typename fp_type>
    class TVelocityY{
    public:
        VelocityY<fp_type,fp_type,PotentialZ> ff;

        fp_type  operator()(fp_type t,fp_type x,fp_type y,fp_type z)
        {
            return ff(t,x,y,z);
        }
    };

    TVelocityX<real_t> vx;
    TVelocityY<real_t> vy;

    class VectorLaplacianCoeff:public VectorCoefficient
    {
    public:
        VectorLaplacianCoeff():VectorCoefficient(2)
        {

        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip) override
        {
            V.SetSize(2);
            Vector x(2); x=0.0;
            T.Transform(ip, x);
            real_t t=VectorCoefficient::GetTime();

            V[0]=vlx(t,x[0],x[1],0.0);
            V[1]=vly(t,x[0],x[1],0.0);
        }

        ScalarLaplacian<real_t,real_t,TVelocityX> vlx;
        ScalarLaplacian<real_t,real_t,TVelocityY> vly;
    };

    class TimeDerivativeCoeff:public VectorCoefficient
    {
    public:
        TimeDerivativeCoeff():VectorCoefficient(2)
        {

        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip) override
        {
            V.SetSize(2);
            Vector x(2); x=0.0;
            T.Transform(ip, x);
            real_t t=VectorCoefficient::GetTime();

            V[0]=vlx.Grad_t(t,x[0],x[1],0.0);
            V[1]=vly.Grad_t(t,x[0],x[1],0.0);
        }

        PotentialGradients<real_t,real_t,TVelocityX> vlx;
        PotentialGradients<real_t,real_t,TVelocityY> vly;

    };

    std::unique_ptr<VectorLaplacianCoeff> vlc;
    std::unique_ptr<TimeDerivativeCoeff> tdv;
};

template<template<typename> typename SPotential>
class ADScalar2DCoeff:public Coefficient
{
public:
    ADScalar2DCoeff()
    {
        gradc.reset(new GradCoeff());
        laplc.reset(new LaplCoeff());
    }

    virtual
    ~ADScalar2DCoeff()
    {

    }

    virtual
    real_t  Eval(ElementTransformation &T,
                 const IntegrationPoint &ip) override
    {

        Vector x(2); x=0.0;
        T.Transform(ip, x);
        real_t t=Coefficient::GetTime();
        return func(t,x[0],x[1]);
    }

    VectorCoefficient* GetGradient(){return gradc.get();}
    Coefficient* GetLaplacian(){return laplc.get();}

private:

    SPotential<real_t> func;

    template<typename fp_type>
    class Spotential3D{
    public:
        fp_type operator()(fp_type t, fp_type x,fp_type y, fp_type z)
        {
           return lfunc(t,x,y);
        }

        SPotential<fp_type> lfunc;
    };

    class GradCoeff:public VectorCoefficient
    {
    public:
        GradCoeff():VectorCoefficient(2)
        {

        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip) override
        {
            V.SetSize(2);
            Vector x(2); x=0.0;
            T.Transform(ip, x);
            real_t t=VectorCoefficient::GetTime();

            V[0]=pg.Grad_x(t,x[0],x[1],0.0);
            V[1]=pg.Grad_y(t,x[0],x[1],0.0);
        }


        PotentialGradients<real_t,real_t,Spotential3D> pg;
    };

    class LaplCoeff:public Coefficient
    {
    public:
        virtual
        real_t  Eval(ElementTransformation &T,
                     const IntegrationPoint &ip) override
        {

            Vector x(2); x=0.0;
            T.Transform(ip, x);
            real_t t=Coefficient::GetTime();
            return func(t,x[0],x[1],0.0);
        }

        ScalarLaplacian<real_t,real_t,Spotential3D> func;
    };


    std::unique_ptr<GradCoeff> gradc;
    std::unique_ptr<LaplCoeff> laplc;

};

template<template<typename> typename SPotential>
class ADScalar3DCoeff:public Coefficient
{
public:
    ADScalar3DCoeff()
    {
        gradc.reset(new GradCoeff());
        laplc.reset(new LaplCoeff());
    }

    virtual
    ~ADScalar3DCoeff()
    {

    }

    virtual
    real_t  Eval(ElementTransformation &T,
                 const IntegrationPoint &ip) override
    {

        Vector x(3); x=0.0;
        T.Transform(ip, x);
        real_t t=Coefficient::GetTime();
        return func(t,x[0],x[1],x[2]);
    }

    VectorCoefficient* GetGradient(){return gradc.get();}
    Coefficient* GetLaplacian(){return laplc.get();}

private:

    SPotential<real_t> func;

    class GradCoeff:public VectorCoefficient
    {
    public:
        GradCoeff():VectorCoefficient(2)
        {

        }

        virtual void Eval(Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip) override
        {
            Vector x(3); x=0.0;
            T.Transform(ip, x);
            real_t t=VectorCoefficient::GetTime();

            V[0]=pg.Grad_x(t,x[0],x[1],x[2]);
            V[1]=pg.Grad_y(t,x[0],x[1],x[2]);
            V[2]=pg.Grad_z(t,x[0],x[1],x[2]);
        }

        PotentialGradients<real_t,real_t,SPotential> pg;
    };


    class LaplCoeff:public Coefficient
    {
    public:
        virtual
        real_t  Eval(ElementTransformation &T,
                     const IntegrationPoint &ip) override
        {

            Vector x(3); x=0.0;
            T.Transform(ip, x);
            real_t t=Coefficient::GetTime();
            return func(t,x[0],x[1],x[2]);
        }

        ScalarLaplacian<real_t,real_t,SPotential> func;
    };


    std::unique_ptr<GradCoeff> gradc;
    std::unique_ptr<LaplCoeff> laplc;


};



}
#endif // MMS_COEF_HPP
