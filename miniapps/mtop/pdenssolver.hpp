#ifndef PDENSSOLVER_H
#define PDENSSOLVER_H

#include <mfem.hpp>
#include <map>
#include <vector>
#include <tuple>

namespace mfem {

class ParFilter{
public:
    ParFilter(){}
    virtual ~ParFilter(){}

    //input and output must be a true-dof vector.
    virtual void FFilter(mfem::Vector& in, mfem::Vector& out)=0;
    //output must be a true-dof vector.
    virtual void FFilter(mfem::Coefficient& in, mfem::Vector& out)=0;
    //input and output must be a true-dof vector.
    virtual void RFilter(mfem::Vector& in, mfem::Vector& out)=0;
};


class PDEFilter: public ParFilter{
public:
    //The input parameter r is the support radius of a cone filter.
    //The diffusion parameter is obtained as r^2/((2*sqrt(3))^2).
    //For details see:
    //Lazarov, B. S. & Sigmund, O.
    //Filters in topology optimization based on Helmholtz-type differential equations
    //International Journal for Numerical Methods in Engineering, 2011, 86, 765-781
    //int order is utilized for the RHS of the filter
    PDEFilter(mfem::ParMesh* mesh,  mfem::ParFiniteElementSpace *pfin,  //input field
                                    mfem::ParFiniteElementSpace *pfout, //filtered field
                                    double r=0.0);


    virtual ~PDEFilter();

    //in -true-dof vector derived from pfin
    //out -true-dof vector derived from pfout
    virtual void FFilter(mfem::Vector& in, mfem::Vector& out);
    virtual void FFilter(mfem::Coefficient& in, mfem::Vector& out);

    //in  -gradients true-dof vector derived from pfout
    //out -gradients true-dof vector derived from pfin
    virtual void RFilter(mfem::Vector& in, mfem::Vector& out);

    void ClearLenScale(); //clear all length scales,i.e., set them to zero
    void SetDiffusion(double a); //set directly the default diffusion parameter
    void SetDiffusion(int mark, double a);
    void SetLenScale(double r); //set the default length scale
    void SetLenScale(int mark, double r); //set length scale for region with a specified mark

private:
    //define coefficients
    double default_diffusion;
    std::map<int,double> mcmap; //<mark,diffusion coefficient>

    struct{
       mfem::ParMesh* mesh;
       mfem::ParFiniteElementSpace *pfin;
       mfem::ParFiniteElementSpace *pfout;

       mfem::Coefficient* dc; //diffusion coefficient
       mfem::Coefficient* mc; //mass coefficient

       mfem::ParBilinearForm *a;
       mfem::ParLinearForm *bl;
       mfem::ParLinearForm *rl;

       mfem::HypreParMatrix *A;//assembled matrix
       mfem::Vector B;

       mfem::ParGridFunction gfin;//input density field
       mfem::ParGridFunction gfft;//filtered density field

       mfem::HypreSolver *prec;
       mfem::HyprePCG    *solv;
    }mfem_solver;

    bool realloc_required;

    void Allocate();

};




}





#endif
