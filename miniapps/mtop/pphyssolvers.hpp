#ifndef PPHYSSOLVERS_H
#define PPHYSSOLVERS_H

#include <mfem.hpp>
#include <map>
#include <vector>
#include <tuple>

namespace mfem {

//block form of the PPhysSolvers
class BPPhysSolvers
{
protected:


public:
    virtual void UpdateDesign(mfem::BlockVector& desf)=0;
    //solve for the the state field
    //for non-linear problems the initial guess is solf
    virtual void FSolve(mfem::BlockVector& solf)=0;
    //solve the adjoint problem
    //the method should be called always after
    virtual void ASolve(const mfem::BlockVector& solf, const mfem::BlockVector& arhs,
                        mfem::BlockVector& adjf)=0;
    //return adjf'*dr/ddesign
    virtual void GradD(const mfem::BlockVector& solf, const mfem::BlockVector& adjf,
                      mfem::BlockVector& grad)=0;

    const mfem::ParFiniteElementSpace* GetSFES(int k)=0; //return solver FES
    const mfem::ParFiniteElementSpace* GetDFES(int k)=0; //return design FES


};

class PPhysSolvers
{
protected:


public:
    virtual void UpdateDesign(mfem::Vector& desf)=0;
    //solve for the the state field
    //for non-linear problems the initial guess is solf
    virtual void FSolve(mfem::Vector& solf)=0;
    //solve the adjoint problem
    //the method should be called always after
    virtual void ASolve(const mfem::Vector& solf, const mfem::Vector& arhs,
                        mfem::Vector& adjf)=0;
    //return adjf'*dr/ddesign
    virtual void GradD(const mfem::Vector& solf, const mfem::Vector& adjf,
                      mfem::Vector& grad)=0;

    const mfem::ParFiniteElementSpace* GetSFES()=0; //return solver FES
    const mfem::ParFiniteElementSpace* GetDFES()=0; //return design FES


};


class ElastSolver3D:public PPhysSolvers
{
  public:
    ElastSolver(mfem::ParMesh* mesh,
                mfem::ParFiniteElementSpace* desfes);

    virtual ~ElastSolver() override;

    virtual void UpdateDesign(mfem::Vector& desf) override;

    virtual void FSolve(mfem::Vector& solf) override;

    virtual void ASolve(const mfem::Vector& solf,
                        const mfem::Vector& arhs,
                        mfem::Vector& adjf) override;

    virtual void GradD(const mfem::Vector& solf,
                       const mfem::Vector& adjf,
                       mfem::Vector& grad) override;

    const mfem::ParFiniteElementSpace* GetSFES() override;
    const mfem::ParFiniteElementSpace* GetDFES() override;

    void SetOrder(int order){ mfem_solv.order=order; }

    void SetMaterial(double lam, double mu) {
        mfem_solv.lam=lam;
        mfem_solv.mu=mu;
    }
    //solver BC


    //solver loads and BC
private:
    //BC map <mark, dof, val>
    std::vector< std::tuple<int,int,double> > bcmap;
    //load map <mark, pressure>
    std::map<int, double> lcmap;
    struct{
        int order; //order of the elements

        //Lame parameters
        double lam;
        double mu;

        mfem::Vector* pdesf;//pointer to the design field
        mfem::ParFiniteElementSpace* solfes;
        mfem::ParFiniteElementSpace* desfes;
    } mfem_solv;

};

}








#endif
