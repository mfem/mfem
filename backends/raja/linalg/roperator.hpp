// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef LAGHOS_RAJA_OPERATOR
#define LAGHOS_RAJA_OPERATOR

namespace mfem {

  // ***************************************************************************
  class RajaOperator : public rmemcpy {
  protected:
    int height;
    int width;
  public:
    explicit RajaOperator(int s = 0) { height = width = s; }
    RajaOperator(int h, int w) { height = h; width = w; }
    inline int Height() const { return height; }
    inline int Width() const { return width; }
    virtual void Mult(const RajaVector &x, RajaVector &y) const  { assert(false); };
    virtual void MultTranspose(const RajaVector &x, RajaVector &y) const { assert(false); }
    virtual const RajaOperator *GetProlongation() const { assert(false); return NULL; }
    virtual const RajaOperator *GetRestriction() const  { assert(false); return NULL; }
    virtual void RecoverFEMSolution(const RajaVector &X,
                                    const RajaVector &b,
                                    RajaVector &x){assert(false);}
  };


  // ***************************************************************************
  class RajaTimeDependentOperator : public RajaOperator{
  private:
    double t;
  public:
    explicit RajaTimeDependentOperator(int n = 0, double t_ = 0.0) : RajaOperator(n), t(t_) {}
    void SetTime(const double _t) { t = _t; }
  };
  
  // ***************************************************************************
  class RajaSolverOperator : public RajaOperator{
  public:
    bool iterative_mode;
    explicit RajaSolverOperator(int s = 0,
                                bool iter_mode = false) :
      RajaOperator(s),
      iterative_mode(iter_mode) { }
    virtual void SetOperator(const RajaOperator &op) = 0;
  };

  // ***************************************************************************
  class RajaRAPOperator : public RajaOperator{
  private:
    const RajaOperator &Rt;
    const RajaOperator &A;
    const RajaOperator &P;
    mutable RajaVector Px;
    mutable RajaVector APx;
  public:
    /// Construct the RAP operator given R^T, A and P.
    RajaRAPOperator(const RajaOperator &Rt_, const RajaOperator &A_, const RajaOperator &P_)
      : RajaOperator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }
    /// Operator application.
    void Mult(const RajaVector & x, RajaVector & y) const {
      push(SkyBlue);
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
      pop();
    }
    /// Application of the transpose.
    void MultTranspose(const RajaVector & x, RajaVector & y) const {
      push(SkyBlue);
      Rt.Mult(x, APx);
      A.MultTranspose(APx, Px);
      P.MultTranspose(Px, y);
      pop();
    }
  };
  
} // mfem

#endif // LAGHOS_RAJA_OPERATOR
