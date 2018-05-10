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
#ifndef LAGHOS_RAJA_SOLVERS
#define LAGHOS_RAJA_SOLVERS

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem {

  // ***************************************************************************
  /*extern "C" void d_Mult(const RajaVector &b, RajaVector &x,
                         const int N,
                         const RajaOperator *oper,
                         RajaSolverOperator *prec,
                         const bool iterative_mode,
                         RajaVector &r,
                         RajaVector &d,
                         RajaVector &z,
                         double &final_norm);*/


  // ***************************************************************************
  class RajaIterativeSolver : public RajaSolverOperator{
#ifdef MFEM_USE_MPI
  private:
    int dot_prod_type; // 0 - local, 1 - global over 'comm'
    MPI_Comm comm;
#endif
  protected:
    const RajaOperator *oper;
    RajaSolverOperator *prec;
    int max_iter, print_level;
    double rel_tol, abs_tol;
    // stats
    mutable int final_iter, converged;
    mutable double final_norm;
    double Dot(const RajaVector &x,
               const RajaVector &y) const {
#ifndef MFEM_USE_MPI
      return (x * y);
#else
      if (dot_prod_type == 0){
        return (x * y);
      }
      double local_dot = (x * y);
      double global_dot;
      MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      return global_dot;
#endif
    }
    double Norm(const RajaVector &x) const { return sqrt(Dot(x, x)); }
  public:
    RajaIterativeSolver(): RajaSolverOperator(0, true){
      oper = NULL;
      prec = NULL;
      max_iter = 10;
      print_level = -1;
      rel_tol = abs_tol = 0.0;
#ifdef MFEM_USE_MPI
      dot_prod_type = 0;
#endif
    }
    
#ifdef MFEM_USE_MPI
  RajaIterativeSolver(MPI_Comm _comm)
    : RajaSolverOperator(0, true)
  {
    oper = NULL;
    prec = NULL;
    max_iter = 10;
    print_level = -1;
    rel_tol = abs_tol = 0.0;
    dot_prod_type = 1;
    comm = _comm;
  }
#endif

    void SetRelTol(double rtol) { rel_tol = rtol; }
    void SetAbsTol(double atol) { abs_tol = atol; }
    void SetMaxIter(int max_it) { max_iter = max_it; }
    void SetPrintLevel(int print_lvl){
#ifndef MFEM_USE_MPI
      print_level = print_lvl;
#else
      if (dot_prod_type == 0)
      {
        print_level = print_lvl;
      }
      else
      {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0)
        {
          print_level = print_lvl;
        }
      }
#endif
    }
    int GetNumIterations() const { return final_iter; }
    int GetConverged() const { return converged; }
    double GetFinalNorm() const { return final_norm; }
    /// This should be called before SetOperator
    virtual void SetPreconditioner(RajaSolverOperator &pr){
      prec = &pr;
      prec->iterative_mode = false;
    }
    /// Also calls SetOperator for the preconditioner if there is one
    virtual void SetOperator(const RajaOperator &op){
      oper = &op;
      height = op.Height();
      width = op.Width();
      if (prec)
      {
        prec->SetOperator(*oper);
      }
    }
  };

  // ***************************************************************************
  // Conjugate gradient method
  // ***************************************************************************
  class RajaCGSolver : public RajaIterativeSolver{
  protected:
    mutable RajaVector r, d, z;
    void UpdateVectors()   {
      r.SetSize(width);
      d.SetSize(width);
      z.SetSize(width);
    }
  public:
    RajaCGSolver() { }
#ifdef MFEM_USE_MPI
    RajaCGSolver(MPI_Comm _comm) : RajaIterativeSolver(_comm) { }
#endif
    virtual void SetOperator(const RajaOperator &op) {
      RajaIterativeSolver::SetOperator(op);
      UpdateVectors();
    }
    void h_Mult(const RajaVector &b, RajaVector &x) const ;
    virtual void Mult(const RajaVector &b, RajaVector &x) const {
      if (rconfig::Get().Dcg()){
#ifdef __NVCC__
        //d_Mult(b,x,b.Size(),oper,prec,iterative_mode,r,d,z,final_norm);
#endif
      }
      else h_Mult(b,x);
    }
  };

} // mfem

#endif // LAGHOS_RAJA_SOLVERS
