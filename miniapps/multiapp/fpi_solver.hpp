// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#ifndef MFEM_FPISOLVER_HPP
#define MFEM_FPISOLVER_HPP

#include "mfem.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <type_traits> // std::is_base_of
#include <experimental/type_traits>
#include <utility>


namespace mfem
{

using namespace std;

/**
 * @brief A class to handle fixed point iteration relaxation methods
 *
 * This class provides a base for implementing various relaxation strategies
 * for fixed point iteration solvers.
 */
class FPIRelaxation
{
#ifdef MFEM_USE_MPI    
private:
   MPI_Comm comm = MPI_COMM_NULL;
#endif 

protected:
   const Operator *op = nullptr; ///< Operator being solved

public:
    FPIRelaxation() = default;

#ifdef MFEM_USE_MPI    
    FPIRelaxation(MPI_Comm comm_){comm = comm_;}
    void SetComm(MPI_Comm comm_) {comm = comm_;}
#endif

    /// @brief Set the operator for the relaxation method.
    virtual void Init(const Operator &op_){op = &op_;}

    /// @brief Compute the relaxation factor for Fixed Point Iteration.
    /// @param state Current state vector
    /// @param residual Current residual vector
    /// @param res_norm Norm of the current residual vector
    /// @param rfactor Current relaxation factor
    /// @return The computed relaxation factor
    virtual real_t Eval(const Vector &state, const Vector &residual, real_t res_norm, real_t rfactor)
    {
        return rfactor; // Default implementation returns the fixed factor
    };

    real_t Dot(const Vector &x, const Vector &y) const
    {
    #ifndef MFEM_USE_MPI
        return (x * y);
    #else
        return InnerProduct(comm, x, y);
    #endif
    }    
};

/**
 * @brief A class to implement Aitken relaxation for Fixed Point Iteration
 * to accelerate the convergence of fixed point iterations.
 */
class AitkenRelaxation : public FPIRelaxation
{
protected:
    Vector rold; // Old residual vector
    real_t rold_norm = 0.0; ///< Norm of the old residual vector
    
public:
    AitkenRelaxation() = default;

#ifdef MFEM_USE_MPI
    AitkenRelaxation(MPI_Comm comm_) : FPIRelaxation(comm_){}
#endif

    /// @brief Set the operator for the relaxation method.
    void Init(const Operator &op_) override
    {
        FPIRelaxation::Init(op_);
        rold.SetSize(op->Width()); // Initialize the old residual vector size
        rold = 0.0;      // Initialize the old residual vector to zero
        rold_norm = 0.0; // Initialize the old residual norm
    }

    /// @brief Compute the Aitken relaxation factor for Fixed Point Iteration.
    /// @param state Current state vector
    /// @param residual Current residual vector
    /// @param res_norm Norm of the current residual vector
    /// @param rfactor Current relaxation factor
    /// @return The computed relaxation factor
    real_t Eval(const Vector &state, const Vector &residual, real_t res_norm, real_t rfactor) override
    {
        real_t num   = Dot(rold, residual) - (rold_norm * rold_norm);
        real_t denom = rold.DistanceSquaredTo(residual);

        rold_norm = res_norm; // Update the old residual norm
        rold = residual;      // Update the old residual vector
        return -rfactor * num / denom;
    };
};

/**
 * @brief A class to implement steepest descent relaxation for Fixed Point Iteration
 * to accelerate the convergence of fixed point iterations.
 */
class SteepestDescentRelaxation : public FPIRelaxation
{
protected:
    Vector z;

public:
    SteepestDescentRelaxation() = default;

#ifdef MFEM_USE_MPI
    SteepestDescentRelaxation(MPI_Comm comm_) : FPIRelaxation(comm_){}
#endif

    /// @brief Set the operator for the relaxation method.
    void Init(const Operator &op_) override
    {
        FPIRelaxation::Init(op_);
        z.SetSize(op->Width()); // Initialize the old residual vector size
        z = 0.0;      // Initialize the old residual vector to zero
    }

    /// @brief Compute the steepest descent relaxation factor for Fixed Point Iteration.
    /// @param state Current state vector
    /// @param residual Current residual vector
    /// @param res_norm Norm of the current residual vector
    /// @param rfactor Current relaxation factor
    /// @return The computed relaxation factor
    real_t Eval(const Vector &state, const Vector &residual, real_t res_norm, real_t rfactor) override
    {
        MFEM_VERIFY(op,"Operator not set; set using Init(Operator&)")

        Operator *J = &op->GetGradient(state);
        real_t num = res_norm * res_norm;
        J->Mult(residual, z); // rold = F'(x) * rnew;
        real_t denom = Dot(z, residual);
        return num/denom;
    };
};


/// Fixed point iteration solver: x <- f(x)
class FPISolver : public IterativeSolver
{
protected:
   mutable Vector r, z;
   mutable real_t rfactor = 1.0;

   FPIRelaxation *relax_method = nullptr; ///< Relaxation strategy for FPI
   bool relaxation_owned = false;
 
   void UpdateVectors()
   {
        r.SetSize(width);
        z.SetSize(width);
    }
 
public:
   FPISolver() { }
 
#ifdef MFEM_USE_MPI
   FPISolver(MPI_Comm comm_) : IterativeSolver(comm_) { }
#endif
 
   virtual void SetOperator(const Operator &op)
   { 
        IterativeSolver::SetOperator(op); 
        UpdateVectors();
        if(!relax_method)
        {
            relax_method = new FPIRelaxation(); // Default relaxation strategy
            relaxation_owned = true;
        }
        relax_method->Init(*oper); 
   }

   void SetRelaxation(real_t r, FPIRelaxation *relaxation = nullptr)
   {
        rfactor = r;

        if (relaxation_owned && relaxation)
        {
            delete relax_method; // Delete the previous relaxation strategy
            relax_method = relaxation; // Set the new relaxation strategy
            relaxation_owned = false;
        }
        else if (relaxation)
        {
            relax_method = relaxation;
        }
#ifdef MFEM_USE_MPI    
        relax_method->SetComm(this->GetComm());
#endif
   }
    
   /// Iterative solution of the (non)linear system using Fixed Point Iteration
   void Mult(const Vector &b, Vector &x) const override
   {
        int i;
        real_t factor = rfactor;
        
        // FPI with fixed number of iterations and given
        // initial guess
        if (rel_tol == 0.0 && iterative_mode)
        {
            if (factor == 1.0)
            {
                for (i = 0; i < max_iter; i++)
                {
                    oper->Mult(x, z);  // z = F(x)
                    x = z;
                }
            }
            else
            {
                for (i = 0; i < max_iter; i++)
                {
                    oper->Mult(x, z);  // z = F(x)
                    subtract(z, x, r); // r = z - x
                    x.Add(factor, r);  // x = x + factor * r
                }
            }
            converged = true;
            final_iter = i;
            return;
        }

        real_t r0, nom, nom0, nomold = 1, cf;

        if (iterative_mode)
        {
            oper->Mult(x, z);  // z = F(x)
            subtract(z, x, r); // r = z - x
        }
        else
        {            
            x = 0.0;
            oper->Mult(x, r);  // r = F(x)
        }
        
        nom0 = nom = sqrt(Dot(r, r));
        initial_norm = nom0;

        if (print_options.iterations | print_options.first_and_last)
        {
            mfem::out << "   Iteration : " << setw(3) << right << 0 << "  ||Br|| = "
                        << nom << (print_options.first_and_last ? " ..." : "") << '\n';
        }
 
        r0 = std::max(nom*rel_tol, abs_tol);
        if (nom <= r0)
        {
            converged = true;
            final_iter = 0;
            final_norm = nom;
            return;
        }        

        // start iteration
        converged = false;
        final_iter = max_iter;
        for (i = 1; true; )
        {    
            oper->Mult(x, z);  // z = F(x)
            subtract(z, x, r); // r = z - x
            nom = sqrt(Dot(r, r));

            factor = relax_method->Eval(x,r,nom,factor);
            x.Add(factor, r); // x = x + factor * r
            
            cf = nom/nomold;
            nomold = nom;
        
            bool done = false;
            if (nom < r0)
            {
                converged = true;
                final_iter = i;
                done = true;
            }
        
            if (++i > max_iter)
            {
                done = true;
            }
        
            if (print_options.iterations || (done && print_options.first_and_last))
            {
                mfem::out << "   Iteration : " << setw(3) << right << (i-1)
                        << "  ||r|| = " << setw(11) << left << nom
                        << "\tConv. rate: " << cf << '\n';
            }
        
            if (done) { break; }
        }
 
        if (print_options.summary || (print_options.warnings && !converged))
        {
            const auto rf = pow (nom/nom0, 1.0/final_iter);
            mfem::out << "FPI: Number of iterations: " << final_iter << '\n'
                        << "Conv. rate: " << cf << '\n'
                        << "Average reduction factor: "<< rf << '\n';
        }
        if (print_options.warnings && !converged)
        {
            mfem::out << "FPI: No convergence!" << '\n';
        }
        
        final_norm = nom;
   }
};


} //mfem namespace


#endif
