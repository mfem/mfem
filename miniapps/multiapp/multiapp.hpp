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


#ifndef MFEM_MULTIAPP_HPP
#define MFEM_MULTIAPP_HPP

#include "mfem.hpp"

#include <memory>
#include <vector>
#include <type_traits> // std::is_base_of
#include <experimental/type_traits>
#include<utility>


namespace mfem
{

using namespace std;

/**
 * @brief A class to handle variables and transfers on shared boundaries
 * TODO: Add transfer maps using FindPoints and for time-interpolation
 */
class LinkedFields 
{

protected:

    using Pair = std::pair<ParGridFunction*, ParTransferMap*>;

    ParGridFunction &source;
    std::vector<Pair> destinations;

    int ndest;

public:

    LinkedFields(ParGridFunction &src, ParGridFunction &dst) : source(src)
    {
        destinations.push_back(std::make_pair(&dst, new ParTransferMap(source, dst)));
    }

    LinkedFields(ParGridFunction &src, ParGridFunction &dst, ParTransferMap &transfer_map) : source(src)
    {
        destinations.push_back(std::make_pair(&dst, &transfer_map));
    }
    
    void AddDestination(ParGridFunction &dst){
        destinations.push_back(std::make_pair(&dst, new ParTransferMap(source, dst)));
        ndest++;
    }

    void AddDestination(ParGridFunction &dst, ParTransferMap &transfer_map){
        destinations.push_back(std::make_pair(&dst, &transfer_map));
        ndest++;
    }

    // This should be virtual to use other transfer operations (e.g., findpts)
    virtual void Transfer(const Vector &vsrc, const int idest=-1)
    {
        source.SetFromTrueDofs(vsrc);
        Transfer(idest);
    }

    // Transfer from internally stored grid function to the destination
    virtual void Transfer(const int idest = -1)
    {
        if(idest >= 0)
        {
            ParGridFunction *destination = destinations[idest].first;
            ParTransferMap *transfer_map = destinations[idest].second;
            transfer_map->Transfer(source,*destination);
        }
        else
        {
            for (auto &destination : destinations) {
                destination.second->Transfer(source,*destination.first);
            }
        }        
    }

    virtual ~LinkedFields() {
        for (auto &dest : destinations) {
            if (dest.second) delete dest.second; // Delete the transfer map
        }
    }
};

/// Fixed point iteration solver: x <- f(x)
class FPISolver : public IterativeSolver
{
public:
    enum class Relaxation
    {
        FIXED,  ///< Fixed relaxation factor
        AITKEN, ///< Aitken relaxation method
        STEEPEST_DESCENT ///< Steepest descent relaxation method
    };

protected:
   mutable Vector r, z, rold;
   mutable real_t r_fac = 1.0;
   Relaxation type = Relaxation::FIXED;
   real_t (FPISolver::*compute_relaxation)(const real_t fac, const Vector &rn, const real_t rn_norm, Vector &ro, real_t ro_norm) const = nullptr;
 
   void UpdateVectors()
   {
        r.SetSize(width);
        z.SetSize(width);
        if (type != Relaxation::FIXED)
        {
            rold.SetSize(width);
            rold = 0.0;
        }
    }
 
public:
   FPISolver() { }
 
#ifdef MFEM_USE_MPI
   FPISolver(MPI_Comm comm_) : IterativeSolver(comm_) { }
#endif
 
   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); compute_relaxation = &FPISolver::FixedRelaxation; }

   
   void SetRelaxationFactor(real_t r, Relaxation relax_type = Relaxation::FIXED)
   {
        r_fac = r;
        type = relax_type;
        UpdateVectors();

        if (relax_type == Relaxation::AITKEN)
        {
            compute_relaxation = &FPISolver::AitkenRelaxation;
        }
        else if (relax_type == Relaxation::STEEPEST_DESCENT)
        {
            if (prec == nullptr)
            {
                mfem_error("Steepest descent relaxation requires a preconditioner (action of Jacobian) to be set first.");
            }
            compute_relaxation = &FPISolver::SteepestDescentRelaxation;
        }
        else // Relaxation::FIXED
        {
            compute_relaxation = &FPISolver::FixedRelaxation;
        }
   }
   
   real_t FixedRelaxation(const real_t fac, const Vector &rn, const real_t rn_norm, Vector &ro, real_t ro_norm) const { return fac; }

   real_t AitkenRelaxation(const real_t fac, const Vector &rn, const real_t rn_norm, Vector &ro, real_t ro_norm) const
   {
        real_t num   = Dot(ro, rn) - (ro_norm * ro_norm);
        real_t denom = ro.DistanceSquaredTo(rn);
        ro = rn; // Update here since old is not needed in general
        return -fac * num / denom;
   }

   real_t SteepestDescentRelaxation(const real_t fac, const Vector &rn, const real_t rn_norm, Vector &ro, real_t ro_norm) const
   {
        real_t num = rn_norm * rn_norm;
        prec->Mult(rn, ro); // rold = F'(x) * rnew; the preconditioner is used to apply the Jacobian
        real_t denom = Dot(ro, rn);
        ro = rn; // Update here since old is not needed in general
        return num/denom;
   }

 
   /// Iterative solution of the (non)linear system using Fixed Point Iteration
   virtual void Mult(const Vector &b, Vector &x) const 
   {
        int i;
        real_t fac = r_fac;
        
        // FPI with fixed number of iterations and given
        // initial guess
        if (rel_tol == 0.0 && iterative_mode)
        {
            if (fac == 1.0)
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
                    x.Add(fac, r); // x = x + fac * r
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

            fac = (this->*compute_relaxation)(fac,r,nom,rold,nomold);
            x.Add(fac, r); // x = x + fac * r
            
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



/** @brief This class is used to define the interface for applications and miniapps.
    It is used to define the Solve method which is used to solve the problem
    defined by the application or miniapp.
 */
class Application : public TimeDependentOperator
{

protected:

    int oper_index = -1;
    std::vector<LinkedFields*> linked_fields;

public:

    Application(int n=0) : TimeDependentOperator(n) {}

    virtual void Initialize() {
        mfem_error("Application::Initialize() is not overridden!");
    }

    virtual void Assemble() {
        mfem_error("Application::Assemble() is not overridden!");
    }

    virtual void Finalize() {
        mfem_error("Application::Finalize() is not overridden!");
    }

    /**
     * @brief Given a vector @a x, solve the problem defined by the application 
     * and return the solution in @a y.
     */
    virtual void Solve(const Vector &x, Vector &y) const {
        MFEM_ABORT("Not implemented for this Application.");
    };

    /**
     * @brief Apply the operator defined by the application to the vector @a x 
     * and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const override {
        MFEM_ABORT("Not implemented for this Application.");
    }

    virtual void UpdateOperator() {
        mfem_error("Application::UpdateOperator() is not overridden!");
    }
    
    virtual void PerformOperation(const int op, const Vector &x, Vector &y) {
        mfem_error("Application::PerformOperation() is not overridden!");
    }

    void AddLinkedFields(LinkedFields *field) {
        linked_fields.push_back(field);
    }

    virtual void Step(Vector &x, real_t &t, real_t &dt) {
        mfem_error("Application::Step() is not overridden!");
    }

    virtual void Transfer(Vector &x) {
        int n = linked_fields.size();
        if (n == 1)
        {
            linked_fields[0]->Transfer(x);
        }
        else
        {
            Transfer();
        }
    }

    virtual void Transfer() {
        int n = linked_fields.size();
        for (int i = 0; i < n; i++)
        {
            linked_fields[i]->Transfer();
        }
    }
};



template <typename App>
class AbstractApplication : public Application
{
private:

protected:

    /// @brief A type trait to check if the erased class has the functions Step, Mult, and Solve with the needed signatures.
    template<class T>
    using CheckStep = decltype(std::declval<T&>().Step(std::declval<Vector&>(),std::declval<real_t&>(),std::declval<real_t&>()));

    template<class T>
    using CheckMult = decltype(std::declval<T&>().Mult(std::declval<const Vector&>(),std::declval<Vector&>()));

    template<class T>
    using CheckSolve = decltype(std::declval<T&>().Solve(std::declval<const Vector&>(),std::declval<Vector&>()));
    

    // Define a template class 'check' to test for the existence of member functions
    template <typename C, template<typename> typename Func, typename R>
    class CheckForMemberFunction {
        template<typename T> static constexpr auto check(T*) -> typename std::is_same< Func<T>, R >::type;
        template<typename> static constexpr std::false_type check(...);
        typedef decltype(check<C>(0)) type;

    public:
        static constexpr bool value  = type::value;
    };    

    App *app; ///< Pointer to the application

public:
    

    constexpr bool HasMult(){return CheckForMemberFunction<App,CheckMult,void>::value;}
    constexpr bool HasSolve(){return CheckForMemberFunction<App,CheckSolve,void>::value;}
    constexpr bool HasStep(){return CheckForMemberFunction<App,CheckStep,void>::value;}

    AbstractApplication(App *app_) : app(app_) {}

    // void Initialize() override
    // {
    //     app->Initialize();
    // }

    // void Assemble() override
    // {
    //     app->Assemble();
    // }

    void Solve(const Vector &x, Vector &y) const override
    {
        if constexpr (CheckForMemberFunction<App,CheckSolve,void>::value)
        {
            app->Solve(x,y);
        }
        else
        {
            MFEM_ABORT("The AbstractApplication does not have the function, Solve.");
        }
    }

    void Mult(const Vector &x, Vector &y) const override
    {
        if constexpr (CheckForMemberFunction<App,CheckMult,void>::value)
        {
            app->Mult(x,y);
        }
        else
        {
            MFEM_ABORT("The AbstractApplication does not have the function, Mult.");            
        }
    }

    // void PerformOperation(const int op, const Vector &x, Vector &y) override
    // {
    //     app->PerformOperation(x,y);
    // }

    void Step(Vector &x, real_t &t, real_t &dt) override
    {
        if constexpr (CheckForMemberFunction<App,CheckStep,void>::value)
        {
            app->Step(x,t,dt);
        }
        else
        {
            MFEM_ABORT("The AbstractApplication does not have the function, Step.");
        }        
    }
};





/**
 * @brief A class to couple multiple applications together.
 */
class CoupledApplication : public Application
{

private:

    class SchwarzOperator : public TimeDependentOperator
    {
        protected:
            CoupledApplication *app;
            real_t dt= 0.0; ///< Time step size, used for time-dependent applications
            const Vector *xp = nullptr; ///< Pointer to the input vector, used in Mult() methods
        public:
            SchwarzOperator(CoupledApplication *app_) : TimeDependentOperator(app_->Height(),app_->Width()), app(app_){}
            void SetTime(real_t dt_) override { dt = dt_; }
            void SetState(const Vector *x){xp = x;}
    };
    class AdditiveSchwarzOperator : public SchwarzOperator
    {
        public:
            AdditiveSchwarzOperator(CoupledApplication *app_) : SchwarzOperator(app_){}
            void Mult(const Vector &x, Vector &y) const override { 
                y = x;
                app->AdditiveSchwarzImplicitSolve(dt,*xp,y);
            }
    };

    class AlternatingSchwarzOperator : public SchwarzOperator
    {
        public:
            AlternatingSchwarzOperator(CoupledApplication *app_) : SchwarzOperator(app_){}        
            void Mult(const Vector &x, Vector &y) const override { 
                y = x;
                app->AlternatingSchwarzImplicitSolve(dt,*xp,y);
            }
    };    

public:

    enum class Scheme
    {
        DECOUPLED,           ///< No coupling, solve each app independently
        MONOLITHIC,          ///< Solve all apps together
        ADDITIVE_SCHWARZ,    ///< Jacobi-type: solve each app and transfer data to the next after all apps
        ALTERNATING_SCHWARZ  ///< Gauss-Seidel-type: solve each app and transfer data to the next immediately
    };

protected:

    int napps = 0; ///< The number of applications
    Scheme scheme = Scheme::ADDITIVE_SCHWARZ; ///< Current coupling type
    
    std::vector<Application*> apps;
    SchwarzOperator *implicit_op = nullptr; ///< Coupled operator for the applications
    Solver *solver;
    real_t dt_ = 0.0; ///< Timestep used to compute u = u_0 + dt*dudt for transfer to apps in multistage methods

    mfem::Array<int> offsets;
    mutable Vector xt,kt;
    
public:

    /**
     * @brief Construct a new CoupledApplication object. The total number of applications
     * @param napp Total number of applications
     */
    template<typename... Args>
    CoupledApplication(const int napp, Args... args) : Application(args...) {
        apps.reserve(napp);
        offsets.Reserve(napp+1);
    }

    /**
     * @brief Construct a new CoupledApplication object. The total number of applications
     */
    CoupledApplication(const int napp) : Application() {
        apps.reserve(napp);
        offsets.Reserve(napp+1);
    }

    /**
     * @brief Construct a new CoupledApplication object for an abstract non/mfem application
     * or templated application, and derived class.
     */
    template <class AppType>
    CoupledApplication(const AppType &app) : Application() {
        AddApplication(app);
    }

    /**
     * @brief Add an application to the list of coupled application and return pointer to it.
     */
    template <class AppType>
    Application* AddApplication(AppType *app) {

        // Add operator to list of operators
        if constexpr(std::is_base_of<Application, AppType>::value) {
            apps.push_back(app);
        } else {
            apps.push_back(new AbstractApplication<AppType>(app));
        }
        napps++;

        // Update size of the coupled operator
        Application* app_ = apps.back();

        offsets.Append( app_->Width());
        this->width  += app_->Width();
        this->height += app_->Height();

        return app_;
    }

    Application* GetApplication(const int i) { return apps[i]; }

    virtual void SetCouplingScheme(Scheme scheme_) { scheme = scheme_; }
    Scheme GetCouplingScheme() const { return scheme; }

    void SetOffsets(Array<int> off_sets)
    {
        this->offsets = off_sets;
    }

    void SetSolver(Solver *solver_){
        solver = solver_;
    }

    void Initialize()
    {
        for (auto &app : apps)
        {
            app->Initialize();
        }
    }

    void Assemble()
    {
        for (auto &app : apps)
        {
            app->Assemble();
        }
    }

    void Finalize(bool do_apps = false)
    {
        if (do_apps)
        {
            for (auto &app : apps)
            {
                app->Finalize();
            }
        }

        
        int max_size = offsets.Max();

        if (scheme == Scheme::MONOLITHIC)
        {   // Set vector size for the full coupled operator
            xt.SetSize(this->Width());
        }
        else if (scheme == Scheme::ADDITIVE_SCHWARZ)
        {   // Set vector size for the the largest operator
            xt.SetSize(max_size);
            implicit_op = new AdditiveSchwarzOperator(this);
        }
        else if (scheme == Scheme::ALTERNATING_SCHWARZ)
        {   // Set vector size for the the largest operator
            xt.SetSize(max_size);
            implicit_op = new AlternatingSchwarzOperator(this);         
        }

        if(implicit_op && solver) solver->SetOperator(*implicit_op);

        offsets.Prepend(0);
        offsets.PartialSum(); // Compute block sizes for the coupled operator
    }

    /**
    * @brief Given a vector @a x, solve the problem defined by the application 
    * or miniapp and return the solution in @a y.
    */
    virtual void Solve(const Vector &x, Vector &y) const
    {
        for (auto &app : apps)
        {
            app->Solve(x, y);
        }
    }


    void SetTime(const real_t t) override
    {
        if (isExplicit()){
            // Compute dt based on the current time and the previous time
            // Used to compute u = u_0 + dt*dudt for transfer to apps in multistage methods
            // Note: this is only needed for explicit operators; ImplicitSolve passes dt directly
            this->dt_ = t - this->GetTime();
        }

        TimeDependentOperator::SetTime(t);
        for (auto &app : apps)
        {
            app->SetTime(t);
        }
    }

    /**
     * @brief Apply the operator defined by the application to the vector @a x 
     * and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const
    {
        BlockVector xb(x.GetData(), offsets);
        BlockVector yb(y.GetData(), offsets);

        if (scheme == Scheme::ADDITIVE_SCHWARZ)
        {   
            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &yi = yb.GetBlock(i);
                apps[i]->Mult(xi,yi);
            }

            // Transfer data. If app is steady-state and Mult solves F(x) = 0, then
            // dt_ = 0 and x is transferred to all applications. If app is explicit, time-dependent,
            // then Mult computes dxdt = F(x), updates xt = xi + dt*dxdt, and transfers xt to all applications.
            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &yi = yb.GetBlock(i);
                int isize  = xi.Size();

                xt.SetSize(isize);
                add(1.0,xi,dt_,yi,xt); ///< Compute the solution (xt = xi + dt*ki)
                apps[i]->Transfer(xt);  ///< Transfer the data to all applications
            }            
        }
        else if (scheme == Scheme::ALTERNATING_SCHWARZ)
        {
            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &yi = yb.GetBlock(i);
                apps[i]->Mult(xi,yi);

                int isize  = xi.Size();
                xt.SetSize(isize);
                add(1.0,xi,dt_,yi,xt); ///< Compute the solution (xt = xi + dt*ki)                
                apps[i]->Transfer(xt);  ///< Transfer the data to all applications
            }
        }
        else 
        {
            for (int i=0; i < napps; i++)
            {
                apps[i]->Mult(xb.GetBlock(i), yb.GetBlock(i));
            }
        }
    }

    /**
     * @brief Perform the specified operation for each application to the vector @a x
     */
    void PerformOperation(const int op, const Vector &x, Vector &y)
    {
        for (auto &app : apps) {
            app->PerformOperation(op, x, y);
        }
    }

    virtual void Transfer(Vector &x)
    {
        BlockVector xb(x.GetData(), offsets);
        for (int i=0; i < napps; i++)
        {
            apps[i]->Transfer(xb.GetBlock(i));
        }
    }

    /**
     * @brief Advance the time step for each application. If the coupling type is ONE_WAY,
     * the data from the current application is transferred to all others. This coupling
     * is forward and requires the applications to be ordered in the desired sequence.
     */
    void Step(Vector &x, real_t &t, real_t &dt) override
    {
        BlockVector xb(x.GetData(), offsets);

        if (scheme == Scheme::ADDITIVE_SCHWARZ)
        {
            // Step forward all operators
            // TODO: Add time-interpolation to enable different time step for each operator; 
            //       currently, all operators are stepped forward with the same time step
            for (int i=0; i < napps; i++)
            {
                real_t t0 = t;   ///< Store the current time
                real_t dt0 = dt; ///< Store the current time step

                Vector &xi = xb.GetBlock(i);
                apps[i]->Step(xi,t0,dt0); ///< Advance the time step for application
            }

            // Transfer the data after all applications have been stepped forward
            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                apps[i]->Transfer(xi);
            }
        }
        else if (scheme == Scheme::ALTERNATING_SCHWARZ)
        {
            for (int i=0; i < napps; i++)
            {
                real_t t0 = t;   ///< Store the current time
                real_t dt0 = dt; ///< Store the current time step

                Vector &xi = xb.GetBlock(i);
                apps[i]->Step(xi,t0,dt0); ///< Advance the time step for application
                apps[i]->Transfer(xi);    ///< Transfer the data to all applications
            }
        }
        else if (scheme == Scheme::DECOUPLED)
        {
            for (int i=0; i < napps; i++)
            {
                real_t t0 = t;   ///< Store the current time
                real_t dt0 = dt; ///< Store the current time step
                Vector &xi = xb.GetBlock(i);
                apps[i]->Step(xi,t0,dt0); ///< Advance the time step for application
            }
        }

    }

    /**
     * @brief Solves the implicit system. The PARTITIONED implicit solve is handled by/within 
     * each individual app. This function performs an implicit solve for the monolithic system
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ){

        if (scheme == Scheme::MONOLITHIC) ///< Solve all physics simultaneously
        {
            
        }
        else if (scheme == Scheme::ADDITIVE_SCHWARZ || scheme == Scheme::ALTERNATING_SCHWARZ) ///< Solve each physics independently
        {
            implicit_op->SetTime(dt); ///< Set the time step for the implicit operator
            implicit_op->SetState(&x); ///< Set the input vector for the implicit operator
            solver->Mult(kt,k); ///< Solve the implicit system using the solver (kt is unused for FPISolver)
        }
        else
        {
            BlockVector xb(x.GetData(), offsets);
            BlockVector kb(k.GetData(), offsets);

            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &ki = kb.GetBlock(i);
                apps[i]->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
            }
        }
    }

    /**
     * @brief Solves the implicit system app-by-app for the PARTITIONED coupling 
     */
    void AdditiveSchwarzImplicitSolve(const real_t dt, const Vector &x, Vector &k )
    {
        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        for (int i=0; i < napps; i++)
        {
            Vector &xi = xb.GetBlock(i);
            Vector &ki = kb.GetBlock(i);
            int isize  = xi.Size();

            xt.SetSize(isize);
            add(1.0,xi,dt,ki,xt); ///< Compute the solution (xt = xi + dt*ki)
            apps[i]->Transfer(xt);  ///< Transfer the data to all applications
        }

        for (int i=0; i < napps; i++)
        {
            Vector &xi = xb.GetBlock(i);
            Vector &ki = kb.GetBlock(i);
            apps[i]->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
        }
    }

    /**
     * @brief Solves the implicit system app-by-app for the PARTITIONED coupling
     * @note k on input is the initial gues for dudt, and is used to transfer solution
     */
    void AlternatingSchwarzImplicitSolve(const real_t dt, const Vector &x, Vector &k )
    {
        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);
       
        for (int i=0; i < napps; i++)
        {
            Vector &xi = xb.GetBlock(i);
            Vector &ki = kb.GetBlock(i);
            int isize  = xi.Size();

            xt.SetSize(isize);
            add(1.0,xi,dt,ki,xt); ///< Compute the solution (xt = xi + dt*ki)
            
            apps[i]->Transfer(xt);  ///< Transfer the data to all applications
            apps[i]->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
        }
    }

};

}


#endif
