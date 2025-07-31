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

#include <algorithm>
#include <memory>
#include <vector>
#include <type_traits> // std::is_base_of
#include <experimental/type_traits>
#include <utility>


namespace mfem
{

using namespace std;

/* TODO LIST
    1) ADD DESTRUCTORS FOR ALL CLASSES
    2) Handle steady-state and time-dependent applications (DAEs)
    3) Add FindPts transfer maps
*/



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



class FPIRelaxation
{
#ifdef MFEM_USE_MPI    
private:
   MPI_Comm comm = MPI_COMM_NULL;
#endif 

public:
    FPIRelaxation() = default;

#ifdef MFEM_USE_MPI    
    FPIRelaxation(MPI_Comm comm_){comm = comm_;}
    void SetComm(MPI_Comm comm_) {comm = comm_;}
#endif

    /// @brief Compute the relaxation factor for Fixed Point Iteration.
    /// @param fac Current relaxation factor
    /// @param rnew Current residual vector
    /// @param rnew_norm Norm of the current residual vector
    /// @param rold Old residual vector
    /// @param rold_norm Norm of the old residual vector
    virtual real_t Eval(real_t fac, const Vector &rnew, real_t rnew_norm, Vector &rold, real_t rold_norm)
    {
        return fac; // Default implementation returns the fixed factor
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

class AitkenRelaxation : public FPIRelaxation
{
public:
    AitkenRelaxation() = default;

#ifdef MFEM_USE_MPI
    AitkenRelaxation(MPI_Comm comm_) : FPIRelaxation(comm_){}
#endif

    /// @brief Compute the Aitken relaxation factor for Fixed Point Iteration.
    virtual real_t Eval(real_t fac, const Vector &rnew, real_t rnew_norm, Vector &rold, real_t rold_norm) override
    {   
        real_t num   = Dot(rold, rnew) - (rold_norm * rold_norm);
        real_t denom = rold.DistanceSquaredTo(rnew);
        return -fac * num / denom;
    }
};

class SteepestDescentRelaxation : public FPIRelaxation
{
protected:
    const Operator *J = nullptr; ///< Preconditioner to apply the Jacobian

    /// @brief Set the preconditioner for the steepest descent relaxation.
    void SetJacobian(const Operator &jacobian) { J = &jacobian; }

    /// @brief Get the preconditioner for the steepest descent relaxation.
    const Operator *GetJacobian() const { return J; }

public:
    SteepestDescentRelaxation() = default;

#ifdef MFEM_USE_MPI
    SteepestDescentRelaxation(MPI_Comm comm_) : FPIRelaxation(comm_){}
#endif

    /// @brief Compute the steepest descent relaxation factor for Fixed Point Iteration.
    virtual real_t Eval(real_t fac, const Vector &rnew, real_t rnew_norm, Vector &rold, real_t rold_norm) override
    {
        if (!J){
            MFEM_ABORT("Jacobian operator not set.");
        }

        real_t num = rnew_norm * rnew_norm;
        J->Mult(rnew, rold); // rold = F'(x) * rnew;
        real_t denom = Dot(rold, rnew);
        return num/denom;
    }
};


/// Fixed point iteration solver: x <- f(x)
class FPISolver : public IterativeSolver
{
protected:
   mutable Vector r, z, rold;
   mutable real_t rfac = 1.0;

   FPIRelaxation *relax_method_ = nullptr; ///< Relaxation strategy for FPI
   bool relaxation_owned;
 
   void UpdateVectors()
   {
        r.SetSize(width);
        z.SetSize(width);
        rold.SetSize(width);
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
    relax_method_ = new FPIRelaxation(); // Default relaxation strategy
    relaxation_owned = true;
   }

   
   void SetRelaxation(real_t r, FPIRelaxation *relax_method = nullptr)
   {
        rfac = r;

        if (relaxation_owned && relax_method)
        {
            delete relax_method_; // Delete the previous relaxation strategy
            relax_method_ = relax_method; // Set the new relaxation strategy
            relaxation_owned = false;
        }
        else if (relax_method)
        {
            relax_method_ = relax_method;
        }
#ifdef MFEM_USE_MPI    
        relax_method_->SetComm(this->GetComm());
#endif
   }
    
   /// Iterative solution of the (non)linear system using Fixed Point Iteration
   void Mult(const Vector &b, Vector &x) const override
   {
        int i;
        real_t fac = rfac;
        
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

            fac = relax_method_->Eval(fac,r,nom,rold,nomold);
            x.Add(fac, r); // x = x + fac * r
            
            cf = nom/nomold;
            nomold = nom;
            rold = r; // Update rold to the current residual
        
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

    int oper_index = numeric_limits<int>::max();
    std::vector<LinkedFields*> linked_fields;
    // Track steady-state here. 

public:
    Application(int n=0) : TimeDependentOperator(n) {}

    Application(int h, int w) : TimeDependentOperator(h,w) {}

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

    virtual void Update() {
        mfem_error("Application::Update() is not overridden!");
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

    virtual void Transfer(const Vector &x) {
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
class AbstractOperator : public Application
{
protected:
 
    // Define a template class 'check' to test for the existence of member functions
    template <typename C>
    class CheckMember{
        private:        

        /// @brief A type trait to check if the erased class has the functions Step, Mult, and Solve
        /// with the needed signatures.
        template<class T>
        using Step = decltype(std::declval<T&>().Step(std::declval<Vector&>(),std::declval<real_t&>(),std::declval<real_t&>()));

        template<class T>
        using StepPtr = decltype(std::declval<T&>().Step(std::declval<const int>(),std::declval<real_t*>(),std::declval<real_t&>(),std::declval<real_t&>()));

        template<class T>
        using Mult = decltype(std::declval<T&>().Mult(std::declval<const Vector&>(),std::declval<Vector&>()));

        template<class T>
        using MultPtr = decltype(std::declval<T&>().Mult(std::declval<const int>(),std::declval<const real_t*>(),std::declval<const int>(),std::declval<real_t*>()));

        template<class T>
        using Solve = decltype(std::declval<T&>().Solve(std::declval<const Vector&>(),std::declval<Vector&>()));

        template<class T>
        using SolvePtr = decltype(std::declval<T&>().Solve(std::declval<const int>(),std::declval<const real_t*>(),std::declval<const int>(),std::declval<real_t*>()));        
        // ---------------------------------------------------------------------
        
        template <typename T, template<typename> typename Func, typename R>
        static constexpr auto Check(T*) -> typename std::is_same< Func<T>, R>::type;
        
        template <typename, template<typename> typename, typename >
        static constexpr std::false_type Check(...);

        // --- Check for the existence of the member functions
        typedef decltype(Check<C,Mult,void>(0)) Has_Mult;
        typedef decltype(Check<C,Step,void>(0)) Has_Step;
        typedef decltype(Check<C,Solve,void>(0)) Has_Solve;

        typedef decltype(Check<C,MultPtr,void>(0)) Has_MultPtr;
        typedef decltype(Check<C,StepPtr,void>(0)) Has_StepPtr;
        typedef decltype(Check<C,SolvePtr,void>(0)) Has_SolvePtr;

    public:
        static constexpr bool HasMult  = Has_Mult::value;
        static constexpr bool HasStep  = Has_Step::value;
        static constexpr bool HasSolve = Has_Solve::value;
        static constexpr bool HasMultPtr  = Has_MultPtr::value;
        static constexpr bool HasStepPtr  = Has_StepPtr::value;
        static constexpr bool HasSolvePtr = Has_SolvePtr::value;

    };    


    App *app; ///< Pointer to the application

public:

    constexpr bool HasMult(){return CheckMember<App>::HasMult;}
    constexpr bool HasStep(){return CheckMember<App>::HasStep;}
    constexpr bool HasSolve(){return CheckMember<App>::HasSolve;}


    AbstractOperator(App *app_) : app(app_) {}
    

    // void PerformOperation(const int op, const Vector &x, Vector &y) override
    // {
    //     app->PerformOperation(x,y);
    // }

    void Solve(const Vector &x, Vector &y) const override
    {
        if constexpr (CheckMember<App>::HasSolve)
        {
            app->Solve(x,y);
        }
        else if constexpr (CheckMember<App>::HasSolvePtr)
        {
            app->Solve(x.Size(), x.GetData(), y.Size(), y.GetData());
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, Solve.");
        }
    }

    void Mult(const Vector &x, Vector &y) const override
    {
        if constexpr (CheckMember<App>::HasMult)
        {
            app->Mult(x,y);
        }
        else if constexpr (CheckMember<App>::HasMultPtr)
        {
            app->Mult(x.Size(), x.GetData(), y.Size(), y.GetData());
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, Mult.");            
        }
    }

    void Step(Vector &x, real_t &t, real_t &dt) override
    {
        if constexpr (CheckMember<App>::HasStep)
        {
            app->Step(x,t,dt);
        }
        else if constexpr (CheckMember<App>::HasStepPtr)
        {
            app->Step(x.Size(), x.GetData(), t, dt);
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, Step.");
        }        
    }
};





/**
 * @brief A class to couple multiple applications together.
 */
class CoupledApplication : public Application
{
private:

    class ImplicitOperator : public TimeDependentOperator
    {
        protected:
            CoupledApplication *app;
            const Vector *u = nullptr; ///< Pointer to the input vector, used in Mult() methods
            bool is_linear = false;    ///< Flag to indicate if the operator is linear
            real_t dt= 0.0;            ///< Time step size, used for time-dependent applications
        public:
            ImplicitOperator(CoupledApplication *app_) : TimeDependentOperator(app_->Height(),app_->Width()), app(app_){}
            virtual void SetTimeStep(real_t dt_){ dt = dt_;}
            virtual void SetState(const Vector *u_){u = u_;}
            virtual void SetLinearity(const bool is_linear_){is_linear = is_linear_;}
            virtual bool IsLinear() const { return is_linear; }
    };
    class AdditiveSchwarzOperator : public ImplicitOperator
    {
        public:
            AdditiveSchwarzOperator(CoupledApplication *app_) : ImplicitOperator(app_){}
            void Mult(const Vector &ki, Vector &k) const override { 
                k = ki;
                app->AdditiveSchwarzImplicitSolve(dt,*u,k);
            }
    };
    class AlternatingSchwarzOperator : public ImplicitOperator
    {
        public:
            AlternatingSchwarzOperator(CoupledApplication *app_) : ImplicitOperator(app_){}        
            void Mult(const Vector &ki, Vector &k) const override { 
                k = ki;
                app->AlternatingSchwarzImplicitSolve(dt,*u,k);
            }
    };
    class MonolithicOperator : public ImplicitOperator
    {
        protected:
            mutable Vector upk;
            mutable future::FDJacobian grad;
            mutable real_t du = 1.0;
        public:
            MonolithicOperator(CoupledApplication *app_, real_t eps=1.0e-6) : ImplicitOperator(app_),
                                                                              upk(Width()), grad(*this,eps) {}
            /**
             * @brief Set the du coefficient in upk = du*u + dt*k, depending on the linearity
             *        of the operator. For linear problems, du = 0, for nonlinear problems, du = 1.
             */
            void SetLinearity(bool is_linear_) override {
                ImplicitOperator::SetLinearity(is_linear_);
                du = is_linear ? 0.0 : 1.0; // Set du to 0 for linear problems, 1 for nonlinear
            }
                                                                              /**
             * @brief Update the current state/point of linearization for the nonlinear operator. For
             *        Jacobian-free methods, i.e. J(k)*v = (F(k+eps*v) - F(k))/eps, this sets the F(k)
             */
            Operator& GetGradient(const Vector &k) const override {
                grad.Update(k);
                return const_cast<future::FDJacobian&>(grad);
            }

            /**
             * @brief Computes the action of the implicit, monolithic operator on vector x.
             *        ImplicitMult assumes ODE of the form F(u,k,t) = M k - g(u,t), G(u,t) = 0
             *        For fully implicit, monolithic solver, we take F(u+dt*k,k,t) = M k - g(u+dt*k,t)
             */
            void Mult(const Vector &k, Vector &y) const override {
                add(du,*u,dt,k,upk); // upk = u + dt*k
                app->Transfer(upk);
                app->ImplicitMult(upk,k,y); //y = f(upk,k,t)
            }
    };

public:

    enum class Scheme
    {
        NONE,                ///< No coupling, solve each app independently
        MONOLITHIC,          ///< Solve all apps simultaneously
        ADDITIVE_SCHWARZ,    ///< Jacobi-type: solve each app and transfer data to the next after all apps
        ALTERNATING_SCHWARZ  ///< Gauss-Seidel-type: solve each app and transfer data to the next immediately
    };

protected:

    int napps = 0; ///< The number of applications
    Scheme scheme = Scheme::ADDITIVE_SCHWARZ; ///< Current coupling type
    
    std::vector<Application*> apps;
    ImplicitOperator *implicit_op = nullptr; ///< Coupled operator for the applications
    Solver *solver = nullptr;   /// Solver for implicit solve (e.g., fixed point, linear, nonlinear)
    real_t dt_ = 0.0; ///< Timestep used to compute u = u_0 + dt*dudt for transfer to apps in multistage methods

    mfem::Array<int> offsets;   // Block offsets for each operator
    mutable Vector xt,b;

public:

    /**
     * @brief Construct a new CoupledApplication object. The total number of applications
     * @param napp Total number of applications
     */
    template<typename... Args>
    CoupledApplication(const int napp, Args... args) : Application(args...) {
        apps.reserve(napp);
        offsets.Reserve(napp+1);
        offsets.Prepend(0);
    }

    /**
     * @brief Construct a new CoupledApplication object. The total number of applications
     */
    CoupledApplication(const int napp) : Application() {
        apps.reserve(napp);
        offsets.Reserve(napp+1);
        offsets.Prepend(0);
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
        if constexpr(std::is_base_of<Application, AppType>::value)
        {
            apps.push_back(app);
        } 
        else
        {
            apps.push_back(new AbstractOperator<AppType>(app));
        }
        napps++;

        // Update size of the coupled operator and the block offsets
        Application* app_ = apps.back();

        int sum = offsets.Last();
        offsets.Append(sum + app_->Width());
        this->width  += app_->Width();
        this->height += app_->Height();

        return app_;
    }

    Application* GetApplication(const int i) { return apps[i]; }

    virtual void SetCouplingScheme(Scheme scheme_) { scheme = scheme_; }
    Scheme GetCouplingScheme() const { return scheme; }

    void SetOffsets(Array<int> off_sets) { this->offsets = off_sets; }

    void SetSolver(Solver *solver_){ solver = solver_; }

    ImplicitOperator* GetImplicitOperator(){ return implicit_op; }

    void Initialize(bool do_apps = false)
    {
        if (do_apps)
        {
            for (auto &app : apps)
            {
                app->Initialize();
            }
        }
    }


    ImplicitOperator *BuildImplicitOperator(){

        if (implicit_op) delete implicit_op;

        if (scheme == Scheme::MONOLITHIC)
        {
            implicit_op = new MonolithicOperator(this, 1e-6); // eps = 1e-6
        }
        else if (scheme == Scheme::ADDITIVE_SCHWARZ)
        {
            implicit_op = new AdditiveSchwarzOperator(this);
        }
        else if (scheme == Scheme::ALTERNATING_SCHWARZ)
        {
            implicit_op = new AlternatingSchwarzOperator(this);
        }

        return implicit_op;
    }

    void Assemble(bool do_apps = false)
    {
        if (do_apps)
        {
            for (auto &app : apps)
            {
                app->Assemble();
            }
        }

        BuildImplicitOperator();
        auto app = *std::max_element(apps.begin(), apps.end(), [](auto a, auto b) { return a->Width() < b->Width();});
        int max_size = app->Width(); // size of largest operator
                    
        if (scheme == Scheme::MONOLITHIC)
        {   // Set vector size for the full coupled operator
            xt.SetSize(this->Width());
        }
        else if (scheme == Scheme::ADDITIVE_SCHWARZ || scheme == Scheme::ALTERNATING_SCHWARZ)
        {   // Set vector size for the the largest operator
            xt.SetSize(max_size);
        }

        if(implicit_op && solver) solver->SetOperator(*implicit_op);
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
        
        if (implicit_op)
        {
            if (!solver){
                MFEM_ABORT("Solver for implicit operator not defined; set solver using SetSolver(&).");
            }

            if(implicit_op->IsLinear())
            {  // Use b to store the rhs for the linear implicit solve.
               // b=0 (or unused) for nonlinear Newton solves
                b.SetSize(Width());
            }
        }
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
     * @brief 
     * 
     */
    void PerformOperation(const int op, const Vector &x, Vector &y)
    {
        for (auto &app : apps) {
            app->PerformOperation(op, x, y);
        }
    }

    virtual void Transfer(const Vector &x)
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
        else if (scheme == Scheme::NONE)
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
    void ImplicitMult(const Vector &u, const Vector &k, Vector &v) const override {

        BlockVector ub(u.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);
        BlockVector vb(v.GetData(), offsets);

        for (int i=0; i < napps; i++)
        {
            Vector &ui = ub.GetBlock(i);
            Vector &ki = kb.GetBlock(i);
            Vector &vi = vb.GetBlock(i);
            apps[i]->ImplicitMult(ui,ki,vi); ///< Solve the implicit system for the application
        }
    }

    /**
     * @brief Solves the implicit system. The PARTITIONED implicit solve is handled by/within 
     * each individual app. This function performs an implicit solve for the monolithic system
     */
    void ExplicitMult(const Vector &u, Vector &v) const override {

        BlockVector ub(u.GetData(), offsets);
        BlockVector vb(v.GetData(), offsets);

        for (int i=0; i < napps; i++)
        {
            Vector &ui = ub.GetBlock(i);
            Vector &vi = vb.GetBlock(i);
            apps[i]->ExplicitMult(ui,vi); ///< Solve the implicit system for the application
        }
    }

    /**
     * @brief Solves the implicit system. The PARTITIONED implicit solve is handled by/within 
     * each individual app. This function performs an implicit solve for the monolithic system
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ){

        if (scheme == Scheme::MONOLITHIC) ///< Solve all physics simultaneously
        {
            implicit_op->SetTimeStep(dt); ///< Set the time step for the implicit operator
            implicit_op->SetState(&x); ///< Set the input vector for the implicit operator
            if (implicit_op->IsLinear()){
                ExplicitMult(x,b);  // Compute the rhs for the linear implicit system
            }
            solver->Mult(b,k);
        }
        else if (scheme == Scheme::ADDITIVE_SCHWARZ || scheme == Scheme::ALTERNATING_SCHWARZ) ///< Solve each physics independently
        {
            implicit_op->SetTimeStep(dt); ///< Set the time step for the implicit operator
            implicit_op->SetState(&x); ///< Set the input vector for the implicit operator
            solver->Mult(b,k); ///< Solve the implicit system using the solver (b is unused for FPISolver)
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
