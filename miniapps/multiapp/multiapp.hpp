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




/** @brief This class is used to define the interface for applications and miniapps.
    It is used to define the Solve method which is used to solve the problem
    defined by the application or miniapp.
 */
class Application : public TimeDependentOperator
{
private:
    std::string name;

protected:

    int oper_index = -1;
    std::vector<LinkedFields*> linked_fields;

public:

    Application(int n=0, std::string name_="Unknown") : TimeDependentOperator(n), name(name_) {}

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
     * @brief Set the name of the application.
     */
    std::string SetName(std::string n) { name = n;}

    /**
     * @brief Get the name of the application.
     */
    std::string GetName() const { return name;}

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

    // std::string GetName() const
    // {
    //     return app->GetName();
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
    
private:

    std::vector<Application*> apps;
    Solver *solver;

    mfem::Array<int> offsets;
    Vector xtmp, ktmp;
    
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

        ktmp.SetSize(this->Width());        
        int max_size = offsets.Max();

        if (scheme == Scheme::MONOLITHIC)
        {   // Set vector size for the full coupled operator
            xtmp.SetSize(this->Width());
        }
        else if (scheme == Scheme::ADDITIVE_SCHWARZ)
        {   // Set vector size for the the largest operator
            xtmp.SetSize(max_size);
        }
        else if (scheme == Scheme::ALTERNATING_SCHWARZ)
        {   // Set vector size for the the largest operator
            xtmp.SetSize(max_size);            
        }

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

    /**
     * @brief Apply the operator defined by the application to the vector @a x 
     * and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const
    {
        BlockVector xb(x.GetData(), offsets);
        BlockVector yb(y.GetData(), offsets);

        for (int i=0; i < napps; i++)
        {
            apps[i]->Mult(xb.GetBlock(i), yb.GetBlock(i));
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
        else if (scheme != Scheme::DECOUPLED) ///< Solve each physics independently
        {
            PartitionedImplicitSolve(dt,x,k);
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
    void PartitionedImplicitSolve(const real_t dt, const Vector &x, Vector &k ){

        BlockVector xb(x.GetData(), offsets);
        BlockVector kb(k.GetData(), offsets);

        k    = 0.0; ///< Reset the vector for the implicit system
        ktmp = 0.0; ///< Reset the temporary vector for the implicit system
        real_t err = 1.0;
        real_t tol = 1e-5;


        for (int iter=0; iter < 3; iter++)
        {
        

        if (scheme == Scheme::ADDITIVE_SCHWARZ)
        {
            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &ki = kb.GetBlock(i);
                apps[i]->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
            }

            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &ki = kb.GetBlock(i);
                int isize  = xi.Size();

                xtmp.SetSize(isize);
                add(1.0,xi,dt,ki,xtmp); ///< Compute the solution (xtmp = 1.0*xi + dt*ki)
                apps[i]->Transfer(xtmp);  ///< Transfer the data to all applications
            }
        }
        else if (scheme == Scheme::ALTERNATING_SCHWARZ)
        {
            for (int i=0; i < napps; i++)
            {
                Vector &xi = xb.GetBlock(i);
                Vector &ki = kb.GetBlock(i);
                int isize  = xi.Size();

                xtmp.SetSize(isize);
                add(1.0,xi,dt,ki,xtmp); ///< Compute the solution (xtmp = 1.0*xi + dt*ki)
                
                apps[i]->Transfer(xtmp);  ///< Transfer the data to all applications
                apps[i]->ImplicitSolve(dt,xi,ki); ///< Solve the implicit system for the application
            }
        }
            err = ktmp.DistanceTo(k);
            if( err < tol && iter > 2 )
            {
                return;
            }
            
            ktmp = k;
        }
    }
};

}


#endif
