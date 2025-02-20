// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#ifndef MFEM_APPLICATION_HPP
#define MFEM_APPLICATION_HPP

#include "../linalg/operator.hpp"

#include <memory>
#include <vector>
#include <type_traits> // std::is_base_of

namespace mfem
{

/** @brief This class is used to define the interface for applications and miniapps.
    It is used to define the Solve method which is used to solve the problem
    defined by the application or miniapp.
 */
class MFEMApplication : public TimeDependentOperator
{
private:
    std::string name;

protected:

    int current_operation = -1;

public:

    MFEMApplication(std::string name_) : name(name_) {}

    virtual void Initialize() {
        mfem_error("MFEMApplication::Initialize() is not overridden!");
    }

    virtual void Assemble() {
        mfem_error("MFEMApplication::Assemble() is not overridden!");
    }

    std::string GetName() const {
        return name;
    }

    /**
     * @brief Given a vector @a x, solve the problem defined by the application 
     * and return the solution in @a y.
     */
    virtual void Solve(const Vector &x, Vector &y) const {
        MFEM_ABORT("Not implemented for this MFEMApplication.");
    };

    /**
     * @brief Apply the operator defined by the application to the vector @a x 
     * and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const override {
        MFEM_ABORT("Not implemented for this MFEMApplication.");
    }

    /**
     * @brief Set the Operation to perform
     */
    virtual void SetOperation(const int op) {
        current_operation = op;
    }

    /**
     * @brief Get the Operation to perform
     */
    virtual int GetOperation() const {
        return current_operation;
    }   

    virtual void PerformOperation(const Vector &x, Vector &y) {
        mfem_error("MFEMApplication::PerformOperation() is not overridden!");
    }

    // Need something to fetch/link variables for multiapp integration
    // Need something to get/set/fetch/perform internal operations
};


template <typename App>
class AbstractApplication : public MFEMApplication
{
private:

protected:
    App app;

public:
    
    AbstractApplication(App &app_) : app(app_) {}

    void Initialize() override
    {
        app.Initialize();
    }

    void Assemble() override
    {
        app.Assemble();
    }

    std::string GetName() const
    {
        return app.GetName();
    }

    void Solve(const Vector &x, Vector &y) const override
    {
        app.Solve(x, y);
    }

    void Mult(const Vector &x, Vector &y) const override
    {
        app.Mult(x, y);
    }

    void PerformOperation(const Vector &x, Vector &y) override
    {
        app.PerformOperation(const Vector &x, Vector &y);
    }
};



class MFEMCoupledApplication
{

protected:

    
private:
    std::vector<std::shared_ptr<MFEMApplication>> apps;

public:

    /**
     * @brief Construct a new MFEMCoupledApplication object. The total number of applications
     * @param napp Total number of applications
     */
    MFEMCoupledApplication(const int napp) {
        apps.reserve(napp);
    }

    /**
     * @brief Construct a new MFEMCoupledApplication object for an abstract non/mfem application
     * or templated application, and derived class.
     */
    template <class AppType>
    MFEMCoupledApplication(const AppType &app) {
        AddApplication(app);
    }

    /**
     * @brief Add an application to the list of coupled application.
     */
    template <class AppType>
    void AddApplication(const AppType &app) {
        if constexpr(std::is_base_of<MFEMApplication, AppType>::value) {
            apps.push_back(std::make_shared<AppType>(app));
        } else {
            apps.push_back(std::make_shared<AbstractApplication<AppType>>(app));
        }
    }

    MFEMApplication* GetApplication(const int i) {
        return apps[i].get();
    }

    void Initialize() {
        for (const auto &app : apps) {
            app->Initialize();
        }
    }

    void Assemble() {
        for (const auto &app : apps) {
            app->Assemble();
        }
    }

    /**
    * @brief Given a vector @a x, solve the problem defined by the application 
    * or miniapp and return the solution in @a y.
    */
    virtual void Solve(const Vector &x, Vector &y) const{
        for (const auto &app : apps) {
            app->Solve(x, y);
        }
    }

    /**
     * @brief Apply the operator defined by the application to the vector @a x 
     * and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const {
        for (const auto &app : apps) {
            app->Mult(x, y);
        }
    }

    /**
     * @brief Perform the specified operation for each application to the vector @a x
     */
    void PerformOperation(const Vector &x, Vector &y) {
        for (const auto &app : apps) {
            app->PerformOperation(x, y);
        }
    }

};

}


#endif
