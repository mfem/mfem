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
#include "fpi_solver.hpp"

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
        if(vsrc.Size() == source.FESpace()->GetVSize() )
        {   // Set GridFunction if the source vector is the same size.
            source.SetFromTrueDofs(vsrc);
        }        
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
public: 

    /**
     * @brief Enum used to keep track of the type of operator
     */
    enum OperatorType{
        ANY_TYPE,       ///< Any MFEM Operator
        MFEM_TDO,       ///< MFEM TimeDependentOperator
        MFEM_SOLVER,    ///< MFEM Solver
        MFEM_ODESOLVER, ///< MFEM ODESolver
        NOT_MFEM_OBJECT ///<  
    };

    /**
     * @brief Used to determine which operation shoudl be performed by 
     * PerformOperation(const int, const Vector &, Vector&)
     */
    enum OperationID{
        MULT,
        IMPLICIT_SOLVE,
        STEP,
        IMPLICIT_MULT,
        EXPLICIT_MULT,
        SOLVE,
        DEFAULT,
        NONE
    };

protected:

    int oper_index = numeric_limits<int>::max();
    std::vector<LinkedFields*> linked_fields;
    OperatorType operator_type = OperatorType::ANY_TYPE; ///< Type of the operator
    OperationID operation_id = OperationID::DEFAULT;

    std::function<void (Vector&)> pre_process_func; ///< Pre-process function
    std::function<void (Vector&)> post_process_func; ///< Post-process function

public:


    Application(int n=0) : TimeDependentOperator(n) {}

    Application(int h, int w) : TimeDependentOperator(h,w) {}

    virtual void Initialize()
    {
        mfem_error("Application::Initialize() is not overridden!");
    }

    virtual void Assemble()
    {
        mfem_error("Application::Assemble() is not overridden!");
    }

    virtual void Finalize()
    {
        mfem_error("Application::Finalize() is not overridden!");
    }

    /**
     * @brief Set the index of the operator
     */
    void SetOperatorIndex(int index){ oper_index = index; }

    /**
     * @brief Get the index of the operator
     */
    int GetOperatorIndex() const { return oper_index; }

    /**
     * @brief Set the Operation ID to call the appropriate operation
     * from Mult()
     */
    virtual void SetOperationID(OperationID id){ operation_id = id; }

    /**
     * @brief Get the current OperationID
     */
    OperationID GetOperationID() const { return operation_id; }

    /**
     * @brief Set the Operator Type object 
     */
    void SetOperatorType(OperatorType type)
    {
        operator_type = type;
    }

    /**
     * @brief Get the Operator Type
     */
    OperatorType GetOperatorType() const
    {
        return operator_type;
    }

    /**
     * @brief Set the pre-processing function
     * @param func Pre-processing function
     * @note This is primarily used for type-erased objects
     */
    void SetPreProcessFunction(std::function<void (Vector&)> func)
    {
        pre_process_func = std::move(func);
    }

    /**
     * @brief Set the post-processing function
     * @param func Post-processing function
     * @note This is primarily used for type-erased objects
     */
    void SetPostProcessFunction(std::function<void (Vector&)> func)
    {
        post_process_func = std::move(func);
    }


    /**
     * @brief Pre-process the input vector @a x before performing operations such as
     * Mult, Solve, ImplicitSolve, Step, etc. when coupled with other operators.
     * @note This method is always called before the main operation is performed, but must
     * be overridden to have an effect. Currently, it does nothing.
     * 
     * The following rules describe some common behavior of the method for particular
     * operators when coupled with other operators:
     * - ODESolver: Update the input @a x [in] with initial condition from another application.
     * - TimeDependentOperator: Set initial guess for @a k in linear solve; not typically needed.
     */
    virtual void PreProcess(Vector &x){
        if(pre_process_func) pre_process_func(x);
    }


    /**
     * @brief Post-process the input vector @a x after performing operations such as
     * Mult, Solve, ImplicitSolve, Step, etc. when coupled with other operators.
     * @note This method is always called after the main operation is performed, but must 
     * be overridden to have an effect. Currently, it does nothing.
     * 
     * The following rules describe some common behavior of the method for particular
     * operators when coupled with other operators:
     * - ODESolver: Update the input @a x [in] with initial condition from another application.
     * - TimeDependentOperator: Prepare to communicate stage updated: un = ui+dt*k.
     */
    virtual void PostProcess(Vector &x){
        if(post_process_func) post_process_func(x);
    }


    /**
     * @brief Solve the problem defined by the operator, given an input @a x 
     * and return the solution in @a y.
     */
    virtual void Solve(const Vector &x, Vector &y) const
    {
        MFEM_ABORT("Not implemented for this Application.");
    };

    /**
     * @brief Apply the operator the vector @a x and return the result in @a y.
     * For @a TimeDependentOperator, this computes (u,t) -> k(u,t).
     */
    virtual void Mult(const Vector &x, Vector &y) const override
    {
        MFEM_ABORT("Not implemented for this Application.");
    }

    /**
     * @brief Update the operator state. 
     */
    virtual void Update()
    {
        mfem_error("Application::Update() is not overridden!");
    }
    
    /**
     * @brief Perform an operation index by @a id on the vector @a x and store the result in @a y.
     */
    virtual void PerformOperation(const int id, const Vector &x, Vector &y)
    {
        mfem_error("Application::PerformOperation() is not overridden!");
    }

    /**
     * @brief Add a LinkedField @a field to the operator.
     * @note The source field in the @a LinkedField should
     * be from the current operator.
     * 
     */
    void AddLinkedFields(LinkedFields *field)
    {
        linked_fields.push_back(field);
    }

    /**
     * @brief Perform a time step from time @a t [in] to time @a t [out] based
     * on the requested step size @a dt [in].
     * @note This function is called if the operator is an ODESolver.
     * 
     * @param x Approximate initial solution
     * @param t Time associated with the approximate solution @a x
     * @param dt Time step size
     */
    virtual void Step(Vector &x, real_t &t, real_t &dt)
    {
        mfem_error("Application::Step() is not overridden!");
    }

    /**
     * @brief Transfer the data defined in the LinkedFields for current operator.
     * 
     * @param x Information to transfer, e.g., current solution vector, stage update, etc.
     */
    virtual void Transfer(const Vector &x)
    {
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

    virtual void Transfer()
    {
        int n = linked_fields.size();
        for (int i = 0; i < n; i++)
        {
            linked_fields[i]->Transfer();
        }
    }
};



/**
 * @brief An abstract class to define the interface for operators that can be used
 * with applications. It provides a way to check if the application has the required
 * member functions and implements the Mult, Solve, and Step methods.
 */
template <typename OpType>
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
        // ---------------------------------------------------------------------
        
        template <typename T, template<typename> typename Func, typename R>
        static constexpr auto Check(T*) -> typename std::is_same< Func<T>, R>::type;
        
        template <typename, template<typename> typename, typename >
        static constexpr std::false_type Check(...);

        // --- Check for the existence of the member functions
        typedef decltype(Check<C,Mult,void>(0)) Has_Mult;
        typedef decltype(Check<C,Step,void>(0)) Has_Step;

        typedef decltype(Check<C,MultPtr,void>(0)) Has_MultPtr;
        typedef decltype(Check<C,StepPtr,void>(0)) Has_StepPtr;

    public:
        static constexpr bool HasMult  = Has_Mult::value;
        static constexpr bool HasStep  = Has_Step::value;
        static constexpr bool HasMultPtr  = Has_MultPtr::value;
        static constexpr bool HasStepPtr  = Has_StepPtr::value;
    };    


    OpType *op;  ///< Pointer to the operator
    Application *nested_op = nullptr; ///< Pointer to the nested operator, if any

public:

    constexpr bool HasMult(){return CheckMember<OpType>::HasMult;}
    constexpr bool HasStep(){return CheckMember<OpType>::HasStep;}


    AbstractOperator(OpType *op_) : op(op_)
    {    
        // If the operator is an ODESolver, we need access to its time-dependent operator 
        if constexpr (std::is_base_of<ODESolver, OpType>::value)
        {
            TimeDependentOperator* tdo = op->GetTimeDependentOperator();
            this->height = tdo->Height();
            this->width  = tdo->Width();
            nested_op = dynamic_cast<Application*>(tdo);
        }
    }

    void Mult(const Vector &x, Vector &y) const override
    {
        if constexpr (CheckMember<OpType>::HasMult)
        {
            op->Mult(x,y);
        }
        else if constexpr (CheckMember<OpType>::HasMultPtr)
        {
            op->Mult(x.Size(), x.GetData(), y.Size(), y.GetData());
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, Mult(const Vector&, Vector&) or Mult(int, double*, int, double*).");            
        }
    }

    void Step(Vector &x, real_t &t, real_t &dt) override
    {
        if constexpr (CheckMember<OpType>::HasStep)
        {
            op->Step(x,t,dt);
        }
        else if constexpr (CheckMember<OpType>::HasStepPtr)
        {
            op->Step(x.Size(), x.GetData(), t, dt);
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, Step(Vector&, real_t&, real_t&) or Step(int, double*, double&, double&).");
        }        
    }

    void SetOperationID(OperationID id) override
    {
        Application::SetOperationID(id);
        if (nested_op) nested_op->SetOperationID(id);
    }
};




class CouplingOperator; // forward declaration for CoupledApplication

/**
 * @brief A class to couple multiple operators together.
 */
class CoupledApplication : public Application
{
public:

    enum class Scheme
    {
        NONE,                ///< No coupling, solve each operator independently
        MONOLITHIC,          ///< Solve all operators simultaneously
        ADDITIVE_SCHWARZ,    ///< Jacobi-type: solve each operator and transfer data to the next after all operators
        ALTERNATING_SCHWARZ  ///< Gauss-Seidel-type: solve each operator and transfer data to the next immediately
    };

protected:

    int nops = 0; ///< The number of applications
    std::vector<Application*> operators;    
    mfem::Array<int> offsets;   // Block offsets for each operator
    int max_op_size=0;          // Largest operator size


    /// Operator for coupling scheme
    Scheme scheme = Scheme::ADDITIVE_SCHWARZ; ///< Current coupling type
    CouplingOperator *coupling_op = nullptr;
    bool own_coupling_op = true;
    
    /// Solver for the coupling operator (e.g., FPISolver for Schwarz and Newton/Krylov for Monolithic)
    Solver *solver = nullptr; 
    bool own_solver = false;

    real_t dt_ = 0.0; ///< Timestep used to compute u = u_0 + dt*dudt for transfer to operators in multistage methods
    mutable Vector b;
    
public:

    /**
     * @brief Construct a new CoupledApplication object.
     * @param nop Total number of operators to couple
     */
    template<typename... Args>
    CoupledApplication(const int nop, Args... args) : Application(args...) {
        operators.reserve(nop);
        offsets.Reserve(nop+1);
        offsets.Prepend(0);
    }

    /**
     * @brief Construct a new CoupledApplication object.
     * @param nop Total number of operators to couple
     */
    CoupledApplication(const int nop) : Application() {
        operators.reserve(nop);
        offsets.Reserve(nop+1);
        offsets.Prepend(0);
    }

    /**
     * @brief Construct a new CoupledApplication object for an abstract non/mfem operator
     * or templated operator, and derived class.
     */
    template <class OpType>
    CoupledApplication(const OpType &op) : Application() {
        AddOperator(op);
    }


    /**
     * @brief Add an operator to the list of coupled operator and return pointer to it.
     */
    template <class OpType>
    Application* AddOperator(OpType *op_) {

        // Add operator to list of operators
        if constexpr(std::is_base_of<Application, OpType>::value)
        {
            operators.push_back(op_);
        } 
        else
        {
            operators.push_back(new AbstractOperator<OpType>(op_));
        }
        nops++;

        // Update size of the coupled operator and the block offsets
        Application* op = operators.back();
        op->SetOperatorIndex(nops-1); // Set the index of the operator

        int sum = offsets.Last();
        offsets.Append(sum + op->Width());

        this->width  += op->Width();
        this->height += op->Height();

        max_op_size = std::max(max_op_size, op->Width()); // Largest operator

        return op;
    }


    /**
     * @brief Get the number of coupled operators
     */
    int Size(){return nops;}

    /**
     * @brief Get the size of the largest operator
     */
    int Max(){return max_op_size;}


    /**
     * @brief Get the Application object at index i
     */
    Application* GetOperator(const int i) { return operators[i]; }

    /**
     * @brief Set the operator coupling scheme
     * @note Currently supported options are provided in enum Scheme
     */
    virtual void SetCouplingScheme(Scheme scheme_) { scheme = scheme_; }

    /**
     * @brief Get the current operator coupling scheme
     */
    Scheme GetCouplingScheme() const { return scheme; }

    /**
     * @brief Set the offset used by BlockVector for the coupled operator
     */
    void SetOffsets(Array<int> off_sets) { this->offsets = off_sets; }

    /**
     * @brief Return the offset used by BlockVector for the coupled operator
     */
    const Array<int> GetOffsets(){return offsets;}


    /**
     * @brief Set the solver used for fixed point iterations or Newton solver in
     * the implicit solve
     * @param own If 'true', own @a solver_
     */
    void SetSolver(Solver *solver_, bool own = false){ solver = solver_; own_solver = own;}


    /**
     * @brief Get the CouplingOperator corresponding to the coupling scheme
     */
    CouplingOperator* GetCouplingOperator(){ return coupling_op; }

    /**
     * @brief Set the CouplingOperator corresponding to the coupling scheme
     * @param op CouplingOperator to use
     * @param own If 'true', own @a op
     */
    void SetCouplingOperator(CouplingOperator* op, bool own=false);

    /**
     * @brief Build and return the appropriate CouplingOperator for the currently
     * set coupling scheme
     */
    CouplingOperator *BuildCouplingOperator();


    /**
     * @brief Initialize the CoupledOperator
     * @param do_ops If 'true', call Initialize on all operator
     */
    void Initialize(bool do_ops = false);


    /**
     * @brief Assemble the CoupledOperator
     * @param do_ops If 'true', call Assemble on all operator
     */
    void Assemble(bool do_ops = false);


    /**
     * @brief Finalize the CoupledOperator
     * @param do_ops If 'true', call Finalize on all operator
     */
    void Finalize(bool do_ops = false);


    /**
     * @brief Pre-process the Vector @a x before an operation (e.g., Mult, ImplictSolve, etc.)
     * @note See Application::PreProcess(Vector&) for when this can be used
     * @param do_ops If 'true', call PreProcess for all operators
     */
    void PreProcess(Vector &x, bool do_ops = false);

    
    /**
     * @brief Post-process the Vector @a x after an operation (e.g., Mult, ImplictSolve, etc.)
     * @note See Application::PostProcess(Vector&) for when this can be used
     * @param do_ops If 'true', call PostProcess for all operators
     */
    void PostProcess(Vector &x, bool do_ops = false);
    
    
    /**
     * @brief Transfer Vector @a x to operators via LinkedFields
     */
    virtual void Transfer(const Vector &x) override;


    void SetTime(const real_t t) override
    {
        if (isExplicit()){
            // Compute dt based on the current time and the previous time
            // Used to compute u = u_0 + dt*dudt for transfer to operators in multistage methods
            // Note: this is only needed for explicit operators; ImplicitSolve passes dt directly
            this->dt_ = t - this->GetTime();
        }

        TimeDependentOperator::SetTime(t);
        for (auto &op : operators)
        {
            op->SetTime(t);
        }
    }


    /**
     * @brief Perform operation, defined by @a op, on @a x and return in @a y
     * @note @a id is typically selected from enum Appliction::OperationID
     */
    void PerformOperation(const int id, const Vector &x, Vector &y)
    {
        for (auto &op : operators) {
            op->PerformOperation(id, x, y);
        }
    }

    
    /**
     * @brief Apply the operator defined by the application to the vector @a x 
     * and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const override;


    /**
     * @brief Advance the time step for each application. \Phi_i(x_i; t, dt) = x_i(t) -> x_i(t+dt).
     * @note This is used when the operator to be coupled are of type ODESolver.
     * @note For Additive Schwarz, forms a Jacobi-type coupling, where the information @a x at time @a t is
     *       first transfered, followed by advancing all operators
     * @note The Alternating Schwarz forms a Gauss-Seidel-type coupling, where each operator, \Phi_i, is advanced
     *       sequentially, and the new state, x_i(t+dt), is transfered to all other operators. This requires that
     *       the operators are ordered.
     */
    void Step(Vector &x, real_t &t, real_t &dt) override;


    /**
     * @brief Solve for the unknown k, at the current time t, the following equation: F(u + gamma k, k, t) = G(u + gamma k, t).
     * The solution procedure is determinied by the coupling scheme (e.g., partitioned vs monolithic solves).
     */
    void ImplicitSolve(const real_t dt, const Vector &x, Vector &k );

    /**
     * @brief Perform the action of the implicit part of all operators, F_i: v_i = F(u_i, k_i, t)
     * where t is the current time. This function performs an implicit solve for the monolithic system
     */
    void ImplicitMult(const Vector &u, const Vector &k, Vector &v) const override;

    
    /**
     * @brief Perform the action of the explicit part of all operators, G_i: v_i = G(u_i, t)
     * where t is the current time.
     */
    void ExplicitMult(const Vector &u, Vector &v) const override;

    
    /**
     * @brief Destroy the Coupled Application object
     */
    ~CoupledApplication();
};





class CouplingOperator : public Application
{
    protected:
        CoupledApplication *coupled_operator;
        const Vector *u = nullptr; ///< Pointer to the input vector, used in Mult() methods
        bool is_linear = false;    ///< Flag to indicate if the operator is linear
        real_t dt= 0.0;            ///< Time step size, used for time-dependent applications
        friend class CoupledApplication;
    public:
        CouplingOperator(CoupledApplication *op) : Application(op->Height(),
                                                   op->Width()),
                                                   coupled_operator(op){}

        virtual void SetTimeStep(real_t dt_){ dt = dt_;}
        virtual void SetState(const Vector *u_){u = u_;}
        virtual void SetLinearity(const bool is_linear_){is_linear = is_linear_;}
        virtual bool IsLinear() const { return is_linear; }
};

class AdditiveSchwarzOperator : public CouplingOperator
{
    protected:
        mutable Vector z;
    public:
        AdditiveSchwarzOperator(CoupledApplication *op) : CouplingOperator(op){
            z.SetSize(coupled_operator->Max());
        }
        void Mult(const Vector &ki, Vector &k) const override;
        void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) override;
        void Step(Vector &x, real_t &t, real_t &dt) override;
};

class AlternatingSchwarzOperator : public CouplingOperator
{
    protected:
        mutable Vector z;
    public:
        AlternatingSchwarzOperator(CoupledApplication *op) : CouplingOperator(op){
            z.SetSize(coupled_operator->Max());
        }
        void Mult(const Vector &ki, Vector &k) const override;
        void ImplicitSolve(const real_t dt, const Vector &x, Vector &k ) override;
        void Step(Vector &x, real_t &t, real_t &dt) override;
};

class MonolithicOperator : public CouplingOperator
{
    protected:
        mutable future::FDJacobian grad;
        mutable real_t du = 1.0;
        mutable Vector upk;

    public:
        MonolithicOperator(CoupledApplication *op, real_t eps=1.0e-6) : CouplingOperator(op),
                                                                        grad(*this,eps), 
                                                                        upk(Width()) {}
        /**
         * @brief Set the du coefficient in upk = du*u + dt*k, depending on the linearity
         *        of the operator. For linear problems, du = 0, for nonlinear problems, du = 1.
         */
        void SetLinearity(bool is_linear_) override {
            CouplingOperator::SetLinearity(is_linear_);
            du = is_linear ? 0.0 : 1.0; // Set du to 0 for linear problems, 1 for nonlinear
        }

        /**
         * @brief Computes the action of the implicit, monolithic operator on vector x.
         *        ImplicitMult assumes ODE of the form F(u,k,t) = M k - g(u,t), G(u,t) = 0
         *        For fully implicit, monolithic solver, we take F(u+dt*k,k,t) = M k - g(u+dt*k,t)
         */
        void Mult(const Vector &ki, Vector &k) const override;
        
        /**
         * @brief Update the current state/point of linearization for the nonlinear operator. For
         *        Jacobian-free methods, i.e. J(k)*v = (F(k+eps*v) - F(k))/eps, this sets the F(k)
         */
        Operator& GetGradient(const Vector &k) const override;
};


} //mfem namespace


#endif
