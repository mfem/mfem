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

namespace mfem
{

/// Forward declarations needed below
class Field;
class GraphNode;
class DAGraph;
class DualGraph;
struct GraphOperation;




// TODO: Should move these to a util namespace or a util file. 
template <typename T, std::size_t... I>
auto ArrayToTuple_impl( const Array<T>& v, std::index_sequence<I...>)
{
    return std::make_tuple(v[I]...);
}

template <int N, typename T>
auto ArrayToTuple(const Array<T>& v)
{
    return ArrayToTuple_impl(v,std::make_index_sequence<N>{});
}

// Pack an iterable item into a MultiVector using template
// Check if type is iterable has begin and end methods
// Also take in a get function that fetches the vector from item
template <typename R, typename T,
          typename = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
void IterableToMultiVector(const T &items, MultiVector &mv,
                           std::function<R(typename T::value_type)> get)
{
    int i = 0, nblocks = std::distance(std::begin(items), std::end(items));
    mv.SetNumBlocks(nblocks);
    for (auto &item : items) { mv.MakeRef(i++, get(item)); }
}

template <typename MV, typename I, typename Set>
void MultiVectorToIterable( MV& mv, I& items, Set&& set)
{
    int i = 0;
    for (auto& item : items) { set(item, mv[i++]); }
}

void BlockVectorToMultiVector(BlockVector &bv, MultiVector &mv);

int GetValidID(int id, int& current, int lb=0, int ub = std::numeric_limits<int>::max());

/// @brief Abstract base class for recording operations
class AbstractTape
{
protected:
    bool is_recording = false; ///< Flag to indicate if recording is active

public:

    virtual void Watch(std::initializer_list<Field*> fields_list)
    {
        MFEM_ABORT("This method is not overridden for this class!");
    }

    bool IsRecording() const { return is_recording; }
    virtual void StartRecording() { is_recording = true; }
    virtual void PauseRecording() { is_recording = false; }

    virtual void StopRecording(std::initializer_list<Field*> outputs_list)
    {
        MFEM_ABORT("This method is not overridden for this class!");
    }

    virtual void RegisterOperation(GraphOperation *operation)
    {
        MFEM_ABORT("This method is not overridden for this class!");
    };

    virtual void ClearTape() 
    {
        MFEM_ABORT("This method is not overridden for this class!");
    };

    virtual ~AbstractTape() = default;
};


/// @brief Base class for storing data (Vector) and distinguishing
/// fields variables
class Field
{
public:
    friend class GraphNode;

private:
    inline static int next_id = 0;

protected:
    Vector *data = nullptr;
    Vector *adjoint = nullptr; // For storing derivative info
    int id = -1; // initialized to invalid id
    bool own_adj = false; // Whether this Field owns the adjoint Vector
    bool own_data = false; // Whether this Field owns the data Vector

    std::string name; // Optional name for the field
    Operator *oper  = nullptr; // Operator that outputs this field
    AbstractTape *tape = nullptr; // Optional tape tracking this field

public:

    ///@brief Constructor for a Field of type Type with optional ID
    Field(Vector *field, Vector *adjoint, int id_ = -1) :
          data(field), adjoint(adjoint), id(GetValidID(id_,next_id)),
          name("Field_" + std::to_string(id)) { }

    ///@brief Constructor for an input field
    Field(Vector *field, int id_ = -1) :
          Field(field, nullptr, id_) { }

    template<typename T,
             typename std::enable_if<std::is_base_of<Vector,T>::value,bool>::type = true>
    Field(const T &v, int id_ = -1) : Field(nullptr, nullptr, id_)
    {
        AllocateData(v);
    }

    Field(int id_ = -1) : Field(nullptr, nullptr, id_) { }

    ///@brief Get the stored internally stored data pointer
    Vector* Data() const { return data; }
    Vector* Adjoint() const { return adjoint; }
    Operator* GetOperator() const { return oper; }

    template<typename T,
             typename std::enable_if<std::is_base_of<Vector,T>::value,bool>::type = true>
    void AllocateData(const T &v)
    {
        if(data && own_data) { delete data; }
        if(adjoint && own_adj) { delete adjoint; }
        data = new T(v); // Create a new Vector on device/host
        adjoint = new T(v); // Might be unused (e.g., gradient calculation)
        own_data = true;
        own_adj = true;
    }

    ///@brief Set the internally stored data pointer
    virtual void SetData(Vector *field, Vector *adj)
    {
        if(data && own_data) { delete data; }
        if(adjoint && own_adj) { delete adjoint; }
        data = field;
        adjoint = adj;
        own_data = false;
        own_adj = false;
    }

    virtual void SetData(Vector *field)
    {
        if(data && own_data) { delete data; }
        data = field;
        own_data = false;
        if(!adjoint)
        {
            adjoint = field;// If not set, use the same as data
            own_adj = false;
        }
    }

    virtual void SetAdjoint(Vector *adj)
    {
        if(adjoint && own_adj) { delete adjoint; }
        adjoint = adj;
        own_adj = false;
    }

    virtual void SetOperator(Operator *op) { oper = op; }

    void SetTape(AbstractTape *t)
    {
        if((tape && t) && (tape != t))
        {
            MFEM_ABORT("Tape is already set for this Field.");
        }
        tape = t;
    }

    AbstractTape* GetTape() const { return tape; }

    std::string Name() const { return name; }
    void SetName(const std::string &n) { name = n; }
    int ID() const { return id; }

    void SetID(int i)
    {
        MFEM_ASSERT(i >= 0, "ID must be non-negative.");
        id = i;
    }

    virtual ~Field()
    {
        if(data && own_data) { delete data; }
        if(adjoint && own_adj) { delete adjoint; }
    }
};


struct GraphOperation
{
    using InputType = std::initializer_list<Field*>;
    using OutputType = InputType;
    using ExecuteFunc = std::function<void(const MultiVector&, MultiVector&)>;
    using GradFunc = std::function<void(const MultiVector&, const MultiVector&, MultiVector&)>;
    using IndexMap = GenericFieldMap<int, int>;

protected:
    Operator *op;

public:
    Array<Field*> inputs, outputs;
    IndexMap input_index, output_index; ///< Field::ID to Array index
    ExecuteFunc execute;
    GradFunc grad, grad_transpose;
    mutable MultiVector xmv, ymv;

    GraphOperation(Operator &oper, InputType in, OutputType out,
                   ExecuteFunc exec = nullptr, GradFunc grad = nullptr,
                   GradFunc grad_transpose = nullptr) :
                   op(&oper), inputs(in), outputs(out),
                   execute(exec), grad(grad), grad_transpose(grad_transpose),
                   xmv(inputs.Size()), ymv(outputs.Size())
    {
        int i = 0, o = 0;
        for(auto *f : in) { input_index.Register(f->ID(), i++); }
        for(auto *f : out) { output_index.Register(f->ID(), o++); }
        if(!exec)
        {
            execute = [this](const MultiVector &x, MultiVector &y)
                            { op->MultMV(x, y); };
        }
        if(!grad)
        {
            grad = [this](const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                          { op->GetGradientMV(x).MultMV(dx, dy); };
        }
        if(!grad_transpose)
        {
            grad_transpose = [this](const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                                    { op->GetGradientMV(x).MultTransposeMV(dx, dy); };
        }
    }

    GraphOperation(ExecuteFunc exec, InputType in, OutputType out,
                   GradFunc grad = nullptr, GradFunc grad_transpose = nullptr) :
                   op(nullptr), inputs(in), outputs(out), execute(exec),
                   grad(grad), grad_transpose(grad_transpose),
                   xmv(inputs.Size()), ymv(outputs.Size())
    {
        int i = 0, o = 0;
        for(auto *f : in) { input_index.Register(f->ID(), i++); }
        for(auto *f : out) { output_index.Register(f->ID(), o++); }
    }

    virtual void Execute() const;

    GraphOperation *GetGradient() const;

    Operator &GetOperator() const { return *op; }

    virtual ~GraphOperation() // Do NOT delete op, inputs or outputs - they are not owned;
    { }
};

// @brief A GraphOperation that stores the internal state of the operator
// and allows for state-dependent execution. Can be registered with the DAG
template<typename OpType, typename StateType>
struct StateDependentGraphOperation : GraphOperation
{
    using ExecuteFunc = std::function<void(StateType &state,
                                           const MultiVector&, MultiVector&)>;
    using GradFunc = std::function<void(StateType &state,
                                        const MultiVector&, const MultiVector&, MultiVector&)>;

    mutable StateType *state; // State to be stored (not owned)
    ExecuteFunc execute;
    GradFunc grad, grad_transpose;

    // Only use if OpType extended from mfem::Operator
    template<typename = std::enable_if_t<std::is_base_of<Operator, OpType>::value>>
    StateDependentGraphOperation(OpType &oper, InputType in, OutputType out, StateType &state,
                                 ExecuteFunc exec = nullptr, GradFunc grad = nullptr,
                                 GradFunc grad_transpose = nullptr) :
                                 GraphOperation(oper, in, out, nullptr, nullptr, nullptr),
                                 state(&state), execute(exec), grad(grad), grad_transpose(grad_transpose)
                                 { }

    StateDependentGraphOperation(ExecuteFunc exec, InputType in, OutputType out, StateType &state,
                                 GradFunc grad = nullptr, GradFunc grad_transpose = nullptr) :
                                 GraphOperation(nullptr, in, out, nullptr, nullptr),
                                 state(&state), execute(exec), grad(grad), grad_transpose(grad_transpose)
                                 { }

    void Execute() const override
    {
        auto get = [](const Field *f) -> Vector& { return *f->Data(); };
        IterableToMultiVector<Vector&>(inputs, xmv, get);
        IterableToMultiVector<Vector&>(outputs, ymv, get);
        execute(*state, xmv, ymv);
    }
};

struct GraphOperationGradient : GraphOperation
{
    ExecuteFunc execute_primal;
    mutable Array<Vector*> primal;
    mutable MultiVector pmv;

    GraphOperationGradient(GraphOperation &oper);

    void Execute() const override;

    ~GraphOperationGradient()
    {
        primal.DeleteAll();
        pmv.SetNumBlocks(0);
    }
};

class GraphNode : public Operator
{
public:
    enum ExecutionMode
    {
        GRADIENT_MODE, ///< Node is being executed as part of a gradient evaluation
        DEFAULT_MODE   ///< Node is being executed as default, operator evaluation
    };

private:
    inline static int next_id = 0;

protected:
    int id = -1;
    std::string name; ///< Optional name for the node
    mutable ExecutionMode exec_mode = DEFAULT_MODE;
    Array<int> input_offsets;  ///< Offsets for input fields
    Array<int> output_offsets; ///< Offsets for output fields

public:

    GraphNode(int h, int w) : Operator(h,w), id(GetValidID(-1,next_id)),
                              name("GraphNode_" + std::to_string(id))
                              { }

    GraphNode(int s = 0) : GraphNode(s, s) { }

    void SetExecutionMode(ExecutionMode mode) { exec_mode = mode; }
    ExecutionMode GetExecutionMode() const { return exec_mode; }

    void SetID(int id_) { id = id_; }
    int ID() const { return id; }

    void SetName(const std::string &n) { name = n; }
    std::string Name() const { return name; }


    virtual void SetOffsets(const Array<int> &inoff, const Array<int> &outoff)
    {
        input_offsets = inoff;
        output_offsets = outoff;
    }

    auto GetOffsets() const
    {
        return std::make_tuple(std::cref(input_offsets), std::cref(output_offsets));
    }

    // Computes J(x) * dx = dy, where J is the Jacobian of the operator
    virtual void GradientMult(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const
    {
        MFEM_ABORT("This method is not overridden for this class!");
    }

    // Computes J(x)^T * dx = dy, where J is the Jacobian of the operator
    virtual void GradientMultTranspose(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const
    {
        MFEM_ABORT("This method is not overridden for this class!");
    }

    virtual void RegisterFields(std::initializer_list<Field*> inputs,
                                std::initializer_list<Field*> outputs);

    virtual void RegisterFields(std::initializer_list<Field*> inputs,
                                std::initializer_list<Field*> outputs,
                                GraphOperation::ExecuteFunc execute,
                                GraphOperation::GradFunc grad = nullptr,
                                GraphOperation::GradFunc grad_transpose = nullptr);

    // Register fields with state-dependent execution
    template<typename StateType,
             typename GraphOpType = StateDependentGraphOperation<GraphNode, StateType>,
             typename ExecuteFunc = typename GraphOpType::ExecuteFunc,
             typename GradFunc =typename GraphOpType::GradFunc,
            // Disable to avoid conflict with other RegisterFields method when StateType is a lambda
             std::enable_if_t<!std::is_same_v<StateType, GraphOpType::ExecuteFunc>, bool> = true
            >
    void RegisterFields(std::initializer_list<Field*> inputs,
                        std::initializer_list<Field*> outputs,
                        StateType &state,
                        ExecuteFunc execute = nullptr,
                        GradFunc grad = nullptr,
                        GradFunc grad_transpose = nullptr)
    {
        // Tape operations
        // Check if all input fields have the same, non-null tape; if so, register this node with that tape
        bool has_same_tape = std::empty(inputs) || std::equal(inputs.begin(), inputs.end(), inputs.begin(),
                            [](const Field *a, const Field *b)
                            { return (a->GetTape() && (a->GetTape() == b->GetTape())); });

        auto in = inputs.begin();
        auto tape = (*in)->GetTape();
        if(!has_same_tape && tape)
        {
            MFEM_ABORT("Input fields are being recorded on different tapes. Cannot register operation.");
        }
        else if(tape->IsRecording()) // All tapes are the same and recording is active
        {
            // Default functions if not provided
            auto def_exec = execute ? execute : [this](StateType &s, const MultiVector &x, MultiVector &y) { this->MultMV(x, y); };
            auto def_grad = grad ? grad : [this](StateType &s, const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                                                { this->GradientMult(x, dx, dy); };
            auto def_grad_transpose = grad_transpose ? grad_transpose :
                                     [this](StateType &s, const MultiVector &x, const MultiVector &dx, MultiVector &dy)
                                           { this->GradientMultTranspose(x, dx, dy); };

            // Register the operation on the tape
            auto *op = new StateDependentGraphOperation<GraphNode, StateType>(*this, inputs, outputs, state,
                                                                              def_exec, def_grad, def_grad_transpose);
            tape->RegisterOperation(op);
        }
        else
        {   // Same tape but not recording, or no tape at all
            // Should we abort or do nothing?
            // MFEM_ABORT("Input fields are not being recorded on a tape. Cannot register operation.");
        }
    }

    virtual ~GraphNode()
    { }
};


/**
   @brief A class to store and coupled multiple operators together.
 */
class DAGraph : public GraphNode, public AbstractTape
{
public:
    inline static int max_ops = 1000;
    inline static int max_fields = 1000;
    inline static int max_derivatives = 1;

    using IndexMap = GenericFieldMap<int, int>;

protected:
    Array<GraphOperation*> operations; ///< List of operations in the graph
    Array<bool> op_owned; ///< Ownership of operations (unused for now)
    Array<int> op_depth;  ///< Depth of each operation in the graph
    IndexMap id_to_op_index; ///< Map from field ID to operation index

    Array<Field*> fields;  ///< List of all fields used in operations (not owned)
    IndexMap id_to_field_index; ///< Map from field ID to index in fields array

    // Experimental state memory management by dag (not Field)
    Array2D<Vector*> state_memory; ///< Owned: use id->field_index to index into this array
                                   /// nfields x ngrad (to store primal and daul)

    int max_depth = 0; ///< Maximum depth of the graph
    bool is_sorted = false; ///< Is the graph topologically sorted?
    bool is_assembled = false; ///< Is the graph assembled?

    GraphOperation* dag_op = nullptr; ///< Pointer to the operation representing the entire DAG
    mutable DualGraph *grad_dag = nullptr; ///< Gradient dag operator

    std::function<void(Field *f, const Vector &v)> get_inputs_func; ///< Function to get inputs for the DAG
    std::function<void(Field *f, Vector &v)> get_outputs_func; ///< Function to get outputs for the DAG

    friend class DualGraph; ///< Allow DualGraph to access protected members
public:
    /**
       @brief Construct a new DAG with a total number of expected operators.
       @param nop (Optional) Total number of expected operators
     */
    DAGraph(int nops = 0, int nfields = 0);

protected:
    // Force tape use for now
    void AddOperation(GraphOperation *op)
    {
        operations.push_back(op);
        // Register input fields and id->index mapping
        for(auto *f : op->inputs)
        {
            if(!id_to_field_index.Has(f->ID()))
            {
                fields.push_back(f);
                id_to_field_index.Register(f->ID(), fields.Size() - 1);
            }
        }
        // Register outputs fields and id->index mapping and
        // id->operation that outputs the field
        for(auto *f : op->outputs)
        {
            id_to_op_index.Register(f->ID(), operations.Size() - 1);
            if(!id_to_field_index.Has(f->ID()))
            {
                fields.push_back(f);
                id_to_field_index.Register(f->ID(), fields.Size() - 1);
            }
        }
        op_owned.push_back(true); // For now, we own the operations
        is_sorted = false;
        is_assembled = false;
    }
public:

    /// @brief Get the number of coupled operators
    int Size() const {return operations.Size();}

    /// @brief Get the operator at index @a i
    GraphOperation &GetOperation(const int i)
    {
        int nop = operations.Size();
        MFEM_ASSERT(i >= 0 && i < nop,
                    "index [" << i << "] is out of range [0," << nop << ")");
        return *operations[i];
    }

    virtual void Assemble();
    bool IsAssembled() const { return is_assembled; }

    virtual void Sort();
    bool IsSorted() const { return is_sorted; }

    void ComputeDepth(bool reverse = false);

protected:
    // This changes intermediate state; restrict user call
    virtual void UpdateState(const MultiVector &x);

    // Experimental: Get state memory to copy and store primal in dual graph
    // virtual void GetState(Field &field, Array<Vector*> &state, int igrad = 0)
    virtual void GetState(Field &field, MultiVector &state, int igrad = 0)
    {
        MFEM_ASSERT(igrad >= 0, "The ith gradient must be non-negative.");

        // MFEM_ASSERT(state.Size() > igrad, "State size " << state.Size()
        //             << " is less than the requested gradient indices " << igrad);
        MFEM_ASSERT(state.NumBlocks() > igrad, "State size " << state.NumBlocks()
                    << " is less than the requested gradient indices " << igrad);

        bool dag_has_field = id_to_field_index.Has(field.ID());
        MFEM_ASSERT(dag_has_field, "Field with ID " << field.ID() << " is not registered in the DAG.");

        int idx = id_to_field_index.Get(field.ID());

        // For now enforce the two are the same fields
        MFEM_ASSERT(&field == fields[idx], "Field with ID " << field.ID()
                    << " does not match the registered field in the DAG.");

        // Get all the gradients for this field upto igrad (0 for primal, 1 for primal & dual, etc.)
        for(int i = 0; i <= igrad; ++i)
        {
            // Force storage as const to avoid accidental modification of the state memory
            state.MakeRef(i, std::as_const(*state_memory(idx, i)));
            // Possibly allocate new and copy to avoid changing the state memory in the DAG
            // and handle the copy operation, if state is Array<Vector*> instead of MultiVector
            // state[i] = field.CreateCopy(state_memory(idx, i));
        }
    }
public:

    void SetOffsets(const Array<int> &inoff, const Array<int> &outoff) override
    {
        GraphNode::SetOffsets(inoff, outoff);
        width = inoff.Last();
        height = outoff.Last();
    }

    void Mult(const Vector &x, Vector &y) const override;
    void MultMV(const MultiVector &x, MultiVector &y) const override;

    void MultTranspose(const Vector &x, Vector &y) const override;
    void MultTransposeMV(const MultiVector &x, MultiVector &y) const override;

    Operator& GetGradient(const Vector &x) const override;
    Operator& GetGradientMV(const MultiVector &x) const override;

    // Taping operations
    void Watch(std::initializer_list<Field*> fields_list) override;
    void StopRecording(std::initializer_list<Field*> outputs_list) override;
    void ClearTape() override;
protected: // This should be protected
    void RegisterOperation(GraphOperation *op) override;
public:

    virtual void Reset();
    ~DAGraph();
};


class DualGraph : public DAGraph
{
protected:
    mutable const DAGraph *primal_dag = nullptr; ///< Pointer to the primal dag operator
public:
    DualGraph(const DAGraph &primal_dag);
    void Assemble() override;
    void UpdateState(const MultiVector &x) override;
};

} //mfem namespace

#endif
