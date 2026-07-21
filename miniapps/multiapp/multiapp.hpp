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

#ifdef MFEM_USE_MPI

namespace mfem
{

/// Forward declarations needed below
class Field;
class FieldCollection;
class GraphNode;
class DAGraph;
class GraphGradient;




/// @brief Base class for storing data (Vector) and distinguishing
/// fields variables
class Field
{
public:
    enum Type
    {
        INPUT , ///< Input field
        OUTPUT, ///< Output field
        DEFAULT ///< Any field
    };

    friend class GraphNode;

private:
    Type type = Type::DEFAULT;
    inline static int next_id = 0;

protected:
    Vector *data = nullptr;
    Vector *adjoint = nullptr; // For storing derivative info
    int id = -1; // initialized to invalid id

    std::string name; // Optional name for the field
    Operator *oper  = nullptr; // Operator that outputs this field

    int GetValidID(int id_, int lb=0, int ub = std::numeric_limits<int>::max())
    {
        return (id_ >= lb && id_ <= ub) ? id_ : next_id++;
    }

public:

    ///@brief Constructor for a Field of type Type with optional ID
    Field(Vector *field, Vector *adjoint, Type type, int id_ = -1) :
          type(type), data(field), adjoint(adjoint), id(GetValidID(id_)),
          name("Field_" + std::to_string(id)) { }

    ///@brief Constructor for a Field of Default type with optional ID
    Field(Vector *field, Vector *adjoint, int id_ = -1) : 
          Field(field, adjoint, Type::DEFAULT, id_) { }

    ///@brief Constructor for an input field
    Field(Vector *field, int id_ = -1) :
          Field(field, nullptr, Type::DEFAULT, id_) { }

    ///@brief Constructor for a Field of type Type
    Field(Vector *field, Type type, int id_ = -1) :
          Field(field, nullptr, type, id_) { }

    ///@brief Get the stored internally stored data pointer
    Vector* Data() const { return data; }
    Vector* Adjoint() const { return adjoint; }
    Operator* GetOperator() const { return oper; }

    ///@brief Set the internally stored data pointer
    virtual void SetData(Vector *field) { data = field; }
    virtual void SetAdjoint(Vector *adj) { adjoint = adj; }
    virtual void SetOperator(Operator *op) { oper = op; }

    std::string Name() const { return name; }
    void SetName(const std::string &n) { name = n; }
    int ID() const { return id; }

    void SetID(int id_)
    {
        MFEM_ASSERT(id_ >= 0, "Field::SetID: ID must be non-negative.");
        id = id_;
    }

    bool IsInput() const {return (type == Type::INPUT);}
    bool IsOutput() const {return (type == Type::OUTPUT);}
    bool IsDefault() const {return (type == Type::DEFAULT);}

    virtual ~Field() = default;

protected:
    void MakeInput() { type = Type::INPUT; }
    void MakeOutput() { type = Type::OUTPUT; }

    ///@brief Set the type of the field (prevents changing type of input/output fields)
    void SetType(Type t)
    {
        type = t;
    }
};

/// @brief A collection of Fields, each identified by a name
class FieldCollection
{
private:
    std::string name; /// Name of the collection
    Operator *oper = nullptr; /// Operator associated with this collection (not owned)
    NamedFieldsMap<Field> fields;
    NamedFieldsMap<int> index_map; /// Map from field name to index in input/output vectors

    std::vector<Field*> input_fields;  // Input fields for this node
    std::vector<Field*> output_fields; // Output fields for this node

public:

    FieldCollection() = default;

    /// @brief Constructor with collection name and optional associated operator
    FieldCollection(std::string collection_name, Operator *op = nullptr):
                    name(collection_name), oper(op) {}

    /// @brief Constructor with associated operator and default collection name
    FieldCollection(Operator *op) : name("FieldCollection"), oper(op) {}

    /// @brief Get the number of fields in the collection
    int Size() const { return fields.NumFields(); }

    /// @brief Set the name of the collection
    void SetName(const std::string &collection_name) { name = collection_name;}

    /// @brief Get the name of the collection
    std::string Name() const { return name; }

    /// @brief Set the operator associated with this collection
    void SetOperator(Operator *op){ oper = op; }

    /// @brief Get the operator associated with this collection
    const Operator* GetOperator() const { return oper; }

    /// @brief Get the field associated with the given name, or nullptr if not found
    Field* GetField(const std::string &field_name) const
    {
        return fields.Get(field_name);
    }

    /// @brief Add a field to the collection with a given name and ownership flag
    void AddField(const std::string &field_name, Field *field, bool own = false)
    {
        if(fields.Has(field_name))
        {
            MFEM_WARNING("FieldCollection::AddField: Field with name "
                         << field_name << " already exists. Replacing existing field.");
        }
        fields.Register(field_name, field, own);
    }

    void AddInput(const std::string &field_name,
                  Field *field, bool own = false)
    {
        bool has_field = fields.Has(field_name);
        auto i = index_map.Get(field_name);
        if(has_field && i != nullptr)
        {
            input_fields[*i] = field;
        }
        else
        {
            input_fields.push_back(field);
            index_map.Register(field_name, new int(input_fields.size() - 1), true);
        }
        AddField(field_name, field, own);
    }

    void AddOutput(const std::string &field_name,
                   Field *field, bool own = false)
    {
        bool has_field = fields.Has(field_name);
        auto i = index_map.Get(field_name);
        if(has_field && i != nullptr)
        {
            output_fields[*i] = field;
        }
        else
        {
            output_fields.push_back(field);
            index_map.Register(field_name, new int(output_fields.size() - 1), true);
        }

        AddField(field_name, field, own);
        if(field->GetOperator() == nullptr)
        {
            field->SetOperator(oper);
        }
    }

    std::vector<Field*>& InputFields() { return input_fields; }
    std::vector<Field*>& OutputFields() { return output_fields; }
    
    Field* InputField(int i) const { return input_fields[i]; }

    Field *InputField(const std::string &field_name) const
    {
        auto idx = index_map.Get(field_name);
        if(!idx)
        {
            MFEM_WARNING("FieldCollection::InputField: Field with name "
                         << field_name << " does not exist in the collection.");
            return nullptr;
        }

        int index = *idx;
        MFEM_VERIFY(index >= 0 && index < static_cast<int>(input_fields.size()),
                    "FieldCollection::InputField: Invalid index for field name: "
                    << field_name << ".");
        return input_fields[index];
    }

    Field* OutputField(int i) const { return output_fields[i]; }
    Field *OutputField(const std::string &field_name) const
    {
        auto idx = index_map.Get(field_name);
        if(!idx)
        {
            MFEM_WARNING("FieldCollection::OutputField: Field with name "
                         << field_name << " does not exist in the collection.");
            return nullptr;
        }

        int index = *idx;
        MFEM_VERIFY(index >= 0 && index < static_cast<int>(input_fields.size()),
                    "FieldCollection::OutputField: Invalid index for field name: "
                    << field_name << ".");
        return output_fields[index];
    }

    NamedFieldsMap<Field> &Fields() { return fields; }
    NamedFieldsMap<Field> Fields() const { return fields; }

    virtual void Save (std::ostream &out) const
    {
        out << "\"Fields\":\n";
        out << "{\n";
        for (auto f = fields.begin(); f != fields.end(); ++f)
        {
            std::string f_name = f->first;
            Field *f_obj = f->second;
            // out << "  " << f_name << ": ID " << f_obj->ID() << ",\n";
            // out << f_obj->ID() << ": " << f_name << ",\n";
            out << '\"' << f_obj->ID() << "\": \"" << f_name << "\"";
            if(f != std::prev(fields.end())) out << ",";
            out << "\n";
        }
        out << "},\n";

        out << "\"Inputs\":\n";
        out << "{\n";
        for (size_t i = 0; i < input_fields.size(); ++i)
        {
            Field *f_obj = input_fields[i];
            out << '\"' << f_obj->ID() << "\": \"" << f_obj->Name() << "\"";
            if(i != input_fields.size() - 1) out << ",";
            out << "\n";
        }
        out << "},\n";

        out << "\"Outputs\":\n";
        out << "{\n";
        for (size_t i = 0; i < output_fields.size(); ++i)
        {
            Field *f_obj = output_fields[i];
            out << '\"' << f_obj->ID() << "\": \"" << f_obj->Name() << "\"";
            if(i != output_fields.size() - 1) out << ",";
            out << "\n";
        }


        out << "}\n";
    }

    Field* HasField(const Field &field) const
    {
        for (auto f = fields.begin(); f != fields.end(); ++f)
        {
            if(f->second == &field)
            {
                return f->second;
            }
        }
        return nullptr;
    }

    Field* HasField(const std::string &field_name) const
    {
        return fields.Get(field_name);
    }

    Field* HasField(const int id) const
    {
        for (auto f = fields.begin(); f != fields.end(); ++f)
        {
            if(f->second->ID() == id)
            {
                return f->second;
            }
        }
        return nullptr;
    }

    ~FieldCollection(){}

};


class GraphNode : public Operator
{
public:
    enum ExecutionMode
    {
        GRADIENT_MODE, ///< Node is being executed as part of a gradient evaluation
        DEFAULT_MODE   ///< Node is being executed in default mode (e.g. operator evaluation)
    };

private:
    inline static int next_id = 0;

protected:
    int id = -1;
    int node_index = std::numeric_limits<int>::min();
    mutable ExecutionMode exec_mode = DEFAULT_MODE;

    std::string name;
    FieldCollection field_collection; // Collection of fields associated with this node

    int GetValidID(int id_, int lb=0, int ub = std::numeric_limits<int>::max())
    {
        return (id_ >= lb && id_ <= ub) ? id_ : next_id++;
    }

public:

    GraphNode(int h, int w) : Operator(h,w), id(GetValidID(-1)),
                              name("Node_" + std::to_string(id)),
                              field_collection(this) { }

    GraphNode(int s = 0) : GraphNode(s, s) { }

    virtual void Execute(const Vector &x, Vector &y)
    {
        MFEM_ABORT("GraphNode::Execute() not implemented");
    }

    virtual void Mult(const Vector &x, Vector &y) const override
    {
        MFEM_ABORT("GraphNode::Mult() not implemented");
    }

    void SetNodeIndex(int index){ node_index = index; }
    int GetNodeIndex() const { return node_index; }

    void SetExecutionMode(ExecutionMode mode) { exec_mode = mode; }
    ExecutionMode GetExecutionMode() const { return exec_mode; }

    void SetName(const std::string &name_) { name = name_; }
    std::string Name() const { return name; }

    void SetID(int id_) { id = id_; }
    int ID() const { return id; }

    NamedFieldsMap<Field>& Fields() { return field_collection.Fields(); }
    Field* Fields(const std::string &f) { return field_collection.GetField(f); }

    NamedFieldsMap<Field> Fields() const { return field_collection.Fields(); }
    Field* Fields(const std::string &f) const { return field_collection.GetField(f); }

    std::vector<Field*>& InputFields() { return field_collection.InputFields(); }
    std::vector<Field*>& OutputFields() { return field_collection.OutputFields(); }
    Field* InputField(int i) const { return field_collection.InputField(i); }
    Field* OutputField(int i) const { return field_collection.OutputField(i); }


    virtual void AddInput(const std::string &field_name,
                          Field *field, bool own = false)
    { field_collection.AddInput(field_name, field, own); }

    virtual void AddInput(Field *field, bool own = false)
    { AddInput(field->Name(), field, own); }

    template<bool OwnInputs = false,
             typename... Args,
             bool AreFields = std::conjunction<std::is_base_of<Field, std::remove_pointer_t<Args>> ...>::value,
             typename std::enable_if<AreFields, bool>::type = true >
    void AddInputs(Args... args)
    {
        ((AddInput(std::forward<Args>(args), OwnInputs)), ...);
    }

    virtual void AddOutput(const std::string &field_name,
                           Field *field, bool own = false)
    { field_collection.AddOutput(field_name, field, own); }

    virtual void AddOutput(Field *field, bool own = false)
    { AddOutput(field->Name(), field, own); }

    template<bool OwnOutputs = false,
             typename... Args,
             bool AreFields = std::conjunction<std::is_base_of<Field, std::remove_pointer_t<Args>> ...>::value,
             typename std::enable_if<AreFields, bool>::type = true >
    void AddOutputs(Args... args)
    {
        ((AddOutput(std::forward<Args>(args), OwnOutputs)), ...);
    }


    virtual void Save (std::ostream &out) const
    {
        out << "\"Node-" << id << "\" : " << std::endl;
        out << "{\n";
        out << "\"Name\": \"" << name << "\",\n";
        field_collection.Save(out);
        out << "}";
    }

    virtual void GradientMult(const Vector &x, const Vector &dx, Vector &dy) const
    {
        MFEM_ABORT("GraphNode::GradientMult() not implemented");
    }

    virtual void GradientMultTranspose(const Vector &x, const Vector &dx, Vector &dy) const
    {
        MFEM_ABORT("GraphNode::GradientMultTranspose() not implemented");
    }

    using Operator::GetGradient;
    virtual Operator &GetGradient(Field* fy, Field* fx, Vector &x) const
    {
        MFEM_ABORT("GraphNode::GetGradient() not implemented");
    }

    virtual void operator()(const Vector &x, Vector &y) const
    {
        Mult(x, y);
    }

    virtual void operator()(const Vector &x0, const Vector &x, Vector &y) const
    {
        MFEM_ABORT("GraphNode::operator()(const Vector&, const Vector&, Vector&) not implemented");
    }

    virtual ~GraphNode() = default;
};


/**
   @brief An abstract, type-erased class to define the interface for 
   operators, not inherited from @a GraphNode. It performs SFINAE 
   checks for stored operator's member functions and override the Mult
   to call the stored object's functions.
 */
template <typename OpType>
class AbstractOperator : public GraphNode
{
protected:
    /// Define a template class 'check' to test for the existence of member functions
    template <typename C>
    class CheckMember{
        private:

        /// @brief A type trait to check if the erased class has the functions Execute and Mult
        /// with the needed signatures.
        template<class T>
        using Execute = decltype(std::declval<T&>().Execute(std::declval<const Vector&>(),
                                                            std::declval<Vector&>()));

        template<class T>
        using ExecutePtr = decltype(std::declval<T&>().Execute(std::declval<const int>(),
                                                               std::declval<const real_t*>(),
                                                               std::declval<const int>(),
                                                               std::declval<real_t*>()));

        template<class T>
        using Mult = decltype(std::declval<T&>().Mult(std::declval<const Vector&>(),
                                                      std::declval<Vector&>()));

        template<class T>
        using MultPtr = decltype(std::declval<T&>().Mult(std::declval<const int>(),
                                                         std::declval<const real_t*>(),
                                                         std::declval<const int>(),
                                                         std::declval<real_t*>()));
        // ---------------------------------------------------------------------
        
        template <typename T, template<typename> typename Func, typename R>
        static constexpr auto Check(T*) -> typename std::is_same< Func<T>, R>::type;

        template <typename, template<typename> typename, typename >
        static constexpr std::false_type Check(...);

        // --- Check for the existence of the member functions
        typedef decltype(Check<C,Execute,void>(0)) Has_Execute;
        typedef decltype(Check<C,Mult,void>(0)) Has_Mult;

        typedef decltype(Check<C,ExecutePtr,void>(0)) Has_ExecutePtr;
        typedef decltype(Check<C,MultPtr,void>(0)) Has_MultPtr;
    public:
        static constexpr bool HasExecute  = Has_Execute::value;
        static constexpr bool HasMult  = Has_Mult::value;
        static constexpr bool HasExecutePtr  = Has_ExecutePtr::value;
        static constexpr bool HasMultPtr  = Has_MultPtr::value;
    };

    OpType *op;  ///< Pointer to the operator

public:

    constexpr bool HasExecute(){return CheckMember<OpType>::HasStep;}
    constexpr bool HasMult(){return CheckMember<OpType>::HasMult;}


    /// @brief Constructor for the type-erased AbstractOperator class
    AbstractOperator(OpType *op_, int h, int w) : GraphNode(h,w), op(op_)
    { }

    /// @brief Constructor for the type-erased AbstractOperator class.
    AbstractOperator(OpType *op_, int s = 0) : AbstractOperator(op_,s,s) {}

    /**
       @brief Perform Mult operation with the stored operator, if it exists.
     */
    void Execute(const Vector &x, Vector &y) override
    {
        if constexpr (CheckMember<OpType>::HasExecute)
        {
            op->Execute(x,y);
        }
        else if constexpr (CheckMember<OpType>::HasExecutePtr)
        {
            op->Execute(x.Size(), x.GetData(), y.Size(), y.GetData());
        }
        else
        {
            MFEM_ABORT("The AbstractOperator does not have the function, "
                       "Execute(const Vector&, Vector&) or "
                       "Execute(int, double*, int, double*).");
        }
    }

    /**
       @brief Perform Mult operation with the stored operator, if it exists.
     */
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
            MFEM_ABORT("The AbstractOperator does not have the function, "
                       "Mult(const Vector&, Vector&) or "
                       "Mult(int, double*, int, double*).");
        }
    }
};

class DataNode : public GraphNode
{
protected:
    Field *field = nullptr;

public:

    DataNode(Field &f, int sz, std::string name = "") : GraphNode(sz), field(&f)
    {
        if(!name.empty()) SetName(name);
        // field_collection.AddField(name, field, false);
    }

    Field* GetField() const { return field; }

    virtual void SetData(const Vector &v)
    {
        Vector *vec = field->Data();
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        MFEM_ASSERT(vec != nullptr, "Input field data is not set.");
        *vec = v;
    }

    virtual void GetData(Vector &v) const
    {
        Vector *vec = field->Data();
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        MFEM_ASSERT(vec != nullptr, "Input field data is not set.");
        v = *vec;
    }

    virtual void SetAdjoint(const Vector &v)
    {
        Vector *vec = field->Adjoint();
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        MFEM_ASSERT(vec != nullptr, "Input field adjoint is not set.");
        *vec = v;
    }

    virtual void GetAdjoint(Vector &v) const
    {
        Vector *vec = field->Adjoint();
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        MFEM_ASSERT(vec != nullptr, "Input field adjoint is not set.");
        v = *vec;
    }

private: // Hide all other functions from user
    using GraphNode::Fields;
    using GraphNode::AddInput;
    using GraphNode::AddOutput;
    using GraphNode::AddInputs;
    using GraphNode::AddOutputs;
    using GraphNode::InputField;
    using GraphNode::OutputField;
    using GraphNode::InputFields;
    using GraphNode::OutputFields;
    using GraphNode::Execute;
    using GraphNode::Mult;
    using GraphNode::GetGradient;
    using GraphNode::GradientMult;
    using GraphNode::GradientMultTranspose;
};

/**
   @brief A class to store and coupled multiple operators together.
 */
class DAGraph : public GraphNode
{
public:
    enum GradMode
    {
        FD, // Finite difference Jacobian
        FORWARD,
        BACKWARD,
        JACOBIAN
    };

protected:
    Array<GraphNode*> nodes; ///< Vector of individual operators
    Array<bool> node_owned; ///< Whether the operators are owned
    Array<int> node_depth; ///< Depth of each operator in the graph


    Array<int> in_offsets;  ///< Block offsets for input fields
    Array<int> out_offsets; ///< Block offsets for output fields
    int max_width=0;        ///< Largest operator width
    int max_height=0;       ///< Largest operator height
    int nnodes = 0;         ///< The number of nodes
    bool sorted = false;    ///< Whether the nodes are topologically sorted
    
    mutable Operator *grad = nullptr; ///< Jacobain operator
    GradMode grad_mode = GradMode::FD;
    mutable Vector fx; ///< Temporary vector for function evaluation
    // mutable Vector dx, dy; ///< Temporary vectors for Jacobian computations

    // Input and output data nodes
    std::vector<DataNode*> input_nodes;
    std::vector<DataNode*> output_nodes;

    mutable Vector ytmp; ///< Temporary vector (used in forward pass in gradient computations)
    mutable Vector xgrad; ///< Point of linearization for gradient computations

    friend class GraphGradient;

public:
    /**
       @brief Construct a new CoupledOperator object.
       @param nop Total number of operators to couple
     */
    DAGraph(const int nop) : GraphNode()
    {
        nodes.Reserve(nop);
        node_owned.Reserve(nop);

        in_offsets.Reserve(nop+1);
        out_offsets.Reserve(nop+1);
        in_offsets.Prepend(0);
        out_offsets.Prepend(0);
    }

    /**
       @brief Construct a new CoupledOperator object for an 
       abstract non/mfem operator.
     */
    template <class OpType>
    DAGraph(const OpType &op) : DAGraph(1)
    {
        AddOperator(op);
    }

    /**
       @brief Add an operator to the list of coupled operator and
       return pointer to it. Not owned unless it's not derived from GraphNode.
     */
    template <class OpType>
    GraphNode* AddOperator(OpType *op_, int h, int w)
    {
        // Add operator to list of operators
        if constexpr(std::is_base_of<GraphNode, OpType>::value)
        {
            nodes.push_back(op_);
            node_owned.Append(false);
        } 
        else
        {
            nodes.push_back(new AbstractOperator<OpType>(op_,h,w));
            node_owned.Append(true);
        }
        nnodes++;

        // Update size of the coupled operator and the block offsets
        GraphNode* op = nodes.Last();
        op->SetNodeIndex(nnodes-1); // Set the index of the operator

        int ht = op->Height();
        int wt = op->Width();

        max_width = std::max(max_width, wt);
        max_height = std::max(max_height, ht);
        sorted = false;

        return op;
    }

    /// @brief Add an operator to the list of coupled operator and return pointer to it.
    template <class OpType>
    GraphNode* AddOperator(OpType *op_, int s = 0) { return AddOperator(op_,s,s);}

    //TODO: Support ownership option
    DataNode* AddInputNode(DataNode *node, bool own = false)
    {
        int index = input_nodes.size();
        node->SetNodeIndex(index);
        in_offsets.Append(in_offsets.Last() + node->Width());
        width += node->Width();

        auto field = node->GetField();
        if(field)
        {   // Add the node's field to the DAG's
            field_collection.AddField(node->Name(), field, false);
        }

        input_nodes.push_back(node);
        return node;
    }

    //TODO: Support ownership option
    DataNode* AddOutputNode(DataNode *node, bool own = false)
    {
        int index = output_nodes.size();
        node->SetNodeIndex(index);
        out_offsets.Append(out_offsets.Last() + node->Height());
        height += node->Height();

        auto field = node->GetField();
        if(field)
        {   // Add the node's field to the DAG's
            field_collection.AddField(node->Name(), field, false);
        }

        output_nodes.push_back(node);
        return node;
    }

    /// @brief Get the number of coupled operators
    int Size(){return nnodes;}

    /// @brief Get the size of the largest operator
    int MaxWidth(){return max_width;}
    int MaxHeight(){return max_height;}

    /// @brief Get the operator at index @a i
    GraphNode* GetNode(const int i)
    {
        MFEM_ASSERT(i >= 0 && i < nnodes,
               "index [" << i << "] is out of range [0," << nnodes << ")");
        return nodes[i];
    }

    /// @brief Specify whether the operator at index @a i is owned.
    void OwnNode(const int i, bool own = true)
    {
        MFEM_ASSERT(i >= 0 && i < nnodes,
               "index [" << i << "] is out of range [0," << nnodes << ")");
        node_owned[i] = own;
    }

    void Assemble();

    void TopologicalSort();

    void ComputeDepth();

    /// @brief Set the gradient mode for the coupled operator
    void SetGradientMode(GradMode mode)
    {
        if(mode != grad_mode)
        {
            if(grad) { delete grad; grad = nullptr; }
            grad_mode = mode;
        }
    }

    /// @brief Return the input offsets for block starts.
    Array<int>& InputOffsets() { return in_offsets; }

    /// @brief Read only access to the input offsets for block starts.
    const Array<int>& InputOffsets() const { return in_offsets; }

    /// @brief Return the output offsets for block starts.
    Array<int>& OutputOffsets() { return out_offsets; }

    /// @brief Read only access to the output offsets for block starts.
    const Array<int>& OutputOffsets() const { return out_offsets; }

    /**
       @brief Apply the operator to the vector @a x 
       and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const override;

    virtual void Execute(const Vector &x, Vector &y) override;

    virtual void Save (std::ostream &out) const
    {
        out << "\"DAGraph\":\n";
        out << "{\n";
        // out << "\"nodes\" : " << nnodes << ",\n";
        out << "\"Nodes\":\n";
        out << "{\n";
        for (int i = 0; i < nodes.Size(); i++)
        {
            nodes[i]->Save(out);
            if(i != nodes.Size()-1) out << ",";
            out << "\n";
        }
        out << "},\n"; // End of Nodes
        field_collection.Save(out);
        out << "}\n";
    }

    Operator& GetGradient(const Vector &x) const override;

    /// @brief Destroy the Coupled Application object
    ~DAGraph();
};



class GraphGradient : public Operator
{
public:
    using GradMode = DAGraph::GradMode;

protected:
    mutable DAGraph *graph = nullptr; ///< Pointer to the DAGraph for which this is the gradient operator
    mutable GradMode grad_mode; ///< Gradient mode

public:
    GraphGradient(DAGraph *graph_, GradMode mode = GradMode::FORWARD) :
                  Operator(graph_->Height(), graph_->Width()),
                  graph(graph_), grad_mode(mode) {}

    void Mult(const Vector &x, Vector &y) const override;

    Operator &GetGradient(const Vector &x) const override;

    virtual void Forward(const Vector &x, Vector &y) const;

    virtual void Backward(const Vector &x, Vector &y) const;

    void SetGradientMode(GradMode mode) { grad_mode = mode; }

    GradMode GetGradientMode() const { return grad_mode; }

    ~GraphGradient() = default;
};


} //mfem namespace

#endif // MFEM_USE_MPI

#endif
