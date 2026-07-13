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
class FieldEdge;
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
        SOURCE, ///< Source field
        TARGET, ///< Target field
        DEFAULT ///< Any field
    };

    friend class GraphNode;
    friend class FieldEdge;

private:
    Type type = Type::DEFAULT;
    inline static int next_id = 0;

protected:
    Vector *data = nullptr;
    Vector *adjoint = nullptr; // For storing derivative info
    int id = -1; // initialized to invalid id

    Operator *node  = nullptr; // Node that owns this field
    FieldEdge *edge = nullptr; // Edge for this source field, if applicable
    Field *source = nullptr; // source field for this target field, if applicable

    int GetValidID(int id_, int lb=0, int ub = std::numeric_limits<int>::max())
    {
        return (id_ >= lb && id_ <= ub) ? id_ : next_id++;
    }

public:

    ///@brief Constructor for a Field of type Type with optional ID
    Field(Vector *field, Vector *adjoint, Type type, int id_ = -1) :
          type(type), data(field), adjoint(adjoint), id(GetValidID(id_)) { }

    ///@brief Constructor for a Source field
    Field(Vector *field, int id_ = -1) :
          Field(field, nullptr, Type::SOURCE, id_) { }

    ///@brief Constructor for a Field of type Type
    Field(Vector *field, Type type, int id_ = -1) :
          Field(field, nullptr, type, id_) { }

    ///@brief Get the stored internally stored data pointer
    Vector* Data() const { return data; }
    Vector* Adjoint() const { return adjoint; }
    Operator* Node() const { return node; }
    Field* SourceField() const { return source; }
    FieldEdge* GetEdge() const { return edge; }

    ///@brief Set the internally stored data pointer
    virtual void SetData(Vector *field) { data = field; }
    virtual void SetAdjoint(Vector *adj) { adjoint = adj; }
    virtual void SetNode(Operator *op) { node = op; }
    void SetSource(Field *src) { source = src; }
    void SetEdge(FieldEdge *fe) { edge = fe; }

    int ID() const { return id; }

    void SetID(int id_)
    {
        MFEM_ASSERT(id_ >= 0, "Field::SetID: ID must be non-negative.");
        id = id_;
    }

    virtual Operator* SourceNode() const
    {
        if(IsSource())
        {
            MFEM_VERIFY(node != nullptr, "Source field: " << ID()
                        << " does not have an associated GraphNode.");
            return node;
        }
        else
        {
            MFEM_VERIFY(source != nullptr, "Field: " << ID()
                        << " does not have an associated source field.");
            MFEM_VERIFY(source->Node() != nullptr, "Source field: "
                        << source->ID() << " for field: " << ID()
                        << " does not have an associated GraphNode.");
            return source->Node();
        }
        return node;
    }

    void SetFieldEdge(FieldEdge *fe)
    {
        MFEM_ASSERT(IsSource(), "FieldEdge only associated with source fields. "
                    << "Field ID: " << id << " is not a source field.");
        edge = fe;
    }

    bool IsSource() const {return (type == Type::SOURCE);}
    bool IsTarget() const {return (type == Type::TARGET); }
    bool IsSourceOrTarget() const { return (type != Type::DEFAULT); }

    virtual ~Field() = default;

protected:
    virtual void MakeSource() { type = Type::SOURCE; }
    virtual void MakeTarget() { type = Type::TARGET; }

    void SetType(Type t)
    {
        if (type != t && (IsSourceOrTarget()))
        { // Warn changing source/target to other or default
            MFEM_WARNING("Changing field type from " << (IsSource() ? "SOURCE" : "TARGET")
                         << " to " << (t == Type::SOURCE ? "SOURCE" : (t == Type::TARGET ? "TARGET" : "DEFAULT"))
                         << " for field ID: " << ID());
        }
        // TODO: If SOURCE -> else; nullify field edge; if else -> SOURCE, nullify source field.
        type = t;
    }
};


/**
   @brief A class for edges from sources to multiple target fields.
 */
class FieldEdge
{
private:
    inline static int next_id = 0;

protected:
    int id = -1;

    Field *source = nullptr;
    bool own_source = false;
    std::vector<Field*> targets;
    std::vector<bool> targets_owned;

    int GetValidID(int id_, int lb=0, int ub = std::numeric_limits<int>::max())
    {
        return (id_ >= lb && id_ <= ub) ? id_ : next_id++;
    }

public:

    FieldEdge(int id_ = -1) : id(GetValidID(id_)) { }

    /**
     * @brief Construct a new FieldEdge with only a source and empty target
     */
    FieldEdge(Field *src, bool own=false, int id_ = -1) :
              id(GetValidID(id_)), source(src), own_source(own)
    {
        if(source)
        {
            source->SetFieldEdge(this);
            source->SetType(Field::Type::SOURCE);
        }
    }

    FieldEdge(Field *src, Field *tar, bool own_src=false, bool own_tar=false, int id_ = -1) :
              FieldEdge(src, own_src, id_)
    {
        AddTarget(tar, own_tar);
    }

    virtual void Execute(const Vector &x, Vector &y) 
    {
        MFEM_ABORT("FieldEdge::Execute() not implemented");
    }

    int ID() const { return id; }

    void SetID(int id_)
    {
        MFEM_ASSERT(id_ >= 0, "FieldEdge::SetID: ID must be non-negative.");
        id = id_;
    }

    /**
       @brief Set the source @a Field.
       @param src Source  @a Field
     */
    void SetSource(Field *src, bool own=false)
    {
        if(own_source && source) delete source;
        source = src;
        own_source = own;
        if(source)
        {
            source->SetFieldEdge(this);
            source->SetType(Field::Type::SOURCE);
        }
    }

    ///@brief Get the source @a Field
    Field* SourceField() const { return source; }

    ///@brief Adds the target @a Field, @a tar, to the list of targets
    virtual void AddTarget(Field *tar, bool own=false)
    {
        tar->SetType(Field::Type::TARGET);
        targets.push_back(tar);
        targets_owned.push_back(own);
        if(source)
        {
            tar->SetID(source->ID());
            tar->SetSource(source);
            Vector *srcv = source->Data();
            Vector *tarv = tar->Data();
            if(srcv && tarv)
            {
                // Make target data a reference to source data
                tarv->SetSize(srcv->Size());
                tarv->MakeRef(*srcv,0);
            }

            // Make target adjoint a reference to source adjoint if it exists
            Vector *src_adj = source->Adjoint();
            Vector *tar_adj = tar->Adjoint();
            if(src_adj && tar_adj)
            {
                tar_adj->SetSize(src_adj->Size());
                tar_adj->MakeRef(*src_adj,0);
            }
        }
    }

    ///@brief Get all target fields
    std::vector<Field*>& Targets() { return targets; }

    bool HasTargets() const { return !targets.empty(); }

    virtual ~FieldEdge()
    {
        for (size_t i=0; i < targets.size(); i++)
        {
            if(targets_owned[i] && targets[i]) delete targets[i];
        }
        if(own_source && source) delete source;
    }
};



/// @brief A collection of Fields and FieldEdge, each identified by a name
class FieldCollection
{
private:
    std::string name; /// Name of the collection
    Operator *src_op = nullptr; /// Source operator (not owned)

    /// Fields for source operator. Contains all source fields and fields that
    /// may be targets of other operator.
    NamedFieldsMap<Field> fields;

    /// FieldEdge for source operator.
    NamedFieldsMap<FieldEdge> edges;

public:

    FieldCollection() = default;

    /// @brief Constructor with collection name and optional source operator
    FieldCollection(std::string collection_name, Operator *op = nullptr):
                    name(collection_name), src_op(op) {}

    /// @brief Constructor with source operator
    FieldCollection(Operator *src) : name("FieldCollection"), src_op(src) {}

    /// @brief Get the number of linked fields in the collection
    int Size() const { return edges.NumFields(); }

    /// @brief Set the name of the collection
    void SetName(const std::string &collection_name) { name = collection_name;}

    /// @brief Get the name of the collection
    std::string Name() const { return name; }

    /// @brief Set the source operator
    void SetOperator(Operator *op){ src_op = op; }

    /// @brief Get the source operator
    const Operator* GetOperator() const { return src_op; }

    /// @brief Get the ParGridFunction for a given source name
    Field *GetSourceField(const std::string &src_name) const
    {
        FieldEdge *edge = edges.Get(src_name);
        if(!edge)
        {
            // MFEM_WARNING("FieldCollection::GetSourceField: Source field "
            //              + src_name + " not found!");
            return nullptr;
        }
        return edge->SourceField();
    }

    /// @brief Get the ParGridFunction for a given field name
    Field* GetField(const std::string &field_name) const
    {
        return fields.Get(field_name);
    }

    FieldEdge* GetFieldEdge(const std::string &src_name) const
    {
        return edges.Get(src_name);
    }

    /// @brief Add a ParGridFunction as a field (does not specify source or target)
    void AddField(const std::string &field_name, Field *field, bool own = false)
    {
        fields.Register(field_name, field, own);
        if(field->Node() == nullptr)
        {
            field->SetNode(src_op);
        }
    }

    /// @brief Add a FieldEdge to the collection with name src_name
    void AddFieldEdge(const std::string &src_name,
                      FieldEdge *edge, bool own = false)
    {
        FieldEdge *edge_exist = edges.Get(src_name);
        if(edge_exist)
        {
            auto targets = edge->Targets();
            for (auto &dest : targets) {
                // auto [target, owned] = dest;
                // edge_exist->AddTarget(target, owned);
                MFEM_ABORT("TO DO")
            }
            return;
        }
        edges.Register(src_name, edge, own);
        fields.Register(src_name, edge->SourceField(), false);
    }

    void AddSourceField(const std::string &src_name, Field *src, bool own=false)
    {
        fields.Register(src_name, src, false);
        if(src->Node() == nullptr)
        {
            src->SetNode(src_op);
        }
        FieldEdge *edge = edges.Get(src_name);
        if(!edge)
        {
            edge = new FieldEdge(src, own);
            edges.Register(src_name, edge, true);
            return;
        }
        edge->SetSource(src, own);
    }

    void AddTargetField(const std::string &src_name, Field *tar,
                        bool own = false)
    {
        FieldEdge *edge = edges.Get(src_name);
        if(!edge)
        {
            edge = new FieldEdge();
            edges.Register(src_name, edge, true);
        }
        edge->AddTarget(tar, own);
    }

    Field* operator[](const std::string &field_name) const
    {
        return GetField(field_name);
    }

    NamedFieldsMap<Field> &GetFields() { return fields; }
    NamedFieldsMap<FieldEdge> &GetEdges() { return edges; }

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
        out << "\"FieldEdge\":\n";
        out << "{\n";
        for (auto edge_pair = edges.begin(); edge_pair != edges.end(); ++edge_pair)
        {
            std::string edge_name = edge_pair->first;
            FieldEdge *edge = edge_pair->second;
            // out << "  " << lf_name << ": ID " << lf_obj->ID() << ",\n";
            out << '\"' << edge->SourceField()->ID() << "\": \"" << edge_name << "\"";
            if(edge_pair != std::prev(edges.end())) out << ",";
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
    FieldCollection fields; // Collection of fields associated with this node

    int GetValidID(int id_, int lb=0, int ub = std::numeric_limits<int>::max())
    {
        return (id_ >= lb && id_ <= ub) ? id_ : next_id++;
    }

public:

    GraphNode(int s = 0) : Operator(s), id(GetValidID(-1)), fields(this)
    { }

    GraphNode(int h, int w) : Operator(h,w), id(GetValidID(-1)), fields(this)
    { }

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

    FieldCollection& Fields() { return fields; }
    const FieldCollection& Fields() const { return fields; }

    Field* Fields(const std::string &field_name) const 
    { return fields.GetField(field_name); }

    Field* Fields(const std::string &field_name)
    { return fields.GetField(field_name); }

    FieldEdge* Edge(const std::string &src_name) const
    { return fields.GetFieldEdge(src_name); }

    FieldEdge* Edge(const std::string &src_name)
    { return fields.GetFieldEdge(src_name); }

    void AddField(const std::string &field_name, Field *field, bool own = false)
    {
        fields.AddField(field_name, field, own);
    }

    /// @brief Add a FieldEdge to the collection with name src_name
    void AddFieldEdge(const std::string &src_name, FieldEdge *field)
    {
        fields.AddFieldEdge(src_name, field);
    }

    virtual void Save (std::ostream &out) const
    {
        out << "\"Node-" << id << "\" : " << std::endl;
        out << "{\n";
        out << "\"Name\": \"" << name << "\",\n";
        fields.Save(out);
        out << "}";
    }

    virtual void JVP(const Vector &x, Vector &y) const
    {
        MFEM_ABORT("GraphNode::JVP() not implemented");
    }

    virtual void VJP(const Vector &x, Vector &y) const
    {
        MFEM_ABORT("GraphNode::VJP() not implemented");
    }

    virtual void GetJacobian(Field* y, Field* x, Vector &x0, Operator *dydx)
    {
        MFEM_ABORT("GraphNode::GetJacobian() not implemented");
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
    Vector data, adjoint;
    Field *field = nullptr;

public:

    DataNode(std::string name, int sz) : GraphNode(sz)
    {
        SetName(name);
    }

    DataNode(std::string name, int sz, Field::Type type) : DataNode(name, sz)
    {
        field = new Field(&data, &adjoint, type);
        fields.AddField(name, field, true); // transfer ownership
    }

    Field* GetField() const { return field; }

    virtual void SetData(const Vector &v)
    {
        MFEM_ABORT("DataNode::SetData() not implemented.");
    }

    virtual void GetData(Vector &v) const
    {
        MFEM_ABORT("DataNode::GetData() not implemented.");
    }

    virtual void SetAdjoint(const Vector &v)
    {
        MFEM_ABORT("DataNode::SetAdjoint() not implemented.");
    }

    virtual void GetAdjoint(Vector &v) const
    {
        MFEM_ABORT("DataNode::GetAdjoint() not implemented.");
    }

private: // Hide all other functions from user
    using GraphNode::Execute;
    using GraphNode::Mult;
    // using GraphNode::GetDerivative;
    using GraphNode::JVP;
    using GraphNode::VJP;
};

class InputNode : public DataNode
{
public:
    InputNode(std::string name, int sz) : DataNode(name, sz)
    {
        data.SetSize(sz);
        adjoint.SetSize(sz);
        field = new Field(&data, &adjoint, Field::Type::SOURCE);
        fields.AddSourceField(name, field, true); // transfer ownership
    }

    void AddTargetField(Field *target, bool own=false)
    {
        fields.AddTargetField(Name(), target, own);
    }

    void SetData(const Vector &v) override
    {
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        data = v; 
    }

    void GetData(Vector &v) const override
    {
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        v = data;
    }

    void SetAdjoint(const Vector &v) override
    {
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        adjoint = v; 
    }

    void GetAdjoint(Vector &v) const override
    {
        MFEM_ASSERT(v.Size() == Width(), "Vector size does not match node size.");
        v = adjoint;
    }
};

class OutputNode : public DataNode
{
public:
    OutputNode(std::string name, int sz) : DataNode(name, sz)
    {
        data.SetSize(sz);
        adjoint.SetSize(sz);
        field = new Field(&data, &adjoint, Field::Type::TARGET);
        fields.AddField(name, field, true); // transfer ownership
    }

    void SetData(const Vector &v) override
    {
        MFEM_ASSERT(v.Size() == Height(), "Vector size does not match node size.");
        data = v;
    }

    void GetData(Vector &v) const override
    {
        MFEM_ASSERT(v.Size() == Height(), "Vector size does not match node size.");
        v = data;
    }

    void SetAdjoint(const Vector &v) override
    {
        MFEM_ASSERT(v.Size() == Height(), "Vector size does not match node size.");
        adjoint = v;
    }

    void GetAdjoint(Vector &v) const override
    {
        MFEM_ASSERT(v.Size() == Height(), "Vector size does not match node size.");
        v = adjoint;
    }
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
    std::vector<GraphNode*> nodes;  ///< Vector of individual operators
    Array<bool> nodes_owned; ///< Whether the operators are owned

    Array<int> in_offsets;  ///< Block offsets for input fields
    Array<int> out_offsets; ///< Block offsets for output fields
    int max_width=0;        ///< Largest operator width
    int max_height=0;       ///< Largest operator height
    int nnodes = 0;         ///< The number of nodes
    
    mutable Operator *grad = nullptr; ///< Jacobain operator
    GradMode grad_mode = GradMode::FD;
    bool own_blocks = false; ///< Whether the BlockOperator owns the individual blocks

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
        nodes.reserve(nop);
        nodes_owned.Reserve(nop);

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
            nodes_owned.Append(false);
        } 
        else
        {
            nodes.push_back(new AbstractOperator<OpType>(op_,h,w));
            nodes_owned.Append(true);
        }
        nnodes++;

        // Update size of the coupled operator and the block offsets
        GraphNode* op = nodes.back();
        op->SetNodeIndex(nnodes-1); // Set the index of the operator

        int ht = op->Height();
        int wt = op->Width();

        max_width = std::max(max_width, wt);
        max_height = std::max(max_height, ht);

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

        auto edge = node->Edge(node->Name());
        if(edge)
        {   // Add the node's linkefield to the DAG's
            fields.AddFieldEdge(node->Name(), edge, false);
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

        auto field = node->Fields(name);
        if(field)
        {   // Add the node's target field to the DAG's
            fields.AddField(name, field, false);
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
    GraphNode* GetNode(const int i) { return nodes[i]; }

    /// @brief Specify whether the operator at index @a i is owned.
    void OwnNode(const int i, bool own = true)
    {
        MFEM_ASSERT(i >= 0 && i < nnodes,
               "index [" << i << "] is out of range [0," << nnodes << ")");
        nodes_owned[i] = own;
    }

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
        for (size_t i = 0; i < nodes.size(); i++)
        {
            nodes[i]->Save(out);
            if(i != nodes.size()-1) out << ",";
            out << "\n";
        }
        out << "},\n"; // End of Nodes
        fields.Save(out);
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







/**
   @brief MFEM SubMesh transfer between two FiniteElementSpaces 
   using ParTransferMap.
   NOT USED YET

class SubMeshTransfer : public GraphNode
{
protected:
    ParFiniteElementSpace *src_fes = nullptr, *tar_fes = nullptr;
    ParTransferMap *transfer_map = nullptr;
    bool own_map = false;

public:

    SubMeshTransfer(ParFiniteElementSpace *src,
                    ParFiniteElementSpace *tar) : GraphNode(),
                    src_fes(src), tar_fes(tar),
                    transfer_map(new ParTransferMap(src_fes, tar_fes)), own_map(true) {}

    SubMeshTransfer(ParGridFunction *src, ParGridFunction *tar) :
                    SubMeshTransfer(src->ParFESpace(), tar->ParFESpace()) {}

    void SetTransferMap(ParTransferMap *map, bool own=false)
    {
        if(own_map && transfer_map) delete transfer_map;
        transfer_map = map;
        own_map = own;
    }

    void Execute(const Vector &src, Vector &tar) override
    {
        MFEM_ASSERT(transfer_map != nullptr, "SubMeshTransfer::Execute: transfer map not set!");

        // Loop through all the edges and perform the operator
        NamedFieldsMap<FieldEdge> &edges = Fields().GetEdges();
        for (auto edge_pair = edges.begin(); edge_pair != edges.end(); ++edge_pair)
        {
            std::string edge_name = edge_pair->first;
            FieldEdge *edge = edge_pair->second;
            ParGridFunction &src_gf = dynamic_cast<ParGridFunction&>(*edge->SourceField()->Data());

            // Loop through all the targets for this source field and perform the transfer
            auto &targets = edge->GetTargets();
            for (auto &target : targets)
            {
                ParGridFunction &tar_gf = dynamic_cast<ParGridFunction&>(*target->Data());
                transfer_map->Transfer(src_gf, tar_gf);
            }
        }
    }

    ~SubMeshTransfer()
    {
        if(own_map && transfer_map) delete transfer_map;
    }
};
 */




} //mfem namespace

#endif // MFEM_USE_MPI

#endif
