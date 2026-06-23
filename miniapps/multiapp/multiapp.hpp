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
class LinkedFields;
class FieldCollection;
class GraphNode;
class Application;
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
    friend class LinkedFields;

private:
    Type type = Type::DEFAULT;
    inline static int next_id = 0;
    // static int next_id;

protected:
    Vector *data = nullptr;
    Vector *adjoint = nullptr; // For storing derivative info
    GraphNode *node = nullptr;
    int id = -1; // initialized to invalid id

    Field *source = nullptr; // source field for this target field, if applicable
    LinkedFields *linked_fields = nullptr; // LinkedFields for this source field, if applicable

public:

    ///@brief Constructor for a Source field
    Field(Vector *field, int id_ = -1) : type(Type::SOURCE), data(field)
    {
        id = id_;
        if(id < 0) id = next_id++;
    }

    ///@brief Constructor for a Field of type Type
    Field(Vector *field, Type type, int id_ = -1) : type(type), data(field)
    {
        id = id_;
        if(id < 0) id = next_id++;
    }

    Field(Vector *field, Vector *adjoint, Type type, int id_ = -1) :
          type(type), data(field), adjoint(adjoint)
    {
        id = id_;
        if(id < 0) id = next_id++;
    }

    ///@brief Get the stored internally stored data pointer
    virtual Vector* Data() const { return data; }

    ///@brief Set the internally stored data pointer
    virtual void SetData(Vector *field) { data = field; }

    virtual Vector* Adjoint() const { return adjoint; }
    virtual void SetAdjoint(Vector *v) { adjoint = v; }

    virtual void SetNode(GraphNode *op) { node = op; }
    virtual GraphNode* GetNode() const { return node; }

    ///@brief Update the stored field with new values
    virtual void Update(const Vector &f) { }

    void SetID(int id_) { id = id_; }
    int GetID() const { return id; }

    void SetSource(Field *src) { source = src; }
    Field* GetSource() const { return source; }
    
    virtual GraphNode* GetSourceNode() const
    {
        if(IsSource())
        {
            MFEM_VERIFY(node != nullptr, "Source field: " << GetID()
                        << " does not have an associated GraphNode.");
            return node;
        }
        else
        {
            MFEM_VERIFY(source != nullptr, "Field: " << GetID()
                        << " does not have an associated source field.");
            MFEM_VERIFY(source->GetNode() != nullptr, "Source field: "
                        << source->GetID() << " for field: " << GetID()
                        << " does not have an associated GraphNode.");
            return source->GetNode();
        }
        return node;
    }

    void SetLinkedFields(LinkedFields *lf)
    {
        MFEM_ASSERT(IsSource(), "LinkedFields only associated with source fields. "
                    << "Field ID: " << id << " is not a source field.");
        linked_fields = lf;
    }
    LinkedFields* GetLinkedFields() const { return linked_fields; }

    virtual void GetDerivative(Field* x, Vector &x0, Vector &dydx);

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
                         << " for field ID: " << GetID());
        }
        // TODO: If SOURCE -> else; nullify linked fields; if else -> SOURCE, nullify source field.
        type = t;
    }
};


class FieldEdge 
{
protected:
    inline static int next_id = 0;
    int id = -1;

public:
    FieldEdge(int id_ = -1)
    {
        id = id_;
        if(id < 0) id = next_id++;
    }

    virtual void Execute(const Vector &x, Vector &y) 
    {
        MFEM_ABORT("FieldEdge::Execute() not implemented");
    }

    void SetID(int id_) { id = id_; }
    int GetID() const { return id; }

    virtual ~FieldEdge() = default;
};

/**
   @brief A class to link sources and multiple target fields.
   TODO: Fold this into FieldEdge
 */
class LinkedFields : public FieldEdge
{
protected:

    Field *source = nullptr;
    bool own_source = false;
    std::vector<Field*> targets;
    std::vector<bool> targets_owned;
    int ndest = 0;

public:

    LinkedFields(int id_ = -1) : FieldEdge(id_) {}

    /**
       @brief Constructor given a source @a Vector.

       @param src Source @a Vector
     */
    LinkedFields(Vector *src): FieldEdge(-1)
    {
        source = new Field(src, Field::Type::SOURCE);
        own_source = true;
        source->SetLinkedFields(this);
    }

    /**
     * @brief Construct a new LinkedFields with only a source and empty target
     */
    LinkedFields(Field *src, bool own=false) : FieldEdge(-1),
                 source(src), own_source(own)
    {
        if(source)
        {
            source->SetLinkedFields(this);
            source->SetType(Field::Type::SOURCE);
        }
    }


    /**
       @brief Constructor given a source and target @a Vector.

       @param src Source  @a Vector
       @param tar Target  @a Vector
     */
    LinkedFields(Vector *src, Vector *tar) : LinkedFields(src)
    {
        AddTarget(tar);
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
            source->SetLinkedFields(this);
            source->SetType(Field::Type::SOURCE);
        }
    }

    ///@brief Set the source @a Vector (does not own).
    void SetSource(Vector *src)
    {
        if(own_source && source) delete source;
        source = new Field(src, Field::Type::SOURCE);
        own_source = true;
        source->SetLinkedFields(this);
    }

    ///@brief Get the source @a Field
    Field* GetSource() const { return source; }

    ///@brief Adds the target @a Field, @a tar, to the list of targets
    void AddTarget(Field *tar, bool own=false)
    {
        tar->SetType(Field::Type::TARGET);
        targets.push_back(tar);
        targets_owned.push_back(own);
        if(source)
        {
            tar->SetID(source->GetID());
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
        ndest++;
    }

    /**
       @brief Adds the @a Vector, @a tar to the list of targets
     */
    void AddTarget(Vector *tar)
    {
        Field *target = new Field(tar, Field::Type::TARGET);
        AddTarget(target, true);
    }

    void UpdateTargets()
    {
        for (size_t i=0; i < targets.size(); i++)
        {
            Field *tar = targets[i];
            if(source)
            {
                tar->SetID(source->GetID());
                tar->SetSource(source);
                Vector *srcv = source->Data();
                Vector *tarv = tar->Data();
                if(srcv && tarv)
                {
                    tarv->MakeRef(*srcv,0,srcv->Size());
                }

                Vector *src_adj = source->Adjoint();
                Vector *tar_adj = tar->Adjoint();
                if(src_adj && tar_adj)
                {
                    tar_adj->MakeRef(*src_adj,0,src_adj->Size());
                }
            }
        }
    }

    ///@brief Get all target fields
    std::vector<Field*>& GetTargets() { return targets; }

    bool HasTargets() const { return !targets.empty(); }

    virtual ~LinkedFields()
    {
        for (size_t i=0; i < targets.size(); i++)
        {
            if(targets_owned[i] && targets[i]) delete targets[i];
        }
        if(own_source && source) delete source;
    }
};



/// @brief A collection of Fields and LinkedFields, each identified by a name
class FieldCollection
{
private:
    std::string name; /// Name of the collection
    GraphNode *src_op = nullptr; /// Source Application (not owned)

    /// Fields for source Application. Contains all source fields and fields that
    /// may be targets of other applications.
    NamedFieldsMap<Field> fields;

    /// LinkedFields for source Application.
    NamedFieldsMap<mfem::LinkedFields> linked_fields;

public:

    FieldCollection() = default;

    /// @brief Constructor with collection name and optional source Application
    FieldCollection(std::string collection_name,
                    GraphNode *op = nullptr):
                    name(collection_name), 
                    src_op(op){}

    /// @brief Constructor with source Application
    FieldCollection(GraphNode *src):src_op(src){}

    /// @brief Get the number of linked fields in the collection
    int Size() const { return linked_fields.NumFields(); }

    /// @brief Set the name of the collection
    void SetName(const std::string &collection_name){ name = collection_name;}

    /// @brief Get the name of the collection
    std::string GetName() const { return name; }

    /// @brief Set the source Application
    void SetSourceOperator(GraphNode *op){ src_op = op; }

    /// @brief Get the source Application
    const GraphNode* GetSourceOperator() const { return src_op; }

    /// @brief Get the ParGridFunction for a given source name
    Field *GetSourceField(const std::string &src_name) const
    {
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            // MFEM_WARNING("FieldCollection::GetSourceField: Source field "
            //              + src_name + " not found!");
            return nullptr;
        }
        return lf->GetSource();
    }

    /// @brief Get the ParGridFunction for a given field name
    Field* GetField(const std::string &field_name) const
    {
        return fields.Get(field_name);
    }

    LinkedFields* GetLinkedFields(const std::string &src_name) const
    {
        return linked_fields.Get(src_name);
    }

    /// @brief Add a ParGridFunction as a field (does not specify source or target)
    void AddField(const std::string &field_name,
                  Field *field, bool own = false)
    {
        fields.Register(field_name, field, own);
        if(field->GetNode() == nullptr)
        {
            field->SetNode(src_op);
        }
    }

    void AddField(const std::string &field_name,
                  Vector *field)
    {
        Field *f = new Field(field, Field::Type::DEFAULT);
        f->SetNode(src_op);
        fields.Register(field_name, f, true);
    }

    /// @brief Add a LinkedFields to the collection with name src_name
    void AddLinkedFields(const std::string &src_name,
                         LinkedFields *lf, bool own = false)
    {
        LinkedFields *lf_exist = linked_fields.Get(src_name);
        if(lf_exist)
        {
            auto targets = lf->GetTargets();
            for (auto &dest : targets) {
                // auto [target, owned] = dest;
                // lf_exist->AddTarget(target, owned);
                MFEM_ABORT("TO DO")
            }
            return;
        }
        linked_fields.Register(src_name, lf, own);
        fields.Register(src_name, lf->GetSource(), false);
    }

    // /// @brief Add a source ParGridFunction to the collection. If src_name does not
    // /// exist, a new LinkedFields is created and owned.
    void AddSourceField(const std::string &src_name, Vector *src)
    {
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            lf = new LinkedFields(src);
            linked_fields.Register(src_name, lf, true);
        }
        else
        {
            lf->SetSource(src);
        }

        Field *src_field = lf->GetSource();
        if(src_field->GetNode() == nullptr)
        {
            src_field->SetNode(src_op);
        }

        fields.Register(src_name, lf->GetSource(), false);
    }

    void AddSourceField(const std::string &src_name, Field *src, bool own=false)
    {
        fields.Register(src_name, src, false);
        if(src->GetNode() == nullptr)
        {
            src->SetNode(src_op);
        }
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            lf = new LinkedFields(src, own);
            linked_fields.Register(src_name, lf, true);
            return;
        }
        lf->SetSource(src, own);
    }

    /// @brief Add a target field
    /// to the source named src_name. If src_name does not exist, a new 
    /// LinkedFields is created and owned.
    void AddTargetField(const std::string &src_name,
                        Vector *tar)
    {
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            lf = new LinkedFields();
            linked_fields.Register(src_name, lf, true);
        }
        lf->AddTarget(tar);
    }

    void AddTargetField(const std::string &src_name,
                        Field *tar,
                        bool own = false)
    {
        LinkedFields *lf = linked_fields.Get(src_name);
        if(!lf)
        {
            lf = new LinkedFields();
            linked_fields.Register(src_name, lf, true);
        }
        lf->AddTarget(tar, own);
    }

    Field* operator[](const std::string &field_name) const
    {
        return GetField(field_name);
    }

    NamedFieldsMap<Field> &GetFields() { return fields; }
    NamedFieldsMap<LinkedFields> &GetLinkedFields() { return linked_fields; }

    virtual void Save (std::ostream &out) const
    {
        out << "\"Fields\":\n";
        out << "{\n";
        for (auto f = fields.begin(); f != fields.end(); ++f)
        {
            std::string f_name = f->first;
            Field *f_obj = f->second;
            // out << "  " << f_name << ": ID " << f_obj->GetID() << ",\n";
            // out << f_obj->GetID() << ": " << f_name << ",\n";
            out << '\"' << f_obj->GetID() << "\": \"" << f_name << "\"";
            if(f != std::prev(fields.end())) out << ",";
            out << "\n";
        }
        out << "},\n";
        out << "\"LinkedFields\":\n";
        out << "{\n";
        for (auto lf = linked_fields.begin(); lf != linked_fields.end(); ++lf)
        {
            std::string lf_name = lf->first;
            LinkedFields *lf_obj = lf->second;
            // out << "  " << lf_name << ": ID " << lf_obj->GetID() << ",\n";
            out << '\"' << lf_obj->GetSource()->GetID() << "\": \"" << lf_name << "\"";
            if(lf != std::prev(linked_fields.end())) out << ",";
            out << "\n";
        }
        out << "}\n";
    }

    Field* HasField(const Field &field) const
    {
        // Field *f = HasField(field.GetID());
        // return (f == &field) ? f : nullptr;
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
            if(f->second->GetID() == id)
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
        DEFAULT_MODE   ///< Node is being executed in default mode (e.g. application evaluation)
    };

protected:
    inline static int next_id = 0;
    int id = -1;
    int node_index = std::numeric_limits<int>::min();
    mutable ExecutionMode exec_mode = DEFAULT_MODE;

    std::string name;
    FieldCollection fields; // Collection of fields associated with this node
    std::vector<GraphNode*> dependencies; // GraphNodes that this node depends on (WIP)
    std::vector<GraphNode*> dependents;   // GraphNodes that depend on this node (WIP)

public:

    GraphNode(int s = 0) : Operator(s), fields(this)
    {
        id = next_id++;
    }

    GraphNode(int h, int w) : Operator(h,w), fields(this)
    {
        id = next_id++;
    }

    virtual void Execute(const Vector &x, Vector &y)
    {
        MFEM_ABORT("GraphNode::Execute() not implemented");
    }

    virtual void Mult(const Vector &x, Vector &y) const override
    {
        MFEM_ABORT("GraphNode::Mult() not implemented");
    }

    virtual Operator* GetDerivative(Field *y, Vector &x)
    {
        MFEM_ABORT("GraphNode::GetDerivative not implemented");
    }

    // TODO: Consider returning bool to indicate ownership of dydx operator
    [[nodiscard]] virtual bool GetDerivative(Field* y, Vector &x, Operator* &dydx)
    {
        MFEM_ABORT("GraphNode::GetDerivative not implemented");
    }

    [[nodiscard]] virtual bool GetDerivative(Field* y, Field* x, Vector &x0, Operator *dydx)
    {
        MFEM_ABORT("GraphNode::GetDerivative not implemented");
    }

    virtual void GetDerivative(Field* y, Field* x, Vector &x0, Vector &dydx)
    {
        MFEM_ABORT("GraphNode::GetDerivative not implemented");
    }

    void SetNodeIndex(int index){ node_index = index; }
    int GetNodeIndex() const { return node_index; }

    void SetExecutionMode(ExecutionMode mode) { exec_mode = mode; }
    ExecutionMode GetExecutionMode() const { return exec_mode; }

    void SetName(const std::string &name_) { name = name_; }
    std::string GetName() const { return name; }

    // WIP
    void AddDependency(GraphNode *node)
    {
        dependencies.push_back(node);
        node->dependents.push_back(this);
    }
    // WIP
    void AddDependent(GraphNode *node)
    {
        dependents.push_back(node);
        node->dependencies.push_back(this);
    }

    const std::vector<GraphNode*>& GetDependencies() const { return dependencies; }
    const std::vector<GraphNode*>& GetDependents() const { return dependents; }

    void SetID(int id_) { id = id_; }
    int GetID() const { return id; }

    FieldCollection& Fields() { return fields; }
    const FieldCollection& Fields() const { return fields; }

    Field* Fields(const std::string &field_name) const 
    { return fields.GetField(field_name); }

    Field* Fields(const std::string &field_name)
    { return fields.GetField(field_name); }

    LinkedFields* LinkedField(const std::string &src_name) const
    { return fields.GetLinkedFields(src_name); }

    LinkedFields* LinkedField(const std::string &src_name)
    { return fields.GetLinkedFields(src_name); }

    void AddField(const std::string &field_name, Field *field, bool own = false)
    {
        fields.AddField(field_name, field, own);
    }

    /// @brief Add a LinkedFields to the collection with name src_name
    void AddLinkedFields(const std::string &src_name, LinkedFields *field)
    {
        fields.AddLinkedFields(src_name, field);
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


/** @brief This class is used to define the interface for applications and miniapps.
 */
class Application : public GraphNode
{
public: 
    /**
       @brief Enum used to keep track of the type of operator
     */
    enum OperatorType {
        ANY_TYPE,       ///< Any MFEM Operator
        MFEM_TDO,       ///< MFEM TimeDependentOperator
        MFEM_SOLVER,    ///< MFEM Solver
        MFEM_ODESOLVER, ///< MFEM ODESolver
        NOT_MFEM_OBJECT ///< Object is not derived from an MFEM class
    };

protected:
    OperatorType operator_type = OperatorType::ANY_TYPE; ///< Type of the operator

public:
    /**
       @brief Construct a new Application object.
       @param n Size of the operator
     */
    Application(int n=0) : GraphNode(n) {}

    /**
       @brief Construct a new Application object.
       @param h Height of the operator
       @param w Width of the operator
     */
    Application(int h, int w) : GraphNode(h,w) {}

    /**
       @brief Set the Operator Type object. 
     */
    void SetOperatorType(OperatorType type) { operator_type = type; }

    /**
       @brief Get the Operator Type.
     */
    OperatorType GetOperatorType() const { return operator_type; }

    virtual void Mult(const Vector &x, Vector &y) const override
    {
        MFEM_ABORT("Not implemented for this Application.");
    }

    /**
       @brief Update the operator state. 
     */
    virtual void Update()
    {
        mfem_error("Application::Update() is not overridden!");
    }
};


/**
   @brief An abstract, type-erased class to define the interface for 
   operators (not inherited from @a Application) to be used
   with applications. It performs SFINAE checks for stored operator's
   member functions and override the Mult, ImplicitSolve and Step methods
   to call the stored object's functions.
 */
template <typename OpType>
class AbstractOperator : public Application
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
    AbstractOperator(OpType *op_, int h, int w) : Application(h,w), op(op_)
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
    using GraphNode::GetDerivative;
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
        fields.AddTargetField(GetName(), target, own);
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
    int nnodes = 0;         ///< The number of applications
    
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
       return pointer to it. Not owned unless it's not derived from Application.
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

        auto lf = node->LinkedField(node->GetName());
        if(lf)
        {   // Add the node's linkefield to the DAG's
            fields.AddLinkedFields(node->GetName(), lf, false);
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

    virtual void GetDerivative(Field* y, Field* x, Vector &x0, Vector &dydx) override;

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
 */
class SubMeshTransfer : public GraphNode
{
protected:
    ParFiniteElementSpace *src_fes = nullptr, *tar_fes = nullptr;
    ParTransferMap *transfer_map = nullptr;
    bool own_map = false;

public:
    /**
       @brief Construct a new SubMeshTransfer between 
       two @a ParFiniteElementSpace.
       
       @param src Source @a ParFiniteElementSpace
       @param tar Target @a ParFiniteElementSpace
     */
    SubMeshTransfer(ParFiniteElementSpace *src,
                    ParFiniteElementSpace *tar) : GraphNode(),
                    src_fes(src), tar_fes(tar),
                    transfer_map(new ParTransferMap(src_fes, tar_fes)), own_map(true) {}

    /**
       @brief Construct a new SubMeshTransfer between
       @a ParFiniteElementSpace of two @a ParGridFunction.
       @param src Source @a ParGridFunction
       @param tar Target @a ParGridFunction
     */
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

        // Loop through all the linked fields and perform the operator
        NamedFieldsMap<LinkedFields> &linked_fields = Fields().GetLinkedFields();
        for (auto lf = linked_fields.begin(); lf != linked_fields.end(); ++lf)
        {
            std::string lf_name = lf->first;
            LinkedFields *lf_obj = lf->second;
            ParGridFunction &src_gf = dynamic_cast<ParGridFunction&>(*lf_obj->GetSource()->Data());

            // Loop through all the targets for this source field and perform the transfer
            auto &targets = lf_obj->GetTargets();
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





} //mfem namespace

#endif // MFEM_USE_MPI

#endif
