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
class FieldCollection;
class GraphNode;
class DAGraph;
class GraphGradient;


/// @brief Base class for storing data (Vector) and distinguishing
/// fields variables
class Field
{
public:
    enum Type ///< Not used for now, but could be used to distinguish between input/output fields
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
    virtual void SetDataAndAdjoint(Vector *field, Vector *adj) { data = field; adjoint = adj; }
    virtual void SetData(Vector *field) { data = field; }
    virtual void SetAdjoint(Vector *adj) { adjoint = adj; }
    virtual void SetOperator(Operator *op) { oper = op; }

    virtual void GetData(Vector &field) const { field = *data; }
    virtual void GetAdjoint(Vector &adj) const { adj = *adjoint; }

    std::string Name() const { return name; }
    void SetName(const std::string &n) { name = n; }
    int ID() const { return id; }

    void SetID(int i)
    {
        MFEM_ASSERT(i >= 0, "ID must be non-negative.");
        id = i;
    }

    bool IsInput() const {return (type == Type::INPUT);}
    bool IsOutput() const {return (type == Type::OUTPUT);}
    bool IsDefault() const {return (type == Type::DEFAULT);}

    virtual ~Field() = default;

protected:

    ///@brief Set the type of the field (prevents changing type of input/output fields)
    void SetType(Type t)
    {
        type = t;
    }
};

/// @brief A collection of Fields, each identified by a name
class FieldCollection
{
public:
    using FieldMap = GenericFieldMap<int, Field*>; // ID -> Field
    using StrToInt = GenericFieldMap<std::string, int>; // Name -> ID
    using IntToBool = GenericFieldMap<int, bool>; // ID -> Ownership

private:
    std::string name; /// Name of the collection
    Operator *oper = nullptr; /// Operator associated with this collection (not owned)
    FieldMap fields;  /// Map from field ID to Field pointer
    StrToInt named_map; /// Map from field name to IDs
    IntToBool ownership_map; /// Map from field ID to ownership flag

    Array<Field*> input_fields;  // Input fields for this node
    Array<Field*> output_fields; // Output fields for this node

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

    /// @brief Add a field and ownership flag to the collection
    void AddField(Field *field, bool own = false)
    {
        if(fields.Has(field->ID()))
        {
            MFEM_ABORT("FieldCollection::AddField: Field with ID "
                       << field->ID() << " already exists.");
        }
        fields.Register(field->ID(), field, own);
        ownership_map.Register(field->ID(), own);
    }

    /// @brief Add a field to the collection with a given name and ownership flag
    void AddField(const std::string &field_name,
                  Field *field, bool own = false)
    {
        AddField(field, own);
        named_map.Register(field_name, field->ID());
    }

    void SetFieldOwnership(int field_id, bool own)
    {
        if(!fields.Has(field_id))
        {
            MFEM_ABORT("FieldCollection::SetFieldOwnership: Field with ID "
                       << field_id << " does not exist.");
        }
        ownership_map.Register(field_id, own);
    }

    void AddInput(Field *field, bool own = false)
    {
        AddField(field, own);
        input_fields.push_back(field);
    }
    void AddInput(const std::string &field_name,
                  Field *field, bool own = false)
    {
        AddField(field_name, field, own);
        input_fields.push_back(field);
    }

    void AddOutput(Field *field, bool own = false)
    {
        AddField(field, own);
        output_fields.push_back(field);
    }
    void AddOutput(const std::string &field_name,
                   Field *field, bool own = false)
    {
        AddField(field_name, field, own);
        output_fields.push_back(field);
        if(field->GetOperator() == nullptr)
        {
            field->SetOperator(oper);
        }
    }

    Array<Field*>& InputFields() { return input_fields; }
    Array<Field*>& OutputFields() { return output_fields; }

    Field* InputField(int i) const { return input_fields[i]; }
    Field *InputField(const std::string &field_name) const
    {
        return fields.Get(named_map.Get(field_name));
    }

    Field* OutputField(int i) const { return output_fields[i]; }
    Field *OutputField(const std::string &field_name) const
    {
        return fields.Get(named_map.Get(field_name));
    }

    FieldMap &Fields() { return fields; }
    FieldMap Fields() const { return fields; }

    virtual void Save (std::ostream &out) const
    {
        out << "\"Fields\":\n";
        out << "{\n";
        for (auto f = fields.begin(); f != fields.end(); ++f)
        {
            std::string f_name = f->second->Name();
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
        for (int i = 0; i < input_fields.Size(); ++i)
        {
            Field *f_obj = input_fields[i];
            out << '\"' << f_obj->ID() << "\": \"" << f_obj->Name() << "\"";
            if(i != input_fields.Size() - 1) out << ",";
            out << "\n";
        }
        out << "},\n";

        out << "\"Outputs\":\n";
        out << "{\n";
        for (int i = 0; i < output_fields.Size(); ++i)
        {
            Field *f_obj = output_fields[i];
            out << '\"' << f_obj->ID() << "\": \"" << f_obj->Name() << "\"";
            if(i != output_fields.Size() - 1) out << ",";
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
        return named_map.Has(field_name) ? fields.Get(named_map.Get(field_name)) : nullptr;
    }

    Field* HasField(const int id) const
    {
        return fields.Get(id);
    }

    void Clear()
    {
        for (auto f = fields.begin(); f != fields.end(); ++f)
        {
            int id = f->second->ID();
            if(ownership_map.Has(id) && ownership_map.Get(id))
            {
                delete f->second;
            }
        }
        fields.clear();
        named_map.clear();
        ownership_map.clear();
        input_fields.SetSize(0);
        output_fields.SetSize(0);
    }

    virtual ~FieldCollection()
    {
        Clear();
    }

};

// TODO: Should move these to a util namespace or a util file. 
template <typename T, std::size_t... I>
auto ArrayToTuple_Helper( const Array<T>& v, std::index_sequence<I...>)
{
    return std::make_tuple(v[I]...);
}

template <int N, typename T>
auto ArrayToTuple(const Array<T>& v)
{
    return ArrayToTuple_Helper(v,std::make_index_sequence<N>{});
}

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
    int node_index = -1;
    mutable ExecutionMode exec_mode = DEFAULT_MODE;

    std::string name;
    mutable FieldCollection field_collection; ///< Collection of fields associated with this node

    // Offsets to be used for operation on BlockVector
    Array<int> input_offsets;  ///< Offsets for input fields
    Array<int> output_offsets; ///< Offsets for output fields

    int GetValidID(int id_, int lb=0, int ub = std::numeric_limits<int>::max())
    {
        return (id_ >= lb && id_ <= ub) ? id_ : next_id++;
    }

public:

    GraphNode(int h, int w) : Operator(h,w), id(GetValidID(-1)),
                              name("Node_" + std::to_string(id)),
                              field_collection(this) { }

    GraphNode(int s = 0) : GraphNode(s, s) { }

    void SetNodeIndex(int index){ node_index = index; }
    int GetNodeIndex() const { return node_index; }

    void SetExecutionMode(ExecutionMode mode) { exec_mode = mode; }
    ExecutionMode GetExecutionMode() const { return exec_mode; }

    void SetName(const std::string &name_) { name = name_; }
    std::string Name() const { return name; }

    void SetID(int id_) { id = id_; }
    int ID() const { return id; }

    FieldCollection& Fields() { return field_collection; }
    const FieldCollection& Fields() const { return field_collection; }

    Array<Field*>& InputFields() const { return field_collection.InputFields(); }
    Array<Field*>& OutputFields() const { return field_collection.OutputFields(); }
    Field* InputField(int i) const { return InputFields()[i]; }
    Field* OutputField(int i) const { return OutputFields()[i]; }


    virtual void AddInput(const std::string &field_name,
                          Field *field, bool own = false)
    { field_collection.AddInput(field_name, field, own); }

    virtual void AddInput(Field *field, bool own = false)
    { field_collection.AddInput(field, own); }

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
    { field_collection.AddOutput(field, own); }

    template<bool OwnOutputs = false,
             typename... Args,
             bool AreFields = std::conjunction<std::is_base_of<Field, std::remove_pointer_t<Args>> ...>::value,
             typename std::enable_if<AreFields, bool>::type = true >
    void AddOutputs(Args... args)
    {
        ((AddOutput(std::forward<Args>(args), OwnOutputs)), ...);
    }

    /// @brief Return the input offsets for block starts.
    Array<int>& InputOffsets() { return input_offsets; }

    /// @brief Read only access to the input offsets for block starts.
    const Array<int>& InputOffsets() const { return input_offsets; }

    void SetInputOffsets(const Array<int> &offsets) { input_offsets = offsets; }

    /// @brief Return the output offsets for block starts.
    Array<int>& OutputOffsets() { return output_offsets; }

    /// @brief Read only access to the output offsets for block starts.
    const Array<int>& OutputOffsets() const { return output_offsets; }

    void SetOutputOffsets(const Array<int> &offsets) { output_offsets = offsets; }

    virtual void MultMV(const MultiVector &x, MultiVector &y) const
    {
        MFEM_ABORT("This method is not overridden for this class!");
        // Include a default implementation that calls Mult() with Vector arguments
    }

    virtual void MultTransposeMV(const MultiVector &x, MultiVector &y) const
    {
        MFEM_ABORT("This method is not overridden for this class!");
        // Include a default implementation that calls MultTranspose() with Vector arguments
    }

    virtual void GradientMult(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const
    {
        MFEM_ABORT("This method is not overridden for this class!");
        GetGradientMV(x).MultMV(dx, dy);
    }

    virtual void GradientMultTranspose(const MultiVector &x, const MultiVector &dx, MultiVector &dy) const
    {
        MFEM_ABORT("This method is not overridden for this class!");
        GetGradientMV(x).MultTransposeMV(dx, dy); // Not yet implemented
    }

    virtual void Save (std::ostream &out) const
    {
        out << "\"Node-" << id << "\" : " << std::endl;
        out << "{\n";
        out << "\"Name\": \"" << name << "\",\n";
        field_collection.Save(out);
        out << "}";
    }

    // Variadic template that takes in an arbitrary number of Fields as inputs and returns a tuple of the N output fields
    template<int N = 1, // Number of output fields
             bool OwnInputs = false,  // Whether to own the input fields
             bool OwnOutputs = false, // Whether to own the output fields
             typename... Args, // Parameter pack for input fields
             bool AreFields = std::conjunction<std::is_base_of<Field, std::remove_pointer_t<Args>> ...>::value,
             typename std::enable_if<AreFields, bool>::type = true >
    constexpr auto operator()(Args... args)
    {
        // Add the input fields to the node
        (AddInput(std::forward<Args>(args), OwnInputs), ...);
        // (AddInput(args), ...);

        if(OutputFields().Size() == 0)
        {
            // Add 'N' number of output fields to the node if none exist
            for(int i = 0; i < N; ++i)
            {
                AddOutput(new Field(nullptr, nullptr, Field::Type::OUTPUT), OwnOutputs);
            }
        }
        else
        {
            MFEM_ASSERT(OutputFields().Size() == N,
                        "Number of output fields " << OutputFields().Size()
                        << " does not match the specified number of outputs " << N);

            // Set output ownership for the output fields if existing fields are used
            auto outputs = OutputFields();
            for(int i = 0; i < N; ++i)
            {
                field_collection.SetFieldOwnership(outputs[i]->ID(), OwnOutputs);
            }
        }

        if constexpr (N == 1)
        {
            return OutputField(0);
        }
        else if constexpr (N > 1)
        {
            // Build and return a tuple of pointer to output fields
            return ArrayToTuple<N>(OutputFields());
        }
    }


    virtual ~GraphNode()
    {
        // Clear collection of fields associated with this node
        field_collection.Clear();
    }
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

        /// @brief A type trait to check if the erased class has the function Mult
        /// with the needed signatures.
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
        typedef decltype(Check<C,Mult,void>(0)) Has_Mult;
        typedef decltype(Check<C,MultPtr,void>(0)) Has_MultPtr;
    public:
        static constexpr bool HasMult  = Has_Mult::value;
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


/**
   @brief A class to store and coupled multiple operators together.
 */
class DAGraph : public GraphNode
{
public:

    using IntToIntMap = GenericFieldMap<int, int>;
    using IntToFieldMap = GenericFieldMap<int, Field*>;

    enum class GradMode
    {
        FINITE_DIFF = 0,  ///< Finite difference Jacobian
        MATRIX_FREE = 1,  ///< Matrix-free Jacobian
        ASSEMBLED = 2,    ///< Assembled Jacobian
        NONE = 3          ///< Not implemented
    };

protected:
    Array<GraphNode*> nodes; ///< Vector of individual operators
    Array<bool> node_owned; ///< Whether the operators are owned
    Array<int> node_depth; ///< Depth of each operator in the graph

    int max_width  = 0;     ///< Largest operator width
    int max_height = 0;     ///< Largest operator height
    int nnodes     = 0;     ///< The number of nodes
    bool sorted    = false; ///< True if the nodes are topologically sorted
    bool assembled = false; ///< True if the graph is assembled

    GradMode grad_mode = GradMode::MATRIX_FREE; ///< Gradient mode for the graph
    mutable Operator *grad = nullptr; ///< Gradient operator
    mutable MultiVector xmv_node, ymv_node; ///< Temporary multivectors for evaluating nodes

    IntToFieldMap fid_to_field; ///< Map from Field ID to Field pointer
    IntToIntMap fid_to_index; ///< Map from ID to index in an array; needed since ordering is not unique

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

    /// @brief Get the number of coupled operators
    int Size(){return nnodes;}

    /// @brief Get the size of the largest operator
    int MaxWidth() const {return max_width;}
    int MaxHeight() const {return max_height;}

    IntToIntMap &GetFieldIdToIndexMap() { return fid_to_index; }
    IntToIntMap GetFieldIdToIndexMap() const { return fid_to_index; }

    IntToFieldMap &GetFieldIdToFieldMap() { return fid_to_field; }
    IntToFieldMap GetFieldIdToFieldMap() const { return fid_to_field; }

    /// @brief Get the operator at index @a i
    GraphNode* GetNode(const int i)
    {
        MFEM_ASSERT(i >= 0 && i < nnodes,
               "index [" << i << "] is out of range [0," << nnodes << ")");
        return nodes[i];
    }

    Array<GraphNode*>& Nodes() { return nodes; }

    /// @brief Specify whether the operator at index @a i is owned.
    void OwnNode(const int i, bool own = true)
    {
        MFEM_ASSERT(i >= 0 && i < nnodes,
               "index [" << i << "] is out of range [0," << nnodes << ")");
        node_owned[i] = own;
    }

    void Assemble();
    bool IsAssembled() const { return assembled; }

    void TopologicalSort();
    bool IsSorted() const { return sorted; }

    void ComputeDepth();

    void ValidateOffsets();

    void ValidateNode(GraphNode &node);

    void CollectFieldMaps();

    using GraphNode::AddInput;
    /* TODO: Remove in favour of AddInput(s) and SetInputOffsets
    void AddInput(Field *field, int sz, bool own = false)
    {
        if(input_offsets.Size() == 0)
        {   // First entry
            input_offsets.Append(0);
        }
        input_offsets.Append(input_offsets.Last() + sz);
        AddInput(field, own);
    }
    */

    using GraphNode::AddOutput;
    /* TODO: Remove in favour of AddOutput(s) and SetOutputOffsets
    void AddOutput(Field *field, int sz, bool own = false)
    {
        if(output_offsets.Size() == 0)
        {   // First entry
            output_offsets.Append(0);
        }
        output_offsets.Append(output_offsets.Last() + sz);
        AddOutput(field, own);
    }
    */

    /// @brief Set the gradient mode for the coupled operator
    void SetGradientMode(GradMode mode)
    {
        if(mode != grad_mode)
        {
            if(grad) { delete grad; grad = nullptr; }
            grad_mode = mode;
        }
    }

    /**
       @brief Apply the operator to the vector @a x 
       and return the result in @a y.
     */
    virtual void Mult(const Vector &x, Vector &y) const override;

    virtual void MultMV(const MultiVector &x, MultiVector &y) const override;

    virtual void Execute(const MultiVector &x, MultiVector &y) const;

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
protected:
    mutable DAGraph *graph = nullptr; ///< Pointer to the DAGraph for which this is the gradient operator
    Array<Vector*> x_work; ///< Array to store linearization point (intermediate fields)
    mutable MultiVector xlin;
    mutable MultiVector x0_mv, dx_mv, dy_mv;

public:
    GraphGradient(DAGraph &dag);

    void Update(const Vector &x);

    void Mult(const Vector &x, Vector &y) const override;

    void MultMV(const MultiVector &x, MultiVector &y) const override;

    void MultTranspose(const Vector &x, Vector &y) const override;

    void MultTransposeMV(const MultiVector &x, MultiVector &y) const override;

    Operator &GetGradient(const Vector &x) const override;

    void Forward(const MultiVector &x, MultiVector &y) const;

    void Reverse(const MultiVector &x, MultiVector &y) const;

    ~GraphGradient()
    {
        for (auto &v : x_work)
        {
            if(v) { delete v; v = nullptr; }
        }
        x_work.DeleteAll();
    }
};

} //mfem namespace

#endif
