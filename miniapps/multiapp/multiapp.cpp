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

#include "multiapp.hpp"

namespace mfem
{

DAGraph::DAGraph(const int nop) : GraphNode()
{
    nodes.Reserve(nop);
    node_owned.Reserve(nop);
    tape = new GraphTape(*this); // Create a tape for this DAGraph
}

DAGraph::~DAGraph()
{
    for(int i=0; i < nnodes; i++)
    {
        if(node_owned[i] && nodes[i]) delete nodes[i];
    }
    if(grad) delete grad;
    if(tape) delete tape;
}

void DAGraph::Assemble()
{
    // Sort graph nodes topologically to ensure correct execution order
    // Ordering is not unique, hence, id->index maps are needed
    TopologicalSort();

    // Collect all fields from the nodes into the field map
    CollectFieldMaps();

    // Compute depth of the graph nodes
    ComputeDepth();

    // Validate each node
    for (auto &node : nodes)
    {
        ValidateNode(*node);
    }

    // Update width and height of the DAG from offsets
    // Check that the input and output offsets are consistent
    ValidateOffsets();

    // Delete any existing gradient operator as node ordering may have changed
    if (grad) delete grad;

    assembled = true;
}

void DAGraph::ValidateOffsets()
{
    // Check that the input and output offsets are consistent
    // with the number of inputs and outputs
    if(InputFields().Size() > 1 && input_offsets.Size() > 0)
    {
        MFEM_ASSERT(input_offsets.Size() == InputFields().Size() + 1,
                    "Input offsets size inconsistent with number of input fields");
    }
    else
    {
        input_offsets = Array<int>({0, nodes[0]->Width()});
    }

    if(OutputFields().Size() > 1 && output_offsets.Size() > 0)
    {
        MFEM_ASSERT(output_offsets.Size() == OutputFields().Size() + 1,
                    "Output offsets size inconsistent with number of output fields");
    }
    else
    {
        output_offsets = Array<int>({0, nodes.Last()->Height()});
    }
}

void DAGraph::ValidateNode(GraphNode &node)
{
    // Validate that the node's input and output fields are consistent with the graph's field map
    auto inputs = node.InputFields();
    auto outputs = node.OutputFields();

    // Check that all input and output fields are registered in the graph's field map
    for(auto input_field : inputs)
    {
        MFEM_ASSERT(fid_to_index.Has(input_field->ID()),
                    "Input field ID " << input_field->ID() << " not found in graph's field map");
    }
    for(auto output_field : outputs)
    {
        MFEM_ASSERT(fid_to_index.Has(output_field->ID()),
                    "Output field ID " << output_field->ID() << " not found in graph's field map");
    }
}

void DAGraph::TopologicalSort()
{
    Array<int> sorted_indices;
    sorted_indices.Reserve(nnodes);

    Array<bool> visited(nnodes);
    visited = false; // Initialize all nodes as unvisited

    // Perform a depth-first search to sort the nodes topologically
    std::function<void(int)> DepthFirstSearch = [&](int node_index)
    {
        if(visited[node_index]) return;
        visited[node_index] = true;
        auto node = nodes[node_index];
        // Visit all nodes that this node depends on
        for(auto input_field : node->InputFields())
        {
            for(int j=0; j < nnodes; j++)
            {
                auto other_node = nodes[j];
                if(other_node == node) continue;
                for(auto output_field : other_node->OutputFields())
                {
                    if(input_field->ID() == output_field->ID()) // Compare by unique ID
                    {
                        DepthFirstSearch(j);
                    }
                }
            }
        }
        sorted_indices.push_back(node_index);
    };

    for(int i=0; i < nnodes; i++)
    {
        DepthFirstSearch(i);
    }

    // check if already sorted
    bool already_sorted = true;
    for(int i=0; i < nnodes; i++)
    {
        if(sorted_indices[i] != i)
        {
            already_sorted = false;
            break;
        }
    }

    if(!already_sorted)
    {
        nodes.Permute(sorted_indices);
        node_owned.Permute(sorted_indices);
    }

    // Update the node indices after sorting
    for(int i=0; i < nnodes; i++)
    {
        nodes[i]->SetNodeIndex(i);
    }

    sorted = true;
}

void DAGraph::ComputeDepth()
{
    // Compute depth of ordered nodes
    node_depth.SetSize(nnodes);
    node_depth = 0;
    for(int i=0; i < nnodes; i++)
    {
        int max_depth = 0;
        auto node = nodes[i];
        for(auto input_field : node->InputFields())
        {
            for(int j=0; j < i; j++)
            {
                auto other_node = nodes[j];
                if(other_node == node) continue;
                for(auto output_field : other_node->OutputFields())
                {
                    if(input_field->ID() == output_field->ID()) // Compare by unique ID
                    {
                        max_depth = std::max(max_depth, node_depth[j] + 1);
                    }
                }
            }
        }
        node_depth[i] = max_depth;
    }
}

void DAGraph::CollectFieldMaps()
{
    MFEM_ASSERT(sorted, "DAGraph must be topologically sorted before collecting fields");

    fid_to_index.clear();
    FieldCollection& field_set = Fields();

    int nfields = 0;
    for (auto f : InputFields())
    {
        fid_to_index.Register(f->ID(), nfields++);
        if(!field_set.HasField(f->ID())) // Inputs would have already been added with AddInput
        {
            field_set.AddField(f, false); // If not, Add input field to collection
        }
    }

    for (auto &node : nodes)
    {
        for (auto f : node->OutputFields())
        {
            if (!fid_to_index.Has(f->ID()))
            {
                fid_to_index.Register(f->ID(), nfields++);
            }
            if(!field_set.HasField(f->ID()))
            {
                field_set.AddField(f, false); // Add output field to graph's field collection
            }
        }
    }
}

void DAGraph::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(width == x.Size(), "Input vector size (" << x.Size()
                << ") must match matrix width (" << width << ")");

    MFEM_ASSERT(height == y.Size(), "Output vector size (" << y.Size()
                << ") must match matrix height (" << height << ")");

    auto inputs  = InputFields();
    auto outputs = OutputFields();

    BlockVector xb(x.GetData(), input_offsets);
    BlockVector yb(y.GetData(), output_offsets);
    MultiVector xmv(inputs.Size()), ymv(outputs.Size());

    // Set the data pointers of the input and output fields
    // of the graph to point to the corresponding blocks of
    // the input and output vectors
    for(int i=0; i < inputs.Size(); i++)
    {
        xmv.MakeRef(i, xb.GetBlock(i));
    }

    for(int i=0; i < outputs.Size(); i++)
    {
        ymv.MakeRef(i, yb.GetBlock(i));
    }

    MultMV(xmv, ymv);
}

void DAGraph::MultMV(const MultiVector &x, MultiVector &y) const
{
    auto inputs  = InputFields();
    auto outputs = OutputFields();

    MFEM_ASSERT(inputs.Size() == x.NumBlocks(), "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << inputs.Size() << ")");

    MFEM_ASSERT(outputs.Size() == y.NumBlocks(), "Number of output blocks (" << y.NumBlocks()
                << ") must match number of output fields (" << outputs.Size() << ")");

    for(int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetData(const_cast<Vector*>(&x[i]));
    }
    for (int i=0; i < outputs.Size(); i++)
    {
        outputs[i]->SetData(&y[i]);
    }

    auto index_map = GetFieldIdToIndexMap();
    auto field_set = Fields();
    int nfields = index_map.NumFields();
    MultiVector ymv(nfields); // TODO: Should this be a member variable?

    // Assemble the multivector from the individual fields based on their IDs
    // This multivector contains all input, output, and intermediate fields in the graph
    for (auto const& [id, idx] : index_map)
    {
        if(Field* field = field_set.Get(id))
        {
            ymv.MakeRef(idx, *field->Data());
        }
        else
        {
            MFEM_ABORT("Field ID " << id << " not found in field map");
        }
    }
    
    Execute(x, ymv);

    for(auto &f : inputs)
    {
        f->SetData(nullptr);
    }
    for(auto &f : outputs)
    {
        f->SetData(nullptr);
    }

    // TODO: Handle case where execution mode is GRADIENT_MODE to update grad
}

void DAGraph::Execute(const MultiVector &x, MultiVector &y) const
{
    MFEM_ASSERT(assembled, "DAGraph must be assembled before calling Execute()");

    MFEM_ASSERT(x.NumBlocks() == InputFields().Size(),
                "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << InputFields().Size() << ")");
    
    auto index_map = GetFieldIdToIndexMap();
    
    MFEM_ASSERT(y.NumBlocks() == index_map.NumFields(),
                "Number of output blocks (" << y.NumBlocks()
                << ") must match number of fields (" << index_map.NumFields() << ")");

    auto inputs  = InputFields();
    for(int i=0; i < inputs.Size(); i++)
    {
        int idx = index_map.Get(inputs[i]->ID());
        if(&y[idx] != &x[i]) // copy data, if address is different
        {
            y[idx] = x[i];
        }
    }

    for (auto node : nodes)
    {
        auto node_inputs = node->InputFields();
        auto node_outputs = node->OutputFields();
        xmv_node.SetNumBlocks(node_inputs.Size());
        ymv_node.SetNumBlocks(node_outputs.Size());

        for (int i=0; i < node_inputs.Size(); i++)
        {
            int idx = index_map.Get(node_inputs[i]->ID());
            xmv_node.MakeRef(i, y[idx]);
        }
        for (int i=0; i < node_outputs.Size(); i++)
        {
            int idx = index_map.Get(node_outputs[i]->ID());
            ymv_node.MakeRef(i, y[idx]);
        }
        node->MultMV(xmv_node, ymv_node);
    }
}

Operator& DAGraph::GetGradient(const Vector &x) const
{
    // TODO: Should/could be removed
    if(grad_mode == GradMode::FINITE_DIFF)
    {
        if(!grad)
        {
            grad = new future::FDJacobian(*this, x, 1e-6);
        }
        else
        {
            grad->GetGradient(x); // Update the FDJacobian with new point x
        }
        return *grad;
    }

    MFEM_ASSERT(static_cast<int>(grad_mode) < static_cast<int>(GradMode::NONE),
                "DAGraph::GetGradient() called with invalid grad_mode: "
                << static_cast<int>(grad_mode));

    if(!grad)
    {
        grad = new GraphGradient(const_cast<DAGraph&>(*this));
    }

    if(grad_mode == GradMode::ASSEMBLED)
    {
        return grad->GetGradient(x); // Assemble the Jacobian matrix
    }
    else // GradMode::MATRIX_FREE
    {
        dynamic_cast<GraphGradient*>(grad)->Update(x); // Update the GraphGradient with new point x
    }

    return *grad;
}

GraphGradient::GraphGradient(DAGraph &dag) : Operator(dag.Height(), dag.Width()),
                                             graph(&dag)
{
    MFEM_ASSERT(graph->IsAssembled(), "GraphGradient requires an assembled DAGraph.");
    MFEM_ASSERT(graph->IsSorted(), "GraphGradient requires a topologically sorted DAGraph.");

    auto index_map = graph->GetFieldIdToIndexMap();
    FieldCollection& field_set = graph->Fields();

    x_arr.DeleteAll(); // Clear any existing pointers
    x_arr.SetSize(index_map.NumFields());
    x_arr = nullptr; // Initialize all pointers to nullptr
    xlin.SetNumBlocks(index_map.NumFields());

    for (auto const& [id, idx] : index_map)
    {
        MFEM_ASSERT(idx >= 0 && idx < x_arr.Size(), "Index out of bounds for field ID: " << id);
        MFEM_ASSERT(field_set.HasField(id), "Field ID not found in field_set: " << id);

        if(x_arr[idx] == nullptr)
        {
            x_arr[idx] = new Vector(); // Allocate a new Vector for this field
        }
        xlin.MakeRef(idx, *x_arr[idx]); // Make xlin refer to the allocated Vector
    }

}

void GraphGradient::Update(const Vector &x)
{
    MFEM_ASSERT(graph != nullptr, "GraphGradient operator requires a non-null DAGraph pointer.");

    auto set_exec_mode = [&](DAGraph::ExecutionMode mode)
    {
        for (auto &node : graph->Nodes())
        {
            node->SetExecutionMode(mode);
        }
    };

    auto inputs = graph->InputFields();
    BlockVector xb(x.GetData(), graph->InputOffsets());
    MultiVector xmv(inputs.Size());
    for(int i=0; i < inputs.Size(); i++)
    {
        xmv.MakeRef(i, xb.GetBlock(i));
    }

    set_exec_mode(DAGraph::ExecutionMode::GRADIENT_MODE);
    graph->Execute(xmv, xlin); // Forward pass to populate fields for gradient computations
    set_exec_mode(DAGraph::ExecutionMode::DEFAULT_MODE); // Reset execution mode for forward pass
}

void GraphGradient::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(x.Size() == graph->Width(), "Input vector size (" << x.Size()
                << ") must match graph width (" << graph->Width() << ")");

    MFEM_ASSERT(y.Size() == graph->Height(), "Output vector size (" << y.Size()
                << ") must match graph height (" << graph->Height() << ")");

    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    auto inputs  = graph->InputFields();
    auto outputs = graph->OutputFields();

    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);
    MultiVector xmv(inputs.Size()), ymv(outputs.Size());

    for(int i=0; i < inputs.Size(); i++)
    {
        xmv.MakeRef(i, xb.GetBlock(i));
    }

    for(int i=0; i < outputs.Size(); i++)
    {
        ymv.MakeRef(i, yb.GetBlock(i));
    }

    MultMV(xmv, ymv); // Forward mode: compute JVP, y = J(z) * x
}

void GraphGradient::MultMV(const MultiVector &x, MultiVector &y) const
{
    auto inputs  = graph->InputFields();
    auto outputs = graph->OutputFields();

    MFEM_ASSERT(inputs.Size() == x.NumBlocks(), "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << inputs.Size() << ")");

    MFEM_ASSERT(outputs.Size() == y.NumBlocks(), "Number of output blocks (" << y.NumBlocks()
                << ") must match number of output fields (" << outputs.Size() << ")");

    for(int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetAdjoint(const_cast<Vector*>(&x[i]));
    }
    for (int i=0; i < outputs.Size(); i++)
    {
        outputs[i]->SetAdjoint(&y[i]);
    }

    auto index_map = graph->GetFieldIdToIndexMap();
    FieldCollection& field_set = graph->Fields();
    int nfields = index_map.NumFields();
    MultiVector ymv(nfields); // TODO: Should this be a class member?

    // Assemble the multivector from the individual fields based on their IDs
    // This multivector contains all input, output, and intermediate fields in the graph
    for (auto const& [id, idx] : index_map)
    {
        if(Field* field = field_set.Get(id))
        {
            ymv.MakeRef(idx, *field->Adjoint());
        }
        else
        {
            MFEM_ABORT("Field ID " << id << " not found in field map");
        }
    }

    Forward(x, ymv); // Forward mode: compute JVP, y = J(z) * x

    for (auto &f : inputs)
    {
        f->SetAdjoint(nullptr);
    }
    for (auto &f : outputs)
    {
        f->SetAdjoint(nullptr);
    }
}

void GraphGradient::MultTranspose(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(x.Size() == graph->Height(), "Input vector size (" << x.Size()
                << ") must match graph height (" << graph->Height() << ")");
    MFEM_ASSERT(y.Size() == graph->Width(), "Output vector size (" << y.Size()
                << ") must match graph width (" << graph->Width() << ")");

    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    auto inputs  = graph->InputFields();
    auto outputs = graph->OutputFields();

    BlockVector xb(x.GetData(), out_offsets);
    BlockVector yb(y.GetData(), in_offsets);
    MultiVector xmv(outputs.Size()), ymv(inputs.Size());

    for(int i=0; i < inputs.Size(); i++)
    {
        xmv.MakeRef(i, xb.GetBlock(i));
    }

    for(int i=0; i < outputs.Size(); i++)
    {
        ymv.MakeRef(i, yb.GetBlock(i));
    }
    MultTransposeMV(xmv, ymv); // Reverse mode: compute VJP, y = J(z)^T * x
}

void GraphGradient::MultTransposeMV(const MultiVector &x, MultiVector &y) const
{
    auto inputs  = graph->InputFields();
    auto outputs = graph->OutputFields();

    MFEM_ASSERT(outputs.Size() == x.NumBlocks(), "Number of input blocks (" << x.NumBlocks()
                << ") must match number of output fields (" << outputs.Size() << ")");

    MFEM_ASSERT(inputs.Size() == y.NumBlocks(), "Number of output blocks (" << y.NumBlocks()
                << ") must match number of input fields (" << inputs.Size() << ")");

    for(int i=0; i < outputs.Size(); i++)
    {
        outputs[i]->SetAdjoint(const_cast<Vector*>(&x[i]));
    }
    for (int i=0; i < inputs.Size(); i++)
    {
        inputs[i]->SetAdjoint(&y[i]);
    }

    auto index_map = graph->GetFieldIdToIndexMap();
    FieldCollection& field_set = graph->Fields();
    int nfields = index_map.NumFields();
    MultiVector ymv(nfields); // TODO: Should this be a class member?

    for(auto const& [id, idx] : index_map)
    {
        if(Field* field = field_set.Get(id))
        {
            ymv.MakeRef(idx, *field->Adjoint());
        }
        else
        {
            MFEM_ABORT("Field ID " << id << " not found in field map");
        }
    }

    Reverse(x, ymv); // Reverse mode: compute VJP, y = J(z)^T * x

    for (auto &f : outputs)
    {
        f->SetAdjoint(nullptr);
    }
    for (auto &f : inputs)
    {
        f->SetAdjoint(nullptr);
    }
}

void GraphGradient::Forward(const MultiVector &x, MultiVector &y) const
{
    MFEM_ASSERT(x.NumBlocks() == graph->InputFields().Size(),
                "Number of input blocks (" << x.NumBlocks()
                << ") must match number of input fields (" << graph->InputFields().Size() << ")");
    
    auto index_map  = graph->GetFieldIdToIndexMap();

    MFEM_ASSERT(y.NumBlocks() == index_map.NumFields(),
                "Number of output blocks (" << y.NumBlocks()
                << ") must match number of fields (" << index_map.NumFields() << ")");

    auto inputs  = graph->InputFields();
    for(int i=0; i < inputs.Size(); i++)
    {
        int idx = index_map.Get(inputs[i]->ID());
        if(&y[idx] != &x[i]) // copy data, if address is different
        {
            y[idx] = x[i];
        }
    }

    auto nodes = graph->Nodes();
    for (auto node : nodes)
    {
        auto node_inputs = node->InputFields();
        auto node_outputs = node->OutputFields();
        x0_mv.SetNumBlocks(node_inputs.Size());
        dx_mv.SetNumBlocks(node_inputs.Size());
        dy_mv.SetNumBlocks(node_outputs.Size());

        for(int i=0; i < node_inputs.Size(); i++)
        {
            int idx = index_map.Get(node_inputs[i]->ID());
            x0_mv.MakeRef(i, xlin[idx]);
            dx_mv.MakeRef(i, y[idx]);
        }
        for(int i=0; i < node_outputs.Size(); i++)
        {
            int idx = index_map.Get(node_outputs[i]->ID());
            dy_mv.MakeRef(i, y[idx]);
        }
        node->GradientMult(x0_mv, dx_mv, dy_mv); // Compute JVP for the node
    }
}

void GraphGradient::Reverse(const MultiVector &x, MultiVector &y) const
{
    MFEM_ASSERT(x.NumBlocks() == graph->OutputFields().Size(),
                "Number of input blocks (" << x.NumBlocks()
                << ") must match number of output fields (" << graph->OutputFields().Size() << ")");

    auto index_map  = graph->GetFieldIdToIndexMap();
    int nnodes  = graph->Size();

    MFEM_ASSERT(y.NumBlocks() == index_map.NumFields(),
                "Number of output blocks (" << y.NumBlocks()
                << ") must match number of fields (" << index_map.NumFields() << ")");

    auto outputs  = graph->OutputFields();
    for(int i=0; i < outputs.Size(); i++)
    {
        int idx = index_map.Get(outputs[i]->ID());
        if(&y[idx] != &x[i]) // copy data, if address is different
        {
            y[idx] = x[i];
        }
    }

    for (int i=nnodes-1; i >= 0; i--)
    {
        auto node = graph->GetNode(i);
        auto node_inputs = node->InputFields();
        auto node_outputs = node->OutputFields();
        x0_mv.SetNumBlocks(node_inputs.Size());
        dx_mv.SetNumBlocks(node_outputs.Size());
        dy_mv.SetNumBlocks(node_inputs.Size());

        for(int i=0; i < node_inputs.Size(); i++)
        {
            int idx = index_map.Get(node_inputs[i]->ID());
            x0_mv.MakeRef(i, xlin[idx]);
            dy_mv.MakeRef(i, y[idx]);
        }
        for(int i=0; i < node_outputs.Size(); i++)
        {
            int idx = index_map.Get(node_outputs[i]->ID());
            dx_mv.MakeRef(i, y[idx]);
        }
        node->GradientMultTranspose(x0_mv, dx_mv, dy_mv); // Compute JVP for the node
    }
}

Operator& GraphGradient::GetGradient(const Vector &x) const
{
    // Used to build Jacobian matrix
    MFEM_ABORT("GraphGradient::GetGradient() not implemented");
}


} // namespace mfem
