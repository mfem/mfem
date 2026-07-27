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

#ifdef MFEM_USE_MPI

namespace mfem
{


void AssembleBlockVectorFromFields(const Array<int> &offsets, const Array<Field*> &fields, Vector &vec)
{
    int nblocks = offsets.Size() - 1;
    MFEM_ASSERT(nblocks == fields.Size(),
                "Number of blocks in offsets does not match number of fields");

    vec.SetSize(offsets.Last());
    for(int i=0; i < nblocks; i++)
    {
        auto field = fields[i];
        MFEM_ASSERT(field->Data() != nullptr,
                    "Field data is null for field: " << field->Name()
                    << " with ID: " << field->ID());
        MFEM_ASSERT(field->Data()->Size() == offsets[i+1] - offsets[i],
                    "Field size mismatch for field: " << field->Name()
                    << " with ID: " << field->ID());

        vec.SetVector(*field->Data(),offsets[i]);
    }
}

void AssembleFieldsFromBlockVector(const Array<int> &offsets, const Vector &vec, Array<Field*> &fields)
{
    int nblocks = offsets.Size() - 1;
    MFEM_ASSERT(nblocks == fields.Size(),
                "Number of blocks in offsets does not match number of fields");

    BlockVector xb(vec.GetData(), offsets);
    for(int i=0; i < nblocks; i++)
    {
        auto field = fields[i];
        MFEM_ASSERT(field->Data() != nullptr,
                    "Field data is null for field: " << field->Name()
                    << " with ID: " << field->ID());
        MFEM_ASSERT(field->Data()->Size() == offsets[i+1] - offsets[i],
                    "Field size mismatch for field: " << field->Name()
                    << " with ID: " << field->ID());

        *field->Data() = xb.GetBlock(i);
    }
}


DAGraph::~DAGraph()
{
    for(int i=0; i < nnodes; i++)
    {
        if(node_owned[i] && nodes[i]) delete nodes[i];
    }
    if(grad) delete grad;
}

void DAGraph::Assemble()
{
    // Sort graph nodes topologically to ensure correct execution order
    TopologicalSort();

    // Compute depth of the graph nodes
    ComputeDepth();

    // Collect all fields from the nodes into the field map
    CollectFields();

    // Update width and height of the DAG from offsets
    // Check that the input and output offsets are consistent
    ValidateOffsets();
    width  = input_offset.Last();
    height = output_offset.Last();

    assembled = true;
}

void DAGraph::ValidateOffsets()
{
    // Check that the input and output offsets are consistent
    // with the number of inputs and outputs
    if(InputFields().Size() > 1)
    {
        MFEM_ASSERT(input_offset.Size() == InputFields().Size() + 1,
                    "Input offsets size inconsistent with number of input fields");
    }
    else
    {
        input_offset = Array<int>({0, nodes[0]->Width()});
    }

    if(OutputFields().Size() > 1)
    {
        MFEM_ASSERT(output_offset.Size() == OutputFields().Size() + 1,
                    "Output offsets size inconsistent with number of output fields");
    }
    else
    {
        output_offset = Array<int>({0, nodes.Last()->Height()});
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

    nodes.Permute(sorted_indices);
    node_owned.Permute(sorted_indices);

    sorted = true;
}

void DAGraph::ComputeDepth()
{
    // Compute depth of ordered nodes
    node_depth.SetSize(nnodes);
    node_depth = 0;
    for(int i=0; i < nnodes; i++)
    {
        int max_dep = 0;
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
                        max_dep = std::max(max_dep, node_depth[j] + 1);
                    }
                }
            }
        }
        node_depth[i] = max_dep;
    }
}

void DAGraph::CollectFields()
{ }

void DAGraph::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(assembled, "DAGraph must be assembled before calling Mult()");

    BlockVector xb(x.GetData(), input_offset);
    BlockVector yb(y.GetData(), output_offset);

    auto inputs = InputFields();
    int ninputs = inputs.Size();
    for(int i=0; i < ninputs; i++)
    {
        auto field = inputs[i];
        field->SetData(&xb.GetBlock(i));
    }

    auto outputs = OutputFields();
    int noutputs = outputs.Size();
    for(int i=0; i < noutputs; i++)
    {
        auto field = outputs[i];
        field->SetData(&yb.GetBlock(i));
    }

    if(input_type == InputType::VECTOR)
    {
        x_node.SetSize(MaxWidth());
        y_node.SetSize(MaxHeight());

        for (int i=0; i < nnodes; i++)
        {
            auto node = nodes[i];
            x_node.SetSize(node->Width());
            y_node.SetSize(node->Height());

            AssembleBlockVectorFromFields(node->InputOffsets(), node->InputFields(), x_node);

            node->Mult(x_node, y_node);

            AssembleFieldsFromBlockVector(node->OutputOffsets(), y_node, node->OutputFields());
        }
    }
    else if(input_type == InputType::MULTIVECTOR)
    {
        MFEM_ABORT("DAGraph::Mult() not implemented for input type: MULTIVECTOR");
    }
    else if(input_type == InputType::NONE)
    {
        Vector x_unused, y_unused;
        for (int i=0; i < nnodes; i++)
        {
            auto node = nodes[i];
            node->Mult(x_unused, y_unused);
        }
    }
    else
    {
        MFEM_ABORT("DAGraph::Mult() not implemented for input type: "
                    << static_cast<int>(input_type));
    }
}

void DAGraph::Execute(const Vector &x, Vector &y)
{
    // Call Mult for now
    Mult(x, y);
}

Operator& DAGraph::GetGradient(const Vector &x) const
{
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

    // Forward pass to to populate (intermediate) fields for differentiation
    fx.SetSize(output_offset.Last());
    xlin.SetSize(x.Size());
    xlin = x; // Store a copy of the input for use in gradient computations

    // Loop through nodes and set execution mode for forward pass
    // This can be used to inform the nodes to build and store Jacobian at x
    for(auto node : nodes)
    {
        node->SetExecutionMode(ExecutionMode::GRADIENT_MODE);
    }

    Mult(xlin, fx); // Forward pass to populate fields for gradient computations

    // Reset execution mode for forward pass
    for(auto node : nodes)
    {
        node->SetExecutionMode(ExecutionMode::DEFAULT_MODE);
    }

    if(grad_mode == GradMode::ASSEMBLED ||
       grad_mode == GradMode::MATRIX_FREE)
    {
        // TODO: Destroy and reallocate GraphGradient to internally 
        //      store the new point of linearization xlin for gradient computations
        //      and populate intermediate fields
        if(!grad)
        {
            grad = new GraphGradient(const_cast<DAGraph*>(this));
        }
        if(grad_mode == GradMode::ASSEMBLED)
        {
            return grad->GetGradient(xlin); // Assembled the Jacobian
        }
        return *grad; // GradMode::MATRIX_FREE
    }
    else
    {
        MFEM_ABORT("DAGraph::GetGradient() not implemented for gradient mode: "
                    << static_cast<int>(grad_mode));
    }

    return *grad;
}

void GraphGradient::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(graph != nullptr, "GraphGradient operator requires a non-null DAGraph pointer.");

    Forward(x, y); // Forward mode: compute JVP, y = J(x) * x
}

void GraphGradient::MultTranspose(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(graph != nullptr, "GraphGradient operator requires a non-null DAGraph pointer.");

    Backward(x, y); // Backward mode: compute VJP, y = J^T(x) * x
}

void GraphGradient::Forward(const Vector &x, Vector &y) const
{
    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);

    auto inputs = graph->InputFields();
    int ninputs = inputs.Size();
    for(int i=0; i < ninputs; i++)
    {
        auto field = inputs[i];
        field->SetAdjoint(&xb.GetBlock(i));
    }

    auto outputs = graph->OutputFields();
    int noutputs = outputs.Size();
    for(int i=0; i < noutputs; i++)
    {
        auto field = outputs[i];
        field->SetAdjoint(&yb.GetBlock(i));
    }

    Vector x0, dx, dy;
    int nnodes  = graph->Size();
    for (int i=0; i < nnodes; i++)
    {
        auto node = graph->GetNode(i);
        node->GradientMult(x0, dx, dy); // Compute JVP for the node
    }
}

void GraphGradient::Backward(const Vector &x, Vector &y) const
{
    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    // Note the switch of input and output offsets for backward mode
    BlockVector xb(x.GetData(), out_offsets);
    BlockVector yb(y.GetData(), in_offsets);

    auto outputs = graph->OutputFields();
    int noutputs = outputs.Size();
    for(int i=0; i < noutputs; i++)
    {
        auto field = outputs[i];
        field->SetAdjoint(&xb.GetBlock(i));
    }

    auto inputs = graph->InputFields();
    int ninputs = inputs.Size();
    for(int i=0; i < ninputs; i++)
    {
        auto field = inputs[i];
        field->SetAdjoint(&yb.GetBlock(i));
    }

    Vector x0, dx, dy;
    int nnodes  = graph->Size();
    for (int i=nnodes-1; i >= 0; i--)
    {
        auto node = graph->GetNode(i);
        node->GradientMultTranspose(x0, dx, dy); // Compute VJP for the node
    }
}

Operator& GraphGradient::GetGradient(const Vector &x) const
{
    // Used to build Jacobian matrix
    MFEM_ABORT("GraphGradient::GetGradient() not implemented");
}


} // namespace mfem
#endif // MFEM_USE_MPI
