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

void Field::GetDerivative(Field* x, Vector &x0, Vector &dydx)
{
    MFEM_ABORT("Field::GetDerivative() not implemented");
}

DAGraph::~DAGraph()
{
    for(int i=0; i < nnodes; i++)
    {
        if(nodes_owned[i] && nodes[i]) delete nodes[i];
    }
    if(grad) delete grad;
}

void DAGraph::Mult(const Vector &x, Vector &y) const
{
    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);

    int ninputs = input_nodes.size();
    for(int i=0; i < ninputs; i++)
    {
        auto data_node = input_nodes[i];
        int index = data_node->GetNodeIndex();
        data_node->SetData(xb.GetBlock(index));
    }

    Vector xtmp, ytmp;
    for (int i=0; i < nnodes; i++)
    {
        nodes[i]->Mult(xtmp, ytmp);
    }

    int noutputs = output_nodes.size();
    for(int i=0; i < noutputs; i++)
    {
        auto data_node = output_nodes[i];
        int index = data_node->GetNodeIndex();
        data_node->GetData(yb.GetBlock(index));
    }
}

void DAGraph::Execute(const Vector &x, Vector &y)
{
    // Call Mult for now
    Mult(x, y);
}

Operator& DAGraph::GetGradient(const Vector &x) const
{
    if(grad_mode == GradMode::FD)
    {
        if(!grad)
        {
            grad = new future::FDJacobian(*this, x, 1e-6);
        }
        else
        {
            auto *fdj = dynamic_cast<future::FDJacobian*>(grad);
            fdj->Update(x);
        }
        return *grad;
    }

    // Forward pass to to populate (intermediate) fields for differentiation
    ytmp.SetSize(out_offsets.Last());
    xgrad.SetSize(x.Size());
    xgrad = x; // Store a copy of the input for use in gradient computations

    // Loop through nodes and set execution mode for forward pass
    // This can be used to inform the nodes to build and store Jacobian at x
    for(auto node : nodes)
    {
        node->SetExecutionMode(ExecutionMode::GRADIENT_MODE);
    }

    Mult(xgrad, ytmp); // Forward pass to populate fields for gradient computations

    // Reset execution mode for forward pass
    for(auto node : nodes)
    {
        node->SetExecutionMode(ExecutionMode::DEFAULT_MODE);
    }

    if(grad_mode == GradMode::FORWARD || grad_mode == GradMode::BACKWARD)
    {
        if(!grad)
        {
            grad = new GraphGradient(const_cast<DAGraph*>(this), grad_mode);
        }
        else
        {
            auto *gg = dynamic_cast<GraphGradient*>(grad);
            gg->SetGradientMode(grad_mode);
        }
        return *grad;
    }
    else
    {
        MFEM_ABORT("DAGraph::GetGradient() not implemented for gradient mode: "
                    << static_cast<int>(grad_mode));
    }

    return *grad;
}

void DAGraph::GetDerivative(Field* y, Field* x, Vector &x0, Vector &dydx)
{
    MFEM_ABORT("DAGraph::GetDerivative() not implemented.");
}

void GraphGradient::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(graph != nullptr, "GraphGradient operator requires a non-null DAGraph pointer.");

    if(grad_mode == GradMode::FD)
    {
        MFEM_ABORT("GraphGradient::Mult() not implemented for finite difference gradient.");
    }
    else if(grad_mode == GradMode::FORWARD)
    {
        // Forward mode: compute JVP, y = J(x) * x
        Forward(x, y);
    }
    else if(grad_mode == GradMode::BACKWARD)
    {
        // Backward mode: compute VJP, y = J^T(x) * x
        Backward(x, y);
    }
    else
    {
        MFEM_ABORT("GraphGradient::Mult() not implemented for gradient mode: "
                    << static_cast<int>(grad_mode));
    }
}

void GraphGradient::Forward(const Vector &x, Vector &y) const
{
    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    BlockVector xb(x.GetData(), in_offsets);
    BlockVector yb(y.GetData(), out_offsets);

    int nnodes  = graph->Size();
    auto fields = graph->Fields();

    int ninputs = graph->input_nodes.size();
    for(int i=0; i < ninputs; i++)
    {
        auto data_node = graph->input_nodes[i]; 
        int index = data_node->GetNodeIndex();
        data_node->SetAdjoint(xb.GetBlock(index)); // Seed input adjoint from input block
    }

    Vector xtmp, ytmp;
    for (int i=0; i < nnodes; i++)
    {
        auto node = graph->GetNode(i);
        node->JVP(xtmp, ytmp);
    }

    int noutputs = graph->output_nodes.size();
    for(int i=0; i < noutputs; i++)
    {
        auto data_node = graph->output_nodes[i];
        int index = data_node->GetNodeIndex();
        data_node->GetAdjoint(yb.GetBlock(index));
    }
}

void GraphGradient::Backward(const Vector &x, Vector &y) const
{
    auto in_offsets  = graph->InputOffsets();
    auto out_offsets = graph->OutputOffsets();

    // Note the switch of input and output offsets for backward mode
    BlockVector xb(x.GetData(), out_offsets);
    BlockVector yb(y.GetData(), in_offsets);

    int nnodes  = graph->Size();
    auto fields = graph->Fields();

    int noutputs = graph->output_nodes.size();
    for(int i=0; i < noutputs; i++)
    {
        auto data_node = graph->output_nodes[i];
        int index = data_node->GetNodeIndex();
        data_node->SetAdjoint(xb.GetBlock(index)); // Seed output adjoint from output block
    }

    Vector xtmp, ytmp;
    for (int i=nnodes-1; i >= 0; i--)
    {
        auto node = graph->GetNode(i);
        node->VJP(xtmp, ytmp);
    }

    int ninputs = graph->input_nodes.size();
    for(int i=0; i < ninputs; i++)
    {
        auto data_node = graph->input_nodes[i];
        int index = data_node->GetNodeIndex();
        data_node->GetAdjoint(yb.GetBlock(index));
    }
}

Operator& GraphGradient::GetGradient(const Vector &x) const
{
    // Used to build Jacobian matrix
    MFEM_ABORT("GraphGradient::GetGradient() not implemented");
}


} // namespace mfem
#endif // MFEM_USE_MPI
