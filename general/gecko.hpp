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

#ifndef MFEM_GECKO_HPP
#define MFEM_GECKO_HPP

// This file collects the sources of the Gecko library as a single module.
// The original library can be found at https://github.com/LLNL/gecko
// Used here with permission.

// ------------------------------------------------------------------------------

// BSD 3-Clause License
//
// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// ------------------------------------------------------------------------------

// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and other
// gecko project contributors. See the above license for details.
// SPDX-License-Identifier: BSD-3-Clause
// LLNL-CODE-800597

/* ------------------------------------------------------------------------------

Gecko
=====

Gecko is a C++ library for solving graph linear arrangement problems.  Gecko
orders graph nodes, representing data elements, connected by undirected edges,
representing affinity relations, with the goal of minimizing a chosen
functional of edge length.  Gecko was primarily designed to minimize the
product of edge lengths but can also be used to reduce bandwidth (maximum
edge length) and 1-sum (sum of edge lengths), among others.
Minimum-edge-product orderings generalize space-filling curve orderings to
geometryless graphs and find applications in data locality optimization,
graph partitioning, and dimensionality reduction.


Author
------

Gecko was written by [Peter Lindstrom](https://people.llnl.gov/pl) at
Lawrence Livermore National Laboratory.


Algorithm
=========

Gecko orders the nodes of an undirected and optionally weighted graph in an
effort to place nodes connected by an edge in consecutive positions in the
linear ordering (aka. *layout*).  Such orderings promote good data locality,
e.g., to improve cache utilization, but also find applications in graph
partitioning and dimensionality reduction.

The gecko ordering method is inspired by algebraic multigrid methods, and
uses V-cycles to coarsen, optimize, and refine the graph layout.
The graph constitutes an abstract representation of the relationship
between elements in a data set, e.g., a graph node may represent a
vertex or a cell in a mesh, a pixel in an image, a node in a binary
search tree, an element in a sparse matrix, etc.  The graph edges
represent node affinities, or a desire that adjacent nodes be stored
close together on linear storage (e.g., disk or main memory).  Such a
data layout is likely to improve cache utilization in block-based
caches common on today's computer architectures.  For instance, the
edges may connect adjacent pixels in an image, as many image
processing operations involve accessing local neighborhoods.  The
resulting node layouts are "cache-oblivious" in the sense that no
particular knowledge of the cache parameters (number and size of
blocks, associativity, replacement policy, etc.) are accounted for.
Rather, the expectation is that the layouts will provide good
locality across all levels of cache.  Note that the ordering method
accepts any undirected graph, whether it represent a structured or
unstructured data set, and is also oblivious of any geometric
structure inherent in the data set.

The optimization algorithm attempts to order the nodes of the graph
so as to minimize the geometric mean edge length, or equivalently
the product

    product |p(i) - p(j)|^w(i, j)

or weighted sum

    sum w(i, j) log(|p(i) - p(j)|)

where *i* and *j* are nodes joined by an edge, *w*(*i*, *j*) is a positive
edge weight (equal to one unless otherwise specified), *p*(*i*) is
the integer position of node *i* in the linear layout of the graph
(with *p*(*i*) = *p*(*j*) if and only if *i* = *j*), and where the product
or sum is over all edges of the graph.

The algorithm is described in further detail in the paper

* Peter Lindstrom
  The Minimum Edge Product Linear Ordering Problem
  LLNL technical report LLNL-TR-496076, August 26, 2011.


Ordering Parameters
-------------------

The `Graph::order()` function and the `gecko` command-line executable take a
number of parameters that govern the layout process.  These parameters are
described below:

* The **functional** is the objective being optimized and expresses the cost
  of the graph layout in terms of some average of its edge lengths
  |*p*(*i*) - *p*(*j*)|.  The predefined functionals are
  * `h` (harmonic mean)
  * `g` (geometric mean)
  * `s` (square mean root)
  * `a` (arithmetic mean)
  * `r` (root mean square)
  * `m` (maximum)

  Note that the algorithm has not been well tuned or tested to optimize
  functionals other than the geometric mean.

* The number of **iterations** specifies the number of multigrid V-cycles
  to perform.  Usually a handful of cycles is sufficient.  The default is
  a single cycle.

* The optimization **window** is the number of consecutive nodes optimized
  concurrently using exhaustive search.  The larger the window, the higher
  the quality.  Note that the running time increases exponentially with the
  window size.  Usually a window no larger than six nodes is sufficient.
  The default is a window size of two nodes.

* The **period** is the number of V-cycles to run between increments of the
  window size.  Usually it is beneficial to start with a small window to get
  a rough layout, and to then increase the window size to fine-tune the
  layout.  The default is a period of one cycle.

* The random **seed** allows injecting some randomness in the optimization
  process.  When the seed is nonzero, the nodes are randomly shuffled prior
  to invoking the ordering algorithm, thereby affecting subsequent coarsening
  and ordering decisions.  In effect, this randomization allows different
  directions to be explored in the combinatorial optimization space.  Fixing
  the seed allows for reproducibility, i.e., the same seed always leads to
  the same layout.  Since the global optimum is seldom (if ever) reached,
  it often makes sense to run several instances of the algorithm, each with
  a new random seed, and to pick the best layout found.  In the gecko
  executable, the current time is used as random seed if not specified.

A reasonable parameter choice for good-quality layouts is:

* iterations = 4
* window = 4
* period = 2

*/

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <cfloat>
#include <limits>

// ----- types.h ----------------------------------------------------------------

#define GECKO_FLOAT_EPSILON std::numeric_limits<Float>::epsilon()
#define GECKO_FLOAT_MAX std::numeric_limits<Float>::max()

namespace Gecko
{

typedef unsigned int uint;

// precision for node positions and computations
#if GECKO_WITH_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif
}

// ----- functional.h -----------------------------------------------------------

namespace Gecko
{

// abstract base class for weighted terms and sums
class WeightedValue
{
public:
   WeightedValue(Float value, Float weight) : value(value), weight(weight) {}
   Float value;
   Float weight;
};

// weighted sum of terms
class WeightedSum : public WeightedValue
{
public:
   WeightedSum(Float value = 0, Float weight = 0) : WeightedValue(value, weight) {}
};

// abstract base class for ordering functionals
class Functional
{
public:
   virtual ~Functional() {}

   virtual WeightedSum sum(const WeightedSum& s, const WeightedValue& t) const
   {
      WeightedSum tot = s;
      accumulate(tot, sum(t));
      return tot;
   }

   virtual WeightedSum sum(const WeightedSum& s, const WeightedSum& t) const
   {
      WeightedSum tot = s;
      accumulate(tot, t);
      return tot;
   }

   // add weighted term to weighted sum
   virtual void accumulate(WeightedSum& s, const WeightedValue& t) const
   {
      accumulate(s, sum(t));
   }

   // add two weighted sums
   virtual void accumulate(WeightedSum& s, const WeightedSum& t) const
   {
      s.value += t.value;
      s.weight += t.weight;
   }

   // is s potentially less than t?
   virtual bool less(const WeightedSum& s, const WeightedSum& t) const
   {
      return s.value < t.value;
   }

   // transform term into weighted sum
   virtual WeightedSum sum(const WeightedValue& term) const = 0;

   // compute weighted mean from a weighted sum
   virtual Float mean(const WeightedSum& sum) const = 0;

   // compute k'th iteration bond for edge of length l and weight w
   virtual Float bond(Float w, Float l, uint k) const = 0;

   // compute position that minimizes weighted distance to a point set
   virtual Float optimum(const std::vector<WeightedValue>& v) const = 0;
};

// functionals with quasiconvex terms, e.g., p-means with p < 1.
class FunctionalQuasiconvex : public Functional
{
protected:
   Float optimum(const std::vector<WeightedValue>& v, Float lmin) const
   {
      // Compute the optimum as the node position that minimizes the
      // functional.  Any nodes coincident with each candidate position
      // are excluded from the functional.
      Float x = v[0].value;
      Float min = GECKO_FLOAT_MAX;
      switch (v.size())
      {
         case 1:
            // Only one choice.
            break;
         case 2:
            // Functional is the same for both nodes; pick node with
            // larger weight.
            if (v[1].weight > v[0].weight)
            {
               x = v[1].value;
            }
            break;
         default:
            for (std::vector<WeightedValue>::const_iterator p = v.begin(); p != v.end();
                 p++)
            {
               WeightedSum s;
               for (std::vector<WeightedValue>::const_iterator q = v.begin(); q != v.end();
                    q++)
               {
                  Float l = std::fabs(p->value - q->value);
                  if (l > lmin)
                  {
                     accumulate(s, WeightedValue(l, q->weight));
                  }
               }
               Float f = mean(s);
               if (f < min)
               {
                  min = f;
                  x = p->value;
               }
            }
            break;
      }
      return x;
   }
private:
   using Functional::optimum; // silence overload vs. override warning
};

// harmonic mean (p = -1)
class FunctionalHarmonic : public FunctionalQuasiconvex
{
public:
   using Functional::sum;
   bool less(const WeightedSum& s, const WeightedSum& t) const
   {
      // This is only a loose bound when s.weight < t.weight.
      return s.value - s.weight > t.value - t.weight;
   }
   WeightedSum sum(const WeightedValue& term) const
   {
      return WeightedSum(term.weight / term.value, term.weight);
   }
   Float mean(const WeightedSum& sum) const
   {
      return sum.weight > 0 ? sum.weight / sum.value : 0;
   }
   Float bond(Float w, Float l, uint k) const
   {
      return w * std::pow(l, -Float(3) * Float(k) / Float(k + 1));
   }
   Float optimum(const std::vector<WeightedValue>& v) const
   {
      return FunctionalQuasiconvex::optimum(v, Float(0.5));
   }
};

// geometric mean (p = 0)
class FunctionalGeometric : public FunctionalQuasiconvex
{
public:
   using Functional::sum;
   WeightedSum sum(const WeightedValue& term) const
   {
      return WeightedSum(term.weight * std::log(term.value), term.weight);
   }
   Float mean(const WeightedSum& sum) const
   {
      return sum.weight > 0 ? std::exp(sum.value / sum.weight) : 0;
   }
   Float bond(Float w, Float l, uint k) const
   {
      return w * std::pow(l, -Float(2) * Float(k) / Float(k + 1));
   }
   Float optimum(const std::vector<WeightedValue>& v) const
   {
      return FunctionalQuasiconvex::optimum(v, Float(0.5));
   }
};

// square mean root (p = 1/2)
class FunctionalSMR : public FunctionalQuasiconvex
{
public:
   using Functional::sum;
   WeightedSum sum(const WeightedValue& term) const
   {
      return WeightedSum(term.weight * std::sqrt(term.value), term.weight);
   }
   Float mean(const WeightedSum& sum) const
   {
      return sum.weight > 0 ? (sum.value / sum.weight) * (sum.value / sum.weight) : 0;
   }
   Float bond(Float w, Float l, uint k) const
   {
      return w * std::pow(l, -Float(1.5) * Float(k) / Float(k + 1));
   }
   Float optimum(const std::vector<WeightedValue>& v) const
   {
      return FunctionalQuasiconvex::optimum(v, Float(0.0));
   }
};

// arithmetic mean (p = 1)
class FunctionalArithmetic : public Functional
{
public:
   using Functional::sum;
   WeightedSum sum(const WeightedValue& term) const
   {
      return WeightedSum(term.weight * term.value, term.weight);
   }
   Float mean(const WeightedSum& sum) const
   {
      return sum.weight > 0 ? sum.value / sum.weight : 0;
   }
   Float bond(Float w, Float l, uint k) const
   {
      return w * std::pow(l, -Float(1) * Float(k) / Float(k + 1));
   }
   Float optimum(const std::vector<WeightedValue>& v) const
   {
      // Compute the optimum as the weighted median.  Since the median may
      // not be unique, the largest interval [x, y] is computed and its
      // centroid is chosen.  The optimum must occur at a node, and hence
      // we consider each node position pi at a time and the relative
      // positions of the remaining nodes pj.
      Float x = 0;
      Float y = 0;
      Float min = GECKO_FLOAT_MAX;
      for (std::vector<WeightedValue>::const_iterator p = v.begin(); p != v.end();
           p++)
      {
         // Compute f = |sum_{j:pj<pi} wj - sum_{j:pj>pi} wj|.
         Float f = 0;
         for (std::vector<WeightedValue>::const_iterator q = v.begin(); q != v.end();
              q++)
            if (q->value < p->value)
            {
               f += q->weight;
            }
            else if (q->value > p->value)
            {
               f -= q->weight;
            }
         f = std::fabs(f);
         // Update interval if f is minimal.
         if (f <= min)
         {
            if (f < min)
            {
               min = f;
               x = y = p->value;
            }
            else
            {
               x = std::min(x, p->value);
               y = std::max(y, p->value);
            }
         }
      }
      return (x + y) / 2;
   }
};

// root mean square (p = 2)
class FunctionalRMS : public Functional
{
public:
   using Functional::sum;
   WeightedSum sum(const WeightedValue& term) const
   {
      return WeightedSum(term.weight * term.value * term.value, term.weight);
   }
   Float mean(const WeightedSum& sum) const
   {
      return sum.weight > 0 ? std::sqrt(sum.value / sum.weight) : 0;
   }
   Float bond(Float w, Float, uint) const
   {
      return w;
   }
   Float optimum(const std::vector<WeightedValue>& v) const
   {
      // Compute the optimum as the weighted mean.
      WeightedSum s;
      for (std::vector<WeightedValue>::const_iterator p = v.begin(); p != v.end();
           p++)
      {
         s.value += p->weight * p->value;
         s.weight += p->weight;
      }
      return s.value / s.weight;
   }
};

// maximum (p = infinity)
class FunctionalMaximum : public Functional
{
public:
   using Functional::sum;
   using Functional::accumulate;
   WeightedSum sum(const WeightedValue& term) const
   {
      return WeightedSum(term.value, term.weight);
   }
   void accumulate(WeightedSum& s, const WeightedSum& t) const
   {
      s.value = std::max(s.value, t.value);
   }
   Float mean(const WeightedSum& sum) const
   {
      return sum.value;
   }
   Float bond(Float, Float, uint) const
   {
      return Float(1);
   }
   Float optimum(const std::vector<WeightedValue>& v) const
   {
      // Compute the optimum as the midrange.
      Float min = v[0].value;
      Float max = v[0].value;
      for (std::vector<WeightedValue>::const_iterator p = v.begin() + 1; p != v.end();
           p++)
      {
         if (p->value < min)
         {
            min = p->value;
         }
         else if (p->value > max)
         {
            max = p->value;
         }
      }
      return (min + max) / 2;
   }
};

}

// ----- progress.h -------------------------------------------------------------

namespace Gecko
{

class Graph;

// Callbacks between iterations and phases.
class Progress
{
public:
   virtual ~Progress() {}
   virtual void beginorder(const Graph* /*graph*/, Float /*cost*/) const {}
   virtual void endorder(const Graph* /*graph*/, Float /*cost*/) const {}
   virtual void beginiter(const Graph* /*graph*/, uint /*iter*/, uint /*maxiter*/,
                          uint /*window*/) const {}
   virtual void enditer(const Graph* /*graph*/, Float /*mincost*/,
                        Float /*cost*/) const {}
   virtual void beginphase(const Graph* /*graph*/, std::string /*name*/) const {};
   virtual void endphase(const Graph* /*graph*/, bool /*show*/) const {};
   virtual bool quit() const { return false; }
};

}

// ----- graph.h ----------------------------------------------------------------

namespace Gecko
{

// Multilevel graph arc.
class Arc
{
public:
   typedef uint Index;
   typedef std::vector<Index>::const_iterator ConstPtr;
   enum { null = 0 };
};

// Multilevel graph node.
class Node
{
public:
   typedef uint Index;
   typedef std::vector<Node>::const_iterator ConstPtr;
   enum { null = 0 };

   // comparator for sorting node indices
   class Comparator
   {
   public:
      Comparator(ConstPtr node_) : node(node_) {}
      bool operator()(uint k, uint l) const { return node[k].pos < node[l].pos; }
   private:
      const ConstPtr node;
   };

   // constructor
   Node(Float pos = -1, Float length = 1, Arc::Index arc = Arc::null,
        Node::Index parent = Node::null) : pos(pos), hlen(Float(0.5) * length),
      arc(arc), parent(parent) {}

   Float pos;          // start position at full resolution
   Float hlen;         // half of node length (number of full res nodes)
   Arc::Index arc;     // one past index of last incident arc
   Node::Index parent; // parent in next coarser resolution
};

// Multilevel graph.
class Graph
{
public:
   // constructor of graph with given (initial) number of nodes
   Graph(uint nodes = 0) : level(0), last_node(Node::null) { init(nodes); }

   // number of nodes and edges
   uint nodes() const { return uint(node.size() - 1); }
   uint edges() const { return uint((adj.size() - 1) / 2); }

   // insert node and return its index
   Node::Index insert_node(Float length = 1);

   // outgoing arcs {begin, ..., end-1} originating from node i
   Arc::Index node_begin(Node::Index i) const { return node[i - 1].arc; }
   Arc::Index node_end(Node::Index i) const { return node[i].arc; }

   // node degree and neighbors
   uint node_degree(Node::Index i) const { return node_end(i) - node_begin(i); }
   std::vector<Node::Index> node_neighbors(Node::Index i) const;

   // insert directed edge (i, j)
   Arc::Index insert_arc(Node::Index i, Node::Index j, Float w = 1, Float b = 1);

   // remove arc or edge
   bool remove_arc(Arc::Index a);
   bool remove_arc(Node::Index i, Node::Index j);
   bool remove_edge(Node::Index i, Node::Index j);

   // index of arc (i, j) or null if not present
   Arc::Index arc_index(Node::Index i, Node::Index j) const;

   // arc source and target nodes and weight
   Node::Index arc_source(Arc::Index a) const;
   Node::Index arc_target(Arc::Index a) const { return adj[a]; }
   Float arc_weight(Arc::Index a) const { return weight[a]; }

   // reverse arc (j, i) of arc a = (i, j)
   Arc::Index reverse_arc(Arc::Index a) const;

   // order graph
   void order(Functional* functional, uint iterations = 1, uint window = 2,
              uint period = 2, uint seed = 0, Progress* progress = 0);

   // optimal permutation found
   const std::vector<Node::Index>& permutation() const { return perm; }

   // node of given rank in reordered graph (0 <= rank <= nodes() - 1)
   Node::Index permutation(uint rank) const { return perm[rank]; }

   // position of node i in reordered graph (1 <= i <= nodes())
   uint rank(Node::Index i) const { return static_cast<uint>(std::floor(node[i].pos)); }

   // cost of current layout
   Float cost() const;

   // return first directed arc if one exists or null otherwise
   Arc::Index directed() const;

protected:
   friend class Subgraph;
   friend class Drawing;

   // constructor/destructor
   Graph(uint nodes, uint level) : level(level), last_node(Node::null) { init(nodes); }

   // arc length
   Float length(Node::Index i, Node::Index j) const { return std::fabs(node[i].pos - node[j].pos); }
   Float length(Arc::Index a) const
   {
      Node::Index i = arc_source(a);
      Node::Index j = arc_target(a);
      return length(i, j);
   }

   // coarsen graph
   Graph* coarsen();

   // refine graph
   void refine(const Graph* graph);

   // perform m sweeps of compatible or Gauss-Seidel relaxation
   void relax(bool compatible, uint m = 1);

   // optimize using n-node window
   void optimize(uint n);

   // place all nodes according to their positions
   void place(bool sort = false);

   // place nodes {k, ..., k + n - 1} according to their positions
   void place(bool sort, uint k, uint n);

   // perform V cycle using n-node window
   void vcycle(uint n, uint work = 0);

   // randomly shuffle nodes
   void shuffle(uint seed = 0);

   // recompute arc bonds for iteration i
   void reweight(uint i);

   // compute cost
   WeightedSum cost(const std::vector<Arc::Index>& subset, Float pos) const;

   // node attributes
   bool persistent(Node::Index i) const { return node[i].parent != Node::null; }
   bool placed(Node::Index i) const { return node[i].pos >= Float(0); }

   Functional* functional;        // ordering functional
   Progress* progress;            // progress callbacks
   std::vector<Node::Index> perm; // ordered list of indices to nodes
   std::vector<Node> node;        // statically ordered list of nodes
   std::vector<Node::Index> adj;  // statically ordered list of adjacent nodes
   std::vector<Float> weight;     // statically ordered list of arc weights
   std::vector<Float> bond;       // statically ordered list of coarsening weights

private:
   // initialize graph with given number of nodes
   void init(uint nodes);

   // find optimal position of node i while fixing all other nodes
   Float optimal(Node::Index i) const;

   // add contribution of fine arc to coarse graph
   void update(Node::Index i, Node::Index j, Float w, Float b);

   // transfer contribution of fine arc a to coarse node p
   void transfer(Graph* g, const std::vector<Float>& part, Node::Index p,
                 Arc::Index a, Float f = 1) const;

   // swap the positions of nodes
   void swap(uint k, uint l);

   // random number generator
   static uint random(uint seed = 0);

   uint level;            // level of coarsening
   Node::Index last_node; // last node with outgoing arcs
};

}

#endif // MFEM_GECKO_HPP
