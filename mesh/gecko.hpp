// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_GECKO
#define MFEM_GECKO

/*
Documentation for Gecko, version 1.0.0, August 26, 2011.

OVERVIEW

  The Gecko library is a beta version of an algorithm for linear
  ordering of undirected graphs, e.g. for sparse matrix ordering.
  The multilevel algorithm is documented in more detail in

    Peter Lindstrom
    The Edge Product Linear Ordering Problem
    LLNL technical report LLNL-JRNL-496076, August 26, 2011, in review

  and improves substantially on the ordering algorithm proposed in

    Sung-Eui Yoon and Peter Lindstrom
    Mesh Layouts for Block-Based Caches
    IEEE Transactions on Visualization and Computer Graphics,
      12(5):1213-1220, September-October 2006

  This latter paper introduced the edge product (aka. geometric mean)
  ordering functional, but outlined a rather naive strategy for its
  minimization (the recursive method employed by the OpenCCL library).
  Gecko consistently produces orderings of higher quality.

  The Gecko algorithm and C++ implementation were developed by Peter
  Lindstrom at LLNL.  The optimization method in Gecko has been designed
  specifically to minimize the geometric mean functional, but also has
  an interface for specifying other ordering functionals, e.g. the
  arithmetic mean (aka. 1-sum, edge sum) and maximum (aka. infinity-sum,
  bandwidth).

  The Gecko software is part of the LOCAL Toolkit (UCRL-CODE-232243)
  and may be freely used for noncommercial purposes.  Please contact
  the author for commercial use.

DETAILS

  The ordering method is inspired by algebraic multigrid methods, and
  uses V-cycles to coarsen, optimize, and refine the graph layout.
  The graph constitutes an abstract representation of the relationship
  between elements in a data set, e.g. a graph node may represent a
  vertex or a cell in a mesh, a pixel in an image, a node in a binary
  search tree, an element in a sparse matrix, etc.  The graph edges
  represent node affinities, or a desire that adjacent nodes be stored
  close together on linear storage (e.g. disk or main memory).  Such a
  data layout is likely to improve cache utilization in block-based
  caches common on today's computer architectures.  For instance, the
  edges may connect adjacent pixels in an image, as many image
  processing operations involve accessing local neighborhoods.  The
  resulting node layouts are "cache-oblivious," in the sense that no
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

    product  |p(i) - p(j)|^w(i, j)
    ij in E

  or weighted sum

      sum    w(i, j) log(|p(i) - p(j)|)
    ij in E

  where i and j are nodes joined by an edge, w(i, j) is a positive
  edge weight (equal to one unless otherwise specified), and p(i) is
  the integer position of node i in the linear layout of the graph
  (with p(i) = p(j) if and only if i = j).

API

  The interface to the library is quite straightforward, and is
  contained entirely in the public section of the Graph class
  (see inc/graph.h).  A graph is first created by inserting nodes
  and edges.  A single call is then made to order the graph, after
  which the best node permutation found can be queried.  Alternatively,
  the final position of each node may be requested.  Since the layout
  process can be quite slow for large graphs, callbacks can be
  specified (via inheritance) for progress reporting, and the
  algorithm can be pre-empted asynchronously by passing a pointer to
  an external Boolean that is checked periodically.

EXECUTABLE

  The examples directory contains example code that uses the library.
  The executable 'order' reads a graph in Chaco/Metis format from
  standard input and writes the reordered position of each node
  in the graph to standard output.  For information on the file
  format, please see the Metis manual at:

    http://glaros.dtc.umn.edu/gkhome/views/metis/

  In the most simple form of this format, the first line specifies the
  number of nodes N and edges E in the graph, which is followed by N
  lines that, for each node, specify the indices of its neighbors (with
  the first node having index one).  A couple of example graphs are
  also included in the data directory.

  Each line in the output specifies the position (relative to zero)
  of the corresponding input node in the reordered layout.  The graph
  may be weighted to emphasize the relative importance that adjacent
  nodes to be placed together.

  A number of command-line options are provided that greatly control
  the quality of the layout and the running time:

    order <functional> [iterations [window [period [seed [psfile]]]]]

  'functional' is one of h (harmonic mean), g (geometric mean),
  s (square mean root), a (arithmetic mean), r (root mean square),
  and m (maximum), and specifies how edge lengths |p(i) - p(j)| are
  reduced into a single cost measure to be minimized.  Note that the
  algorithm has not been well tuned or tested to optimize functionals
  other than the geometric mean.

  'iterations' specifies the number of V-cycles to perform.  Usually a
  handful of cycles is sufficient.  The default is a single cycle.

  'window' is the number of consecutive nodes optimized concurrently.
  The larger the window, the higher the quality.  However, the running
  time increases exponentially with the window size.  Usually a window
  no larger than six nodes is sufficient.  The default is a window size
  of two nodes.

  'period' is the number of cycles to run between increments of the
  window size.  Usually it is beneficial to start with a small window
  to get a rough layout, and to then increase the window size to
  fine-tune the layout.  The default is a period of one cycle.

  'seed' specifies a random seed (for the sake of reproducibility).
  When the seed is nonzero, the nodes are randomly shuffled prior to
  invoking the ordering algorithm, thereby affecting subsequent
  coarsening and ordering decisions.  In effect, this randomization
  allows different directions to be explored in the combinatorial
  optimization space.  Since the global optimum is seldom (if ever)
  reached, it often makes sense to run several instances of the code,
  each with a new random seed, and to pick the best layout found.
  If the seed is zero, then no shuffling is done.  The default is to
  use the current time as random seed.

  A reasonable parameter choice for good-quality layouts is cycles = 4,
  window = 4, period = 2.  As the window gets progressively larger, the
  running time of each cycle will rapidly increase.

VERSIONS

  1.0.0 (August 26, 2011): Initial beta release.
  1.0.1 (November 28, 2019): Adjusted code style and incorporated into MFEM.
*/

#include "../config/config.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <limits>

#define GECKO_PART_FRAC 4    // ratio of max to min weight for aggregation
#define GECKO_CR_SWEEPS 1    // number of compatible relaxation sweeps
#define GECKO_GS_SWEEPS 1    // number of Gauss-Seidel relaxation sweeps
#define GECKO_ADJLIST 0      // use adjacency list (1) or matrix (0)
#define GECKO_NONRECURSIVE 0 // use nonrecursive permutation algorithm
#define GECKO_WINDOW_MAX 16  // max number of nodes in subgraph

namespace gecko
{

typedef unsigned int uint;

// precision for node positions and computations
#ifdef GECKO_DOUBLE_PRECISION
typedef double Float;
#else
typedef float Float;
#endif

#define GECKO_FLOAT_EPSILON std::numeric_limits<Float>::epsilon()
#define GECKO_FLOAT_MAX std::numeric_limits<Float>::max()

class Functional;
class Progress;


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
//  bool operator<(const WeightedSum& sum) const { return value < sum.value; }
};


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
  class Comparator {
  public:
    Comparator(ConstPtr node) : _node(node) {}
    bool operator()(uint k, uint l) const { return _node[k].pos < _node[l].pos; }
  private:
    const ConstPtr _node;
  };

  // constructor
  Node(Float pos = -1, Float length = 1, Arc::Index arc = Arc::null,
       Node::Index parent = Node::null)
     : pos(pos), hlen(Float(0.5) * length), arc(arc), parent(parent) {}

  Float pos;          // start position at full resolution
  Float hlen;         // half of node length (number of full res nodes)
  Arc::Index arc;     // one past index of last incident arc
  Node::Index parent; // parent in next coarser resolution
};


// Multilevel graph.
class Graph
{
public:
  Graph(uint level = 0);

  // insert node
  Node::Index insert(Float length = 1);

  // insert directed edge (i, j)
  Arc::Index insert(Node::Index i, Node::Index j, Float w = 1, Float b = 1);

  // order graph
  void order(Functional* functional, uint iterations = 1, uint window = 2,
             uint period = 2, uint seed = 0, Progress* progress = 0);

  // optimal permutation found
  const std::vector<Node::Index>& permutation() const { return perm; }

  // position of node i in reordered graph
  uint rank(Node::Index i) const { return (uint)floor(node[i].pos); }

  // cost of current layout
  Float cost() const;

  // number of nodes and edges
  uint nodes() const { return node.size() - 1; }
  uint edges() const { return (adj.size() - 1) / 2; }

protected:
  // remove arc
  void remove(Node::Index i, Node::Index j);

  // find arc
  Arc::Index find(Node::Index i, Node::Index j) const;

  // arc source and target nodes
  Node::Index source(Arc::Index a) const;
  Node::Index target(Arc::Index a) const { return adj[a]; }

  // reverse arc (j, i) of arc a = (i, j)
  Arc::Index reverse(Arc::Index a) const;

  // arc length
  Float length(Node::Index i, Node::Index j) const
  { return fabs(node[i].pos - node[j].pos); }

  Float length(Arc::Index a) const
  {
    Node::Index i = source(a);
    Node::Index j = target(a);
    return length(i, j);
  }

  // is graph directed?
  bool directed(Node::Index* i = 0, Node::Index* j = 0) const;

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

  Arc::Index begin(Node::Index i) const { return node[i - 1].arc; }
  Arc::Index end(Node::Index i) const { return node[i].arc; }
  uint degree(Node::Index i) const { return end(i) - begin(i); }
  bool persistent(Node::Index i) const { return node[i].parent != Node::null; }
  bool placed(Node::Index i) const { return node[i].pos >= 0; }

  Functional* functional;        // ordering functional
  Progress* progress;            // progress callbacks
  std::vector<Node::Index> perm; // ordered list of indices to nodes
  std::vector<Node> node;        // statically ordered list of nodes
  std::vector<Node::Index> adj;  // statically ordered list of adjacent nodes
  std::vector<Float> weight;     // statically ordered list of arc weights
  std::vector<Float> bond;       // statically ordered list of coarsening weights

private:
  // find optimal position of node i while fixing all other nodes
  Float optimal(Node::Index i) const;

  // add contribution of fine arc to coarse graph
  void update(Node::Index i, Node::Index j, Float w, Float b);

  // transfer contribution of fine arc a to coarse node p
  void transfer(Graph* g, const std::vector<Float>& part, Node::Index p,
                Arc::Index a, Float f = 1) const;

  // swap the positions of nodes
  void swap(uint k, uint l);

  uint level; // level of coarsening

  friend class Subgraph;
  friend class Drawing;
};


//// functional.h ///////////////////////////////////////////////////////////////

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

  // compute k'th iteration bond for egde of length l and weight w
  virtual Float bond(Float w, Float l, uint k) const = 0;

  // compute position that minimizes weighted distance to a point set
  virtual Float optimum(const std::vector<WeightedValue>& v) const = 0;
};


// functionals with quasiconvex terms, e.g. p-means with p < 1.
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
    switch (v.size()) {
      case 1:
        // Only one choice.
        break;
      case 2:
        // Functional is the same for both nodes; pick node with
        // larger weight.
        if (v[1].weight > v[0].weight)
          x = v[1].value;
        break;
      default:
        for (auto p = v.begin(); p != v.end(); p++) {
          WeightedSum s;
          for (auto q = v.begin(); q != v.end(); q++) {
            Float l = fabs(p->value - q->value);
            if (l > lmin)
              accumulate(s, WeightedValue(l, q->weight));
          }
          Float f = mean(s);
          if (f < min) {
            min = f;
            x = p->value;
          }
        }
        break;
    }
    return x;
  }
};


// harmonic mean (p = -1)
class FunctionalHarmonic : public FunctionalQuasiconvex
{
public:
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
    return w * pow(l, -Float(3) * k / (k + 1));
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
  WeightedSum sum(const WeightedValue& term) const
  {
    return WeightedSum(term.weight * log(term.value), term.weight);
  }
  Float mean(const WeightedSum& sum) const
  {
    return sum.weight > 0 ? exp(sum.value / sum.weight) : 0;
  }
  Float bond(Float w, Float l, uint k) const
  {
    return w * pow(l, -Float(2) * k / (k + 1));
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
  WeightedSum sum(const WeightedValue& term) const
  {
    return WeightedSum(term.weight * sqrt(term.value), term.weight);
  }
  Float mean(const WeightedSum& sum) const
  {
    return sum.weight > 0 ? (sum.value / sum.weight) * (sum.value / sum.weight) : 0;
  }
  Float bond(Float w, Float l, uint k) const
  {
    return w * pow(l, -Float(1.5) * k / (k + 1));
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
    return w * pow(l, -Float(1) * k / (k + 1));
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
    for (auto p = v.begin(); p != v.end(); p++)
    {
      // Compute f = |sum_{j:pj<pi} wj - sum_{j:pj>pi} wj|.
      Float f = 0;
      for (auto q = v.begin(); q != v.end(); q++)
      {
        if (q->value < p->value)
        {
          f += q->weight;
        }
        else if (q->value > p->value)
        {
          f -= q->weight;
        }
      }
      f = fabs(f);
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
  WeightedSum sum(const WeightedValue& term) const
  {
    return WeightedSum(term.weight * term.value * term.value, term.weight);
  }
  Float mean(const WeightedSum& sum) const
  {
    return sum.weight > 0 ? sqrt(sum.value / sum.weight) : 0;
  }
  Float bond(Float w, Float l, uint k) const
  {
    return w;
  }
  Float optimum(const std::vector<WeightedValue>& v) const
  {
    // Compute the optimum as the weighted mean.
    WeightedSum s;
    for (auto p = v.begin(); p != v.end(); p++) {
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
  Float bond(Float w, Float l, uint k) const
  {
    return Float(1);
  }
  Float optimum(const std::vector<WeightedValue>& v) const
  {
    // Compute the optimum as the midrange.
    Float min = v[0].value;
    Float max = v[0].value;
    for (auto p = v.begin() + 1; p != v.end(); p++) {
      if (p->value < min)
        min = p->value;
      else if (p->value > max)
        max = p->value;
    }
    return (min + max) / 2;
  }
};


//// progress.h /////////////////////////////////////////////////////////////////

// Callbacks between iterations and phases.
class Progress
{
public:
  virtual ~Progress() {}
  virtual void beginorder(const Graph* graph, Float cost) const {}
  virtual void endorder(const Graph* graph, Float cost) const {}
  virtual void beginiter(const Graph* graph, uint iter, uint maxiter, uint window) const {}
  virtual void enditer(const Graph* graph, Float mincost, Float cost) const {}
  virtual void beginphase(const Graph* graph, std::string name) const {};
  virtual void endphase(const Graph* graph, bool show) const {};
  virtual bool quit() const { return false; }
};


} // namespace gecko

#endif // MFEM_GECKO
