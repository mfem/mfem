
// #include <memory>
// #include "mortarassemble.hpp"


// namespace mfem
// {

// void Print(const IntegrationRule &ir, std::ostream &os)
// {
//    os << "points:\n[\n";
//    for (int i = 0; i < ir.Size(); ++i)
//    {
//       os << "\t" << ir[i].x << ", " << ir[i].y << ", " << ir[i].z << "\n";
//    }

//    os << "]\n";

//    os << "weights:\n[\n\t";
//    for (int i = 0; i < ir.Size(); ++i)
//    {
//       os << ir[i].weight << " ";
//    }
//    os << "\n]\n";
// }

// double SumOfWeights(const IntegrationRule &ir)
// {
//    double ret = 0;
//    for (int i = 0; i < ir.Size(); ++i)
//    {
//       ret += ir[i].weight;
//    }
//    return ret;
// }

// void MakeCompositeQuadrature2D(const DenseMatrix &polygon, const double weight,
//                                const int order, IntegrationRule &c_ir)
// {
//    const IntegrationRule &ir = IntRules.Get(Geometry::TRIANGLE, order);
//    const int n_triangles     = polygon.Width() - 2;
//    const int n_quad_points   = n_triangles * ir.Size();

//    assert(fabs(SumOfWeights(ir) - 0.5) < 1e-8);

//    double triangle[3*2] = { polygon.Data()[0], polygon.Data()[1], 0., 0., 0., 0. };

//    Vector o, u, v, p;
//    polygon.GetColumn(0, o);

//    c_ir.SetSize(n_quad_points);
//    Intersector isector;

//    double relative_weight = 0;

//    int quad_index = 0;
//    for (int i = 2; i < polygon.Width(); ++i)
//    {
//       polygon.GetColumn(i-1, u);
//       polygon.GetColumn(i, v);

//       triangle[2] = u[0];
//       triangle[3] = u[1];

//       triangle[4] = v[0];
//       triangle[5] = v[1];


//       u -= o;
//       v -= o;

//       const double scale = fabs(isector.polygon_area_2(3, triangle)) / ( weight );
//       relative_weight += scale;

//       for (int k = 0; k < ir.Size(); ++k, ++quad_index)
//       {
//          auto &qp    = ir[k];
//          auto &c_qp  = c_ir[quad_index];

//          p = o;
//          add(p, qp.x, u, p);
//          add(p, qp.y, v, p);

//          c_qp.x      = p(0);
//          c_qp.y      = p(1);
//          c_qp.z      = 0.0;
//          c_qp.weight = scale * qp.weight;
//       }
//    }

//    assert(relative_weight <= 1.0001);
//    assert(quad_index == n_quad_points);
// }

// void MakeCompositeQuadrature3D(const Polyhedron &polyhedron,
//                                const double weight, const int order, IntegrationRule &c_ir)
// {
//    using std::min;

//    Intersector isector;

//    const IntegrationRule &ir    = IntRules.Get(Geometry::TETRAHEDRON, order);
//    const int n_sub_elements      = isector.n_volume_elements(polyhedron);
//    const int total_n_quad_points = n_sub_elements * ir.Size();

//    std::vector<double> ref_quad_points (ir.Size() * 3);
//    std::vector<double> ref_quad_weights(ir.Size());

//    for (int i = 0; i < ir.Size(); ++i)
//    {
//       const int offset = i * 3;

//       ref_quad_points[offset    ] = ir[i].x;
//       ref_quad_points[offset + 1] = ir[i].y;
//       ref_quad_points[offset + 2] = ir[i].z;

//       ref_quad_weights[i]     = ir[i].weight;
//    }

//    c_ir.SetSize(total_n_quad_points);

//    //global quadrature points
//    double quad_points     [MAX_QUAD_POINTS * 3];
//    double quad_weights    [MAX_QUAD_POINTS];

//    double barycenter_p[3];
//    isector.row_average( polyhedron.n_nodes, polyhedron.n_dims, polyhedron.points,
//                         barycenter_p);

//    const int max_n_sub_els = isector.max_n_elements_from_facets(polyhedron);
//    const int max_n_sub_inc = std::max(1,
//                                       MAX_QUAD_POINTS / (max_n_sub_els * ir.Size()));

//    const double scale = 1.0 / ( weight );

//    int mfem_quad_index = 0;

//    if (n_sub_elements == 1)
//    {
//       isector.tetrahedron_transform(polyhedron.points, total_n_quad_points,
//                                     &ref_quad_points[0], quad_points);

//       const double w = fabs(isector.m_tetrahedron_volume(polyhedron.points) * scale);
//       for (int i = 0; i < total_n_quad_points; ++i, ++mfem_quad_index)
//       {
//          const int offset = i * 3;
//          c_ir[mfem_quad_index].x = quad_points[offset    ];
//          c_ir[mfem_quad_index].y = quad_points[offset + 1];
//          c_ir[mfem_quad_index].z = quad_points[offset + 2];

//          c_ir[mfem_quad_index].weight = ref_quad_weights[i] * w;
//       }

//    }
//    else
//    {

//       for (int begin_k = 0; begin_k < polyhedron.n_elements;)
//       {
//          const int end_k = min(begin_k + max_n_sub_inc, polyhedron.n_elements);
//          assert(end_k > begin_k && "end_k > begin_k");

//          const int n_quad_points =
//             isector.make_quadrature_points_from_polyhedron_in_range_around_point(
//                polyhedron,
//                begin_k,
//                end_k,
//                scale,
//                ir.Size(),
//                &ref_quad_points [0],
//                &ref_quad_weights[0],
//                barycenter_p,
//                quad_points,
//                quad_weights
//             );

//          for (int i = 0; i < n_quad_points; ++i, ++mfem_quad_index)
//          {
//             const int offset = i * 3;
//             c_ir[mfem_quad_index].x = quad_points[offset    ];
//             c_ir[mfem_quad_index].y = quad_points[offset + 1];
//             c_ir[mfem_quad_index].z = quad_points[offset + 2];

//             c_ir[mfem_quad_index].weight = quad_weights[i];
//          }

//          begin_k = end_k;
//       }
//    }
// }

// void TransformToReference(ElementTransformation &Trans, const int type,
//                           const IntegrationRule &global_ir, IntegrationRule &ref_ir)
// {
//    const int dim = Trans.GetSpaceDim();
//    Vector p(dim);
//    p = 0.0;

//    ref_ir.SetSize(global_ir.Size());

//    for (int i = 0; i < global_ir.Size(); ++i)
//    {
//       p(0) = global_ir[i].x;
//       p(1) = global_ir[i].y;

//       if (p.Size() > 2)
//       {
//          p(2) = global_ir[i].z;
//       }

//       Trans.TransformBack(p, ref_ir[i]);
//       ref_ir[i].weight = global_ir[i].weight;

//       assert(ref_ir[i].x >= -1e-8);
//       assert(ref_ir[i].y >= -1e-8);
//       assert(ref_ir[i].z >= -1e-8);

//       assert(ref_ir[i].x <= 1 + 1e-8);
//       assert(ref_ir[i].y <= 1 + 1e-8);
//       assert(ref_ir[i].z <= 1 + 1e-8);

//       if (type != Geometry::TRIANGLE && dim == 2)
//       {
//          ref_ir[i].weight *= 2;
//       }
//       else if (type != Geometry::TETRAHEDRON && dim == 3)
//       {
//          ref_ir[i].weight *= 6;
//       }
//       else
//       {
//          assert(dim == 2 || dim == 3);
//       }
//    }
// }

// void MortarAssemble(
//    const FiniteElement &trial_fe,
//    const IntegrationRule &trial_ir,
//    const FiniteElement &test_fe,
//    const IntegrationRule &test_ir,
//    ElementTransformation &Trans,
//    DenseMatrix &elmat)
// {
//    int tr_nd = trial_fe.GetDof();
//    int te_nd = test_fe.GetDof();
//    double w;

//    Vector shape, te_shape;

//    elmat.SetSize (te_nd, tr_nd);
//    shape.SetSize (tr_nd);
//    te_shape.SetSize (te_nd);

//    elmat = 0.0;
//    for (int i = 0; i < test_ir.GetNPoints(); i++)
//    {
//       const IntegrationPoint &trial_ip = trial_ir.IntPoint(i);
//       const IntegrationPoint &test_ip  = test_ir.IntPoint(i);
//       Trans.SetIntPoint (&test_ip);

//       trial_fe.CalcShape(trial_ip, shape);
//       test_fe.CalcShape(test_ip, te_shape);

//       w = Trans.Weight() * test_ip.weight;

//       te_shape *= w;
//       AddMultVWt(te_shape, shape, elmat);
//    }
// }

// double Sum(const DenseMatrix &mat)
// {
//    Vector rs(mat.Width());
//    mat.GetRowSums(rs);
//    return rs.Sum();
// }


// bool Intersect2D(const DenseMatrix &poly1, const DenseMatrix &poly2,
//                  DenseMatrix &intersection)
// {
//    Intersector isector;
//    double result_buffer[MAX_N_ISECT_POINTS * 2];
//    int n_vertices_result;

//    assert( isector.polygon_area_2(poly1.Width(),  poly1.Data()) > 0 );
//    assert( isector.polygon_area_2(poly2.Width(),  poly2.Data()) > 0 );

//    if (!isector.intersect_convex_polygons(poly1.Width(), poly1.Data(),
//                                           poly2.Width(), poly2.Data(),
//                                           &n_vertices_result, result_buffer,
//                                           DEFAULT_TOLLERANCE))
//    {
//       return false;
//    }

//    assert( isector.polygon_area_2(n_vertices_result,  result_buffer) > 0 );

//    intersection.SetSize(2, n_vertices_result);
//    std::copy(result_buffer, result_buffer + n_vertices_result * 2,
//              intersection.Data());
//    return true;
// }


// void MakePolyhedron(const Mesh &m, const int el_index, Polyhedron &polyhedron)
// {
//    //TODO check counter-clockwise faces
//    using namespace std;

//    const int dim = m.Dimension();

//    assert(m.GetElement(el_index));

//    const Element &e = *m.GetElement(el_index);
//    const int e_type = e.GetType();

//    Array<int> faces, cor;
//    m.GetElementFaces(el_index, faces, cor);

//    DenseMatrix pts;
//    m.GetPointMatrix(el_index, pts);

//    Array<int> vertices;
//    m.GetElementVertices(el_index, vertices);

//    polyhedron.n_elements = faces.Size();
//    polyhedron.n_nodes     = vertices.Size();
//    polyhedron.n_dims   = 3;
//    polyhedron.el_ptr[0]  = 0;

//    for (int i = 0; i < vertices.Size(); ++i)
//    {
//       const int offset = i * dim;

//       for (int j = 0; j < dim; ++j)
//       {
//          polyhedron.points[offset + j] = pts(j, i);
//       }
//    }

//    Array<int> f2v;
//    for (int i = 0; i < faces.Size(); ++i)
//    {
//       m.GetFaceVertices(faces[i], f2v);
//       const int eptr = polyhedron.el_ptr[i];

//       for (int j = 0; j < f2v.Size(); ++j)
//       {
//          const int v_offset = vertices.Find(f2v[j]);
//          polyhedron.el_index[eptr + j] = v_offset;
//       }

//       polyhedron.el_ptr[i + 1] = polyhedron.el_ptr[i] + f2v.Size();
//    }
// }

// bool Intersect3D(const Mesh &m1, const int el1, const Mesh &m2, const int el2,
//                  Polyhedron &intersection)
// {
//    Intersector isector;
//    Polyhedron p1, p2;
//    MakePolyhedron(m1, el1, p1);
//    MakePolyhedron(m2, el2, p2);
//    return isector.intersect_convex_polyhedra(p1, p2, &intersection);
// }

// }
