class MeshConnections
{
   private:
   Mesh *mesh;
   Table *elem_to_elem, *elem_to_face, *elem_to_edge, *elem_to vert;
   Table *face_to_elem, *face_to_face, *face_to_edge, *face_to vert;
   Table *edge_to_elem, *edge_to_face, *edge_to_edge, *edge_to vert;
   Table *vert_to_elem, *vert_to_face, *vert_to_edge, *vert_to vert;

   public:
   //Queries
   bool IsFaceInElem(int elem, int face);
   bool IsEdgeInElem(int elem, int edge);
   bool IsVertInElem(int elem, int vert);
   bool IsEdgeInFace(int face, int edge);
   bool IsVertInFace(int face, int vert);
   bool IsVertInEdge(int edge, int vert);

   //Neighbor mappings
   void NDNeighbors(int nd, int id, Array<int> &neighbors);
   void NeighborsToElem(int elem, Array<int> &elems);
   void NeighborsToFace(int face, Array<int> &faces);
   void NeighborsToEdge(int edge, Array<int> &edges);
   void NeighborsToVert(int vert, Array<int> &verts);

   //Singular forward mappings
   void NDPartsInMDPart(int nd, int md, int mdpart, Array<int> &ndparts);
   void FacesInElem(int elem, Array<int> &faces);
   void EdgesInElem(int elem, Array<int> &edges);
   void VertsInElem(int elem, Array<int> &verts);
   void EdgesInFace(int face, Array<int> &edges);
   void VertsInFace(int face, Array<int> &verts);
   void VertsInEdge(int edge, Array<int> &verts);

   //Mutiple forward mappings
   void NDPartsInMDParts(int nd, int md, const Array<int> &mdparts, Array<int> &ndparts);
   void FacesInElems(const Array<int> &elems, Array<int> &faces);
   void EdgesInElems(const Array<int> &elems, Array<int> &edges);
   void VertsInElems(const Array<int> &elems, Array<int> &verts);
   void EdgesInFaces(const Array<int> &faces, Array<int> &edges);
   void VertsInFaces(const Array<int> &edges, Array<int> &verts);
   void VertsInEdges(const Array<int> &edges, Array<int> &verts);
      
   //Singular reverse mappings
   void NDPartsWithMDPart(int nd, int md, int mdpart, Array<int> &ndparts);
   void ElemsWithFace(int face, Array<int> &elems);
   void ElemsWithEdge(int edge, Array<int> &elems);
   void ElemsWithVert(int vert, Array<int> &elems);
   void FacesWithEdge(int edge, Array<int> &faces);
   void FacesWithVert(int vert, Array<int> &faces);
   void EdgesWithVert(int vert, Array<int> &edges);

   //Multiple Reverse mappings for the bigger containing
   //ANY instances in the smaller
   //E.G.  ElemsWithAnyFaces will return the elements
   //      that have ANY of their faces denoted in the "faces" array
   void NDPartsTouchedByMDParts(int nd, int md, const Array<int> &mdparts, Array<int> &ndparts);
   void ElemsTouchedByFaces(const Array<int> &faces, Array<int> &elems);
   void ElemsTouchedByEdges(const Array<int> &edges, Array<int> &elems);
   void ElemsTouchedByVerts(const Array<int> &verts, Array<int> &elems);
   void FacesTouchedByEdges(const Array<int> &edges, Array<int> &faces);
   void FacesTouchedByVerts(const Array<int> &verts, Array<int> &faces);
   void EdgesTouchedByVerts(const Array<int> &verts, Array<int> &edges);   

   //Multiple Reverse mappings for the bigger containing
   //ANY instances in the smaller
   //E.G.  ElemsWithAllFaces will return the elements
   //      that have ALL of their faces denoted in the "faces" array
   void NDPartsContainedByMDParts(int nd, int md, const Array<int> &mdparts, Array<int> &ndparts);
   void ElemsContainedByFaces(const Array<int> &faces, Array<int> &elems);
   void ElemsContainedByEdges(const Array<int> &edges, Array<int> &elems);
   void ElemsContainedByVerts(const Array<int> &verts, Array<int> &elems);
   void FacesContainedByEdges(const Array<int> &edges, Array<int> &faces);
   void FacesContainedByVerts(const Array<int> &verts, Array<int> &faces);
   void EdgesContainedByVerts(const Array<int> &verts, Array<int> &edges);

   //Raw table access
   //Make these private???
   Table *ElemToVertTable();
   Table *ElemToFaceTable();
   Table *ElemToEdgeTable();
   Table *ElemToVertTable();
   Table *FaceToVertTable();
   Table *FaceToFaceTable();
   Table *FaceToEdgeTable();
   Table *FaceToVertTable();
   Table *EdgeToVertTable();
   Table *EdgeToFaceTable();
   Table *EdgeToEdgeTable();
   Table *EdgeToVertTable();
   Table *VertToVertTable();
   Table *VertToFaceTable();
   Table *VertToEdgeTable();
   Table *VertToVertTable();
};