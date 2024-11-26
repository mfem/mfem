list(APPEND DATA_FILES
  amr-hex.mesh
  amr-quad.mesh
  ball-nurbs.mesh
  beam-hex-nurbs.mesh
  beam-hex.mesh
  beam-hex.vtk
  beam-quad-amr.mesh
  beam-quad-nurbs-sf.mesh
  beam-quad-nurbs.mesh
  beam-quad.mesh
  beam-quad.vtk
  beam-tet.mesh
  beam-tet.vtk
  beam-tri.mesh
  beam-tri.vtk
  beam-wedge.mesh
  beam-wedge.vtk
  compass.geo
  compass.mesh
  compass.msh
  cube-nurbs.mesh
  diag-segment-2d.mesh
  diag-segment-3d.mesh
  disc-nurbs.mesh
  escher-p2.mesh
  escher-p2.vtk
  escher-p3.mesh
  escher.mesh
  escher.vtk
  fichera-amr.mesh
  fichera-mixed-p2.mesh
  fichera-mixed-p2.vtk
  fichera-mixed.mesh
  fichera-q2.mesh
  fichera-q2.vtk
  fichera-q3.mesh
  fichera-quad-mixed.mesh
  fichera-quad.mesh
  fichera.mesh
  fichera.vtk
  hexagon.mesh
  inline-hex.mesh
  inline-pyramid.mesh
  inline-quad.mesh
  inline-segment.mesh
  inline-tet.mesh
  inline-tri.mesh
  inline-wedge.mesh
  klein-bottle.mesh
  klein-donut.mesh
  l-shape.mesh
  llnl-p3.mesh
  mobius-strip.mesh
  octahedron.mesh
  periodic-annulus-sector.geo
  periodic-annulus-sector.msh
  periodic-cube.geo
  periodic-cube.mesh
  periodic-cube.msh
  periodic-hexagon.mesh
  periodic-segment.mesh
  periodic-square.geo
  periodic-square.mesh
  periodic-square.msh
  periodic-torus-sector.geo
  periodic-torus-sector.msh
  pipe-nurbs-2d.mesh
  pipe-nurbs-log.mesh
  pipe-nurbs.mesh
  ref-cube.mesh
  ref-prism.mesh
  ref-pyramid.mesh
  ref-segment.mesh
  ref-square.mesh
  ref-tetrahedron.mesh
  ref-triangle.mesh
  rt-2d-p4-tri.mesh
  rt-2d-q3.mesh
  segment-nurbs.mesh
  square-disc-nurbs-patch.mesh
  square-disc-nurbs.mesh
  square-disc-p2.mesh
  square-disc-p2.vtk
  square-disc-p3.mesh
  square-disc-surf.mesh
  square-disc.mesh
  square-disc.vtk
  square-mixed.mesh
  square-nurbs-pw.mesh
  square-nurbs.mesh
  star-hilbert.mesh
  star-mixed-p2.mesh
  star-mixed-p2.vtk
  star-mixed.mesh
  star-q2.mesh
  star-q2.vtk
  star-q3.fms
  star-q3.mesh
  star-surf.mesh
  star.mesh
  star.vtk
  tinyzoo-3d.mesh
  toroid-hex.mesh
  toroid-wedge.mesh
)

foreach(DATA_FILE ${DATA_FILES})
  set(SRC_DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${DATA_FILE})
  set(GEN_DATA_FILE ${CMAKE_CURRENT_BINARY_DIR}/${DATA_FILE})
  add_custom_command(
    OUTPUT ${GEN_DATA_FILE}
    DEPENDS ${SRC_DATA_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${SRC_DATA_FILE} ${GEN_DATA_FILE})
  list(APPEND GENERATED_DATA_FILES ${GEN_DATA_FILE})
endforeach()

add_custom_target(copy_data DEPENDS ${GENERATED_DATA_FILES})
