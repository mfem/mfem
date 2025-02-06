#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11
from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

path = f'C:\\Users\\Tarik-Personal\\Documents\\GitHub\\mfem\\scripts\\results\\mesh_detj_bounds_comparison\\mesh_detj_bounds_comparison.vtu'
solution = XMLUnstructuredGridReader(FileName=[path])

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.OrientationAxesVisibility = 0
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]

solutionDisplay = Show(solution, renderView1, 'UnstructuredGridRepresentation')
solutionDisplay.Ambient = 0.25
solutionDisplay.SetScalarBarVisibility(renderView1, False)

edges = XMLUnstructuredGridReader(FileName=[solution])
edgesDisplay = Show(bounds_2, renderView1, 'UnstructuredGridRepresentation')
edgesDisplay.NonlinearSubdivisionLevel = 3
edgesDisplay.SetRepresentationType('Wireframe')
edgesDisplay.AmbientColor = [0.0, 0.0, 0.0]
edgesDisplay.DiffuseColor = [0.0, 0.0, 0.0]
edgesDisplay.LineWidth = 1.5



ColorBy(solutionDisplay, ('POINTS', 'u'))
uLUT = GetColorTransferFunction('u')
uLUT.ApplyPreset('Rainbow Desaturated', True)
uLUT.NumberOfTableValues = 10

# create a new 'Contour'
contour1 = Contour(Input=solvtu)
contour1.Isosurfaces = [-7.60548e-05, 0.08307875068, 0.16623355616, 0.24938836164, 0.33254316712, 
                        0.41569797259999997, 0.49885277808, 0.58200758356, 0.66516238904, 0.74831719452, 0.831472]

contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')
ColorBy(contour1Display, None)
contour1Display.LineWidth = 2.0


bounds = XMLUnstructuredGridReader(FileName=[boundpath])
bounds_Display = Show(bounds, renderView1, 'UnstructuredGridRepresentation')
bounds_Display.NonlinearSubdivisionLevel = 3
bounds_Display.Opacity = 0.5


bounds_3 = XMLUnstructuredGridReader(FileName=[loboundpath])
bounds_3Display = Show(bounds_3, renderView1, 'UnstructuredGridRepresentation')
bounds_3Display.NonlinearSubdivisionLevel = 1
bounds_3Display.SetRepresentationType('Points')
bounds_3Display.AmbientColor = [0.0, 0.0, 0.0]
bounds_3Display.DiffuseColor = [0.0, 0.0, 0.0]
bounds_3Display.PointSize = 6


renderView1.OrientationAxesVisibility = 0
renderView1.CameraPosition = [-1.016570841301479, -1.3554137382029485, 0.9299840088072358]
renderView1.CameraFocalPoint = [0.38012337357455567, 0.3057432865904366, 0.4010583889725267]
renderView1.CameraParallelScale = 1
renderView1.Background = [1, 1, 1]
renderView1.CameraViewUp = [0.15031599573568624, 0.1803791948828235, 0.9720434390907711]

SaveScreenshot(outpath, renderView1, ImageResolution=[800, 700])