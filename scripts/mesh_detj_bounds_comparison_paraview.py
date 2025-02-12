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
solutionDisplay.NonlinearSubdivisionLevel = 3




edges = XMLUnstructuredGridReader(FileName=[path])
edgesDisplay = Show(edges, renderView1, 'UnstructuredGridRepresentation')
ColorBy(edgesDisplay, None)
edgesDisplay.NonlinearSubdivisionLevel = 3
edgesDisplay.SetRepresentationType('Wireframe')
edgesDisplay.AmbientColor = [1,1,1]
edgesDisplay.DiffuseColor = [1,1,1]
edgesDisplay.LineWidth = 1.5


renderView1.OrientationAxesVisibility = 0
renderView1.CameraPosition = [3.5, 1.275, 6.55]
renderView1.CameraFocalPoint = [3.5, 1.275, 0]
renderView1.CameraParallelScale = 0.5
renderView1.Background = [1, 1, 1]
renderView1.CameraViewUp = [0, 1, 0]

for var in ['bruteboundmin', 'bernboundmin', 'cusboundmin', 'bernerrmin', 'cuserrmin']:
    solutionDisplay.SetScalarBarVisibility(renderView1, False)
    ColorBy(solutionDisplay, ('POINTS', var))
    LUT = GetColorTransferFunction(var)
    LUT.ApplyPreset('Jet', True)
    LUT.NumberOfTableValues = 1024
    solutionDisplay.SetScalarBarVisibility(renderView1, True)
    LUTColorBar = GetScalarBar(LUT, renderView1)
    LUTColorBar.RangeLabelFormat = '%-#6.0g'
    LUTColorBar.TextPosition = 'Ticks right/top, annotations left/bottom'
    LUTColorBar.LabelColor = [0, 0, 0]
    LUTColorBar.TitleColor = [1, 1, 1]
    LUTColorBar.LabelBold = 1
    LUTColorBar.TitleBold = 1
    LUTColorBar.LabelFontSize = 12
    LUTColorBar.TitleFontSize = 14
    LUTColorBar.ComponentTitle = ''
    LUTColorBar.Orientation = 'Horizontal'
    LUTColorBar.WindowLocation = 'Any Location'
    LUTColorBar.ScalarBarLength = 0.6
    LUTColorBar.ScalarBarThickness = 12
    LUTColorBar.Position = [0.2, 0.02]
    LUTColorBar.HorizontalTitle = False
    LUTColorBar.AddRangeLabels = 1
    LUTColorBar.RangeLabelFormat = '%g'
    solutionDisplay.SetScalarBarVisibility(renderView1, True)
    if 'boundmin' in var:
        LUTColorBar.Title = 'min det(J)'
        LUT.RescaleTransferFunction(0, 0.3)
        pvar = var.replace('boundmin', '')
        outpath = path.replace('.vtu', f'_{pvar}.jpg')
    else:
        LUTColorBar.Title = 'Error'
        LUT.UseLogScale = 1
        LUT.MapControlPointsToLogSpace()
        LUT.RescaleTransferFunction(1e-4, 1e-1)
        pvar = var.replace('min', '')
        outpath = path.replace('.vtu', f'_{pvar}.jpg')
    SaveScreenshot(outpath, renderView1, ImageResolution=[1200, 600])
    solutionDisplay.SetScalarBarVisibility(renderView1, False)