#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11
from paraview.simple import *

path = '/Users/mittal3/LLNL/mfem-detJ/mfem/miniapps/meshing/ParaView/Blade/Blade.pvd'
outpathpre = f'/Users/mittal3/LLNL/mfem-detJ/mfem/scripts/results/blade/blade.pvd'

# solution = PVDReader(FileName=path)
solution2 = PVDReader(FileName=path)

renderView1 = GetActiveViewOrCreate('RenderView')
solutionDisplay2 = GetDisplayProperties(solution2, view=renderView1)
Show(solution2, renderView1)
# solutionDisplay2.NonlinearSubdivisionLevel=3
# ColorBy(solutionDisplay2, None)
solutionDisplay2.SetRepresentationType('Wireframe')
solutionDisplay2.AmbientColor = [0.0, 0.0, 0.0]
solutionDisplay2.DiffuseColor = [0.0, 0.0, 0.0]
solutionDisplay2.LineWidth = 6.0

renderView1.OrientationAxesVisibility = 0
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.CameraParallelScale = 1.2
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.7828632205472154, 0.1142785472132392, 3.5841201523072455]
renderView1.CameraFocalPoint = [0.7828632205472154, 0.1142785472132392, 0.0]
renderView1.CameraParallelScale = 1.0

outpath = outpathpre.replace('.pvd', f'_opt.jpg')

# renderView1.CameraPosition = [0.4693038985133171, 0.47868532687425613, 3.5841201523072455]
# renderView1.CameraFocalPoint = [0.4693038985133171, 0.47868532687425613, 0.0]
# renderView1.CameraViewUp = [0,1,0]
# renderView1.CameraParallelScale = 0.75

SaveScreenshot(outpath, renderView1, ImageResolution=[2000, 1600])


csv_file = CSVReader(FileName=['/Users/mittal3/LLNL/mfem-detJ/mfem/miniapps/meshing/bladeoptxyz_qp.txt'])
tableToPoints = TableToPoints(Input=csv_file)
tableToPoints.XColumn = "x"
tableToPoints.YColumn = "y"
tableToPoints.ZColumn = ""
tableToPoints.a2DPoints = 1
pointDisplay = Show(tableToPoints, renderView1)
pointDisplay.SetRepresentationType("Points")
pointDisplay.PointSize = 20
pointDisplay.RenderPointsAsSpheres = 1
ColorBy(pointDisplay, ("POINTS", "color"))
outpath = outpathpre.replace('.pvd', f'_qp.jpg')
pointDisplay.SetScalarBarVisibility(renderView1, False)

SaveScreenshot(outpath, renderView1, ImageResolution=[2000, 1600])


renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [1.0415602094284788, 0.08141450523812808, 3.5841201523072455]
renderView1.CameraFocalPoint = [1.0415602094284788, 0.08141450523812808, 0.0]
renderView1.CameraParallelScale = 0.09127190366170114
solutionDisplay2.NonlinearSubdivisionLevel = 4
outpath = outpathpre.replace('.pvd', f'_qp_zoom.jpg')
SaveScreenshot(outpath, renderView1, ImageResolution=[2000, 1600])


csv_file2 = CSVReader(FileName=['/Users/mittal3/LLNL/mfem-detJ/mfem/miniapps/meshing/bladeoptxyz_nodes.txt'])
tableToPoints2 = TableToPoints(Input=csv_file2)
tableToPoints2.XColumn = "x"
tableToPoints2.YColumn = "y"
tableToPoints2.ZColumn = ""
tableToPoints2.a2DPoints = 1
pointDisplay2 = Show(tableToPoints2, renderView1)
pointDisplay2.SetRepresentationType("Points")
pointDisplay2.PointSize = 20
pointDisplay2.RenderPointsAsSpheres = 1
ColorBy(pointDisplay2, ("POINTS", "color"))
outpath = outpathpre.replace('.pvd', f'_nodes.jpg')
pointDisplay2.SetScalarBarVisibility(renderView1, False)
Hide(tableToPoints, renderView1)
outpath = outpathpre.replace('.pvd', f'_nodes_zoom.jpg')


# pointDisplay2.UseSeparateColorMap = 1
# pointDisplay2.SetScalarBarVisibility(renderView1, False)
# pointDisplay2.RescaleTransferFunctionToDataRange(True, False)
# pointDisplay2.SetScalarBarVisibility(renderView1, True)

tableToPoints2Display = GetDisplayProperties(tableToPoints2, view=renderView1)
tableToPoints2Display.UseSeparateColorMap = 1
ColorBy(tableToPoints2Display, ('POINTS', 'color'), True)
tableToPoints2Display.RescaleTransferFunctionToDataRange(True, False)
tableToPoints2Display.SetScalarBarVisibility(renderView1, True)
separate_tableToPoints2Display_colorLUT = GetColorTransferFunction('color', tableToPoints2Display, separate=True)
separate_tableToPoints2Display_colorPWF = GetOpacityTransferFunction('color', tableToPoints2Display, separate=True)
separate_tableToPoints2Display_colorTF2D = GetTransferFunction2D('color', tableToPoints2Display, separate=True)
separate_tableToPoints2Display_colorLUT.ApplyPreset('Black, Blue and White', True)
pointDisplay2.SetScalarBarVisibility(renderView1, False)

# pointDisplay2_colorLUT.ApplyPreset('X Ray', True)
# pointDisplay2_colorLUT.ApplyPreset('Rainbow Uniform', True)

SaveScreenshot(outpath, renderView1, ImageResolution=[2000, 1600])
Show(tableToPoints, renderView1)
outpath = outpathpre.replace('.pvd', f'_nodes_qp_zoom.jpg')
SaveScreenshot(outpath, renderView1, ImageResolution=[2000, 1600])


# Hide(tableToPoints2, renderView1)

