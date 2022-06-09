## Script for visualizing the surrogate material with random imperfections.
## The script was generated via ParaView's trace method with ParaView 5.10.1.
## The script was post-processed to simplify the variable names and formatted 
## with the python formatter black.

import os
from paraview.simple import *


my_working_dir = os.getcwd()
filename = "SurrogateMaterial.pvd"
paraview.simple._DisableFirstRenderCameraReset()


# create a new 'PVD Reader'
pvd = PVDReader(
    registrationName=filename,
    FileName="{}/ParaView/SurrogateMaterial/{}".format(
        my_working_dir, filename
    ),
)
pvd.CellArrays = ["attribute"]
pvd.PointArrays = ["imperfect_topology", "random_field", "topological_support"]

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# show data in view
pvd_display = Show(pvd, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
pvd_display.Representation = "Surface"
pvd_display.ColorArrayName = [None, ""]
pvd_display.SelectTCoordArray = "None"
pvd_display.SelectNormalArray = "None"
pvd_display.SelectTangentArray = "None"
pvd_display.OSPRayScaleArray = "imperfect_topology"
pvd_display.OSPRayScaleFunction = "PiecewiseFunction"
pvd_display.SelectOrientationVectors = "None"
pvd_display.ScaleFactor = 0.1
pvd_display.SelectScaleArray = "None"
pvd_display.GlyphType = "Arrow"
pvd_display.GlyphTableIndexArray = "None"
pvd_display.GaussianRadius = 0.005
pvd_display.SetScaleArray = ["POINTS", "imperfect_topology"]
pvd_display.ScaleTransferFunction = "PiecewiseFunction"
pvd_display.OpacityArray = ["POINTS", "imperfect_topology"]
pvd_display.OpacityTransferFunction = "PiecewiseFunction"
pvd_display.DataAxesGrid = "GridAxesRepresentation"
pvd_display.PolarAxes = "PolarAxesRepresentation"
pvd_display.ScalarOpacityUnitDistance = 0.05412658773652741
pvd_display.OpacityArrayName = ["POINTS", "imperfect_topology"]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
pvd_display.ScaleTransferFunction.Points = [
    -0.4235184784443208,
    0.0,
    0.5,
    0.0,
    0.1197762226214436,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
pvd_display.OpacityTransferFunction.Points = [
    -0.4235184784443208,
    0.0,
    0.5,
    0.0,
    0.1197762226214436,
    1.0,
    0.5,
    0.0,
]

# reset view to fit data
renderView1.ResetCamera(False)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(pvd_display, ("POINTS", "imperfect_topology"))

# rescale color and/or opacity maps used to include current data range
pvd_display.RescaleTransferFunctionToDataRange(True, True)

# change representation type
pvd_display.SetRepresentationType("Volume")

# get color transfer function/color map for 'imperfect_topology'
imperfect_topologyLUT = GetColorTransferFunction("imperfect_topology")

# get opacity transfer function/opacity map for 'imperfect_topology'
imperfect_topologyPWF = GetOpacityTransferFunction("imperfect_topology")

# Properties modified on pvd_display
pvd_display.SelectMapper = "Resample To Image"

# get separate color transfer function/color map for 'imperfect_topology'
pvd_display_surrogate_LUT = GetColorTransferFunction(
    "imperfect_topology", pvd_display, separate=True
)

# get separate opacity transfer function/opacity map for 'imperfect_topology'
pvd_display_surrogate_PWF = GetOpacityTransferFunction(
    "imperfect_topology", pvd_display, separate=True
)

# Properties modified on pvd_display
pvd_display.LookupTable = pvd_display_surrogate_LUT
pvd_display.UseSeparateColorMap = 1
pvd_display.ScalarOpacityFunction = pvd_display_surrogate_PWF
pvd_display.UseSeparateOpacityArray = 1

# Properties modified on pvd_display_surrogate_LUT
pvd_display_surrogate_LUT.AutomaticRescaleRangeMode = (
    "Grow and update on 'Apply'"
)
pvd_display_surrogate_LUT.InterpretValuesAsCategories = 0
pvd_display_surrogate_LUT.AnnotationsInitialized = 0
pvd_display_surrogate_LUT.ShowCategoricalColorsinDataRangeOnly = 0
pvd_display_surrogate_LUT.RescaleOnVisibilityChange = 0
pvd_display_surrogate_LUT.EnableOpacityMapping = 0
pvd_display_surrogate_LUT.ScalarOpacityFunction = pvd_display_surrogate_PWF
pvd_display_surrogate_LUT.RGBPoints = [
    -0.4235184784443208,
    0.231373,
    0.298039,
    0.752941,
    -0.15187112791143859,
    0.865003,
    0.865003,
    0.865003,
    0.1197762226214436,
    0.705882,
    0.0156863,
    0.14902,
]
pvd_display_surrogate_LUT.UseLogScale = 0
pvd_display_surrogate_LUT.UseOpacityControlPointsFreehandDrawing = 0
pvd_display_surrogate_LUT.ShowDataHistogram = 0
pvd_display_surrogate_LUT.AutomaticDataHistogramComputation = 0
pvd_display_surrogate_LUT.DataHistogramNumberOfBins = 10
pvd_display_surrogate_LUT.ColorSpace = "Diverging"
pvd_display_surrogate_LUT.UseBelowRangeColor = 0
pvd_display_surrogate_LUT.BelowRangeColor = [0.0, 0.0, 0.0]
pvd_display_surrogate_LUT.UseAboveRangeColor = 0
pvd_display_surrogate_LUT.AboveRangeColor = [0.5, 0.5, 0.5]
pvd_display_surrogate_LUT.NanColor = [1.0, 1.0, 0.0]
pvd_display_surrogate_LUT.NanOpacity = 1.0
pvd_display_surrogate_LUT.Discretize = 1
pvd_display_surrogate_LUT.NumberOfTableValues = 256
pvd_display_surrogate_LUT.ScalarRangeInitialized = 1.0
pvd_display_surrogate_LUT.HSVWrap = 0
pvd_display_surrogate_LUT.VectorComponent = 0
pvd_display_surrogate_LUT.VectorMode = "Magnitude"
pvd_display_surrogate_LUT.AllowDuplicateScalars = 1
pvd_display_surrogate_LUT.Annotations = []
pvd_display_surrogate_LUT.ActiveAnnotatedValues = []
pvd_display_surrogate_LUT.IndexedColors = []
pvd_display_surrogate_LUT.IndexedOpacities = []

# Properties modified on pvd_display_surrogate_PWF
pvd_display_surrogate_PWF.Points = [
    -0.4235184784443208,
    0.0,
    0.5,
    0.0,
    0.1197762226214436,
    1.0,
    0.5,
    0.0,
]
pvd_display_surrogate_PWF.AllowDuplicateScalars = 1
pvd_display_surrogate_PWF.UseLogScale = 0
pvd_display_surrogate_PWF.ScalarRangeInitialized = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
pvd_display_surrogate_LUT.ApplyPreset("X Ray", True)

# Properties modified on pvd_display_surrogate_PWF
pvd_display_surrogate_PWF.Points = [
    -0.4235184784443208,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    1.0,
    0.5,
    0.0,
    0.1197762226214436,
    1.0,
    0.5,
    0.0,
]

# Properties modified on pvd_display_surrogate_LUT
pvd_display_surrogate_LUT.RGBPoints = [
    -0.4235184784443208,
    1.0,
    1.0,
    1.0,
    -0.1529620736837387,
    0.28627450980392155,
    0.28627450980392155,
    0.28627450980392155,
    0.1197762226214436,
    0.0,
    0.0,
    0.0,
]

# show color bar/color legend
pvd_display.SetScalarBarVisibility(renderView1, True)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1329, 1217)

# current camera placement for renderView1
renderView1.CameraPosition = [
    1.369700786223454,
    1.775310657560424,
    3.4687296427700742,
]
renderView1.CameraFocalPoint = [
    0.49999999999999994,
    0.49999999999999994,
    0.49999999999999994,
]
renderView1.CameraViewUp = [
    -0.08957645913555978,
    0.9243636497691667,
    -0.3708475440853837,
]
renderView1.CameraParallelScale = 0.8660254037844386

# save screenshot
SaveScreenshot(
    "/Users/tobias/mfem/build/miniapps/materials/my-surrogate.png",
    renderView1,
    ImageResolution=[1329, 1217],
)
