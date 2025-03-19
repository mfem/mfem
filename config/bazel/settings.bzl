"""
This module defines build settings: 'mode' and 'precision'.
"""

# Mode ########################################################################
ModeInfo = provider(doc = "serial or parallel", fields = ["type"])
def PrintMode(ctx):
    ctx.actions.write(output = ctx.outputs.log, 
                      content = "Compiling in " + ctx.attr.mode[ModeInfo].type + "!")
print_mode = rule(implementation = PrintMode, attrs = {"mode": attr.label()})
def Mode(ctx): return ModeInfo(type = ctx.label.name)
mode = rule(implementation = Mode)

# Precision ###################################################################
PrecisionInfo = provider(doc = "single or double", fields = ["type"])
def PrintPrecision(ctx):
    ctx.actions.write(output = ctx.outputs.log, 
                      content = "Compiling in " + ctx.attr.precision[PrecisionInfo].type + "!")
print_precision = rule(implementation = PrintPrecision, attrs = {"precision": attr.label()})
def Precision(ctx): return PrecisionInfo(type = ctx.label.name)
precision = rule(implementation = Precision)
