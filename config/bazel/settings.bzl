# buildifier: disable=module-docstring
# Mode ########################################################################
ModeInfo = provider(doc = "", fields = ["type"])

# buildifier: disable=print
def _print_mode(ctx):
    print("Compiling in " + ctx.attr.mode[ModeInfo].type + "!")

print_mode = rule(implementation = _print_mode, attrs = {"mode": attr.label()})

# Mode ########################################################################
def _mode_impl(ctx):
    return ModeInfo(type = ctx.label.name)

mode = rule(implementation = _mode_impl)

# Precision ###################################################################
PrecisionInfo = provider(doc = "", fields = ["type"])

# buildifier: disable=print
def _print_precision(ctx):
    print("Compiling in " + ctx.attr.precision[PrecisionInfo].type + "!")

print_precision = rule(implementation = _print_precision, attrs = {"precision": attr.label()})

# Mode ########################################################################
def _precision_impl(ctx):
    return PrecisionInfo(type = ctx.label.name)

precision = rule(implementation = _precision_impl)
