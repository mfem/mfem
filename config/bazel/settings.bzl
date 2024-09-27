# buildifier: disable=module-docstring
ModeInfo = provider(doc = "", fields = ["type"])

# Print Mode ##################################################################
# buildifier: disable=print
def _impl(ctx):
    print("Compiling in " + ctx.attr.mode[ModeInfo].type + "!")

print_mode = rule(implementation = _impl, attrs = {"mode": attr.label()})

# Mode ########################################################################
def _mode_impl(ctx):
    return ModeInfo(type = ctx.label.name)

mode = rule(implementation = _mode_impl)
