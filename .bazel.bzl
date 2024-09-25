"""
Bazel BUILD helper functions
"""

load("@rules_cc//cc:defs.bzl", "cc_binary")

BAZEL_SH = "BAZEL_SH"

def auto_config_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sConfiguration Error:%s %s\n" % (red, no_color, msg))

def auto_config_success(msg):
    """Output success message when auto configuration succeeds."""
    red = "\033[0;32m"
    no_color = "\033[m"
    fail("%sConfiguration Success:%s %s\n" % (red, no_color, msg))

def get_environ(ctx, name, default_value = None):
    """Returns the value of an environment variable on the execution platform.

    Args:
      ctx: the ctx
      name: the name of environment variable
      default_value: the value to return if not set

    Returns:
      The value of the environment variable 'name' on the execution platform
      or 'default_value' if it's not set.
    """
    cmd = "echo -n \"$%s\"" % name
    result = execute(
        ctx,
        [get_bash_bin(ctx), "-c", cmd],
        empty_stdout_fine = True,
    )
    if len(result.stdout) == 0:
        return default_value
    return result.stdout

def get_host_environ(ctx, name, default_value = None):
    """Returns the value of an environment variable on the host platform.

    The host platform is the machine that Bazel runs on.

    Args:
      ctx: the ctx
      name: the name of environment variable
      default_value: default_value

    Returns:
      The value of the environment variable 'name' on the host platform.
    """
    if name in ctx.os.environ:
        return ctx.os.environ.get(name).strip()

    if hasattr(ctx.attr, "environ") and name in ctx.attr.environ:
        return ctx.attr.environ.get(name).strip()

    return default_value

def raw_exec(ctx, cmdline):
    """Executes a command via ctx.execute() and returns the result.

    This method is useful for debugging purposes. For example, to print all
    commands executed as well as their return code.

    Args:
      ctx: the ctx
      cmdline: the list of args

    Returns:
      The 'exec_result' of ctx.execute().
    """

    # return ctx.execute(cmdline)
    output = ""
    ctx.actions.run_shell(outputs = [output], command = cmdline)
    return output

def execute(
        ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Args:
      ctx: the ctx object
      cmdline: list of strings, the command to execute
      error_msg: string, a summary of the error if the command fails
      error_details: string, details about the error or steps to fix it
      empty_stdout_fine: bool, if True, an empty stdout result is fine,
        otherwise it's an error
    Returns:
      The result of ctx.execute(cmdline)
    """
    result = raw_exec(ctx, cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def generate_examples():
    for n in range(1, 40):
        example = "ex" + str(n)
        cc_binary(
            name = example,
            srcs = ["examples/" + example + ".cpp"],
            deps = [
                "mfem",
                "mfem_hpp",
                "examples_hpp",
            ],
        )

# def add_mpi_local_repository():
#     new_local_repository(
#         # repo_mapping, # _not_ supported in `MODULE.bazel`
#         name = "mpi",
#         # build_file = "bazel/mpi",
#         # build_file_content = "",
#         path = "/opt/homebrew/opt/openmpi",
#     )

# def _my_rule_impl(ctx):
#     output = ctx.actions.declare_file("output.txt")
#     ctx.actions.run_shell(
#         outputs = [output],
#         command = "echo 'Hello, World!' > {}".format(output.path),
#     )
#     return [DefaultInfo(files = depset([output]))]

# my_rule = rule(
#     implementation = _my_rule_impl,
#     attrs = {},
# )

def _mpi_path_impl(ctx):
    out_file = ctx.actions.declare_file("%s.out" % ctx.attr.name)

    ctx.actions.run_shell(
        outputs = [out_file],
        command = "which mpicxx > %s" % (out_file.path),
        use_default_shell_env = True,
    )
    return [DefaultInfo(files = depset([out_file]))]

mpi_path = rule(_mpi_path_impl)

def which(ctx, program_name):
    result = execute(ctx, ["which", program_name])
    return result.stdout.rstrip()

def _get_bash_bin(ctx):
    """Gets the bash bin path.

    Args:
      ctx: the ctx

    Returns:
      The bash bin path.
    """

    # bash_bin = get_host_environ(ctx, BAZEL_SH)
    # if bash_bin != None:
    # return bash_bin
    bash_bin_path = which(ctx, "bash")
    if bash_bin_path == None:
        auto_config_fail("Cannot find bash in PATH, please make sure " +
                         "bash is installed and add its directory in PATH, or --define " +
                         "%s='/path/to/bash'.\nPATH=%s" % (
                             BAZEL_SH,
                             get_environ("PATH", ""),
                         ))
    auto_config_success("Bash in PATH: %s" % (bash_bin_path))
    return bash_bin_path

get_bash_bin = rule(_get_bash_bin)
