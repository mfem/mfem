#!/usr/sbin/dtrace -s

#pragma D option quiet

pid$target:libsystem_kernel.dylib:mprotect:entry
{
    printf("mprotect called: pid=%d, addr=%p, len=%d, prot=%d\n",
           pid, arg0, arg1, arg2);
}
