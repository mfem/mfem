
NOTMAC := $(subst Darwin,,$(shell uname -s))

ifneq ($(NOTMAC),)
   PICFLAG := -Xcompiler $(PICFLAG)
   BUILD_SOFLAGS = -shared -Xarchive -Wl,-soname,libmfem.$(SO_VER)
endif

