CC = mpicc
CXX = mpicxx
FC = mpif90
CFLAGS = --std=gnu11 -g -O2

CFLAGS +=-fno-common -fomit-frame-pointer

CFLAGS += -Wconversion -Wno-sign-conversion \
          -Wcast-align -Wchar-subscripts -Wall -W \
          -Wpointer-arith -Wwrite-strings -Wformat-security -pedantic \
          -Wextra -Wno-unused-parameter

# list of libraries to build
TPLS ?= occa p4est

UNAME_S := $(shell uname -s)

# occa flags
CPPFLAGS += -Iocca/include
LDFLAGS += -Locca/lib
LDLIBS += -locca
ifeq ($(UNAME_S),Linux)
 LDFLAGS += -Wl,-rpath=$(CURDIR)/occa/lib,--enable-new-dtags
endif

# p4est flags
CPPFLAGS += -Ip4est/local/include
LDFLAGS += -Lp4est/local/lib
LDLIBS += -lp4est -lsc
ifeq ($(UNAME_S),Linux)
 LDFLAGS += -Wl,-rpath=$(CURDIR)/p4est/local/lib,--enable-new-dtags
endif

all: tutorial

occa:
	tar xzf tpl/occa-*.tar.gz && mv occa-* occa
	patch -p1 < tpl/occa-glibcfix.patch
	cd occa && $(MAKE) OCCA_DEVELOPER=1 DEBUG=1 CC=$(CC) CXX=$(CXX) FC=$(FC)

p4est:
	tar xzf tpl/p4est-*.tar.gz && mv p4est-* p4est
	cd p4est && ./configure CC=$(CC) --enable-mpi --enable-debug --without-blas && $(MAKE) install

# Dependencies
tutorial: tutorial.c | $(TPLS)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) \
        $< $(LOADLIBES) $(LDLIBS) -o $@
ifeq ($(UNAME_S),Darwin)
	install_name_tool -add_rpath $$(pwd)/occa/lib $@
endif

# Rules
.PHONY: clean realclean test

rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $2,$d))

test: $(patsubst %.test,%.stdout,$(call rwildcard,,%.test))

%.stdout: %.test tutorial
	./$< > $@ 2> $(patsubst %.stdout,%.stderr,$@) \
		|| (touch --date=@0 $@; false)
	git diff --exit-code --src-prefix=expected/ --dst-prefix=actual/ \
		$@ $(patsubst %.stdout,%.stderr,$@) \
		|| (touch --date=@0 $@; false)

clean:
	rm -rf tutorial

realclean:
	rm -rf $(TPLS)
	git clean -X -d -f
