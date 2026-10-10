CXX ?= g++
AR ?= ar
BUILD ?= build
EIGEN3_INCLUDE ?= /usr/include/eigen3
PF_WITH_MUMPS ?= 0
PF_MUMPS_INCLUDE ?= /usr/include
PF_MUMPS_LIBDIR ?= /usr/lib
PF_MUMPS_LIB ?= dmumps_seq
PF_WITH_METIS ?= 0
PF_METIS_INCLUDE ?= /usr/include
PF_METIS_LIBDIR ?= /usr/lib
PF_METIS_LIB ?= metis
LDLIBS += -ldl
PF_WITH_IPOPT ?= 1
PF_WITH_OPENMP ?= 1
PF_NATIVE ?= 0
IPOPT_INCLUDE ?= /usr/include/coin
IPOPT_LIBDIR ?= /usr/lib
CPPFLAGS += -Iinclude -I$(EIGEN3_INCLUDE) -DEIGEN_DONT_PARALLELIZE
CXXFLAGS ?= -O3 -DNDEBUG
CXXFLAGS += -std=c++17 -fno-fast-math -ffp-contract=off -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -MMD -MP
ifeq ($(PF_NATIVE),1)
# The same flags are used for the library, CLI, workers and tests.
CXXFLAGS += -march=native
endif
ifeq ($(PF_WITH_OPENMP),1)
CXXFLAGS += -fopenmp
LDLIBS += -fopenmp
endif
ifeq ($(PF_WITH_IPOPT),1)
CPPFLAGS += -DPF_WITH_IPOPT -I$(IPOPT_INCLUDE)
LDLIBS += -L$(IPOPT_LIBDIR) -Wl,-rpath,$(IPOPT_LIBDIR) -lipopt
endif
ifeq ($(PF_WITH_MUMPS),1)
CPPFLAGS += -DPF_WITH_MUMPS -I$(PF_MUMPS_INCLUDE)
LDLIBS += -L$(PF_MUMPS_LIBDIR) -Wl,-rpath,$(PF_MUMPS_LIBDIR) -l$(PF_MUMPS_LIB)
endif
ifeq ($(PF_WITH_METIS),1)
CPPFLAGS += -DPF_WITH_METIS -isystem $(PF_METIS_INCLUDE)
LDLIBS += -L$(PF_METIS_LIBDIR) -Wl,-rpath,$(PF_METIS_LIBDIR) -l$(PF_METIS_LIB)
endif
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SOURCES))
.PHONY: all test clean
all: $(BUILD)/pf_local_incumbent_tests $(BUILD)/pf_linear_check $(BUILD)/pf_local_bench $(BUILD)/pf_lj_newton_tests $(BUILD)/pf_solver $(BUILD)/pf_local_worker $(BUILD)/pf_tests $(BUILD)/pf_probe $(BUILD)/pf_rng_probe $(BUILD)/pf_eigen_boundary_tests
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BUILD)/libpf.a: $(OBJECTS)
	$(AR) rcs $@ $^
$(BUILD)/pf_solver: app/main.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_local_worker: app/worker.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) -Isrc $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_tests: tests/test_core.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_probe: tests/probe.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_eigen_boundary_tests: tests/test_eigen_boundary.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_rng_probe: tests/rng_probe.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_local_bench: app/local_bench.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_lj_newton_tests: tests/test_lj_newton.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_linear_check: app/linear_check.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
$(BUILD)/pf_local_incumbent_tests: tests/test_local_incumbent.cpp $(BUILD)/libpf.a | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BUILD)/libpf.a $(LDFLAGS) $(LDLIBS) -o $@
test: all
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(BUILD)/pf_local_incumbent_tests
	$(BUILD)/pf_linear_check --backend ldl
ifeq ($(PF_WITH_METIS),1)
	$(BUILD)/pf_linear_check --backend ldl --ordering metis
ifeq ($(PF_WITH_MUMPS),1)
	$(BUILD)/pf_linear_check --backend mumps --ordering metis
endif
endif
	$(BUILD)/pf_lj_newton_tests
	$(BUILD)/pf_eigen_boundary_tests
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(BUILD)/pf_tests
	python3 tests/numpy_random_parity.py $(BUILD)/pf_rng_probe $(BUILD)/pf_solver
	python3 tests/python_parity.py $(BUILD)/pf_probe
	python3 tests/integration.py $(BUILD)/pf_solver
clean:
	rm -rf $(BUILD)
-include $(OBJECTS:.o=.d)
