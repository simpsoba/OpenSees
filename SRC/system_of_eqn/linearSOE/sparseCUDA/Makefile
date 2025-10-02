include ../../../../Makefile.def

#==============================================================================
# sparseCuda Makefile
#==============================================================================

OBJS       = CudaGenBcsrLinSOE.o CudaGenBcsrLinSolver.o AmgXLinSolver.o

# Include directories
INCLUDES   += -I$(CORE)/system_of_eqn/linearSOE/sparseCuda

ifdef CUDA
    # CUDA available - use NVCC for sparseCuda files
    ifdef AMGX
        # AMGX also available
        CXXFLAGS += -D_CUDA -D_AMGX
        $(info CUDA and AMGX available - compiling full sparseCuda implementation with NVCC)
    else
        # CUDA available but no AMGX
        CXXFLAGS += -D_CUDA
        $(info CUDA available but AMGX not available - using sparseCuda with dummy AMGX (compiled with NVCC))
    endif
    
    # Force NVCC compilation for sparseCuda files
    CXX = nvcc
    CXXFLAGS += -x c++ --std=c++11
else
    # No CUDA available - use GCC with dummy implementations
    CXXFLAGS += -D_NO_CUDA -D_NO_AMGX
    $(info CUDA not available - sparseCuda files will use dummy implementations (compiled with GCC))
endif

all:         $(OBJS)

# Miscellaneous
tidy:	
	@$(RM) $(RMFLAGS) Makefile.bak *~ #*# core

clean: tidy
	@$(RM) $(RMFLAGS) $(OBJS) *.o

spotless: clean
	@$(RM) $(RMFLAGS)

wipe: spotless

# DO NOT DELETE THIS LINE -- make depend depends on it.
