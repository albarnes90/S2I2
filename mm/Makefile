PROGRAM = mm
PROGSRC = mm.cc
CXXFLAGS = -g -std=c++0x -pg
CXX = g++
CXXLIBS = -L/home/crawdad/lib -llapack_pg -lblas_pg -lgfortran
#CXXLIBS = -L/home/crawdad/lib -llapack3 -lblas3 -lgfortran
#CXXLIBS = -L$(ATLAS_LIB) -llapack -lf77blas -latlas -lgfortran
#CXX = icpc
#CXXLIBS = -mkl

OBJECTS = $(PROGSRC:%.cc=%.o)

$(PROGRAM): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(CXXLIBS) -o $@

%.o: $.cc
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJECTS) $(PROGRAM)
