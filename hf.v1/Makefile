CXX = g++
CXXFLAGS = -g -Wall -pg
CXXLIBS = -L/home/crawdad/lib -llapack_pg -lblas_pg -lgfortran
#CXX = icpc
#CXXLIBS = -mkl

scf: scf.o diag.o mmult.o
	$(CXX) $(CXXFLAGS) $^ $(CXXLIBS) -o $@

clean:
	rm -f *.o scf
