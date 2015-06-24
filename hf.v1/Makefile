CXX = g++
CXXFLAGS = -g -Wall
#CXXLIBS = -L/home/crawdad/lib -llapack3 -lblas3 -lgfortran
CXX = icpc
CXXLIBS = -mkl

scf: scf.o diag.o mmult.o
	$(CXX) $^ $(CXXLIBS) -o $@

clean:
	rm -f *.o scf
