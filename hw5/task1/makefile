PROGRAM = main
LIBS = -fopenmp -lphread
# CXXFLAGS = -Wall -fPIC


all: $(PROGRAM)

$(PROGRAM) : heated_plate_openmp.cpp parallel.cpp
	g++ -g  heated_plate_openmp.cpp parallel.cpp  -o   main -fopenmp -lpthread

run:
	for n in 1, 2, 4, 8, 16; do \
		for size in 128 256 512 1024 2048 4096 8192; do \
			echo "Running program with size $$size and $$n threads" | tee -a output.txt; \
			./$(PROGRAM) $$size $$size $$n | tee -a output.txt; \
		done; \
		echo "End of program with $$n threads" | tee -a output.txt; \
		echo "" | tee -a output.txt; \
	done; \
	echo "End of program" | tee -a output.txt; \
	echo "" | tee -a output.txt;