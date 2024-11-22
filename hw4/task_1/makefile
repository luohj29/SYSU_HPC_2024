PROGRAM = task1
LIBS = -fopenmp
# CXXFLAGS = -Wall -fPIC


all: $(PROGRAM)

$(PROGRAM) : task1.c
	gcc -fopenmp task1.c -o task1

run:
	for n in 1, 2, 4, 8, 16, 32, 64, 80, 128; do \
		for size in 128 256 512 1024 2048; do \
			echo "Running program with size $$size and $$n threads" | tee -a output.txt; \
			time ./$(PROGRAM) $$n $$size| tee -a output.txt; \
		done; \
		echo "End of program with $$n threads" | tee -a output.txt; \
		echo "" | tee -a output.txt; \
	done; \
	echo "End of program" | tee -a output.txt; \
	echo "" | tee -a output.txt;