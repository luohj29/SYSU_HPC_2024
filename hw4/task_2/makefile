PROGRAM = task2
LIBS = -fopenmp
# CXXFLAGS = -Wall -fPIC


all: $(PROGRAM)

$(PROGRAM) : task2.c
	gcc -fopenmp task2.c -o task2

run:
	echo "mode 0 for default, 1 for static, 2 for dynamic" | tee -a output_2.txt; \
	for mode in 0 ,1, 2; do \
		for n in 1, 2, 4, 8, 16, 32, 64, 80, 128; do \
			for size in 128 256 512 1024 2048; do \
				echo "Running program with size $$size and $$n threads and mode $$mode" | tee -a output_2.txt; \
				time ./$(PROGRAM) $$n $$size $$mode | tee -a output_2.txt; \
			done; \
			echo "End of program with $$n threads" | tee -a output_2.txt; \
			echo "" | tee -a output_2.txt; \
		done; \
		echo "End of program with $$mode " | tee -a output_2.txt; \
	done; \
	echo "End of program" | tee -a output_2.txt; \
	echo "" | tee -a output.txt;