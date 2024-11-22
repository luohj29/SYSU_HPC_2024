PROGRAM = task0
LIBS = -lmm -lphread
CXXFLAGS = -Wall -fPIC

# 目标文件
OBJ = matrix_multiply_s.o
SHARED_LIB = libmm.so

# 默认目标
all: $(SHARED_LIB) $(PROGRAM)

# 编译共享库
$(SHARED_LIB): $(OBJ)
	$(CXX) -shared -o $@ $^


$(PROGRAM) : task0.cpp $(SHARED_LIB)
	g++ task0.cpp -o task0 -lpthread -lmm

# 规则用于编译 matrix_multiply_s.cpp
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# 安装共享库
install: $(SHARED_LIB)
	sudo cp $(SHARED_LIB) /usr/local/lib
run: 
	for n in 1 2 3 4 5 6 7 8; do \
		for size in 128 256 512 1024 2048; do \
			echo "Running program with size $$size and $$n threads" | tee -a output.txt; \
			time ./$(PROGRAM) $$n $$size | tee -a output.txt; \
		done; \
		echo "End of program with $$n threads" | tee -a output.txt; \
		echo "" | tee -a output.txt; \
	done; \
	echo "End of program" | tee -a output.txt; \
	echo "" | tee -a output.txt;