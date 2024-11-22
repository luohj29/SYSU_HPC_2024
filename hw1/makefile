# 定义变量
CXX = g++
CXXFLAGS = -Wall -fPIC
LIBS = -lmm
MPI_CXX = mpic++

# 目标文件
OBJ = matrix_multiply_s.o
SHARED_LIB = libmm.so

# 可执行文件
MPV1 = mp_v1
MPV2 = mp_v2

# 默认目标
all: $(SHARED_LIB) $(MPV1) $(MPV2)

# 编译共享库
$(SHARED_LIB): $(OBJ)
	$(CXX) -shared -o $@ $^

# 规则用于编译 mp_v1
$(MPV1): mp_v1.cpp $(SHARED_LIB)
	$(MPI_CXX) -L./ -o $@ $< $(LIBS)

# 规则用于编译 mp_v2
$(MPV2): mp_v2.cpp $(SHARED_LIB)
	$(MPI_CXX) -L./ -o $@ $< $(LIBS)

# 规则用于编译 matrix_multiply_s.cpp
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# 安装共享库
install: $(SHARED_LIB)
	sudo cp $(SHARED_LIB) /usr/local/lib

# 清理目标
clean:
	rm -f $(OBJ) $(SHARED_LIB) $(MPV1) $(MPV2)
