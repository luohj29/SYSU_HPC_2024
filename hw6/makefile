# 项目配置
PROGRAM1 = gemmMain
PROGRAM2 = convMain

# 源文件
SRC_DIR = src
BUILD_DIR = build
SRC_FILES1 = $(SRC_DIR)/gemm_main.cu $(SRC_DIR)/gemm.cu $(SRC_DIR)/utils.cu
SRC_FILES2 = $(SRC_DIR)/conv_main.cu $(SRC_DIR)/conv.cu $(SRC_DIR)/gemm.cu $(SRC_DIR)/class.cu $(SRC_DIR)/utils.cu

# 目标文件
OBJ_FILES1 = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRC_FILES1))
OBJ_FILES2 = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRC_FILES2))

# 编译器和选项
NVCC = nvcc
CFLAGS = 
LDFLAGS = -lcublas
DEBUG_CFLAGS = -G -g -lineinfo -Xcompiler -rdynamic

# 默认目标
.PHONY: all
all: build $(PROGRAM1) $(PROGRAM2)

# 生成 build 目录
build:
	mkdir -p $(BUILD_DIR)

# 编译生成可执行文件
$(PROGRAM1): $(OBJ_FILES1)
	$(NVCC) $(OBJ_FILES1) $(LDFLAGS) -o $@

$(PROGRAM2): $(OBJ_FILES2)
	$(NVCC) $(OBJ_FILES2) $(LDFLAGS) -lcudnn -o $@

debug: build
	$(NVCC) $(SRC_FILES1) $(DEBUG_CFLAGS) $(LDFLAGS) -o $(PROGRAM1)
	$(NVCC) $(SRC_FILES2) $(DEBUG_CFLAGS) $(LDFLAGS) -o $(PROGRAM2)

# 编译 .cu 文件为 .o 文件
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

# 清理目标文件和可执行文件
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*.o $(PROGRAM1) $(PROGRAM2) gemm_output.txt conv_output.txt

# 运行程序
.PHONY: run
run1: $(PROGRAM1)
	@for n in 8 16 32; do \
		for size in 512 1024 2048 4096 8192; do \
			echo "Running $(PROGRAM1) with size $$size and $$n BLOCK" | tee -a gemm_output.txt; \
			./$(PROGRAM1) $$size $$size $$size $$n | tee -a gemm_output.txt; \
		done; \
		echo "End of $(PROGRAM1) with $$n BLOCK SIZE" | tee -a gemm_output.txt; \
		echo "" | tee -a gemm_output.txt; \
	done; \
	echo "End of $(PROGRAM1)" | tee -a gemm_output.txt; \
	echo "" | tee -a gemm_output.txt;

run2: $(PROGRAM2)
	@for stride in 1 2 3; do \
		for size in 512 1024 2048 4096 8192; do \
			echo "Running $(PROGRAM2) with size $$size and $$stride stride" | tee -a conv_output.txt; \
			./$(PROGRAM2) $$size $$size 3 3 3 1 $$stride | tee -a conv_output.txt; \
		done; \
		echo "End of $(PROGRAM2) with $$stride stride" | tee -a conv_output.txt; \
		echo "" | tee -a conv_output.txt; \
	done; \
	echo "End of $(PROGRAM2)" | tee -a conv_output.txt; \
	echo "" | tee -a conv_output.txt;
