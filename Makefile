# Compiler and flags
CXX = g++
CXXFLAGS = -I./onnxruntime/include -std=c++17
LDFLAGS = -L./onnxruntime/lib -lonnxruntime

# Target executable
TARGET = onnx_inference

# Source files
SRC = model_cpp_test.cpp

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean the build
clean:
	rm -f $(TARGET)
