
# ===============================================================
# Toolchain
# ===============================================================
CXX       ?= g++
INCLUDES  ?= -Iinclude

# ---------------- Debug / Sanitized (for tests and debug demos) ----
CXXFLAGS  ?= -std=c++17 -O0 -g -Wall -Wextra -Wpedantic -fno-omit-frame-pointer \
             -fsanitize=address,undefined -MMD -MP
LDFLAGS   ?= -fsanitize=address,undefined

# ---------------- Fast / Release (no sanitizers; for demos) --------
# Detect host for decent arch flags
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH_FAST := -mcpu=apple-m2
  else
    ARCH_FAST := -march=native
  endif
else
  ARCH_FAST := -march=native
endif

FAST_CXXFLAGS := -std=c++17 -O3 -DNDEBUG -Wall -Wextra -fno-omit-frame-pointer \
                 -ffast-math $(ARCH_FAST) -MMD -MP
FAST_LDFLAGS  :=

# Some platforms want pthread explicitly
ifeq ($(UNAME_S),Linux)
  THREAD_LIBS := -pthread
endif

# ===============================================================
# Layout
# ===============================================================
BUILD_DIR      := build
OBJ_DIR        := $(BUILD_DIR)/obj
TEST_BIN       := $(BUILD_DIR)/tests

DEMO_DIR       := $(BUILD_DIR)/demo_slow
DEMO_BIN       := $(DEMO_DIR)/mnist_demo

FAST_BUILD_DIR := $(BUILD_DIR)/fast
FAST_OBJ_DIR   := $(FAST_BUILD_DIR)/obj
FAST_MNIST_BIN := $(FAST_BUILD_DIR)/mnist_demo
FAST_LSTM_BIN  := $(FAST_BUILD_DIR)/lstm_shakespeare

# ===============================================================
# Sources
# ===============================================================
SRC_FILES   := $(shell find src -type f -name '*.cpp')
TEST_FILES  := $(wildcard tests/*.cpp)

MNIST_EXAMPLE := examples/mnist_demo.cpp
LSTM_EXAMPLE  := examples/lstm_shakespeare.cpp

# Objects (sanitized)
OBJS        := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES)) \
               $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_FILES))

# Objects (fast)
FAST_OBJS   := $(patsubst %.cpp,$(FAST_OBJ_DIR)/%.o,$(SRC_FILES))

# Dependency files
DEPS        := $(OBJS:.o=.d)
FAST_DEPS   := $(FAST_OBJS:.o=.d)

# ===============================================================
# Phonies
# ===============================================================
.PHONY: all test demo run-demo fast-mnist run-fast-mnist fast-lstm run-fast-lstm clean

all: test fast-mnist fast-lstm

# ===============================================================
# Tests (sanitized)
# ===============================================================
test: $(TEST_BIN)

$(TEST_BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(THREAD_LIBS)

# Compile rule (sanitized) for any .cpp
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ===============================================================
# Debug demo (optional; sanitized)
# ===============================================================
demo: $(DEMO_BIN)

$(DEMO_BIN): $(OBJ_DIR)/$(MNIST_EXAMPLE:.cpp=.o) $(filter-out $(OBJ_DIR)/tests/%.o,$(filter $(OBJ_DIR)/src/%,$(OBJS)))
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) $(THREAD_LIBS)

run-demo: demo
	./$(DEMO_BIN) $(ARGS)

# ===============================================================
# Fast demos (no sanitizers)
# ===============================================================
fast-mnist: $(FAST_MNIST_BIN)

$(FAST_MNIST_BIN): $(FAST_OBJ_DIR)/$(MNIST_EXAMPLE:.cpp=.o) $(FAST_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(FAST_CXXFLAGS) $(INCLUDES) $^ -o $@ $(FAST_LDFLAGS) $(THREAD_LIBS)

run-fast-mnist: fast-mnist
	./$(FAST_MNIST_BIN) $(ARGS)

fast-lstm: $(FAST_LSTM_BIN)

$(FAST_LSTM_BIN): $(FAST_OBJ_DIR)/$(LSTM_EXAMPLE:.cpp=.o) $(FAST_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(FAST_CXXFLAGS) $(INCLUDES) $^ -o $@ $(FAST_LDFLAGS) $(THREAD_LIBS)

run-fast-lstm: fast-lstm
	./$(FAST_LSTM_BIN) $(ARGS)

# Compile rule (fast) for any .cpp
$(FAST_OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(FAST_CXXFLAGS) $(INCLUDES) -c $< -o $@

# ===============================================================
# Clean
# ===============================================================
clean:
	rm -rf $(BUILD_DIR)

# ===============================================================
# Includes
# ===============================================================
-include $(DEPS)
-include $(FAST_DEPS)
