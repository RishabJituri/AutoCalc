# --- toolchain ---
CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O0 -g -Wall -Wextra -Wpedantic -fno-omit-frame-pointer \
             -fsanitize=address,undefined -MMD -MP
LDFLAGS   ?= -fsanitize=address,undefined
INCLUDES  ?= -Iinclude

# --- layout ---
BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj
TEST_BIN  := $(BUILD_DIR)/tests
DEMO_BIN  := $(BUILD_DIR)/mnist_demo

# --- sources ---
SRC_FILES  := $(shell find src -type f -name '*.cpp')
TEST_FILES := $(wildcard tests/*.cpp)

# Try common locations for the demo
DEMO_SRC   := $(firstword $(wildcard mnist_demo.cpp examples/mnist_demo.cpp demo/mnist_demo.cpp))

# --- objects ---
OBJS       := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
TEST_OBJS  := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_FILES))
DEMO_OBJS  := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(DEMO_SRC))
DEPS       := $(OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(DEMO_OBJS:.o=.d)

# --- targets ---
.PHONY: all test demo run-tests run-demo clean fast-demo run-fast-demo

# Only build the demo binary if the source was found
ifeq ($(strip $(DEMO_SRC)),)
ALL_BINS := $(TEST_BIN)
else
ALL_BINS := $(TEST_BIN) $(DEMO_BIN)
endif

all: $(ALL_BINS)

# tests exe
$(TEST_BIN): $(OBJS) $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

# demo exe (guarded)
ifeq ($(strip $(DEMO_SRC)),)
$(DEMO_BIN):
	@echo "error: mnist_demo.cpp not found (tried: ./, examples/, demo/)."
	@echo "       Put the demo source at one of those paths or set DEMO_SRC."
	@false
else
$(DEMO_BIN): $(OBJS) $(DEMO_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)
endif

# generic compile rule
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# convenience
run-tests: $(TEST_BIN)
	./$(TEST_BIN)

run-demo: $(DEMO_BIN)
	./$(DEMO_BIN) $(ARGS)

test: run-tests
demo: run-demo

# ------------------ FAST DEMO (no sanitizers) ------------------
FAST_BUILD_DIR := build-fast
FAST_OBJ_DIR   := $(FAST_BUILD_DIR)/obj
FAST_DEMO_BIN  := $(FAST_BUILD_DIR)/mnist_demo

# High-opt flags; no sanitizers here
FAST_CXXFLAGS := -std=c++17 -O3 -DNDEBUG -march=native -MMD -MP $(INCLUDES)
FAST_LDFLAGS  :=

FAST_DEMO_SRC  := $(DEMO_SRC)
FAST_DEMO_OBJS := $(patsubst %.cpp,$(FAST_OBJ_DIR)/%.o,$(SRC_FILES) $(FAST_DEMO_SRC))
FAST_DEPS      := $(FAST_DEMO_OBJS:.o=.d)

ifeq ($(strip $(FAST_DEMO_SRC)),)
fast-demo:
	@echo "error: mnist_demo.cpp not found (tried: ./, examples/, demo/)."
	@false
else
fast-demo: $(FAST_DEMO_BIN)

$(FAST_DEMO_BIN): $(FAST_DEMO_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(FAST_LDFLAGS)

$(FAST_OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(FAST_CXXFLAGS) -c $< -o $@
endif

run-fast-demo: fast-demo
	./$(FAST_DEMO_BIN) $(ARGS)
# ---------------------------------------------------------------

clean:
	rm -rf $(BUILD_DIR) $(FAST_BUILD_DIR)

-include $(DEPS)
-include $(FAST_DEPS)
