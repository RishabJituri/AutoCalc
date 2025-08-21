CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O0 -g -Wall -Wextra -Wpedantic -fno-omit-frame-pointer \
             -fsanitize=address,undefined -MMD -MP
LDFLAGS   ?= -fsanitize=address,undefined
INCLUDES  ?= -Iinclude

BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj
BIN       := $(BUILD_DIR)/tests

SRC_FILES  := $(wildcard src/*.cpp)
TEST_FILES := $(wildcard tests/*.cpp)
OBJS       := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
TEST_OBJS  := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(TEST_FILES))
DEPS       := $(OBJS:.o=.d) $(TEST_OBJS:.o=.d)

.PHONY: all run test clean
all: $(BIN)

$(BIN): $(OBJS) $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

run: $(BIN)
	./$(BIN)

test: run
clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
