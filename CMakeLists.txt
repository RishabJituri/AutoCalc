cmake_minimum_required(VERSION 3.20)
project(autocalc LANGUAGES CXX)

# ------------------------ C++ standard ------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------ Options/Args ------------------------
# Allow: cmake -DARGS="--epochs 1 --data ./Data" ..
set(ARGS "" CACHE STRING "Arguments passed to run-* convenience targets")

# ------------------------ Include paths -----------------------
# Preserve Makefile's explicit include flags
# (These are only used to seed include directories; keep them explicit as requested.)
set(AUTO_INCLUDES "-Iinclude")

# Parse -I... items into include_directories
separate_arguments(AUTO_INCLUDE_TOKENS NATIVE_COMMAND "${AUTO_INCLUDES}")
set(_incs)
foreach(tok ${AUTO_INCLUDE_TOKENS})
  if(tok MATCHES "^-I(.+)$")
    list(APPEND _incs "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(_incs)
  include_directories(${_incs})
endif()

# ------------------------ Flags: Debug/Sanitized ---------------
# Keep original flags from Makefile
set(AUTO_CXXFLAGS_DBG "-std=c++17 -O0 -g -Wall -Wextra -Wpedantic -fno-omit-frame-pointer -fsanitize=address,undefined -MMD -MP")
set(AUTO_LDFLAGS_DBG  "-fsanitize=address,undefined")

# ------------------------ Flags: Fast/Release ------------------
set(AUTO_FAST_CXXFLAGS "-std=c++17 -O3 -DNDEBUG -Wall -Wextra -fno-omit-frame-pointer -ffast-math ${AUTO_FAST_ARCH} -MMD -MP")
set(AUTO_FAST_LDFLAGS  "")

# Extract arch flags (-mcpu/-march) for fast targets so they can be isolated
set(AUTO_FAST_ARCH "")
string(REGEX MATCHALL "-m(cpu|arch)=[^ ]+" _arch_tokens "${AUTO_FAST_CXXFLAGS}")
if(_arch_tokens)
  list(GET _arch_tokens 0 AUTO_FAST_ARCH)
endif()

# Common warnings from Makefile
set(AUTO_WARN_FLAGS "")
string(REGEX MATCHALL "-W[^ ]+" _warn_tokens "${AUTO_CXXFLAGS_DBG}")
foreach(w ${_warn_tokens})
  if(NOT w MATCHES "-Wall|-Wextra|-Wpedantic")
    continue()
  endif()
  list(APPEND AUTO_WARN_FLAGS ${w})
endforeach()

# Common extras (frame-pointer etc.)
set(AUTO_COMMON_FLAGS "")
foreach(flag "-fno-omit-frame-pointer" "-ffast-math")
  string(FIND "${AUTO_CXXFLAGS_DBG} ${AUTO_FAST_CXXFLAGS}" "${flag}" _found)
  if(NOT _found EQUAL -1)
    list(APPEND AUTO_COMMON_FLAGS ${flag})
  endif()
endforeach()

# ------------------------ Source discovery ---------------------
# Mirror Makefile's globbing approach
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB TEST_FILES CONFIGURE_DEPENDS "tests/*.cpp")
set(MNIST_EXAMPLE "examples/mnist_demo.cpp")
set(RESNET_EXAMPLE "examples/ResNet_demo.cpp")
set(LSTM_EXAMPLE  "examples/lstm_shakespeare.cpp")

# ------------------------ Output layout ------------------------
# Recreate output paths similar to the Makefile
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)  # default
set(LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)

# Helper function to set an exe's output directory
function(place_exe tgt dir)
  set_target_properties(${tgt} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${dir}"
  )
endfunction()

# ------------------------ Threads (Linux) ----------------------
find_package(Threads)
set(THREADS_LIB "")
if(Threads_FOUND)
  set(THREADS_LIB Threads::Threads)
endif()

# ------------------------ Python bindings (pybind11) ------------------------
find_package(Python COMPONENTS Interpreter Development QUIET)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via find_package(pybind11). You can install it into your active Python: python3 -m pip install --user pybind11")
endif()

if(pybind11_FOUND AND Python_FOUND)
  # Build core library from project sources and make it PIC so it can be linked
  add_library(autocalc_lib STATIC ${SRC_FILES})
  set_target_properties(autocalc_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(_incs)
    target_include_directories(autocalc_lib PUBLIC ${_incs})
  endif()
  target_compile_features(autocalc_lib PRIVATE cxx_std_17)

  add_library(ag_python MODULE bindings/variable.cpp bindings/nn.cpp bindings/data.cpp)
  target_compile_features(ag_python PRIVATE cxx_std_17)
  if(_incs)
    target_include_directories(ag_python PRIVATE ${_incs})
  endif()
  # Link the core static library so the module exposes the project symbols
  target_link_libraries(ag_python PRIVATE autocalc_lib pybind11::module ${THREADS_LIB})
  # Build the compiled backend as a submodule named _backend inside the `ag` package.
  # This makes the compiled extension importable as `ag._backend` while the Python shim
  # remains the package entrypoint.
  set_target_properties(ag_python PROPERTIES PREFIX "" OUTPUT_NAME "_backend")
  # Place the compiled extension under build/python/ag so `build/python` added to PYTHONPATH
  # allows importing `ag` (the Python package) and `ag._backend` (the compiled submodule).
  set_target_properties(ag_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/ag")
  # Ensure directory exists at configure time for some build systems
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/python/ag")
  add_custom_target(build-bindings DEPENDS ag_python)
else()
  message(STATUS "Python bindings target skipped (pybind11 or Python not available).")
endif()

# ===============================================================
# Targets
# ===============================================================

# -------- Tests (sanitized) --------
add_executable(tests ${SRC_FILES} ${TEST_FILES})
target_compile_options(tests PRIVATE ${AUTO_WARN_FLAGS} -O0 -g -fno-omit-frame-pointer)
# Apply sanitizer/linker flags from Makefile
separate_arguments(_dbg_cxx NATIVE_COMMAND "${AUTO_CXXFLAGS_DBG}")
separate_arguments(_dbg_ld  NATIVE_COMMAND "${AUTO_LDFLAGS_DBG}")
target_compile_options(tests PRIVATE ${_dbg_cxx})
target_link_options(tests PRIVATE ${_dbg_ld})
target_link_libraries(tests PRIVATE ${THREADS_LIB})
place_exe(tests "tests")

# -------- Sanity pool (sanitized) --------
if(EXISTS ${CMAKE_SOURCE_DIR}/parallel_sanity/sanity_pool.cpp)
  add_executable(sanity_pool parallel_sanity/sanity_pool.cpp ${SRC_FILES})
  target_compile_options(sanity_pool PRIVATE ${AUTO_WARN_FLAGS} -O0 -g -fno-omit-frame-pointer)
  target_compile_options(sanity_pool PRIVATE ${_dbg_cxx})
  target_link_options(sanity_pool PRIVATE ${_dbg_ld})
  target_link_libraries(sanity_pool PRIVATE ${THREADS_LIB})
  place_exe(sanity_pool ".")
  add_custom_target(run-sanity
    COMMAND $<TARGET_FILE:sanity_pool> ${ARGS}
    DEPENDS sanity_pool
    USES_TERMINAL
  )
endif()

# -------- Debug demo (sanitized) --------
if(EXISTS ${CMAKE_SOURCE_DIR}/${MNIST_EXAMPLE})
  add_executable(mnist_demo_dbg ${MNIST_EXAMPLE} ${SRC_FILES})
  # Warning: tests/*.cpp are NOT added here (mirrors Makefile's filter-out)
  target_compile_options(mnist_demo_dbg PRIVATE ${AUTO_WARN_FLAGS} -O0 -g -fno-omit-frame-pointer)
  target_compile_options(mnist_demo_dbg PRIVATE ${_dbg_cxx})
  target_link_options(mnist_demo_dbg PRIVATE ${_dbg_ld})
  target_link_libraries(mnist_demo_dbg PRIVATE ${THREADS_LIB})
  place_exe(mnist_demo_dbg "demo_slow")
  add_custom_target(run-demo
    COMMAND $<TARGET_FILE:mnist_demo_dbg> ${ARGS}
    DEPENDS mnist_demo_dbg
    USES_TERMINAL
  )
endif()

# -------- Fast demos (release-ish) --------
# Common fast flags
separate_arguments(_fast_cxx NATIVE_COMMAND "${AUTO_FAST_CXXFLAGS}")
separate_arguments(_fast_ld  NATIVE_COMMAND "${AUTO_FAST_LDFLAGS}")

# Option to force fast-only flags for demos and core library (no sanitizers, no debug)
option(AUTOCALC_FORCE_FAST "Enforce fast compile flags (no sanitizers) for fast targets and core library" OFF)
if(AUTOCALC_FORCE_FAST)
  message(STATUS "AUTOCALC_FORCE_FAST=ON: forcing fast flags and disabling sanitizers for core/fast targets")
  # Compose the set of flags we want for fast targets and the core library
  set(_FORCE_FAST_FLAGS ${AUTO_WARN_FLAGS} -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx} -g0 -fno-sanitize=address -fno-sanitize=undefined)
  set(_FORCE_FAST_LDFLAGS ${_fast_ld})
  # If the core static lib exists (pybind11 path), force-recompile it with fast flags
  if(TARGET autocalc_lib)
    target_compile_options(autocalc_lib PRIVATE ${_FORCE_FAST_FLAGS})
    target_link_options(autocalc_lib PRIVATE ${_FORCE_FAST_LDFLAGS})
  endif()
  # Ensure any fast_* targets created below will also use the same flags by exposing a variable
  set(AUTOCALC__FORCE_FAST_FLAGS "${_FORCE_FAST_FLAGS}")
  set(AUTOCALC__FORCE_FAST_LDFLAGS "${_FORCE_FAST_LDFLAGS}")
endif()

# MNIST
if(EXISTS ${CMAKE_SOURCE_DIR}/${MNIST_EXAMPLE})
  add_executable(fast_mnist ${MNIST_EXAMPLE} ${SRC_FILES})
  # Force only fast flags for this target, disable sanitizers and debug info
  target_compile_options(fast_mnist PRIVATE ${AUTO_WARN_FLAGS} -DNDEBUG -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx} -g0 -fno-sanitize=address -fno-sanitize=undefined)
  target_link_options(fast_mnist PRIVATE ${_fast_ld} -fno-sanitize=address -fno-sanitize=undefined)
  target_link_libraries(fast_mnist PRIVATE ${THREADS_LIB})
  place_exe(fast_mnist "fast")
  add_custom_target(run-fast-mnist
    COMMAND $<TARGET_FILE:fast_mnist> ${ARGS}
    DEPENDS fast_mnist
    USES_TERMINAL
  )
endif()

# ResNet
if(EXISTS ${CMAKE_SOURCE_DIR}/${RESNET_EXAMPLE})
  add_executable(fast_resnet ${RESNET_EXAMPLE} ${SRC_FILES})
  target_compile_options(fast_resnet PRIVATE ${AUTO_WARN_FLAGS} -DNDEBUG -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx} -g0 -fno-sanitize=address -fno-sanitize=undefined)
  target_link_options(fast_resnet PRIVATE ${_fast_ld} -fno-sanitize=address -fno-sanitize=undefined)
  target_link_libraries(fast_resnet PRIVATE ${THREADS_LIB})
  place_exe(fast_resnet "fast")
  add_custom_target(run-fast-resnet
    COMMAND $<TARGET_FILE:fast_resnet> ${ARGS}
    DEPENDS fast_resnet
    USES_TERMINAL
  )
endif()

# LSTM
if(EXISTS ${CMAKE_SOURCE_DIR}/${LSTM_EXAMPLE})
  add_executable(fast_lstm ${LSTM_EXAMPLE} ${SRC_FILES})
  target_compile_options(fast_lstm PRIVATE ${AUTO_WARN_FLAGS} -DNDEBUG -O3 ${AUTO_COMMON_FLAGS} ${_fast_cxx} -g0 -fno-sanitize=address -fno-sanitize=undefined)
  target_link_options(fast_lstm PRIVATE ${_fast_ld} -fno-sanitize=address -fno-sanitize=undefined)
  target_link_libraries(fast_lstm PRIVATE ${THREADS_LIB})
  place_exe(fast_lstm "fast")
  add_custom_target(run-fast-lstm
    COMMAND $<TARGET_FILE:fast_lstm> ${ARGS}
    DEPENDS fast_lstm
    USES_TERMINAL
  )
endif()

# Convenience "all" similar to Makefile: build tests + fast mnist + fast lstm
add_custom_target(build-all
  DEPENDS tests
)
if(TARGET fast_mnist)
  add_dependencies(build-all fast_mnist)
endif()
if(TARGET fast_lstm)
  add_dependencies(build-all fast_lstm)
endif()

# ------------------------ Notes ------------------------
message(STATUS "Includes: ${_incs}")
message(STATUS "Debug CXXFLAGS: ${AUTO_CXXFLAGS_DBG}")
message(STATUS "Fast  CXXFLAGS: ${AUTO_FAST_CXXFLAGS}")
