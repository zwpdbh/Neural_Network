# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Users/zw/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/173.4674.29/CLion.app/Contents/bin/cmake/bin/cmake"

# The command to remove a file.
RM = "/Users/zw/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/173.4674.29/CLion.app/Contents/bin/cmake/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zw/code/C_and_C++_Projects/Neural_Network

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/neural_network.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/neural_network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neural_network.dir/flags.make

CMakeFiles/neural_network.dir/main.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neural_network.dir/main.cpp.o"
	/usr/local/opt/llvm/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neural_network.dir/main.cpp.o -c /Users/zw/code/C_and_C++_Projects/Neural_Network/main.cpp

CMakeFiles/neural_network.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/main.cpp.i"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/Neural_Network/main.cpp > CMakeFiles/neural_network.dir/main.cpp.i

CMakeFiles/neural_network.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/main.cpp.s"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/Neural_Network/main.cpp -o CMakeFiles/neural_network.dir/main.cpp.s

CMakeFiles/neural_network.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/neural_network.dir/main.cpp.o.requires

CMakeFiles/neural_network.dir/main.cpp.o.provides: CMakeFiles/neural_network.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/neural_network.dir/build.make CMakeFiles/neural_network.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/neural_network.dir/main.cpp.o.provides

CMakeFiles/neural_network.dir/main.cpp.o.provides.build: CMakeFiles/neural_network.dir/main.cpp.o


CMakeFiles/neural_network.dir/dataset.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/dataset.cpp.o: ../dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/neural_network.dir/dataset.cpp.o"
	/usr/local/opt/llvm/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neural_network.dir/dataset.cpp.o -c /Users/zw/code/C_and_C++_Projects/Neural_Network/dataset.cpp

CMakeFiles/neural_network.dir/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/dataset.cpp.i"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/Neural_Network/dataset.cpp > CMakeFiles/neural_network.dir/dataset.cpp.i

CMakeFiles/neural_network.dir/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/dataset.cpp.s"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/Neural_Network/dataset.cpp -o CMakeFiles/neural_network.dir/dataset.cpp.s

CMakeFiles/neural_network.dir/dataset.cpp.o.requires:

.PHONY : CMakeFiles/neural_network.dir/dataset.cpp.o.requires

CMakeFiles/neural_network.dir/dataset.cpp.o.provides: CMakeFiles/neural_network.dir/dataset.cpp.o.requires
	$(MAKE) -f CMakeFiles/neural_network.dir/build.make CMakeFiles/neural_network.dir/dataset.cpp.o.provides.build
.PHONY : CMakeFiles/neural_network.dir/dataset.cpp.o.provides

CMakeFiles/neural_network.dir/dataset.cpp.o.provides.build: CMakeFiles/neural_network.dir/dataset.cpp.o


CMakeFiles/neural_network.dir/neuralnetwork.cpp.o: CMakeFiles/neural_network.dir/flags.make
CMakeFiles/neural_network.dir/neuralnetwork.cpp.o: ../neuralnetwork.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/neural_network.dir/neuralnetwork.cpp.o"
	/usr/local/opt/llvm/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neural_network.dir/neuralnetwork.cpp.o -c /Users/zw/code/C_and_C++_Projects/Neural_Network/neuralnetwork.cpp

CMakeFiles/neural_network.dir/neuralnetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neural_network.dir/neuralnetwork.cpp.i"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zw/code/C_and_C++_Projects/Neural_Network/neuralnetwork.cpp > CMakeFiles/neural_network.dir/neuralnetwork.cpp.i

CMakeFiles/neural_network.dir/neuralnetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neural_network.dir/neuralnetwork.cpp.s"
	/usr/local/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zw/code/C_and_C++_Projects/Neural_Network/neuralnetwork.cpp -o CMakeFiles/neural_network.dir/neuralnetwork.cpp.s

CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.requires:

.PHONY : CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.requires

CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.provides: CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.requires
	$(MAKE) -f CMakeFiles/neural_network.dir/build.make CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.provides.build
.PHONY : CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.provides

CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.provides.build: CMakeFiles/neural_network.dir/neuralnetwork.cpp.o


# Object files for target neural_network
neural_network_OBJECTS = \
"CMakeFiles/neural_network.dir/main.cpp.o" \
"CMakeFiles/neural_network.dir/dataset.cpp.o" \
"CMakeFiles/neural_network.dir/neuralnetwork.cpp.o"

# External object files for target neural_network
neural_network_EXTERNAL_OBJECTS =

../bin/neural_network: CMakeFiles/neural_network.dir/main.cpp.o
../bin/neural_network: CMakeFiles/neural_network.dir/dataset.cpp.o
../bin/neural_network: CMakeFiles/neural_network.dir/neuralnetwork.cpp.o
../bin/neural_network: CMakeFiles/neural_network.dir/build.make
../bin/neural_network: CMakeFiles/neural_network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/neural_network"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neural_network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neural_network.dir/build: ../bin/neural_network

.PHONY : CMakeFiles/neural_network.dir/build

CMakeFiles/neural_network.dir/requires: CMakeFiles/neural_network.dir/main.cpp.o.requires
CMakeFiles/neural_network.dir/requires: CMakeFiles/neural_network.dir/dataset.cpp.o.requires
CMakeFiles/neural_network.dir/requires: CMakeFiles/neural_network.dir/neuralnetwork.cpp.o.requires

.PHONY : CMakeFiles/neural_network.dir/requires

CMakeFiles/neural_network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neural_network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neural_network.dir/clean

CMakeFiles/neural_network.dir/depend:
	cd /Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zw/code/C_and_C++_Projects/Neural_Network /Users/zw/code/C_and_C++_Projects/Neural_Network /Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug /Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug /Users/zw/code/C_and_C++_Projects/Neural_Network/cmake-build-debug/CMakeFiles/neural_network.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neural_network.dir/depend

