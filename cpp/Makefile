# ctbrain_cv — extração de parênquima cerebral em TC com OpenCV (C++17)
#
#   make            # build release em bin/ctbrain_cv
#   make debug      # build com -O0 -g e sanitizers
#   make clean
#
# Dependências: OpenCV 4 (pkg-config opencv4) e zlib.
#   macOS:  brew install opencv pkg-config
#   Ubuntu: sudo apt install libopencv-dev zlib1g-dev pkg-config

CXX      ?= c++
TARGET   := bin/ctbrain_cv
BUILD    := build
SRC      := $(wildcard src/*.cpp)
OBJ      := $(SRC:src/%.cpp=$(BUILD)/%.o)
DEP      := $(OBJ:.o=.d)

# headers do OpenCV como -isystem: avisos de terceiros não poluem o build
OPENCV_CFLAGS := $(patsubst -I%,-isystem %,$(shell pkg-config --cflags opencv4 2>/dev/null))
# só os módulos usados (core + imgproc), não a biblioteca inteira
OPENCV_LIBS   := $(shell pkg-config --libs-only-L opencv4 2>/dev/null) -lopencv_imgproc -lopencv_core

OPTFLAGS ?= -O2 -DNDEBUG
SANITIZE := -fsanitize=address,undefined
CXXFLAGS += $(OPTFLAGS) -std=c++17 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
CPPFLAGS += -Iinclude $(OPENCV_CFLAGS) -MMD -MP
LDLIBS   += $(OPENCV_LIBS) -lz

.PHONY: all debug clean check-deps help

all: check-deps $(TARGET)

debug:
	$(MAKE) clean
	$(MAKE) OPTFLAGS="-O0 -g $(SANITIZE)" LDFLAGS="$(SANITIZE)"

$(TARGET): $(OBJ) | bin
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD) bin:
	@mkdir -p $@

check-deps:
	@pkg-config --exists opencv4 || { \
	  echo "OpenCV 4 não encontrado via pkg-config (macOS: brew install opencv pkg-config)"; exit 1; }

clean:
	rm -rf $(BUILD) bin

help:
	@sed -n '1,12p' Makefile

-include $(DEP)
