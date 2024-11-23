#!/usr/bin/make -f

NAME = MusicGenVST
CXXFLAGS += -std=c++17

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
    CXXFLAGS += -I/usr/include/jsoncpp -I/usr/include -I/opt/homebrew/include -DLINUX
    LDFLAGS += -lcurl -ljsoncpp -lsndfile
endif

ifeq ($(UNAME_S), Darwin)
    CXXFLAGS += -I/usr/local/include/jsoncpp -I/usr/local/include -I/opt/homebrew/include -DDARWIN -arch arm64
    LDFLAGS += -lcurl -ljsoncpp -lsndfile
endif
ifeq ($(OS), Windows_NT)
endif

# --------------------------------------------------------------
# Files to build

FILES_DSP = \
	MusicGen.cpp

FILES_UI  = \
	MusicGenUI.cpp \
	./src/SimpleButton.cpp \
	./src/Knob.cpp \
	./src/TextInput.cpp \
	./src/Panel.cpp \
	./src/Label.cpp \
	./src/Checkbox.cpp \
	./src/ValueIndicator.cpp \
	./src/WidgetGroup.cpp \
	./src/Layout.cpp \
	./src/WAIVEWidget.cpp \
	tinyfiledialogs.c

# --------------------------------------------------------------
# Do some magic

include ../../dpf/Makefile.plugins.mk

# --------------------------------------------------------------
# VST2, LV2, and Standalone targets

TARGETS += vst
TARGETS += vst3
ifeq ($(HAVE_DGL), true)
TARGETS += lv2_sep
else
TARGETS += lv2_dsp
endif

TARGETS += jack

all: $(TARGETS)

# --------------------------------------------------------------
