#!/usr/bin/make -f

NAME = MusicGenVST

# --------------------------------------------------------------
# Files to build

FILES_DSP = \
	MusicGen.cpp

FILES_UI  = \
	MusicGenUI.cpp \
	./src/NanoButton.cpp

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
