#!/usr/bin/make -f
# Makefile for DISTRHO Plugins #
# ---------------------------- #
# Created by falkTX
#
# Modified by REIS0

VST2 ?= true
LV2 ?= false
STANDALONE ?= false

# --------------------------------------------------------------
# Project name, used for binaries

NAME = MusicGenVST

# --------------------------------------------------------------
# Files to build

FILES_DSP = \
	MusicGen.cpp

FILES_UI  = \
	MusicGenUI.cpp

# --------------------------------------------------------------
# Do some magic

include ../../dpf/Makefile.plugins.mk

# --------------------------------------------------------------
# VST2, LV2, and Standalone targets

ifeq ($(VST2), true)
TARGETS += vst
endif

ifeq ($(LV2), true)
ifeq ($(HAVE_DGL), true)
TARGETS += lv2_sep
else
TARGETS += lv2_dsp
endif
endif

ifeq ($(STANDALONE), true)
TARGETS += jack
endif

all: $(TARGETS)

# --------------------------------------------------------------
