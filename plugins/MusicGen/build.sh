#!/bin/bash
make clean
make 
cp -r ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
open  ../../bin/MusicGenVST.app