#!/bin/bash
# make clean
make 
sudo cp -r ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
sudo cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
../../bin/MusicGenVST.app/Contents/MacOS/MusicGenVST