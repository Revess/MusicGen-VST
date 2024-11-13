#!/bin/bash
# make clean
make 


UNAME_S=$(uname -s)

if [ "$UNAME_S" = "Linux" ]; then
    sudo cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
    ../../bin/MusicGenVST
elif [ "$UNAME_S" = "Darwin" ]; then
    sudo cp -r ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
    sudo cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
    ../../bin/MusicGenVST.app/Contents/MacOS/MusicGenVST
elif [ "$OS" = "Windows_NT" ]; then
    ../../bin/MusicGenVST
else
    echo "Unsupported operating system: $UNAME_S"
    exit 1
fi
