#!/bin/bash
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

export PKG_CONFIG_PATH="/opt/homebrew/opt/qt/lib/pkgconfig:$PKG_CONFIG_PATH"

if [ "$UNAME_S" = "Linux" ]; then
    make 
    sudo cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
    ../../bin/MusicGenVST
elif [ "$UNAME_S" = "Darwin" ]; then
    if [ "$UNAME_M" = "x86_64" ]; then
        arch -x86_64 make
        echo "Running on macOS x64"
        codesign --deep --force --sign - ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
        sudo rm -rf ../../vst2/MusicGenVST.vst
        sudo cp -r ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
        codesign --deep --force --sign - ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
        rm -rf ../../vst3/MusicGenVST.vst3
        cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
        arch -x86_64 ../../bin/MusicGenVST.app/Contents/MacOS/MusicGenVST
    elif [ "$UNAME_M" = "arm64" ]; then
        arch -arm64 make
        echo "Running on macOS arm64"
        codesign --deep --force --sign - ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
        sudo rm -rf ../../vst2/MusicGenVST.vst
        sudo cp -r ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
        codesign --deep --force --sign - ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
        rm -rf ../../vst3/MusicGenVST.vst3
        cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
        arch -arm64 ../../bin/MusicGenVST.app/Contents/MacOS/MusicGenVST
    fi
elif [ "$OS" = "Windows_NT" ]; then
    make 
    ../../bin/MusicGenVST
else
    echo "Unsupported operating system: $UNAME_S"
    exit 1
fi