#!/bin/bash
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

export PKG_CONFIG_PATH="/opt/homebrew/opt/qt/lib/pkgconfig:$PKG_CONFIG_PATH"

if [ "$UNAME_S" = "Linux" ]; then
    make 
    sudo rm -rf ../../vst2/MusicGenVST.vst
    sudo cp -r ../../bin/MusicGenVST.vst ../../vst2/MusicGenVST.vst
    rm -rf ../../vst3/MusicGenVST.vst3
    sudo cp -r ../../bin/MusicGenVST.vst3 ../../vst3/MusicGenVST.vst3
    ../../bin/MusicGenVST
    zip -r ../../release/standalone-x86_64-linux.zip ../../bin/MusicGenVST.app
    zip -r ../../release/vst2-x86_64-linux.zip ../../bin/MusicGenVST.vst
    zip -r ../../release/vst3-x86_64-linux.zip ../../bin/MusicGenVST.vst3
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
        zip -r ../../release/standalone-x86_64-osx.zip ../../bin/MusicGenVST.app
        zip -r ../../release/vst2-x86_64-osx.zip ../../bin/MusicGenVST.vst
        zip -r ../../release/vst3-x86_64-osx.zip ../../bin/MusicGenVST.vst3
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
        zip -r ../../release/standalone-arm64-osx.zip ../../bin/MusicGenVST.app
        zip -r ../../release/vst2-arm64-osx.zip ../../bin/MusicGenVST.vst
        zip -r ../../release/vst3-arm64-osx.zip ../../bin/MusicGenVST.vst3
    fi
elif [ "$OS" = "Windows_NT" ]; then
    make 
    ../../bin/MusicGenVST
else
    echo "Unsupported operating system: $UNAME_S"
    exit 1
fi