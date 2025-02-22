## Install
```sudo apt update```
```sudo apt install git cmake g++ pkg-config libx11-dev libgl1-mesa-dev libasound2-dev libjack-jackd2-dev meson```

```git clone https://github.com/DISTRHO/DPF.git```

```git submodule update --init --recursive```

### OSX
#### ARM64
```
arch -arm64 brew install jsoncpp curl sndfile
cd ./plugins/MusicGen
chmod +x ./build.sh
./build.sh
```

#### AMD64
```
arch -x86_64 brew install jsoncpp curl sndfile
cd ./plugins/MusicGen
chmod +x ./build.sh
./build.sh
```

# Ignor
```
sudo apt update
sudo apt install meson ninja-build
git clone https://github.com/lv2/pugl.git
cd pugl
meson setup build
meson compile -C build
sudo meson install -C build
```

```jupyterlab-markup```

## Linux make setup
```sudo apt-get install libcurl4-openssl-dev libjsoncpp-dev```

brew install libsndfile jsoncpp