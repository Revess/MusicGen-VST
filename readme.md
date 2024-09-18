## Install
```git clone https://github.com/DISTRHO/DPF.git```

```git submodule update --init --recursive```


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