## Installation
```bash
git clone https://github.com/juj/emsdk.git
cd emsdk
emsdk update
emsdk install latest
emsdk activate latest
```

## Command
```bash
emcc SDK/cpp/validate.cpp -s EXPORTED_RUNTIME_METHODS=['ccall','UTF8ToString'] -s EXPORTED_FUNCTIONS=['_malloc','_free'] -o SDK/cpp/validate.js
```