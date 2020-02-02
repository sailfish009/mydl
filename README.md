# mydl
my custom "[detectron2](https://github.com/facebookresearch/detectron2)" as [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 

released under MIT license

thanks for hint, xuan

note:

_C.so : detectron2(name replaced with mydl) C compiled module

git clone https://github.com/sailfish009/mydl_C

cd mydl_C

CC=gcc-8 CXX=g++-8 python3 setup.py build

cp build/lib.linux-x86_64-3.7/mydl/_C.cpython-xxm-x86_64-linux-gnu.so ../mydl/_C.so
