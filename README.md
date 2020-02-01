# mydl
my custom "[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)" as [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 

released under MIT license

thanks for hint, xuan

note:

_C.so : maskrcnn-benchmark(name replaced with mydl) C compiled module

git clone https://github.com/sailfish009/maskrcnn-benchmark

cd maskrcnn-benchmark

python3 setup.py build

cp build/lib.linux-x86_64-3.7/mydl/_C.cpython-xxm-x86_64-linux-gnu.so ../mydl/_C.so
