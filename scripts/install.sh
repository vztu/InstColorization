pip install --user -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install --user cython pyyaml==5.1
pip install --user -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install --user dominate==2.4.0
# pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
pip install --user detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

pip install --user visdom

conda install -c anaconda ipython
