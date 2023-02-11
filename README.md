# Installation
To setup `mot-mmtrack`, run the following command:
```bash
conda create -n mot-mmtrack python=3.10
pip install torch torchvision openmim
mim install -r requirements/mminstall.txt
pip install -v -e .
pip install mmyolo
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

# Usage
Run the demo:
```bash
python demo/demo_mot_vis.py \
    configs/mot/deepsort/my_config.py \
    --input demo/demo.mp4 \
    --output outputs
```