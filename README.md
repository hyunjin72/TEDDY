## Running the Code
- Create a virtual environment
```
sudo apt-get install python3-venv
python3 -m venv myvenv
source myvenv/bin/activate
```
- Install required packages
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.8 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
```
- Install following packages
```
pip install -r requirements.txt
```
- To reproduce the performance of Teddy on Citeseer (GIN), run the following script:
```
python main_pruning_proposed.py
```
