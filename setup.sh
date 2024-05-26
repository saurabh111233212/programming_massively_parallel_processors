sudo apt-get update
sudo apt-get install -y cuda
export PATH=/usr/local/cuda/bin:$PATH
source ~/.bashrc
source ~/.zshrc
nvcc --version