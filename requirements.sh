# create conda environment
# conda create --name myEnv python=3.7
# conda activate myEnv

# install dependencies
conda install jupyter pyyaml
conda install -c anaconda cloudpickle
conda install pytorch torchvision -c pytorch

# use pip to install opencv
# otherwise cv2.imshow will not work
pip install opencv-contrib-python
pip install cython matplotlib tqdm scipy numpy 
