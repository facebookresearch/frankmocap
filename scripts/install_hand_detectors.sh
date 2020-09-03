mkdir -p detectors
cd detectors

pip install gdown

# Install 100-DOH hand-object detectors
git clone https://github.com/ddshan/hand_object_detector
# compile
cd hand_object_detector/lib
python setup.py build develop
cd ../../

# Install 100-DOH hand-only detectors
git clone git@github.com:ddshan/hand_detector.d2.git
mv hand_detector.d2 hand_only_detector

# downloading weights
gdown https://drive.google.com/uc?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE
gdown https://drive.google.com/uc?id=1OqgexNM52uxsPG3i8GuodDOJAGFsYkPg
mkdir -p ../data/weights/hand_detector
mv *pth ../data/weights/hand_detector
