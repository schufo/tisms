# Joint phoneme alignment and text-informed speech separation on highly corrupted speech

Here you find the code to reproduce the experiments of the paper "Joint phoneme alignment and text-informed speech separation on highly corrupted speech" by Kilian Schulze-Forster, Clement S. J. Doire, Gaël Richard, Roland Badeau. Accepted at *IEEE International Conference on Audio, Speech, and Signal Processing, 2020.*

The paper and audio examples are availabe [here](https://schufo.github.io/publications/2020-ICASSP)

## Download
Clone the repository to your machine:
<pre>
git clone https://github.com/schufo/tisms.git
</pre>

Make sure that your working directory is 'tisms/' for all steps described below.

## Virtual Environment
The project was done in a conda environment with python 3.6. You can create one with the following command:
<pre>
conda create -n tisms_env python=3.6
</pre>

Activate the environment:
<pre>
source activate tisms_env
</pre>

Then install pytorch. I was using version 1.1.0 but later versions should work as well (I did not test it though).
<pre>
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
</pre>

The you can run the following command to install all other required packages with pip:
<pre>
pip install -r requirements.txt
</pre>


## Data Preprocessing

At the bottom of the script 01\_musdb\_pre\_processing.py enter the correct links to your MUSDB dataset and to the directory where you want to save the preprocessed MUDSB data.

Run 01\_musdb\_pre\_processing.py

Run 02\_make\_timit\_phoneme\_vocabulary.py

In the folder 'data' you find three python files containing the data set classes for training, validation, and testing. Enter the correct path to your TIMIT dataset and to the preprocessed MUSDB data at the top of all three files.


## Training

To train the Baseline (BL) model run the following from the command line
<pre>
python 03_train_BL.py
</pre>

With the following commands you can train the three versions of the text-informed models:
<pre>
python 04_train_informed_models.py with 'tag="V1"'

python 04_train_informed_models.py with 'tag="V2"' 'side_info_encoder_bidirectional=False'

python 04_train_informed_models.py with 'tag="V3"' 'model="InformedSeparatorWithSplitAttention"'
</pre>

To train the model with Optimal Attention (OA) weights run:
<pre>
python 05_train_OA.py
</pre>

For this project I tried the experiment tracking package "sacred". Since scripts we want to run for testing need to access configuration files by their tag we assigned during training we now need to run:
<pre>
python 06_copy_configs.py
</pre>


## Evaluation

eval alignment: python 07_eval_alignment.py with 'test_snr=100' 'tag="V1"'

'test_snr=100' means clean speech, other wise run with 'test_snr=-5'

eval separation python 08_eval_separation.py with 'tag="BL"'

run with all tags

what is saved after eval? script to show results?