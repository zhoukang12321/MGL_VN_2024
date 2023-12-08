# MGL for VN
Multimodal Graph Learning and Action Boost Memory for Visual Navigation

DETR pretrained file can be found in https://github.com/sx-zhang/HOZ.git
VTNet pretrained model can be found in https://github.com/xiaobaishu0097/ICLR_VTNet.git
Multimodal Fusion model can be found in https://github.com/barmayo/spatial_attention.git
We modified the multimodal fusion using GAT network and use DETR module as detector and structure of DETR like VTNet as another visual inputs. Besides, we build another SG datasets like Visual Genome datasets in Scene Prior paper.
We public some codes in this branch and when the paper are accepted, we will release whole datasets and codes.



Setup
Clone the repository with git clone https://github.com/zhoukang12321/MGL_VN_2024.git && cd MGL_VN_2024.

Install the necessary packages. If you are using pip then simply run pip install -r requirements.txt.

Download the pretrained models and data to the MGL_VN_2024 directory. Untar with

tar -xzf pretrained_models.tar.gz
tar -xzf data.tar.gz
The data folder contains:

thor_offline_data which is organized into sub-folders, each of which corresponds to a scene in AI2-THOR. For each room we have scraped the ResNet features of all possible locations in addition to a metadata and NetworkX graph of possible navigations in the scene.
thor_glove which contains the GloVe embeddings for the navigation targets.
gcn which contains the necessary data for the Graph Convolutional Network (GCN) in Scene Priors, including the adjacency matrix.
Note that the starting positions and scenes for the test and validation set may be found in test_val_split.

If you wish to access the RGB images in addition to the ResNet features, replace thor_offline_data with thor_offlline_data_with_images. If you wish to run your model on the image files, add the command line argument --images_file_name images.hdf5.

Evaluation using Pretrained Models
Use the following code to run the pretrained models on the test set. Add the argument --gpu-ids 0 1 to speed up the evaluation by using GPUs.

SAVN
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/savn_pretrained.dat \
    --model SAVN \
    --results_json savn_test.json 

cat savn_test.json 
Scene Priors
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/gcn_pretrained.dat \
    --model GCN \
    --glove_dir ./data/gcn \
    --results_json scene_priors_test.json

cat scene_priors_test.json 
Non-Adaptvie-A3C
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/nonadaptivea3c_pretrained.dat \
    --results_json nonadaptivea3c_test.json

cat nonadaptivea3c_test.json
The result may vary depending on system and set-up though we obtain:

Model	SPL ≥ 1	Success ≥ 1	SPL ≥ 5	Success ≥ 5
SAVN	16.13	42.20	14.30	30.09
Scene Priors	14.86	36.90	11.49	24.70
Non-Adaptive A3C	14.10	32.40	10.73	19.16
The results in the initial submission (shown below) were the best (in terms of success on the validation set). After the initial submission, we trained the model 5 times from scratch to obtain error bars, which you may find in results.

Model	SPL ≥ 1	Success ≥ 1	SPL ≥ 5	Success ≥ 5
SAVN	16.13	42.10	13.19	30.54
Non-Adaptive A3C	13.73	32.90	10.88	20.66
How to Train your SAVN
You may train your own models by using the commands below.

Training SAVN
python main.py \
    --title savn_train \
    --model SAVN \
    --gpu-ids 0 1 \
    --workers 12
Training Non-Adaptvie A3C
python main.py \
    --title nonadaptivea3c_train \
    --gpu-ids 0 1 \
    --workers 12
How to Evaluate your Trained Model
You may use the following commands for evaluating models you have trained.

SAVN
python full_eval.py \
    --title savn \
    --model SAVN \
    --results_json savn_results.json \
    --gpu-ids 0 1
    
cat savn_results.json
Non-Adaptive A3C
python full_eval.py \
    --title nonadaptivea3c \
    --results_json nonadaptivea3c_results.json \
    --gpu-ids 0 1
    
cat nonadaptivea3c_results.json
Random Agent
python main.py \
    --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --title random_test \
    --agent_type RandomNavigationAgent \
    --results_json random_results.json
    
cat random_results.json
