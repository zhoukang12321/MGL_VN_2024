# MGL_VN_2024
Code and datasets for paper Multimodal Graph Learning and Action Boost Memory for Visual Navigation

1.Prepare Pretrained DETR pth file from HOZ https://github.com/sx-zhang/HOZ.git.
2.Prepare pretrained Faster-rcnn file from VGM https://github.com/rllab-snu/Visual-Graph-Memory.git and VTNet https://github.com/xiaobaishu0097/ICLR_VTNet.git
3.Download AI2THOR datasets https://prior-datasets.s3.us-east-2.amazonaws.com/savn/data.tar.gz and pretrained model https://prior-datasets.s3.us-east-2.amazonaws.com/savn/pretrained_models.tar.gz
4.The data folder contains:

thor_offline_data which is organized into sub-folders, each of which corresponds to a scene in AI2-THOR. For each room we have scraped the ResNet features of all possible locations in addition to a metadata and NetworkX graph of possible navigations in the scene.
thor_glove which contains the GloVe embeddings for the navigation targets.
gcn which contains the necessary data for the Graph Convolutional Network (GCN) in Scene Priors, including the adjacency matrix.
Note that the starting positions and scenes for the test and validation set may be found in test_val_split.

If you wish to access the RGB images in addition to the ResNet features, replace thor_offline_data with thor_offlline_data_with_images. If you wish to run your model on the image files, add the command line argument --images_file_name images.hdf5.

Install the necessary packages. If you are using pip then simply run pip install -r requirements.txt.
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/xxx_pretrained.dat \
    --model xxx \
    --results_json savn_test.json 
