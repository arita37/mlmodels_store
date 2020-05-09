
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_dataloader GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'c650f5b1ef2efd9067a12b489479a213849a404f', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c650f5b1ef2efd9067a12b489479a213849a404f

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 317826.43it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 409809.58it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 567224.96it/s] 33%|███▎      | 3309568/9912422 [00:00<00:08, 800496.32it/s] 67%|██████▋   | 6635520/9912422 [00:00<00:02, 1130150.31it/s]9920512it [00:00, 10119349.81it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 143778.13it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 314072.48it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 406141.79it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 562187.25it/s]1654784it [00:00, 2826714.67it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52926.03it/s]            ['resnet34_benchmark_mnist.json', 'resnet18_benchmark_new.json', 'keras_reuters_charcnn.json', 'kears_textcnn.json', 'namentity_crm_bilstm_new.json', 'armdn_dataloader.json', 'matchzoo_models_new.json', 'resnet18_benchmark_mnist.json', 'torchhub_cnn_dataloader.json']





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json
#####  Load JSON data_pars
{'input_pars': {'path': 'dataset/vision/MNIST'}, 'loader': {'uri': 'preprocess/image.py::torchvision_dataset_MNIST_load', 'pass_data_pars': False}, 'output': {'format_func': {'uri': 'preprocess/image.py::wrap_torch_datasets', 'args': {'args_list': [{'batch_size': 4}, {'batch_size': 4}], 'shuffle': True}, 'pass_data_pars': False}}}

 #####  Load DataLoader 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json 'data_info'





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_new.json
#####  Load JSON data_pars
{'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True}}]}

 #####  Load DataLoader 

 #####  compute DataLoader 
Mismatch input / output data type {'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True}} 
URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True}
postional parameteres :  ['data_info']
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Processing...
Done!
((<torch.utils.data.dataloader.DataLoader object at 0x7fc54cd81b38>, <torch.utils.data.dataloader.DataLoader object at 0x7fc4e6d8c6a0>), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json
#####  Load JSON data_pars
{'input_pars': {'path': 'dataset/text/reuters/reuters.npz', 'test_size': 0.2}, 'loader': {'uri': 'test_dataloader.py::load_npz', 'pass_data_pars': False}, 'preprocessor': {'uri': 'test_dataloader.py::SingleFunctionPreprocessor', 'arg': {'func_dict': {'uri': 'test_dataloader.py::tokenize_x', 'arg': {'no_classes': 46}}}, 'pass_data_pars': False}, 'split_train_test': {'uri': 'test_dataloader.py::timeseries_split', 'arg': {}, 'pass_data_pars': False, 'testsize_keyword': 'test_size'}, 'alphabet_size': 69, 'input_size': 10000, 'num_of_classes': 46}

 #####  Load DataLoader 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json 'data_info'





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//kears_textcnn.json
#####  Load JSON data_pars
{'data_info': {'dataset': 'mlmodels/dataset/text/imdb', 'pass_data_pars': False, 'train': True, 'maxlen': 40, 'max_features': 5}, 'preprocessors': [{'name': 'loader', 'uri': 'mlmodels/preprocess/generic.py::NumpyDataset', 'args': {'encoding': "'ISO-8859-1'"}}]}

 #####  Load DataLoader 

 #####  compute DataLoader 
Mismatch input / output data type {'name': 'loader', 'uri': 'mlmodels/preprocess/generic.py::NumpyDataset', 'args': {'encoding': "'ISO-8859-1'"}} 
URL:  mlmodels/preprocess/generic.py::NumpyDataset {'encoding': "'ISO-8859-1'"}
cls_name : NumpyDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//kears_textcnn.json Object arrays cannot be loaded when allow_pickle=False





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//namentity_crm_bilstm_new.json
#####  Load JSON data_pars
{'data_info': {'data_path': 'dataset/text/', 'dataset': 'ner_dataset.csv', 'pass_data_pars': False, 'train': True}, 'preprocessors': [{'name': 'loader', 'uri': 'mlmodels/preprocess/generic.py::pandasDataset', 'args': {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []}}, {'uri': 'mlmodels/preprocess/text_keras.py::Preprocess_namentity', 'args': {'max_len': 75}, 'internal_states': ['word_count']}, {'name': 'split_xy', 'uri': 'mlmodels/dataloader.py::split_xy_from_dict', 'args': {'col_Xinput': ['X'], 'col_yinput': ['y']}}, {'name': 'split_train_test', 'uri': 'sklearn.model_selection::train_test_split', 'args': {'test_size': 0.5}}, {'name': 'saver', 'uri': 'mlmodels/dataloader.py::pickle_dump', 'args': {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'}}], 'output': {'shape': [[75], [75, 18]], 'max_len': 75}}

 #####  Load DataLoader 

 #####  compute DataLoader 
Mismatch input / output data type {'name': 'loader', 'uri': 'mlmodels/preprocess/generic.py::pandasDataset', 'args': {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []}} 
URL:  mlmodels/preprocess/generic.py::pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []}
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//namentity_crm_bilstm_new.json 'data_info' is required fields in pandasDataset





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json
#####  Load JSON data_pars
{'input_pars': {'path': 'dataset/timeseries/HOBBIES_1_001_CA_1_validation.csv', 'col_Xinput': ['Demand'], 'col_yinput': ['Demand'], 'test_size': 0.1}, 'preprocessor': {'uri': 'test_dataloader.py::SingleFunctionPreprocessor', 'arg': {'func_dict': {'uri': 'preprocess/timeseries.py::pd_fillna', 'arg': {'method': 'pad'}}}, 'pass_data_pars': False}, 'split_xy': {'uri': 'test_dataloader.py::split_timeseries_df', 'arg': {'length': 60, 'shift': 1}}, 'split_train_test': {'uri': 'test_dataloader.py::identical_test_set_split', 'arg': {}, 'pass_data_pars': False, 'testsize_keyword': 'test_size'}}

 #####  Load DataLoader 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json 'data_info'





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json

#####  Load JSON data_pars
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json Extra data: line 172 column 2 (char 5004)





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json
#####  Load JSON data_pars
{'input_pars': {'path': 'dataset/vision/MNIST'}, 'loader': {'uri': 'preprocess/image.py::torchvision_dataset_MNIST_load', 'pass_data_pars': False}, 'output': {'format_func': {'uri': 'preprocess/image.py::torch_datasets_wrapper', 'args': {'args_list': [{'batch_size': 4}, {'batch_size': 4}], 'shuffle': True}, 'pass_data_pars': False}}}

 #####  Load DataLoader 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json 'data_info'





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json
#####  Load JSON data_pars
{'input_pars': {'path': 'dataset/vision/MNIST'}, 'loader': {'uri': 'preprocess/image.py::torchvision_dataset_MNIST_load', 'pass_data_pars': False}, 'output': {'format_func': {'uri': 'preprocess/image.py::wrap_torch_datasets', 'args': {'args_list': [{'batch_size': 4}, {'batch_size': 1}], 'shuffle': True}, 'pass_data_pars': False}}}

 #####  Load DataLoader 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json 'data_info'
