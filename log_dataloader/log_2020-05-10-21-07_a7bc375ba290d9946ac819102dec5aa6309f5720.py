
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a7bc375ba290d9946ac819102dec5aa6309f5720', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'a7bc375ba290d9946ac819102dec5aa6309f5720', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a7bc375ba290d9946ac819102dec5aa6309f5720

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a7bc375ba290d9946ac819102dec5aa6309f5720

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
['/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_textcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//namentity_crm_bilstm_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//charcnn.json']





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 

#####  Load JSON data_pars
{'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torchvision_dataset_MNIST_load', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'ztest/mnist/'}}, 'shuffle': True, 'download': True}}]}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels.preprocess.generic::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torchvision_dataset_MNIST_load', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'ztest/mnist/'}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2225359e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2225359e18>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2225359e18> , (data_info, **args) 

  #### If transformer URI is Provided 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3686400/9912422 [00:00<00:00, 36838476.67it/s] 97%|█████████▋| 9641984/9912422 [00:00<00:00, 41038563.28it/s]9920512it [00:00, 32090111.57it/s]                             
0it [00:00, ?it/s]32768it [00:00, 923729.58it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 456651.31it/s]1654784it [00:00, 11668438.31it/s]                         
0it [00:00, ?it/s]8192it [00:00, 134821.79it/s]
torchvision_dataset_MNIST_load() missing 1 required positional argument: 'path'
dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f2225358fd0>, <torch.utils.data.dataloader.DataLoader object at 0x7f2223699748>), {})
