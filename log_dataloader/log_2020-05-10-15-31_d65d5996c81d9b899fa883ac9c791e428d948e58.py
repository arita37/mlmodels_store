
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_dataloader GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d65d5996c81d9b899fa883ac9c791e428d948e58', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/adata2', 'sha': 'd65d5996c81d9b899fa883ac9c791e428d948e58', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d65d5996c81d9b899fa883ac9c791e428d948e58

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d65d5996c81d9b899fa883ac9c791e428d948e58

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
['/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_textcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//namentity_crm_bilstm_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//charcnn.json']





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 

#####  Load JSON data_pars
{'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torchvision_dataset_MNIST_load', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True}}]}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels.preprocess.generic::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torchvision_dataset_MNIST_load', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5e32c26ea0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5e32c26ea0>

  function with postional parmater data_info >> get_dataset_torch(data_info, **args) 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███▏      | 3121152/9912422 [00:00<00:00, 31127539.62it/s]9920512it [00:00, 32996579.11it/s]                             
0it [00:00, ?it/s]32768it [00:00, 584158.05it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 159999.90it/s]1654784it [00:00, 11167498.22it/s]                         
0it [00:00, ?it/s]8192it [00:00, 155189.53it/s]
torchvision_dataset_MNIST_load() missing 1 required positional argument: 'path'
dset :  <class 'torchvision.datasets.mnist.MNIST'>
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
((<torch.utils.data.dataloader.DataLoader object at 0x7f5e32c28f28>, <torch.utils.data.dataloader.DataLoader object at 0x7f5e30f66748>), {})
