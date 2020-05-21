
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51225cb7afa33c64ee4aff9194b137745935d62b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '51225cb7afa33c64ee4aff9194b137745935d62b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51225cb7afa33c64ee4aff9194b137745935d62b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51225cb7afa33c64ee4aff9194b137745935d62b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51225cb7afa33c64ee4aff9194b137745935d62b

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  



###### Test_run_model  #############################################################



 ####################################################################################################
dataset/json/refactor/resnet18_benchmark_mnist.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 1,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/MNIST/",
        "dataset": "MNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels/preprocess/generic.py::get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:MNIST",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars": false,
              "arg": {}
            },
            "shuffle": true,
            "download": true
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/MNIST/restnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/MNIST/restnet18/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f7d43a15a90> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d429e10d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d429e10d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d429e10d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3129344/9912422 [00:00<00:00, 30909814.66it/s]9920512it [00:00, 34733721.02it/s]                             
0it [00:00, ?it/s]32768it [00:00, 595504.88it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 484890.41it/s]1654784it [00:00, 11722416.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186785.42it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0032516783475875855 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01859379529953003 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 2 	 Loss: 0.001762149840593338 	 Accuracy: 1
Train Epoch: 2 	 Loss: 0.02358200669288635 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.0023236317038536074 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.022027207493782043 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0014015940030415854 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.01087883648276329 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0016163067867358525 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.015753150641918182 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d429c3f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d429c3f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d429c3f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([1, 2, 3, 6, 3, 6, 7, 4, 6, 5]), array([5, 3, 3, 6, 3, 6, 7, 4, 5, 5]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/resnet34_benchmark_mnist.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet34",
      "num_classes": 10,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 1,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/MNIST/",
        "dataset": "MNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels/preprocess/generic.py::get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:MNIST",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars": false,
              "arg": {}
            },
            "shuffle": true,
            "download": true
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 5,
      "epochs": 2,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/MNIST/restnet34/checkpoints/",
      "path": "ztest/model_tch/torchhub/MNIST/restnet34/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f7d43a156a0> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d40137840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d40137840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d40137840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0022088311513264974 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.016648166418075563 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.001814894954363505 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03079499578475952 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d40151d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d40151d90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d40151d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([6, 3, 9, 4, 0, 9, 1, 6, 1, 1]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/model_list_CIFAR.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 1,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/cifar10/",
        "dataset": "CIFAR10",
        "data_type": "tf_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tf_dataset_start",
          "uri": "mlmodels/preprocess/generic.py::tf_dataset_download",
          "arg": {
            "train_samples": 2000,
            "test_samples": 500,
            "shuffle": true,
            "download": true
          }
        },
        {
          "uri": "mlmodels/preprocess/generic.py::get_dataset_torch",
          "args": {
            "dataloader": "mlmodels/preprocess/generic.py:NumpyDataset",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_generic",
              "pass_data_pars": false,
              "arg": {
                "fixed_size": 256
              }
            },
            "shuffle": true,
            "download": true
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 2,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/cifar10/resnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/cifar10/resnet18/"
    }
  },
  "resnet152": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet152",
      "num_classes": 10,
      "pretrained": 1,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "dataset": "torchvision.datasets:CIFAR",
      "data_path": "dataset/vision/",
      "train_batch_size": 100,
      "test_batch_size": 10,
      "transform_uri": "mlmodels.preprocess.image:torch_transform_data_augment"
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/resnet152/checkpoints/",
      "path": "ztest/model_tch/torchhub/resnet152/"
    }
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://download.pytorch.org/models/resnet18-5c106cde.pth" to /home/runner/.cache/torch/checkpoints/resnet18-5c106cde.pth
  0%|          | 0.00/44.7M [00:00<?, ?B/s]  6%|▌         | 2.62M/44.7M [00:00<00:01, 27.5MB/s] 12%|█▏        | 5.24M/44.7M [00:00<00:01, 27.5MB/s] 17%|█▋        | 7.77M/44.7M [00:00<00:01, 27.2MB/s] 21%|██        | 9.46M/44.7M [00:00<00:01, 18.9MB/s] 26%|██▌       | 11.6M/44.7M [00:00<00:01, 19.8MB/s] 31%|███       | 13.8M/44.7M [00:00<00:01, 20.5MB/s] 36%|███▌      | 16.1M/44.7M [00:00<00:01, 21.6MB/s] 42%|████▏     | 18.7M/44.7M [00:00<00:01, 22.9MB/s] 47%|████▋     | 21.1M/44.7M [00:00<00:01, 23.4MB/s] 52%|█████▏    | 23.4M/44.7M [00:01<00:00, 23.9MB/s] 58%|█████▊    | 25.9M/44.7M [00:01<00:00, 24.5MB/s] 63%|██████▎   | 28.3M/44.7M [00:01<00:00, 24.2MB/s] 69%|██████▉   | 31.0M/44.7M [00:01<00:00, 25.3MB/s] 75%|███████▍  | 33.4M/44.7M [00:01<00:00, 24.9MB/s] 80%|████████  | 35.8M/44.7M [00:01<00:00, 24.9MB/s] 86%|████████▌ | 38.4M/44.7M [00:01<00:00, 25.6MB/s] 92%|█████████▏| 40.9M/44.7M [00:01<00:00, 25.5MB/s] 97%|█████████▋| 43.4M/44.7M [00:01<00:00, 25.7MB/s]100%|██████████| 44.7M/44.7M [00:01<00:00, 24.3MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f7d43863ac8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7d429b1e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7d429b1e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f7d429b1e18> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:37,  1.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:37,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:36,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:35,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:35,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:34,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:34,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:33,  1.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:05,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:04,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:04,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:03,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:03,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:03,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:02,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:02,  2.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:43,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:43,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:43,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:42,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:42,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:42,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:41,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:41,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:41,  3.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:28,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:28,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:28,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:28,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:27,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:27,  4.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:27,  4.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:19,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:19,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:19,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:19,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:18,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:18,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:18,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:18,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:18,  6.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:13,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:13,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:12,  8.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:08, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 16.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 21.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 27.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 27.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 34.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 46.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 46.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 53.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 58.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 64.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.67s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.67s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.67s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.60 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.67s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 70.60 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.31 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.18s/ url]
0 examples [00:00, ? examples/s]2020-05-21 14:30:17.730037: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 14:30:17.757919: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-21 14:30:17.758128: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562a2d2336b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 14:30:17.758148: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  8.24 examples/s]88 examples [00:00, 11.72 examples/s]181 examples [00:00, 16.65 examples/s]273 examples [00:00, 23.61 examples/s]367 examples [00:00, 33.36 examples/s]460 examples [00:00, 46.94 examples/s]556 examples [00:00, 65.67 examples/s]639 examples [00:00, 90.68 examples/s]732 examples [00:00, 124.31 examples/s]818 examples [00:01, 166.55 examples/s]913 examples [00:01, 221.21 examples/s]1008 examples [00:01, 287.24 examples/s]1098 examples [00:01, 359.66 examples/s]1187 examples [00:01, 433.84 examples/s]1275 examples [00:01, 501.32 examples/s]1361 examples [00:01, 571.81 examples/s]1451 examples [00:01, 641.70 examples/s]1538 examples [00:01, 664.05 examples/s]1622 examples [00:01, 707.95 examples/s]1706 examples [00:02, 742.38 examples/s]1791 examples [00:02, 771.04 examples/s]1875 examples [00:02, 784.43 examples/s]1963 examples [00:02, 810.14 examples/s]2056 examples [00:02, 842.15 examples/s]2149 examples [00:02, 864.60 examples/s]2241 examples [00:02, 880.45 examples/s]2333 examples [00:02, 889.85 examples/s]2427 examples [00:02, 902.68 examples/s]2519 examples [00:02, 905.70 examples/s]2611 examples [00:03, 864.41 examples/s]2703 examples [00:03, 879.16 examples/s]2797 examples [00:03, 894.82 examples/s]2891 examples [00:03, 905.59 examples/s]2985 examples [00:03, 912.13 examples/s]3077 examples [00:03, 906.52 examples/s]3170 examples [00:03, 911.69 examples/s]3262 examples [00:03, 911.73 examples/s]3354 examples [00:03, 849.55 examples/s]3442 examples [00:04, 857.56 examples/s]3529 examples [00:04, 861.02 examples/s]3622 examples [00:04, 878.64 examples/s]3711 examples [00:04, 878.83 examples/s]3802 examples [00:04, 887.45 examples/s]3893 examples [00:04, 893.81 examples/s]3983 examples [00:04, 869.12 examples/s]4071 examples [00:04, 866.18 examples/s]4160 examples [00:04, 873.01 examples/s]4248 examples [00:04, 849.27 examples/s]4334 examples [00:05, 837.83 examples/s]4426 examples [00:05, 860.50 examples/s]4515 examples [00:05, 868.70 examples/s]4608 examples [00:05, 885.53 examples/s]4703 examples [00:05, 901.08 examples/s]4795 examples [00:05, 905.06 examples/s]4887 examples [00:05, 909.01 examples/s]4979 examples [00:05, 907.85 examples/s]5071 examples [00:05, 910.08 examples/s]5163 examples [00:05, 908.79 examples/s]5255 examples [00:06, 909.51 examples/s]5347 examples [00:06, 912.04 examples/s]5439 examples [00:06, 907.31 examples/s]5530 examples [00:06, 893.50 examples/s]5621 examples [00:06, 897.26 examples/s]5711 examples [00:06, 896.94 examples/s]5804 examples [00:06, 905.73 examples/s]5896 examples [00:06, 909.84 examples/s]5988 examples [00:06, 912.63 examples/s]6080 examples [00:06, 884.55 examples/s]6169 examples [00:07, 883.46 examples/s]6261 examples [00:07, 891.73 examples/s]6357 examples [00:07, 908.72 examples/s]6449 examples [00:07, 912.02 examples/s]6542 examples [00:07, 914.36 examples/s]6634 examples [00:07, 902.60 examples/s]6728 examples [00:07, 912.39 examples/s]6820 examples [00:07, 885.83 examples/s]6909 examples [00:07, 884.75 examples/s]7002 examples [00:08, 896.95 examples/s]7094 examples [00:08, 903.42 examples/s]7188 examples [00:08, 913.49 examples/s]7281 examples [00:08, 917.61 examples/s]7374 examples [00:08, 920.00 examples/s]7467 examples [00:08, 911.14 examples/s]7560 examples [00:08, 915.44 examples/s]7655 examples [00:08, 923.22 examples/s]7748 examples [00:08, 924.40 examples/s]7843 examples [00:08, 931.25 examples/s]7937 examples [00:09, 916.30 examples/s]8029 examples [00:09, 865.88 examples/s]8122 examples [00:09, 881.75 examples/s]8211 examples [00:09, 878.29 examples/s]8300 examples [00:09, 857.76 examples/s]8387 examples [00:09, 854.98 examples/s]8473 examples [00:09, 854.02 examples/s]8559 examples [00:09, 847.44 examples/s]8648 examples [00:09, 857.86 examples/s]8736 examples [00:09, 863.24 examples/s]8823 examples [00:10, 828.19 examples/s]8907 examples [00:10, 820.17 examples/s]9000 examples [00:10, 849.11 examples/s]9093 examples [00:10, 870.76 examples/s]9187 examples [00:10, 887.94 examples/s]9281 examples [00:10, 900.97 examples/s]9373 examples [00:10, 906.15 examples/s]9465 examples [00:10, 908.86 examples/s]9557 examples [00:10, 906.91 examples/s]9648 examples [00:10, 903.56 examples/s]9739 examples [00:11, 891.11 examples/s]9832 examples [00:11, 902.00 examples/s]9926 examples [00:11, 912.68 examples/s]10018 examples [00:11, 856.64 examples/s]10112 examples [00:11, 877.57 examples/s]10204 examples [00:11, 889.06 examples/s]10297 examples [00:11, 899.99 examples/s]10389 examples [00:11, 905.57 examples/s]10481 examples [00:11, 909.75 examples/s]10575 examples [00:12, 916.06 examples/s]10667 examples [00:12, 912.75 examples/s]10760 examples [00:12, 916.67 examples/s]10853 examples [00:12, 920.12 examples/s]10946 examples [00:12, 922.84 examples/s]11039 examples [00:12, 921.29 examples/s]11132 examples [00:12, 914.77 examples/s]11224 examples [00:12, 882.79 examples/s]11313 examples [00:12, 878.08 examples/s]11401 examples [00:12, 873.42 examples/s]11489 examples [00:13, 828.20 examples/s]11578 examples [00:13, 844.25 examples/s]11664 examples [00:13, 846.73 examples/s]11757 examples [00:13, 868.90 examples/s]11850 examples [00:13, 885.59 examples/s]11943 examples [00:13, 898.05 examples/s]12035 examples [00:13, 903.06 examples/s]12127 examples [00:13, 907.67 examples/s]12220 examples [00:13, 912.71 examples/s]12313 examples [00:13, 915.63 examples/s]12406 examples [00:14, 919.04 examples/s]12498 examples [00:14, 911.81 examples/s]12590 examples [00:14, 894.09 examples/s]12680 examples [00:14, 887.66 examples/s]12769 examples [00:14, 885.42 examples/s]12858 examples [00:14, 883.39 examples/s]12947 examples [00:14, 872.22 examples/s]13035 examples [00:14, 874.11 examples/s]13123 examples [00:14, 868.91 examples/s]13212 examples [00:15, 874.79 examples/s]13301 examples [00:15, 876.71 examples/s]13389 examples [00:15, 869.08 examples/s]13476 examples [00:15, 838.58 examples/s]13570 examples [00:15, 865.12 examples/s]13666 examples [00:15, 890.58 examples/s]13760 examples [00:15, 902.45 examples/s]13851 examples [00:15, 901.82 examples/s]13942 examples [00:15, 901.94 examples/s]14036 examples [00:15, 912.06 examples/s]14129 examples [00:16, 916.87 examples/s]14221 examples [00:16, 906.69 examples/s]14313 examples [00:16, 908.84 examples/s]14409 examples [00:16, 920.92 examples/s]14504 examples [00:16, 926.90 examples/s]14597 examples [00:16, 925.45 examples/s]14691 examples [00:16, 928.15 examples/s]14784 examples [00:16, 917.86 examples/s]14879 examples [00:16, 926.19 examples/s]14976 examples [00:16, 938.12 examples/s]15071 examples [00:17, 940.19 examples/s]15168 examples [00:17, 947.04 examples/s]15263 examples [00:17, 942.33 examples/s]15361 examples [00:17, 951.53 examples/s]15457 examples [00:17, 941.14 examples/s]15556 examples [00:17, 954.23 examples/s]15655 examples [00:17, 963.27 examples/s]15752 examples [00:17, 940.89 examples/s]15847 examples [00:17, 927.53 examples/s]15945 examples [00:17, 941.53 examples/s]16044 examples [00:18, 955.15 examples/s]16140 examples [00:18, 948.45 examples/s]16235 examples [00:18, 946.86 examples/s]16330 examples [00:18, 935.92 examples/s]16424 examples [00:18, 936.05 examples/s]16518 examples [00:18, 925.01 examples/s]16611 examples [00:18, 918.34 examples/s]16703 examples [00:18, 877.96 examples/s]16797 examples [00:18, 895.39 examples/s]16891 examples [00:18, 907.30 examples/s]16983 examples [00:19, 909.72 examples/s]17075 examples [00:19, 874.59 examples/s]17165 examples [00:19, 879.12 examples/s]17257 examples [00:19, 889.39 examples/s]17350 examples [00:19, 900.35 examples/s]17445 examples [00:19, 913.77 examples/s]17539 examples [00:19, 919.70 examples/s]17634 examples [00:19, 927.27 examples/s]17731 examples [00:19, 937.02 examples/s]17828 examples [00:20, 945.31 examples/s]17928 examples [00:20, 958.64 examples/s]18025 examples [00:20, 959.61 examples/s]18122 examples [00:20, 923.72 examples/s]18215 examples [00:20, 924.82 examples/s]18309 examples [00:20, 927.12 examples/s]18404 examples [00:20, 933.62 examples/s]18498 examples [00:20, 912.33 examples/s]18592 examples [00:20, 918.29 examples/s]18691 examples [00:20, 937.25 examples/s]18787 examples [00:21, 943.18 examples/s]18886 examples [00:21, 954.74 examples/s]18982 examples [00:21, 926.18 examples/s]19075 examples [00:21, 919.81 examples/s]19168 examples [00:21, 916.82 examples/s]19261 examples [00:21, 920.73 examples/s]19354 examples [00:21, 919.47 examples/s]19447 examples [00:21, 919.03 examples/s]19539 examples [00:21, 915.30 examples/s]19631 examples [00:21, 913.20 examples/s]19723 examples [00:22, 912.45 examples/s]19815 examples [00:22, 906.24 examples/s]19906 examples [00:22, 892.01 examples/s]20001 examples [00:22, 854.06 examples/s]20099 examples [00:22, 885.95 examples/s]20197 examples [00:22, 910.46 examples/s]20294 examples [00:22, 926.44 examples/s]20388 examples [00:22, 885.59 examples/s]20481 examples [00:22, 896.78 examples/s]20575 examples [00:23, 908.53 examples/s]20670 examples [00:23, 918.44 examples/s]20766 examples [00:23, 928.49 examples/s]20861 examples [00:23, 934.07 examples/s]20955 examples [00:23, 932.21 examples/s]21049 examples [00:23, 930.76 examples/s]21143 examples [00:23, 917.34 examples/s]21235 examples [00:23, 885.32 examples/s]21324 examples [00:23, 875.97 examples/s]21417 examples [00:23, 891.47 examples/s]21509 examples [00:24, 897.40 examples/s]21604 examples [00:24, 910.03 examples/s]21698 examples [00:24, 918.28 examples/s]21790 examples [00:24, 914.54 examples/s]21887 examples [00:24, 929.03 examples/s]21985 examples [00:24, 941.69 examples/s]22081 examples [00:24, 945.55 examples/s]22177 examples [00:24, 948.27 examples/s]22272 examples [00:24, 929.35 examples/s]22366 examples [00:24, 924.48 examples/s]22459 examples [00:25, 914.90 examples/s]22551 examples [00:25, 910.03 examples/s]22643 examples [00:25, 900.05 examples/s]22734 examples [00:25, 851.54 examples/s]22831 examples [00:25, 883.33 examples/s]22925 examples [00:25, 898.21 examples/s]23016 examples [00:25, 894.27 examples/s]23109 examples [00:25, 904.32 examples/s]23200 examples [00:25, 900.13 examples/s]23293 examples [00:25, 907.45 examples/s]23387 examples [00:26, 916.23 examples/s]23484 examples [00:26, 929.58 examples/s]23581 examples [00:26, 940.33 examples/s]23676 examples [00:26, 940.47 examples/s]23771 examples [00:26, 939.50 examples/s]23866 examples [00:26, 929.97 examples/s]23960 examples [00:26, 930.41 examples/s]24055 examples [00:26, 935.57 examples/s]24149 examples [00:26, 923.05 examples/s]24242 examples [00:27, 912.07 examples/s]24337 examples [00:27, 921.46 examples/s]24434 examples [00:27, 934.09 examples/s]24528 examples [00:27, 929.96 examples/s]24622 examples [00:27, 926.86 examples/s]24716 examples [00:27, 929.48 examples/s]24809 examples [00:27, 923.57 examples/s]24903 examples [00:27, 926.12 examples/s]24996 examples [00:27, 902.99 examples/s]25087 examples [00:27, 873.50 examples/s]25175 examples [00:28, 875.18 examples/s]25269 examples [00:28, 893.30 examples/s]25365 examples [00:28, 910.39 examples/s]25457 examples [00:28, 911.66 examples/s]25549 examples [00:28, 897.13 examples/s]25639 examples [00:28, 896.76 examples/s]25735 examples [00:28, 912.26 examples/s]25830 examples [00:28, 921.29 examples/s]25925 examples [00:28, 928.03 examples/s]26018 examples [00:28, 909.03 examples/s]26115 examples [00:29, 924.14 examples/s]26210 examples [00:29, 929.19 examples/s]26310 examples [00:29, 947.09 examples/s]26405 examples [00:29, 942.74 examples/s]26500 examples [00:29, 935.72 examples/s]26595 examples [00:29, 939.04 examples/s]26689 examples [00:29, 929.78 examples/s]26785 examples [00:29, 938.46 examples/s]26879 examples [00:29, 938.50 examples/s]26973 examples [00:29, 923.53 examples/s]27066 examples [00:30, 899.98 examples/s]27157 examples [00:30, 887.04 examples/s]27246 examples [00:30, 859.40 examples/s]27333 examples [00:30, 830.49 examples/s]27419 examples [00:30, 837.05 examples/s]27505 examples [00:30, 842.05 examples/s]27591 examples [00:30, 846.00 examples/s]27676 examples [00:30, 846.55 examples/s]27761 examples [00:30, 846.24 examples/s]27847 examples [00:31, 848.45 examples/s]27932 examples [00:31, 847.95 examples/s]28017 examples [00:31, 845.66 examples/s]28109 examples [00:31, 864.75 examples/s]28196 examples [00:31, 858.75 examples/s]28284 examples [00:31, 862.31 examples/s]28378 examples [00:31, 881.93 examples/s]28467 examples [00:31, 875.15 examples/s]28558 examples [00:31, 882.80 examples/s]28647 examples [00:31, 879.81 examples/s]28736 examples [00:32, 875.09 examples/s]28824 examples [00:32, 868.99 examples/s]28912 examples [00:32, 871.79 examples/s]29002 examples [00:32, 879.05 examples/s]29090 examples [00:32, 875.55 examples/s]29178 examples [00:32, 872.17 examples/s]29267 examples [00:32, 875.19 examples/s]29355 examples [00:32, 870.42 examples/s]29444 examples [00:32, 874.29 examples/s]29532 examples [00:32, 873.86 examples/s]29620 examples [00:33, 849.95 examples/s]29713 examples [00:33, 870.93 examples/s]29801 examples [00:33, 861.98 examples/s]29888 examples [00:33, 852.94 examples/s]29974 examples [00:33, 855.01 examples/s]30060 examples [00:33, 810.99 examples/s]30152 examples [00:33, 838.58 examples/s]30241 examples [00:33, 851.16 examples/s]30328 examples [00:33, 855.79 examples/s]30417 examples [00:33, 864.65 examples/s]30505 examples [00:34, 867.80 examples/s]30595 examples [00:34, 872.60 examples/s]30684 examples [00:34, 875.05 examples/s]30773 examples [00:34, 879.07 examples/s]30861 examples [00:34, 877.33 examples/s]30949 examples [00:34, 875.99 examples/s]31037 examples [00:34, 865.92 examples/s]31130 examples [00:34, 881.34 examples/s]31221 examples [00:34, 887.38 examples/s]31310 examples [00:35, 868.96 examples/s]31404 examples [00:35, 886.85 examples/s]31493 examples [00:35, 876.25 examples/s]31583 examples [00:35, 881.28 examples/s]31675 examples [00:35, 891.26 examples/s]31767 examples [00:35, 898.10 examples/s]31860 examples [00:35, 904.83 examples/s]31951 examples [00:35, 883.41 examples/s]32044 examples [00:35, 895.35 examples/s]32137 examples [00:35, 902.79 examples/s]32228 examples [00:36, 887.54 examples/s]32319 examples [00:36, 891.80 examples/s]32411 examples [00:36, 899.30 examples/s]32505 examples [00:36, 908.99 examples/s]32599 examples [00:36, 915.82 examples/s]32691 examples [00:36, 898.32 examples/s]32781 examples [00:36, 881.62 examples/s]32870 examples [00:36, 879.48 examples/s]32959 examples [00:36, 873.55 examples/s]33047 examples [00:36, 865.91 examples/s]33134 examples [00:37, 863.44 examples/s]33223 examples [00:37, 869.27 examples/s]33311 examples [00:37, 870.92 examples/s]33399 examples [00:37, 871.41 examples/s]33488 examples [00:37, 875.25 examples/s]33576 examples [00:37, 875.55 examples/s]33665 examples [00:37, 877.40 examples/s]33753 examples [00:37, 863.02 examples/s]33847 examples [00:37, 883.16 examples/s]33941 examples [00:37, 896.65 examples/s]34031 examples [00:38, 896.46 examples/s]34124 examples [00:38, 904.15 examples/s]34215 examples [00:38, 899.02 examples/s]34310 examples [00:38, 911.74 examples/s]34404 examples [00:38, 917.51 examples/s]34496 examples [00:38, 915.20 examples/s]34588 examples [00:38, 904.67 examples/s]34680 examples [00:38, 906.44 examples/s]34771 examples [00:38, 906.53 examples/s]34862 examples [00:38, 906.37 examples/s]34953 examples [00:39, 886.79 examples/s]35045 examples [00:39, 895.32 examples/s]35135 examples [00:39, 884.53 examples/s]35224 examples [00:39, 882.89 examples/s]35313 examples [00:39, 881.62 examples/s]35402 examples [00:39, 878.46 examples/s]35490 examples [00:39, 870.61 examples/s]35578 examples [00:39, 864.91 examples/s]35666 examples [00:39, 868.84 examples/s]35753 examples [00:40, 867.41 examples/s]35840 examples [00:40, 864.79 examples/s]35930 examples [00:40, 874.38 examples/s]36020 examples [00:40, 879.88 examples/s]36115 examples [00:40, 898.51 examples/s]36209 examples [00:40, 908.45 examples/s]36301 examples [00:40, 909.04 examples/s]36396 examples [00:40, 920.58 examples/s]36489 examples [00:40, 872.57 examples/s]36577 examples [00:40, 849.11 examples/s]36667 examples [00:41, 862.60 examples/s]36754 examples [00:41, 861.29 examples/s]36842 examples [00:41, 866.10 examples/s]36929 examples [00:41, 854.69 examples/s]37016 examples [00:41, 858.63 examples/s]37104 examples [00:41, 862.25 examples/s]37193 examples [00:41, 868.96 examples/s]37281 examples [00:41, 872.10 examples/s]37369 examples [00:41, 872.41 examples/s]37457 examples [00:41, 871.55 examples/s]37545 examples [00:42, 873.50 examples/s]37633 examples [00:42, 867.87 examples/s]37723 examples [00:42, 875.56 examples/s]37812 examples [00:42, 878.62 examples/s]37905 examples [00:42, 893.16 examples/s]38000 examples [00:42, 907.13 examples/s]38092 examples [00:42, 910.89 examples/s]38187 examples [00:42, 920.90 examples/s]38280 examples [00:42, 922.42 examples/s]38373 examples [00:42, 881.66 examples/s]38464 examples [00:43, 887.24 examples/s]38554 examples [00:43, 888.14 examples/s]38648 examples [00:43, 900.30 examples/s]38741 examples [00:43, 906.65 examples/s]38834 examples [00:43, 911.17 examples/s]38928 examples [00:43, 917.81 examples/s]39021 examples [00:43, 920.07 examples/s]39114 examples [00:43, 922.46 examples/s]39207 examples [00:43, 886.37 examples/s]39297 examples [00:44, 889.22 examples/s]39391 examples [00:44, 901.20 examples/s]39482 examples [00:44, 899.27 examples/s]39575 examples [00:44, 905.77 examples/s]39666 examples [00:44, 905.76 examples/s]39757 examples [00:44, 906.61 examples/s]39848 examples [00:44, 898.10 examples/s]39938 examples [00:44, 877.61 examples/s]40026 examples [00:44, 795.18 examples/s]40114 examples [00:44, 817.84 examples/s]40203 examples [00:45, 836.66 examples/s]40291 examples [00:45, 847.18 examples/s]40380 examples [00:45, 857.75 examples/s]40469 examples [00:45, 864.73 examples/s]40556 examples [00:45, 862.89 examples/s]40645 examples [00:45, 868.85 examples/s]40733 examples [00:45, 870.12 examples/s]40821 examples [00:45, 870.74 examples/s]40910 examples [00:45, 874.05 examples/s]40998 examples [00:45, 875.69 examples/s]41086 examples [00:46, 868.85 examples/s]41173 examples [00:46, 811.19 examples/s]41261 examples [00:46, 829.62 examples/s]41348 examples [00:46, 839.61 examples/s]41435 examples [00:46, 846.93 examples/s]41523 examples [00:46, 854.87 examples/s]41609 examples [00:46, 855.98 examples/s]41695 examples [00:46, 855.69 examples/s]41781 examples [00:46, 854.82 examples/s]41870 examples [00:46, 862.80 examples/s]41963 examples [00:47, 880.14 examples/s]42055 examples [00:47, 889.63 examples/s]42146 examples [00:47, 892.75 examples/s]42236 examples [00:47, 883.45 examples/s]42325 examples [00:47, 876.74 examples/s]42413 examples [00:47, 861.47 examples/s]42500 examples [00:47, 856.35 examples/s]42586 examples [00:47, 853.54 examples/s]42672 examples [00:47, 852.30 examples/s]42760 examples [00:48, 859.96 examples/s]42848 examples [00:48, 864.08 examples/s]42937 examples [00:48, 868.98 examples/s]43024 examples [00:48, 868.53 examples/s]43117 examples [00:48, 884.54 examples/s]43211 examples [00:48, 900.42 examples/s]43302 examples [00:48, 896.58 examples/s]43394 examples [00:48, 901.87 examples/s]43485 examples [00:48, 888.77 examples/s]43577 examples [00:48, 897.88 examples/s]43668 examples [00:49, 899.02 examples/s]43760 examples [00:49, 905.14 examples/s]43852 examples [00:49, 907.50 examples/s]43943 examples [00:49, 901.25 examples/s]44034 examples [00:49, 899.89 examples/s]44125 examples [00:49, 902.54 examples/s]44218 examples [00:49, 908.04 examples/s]44309 examples [00:49, 907.08 examples/s]44400 examples [00:49, 896.70 examples/s]44490 examples [00:49, 897.63 examples/s]44580 examples [00:50, 895.01 examples/s]44672 examples [00:50, 900.48 examples/s]44763 examples [00:50, 901.29 examples/s]44854 examples [00:50, 889.83 examples/s]44944 examples [00:50, 890.37 examples/s]45034 examples [00:50, 892.02 examples/s]45125 examples [00:50, 896.94 examples/s]45216 examples [00:50, 899.83 examples/s]45308 examples [00:50, 902.91 examples/s]45399 examples [00:50, 892.48 examples/s]45489 examples [00:51, 887.75 examples/s]45578 examples [00:51, 879.89 examples/s]45667 examples [00:51, 880.39 examples/s]45756 examples [00:51, 865.52 examples/s]45848 examples [00:51, 880.94 examples/s]45940 examples [00:51, 891.25 examples/s]46030 examples [00:51, 880.73 examples/s]46119 examples [00:51, 868.29 examples/s]46209 examples [00:51, 875.13 examples/s]46297 examples [00:51, 873.28 examples/s]46385 examples [00:52, 871.85 examples/s]46473 examples [00:52, 865.89 examples/s]46560 examples [00:52, 863.00 examples/s]46647 examples [00:52, 863.38 examples/s]46734 examples [00:52, 853.48 examples/s]46820 examples [00:52, 830.17 examples/s]46904 examples [00:52, 828.22 examples/s]46987 examples [00:52, 824.45 examples/s]47074 examples [00:52, 837.33 examples/s]47162 examples [00:53, 848.04 examples/s]47247 examples [00:53, 839.02 examples/s]47339 examples [00:53, 860.47 examples/s]47430 examples [00:53, 874.42 examples/s]47522 examples [00:53, 887.24 examples/s]47615 examples [00:53, 897.35 examples/s]47706 examples [00:53, 901.10 examples/s]47799 examples [00:53, 906.93 examples/s]47891 examples [00:53, 909.91 examples/s]47983 examples [00:53, 908.48 examples/s]48074 examples [00:54, 880.69 examples/s]48165 examples [00:54, 886.99 examples/s]48254 examples [00:54, 876.50 examples/s]48342 examples [00:54, 870.23 examples/s]48430 examples [00:54, 867.45 examples/s]48517 examples [00:54, 865.88 examples/s]48605 examples [00:54, 868.43 examples/s]48692 examples [00:54, 867.54 examples/s]48779 examples [00:54, 862.52 examples/s]48867 examples [00:54, 865.79 examples/s]48954 examples [00:55, 865.21 examples/s]49043 examples [00:55, 870.06 examples/s]49131 examples [00:55, 870.14 examples/s]49219 examples [00:55, 861.91 examples/s]49309 examples [00:55, 870.89 examples/s]49401 examples [00:55, 884.49 examples/s]49494 examples [00:55, 895.25 examples/s]49584 examples [00:55, 878.16 examples/s]49677 examples [00:55, 892.73 examples/s]49769 examples [00:55, 898.85 examples/s]49862 examples [00:56, 906.59 examples/s]49953 examples [00:56, 874.36 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5274/50000 [00:00<00:00, 52739.67 examples/s] 32%|███▏      | 15820/50000 [00:00<00:00, 62044.21 examples/s] 53%|█████▎    | 26497/50000 [00:00<00:00, 70961.07 examples/s] 75%|███████▍  | 37331/50000 [00:00<00:00, 79153.62 examples/s] 97%|█████████▋| 48330/50000 [00:00<00:00, 86422.30 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 734.03 examples/s]166 examples [00:00, 779.94 examples/s]259 examples [00:00, 817.95 examples/s]350 examples [00:00, 843.50 examples/s]439 examples [00:00, 856.25 examples/s]525 examples [00:00, 854.62 examples/s]613 examples [00:00, 859.49 examples/s]703 examples [00:00, 868.74 examples/s]786 examples [00:00, 825.56 examples/s]880 examples [00:01, 856.39 examples/s]973 examples [00:01, 875.18 examples/s]1062 examples [00:01, 879.50 examples/s]1156 examples [00:01, 894.67 examples/s]1246 examples [00:01, 883.17 examples/s]1335 examples [00:01, 873.39 examples/s]1425 examples [00:01, 879.14 examples/s]1521 examples [00:01, 901.85 examples/s]1614 examples [00:01, 908.92 examples/s]1705 examples [00:01, 897.13 examples/s]1795 examples [00:02, 893.64 examples/s]1885 examples [00:02, 885.90 examples/s]1977 examples [00:02, 894.83 examples/s]2067 examples [00:02, 894.31 examples/s]2157 examples [00:02, 876.22 examples/s]2250 examples [00:02, 889.91 examples/s]2342 examples [00:02, 897.51 examples/s]2436 examples [00:02, 908.13 examples/s]2530 examples [00:02, 915.12 examples/s]2622 examples [00:02, 909.51 examples/s]2717 examples [00:03, 920.86 examples/s]2810 examples [00:03, 919.74 examples/s]2903 examples [00:03, 922.50 examples/s]2997 examples [00:03, 925.86 examples/s]3092 examples [00:03, 931.25 examples/s]3186 examples [00:03, 918.49 examples/s]3278 examples [00:03, 889.91 examples/s]3368 examples [00:03, 875.45 examples/s]3456 examples [00:03, 871.84 examples/s]3545 examples [00:03, 877.16 examples/s]3634 examples [00:04, 878.81 examples/s]3722 examples [00:04, 878.13 examples/s]3810 examples [00:04, 878.10 examples/s]3899 examples [00:04, 879.91 examples/s]3988 examples [00:04, 878.84 examples/s]4076 examples [00:04, 877.85 examples/s]4164 examples [00:04, 854.51 examples/s]4256 examples [00:04, 871.52 examples/s]4351 examples [00:04, 892.60 examples/s]4441 examples [00:04, 890.14 examples/s]4535 examples [00:05, 903.41 examples/s]4626 examples [00:05, 904.90 examples/s]4721 examples [00:05, 915.28 examples/s]4813 examples [00:05, 899.05 examples/s]4904 examples [00:05, 893.84 examples/s]4996 examples [00:05, 899.21 examples/s]5087 examples [00:05, 889.40 examples/s]5177 examples [00:05, 889.39 examples/s]5266 examples [00:05, 887.33 examples/s]5355 examples [00:06, 882.15 examples/s]5444 examples [00:06, 883.06 examples/s]5533 examples [00:06, 878.78 examples/s]5621 examples [00:06, 877.90 examples/s]5709 examples [00:06, 872.86 examples/s]5798 examples [00:06, 875.31 examples/s]5887 examples [00:06, 879.08 examples/s]5975 examples [00:06, 876.82 examples/s]6063 examples [00:06, 875.10 examples/s]6153 examples [00:06, 880.87 examples/s]6242 examples [00:07, 880.12 examples/s]6331 examples [00:07, 882.85 examples/s]6420 examples [00:07, 852.18 examples/s]6509 examples [00:07, 862.56 examples/s]6598 examples [00:07, 869.31 examples/s]6686 examples [00:07, 866.94 examples/s]6773 examples [00:07, 864.94 examples/s]6860 examples [00:07, 861.77 examples/s]6947 examples [00:07, 843.30 examples/s]7036 examples [00:07, 855.73 examples/s]7123 examples [00:08, 859.55 examples/s]7215 examples [00:08, 875.19 examples/s]7303 examples [00:08, 876.14 examples/s]7393 examples [00:08, 881.06 examples/s]7482 examples [00:08, 883.71 examples/s]7571 examples [00:08, 862.62 examples/s]7658 examples [00:08, 853.89 examples/s]7745 examples [00:08, 858.57 examples/s]7834 examples [00:08, 867.59 examples/s]7923 examples [00:08, 873.14 examples/s]8012 examples [00:09, 876.40 examples/s]8101 examples [00:09, 878.53 examples/s]8189 examples [00:09, 874.61 examples/s]8279 examples [00:09, 881.57 examples/s]8369 examples [00:09, 884.69 examples/s]8458 examples [00:09, 881.81 examples/s]8550 examples [00:09, 892.51 examples/s]8642 examples [00:09, 899.00 examples/s]8732 examples [00:09, 864.00 examples/s]8820 examples [00:09, 867.72 examples/s]8910 examples [00:10, 876.66 examples/s]8998 examples [00:10, 874.95 examples/s]9086 examples [00:10, 874.95 examples/s]9175 examples [00:10, 878.79 examples/s]9264 examples [00:10, 880.00 examples/s]9353 examples [00:10, 881.66 examples/s]9442 examples [00:10, 879.46 examples/s]9530 examples [00:10, 874.08 examples/s]9620 examples [00:10, 879.27 examples/s]9708 examples [00:11, 877.22 examples/s]9797 examples [00:11, 878.13 examples/s]9885 examples [00:11, 876.94 examples/s]9973 examples [00:11, 875.95 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKB7V1G/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKB7V1G/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d121818c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d121818c8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d121818c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.45359819412231445 	 Accuracy: 42
Train Epoch: 1 	 Loss: 5.670539474487304 	 Accuracy: 260
model saves at 260 accuracy
Train Epoch: 2 	 Loss: 0.4484265828132629 	 Accuracy: 56
Train Epoch: 2 	 Loss: 14.71997013092041 	 Accuracy: 160

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7cde6dd6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7cde6dd6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f7cde6dd6a8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cde6dd620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cde6dd620>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cde6dd620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([5, 1, 5, 1, 1, 8, 8, 1, 1, 1]), array([0, 3, 7, 2, 1, 8, 0, 1, 3, 8]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/torchhub_cnn_dataloader.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub",
      "repo_uri": "facebookresearch/pytorch_GAN_zoo:hub",
      "model": "PGAN",
      "model_name": "celebAHQ-512",
      "num_classes": 1000,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 1,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/MNIST",
        "dataset": "MNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels.preprocess.generic::get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:MNIST",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars": false,
              "arg": {
                "fixed_size": 256,
                "path": "dataset/vision/MNIST/"
              }
            },
            "shuffle": true,
            "download": true
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 5,
      "epochs": 2,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/pytorch_GAN_zoo/checkpoints/",
      "path": "ztest/model_tch/torchhub/pytorch_GAN_zoo/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f7cdd3b9278> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
img_01.png
0

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/resnet18_benchmark_FashionMNIST.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 0,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/FashionMNIST",
        "dataset": "FashionMNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels/preprocess/generic.py::get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:FashionMNIST",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars": false,
              "arg": {}
            },
            "shuffle": true,
            "download": true
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/FashionMNIST/resnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/FashionMNIST/resnet18/"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f7cdd3b9278> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdadf70d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdadf70d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdadf70d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:37, 270831.68it/s]  0%|          | 131072/26421880 [00:00<01:18, 333841.01it/s]  2%|▏         | 425984/26421880 [00:00<00:58, 442525.57it/s]  4%|▍         | 1081344/26421880 [00:00<00:41, 612333.36it/s] 10%|▉         | 2523136/26421880 [00:00<00:27, 859122.65it/s] 25%|██▍       | 6479872/26421880 [00:01<00:16, 1210273.71it/s] 37%|███▋      | 9830400/26421880 [00:01<00:09, 1702541.10it/s] 51%|█████▏    | 13549568/26421880 [00:01<00:05, 2380815.03it/s] 65%|██████▍   | 17047552/26421880 [00:01<00:02, 3304727.19it/s] 80%|████████  | 21200896/26421880 [00:01<00:01, 4565329.58it/s] 94%|█████████▎| 24739840/26421880 [00:01<00:00, 5742606.12it/s]26427392it [00:01, 14790211.58it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 111414.62it/s]           
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 264017.29it/s]  4%|▍         | 180224/4422102 [00:00<00:12, 341057.66it/s] 11%|█         | 466944/4422102 [00:00<00:08, 451189.99it/s] 34%|███▍      | 1499136/4422102 [00:00<00:04, 630057.82it/s] 77%|███████▋  | 3407872/4422102 [00:01<00:01, 887525.63it/s]4423680it [00:01, 3965763.91it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 39358.28it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0034001386364301044 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02016909670829773 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0023676666220029197 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02072322505712509 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.002559281011422475 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.024793480038642884 	 Accuracy: 4
Train Epoch: 4 	 Loss: 0.0020785045723120373 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.017684122204780578 	 Accuracy: 4
Train Epoch: 5 	 Loss: 0.0024406794011592866 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.011442890167236328 	 Accuracy: 5
model saves at 5 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdadee730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdadee730>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdadee730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([8, 9, 9, 7, 5, 6, 0, 7, 7, 3]), array([8, 9, 9, 7, 5, 0, 3, 7, 8, 4]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/model_list_KMNIST.json
{
  "test": {
    "hypermodel_pars": {
      "learning_rate": {
        "type": "log_uniform",
        "init": 0.01,
        "range": [
          0.001,
          0.1
        ]
      }
    },
    "model_pars": {
      "model_uri": "model_tch.torchhub.py",
      "repo_uri": "pytorch/vision",
      "model": "resnet18",
      "num_classes": 10,
      "pretrained": 1,
      "_comment": "0: False, 1: True",
      "num_layers": 5,
      "size": 6,
      "size_layer": 128,
      "output_size": 6,
      "timestep": 4,
      "epoch": 2
    },
    "data_pars": {
      "data_info": {
        "data_path": "dataset/vision/KMNIST",
        "dataset": "KMNIST",
        "data_type": "tch_dataset",
        "batch_size": 10,
        "train": true
      },
      "preprocessors": [
        {
          "name": "tch_dataset_start",
          "uri": "mlmodels/preprocess/generic.py::get_dataset_torch",
          "args": {
            "dataloader": "torchvision.datasets:KMNIST",
            "to_image": true,
            "transform": {
              "uri": "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars": false,
              "arg": {}
            },
            "shuffle": true,
            "download": true,
            "_comment": "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "
          }
        }
      ]
    },
    "compute_pars": {
      "distributed": "mpi",
      "max_batch_sample": 10,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/KMNIST/resnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/KMNIST/resnet18/"
    }
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_tch.torchhub' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 

  <mlmodels.model_tch.torchhub.Model object at 0x7f7cde6dec18> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdd318f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdd318f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdd318f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110225.05it/s]  0%|          | 40960/18165135 [00:00<02:28, 122371.17it/s]  1%|          | 98304/18165135 [00:00<01:57, 153685.18it/s]  1%|          | 212992/18165135 [00:01<01:28, 202231.68it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272278.40it/s]  5%|▍         | 868352/18165135 [00:01<00:46, 374253.76it/s] 10%|▉         | 1736704/18165135 [00:01<00:31, 520325.04it/s] 19%|█▉        | 3481600/18165135 [00:01<00:20, 729400.74it/s] 32%|███▏      | 5791744/18165135 [00:01<00:12, 1027959.34it/s] 38%|███▊      | 6856704/18165135 [00:02<00:08, 1343402.50it/s] 46%|████▌     | 8265728/18165135 [00:02<00:05, 1707407.24it/s] 53%|█████▎    | 9584640/18165135 [00:02<00:04, 2091048.75it/s] 67%|██████▋   | 12206080/18165135 [00:02<00:02, 2854228.42it/s] 73%|███████▎  | 13254656/18165135 [00:02<00:01, 3416929.94it/s] 78%|███████▊  | 14245888/18165135 [00:03<00:00, 4157754.67it/s] 84%|████████▍ | 15286272/18165135 [00:03<00:00, 5026543.32it/s] 89%|████████▉ | 16211968/18165135 [00:03<00:00, 5569092.79it/s] 94%|█████████▍| 17080320/18165135 [00:03<00:00, 5855100.54it/s] 98%|█████████▊| 17891328/18165135 [00:03<00:00, 5903104.30it/s]18169856it [00:03, 5086266.74it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 109745.89it/s]32768it [00:00, 72711.83it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110159.85it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122338.91it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153795.48it/s]  7%|▋         | 212992/3041136 [00:00<00:13, 202326.12it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 272528.20it/s] 29%|██▊       | 868352/3041136 [00:01<00:05, 374581.41it/s] 57%|█████▋    | 1736704/3041136 [00:01<00:02, 520776.71it/s] 94%|█████████▍| 2867200/3041136 [00:01<00:00, 722653.90it/s]3047424it [00:01, 2027841.99it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 27127.24it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004554310977458954 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.034785932779312134 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0037543396353721618 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02111237132549286 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.004257721026738485 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.036746111631393436 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.003144671618938446 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.0651039686203003 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.00294822096824646 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.01880384647846222 	 Accuracy: 4
model saves at 4 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7cdd3187b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7cdd3187b8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7cdd3187b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([5, 2, 2, 4, 6, 1, 0, 1, 9, 9]), array([0, 3, 0, 4, 5, 1, 9, 6, 4, 9]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

