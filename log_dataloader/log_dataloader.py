
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6a99d88b4362365e9c836391bbcc0ec2c7c12210', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '6a99d88b4362365e9c836391bbcc0ec2c7c12210', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6a99d88b4362365e9c836391bbcc0ec2c7c12210

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6a99d88b4362365e9c836391bbcc0ec2c7c12210

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6a99d88b4362365e9c836391bbcc0ec2c7c12210

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa65413bf98> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6530f20d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6530f20d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6530f20d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 159314.79it/s] 63%|██████▎   | 6217728/9912422 [00:00<00:16, 227341.41it/s]9920512it [00:00, 42224130.94it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 284617.52it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160673.27it/s]1654784it [00:00, 11133569.38it/s]                         
0it [00:00, ?it/s]8192it [00:00, 175231.48it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0036077163616816204 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01987665855884552 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0019933396875858305 	 Accuracy: 1
Train Epoch: 2 	 Loss: 0.016273722052574158 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.002195437322060267 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.0072566565573215485 	 Accuracy: 7
model saves at 7 accuracy
Train Epoch: 4 	 Loss: 0.0013943748126427334 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.012467967242002487 	 Accuracy: 6
Train Epoch: 5 	 Loss: 0.0018324399987856546 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01222967231273651 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6530d5f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6530d5f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6530d5f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([5, 7, 6, 5, 7, 1, 7, 0, 7, 7]), array([5, 8, 4, 5, 0, 1, 2, 0, 0, 8]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa654127630> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6509e1840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6509e1840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6509e1840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0018280904491742452 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02187564015388489 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0020557056268056236 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03186899280548096 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6509e5d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6509e5d90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6509e5d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([7, 2, 7, 2, 2, 2, 2, 2, 2, 2]), array([1, 4, 0, 1, 7, 6, 7, 9, 6, 1]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 15%|█▌        | 6.77M/44.7M [00:00<00:00, 71.0MB/s] 33%|███▎      | 14.6M/44.7M [00:00<00:00, 74.1MB/s] 51%|█████     | 22.6M/44.7M [00:00<00:00, 76.8MB/s] 69%|██████▉   | 31.0M/44.7M [00:00<00:00, 78.6MB/s] 87%|████████▋ | 39.0M/44.7M [00:00<00:00, 77.3MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 79.3MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7fa653f767b8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa6530c1e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa6530c1e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fa6530c1e18> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:22,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:21,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:20,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:19,  1.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:54,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:53,  2.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:36,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.40 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 23.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 29.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 35.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 54.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 59.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 59.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 70.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 72.46 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 72.46 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 32.33 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.01s/ url]
0 examples [00:00, ? examples/s]2020-05-23 03:19:12.581277: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 03:19:12.722140: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-23 03:19:12.722299: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56554176fce0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 03:19:12.722317: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:01,  1.90s/ examples]95 examples [00:02,  1.33s/ examples]191 examples [00:02,  1.07 examples/s]284 examples [00:02,  1.53 examples/s]389 examples [00:02,  2.19 examples/s]492 examples [00:02,  3.12 examples/s]595 examples [00:02,  4.45 examples/s]699 examples [00:02,  6.35 examples/s]803 examples [00:02,  9.05 examples/s]908 examples [00:02, 12.88 examples/s]1013 examples [00:02, 18.30 examples/s]1116 examples [00:03, 25.95 examples/s]1218 examples [00:03, 36.66 examples/s]1319 examples [00:03, 51.55 examples/s]1419 examples [00:03, 72.05 examples/s]1521 examples [00:03, 99.88 examples/s]1622 examples [00:03, 136.84 examples/s]1723 examples [00:03, 184.72 examples/s]1826 examples [00:03, 244.89 examples/s]1929 examples [00:03, 317.38 examples/s]2031 examples [00:03, 399.22 examples/s]2134 examples [00:04, 489.02 examples/s]2236 examples [00:04, 575.44 examples/s]2341 examples [00:04, 665.34 examples/s]2446 examples [00:04, 747.02 examples/s]2551 examples [00:04, 816.72 examples/s]2655 examples [00:04, 871.18 examples/s]2759 examples [00:04, 912.49 examples/s]2862 examples [00:04, 939.95 examples/s]2965 examples [00:04, 965.03 examples/s]3068 examples [00:04, 972.04 examples/s]3170 examples [00:05, 985.82 examples/s]3272 examples [00:05, 985.25 examples/s]3373 examples [00:05, 967.29 examples/s]3472 examples [00:05, 956.79 examples/s]3574 examples [00:05, 973.41 examples/s]3675 examples [00:05, 981.72 examples/s]3778 examples [00:05, 993.51 examples/s]3880 examples [00:05, 996.12 examples/s]3983 examples [00:05, 1005.47 examples/s]4085 examples [00:05, 1007.94 examples/s]4189 examples [00:06, 1014.43 examples/s]4291 examples [00:06, 972.32 examples/s] 4395 examples [00:06, 990.18 examples/s]4499 examples [00:06, 1004.32 examples/s]4600 examples [00:06, 994.68 examples/s] 4702 examples [00:06, 1001.93 examples/s]4803 examples [00:06, 998.17 examples/s] 4906 examples [00:06, 1006.81 examples/s]5007 examples [00:06, 1002.14 examples/s]5109 examples [00:06, 1005.65 examples/s]5210 examples [00:07, 1001.56 examples/s]5311 examples [00:07, 991.17 examples/s] 5411 examples [00:07, 987.61 examples/s]5510 examples [00:07, 965.79 examples/s]5611 examples [00:07, 976.02 examples/s]5713 examples [00:07, 986.62 examples/s]5815 examples [00:07, 994.72 examples/s]5917 examples [00:07, 1002.15 examples/s]6019 examples [00:07, 1005.60 examples/s]6123 examples [00:07, 1012.22 examples/s]6225 examples [00:08, 1002.65 examples/s]6326 examples [00:08, 1001.39 examples/s]6427 examples [00:08, 979.26 examples/s] 6530 examples [00:08, 992.17 examples/s]6630 examples [00:08, 988.02 examples/s]6730 examples [00:08, 989.64 examples/s]6832 examples [00:08, 997.87 examples/s]6934 examples [00:08, 1003.23 examples/s]7036 examples [00:08, 1006.01 examples/s]7137 examples [00:09, 1006.69 examples/s]7238 examples [00:09, 998.94 examples/s] 7340 examples [00:09, 1004.89 examples/s]7441 examples [00:09, 974.00 examples/s] 7539 examples [00:09, 965.07 examples/s]7641 examples [00:09, 980.09 examples/s]7740 examples [00:09, 979.62 examples/s]7839 examples [00:09, 961.43 examples/s]7936 examples [00:09, 938.22 examples/s]8032 examples [00:09, 943.52 examples/s]8129 examples [00:10, 949.65 examples/s]8227 examples [00:10, 956.50 examples/s]8328 examples [00:10, 969.73 examples/s]8428 examples [00:10, 977.12 examples/s]8526 examples [00:10, 964.31 examples/s]8630 examples [00:10, 983.90 examples/s]8731 examples [00:10, 991.48 examples/s]8831 examples [00:10, 991.36 examples/s]8931 examples [00:10, 956.05 examples/s]9029 examples [00:10, 960.64 examples/s]9130 examples [00:11, 970.31 examples/s]9229 examples [00:11, 974.00 examples/s]9327 examples [00:11, 974.74 examples/s]9425 examples [00:11, 952.51 examples/s]9521 examples [00:11, 937.28 examples/s]9615 examples [00:11, 924.69 examples/s]9717 examples [00:11, 950.41 examples/s]9820 examples [00:11, 970.51 examples/s]9918 examples [00:11, 931.93 examples/s]10012 examples [00:12, 905.97 examples/s]10111 examples [00:12, 927.62 examples/s]10209 examples [00:12, 941.60 examples/s]10304 examples [00:12, 934.06 examples/s]10405 examples [00:12, 954.66 examples/s]10508 examples [00:12, 974.49 examples/s]10610 examples [00:12, 987.61 examples/s]10713 examples [00:12, 999.75 examples/s]10815 examples [00:12, 1004.34 examples/s]10916 examples [00:12, 1005.20 examples/s]11019 examples [00:13, 1011.63 examples/s]11121 examples [00:13, 1006.09 examples/s]11222 examples [00:13, 988.83 examples/s] 11323 examples [00:13, 993.74 examples/s]11423 examples [00:13, 963.75 examples/s]11526 examples [00:13, 981.65 examples/s]11627 examples [00:13, 989.63 examples/s]11728 examples [00:13, 995.21 examples/s]11832 examples [00:13, 1006.06 examples/s]11936 examples [00:13, 1013.95 examples/s]12038 examples [00:14, 1007.73 examples/s]12139 examples [00:14, 997.60 examples/s] 12240 examples [00:14, 1000.58 examples/s]12341 examples [00:14, 1002.63 examples/s]12444 examples [00:14, 1009.70 examples/s]12546 examples [00:14, 1010.00 examples/s]12648 examples [00:14, 1006.31 examples/s]12749 examples [00:14, 1005.91 examples/s]12850 examples [00:14, 992.67 examples/s] 12951 examples [00:14, 996.06 examples/s]13052 examples [00:15, 998.26 examples/s]13153 examples [00:15, 999.61 examples/s]13254 examples [00:15, 1001.63 examples/s]13355 examples [00:15, 979.48 examples/s] 13454 examples [00:15, 973.41 examples/s]13552 examples [00:15, 939.93 examples/s]13647 examples [00:15, 928.74 examples/s]13748 examples [00:15, 949.82 examples/s]13848 examples [00:15, 962.82 examples/s]13950 examples [00:16, 977.03 examples/s]14051 examples [00:16, 985.29 examples/s]14151 examples [00:16, 986.79 examples/s]14252 examples [00:16, 992.84 examples/s]14352 examples [00:16, 989.11 examples/s]14452 examples [00:16, 989.64 examples/s]14553 examples [00:16, 994.50 examples/s]14653 examples [00:16, 993.15 examples/s]14754 examples [00:16, 997.97 examples/s]14854 examples [00:16, 991.74 examples/s]14954 examples [00:17, 979.77 examples/s]15056 examples [00:17, 990.63 examples/s]15158 examples [00:17, 997.72 examples/s]15263 examples [00:17, 1010.85 examples/s]15366 examples [00:17, 1012.49 examples/s]15468 examples [00:17, 1013.93 examples/s]15570 examples [00:17, 999.01 examples/s] 15670 examples [00:17, 999.14 examples/s]15770 examples [00:17, 983.04 examples/s]15869 examples [00:17, 978.57 examples/s]15970 examples [00:18, 985.57 examples/s]16073 examples [00:18, 997.58 examples/s]16174 examples [00:18, 999.05 examples/s]16277 examples [00:18, 1005.73 examples/s]16378 examples [00:18, 979.93 examples/s] 16480 examples [00:18, 990.93 examples/s]16583 examples [00:18, 1001.12 examples/s]16684 examples [00:18, 985.22 examples/s] 16783 examples [00:18, 985.17 examples/s]16883 examples [00:18, 988.36 examples/s]16986 examples [00:19, 998.43 examples/s]17089 examples [00:19, 1004.71 examples/s]17190 examples [00:19, 1002.36 examples/s]17292 examples [00:19, 1006.66 examples/s]17394 examples [00:19, 1007.84 examples/s]17496 examples [00:19, 1011.15 examples/s]17598 examples [00:19, 974.54 examples/s] 17697 examples [00:19, 977.30 examples/s]17795 examples [00:19, 961.34 examples/s]17896 examples [00:19, 973.39 examples/s]17997 examples [00:20, 983.59 examples/s]18096 examples [00:20, 977.92 examples/s]18194 examples [00:20, 974.84 examples/s]18292 examples [00:20, 974.21 examples/s]18394 examples [00:20, 985.86 examples/s]18494 examples [00:20, 988.89 examples/s]18596 examples [00:20, 996.02 examples/s]18697 examples [00:20, 998.07 examples/s]18797 examples [00:20, 984.50 examples/s]18898 examples [00:20, 991.96 examples/s]19000 examples [00:21, 998.22 examples/s]19100 examples [00:21, 990.23 examples/s]19200 examples [00:21, 968.49 examples/s]19302 examples [00:21, 982.45 examples/s]19401 examples [00:21, 957.16 examples/s]19504 examples [00:21, 977.01 examples/s]19606 examples [00:21, 987.70 examples/s]19708 examples [00:21, 996.05 examples/s]19808 examples [00:21, 994.15 examples/s]19908 examples [00:22, 984.67 examples/s]20007 examples [00:22, 929.04 examples/s]20106 examples [00:22, 946.20 examples/s]20209 examples [00:22, 969.05 examples/s]20307 examples [00:22, 971.76 examples/s]20405 examples [00:22, 928.49 examples/s]20504 examples [00:22, 944.16 examples/s]20607 examples [00:22, 967.26 examples/s]20707 examples [00:22, 974.46 examples/s]20805 examples [00:22, 974.68 examples/s]20904 examples [00:23, 978.14 examples/s]21006 examples [00:23, 988.76 examples/s]21109 examples [00:23, 1000.23 examples/s]21213 examples [00:23, 1010.09 examples/s]21316 examples [00:23, 1015.90 examples/s]21419 examples [00:23, 1019.72 examples/s]21522 examples [00:23, 1021.44 examples/s]21625 examples [00:23, 1021.20 examples/s]21728 examples [00:23, 1021.45 examples/s]21831 examples [00:23, 1017.97 examples/s]21933 examples [00:24, 1014.46 examples/s]22035 examples [00:24, 1010.31 examples/s]22137 examples [00:24, 1008.88 examples/s]22241 examples [00:24, 1016.59 examples/s]22345 examples [00:24, 1021.01 examples/s]22448 examples [00:24, 981.49 examples/s] 22550 examples [00:24, 992.63 examples/s]22651 examples [00:24, 995.80 examples/s]22751 examples [00:24, 994.37 examples/s]22853 examples [00:24, 1001.28 examples/s]22957 examples [00:25, 1009.81 examples/s]23059 examples [00:25, 1012.12 examples/s]23161 examples [00:25, 1008.81 examples/s]23263 examples [00:25, 1011.61 examples/s]23367 examples [00:25, 1018.92 examples/s]23470 examples [00:25, 1019.76 examples/s]23573 examples [00:25, 937.00 examples/s] 23672 examples [00:25, 950.23 examples/s]23769 examples [00:25, 925.18 examples/s]23863 examples [00:26, 926.22 examples/s]23965 examples [00:26, 950.50 examples/s]24062 examples [00:26, 955.43 examples/s]24159 examples [00:26, 959.27 examples/s]24262 examples [00:26, 978.95 examples/s]24365 examples [00:26, 993.60 examples/s]24468 examples [00:26, 1003.98 examples/s]24569 examples [00:26, 992.13 examples/s] 24670 examples [00:26, 994.98 examples/s]24770 examples [00:26, 993.99 examples/s]24870 examples [00:27, 984.80 examples/s]24969 examples [00:27, 984.91 examples/s]25068 examples [00:27, 969.63 examples/s]25171 examples [00:27, 984.94 examples/s]25273 examples [00:27, 993.82 examples/s]25373 examples [00:27, 969.96 examples/s]25471 examples [00:27, 963.86 examples/s]25574 examples [00:27, 980.12 examples/s]25674 examples [00:27, 983.64 examples/s]25773 examples [00:27, 980.88 examples/s]25874 examples [00:28, 989.37 examples/s]25977 examples [00:28, 998.71 examples/s]26079 examples [00:28, 1004.40 examples/s]26180 examples [00:28, 1003.15 examples/s]26281 examples [00:28, 1002.62 examples/s]26384 examples [00:28, 1009.10 examples/s]26488 examples [00:28, 1015.39 examples/s]26591 examples [00:28, 1016.57 examples/s]26693 examples [00:28, 1005.01 examples/s]26795 examples [00:28, 1007.45 examples/s]26898 examples [00:29, 1012.24 examples/s]27000 examples [00:29, 1012.50 examples/s]27103 examples [00:29, 1017.34 examples/s]27205 examples [00:29, 1014.92 examples/s]27307 examples [00:29, 985.29 examples/s] 27406 examples [00:29, 976.85 examples/s]27504 examples [00:29, 975.23 examples/s]27606 examples [00:29, 986.56 examples/s]27706 examples [00:29, 987.71 examples/s]27805 examples [00:30, 963.55 examples/s]27908 examples [00:30, 980.53 examples/s]28011 examples [00:30, 994.03 examples/s]28114 examples [00:30, 1003.78 examples/s]28215 examples [00:30, 996.10 examples/s] 28317 examples [00:30, 1000.55 examples/s]28418 examples [00:30, 975.58 examples/s] 28522 examples [00:30, 993.13 examples/s]28626 examples [00:30, 1004.40 examples/s]28727 examples [00:30, 1001.36 examples/s]28830 examples [00:31, 1009.17 examples/s]28933 examples [00:31, 1014.44 examples/s]29035 examples [00:31, 1000.70 examples/s]29136 examples [00:31, 1000.09 examples/s]29240 examples [00:31, 1009.80 examples/s]29345 examples [00:31, 1019.17 examples/s]29447 examples [00:31, 1013.09 examples/s]29549 examples [00:31, 1006.70 examples/s]29650 examples [00:31, 995.38 examples/s] 29750 examples [00:31, 989.69 examples/s]29850 examples [00:32, 961.44 examples/s]29950 examples [00:32, 971.18 examples/s]30048 examples [00:32, 882.68 examples/s]30147 examples [00:32, 912.06 examples/s]30250 examples [00:32, 944.44 examples/s]30353 examples [00:32, 966.31 examples/s]30457 examples [00:32, 985.22 examples/s]30561 examples [00:32, 998.94 examples/s]30662 examples [00:32, 988.10 examples/s]30763 examples [00:33, 993.58 examples/s]30863 examples [00:33, 989.45 examples/s]30964 examples [00:33, 994.89 examples/s]31067 examples [00:33, 1002.50 examples/s]31168 examples [00:33, 1003.37 examples/s]31269 examples [00:33, 994.61 examples/s] 31369 examples [00:33, 980.26 examples/s]31468 examples [00:33, 959.77 examples/s]31571 examples [00:33, 977.71 examples/s]31672 examples [00:33, 985.57 examples/s]31774 examples [00:34, 993.67 examples/s]31874 examples [00:34, 981.38 examples/s]31973 examples [00:34, 955.40 examples/s]32076 examples [00:34, 975.03 examples/s]32178 examples [00:34, 986.54 examples/s]32279 examples [00:34, 993.44 examples/s]32379 examples [00:34, 993.38 examples/s]32483 examples [00:34, 1004.41 examples/s]32584 examples [00:34, 990.48 examples/s] 32684 examples [00:34, 950.00 examples/s]32788 examples [00:35, 973.85 examples/s]32892 examples [00:35, 992.75 examples/s]32995 examples [00:35, 1001.78 examples/s]33100 examples [00:35, 1013.46 examples/s]33202 examples [00:35, 1004.47 examples/s]33303 examples [00:35, 1000.27 examples/s]33406 examples [00:35, 1007.53 examples/s]33509 examples [00:35, 1012.83 examples/s]33611 examples [00:35, 1012.83 examples/s]33713 examples [00:35, 1006.68 examples/s]33814 examples [00:36, 1004.18 examples/s]33915 examples [00:36, 984.13 examples/s] 34014 examples [00:36, 977.56 examples/s]34113 examples [00:36, 979.60 examples/s]34213 examples [00:36, 985.57 examples/s]34316 examples [00:36, 997.25 examples/s]34416 examples [00:36, 982.24 examples/s]34516 examples [00:36, 986.97 examples/s]34619 examples [00:36, 997.24 examples/s]34719 examples [00:36, 994.82 examples/s]34819 examples [00:37, 994.16 examples/s]34923 examples [00:37, 1006.13 examples/s]35027 examples [00:37, 1013.37 examples/s]35132 examples [00:37, 1022.04 examples/s]35235 examples [00:37, 1023.04 examples/s]35338 examples [00:37, 1021.27 examples/s]35442 examples [00:37, 1023.93 examples/s]35546 examples [00:37, 1027.31 examples/s]35650 examples [00:37, 1030.94 examples/s]35754 examples [00:38, 1025.81 examples/s]35857 examples [00:38, 1019.91 examples/s]35960 examples [00:38, 1000.27 examples/s]36061 examples [00:38, 997.80 examples/s] 36162 examples [00:38, 998.95 examples/s]36262 examples [00:38, 995.86 examples/s]36362 examples [00:38, 995.67 examples/s]36464 examples [00:38, 1002.16 examples/s]36565 examples [00:38, 967.71 examples/s] 36668 examples [00:38, 984.71 examples/s]36769 examples [00:39, 990.84 examples/s]36870 examples [00:39, 995.73 examples/s]36973 examples [00:39, 1003.95 examples/s]37075 examples [00:39, 1006.34 examples/s]37177 examples [00:39, 1009.84 examples/s]37279 examples [00:39, 1006.54 examples/s]37381 examples [00:39, 1010.15 examples/s]37483 examples [00:39, 992.82 examples/s] 37583 examples [00:39, 982.53 examples/s]37686 examples [00:39, 993.71 examples/s]37787 examples [00:40, 996.53 examples/s]37890 examples [00:40, 1004.03 examples/s]37993 examples [00:40, 1009.84 examples/s]38096 examples [00:40, 1013.52 examples/s]38198 examples [00:40, 1011.71 examples/s]38300 examples [00:40, 1007.24 examples/s]38402 examples [00:40, 1009.69 examples/s]38505 examples [00:40, 1013.24 examples/s]38608 examples [00:40, 1017.72 examples/s]38710 examples [00:40, 995.27 examples/s] 38810 examples [00:41, 988.30 examples/s]38910 examples [00:41, 989.73 examples/s]39010 examples [00:41, 991.23 examples/s]39112 examples [00:41, 999.41 examples/s]39212 examples [00:41, 987.97 examples/s]39313 examples [00:41, 991.91 examples/s]39415 examples [00:41, 998.40 examples/s]39517 examples [00:41, 1002.26 examples/s]39619 examples [00:41, 1005.37 examples/s]39720 examples [00:41, 1005.55 examples/s]39821 examples [00:42, 996.10 examples/s] 39922 examples [00:42, 998.03 examples/s]40022 examples [00:42, 948.18 examples/s]40125 examples [00:42, 970.88 examples/s]40227 examples [00:42, 984.84 examples/s]40326 examples [00:42, 980.15 examples/s]40426 examples [00:42, 985.59 examples/s]40525 examples [00:42, 945.96 examples/s]40625 examples [00:42, 961.36 examples/s]40728 examples [00:43, 980.91 examples/s]40829 examples [00:43, 987.34 examples/s]40930 examples [00:43, 993.40 examples/s]41030 examples [00:43, 989.06 examples/s]41130 examples [00:43, 955.04 examples/s]41229 examples [00:43, 964.03 examples/s]41330 examples [00:43, 976.00 examples/s]41434 examples [00:43, 993.49 examples/s]41538 examples [00:43, 1005.30 examples/s]41642 examples [00:43, 1013.14 examples/s]41747 examples [00:44, 1022.66 examples/s]41850 examples [00:44, 1021.14 examples/s]41953 examples [00:44, 1008.67 examples/s]42056 examples [00:44, 1013.14 examples/s]42158 examples [00:44, 1012.30 examples/s]42262 examples [00:44, 1018.20 examples/s]42364 examples [00:44, 1013.56 examples/s]42468 examples [00:44, 1020.18 examples/s]42572 examples [00:44, 1024.95 examples/s]42675 examples [00:44, 1022.38 examples/s]42778 examples [00:45, 975.84 examples/s] 42879 examples [00:45, 985.51 examples/s]42983 examples [00:45, 999.58 examples/s]43087 examples [00:45, 1009.13 examples/s]43189 examples [00:45, 988.02 examples/s] 43291 examples [00:45, 996.08 examples/s]43391 examples [00:45, 969.08 examples/s]43489 examples [00:45, 954.74 examples/s]43585 examples [00:45, 930.84 examples/s]43683 examples [00:45, 942.72 examples/s]43778 examples [00:46, 940.76 examples/s]43878 examples [00:46, 955.69 examples/s]43980 examples [00:46, 971.43 examples/s]44083 examples [00:46, 986.01 examples/s]44186 examples [00:46, 997.14 examples/s]44289 examples [00:46, 1006.44 examples/s]44390 examples [00:46, 1002.99 examples/s]44492 examples [00:46, 1005.85 examples/s]44595 examples [00:46, 1012.42 examples/s]44697 examples [00:47, 1009.35 examples/s]44798 examples [00:47, 1000.16 examples/s]44899 examples [00:47, 981.50 examples/s] 45000 examples [00:47, 987.79 examples/s]45103 examples [00:47, 999.01 examples/s]45205 examples [00:47, 1003.81 examples/s]45308 examples [00:47, 1009.18 examples/s]45409 examples [00:47, 988.04 examples/s] 45509 examples [00:47, 991.04 examples/s]45610 examples [00:47, 996.10 examples/s]45712 examples [00:48, 1000.54 examples/s]45813 examples [00:48, 1003.05 examples/s]45914 examples [00:48, 1002.10 examples/s]46015 examples [00:48, 1003.79 examples/s]46117 examples [00:48, 1008.51 examples/s]46220 examples [00:48, 1013.71 examples/s]46323 examples [00:48, 1018.50 examples/s]46425 examples [00:48, 1009.17 examples/s]46527 examples [00:48, 1010.20 examples/s]46629 examples [00:48, 978.81 examples/s] 46731 examples [00:49, 990.45 examples/s]46833 examples [00:49, 997.68 examples/s]46934 examples [00:49, 999.50 examples/s]47036 examples [00:49, 1004.43 examples/s]47137 examples [00:49, 999.82 examples/s] 47239 examples [00:49, 1005.29 examples/s]47341 examples [00:49, 1008.66 examples/s]47442 examples [00:49, 999.67 examples/s] 47545 examples [00:49, 1006.82 examples/s]47647 examples [00:49, 1009.63 examples/s]47748 examples [00:50, 1002.70 examples/s]47849 examples [00:50, 1001.60 examples/s]47950 examples [00:50, 982.79 examples/s] 48049 examples [00:50, 954.24 examples/s]48149 examples [00:50, 967.05 examples/s]48252 examples [00:50, 982.98 examples/s]48353 examples [00:50, 989.43 examples/s]48454 examples [00:50, 994.53 examples/s]48555 examples [00:50, 998.98 examples/s]48655 examples [00:50, 986.12 examples/s]48754 examples [00:51, 985.09 examples/s]48855 examples [00:51, 991.16 examples/s]48957 examples [00:51, 996.93 examples/s]49060 examples [00:51, 1004.21 examples/s]49162 examples [00:51, 1008.12 examples/s]49264 examples [00:51, 1009.93 examples/s]49366 examples [00:51, 1011.64 examples/s]49468 examples [00:51, 1004.11 examples/s]49571 examples [00:51, 1009.43 examples/s]49672 examples [00:52, 976.32 examples/s] 49770 examples [00:52, 975.59 examples/s]49869 examples [00:52, 977.00 examples/s]49967 examples [00:52, 973.03 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7022/50000 [00:00<00:00, 70217.89 examples/s] 38%|███▊      | 18982/50000 [00:00<00:00, 80145.25 examples/s] 62%|██████▏   | 30875/50000 [00:00<00:00, 88834.86 examples/s] 84%|████████▍ | 42237/50000 [00:00<00:00, 95054.95 examples/s]                                                               0 examples [00:00, ? examples/s]89 examples [00:00, 881.94 examples/s]190 examples [00:00, 914.79 examples/s]292 examples [00:00, 942.08 examples/s]392 examples [00:00, 952.99 examples/s]485 examples [00:00, 945.45 examples/s]588 examples [00:00, 966.77 examples/s]692 examples [00:00, 985.81 examples/s]795 examples [00:00, 997.60 examples/s]890 examples [00:00, 971.40 examples/s]994 examples [00:01, 988.89 examples/s]1099 examples [00:01, 1004.60 examples/s]1203 examples [00:01, 1014.01 examples/s]1308 examples [00:01, 1021.86 examples/s]1411 examples [00:01, 1023.43 examples/s]1515 examples [00:01, 1025.91 examples/s]1620 examples [00:01, 1030.42 examples/s]1725 examples [00:01, 1035.42 examples/s]1830 examples [00:01, 1038.68 examples/s]1934 examples [00:01, 1012.56 examples/s]2036 examples [00:02, 1005.18 examples/s]2139 examples [00:02, 1012.40 examples/s]2241 examples [00:02, 973.91 examples/s] 2345 examples [00:02, 990.78 examples/s]2448 examples [00:02, 1000.52 examples/s]2551 examples [00:02, 1000.39 examples/s]2655 examples [00:02, 1010.54 examples/s]2758 examples [00:02, 1014.64 examples/s]2862 examples [00:02, 1020.33 examples/s]2965 examples [00:02, 1019.44 examples/s]3068 examples [00:03, 1021.81 examples/s]3171 examples [00:03, 1022.59 examples/s]3275 examples [00:03, 1025.27 examples/s]3379 examples [00:03, 1026.68 examples/s]3482 examples [00:03, 1018.38 examples/s]3584 examples [00:03, 1018.66 examples/s]3688 examples [00:03, 1023.22 examples/s]3793 examples [00:03, 1028.38 examples/s]3896 examples [00:03, 1026.28 examples/s]3999 examples [00:03, 1025.04 examples/s]4102 examples [00:04, 990.76 examples/s] 4205 examples [00:04, 1001.77 examples/s]4307 examples [00:04, 1001.82 examples/s]4408 examples [00:04, 1004.21 examples/s]4512 examples [00:04, 1013.52 examples/s]4616 examples [00:04, 1019.50 examples/s]4721 examples [00:04, 1027.66 examples/s]4825 examples [00:04, 1029.07 examples/s]4928 examples [00:04, 1021.70 examples/s]5031 examples [00:04, 1022.45 examples/s]5134 examples [00:05, 1011.72 examples/s]5236 examples [00:05, 996.83 examples/s] 5336 examples [00:05, 978.81 examples/s]5440 examples [00:05, 994.75 examples/s]5542 examples [00:05, 1000.90 examples/s]5643 examples [00:05, 1001.46 examples/s]5745 examples [00:05, 1004.41 examples/s]5848 examples [00:05, 1010.23 examples/s]5951 examples [00:05, 1014.58 examples/s]6053 examples [00:06, 980.07 examples/s] 6152 examples [00:06, 969.77 examples/s]6254 examples [00:06, 984.23 examples/s]6356 examples [00:06, 993.91 examples/s]6456 examples [00:06, 989.11 examples/s]6556 examples [00:06, 990.29 examples/s]6656 examples [00:06, 985.95 examples/s]6759 examples [00:06, 996.84 examples/s]6863 examples [00:06, 1008.75 examples/s]6964 examples [00:06, 1007.65 examples/s]7065 examples [00:07, 1006.52 examples/s]7167 examples [00:07, 1009.22 examples/s]7268 examples [00:07, 990.67 examples/s] 7372 examples [00:07, 1003.72 examples/s]7475 examples [00:07, 1010.94 examples/s]7577 examples [00:07, 952.59 examples/s] 7674 examples [00:07, 948.49 examples/s]7779 examples [00:07, 975.94 examples/s]7884 examples [00:07, 993.94 examples/s]7989 examples [00:07, 1007.85 examples/s]8091 examples [00:08, 1011.08 examples/s]8193 examples [00:08, 1002.47 examples/s]8294 examples [00:08, 975.16 examples/s] 8395 examples [00:08, 984.42 examples/s]8498 examples [00:08, 997.16 examples/s]8600 examples [00:08, 1003.53 examples/s]8701 examples [00:08, 994.97 examples/s] 8806 examples [00:08, 1009.88 examples/s]8910 examples [00:08, 1017.77 examples/s]9015 examples [00:08, 1026.73 examples/s]9119 examples [00:09, 1028.81 examples/s]9224 examples [00:09, 1034.54 examples/s]9328 examples [00:09, 1006.89 examples/s]9429 examples [00:09, 1002.28 examples/s]9530 examples [00:09, 1001.65 examples/s]9631 examples [00:09, 1004.09 examples/s]9732 examples [00:09, 1003.63 examples/s]9834 examples [00:09, 1007.38 examples/s]9937 examples [00:09, 1011.33 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFASPDV/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFASPDV/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6228ce8c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6228ce8c8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6228ce8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.4935440897941589 	 Accuracy: 46
Train Epoch: 1 	 Loss: 5.011375045776367 	 Accuracy: 180
model saves at 180 accuracy
Train Epoch: 2 	 Loss: 0.4010697054862976 	 Accuracy: 56
Train Epoch: 2 	 Loss: 8.147210884094239 	 Accuracy: 280
model saves at 280 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa5f2e2c620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa5f2e2c620>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fa5f2e2c620> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa5f038c7b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa5f038c7b8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa5f038c7b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([8, 7, 8, 0, 5, 7, 8, 8, 8, 8]), array([1, 7, 8, 0, 5, 7, 7, 3, 3, 2]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa65413bf98> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa5f1ab3b38> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa5f053d0d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa5f053d0d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa5f053d0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:37, 270620.15it/s]  0%|          | 131072/26421880 [00:00<01:19, 332630.15it/s]  2%|▏         | 425984/26421880 [00:00<00:58, 441319.32it/s]  4%|▍         | 1081344/26421880 [00:00<00:41, 610276.26it/s] 13%|█▎        | 3473408/26421880 [00:01<00:26, 856925.45it/s] 28%|██▊       | 7282688/26421880 [00:01<00:15, 1210860.50it/s] 46%|████▌     | 12148736/26421880 [00:01<00:08, 1711535.55it/s] 59%|█████▉    | 15532032/26421880 [00:01<00:04, 2367522.02it/s] 73%|███████▎  | 19210240/26421880 [00:01<00:02, 3279224.93it/s] 89%|████████▉ | 23625728/26421880 [00:01<00:00, 4540084.98it/s]26427392it [00:01, 14772313.99it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 93720.18it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 270870.11it/s]  4%|▍         | 188416/4422102 [00:00<00:11, 353933.05it/s] 11%|█         | 466944/4422102 [00:00<00:08, 464889.75it/s] 36%|███▌      | 1581056/4422102 [00:00<00:04, 651114.32it/s] 87%|████████▋ | 3833856/4422102 [00:01<00:00, 911984.83it/s]4423680it [00:01, 4287686.17it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 35385.49it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0036674314538637795 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01987356746196747 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0027121591170628867 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02700835859775543 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.0026539072891076407 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.011419918060302734 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0025885647336641947 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.01389524519443512 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.0021607056458791095 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.014427491843700409 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa5f0533730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa5f0533730>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa5f0533730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([0, 1, 4, 2, 1, 5, 1, 1, 5, 0]), array([0, 0, 0, 6, 1, 0, 1, 1, 5, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa5f038d978> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa5f1a1bd90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa5f1a1bd90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa5f1a1bd90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110302.54it/s]  0%|          | 40960/18165135 [00:00<02:27, 122487.08it/s]  1%|          | 98304/18165135 [00:00<01:57, 153990.10it/s]  1%|          | 212992/18165135 [00:01<01:28, 202620.28it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272886.13it/s]  3%|▎         | 565248/18165135 [00:01<00:51, 342047.60it/s]  7%|▋         | 1228800/18165135 [00:01<00:35, 474774.16it/s]  9%|▉         | 1638400/18165135 [00:01<00:26, 631423.92it/s] 11%|█▏        | 2048000/18165135 [00:01<00:19, 821165.97it/s] 14%|█▎        | 2473984/18165135 [00:01<00:15, 1044656.86it/s] 16%|█▌        | 2916352/18165135 [00:02<00:11, 1296701.41it/s] 19%|█▊        | 3366912/18165135 [00:02<00:09, 1563967.27it/s] 21%|██        | 3825664/18165135 [00:02<00:07, 1835169.02it/s] 24%|██▎       | 4292608/18165135 [00:02<00:06, 2095834.77it/s] 26%|██▌       | 4767744/18165135 [00:02<00:05, 2336709.06it/s] 29%|██▉       | 5242880/18165135 [00:02<00:04, 2708358.59it/s] 32%|███▏      | 5726208/18165135 [00:02<00:04, 3079986.43it/s] 34%|███▎      | 6103040/18165135 [00:03<00:03, 3200808.82it/s] 36%|███▌      | 6471680/18165135 [00:03<00:03, 3002712.42it/s] 38%|███▊      | 6815744/18165135 [00:03<00:04, 2829631.81it/s] 40%|███▉      | 7241728/18165135 [00:03<00:03, 2877059.10it/s] 43%|████▎     | 7757824/18165135 [00:03<00:03, 3031745.43it/s] 46%|████▌     | 8282112/18165135 [00:03<00:03, 3160048.32it/s] 48%|████▊     | 8806400/18165135 [00:03<00:02, 3260889.75it/s] 51%|█████▏    | 9338880/18165135 [00:04<00:02, 3350337.59it/s] 54%|█████▍    | 9863168/18165135 [00:04<00:02, 3686457.71it/s] 57%|█████▋    | 10387456/18165135 [00:04<00:01, 3977687.46it/s] 59%|█████▉    | 10805248/18165135 [00:04<00:01, 3946214.28it/s] 62%|██████▏   | 11214848/18165135 [00:04<00:01, 3561567.16it/s] 64%|██████▍   | 11591680/18165135 [00:04<00:02, 3277225.38it/s] 66%|██████▋   | 12050432/18165135 [00:04<00:01, 3547759.04it/s] 69%|██████▉   | 12574720/18165135 [00:04<00:01, 3879712.21it/s] 71%|███████▏  | 12984320/18165135 [00:04<00:01, 3854060.95it/s] 74%|███████▎  | 13385728/18165135 [00:05<00:01, 3512553.33it/s] 76%|███████▌  | 13754368/18165135 [00:05<00:01, 3223862.48it/s] 79%|███████▊  | 14278656/18165135 [00:05<00:01, 3319951.45it/s] 82%|████████▏ | 14843904/18165135 [00:05<00:00, 3448036.43it/s] 85%|████████▍ | 15409152/18165135 [00:05<00:00, 3538696.69it/s] 88%|████████▊ | 15974400/18165135 [00:05<00:00, 3907791.76it/s] 91%|█████████ | 16498688/18165135 [00:05<00:00, 4190572.96it/s] 93%|█████████▎| 16941056/18165135 [00:06<00:00, 4113359.61it/s] 96%|█████████▌| 17367040/18165135 [00:06<00:00, 3761565.68it/s] 98%|█████████▊| 17760256/18165135 [00:06<00:00, 3423656.03it/s]100%|█████████▉| 18120704/18165135 [00:06<00:00, 2984798.85it/s]18169856it [00:06, 2819384.18it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110202.78it/s]32768it [00:00, 54105.24it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 109540.87it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 121999.76it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153441.90it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 200742.42it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 271063.71it/s] 28%|██▊       | 860160/3041136 [00:01<00:05, 372390.43it/s] 57%|█████▋    | 1728512/3041136 [00:01<00:02, 517801.63it/s]3047424it [00:01, 1948336.61it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17434.89it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004132438600063324 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.023509946584701538 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0035710445245107016 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.028145468235015868 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0033668922980626425 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.05036019206047058 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0026701502402623496 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.02128412616252899 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0025511623124281567 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.017013410806655884 	 Accuracy: 5
model saves at 5 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa5f1a1b840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa5f1a1b840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa5f1a1b840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([4, 4, 0, 2, 2, 5, 0, 0, 4, 8]), array([5, 4, 0, 9, 5, 5, 0, 0, 3, 1]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

