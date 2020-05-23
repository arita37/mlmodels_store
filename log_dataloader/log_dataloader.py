
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/83d573d3e4d1469b00137d60b7a53c8c5dbf49a9', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '83d573d3e4d1469b00137d60b7a53c8c5dbf49a9', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/83d573d3e4d1469b00137d60b7a53c8c5dbf49a9

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/83d573d3e4d1469b00137d60b7a53c8c5dbf49a9

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/83d573d3e4d1469b00137d60b7a53c8c5dbf49a9

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fbfaa5def98> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbfa9599048>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbfa9599048>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbfa9599048> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 27%|██▋       | 2654208/9912422 [00:00<00:00, 26531794.46it/s]9920512it [00:00, 31633726.18it/s]                             
0it [00:00, ?it/s]32768it [00:00, 383005.81it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8098586.91it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 74077.98it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0035299606323242185 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020069584131240846 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.00224101718266805 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.027803396821022033 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0023963655730088553 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.030407315969467164 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0021137563586235046 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.022990103900432587 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 5 	 Loss: 0.0016630302369594575 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01750285869836807 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbfa957aea0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbfa957aea0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbfa957aea0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([0, 0, 6, 5, 5, 1, 9, 5, 4, 6]), array([0, 0, 6, 9, 5, 1, 9, 8, 4, 6]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fbfaa5cc630> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbfa6d087b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbfa6d087b8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbfa6d087b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0018387967348098755 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.017751184225082397 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0019365756313006084 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.028145942211151125 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbfa6cf6d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbfa6cf6d90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbfa6cf6d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9]), array([1, 9, 5, 5, 4, 4, 6, 0, 2, 2]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 24%|██▎       | 10.6M/44.7M [00:00<00:00, 111MB/s] 51%|█████     | 22.7M/44.7M [00:00<00:00, 115MB/s] 78%|███████▊  | 34.7M/44.7M [00:00<00:00, 118MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 125MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7fbfaa4197b8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbfa9566d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbfa9566d90>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fbfa9566d90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:09,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:08,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:08,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:07,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:44,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:19,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:19,  6.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  8.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:12,  8.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:09, 12.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:09, 12.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:09, 12.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 12.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 16.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 21.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 28.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 36.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 44.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 52.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 66.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 73.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.19s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.78 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.07s/ url]
0 examples [00:00, ? examples/s]2020-05-23 18:51:17.492076: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 18:51:17.562728: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095159999 Hz
2020-05-23 18:51:17.563473: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5574ed9dca60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 18:51:17.563492: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  5.80 examples/s]123 examples [00:00,  8.26 examples/s]242 examples [00:00, 11.77 examples/s]347 examples [00:00, 16.73 examples/s]462 examples [00:00, 23.76 examples/s]569 examples [00:00, 33.62 examples/s]694 examples [00:00, 47.48 examples/s]815 examples [00:00, 66.70 examples/s]938 examples [00:00, 93.12 examples/s]1060 examples [00:01, 128.78 examples/s]1177 examples [00:01, 175.67 examples/s]1301 examples [00:01, 236.58 examples/s]1419 examples [00:01, 310.98 examples/s]1537 examples [00:01, 398.24 examples/s]1656 examples [00:01, 497.17 examples/s]1774 examples [00:01, 600.03 examples/s]1891 examples [00:01, 693.11 examples/s]2006 examples [00:01, 781.55 examples/s]2120 examples [00:01, 862.14 examples/s]2234 examples [00:02, 923.71 examples/s]2349 examples [00:02, 981.03 examples/s]2466 examples [00:02, 1030.75 examples/s]2591 examples [00:02, 1086.25 examples/s]2713 examples [00:02, 1120.67 examples/s]2833 examples [00:02, 1141.48 examples/s]2952 examples [00:02, 1137.70 examples/s]3074 examples [00:02, 1160.97 examples/s]3199 examples [00:02, 1183.61 examples/s]3322 examples [00:03, 1197.15 examples/s]3443 examples [00:03, 1188.36 examples/s]3563 examples [00:03, 1182.39 examples/s]3682 examples [00:03, 1180.44 examples/s]3804 examples [00:03, 1190.42 examples/s]3924 examples [00:03, 1173.87 examples/s]4043 examples [00:03, 1176.80 examples/s]4161 examples [00:03, 1166.05 examples/s]4278 examples [00:03, 1148.72 examples/s]4394 examples [00:03, 1149.52 examples/s]4511 examples [00:04, 1154.36 examples/s]4632 examples [00:04, 1168.95 examples/s]4750 examples [00:04, 1166.58 examples/s]4868 examples [00:04, 1169.59 examples/s]4987 examples [00:04, 1173.99 examples/s]5106 examples [00:04, 1177.42 examples/s]5225 examples [00:04, 1180.47 examples/s]5344 examples [00:04, 1165.63 examples/s]5461 examples [00:04, 1142.74 examples/s]5579 examples [00:04, 1152.26 examples/s]5697 examples [00:05, 1160.27 examples/s]5815 examples [00:05, 1163.44 examples/s]5933 examples [00:05, 1167.91 examples/s]6050 examples [00:05, 1165.21 examples/s]6171 examples [00:05, 1176.25 examples/s]6291 examples [00:05, 1181.65 examples/s]6410 examples [00:05, 1172.40 examples/s]6528 examples [00:05, 1138.94 examples/s]6643 examples [00:05, 1128.59 examples/s]6764 examples [00:05, 1149.57 examples/s]6888 examples [00:06, 1172.90 examples/s]7006 examples [00:06, 1165.87 examples/s]7131 examples [00:06, 1189.10 examples/s]7251 examples [00:06, 1183.04 examples/s]7370 examples [00:06, 1182.40 examples/s]7489 examples [00:06, 1146.28 examples/s]7604 examples [00:06, 1136.48 examples/s]7718 examples [00:06, 1130.34 examples/s]7832 examples [00:06, 1115.73 examples/s]7950 examples [00:06, 1133.79 examples/s]8065 examples [00:07, 1136.21 examples/s]8179 examples [00:07, 1125.34 examples/s]8292 examples [00:07, 1108.17 examples/s]8403 examples [00:07, 1062.92 examples/s]8510 examples [00:07, 1055.02 examples/s]8623 examples [00:07, 1076.09 examples/s]8734 examples [00:07, 1085.51 examples/s]8843 examples [00:07, 1045.25 examples/s]8957 examples [00:07, 1071.91 examples/s]9072 examples [00:08, 1092.60 examples/s]9182 examples [00:08, 1087.18 examples/s]9294 examples [00:08, 1094.73 examples/s]9404 examples [00:08, 1068.36 examples/s]9513 examples [00:08, 1072.45 examples/s]9621 examples [00:08, 1069.86 examples/s]9733 examples [00:08, 1082.46 examples/s]9846 examples [00:08, 1094.67 examples/s]9960 examples [00:08, 1107.56 examples/s]10071 examples [00:08, 1058.62 examples/s]10184 examples [00:09, 1077.41 examples/s]10293 examples [00:09, 1072.82 examples/s]10401 examples [00:09, 1074.82 examples/s]10515 examples [00:09, 1092.62 examples/s]10632 examples [00:09, 1113.43 examples/s]10748 examples [00:09, 1125.69 examples/s]10862 examples [00:09, 1128.59 examples/s]10976 examples [00:09, 1103.18 examples/s]11087 examples [00:09, 1086.03 examples/s]11202 examples [00:09, 1104.30 examples/s]11313 examples [00:10, 1105.93 examples/s]11428 examples [00:10, 1117.29 examples/s]11540 examples [00:10, 1100.04 examples/s]11657 examples [00:10, 1117.61 examples/s]11771 examples [00:10, 1122.67 examples/s]11887 examples [00:10, 1133.24 examples/s]12001 examples [00:10, 1124.27 examples/s]12114 examples [00:10, 1117.18 examples/s]12226 examples [00:10, 1114.01 examples/s]12341 examples [00:11, 1121.87 examples/s]12457 examples [00:11, 1132.63 examples/s]12571 examples [00:11, 1128.20 examples/s]12684 examples [00:11, 1124.70 examples/s]12797 examples [00:11, 1124.53 examples/s]12910 examples [00:11, 1108.64 examples/s]13023 examples [00:11, 1112.86 examples/s]13138 examples [00:11, 1122.62 examples/s]13252 examples [00:11, 1127.21 examples/s]13367 examples [00:11, 1130.83 examples/s]13481 examples [00:12, 1108.22 examples/s]13597 examples [00:12, 1122.06 examples/s]13713 examples [00:12, 1130.84 examples/s]13827 examples [00:12, 1128.86 examples/s]13945 examples [00:12, 1141.19 examples/s]14061 examples [00:12, 1144.10 examples/s]14177 examples [00:12, 1147.02 examples/s]14292 examples [00:12, 1130.79 examples/s]14406 examples [00:12, 1104.56 examples/s]14517 examples [00:12, 1102.20 examples/s]14630 examples [00:13, 1108.68 examples/s]14745 examples [00:13, 1118.94 examples/s]14861 examples [00:13, 1129.40 examples/s]14975 examples [00:13, 1129.87 examples/s]15089 examples [00:13, 1091.37 examples/s]15202 examples [00:13, 1102.18 examples/s]15313 examples [00:13, 1100.89 examples/s]15424 examples [00:13, 1074.58 examples/s]15540 examples [00:13, 1098.57 examples/s]15658 examples [00:13, 1121.50 examples/s]15771 examples [00:14, 1122.41 examples/s]15890 examples [00:14, 1140.90 examples/s]16005 examples [00:14, 1127.72 examples/s]16126 examples [00:14, 1148.81 examples/s]16249 examples [00:14, 1171.79 examples/s]16372 examples [00:14, 1187.01 examples/s]16495 examples [00:14, 1197.17 examples/s]16617 examples [00:14, 1203.26 examples/s]16738 examples [00:14, 1194.47 examples/s]16858 examples [00:14, 1195.14 examples/s]16982 examples [00:15, 1207.28 examples/s]17103 examples [00:15, 1199.80 examples/s]17224 examples [00:15, 1180.82 examples/s]17345 examples [00:15, 1188.89 examples/s]17467 examples [00:15, 1195.50 examples/s]17587 examples [00:15, 1169.62 examples/s]17705 examples [00:15, 1167.93 examples/s]17822 examples [00:15, 1166.62 examples/s]17939 examples [00:15, 1132.35 examples/s]18054 examples [00:16, 1137.52 examples/s]18178 examples [00:16, 1164.97 examples/s]18303 examples [00:16, 1188.01 examples/s]18426 examples [00:16, 1199.31 examples/s]18547 examples [00:16, 1196.06 examples/s]18667 examples [00:16, 1150.42 examples/s]18789 examples [00:16, 1168.21 examples/s]18911 examples [00:16, 1181.27 examples/s]19031 examples [00:16, 1186.23 examples/s]19152 examples [00:16, 1191.69 examples/s]19272 examples [00:17, 1193.95 examples/s]19398 examples [00:17, 1210.73 examples/s]19525 examples [00:17, 1226.26 examples/s]19648 examples [00:17, 1225.52 examples/s]19772 examples [00:17, 1228.09 examples/s]19895 examples [00:17, 1206.21 examples/s]20016 examples [00:17, 1138.35 examples/s]20136 examples [00:17, 1155.91 examples/s]20253 examples [00:17, 1156.06 examples/s]20370 examples [00:17, 1143.32 examples/s]20491 examples [00:18, 1160.42 examples/s]20613 examples [00:18, 1176.61 examples/s]20731 examples [00:18, 1173.73 examples/s]20850 examples [00:18, 1177.69 examples/s]20972 examples [00:18, 1189.04 examples/s]21095 examples [00:18, 1199.22 examples/s]21216 examples [00:18, 1194.43 examples/s]21336 examples [00:18, 1181.20 examples/s]21455 examples [00:18, 1161.89 examples/s]21579 examples [00:18, 1181.88 examples/s]21701 examples [00:19, 1191.17 examples/s]21821 examples [00:19, 1183.74 examples/s]21940 examples [00:19, 1162.63 examples/s]22057 examples [00:19, 1156.81 examples/s]22182 examples [00:19, 1182.81 examples/s]22301 examples [00:19, 1150.04 examples/s]22417 examples [00:19, 1140.38 examples/s]22534 examples [00:19, 1148.46 examples/s]22650 examples [00:19, 1146.34 examples/s]22774 examples [00:20, 1170.93 examples/s]22899 examples [00:20, 1191.88 examples/s]23022 examples [00:20, 1200.88 examples/s]23145 examples [00:20, 1209.24 examples/s]23267 examples [00:20, 1209.52 examples/s]23391 examples [00:20, 1216.84 examples/s]23514 examples [00:20, 1219.77 examples/s]23638 examples [00:20, 1222.96 examples/s]23764 examples [00:20, 1232.50 examples/s]23888 examples [00:20, 1196.98 examples/s]24008 examples [00:21, 1183.74 examples/s]24127 examples [00:21, 1171.92 examples/s]24250 examples [00:21, 1188.56 examples/s]24370 examples [00:21, 1157.79 examples/s]24490 examples [00:21, 1167.07 examples/s]24614 examples [00:21, 1186.09 examples/s]24736 examples [00:21, 1195.30 examples/s]24859 examples [00:21, 1204.41 examples/s]24980 examples [00:21, 1183.69 examples/s]25101 examples [00:21, 1189.96 examples/s]25227 examples [00:22, 1207.52 examples/s]25351 examples [00:22, 1215.46 examples/s]25476 examples [00:22, 1223.67 examples/s]25600 examples [00:22, 1227.57 examples/s]25723 examples [00:22, 1219.77 examples/s]25846 examples [00:22, 1214.06 examples/s]25968 examples [00:22, 1159.80 examples/s]26089 examples [00:22, 1173.28 examples/s]26209 examples [00:22, 1179.68 examples/s]26328 examples [00:22, 1173.10 examples/s]26448 examples [00:23, 1178.38 examples/s]26566 examples [00:23, 1151.87 examples/s]26688 examples [00:23, 1171.30 examples/s]26810 examples [00:23, 1185.16 examples/s]26929 examples [00:23, 1184.79 examples/s]27053 examples [00:23, 1199.84 examples/s]27176 examples [00:23, 1206.50 examples/s]27297 examples [00:23, 1199.53 examples/s]27418 examples [00:23, 1190.76 examples/s]27538 examples [00:23, 1169.30 examples/s]27656 examples [00:24, 1171.84 examples/s]27774 examples [00:24, 1158.31 examples/s]27890 examples [00:24, 1090.27 examples/s]28000 examples [00:24, 1083.83 examples/s]28111 examples [00:24, 1089.47 examples/s]28234 examples [00:24, 1125.87 examples/s]28352 examples [00:24, 1140.62 examples/s]28475 examples [00:24, 1166.03 examples/s]28596 examples [00:24, 1176.87 examples/s]28718 examples [00:25, 1188.77 examples/s]28843 examples [00:25, 1205.15 examples/s]28964 examples [00:25, 1206.53 examples/s]29088 examples [00:25, 1215.38 examples/s]29210 examples [00:25, 1216.19 examples/s]29332 examples [00:25, 1207.87 examples/s]29455 examples [00:25, 1214.35 examples/s]29577 examples [00:25, 1133.62 examples/s]29702 examples [00:25, 1164.67 examples/s]29820 examples [00:25, 1146.10 examples/s]29938 examples [00:26, 1155.33 examples/s]30055 examples [00:26, 1066.59 examples/s]30176 examples [00:26, 1105.44 examples/s]30296 examples [00:26, 1130.75 examples/s]30411 examples [00:26, 1132.53 examples/s]30529 examples [00:26, 1144.29 examples/s]30645 examples [00:26, 1138.55 examples/s]30770 examples [00:26, 1167.11 examples/s]30889 examples [00:26, 1173.79 examples/s]31007 examples [00:27, 1151.74 examples/s]31123 examples [00:27, 1144.30 examples/s]31243 examples [00:27, 1157.99 examples/s]31365 examples [00:27, 1174.12 examples/s]31486 examples [00:27, 1182.79 examples/s]31605 examples [00:27, 1181.71 examples/s]31724 examples [00:27, 1164.09 examples/s]31841 examples [00:27, 1149.18 examples/s]31962 examples [00:27, 1164.18 examples/s]32079 examples [00:27, 1153.66 examples/s]32195 examples [00:28, 1140.96 examples/s]32312 examples [00:28, 1147.70 examples/s]32427 examples [00:28, 1109.62 examples/s]32544 examples [00:28, 1125.03 examples/s]32670 examples [00:28, 1160.75 examples/s]32788 examples [00:28, 1164.38 examples/s]32908 examples [00:28, 1174.27 examples/s]33026 examples [00:28, 1135.15 examples/s]33147 examples [00:28, 1154.81 examples/s]33266 examples [00:28, 1162.89 examples/s]33384 examples [00:29, 1167.66 examples/s]33502 examples [00:29, 1170.35 examples/s]33620 examples [00:29, 1163.46 examples/s]33744 examples [00:29, 1184.59 examples/s]33866 examples [00:29, 1192.73 examples/s]33986 examples [00:29, 1184.16 examples/s]34107 examples [00:29, 1191.55 examples/s]34227 examples [00:29, 1176.87 examples/s]34345 examples [00:29, 1176.67 examples/s]34464 examples [00:29, 1180.38 examples/s]34583 examples [00:30, 1182.23 examples/s]34704 examples [00:30, 1188.07 examples/s]34826 examples [00:30, 1195.83 examples/s]34946 examples [00:30, 1188.82 examples/s]35070 examples [00:30, 1203.64 examples/s]35191 examples [00:30, 1189.15 examples/s]35313 examples [00:30, 1197.32 examples/s]35436 examples [00:30, 1206.44 examples/s]35557 examples [00:30, 1197.06 examples/s]35679 examples [00:30, 1203.14 examples/s]35800 examples [00:31, 1204.30 examples/s]35921 examples [00:31, 1201.89 examples/s]36043 examples [00:31, 1206.84 examples/s]36165 examples [00:31, 1209.96 examples/s]36287 examples [00:31, 1192.41 examples/s]36407 examples [00:31, 1166.08 examples/s]36524 examples [00:31, 1130.82 examples/s]36638 examples [00:31, 1110.91 examples/s]36761 examples [00:31, 1141.99 examples/s]36882 examples [00:32, 1161.34 examples/s]37002 examples [00:32, 1171.81 examples/s]37122 examples [00:32, 1178.43 examples/s]37247 examples [00:32, 1196.61 examples/s]37371 examples [00:32, 1207.82 examples/s]37495 examples [00:32, 1216.46 examples/s]37617 examples [00:32, 1183.85 examples/s]37736 examples [00:32, 1165.20 examples/s]37857 examples [00:32, 1175.92 examples/s]37979 examples [00:32, 1187.35 examples/s]38099 examples [00:33, 1189.79 examples/s]38219 examples [00:33, 1180.10 examples/s]38338 examples [00:33, 1174.03 examples/s]38458 examples [00:33, 1180.04 examples/s]38581 examples [00:33, 1193.80 examples/s]38701 examples [00:33, 1191.21 examples/s]38821 examples [00:33, 1157.75 examples/s]38938 examples [00:33, 1129.45 examples/s]39052 examples [00:33, 1113.09 examples/s]39164 examples [00:33, 1113.27 examples/s]39278 examples [00:34, 1120.16 examples/s]39392 examples [00:34, 1125.55 examples/s]39509 examples [00:34, 1136.23 examples/s]39627 examples [00:34, 1147.46 examples/s]39742 examples [00:34, 1145.43 examples/s]39860 examples [00:34, 1154.40 examples/s]39976 examples [00:34, 1148.16 examples/s]40091 examples [00:34, 1098.83 examples/s]40202 examples [00:34, 1056.44 examples/s]40310 examples [00:35, 1062.87 examples/s]40430 examples [00:35, 1097.94 examples/s]40541 examples [00:35, 1100.46 examples/s]40659 examples [00:35, 1123.07 examples/s]40778 examples [00:35, 1139.82 examples/s]40893 examples [00:35, 1136.95 examples/s]41007 examples [00:35, 1127.88 examples/s]41120 examples [00:35, 1088.31 examples/s]41241 examples [00:35, 1121.49 examples/s]41365 examples [00:35, 1154.52 examples/s]41482 examples [00:36, 1156.38 examples/s]41599 examples [00:36, 1143.47 examples/s]41714 examples [00:36, 1131.96 examples/s]41831 examples [00:36, 1142.64 examples/s]41952 examples [00:36, 1161.90 examples/s]42069 examples [00:36, 1135.68 examples/s]42183 examples [00:36, 1122.44 examples/s]42296 examples [00:36, 1107.20 examples/s]42413 examples [00:36, 1123.26 examples/s]42527 examples [00:36, 1125.55 examples/s]42640 examples [00:37, 1125.79 examples/s]42761 examples [00:37, 1146.41 examples/s]42881 examples [00:37, 1161.87 examples/s]43005 examples [00:37, 1182.37 examples/s]43128 examples [00:37, 1194.59 examples/s]43248 examples [00:37, 1184.40 examples/s]43367 examples [00:37, 1169.46 examples/s]43485 examples [00:37, 1159.13 examples/s]43602 examples [00:37, 1150.71 examples/s]43718 examples [00:37, 1128.55 examples/s]43835 examples [00:38, 1138.55 examples/s]43950 examples [00:38, 1141.87 examples/s]44065 examples [00:38, 1124.09 examples/s]44178 examples [00:38, 1103.46 examples/s]44289 examples [00:38, 1101.05 examples/s]44400 examples [00:38, 1100.13 examples/s]44512 examples [00:38, 1104.67 examples/s]44626 examples [00:38, 1112.83 examples/s]44743 examples [00:38, 1128.27 examples/s]44859 examples [00:38, 1137.30 examples/s]44976 examples [00:39, 1145.26 examples/s]45092 examples [00:39, 1147.98 examples/s]45207 examples [00:39, 1147.31 examples/s]45330 examples [00:39, 1169.85 examples/s]45451 examples [00:39, 1180.45 examples/s]45570 examples [00:39, 1177.71 examples/s]45688 examples [00:39, 1124.23 examples/s]45807 examples [00:39, 1142.01 examples/s]45923 examples [00:39, 1146.96 examples/s]46039 examples [00:40, 1142.63 examples/s]46161 examples [00:40, 1163.16 examples/s]46281 examples [00:40, 1172.59 examples/s]46400 examples [00:40, 1174.61 examples/s]46518 examples [00:40, 1172.70 examples/s]46642 examples [00:40, 1191.17 examples/s]46762 examples [00:40, 1192.85 examples/s]46882 examples [00:40, 1192.80 examples/s]47002 examples [00:40, 1188.76 examples/s]47123 examples [00:40, 1192.37 examples/s]47243 examples [00:41, 1156.80 examples/s]47365 examples [00:41, 1173.83 examples/s]47487 examples [00:41, 1186.75 examples/s]47606 examples [00:41, 1183.88 examples/s]47728 examples [00:41, 1192.00 examples/s]47849 examples [00:41, 1194.55 examples/s]47969 examples [00:41, 1154.14 examples/s]48091 examples [00:41, 1170.41 examples/s]48209 examples [00:41, 1170.16 examples/s]48329 examples [00:41, 1178.82 examples/s]48450 examples [00:42, 1185.87 examples/s]48571 examples [00:42, 1190.70 examples/s]48691 examples [00:42, 1184.43 examples/s]48810 examples [00:42, 1182.73 examples/s]48932 examples [00:42, 1190.92 examples/s]49055 examples [00:42, 1200.30 examples/s]49176 examples [00:42, 1189.21 examples/s]49295 examples [00:42, 1168.50 examples/s]49414 examples [00:42, 1173.81 examples/s]49534 examples [00:42, 1179.76 examples/s]49653 examples [00:43, 1181.66 examples/s]49772 examples [00:43, 1174.07 examples/s]49890 examples [00:43, 1165.85 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 19%|█▉        | 9375/50000 [00:00<00:00, 93747.85 examples/s] 48%|████▊     | 24067/50000 [00:00<00:00, 105165.02 examples/s] 77%|███████▋  | 38349/50000 [00:00<00:00, 114196.97 examples/s]                                                                0 examples [00:00, ? examples/s]96 examples [00:00, 953.68 examples/s]215 examples [00:00, 1012.06 examples/s]328 examples [00:00, 1042.77 examples/s]450 examples [00:00, 1089.56 examples/s]574 examples [00:00, 1130.62 examples/s]695 examples [00:00, 1152.40 examples/s]819 examples [00:00, 1175.28 examples/s]940 examples [00:00, 1183.38 examples/s]1059 examples [00:00, 1184.89 examples/s]1175 examples [00:01, 1173.56 examples/s]1290 examples [00:01, 1153.72 examples/s]1407 examples [00:01, 1156.54 examples/s]1532 examples [00:01, 1181.70 examples/s]1653 examples [00:01, 1188.85 examples/s]1774 examples [00:01, 1194.12 examples/s]1894 examples [00:01, 1191.37 examples/s]2018 examples [00:01, 1204.64 examples/s]2139 examples [00:01, 1171.14 examples/s]2257 examples [00:01, 1134.65 examples/s]2381 examples [00:02, 1161.82 examples/s]2498 examples [00:02, 1158.52 examples/s]2616 examples [00:02, 1163.83 examples/s]2733 examples [00:02, 1163.42 examples/s]2850 examples [00:02, 1142.33 examples/s]2968 examples [00:02, 1151.79 examples/s]3088 examples [00:02, 1163.13 examples/s]3210 examples [00:02, 1178.88 examples/s]3329 examples [00:02, 1178.91 examples/s]3447 examples [00:02, 1171.41 examples/s]3571 examples [00:03, 1188.52 examples/s]3692 examples [00:03, 1192.79 examples/s]3818 examples [00:03, 1209.55 examples/s]3940 examples [00:03, 1163.90 examples/s]4057 examples [00:03, 1157.77 examples/s]4174 examples [00:03, 1149.63 examples/s]4292 examples [00:03, 1158.51 examples/s]4412 examples [00:03, 1167.76 examples/s]4535 examples [00:03, 1183.49 examples/s]4654 examples [00:03, 1112.37 examples/s]4768 examples [00:04, 1118.92 examples/s]4882 examples [00:04, 1124.46 examples/s]5004 examples [00:04, 1149.61 examples/s]5122 examples [00:04, 1156.80 examples/s]5239 examples [00:04, 1151.61 examples/s]5358 examples [00:04, 1162.10 examples/s]5475 examples [00:04, 1158.38 examples/s]5600 examples [00:04, 1181.68 examples/s]5719 examples [00:04, 1180.09 examples/s]5838 examples [00:05, 1178.08 examples/s]5956 examples [00:05, 1153.33 examples/s]6079 examples [00:05, 1172.65 examples/s]6203 examples [00:05, 1190.91 examples/s]6326 examples [00:05, 1200.76 examples/s]6447 examples [00:05, 1179.05 examples/s]6568 examples [00:05, 1184.63 examples/s]6691 examples [00:05, 1195.58 examples/s]6814 examples [00:05, 1205.54 examples/s]6935 examples [00:05, 1202.51 examples/s]7057 examples [00:06, 1207.61 examples/s]7178 examples [00:06, 1139.50 examples/s]7294 examples [00:06, 1145.04 examples/s]7414 examples [00:06, 1158.49 examples/s]7531 examples [00:06, 1160.65 examples/s]7655 examples [00:06, 1183.09 examples/s]7776 examples [00:06, 1188.68 examples/s]7900 examples [00:06, 1201.07 examples/s]8023 examples [00:06, 1207.84 examples/s]8144 examples [00:06, 1161.02 examples/s]8263 examples [00:07, 1167.41 examples/s]8381 examples [00:07, 1163.45 examples/s]8504 examples [00:07, 1180.84 examples/s]8624 examples [00:07, 1186.30 examples/s]8743 examples [00:07, 1153.05 examples/s]8860 examples [00:07, 1155.21 examples/s]8976 examples [00:07, 1151.78 examples/s]9092 examples [00:07, 1142.82 examples/s]9210 examples [00:07, 1151.97 examples/s]9326 examples [00:07, 1149.48 examples/s]9446 examples [00:08, 1163.84 examples/s]9566 examples [00:08, 1173.19 examples/s]9688 examples [00:08, 1185.63 examples/s]9809 examples [00:08, 1192.79 examples/s]9929 examples [00:08, 1183.19 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2TKPT6/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete2TKPT6/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbf792b6840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbf792b6840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbf792b6840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.4516896104812622 	 Accuracy: 62
Train Epoch: 1 	 Loss: 6.63164119720459 	 Accuracy: 140
model saves at 140 accuracy
Train Epoch: 2 	 Loss: 0.41386138200759887 	 Accuracy: 78
Train Epoch: 2 	 Loss: 13.377769088745117 	 Accuracy: 200
model saves at 200 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbf498219d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbf498219d8>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fbf498219d8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbf4859d598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbf4859d598>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbf4859d598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([3, 9, 9, 5, 9, 5, 9, 0, 9, 9]), array([4, 5, 7, 5, 0, 0, 1, 8, 9, 1]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fbfa6cd6978> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fbf48347160> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbf40099048>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbf40099048>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbf40099048> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 40960/26421880 [00:00<01:17, 341032.82it/s]  0%|          | 106496/26421880 [00:00<01:12, 363268.46it/s]  1%|▏         | 376832/26421880 [00:00<00:53, 485203.99it/s]  4%|▎         | 950272/26421880 [00:00<00:38, 656753.27it/s] 10%|▉         | 2564096/26421880 [00:00<00:25, 922021.83it/s] 21%|██        | 5464064/26421880 [00:01<00:16, 1299263.27it/s] 35%|███▌      | 9314304/26421880 [00:01<00:09, 1828115.05it/s] 48%|████▊     | 12804096/26421880 [00:01<00:05, 2553547.38it/s] 60%|██████    | 15982592/26421880 [00:01<00:02, 3480743.81it/s] 76%|███████▌  | 20094976/26421880 [00:01<00:01, 4798414.56it/s] 90%|█████████ | 23879680/26421880 [00:01<00:00, 6499536.70it/s]26427392it [00:01, 16071543.00it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 94445.61it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 271494.72it/s]  4%|▍         | 172032/4422102 [00:00<00:12, 348874.51it/s] 10%|█         | 450560/4422102 [00:00<00:08, 460207.02it/s] 33%|███▎      | 1474560/4422102 [00:00<00:04, 642814.63it/s] 83%|████████▎ | 3653632/4422102 [00:00<00:00, 900776.62it/s]4423680it [00:01, 4408302.51it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 35126.44it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.00326530518134435 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019758899807929994 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0028828375935554505 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.025387633204460143 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.0019947830041249595 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.022077077984809877 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0027358360091845196 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.021479838252067566 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 5 	 Loss: 0.002422266125679016 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.012999752759933472 	 Accuracy: 5
model saves at 5 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbf4008f6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbf4008f6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbf4008f6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([5, 8, 4, 5, 0, 4, 3, 5, 7, 3]), array([5, 8, 4, 5, 0, 4, 0, 7, 7, 3]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fbfa6cd6fd0> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbf4828bea0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbf4828bea0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbf4828bea0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110443.12it/s]  0%|          | 40960/18165135 [00:00<02:27, 122585.91it/s]  1%|          | 98304/18165135 [00:00<01:57, 154107.22it/s]  1%|          | 212992/18165135 [00:01<01:28, 202763.70it/s]  2%|▏         | 425984/18165135 [00:01<01:04, 273056.51it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 375026.12it/s] 10%|▉         | 1736704/18165135 [00:01<00:31, 521500.55it/s] 19%|█▉        | 3473408/18165135 [00:01<00:20, 730979.54it/s] 31%|███▏      | 5685248/18165135 [00:01<00:12, 1010534.41it/s] 40%|███▉      | 7176192/18165135 [00:01<00:07, 1402531.68it/s] 45%|████▍     | 8085504/18165135 [00:02<00:05, 1757870.74it/s] 51%|█████     | 9183232/18165135 [00:02<00:04, 2087791.25it/s] 62%|██████▏   | 11345920/18165135 [00:02<00:02, 2863361.26it/s] 69%|██████▊   | 12443648/18165135 [00:02<00:01, 3081496.52it/s] 82%|████████▏ | 14835712/18165135 [00:02<00:00, 4154901.25it/s] 89%|████████▉ | 16121856/18165135 [00:03<00:00, 4704270.49it/s] 95%|█████████▌| 17260544/18165135 [00:03<00:00, 5633350.18it/s]18169856it [00:03, 5362878.23it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110193.06it/s]32768it [00:00, 52917.17it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110214.09it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122393.53it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153890.70it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 201267.46it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 271718.92it/s] 28%|██▊       | 860160/3041136 [00:01<00:05, 373205.79it/s] 57%|█████▋    | 1720320/3041136 [00:01<00:02, 518776.13it/s] 97%|█████████▋| 2949120/3041136 [00:01<00:00, 721648.78it/s]3047424it [00:01, 1829401.40it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17407.00it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004241213500499726 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.021274624943733215 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0033382355372111004 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.047826371431350705 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.0032524852454662322 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.046948672294616696 	 Accuracy: 2
Train Epoch: 4 	 Loss: 0.0029056810935338337 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.03253430509567261 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0035334856907526654 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.023123254656791686 	 Accuracy: 3

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbf4828b730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbf4828b730>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fbf4828b730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([6, 5, 1, 2, 2, 9, 5, 6, 6, 7]), array([7, 3, 6, 2, 9, 2, 8, 6, 1, 3]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

