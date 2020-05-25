
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f80672162e8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f806624e400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f806624e400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f806624e400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 148390.47it/s] 66%|██████▋   | 6569984/9912422 [00:00<00:15, 211775.42it/s]9920512it [00:00, 40421415.24it/s]                           
0it [00:00, ?it/s]32768it [00:00, 586603.93it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:09, 163617.02it/s]1654784it [00:00, 11462684.87it/s]                         
0it [00:00, ?it/s]8192it [00:00, 190525.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003651718686024348 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020982644081115723 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.002905939588944117 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.01586047315597534 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.0020574920773506165 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.02194436639547348 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0018595914815862974 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.015040547370910644 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 5 	 Loss: 0.0011946581602096558 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.00899277400970459 	 Accuracy: 7
model saves at 7 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f806623af28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f806623af28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f806623af28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([9, 0, 9, 4, 1, 3, 4, 9, 1, 1]), array([9, 0, 9, 4, 1, 8, 1, 9, 1, 1]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f80680c74a8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f80608d0b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f80608d0b70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f80608d0b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0021082589228947957 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.016557615995407105 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0021792054971059164 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.032717459678649904 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f80608d0400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f80608d0400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f80608d0400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([3, 4, 9, 7, 0, 6, 6, 6, 6, 7]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 15%|█▍        | 6.52M/44.7M [00:00<00:00, 68.3MB/s] 30%|███       | 13.5M/44.7M [00:00<00:00, 69.2MB/s] 49%|████▉     | 22.1M/44.7M [00:00<00:00, 74.1MB/s] 69%|██████▉   | 30.9M/44.7M [00:00<00:00, 78.8MB/s] 88%|████████▊ | 39.3M/44.7M [00:00<00:00, 81.3MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 81.3MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f80d4434390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f80662241e0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f80662241e0>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f80662241e0> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:17,  7.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 47.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 53.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 57.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 57.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.81s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 67.31 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.21s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.81s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 67.31 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.21s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.21s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.09 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.21s/ url]
0 examples [00:00, ? examples/s]2020-05-25 00:09:37.813675: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 00:09:37.856070: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-25 00:09:37.856256: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b911c48680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 00:09:37.856276: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.25 examples/s]97 examples [00:00,  1.79 examples/s]193 examples [00:00,  2.56 examples/s]285 examples [00:01,  3.65 examples/s]372 examples [00:01,  5.20 examples/s]462 examples [00:01,  7.41 examples/s]556 examples [00:01, 10.55 examples/s]654 examples [00:01, 15.01 examples/s]750 examples [00:01, 21.29 examples/s]839 examples [00:01, 30.10 examples/s]932 examples [00:01, 42.41 examples/s]1026 examples [00:01, 59.43 examples/s]1121 examples [00:02, 82.64 examples/s]1213 examples [00:02, 113.63 examples/s]1310 examples [00:02, 154.49 examples/s]1403 examples [00:02, 205.97 examples/s]1496 examples [00:02, 267.05 examples/s]1589 examples [00:02, 339.59 examples/s]1684 examples [00:02, 420.37 examples/s]1783 examples [00:02, 507.46 examples/s]1882 examples [00:02, 593.71 examples/s]1978 examples [00:02, 667.66 examples/s]2075 examples [00:03, 735.06 examples/s]2177 examples [00:03, 801.58 examples/s]2278 examples [00:03, 853.59 examples/s]2377 examples [00:03, 888.42 examples/s]2476 examples [00:03, 915.82 examples/s]2577 examples [00:03, 942.12 examples/s]2678 examples [00:03, 959.27 examples/s]2778 examples [00:03, 931.38 examples/s]2874 examples [00:03, 903.37 examples/s]2967 examples [00:03, 896.09 examples/s]3059 examples [00:04, 879.25 examples/s]3149 examples [00:04, 869.26 examples/s]3247 examples [00:04, 899.47 examples/s]3346 examples [00:04, 923.24 examples/s]3448 examples [00:04, 947.92 examples/s]3548 examples [00:04, 961.63 examples/s]3648 examples [00:04, 972.26 examples/s]3749 examples [00:04, 982.57 examples/s]3848 examples [00:04, 979.77 examples/s]3947 examples [00:04, 959.45 examples/s]4044 examples [00:05, 931.64 examples/s]4138 examples [00:05, 916.71 examples/s]4239 examples [00:05, 940.92 examples/s]4335 examples [00:05, 944.44 examples/s]4434 examples [00:05, 955.43 examples/s]4530 examples [00:05, 952.05 examples/s]4626 examples [00:05, 903.99 examples/s]4718 examples [00:05, 890.05 examples/s]4808 examples [00:05, 884.86 examples/s]4898 examples [00:06, 887.51 examples/s]4992 examples [00:06, 899.66 examples/s]5088 examples [00:06, 916.50 examples/s]5185 examples [00:06, 931.64 examples/s]5279 examples [00:06, 932.19 examples/s]5373 examples [00:06, 932.87 examples/s]5467 examples [00:06, 924.32 examples/s]5566 examples [00:06, 941.26 examples/s]5661 examples [00:06, 924.65 examples/s]5754 examples [00:06, 904.29 examples/s]5845 examples [00:07, 900.47 examples/s]5937 examples [00:07, 903.80 examples/s]6028 examples [00:07, 897.82 examples/s]6118 examples [00:07, 897.92 examples/s]6211 examples [00:07, 905.92 examples/s]6308 examples [00:07, 924.02 examples/s]6401 examples [00:07, 912.26 examples/s]6493 examples [00:07, 902.43 examples/s]6588 examples [00:07, 912.24 examples/s]6686 examples [00:07, 929.55 examples/s]6784 examples [00:08, 942.74 examples/s]6884 examples [00:08, 959.02 examples/s]6985 examples [00:08, 971.13 examples/s]7083 examples [00:08, 972.52 examples/s]7181 examples [00:08, 946.31 examples/s]7280 examples [00:08, 958.20 examples/s]7380 examples [00:08, 968.95 examples/s]7479 examples [00:08, 974.66 examples/s]7577 examples [00:08, 969.81 examples/s]7675 examples [00:09, 931.69 examples/s]7771 examples [00:09, 938.63 examples/s]7866 examples [00:09, 930.27 examples/s]7963 examples [00:09, 941.29 examples/s]8060 examples [00:09, 949.49 examples/s]8156 examples [00:09, 908.38 examples/s]8248 examples [00:09, 890.05 examples/s]8345 examples [00:09, 912.36 examples/s]8440 examples [00:09, 921.42 examples/s]8535 examples [00:09, 927.59 examples/s]8628 examples [00:10, 927.24 examples/s]8721 examples [00:10, 923.79 examples/s]8818 examples [00:10, 934.43 examples/s]8915 examples [00:10, 943.32 examples/s]9010 examples [00:10, 942.75 examples/s]9105 examples [00:10, 939.00 examples/s]9200 examples [00:10, 941.70 examples/s]9299 examples [00:10, 953.36 examples/s]9400 examples [00:10, 968.51 examples/s]9499 examples [00:10, 972.28 examples/s]9597 examples [00:11, 936.03 examples/s]9691 examples [00:11, 922.00 examples/s]9788 examples [00:11, 935.04 examples/s]9883 examples [00:11, 939.03 examples/s]9978 examples [00:11, 885.32 examples/s]10068 examples [00:12, 322.96 examples/s]10160 examples [00:12, 400.87 examples/s]10255 examples [00:12, 484.34 examples/s]10344 examples [00:12, 559.76 examples/s]10435 examples [00:12, 631.89 examples/s]10527 examples [00:12, 695.90 examples/s]10618 examples [00:12, 748.05 examples/s]10713 examples [00:12, 798.77 examples/s]10803 examples [00:12, 823.49 examples/s]10893 examples [00:13, 841.47 examples/s]10983 examples [00:13, 846.54 examples/s]11078 examples [00:13, 873.76 examples/s]11180 examples [00:13, 912.63 examples/s]11281 examples [00:13, 937.60 examples/s]11377 examples [00:13, 920.07 examples/s]11471 examples [00:13, 920.76 examples/s]11565 examples [00:13, 922.11 examples/s]11658 examples [00:13, 914.71 examples/s]11752 examples [00:14, 921.49 examples/s]11845 examples [00:14, 906.56 examples/s]11936 examples [00:14, 893.27 examples/s]12026 examples [00:14, 886.52 examples/s]12115 examples [00:14, 865.34 examples/s]12210 examples [00:14, 887.12 examples/s]12300 examples [00:14, 886.03 examples/s]12389 examples [00:14, 882.80 examples/s]12488 examples [00:14, 911.36 examples/s]12589 examples [00:14, 936.75 examples/s]12684 examples [00:15, 919.67 examples/s]12777 examples [00:15, 916.08 examples/s]12879 examples [00:15, 943.42 examples/s]12981 examples [00:15, 962.50 examples/s]13078 examples [00:15, 948.68 examples/s]13176 examples [00:15, 957.80 examples/s]13275 examples [00:15, 965.00 examples/s]13373 examples [00:15, 968.81 examples/s]13471 examples [00:15, 970.21 examples/s]13570 examples [00:15, 974.47 examples/s]13668 examples [00:16, 968.65 examples/s]13765 examples [00:16, 968.61 examples/s]13862 examples [00:16, 935.67 examples/s]13958 examples [00:16, 940.93 examples/s]14057 examples [00:16, 952.96 examples/s]14153 examples [00:16, 951.72 examples/s]14252 examples [00:16, 962.81 examples/s]14352 examples [00:16, 973.02 examples/s]14450 examples [00:16, 974.14 examples/s]14548 examples [00:16, 971.99 examples/s]14646 examples [00:17, 955.68 examples/s]14742 examples [00:17, 954.42 examples/s]14838 examples [00:17, 953.53 examples/s]14934 examples [00:17, 926.21 examples/s]15027 examples [00:17, 914.76 examples/s]15123 examples [00:17, 927.76 examples/s]15218 examples [00:17, 933.28 examples/s]15320 examples [00:17, 955.44 examples/s]15416 examples [00:17, 935.77 examples/s]15516 examples [00:18, 953.26 examples/s]15612 examples [00:18, 945.59 examples/s]15710 examples [00:18, 952.68 examples/s]15806 examples [00:18, 951.42 examples/s]15903 examples [00:18, 954.63 examples/s]16001 examples [00:18, 959.44 examples/s]16098 examples [00:18, 945.94 examples/s]16195 examples [00:18, 952.23 examples/s]16292 examples [00:18, 955.44 examples/s]16389 examples [00:18, 957.73 examples/s]16486 examples [00:19, 960.84 examples/s]16583 examples [00:19, 954.04 examples/s]16679 examples [00:19, 952.05 examples/s]16775 examples [00:19, 923.71 examples/s]16873 examples [00:19, 937.55 examples/s]16973 examples [00:19, 953.60 examples/s]17069 examples [00:19, 937.23 examples/s]17163 examples [00:19, 919.39 examples/s]17256 examples [00:19, 920.45 examples/s]17349 examples [00:19, 887.03 examples/s]17439 examples [00:20, 879.95 examples/s]17535 examples [00:20, 900.38 examples/s]17637 examples [00:20, 930.98 examples/s]17731 examples [00:20, 932.67 examples/s]17829 examples [00:20, 945.23 examples/s]17924 examples [00:20, 945.18 examples/s]18019 examples [00:20, 932.32 examples/s]18113 examples [00:20, 921.29 examples/s]18206 examples [00:20, 912.76 examples/s]18298 examples [00:20, 908.09 examples/s]18396 examples [00:21, 926.64 examples/s]18489 examples [00:21, 924.16 examples/s]18591 examples [00:21, 949.49 examples/s]18692 examples [00:21, 965.18 examples/s]18790 examples [00:21, 968.77 examples/s]18888 examples [00:21, 962.37 examples/s]18985 examples [00:21, 940.96 examples/s]19080 examples [00:21, 922.30 examples/s]19173 examples [00:21, 896.46 examples/s]19263 examples [00:22, 877.14 examples/s]19352 examples [00:22, 874.61 examples/s]19448 examples [00:22, 896.22 examples/s]19538 examples [00:22, 877.32 examples/s]19630 examples [00:22, 887.69 examples/s]19728 examples [00:22, 911.51 examples/s]19822 examples [00:22, 918.30 examples/s]19918 examples [00:22, 928.53 examples/s]20012 examples [00:22, 906.40 examples/s]20107 examples [00:22, 916.91 examples/s]20199 examples [00:23, 907.02 examples/s]20292 examples [00:23, 913.46 examples/s]20390 examples [00:23, 930.21 examples/s]20484 examples [00:23, 905.66 examples/s]20575 examples [00:23, 885.95 examples/s]20664 examples [00:23, 870.66 examples/s]20753 examples [00:23, 875.30 examples/s]20852 examples [00:23, 906.47 examples/s]20950 examples [00:23, 926.62 examples/s]21047 examples [00:23, 937.32 examples/s]21142 examples [00:24, 936.01 examples/s]21240 examples [00:24, 947.20 examples/s]21338 examples [00:24, 954.65 examples/s]21435 examples [00:24, 957.99 examples/s]21533 examples [00:24, 961.70 examples/s]21631 examples [00:24, 967.01 examples/s]21729 examples [00:24, 968.39 examples/s]21827 examples [00:24, 969.20 examples/s]21927 examples [00:24, 977.29 examples/s]22025 examples [00:25, 969.40 examples/s]22122 examples [00:25, 933.65 examples/s]22216 examples [00:25, 926.89 examples/s]22311 examples [00:25, 932.82 examples/s]22405 examples [00:25, 919.62 examples/s]22502 examples [00:25, 932.38 examples/s]22596 examples [00:25, 917.77 examples/s]22691 examples [00:25, 926.97 examples/s]22789 examples [00:25, 941.23 examples/s]22890 examples [00:25, 958.57 examples/s]22988 examples [00:26, 963.68 examples/s]23085 examples [00:26, 956.01 examples/s]23181 examples [00:26, 947.43 examples/s]23282 examples [00:26, 964.48 examples/s]23379 examples [00:26, 957.26 examples/s]23475 examples [00:26, 954.53 examples/s]23571 examples [00:26, 943.47 examples/s]23666 examples [00:26, 944.46 examples/s]23765 examples [00:26, 957.02 examples/s]23863 examples [00:26, 963.28 examples/s]23962 examples [00:27, 969.40 examples/s]24059 examples [00:27, 963.52 examples/s]24156 examples [00:27, 954.72 examples/s]24252 examples [00:27, 939.86 examples/s]24347 examples [00:27, 940.56 examples/s]24446 examples [00:27, 952.56 examples/s]24542 examples [00:27, 948.52 examples/s]24637 examples [00:27, 946.73 examples/s]24733 examples [00:27, 948.35 examples/s]24828 examples [00:27, 944.15 examples/s]24927 examples [00:28, 956.78 examples/s]25023 examples [00:28, 949.94 examples/s]25119 examples [00:28, 921.51 examples/s]25212 examples [00:28, 913.83 examples/s]25304 examples [00:28, 892.02 examples/s]25404 examples [00:28, 920.98 examples/s]25503 examples [00:28, 940.35 examples/s]25598 examples [00:28, 936.69 examples/s]25699 examples [00:28, 957.02 examples/s]25798 examples [00:29, 963.51 examples/s]25895 examples [00:29, 957.81 examples/s]25995 examples [00:29, 967.83 examples/s]26092 examples [00:29, 938.55 examples/s]26192 examples [00:29, 953.86 examples/s]26291 examples [00:29, 961.84 examples/s]26389 examples [00:29, 966.04 examples/s]26486 examples [00:29, 956.09 examples/s]26582 examples [00:29, 923.14 examples/s]26677 examples [00:29, 928.55 examples/s]26771 examples [00:30, 930.20 examples/s]26866 examples [00:30, 936.00 examples/s]26961 examples [00:30, 937.70 examples/s]27055 examples [00:30, 938.23 examples/s]27149 examples [00:30, 934.41 examples/s]27245 examples [00:30, 941.17 examples/s]27344 examples [00:30, 954.11 examples/s]27440 examples [00:30, 955.52 examples/s]27536 examples [00:30, 955.20 examples/s]27636 examples [00:30, 966.39 examples/s]27733 examples [00:31, 924.59 examples/s]27826 examples [00:31, 919.65 examples/s]27925 examples [00:31, 938.36 examples/s]28020 examples [00:31, 939.92 examples/s]28121 examples [00:31, 958.21 examples/s]28218 examples [00:31, 934.44 examples/s]28312 examples [00:31, 930.75 examples/s]28406 examples [00:31, 909.35 examples/s]28498 examples [00:31, 897.19 examples/s]28591 examples [00:31, 904.49 examples/s]28682 examples [00:32, 905.55 examples/s]28773 examples [00:32, 901.52 examples/s]28867 examples [00:32, 911.59 examples/s]28964 examples [00:32, 926.44 examples/s]29060 examples [00:32, 935.93 examples/s]29154 examples [00:32, 928.93 examples/s]29247 examples [00:32, 923.13 examples/s]29341 examples [00:32, 926.68 examples/s]29434 examples [00:32, 918.60 examples/s]29530 examples [00:33, 928.59 examples/s]29623 examples [00:33, 923.22 examples/s]29716 examples [00:33, 919.50 examples/s]29808 examples [00:33, 868.67 examples/s]29901 examples [00:33, 885.70 examples/s]29998 examples [00:33, 907.55 examples/s]30090 examples [00:33, 856.35 examples/s]30187 examples [00:33, 885.24 examples/s]30283 examples [00:33, 906.03 examples/s]30383 examples [00:33, 930.32 examples/s]30483 examples [00:34, 949.35 examples/s]30583 examples [00:34, 962.44 examples/s]30680 examples [00:34, 945.44 examples/s]30784 examples [00:34, 969.82 examples/s]30882 examples [00:34, 968.55 examples/s]30980 examples [00:34, 943.96 examples/s]31078 examples [00:34, 953.04 examples/s]31175 examples [00:34, 957.54 examples/s]31271 examples [00:34, 948.14 examples/s]31366 examples [00:34, 924.38 examples/s]31462 examples [00:35, 933.52 examples/s]31556 examples [00:35, 906.90 examples/s]31654 examples [00:35, 926.82 examples/s]31748 examples [00:35, 927.71 examples/s]31841 examples [00:35, 921.02 examples/s]31934 examples [00:35, 911.01 examples/s]32026 examples [00:35, 908.89 examples/s]32118 examples [00:35, 911.52 examples/s]32210 examples [00:35, 885.68 examples/s]32303 examples [00:36, 895.46 examples/s]32398 examples [00:36, 910.11 examples/s]32496 examples [00:36, 928.02 examples/s]32594 examples [00:36, 941.95 examples/s]32693 examples [00:36, 954.02 examples/s]32793 examples [00:36, 965.05 examples/s]32894 examples [00:36, 976.48 examples/s]32993 examples [00:36, 980.25 examples/s]33092 examples [00:36, 958.12 examples/s]33188 examples [00:36, 942.34 examples/s]33283 examples [00:37, 927.42 examples/s]33376 examples [00:37, 922.09 examples/s]33469 examples [00:37, 919.92 examples/s]33562 examples [00:37, 917.39 examples/s]33654 examples [00:37, 912.90 examples/s]33751 examples [00:37, 928.78 examples/s]33850 examples [00:37, 944.10 examples/s]33951 examples [00:37, 961.83 examples/s]34048 examples [00:37, 954.76 examples/s]34144 examples [00:37, 913.01 examples/s]34237 examples [00:38, 916.42 examples/s]34331 examples [00:38, 920.72 examples/s]34424 examples [00:38, 908.26 examples/s]34516 examples [00:38, 873.51 examples/s]34604 examples [00:38, 851.67 examples/s]34692 examples [00:38, 857.51 examples/s]34781 examples [00:38, 865.31 examples/s]34878 examples [00:38, 892.00 examples/s]34974 examples [00:38, 909.53 examples/s]35067 examples [00:39, 915.51 examples/s]35166 examples [00:39, 935.86 examples/s]35262 examples [00:39, 941.53 examples/s]35357 examples [00:39, 911.29 examples/s]35451 examples [00:39, 919.71 examples/s]35544 examples [00:39, 898.92 examples/s]35635 examples [00:39, 894.06 examples/s]35725 examples [00:39, 884.82 examples/s]35815 examples [00:39, 887.09 examples/s]35906 examples [00:39, 891.52 examples/s]35996 examples [00:40, 879.25 examples/s]36085 examples [00:40, 882.13 examples/s]36185 examples [00:40, 912.05 examples/s]36286 examples [00:40, 938.08 examples/s]36384 examples [00:40, 948.73 examples/s]36480 examples [00:40, 939.78 examples/s]36581 examples [00:40, 958.93 examples/s]36678 examples [00:40, 948.22 examples/s]36777 examples [00:40, 959.74 examples/s]36876 examples [00:40, 965.85 examples/s]36973 examples [00:41, 929.75 examples/s]37067 examples [00:41, 932.51 examples/s]37163 examples [00:41, 938.50 examples/s]37261 examples [00:41, 950.43 examples/s]37357 examples [00:41, 942.01 examples/s]37452 examples [00:41, 920.89 examples/s]37545 examples [00:41, 921.01 examples/s]37638 examples [00:41, 898.90 examples/s]37729 examples [00:41, 894.33 examples/s]37823 examples [00:41, 906.06 examples/s]37920 examples [00:42, 922.54 examples/s]38013 examples [00:42, 914.71 examples/s]38105 examples [00:42, 915.42 examples/s]38202 examples [00:42, 928.19 examples/s]38298 examples [00:42, 934.56 examples/s]38395 examples [00:42, 942.85 examples/s]38494 examples [00:42, 956.23 examples/s]38595 examples [00:42, 970.15 examples/s]38695 examples [00:42, 975.59 examples/s]38793 examples [00:43, 961.41 examples/s]38890 examples [00:43, 943.63 examples/s]38985 examples [00:43, 925.34 examples/s]39078 examples [00:43, 897.24 examples/s]39169 examples [00:43, 874.14 examples/s]39267 examples [00:43, 903.19 examples/s]39358 examples [00:43, 881.19 examples/s]39459 examples [00:43, 914.28 examples/s]39561 examples [00:43, 942.22 examples/s]39658 examples [00:43, 947.80 examples/s]39754 examples [00:44, 903.35 examples/s]39846 examples [00:44, 887.06 examples/s]39936 examples [00:44, 878.70 examples/s]40025 examples [00:44, 848.06 examples/s]40117 examples [00:44, 866.38 examples/s]40207 examples [00:44, 875.70 examples/s]40306 examples [00:44, 905.80 examples/s]40406 examples [00:44, 931.00 examples/s]40504 examples [00:44, 941.42 examples/s]40600 examples [00:45, 944.89 examples/s]40696 examples [00:45, 948.35 examples/s]40793 examples [00:45, 952.43 examples/s]40889 examples [00:45, 949.03 examples/s]40985 examples [00:45, 933.95 examples/s]41079 examples [00:45, 915.21 examples/s]41171 examples [00:45, 897.85 examples/s]41262 examples [00:45, 899.69 examples/s]41353 examples [00:45, 899.41 examples/s]41444 examples [00:45, 900.13 examples/s]41535 examples [00:46, 865.98 examples/s]41625 examples [00:46, 874.49 examples/s]41719 examples [00:46, 891.85 examples/s]41810 examples [00:46, 895.91 examples/s]41900 examples [00:46, 895.95 examples/s]41992 examples [00:46, 901.04 examples/s]42083 examples [00:46, 879.90 examples/s]42183 examples [00:46, 911.81 examples/s]42279 examples [00:46, 925.39 examples/s]42376 examples [00:46, 936.51 examples/s]42474 examples [00:47, 948.96 examples/s]42570 examples [00:47, 910.82 examples/s]42667 examples [00:47, 926.71 examples/s]42765 examples [00:47, 939.92 examples/s]42865 examples [00:47, 954.52 examples/s]42964 examples [00:47, 963.65 examples/s]43061 examples [00:47, 955.21 examples/s]43161 examples [00:47, 967.19 examples/s]43262 examples [00:47, 976.31 examples/s]43362 examples [00:47, 981.35 examples/s]43461 examples [00:48, 981.53 examples/s]43560 examples [00:48, 955.87 examples/s]43656 examples [00:48, 942.41 examples/s]43751 examples [00:48, 942.96 examples/s]43850 examples [00:48, 956.19 examples/s]43946 examples [00:48, 952.86 examples/s]44042 examples [00:48, 942.41 examples/s]44137 examples [00:48, 944.33 examples/s]44232 examples [00:48, 945.95 examples/s]44327 examples [00:49, 924.05 examples/s]44426 examples [00:49, 940.82 examples/s]44521 examples [00:49, 913.77 examples/s]44620 examples [00:49, 934.10 examples/s]44719 examples [00:49, 948.91 examples/s]44815 examples [00:49, 947.63 examples/s]44910 examples [00:49, 934.86 examples/s]45004 examples [00:49, 910.52 examples/s]45096 examples [00:49, 898.32 examples/s]45188 examples [00:49, 903.73 examples/s]45279 examples [00:50, 901.88 examples/s]45370 examples [00:50, 894.41 examples/s]45460 examples [00:50, 888.96 examples/s]45559 examples [00:50, 916.11 examples/s]45655 examples [00:50, 927.51 examples/s]45748 examples [00:50, 908.75 examples/s]45844 examples [00:50, 921.32 examples/s]45938 examples [00:50, 924.44 examples/s]46035 examples [00:50, 936.71 examples/s]46129 examples [00:50, 923.82 examples/s]46222 examples [00:51, 916.68 examples/s]46314 examples [00:51, 900.95 examples/s]46405 examples [00:51, 886.88 examples/s]46505 examples [00:51, 917.12 examples/s]46606 examples [00:51, 941.86 examples/s]46703 examples [00:51, 948.14 examples/s]46799 examples [00:51, 933.62 examples/s]46893 examples [00:51, 900.63 examples/s]46985 examples [00:51, 903.91 examples/s]47076 examples [00:52, 903.79 examples/s]47167 examples [00:52, 895.63 examples/s]47257 examples [00:52, 859.40 examples/s]47344 examples [00:52, 843.02 examples/s]47429 examples [00:52, 834.45 examples/s]47514 examples [00:52, 837.09 examples/s]47603 examples [00:52, 846.36 examples/s]47695 examples [00:52, 866.83 examples/s]47792 examples [00:52, 893.41 examples/s]47889 examples [00:52, 913.01 examples/s]47981 examples [00:53, 896.80 examples/s]48071 examples [00:53, 891.22 examples/s]48161 examples [00:53, 857.55 examples/s]48255 examples [00:53, 878.82 examples/s]48350 examples [00:53, 897.80 examples/s]48449 examples [00:53, 921.21 examples/s]48548 examples [00:53, 940.06 examples/s]48643 examples [00:53, 938.06 examples/s]48740 examples [00:53, 946.46 examples/s]48835 examples [00:53, 945.73 examples/s]48930 examples [00:54, 943.07 examples/s]49025 examples [00:54, 938.72 examples/s]49119 examples [00:54, 937.06 examples/s]49213 examples [00:54, 924.23 examples/s]49313 examples [00:54, 944.14 examples/s]49410 examples [00:54, 951.41 examples/s]49511 examples [00:54, 967.50 examples/s]49608 examples [00:54, 965.29 examples/s]49708 examples [00:54, 974.61 examples/s]49806 examples [00:54, 964.97 examples/s]49903 examples [00:55, 922.71 examples/s]49999 examples [00:55, 931.37 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5712/50000 [00:00<00:00, 57117.19 examples/s] 31%|███       | 15467/50000 [00:00<00:00, 65226.75 examples/s] 51%|█████     | 25484/50000 [00:00<00:00, 72850.01 examples/s] 74%|███████▍  | 36966/50000 [00:00<00:00, 81821.57 examples/s] 97%|█████████▋| 48435/50000 [00:00<00:00, 89517.64 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 737.40 examples/s]169 examples [00:00, 789.50 examples/s]268 examples [00:00, 838.63 examples/s]361 examples [00:00, 860.74 examples/s]464 examples [00:00, 903.42 examples/s]545 examples [00:00, 871.33 examples/s]637 examples [00:00, 883.53 examples/s]735 examples [00:00, 907.93 examples/s]827 examples [00:00, 909.97 examples/s]925 examples [00:01, 928.91 examples/s]1019 examples [00:01, 929.05 examples/s]1113 examples [00:01, 928.12 examples/s]1206 examples [00:01, 912.40 examples/s]1300 examples [00:01, 918.01 examples/s]1398 examples [00:01, 935.48 examples/s]1494 examples [00:01, 941.60 examples/s]1589 examples [00:01, 939.36 examples/s]1683 examples [00:01, 937.90 examples/s]1777 examples [00:01, 913.80 examples/s]1869 examples [00:02, 903.84 examples/s]1960 examples [00:02, 901.59 examples/s]2051 examples [00:02, 904.09 examples/s]2142 examples [00:02, 903.26 examples/s]2234 examples [00:02, 906.72 examples/s]2325 examples [00:02, 897.11 examples/s]2415 examples [00:02, 892.60 examples/s]2508 examples [00:02, 901.38 examples/s]2608 examples [00:02, 927.76 examples/s]2709 examples [00:02, 949.68 examples/s]2808 examples [00:03, 961.33 examples/s]2905 examples [00:03, 951.91 examples/s]3001 examples [00:03, 935.75 examples/s]3095 examples [00:03, 923.47 examples/s]3188 examples [00:03, 916.06 examples/s]3280 examples [00:03, 887.86 examples/s]3370 examples [00:03, 864.63 examples/s]3457 examples [00:03, 852.77 examples/s]3545 examples [00:03, 857.89 examples/s]3637 examples [00:03, 874.66 examples/s]3730 examples [00:04, 890.28 examples/s]3829 examples [00:04, 917.85 examples/s]3925 examples [00:04, 927.60 examples/s]4025 examples [00:04, 945.43 examples/s]4123 examples [00:04, 953.87 examples/s]4219 examples [00:04, 942.50 examples/s]4317 examples [00:04, 953.33 examples/s]4413 examples [00:04, 945.59 examples/s]4508 examples [00:04, 941.56 examples/s]4605 examples [00:05, 949.53 examples/s]4701 examples [00:05, 948.63 examples/s]4800 examples [00:05, 959.93 examples/s]4897 examples [00:05, 951.68 examples/s]4995 examples [00:05, 958.20 examples/s]5093 examples [00:05, 961.99 examples/s]5190 examples [00:05, 958.05 examples/s]5291 examples [00:05, 971.55 examples/s]5389 examples [00:05, 973.82 examples/s]5492 examples [00:05, 988.84 examples/s]5591 examples [00:06, 976.65 examples/s]5689 examples [00:06, 958.19 examples/s]5785 examples [00:06, 935.92 examples/s]5879 examples [00:06, 927.48 examples/s]5972 examples [00:06, 919.61 examples/s]6065 examples [00:06, 920.12 examples/s]6158 examples [00:06, 902.71 examples/s]6253 examples [00:06, 916.13 examples/s]6349 examples [00:06, 928.12 examples/s]6445 examples [00:06, 937.40 examples/s]6542 examples [00:07, 946.54 examples/s]6637 examples [00:07, 907.85 examples/s]6729 examples [00:07, 895.63 examples/s]6824 examples [00:07, 907.63 examples/s]6919 examples [00:07, 918.46 examples/s]7021 examples [00:07, 945.54 examples/s]7117 examples [00:07, 949.30 examples/s]7217 examples [00:07, 961.62 examples/s]7319 examples [00:07, 975.79 examples/s]7420 examples [00:07, 983.80 examples/s]7519 examples [00:08, 982.22 examples/s]7618 examples [00:08, 965.60 examples/s]7719 examples [00:08, 977.87 examples/s]7820 examples [00:08, 987.19 examples/s]7919 examples [00:08, 985.14 examples/s]8018 examples [00:08, 986.28 examples/s]8117 examples [00:08, 983.46 examples/s]8216 examples [00:08, 962.93 examples/s]8313 examples [00:08, 940.07 examples/s]8408 examples [00:09, 929.07 examples/s]8502 examples [00:09, 914.43 examples/s]8599 examples [00:09, 928.41 examples/s]8695 examples [00:09, 935.80 examples/s]8793 examples [00:09, 946.96 examples/s]8893 examples [00:09, 960.48 examples/s]8992 examples [00:09, 967.53 examples/s]9089 examples [00:09, 936.89 examples/s]9185 examples [00:09, 943.08 examples/s]9284 examples [00:09, 954.53 examples/s]9380 examples [00:10, 949.00 examples/s]9476 examples [00:10, 950.76 examples/s]9572 examples [00:10, 932.14 examples/s]9669 examples [00:10, 942.70 examples/s]9764 examples [00:10, 921.47 examples/s]9864 examples [00:10, 941.23 examples/s]9962 examples [00:10, 951.17 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteVE3ADA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteVE3ADA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8048ff5ae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8048ff5ae8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f8048ff5ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.5229832172393799 	 Accuracy: 22
Train Epoch: 1 	 Loss: 4.161836719512939 	 Accuracy: 280
model saves at 280 accuracy
Train Epoch: 2 	 Loss: 0.42581536293029787 	 Accuracy: 54
Train Epoch: 2 	 Loss: 7.415443706512451 	 Accuracy: 280

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f800201d598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f800201d598>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f800201d598> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8000c4f0d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8000c4f0d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f8000c4f0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([5, 6, 8, 0, 6, 6, 5, 0, 0, 3]), array([2, 6, 8, 8, 7, 2, 5, 0, 3, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f8000bd2d68> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f7ffe5a8cc0> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7ffd683620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7ffd683620>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7ffd683620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:36, 272141.97it/s]  1%|          | 196608/26421880 [00:00<01:13, 355919.13it/s]  2%|▏         | 466944/26421880 [00:00<00:55, 467588.63it/s]  6%|▌         | 1638400/26421880 [00:00<00:37, 654415.57it/s] 15%|█▍        | 3833856/26421880 [00:00<00:24, 917270.68it/s] 30%|██▉       | 7913472/26421880 [00:01<00:14, 1297715.56it/s] 47%|████▋     | 12394496/26421880 [00:01<00:07, 1828761.72it/s] 61%|██████    | 16015360/26421880 [00:01<00:04, 2530167.87it/s] 75%|███████▌  | 19881984/26421880 [00:01<00:01, 3515876.86it/s] 90%|████████▉ | 23707648/26421880 [00:01<00:00, 4832318.98it/s]26427392it [00:01, 16055378.08it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 61854.88it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 40960/4422102 [00:00<00:12, 341914.48it/s]  2%|▏         | 106496/4422102 [00:00<00:11, 364125.93it/s] 10%|▉         | 425984/4422102 [00:00<00:08, 491456.45it/s] 21%|██        | 925696/4422102 [00:00<00:05, 659659.14it/s] 79%|███████▊  | 3473408/4422102 [00:00<00:01, 929995.71it/s]4423680it [00:00, 4562730.95it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 35080.93it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0035790524085362752 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019391949772834777 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0026626866559187573 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03988063716888428 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.0022897893289724985 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.02195690894126892 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.002338699539502462 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.012345008254051208 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0025746260384718576 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.012489098012447358 	 Accuracy: 4

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7ffd681d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7ffd681d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7ffd681d08> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([9, 2, 3, 3, 1, 2, 1, 7, 9, 2]), array([9, 2, 1, 1, 1, 2, 1, 5, 9, 4]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f7ffe6f95f8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7ffd67f510>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7ffd67f510>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7ffd67f510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110304.84it/s]  0%|          | 40960/18165135 [00:00<02:27, 122518.28it/s]  1%|          | 98304/18165135 [00:00<01:57, 154017.14it/s]  1%|          | 212992/18165135 [00:01<01:28, 202617.76it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272871.11it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 374799.94it/s] 10%|▉         | 1728512/18165135 [00:01<00:31, 521055.97it/s] 19%|█▉        | 3457024/18165135 [00:01<00:20, 730304.53it/s] 28%|██▊       | 5095424/18165135 [00:01<00:13, 1002249.74it/s] 41%|████      | 7454720/18165135 [00:02<00:07, 1406184.01it/s] 47%|████▋     | 8593408/18165135 [00:02<00:05, 1733057.51it/s] 52%|█████▏    | 9502720/18165135 [00:02<00:04, 1983450.31it/s] 64%|██████▍   | 11657216/18165135 [00:02<00:02, 2677062.71it/s] 69%|██████▉   | 12607488/18165135 [00:03<00:02, 2321931.20it/s] 73%|███████▎  | 13328384/18165135 [00:03<00:02, 2048115.97it/s] 81%|████████▏ | 14786560/18165135 [00:03<00:01, 2676082.54it/s] 87%|████████▋ | 15745024/18165135 [00:04<00:00, 3403229.96it/s] 91%|█████████ | 16490496/18165135 [00:04<00:00, 3442511.43it/s] 94%|█████████▍| 17121280/18165135 [00:04<00:00, 3393702.42it/s] 97%|█████████▋| 17661952/18165135 [00:04<00:00, 3396752.16it/s]100%|█████████▉| 18145280/18165135 [00:04<00:00, 3358845.44it/s]18169856it [00:04, 3835653.23it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110080.97it/s]32768it [00:00, 53298.08it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 109448.92it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 121770.59it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153111.72it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 200298.52it/s] 14%|█▎        | 417792/3041136 [00:01<00:09, 269900.01it/s] 28%|██▊       | 851968/3041136 [00:01<00:05, 370815.00it/s] 44%|████▍     | 1335296/3041136 [00:01<00:03, 504918.36it/s] 61%|██████    | 1843200/3041136 [00:01<00:01, 678363.40it/s] 78%|███████▊  | 2375680/3041136 [00:01<00:00, 895952.93it/s] 96%|█████████▌| 2924544/3041136 [00:02<00:00, 1158810.84it/s]3047424it [00:02, 1511409.36it/s]                             
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 16076.27it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004157121698061625 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02764184820652008 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.004376392066478729 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03780003345012665 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.003419345458348592 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.029118567228317262 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 4 	 Loss: 0.0026549160877863565 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.025120489060878753 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 5 	 Loss: 0.002930254658063253 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.021727227449417113 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7ffd67ec80>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7ffd67ec80>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7ffd67ec80> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([6, 4, 1, 2, 2, 0, 0, 8, 3, 0]), array([9, 7, 1, 1, 6, 9, 0, 4, 3, 7]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

