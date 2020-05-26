
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

  <mlmodels.model_tch.torchhub.Model object at 0x7fd86d736390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd86c76f400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd86c76f400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd86c76f400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3842048/9912422 [00:00<00:00, 38330956.20it/s]9920512it [00:00, 34452284.43it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 259773.63it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160296.61it/s]1654784it [00:00, 11355524.90it/s]                         
0it [00:00, ?it/s]8192it [00:00, 102348.55it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.00342179536819458 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01988762629032135 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.002209602475166321 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.033713329315185545 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.002894797215859095 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.035151515007019046 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 4 	 Loss: 0.0019176630079746247 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.021090044260025025 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 5 	 Loss: 0.001627552092075348 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.010415908336639404 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd86c75bf28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd86c75bf28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd86c75bf28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([4, 7, 9, 5, 9, 1, 9, 8, 2, 1]), array([4, 7, 4, 5, 4, 1, 4, 8, 2, 7]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fd86e5e7550> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd869acab70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd869acab70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd869acab70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0022008558909098305 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011832117319107056 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0019408984382947286 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03773857069015503 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd869aca400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd869aca400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd869aca400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), array([5, 3, 8, 4, 8, 8, 5, 4, 3, 6]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 17%|█▋        | 7.78M/44.7M [00:00<00:00, 81.5MB/s] 32%|███▏      | 14.5M/44.7M [00:00<00:00, 77.8MB/s] 45%|████▌     | 20.3M/44.7M [00:00<00:00, 71.9MB/s] 61%|██████▏   | 27.4M/44.7M [00:00<00:00, 72.4MB/s] 74%|███████▍  | 33.0M/44.7M [00:00<00:00, 67.7MB/s] 87%|████████▋ | 38.9M/44.7M [00:00<00:00, 65.6MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 67.8MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7fd8ba135ef0> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd86c73a1e0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd86c73a1e0>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fd86c73a1e0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:35,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:23,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:07, 15.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:00<00:05, 19.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 26.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 32.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 40.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 65.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 71.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 71.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 75.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 75.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.56 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.29s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.56 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]
0 examples [00:00, ? examples/s]2020-05-26 00:09:04.413368: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 00:09:04.418425: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-26 00:09:04.418569: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e882c1d140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 00:09:04.418585: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
61 examples [00:00, 608.83 examples/s]160 examples [00:00, 686.93 examples/s]255 examples [00:00, 748.90 examples/s]341 examples [00:00, 778.01 examples/s]432 examples [00:00, 811.90 examples/s]521 examples [00:00, 831.86 examples/s]620 examples [00:00, 872.13 examples/s]714 examples [00:00, 891.39 examples/s]812 examples [00:00, 914.24 examples/s]908 examples [00:01, 923.58 examples/s]1003 examples [00:01, 929.35 examples/s]1096 examples [00:01, 925.93 examples/s]1192 examples [00:01, 934.01 examples/s]1286 examples [00:01, 935.12 examples/s]1384 examples [00:01, 946.07 examples/s]1479 examples [00:01, 927.24 examples/s]1575 examples [00:01, 935.04 examples/s]1681 examples [00:01, 968.66 examples/s]1781 examples [00:01, 975.75 examples/s]1879 examples [00:02, 955.16 examples/s]1976 examples [00:02, 957.30 examples/s]2078 examples [00:02, 974.00 examples/s]2176 examples [00:02, 968.10 examples/s]2273 examples [00:02, 959.12 examples/s]2370 examples [00:02, 951.09 examples/s]2466 examples [00:02, 928.71 examples/s]2560 examples [00:02, 921.50 examples/s]2656 examples [00:02, 931.52 examples/s]2750 examples [00:02, 928.79 examples/s]2843 examples [00:03, 928.87 examples/s]2936 examples [00:03, 927.63 examples/s]3030 examples [00:03, 928.38 examples/s]3124 examples [00:03, 931.82 examples/s]3219 examples [00:03, 934.65 examples/s]3315 examples [00:03, 939.98 examples/s]3410 examples [00:03, 923.27 examples/s]3503 examples [00:03, 919.91 examples/s]3597 examples [00:03, 923.47 examples/s]3690 examples [00:03, 922.00 examples/s]3784 examples [00:04, 925.29 examples/s]3877 examples [00:04, 923.49 examples/s]3975 examples [00:04, 937.63 examples/s]4069 examples [00:04, 929.41 examples/s]4169 examples [00:04, 947.95 examples/s]4265 examples [00:04, 949.88 examples/s]4361 examples [00:04, 943.92 examples/s]4460 examples [00:04, 955.48 examples/s]4556 examples [00:04, 953.64 examples/s]4652 examples [00:04, 948.17 examples/s]4747 examples [00:05, 928.34 examples/s]4840 examples [00:05, 911.73 examples/s]4932 examples [00:05, 913.24 examples/s]5029 examples [00:05, 927.68 examples/s]5122 examples [00:05, 925.94 examples/s]5215 examples [00:05, 899.91 examples/s]5306 examples [00:05, 896.09 examples/s]5402 examples [00:05, 912.25 examples/s]5499 examples [00:05, 928.02 examples/s]5598 examples [00:06, 942.54 examples/s]5693 examples [00:06, 936.92 examples/s]5787 examples [00:06, 923.47 examples/s]5880 examples [00:06, 917.40 examples/s]5976 examples [00:06, 927.69 examples/s]6074 examples [00:06, 939.61 examples/s]6169 examples [00:06, 938.54 examples/s]6264 examples [00:06, 939.55 examples/s]6359 examples [00:06, 942.29 examples/s]6458 examples [00:06, 953.92 examples/s]6554 examples [00:07, 945.28 examples/s]6654 examples [00:07, 960.48 examples/s]6751 examples [00:07, 946.60 examples/s]6848 examples [00:07, 953.07 examples/s]6944 examples [00:07, 950.37 examples/s]7040 examples [00:07, 939.17 examples/s]7137 examples [00:07, 946.38 examples/s]7235 examples [00:07, 956.03 examples/s]7334 examples [00:07, 963.68 examples/s]7431 examples [00:07, 921.35 examples/s]7526 examples [00:08, 927.45 examples/s]7621 examples [00:08, 931.78 examples/s]7716 examples [00:08, 937.02 examples/s]7810 examples [00:08, 936.35 examples/s]7904 examples [00:08, 928.46 examples/s]7998 examples [00:08, 929.28 examples/s]8091 examples [00:08, 921.32 examples/s]8184 examples [00:08, 920.05 examples/s]8279 examples [00:08, 927.96 examples/s]8376 examples [00:08, 938.56 examples/s]8473 examples [00:09, 946.22 examples/s]8569 examples [00:09, 947.67 examples/s]8664 examples [00:09, 933.30 examples/s]8762 examples [00:09, 946.17 examples/s]8857 examples [00:09, 944.24 examples/s]8956 examples [00:09, 955.03 examples/s]9052 examples [00:09, 921.97 examples/s]9145 examples [00:09, 874.94 examples/s]9240 examples [00:09, 896.02 examples/s]9331 examples [00:10, 890.42 examples/s]9421 examples [00:10, 863.77 examples/s]9513 examples [00:10, 877.97 examples/s]9606 examples [00:10, 892.80 examples/s]9700 examples [00:10, 904.83 examples/s]9799 examples [00:10, 928.08 examples/s]9893 examples [00:10, 915.61 examples/s]9987 examples [00:10, 921.37 examples/s]10080 examples [00:10, 857.89 examples/s]10172 examples [00:10, 871.58 examples/s]10266 examples [00:11, 890.32 examples/s]10363 examples [00:11, 910.69 examples/s]10455 examples [00:11, 908.22 examples/s]10547 examples [00:11, 910.17 examples/s]10644 examples [00:11, 927.33 examples/s]10743 examples [00:11, 944.00 examples/s]10838 examples [00:11, 939.80 examples/s]10936 examples [00:11, 949.80 examples/s]11032 examples [00:11, 952.10 examples/s]11133 examples [00:11, 965.80 examples/s]11230 examples [00:12, 966.60 examples/s]11329 examples [00:12, 970.52 examples/s]11427 examples [00:12, 954.67 examples/s]11524 examples [00:12, 956.25 examples/s]11620 examples [00:12, 950.13 examples/s]11718 examples [00:12, 956.85 examples/s]11815 examples [00:12, 959.60 examples/s]11911 examples [00:12, 930.96 examples/s]12005 examples [00:12, 921.06 examples/s]12101 examples [00:13, 931.86 examples/s]12196 examples [00:13, 934.96 examples/s]12290 examples [00:13, 925.37 examples/s]12385 examples [00:13, 930.94 examples/s]12481 examples [00:13, 935.96 examples/s]12582 examples [00:13, 954.98 examples/s]12682 examples [00:13, 964.77 examples/s]12779 examples [00:13, 962.48 examples/s]12876 examples [00:13, 956.89 examples/s]12973 examples [00:13, 959.11 examples/s]13069 examples [00:14, 942.72 examples/s]13164 examples [00:14, 933.85 examples/s]13258 examples [00:14, 923.24 examples/s]13351 examples [00:14, 922.54 examples/s]13445 examples [00:14, 927.64 examples/s]13542 examples [00:14, 939.52 examples/s]13637 examples [00:14, 934.97 examples/s]13731 examples [00:14, 934.66 examples/s]13825 examples [00:14, 932.93 examples/s]13921 examples [00:14, 940.71 examples/s]14021 examples [00:15, 956.92 examples/s]14119 examples [00:15, 963.62 examples/s]14216 examples [00:15, 955.06 examples/s]14312 examples [00:15, 934.70 examples/s]14406 examples [00:15, 903.59 examples/s]14497 examples [00:15, 900.87 examples/s]14594 examples [00:15, 919.79 examples/s]14687 examples [00:15, 914.71 examples/s]14779 examples [00:15, 911.95 examples/s]14876 examples [00:15, 927.00 examples/s]14969 examples [00:16, 927.44 examples/s]15065 examples [00:16, 936.93 examples/s]15159 examples [00:16, 912.70 examples/s]15254 examples [00:16, 921.11 examples/s]15350 examples [00:16, 932.17 examples/s]15444 examples [00:16, 925.11 examples/s]15546 examples [00:16, 950.60 examples/s]15644 examples [00:16, 957.23 examples/s]15740 examples [00:16, 948.65 examples/s]15836 examples [00:16, 951.22 examples/s]15934 examples [00:17, 957.67 examples/s]16030 examples [00:17, 957.10 examples/s]16131 examples [00:17, 970.51 examples/s]16229 examples [00:17, 956.67 examples/s]16325 examples [00:17, 944.25 examples/s]16420 examples [00:17, 918.52 examples/s]16519 examples [00:17, 938.57 examples/s]16618 examples [00:17, 952.79 examples/s]16714 examples [00:17, 934.90 examples/s]16812 examples [00:18, 946.81 examples/s]16914 examples [00:18, 965.68 examples/s]17011 examples [00:18, 954.18 examples/s]17112 examples [00:18, 968.59 examples/s]17210 examples [00:18, 953.44 examples/s]17308 examples [00:18, 960.72 examples/s]17410 examples [00:18, 975.54 examples/s]17508 examples [00:18, 966.48 examples/s]17608 examples [00:18, 976.27 examples/s]17706 examples [00:18, 958.90 examples/s]17803 examples [00:19, 937.60 examples/s]17901 examples [00:19, 948.14 examples/s]17996 examples [00:19, 937.17 examples/s]18092 examples [00:19, 943.07 examples/s]18187 examples [00:19, 928.50 examples/s]18284 examples [00:19, 938.43 examples/s]18383 examples [00:19, 950.54 examples/s]18480 examples [00:19, 954.02 examples/s]18576 examples [00:19, 953.34 examples/s]18672 examples [00:19, 947.97 examples/s]18767 examples [00:20, 941.02 examples/s]18869 examples [00:20, 961.30 examples/s]18966 examples [00:20, 939.02 examples/s]19061 examples [00:20, 934.85 examples/s]19155 examples [00:20, 921.35 examples/s]19253 examples [00:20, 936.73 examples/s]19351 examples [00:20, 948.44 examples/s]19446 examples [00:20, 946.64 examples/s]19541 examples [00:20, 926.52 examples/s]19634 examples [00:20, 924.07 examples/s]19733 examples [00:21, 941.00 examples/s]19828 examples [00:21, 934.43 examples/s]19922 examples [00:21, 934.85 examples/s]20016 examples [00:21, 868.64 examples/s]20104 examples [00:21, 856.55 examples/s]20201 examples [00:21, 885.34 examples/s]20296 examples [00:21, 902.66 examples/s]20387 examples [00:21, 902.33 examples/s]20478 examples [00:21, 897.20 examples/s]20569 examples [00:22, 868.76 examples/s]20663 examples [00:22, 886.11 examples/s]20762 examples [00:22, 913.74 examples/s]20854 examples [00:22, 891.05 examples/s]20944 examples [00:22, 879.29 examples/s]21035 examples [00:22, 888.05 examples/s]21126 examples [00:22, 894.00 examples/s]21216 examples [00:22, 890.72 examples/s]21309 examples [00:22, 899.18 examples/s]21400 examples [00:22, 899.59 examples/s]21491 examples [00:23, 862.67 examples/s]21578 examples [00:23, 862.01 examples/s]21672 examples [00:23, 882.42 examples/s]21769 examples [00:23, 904.76 examples/s]21863 examples [00:23, 914.20 examples/s]21962 examples [00:23, 933.96 examples/s]22056 examples [00:23, 930.98 examples/s]22150 examples [00:23, 905.42 examples/s]22241 examples [00:23, 887.14 examples/s]22331 examples [00:24, 878.25 examples/s]22426 examples [00:24, 896.25 examples/s]22516 examples [00:24, 879.51 examples/s]22607 examples [00:24, 885.59 examples/s]22701 examples [00:24, 900.82 examples/s]22793 examples [00:24, 906.16 examples/s]22891 examples [00:24, 926.26 examples/s]22986 examples [00:24, 930.10 examples/s]23080 examples [00:24, 929.38 examples/s]23176 examples [00:24, 937.43 examples/s]23270 examples [00:25, 929.99 examples/s]23364 examples [00:25, 901.86 examples/s]23462 examples [00:25, 923.47 examples/s]23555 examples [00:25, 919.19 examples/s]23648 examples [00:25, 897.38 examples/s]23739 examples [00:25, 883.58 examples/s]23828 examples [00:25, 870.51 examples/s]23916 examples [00:25, 860.49 examples/s]24010 examples [00:25, 880.71 examples/s]24105 examples [00:25, 898.01 examples/s]24196 examples [00:26, 880.37 examples/s]24292 examples [00:26, 902.25 examples/s]24387 examples [00:26, 915.79 examples/s]24481 examples [00:26, 922.29 examples/s]24576 examples [00:26, 928.84 examples/s]24670 examples [00:26, 928.85 examples/s]24763 examples [00:26, 925.95 examples/s]24860 examples [00:26, 937.57 examples/s]24961 examples [00:26, 957.19 examples/s]25057 examples [00:26, 951.15 examples/s]25156 examples [00:27, 962.07 examples/s]25255 examples [00:27, 970.28 examples/s]25357 examples [00:27, 982.50 examples/s]25461 examples [00:27, 997.51 examples/s]25561 examples [00:27, 971.58 examples/s]25659 examples [00:27, 950.43 examples/s]25755 examples [00:27, 951.52 examples/s]25858 examples [00:27, 972.83 examples/s]25956 examples [00:27, 971.55 examples/s]26054 examples [00:28, 963.12 examples/s]26151 examples [00:28, 952.14 examples/s]26247 examples [00:28, 938.86 examples/s]26342 examples [00:28, 938.71 examples/s]26436 examples [00:28, 910.80 examples/s]26535 examples [00:28, 933.10 examples/s]26629 examples [00:28, 934.95 examples/s]26725 examples [00:28, 941.45 examples/s]26823 examples [00:28, 951.61 examples/s]26919 examples [00:28, 952.22 examples/s]27015 examples [00:29, 931.92 examples/s]27109 examples [00:29, 922.88 examples/s]27202 examples [00:29, 923.18 examples/s]27299 examples [00:29, 934.40 examples/s]27393 examples [00:29, 934.53 examples/s]27487 examples [00:29, 931.63 examples/s]27581 examples [00:29, 920.45 examples/s]27674 examples [00:29, 917.16 examples/s]27766 examples [00:29, 910.17 examples/s]27860 examples [00:29, 918.54 examples/s]27955 examples [00:30, 917.74 examples/s]28047 examples [00:30, 901.47 examples/s]28141 examples [00:30, 912.01 examples/s]28234 examples [00:30, 916.19 examples/s]28327 examples [00:30, 920.15 examples/s]28420 examples [00:30, 920.31 examples/s]28513 examples [00:30, 911.74 examples/s]28611 examples [00:30, 929.36 examples/s]28705 examples [00:30, 931.16 examples/s]28799 examples [00:30, 931.37 examples/s]28893 examples [00:31, 917.43 examples/s]28985 examples [00:31, 910.45 examples/s]29081 examples [00:31, 921.37 examples/s]29174 examples [00:31, 919.23 examples/s]29266 examples [00:31, 896.84 examples/s]29361 examples [00:31, 911.44 examples/s]29454 examples [00:31, 914.59 examples/s]29547 examples [00:31, 917.35 examples/s]29646 examples [00:31, 937.49 examples/s]29742 examples [00:32, 944.00 examples/s]29840 examples [00:32, 954.45 examples/s]29936 examples [00:32, 937.34 examples/s]30030 examples [00:32, 875.37 examples/s]30131 examples [00:32, 910.57 examples/s]30230 examples [00:32, 931.55 examples/s]30324 examples [00:32, 920.26 examples/s]30417 examples [00:32, 921.57 examples/s]30515 examples [00:32, 936.22 examples/s]30617 examples [00:32, 958.19 examples/s]30714 examples [00:33, 941.96 examples/s]30811 examples [00:33, 948.49 examples/s]30907 examples [00:33, 935.86 examples/s]31001 examples [00:33, 902.89 examples/s]31096 examples [00:33, 913.79 examples/s]31195 examples [00:33, 933.56 examples/s]31289 examples [00:33, 930.10 examples/s]31383 examples [00:33, 915.85 examples/s]31477 examples [00:33, 920.80 examples/s]31572 examples [00:33, 928.21 examples/s]31670 examples [00:34, 941.51 examples/s]31765 examples [00:34, 941.24 examples/s]31860 examples [00:34, 927.75 examples/s]31961 examples [00:34, 950.03 examples/s]32064 examples [00:34, 970.89 examples/s]32162 examples [00:34, 966.21 examples/s]32259 examples [00:34, 961.82 examples/s]32361 examples [00:34, 977.25 examples/s]32463 examples [00:34, 987.84 examples/s]32562 examples [00:34, 971.17 examples/s]32661 examples [00:35, 975.24 examples/s]32759 examples [00:35, 957.06 examples/s]32855 examples [00:35, 951.15 examples/s]32951 examples [00:35, 935.03 examples/s]33049 examples [00:35, 944.81 examples/s]33144 examples [00:35, 938.09 examples/s]33238 examples [00:35, 909.57 examples/s]33330 examples [00:35, 897.62 examples/s]33420 examples [00:35, 881.69 examples/s]33512 examples [00:36, 892.49 examples/s]33608 examples [00:36, 909.66 examples/s]33704 examples [00:36, 923.06 examples/s]33800 examples [00:36, 932.32 examples/s]33899 examples [00:36, 946.81 examples/s]34001 examples [00:36, 965.99 examples/s]34101 examples [00:36, 973.68 examples/s]34199 examples [00:36, 945.28 examples/s]34294 examples [00:36, 919.82 examples/s]34390 examples [00:36, 930.36 examples/s]34497 examples [00:37, 967.25 examples/s]34601 examples [00:37, 986.97 examples/s]34701 examples [00:37, 972.35 examples/s]34799 examples [00:37, 967.75 examples/s]34897 examples [00:37, 956.39 examples/s]34993 examples [00:37, 922.67 examples/s]35086 examples [00:37, 916.90 examples/s]35178 examples [00:37, 917.49 examples/s]35278 examples [00:37, 938.44 examples/s]35373 examples [00:38, 916.48 examples/s]35468 examples [00:38, 925.12 examples/s]35567 examples [00:38, 941.96 examples/s]35662 examples [00:38, 941.87 examples/s]35757 examples [00:38, 939.61 examples/s]35855 examples [00:38, 948.75 examples/s]35953 examples [00:38, 956.73 examples/s]36050 examples [00:38, 959.41 examples/s]36147 examples [00:38, 956.72 examples/s]36243 examples [00:38, 951.70 examples/s]36339 examples [00:39, 945.93 examples/s]36435 examples [00:39, 949.51 examples/s]36530 examples [00:39, 947.73 examples/s]36625 examples [00:39, 942.79 examples/s]36725 examples [00:39, 957.97 examples/s]36821 examples [00:39, 948.48 examples/s]36921 examples [00:39, 959.52 examples/s]37019 examples [00:39, 964.72 examples/s]37116 examples [00:39, 926.29 examples/s]37216 examples [00:39, 946.57 examples/s]37312 examples [00:40, 948.07 examples/s]37408 examples [00:40, 951.02 examples/s]37504 examples [00:40, 943.69 examples/s]37599 examples [00:40, 934.78 examples/s]37693 examples [00:40, 930.05 examples/s]37791 examples [00:40, 942.87 examples/s]37886 examples [00:40, 899.36 examples/s]37983 examples [00:40, 919.39 examples/s]38076 examples [00:40, 920.79 examples/s]38172 examples [00:40, 929.79 examples/s]38267 examples [00:41, 933.81 examples/s]38371 examples [00:41, 961.94 examples/s]38472 examples [00:41, 974.58 examples/s]38570 examples [00:41, 961.71 examples/s]38667 examples [00:41, 961.30 examples/s]38764 examples [00:41, 962.83 examples/s]38861 examples [00:41, 956.64 examples/s]38957 examples [00:41, 939.71 examples/s]39052 examples [00:41, 907.07 examples/s]39144 examples [00:42, 878.62 examples/s]39239 examples [00:42, 898.08 examples/s]39342 examples [00:42, 933.26 examples/s]39438 examples [00:42, 940.00 examples/s]39535 examples [00:42, 947.90 examples/s]39631 examples [00:42, 930.38 examples/s]39726 examples [00:42, 934.99 examples/s]39820 examples [00:42, 930.02 examples/s]39916 examples [00:42, 937.05 examples/s]40010 examples [00:42, 874.19 examples/s]40105 examples [00:43, 893.86 examples/s]40206 examples [00:43, 925.55 examples/s]40300 examples [00:43, 921.00 examples/s]40396 examples [00:43, 930.04 examples/s]40490 examples [00:43, 896.81 examples/s]40583 examples [00:43, 904.64 examples/s]40676 examples [00:43, 910.14 examples/s]40768 examples [00:43, 890.50 examples/s]40858 examples [00:43, 887.75 examples/s]40956 examples [00:43, 912.96 examples/s]41050 examples [00:44, 918.50 examples/s]41148 examples [00:44, 933.98 examples/s]41243 examples [00:44, 937.96 examples/s]41340 examples [00:44, 944.13 examples/s]41439 examples [00:44, 955.52 examples/s]41536 examples [00:44, 958.37 examples/s]41632 examples [00:44, 957.37 examples/s]41728 examples [00:44, 956.55 examples/s]41824 examples [00:44, 947.44 examples/s]41923 examples [00:44, 957.79 examples/s]42019 examples [00:45, 947.09 examples/s]42114 examples [00:45, 913.61 examples/s]42207 examples [00:45, 915.76 examples/s]42299 examples [00:45, 914.65 examples/s]42394 examples [00:45, 924.44 examples/s]42487 examples [00:45, 916.73 examples/s]42579 examples [00:45, 912.67 examples/s]42675 examples [00:45, 923.80 examples/s]42768 examples [00:45, 920.73 examples/s]42861 examples [00:46, 916.94 examples/s]42953 examples [00:46, 909.58 examples/s]43045 examples [00:46, 912.47 examples/s]43142 examples [00:46, 928.65 examples/s]43238 examples [00:46, 934.83 examples/s]43337 examples [00:46, 945.65 examples/s]43432 examples [00:46, 944.83 examples/s]43532 examples [00:46, 958.92 examples/s]43628 examples [00:46, 937.93 examples/s]43725 examples [00:46, 947.18 examples/s]43820 examples [00:47, 939.38 examples/s]43915 examples [00:47, 933.95 examples/s]44013 examples [00:47, 947.18 examples/s]44113 examples [00:47, 962.29 examples/s]44210 examples [00:47, 960.68 examples/s]44307 examples [00:47, 952.55 examples/s]44403 examples [00:47, 947.63 examples/s]44498 examples [00:47, 942.02 examples/s]44593 examples [00:47, 928.99 examples/s]44686 examples [00:47, 911.95 examples/s]44780 examples [00:48, 919.12 examples/s]44873 examples [00:48, 920.67 examples/s]44970 examples [00:48, 933.31 examples/s]45069 examples [00:48, 948.88 examples/s]45165 examples [00:48, 936.96 examples/s]45264 examples [00:48, 950.89 examples/s]45360 examples [00:48, 939.11 examples/s]45461 examples [00:48, 958.35 examples/s]45561 examples [00:48, 969.33 examples/s]45659 examples [00:48, 961.62 examples/s]45756 examples [00:49, 943.58 examples/s]45851 examples [00:49, 945.28 examples/s]45946 examples [00:49, 943.06 examples/s]46042 examples [00:49, 946.03 examples/s]46137 examples [00:49, 935.90 examples/s]46240 examples [00:49, 960.73 examples/s]46340 examples [00:49, 972.10 examples/s]46438 examples [00:49, 971.07 examples/s]46536 examples [00:49, 912.51 examples/s]46631 examples [00:50, 920.98 examples/s]46726 examples [00:50, 928.21 examples/s]46824 examples [00:50, 942.45 examples/s]46919 examples [00:50, 942.62 examples/s]47016 examples [00:50, 949.23 examples/s]47112 examples [00:50, 940.15 examples/s]47209 examples [00:50, 946.61 examples/s]47304 examples [00:50, 944.60 examples/s]47399 examples [00:50, 944.61 examples/s]47494 examples [00:50, 942.88 examples/s]47589 examples [00:51, 922.44 examples/s]47688 examples [00:51, 939.86 examples/s]47787 examples [00:51, 953.62 examples/s]47883 examples [00:51, 950.28 examples/s]47982 examples [00:51, 960.14 examples/s]48079 examples [00:51, 952.79 examples/s]48175 examples [00:51, 952.29 examples/s]48274 examples [00:51, 961.21 examples/s]48371 examples [00:51, 950.96 examples/s]48469 examples [00:51, 958.37 examples/s]48565 examples [00:52, 947.32 examples/s]48660 examples [00:52, 945.95 examples/s]48755 examples [00:52, 940.00 examples/s]48853 examples [00:52, 948.21 examples/s]48956 examples [00:52, 970.47 examples/s]49054 examples [00:52, 952.80 examples/s]49150 examples [00:52, 950.44 examples/s]49246 examples [00:52, 939.49 examples/s]49346 examples [00:52, 955.36 examples/s]49442 examples [00:52, 933.37 examples/s]49536 examples [00:53, 919.87 examples/s]49633 examples [00:53, 933.26 examples/s]49727 examples [00:53, 922.74 examples/s]49820 examples [00:53, 916.70 examples/s]49912 examples [00:53, 916.58 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6868/50000 [00:00<00:00, 68678.10 examples/s] 36%|███▋      | 18194/50000 [00:00<00:00, 77873.86 examples/s] 60%|██████    | 30119/50000 [00:00<00:00, 86920.86 examples/s] 86%|████████▋ | 43212/50000 [00:00<00:00, 96666.74 examples/s]                                                               0 examples [00:00, ? examples/s]72 examples [00:00, 715.53 examples/s]171 examples [00:00, 779.40 examples/s]267 examples [00:00, 825.07 examples/s]367 examples [00:00, 869.33 examples/s]466 examples [00:00, 901.85 examples/s]549 examples [00:00, 876.71 examples/s]650 examples [00:00, 912.34 examples/s]744 examples [00:00, 920.23 examples/s]845 examples [00:00, 937.76 examples/s]941 examples [00:01, 943.38 examples/s]1039 examples [00:01, 952.26 examples/s]1141 examples [00:01, 970.45 examples/s]1238 examples [00:01, 962.90 examples/s]1338 examples [00:01, 973.38 examples/s]1436 examples [00:01, 967.55 examples/s]1533 examples [00:01, 962.10 examples/s]1635 examples [00:01, 976.70 examples/s]1734 examples [00:01, 978.20 examples/s]1832 examples [00:01, 928.26 examples/s]1926 examples [00:02, 923.95 examples/s]2019 examples [00:02, 921.83 examples/s]2114 examples [00:02, 928.48 examples/s]2208 examples [00:02, 899.45 examples/s]2308 examples [00:02, 924.17 examples/s]2401 examples [00:02, 915.10 examples/s]2499 examples [00:02, 932.39 examples/s]2595 examples [00:02, 937.36 examples/s]2690 examples [00:02, 930.90 examples/s]2785 examples [00:02, 936.03 examples/s]2880 examples [00:03, 939.63 examples/s]2976 examples [00:03, 943.69 examples/s]3071 examples [00:03, 941.32 examples/s]3169 examples [00:03, 950.85 examples/s]3265 examples [00:03, 951.65 examples/s]3361 examples [00:03, 946.66 examples/s]3457 examples [00:03, 950.30 examples/s]3553 examples [00:03, 937.89 examples/s]3647 examples [00:03, 915.04 examples/s]3745 examples [00:03, 933.43 examples/s]3840 examples [00:04, 937.85 examples/s]3937 examples [00:04, 945.44 examples/s]4038 examples [00:04, 961.61 examples/s]4138 examples [00:04, 972.04 examples/s]4237 examples [00:04, 973.54 examples/s]4335 examples [00:04, 956.13 examples/s]4431 examples [00:04, 945.37 examples/s]4529 examples [00:04, 952.53 examples/s]4628 examples [00:04, 959.30 examples/s]4725 examples [00:05, 955.60 examples/s]4822 examples [00:05, 959.82 examples/s]4919 examples [00:05, 955.58 examples/s]5016 examples [00:05, 959.53 examples/s]5112 examples [00:05, 952.92 examples/s]5210 examples [00:05, 960.56 examples/s]5307 examples [00:05, 959.61 examples/s]5403 examples [00:05, 945.19 examples/s]5498 examples [00:05, 921.20 examples/s]5594 examples [00:05, 931.67 examples/s]5688 examples [00:06, 933.76 examples/s]5784 examples [00:06, 940.16 examples/s]5883 examples [00:06, 953.22 examples/s]5984 examples [00:06, 969.52 examples/s]6082 examples [00:06, 968.09 examples/s]6183 examples [00:06, 978.49 examples/s]6281 examples [00:06, 948.47 examples/s]6377 examples [00:06, 941.42 examples/s]6472 examples [00:06, 941.25 examples/s]6570 examples [00:06, 951.25 examples/s]6666 examples [00:07, 950.04 examples/s]6762 examples [00:07, 940.22 examples/s]6866 examples [00:07, 964.49 examples/s]6963 examples [00:07, 959.40 examples/s]7061 examples [00:07, 964.06 examples/s]7160 examples [00:07, 969.75 examples/s]7258 examples [00:07, 954.79 examples/s]7359 examples [00:07, 966.74 examples/s]7456 examples [00:07, 953.14 examples/s]7553 examples [00:07, 955.91 examples/s]7649 examples [00:08, 942.95 examples/s]7745 examples [00:08, 944.12 examples/s]7847 examples [00:08, 963.84 examples/s]7947 examples [00:08, 973.81 examples/s]8045 examples [00:08, 970.08 examples/s]8143 examples [00:08, 968.11 examples/s]8240 examples [00:08, 939.13 examples/s]8336 examples [00:08, 944.80 examples/s]8431 examples [00:08, 945.27 examples/s]8526 examples [00:08, 931.02 examples/s]8622 examples [00:09, 936.83 examples/s]8716 examples [00:09, 934.24 examples/s]8813 examples [00:09, 941.40 examples/s]8909 examples [00:09, 945.24 examples/s]9005 examples [00:09, 946.34 examples/s]9100 examples [00:09, 932.99 examples/s]9194 examples [00:09, 921.67 examples/s]9288 examples [00:09, 925.45 examples/s]9381 examples [00:09, 909.48 examples/s]9474 examples [00:10, 915.15 examples/s]9575 examples [00:10, 939.59 examples/s]9670 examples [00:10, 928.46 examples/s]9766 examples [00:10, 937.27 examples/s]9860 examples [00:10, 936.34 examples/s]9954 examples [00:10, 936.42 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW6GWEJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW6GWEJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd84b4cbf28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd84b4cbf28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd84b4cbf28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.4843907403945923 	 Accuracy: 48
Train Epoch: 1 	 Loss: 5.271389102935791 	 Accuracy: 220
model saves at 220 accuracy
Train Epoch: 2 	 Loss: 0.5168248343467713 	 Accuracy: 56
Train Epoch: 2 	 Loss: 6.102508068084717 	 Accuracy: 120

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd808419bf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd808419bf8>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fd808419bf8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd8084dfae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd8084dfae8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd8084dfae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([9, 9, 1, 6, 9, 5, 1, 5, 6, 9]), array([1, 0, 7, 2, 0, 7, 5, 3, 2, 5]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fd86d736390> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fd807188710> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd804428620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd804428620>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd804428620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:39, 264538.68it/s]  0%|          | 131072/26421880 [00:00<01:21, 323582.48it/s]  2%|▏         | 425984/26421880 [00:00<01:00, 430313.77it/s]  4%|▍         | 1081344/26421880 [00:00<00:42, 594472.14it/s] 11%|█         | 2785280/26421880 [00:00<00:28, 836642.35it/s] 22%|██▏       | 5816320/26421880 [00:01<00:17, 1181120.94it/s] 36%|███▌      | 9404416/26421880 [00:01<00:10, 1653967.07it/s] 50%|█████     | 13336576/26421880 [00:01<00:05, 2320687.10it/s] 63%|██████▎   | 16613376/26421880 [00:01<00:03, 3204024.79it/s] 77%|███████▋  | 20422656/26421880 [00:01<00:01, 4416275.51it/s] 92%|█████████▏| 24297472/26421880 [00:01<00:00, 6015149.10it/s]26427392it [00:01, 15598906.03it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 96370.62it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 271369.28it/s]  4%|▍         | 188416/4422102 [00:00<00:12, 350376.07it/s] 11%|█         | 466944/4422102 [00:00<00:08, 464247.09it/s] 36%|███▌      | 1572864/4422102 [00:00<00:04, 648380.32it/s] 87%|████████▋ | 3833856/4422102 [00:00<00:00, 910008.74it/s]4423680it [00:00, 4450209.56it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 30174.82it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0035843823949495953 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020661169290542604 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0029644269545873007 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02039453101158142 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.0027030829588572183 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.027509865045547484 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.002384417474269867 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.01788171362876892 	 Accuracy: 4
Train Epoch: 5 	 Loss: 0.0023691586355368294 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.010834770619869233 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd804426d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd804426d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd804426d08> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([1, 5, 6, 6, 3, 4, 4, 7, 4, 1]), array([1, 5, 0, 2, 3, 2, 4, 5, 4, 1]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fd807171710> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd804430510>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd804430510>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd804430510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:45, 109943.42it/s]  0%|          | 40960/18165135 [00:00<02:28, 122167.18it/s]  1%|          | 98304/18165135 [00:00<01:57, 153607.27it/s]  1%|          | 212992/18165135 [00:01<01:28, 202052.39it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272181.50it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 373845.63it/s]  6%|▌         | 1024000/18165135 [00:01<00:37, 459410.99it/s] 13%|█▎        | 2367488/18165135 [00:01<00:24, 643753.91it/s] 17%|█▋        | 3104768/18165135 [00:01<00:17, 870819.14it/s] 21%|██        | 3850240/18165135 [00:01<00:12, 1157505.22it/s] 25%|██▌       | 4612096/18165135 [00:02<00:08, 1507098.13it/s] 30%|██▉       | 5390336/18165135 [00:02<00:06, 1946411.66it/s] 34%|███▍      | 6160384/18165135 [00:02<00:04, 2492470.22it/s] 37%|███▋      | 6782976/18165135 [00:02<00:03, 3036062.77it/s] 41%|████      | 7372800/18165135 [00:02<00:03, 3335684.96it/s] 44%|████▎     | 7913472/18165135 [00:02<00:02, 3494190.55it/s] 47%|████▋     | 8626176/18165135 [00:02<00:02, 4021972.17it/s] 52%|█████▏    | 9453568/18165135 [00:02<00:01, 4690995.10it/s] 56%|█████▌    | 10117120/18165135 [00:03<00:01, 5139742.38it/s] 59%|█████▉    | 10739712/18165135 [00:03<00:01, 4949416.02it/s] 62%|██████▏   | 11313152/18165135 [00:03<00:01, 4683898.72it/s] 67%|██████▋   | 12083200/18165135 [00:03<00:01, 5154707.05it/s] 71%|███████▏  | 12951552/18165135 [00:03<00:00, 5766394.42it/s] 75%|███████▍  | 13598720/18165135 [00:03<00:00, 5956264.13it/s] 78%|███████▊  | 14245888/18165135 [00:03<00:00, 3984782.25it/s] 85%|████████▌ | 15491072/18165135 [00:04<00:00, 4837956.91it/s] 89%|████████▉ | 16146432/18165135 [00:04<00:00, 4690815.87it/s] 92%|█████████▏| 16785408/18165135 [00:04<00:00, 4866502.79it/s] 96%|█████████▌| 17432576/18165135 [00:04<00:00, 5137843.29it/s] 99%|█████████▉| 18014208/18165135 [00:04<00:00, 5203212.50it/s]18169856it [00:04, 3900467.66it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110147.31it/s]32768it [00:00, 52442.78it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110192.18it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122373.70it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153833.47it/s]  7%|▋         | 212992/3041136 [00:00<00:13, 202423.82it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 272630.71it/s] 28%|██▊       | 860160/3041136 [00:01<00:05, 374450.44it/s] 50%|████▉     | 1507328/3041136 [00:01<00:02, 515844.44it/s] 72%|███████▏  | 2187264/3041136 [00:01<00:01, 702680.36it/s] 95%|█████████▌| 2899968/3041136 [00:01<00:00, 944070.71it/s]3047424it [00:01, 1839112.76it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17571.64it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004453654944896698 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.024917293548583984 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0034120816985766093 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02154597282409668 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0031075907945632936 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.01917562937736511 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 4 	 Loss: 0.0030045229395230613 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.022717448770999907 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0027230661312739055 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.03291370785236359 	 Accuracy: 2

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd804431c80>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd804431c80>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd804431c80> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([8, 8, 8, 2, 6, 8, 8, 8, 5, 8]), array([3, 7, 6, 2, 5, 5, 4, 3, 0, 6]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

