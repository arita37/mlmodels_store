
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a3348a6e22d26e01dd9ff758dafc8c305dd2ddc7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'a3348a6e22d26e01dd9ff758dafc8c305dd2ddc7', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a3348a6e22d26e01dd9ff758dafc8c305dd2ddc7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a3348a6e22d26e01dd9ff758dafc8c305dd2ddc7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a3348a6e22d26e01dd9ff758dafc8c305dd2ddc7

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fadd9e03630> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadd8dbc0d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadd8dbc0d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fadd8dbc0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3424256/9912422 [00:00<00:00, 34232553.78it/s]9920512it [00:00, 34955746.84it/s]                             
0it [00:00, ?it/s]32768it [00:00, 328226.54it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 126907.88it/s]1654784it [00:00, 9351364.85it/s]                          
0it [00:00, ?it/s]8192it [00:00, 236666.66it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003468757728735606 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019283077597618103 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.002164491395155589 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.015376601338386536 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.002328516036272049 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.026319879770278932 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0027264760037263233 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.03541542100906372 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0017894108394781749 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.012941895961761474 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadd8d9ef28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadd8d9ef28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fadd8d9ef28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([9, 6, 3, 6, 1, 1, 1, 6, 1, 6]), array([9, 9, 3, 2, 1, 1, 1, 6, 1, 6]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fadd9df0668> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadd941a730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadd941a730>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fadd941a730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0019563048879305523 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.026315380573272706 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0018295051654179891 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.04599945306777954 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fadd941ed90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fadd941ed90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fadd941ed90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), array([4, 6, 5, 2, 9, 3, 3, 3, 1, 7]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 19%|█▉        | 8.45M/44.7M [00:00<00:00, 88.6MB/s] 41%|████      | 18.2M/44.7M [00:00<00:00, 92.3MB/s] 62%|██████▏   | 27.8M/44.7M [00:00<00:00, 94.5MB/s] 79%|███████▉  | 35.4M/44.7M [00:00<00:00, 89.5MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 94.8MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7fadd9c3e7f0> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fadd8d84e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fadd8d84e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fadd8d84e18> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:02,  2.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:02,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:02,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:02,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:01,  2.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:44,  3.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:44,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:43,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:43,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:43,  3.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:31,  4.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:31,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:31,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:31,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:30,  4.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:22,  6.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:22,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:22,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:22,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:22,  6.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:16,  8.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:16,  8.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:12, 11.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:12, 11.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:12, 11.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:12, 11.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:12, 11.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:09, 14.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:09, 14.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:09, 14.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:09, 14.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:09, 14.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 17.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:07, 17.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:06, 20.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:06, 20.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:06, 20.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:06, 20.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:06, 20.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 23.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:05, 23.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 26.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:04, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:04, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 28.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:04, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:04, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:03, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:03, 28.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:03, 30.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:03, 30.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 32.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:03, 32.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 32.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 32.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 32.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 33.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 33.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 33.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 33.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:03, 33.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 34.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 34.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 34.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 34.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 34.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 35.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 38.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:02, 38.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 42.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:02, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 42.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 49.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 49.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 55.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 55.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:00, 63.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 63.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 70.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 70.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 76.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 76.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 80.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 80.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 84.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 84.63 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 87.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 87.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 89.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 89.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.30s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 89.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 89.96 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.36s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.30s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 89.96 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.36s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.36s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.21 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.36s/ url]
0 examples [00:00, ? examples/s]2020-05-23 18:26:59.035357: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 18:26:59.043149: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-23 18:26:59.043922: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56472c6b2a20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 18:26:59.043942: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
54 examples [00:00, 535.77 examples/s]165 examples [00:00, 633.90 examples/s]266 examples [00:00, 713.00 examples/s]373 examples [00:00, 792.21 examples/s]484 examples [00:00, 865.23 examples/s]595 examples [00:00, 926.32 examples/s]699 examples [00:00, 955.05 examples/s]804 examples [00:00, 979.19 examples/s]903 examples [00:00, 979.46 examples/s]1007 examples [00:01, 994.85 examples/s]1114 examples [00:01, 1014.11 examples/s]1227 examples [00:01, 1043.82 examples/s]1333 examples [00:01, 1047.92 examples/s]1439 examples [00:01, 1038.31 examples/s]1551 examples [00:01, 1060.45 examples/s]1664 examples [00:01, 1078.86 examples/s]1776 examples [00:01, 1090.74 examples/s]1886 examples [00:01, 1079.51 examples/s]1996 examples [00:01, 1084.58 examples/s]2109 examples [00:02, 1096.85 examples/s]2222 examples [00:02, 1105.62 examples/s]2334 examples [00:02, 1109.71 examples/s]2446 examples [00:02, 1111.50 examples/s]2558 examples [00:02, 1100.05 examples/s]2669 examples [00:02, 1069.07 examples/s]2780 examples [00:02, 1078.58 examples/s]2889 examples [00:02, 1077.08 examples/s]2998 examples [00:02, 1080.17 examples/s]3107 examples [00:02, 1072.08 examples/s]3215 examples [00:03, 1073.75 examples/s]3325 examples [00:03, 1078.87 examples/s]3436 examples [00:03, 1087.41 examples/s]3548 examples [00:03, 1094.87 examples/s]3658 examples [00:03, 1093.31 examples/s]3769 examples [00:03, 1096.83 examples/s]3879 examples [00:03, 1095.82 examples/s]3991 examples [00:03, 1101.19 examples/s]4102 examples [00:03, 1098.78 examples/s]4212 examples [00:03, 1058.54 examples/s]4324 examples [00:04, 1075.25 examples/s]4436 examples [00:04, 1085.56 examples/s]4546 examples [00:04, 1089.20 examples/s]4656 examples [00:04, 1091.53 examples/s]4766 examples [00:04, 1047.55 examples/s]4876 examples [00:04, 1062.47 examples/s]4987 examples [00:04, 1075.21 examples/s]5095 examples [00:04, 1071.30 examples/s]5206 examples [00:04, 1081.30 examples/s]5315 examples [00:04, 1077.43 examples/s]5423 examples [00:05, 1077.84 examples/s]5534 examples [00:05, 1087.08 examples/s]5643 examples [00:05, 1086.59 examples/s]5752 examples [00:05, 1082.08 examples/s]5862 examples [00:05, 1084.73 examples/s]5971 examples [00:05, 1084.94 examples/s]6083 examples [00:05, 1094.14 examples/s]6193 examples [00:05, 1093.80 examples/s]6306 examples [00:05, 1101.06 examples/s]6417 examples [00:05, 1071.22 examples/s]6526 examples [00:06, 1075.73 examples/s]6637 examples [00:06, 1085.17 examples/s]6750 examples [00:06, 1096.99 examples/s]6865 examples [00:06, 1112.34 examples/s]6979 examples [00:06, 1117.92 examples/s]7092 examples [00:06, 1120.15 examples/s]7206 examples [00:06, 1124.23 examples/s]7319 examples [00:06, 1122.03 examples/s]7432 examples [00:06, 1116.79 examples/s]7544 examples [00:07, 1102.00 examples/s]7657 examples [00:07, 1108.17 examples/s]7770 examples [00:07, 1114.54 examples/s]7882 examples [00:07, 1113.68 examples/s]7994 examples [00:07, 1114.46 examples/s]8106 examples [00:07, 1064.97 examples/s]8213 examples [00:07, 1041.94 examples/s]8323 examples [00:07, 1056.77 examples/s]8432 examples [00:07, 1065.29 examples/s]8539 examples [00:07, 1066.30 examples/s]8646 examples [00:08, 1057.29 examples/s]8752 examples [00:08, 1051.05 examples/s]8858 examples [00:08, 1033.93 examples/s]8969 examples [00:08, 1054.93 examples/s]9080 examples [00:08, 1069.49 examples/s]9193 examples [00:08, 1086.62 examples/s]9307 examples [00:08, 1099.65 examples/s]9418 examples [00:08, 1098.52 examples/s]9529 examples [00:08, 1101.73 examples/s]9640 examples [00:08, 1100.56 examples/s]9751 examples [00:09, 1097.74 examples/s]9861 examples [00:09, 1080.30 examples/s]9970 examples [00:09, 1069.54 examples/s]10078 examples [00:09, 998.07 examples/s]10179 examples [00:09, 971.77 examples/s]10290 examples [00:09, 1008.88 examples/s]10404 examples [00:09, 1042.74 examples/s]10516 examples [00:09, 1064.39 examples/s]10628 examples [00:09, 1079.33 examples/s]10741 examples [00:09, 1092.27 examples/s]10852 examples [00:10, 1095.62 examples/s]10964 examples [00:10, 1101.69 examples/s]11076 examples [00:10, 1106.87 examples/s]11187 examples [00:10, 1097.02 examples/s]11297 examples [00:10, 1066.46 examples/s]11404 examples [00:10, 1050.37 examples/s]11516 examples [00:10, 1068.76 examples/s]11626 examples [00:10, 1076.69 examples/s]11734 examples [00:10, 1064.03 examples/s]11843 examples [00:11, 1070.08 examples/s]11952 examples [00:11, 1075.51 examples/s]12061 examples [00:11, 1079.76 examples/s]12171 examples [00:11, 1084.31 examples/s]12280 examples [00:11, 1081.55 examples/s]12391 examples [00:11, 1088.99 examples/s]12500 examples [00:11, 1087.82 examples/s]12613 examples [00:11, 1097.83 examples/s]12724 examples [00:11, 1101.28 examples/s]12835 examples [00:11, 1099.59 examples/s]12948 examples [00:12, 1107.97 examples/s]13059 examples [00:12, 1094.20 examples/s]13170 examples [00:12, 1096.22 examples/s]13283 examples [00:12, 1104.00 examples/s]13394 examples [00:12, 1064.17 examples/s]13501 examples [00:12, 1064.49 examples/s]13610 examples [00:12, 1070.47 examples/s]13721 examples [00:12, 1081.46 examples/s]13833 examples [00:12, 1090.35 examples/s]13944 examples [00:12, 1093.84 examples/s]14054 examples [00:13, 1087.14 examples/s]14164 examples [00:13, 1089.63 examples/s]14275 examples [00:13, 1095.46 examples/s]14385 examples [00:13, 1094.66 examples/s]14495 examples [00:13, 1093.38 examples/s]14605 examples [00:13, 1085.76 examples/s]14714 examples [00:13, 1072.23 examples/s]14825 examples [00:13, 1082.06 examples/s]14936 examples [00:13, 1087.82 examples/s]15046 examples [00:13, 1091.02 examples/s]15157 examples [00:14, 1095.06 examples/s]15267 examples [00:14, 1094.97 examples/s]15380 examples [00:14, 1103.21 examples/s]15492 examples [00:14, 1105.84 examples/s]15603 examples [00:14, 1107.07 examples/s]15714 examples [00:14, 1090.34 examples/s]15824 examples [00:14, 1089.84 examples/s]15936 examples [00:14, 1098.57 examples/s]16049 examples [00:14, 1107.73 examples/s]16161 examples [00:14, 1108.57 examples/s]16272 examples [00:15, 1104.12 examples/s]16383 examples [00:15, 1100.68 examples/s]16494 examples [00:15, 1101.09 examples/s]16606 examples [00:15, 1105.91 examples/s]16717 examples [00:15, 1096.02 examples/s]16827 examples [00:15, 1092.43 examples/s]16939 examples [00:15, 1099.51 examples/s]17050 examples [00:15, 1101.15 examples/s]17161 examples [00:15, 1103.47 examples/s]17273 examples [00:15, 1106.47 examples/s]17385 examples [00:16, 1109.24 examples/s]17496 examples [00:16, 1099.45 examples/s]17606 examples [00:16, 1096.63 examples/s]17719 examples [00:16, 1105.99 examples/s]17831 examples [00:16, 1108.93 examples/s]17944 examples [00:16, 1112.60 examples/s]18056 examples [00:16, 1060.61 examples/s]18169 examples [00:16, 1077.92 examples/s]18281 examples [00:16, 1087.51 examples/s]18391 examples [00:17, 1053.44 examples/s]18503 examples [00:17, 1070.00 examples/s]18613 examples [00:17, 1075.81 examples/s]18724 examples [00:17, 1084.86 examples/s]18836 examples [00:17, 1093.66 examples/s]18946 examples [00:17, 1090.96 examples/s]19056 examples [00:17, 1076.27 examples/s]19164 examples [00:17, 1049.82 examples/s]19275 examples [00:17, 1066.38 examples/s]19385 examples [00:17, 1074.69 examples/s]19495 examples [00:18, 1080.87 examples/s]19604 examples [00:18, 1083.46 examples/s]19713 examples [00:18, 1080.67 examples/s]19826 examples [00:18, 1092.33 examples/s]19936 examples [00:18, 1094.13 examples/s]20046 examples [00:18, 1034.92 examples/s]20151 examples [00:18, 1029.90 examples/s]20255 examples [00:18, 1024.15 examples/s]20358 examples [00:18, 1019.04 examples/s]20466 examples [00:18, 1035.97 examples/s]20576 examples [00:19, 1053.65 examples/s]20686 examples [00:19, 1065.59 examples/s]20793 examples [00:19, 1066.11 examples/s]20904 examples [00:19, 1076.32 examples/s]21015 examples [00:19, 1083.85 examples/s]21127 examples [00:19, 1092.51 examples/s]21237 examples [00:19, 1083.61 examples/s]21346 examples [00:19, 1037.43 examples/s]21458 examples [00:19, 1059.75 examples/s]21570 examples [00:19, 1075.73 examples/s]21682 examples [00:20, 1086.97 examples/s]21793 examples [00:20, 1093.13 examples/s]21903 examples [00:20, 1084.27 examples/s]22014 examples [00:20, 1091.12 examples/s]22124 examples [00:20, 1093.46 examples/s]22236 examples [00:20, 1099.05 examples/s]22348 examples [00:20, 1104.85 examples/s]22459 examples [00:20, 1098.41 examples/s]22572 examples [00:20, 1105.42 examples/s]22683 examples [00:21, 1099.75 examples/s]22794 examples [00:21, 1072.71 examples/s]22902 examples [00:21, 1070.29 examples/s]23010 examples [00:21, 1062.03 examples/s]23121 examples [00:21, 1074.83 examples/s]23232 examples [00:21, 1083.06 examples/s]23345 examples [00:21, 1096.06 examples/s]23456 examples [00:21, 1097.91 examples/s]23566 examples [00:21, 1089.97 examples/s]23677 examples [00:21, 1094.44 examples/s]23790 examples [00:22, 1103.40 examples/s]23901 examples [00:22, 1097.96 examples/s]24011 examples [00:22, 1098.18 examples/s]24121 examples [00:22, 1091.88 examples/s]24233 examples [00:22, 1098.57 examples/s]24345 examples [00:22, 1101.96 examples/s]24456 examples [00:22, 1100.73 examples/s]24567 examples [00:22, 1056.81 examples/s]24674 examples [00:22, 1039.37 examples/s]24785 examples [00:22, 1059.37 examples/s]24892 examples [00:23, 1051.18 examples/s]24998 examples [00:23, 1038.49 examples/s]25107 examples [00:23, 1050.74 examples/s]25214 examples [00:23, 1055.31 examples/s]25320 examples [00:23, 1056.70 examples/s]25428 examples [00:23, 1062.40 examples/s]25537 examples [00:23, 1068.91 examples/s]25645 examples [00:23, 1071.54 examples/s]25753 examples [00:23, 1071.62 examples/s]25861 examples [00:23, 1068.42 examples/s]25971 examples [00:24, 1076.08 examples/s]26082 examples [00:24, 1084.16 examples/s]26193 examples [00:24, 1090.36 examples/s]26303 examples [00:24, 1090.35 examples/s]26413 examples [00:24, 1087.80 examples/s]26525 examples [00:24, 1096.95 examples/s]26635 examples [00:24, 1096.62 examples/s]26746 examples [00:24, 1098.71 examples/s]26856 examples [00:24, 1098.13 examples/s]26966 examples [00:24, 1097.66 examples/s]27076 examples [00:25, 1087.12 examples/s]27186 examples [00:25, 1088.56 examples/s]27295 examples [00:25, 1082.45 examples/s]27406 examples [00:25, 1087.48 examples/s]27517 examples [00:25, 1091.57 examples/s]27627 examples [00:25, 1085.10 examples/s]27737 examples [00:25, 1087.93 examples/s]27848 examples [00:25, 1093.18 examples/s]27958 examples [00:25, 1040.21 examples/s]28067 examples [00:26, 1052.70 examples/s]28174 examples [00:26, 1055.14 examples/s]28284 examples [00:26, 1066.85 examples/s]28391 examples [00:26, 1060.44 examples/s]28498 examples [00:26, 1061.31 examples/s]28609 examples [00:26, 1075.17 examples/s]28718 examples [00:26, 1076.91 examples/s]28830 examples [00:26, 1087.90 examples/s]28939 examples [00:26, 1087.06 examples/s]29048 examples [00:26, 1067.91 examples/s]29159 examples [00:27, 1079.89 examples/s]29268 examples [00:27, 1047.23 examples/s]29381 examples [00:27, 1069.94 examples/s]29491 examples [00:27, 1075.87 examples/s]29599 examples [00:27, 1039.04 examples/s]29710 examples [00:27, 1057.08 examples/s]29818 examples [00:27, 1061.33 examples/s]29925 examples [00:27, 1042.55 examples/s]30030 examples [00:27, 954.80 examples/s] 30128 examples [00:27, 921.97 examples/s]30238 examples [00:28, 968.01 examples/s]30348 examples [00:28, 1002.40 examples/s]30458 examples [00:28, 1027.31 examples/s]30568 examples [00:28, 1047.92 examples/s]30678 examples [00:28, 1062.02 examples/s]30789 examples [00:28, 1075.36 examples/s]30900 examples [00:28, 1084.13 examples/s]31009 examples [00:28, 1085.87 examples/s]31118 examples [00:28, 1066.94 examples/s]31225 examples [00:29, 1018.04 examples/s]31334 examples [00:29, 1036.09 examples/s]31440 examples [00:29, 1040.31 examples/s]31548 examples [00:29, 1049.59 examples/s]31654 examples [00:29, 1049.41 examples/s]31760 examples [00:29, 1039.02 examples/s]31865 examples [00:29, 1026.41 examples/s]31974 examples [00:29, 1043.74 examples/s]32083 examples [00:29, 1055.71 examples/s]32189 examples [00:29, 1034.59 examples/s]32299 examples [00:30, 1051.19 examples/s]32405 examples [00:30, 1043.62 examples/s]32513 examples [00:30, 1051.87 examples/s]32623 examples [00:30, 1063.34 examples/s]32730 examples [00:30, 1048.83 examples/s]32841 examples [00:30, 1064.47 examples/s]32954 examples [00:30, 1082.03 examples/s]33064 examples [00:30, 1085.81 examples/s]33173 examples [00:30, 1083.58 examples/s]33282 examples [00:30, 1050.52 examples/s]33388 examples [00:31, 1041.59 examples/s]33498 examples [00:31, 1056.88 examples/s]33608 examples [00:31, 1068.55 examples/s]33718 examples [00:31, 1077.35 examples/s]33826 examples [00:31, 1073.89 examples/s]33938 examples [00:31, 1085.45 examples/s]34047 examples [00:31, 1086.65 examples/s]34156 examples [00:31, 1074.73 examples/s]34264 examples [00:31, 1066.84 examples/s]34371 examples [00:31, 1064.55 examples/s]34478 examples [00:32, 1062.47 examples/s]34589 examples [00:32, 1074.19 examples/s]34698 examples [00:32, 1076.85 examples/s]34806 examples [00:32, 1063.05 examples/s]34913 examples [00:32, 1047.66 examples/s]35023 examples [00:32, 1060.26 examples/s]35134 examples [00:32, 1073.37 examples/s]35242 examples [00:32, 1072.52 examples/s]35350 examples [00:32, 1072.38 examples/s]35458 examples [00:32, 1063.81 examples/s]35566 examples [00:33, 1065.83 examples/s]35675 examples [00:33, 1072.48 examples/s]35783 examples [00:33, 1070.16 examples/s]35895 examples [00:33, 1082.36 examples/s]36005 examples [00:33, 1086.51 examples/s]36115 examples [00:33, 1088.84 examples/s]36226 examples [00:33, 1092.84 examples/s]36336 examples [00:33, 1094.57 examples/s]36446 examples [00:33, 1071.53 examples/s]36554 examples [00:34, 1071.05 examples/s]36662 examples [00:34, 1064.01 examples/s]36772 examples [00:34, 1074.04 examples/s]36882 examples [00:34, 1079.29 examples/s]36990 examples [00:34, 1074.41 examples/s]37099 examples [00:34, 1076.25 examples/s]37210 examples [00:34, 1084.64 examples/s]37322 examples [00:34, 1092.98 examples/s]37432 examples [00:34, 1094.81 examples/s]37544 examples [00:34, 1100.15 examples/s]37655 examples [00:35, 1094.05 examples/s]37766 examples [00:35, 1097.07 examples/s]37877 examples [00:35, 1100.07 examples/s]37988 examples [00:35, 1099.59 examples/s]38098 examples [00:35, 1063.16 examples/s]38207 examples [00:35, 1068.29 examples/s]38316 examples [00:35, 1073.12 examples/s]38426 examples [00:35, 1079.40 examples/s]38535 examples [00:35, 1081.82 examples/s]38644 examples [00:35, 1082.01 examples/s]38753 examples [00:36, 1075.00 examples/s]38862 examples [00:36, 1078.74 examples/s]38970 examples [00:36, 1068.04 examples/s]39080 examples [00:36, 1076.16 examples/s]39191 examples [00:36, 1083.54 examples/s]39302 examples [00:36, 1091.26 examples/s]39414 examples [00:36, 1097.51 examples/s]39526 examples [00:36, 1103.95 examples/s]39637 examples [00:36, 1091.47 examples/s]39748 examples [00:36, 1094.17 examples/s]39859 examples [00:37, 1096.42 examples/s]39969 examples [00:37, 1094.21 examples/s]40079 examples [00:37, 990.83 examples/s] 40190 examples [00:37, 1022.29 examples/s]40301 examples [00:37, 1046.09 examples/s]40407 examples [00:37, 1046.81 examples/s]40517 examples [00:37, 1059.64 examples/s]40629 examples [00:37, 1075.83 examples/s]40739 examples [00:37, 1081.82 examples/s]40850 examples [00:37, 1087.72 examples/s]40960 examples [00:38, 1064.08 examples/s]41067 examples [00:38, 1042.52 examples/s]41177 examples [00:38, 1058.10 examples/s]41288 examples [00:38, 1072.05 examples/s]41396 examples [00:38, 1045.06 examples/s]41503 examples [00:38, 1051.93 examples/s]41613 examples [00:38, 1065.84 examples/s]41725 examples [00:38, 1081.46 examples/s]41835 examples [00:38, 1086.36 examples/s]41945 examples [00:39, 1087.95 examples/s]42054 examples [00:39, 1079.97 examples/s]42163 examples [00:39, 1082.14 examples/s]42272 examples [00:39, 1077.70 examples/s]42380 examples [00:39, 1067.29 examples/s]42487 examples [00:39, 1063.22 examples/s]42598 examples [00:39, 1075.57 examples/s]42711 examples [00:39, 1088.58 examples/s]42823 examples [00:39, 1097.42 examples/s]42936 examples [00:39, 1106.29 examples/s]43047 examples [00:40, 1090.95 examples/s]43157 examples [00:40, 1077.25 examples/s]43265 examples [00:40, 1069.51 examples/s]43374 examples [00:40, 1073.69 examples/s]43484 examples [00:40, 1080.66 examples/s]43593 examples [00:40, 1058.74 examples/s]43703 examples [00:40, 1070.28 examples/s]43815 examples [00:40, 1082.51 examples/s]43925 examples [00:40, 1087.46 examples/s]44037 examples [00:40, 1095.58 examples/s]44147 examples [00:41, 1096.46 examples/s]44257 examples [00:41, 1092.65 examples/s]44367 examples [00:41, 1085.70 examples/s]44478 examples [00:41, 1090.65 examples/s]44590 examples [00:41, 1097.89 examples/s]44700 examples [00:41, 1061.97 examples/s]44810 examples [00:41, 1070.60 examples/s]44920 examples [00:41, 1078.51 examples/s]45030 examples [00:41, 1084.18 examples/s]45140 examples [00:41, 1087.56 examples/s]45250 examples [00:42, 1088.69 examples/s]45359 examples [00:42, 1086.05 examples/s]45470 examples [00:42, 1090.53 examples/s]45581 examples [00:42, 1095.55 examples/s]45691 examples [00:42, 1064.41 examples/s]45800 examples [00:42, 1071.63 examples/s]45909 examples [00:42, 1075.09 examples/s]46020 examples [00:42, 1082.70 examples/s]46129 examples [00:42, 1081.21 examples/s]46238 examples [00:42, 1079.02 examples/s]46346 examples [00:43, 1063.75 examples/s]46456 examples [00:43, 1073.25 examples/s]46566 examples [00:43, 1079.06 examples/s]46677 examples [00:43, 1087.06 examples/s]46788 examples [00:43, 1092.79 examples/s]46899 examples [00:43, 1096.02 examples/s]47010 examples [00:43, 1098.25 examples/s]47121 examples [00:43, 1101.03 examples/s]47233 examples [00:43, 1104.57 examples/s]47344 examples [00:44, 1103.04 examples/s]47455 examples [00:44, 1099.74 examples/s]47566 examples [00:44, 1100.28 examples/s]47677 examples [00:44, 1101.64 examples/s]47789 examples [00:44, 1104.37 examples/s]47900 examples [00:44, 1100.06 examples/s]48011 examples [00:44, 1051.99 examples/s]48121 examples [00:44, 1064.91 examples/s]48230 examples [00:44, 1071.71 examples/s]48340 examples [00:44, 1079.28 examples/s]48451 examples [00:45, 1085.51 examples/s]48560 examples [00:45, 1084.96 examples/s]48670 examples [00:45, 1088.59 examples/s]48780 examples [00:45, 1091.92 examples/s]48891 examples [00:45, 1096.63 examples/s]49002 examples [00:45, 1099.57 examples/s]49112 examples [00:45, 1096.33 examples/s]49222 examples [00:45, 1091.40 examples/s]49334 examples [00:45, 1098.76 examples/s]49447 examples [00:45, 1105.40 examples/s]49558 examples [00:46, 1106.75 examples/s]49669 examples [00:46, 1106.65 examples/s]49780 examples [00:46, 1098.31 examples/s]49890 examples [00:46, 1093.56 examples/s]50000 examples [00:46, 1091.08 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7378/50000 [00:00<00:00, 73774.44 examples/s] 42%|████▏     | 20863/50000 [00:00<00:00, 85374.38 examples/s] 68%|██████▊   | 34212/50000 [00:00<00:00, 95723.82 examples/s] 96%|█████████▌| 47899/50000 [00:00<00:00, 105211.93 examples/s]                                                                0 examples [00:00, ? examples/s]93 examples [00:00, 925.27 examples/s]204 examples [00:00, 972.84 examples/s]316 examples [00:00, 1011.11 examples/s]428 examples [00:00, 1039.72 examples/s]539 examples [00:00, 1059.68 examples/s]651 examples [00:00, 1076.22 examples/s]763 examples [00:00, 1088.82 examples/s]865 examples [00:00, 1043.21 examples/s]976 examples [00:00, 1059.19 examples/s]1087 examples [00:01, 1072.80 examples/s]1198 examples [00:01, 1082.79 examples/s]1305 examples [00:01, 1069.61 examples/s]1416 examples [00:01, 1079.64 examples/s]1527 examples [00:01, 1087.10 examples/s]1639 examples [00:01, 1096.15 examples/s]1750 examples [00:01, 1097.26 examples/s]1860 examples [00:01, 1097.29 examples/s]1971 examples [00:01, 1100.53 examples/s]2082 examples [00:01, 1101.49 examples/s]2193 examples [00:02, 1088.94 examples/s]2302 examples [00:02, 1088.87 examples/s]2411 examples [00:02, 1070.41 examples/s]2523 examples [00:02, 1082.92 examples/s]2632 examples [00:02, 1080.99 examples/s]2743 examples [00:02, 1087.29 examples/s]2855 examples [00:02, 1095.68 examples/s]2965 examples [00:02, 1090.85 examples/s]3075 examples [00:02, 1091.08 examples/s]3185 examples [00:02, 1088.82 examples/s]3297 examples [00:03, 1097.92 examples/s]3409 examples [00:03, 1103.97 examples/s]3520 examples [00:03, 1096.51 examples/s]3631 examples [00:03, 1098.83 examples/s]3741 examples [00:03, 1073.51 examples/s]3853 examples [00:03, 1085.38 examples/s]3965 examples [00:03, 1094.29 examples/s]4075 examples [00:03, 1081.80 examples/s]4184 examples [00:03, 1046.96 examples/s]4291 examples [00:03, 1051.22 examples/s]4403 examples [00:04, 1068.37 examples/s]4516 examples [00:04, 1084.54 examples/s]4629 examples [00:04, 1096.63 examples/s]4740 examples [00:04, 1099.70 examples/s]4851 examples [00:04, 1090.73 examples/s]4961 examples [00:04, 1089.69 examples/s]5071 examples [00:04, 1089.95 examples/s]5181 examples [00:04, 1091.85 examples/s]5292 examples [00:04, 1096.97 examples/s]5403 examples [00:04, 1099.60 examples/s]5513 examples [00:05, 1099.32 examples/s]5625 examples [00:05, 1102.57 examples/s]5736 examples [00:05, 1102.47 examples/s]5847 examples [00:05, 1102.25 examples/s]5958 examples [00:05, 1030.91 examples/s]6063 examples [00:05, 1026.38 examples/s]6175 examples [00:05, 1051.02 examples/s]6286 examples [00:05, 1065.78 examples/s]6394 examples [00:05, 1058.45 examples/s]6504 examples [00:06, 1070.10 examples/s]6615 examples [00:06, 1079.51 examples/s]6724 examples [00:06, 1075.76 examples/s]6835 examples [00:06, 1085.28 examples/s]6945 examples [00:06, 1089.20 examples/s]7055 examples [00:06, 1090.20 examples/s]7165 examples [00:06, 1092.92 examples/s]7276 examples [00:06, 1097.79 examples/s]7387 examples [00:06, 1100.47 examples/s]7498 examples [00:06, 1075.39 examples/s]7608 examples [00:07, 1080.26 examples/s]7717 examples [00:07, 1082.02 examples/s]7827 examples [00:07, 1085.78 examples/s]7939 examples [00:07, 1094.99 examples/s]8051 examples [00:07, 1100.03 examples/s]8162 examples [00:07, 1096.77 examples/s]8274 examples [00:07, 1101.49 examples/s]8386 examples [00:07, 1105.41 examples/s]8499 examples [00:07, 1109.96 examples/s]8611 examples [00:07, 1110.63 examples/s]8723 examples [00:08, 1085.84 examples/s]8832 examples [00:08, 1081.25 examples/s]8945 examples [00:08, 1094.82 examples/s]9058 examples [00:08, 1102.95 examples/s]9173 examples [00:08, 1113.80 examples/s]9285 examples [00:08, 1115.30 examples/s]9397 examples [00:08, 1111.99 examples/s]9509 examples [00:08, 1104.82 examples/s]9620 examples [00:08, 1102.06 examples/s]9731 examples [00:08, 1102.74 examples/s]9842 examples [00:09, 1101.97 examples/s]9953 examples [00:09, 1098.08 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1L3F4V/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1L3F4V/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fada8ad1840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fada8ad1840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fada8ad1840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.4893384170532227 	 Accuracy: 58
Train Epoch: 1 	 Loss: 10.729439735412598 	 Accuracy: 140
model saves at 140 accuracy
Train Epoch: 2 	 Loss: 0.5464566564559936 	 Accuracy: 52
Train Epoch: 2 	 Loss: 5.392480802536011 	 Accuracy: 280
model saves at 280 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fad7901d6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fad7901d6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fad7901d6a8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad78590620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad78590620>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fad78590620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([9, 0, 6, 2, 0, 2, 2, 2, 2, 4]), array([9, 8, 1, 2, 2, 2, 2, 8, 5, 8]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fadd9e03630> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fad7857e630> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad6e9f4048>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad6e9f4048>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fad6e9f4048> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:39, 265533.60it/s]  1%|          | 196608/26421880 [00:00<01:15, 346114.62it/s]  2%|▏         | 466944/26421880 [00:00<00:56, 455960.79it/s]  6%|▌         | 1638400/26421880 [00:00<00:38, 638089.98it/s] 15%|█▍        | 3833856/26421880 [00:01<00:25, 894402.06it/s] 28%|██▊       | 7389184/26421880 [00:01<00:15, 1264086.43it/s] 43%|████▎     | 11386880/26421880 [00:01<00:08, 1781617.87it/s] 58%|█████▊    | 15351808/26421880 [00:01<00:04, 2496810.02it/s] 70%|███████   | 18595840/26421880 [00:01<00:02, 3450276.20it/s] 83%|████████▎ | 22003712/26421880 [00:01<00:00, 4722807.47it/s] 96%|█████████▋| 25436160/26421880 [00:01<00:00, 6213699.21it/s]26427392it [00:01, 15671805.27it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 59826.48it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 266136.91it/s]  3%|▎         | 131072/4422102 [00:00<00:13, 327392.50it/s] 10%|▉         | 425984/4422102 [00:00<00:09, 434227.36it/s] 18%|█▊        | 794624/4422102 [00:00<00:06, 586040.88it/s] 29%|██▉       | 1286144/4422102 [00:00<00:04, 773305.54it/s] 37%|███▋      | 1646592/4422102 [00:01<00:02, 997801.79it/s] 44%|████▍     | 1966080/4422102 [00:01<00:02, 1172408.28it/s] 51%|█████▏    | 2277376/4422102 [00:01<00:01, 1409769.26it/s] 59%|█████▉    | 2605056/4422102 [00:01<00:01, 1614813.04it/s] 64%|██████▍   | 2842624/4422102 [00:01<00:00, 1709355.52it/s] 71%|███████▏  | 3153920/4422102 [00:01<00:00, 1918833.95it/s] 79%|███████▉  | 3497984/4422102 [00:01<00:00, 2073688.96it/s] 85%|████████▍ | 3743744/4422102 [00:01<00:00, 2065971.35it/s] 92%|█████████▏| 4063232/4422102 [00:02<00:00, 2234519.19it/s]100%|█████████▉| 4415488/4422102 [00:02<00:00, 2335194.01it/s]4423680it [00:02, 1967766.45it/s]                             
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 34306.67it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003179624776045481 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019202130675315857 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 2 	 Loss: 0.0028220378955205283 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02924057447910309 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.00225891183813413 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.014040170073509215 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.002047371576229731 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.009045246869325637 	 Accuracy: 7
model saves at 7 accuracy
Train Epoch: 5 	 Loss: 0.0020298765252033868 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.012930775046348571 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad6e9ea6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad6e9ea6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fad6e9ea6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([3, 3, 4, 7, 1, 8, 5, 5, 4, 6]), array([5, 4, 4, 5, 1, 8, 5, 5, 4, 4]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fad6eedb0b8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad7828ad90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad7828ad90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fad7828ad90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:45, 109925.48it/s]  0%|          | 40960/18165135 [00:00<02:28, 122115.90it/s]  1%|          | 98304/18165135 [00:00<01:57, 153555.10it/s]  1%|          | 204800/18165135 [00:01<01:29, 200856.19it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 271170.94it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 372522.03it/s]  9%|▉         | 1720320/18165135 [00:01<00:31, 517862.36it/s] 14%|█▎        | 2490368/18165135 [00:01<00:22, 698417.74it/s] 31%|███       | 5636096/18165135 [00:01<00:12, 983710.69it/s] 36%|███▌      | 6479872/18165135 [00:02<00:09, 1290916.77it/s] 40%|███▉      | 7217152/18165135 [00:02<00:06, 1685984.27it/s] 47%|████▋     | 8626176/18165135 [00:02<00:04, 2239452.35it/s] 53%|█████▎    | 9707520/18165135 [00:02<00:02, 2825142.47it/s] 60%|█████▉    | 10821632/18165135 [00:02<00:02, 3472905.42it/s] 66%|██████▌   | 11952128/18165135 [00:02<00:01, 4144705.00it/s] 71%|███████   | 12820480/18165135 [00:02<00:01, 4254505.36it/s] 78%|███████▊  | 14147584/18165135 [00:03<00:00, 5275537.91it/s] 82%|████████▏ | 14974976/18165135 [00:03<00:00, 5352820.88it/s] 87%|████████▋ | 15794176/18165135 [00:03<00:00, 5395135.56it/s] 92%|█████████▏| 16621568/18165135 [00:03<00:00, 5616501.06it/s] 96%|█████████▌| 17457152/18165135 [00:03<00:00, 6137361.11it/s]100%|█████████▉| 18161664/18165135 [00:03<00:00, 6272716.85it/s]18169856it [00:03, 4891038.65it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 109935.68it/s]32768it [00:00, 54531.22it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110107.78it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122288.29it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153729.73it/s]  7%|▋         | 204800/3041136 [00:00<00:14, 201051.52it/s] 14%|█▎        | 417792/3041136 [00:01<00:09, 270861.64it/s] 28%|██▊       | 843776/3041136 [00:01<00:05, 371827.82it/s] 54%|█████▍    | 1654784/3041136 [00:01<00:02, 516043.97it/s] 82%|████████▏ | 2506752/3041136 [00:01<00:00, 709528.74it/s]3047424it [00:01, 1981295.50it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17587.40it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004567464292049408 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.029936624765396116 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.003378482381502787 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.0354429612159729 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.0033348531126976013 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.02858726477622986 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 4 	 Loss: 0.002658199777205785 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.02549759614467621 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0032139700601498287 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.022478964686393738 	 Accuracy: 4
model saves at 4 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fad7828a840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fad7828a840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fad7828a840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([3, 5, 6, 4, 4, 1, 1, 1, 4, 1]), array([3, 2, 6, 0, 9, 7, 1, 1, 9, 1]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

