
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/58531b1bea6be1344789995e2201f6ef21c26dc4', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '58531b1bea6be1344789995e2201f6ef21c26dc4', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/58531b1bea6be1344789995e2201f6ef21c26dc4

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/58531b1bea6be1344789995e2201f6ef21c26dc4

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/58531b1bea6be1344789995e2201f6ef21c26dc4

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f0662874358> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f06618ac400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f06618ac400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f06618ac400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 143604.17it/s] 56%|█████▌    | 5521408/9912422 [00:00<00:21, 204918.95it/s]9920512it [00:00, 38260368.74it/s]                           
0it [00:00, ?it/s]32768it [00:00, 593465.75it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 453396.39it/s]1654784it [00:00, 11272715.26it/s]                         
0it [00:00, ?it/s]8192it [00:00, 186748.87it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003314010163148244 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019483379602432252 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.002098155309756597 	 Accuracy: 1
Train Epoch: 2 	 Loss: 0.01990427488088608 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.002546933839718501 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.013755131155252457 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0021501046667496363 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.016533534049987794 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.0011438886026541393 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01625742092728615 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0661899f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0661899f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f0661899f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([7, 1, 8, 2, 9, 4, 7, 3, 9, 1]), array([7, 1, 0, 2, 9, 4, 0, 8, 7, 1]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f0663725518> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0657f2bb70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0657f2bb70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f0657f2bb70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0014203105966250102 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01443051528930664 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0019676421085993447 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.027666662216186525 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0657f2b400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0657f2b400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f0657f2b400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]), array([3, 5, 7, 8, 9, 2, 8, 6, 7, 4]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 27%|██▋       | 11.9M/44.7M [00:00<00:00, 125MB/s] 58%|█████▊    | 25.7M/44.7M [00:00<00:00, 130MB/s] 90%|████████▉ | 40.1M/44.7M [00:00<00:00, 136MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 141MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f06cf7cd828> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f06618781e0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f06618781e0>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f06618781e0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:17,  2.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:17,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:17,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:16,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:16,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:15,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:15,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:14,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:52,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:51,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:50,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:49,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:49,  2.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:32,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:32,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:21,  5.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  8.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.16 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 26.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 57.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 64.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.34 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.34 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.61 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ url]
0 examples [00:00, ? examples/s]2020-05-23 18:09:09.456494: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 18:09:09.469282: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-23 18:09:09.469522: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559cec3093a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 18:09:09.469544: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2 examples [00:00, 19.95 examples/s]110 examples [00:00, 28.28 examples/s]219 examples [00:00, 39.95 examples/s]333 examples [00:00, 56.23 examples/s]442 examples [00:00, 78.57 examples/s]552 examples [00:00, 108.87 examples/s]661 examples [00:00, 149.11 examples/s]772 examples [00:00, 201.35 examples/s]883 examples [00:00, 266.83 examples/s]994 examples [00:01, 345.56 examples/s]1102 examples [00:01, 433.86 examples/s]1210 examples [00:01, 528.74 examples/s]1319 examples [00:01, 625.12 examples/s]1427 examples [00:01, 713.23 examples/s]1537 examples [00:01, 796.71 examples/s]1649 examples [00:01, 870.54 examples/s]1758 examples [00:01, 925.06 examples/s]1867 examples [00:01, 944.23 examples/s]1980 examples [00:01, 993.18 examples/s]2092 examples [00:02, 1027.69 examples/s]2202 examples [00:02, 1015.06 examples/s]2312 examples [00:02, 1037.47 examples/s]2423 examples [00:02, 1057.77 examples/s]2534 examples [00:02, 1070.82 examples/s]2644 examples [00:02, 1078.74 examples/s]2754 examples [00:02, 1082.86 examples/s]2864 examples [00:02, 1086.55 examples/s]2974 examples [00:02, 1088.78 examples/s]3084 examples [00:02, 1089.43 examples/s]3195 examples [00:03, 1092.71 examples/s]3305 examples [00:03, 1078.36 examples/s]3414 examples [00:03, 1077.36 examples/s]3522 examples [00:03, 1076.06 examples/s]3630 examples [00:03, 1059.54 examples/s]3741 examples [00:03, 1074.08 examples/s]3851 examples [00:03, 1080.72 examples/s]3963 examples [00:03, 1090.55 examples/s]4076 examples [00:03, 1099.65 examples/s]4190 examples [00:03, 1110.63 examples/s]4302 examples [00:04, 1108.62 examples/s]4414 examples [00:04, 1111.20 examples/s]4526 examples [00:04, 1109.86 examples/s]4638 examples [00:04, 1103.84 examples/s]4749 examples [00:04, 1098.80 examples/s]4859 examples [00:04, 1097.54 examples/s]4969 examples [00:04, 1090.74 examples/s]5079 examples [00:04, 1087.49 examples/s]5188 examples [00:04, 1046.91 examples/s]5298 examples [00:04, 1061.87 examples/s]5407 examples [00:05, 1069.40 examples/s]5518 examples [00:05, 1080.80 examples/s]5628 examples [00:05, 1085.34 examples/s]5738 examples [00:05, 1088.35 examples/s]5848 examples [00:05, 1089.85 examples/s]5959 examples [00:05, 1094.66 examples/s]6069 examples [00:05, 1087.38 examples/s]6182 examples [00:05, 1098.20 examples/s]6293 examples [00:05, 1099.65 examples/s]6404 examples [00:05, 1101.36 examples/s]6515 examples [00:06, 1079.98 examples/s]6628 examples [00:06, 1093.53 examples/s]6742 examples [00:06, 1105.91 examples/s]6853 examples [00:06, 1081.45 examples/s]6967 examples [00:06, 1095.98 examples/s]7077 examples [00:06, 1084.24 examples/s]7187 examples [00:06, 1088.57 examples/s]7300 examples [00:06, 1098.58 examples/s]7413 examples [00:06, 1106.26 examples/s]7524 examples [00:07, 1093.78 examples/s]7634 examples [00:07, 1094.98 examples/s]7744 examples [00:07, 1095.24 examples/s]7856 examples [00:07, 1100.41 examples/s]7967 examples [00:07, 1094.21 examples/s]8079 examples [00:07, 1099.62 examples/s]8191 examples [00:07, 1103.64 examples/s]8302 examples [00:07, 1102.91 examples/s]8415 examples [00:07, 1109.91 examples/s]8527 examples [00:07, 1066.11 examples/s]8639 examples [00:08, 1080.68 examples/s]8748 examples [00:08, 1081.00 examples/s]8859 examples [00:08, 1087.77 examples/s]8968 examples [00:08, 1074.22 examples/s]9079 examples [00:08, 1084.03 examples/s]9190 examples [00:08, 1090.67 examples/s]9300 examples [00:08, 1080.25 examples/s]9411 examples [00:08, 1086.61 examples/s]9525 examples [00:08, 1101.84 examples/s]9636 examples [00:08, 1098.18 examples/s]9747 examples [00:09, 1099.26 examples/s]9858 examples [00:09, 1101.41 examples/s]9969 examples [00:09, 1098.36 examples/s]10079 examples [00:09, 1041.08 examples/s]10191 examples [00:09, 1062.37 examples/s]10305 examples [00:09, 1082.78 examples/s]10414 examples [00:09, 1082.83 examples/s]10525 examples [00:09, 1089.81 examples/s]10638 examples [00:09, 1098.84 examples/s]10749 examples [00:09, 1088.18 examples/s]10858 examples [00:10, 1075.14 examples/s]10969 examples [00:10, 1082.59 examples/s]11081 examples [00:10, 1091.71 examples/s]11195 examples [00:10, 1103.10 examples/s]11306 examples [00:10, 1104.03 examples/s]11420 examples [00:10, 1111.47 examples/s]11532 examples [00:10, 1100.51 examples/s]11644 examples [00:10, 1105.66 examples/s]11755 examples [00:10, 1097.57 examples/s]11865 examples [00:10, 1059.21 examples/s]11975 examples [00:11, 1069.65 examples/s]12083 examples [00:11, 1070.12 examples/s]12196 examples [00:11, 1086.03 examples/s]12305 examples [00:11, 1061.56 examples/s]12412 examples [00:11, 1052.30 examples/s]12520 examples [00:11, 1057.99 examples/s]12634 examples [00:11, 1078.70 examples/s]12746 examples [00:11, 1089.55 examples/s]12861 examples [00:11, 1104.45 examples/s]12976 examples [00:12, 1117.42 examples/s]13088 examples [00:12, 1102.02 examples/s]13199 examples [00:12, 1067.18 examples/s]13312 examples [00:12, 1083.67 examples/s]13424 examples [00:12, 1091.77 examples/s]13537 examples [00:12, 1102.50 examples/s]13648 examples [00:12, 1093.29 examples/s]13758 examples [00:12, 1090.83 examples/s]13870 examples [00:12, 1098.98 examples/s]13986 examples [00:12, 1114.99 examples/s]14098 examples [00:13, 1111.33 examples/s]14211 examples [00:13, 1116.62 examples/s]14324 examples [00:13, 1118.13 examples/s]14436 examples [00:13, 1117.70 examples/s]14548 examples [00:13, 1115.60 examples/s]14663 examples [00:13, 1123.49 examples/s]14776 examples [00:13, 1117.42 examples/s]14888 examples [00:13, 1108.59 examples/s]14999 examples [00:13, 1106.84 examples/s]15112 examples [00:13, 1109.59 examples/s]15223 examples [00:14, 1103.93 examples/s]15335 examples [00:14, 1106.17 examples/s]15446 examples [00:14, 1104.15 examples/s]15558 examples [00:14, 1107.47 examples/s]15672 examples [00:14, 1114.48 examples/s]15784 examples [00:14, 1109.98 examples/s]15896 examples [00:14, 1109.78 examples/s]16008 examples [00:14, 1112.42 examples/s]16121 examples [00:14, 1115.37 examples/s]16235 examples [00:14, 1121.19 examples/s]16348 examples [00:15, 1119.23 examples/s]16463 examples [00:15, 1127.86 examples/s]16576 examples [00:15, 1104.23 examples/s]16687 examples [00:15, 1104.87 examples/s]16799 examples [00:15, 1108.83 examples/s]16911 examples [00:15, 1111.16 examples/s]17023 examples [00:15, 1062.98 examples/s]17130 examples [00:15, 1055.77 examples/s]17239 examples [00:15, 1061.34 examples/s]17349 examples [00:15, 1071.33 examples/s]17461 examples [00:16, 1084.73 examples/s]17570 examples [00:16, 1069.74 examples/s]17683 examples [00:16, 1085.39 examples/s]17792 examples [00:16, 1070.99 examples/s]17900 examples [00:16, 1073.28 examples/s]18009 examples [00:16, 1076.61 examples/s]18117 examples [00:16, 1075.61 examples/s]18229 examples [00:16, 1086.71 examples/s]18340 examples [00:16, 1091.02 examples/s]18452 examples [00:17, 1096.50 examples/s]18562 examples [00:17, 1068.29 examples/s]18671 examples [00:17, 1072.83 examples/s]18779 examples [00:17, 1073.85 examples/s]18889 examples [00:17, 1079.47 examples/s]19000 examples [00:17, 1086.70 examples/s]19110 examples [00:17, 1090.01 examples/s]19220 examples [00:17, 1071.53 examples/s]19332 examples [00:17, 1084.83 examples/s]19445 examples [00:17, 1095.43 examples/s]19559 examples [00:18, 1106.11 examples/s]19671 examples [00:18, 1109.02 examples/s]19782 examples [00:18, 1093.42 examples/s]19895 examples [00:18, 1102.69 examples/s]20006 examples [00:18, 1041.37 examples/s]20119 examples [00:18, 1064.81 examples/s]20228 examples [00:18, 1069.82 examples/s]20336 examples [00:18, 1071.74 examples/s]20449 examples [00:18, 1086.59 examples/s]20560 examples [00:18, 1093.29 examples/s]20672 examples [00:19, 1099.89 examples/s]20783 examples [00:19, 1099.67 examples/s]20894 examples [00:19, 1101.53 examples/s]21005 examples [00:19, 1092.25 examples/s]21115 examples [00:19, 1090.73 examples/s]21226 examples [00:19, 1095.80 examples/s]21336 examples [00:19, 1081.00 examples/s]21447 examples [00:19, 1087.65 examples/s]21561 examples [00:19, 1102.34 examples/s]21674 examples [00:19, 1110.11 examples/s]21786 examples [00:20, 1103.86 examples/s]21897 examples [00:20, 1066.46 examples/s]22006 examples [00:20, 1071.52 examples/s]22118 examples [00:20, 1085.57 examples/s]22230 examples [00:20, 1093.81 examples/s]22342 examples [00:20, 1101.48 examples/s]22453 examples [00:20, 1085.62 examples/s]22563 examples [00:20, 1089.68 examples/s]22675 examples [00:20, 1097.81 examples/s]22785 examples [00:20, 1088.64 examples/s]22898 examples [00:21, 1097.99 examples/s]23008 examples [00:21, 1085.08 examples/s]23120 examples [00:21, 1093.62 examples/s]23230 examples [00:21, 1085.33 examples/s]23343 examples [00:21, 1097.05 examples/s]23455 examples [00:21, 1103.62 examples/s]23566 examples [00:21, 1083.04 examples/s]23676 examples [00:21, 1085.92 examples/s]23787 examples [00:21, 1090.32 examples/s]23898 examples [00:22, 1093.50 examples/s]24012 examples [00:22, 1105.26 examples/s]24125 examples [00:22, 1111.47 examples/s]24237 examples [00:22, 1108.69 examples/s]24350 examples [00:22, 1114.78 examples/s]24462 examples [00:22, 1101.41 examples/s]24573 examples [00:22, 1093.74 examples/s]24683 examples [00:22, 1067.57 examples/s]24791 examples [00:22, 1070.97 examples/s]24904 examples [00:22, 1085.39 examples/s]25013 examples [00:23, 1070.71 examples/s]25121 examples [00:23, 1067.87 examples/s]25228 examples [00:23, 1051.27 examples/s]25336 examples [00:23, 1058.23 examples/s]25447 examples [00:23, 1069.34 examples/s]25555 examples [00:23, 1013.70 examples/s]25667 examples [00:23, 1042.86 examples/s]25779 examples [00:23, 1063.95 examples/s]25887 examples [00:23, 1066.96 examples/s]25996 examples [00:23, 1072.50 examples/s]26108 examples [00:24, 1084.12 examples/s]26217 examples [00:24, 1082.62 examples/s]26329 examples [00:24, 1092.67 examples/s]26439 examples [00:24, 1094.72 examples/s]26550 examples [00:24, 1097.30 examples/s]26665 examples [00:24, 1112.53 examples/s]26778 examples [00:24, 1115.69 examples/s]26894 examples [00:24, 1126.21 examples/s]27007 examples [00:24, 1124.53 examples/s]27121 examples [00:24, 1128.31 examples/s]27235 examples [00:25, 1128.69 examples/s]27348 examples [00:25, 1119.17 examples/s]27461 examples [00:25, 1120.76 examples/s]27574 examples [00:25, 1121.82 examples/s]27687 examples [00:25, 1120.80 examples/s]27800 examples [00:25, 1116.27 examples/s]27912 examples [00:25, 1104.90 examples/s]28024 examples [00:25, 1109.07 examples/s]28135 examples [00:25, 1104.77 examples/s]28250 examples [00:25, 1115.39 examples/s]28362 examples [00:26, 1115.05 examples/s]28474 examples [00:26, 1092.18 examples/s]28584 examples [00:26, 1069.12 examples/s]28694 examples [00:26, 1077.05 examples/s]28806 examples [00:26, 1088.49 examples/s]28918 examples [00:26, 1096.54 examples/s]29028 examples [00:26, 1084.79 examples/s]29137 examples [00:26, 1084.32 examples/s]29250 examples [00:26, 1096.45 examples/s]29365 examples [00:27, 1109.58 examples/s]29477 examples [00:27, 1107.87 examples/s]29588 examples [00:27, 1091.58 examples/s]29698 examples [00:27, 1093.42 examples/s]29808 examples [00:27, 1085.32 examples/s]29917 examples [00:27, 1085.49 examples/s]30026 examples [00:27, 1029.81 examples/s]30137 examples [00:27, 1051.37 examples/s]30246 examples [00:27, 1048.69 examples/s]30352 examples [00:27, 1049.96 examples/s]30462 examples [00:28, 1063.73 examples/s]30570 examples [00:28, 1067.71 examples/s]30677 examples [00:28, 1065.40 examples/s]30784 examples [00:28, 1058.81 examples/s]30890 examples [00:28, 1047.74 examples/s]30995 examples [00:28, 1031.93 examples/s]31103 examples [00:28, 1045.54 examples/s]31214 examples [00:28, 1063.24 examples/s]31324 examples [00:28, 1073.19 examples/s]31432 examples [00:28, 1063.14 examples/s]31542 examples [00:29, 1072.33 examples/s]31651 examples [00:29, 1075.99 examples/s]31762 examples [00:29, 1083.71 examples/s]31871 examples [00:29, 1071.41 examples/s]31982 examples [00:29, 1080.58 examples/s]32091 examples [00:29, 1058.70 examples/s]32202 examples [00:29, 1071.06 examples/s]32315 examples [00:29, 1085.86 examples/s]32424 examples [00:29, 1083.13 examples/s]32535 examples [00:29, 1088.76 examples/s]32647 examples [00:30, 1097.53 examples/s]32759 examples [00:30, 1102.37 examples/s]32870 examples [00:30, 1101.63 examples/s]32982 examples [00:30, 1104.48 examples/s]33093 examples [00:30, 1093.93 examples/s]33205 examples [00:30, 1098.87 examples/s]33315 examples [00:30, 1092.94 examples/s]33425 examples [00:30, 1092.90 examples/s]33537 examples [00:30, 1099.39 examples/s]33647 examples [00:30, 1069.46 examples/s]33759 examples [00:31, 1083.12 examples/s]33868 examples [00:31, 1057.56 examples/s]33981 examples [00:31, 1077.90 examples/s]34090 examples [00:31, 1067.41 examples/s]34197 examples [00:31, 1035.87 examples/s]34306 examples [00:31, 1050.26 examples/s]34415 examples [00:31, 1060.73 examples/s]34523 examples [00:31, 1064.59 examples/s]34630 examples [00:31, 1044.72 examples/s]34740 examples [00:32, 1058.03 examples/s]34846 examples [00:32, 1056.50 examples/s]34957 examples [00:32, 1071.35 examples/s]35065 examples [00:32, 1025.13 examples/s]35175 examples [00:32, 1044.10 examples/s]35280 examples [00:32, 1032.52 examples/s]35389 examples [00:32, 1048.16 examples/s]35495 examples [00:32, 1041.04 examples/s]35604 examples [00:32, 1054.66 examples/s]35712 examples [00:32, 1061.08 examples/s]35824 examples [00:33, 1077.24 examples/s]35938 examples [00:33, 1093.61 examples/s]36048 examples [00:33, 1066.99 examples/s]36161 examples [00:33, 1084.16 examples/s]36272 examples [00:33, 1089.41 examples/s]36384 examples [00:33, 1096.88 examples/s]36495 examples [00:33, 1098.57 examples/s]36606 examples [00:33, 1098.47 examples/s]36720 examples [00:33, 1108.07 examples/s]36831 examples [00:33, 1098.76 examples/s]36941 examples [00:34, 1095.62 examples/s]37055 examples [00:34, 1107.56 examples/s]37168 examples [00:34, 1112.56 examples/s]37280 examples [00:34, 1108.27 examples/s]37391 examples [00:34, 1103.78 examples/s]37502 examples [00:34, 1095.49 examples/s]37612 examples [00:34, 1086.62 examples/s]37721 examples [00:34, 1066.11 examples/s]37832 examples [00:34, 1077.97 examples/s]37941 examples [00:34, 1081.34 examples/s]38051 examples [00:35, 1084.45 examples/s]38160 examples [00:35, 1083.10 examples/s]38270 examples [00:35, 1087.48 examples/s]38379 examples [00:35, 1084.69 examples/s]38488 examples [00:35, 1079.04 examples/s]38596 examples [00:35, 1067.85 examples/s]38703 examples [00:35, 1055.06 examples/s]38812 examples [00:35, 1064.87 examples/s]38922 examples [00:35, 1074.76 examples/s]39030 examples [00:36, 1073.89 examples/s]39139 examples [00:36, 1076.30 examples/s]39247 examples [00:36, 1075.65 examples/s]39355 examples [00:36, 1070.48 examples/s]39463 examples [00:36, 1066.08 examples/s]39570 examples [00:36, 1063.34 examples/s]39677 examples [00:36, 1034.01 examples/s]39781 examples [00:36, 1034.80 examples/s]39891 examples [00:36, 1052.22 examples/s]40001 examples [00:36, 988.31 examples/s] 40107 examples [00:37, 1008.05 examples/s]40215 examples [00:37, 1026.67 examples/s]40325 examples [00:37, 1046.61 examples/s]40436 examples [00:37, 1064.68 examples/s]40547 examples [00:37, 1076.41 examples/s]40655 examples [00:37, 1061.23 examples/s]40762 examples [00:37, 1043.36 examples/s]40871 examples [00:37, 1055.17 examples/s]40982 examples [00:37, 1070.96 examples/s]41094 examples [00:37, 1082.21 examples/s]41205 examples [00:38, 1089.57 examples/s]41315 examples [00:38, 1089.37 examples/s]41425 examples [00:38, 1059.66 examples/s]41536 examples [00:38, 1072.23 examples/s]41645 examples [00:38, 1076.21 examples/s]41753 examples [00:38, 1075.81 examples/s]41861 examples [00:38, 1034.28 examples/s]41968 examples [00:38, 1019.88 examples/s]42072 examples [00:38, 1025.63 examples/s]42175 examples [00:39, 1021.78 examples/s]42281 examples [00:39, 1030.25 examples/s]42385 examples [00:39, 1020.57 examples/s]42492 examples [00:39, 1033.83 examples/s]42603 examples [00:39, 1055.06 examples/s]42715 examples [00:39, 1071.09 examples/s]42823 examples [00:39, 1059.53 examples/s]42933 examples [00:39, 1070.00 examples/s]43043 examples [00:39, 1076.42 examples/s]43151 examples [00:39, 1073.43 examples/s]43259 examples [00:40, 1063.16 examples/s]43369 examples [00:40, 1071.17 examples/s]43477 examples [00:40, 1071.85 examples/s]43589 examples [00:40, 1084.63 examples/s]43698 examples [00:40, 1084.63 examples/s]43809 examples [00:40, 1090.48 examples/s]43919 examples [00:40, 1088.52 examples/s]44028 examples [00:40, 1086.67 examples/s]44138 examples [00:40, 1087.84 examples/s]44249 examples [00:40, 1092.58 examples/s]44361 examples [00:41, 1098.12 examples/s]44472 examples [00:41, 1100.15 examples/s]44583 examples [00:41, 1087.10 examples/s]44693 examples [00:41, 1088.40 examples/s]44802 examples [00:41, 1080.65 examples/s]44911 examples [00:41, 1077.89 examples/s]45019 examples [00:41, 1059.64 examples/s]45126 examples [00:41, 1023.65 examples/s]45231 examples [00:41, 1030.92 examples/s]45340 examples [00:41, 1047.15 examples/s]45452 examples [00:42, 1065.90 examples/s]45562 examples [00:42, 1074.98 examples/s]45673 examples [00:42, 1083.68 examples/s]45787 examples [00:42, 1099.50 examples/s]45901 examples [00:42, 1109.36 examples/s]46013 examples [00:42, 1105.84 examples/s]46124 examples [00:42, 1102.83 examples/s]46237 examples [00:42, 1108.80 examples/s]46349 examples [00:42, 1111.07 examples/s]46461 examples [00:42, 1113.25 examples/s]46574 examples [00:43, 1118.20 examples/s]46686 examples [00:43, 1110.76 examples/s]46798 examples [00:43, 1108.97 examples/s]46909 examples [00:43, 1105.13 examples/s]47020 examples [00:43, 1082.19 examples/s]47129 examples [00:43, 1078.90 examples/s]47239 examples [00:43, 1085.09 examples/s]47349 examples [00:43, 1089.24 examples/s]47461 examples [00:43, 1095.62 examples/s]47574 examples [00:43, 1103.88 examples/s]47685 examples [00:44, 1105.51 examples/s]47796 examples [00:44, 1095.39 examples/s]47906 examples [00:44, 1062.68 examples/s]48016 examples [00:44, 1072.45 examples/s]48128 examples [00:44, 1084.12 examples/s]48240 examples [00:44, 1091.80 examples/s]48350 examples [00:44, 1089.34 examples/s]48460 examples [00:44, 1054.87 examples/s]48569 examples [00:44, 1062.99 examples/s]48680 examples [00:45, 1076.14 examples/s]48791 examples [00:45, 1085.50 examples/s]48900 examples [00:45, 1085.30 examples/s]49011 examples [00:45, 1089.99 examples/s]49121 examples [00:45, 1090.40 examples/s]49231 examples [00:45, 1089.67 examples/s]49342 examples [00:45, 1094.54 examples/s]49452 examples [00:45, 1077.27 examples/s]49563 examples [00:45, 1085.25 examples/s]49672 examples [00:45, 1077.35 examples/s]49782 examples [00:46, 1083.06 examples/s]49894 examples [00:46, 1092.49 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7356/50000 [00:00<00:00, 73558.84 examples/s] 42%|████▏     | 20944/50000 [00:00<00:00, 85294.67 examples/s] 69%|██████▉   | 34603/50000 [00:00<00:00, 96124.03 examples/s] 96%|█████████▋| 48132/50000 [00:00<00:00, 105265.20 examples/s]                                                                0 examples [00:00, ? examples/s]88 examples [00:00, 879.68 examples/s]199 examples [00:00, 937.66 examples/s]308 examples [00:00, 978.66 examples/s]419 examples [00:00, 1012.32 examples/s]533 examples [00:00, 1045.42 examples/s]636 examples [00:00, 1040.06 examples/s]746 examples [00:00, 1057.33 examples/s]857 examples [00:00, 1070.99 examples/s]967 examples [00:00, 1077.97 examples/s]1073 examples [00:01, 1071.50 examples/s]1182 examples [00:01, 1076.57 examples/s]1289 examples [00:01, 1066.00 examples/s]1395 examples [00:01, 1054.26 examples/s]1509 examples [00:01, 1078.53 examples/s]1623 examples [00:01, 1096.11 examples/s]1734 examples [00:01, 1099.57 examples/s]1847 examples [00:01, 1107.87 examples/s]1959 examples [00:01, 1110.79 examples/s]2071 examples [00:01, 1110.36 examples/s]2183 examples [00:02, 1095.05 examples/s]2293 examples [00:02, 1095.57 examples/s]2403 examples [00:02, 1077.32 examples/s]2514 examples [00:02, 1085.59 examples/s]2625 examples [00:02, 1090.23 examples/s]2737 examples [00:02, 1097.27 examples/s]2848 examples [00:02, 1099.28 examples/s]2960 examples [00:02, 1102.80 examples/s]3072 examples [00:02, 1106.66 examples/s]3183 examples [00:02, 1103.81 examples/s]3294 examples [00:03, 1094.98 examples/s]3404 examples [00:03, 1092.82 examples/s]3516 examples [00:03, 1098.71 examples/s]3627 examples [00:03, 1101.25 examples/s]3741 examples [00:03, 1110.87 examples/s]3857 examples [00:03, 1124.86 examples/s]3970 examples [00:03, 1106.60 examples/s]4081 examples [00:03, 1107.48 examples/s]4193 examples [00:03, 1110.50 examples/s]4305 examples [00:03, 1109.64 examples/s]4417 examples [00:04, 1108.12 examples/s]4528 examples [00:04, 1099.72 examples/s]4639 examples [00:04, 1096.39 examples/s]4749 examples [00:04, 1087.92 examples/s]4861 examples [00:04, 1095.03 examples/s]4971 examples [00:04, 1090.58 examples/s]5081 examples [00:04, 1090.17 examples/s]5191 examples [00:04, 1061.85 examples/s]5298 examples [00:04, 1036.03 examples/s]5406 examples [00:04, 1046.86 examples/s]5517 examples [00:05, 1064.60 examples/s]5627 examples [00:05, 1073.50 examples/s]5737 examples [00:05, 1078.86 examples/s]5848 examples [00:05, 1085.53 examples/s]5957 examples [00:05, 1079.99 examples/s]6068 examples [00:05, 1086.94 examples/s]6179 examples [00:05, 1091.12 examples/s]6292 examples [00:05, 1100.02 examples/s]6403 examples [00:05, 1094.40 examples/s]6513 examples [00:05, 1087.15 examples/s]6625 examples [00:06, 1096.64 examples/s]6736 examples [00:06, 1099.19 examples/s]6849 examples [00:06, 1107.19 examples/s]6962 examples [00:06, 1113.41 examples/s]7075 examples [00:06, 1117.64 examples/s]7187 examples [00:06, 1104.62 examples/s]7298 examples [00:06, 1092.69 examples/s]7409 examples [00:06, 1095.52 examples/s]7520 examples [00:06, 1098.18 examples/s]7630 examples [00:07, 1068.88 examples/s]7739 examples [00:07, 1075.12 examples/s]7849 examples [00:07, 1080.26 examples/s]7958 examples [00:07, 1064.20 examples/s]8065 examples [00:07, 1054.64 examples/s]8174 examples [00:07, 1064.27 examples/s]8282 examples [00:07, 1066.80 examples/s]8391 examples [00:07, 1072.39 examples/s]8499 examples [00:07, 1069.92 examples/s]8610 examples [00:07, 1079.59 examples/s]8719 examples [00:08, 1077.11 examples/s]8830 examples [00:08, 1085.13 examples/s]8940 examples [00:08, 1087.10 examples/s]9050 examples [00:08, 1090.02 examples/s]9160 examples [00:08, 1091.77 examples/s]9271 examples [00:08, 1095.23 examples/s]9382 examples [00:08, 1097.53 examples/s]9492 examples [00:08, 1095.62 examples/s]9602 examples [00:08, 1077.45 examples/s]9712 examples [00:08, 1082.32 examples/s]9821 examples [00:09, 1064.30 examples/s]9935 examples [00:09, 1084.39 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ93MJH/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteJ93MJH/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f065cdcf400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f065cdcf400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f065cdcf400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.47470802068710327 	 Accuracy: 58
Train Epoch: 1 	 Loss: 6.557354068756103 	 Accuracy: 180
model saves at 180 accuracy
Train Epoch: 2 	 Loss: 0.45576890230178835 	 Accuracy: 50
Train Epoch: 2 	 Loss: 5.401175713539123 	 Accuracy: 240
model saves at 240 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f05fdaba510>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f05fdaba510>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f05fdaba510> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f05fc7c3950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f05fc7c3950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f05fc7c3950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([5, 4, 3, 7, 0, 4, 0, 7, 4, 0]), array([1, 6, 3, 5, 8, 2, 7, 2, 2, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f0662874358> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f05fc743550> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f05fa20b6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f05fa20b6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f05fa20b6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:39, 265136.78it/s]  0%|          | 73728/26421880 [00:00<01:51, 237244.07it/s]  0%|          | 114688/26421880 [00:00<01:46, 247037.75it/s]  1%|          | 147456/26421880 [00:00<01:45, 249150.26it/s]  1%|          | 188416/26421880 [00:01<01:49, 239631.31it/s]  1%|          | 262144/26421880 [00:01<01:32, 283351.02it/s]  2%|▏         | 409600/26421880 [00:01<01:11, 365775.06it/s]  3%|▎         | 696320/26421880 [00:01<00:53, 483033.90it/s]  5%|▍         | 1236992/26421880 [00:01<00:38, 657546.42it/s]  9%|▊         | 2310144/26421880 [00:01<00:26, 915147.76it/s] 20%|█▉        | 5193728/26421880 [00:01<00:16, 1289803.41it/s] 32%|███▏      | 8585216/26421880 [00:01<00:09, 1802855.06it/s] 45%|████▍     | 11886592/26421880 [00:02<00:05, 2516602.40it/s] 58%|█████▊    | 15310848/26421880 [00:02<00:03, 3483670.47it/s] 70%|███████   | 18546688/26421880 [00:02<00:01, 4754797.38it/s] 83%|████████▎ | 21987328/26421880 [00:02<00:00, 6398005.28it/s] 99%|█████████▊| 26091520/26421880 [00:02<00:00, 8567586.21it/s]26427392it [00:02, 10665468.28it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 88508.07it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 268303.05it/s]  4%|▍         | 188416/4422102 [00:00<00:12, 351651.13it/s] 11%|█         | 466944/4422102 [00:00<00:08, 462373.40it/s] 33%|███▎      | 1466368/4422102 [00:00<00:04, 647688.69it/s] 73%|███████▎  | 3244032/4422102 [00:00<00:01, 910536.90it/s]4423680it [00:00, 4592239.32it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 39739.10it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0033136102755864463 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02778861176967621 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0028997807105382283 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.020174808084964753 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0027125909626483915 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.01902831768989563 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 4 	 Loss: 0.002403558979431788 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.011314286828041076 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0020223804513613383 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.009260708272457122 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f05fa208e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f05fa208e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f05fa208e18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([9, 0, 8, 8, 9, 9, 1, 0, 4, 3]), array([9, 3, 8, 8, 9, 9, 1, 0, 4, 3]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f05fa26ce10> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f05fa206598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f05fa206598>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f05fa206598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110272.98it/s]  0%|          | 40960/18165135 [00:00<02:27, 122476.40it/s]  1%|          | 98304/18165135 [00:00<01:57, 153943.61it/s]  1%|          | 204800/18165135 [00:01<01:29, 201303.18it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 271765.07it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 373304.28it/s] 10%|▉         | 1728512/18165135 [00:01<00:31, 519026.09it/s] 19%|█▉        | 3465216/18165135 [00:01<00:20, 727564.07it/s] 34%|███▍      | 6193152/18165135 [00:01<00:11, 1010797.06it/s] 42%|████▏     | 7593984/18165135 [00:02<00:07, 1400165.74it/s] 47%|████▋     | 8511488/18165135 [00:02<00:05, 1707430.21it/s] 53%|█████▎    | 9609216/18165135 [00:02<00:04, 2035748.46it/s] 68%|██████▊   | 12271616/18165135 [00:02<00:02, 2772722.42it/s] 73%|███████▎  | 13254656/18165135 [00:02<00:01, 3199208.14it/s] 81%|████████▏ | 14770176/18165135 [00:03<00:00, 4182950.61it/s] 88%|████████▊ | 15941632/18165135 [00:03<00:00, 4863613.40it/s] 94%|█████████▍| 17121280/18165135 [00:03<00:00, 5805474.25it/s]18169856it [00:03, 6702429.60it/s]                              18169856it [00:03, 5289106.12it/s]
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110139.02it/s]32768it [00:00, 52899.67it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 109991.29it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122190.07it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153624.98it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 200926.45it/s] 14%|█▎        | 417792/3041136 [00:01<00:09, 270715.50it/s] 28%|██▊       | 843776/3041136 [00:01<00:05, 370950.82it/s] 55%|█████▌    | 1679360/3041136 [00:01<00:02, 515986.01it/s] 64%|██████▍   | 1949696/3041136 [00:01<00:01, 650266.22it/s]3047424it [00:01, 1815588.98it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17549.42it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0043078807791074114 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.022759110450744628 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0035577967762947084 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.032782558202743534 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.0031796175241470335 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.06643473243713378 	 Accuracy: 2
Train Epoch: 4 	 Loss: 0.0026672591269016267 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.027735094785690306 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 5 	 Loss: 0.002801466683546702 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.02221928608417511 	 Accuracy: 4

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f05fa205d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f05fa205d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f05fa205d08> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([4, 8, 7, 7, 4, 8, 0, 7, 3, 7]), array([1, 2, 8, 3, 2, 2, 4, 7, 6, 0]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

