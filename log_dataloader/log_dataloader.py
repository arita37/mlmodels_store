
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '3aee4395159545a95b0d7c8ed6830ec48eff1164', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3aee4395159545a95b0d7c8ed6830ec48eff1164

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f6997c94390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6996c8d488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6996c8d488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f6996c8d488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 458864.03it/s] 93%|█████████▎| 9256960/9912422 [00:00<00:01, 654122.96it/s]9920512it [00:00, 44588416.95it/s]                           
0it [00:00, ?it/s] 28%|██▊       | 8192/28881 [00:00<00:00, 81889.23it/s]32768it [00:00, 291986.57it/s]                         
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 153547.64it/s]1654784it [00:00, 10665888.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 228128.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003205800712108612 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.021801886916160584 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.002529891053835551 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.014322838366031647 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.001675864855448405 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.02204523515701294 	 Accuracy: 4
Train Epoch: 4 	 Loss: 0.0021562319298585255 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.009238265931606293 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0013590420385201773 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01713257324695587 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6996c78f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6996c78f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f6996c78f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([1, 3, 2, 2, 6, 2, 3, 1, 8, 1]), array([1, 9, 2, 2, 6, 2, 5, 1, 8, 1]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f6998b04550> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f698d2cebf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f698d2cebf8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f698d2cebf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.002002081235249837 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.034770421028137204 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0017318444450696309 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.04979247713088989 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f698d2ce488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f698d2ce488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f698d2ce488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]), array([5, 8, 3, 3, 4, 2, 5, 7, 4, 8]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 19%|█▉        | 8.45M/44.7M [00:00<00:00, 88.6MB/s] 41%|████      | 18.4M/44.7M [00:00<00:00, 92.7MB/s] 64%|██████▍   | 28.5M/44.7M [00:00<00:00, 96.5MB/s] 86%|████████▋ | 38.6M/44.7M [00:00<00:00, 99.0MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 103MB/s] 
  <mlmodels.model_tch.torchhub.Model object at 0x7f69f8ccfe10> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f6996c67268>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f6996c67268>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f6996c67268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:27,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:26,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:26,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:25,  1.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:00,  2.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:00,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:00,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:58,  2.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:41,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:40,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:40,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:40,  3.62 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:27,  5.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:19,  6.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:14,  9.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:14,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:14,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:14,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:13,  9.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:10, 12.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:10, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 20.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 25.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 45.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 45.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 45.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 45.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 45.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 45.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 45.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 45.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 45.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 45.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 45.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 45.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 45.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 45.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 47.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 47.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 51.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 51.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 54.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 54.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 56.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 57.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 57.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 58.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 59.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 59.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 61.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 61.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 63.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 63.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 66.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 66.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 66.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 66.84 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.82s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 66.84 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.82s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.83s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 27.81 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.83s/ url]
0 examples [00:00, ? examples/s]2020-05-23 12:09:49.900820: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 12:09:49.905953: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397215000 Hz
2020-05-23 12:09:49.906472: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5623ac22e540 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 12:09:49.906500: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
62 examples [00:00, 616.49 examples/s]156 examples [00:00, 687.04 examples/s]250 examples [00:00, 746.56 examples/s]338 examples [00:00, 780.81 examples/s]422 examples [00:00, 797.43 examples/s]513 examples [00:00, 826.50 examples/s]608 examples [00:00, 859.20 examples/s]703 examples [00:00, 883.96 examples/s]794 examples [00:00, 891.60 examples/s]882 examples [00:01, 876.06 examples/s]970 examples [00:01, 876.99 examples/s]1064 examples [00:01, 893.82 examples/s]1158 examples [00:01, 904.54 examples/s]1253 examples [00:01, 917.65 examples/s]1345 examples [00:01, 917.82 examples/s]1437 examples [00:01, 900.87 examples/s]1528 examples [00:01, 873.78 examples/s]1620 examples [00:01, 884.46 examples/s]1715 examples [00:01, 901.02 examples/s]1806 examples [00:02, 887.19 examples/s]1898 examples [00:02, 894.95 examples/s]1992 examples [00:02, 907.59 examples/s]2085 examples [00:02, 912.46 examples/s]2182 examples [00:02, 927.75 examples/s]2275 examples [00:02, 909.86 examples/s]2368 examples [00:02, 913.54 examples/s]2460 examples [00:02, 891.97 examples/s]2557 examples [00:02, 912.64 examples/s]2649 examples [00:02, 908.72 examples/s]2741 examples [00:03, 904.95 examples/s]2833 examples [00:03, 906.38 examples/s]2924 examples [00:03, 892.53 examples/s]3018 examples [00:03, 905.73 examples/s]3115 examples [00:03, 922.05 examples/s]3208 examples [00:03, 912.18 examples/s]3300 examples [00:03, 877.66 examples/s]3394 examples [00:03, 894.16 examples/s]3488 examples [00:03, 905.08 examples/s]3580 examples [00:03, 906.08 examples/s]3672 examples [00:04, 908.95 examples/s]3769 examples [00:04, 924.47 examples/s]3862 examples [00:04, 915.47 examples/s]3956 examples [00:04, 922.66 examples/s]4049 examples [00:04, 901.74 examples/s]4140 examples [00:04, 890.32 examples/s]4231 examples [00:04, 889.99 examples/s]4321 examples [00:04, 841.81 examples/s]4406 examples [00:04, 831.80 examples/s]4493 examples [00:05, 842.71 examples/s]4578 examples [00:05, 823.80 examples/s]4661 examples [00:05, 801.85 examples/s]4757 examples [00:05, 841.45 examples/s]4853 examples [00:05, 873.23 examples/s]4948 examples [00:05, 893.75 examples/s]5039 examples [00:05, 871.91 examples/s]5127 examples [00:05, 870.11 examples/s]5219 examples [00:05, 883.04 examples/s]5313 examples [00:05, 897.97 examples/s]5406 examples [00:06, 905.07 examples/s]5499 examples [00:06, 911.51 examples/s]5591 examples [00:06, 903.42 examples/s]5682 examples [00:06, 897.45 examples/s]5772 examples [00:06, 848.76 examples/s]5858 examples [00:06, 820.68 examples/s]5950 examples [00:06, 846.93 examples/s]6036 examples [00:06, 816.58 examples/s]6119 examples [00:06, 771.65 examples/s]6201 examples [00:07, 782.43 examples/s]6280 examples [00:07, 765.86 examples/s]6368 examples [00:07, 796.32 examples/s]6449 examples [00:07, 759.77 examples/s]6529 examples [00:07, 769.08 examples/s]6607 examples [00:07, 759.46 examples/s]6684 examples [00:07, 726.57 examples/s]6758 examples [00:07, 688.09 examples/s]6845 examples [00:07, 732.73 examples/s]6920 examples [00:08, 713.49 examples/s]6997 examples [00:08, 728.90 examples/s]7086 examples [00:08, 769.27 examples/s]7165 examples [00:08, 740.91 examples/s]7241 examples [00:08, 743.45 examples/s]7317 examples [00:08, 733.80 examples/s]7394 examples [00:08, 743.64 examples/s]7483 examples [00:08, 780.02 examples/s]7563 examples [00:08, 783.42 examples/s]7651 examples [00:08, 809.50 examples/s]7739 examples [00:09, 829.15 examples/s]7823 examples [00:09, 806.32 examples/s]7909 examples [00:09, 821.61 examples/s]8000 examples [00:09, 843.93 examples/s]8090 examples [00:09, 859.66 examples/s]8180 examples [00:09, 870.70 examples/s]8268 examples [00:09, 845.71 examples/s]8353 examples [00:09, 805.84 examples/s]8435 examples [00:09, 808.77 examples/s]8519 examples [00:09, 816.21 examples/s]8601 examples [00:10, 773.99 examples/s]8685 examples [00:10, 790.55 examples/s]8776 examples [00:10, 822.77 examples/s]8869 examples [00:10, 851.72 examples/s]8961 examples [00:10, 863.05 examples/s]9049 examples [00:10, 865.77 examples/s]9137 examples [00:10, 868.03 examples/s]9227 examples [00:10, 876.84 examples/s]9315 examples [00:10, 873.15 examples/s]9408 examples [00:11, 887.92 examples/s]9501 examples [00:11, 900.00 examples/s]9592 examples [00:11, 892.33 examples/s]9685 examples [00:11, 899.67 examples/s]9776 examples [00:11, 883.20 examples/s]9868 examples [00:11, 893.50 examples/s]9958 examples [00:11, 894.06 examples/s]10048 examples [00:11, 847.31 examples/s]10135 examples [00:11, 852.79 examples/s]10222 examples [00:11, 856.20 examples/s]10309 examples [00:12, 857.71 examples/s]10395 examples [00:12, 827.26 examples/s]10482 examples [00:12, 838.77 examples/s]10567 examples [00:12, 819.44 examples/s]10655 examples [00:12, 833.51 examples/s]10739 examples [00:12, 834.69 examples/s]10823 examples [00:12, 812.35 examples/s]10912 examples [00:12, 830.91 examples/s]11005 examples [00:12, 858.31 examples/s]11092 examples [00:13, 851.93 examples/s]11185 examples [00:13, 872.72 examples/s]11273 examples [00:13, 842.20 examples/s]11366 examples [00:13, 866.31 examples/s]11461 examples [00:13, 889.67 examples/s]11555 examples [00:13, 903.95 examples/s]11649 examples [00:13, 912.87 examples/s]11741 examples [00:13, 883.22 examples/s]11833 examples [00:13, 893.85 examples/s]11923 examples [00:13, 854.61 examples/s]12010 examples [00:14, 840.21 examples/s]12098 examples [00:14, 850.84 examples/s]12184 examples [00:14, 850.32 examples/s]12276 examples [00:14, 869.89 examples/s]12368 examples [00:14, 883.63 examples/s]12462 examples [00:14, 897.01 examples/s]12553 examples [00:14, 898.53 examples/s]12645 examples [00:14, 901.72 examples/s]12736 examples [00:14, 890.45 examples/s]12826 examples [00:14, 888.42 examples/s]12918 examples [00:15, 897.56 examples/s]13011 examples [00:15, 905.71 examples/s]13102 examples [00:15, 903.87 examples/s]13193 examples [00:15, 891.91 examples/s]13287 examples [00:15, 905.13 examples/s]13381 examples [00:15, 912.80 examples/s]13473 examples [00:15, 897.01 examples/s]13563 examples [00:15, 885.17 examples/s]13652 examples [00:15, 881.28 examples/s]13741 examples [00:15, 859.32 examples/s]13830 examples [00:16, 867.05 examples/s]13919 examples [00:16, 873.52 examples/s]14007 examples [00:16, 869.98 examples/s]14095 examples [00:16, 826.89 examples/s]14179 examples [00:16, 799.20 examples/s]14260 examples [00:16, 795.69 examples/s]14340 examples [00:16, 795.12 examples/s]14420 examples [00:16, 763.94 examples/s]14497 examples [00:16, 678.81 examples/s]14576 examples [00:17, 706.61 examples/s]14659 examples [00:17, 738.36 examples/s]14738 examples [00:17, 752.99 examples/s]14818 examples [00:17, 766.42 examples/s]14896 examples [00:17, 763.80 examples/s]14978 examples [00:17, 759.21 examples/s]15055 examples [00:17, 670.61 examples/s]15125 examples [00:17, 609.56 examples/s]15189 examples [00:18, 578.37 examples/s]15256 examples [00:18, 602.38 examples/s]15319 examples [00:18, 599.38 examples/s]15381 examples [00:18, 585.92 examples/s]15446 examples [00:18, 602.90 examples/s]15512 examples [00:18, 618.71 examples/s]15587 examples [00:18, 652.69 examples/s]15660 examples [00:18, 672.72 examples/s]15731 examples [00:18, 683.26 examples/s]15814 examples [00:18, 720.92 examples/s]15893 examples [00:19, 736.77 examples/s]15968 examples [00:19, 706.47 examples/s]16040 examples [00:19, 690.07 examples/s]16110 examples [00:19, 681.83 examples/s]16179 examples [00:19, 673.62 examples/s]16257 examples [00:19, 700.26 examples/s]16328 examples [00:19, 682.04 examples/s]16397 examples [00:19, 673.28 examples/s]16470 examples [00:19, 687.76 examples/s]16540 examples [00:19, 675.66 examples/s]16608 examples [00:20, 642.48 examples/s]16673 examples [00:20, 634.45 examples/s]16737 examples [00:20, 627.99 examples/s]16801 examples [00:20, 585.99 examples/s]16863 examples [00:20, 580.17 examples/s]16922 examples [00:20, 579.41 examples/s]16990 examples [00:20, 605.34 examples/s]17059 examples [00:20, 626.87 examples/s]17123 examples [00:20, 624.77 examples/s]17200 examples [00:21, 660.53 examples/s]17267 examples [00:21, 635.58 examples/s]17339 examples [00:21, 658.67 examples/s]17406 examples [00:21, 651.86 examples/s]17475 examples [00:21, 662.53 examples/s]17542 examples [00:21, 632.95 examples/s]17627 examples [00:21, 683.72 examples/s]17705 examples [00:21, 709.02 examples/s]17790 examples [00:21, 744.02 examples/s]17866 examples [00:22, 708.12 examples/s]17939 examples [00:22, 694.01 examples/s]18026 examples [00:22, 736.22 examples/s]18102 examples [00:22, 706.48 examples/s]18174 examples [00:22, 709.83 examples/s]18249 examples [00:22, 721.39 examples/s]18322 examples [00:22, 695.58 examples/s]18393 examples [00:22, 687.93 examples/s]18464 examples [00:22, 693.20 examples/s]18535 examples [00:22, 697.56 examples/s]18612 examples [00:23, 715.25 examples/s]18684 examples [00:23, 646.81 examples/s]18764 examples [00:23, 683.62 examples/s]18837 examples [00:23, 694.61 examples/s]18908 examples [00:23, 689.27 examples/s]18978 examples [00:23, 688.73 examples/s]19048 examples [00:23, 684.88 examples/s]19117 examples [00:23, 654.31 examples/s]19184 examples [00:23, 647.54 examples/s]19256 examples [00:24, 667.38 examples/s]19324 examples [00:24, 661.28 examples/s]19391 examples [00:24, 656.70 examples/s]19458 examples [00:24, 660.62 examples/s]19525 examples [00:24, 661.91 examples/s]19592 examples [00:24, 650.95 examples/s]19658 examples [00:24, 646.53 examples/s]19723 examples [00:24, 645.40 examples/s]19788 examples [00:24, 620.57 examples/s]19853 examples [00:24, 627.29 examples/s]19921 examples [00:25, 642.18 examples/s]19991 examples [00:25, 656.87 examples/s]20057 examples [00:25, 643.33 examples/s]20130 examples [00:25, 665.48 examples/s]20199 examples [00:25, 672.45 examples/s]20274 examples [00:25, 692.87 examples/s]20344 examples [00:25, 681.78 examples/s]20413 examples [00:25, 645.55 examples/s]20486 examples [00:25, 666.39 examples/s]20554 examples [00:26, 642.29 examples/s]20629 examples [00:26, 669.74 examples/s]20697 examples [00:26, 652.78 examples/s]20763 examples [00:26, 612.07 examples/s]20828 examples [00:26, 622.95 examples/s]20891 examples [00:26, 605.63 examples/s]20961 examples [00:26, 630.22 examples/s]21033 examples [00:26, 654.44 examples/s]21107 examples [00:26, 676.12 examples/s]21182 examples [00:26, 694.07 examples/s]21253 examples [00:27, 697.79 examples/s]21324 examples [00:27, 698.90 examples/s]21397 examples [00:27, 705.55 examples/s]21468 examples [00:27, 691.47 examples/s]21548 examples [00:27, 714.39 examples/s]21620 examples [00:27, 691.95 examples/s]21700 examples [00:27, 718.67 examples/s]21773 examples [00:27, 705.87 examples/s]21852 examples [00:27, 728.74 examples/s]21926 examples [00:28, 726.10 examples/s]22000 examples [00:28, 729.96 examples/s]22077 examples [00:28, 741.31 examples/s]22152 examples [00:28, 737.78 examples/s]22235 examples [00:28, 759.64 examples/s]22312 examples [00:28, 758.07 examples/s]22388 examples [00:28, 753.15 examples/s]22466 examples [00:28, 759.08 examples/s]22543 examples [00:28, 741.96 examples/s]22618 examples [00:28, 717.48 examples/s]22702 examples [00:29, 748.38 examples/s]22778 examples [00:29, 739.55 examples/s]22853 examples [00:29, 711.58 examples/s]22935 examples [00:29, 738.93 examples/s]23010 examples [00:29, 735.79 examples/s]23085 examples [00:29, 732.21 examples/s]23171 examples [00:29, 766.16 examples/s]23253 examples [00:29, 779.93 examples/s]23332 examples [00:29, 770.67 examples/s]23411 examples [00:29, 773.89 examples/s]23489 examples [00:30, 758.13 examples/s]23566 examples [00:30, 736.88 examples/s]23640 examples [00:30, 721.15 examples/s]23721 examples [00:30, 743.71 examples/s]23799 examples [00:30, 754.16 examples/s]23875 examples [00:30, 691.90 examples/s]23949 examples [00:30, 704.43 examples/s]24021 examples [00:30, 702.17 examples/s]24092 examples [00:30, 677.31 examples/s]24174 examples [00:31, 713.63 examples/s]24262 examples [00:31, 754.70 examples/s]24339 examples [00:31, 711.52 examples/s]24412 examples [00:31, 706.75 examples/s]24484 examples [00:31, 709.37 examples/s]24567 examples [00:31, 740.06 examples/s]24658 examples [00:31, 782.98 examples/s]24749 examples [00:31, 816.91 examples/s]24834 examples [00:31, 823.73 examples/s]24918 examples [00:32, 823.94 examples/s]25002 examples [00:32, 824.25 examples/s]25089 examples [00:32, 835.46 examples/s]25177 examples [00:32, 847.07 examples/s]25267 examples [00:32, 860.43 examples/s]25359 examples [00:32, 876.36 examples/s]25447 examples [00:32, 847.83 examples/s]25533 examples [00:32, 848.75 examples/s]25623 examples [00:32, 862.92 examples/s]25710 examples [00:32, 858.36 examples/s]25797 examples [00:33, 852.71 examples/s]25883 examples [00:33, 845.38 examples/s]25970 examples [00:33, 850.31 examples/s]26056 examples [00:33, 791.44 examples/s]26146 examples [00:33, 820.15 examples/s]26237 examples [00:33, 843.48 examples/s]26324 examples [00:33, 850.38 examples/s]26410 examples [00:33, 842.49 examples/s]26497 examples [00:33, 849.92 examples/s]26584 examples [00:33, 855.55 examples/s]26677 examples [00:34, 874.37 examples/s]26769 examples [00:34, 887.42 examples/s]26858 examples [00:34, 885.55 examples/s]26947 examples [00:34, 876.68 examples/s]27036 examples [00:34, 879.01 examples/s]27127 examples [00:34, 887.36 examples/s]27216 examples [00:34, 886.51 examples/s]27307 examples [00:34, 892.78 examples/s]27398 examples [00:34, 897.15 examples/s]27488 examples [00:34, 883.84 examples/s]27582 examples [00:35, 897.75 examples/s]27672 examples [00:35, 897.06 examples/s]27762 examples [00:35, 891.94 examples/s]27855 examples [00:35, 900.43 examples/s]27947 examples [00:35, 905.45 examples/s]28038 examples [00:35, 891.59 examples/s]28128 examples [00:35, 861.79 examples/s]28215 examples [00:35, 812.82 examples/s]28303 examples [00:35, 830.51 examples/s]28394 examples [00:36, 852.54 examples/s]28486 examples [00:36, 870.44 examples/s]28574 examples [00:36, 870.16 examples/s]28665 examples [00:36, 881.28 examples/s]28758 examples [00:36, 894.64 examples/s]28851 examples [00:36, 902.51 examples/s]28945 examples [00:36, 912.60 examples/s]29037 examples [00:36, 870.74 examples/s]29129 examples [00:36, 883.88 examples/s]29223 examples [00:36, 897.90 examples/s]29315 examples [00:37, 901.54 examples/s]29406 examples [00:37, 890.57 examples/s]29496 examples [00:37, 868.11 examples/s]29585 examples [00:37, 872.05 examples/s]29676 examples [00:37, 882.61 examples/s]29765 examples [00:37, 881.74 examples/s]29854 examples [00:37, 883.04 examples/s]29943 examples [00:37, 874.98 examples/s]30031 examples [00:37, 815.12 examples/s]30118 examples [00:37, 830.69 examples/s]30206 examples [00:38, 843.41 examples/s]30291 examples [00:38, 841.25 examples/s]30380 examples [00:38, 854.53 examples/s]30474 examples [00:38, 877.12 examples/s]30568 examples [00:38, 892.85 examples/s]30662 examples [00:38, 905.31 examples/s]30757 examples [00:38, 915.76 examples/s]30850 examples [00:38, 918.40 examples/s]30946 examples [00:38, 927.94 examples/s]31041 examples [00:39, 933.29 examples/s]31135 examples [00:39, 927.51 examples/s]31228 examples [00:39, 880.91 examples/s]31317 examples [00:39, 867.24 examples/s]31412 examples [00:39, 888.82 examples/s]31506 examples [00:39, 902.70 examples/s]31601 examples [00:39, 913.18 examples/s]31695 examples [00:39, 921.01 examples/s]31788 examples [00:39, 882.95 examples/s]31882 examples [00:39, 897.96 examples/s]31973 examples [00:40, 860.87 examples/s]32066 examples [00:40, 878.94 examples/s]32157 examples [00:40, 886.58 examples/s]32247 examples [00:40, 882.48 examples/s]32339 examples [00:40, 893.35 examples/s]32429 examples [00:40, 865.93 examples/s]32516 examples [00:40, 841.80 examples/s]32608 examples [00:40, 861.39 examples/s]32702 examples [00:40, 881.99 examples/s]32793 examples [00:40, 889.78 examples/s]32885 examples [00:41, 896.03 examples/s]32980 examples [00:41, 910.92 examples/s]33072 examples [00:41, 897.91 examples/s]33166 examples [00:41, 907.78 examples/s]33258 examples [00:41, 909.74 examples/s]33350 examples [00:41, 898.19 examples/s]33440 examples [00:41, 898.70 examples/s]33530 examples [00:41, 895.44 examples/s]33622 examples [00:41, 902.35 examples/s]33715 examples [00:42, 908.24 examples/s]33806 examples [00:42, 861.34 examples/s]33893 examples [00:42, 853.12 examples/s]33979 examples [00:42, 851.75 examples/s]34065 examples [00:42, 845.48 examples/s]34150 examples [00:42, 843.50 examples/s]34240 examples [00:42, 858.63 examples/s]34334 examples [00:42, 880.31 examples/s]34423 examples [00:42, 860.63 examples/s]34510 examples [00:42, 855.06 examples/s]34596 examples [00:43, 855.99 examples/s]34688 examples [00:43, 867.42 examples/s]34778 examples [00:43, 875.86 examples/s]34866 examples [00:43, 842.67 examples/s]34958 examples [00:43, 863.07 examples/s]35050 examples [00:43, 879.03 examples/s]35144 examples [00:43, 895.78 examples/s]35234 examples [00:43, 876.21 examples/s]35322 examples [00:43, 852.39 examples/s]35408 examples [00:43, 853.57 examples/s]35495 examples [00:44, 857.57 examples/s]35581 examples [00:44, 841.26 examples/s]35666 examples [00:44, 822.61 examples/s]35753 examples [00:44, 834.94 examples/s]35837 examples [00:44, 819.36 examples/s]35920 examples [00:44, 811.03 examples/s]36002 examples [00:44, 789.30 examples/s]36082 examples [00:44, 738.55 examples/s]36157 examples [00:44, 740.16 examples/s]36232 examples [00:45, 739.77 examples/s]36311 examples [00:45, 752.31 examples/s]36397 examples [00:45, 781.10 examples/s]36482 examples [00:45, 799.43 examples/s]36563 examples [00:45, 782.53 examples/s]36653 examples [00:45, 811.70 examples/s]36745 examples [00:45, 839.89 examples/s]36835 examples [00:45, 855.14 examples/s]36922 examples [00:45, 853.13 examples/s]37008 examples [00:45, 844.36 examples/s]37095 examples [00:46, 850.56 examples/s]37181 examples [00:46, 849.67 examples/s]37272 examples [00:46, 864.11 examples/s]37359 examples [00:46, 861.73 examples/s]37447 examples [00:46, 866.72 examples/s]37534 examples [00:46, 812.60 examples/s]37617 examples [00:46, 809.72 examples/s]37699 examples [00:46, 797.78 examples/s]37780 examples [00:46, 779.70 examples/s]37860 examples [00:47, 785.37 examples/s]37941 examples [00:47, 792.29 examples/s]38025 examples [00:47, 804.51 examples/s]38111 examples [00:47, 818.46 examples/s]38196 examples [00:47, 827.51 examples/s]38284 examples [00:47, 841.52 examples/s]38370 examples [00:47, 846.64 examples/s]38458 examples [00:47, 854.30 examples/s]38544 examples [00:47, 849.76 examples/s]38630 examples [00:47, 847.92 examples/s]38719 examples [00:48, 858.80 examples/s]38811 examples [00:48, 875.02 examples/s]38904 examples [00:48, 890.51 examples/s]38999 examples [00:48, 905.62 examples/s]39090 examples [00:48, 899.77 examples/s]39181 examples [00:48, 899.93 examples/s]39272 examples [00:48, 869.40 examples/s]39366 examples [00:48, 888.04 examples/s]39458 examples [00:48, 895.70 examples/s]39548 examples [00:48, 886.59 examples/s]39637 examples [00:49, 852.43 examples/s]39723 examples [00:49, 839.82 examples/s]39808 examples [00:49, 819.42 examples/s]39898 examples [00:49, 840.32 examples/s]39984 examples [00:49, 844.23 examples/s]40069 examples [00:49, 804.76 examples/s]40158 examples [00:49, 828.20 examples/s]40246 examples [00:49, 842.97 examples/s]40333 examples [00:49, 848.48 examples/s]40423 examples [00:49, 861.59 examples/s]40510 examples [00:50, 850.57 examples/s]40596 examples [00:50, 835.95 examples/s]40685 examples [00:50, 850.13 examples/s]40775 examples [00:50, 861.54 examples/s]40865 examples [00:50, 870.91 examples/s]40953 examples [00:50, 867.73 examples/s]41040 examples [00:50, 864.84 examples/s]41130 examples [00:50, 872.61 examples/s]41218 examples [00:50, 844.35 examples/s]41304 examples [00:51, 848.39 examples/s]41390 examples [00:51, 838.31 examples/s]41478 examples [00:51, 849.71 examples/s]41567 examples [00:51, 858.89 examples/s]41654 examples [00:51, 860.91 examples/s]41743 examples [00:51, 867.62 examples/s]41830 examples [00:51, 865.49 examples/s]41919 examples [00:51, 872.30 examples/s]42008 examples [00:51, 876.95 examples/s]42096 examples [00:51, 867.58 examples/s]42183 examples [00:52, 812.43 examples/s]42268 examples [00:52, 821.91 examples/s]42355 examples [00:52, 834.02 examples/s]42439 examples [00:52, 782.66 examples/s]42519 examples [00:52, 752.75 examples/s]42608 examples [00:52, 788.14 examples/s]42698 examples [00:52, 816.84 examples/s]42784 examples [00:52, 828.28 examples/s]42872 examples [00:52, 843.12 examples/s]42957 examples [00:53, 837.33 examples/s]43042 examples [00:53, 836.00 examples/s]43126 examples [00:53, 823.91 examples/s]43215 examples [00:53, 841.17 examples/s]43300 examples [00:53, 812.90 examples/s]43388 examples [00:53, 830.33 examples/s]43472 examples [00:53, 832.12 examples/s]43560 examples [00:53, 843.73 examples/s]43646 examples [00:53, 847.74 examples/s]43734 examples [00:53, 856.44 examples/s]43823 examples [00:54, 863.55 examples/s]43911 examples [00:54, 867.20 examples/s]44004 examples [00:54, 883.72 examples/s]44093 examples [00:54, 876.57 examples/s]44184 examples [00:54, 885.74 examples/s]44277 examples [00:54, 898.22 examples/s]44367 examples [00:54, 894.23 examples/s]44460 examples [00:54, 903.97 examples/s]44552 examples [00:54, 908.37 examples/s]44643 examples [00:54, 902.54 examples/s]44734 examples [00:55, 857.96 examples/s]44821 examples [00:55, 849.69 examples/s]44915 examples [00:55, 873.81 examples/s]45003 examples [00:55, 855.18 examples/s]45089 examples [00:55, 850.22 examples/s]45177 examples [00:55, 858.90 examples/s]45264 examples [00:55, 827.27 examples/s]45348 examples [00:55, 814.87 examples/s]45430 examples [00:55, 797.91 examples/s]45511 examples [00:56, 795.78 examples/s]45591 examples [00:56, 794.99 examples/s]45671 examples [00:56, 795.14 examples/s]45751 examples [00:56, 775.65 examples/s]45829 examples [00:56, 776.73 examples/s]45914 examples [00:56, 797.17 examples/s]45996 examples [00:56, 801.25 examples/s]46077 examples [00:56, 792.61 examples/s]46168 examples [00:56, 822.59 examples/s]46260 examples [00:56, 847.31 examples/s]46349 examples [00:57, 859.53 examples/s]46436 examples [00:57, 858.38 examples/s]46525 examples [00:57, 865.00 examples/s]46614 examples [00:57, 871.09 examples/s]46706 examples [00:57, 882.59 examples/s]46796 examples [00:57, 885.56 examples/s]46889 examples [00:57, 895.84 examples/s]46982 examples [00:57, 903.33 examples/s]47074 examples [00:57, 906.92 examples/s]47168 examples [00:57, 915.78 examples/s]47260 examples [00:58, 879.42 examples/s]47349 examples [00:58, 844.62 examples/s]47435 examples [00:58, 846.26 examples/s]47526 examples [00:58, 862.62 examples/s]47615 examples [00:58, 868.93 examples/s]47704 examples [00:58, 862.53 examples/s]47794 examples [00:58, 871.24 examples/s]47887 examples [00:58, 885.49 examples/s]47980 examples [00:58, 896.37 examples/s]48072 examples [00:58, 903.27 examples/s]48163 examples [00:59, 900.29 examples/s]48256 examples [00:59, 901.25 examples/s]48347 examples [00:59, 870.19 examples/s]48438 examples [00:59, 879.93 examples/s]48529 examples [00:59, 887.93 examples/s]48618 examples [00:59, 863.00 examples/s]48705 examples [00:59, 833.19 examples/s]48789 examples [00:59, 816.27 examples/s]48871 examples [00:59, 816.89 examples/s]48957 examples [01:00, 828.35 examples/s]49041 examples [01:00, 822.94 examples/s]49124 examples [01:00, 818.39 examples/s]49215 examples [01:00, 842.83 examples/s]49307 examples [01:00, 863.23 examples/s]49396 examples [01:00, 870.80 examples/s]49484 examples [01:00, 850.67 examples/s]49570 examples [01:00, 832.02 examples/s]49655 examples [01:00, 836.67 examples/s]49742 examples [01:00, 844.13 examples/s]49831 examples [01:01, 857.32 examples/s]49920 examples [01:01, 864.60 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5821/50000 [00:00<00:00, 58209.92 examples/s] 31%|███       | 15476/50000 [00:00<00:00, 66081.88 examples/s] 52%|█████▏    | 26046/50000 [00:00<00:00, 74452.13 examples/s] 72%|███████▏  | 36048/50000 [00:00<00:00, 80633.80 examples/s] 94%|█████████▎| 46762/50000 [00:00<00:00, 87096.41 examples/s]                                                               0 examples [00:00, ? examples/s]76 examples [00:00, 758.95 examples/s]157 examples [00:00, 772.65 examples/s]242 examples [00:00, 793.40 examples/s]331 examples [00:00, 820.00 examples/s]419 examples [00:00, 835.30 examples/s]508 examples [00:00, 849.82 examples/s]592 examples [00:00, 845.91 examples/s]685 examples [00:00, 867.53 examples/s]779 examples [00:00, 886.62 examples/s]872 examples [00:01, 898.76 examples/s]966 examples [00:01, 908.78 examples/s]1056 examples [00:01, 892.59 examples/s]1145 examples [00:01, 862.32 examples/s]1235 examples [00:01, 872.57 examples/s]1329 examples [00:01, 889.70 examples/s]1425 examples [00:01, 907.57 examples/s]1519 examples [00:01, 916.81 examples/s]1613 examples [00:01, 921.68 examples/s]1706 examples [00:01, 902.65 examples/s]1797 examples [00:02, 897.31 examples/s]1887 examples [00:02, 836.14 examples/s]1972 examples [00:02, 825.62 examples/s]2056 examples [00:02, 811.30 examples/s]2138 examples [00:02, 807.19 examples/s]2227 examples [00:02, 828.36 examples/s]2320 examples [00:02, 854.98 examples/s]2411 examples [00:02, 869.85 examples/s]2499 examples [00:02, 844.00 examples/s]2588 examples [00:02, 855.30 examples/s]2682 examples [00:03, 876.87 examples/s]2777 examples [00:03, 896.97 examples/s]2868 examples [00:03, 861.45 examples/s]2955 examples [00:03, 800.55 examples/s]3039 examples [00:03, 810.14 examples/s]3134 examples [00:03, 845.74 examples/s]3227 examples [00:03, 866.08 examples/s]3318 examples [00:03, 876.41 examples/s]3407 examples [00:03, 879.74 examples/s]3496 examples [00:04, 867.27 examples/s]3592 examples [00:04, 891.56 examples/s]3688 examples [00:04, 910.94 examples/s]3782 examples [00:04, 917.09 examples/s]3874 examples [00:04, 903.52 examples/s]3965 examples [00:04, 895.04 examples/s]4055 examples [00:04, 890.18 examples/s]4147 examples [00:04, 896.49 examples/s]4240 examples [00:04, 903.89 examples/s]4332 examples [00:04, 908.44 examples/s]4423 examples [00:05, 902.22 examples/s]4514 examples [00:05, 898.79 examples/s]4607 examples [00:05, 907.38 examples/s]4698 examples [00:05, 907.71 examples/s]4791 examples [00:05, 911.81 examples/s]4883 examples [00:05, 885.82 examples/s]4977 examples [00:05, 899.07 examples/s]5072 examples [00:05, 910.81 examples/s]5165 examples [00:05, 915.81 examples/s]5258 examples [00:05, 917.48 examples/s]5350 examples [00:06, 880.15 examples/s]5443 examples [00:06, 893.31 examples/s]5533 examples [00:06, 894.93 examples/s]5623 examples [00:06, 894.15 examples/s]5717 examples [00:06, 905.10 examples/s]5810 examples [00:06, 910.53 examples/s]5903 examples [00:06, 914.16 examples/s]5995 examples [00:06, 904.19 examples/s]6086 examples [00:06, 902.61 examples/s]6180 examples [00:07, 903.16 examples/s]6274 examples [00:07, 912.23 examples/s]6367 examples [00:07, 917.03 examples/s]6459 examples [00:07, 899.85 examples/s]6550 examples [00:07, 897.58 examples/s]6640 examples [00:07, 898.00 examples/s]6730 examples [00:07, 889.94 examples/s]6822 examples [00:07, 898.34 examples/s]6912 examples [00:07, 893.24 examples/s]7002 examples [00:07, 891.47 examples/s]7094 examples [00:08, 898.02 examples/s]7184 examples [00:08, 893.31 examples/s]7274 examples [00:08, 889.81 examples/s]7364 examples [00:08, 886.46 examples/s]7453 examples [00:08, 857.05 examples/s]7539 examples [00:08, 851.48 examples/s]7625 examples [00:08, 817.61 examples/s]7713 examples [00:08, 833.33 examples/s]7805 examples [00:08, 857.27 examples/s]7892 examples [00:08, 830.95 examples/s]7984 examples [00:09, 854.98 examples/s]8072 examples [00:09, 854.93 examples/s]8158 examples [00:09, 849.49 examples/s]8250 examples [00:09, 868.96 examples/s]8340 examples [00:09, 876.57 examples/s]8431 examples [00:09, 883.75 examples/s]8520 examples [00:09, 871.01 examples/s]8613 examples [00:09, 886.28 examples/s]8707 examples [00:09, 899.70 examples/s]8798 examples [00:09, 897.28 examples/s]8891 examples [00:10, 905.66 examples/s]8987 examples [00:10, 918.69 examples/s]9080 examples [00:10, 918.90 examples/s]9172 examples [00:10, 884.72 examples/s]9261 examples [00:10, 850.56 examples/s]9355 examples [00:10, 874.99 examples/s]9450 examples [00:10, 893.63 examples/s]9540 examples [00:10, 871.11 examples/s]9631 examples [00:10, 880.31 examples/s]9722 examples [00:11, 887.24 examples/s]9815 examples [00:11, 899.58 examples/s]9906 examples [00:11, 897.14 examples/s]9996 examples [00:11, 892.55 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIHKM07/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteIHKM07/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f698e170488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f698e170488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f698e170488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.4845977163314819 	 Accuracy: 50
Train Epoch: 1 	 Loss: 2.2645348072052003 	 Accuracy: 280
model saves at 280 accuracy
Train Epoch: 2 	 Loss: 0.46170180320739745 	 Accuracy: 62
Train Epoch: 2 	 Loss: 10.048411178588868 	 Accuracy: 180

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f6932908730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f6932908730>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f6932908730> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f692fe45950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f692fe45950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f692fe45950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([7, 9, 7, 8, 7, 7, 8, 8, 9, 8]), array([9, 1, 5, 8, 7, 3, 2, 5, 3, 9]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f699455c860> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f692fdcdac8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f692e10c6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f692e10c6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f692e10c6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:39, 265652.68it/s]  1%|          | 221184/26421880 [00:00<01:14, 351766.62it/s]  2%|▏         | 466944/26421880 [00:00<00:56, 457773.22it/s]  5%|▍         | 1245184/26421880 [00:00<00:39, 637831.59it/s] 11%|█         | 2875392/26421880 [00:00<00:26, 896158.21it/s] 25%|██▍       | 6537216/26421880 [00:01<00:15, 1266444.64it/s] 38%|███▊      | 10002432/26421880 [00:01<00:09, 1765046.33it/s] 52%|█████▏    | 13852672/26421880 [00:01<00:05, 2472710.48it/s] 68%|██████▊   | 17924096/26421880 [00:01<00:02, 3442830.64it/s] 81%|████████▏ | 21512192/26421880 [00:01<00:01, 4717350.43it/s] 94%|█████████▎| 24748032/26421880 [00:01<00:00, 6337853.48it/s]26427392it [00:01, 15681962.27it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 90902.69it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 265784.53it/s]  4%|▎         | 155648/4422102 [00:00<00:12, 333721.67it/s] 10%|▉         | 425984/4422102 [00:00<00:09, 441921.17it/s] 27%|██▋       | 1212416/4422102 [00:00<00:05, 616327.66it/s] 60%|█████▉    | 2646016/4422102 [00:00<00:02, 861208.68it/s]4423680it [00:00, 4616416.27it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 31065.07it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003345588207244873 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.032127443075180055 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.003049388249715169 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.022284475803375243 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0026106201112270355 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.021454837322235108 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0020352649092674254 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.029407901048660277 	 Accuracy: 4
Train Epoch: 5 	 Loss: 0.002488917867342631 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.010094940423965455 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f692e10ae18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f692e10ae18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f692e10ae18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([8, 7, 8, 4, 9, 0, 7, 5, 7, 2]), array([8, 5, 8, 2, 9, 0, 7, 5, 7, 2]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f692fe4d978> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f692e176598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f692e176598>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f692e176598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110161.09it/s]  0%|          | 40960/18165135 [00:00<02:28, 122211.78it/s]  1%|          | 98304/18165135 [00:00<01:57, 153680.07it/s]  1%|          | 204800/18165135 [00:01<01:29, 201028.11it/s]  2%|▏         | 417792/18165135 [00:01<01:05, 270818.76it/s]  5%|▍         | 851968/18165135 [00:01<00:46, 372080.62it/s]  9%|▉         | 1703936/18165135 [00:01<00:31, 517128.33it/s] 19%|█▉        | 3416064/18165135 [00:01<00:20, 724780.21it/s] 33%|███▎      | 5971968/18165135 [00:01<00:12, 1005203.63it/s] 41%|████      | 7430144/18165135 [00:02<00:07, 1393946.24it/s] 48%|████▊     | 8757248/18165135 [00:02<00:04, 1899282.21it/s] 54%|█████▍    | 9764864/18165135 [00:02<00:03, 2315116.66it/s] 59%|█████▉    | 10706944/18165135 [00:02<00:02, 2918820.15it/s] 66%|██████▋   | 12042240/18165135 [00:02<00:01, 3497747.58it/s] 77%|███████▋  | 14024704/18165135 [00:02<00:00, 4645154.47it/s] 83%|████████▎ | 15155200/18165135 [00:02<00:00, 5241754.20it/s] 89%|████████▉ | 16171008/18165135 [00:03<00:00, 5713628.74it/s] 94%|█████████▍| 17104896/18165135 [00:03<00:00, 6335372.53it/s]18169856it [00:03, 5574007.18it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110212.50it/s]32768it [00:00, 54016.83it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110029.33it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122238.02it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153709.14it/s]  7%|▋         | 212992/3041136 [00:00<00:13, 202276.52it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 272436.81it/s] 29%|██▊       | 868352/3041136 [00:01<00:05, 374403.93it/s] 57%|█████▋    | 1728512/3041136 [00:01<00:02, 520406.50it/s] 94%|█████████▍| 2859008/3041136 [00:01<00:00, 722191.88it/s]3047424it [00:01, 2026390.48it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17623.18it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004196008284886678 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.03067164969444275 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0034171117544174196 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03707311630249024 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.00364110670487086 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.026695759654045106 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0032480327486991882 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.02621695077419281 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0028163289626439414 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.017576359152793885 	 Accuracy: 4
model saves at 4 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f692e111d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f692e111d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f692e111d08> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([2, 9, 0, 9, 0, 1, 0, 8, 2, 8]), array([4, 9, 0, 9, 9, 9, 0, 1, 2, 8]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

