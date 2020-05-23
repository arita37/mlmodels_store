
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/e9d2938b3714839d6ed21240c11746b217db3d5e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'e9d2938b3714839d6ed21240c11746b217db3d5e', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/e9d2938b3714839d6ed21240c11746b217db3d5e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/e9d2938b3714839d6ed21240c11746b217db3d5e

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/e9d2938b3714839d6ed21240c11746b217db3d5e

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fda2e7a9f98> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fda2d763048>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fda2d763048>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fda2d763048> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 146328.09it/s] 74%|███████▍  | 7364608/9912422 [00:00<00:12, 208860.01it/s]9920512it [00:00, 42001898.90it/s]                           
0it [00:00, ?it/s]32768it [00:00, 468908.49it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 149921.08it/s]1654784it [00:00, 10715701.70it/s]                         
0it [00:00, ?it/s]8192it [00:00, 159049.30it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0031904931167761486 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.019547202825546263 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.003047777573267619 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02087398409843445 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.00197514146566391 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.02126371693611145 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.002015630548199018 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.016948385894298555 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.002529357840617498 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.013173593759536743 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fda2d744ea0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fda2d744ea0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fda2d744ea0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 1, 8, 0, 3, 8, 6, 1, 4, 3]), array([6, 8, 6, 0, 5, 8, 2, 1, 4, 3]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fda2e796630> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fda2ddc17b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fda2ddc17b8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fda2ddc17b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0021691522200902305 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.041368162631988524 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0021412725448608397 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.054916202545166014 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fda2ddc6d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fda2ddc6d90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fda2ddc6d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), array([6, 7, 6, 8, 2, 0, 5, 3, 2, 0]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 27%|██▋       | 12.0M/44.7M [00:00<00:00, 125MB/s] 60%|█████▉    | 26.7M/44.7M [00:00<00:00, 133MB/s] 91%|█████████▏| 40.8M/44.7M [00:00<00:00, 137MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 144MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7fda9b94b3c8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fda2ddbdd90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fda2ddbdd90>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fda2ddbdd90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:15,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:14,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:14,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:13,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:12,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:51,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:50,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:50,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:49,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:48,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:48,  3.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:32,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:31,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:31,  4.25 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:21,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:21,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:21,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:20,  5.96 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 33.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 50.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 50.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
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

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 66.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.68 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.68 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.19 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ url]
0 examples [00:00, ? examples/s]2020-05-23 19:31:08.441770: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 19:31:08.445772: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-23 19:31:08.445913: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c03a84e260 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 19:31:08.445926: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
76 examples [00:00, 726.15 examples/s]181 examples [00:00, 799.61 examples/s]289 examples [00:00, 866.36 examples/s]405 examples [00:00, 935.48 examples/s]520 examples [00:00, 989.21 examples/s]631 examples [00:00, 1020.96 examples/s]741 examples [00:00, 1042.41 examples/s]856 examples [00:00, 1072.14 examples/s]968 examples [00:00, 1083.98 examples/s]1083 examples [00:01, 1101.52 examples/s]1193 examples [00:01, 1099.81 examples/s]1305 examples [00:01, 1104.76 examples/s]1416 examples [00:01, 1103.88 examples/s]1530 examples [00:01, 1111.76 examples/s]1643 examples [00:01, 1110.80 examples/s]1754 examples [00:01, 1068.31 examples/s]1867 examples [00:01, 1084.16 examples/s]1976 examples [00:01, 1044.43 examples/s]2087 examples [00:01, 1061.45 examples/s]2202 examples [00:02, 1084.98 examples/s]2312 examples [00:02, 1086.86 examples/s]2425 examples [00:02, 1096.94 examples/s]2539 examples [00:02, 1107.55 examples/s]2652 examples [00:02, 1113.02 examples/s]2764 examples [00:02, 1110.70 examples/s]2876 examples [00:02, 1098.92 examples/s]2988 examples [00:02, 1104.57 examples/s]3101 examples [00:02, 1109.58 examples/s]3213 examples [00:02, 1094.16 examples/s]3327 examples [00:03, 1107.44 examples/s]3438 examples [00:03, 1106.51 examples/s]3549 examples [00:03, 1074.75 examples/s]3662 examples [00:03, 1088.63 examples/s]3772 examples [00:03, 1088.79 examples/s]3882 examples [00:03, 1067.70 examples/s]3991 examples [00:03, 1071.83 examples/s]4102 examples [00:03, 1082.46 examples/s]4213 examples [00:03, 1088.27 examples/s]4323 examples [00:03, 1090.14 examples/s]4433 examples [00:04, 1089.67 examples/s]4543 examples [00:04, 1078.30 examples/s]4652 examples [00:04, 1079.46 examples/s]4762 examples [00:04, 1084.73 examples/s]4874 examples [00:04, 1092.56 examples/s]4984 examples [00:04, 1087.75 examples/s]5093 examples [00:04, 1075.48 examples/s]5202 examples [00:04, 1077.86 examples/s]5310 examples [00:04, 1060.41 examples/s]5424 examples [00:04, 1082.25 examples/s]5538 examples [00:05, 1096.33 examples/s]5648 examples [00:05, 1021.58 examples/s]5752 examples [00:05, 1005.27 examples/s]5862 examples [00:05, 1030.70 examples/s]5973 examples [00:05, 1052.04 examples/s]6079 examples [00:05, 967.25 examples/s] 6184 examples [00:05, 988.81 examples/s]6287 examples [00:05, 999.89 examples/s]6392 examples [00:05, 1013.53 examples/s]6500 examples [00:06, 1032.39 examples/s]6604 examples [00:06, 1010.26 examples/s]6706 examples [00:06, 1007.56 examples/s]6813 examples [00:06, 1023.89 examples/s]6924 examples [00:06, 1046.01 examples/s]7032 examples [00:06, 1053.69 examples/s]7140 examples [00:06, 1058.97 examples/s]7247 examples [00:06, 1039.31 examples/s]7353 examples [00:06, 1043.92 examples/s]7462 examples [00:06, 1054.87 examples/s]7572 examples [00:07, 1067.94 examples/s]7679 examples [00:07, 1033.38 examples/s]7789 examples [00:07, 1051.50 examples/s]7899 examples [00:07, 1065.36 examples/s]8009 examples [00:07, 1074.37 examples/s]8117 examples [00:07, 1018.88 examples/s]8222 examples [00:07, 1025.29 examples/s]8326 examples [00:07, 1002.61 examples/s]8434 examples [00:07, 1022.17 examples/s]8538 examples [00:08, 1025.71 examples/s]8641 examples [00:08, 1020.95 examples/s]8750 examples [00:08, 1040.27 examples/s]8861 examples [00:08, 1058.84 examples/s]8968 examples [00:08, 1014.93 examples/s]9078 examples [00:08, 1037.84 examples/s]9189 examples [00:08, 1055.92 examples/s]9296 examples [00:08, 1055.10 examples/s]9402 examples [00:08, 1049.60 examples/s]9512 examples [00:08, 1064.01 examples/s]9624 examples [00:09, 1079.02 examples/s]9733 examples [00:09, 1078.12 examples/s]9843 examples [00:09, 1082.49 examples/s]9954 examples [00:09, 1089.15 examples/s]10063 examples [00:09, 1040.17 examples/s]10176 examples [00:09, 1063.08 examples/s]10284 examples [00:09, 1066.35 examples/s]10391 examples [00:09, 1066.63 examples/s]10504 examples [00:09, 1082.66 examples/s]10618 examples [00:09, 1097.29 examples/s]10728 examples [00:10, 1081.65 examples/s]10839 examples [00:10, 1088.87 examples/s]10949 examples [00:10, 1072.36 examples/s]11057 examples [00:10, 1043.57 examples/s]11162 examples [00:10, 1014.82 examples/s]11271 examples [00:10, 1028.65 examples/s]11375 examples [00:10, 1028.20 examples/s]11483 examples [00:10, 1041.15 examples/s]11589 examples [00:10, 1045.47 examples/s]11700 examples [00:11, 1062.19 examples/s]11810 examples [00:11, 1073.11 examples/s]11921 examples [00:11, 1081.98 examples/s]12031 examples [00:11, 1086.69 examples/s]12140 examples [00:11, 1085.45 examples/s]12249 examples [00:11, 1043.02 examples/s]12359 examples [00:11, 1057.47 examples/s]12470 examples [00:11, 1069.72 examples/s]12581 examples [00:11, 1079.35 examples/s]12691 examples [00:11, 1084.07 examples/s]12803 examples [00:12, 1093.58 examples/s]12914 examples [00:12, 1097.21 examples/s]13024 examples [00:12, 1090.76 examples/s]13134 examples [00:12, 1092.63 examples/s]13244 examples [00:12, 1075.65 examples/s]13356 examples [00:12, 1086.68 examples/s]13465 examples [00:12, 1065.75 examples/s]13575 examples [00:12, 1073.39 examples/s]13686 examples [00:12, 1083.86 examples/s]13798 examples [00:12, 1093.73 examples/s]13908 examples [00:13, 1094.23 examples/s]14021 examples [00:13, 1103.25 examples/s]14133 examples [00:13, 1105.67 examples/s]14244 examples [00:13, 1101.73 examples/s]14358 examples [00:13, 1110.25 examples/s]14470 examples [00:13, 1111.68 examples/s]14582 examples [00:13, 1100.88 examples/s]14693 examples [00:13, 1023.39 examples/s]14805 examples [00:13, 1048.59 examples/s]14917 examples [00:13, 1066.77 examples/s]15026 examples [00:14, 1073.15 examples/s]15138 examples [00:14, 1085.10 examples/s]15247 examples [00:14, 1085.36 examples/s]15359 examples [00:14, 1093.57 examples/s]15472 examples [00:14, 1103.11 examples/s]15583 examples [00:14, 1084.75 examples/s]15692 examples [00:14, 1057.00 examples/s]15798 examples [00:14, 1048.94 examples/s]15911 examples [00:14, 1070.20 examples/s]16023 examples [00:15, 1084.16 examples/s]16135 examples [00:15, 1093.69 examples/s]16245 examples [00:15, 1089.46 examples/s]16355 examples [00:15, 1074.63 examples/s]16466 examples [00:15, 1083.98 examples/s]16578 examples [00:15, 1093.89 examples/s]16692 examples [00:15, 1107.30 examples/s]16803 examples [00:15, 1104.61 examples/s]16914 examples [00:15, 1098.24 examples/s]17025 examples [00:15, 1101.49 examples/s]17139 examples [00:16, 1110.41 examples/s]17251 examples [00:16, 1110.58 examples/s]17365 examples [00:16, 1118.50 examples/s]17477 examples [00:16, 1114.47 examples/s]17589 examples [00:16, 1115.67 examples/s]17703 examples [00:16, 1121.16 examples/s]17816 examples [00:16, 1119.71 examples/s]17928 examples [00:16, 1117.72 examples/s]18040 examples [00:16, 1051.82 examples/s]18146 examples [00:16, 1043.78 examples/s]18255 examples [00:17, 1056.69 examples/s]18368 examples [00:17, 1075.45 examples/s]18479 examples [00:17, 1085.13 examples/s]18588 examples [00:17, 1081.53 examples/s]18699 examples [00:17, 1087.90 examples/s]18808 examples [00:17, 1074.51 examples/s]18916 examples [00:17, 1061.37 examples/s]19023 examples [00:17, 1012.26 examples/s]19133 examples [00:17, 1036.53 examples/s]19247 examples [00:17, 1064.94 examples/s]19359 examples [00:18, 1078.50 examples/s]19473 examples [00:18, 1093.75 examples/s]19588 examples [00:18, 1107.40 examples/s]19700 examples [00:18, 1108.66 examples/s]19815 examples [00:18, 1119.29 examples/s]19928 examples [00:18, 1116.04 examples/s]20040 examples [00:18, 1037.99 examples/s]20149 examples [00:18, 1050.98 examples/s]20263 examples [00:18, 1074.13 examples/s]20372 examples [00:19, 1056.83 examples/s]20484 examples [00:19, 1073.80 examples/s]20596 examples [00:19, 1086.46 examples/s]20706 examples [00:19, 1090.23 examples/s]20817 examples [00:19, 1093.95 examples/s]20931 examples [00:19, 1106.67 examples/s]21044 examples [00:19, 1111.07 examples/s]21156 examples [00:19, 1100.83 examples/s]21267 examples [00:19, 1082.70 examples/s]21376 examples [00:19, 1061.28 examples/s]21488 examples [00:20, 1076.23 examples/s]21601 examples [00:20, 1089.44 examples/s]21711 examples [00:20, 1088.35 examples/s]21822 examples [00:20, 1094.26 examples/s]21932 examples [00:20, 1094.64 examples/s]22042 examples [00:20, 1067.41 examples/s]22149 examples [00:20, 1063.59 examples/s]22261 examples [00:20, 1078.70 examples/s]22374 examples [00:20, 1090.80 examples/s]22484 examples [00:20, 1086.61 examples/s]22595 examples [00:21, 1091.91 examples/s]22705 examples [00:21, 1010.16 examples/s]22817 examples [00:21, 1040.60 examples/s]22928 examples [00:21, 1059.12 examples/s]23040 examples [00:21, 1076.30 examples/s]23151 examples [00:21, 1084.04 examples/s]23260 examples [00:21, 1071.67 examples/s]23372 examples [00:21, 1083.14 examples/s]23481 examples [00:21, 1081.52 examples/s]23592 examples [00:21, 1087.36 examples/s]23703 examples [00:22, 1092.38 examples/s]23815 examples [00:22, 1099.15 examples/s]23926 examples [00:22, 1095.99 examples/s]24036 examples [00:22, 1096.29 examples/s]24146 examples [00:22, 1085.31 examples/s]24257 examples [00:22, 1092.56 examples/s]24367 examples [00:22, 1089.16 examples/s]24478 examples [00:22, 1093.04 examples/s]24588 examples [00:22, 1062.70 examples/s]24697 examples [00:23, 1070.29 examples/s]24808 examples [00:23, 1080.73 examples/s]24917 examples [00:23, 1070.81 examples/s]25026 examples [00:23, 1075.27 examples/s]25134 examples [00:23, 1075.34 examples/s]25242 examples [00:23, 1070.64 examples/s]25350 examples [00:23, 1038.81 examples/s]25462 examples [00:23, 1059.65 examples/s]25572 examples [00:23, 1071.43 examples/s]25681 examples [00:23, 1076.30 examples/s]25793 examples [00:24, 1088.01 examples/s]25904 examples [00:24, 1093.71 examples/s]26019 examples [00:24, 1108.53 examples/s]26132 examples [00:24, 1112.89 examples/s]26244 examples [00:24, 1114.78 examples/s]26356 examples [00:24, 1104.12 examples/s]26467 examples [00:24, 1094.37 examples/s]26577 examples [00:24, 1079.13 examples/s]26689 examples [00:24, 1088.82 examples/s]26798 examples [00:24, 1076.07 examples/s]26910 examples [00:25, 1088.36 examples/s]27022 examples [00:25, 1095.07 examples/s]27134 examples [00:25, 1100.08 examples/s]27245 examples [00:25, 1091.96 examples/s]27355 examples [00:25, 1082.09 examples/s]27467 examples [00:25, 1091.41 examples/s]27577 examples [00:25, 1067.17 examples/s]27690 examples [00:25, 1082.79 examples/s]27799 examples [00:25, 1052.30 examples/s]27905 examples [00:25, 1028.98 examples/s]28016 examples [00:26, 1051.27 examples/s]28122 examples [00:26, 1019.84 examples/s]28235 examples [00:26, 1048.27 examples/s]28345 examples [00:26, 1060.51 examples/s]28452 examples [00:26, 1063.14 examples/s]28561 examples [00:26, 1069.87 examples/s]28670 examples [00:26, 1075.62 examples/s]28781 examples [00:26, 1083.55 examples/s]28892 examples [00:26, 1090.71 examples/s]29002 examples [00:27, 1086.13 examples/s]29111 examples [00:27, 1076.11 examples/s]29222 examples [00:27, 1082.50 examples/s]29331 examples [00:27, 1052.80 examples/s]29441 examples [00:27, 1064.91 examples/s]29548 examples [00:27, 1066.04 examples/s]29655 examples [00:27, 1056.68 examples/s]29765 examples [00:27, 1069.30 examples/s]29878 examples [00:27, 1085.55 examples/s]29990 examples [00:27, 1094.78 examples/s]30100 examples [00:28, 1043.00 examples/s]30211 examples [00:28, 1059.90 examples/s]30321 examples [00:28, 1069.22 examples/s]30429 examples [00:28, 1063.79 examples/s]30543 examples [00:28, 1083.95 examples/s]30654 examples [00:28, 1090.87 examples/s]30766 examples [00:28, 1097.04 examples/s]30879 examples [00:28, 1104.17 examples/s]30992 examples [00:28, 1109.34 examples/s]31104 examples [00:28, 1070.35 examples/s]31212 examples [00:29, 1048.93 examples/s]31318 examples [00:29, 1040.05 examples/s]31429 examples [00:29, 1060.08 examples/s]31543 examples [00:29, 1082.68 examples/s]31655 examples [00:29, 1090.77 examples/s]31765 examples [00:29, 1090.27 examples/s]31875 examples [00:29, 1083.51 examples/s]31984 examples [00:29, 1063.39 examples/s]32091 examples [00:29, 1064.05 examples/s]32198 examples [00:29, 1052.84 examples/s]32304 examples [00:30, 1053.85 examples/s]32410 examples [00:30, 1039.32 examples/s]32522 examples [00:30, 1061.02 examples/s]32635 examples [00:30, 1079.44 examples/s]32750 examples [00:30, 1098.84 examples/s]32861 examples [00:30, 1101.94 examples/s]32972 examples [00:30, 1084.93 examples/s]33081 examples [00:30, 1085.53 examples/s]33190 examples [00:30, 1075.83 examples/s]33298 examples [00:31, 1058.74 examples/s]33407 examples [00:31, 1065.29 examples/s]33518 examples [00:31, 1076.70 examples/s]33629 examples [00:31, 1086.16 examples/s]33738 examples [00:31, 1081.36 examples/s]33848 examples [00:31, 1085.61 examples/s]33957 examples [00:31, 1035.34 examples/s]34064 examples [00:31, 1044.59 examples/s]34170 examples [00:31, 1048.27 examples/s]34276 examples [00:31, 1038.91 examples/s]34385 examples [00:32, 1052.59 examples/s]34491 examples [00:32, 1014.08 examples/s]34600 examples [00:32, 1034.04 examples/s]34704 examples [00:32, 1033.65 examples/s]34808 examples [00:32, 1033.65 examples/s]34912 examples [00:32, 1029.04 examples/s]35016 examples [00:32, 1031.79 examples/s]35120 examples [00:32, 1025.02 examples/s]35223 examples [00:32, 996.46 examples/s] 35334 examples [00:32, 1027.38 examples/s]35445 examples [00:33, 1047.76 examples/s]35556 examples [00:33, 1063.07 examples/s]35669 examples [00:33, 1079.90 examples/s]35780 examples [00:33, 1086.31 examples/s]35892 examples [00:33, 1093.79 examples/s]36002 examples [00:33, 1090.58 examples/s]36112 examples [00:33, 1090.31 examples/s]36223 examples [00:33, 1094.28 examples/s]36333 examples [00:33, 1095.81 examples/s]36443 examples [00:33, 1093.06 examples/s]36553 examples [00:34, 1041.57 examples/s]36661 examples [00:34, 1051.12 examples/s]36772 examples [00:34, 1067.94 examples/s]36882 examples [00:34, 1075.22 examples/s]36994 examples [00:34, 1086.47 examples/s]37103 examples [00:34, 1087.51 examples/s]37214 examples [00:34, 1093.63 examples/s]37329 examples [00:34, 1109.72 examples/s]37441 examples [00:34, 1076.94 examples/s]37550 examples [00:35, 1080.62 examples/s]37662 examples [00:35, 1090.42 examples/s]37772 examples [00:35, 1044.67 examples/s]37884 examples [00:35, 1063.83 examples/s]37995 examples [00:35, 1075.83 examples/s]38105 examples [00:35, 1082.05 examples/s]38216 examples [00:35, 1088.72 examples/s]38326 examples [00:35, 1071.13 examples/s]38434 examples [00:35, 1053.30 examples/s]38542 examples [00:35, 1059.50 examples/s]38653 examples [00:36, 1073.30 examples/s]38761 examples [00:36, 1065.48 examples/s]38868 examples [00:36, 1043.99 examples/s]38980 examples [00:36, 1065.11 examples/s]39092 examples [00:36, 1079.17 examples/s]39203 examples [00:36, 1085.44 examples/s]39312 examples [00:36, 1075.61 examples/s]39420 examples [00:36, 1040.46 examples/s]39526 examples [00:36, 1045.55 examples/s]39635 examples [00:36, 1058.14 examples/s]39746 examples [00:37, 1071.84 examples/s]39854 examples [00:37, 1073.95 examples/s]39965 examples [00:37, 1083.95 examples/s]40074 examples [00:37, 969.34 examples/s] 40174 examples [00:37, 971.10 examples/s]40281 examples [00:37, 998.52 examples/s]40384 examples [00:37, 1005.71 examples/s]40487 examples [00:37, 1009.63 examples/s]40589 examples [00:37, 1009.70 examples/s]40695 examples [00:38, 1023.84 examples/s]40803 examples [00:38, 1037.21 examples/s]40912 examples [00:38, 1051.37 examples/s]41018 examples [00:38, 1047.82 examples/s]41123 examples [00:38, 1034.66 examples/s]41229 examples [00:38, 1038.93 examples/s]41334 examples [00:38, 971.82 examples/s] 41433 examples [00:38, 966.77 examples/s]41542 examples [00:38, 999.24 examples/s]41651 examples [00:38, 1023.78 examples/s]41760 examples [00:39, 1040.83 examples/s]41865 examples [00:39, 1039.90 examples/s]41970 examples [00:39, 1038.72 examples/s]42081 examples [00:39, 1056.77 examples/s]42192 examples [00:39, 1071.46 examples/s]42302 examples [00:39, 1077.60 examples/s]42414 examples [00:39, 1089.81 examples/s]42524 examples [00:39, 1087.39 examples/s]42635 examples [00:39, 1093.92 examples/s]42745 examples [00:39, 1064.40 examples/s]42857 examples [00:40, 1080.26 examples/s]42966 examples [00:40, 1053.44 examples/s]43072 examples [00:40, 1048.94 examples/s]43183 examples [00:40, 1064.86 examples/s]43294 examples [00:40, 1077.75 examples/s]43402 examples [00:40, 1059.40 examples/s]43512 examples [00:40, 1070.53 examples/s]43621 examples [00:40, 1075.60 examples/s]43730 examples [00:40, 1078.16 examples/s]43838 examples [00:40, 1068.54 examples/s]43945 examples [00:41, 1059.71 examples/s]44056 examples [00:41, 1073.26 examples/s]44165 examples [00:41, 1076.63 examples/s]44273 examples [00:41, 1072.04 examples/s]44381 examples [00:41, 1056.89 examples/s]44493 examples [00:41, 1072.50 examples/s]44605 examples [00:41, 1083.79 examples/s]44714 examples [00:41, 1080.90 examples/s]44825 examples [00:41, 1087.24 examples/s]44934 examples [00:42, 1072.02 examples/s]45042 examples [00:42, 1069.09 examples/s]45153 examples [00:42, 1080.17 examples/s]45262 examples [00:42, 1079.87 examples/s]45372 examples [00:42, 1085.40 examples/s]45481 examples [00:42, 1083.57 examples/s]45591 examples [00:42, 1085.81 examples/s]45700 examples [00:42, 1005.12 examples/s]45809 examples [00:42, 1028.29 examples/s]45918 examples [00:42, 1045.60 examples/s]46027 examples [00:43, 1056.65 examples/s]46136 examples [00:43, 1064.13 examples/s]46246 examples [00:43, 1073.99 examples/s]46356 examples [00:43, 1079.80 examples/s]46468 examples [00:43, 1090.99 examples/s]46578 examples [00:43, 1088.16 examples/s]46687 examples [00:43, 1087.07 examples/s]46797 examples [00:43, 1089.22 examples/s]46906 examples [00:43, 1085.02 examples/s]47015 examples [00:43, 1056.86 examples/s]47123 examples [00:44, 1061.44 examples/s]47232 examples [00:44, 1069.39 examples/s]47341 examples [00:44, 1074.92 examples/s]47449 examples [00:44, 1061.36 examples/s]47556 examples [00:44, 1027.81 examples/s]47666 examples [00:44, 1046.24 examples/s]47776 examples [00:44, 1059.96 examples/s]47883 examples [00:44, 1027.95 examples/s]47987 examples [00:44, 1029.14 examples/s]48092 examples [00:44, 1033.92 examples/s]48202 examples [00:45, 1050.87 examples/s]48311 examples [00:45, 1061.30 examples/s]48418 examples [00:45, 1008.38 examples/s]48524 examples [00:45, 1021.28 examples/s]48627 examples [00:45, 1019.23 examples/s]48735 examples [00:45, 1034.88 examples/s]48842 examples [00:45, 1044.50 examples/s]48951 examples [00:45, 1057.03 examples/s]49060 examples [00:45, 1066.60 examples/s]49171 examples [00:46, 1077.52 examples/s]49282 examples [00:46, 1084.58 examples/s]49393 examples [00:46, 1091.82 examples/s]49505 examples [00:46, 1099.78 examples/s]49616 examples [00:46, 1086.13 examples/s]49725 examples [00:46, 1069.03 examples/s]49833 examples [00:46, 1037.80 examples/s]49938 examples [00:46, 1034.32 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5685/50000 [00:00<00:00, 56849.11 examples/s] 37%|███▋      | 18715/50000 [00:00<00:00, 68419.15 examples/s] 65%|██████▍   | 32400/50000 [00:00<00:00, 80493.70 examples/s] 93%|█████████▎| 46407/50000 [00:00<00:00, 92265.52 examples/s]                                                               0 examples [00:00, ? examples/s]83 examples [00:00, 828.17 examples/s]183 examples [00:00, 872.79 examples/s]283 examples [00:00, 907.14 examples/s]387 examples [00:00, 922.58 examples/s]497 examples [00:00, 969.41 examples/s]607 examples [00:00, 1004.09 examples/s]713 examples [00:00, 1017.66 examples/s]813 examples [00:00, 1008.57 examples/s]921 examples [00:00, 1026.66 examples/s]1031 examples [00:01, 1045.39 examples/s]1143 examples [00:01, 1065.01 examples/s]1253 examples [00:01, 1073.73 examples/s]1365 examples [00:01, 1085.73 examples/s]1474 examples [00:01, 1069.98 examples/s]1581 examples [00:01, 1058.90 examples/s]1687 examples [00:01, 1053.24 examples/s]1793 examples [00:01, 1047.02 examples/s]1904 examples [00:01, 1063.25 examples/s]2016 examples [00:01, 1077.81 examples/s]2128 examples [00:02, 1089.40 examples/s]2239 examples [00:02, 1093.22 examples/s]2349 examples [00:02, 1074.53 examples/s]2464 examples [00:02, 1093.40 examples/s]2574 examples [00:02, 1045.93 examples/s]2680 examples [00:02, 1030.75 examples/s]2794 examples [00:02, 1060.13 examples/s]2901 examples [00:02, 1038.52 examples/s]3006 examples [00:02, 1040.20 examples/s]3118 examples [00:02, 1061.81 examples/s]3225 examples [00:03, 1060.32 examples/s]3332 examples [00:03, 1050.61 examples/s]3442 examples [00:03, 1064.86 examples/s]3549 examples [00:03, 1012.03 examples/s]3657 examples [00:03, 1029.96 examples/s]3770 examples [00:03, 1057.57 examples/s]3879 examples [00:03, 1066.69 examples/s]3987 examples [00:03, 1054.80 examples/s]4093 examples [00:03, 1026.05 examples/s]4197 examples [00:04, 1019.72 examples/s]4302 examples [00:04, 1026.81 examples/s]4407 examples [00:04, 1032.88 examples/s]4515 examples [00:04, 1045.89 examples/s]4624 examples [00:04, 1057.69 examples/s]4730 examples [00:04, 1057.07 examples/s]4836 examples [00:04, 1049.61 examples/s]4947 examples [00:04, 1065.59 examples/s]5054 examples [00:04, 1053.90 examples/s]5165 examples [00:04, 1067.70 examples/s]5275 examples [00:05, 1074.38 examples/s]5390 examples [00:05, 1093.98 examples/s]5500 examples [00:05, 1094.30 examples/s]5610 examples [00:05, 1074.95 examples/s]5718 examples [00:05, 1054.38 examples/s]5829 examples [00:05, 1068.40 examples/s]5938 examples [00:05, 1072.96 examples/s]6046 examples [00:05, 1023.94 examples/s]6149 examples [00:05, 1017.14 examples/s]6263 examples [00:05, 1050.99 examples/s]6369 examples [00:06, 1040.88 examples/s]6475 examples [00:06, 1044.63 examples/s]6585 examples [00:06, 1060.61 examples/s]6692 examples [00:06, 1043.34 examples/s]6797 examples [00:06, 1030.01 examples/s]6909 examples [00:06, 1055.15 examples/s]7021 examples [00:06, 1073.45 examples/s]7130 examples [00:06, 1075.54 examples/s]7238 examples [00:06, 1064.79 examples/s]7351 examples [00:06, 1082.39 examples/s]7460 examples [00:07, 1076.60 examples/s]7568 examples [00:07, 1074.57 examples/s]7676 examples [00:07, 1067.49 examples/s]7783 examples [00:07, 1060.96 examples/s]7892 examples [00:07, 1068.73 examples/s]7999 examples [00:07, 1059.95 examples/s]8106 examples [00:07, 1012.21 examples/s]8214 examples [00:07, 1028.85 examples/s]8324 examples [00:07, 1048.37 examples/s]8430 examples [00:08, 1048.88 examples/s]8542 examples [00:08, 1068.20 examples/s]8654 examples [00:08, 1081.27 examples/s]8763 examples [00:08, 1069.64 examples/s]8873 examples [00:08, 1076.59 examples/s]8986 examples [00:08, 1090.70 examples/s]9099 examples [00:08, 1100.99 examples/s]9210 examples [00:08, 1080.01 examples/s]9319 examples [00:08, 1031.27 examples/s]9428 examples [00:08, 1047.34 examples/s]9534 examples [00:09, 1049.76 examples/s]9646 examples [00:09, 1068.52 examples/s]9754 examples [00:09, 1002.49 examples/s]9856 examples [00:09, 978.78 examples/s] 9955 examples [00:09, 963.03 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteT9L9EA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteT9L9EA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd9fd4957b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd9fd4957b8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd9fd4957b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.526650230884552 	 Accuracy: 44
Train Epoch: 1 	 Loss: 8.700626754760743 	 Accuracy: 120
model saves at 120 accuracy
Train Epoch: 2 	 Loss: 0.4384253716468811 	 Accuracy: 40
Train Epoch: 2 	 Loss: 37.50219650268555 	 Accuracy: 120

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd9cda2c730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd9cda2c730>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fd9cda2c730> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd9c5f36510>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd9c5f36510>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd9c5f36510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([9, 9, 8, 9, 9, 2, 9, 9, 2, 1]), array([0, 7, 5, 7, 8, 2, 0, 7, 7, 5]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fda2aea0898> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fd9cd9bec50> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd9c704ef28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd9c704ef28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd9c704ef28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 40960/26421880 [00:00<01:20, 326965.93it/s]  0%|          | 106496/26421880 [00:00<01:13, 356531.47it/s]  1%|▏         | 376832/26421880 [00:00<00:54, 475523.28it/s]  4%|▎         | 950272/26421880 [00:00<00:39, 645965.92it/s] 12%|█▏        | 3080192/26421880 [00:00<00:25, 907975.27it/s] 26%|██▌       | 6897664/26421880 [00:01<00:15, 1278172.75it/s] 43%|████▎     | 11468800/26421880 [00:01<00:08, 1804335.04it/s] 57%|█████▋    | 15056896/26421880 [00:01<00:04, 2514874.28it/s] 73%|███████▎  | 19251200/26421880 [00:01<00:02, 3462491.67it/s] 87%|████████▋ | 22896640/26421880 [00:01<00:00, 4751808.48it/s]26427392it [00:01, 16700626.94it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 60203.48it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 270384.83it/s]  4%|▍         | 180224/4422102 [00:00<00:12, 348360.35it/s] 11%|█         | 466944/4422102 [00:00<00:08, 461451.89it/s] 35%|███▌      | 1548288/4422102 [00:00<00:04, 644736.40it/s] 87%|████████▋ | 3833856/4422102 [00:00<00:00, 904641.08it/s]4423680it [00:01, 4420781.51it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 34611.40it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003729278365770976 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020902223825454713 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.002911680499712626 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.027365025520324708 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.002619066963593165 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.013790334463119508 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.001963585500915845 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.009912899911403657 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0017389934659004212 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01061627620458603 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd9c704e620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd9c704e620>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd9c704e620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([0, 7, 7, 4, 6, 3, 0, 2, 6, 4]), array([3, 7, 7, 4, 6, 3, 0, 4, 2, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fda2aea0940> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd9cc5e6d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd9cc5e6d90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd9cc5e6d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110315.65it/s]  0%|          | 40960/18165135 [00:00<02:27, 122494.07it/s]  1%|          | 98304/18165135 [00:00<01:57, 153969.73it/s]  1%|          | 212992/18165135 [00:01<01:28, 202558.13it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272801.37it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 374694.04it/s] 10%|▉         | 1728512/18165135 [00:01<00:31, 520904.55it/s] 15%|█▌        | 2744320/18165135 [00:01<00:21, 709625.27it/s] 30%|███       | 5529600/18165135 [00:01<00:12, 997345.18it/s] 35%|███▍      | 6316032/18165135 [00:02<00:11, 1072395.80it/s] 46%|████▋     | 8429568/18165135 [00:02<00:06, 1480936.27it/s] 52%|█████▏    | 9461760/18165135 [00:02<00:04, 1953527.33it/s] 57%|█████▋    | 10264576/18165135 [00:02<00:03, 2374258.47it/s] 60%|██████    | 10969088/18165135 [00:03<00:02, 2752434.67it/s] 64%|██████▍   | 11591680/18165135 [00:03<00:02, 2603117.41it/s] 67%|██████▋   | 12099584/18165135 [00:03<00:02, 2808068.93it/s] 70%|██████▉   | 12632064/18165135 [00:03<00:01, 3001388.68it/s] 73%|███████▎  | 13180928/18165135 [00:03<00:01, 3177172.67it/s] 76%|███████▌  | 13737984/18165135 [00:03<00:01, 3325822.29it/s] 79%|███████▊  | 14303232/18165135 [00:04<00:01, 3452290.64it/s] 82%|████████▏ | 14868480/18165135 [00:04<00:00, 3546015.09it/s] 85%|████████▌ | 15441920/18165135 [00:04<00:00, 3630448.47it/s] 88%|████████▊ | 16023552/18165135 [00:04<00:00, 3708462.39it/s] 91%|█████████▏| 16605184/18165135 [00:04<00:00, 3764454.40it/s] 95%|█████████▍| 17195008/18165135 [00:04<00:00, 3809752.18it/s] 98%|█████████▊| 17776640/18165135 [00:05<00:00, 3849312.07it/s]18169856it [00:05, 3599065.37it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110192.71it/s]32768it [00:00, 52909.39it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 109972.98it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122219.98it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153661.28it/s]  7%|▋         | 212992/3041136 [00:01<00:13, 202217.78it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 272365.38it/s] 28%|██▊       | 860160/3041136 [00:01<00:05, 374101.09it/s] 46%|████▌     | 1384448/3041136 [00:01<00:03, 511145.86it/s] 64%|██████▎   | 1933312/3041136 [00:01<00:01, 689140.05it/s] 82%|████████▏ | 2506752/3041136 [00:01<00:00, 914278.78it/s]3047424it [00:01, 1654524.56it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 18123.48it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.00401374747355779 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.03384067904949188 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.003557303448518117 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03037525463104248 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.002295521299044291 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.030662638664245605 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.002665768325328827 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.021565177679061888 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.0032526809374491374 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.019522085666656495 	 Accuracy: 4
model saves at 4 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd9cc5e6840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd9cc5e6840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fd9cc5e6840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([4, 7, 8, 6, 8, 1, 1, 9, 1, 4]), array([4, 5, 3, 9, 8, 1, 1, 7, 1, 0]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

