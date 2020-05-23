
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/1304d63f8392925ad75755daa3312ccffd819b56', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '1304d63f8392925ad75755daa3312ccffd819b56', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/1304d63f8392925ad75755daa3312ccffd819b56

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/1304d63f8392925ad75755daa3312ccffd819b56

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/1304d63f8392925ad75755daa3312ccffd819b56

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa69e47ff60> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa69d4390d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa69d4390d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa69d4390d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 159722.10it/s] 83%|████████▎ | 8183808/9912422 [00:00<00:07, 227983.36it/s]9920512it [00:00, 45200556.15it/s]                           
0it [00:00, ?it/s]32768it [00:00, 475185.85it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 158435.82it/s]1654784it [00:00, 11001460.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 160265.21it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0033615288933118185 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01932040047645569 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.002521169304847717 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.044241687059402464 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.002426634728908539 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.029714333534240724 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0018089506427447 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.01562684863805771 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0012855009486277898 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.00736866144835949 	 Accuracy: 7
model saves at 7 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa69d41af28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa69d41af28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa69d41af28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 6, 7, 3, 0, 3, 3, 0, 0, 5]), array([8, 6, 7, 3, 6, 2, 3, 3, 0, 5]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa69e46c5f8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa69da96730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa69da96730>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa69da96730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0020397337675094602 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018085808992385863 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.002617170770963033 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.027463670253753663 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa69da9ad90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa69da9ad90>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa69da9ad90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6]), array([7, 5, 9, 1, 2, 4, 3, 5, 3, 2]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 19%|█▉        | 8.57M/44.7M [00:00<00:00, 89.9MB/s] 38%|███▊      | 17.1M/44.7M [00:00<00:00, 89.6MB/s] 59%|█████▉    | 26.4M/44.7M [00:00<00:00, 91.8MB/s] 80%|████████  | 35.9M/44.7M [00:00<00:00, 94.0MB/s] 99%|█████████▉| 44.3M/44.7M [00:00<00:00, 92.4MB/s]100%|██████████| 44.7M/44.7M [00:00<00:00, 93.0MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7fa69a7e0470> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa69d400e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa69d400e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fa69d400e18> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:56,  2.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:56,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:54,  2.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:17,  7.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:17,  7.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:17,  7.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:01<00:12,  9.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12,  9.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 13.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:08, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 22.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 22.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 28.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 47.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 57.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 63.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 63.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.03 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 71.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 71.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.83s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.83s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.83s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 71.35 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.83s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 71.35 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.14s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.15s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.48 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.15s/ url]
0 examples [00:00, ? examples/s]2020-05-23 18:33:11.465130: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 18:33:11.532159: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-23 18:33:11.532835: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563975017900 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 18:33:11.532858: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  4.86 examples/s]105 examples [00:00,  6.93 examples/s]205 examples [00:00,  9.88 examples/s]304 examples [00:00, 14.05 examples/s]402 examples [00:00, 19.95 examples/s]489 examples [00:00, 28.22 examples/s]592 examples [00:00, 39.84 examples/s]693 examples [00:00, 55.96 examples/s]791 examples [00:01, 78.04 examples/s]886 examples [00:01, 107.66 examples/s]989 examples [00:01, 147.17 examples/s]1095 examples [00:01, 198.33 examples/s]1197 examples [00:01, 261.48 examples/s]1297 examples [00:01, 335.42 examples/s]1399 examples [00:01, 419.74 examples/s]1499 examples [00:01, 505.53 examples/s]1599 examples [00:01, 593.45 examples/s]1699 examples [00:01, 674.03 examples/s]1799 examples [00:02, 688.76 examples/s]1891 examples [00:02, 721.71 examples/s]1980 examples [00:02, 741.43 examples/s]2077 examples [00:02, 797.25 examples/s]2167 examples [00:02, 823.98 examples/s]2268 examples [00:02, 870.83 examples/s]2373 examples [00:02, 917.05 examples/s]2471 examples [00:02, 933.39 examples/s]2576 examples [00:02, 965.49 examples/s]2679 examples [00:02, 981.42 examples/s]2783 examples [00:03, 998.22 examples/s]2885 examples [00:03, 971.16 examples/s]2984 examples [00:03, 960.18 examples/s]3081 examples [00:03, 948.30 examples/s]3179 examples [00:03, 956.68 examples/s]3280 examples [00:03, 971.81 examples/s]3378 examples [00:03, 970.23 examples/s]3480 examples [00:03, 981.58 examples/s]3580 examples [00:03, 984.15 examples/s]3683 examples [00:04, 994.98 examples/s]3784 examples [00:04, 997.84 examples/s]3884 examples [00:04, 965.43 examples/s]3990 examples [00:04, 990.04 examples/s]4091 examples [00:04, 994.74 examples/s]4191 examples [00:04, 981.61 examples/s]4293 examples [00:04, 992.62 examples/s]4393 examples [00:04, 986.44 examples/s]4493 examples [00:04, 988.22 examples/s]4593 examples [00:04, 989.54 examples/s]4696 examples [00:05, 999.16 examples/s]4797 examples [00:05, 1002.08 examples/s]4900 examples [00:05, 1007.99 examples/s]5002 examples [00:05, 1011.39 examples/s]5104 examples [00:05, 987.99 examples/s] 5209 examples [00:05, 1003.51 examples/s]5310 examples [00:05, 1004.58 examples/s]5414 examples [00:05, 1012.74 examples/s]5516 examples [00:05, 1008.92 examples/s]5617 examples [00:05, 971.19 examples/s] 5715 examples [00:06, 954.05 examples/s]5815 examples [00:06, 966.85 examples/s]5912 examples [00:06, 952.76 examples/s]6008 examples [00:06, 952.04 examples/s]6104 examples [00:06, 912.69 examples/s]6196 examples [00:06, 910.06 examples/s]6288 examples [00:06, 911.46 examples/s]6380 examples [00:06, 906.66 examples/s]6471 examples [00:06, 895.85 examples/s]6563 examples [00:07, 900.32 examples/s]6659 examples [00:07, 915.30 examples/s]6751 examples [00:07, 903.08 examples/s]6846 examples [00:07, 914.15 examples/s]6938 examples [00:07, 888.28 examples/s]7028 examples [00:07, 882.10 examples/s]7132 examples [00:07, 922.94 examples/s]7226 examples [00:07, 927.90 examples/s]7320 examples [00:07, 915.27 examples/s]7418 examples [00:07, 930.93 examples/s]7512 examples [00:08, 877.59 examples/s]7601 examples [00:08, 879.79 examples/s]7695 examples [00:08, 895.17 examples/s]7795 examples [00:08, 923.81 examples/s]7896 examples [00:08, 945.28 examples/s]7998 examples [00:08, 965.47 examples/s]8100 examples [00:08, 981.03 examples/s]8199 examples [00:08, 930.42 examples/s]8302 examples [00:08, 956.39 examples/s]8403 examples [00:08, 970.54 examples/s]8503 examples [00:09, 977.78 examples/s]8605 examples [00:09, 987.40 examples/s]8709 examples [00:09, 1000.35 examples/s]8810 examples [00:09, 971.36 examples/s] 8909 examples [00:09, 974.86 examples/s]9010 examples [00:09, 985.10 examples/s]9109 examples [00:09, 973.43 examples/s]9207 examples [00:09, 973.99 examples/s]9306 examples [00:09, 977.32 examples/s]9404 examples [00:10, 956.21 examples/s]9502 examples [00:10, 962.28 examples/s]9602 examples [00:10, 971.72 examples/s]9704 examples [00:10, 983.12 examples/s]9803 examples [00:10, 982.09 examples/s]9902 examples [00:10, 951.98 examples/s]9999 examples [00:10, 956.67 examples/s]10095 examples [00:10, 904.88 examples/s]10195 examples [00:10, 930.48 examples/s]10298 examples [00:10, 956.31 examples/s]10396 examples [00:11, 962.86 examples/s]10499 examples [00:11, 980.41 examples/s]10602 examples [00:11, 994.22 examples/s]10705 examples [00:11, 1003.18 examples/s]10808 examples [00:11, 1008.69 examples/s]10910 examples [00:11, 990.64 examples/s] 11010 examples [00:11, 981.44 examples/s]11109 examples [00:11, 974.79 examples/s]11213 examples [00:11, 991.41 examples/s]11315 examples [00:11, 998.82 examples/s]11416 examples [00:12, 991.89 examples/s]11517 examples [00:12, 995.53 examples/s]11620 examples [00:12, 1005.53 examples/s]11722 examples [00:12, 1007.09 examples/s]11823 examples [00:12, 1005.50 examples/s]11924 examples [00:12, 938.13 examples/s] 12022 examples [00:12, 948.17 examples/s]12122 examples [00:12, 961.70 examples/s]12227 examples [00:12, 985.23 examples/s]12330 examples [00:12, 997.74 examples/s]12433 examples [00:13, 1007.02 examples/s]12535 examples [00:13, 1008.05 examples/s]12637 examples [00:13, 1005.89 examples/s]12742 examples [00:13, 1016.71 examples/s]12844 examples [00:13, 998.04 examples/s] 12944 examples [00:13, 993.36 examples/s]13044 examples [00:13, 994.40 examples/s]13149 examples [00:13, 1009.28 examples/s]13254 examples [00:13, 1019.51 examples/s]13358 examples [00:14, 1024.26 examples/s]13461 examples [00:14, 993.53 examples/s] 13566 examples [00:14, 1008.89 examples/s]13671 examples [00:14, 1019.07 examples/s]13774 examples [00:14, 1014.35 examples/s]13876 examples [00:14, 1013.36 examples/s]13978 examples [00:14, 1011.76 examples/s]14080 examples [00:14, 1005.48 examples/s]14183 examples [00:14, 1011.63 examples/s]14285 examples [00:14, 1013.01 examples/s]14387 examples [00:15, 1009.18 examples/s]14488 examples [00:15, 1000.67 examples/s]14589 examples [00:15, 993.48 examples/s] 14689 examples [00:15, 976.80 examples/s]14787 examples [00:15, 969.78 examples/s]14887 examples [00:15, 976.78 examples/s]14985 examples [00:15, 958.97 examples/s]15085 examples [00:15, 969.59 examples/s]15184 examples [00:15, 974.91 examples/s]15282 examples [00:15, 976.14 examples/s]15380 examples [00:16, 965.29 examples/s]15479 examples [00:16, 972.05 examples/s]15582 examples [00:16, 988.13 examples/s]15684 examples [00:16, 995.82 examples/s]15784 examples [00:16, 971.08 examples/s]15882 examples [00:16, 907.81 examples/s]15974 examples [00:16, 911.12 examples/s]16076 examples [00:16, 941.01 examples/s]16181 examples [00:16, 970.92 examples/s]16282 examples [00:16, 980.59 examples/s]16384 examples [00:17, 989.63 examples/s]16488 examples [00:17, 1004.00 examples/s]16591 examples [00:17, 1010.47 examples/s]16697 examples [00:17, 1022.29 examples/s]16800 examples [00:17, 999.70 examples/s] 16902 examples [00:17, 1005.38 examples/s]17003 examples [00:17, 991.95 examples/s] 17103 examples [00:17, 980.30 examples/s]17205 examples [00:17, 991.29 examples/s]17308 examples [00:18, 1001.31 examples/s]17409 examples [00:18, 998.73 examples/s] 17509 examples [00:18, 977.70 examples/s]17609 examples [00:18, 983.28 examples/s]17712 examples [00:18, 995.09 examples/s]17812 examples [00:18, 994.17 examples/s]17912 examples [00:18, 973.51 examples/s]18010 examples [00:18, 937.56 examples/s]18112 examples [00:18, 958.38 examples/s]18215 examples [00:18, 976.19 examples/s]18317 examples [00:19, 987.78 examples/s]18420 examples [00:19, 998.34 examples/s]18525 examples [00:19, 1012.53 examples/s]18628 examples [00:19, 1017.17 examples/s]18731 examples [00:19, 1020.48 examples/s]18834 examples [00:19, 1017.22 examples/s]18936 examples [00:19, 975.62 examples/s] 19039 examples [00:19, 990.70 examples/s]19142 examples [00:19, 1000.75 examples/s]19245 examples [00:19, 1008.38 examples/s]19347 examples [00:20, 1009.76 examples/s]19449 examples [00:20, 1008.23 examples/s]19553 examples [00:20, 1015.95 examples/s]19655 examples [00:20, 1017.01 examples/s]19757 examples [00:20, 1006.90 examples/s]19858 examples [00:20, 999.55 examples/s] 19959 examples [00:20, 976.25 examples/s]20057 examples [00:20, 936.64 examples/s]20162 examples [00:20, 966.62 examples/s]20267 examples [00:20, 987.43 examples/s]20367 examples [00:21, 968.03 examples/s]20465 examples [00:21, 960.21 examples/s]20568 examples [00:21, 978.45 examples/s]20669 examples [00:21, 985.21 examples/s]20768 examples [00:21, 979.94 examples/s]20867 examples [00:21, 982.36 examples/s]20966 examples [00:21, 957.96 examples/s]21064 examples [00:21, 962.28 examples/s]21161 examples [00:21, 949.57 examples/s]21257 examples [00:22, 951.16 examples/s]21354 examples [00:22, 955.51 examples/s]21450 examples [00:22, 931.96 examples/s]21552 examples [00:22, 956.03 examples/s]21654 examples [00:22, 974.32 examples/s]21757 examples [00:22, 987.60 examples/s]21859 examples [00:22, 996.87 examples/s]21959 examples [00:22, 961.61 examples/s]22056 examples [00:22, 961.06 examples/s]22157 examples [00:22, 974.96 examples/s]22255 examples [00:23, 938.81 examples/s]22352 examples [00:23, 947.67 examples/s]22448 examples [00:23, 921.76 examples/s]22552 examples [00:23, 952.22 examples/s]22650 examples [00:23, 959.93 examples/s]22754 examples [00:23, 980.26 examples/s]22853 examples [00:23, 979.33 examples/s]22957 examples [00:23, 995.59 examples/s]23062 examples [00:23, 1010.72 examples/s]23167 examples [00:23, 1020.49 examples/s]23272 examples [00:24, 1028.68 examples/s]23377 examples [00:24, 1033.01 examples/s]23481 examples [00:24, 1024.11 examples/s]23585 examples [00:24, 1025.98 examples/s]23688 examples [00:24, 1026.00 examples/s]23791 examples [00:24, 1025.13 examples/s]23894 examples [00:24, 1019.51 examples/s]23996 examples [00:24, 1015.65 examples/s]24098 examples [00:24, 1014.10 examples/s]24201 examples [00:24, 1016.80 examples/s]24305 examples [00:25, 1021.90 examples/s]24408 examples [00:25, 1023.52 examples/s]24511 examples [00:25, 1015.13 examples/s]24613 examples [00:25, 1014.66 examples/s]24715 examples [00:25, 1013.18 examples/s]24817 examples [00:25, 1010.39 examples/s]24919 examples [00:25, 973.94 examples/s] 25017 examples [00:25, 928.56 examples/s]25122 examples [00:25, 960.27 examples/s]25226 examples [00:26, 981.46 examples/s]25331 examples [00:26, 999.13 examples/s]25435 examples [00:26, 1009.05 examples/s]25540 examples [00:26, 1018.54 examples/s]25643 examples [00:26, 1001.41 examples/s]25744 examples [00:26, 994.56 examples/s] 25844 examples [00:26, 993.82 examples/s]25945 examples [00:26, 997.88 examples/s]26047 examples [00:26, 1004.14 examples/s]26148 examples [00:26, 994.27 examples/s] 26248 examples [00:27, 991.70 examples/s]26348 examples [00:27, 988.08 examples/s]26447 examples [00:27, 976.86 examples/s]26545 examples [00:27, 977.21 examples/s]26643 examples [00:27, 962.78 examples/s]26743 examples [00:27, 973.42 examples/s]26847 examples [00:27, 989.96 examples/s]26947 examples [00:27, 980.11 examples/s]27047 examples [00:27, 984.51 examples/s]27148 examples [00:27, 991.07 examples/s]27248 examples [00:28, 955.62 examples/s]27344 examples [00:28, 889.18 examples/s]27435 examples [00:28, 873.12 examples/s]27535 examples [00:28, 905.48 examples/s]27634 examples [00:28, 928.32 examples/s]27729 examples [00:28, 931.79 examples/s]27823 examples [00:28, 924.53 examples/s]27916 examples [00:28, 892.41 examples/s]28019 examples [00:28, 928.60 examples/s]28121 examples [00:29, 951.84 examples/s]28218 examples [00:29, 956.03 examples/s]28315 examples [00:29, 948.57 examples/s]28416 examples [00:29, 965.84 examples/s]28521 examples [00:29, 988.25 examples/s]28623 examples [00:29, 995.61 examples/s]28727 examples [00:29, 1006.17 examples/s]28828 examples [00:29, 1000.48 examples/s]28929 examples [00:29, 975.64 examples/s] 29032 examples [00:29, 990.84 examples/s]29134 examples [00:30, 997.96 examples/s]29239 examples [00:30, 1012.26 examples/s]29341 examples [00:30, 1006.85 examples/s]29442 examples [00:30, 984.82 examples/s] 29541 examples [00:30, 942.99 examples/s]29637 examples [00:30, 947.45 examples/s]29735 examples [00:30, 956.11 examples/s]29831 examples [00:30, 948.90 examples/s]29931 examples [00:30, 962.33 examples/s]30028 examples [00:30, 922.67 examples/s]30131 examples [00:31, 952.07 examples/s]30233 examples [00:31, 970.55 examples/s]30331 examples [00:31, 967.73 examples/s]30434 examples [00:31, 984.96 examples/s]30537 examples [00:31, 996.88 examples/s]30641 examples [00:31, 1006.53 examples/s]30742 examples [00:31, 999.09 examples/s] 30843 examples [00:31, 997.17 examples/s]30943 examples [00:31, 960.68 examples/s]31040 examples [00:32, 954.21 examples/s]31136 examples [00:32, 949.31 examples/s]31232 examples [00:32, 935.78 examples/s]31331 examples [00:32, 949.50 examples/s]31433 examples [00:32, 967.50 examples/s]31535 examples [00:32, 980.35 examples/s]31636 examples [00:32, 986.62 examples/s]31737 examples [00:32, 991.14 examples/s]31837 examples [00:32, 967.41 examples/s]31939 examples [00:32, 980.42 examples/s]32040 examples [00:33, 987.59 examples/s]32141 examples [00:33, 993.04 examples/s]32241 examples [00:33, 978.03 examples/s]32339 examples [00:33, 972.93 examples/s]32437 examples [00:33, 973.42 examples/s]32535 examples [00:33, 965.92 examples/s]32633 examples [00:33, 968.62 examples/s]32735 examples [00:33, 982.51 examples/s]32838 examples [00:33, 995.93 examples/s]32943 examples [00:33, 1010.95 examples/s]33045 examples [00:34, 1012.15 examples/s]33147 examples [00:34, 1013.62 examples/s]33251 examples [00:34, 1021.16 examples/s]33354 examples [00:34, 1011.71 examples/s]33458 examples [00:34, 1019.30 examples/s]33562 examples [00:34, 1025.06 examples/s]33665 examples [00:34, 1002.46 examples/s]33766 examples [00:34, 951.04 examples/s] 33862 examples [00:34, 938.50 examples/s]33957 examples [00:35, 923.29 examples/s]34062 examples [00:35, 957.43 examples/s]34159 examples [00:35, 960.64 examples/s]34263 examples [00:35, 980.98 examples/s]34362 examples [00:35, 976.43 examples/s]34465 examples [00:35, 989.44 examples/s]34568 examples [00:35, 1000.09 examples/s]34670 examples [00:35, 1003.16 examples/s]34773 examples [00:35, 1009.34 examples/s]34875 examples [00:35, 1002.22 examples/s]34979 examples [00:36, 1012.45 examples/s]35084 examples [00:36, 1021.88 examples/s]35188 examples [00:36, 1025.33 examples/s]35293 examples [00:36, 1030.88 examples/s]35397 examples [00:36, 1017.38 examples/s]35499 examples [00:36, 1008.98 examples/s]35602 examples [00:36, 1013.59 examples/s]35704 examples [00:36, 1014.90 examples/s]35807 examples [00:36, 1017.74 examples/s]35909 examples [00:36, 1015.55 examples/s]36011 examples [00:37, 1016.66 examples/s]36114 examples [00:37, 1018.98 examples/s]36216 examples [00:37, 1017.36 examples/s]36318 examples [00:37, 1017.49 examples/s]36420 examples [00:37, 997.37 examples/s] 36520 examples [00:37, 965.93 examples/s]36623 examples [00:37, 981.93 examples/s]36726 examples [00:37, 993.60 examples/s]36826 examples [00:37, 985.76 examples/s]36925 examples [00:37, 974.61 examples/s]37023 examples [00:38, 974.38 examples/s]37121 examples [00:38, 975.31 examples/s]37220 examples [00:38, 978.38 examples/s]37325 examples [00:38, 996.48 examples/s]37425 examples [00:38, 980.44 examples/s]37524 examples [00:38, 981.59 examples/s]37626 examples [00:38, 991.60 examples/s]37727 examples [00:38, 995.87 examples/s]37831 examples [00:38, 1006.30 examples/s]37932 examples [00:38, 1004.85 examples/s]38034 examples [00:39, 1007.82 examples/s]38136 examples [00:39, 1010.55 examples/s]38238 examples [00:39, 1012.96 examples/s]38343 examples [00:39, 1022.29 examples/s]38446 examples [00:39, 1019.03 examples/s]38550 examples [00:39, 1024.82 examples/s]38653 examples [00:39, 1000.54 examples/s]38754 examples [00:39, 956.73 examples/s] 38858 examples [00:39, 977.49 examples/s]38959 examples [00:39, 985.40 examples/s]39064 examples [00:40, 1001.73 examples/s]39169 examples [00:40, 1014.15 examples/s]39272 examples [00:40, 1018.22 examples/s]39375 examples [00:40, 1004.58 examples/s]39476 examples [00:40, 994.08 examples/s] 39580 examples [00:40, 1006.72 examples/s]39681 examples [00:40, 1001.97 examples/s]39782 examples [00:40, 988.12 examples/s] 39885 examples [00:40, 998.93 examples/s]39986 examples [00:41, 957.58 examples/s]40083 examples [00:41, 886.07 examples/s]40180 examples [00:41, 909.19 examples/s]40277 examples [00:41, 926.50 examples/s]40377 examples [00:41, 946.65 examples/s]40478 examples [00:41, 962.93 examples/s]40580 examples [00:41, 978.98 examples/s]40682 examples [00:41, 988.44 examples/s]40785 examples [00:41, 998.41 examples/s]40887 examples [00:41, 1002.20 examples/s]40988 examples [00:42, 999.07 examples/s] 41089 examples [00:42, 957.27 examples/s]41188 examples [00:42, 966.73 examples/s]41289 examples [00:42, 976.70 examples/s]41389 examples [00:42, 983.46 examples/s]41489 examples [00:42, 985.89 examples/s]41590 examples [00:42, 990.32 examples/s]41692 examples [00:42, 997.98 examples/s]41794 examples [00:42, 1003.67 examples/s]41895 examples [00:42, 985.20 examples/s] 41994 examples [00:43, 969.80 examples/s]42092 examples [00:43, 934.75 examples/s]42189 examples [00:43, 944.33 examples/s]42289 examples [00:43, 958.90 examples/s]42386 examples [00:43, 960.43 examples/s]42483 examples [00:43, 945.38 examples/s]42582 examples [00:43, 955.73 examples/s]42683 examples [00:43, 971.06 examples/s]42784 examples [00:43, 976.34 examples/s]42882 examples [00:44, 972.84 examples/s]42980 examples [00:44, 949.55 examples/s]43076 examples [00:44, 933.93 examples/s]43178 examples [00:44, 955.99 examples/s]43280 examples [00:44, 973.34 examples/s]43378 examples [00:44, 960.94 examples/s]43476 examples [00:44, 964.60 examples/s]43573 examples [00:44, 956.52 examples/s]43675 examples [00:44, 974.28 examples/s]43777 examples [00:44, 986.69 examples/s]43876 examples [00:45, 985.49 examples/s]43979 examples [00:45, 996.64 examples/s]44081 examples [00:45, 1002.65 examples/s]44183 examples [00:45, 1006.87 examples/s]44286 examples [00:45, 1011.30 examples/s]44388 examples [00:45, 999.12 examples/s] 44488 examples [00:45, 993.33 examples/s]44590 examples [00:45, 1000.99 examples/s]44692 examples [00:45, 1004.36 examples/s]44793 examples [00:45, 1001.78 examples/s]44894 examples [00:46, 988.64 examples/s] 44996 examples [00:46, 997.34 examples/s]45096 examples [00:46, 988.05 examples/s]45197 examples [00:46, 992.02 examples/s]45299 examples [00:46, 999.07 examples/s]45399 examples [00:46, 993.53 examples/s]45499 examples [00:46, 994.20 examples/s]45602 examples [00:46, 1002.31 examples/s]45703 examples [00:46, 971.98 examples/s] 45806 examples [00:46, 987.05 examples/s]45905 examples [00:47, 979.19 examples/s]46004 examples [00:47, 968.34 examples/s]46101 examples [00:47, 924.50 examples/s]46202 examples [00:47, 948.55 examples/s]46302 examples [00:47, 960.57 examples/s]46400 examples [00:47, 965.33 examples/s]46499 examples [00:47, 971.45 examples/s]46597 examples [00:47, 973.72 examples/s]46695 examples [00:47, 950.49 examples/s]46791 examples [00:48, 941.46 examples/s]46887 examples [00:48, 945.05 examples/s]46990 examples [00:48, 967.44 examples/s]47094 examples [00:48, 986.05 examples/s]47199 examples [00:48, 1003.39 examples/s]47300 examples [00:48, 1002.32 examples/s]47401 examples [00:48, 966.51 examples/s] 47499 examples [00:48, 941.59 examples/s]47594 examples [00:48, 917.45 examples/s]47692 examples [00:48, 933.57 examples/s]47786 examples [00:49, 933.78 examples/s]47886 examples [00:49, 950.61 examples/s]47982 examples [00:49, 937.65 examples/s]48076 examples [00:49, 934.51 examples/s]48175 examples [00:49, 948.17 examples/s]48277 examples [00:49, 968.44 examples/s]48375 examples [00:49, 939.21 examples/s]48470 examples [00:49, 932.57 examples/s]48572 examples [00:49, 956.72 examples/s]48673 examples [00:49, 971.95 examples/s]48774 examples [00:50, 981.98 examples/s]48873 examples [00:50, 899.43 examples/s]48973 examples [00:50, 926.01 examples/s]49069 examples [00:50, 934.52 examples/s]49170 examples [00:50, 955.69 examples/s]49271 examples [00:50, 970.91 examples/s]49369 examples [00:50, 970.31 examples/s]49473 examples [00:50, 988.16 examples/s]49577 examples [00:50, 1002.74 examples/s]49678 examples [00:51, 957.16 examples/s] 49775 examples [00:51, 958.86 examples/s]49876 examples [00:51, 971.19 examples/s]49974 examples [00:51, 956.33 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5728/50000 [00:00<00:00, 57277.32 examples/s] 33%|███▎      | 16343/50000 [00:00<00:00, 66455.38 examples/s] 55%|█████▍    | 27300/50000 [00:00<00:00, 75349.47 examples/s] 77%|███████▋  | 38565/50000 [00:00<00:00, 83659.58 examples/s]100%|█████████▉| 49835/50000 [00:00<00:00, 90666.38 examples/s]                                                               0 examples [00:00, ? examples/s]83 examples [00:00, 824.24 examples/s]184 examples [00:00, 872.05 examples/s]282 examples [00:00, 899.71 examples/s]382 examples [00:00, 927.24 examples/s]483 examples [00:00, 950.38 examples/s]583 examples [00:00, 963.66 examples/s]685 examples [00:00, 979.10 examples/s]779 examples [00:00, 965.21 examples/s]879 examples [00:00, 974.72 examples/s]978 examples [00:01, 978.55 examples/s]1080 examples [00:01, 990.32 examples/s]1181 examples [00:01, 995.53 examples/s]1280 examples [00:01, 966.73 examples/s]1377 examples [00:01, 961.07 examples/s]1479 examples [00:01, 975.81 examples/s]1584 examples [00:01, 996.46 examples/s]1688 examples [00:01, 1008.86 examples/s]1789 examples [00:01, 981.13 examples/s] 1888 examples [00:01, 977.23 examples/s]1987 examples [00:02, 980.02 examples/s]2086 examples [00:02, 965.05 examples/s]2188 examples [00:02, 978.64 examples/s]2289 examples [00:02, 986.65 examples/s]2394 examples [00:02, 1002.48 examples/s]2500 examples [00:02, 1018.03 examples/s]2605 examples [00:02, 1026.97 examples/s]2709 examples [00:02, 1030.42 examples/s]2813 examples [00:02, 996.86 examples/s] 2913 examples [00:02, 993.28 examples/s]3016 examples [00:03, 1003.74 examples/s]3117 examples [00:03, 1000.61 examples/s]3219 examples [00:03, 1005.45 examples/s]3320 examples [00:03, 988.94 examples/s] 3423 examples [00:03, 1000.75 examples/s]3525 examples [00:03, 1003.66 examples/s]3628 examples [00:03, 1009.90 examples/s]3730 examples [00:03, 1008.13 examples/s]3831 examples [00:03, 988.12 examples/s] 3932 examples [00:03, 993.44 examples/s]4034 examples [00:04, 998.96 examples/s]4134 examples [00:04, 978.01 examples/s]4235 examples [00:04, 986.55 examples/s]4334 examples [00:04, 938.53 examples/s]4431 examples [00:04, 946.71 examples/s]4534 examples [00:04, 969.25 examples/s]4632 examples [00:04, 970.96 examples/s]4730 examples [00:04, 957.29 examples/s]4826 examples [00:04, 954.08 examples/s]4922 examples [00:05, 942.02 examples/s]5021 examples [00:05, 953.89 examples/s]5123 examples [00:05, 972.08 examples/s]5226 examples [00:05, 986.83 examples/s]5327 examples [00:05, 992.57 examples/s]5431 examples [00:05, 1004.10 examples/s]5534 examples [00:05, 1010.26 examples/s]5636 examples [00:05, 1012.86 examples/s]5738 examples [00:05, 1014.98 examples/s]5840 examples [00:05, 1010.72 examples/s]5943 examples [00:06, 1015.73 examples/s]6045 examples [00:06, 1014.73 examples/s]6147 examples [00:06, 1004.81 examples/s]6248 examples [00:06, 951.04 examples/s] 6346 examples [00:06, 957.27 examples/s]6443 examples [00:06, 956.19 examples/s]6545 examples [00:06, 972.10 examples/s]6645 examples [00:06, 979.09 examples/s]6744 examples [00:06, 957.53 examples/s]6847 examples [00:06, 976.78 examples/s]6946 examples [00:07, 978.49 examples/s]7048 examples [00:07, 988.43 examples/s]7148 examples [00:07, 965.32 examples/s]7245 examples [00:07, 873.67 examples/s]7335 examples [00:07, 875.26 examples/s]7429 examples [00:07, 891.70 examples/s]7528 examples [00:07, 918.70 examples/s]7621 examples [00:07, 895.32 examples/s]7712 examples [00:07, 879.22 examples/s]7809 examples [00:08, 902.43 examples/s]7905 examples [00:08, 917.97 examples/s]8002 examples [00:08, 932.30 examples/s]8096 examples [00:08, 923.84 examples/s]8196 examples [00:08, 944.23 examples/s]8299 examples [00:08, 965.98 examples/s]8400 examples [00:08, 976.44 examples/s]8501 examples [00:08, 984.13 examples/s]8602 examples [00:08, 990.56 examples/s]8702 examples [00:08, 979.22 examples/s]8801 examples [00:09, 946.48 examples/s]8904 examples [00:09, 969.40 examples/s]9007 examples [00:09, 986.70 examples/s]9111 examples [00:09, 1001.34 examples/s]9212 examples [00:09, 998.64 examples/s] 9314 examples [00:09, 1004.41 examples/s]9415 examples [00:09, 1001.32 examples/s]9516 examples [00:09, 996.98 examples/s] 9616 examples [00:09, 997.13 examples/s]9716 examples [00:09, 997.05 examples/s]9816 examples [00:10, 966.02 examples/s]9913 examples [00:10, 956.25 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP3FOKD/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP3FOKD/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa68021a840>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa68021a840>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa68021a840> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.47548943996429444 	 Accuracy: 56
Train Epoch: 1 	 Loss: 9.043111419677734 	 Accuracy: 240
model saves at 240 accuracy
Train Epoch: 2 	 Loss: 0.4423569655418396 	 Accuracy: 50
Train Epoch: 2 	 Loss: 17.754297256469727 	 Accuracy: 260
model saves at 260 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa63d1b1d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa63d1b1d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7fa63d1b1d08> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6356a5bf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6356a5bf8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6356a5bf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([7, 8, 1, 1, 1, 7, 1, 8, 1, 2]), array([4, 8, 1, 0, 1, 2, 2, 8, 2, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa69ab76f28> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa637dce630> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa6358910d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa6358910d0>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa6358910d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:37, 271809.00it/s]  1%|          | 163840/26421880 [00:00<01:15, 345571.72it/s]  1%|▏         | 376832/26421880 [00:00<00:58, 447067.24it/s]  5%|▌         | 1409024/26421880 [00:00<00:40, 624506.06it/s] 12%|█▏        | 3104768/26421880 [00:00<00:26, 871550.09it/s] 25%|██▍       | 6594560/26421880 [00:01<00:16, 1231106.20it/s] 44%|████▍     | 11657216/26421880 [00:01<00:08, 1739349.27it/s] 58%|█████▊    | 15450112/26421880 [00:01<00:04, 2413689.77it/s] 72%|███████▏  | 19144704/26421880 [00:01<00:02, 3349113.88it/s] 89%|████████▉ | 23609344/26421880 [00:01<00:00, 4635329.89it/s]26427392it [00:01, 16334170.78it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 108784.38it/s]           
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 272350.85it/s]  4%|▎         | 155648/4422102 [00:00<00:12, 346096.07it/s] 10%|▉         | 425984/4422102 [00:00<00:08, 455021.81it/s] 30%|██▉       | 1318912/4422102 [00:00<00:04, 634331.51it/s] 79%|███████▊  | 3473408/4422102 [00:00<00:01, 888458.97it/s]4423680it [00:00, 4425125.39it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 30131.24it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.003895607848962148 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018603525280952455 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.002850460777680079 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.016028272211551665 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 3 	 Loss: 0.0028886542121569314 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.021378687381744386 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.002058789094289144 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.012035605490207673 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.0019705144266287484 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.011101609230041503 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa635887730>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa635887730>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa635887730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([1, 8, 3, 4, 3, 8, 1, 9, 6, 3]), array([1, 4, 0, 2, 3, 8, 1, 9, 6, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7fa69ab76b70> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa637d39048>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa637d39048>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa637d39048> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110174.33it/s]  0%|          | 40960/18165135 [00:00<02:28, 122390.15it/s]  1%|          | 98304/18165135 [00:00<01:57, 153865.04it/s]  1%|          | 212992/18165135 [00:01<01:28, 202020.77it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272141.78it/s]  5%|▍         | 868352/18165135 [00:01<00:46, 374068.59it/s] 10%|▉         | 1736704/18165135 [00:01<00:31, 519780.04it/s] 19%|█▉        | 3489792/18165135 [00:01<00:20, 728709.49it/s] 30%|██▉       | 5447680/18165135 [00:01<00:12, 1005099.62it/s] 38%|███▊      | 6914048/18165135 [00:02<00:08, 1394302.52it/s] 46%|████▌     | 8364032/18165135 [00:02<00:05, 1840267.45it/s] 51%|█████     | 9199616/18165135 [00:02<00:04, 2082376.63it/s] 62%|██████▏   | 11296768/18165135 [00:02<00:02, 2784801.79it/s] 67%|██████▋   | 12165120/18165135 [00:03<00:02, 2445824.07it/s] 79%|███████▉  | 14327808/18165135 [00:03<00:01, 3286494.38it/s] 84%|████████▍ | 15294464/18165135 [00:03<00:00, 3719088.79it/s] 89%|████████▉ | 16130048/18165135 [00:03<00:00, 4317371.70it/s] 93%|█████████▎| 16924672/18165135 [00:03<00:00, 4552475.25it/s] 97%|█████████▋| 17637376/18165135 [00:03<00:00, 4640452.07it/s]18169856it [00:03, 4675180.06it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110107.25it/s]32768it [00:00, 49789.61it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110131.78it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122312.06it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153781.65it/s]  7%|▋         | 212992/3041136 [00:01<00:13, 202361.09it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 272280.21it/s] 29%|██▊       | 868352/3041136 [00:01<00:05, 374257.79it/s] 53%|█████▎    | 1605632/3041136 [00:01<00:02, 517874.90it/s] 78%|███████▊  | 2383872/3041136 [00:01<00:00, 709650.40it/s]3047424it [00:01, 1813890.79it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17592.26it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004304149508476257 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02283863389492035 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.003220496733983358 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.025674896955490113 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.002598364273707072 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.03197213339805603 	 Accuracy: 2
Train Epoch: 4 	 Loss: 0.002897824247678121 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.032360450744628906 	 Accuracy: 2
Train Epoch: 5 	 Loss: 0.00291962331533432 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.016776948094367982 	 Accuracy: 4
model saves at 4 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa637d397b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa637d397b8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7fa637d397b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([6, 2, 7, 5, 2, 2, 9, 2, 2, 2]), array([6, 4, 0, 1, 1, 7, 9, 0, 4, 2]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

