
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

  <mlmodels.model_tch.torchhub.Model object at 0x7f202855b390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2027554488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2027554488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f2027554488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3842048/9912422 [00:00<00:00, 38035742.88it/s]9920512it [00:00, 31124348.23it/s]                             
0it [00:00, ?it/s]32768it [00:00, 576520.19it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7475509.99it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 69480.00it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0033814523220062256 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.020572691917419433 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.002238053798675537 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.013675601661205291 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 3 	 Loss: 0.0024172293643156687 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.014128903925418854 	 Accuracy: 5
Train Epoch: 4 	 Loss: 0.002294282426436742 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.020909676432609558 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.0013696183711290359 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.007956363290548324 	 Accuracy: 7
model saves at 7 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2027540f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2027540f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f2027540f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([3, 3, 0, 5, 4, 0, 3, 1, 8, 1]), array([3, 3, 0, 5, 6, 0, 2, 1, 6, 1]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f20293cb550> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f20248aabf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f20248aabf8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f20248aabf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0019624786774317424 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.023775751113891602 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0022093315521876018 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.04941189956665039 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f20248aa488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f20248aa488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f20248aa488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([8, 7, 3, 4, 2, 8, 6, 2, 4, 2]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s] 17%|█▋        | 7.59M/44.7M [00:00<00:00, 79.5MB/s] 43%|████▎     | 19.3M/44.7M [00:00<00:00, 88.9MB/s] 68%|██████▊   | 30.2M/44.7M [00:00<00:00, 95.2MB/s] 94%|█████████▍| 41.9M/44.7M [00:00<00:00, 102MB/s] 100%|██████████| 44.7M/44.7M [00:00<00:00, 111MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f202488d2e8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f202752e268>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f202752e268>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f202752e268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:05,  2.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:04,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:04,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:03,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:45,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:45,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:44,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:44,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:44,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:43,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:43,  3.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:30,  4.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:30,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:30,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:30,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:30,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:30,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:29,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:29,  4.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:21,  6.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:21,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:21,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:20,  6.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:14,  9.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:14,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:14,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:14,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:14,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 12.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 12.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 16.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 32.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 39.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 59.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 59.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 66.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 68.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 70.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 69.99 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.70s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.99 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.91 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ url]
0 examples [00:00, ? examples/s]2020-05-23 06:09:40.948401: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 06:09:40.961258: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-23 06:09:40.961477: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557186928700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 06:09:40.961493: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
17 examples [00:00, 169.58 examples/s]104 examples [00:00, 223.39 examples/s]188 examples [00:00, 286.05 examples/s]275 examples [00:00, 358.03 examples/s]362 examples [00:00, 434.35 examples/s]449 examples [00:00, 510.12 examples/s]533 examples [00:00, 577.10 examples/s]619 examples [00:00, 638.98 examples/s]708 examples [00:00, 696.82 examples/s]790 examples [00:01, 722.03 examples/s]881 examples [00:01, 769.64 examples/s]973 examples [00:01, 808.43 examples/s]1063 examples [00:01, 833.86 examples/s]1153 examples [00:01, 851.36 examples/s]1241 examples [00:01, 843.80 examples/s]1329 examples [00:01, 852.03 examples/s]1416 examples [00:01, 851.80 examples/s]1503 examples [00:01, 844.38 examples/s]1594 examples [00:01, 861.32 examples/s]1683 examples [00:02, 868.36 examples/s]1774 examples [00:02, 879.90 examples/s]1865 examples [00:02, 887.15 examples/s]1954 examples [00:02, 885.97 examples/s]2043 examples [00:02, 879.46 examples/s]2133 examples [00:02, 881.97 examples/s]2222 examples [00:02, 871.14 examples/s]2310 examples [00:02, 864.64 examples/s]2399 examples [00:02, 869.89 examples/s]2489 examples [00:02, 878.36 examples/s]2579 examples [00:03, 882.41 examples/s]2669 examples [00:03, 886.30 examples/s]2759 examples [00:03, 888.98 examples/s]2850 examples [00:03, 893.29 examples/s]2943 examples [00:03, 903.19 examples/s]3037 examples [00:03, 912.42 examples/s]3134 examples [00:03, 926.58 examples/s]3228 examples [00:03, 930.07 examples/s]3322 examples [00:03, 924.77 examples/s]3417 examples [00:03, 930.71 examples/s]3511 examples [00:04, 928.40 examples/s]3605 examples [00:04, 931.18 examples/s]3699 examples [00:04, 899.20 examples/s]3790 examples [00:04, 883.21 examples/s]3881 examples [00:04, 889.95 examples/s]3971 examples [00:04, 863.67 examples/s]4065 examples [00:04, 883.75 examples/s]4156 examples [00:04, 889.03 examples/s]4246 examples [00:04, 889.13 examples/s]4336 examples [00:04, 890.92 examples/s]4426 examples [00:05, 892.19 examples/s]4518 examples [00:05, 899.82 examples/s]4609 examples [00:05, 901.88 examples/s]4700 examples [00:05, 893.94 examples/s]4790 examples [00:05, 867.08 examples/s]4880 examples [00:05, 874.48 examples/s]4969 examples [00:05, 877.63 examples/s]5057 examples [00:05, 878.08 examples/s]5145 examples [00:05, 869.82 examples/s]5238 examples [00:06, 884.99 examples/s]5329 examples [00:06, 891.93 examples/s]5420 examples [00:06, 897.10 examples/s]5510 examples [00:06, 889.20 examples/s]5599 examples [00:06, 873.48 examples/s]5687 examples [00:06, 869.69 examples/s]5775 examples [00:06, 863.19 examples/s]5863 examples [00:06, 867.63 examples/s]5950 examples [00:06, 865.56 examples/s]6037 examples [00:06, 845.05 examples/s]6125 examples [00:07, 852.64 examples/s]6213 examples [00:07, 859.93 examples/s]6306 examples [00:07, 877.02 examples/s]6397 examples [00:07, 886.44 examples/s]6488 examples [00:07, 891.73 examples/s]6578 examples [00:07, 891.79 examples/s]6668 examples [00:07, 877.72 examples/s]6764 examples [00:07, 899.54 examples/s]6859 examples [00:07, 911.62 examples/s]6951 examples [00:07, 906.27 examples/s]7045 examples [00:08, 915.92 examples/s]7137 examples [00:08, 915.16 examples/s]7232 examples [00:08, 925.04 examples/s]7328 examples [00:08, 933.38 examples/s]7422 examples [00:08, 916.84 examples/s]7515 examples [00:08, 919.53 examples/s]7608 examples [00:08, 912.97 examples/s]7700 examples [00:08, 912.81 examples/s]7792 examples [00:08, 911.88 examples/s]7884 examples [00:08, 872.63 examples/s]7972 examples [00:09, 868.84 examples/s]8066 examples [00:09, 886.72 examples/s]8159 examples [00:09, 898.11 examples/s]8250 examples [00:09, 900.44 examples/s]8341 examples [00:09, 880.37 examples/s]8430 examples [00:09, 878.29 examples/s]8519 examples [00:09, 880.81 examples/s]8608 examples [00:09, 879.58 examples/s]8697 examples [00:09, 879.40 examples/s]8785 examples [00:10, 878.51 examples/s]8873 examples [00:10, 875.33 examples/s]8962 examples [00:10, 877.81 examples/s]9051 examples [00:10, 879.14 examples/s]9140 examples [00:10, 879.84 examples/s]9229 examples [00:10, 881.83 examples/s]9318 examples [00:10, 868.78 examples/s]9405 examples [00:10, 860.16 examples/s]9496 examples [00:10, 872.93 examples/s]9586 examples [00:10, 878.53 examples/s]9675 examples [00:11, 879.62 examples/s]9766 examples [00:11, 885.99 examples/s]9855 examples [00:11, 882.34 examples/s]9945 examples [00:11, 886.60 examples/s]10034 examples [00:11, 837.31 examples/s]10121 examples [00:11, 846.52 examples/s]10215 examples [00:11, 872.35 examples/s]10309 examples [00:11, 891.58 examples/s]10402 examples [00:11, 899.91 examples/s]10496 examples [00:11, 909.96 examples/s]10588 examples [00:12, 912.27 examples/s]10680 examples [00:12, 905.46 examples/s]10775 examples [00:12, 916.88 examples/s]10869 examples [00:12, 920.71 examples/s]10962 examples [00:12, 908.94 examples/s]11053 examples [00:12, 872.21 examples/s]11141 examples [00:12, 872.89 examples/s]11233 examples [00:12, 885.46 examples/s]11323 examples [00:12, 889.12 examples/s]11416 examples [00:12, 900.34 examples/s]11510 examples [00:13, 909.46 examples/s]11602 examples [00:13, 911.77 examples/s]11697 examples [00:13, 921.60 examples/s]11792 examples [00:13, 928.34 examples/s]11885 examples [00:13, 925.62 examples/s]11978 examples [00:13, 913.08 examples/s]12070 examples [00:13, 871.15 examples/s]12158 examples [00:13, 864.11 examples/s]12250 examples [00:13, 878.77 examples/s]12339 examples [00:14, 856.67 examples/s]12428 examples [00:14, 865.94 examples/s]12515 examples [00:14, 854.14 examples/s]12606 examples [00:14, 869.36 examples/s]12700 examples [00:14, 887.56 examples/s]12792 examples [00:14, 894.75 examples/s]12889 examples [00:14, 913.65 examples/s]12983 examples [00:14, 920.29 examples/s]13078 examples [00:14, 927.09 examples/s]13171 examples [00:14, 901.04 examples/s]13262 examples [00:15, 895.95 examples/s]13352 examples [00:15, 892.30 examples/s]13442 examples [00:15, 876.36 examples/s]13535 examples [00:15, 889.08 examples/s]13627 examples [00:15, 897.85 examples/s]13717 examples [00:15, 856.87 examples/s]13805 examples [00:15, 862.31 examples/s]13892 examples [00:15, 826.52 examples/s]13983 examples [00:15, 849.54 examples/s]14080 examples [00:15, 880.30 examples/s]14171 examples [00:16, 888.74 examples/s]14265 examples [00:16, 902.63 examples/s]14360 examples [00:16, 914.69 examples/s]14456 examples [00:16, 926.06 examples/s]14552 examples [00:16, 934.93 examples/s]14646 examples [00:16, 934.90 examples/s]14743 examples [00:16, 944.03 examples/s]14838 examples [00:16, 912.17 examples/s]14930 examples [00:16, 911.91 examples/s]15024 examples [00:16, 917.84 examples/s]15116 examples [00:17, 908.28 examples/s]15210 examples [00:17, 915.23 examples/s]15303 examples [00:17, 918.72 examples/s]15396 examples [00:17, 921.37 examples/s]15489 examples [00:17, 923.89 examples/s]15582 examples [00:17, 919.66 examples/s]15677 examples [00:17, 926.37 examples/s]15771 examples [00:17, 928.73 examples/s]15864 examples [00:17, 918.90 examples/s]15956 examples [00:18, 909.82 examples/s]16048 examples [00:18, 903.21 examples/s]16139 examples [00:18, 887.21 examples/s]16228 examples [00:18, 884.31 examples/s]16317 examples [00:18, 872.42 examples/s]16405 examples [00:18, 858.01 examples/s]16491 examples [00:18, 799.03 examples/s]16586 examples [00:18, 838.25 examples/s]16681 examples [00:18, 865.98 examples/s]16776 examples [00:18, 887.12 examples/s]16871 examples [00:19, 902.33 examples/s]16962 examples [00:19, 882.82 examples/s]17051 examples [00:19, 876.18 examples/s]17141 examples [00:19, 881.39 examples/s]17230 examples [00:19, 879.28 examples/s]17320 examples [00:19, 882.42 examples/s]17409 examples [00:19, 882.12 examples/s]17498 examples [00:19, 870.46 examples/s]17592 examples [00:19, 887.76 examples/s]17685 examples [00:19, 899.34 examples/s]17776 examples [00:20, 892.94 examples/s]17866 examples [00:20, 883.93 examples/s]17955 examples [00:20, 883.01 examples/s]18046 examples [00:20, 890.65 examples/s]18136 examples [00:20, 892.46 examples/s]18226 examples [00:20, 875.40 examples/s]18315 examples [00:20, 877.89 examples/s]18408 examples [00:20, 891.93 examples/s]18498 examples [00:20, 888.86 examples/s]18588 examples [00:21, 891.25 examples/s]18678 examples [00:21, 885.53 examples/s]18767 examples [00:21, 873.44 examples/s]18855 examples [00:21, 875.12 examples/s]18943 examples [00:21, 869.57 examples/s]19031 examples [00:21, 869.63 examples/s]19122 examples [00:21, 880.49 examples/s]19216 examples [00:21, 896.51 examples/s]19311 examples [00:21, 910.74 examples/s]19406 examples [00:21, 920.05 examples/s]19501 examples [00:22, 927.64 examples/s]19595 examples [00:22, 928.93 examples/s]19688 examples [00:22, 926.60 examples/s]19781 examples [00:22, 927.39 examples/s]19876 examples [00:22, 933.92 examples/s]19970 examples [00:22, 933.26 examples/s]20064 examples [00:22, 862.93 examples/s]20159 examples [00:22, 887.29 examples/s]20249 examples [00:22, 879.67 examples/s]20345 examples [00:22, 900.79 examples/s]20439 examples [00:23, 912.18 examples/s]20532 examples [00:23, 915.95 examples/s]20626 examples [00:23, 921.20 examples/s]20722 examples [00:23, 929.90 examples/s]20816 examples [00:23, 928.34 examples/s]20910 examples [00:23, 928.80 examples/s]21005 examples [00:23, 933.72 examples/s]21101 examples [00:23, 939.69 examples/s]21196 examples [00:23, 938.15 examples/s]21290 examples [00:23, 925.83 examples/s]21383 examples [00:24, 910.20 examples/s]21475 examples [00:24, 901.80 examples/s]21566 examples [00:24, 893.02 examples/s]21656 examples [00:24, 888.32 examples/s]21745 examples [00:24, 875.61 examples/s]21833 examples [00:24, 871.37 examples/s]21921 examples [00:24, 868.06 examples/s]22009 examples [00:24, 871.19 examples/s]22097 examples [00:24, 873.37 examples/s]22185 examples [00:24, 873.95 examples/s]22273 examples [00:25, 874.72 examples/s]22361 examples [00:25, 876.06 examples/s]22449 examples [00:25, 868.08 examples/s]22539 examples [00:25, 874.33 examples/s]22627 examples [00:25, 868.25 examples/s]22714 examples [00:25, 857.19 examples/s]22806 examples [00:25, 873.56 examples/s]22895 examples [00:25, 878.22 examples/s]22983 examples [00:25, 862.98 examples/s]23077 examples [00:26, 882.11 examples/s]23166 examples [00:26, 879.79 examples/s]23255 examples [00:26, 879.21 examples/s]23344 examples [00:26, 881.82 examples/s]23433 examples [00:26, 882.34 examples/s]23522 examples [00:26, 873.80 examples/s]23610 examples [00:26, 875.40 examples/s]23698 examples [00:26, 876.66 examples/s]23789 examples [00:26, 880.73 examples/s]23882 examples [00:26, 894.61 examples/s]23972 examples [00:27, 895.82 examples/s]24063 examples [00:27, 897.66 examples/s]24153 examples [00:27, 879.19 examples/s]24246 examples [00:27, 892.08 examples/s]24339 examples [00:27, 901.44 examples/s]24430 examples [00:27, 900.50 examples/s]24521 examples [00:27, 896.65 examples/s]24615 examples [00:27, 906.42 examples/s]24707 examples [00:27, 910.19 examples/s]24799 examples [00:27, 898.87 examples/s]24891 examples [00:28, 904.80 examples/s]24982 examples [00:28, 903.88 examples/s]25073 examples [00:28, 895.85 examples/s]25166 examples [00:28, 903.26 examples/s]25257 examples [00:28, 902.44 examples/s]25351 examples [00:28, 910.80 examples/s]25444 examples [00:28, 915.75 examples/s]25536 examples [00:28, 907.83 examples/s]25627 examples [00:28, 896.52 examples/s]25717 examples [00:28, 875.65 examples/s]25814 examples [00:29, 899.82 examples/s]25908 examples [00:29, 909.26 examples/s]26000 examples [00:29, 904.49 examples/s]26091 examples [00:29, 904.57 examples/s]26182 examples [00:29, 891.46 examples/s]26272 examples [00:29, 867.25 examples/s]26365 examples [00:29, 883.41 examples/s]26454 examples [00:29, 862.23 examples/s]26543 examples [00:29, 869.15 examples/s]26635 examples [00:30, 881.28 examples/s]26729 examples [00:30, 895.66 examples/s]26822 examples [00:30, 903.59 examples/s]26913 examples [00:30, 892.88 examples/s]27003 examples [00:30, 866.34 examples/s]27091 examples [00:30, 867.53 examples/s]27181 examples [00:30, 875.98 examples/s]27273 examples [00:30, 887.34 examples/s]27362 examples [00:30, 878.73 examples/s]27452 examples [00:30, 884.91 examples/s]27541 examples [00:31, 884.32 examples/s]27633 examples [00:31, 892.32 examples/s]27724 examples [00:31, 895.36 examples/s]27814 examples [00:31, 882.38 examples/s]27903 examples [00:31, 882.22 examples/s]27992 examples [00:31, 880.95 examples/s]28081 examples [00:31, 881.36 examples/s]28170 examples [00:31, 881.80 examples/s]28259 examples [00:31, 879.15 examples/s]28347 examples [00:31, 875.11 examples/s]28435 examples [00:32, 851.59 examples/s]28524 examples [00:32, 860.12 examples/s]28613 examples [00:32, 866.43 examples/s]28700 examples [00:32, 866.06 examples/s]28791 examples [00:32, 877.08 examples/s]28881 examples [00:32, 883.74 examples/s]28971 examples [00:32, 886.72 examples/s]29060 examples [00:32, 859.05 examples/s]29147 examples [00:32, 848.87 examples/s]29235 examples [00:32, 857.31 examples/s]29321 examples [00:33, 853.14 examples/s]29413 examples [00:33, 872.14 examples/s]29505 examples [00:33, 885.02 examples/s]29594 examples [00:33, 886.36 examples/s]29688 examples [00:33, 901.23 examples/s]29782 examples [00:33, 910.27 examples/s]29876 examples [00:33, 916.37 examples/s]29969 examples [00:33, 917.23 examples/s]30061 examples [00:33, 822.16 examples/s]30154 examples [00:34, 849.39 examples/s]30245 examples [00:34, 864.43 examples/s]30339 examples [00:34, 884.33 examples/s]30434 examples [00:34, 900.79 examples/s]30525 examples [00:34, 894.49 examples/s]30616 examples [00:34, 897.96 examples/s]30709 examples [00:34, 902.11 examples/s]30800 examples [00:34, 840.81 examples/s]30892 examples [00:34, 861.40 examples/s]30984 examples [00:34, 877.33 examples/s]31073 examples [00:35, 862.39 examples/s]31162 examples [00:35, 869.72 examples/s]31251 examples [00:35, 873.55 examples/s]31339 examples [00:35, 871.62 examples/s]31427 examples [00:35, 867.84 examples/s]31517 examples [00:35, 876.65 examples/s]31605 examples [00:35, 867.71 examples/s]31692 examples [00:35, 848.87 examples/s]31778 examples [00:35, 851.81 examples/s]31867 examples [00:35, 862.25 examples/s]31958 examples [00:36, 874.28 examples/s]32046 examples [00:36, 867.86 examples/s]32135 examples [00:36, 873.13 examples/s]32226 examples [00:36, 881.37 examples/s]32315 examples [00:36, 868.69 examples/s]32403 examples [00:36, 870.56 examples/s]32492 examples [00:36, 875.15 examples/s]32585 examples [00:36, 890.31 examples/s]32676 examples [00:36, 894.86 examples/s]32766 examples [00:36, 896.31 examples/s]32856 examples [00:37, 890.08 examples/s]32946 examples [00:37, 890.02 examples/s]33037 examples [00:37, 894.95 examples/s]33127 examples [00:37, 889.05 examples/s]33216 examples [00:37, 879.08 examples/s]33306 examples [00:37, 884.91 examples/s]33395 examples [00:37, 885.50 examples/s]33486 examples [00:37, 891.82 examples/s]33576 examples [00:37, 892.31 examples/s]33666 examples [00:38, 889.83 examples/s]33756 examples [00:38, 880.84 examples/s]33847 examples [00:38, 887.20 examples/s]33942 examples [00:38, 902.35 examples/s]34033 examples [00:38, 893.80 examples/s]34123 examples [00:38, 888.02 examples/s]34213 examples [00:38, 889.88 examples/s]34303 examples [00:38, 880.82 examples/s]34392 examples [00:38, 830.52 examples/s]34483 examples [00:38, 851.84 examples/s]34576 examples [00:39, 872.23 examples/s]34668 examples [00:39, 886.02 examples/s]34760 examples [00:39, 893.27 examples/s]34852 examples [00:39, 899.75 examples/s]34943 examples [00:39, 900.18 examples/s]35034 examples [00:39, 893.00 examples/s]35127 examples [00:39, 902.25 examples/s]35218 examples [00:39, 878.84 examples/s]35308 examples [00:39, 884.48 examples/s]35397 examples [00:39, 877.95 examples/s]35485 examples [00:40, 871.53 examples/s]35576 examples [00:40, 882.17 examples/s]35665 examples [00:40, 881.02 examples/s]35755 examples [00:40, 883.53 examples/s]35848 examples [00:40, 894.81 examples/s]35938 examples [00:40, 889.19 examples/s]36027 examples [00:40, 875.45 examples/s]36115 examples [00:40, 873.91 examples/s]36205 examples [00:40, 880.73 examples/s]36294 examples [00:40, 878.95 examples/s]36382 examples [00:41, 874.02 examples/s]36470 examples [00:41, 851.37 examples/s]36556 examples [00:41, 847.06 examples/s]36641 examples [00:41, 843.49 examples/s]36728 examples [00:41, 850.82 examples/s]36814 examples [00:41, 850.72 examples/s]36900 examples [00:41, 829.13 examples/s]36988 examples [00:41, 842.34 examples/s]37081 examples [00:41, 866.66 examples/s]37175 examples [00:42, 884.77 examples/s]37264 examples [00:42, 852.27 examples/s]37350 examples [00:42, 829.41 examples/s]37440 examples [00:42, 847.95 examples/s]37527 examples [00:42, 854.13 examples/s]37613 examples [00:42, 854.06 examples/s]37702 examples [00:42, 862.71 examples/s]37791 examples [00:42, 870.46 examples/s]37880 examples [00:42, 876.02 examples/s]37968 examples [00:42, 858.23 examples/s]38057 examples [00:43, 866.17 examples/s]38149 examples [00:43, 880.20 examples/s]38243 examples [00:43, 895.14 examples/s]38337 examples [00:43, 905.72 examples/s]38430 examples [00:43, 912.42 examples/s]38522 examples [00:43, 905.79 examples/s]38613 examples [00:43, 904.57 examples/s]38707 examples [00:43, 912.50 examples/s]38800 examples [00:43, 915.25 examples/s]38893 examples [00:43, 916.87 examples/s]38985 examples [00:44, 916.58 examples/s]39077 examples [00:44, 898.16 examples/s]39167 examples [00:44, 862.59 examples/s]39255 examples [00:44, 865.16 examples/s]39343 examples [00:44, 867.76 examples/s]39430 examples [00:44, 863.78 examples/s]39518 examples [00:44, 868.58 examples/s]39610 examples [00:44, 881.92 examples/s]39699 examples [00:44, 882.01 examples/s]39788 examples [00:44, 848.26 examples/s]39879 examples [00:45, 865.46 examples/s]39967 examples [00:45, 868.71 examples/s]40055 examples [00:45, 829.43 examples/s]40148 examples [00:45, 855.83 examples/s]40244 examples [00:45, 884.59 examples/s]40334 examples [00:45, 847.96 examples/s]40420 examples [00:45, 846.41 examples/s]40513 examples [00:45, 868.76 examples/s]40603 examples [00:45, 877.69 examples/s]40695 examples [00:46, 888.57 examples/s]40785 examples [00:46, 874.50 examples/s]40875 examples [00:46, 880.03 examples/s]40964 examples [00:46, 881.67 examples/s]41055 examples [00:46, 887.41 examples/s]41147 examples [00:46, 896.79 examples/s]41237 examples [00:46, 889.50 examples/s]41327 examples [00:46, 892.53 examples/s]41417 examples [00:46, 891.11 examples/s]41507 examples [00:46, 881.69 examples/s]41596 examples [00:47, 880.48 examples/s]41685 examples [00:47, 871.09 examples/s]41773 examples [00:47, 853.69 examples/s]41859 examples [00:47, 837.49 examples/s]41951 examples [00:47, 859.39 examples/s]42041 examples [00:47, 870.82 examples/s]42132 examples [00:47, 881.70 examples/s]42227 examples [00:47, 899.47 examples/s]42321 examples [00:47, 908.12 examples/s]42412 examples [00:47, 903.37 examples/s]42505 examples [00:48, 909.86 examples/s]42597 examples [00:48, 887.49 examples/s]42691 examples [00:48, 901.58 examples/s]42784 examples [00:48, 907.66 examples/s]42877 examples [00:48, 911.64 examples/s]42970 examples [00:48, 915.95 examples/s]43062 examples [00:48, 912.83 examples/s]43154 examples [00:48, 914.43 examples/s]43246 examples [00:48, 916.06 examples/s]43338 examples [00:48, 909.77 examples/s]43430 examples [00:49, 907.55 examples/s]43521 examples [00:49, 888.07 examples/s]43612 examples [00:49, 893.96 examples/s]43703 examples [00:49, 895.82 examples/s]43796 examples [00:49, 904.94 examples/s]43887 examples [00:49, 904.91 examples/s]43978 examples [00:49, 897.70 examples/s]44072 examples [00:49, 906.87 examples/s]44163 examples [00:49, 907.69 examples/s]44254 examples [00:50, 904.04 examples/s]44346 examples [00:50, 906.71 examples/s]44437 examples [00:50, 899.64 examples/s]44527 examples [00:50, 881.72 examples/s]44620 examples [00:50, 895.22 examples/s]44712 examples [00:50, 901.04 examples/s]44803 examples [00:50, 900.28 examples/s]44894 examples [00:50, 884.48 examples/s]44987 examples [00:50, 896.95 examples/s]45081 examples [00:50, 909.14 examples/s]45174 examples [00:51, 913.89 examples/s]45266 examples [00:51, 913.90 examples/s]45358 examples [00:51, 902.33 examples/s]45449 examples [00:51, 896.18 examples/s]45539 examples [00:51, 893.58 examples/s]45632 examples [00:51, 902.12 examples/s]45726 examples [00:51, 910.87 examples/s]45818 examples [00:51, 904.23 examples/s]45911 examples [00:51, 911.31 examples/s]46004 examples [00:51, 915.47 examples/s]46096 examples [00:52, 875.05 examples/s]46185 examples [00:52, 877.61 examples/s]46274 examples [00:52, 858.11 examples/s]46361 examples [00:52, 847.66 examples/s]46453 examples [00:52, 866.62 examples/s]46541 examples [00:52, 868.17 examples/s]46634 examples [00:52, 884.03 examples/s]46725 examples [00:52, 889.83 examples/s]46820 examples [00:52, 905.99 examples/s]46912 examples [00:52, 908.82 examples/s]47005 examples [00:53, 913.05 examples/s]47097 examples [00:53, 914.95 examples/s]47190 examples [00:53, 917.52 examples/s]47282 examples [00:53, 887.25 examples/s]47372 examples [00:53, 890.02 examples/s]47465 examples [00:53, 900.86 examples/s]47562 examples [00:53, 918.57 examples/s]47655 examples [00:53, 912.86 examples/s]47748 examples [00:53, 916.60 examples/s]47840 examples [00:54, 892.57 examples/s]47934 examples [00:54, 905.32 examples/s]48025 examples [00:54, 889.13 examples/s]48117 examples [00:54, 897.05 examples/s]48207 examples [00:54, 893.15 examples/s]48298 examples [00:54, 897.96 examples/s]48392 examples [00:54, 909.59 examples/s]48488 examples [00:54, 921.68 examples/s]48581 examples [00:54, 893.17 examples/s]48674 examples [00:54, 903.73 examples/s]48767 examples [00:55, 911.22 examples/s]48859 examples [00:55, 880.22 examples/s]48948 examples [00:55, 880.04 examples/s]49037 examples [00:55, 882.16 examples/s]49129 examples [00:55, 890.84 examples/s]49220 examples [00:55, 894.44 examples/s]49310 examples [00:55, 890.49 examples/s]49400 examples [00:55, 880.37 examples/s]49489 examples [00:55, 877.33 examples/s]49580 examples [00:55, 884.32 examples/s]49669 examples [00:56, 882.60 examples/s]49758 examples [00:56, 879.50 examples/s]49848 examples [00:56, 881.72 examples/s]49937 examples [00:56, 872.55 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6524/50000 [00:00<00:00, 65239.91 examples/s] 36%|███▌      | 17892/50000 [00:00<00:00, 74801.55 examples/s] 59%|█████▉    | 29669/50000 [00:00<00:00, 83994.25 examples/s] 82%|████████▏ | 41114/50000 [00:00<00:00, 91276.62 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 748.63 examples/s]170 examples [00:00, 797.78 examples/s]264 examples [00:00, 834.00 examples/s]355 examples [00:00, 853.78 examples/s]439 examples [00:00, 847.93 examples/s]531 examples [00:00, 867.94 examples/s]623 examples [00:00, 882.57 examples/s]719 examples [00:00, 904.03 examples/s]812 examples [00:00, 909.58 examples/s]904 examples [00:01, 912.46 examples/s]1000 examples [00:01, 924.49 examples/s]1099 examples [00:01, 940.69 examples/s]1196 examples [00:01, 948.85 examples/s]1292 examples [00:01, 949.47 examples/s]1389 examples [00:01, 954.38 examples/s]1485 examples [00:01, 952.10 examples/s]1581 examples [00:01, 952.52 examples/s]1678 examples [00:01, 957.50 examples/s]1774 examples [00:01, 950.66 examples/s]1870 examples [00:02, 942.81 examples/s]1965 examples [00:02, 905.82 examples/s]2056 examples [00:02, 901.74 examples/s]2147 examples [00:02, 904.15 examples/s]2238 examples [00:02, 897.52 examples/s]2328 examples [00:02, 845.36 examples/s]2414 examples [00:02, 846.94 examples/s]2505 examples [00:02, 863.93 examples/s]2602 examples [00:02, 890.86 examples/s]2692 examples [00:02, 893.48 examples/s]2783 examples [00:03, 896.76 examples/s]2876 examples [00:03, 904.60 examples/s]2971 examples [00:03, 917.36 examples/s]3068 examples [00:03, 930.31 examples/s]3162 examples [00:03, 914.60 examples/s]3255 examples [00:03, 917.14 examples/s]3348 examples [00:03, 918.08 examples/s]3440 examples [00:03, 892.24 examples/s]3532 examples [00:03, 900.28 examples/s]3626 examples [00:03, 909.70 examples/s]3720 examples [00:04, 917.47 examples/s]3815 examples [00:04, 926.74 examples/s]3908 examples [00:04, 915.76 examples/s]4000 examples [00:04, 910.51 examples/s]4092 examples [00:04, 900.55 examples/s]4183 examples [00:04, 895.48 examples/s]4273 examples [00:04, 890.83 examples/s]4364 examples [00:04, 894.97 examples/s]4454 examples [00:04, 894.19 examples/s]4545 examples [00:04, 897.92 examples/s]4636 examples [00:05, 900.55 examples/s]4727 examples [00:05, 900.43 examples/s]4820 examples [00:05, 906.82 examples/s]4913 examples [00:05, 911.91 examples/s]5005 examples [00:05, 908.71 examples/s]5096 examples [00:05, 891.11 examples/s]5186 examples [00:05, 877.67 examples/s]5278 examples [00:05, 888.55 examples/s]5368 examples [00:05, 890.36 examples/s]5458 examples [00:06, 886.84 examples/s]5547 examples [00:06, 887.12 examples/s]5638 examples [00:06, 893.36 examples/s]5729 examples [00:06, 896.56 examples/s]5820 examples [00:06, 899.19 examples/s]5910 examples [00:06, 889.51 examples/s]6001 examples [00:06, 893.43 examples/s]6095 examples [00:06, 906.41 examples/s]6189 examples [00:06, 914.88 examples/s]6281 examples [00:06, 914.98 examples/s]6373 examples [00:07, 897.04 examples/s]6463 examples [00:07, 890.70 examples/s]6554 examples [00:07, 896.33 examples/s]6649 examples [00:07, 910.75 examples/s]6742 examples [00:07, 915.11 examples/s]6836 examples [00:07, 921.99 examples/s]6929 examples [00:07, 923.56 examples/s]7025 examples [00:07, 933.25 examples/s]7119 examples [00:07, 920.24 examples/s]7212 examples [00:07, 918.31 examples/s]7305 examples [00:08, 920.84 examples/s]7404 examples [00:08, 939.54 examples/s]7502 examples [00:08, 950.70 examples/s]7598 examples [00:08, 952.30 examples/s]7694 examples [00:08, 920.68 examples/s]7787 examples [00:08, 917.62 examples/s]7879 examples [00:08, 883.73 examples/s]7968 examples [00:08, 871.54 examples/s]8061 examples [00:08, 885.88 examples/s]8154 examples [00:08, 898.28 examples/s]8246 examples [00:09, 903.61 examples/s]8337 examples [00:09, 896.79 examples/s]8427 examples [00:09, 893.91 examples/s]8518 examples [00:09, 897.14 examples/s]8608 examples [00:09, 893.93 examples/s]8698 examples [00:09, 878.91 examples/s]8788 examples [00:09, 882.75 examples/s]8877 examples [00:09, 875.49 examples/s]8965 examples [00:09, 871.89 examples/s]9053 examples [00:10, 872.65 examples/s]9141 examples [00:10, 869.08 examples/s]9229 examples [00:10, 869.90 examples/s]9317 examples [00:10, 862.20 examples/s]9404 examples [00:10, 833.31 examples/s]9497 examples [00:10, 857.78 examples/s]9590 examples [00:10, 878.18 examples/s]9682 examples [00:10, 887.48 examples/s]9772 examples [00:10, 887.92 examples/s]9864 examples [00:10, 896.98 examples/s]9954 examples [00:11, 892.56 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW70GLO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW70GLO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f200a30fe18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f200a30fe18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f200a30fe18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.496017370223999 	 Accuracy: 32
Train Epoch: 1 	 Loss: 5.196482563018799 	 Accuracy: 140
model saves at 140 accuracy
Train Epoch: 2 	 Loss: 0.44661827325820924 	 Accuracy: 54
Train Epoch: 2 	 Loss: 6.555672740936279 	 Accuracy: 200
model saves at 200 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1fc31e6620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1fc31e6620>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f1fc31e6620> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fc1f4f950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fc1f4f950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fc1f4f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([7, 1, 4, 1, 4, 2, 4, 0, 1, 4]), array([7, 2, 5, 0, 2, 5, 0, 9, 3, 2]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f2024e21978> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f1fc1f3c898> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fbf9946a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fbf9946a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fbf9946a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:37, 270424.91it/s]  1%|          | 180224/26421880 [00:00<01:15, 347934.14it/s]  2%|▏         | 409600/26421880 [00:00<00:57, 453218.32it/s]  6%|▌         | 1523712/26421880 [00:00<00:39, 633678.96it/s] 13%|█▎        | 3334144/26421880 [00:01<00:26, 885373.58it/s] 28%|██▊       | 7503872/26421880 [00:01<00:15, 1253257.80it/s] 44%|████▍     | 11583488/26421880 [00:01<00:08, 1767101.28it/s] 57%|█████▋    | 15089664/26421880 [00:01<00:04, 2468095.53it/s] 70%|███████   | 18513920/26421880 [00:01<00:02, 3370683.49it/s] 86%|████████▌ | 22732800/26421880 [00:01<00:00, 4655836.58it/s]26427392it [00:01, 15319548.42it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 102090.14it/s]           
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 269631.60it/s]  3%|▎         | 131072/4422102 [00:00<00:12, 332116.56it/s] 10%|▉         | 425984/4422102 [00:00<00:09, 440336.54it/s] 24%|██▍       | 1081344/4422102 [00:00<00:05, 609120.41it/s] 78%|███████▊  | 3465216/4422102 [00:00<00:01, 857636.08it/s]4423680it [00:01, 4411790.91it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 38703.29it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0033664461970329285 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018586045265197755 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0032727158268292746 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.01641995644569397 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0026969702343146006 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.01244019228219986 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.002552768607934316 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.011088847935199738 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0020980394879976908 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.008445573687553406 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fbf991e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fbf991e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fbf991e18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([4, 3, 8, 4, 1, 4, 4, 4, 9, 4]), array([4, 3, 8, 2, 1, 4, 4, 6, 9, 4]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f1fc323b080> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fc1dff598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fc1dff598>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fc1dff598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110119.07it/s]  0%|          | 40960/18165135 [00:00<02:28, 122279.59it/s]  1%|          | 98304/18165135 [00:00<01:57, 153742.06it/s]  1%|          | 188416/18165135 [00:01<01:30, 198038.49it/s]  2%|▏         | 385024/18165135 [00:01<01:06, 265787.23it/s]  4%|▍         | 778240/18165135 [00:01<00:47, 363961.30it/s]  9%|▊         | 1564672/18165135 [00:01<00:32, 505046.37it/s] 17%|█▋        | 3145728/18165135 [00:01<00:21, 707047.53it/s] 32%|███▏      | 5808128/18165135 [00:01<00:12, 982241.55it/s] 40%|███▉      | 7225344/18165135 [00:02<00:08, 1362235.99it/s] 45%|████▍     | 8110080/18165135 [00:02<00:06, 1663210.67it/s] 49%|████▉     | 8929280/18165135 [00:02<00:04, 1885670.90it/s] 64%|██████▍   | 11714560/18165135 [00:02<00:02, 2584448.16it/s] 70%|██████▉   | 12705792/18165135 [00:03<00:02, 2433586.40it/s] 83%|████████▎ | 15024128/18165135 [00:03<00:00, 3260497.48it/s] 88%|████████▊ | 16023552/18165135 [00:03<00:00, 3222659.08it/s] 93%|█████████▎| 16818176/18165135 [00:03<00:00, 3632094.69it/s] 98%|█████████▊| 17743872/18165135 [00:03<00:00, 4287930.29it/s]18169856it [00:03, 4555033.42it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110398.05it/s]32768it [00:00, 52843.01it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110275.81it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122407.86it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153906.43it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 201258.96it/s] 14%|█▎        | 417792/3041136 [00:01<00:10, 256554.08it/s] 37%|███▋      | 1122304/3041136 [00:01<00:05, 358176.56it/s] 47%|████▋     | 1417216/3041136 [00:01<00:03, 474833.29it/s] 57%|█████▋    | 1728512/3041136 [00:01<00:02, 618108.26it/s] 67%|██████▋   | 2048000/3041136 [00:01<00:01, 785933.87it/s] 78%|███████▊  | 2375680/3041136 [00:02<00:00, 973740.97it/s] 89%|████████▉ | 2711552/3041136 [00:02<00:00, 1173921.66it/s]3047424it [00:02, 1313447.41it/s]                             
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17176.45it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0040072900255521135 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.03890702724456787 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.003338549127181371 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.035386595845222475 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.004028462290763855 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.046309571504592896 	 Accuracy: 2
Train Epoch: 4 	 Loss: 0.00293719944357872 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.050854619145393375 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 5 	 Loss: 0.002887967427571615 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.03402546799182892 	 Accuracy: 4

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fbf998d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fbf998d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fbf998d08> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([8, 8, 6, 6, 9, 7, 9, 8, 6, 9]), array([3, 3, 4, 0, 5, 0, 9, 8, 1, 9]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

