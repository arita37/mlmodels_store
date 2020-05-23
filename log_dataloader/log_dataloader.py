
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '7423a9c1aea8d708841a3941e104542978e088ce', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7423a9c1aea8d708841a3941e104542978e088ce

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f5c0b754390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c0a74e488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c0a74e488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c0a74e488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 158218.05it/s] 73%|███████▎  | 7192576/9912422 [00:00<00:12, 225808.57it/s]9920512it [00:00, 43810876.86it/s]                           
0it [00:00, ?it/s]32768it [00:00, 519716.22it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 153025.18it/s]1654784it [00:00, 10689988.03it/s]                         
0it [00:00, ?it/s]8192it [00:00, 123021.34it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.00290068781375885 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01915331470966339 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0025899816354115804 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.036784974217414855 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.0022038836777210236 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.019994078636169435 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.002178438941637675 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.02236473435163498 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.001456481084227562 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.009107798099517823 	 Accuracy: 7
model saves at 7 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c0a738f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c0a738f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c0a738f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([3, 6, 7, 5, 7, 2, 3, 9, 4, 4]), array([8, 6, 7, 5, 7, 2, 3, 9, 4, 4]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f5c0c5c4550> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c00dbebf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c00dbebf8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c00dbebf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.002100431720415751 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018359631538391114 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.002224269946416219 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.015841343641281126 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c00dbe488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c00dbe488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c00dbe488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([7, 7, 7, 7, 7, 0, 7, 0, 0, 7]), array([7, 7, 7, 7, 4, 6, 3, 6, 6, 7]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s]  5%|▌         | 2.38M/44.7M [00:00<00:01, 24.9MB/s] 10%|█         | 4.56M/44.7M [00:00<00:01, 24.3MB/s] 13%|█▎        | 6.00M/44.7M [00:00<00:02, 19.0MB/s] 16%|█▌        | 7.17M/44.7M [00:00<00:02, 15.2MB/s] 21%|██        | 9.30M/44.7M [00:00<00:02, 16.8MB/s] 24%|██▍       | 10.7M/44.7M [00:00<00:02, 13.1MB/s] 29%|██▉       | 13.1M/44.7M [00:00<00:02, 15.0MB/s] 33%|███▎      | 14.6M/44.7M [00:00<00:02, 13.2MB/s] 36%|███▌      | 16.1M/44.7M [00:01<00:02, 13.9MB/s] 42%|████▏     | 18.7M/44.7M [00:01<00:01, 16.2MB/s] 47%|████▋     | 20.9M/44.7M [00:01<00:01, 17.8MB/s] 52%|█████▏    | 23.0M/44.7M [00:01<00:01, 18.2MB/s] 56%|█████▌    | 24.9M/44.7M [00:01<00:01, 16.5MB/s] 59%|█████▉    | 26.6M/44.7M [00:01<00:01, 14.4MB/s] 64%|██████▍   | 28.8M/44.7M [00:01<00:01, 16.0MB/s] 68%|██████▊   | 30.5M/44.7M [00:01<00:00, 16.5MB/s] 72%|███████▏  | 32.2M/44.7M [00:02<00:01, 12.3MB/s] 75%|███████▌  | 33.6M/44.7M [00:02<00:00, 12.9MB/s] 79%|███████▊  | 35.1M/44.7M [00:02<00:00, 13.6MB/s] 84%|████████▎ | 37.3M/44.7M [00:02<00:00, 15.5MB/s] 87%|████████▋ | 39.0M/44.7M [00:02<00:00, 15.2MB/s] 91%|█████████ | 40.6M/44.7M [00:02<00:00, 14.4MB/s] 96%|█████████▋| 43.0M/44.7M [00:02<00:00, 16.6MB/s]100%|██████████| 44.7M/44.7M [00:02<00:00, 16.4MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f5c58113ef0> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5c0a720268>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5c0a720268>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f5c0a720268> , (data_info, **args) 

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
Dl Size...:   4%|▍         | 7/162 [00:00<01:15,  2.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:52,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:52,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:51,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:51,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:50,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:50,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:49,  2.91 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:33,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:33,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:32,  4.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:21,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:21,  5.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:14,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:14,  8.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  8.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 26.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 33.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 49.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 57.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 65.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.36 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.36 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.37s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.04 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.38s/ url]
0 examples [00:00, ? examples/s]2020-05-23 00:09:11.369625: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-23 00:09:11.384394: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-23 00:09:11.384568: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564f56229d70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 00:09:11.384581: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
16 examples [00:00, 158.60 examples/s]118 examples [00:00, 212.32 examples/s]222 examples [00:00, 278.88 examples/s]327 examples [00:00, 357.55 examples/s]430 examples [00:00, 444.26 examples/s]536 examples [00:00, 537.85 examples/s]640 examples [00:00, 628.89 examples/s]743 examples [00:00, 711.53 examples/s]848 examples [00:00, 786.22 examples/s]952 examples [00:01, 847.03 examples/s]1054 examples [00:01, 890.94 examples/s]1157 examples [00:01, 928.19 examples/s]1258 examples [00:01, 935.63 examples/s]1365 examples [00:01, 971.41 examples/s]1475 examples [00:01, 1005.64 examples/s]1581 examples [00:01, 1018.98 examples/s]1690 examples [00:01, 1037.02 examples/s]1796 examples [00:01, 1015.65 examples/s]1899 examples [00:01, 1013.47 examples/s]2004 examples [00:02, 1022.06 examples/s]2107 examples [00:02, 1022.76 examples/s]2210 examples [00:02, 1015.35 examples/s]2312 examples [00:02, 1004.53 examples/s]2417 examples [00:02, 1015.39 examples/s]2529 examples [00:02, 1043.80 examples/s]2634 examples [00:02, 1020.92 examples/s]2743 examples [00:02, 1039.06 examples/s]2848 examples [00:02, 1034.36 examples/s]2952 examples [00:02, 1035.94 examples/s]3057 examples [00:03, 1039.78 examples/s]3162 examples [00:03, 1032.05 examples/s]3266 examples [00:03, 1029.21 examples/s]3369 examples [00:03, 1024.98 examples/s]3475 examples [00:03, 1033.02 examples/s]3580 examples [00:03, 1035.27 examples/s]3685 examples [00:03, 1038.99 examples/s]3789 examples [00:03, 1035.57 examples/s]3893 examples [00:03, 1036.86 examples/s]3998 examples [00:03, 1038.39 examples/s]4102 examples [00:04, 1032.80 examples/s]4207 examples [00:04, 1036.94 examples/s]4311 examples [00:04, 1033.28 examples/s]4415 examples [00:04, 1019.99 examples/s]4518 examples [00:04, 1006.51 examples/s]4628 examples [00:04, 1030.61 examples/s]4736 examples [00:04, 1044.62 examples/s]4845 examples [00:04, 1056.24 examples/s]4951 examples [00:04, 1045.34 examples/s]5063 examples [00:04, 1064.22 examples/s]5170 examples [00:05, 1063.97 examples/s]5277 examples [00:05, 1059.65 examples/s]5384 examples [00:05, 1025.84 examples/s]5487 examples [00:05, 1014.82 examples/s]5597 examples [00:05, 1038.34 examples/s]5709 examples [00:05, 1059.38 examples/s]5820 examples [00:05, 1072.85 examples/s]5929 examples [00:05, 1077.51 examples/s]6041 examples [00:05, 1087.50 examples/s]6153 examples [00:05, 1094.52 examples/s]6265 examples [00:06, 1100.14 examples/s]6376 examples [00:06, 1061.28 examples/s]6483 examples [00:06, 1032.31 examples/s]6587 examples [00:06, 1029.35 examples/s]6693 examples [00:06, 1037.30 examples/s]6799 examples [00:06, 1043.52 examples/s]6904 examples [00:06, 1037.52 examples/s]7008 examples [00:06, 1007.65 examples/s]7119 examples [00:06, 1034.08 examples/s]7230 examples [00:07, 1053.12 examples/s]7337 examples [00:07, 1056.68 examples/s]7443 examples [00:07, 1040.73 examples/s]7548 examples [00:07, 1039.92 examples/s]7657 examples [00:07, 1054.37 examples/s]7763 examples [00:07, 1055.95 examples/s]7871 examples [00:07, 1058.82 examples/s]7977 examples [00:07, 1044.64 examples/s]8082 examples [00:07, 1028.80 examples/s]8188 examples [00:07, 1036.83 examples/s]8301 examples [00:08, 1060.64 examples/s]8410 examples [00:08, 1068.41 examples/s]8517 examples [00:08, 1061.43 examples/s]8624 examples [00:08, 1017.41 examples/s]8733 examples [00:08, 1038.03 examples/s]8843 examples [00:08, 1055.67 examples/s]8955 examples [00:08, 1072.03 examples/s]9064 examples [00:08, 1076.16 examples/s]9173 examples [00:08, 1079.07 examples/s]9283 examples [00:08, 1083.94 examples/s]9392 examples [00:09, 1082.87 examples/s]9501 examples [00:09, 1080.46 examples/s]9610 examples [00:09, 1075.75 examples/s]9718 examples [00:09, 1061.66 examples/s]9825 examples [00:09, 1041.59 examples/s]9933 examples [00:09, 1050.47 examples/s]10039 examples [00:09, 1000.96 examples/s]10142 examples [00:09, 1008.16 examples/s]10246 examples [00:09, 1015.55 examples/s]10351 examples [00:10, 1022.98 examples/s]10454 examples [00:10, 1013.94 examples/s]10557 examples [00:10, 1016.93 examples/s]10659 examples [00:10, 1016.50 examples/s]10761 examples [00:10, 1015.97 examples/s]10865 examples [00:10, 1022.03 examples/s]10969 examples [00:10, 1026.16 examples/s]11072 examples [00:10, 1027.03 examples/s]11175 examples [00:10, 1026.64 examples/s]11278 examples [00:10, 1023.56 examples/s]11383 examples [00:11, 1030.59 examples/s]11487 examples [00:11, 1021.53 examples/s]11590 examples [00:11, 997.58 examples/s] 11690 examples [00:11, 989.89 examples/s]11790 examples [00:11, 971.69 examples/s]11900 examples [00:11, 1005.16 examples/s]12006 examples [00:11, 1017.44 examples/s]12109 examples [00:11, 1008.49 examples/s]12211 examples [00:11, 1005.01 examples/s]12317 examples [00:11, 1019.98 examples/s]12426 examples [00:12, 1039.97 examples/s]12533 examples [00:12, 1048.77 examples/s]12641 examples [00:12, 1055.49 examples/s]12747 examples [00:12, 1041.88 examples/s]12857 examples [00:12, 1056.09 examples/s]12965 examples [00:12, 1061.69 examples/s]13072 examples [00:12, 1060.39 examples/s]13184 examples [00:12, 1076.15 examples/s]13292 examples [00:12, 1050.43 examples/s]13398 examples [00:12, 1051.55 examples/s]13504 examples [00:13, 1051.14 examples/s]13610 examples [00:13, 1040.92 examples/s]13715 examples [00:13, 1036.75 examples/s]13819 examples [00:13, 1010.28 examples/s]13921 examples [00:13, 1005.08 examples/s]14026 examples [00:13, 1017.27 examples/s]14130 examples [00:13, 1023.15 examples/s]14233 examples [00:13, 1005.38 examples/s]14336 examples [00:13, 1010.20 examples/s]14441 examples [00:14, 1021.12 examples/s]14547 examples [00:14, 1030.93 examples/s]14654 examples [00:14, 1040.54 examples/s]14759 examples [00:14, 1020.85 examples/s]14862 examples [00:14, 1018.50 examples/s]14964 examples [00:14, 998.64 examples/s] 15065 examples [00:14, 1001.59 examples/s]15169 examples [00:14, 1010.82 examples/s]15274 examples [00:14, 1019.65 examples/s]15377 examples [00:14, 1016.20 examples/s]15479 examples [00:15, 983.10 examples/s] 15578 examples [00:15, 933.71 examples/s]15681 examples [00:15, 960.50 examples/s]15783 examples [00:15, 976.15 examples/s]15887 examples [00:15, 991.69 examples/s]15987 examples [00:15, 993.94 examples/s]16091 examples [00:15, 1005.75 examples/s]16194 examples [00:15, 1012.64 examples/s]16299 examples [00:15, 1021.18 examples/s]16402 examples [00:15, 1020.44 examples/s]16505 examples [00:16, 1022.07 examples/s]16611 examples [00:16, 1032.32 examples/s]16715 examples [00:16, 1033.33 examples/s]16819 examples [00:16, 1033.91 examples/s]16923 examples [00:16, 1033.48 examples/s]17036 examples [00:16, 1060.18 examples/s]17148 examples [00:16, 1077.10 examples/s]17256 examples [00:16, 1049.16 examples/s]17362 examples [00:16, 1047.20 examples/s]17467 examples [00:16, 1041.54 examples/s]17572 examples [00:17, 1021.46 examples/s]17675 examples [00:17, 1012.18 examples/s]17777 examples [00:17, 996.62 examples/s] 17885 examples [00:17, 1019.48 examples/s]17988 examples [00:17, 1016.24 examples/s]18090 examples [00:17, 1015.61 examples/s]18192 examples [00:17, 1008.78 examples/s]18294 examples [00:17, 1012.05 examples/s]18407 examples [00:17, 1043.87 examples/s]18516 examples [00:17, 1056.57 examples/s]18628 examples [00:18, 1072.57 examples/s]18740 examples [00:18, 1082.51 examples/s]18851 examples [00:18, 1088.40 examples/s]18961 examples [00:18, 1091.18 examples/s]19071 examples [00:18, 1085.18 examples/s]19184 examples [00:18, 1096.38 examples/s]19295 examples [00:18, 1099.09 examples/s]19405 examples [00:18, 1093.07 examples/s]19515 examples [00:18, 1067.83 examples/s]19622 examples [00:19, 1052.27 examples/s]19728 examples [00:19, 1040.51 examples/s]19833 examples [00:19, 1038.03 examples/s]19939 examples [00:19, 1043.72 examples/s]20044 examples [00:19, 989.25 examples/s] 20148 examples [00:19, 1003.49 examples/s]20252 examples [00:19, 1013.17 examples/s]20357 examples [00:19, 1021.97 examples/s]20462 examples [00:19, 1027.27 examples/s]20565 examples [00:19, 1025.28 examples/s]20668 examples [00:20, 1017.04 examples/s]20770 examples [00:20, 1009.02 examples/s]20872 examples [00:20, 1009.83 examples/s]20982 examples [00:20, 1034.13 examples/s]21086 examples [00:20, 988.47 examples/s] 21186 examples [00:20, 970.88 examples/s]21294 examples [00:20, 999.22 examples/s]21404 examples [00:20, 1025.00 examples/s]21516 examples [00:20, 1049.82 examples/s]21625 examples [00:20, 1059.80 examples/s]21734 examples [00:21, 1066.18 examples/s]21844 examples [00:21, 1076.04 examples/s]21954 examples [00:21, 1081.44 examples/s]22065 examples [00:21, 1089.46 examples/s]22175 examples [00:21, 1080.70 examples/s]22285 examples [00:21, 1085.24 examples/s]22396 examples [00:21, 1089.78 examples/s]22506 examples [00:21, 1087.73 examples/s]22615 examples [00:21, 1067.05 examples/s]22722 examples [00:21, 1066.82 examples/s]22829 examples [00:22, 1057.81 examples/s]22935 examples [00:22, 1052.14 examples/s]23041 examples [00:22, 1042.45 examples/s]23146 examples [00:22, 1037.73 examples/s]23250 examples [00:22, 1031.94 examples/s]23354 examples [00:22, 1032.81 examples/s]23458 examples [00:22, 1011.28 examples/s]23563 examples [00:22, 1021.08 examples/s]23666 examples [00:22, 1021.36 examples/s]23769 examples [00:23, 1021.75 examples/s]23872 examples [00:23, 1011.60 examples/s]23975 examples [00:23, 1016.22 examples/s]24078 examples [00:23, 1019.75 examples/s]24184 examples [00:23, 1029.87 examples/s]24288 examples [00:23, 1021.92 examples/s]24391 examples [00:23, 984.64 examples/s] 24491 examples [00:23, 986.99 examples/s]24590 examples [00:23, 971.62 examples/s]24688 examples [00:23, 971.50 examples/s]24795 examples [00:24, 997.45 examples/s]24902 examples [00:24, 1017.92 examples/s]25012 examples [00:24, 1041.21 examples/s]25123 examples [00:24, 1059.04 examples/s]25233 examples [00:24, 1070.42 examples/s]25343 examples [00:24, 1078.08 examples/s]25451 examples [00:24, 1078.39 examples/s]25559 examples [00:24, 1078.68 examples/s]25669 examples [00:24, 1083.00 examples/s]25778 examples [00:24, 1060.94 examples/s]25885 examples [00:25, 1047.51 examples/s]25990 examples [00:25, 1029.43 examples/s]26095 examples [00:25, 1035.50 examples/s]26199 examples [00:25, 1035.47 examples/s]26303 examples [00:25, 1026.26 examples/s]26408 examples [00:25, 1032.42 examples/s]26512 examples [00:25, 1016.66 examples/s]26614 examples [00:25, 1009.63 examples/s]26718 examples [00:25, 1015.76 examples/s]26826 examples [00:25, 1033.10 examples/s]26930 examples [00:26, 1033.29 examples/s]27034 examples [00:26, 1035.24 examples/s]27138 examples [00:26, 992.57 examples/s] 27238 examples [00:26, 988.11 examples/s]27341 examples [00:26, 998.22 examples/s]27447 examples [00:26, 1015.76 examples/s]27549 examples [00:26, 974.50 examples/s] 27657 examples [00:26, 1001.51 examples/s]27765 examples [00:26, 1021.61 examples/s]27874 examples [00:27, 1039.27 examples/s]27981 examples [00:27, 1046.21 examples/s]28086 examples [00:27, 1037.07 examples/s]28192 examples [00:27, 1041.92 examples/s]28297 examples [00:27, 1027.13 examples/s]28400 examples [00:27, 1006.71 examples/s]28507 examples [00:27, 1024.31 examples/s]28610 examples [00:27, 1021.20 examples/s]28713 examples [00:27, 1013.90 examples/s]28819 examples [00:27, 1025.12 examples/s]28922 examples [00:28, 1025.95 examples/s]29026 examples [00:28, 1027.47 examples/s]29129 examples [00:28, 1023.49 examples/s]29233 examples [00:28, 1025.65 examples/s]29336 examples [00:28, 1018.34 examples/s]29442 examples [00:28, 1029.06 examples/s]29545 examples [00:28, 1023.32 examples/s]29648 examples [00:28, 995.27 examples/s] 29748 examples [00:28, 982.77 examples/s]29847 examples [00:28, 971.62 examples/s]29948 examples [00:29, 982.78 examples/s]30047 examples [00:29, 945.19 examples/s]30150 examples [00:29, 968.23 examples/s]30259 examples [00:29, 1000.80 examples/s]30369 examples [00:29, 1026.72 examples/s]30473 examples [00:29, 1018.67 examples/s]30576 examples [00:29, 996.04 examples/s] 30677 examples [00:29, 993.72 examples/s]30784 examples [00:29, 1013.41 examples/s]30893 examples [00:30, 1032.41 examples/s]30999 examples [00:30, 1039.04 examples/s]31104 examples [00:30, 1033.80 examples/s]31208 examples [00:30, 1030.18 examples/s]31312 examples [00:30, 1026.30 examples/s]31415 examples [00:30, 1007.99 examples/s]31516 examples [00:30, 1001.59 examples/s]31617 examples [00:30, 999.65 examples/s] 31719 examples [00:30, 1005.61 examples/s]31820 examples [00:30, 1006.15 examples/s]31923 examples [00:31, 1011.60 examples/s]32027 examples [00:31, 1017.13 examples/s]32129 examples [00:31, 1014.54 examples/s]32233 examples [00:31, 1019.84 examples/s]32336 examples [00:31, 1016.71 examples/s]32439 examples [00:31, 1020.26 examples/s]32542 examples [00:31, 1019.66 examples/s]32644 examples [00:31, 1012.39 examples/s]32748 examples [00:31, 1019.17 examples/s]32850 examples [00:31, 1013.94 examples/s]32956 examples [00:32, 1025.17 examples/s]33059 examples [00:32, 1005.24 examples/s]33160 examples [00:32, 990.68 examples/s] 33260 examples [00:32, 972.76 examples/s]33362 examples [00:32, 986.00 examples/s]33464 examples [00:32, 994.54 examples/s]33569 examples [00:32, 1009.08 examples/s]33671 examples [00:32, 930.11 examples/s] 33779 examples [00:32, 969.85 examples/s]33878 examples [00:32, 959.83 examples/s]33979 examples [00:33, 973.73 examples/s]34078 examples [00:33, 976.50 examples/s]34179 examples [00:33, 984.39 examples/s]34281 examples [00:33, 993.72 examples/s]34384 examples [00:33, 1004.18 examples/s]34487 examples [00:33, 1011.58 examples/s]34589 examples [00:33, 1009.94 examples/s]34691 examples [00:33, 1012.72 examples/s]34793 examples [00:33, 1014.24 examples/s]34895 examples [00:33, 1012.01 examples/s]34998 examples [00:34, 1015.61 examples/s]35104 examples [00:34, 1026.37 examples/s]35207 examples [00:34, 1026.83 examples/s]35316 examples [00:34, 1044.02 examples/s]35424 examples [00:34, 1054.13 examples/s]35535 examples [00:34, 1069.00 examples/s]35643 examples [00:34, 1071.36 examples/s]35751 examples [00:34, 1071.79 examples/s]35859 examples [00:34, 1069.96 examples/s]35967 examples [00:35, 1046.90 examples/s]36075 examples [00:35, 1055.01 examples/s]36182 examples [00:35, 1058.37 examples/s]36288 examples [00:35, 1039.29 examples/s]36394 examples [00:35, 1043.81 examples/s]36503 examples [00:35, 1054.85 examples/s]36609 examples [00:35, 1026.67 examples/s]36713 examples [00:35, 1030.48 examples/s]36817 examples [00:35, 1001.14 examples/s]36918 examples [00:35, 1002.68 examples/s]37023 examples [00:36, 1015.37 examples/s]37125 examples [00:36, 995.85 examples/s] 37230 examples [00:36, 1009.35 examples/s]37338 examples [00:36, 1029.31 examples/s]37442 examples [00:36, 1028.85 examples/s]37546 examples [00:36, 1026.32 examples/s]37649 examples [00:36, 1024.03 examples/s]37752 examples [00:36, 1024.54 examples/s]37857 examples [00:36, 1031.53 examples/s]37961 examples [00:36, 1032.96 examples/s]38066 examples [00:37, 1036.69 examples/s]38170 examples [00:37, 1037.49 examples/s]38274 examples [00:37, 1028.33 examples/s]38377 examples [00:37, 1028.24 examples/s]38480 examples [00:37, 1028.75 examples/s]38583 examples [00:37, 1027.61 examples/s]38689 examples [00:37, 1034.64 examples/s]38793 examples [00:37, 1014.18 examples/s]38898 examples [00:37, 1023.06 examples/s]39003 examples [00:37, 1030.30 examples/s]39109 examples [00:38, 1038.96 examples/s]39216 examples [00:38, 1045.45 examples/s]39321 examples [00:38, 1040.10 examples/s]39426 examples [00:38, 1023.08 examples/s]39530 examples [00:38, 1026.32 examples/s]39635 examples [00:38, 1031.68 examples/s]39739 examples [00:38, 1030.26 examples/s]39843 examples [00:38, 1026.60 examples/s]39946 examples [00:38, 989.74 examples/s] 40046 examples [00:39, 956.15 examples/s]40143 examples [00:39, 952.18 examples/s]40244 examples [00:39, 967.74 examples/s]40347 examples [00:39, 983.06 examples/s]40452 examples [00:39, 1001.66 examples/s]40554 examples [00:39, 1005.98 examples/s]40659 examples [00:39, 1016.66 examples/s]40762 examples [00:39, 1018.14 examples/s]40864 examples [00:39, 1014.00 examples/s]40966 examples [00:39, 998.15 examples/s] 41066 examples [00:40, 995.96 examples/s]41166 examples [00:40, 993.83 examples/s]41271 examples [00:40, 1009.54 examples/s]41374 examples [00:40, 1014.12 examples/s]41482 examples [00:40, 1032.03 examples/s]41586 examples [00:40, 1022.97 examples/s]41689 examples [00:40, 1024.74 examples/s]41796 examples [00:40, 1036.60 examples/s]41900 examples [00:40, 1012.06 examples/s]42002 examples [00:40, 1007.07 examples/s]42103 examples [00:41, 994.35 examples/s] 42207 examples [00:41, 1007.56 examples/s]42308 examples [00:41, 1005.07 examples/s]42412 examples [00:41, 1013.09 examples/s]42514 examples [00:41, 991.81 examples/s] 42618 examples [00:41, 1004.64 examples/s]42720 examples [00:41, 1006.47 examples/s]42830 examples [00:41, 1030.46 examples/s]42939 examples [00:41, 1045.45 examples/s]43044 examples [00:41, 999.36 examples/s] 43154 examples [00:42, 1025.77 examples/s]43262 examples [00:42, 1040.28 examples/s]43371 examples [00:42, 1052.16 examples/s]43477 examples [00:42, 1033.78 examples/s]43585 examples [00:42, 1046.50 examples/s]43693 examples [00:42, 1053.98 examples/s]43799 examples [00:42, 1050.77 examples/s]43905 examples [00:42, 1044.59 examples/s]44010 examples [00:42, 1034.43 examples/s]44114 examples [00:42, 1032.51 examples/s]44224 examples [00:43, 1049.02 examples/s]44333 examples [00:43, 1059.19 examples/s]44440 examples [00:43, 1061.65 examples/s]44547 examples [00:43, 1053.65 examples/s]44653 examples [00:43, 1029.68 examples/s]44762 examples [00:43, 1045.04 examples/s]44871 examples [00:43, 1055.49 examples/s]44979 examples [00:43, 1062.32 examples/s]45086 examples [00:43, 1053.75 examples/s]45192 examples [00:44, 1053.59 examples/s]45302 examples [00:44, 1065.87 examples/s]45409 examples [00:44, 1061.15 examples/s]45516 examples [00:44, 1059.42 examples/s]45622 examples [00:44, 1046.74 examples/s]45732 examples [00:44, 1059.92 examples/s]45839 examples [00:44, 1051.75 examples/s]45946 examples [00:44, 1056.62 examples/s]46054 examples [00:44, 1061.64 examples/s]46161 examples [00:44, 1060.35 examples/s]46268 examples [00:45, 1001.20 examples/s]46374 examples [00:45, 1015.70 examples/s]46483 examples [00:45, 1034.67 examples/s]46593 examples [00:45, 1051.49 examples/s]46700 examples [00:45, 1054.73 examples/s]46809 examples [00:45, 1063.98 examples/s]46919 examples [00:45, 1072.24 examples/s]47029 examples [00:45, 1078.66 examples/s]47139 examples [00:45, 1082.59 examples/s]47248 examples [00:45, 1078.04 examples/s]47356 examples [00:46, 1047.42 examples/s]47461 examples [00:46, 1009.24 examples/s]47564 examples [00:46, 1014.72 examples/s]47673 examples [00:46, 1035.54 examples/s]47779 examples [00:46, 1040.39 examples/s]47884 examples [00:46, 1037.81 examples/s]47988 examples [00:46, 1036.75 examples/s]48096 examples [00:46, 1048.50 examples/s]48201 examples [00:46, 1025.45 examples/s]48307 examples [00:46, 1035.00 examples/s]48417 examples [00:47, 1052.03 examples/s]48524 examples [00:47, 1056.38 examples/s]48635 examples [00:47, 1070.33 examples/s]48743 examples [00:47, 1017.07 examples/s]48851 examples [00:47, 1032.71 examples/s]48962 examples [00:47, 1052.72 examples/s]49072 examples [00:47, 1064.95 examples/s]49179 examples [00:47, 1053.27 examples/s]49285 examples [00:47, 1048.42 examples/s]49391 examples [00:48, 1032.24 examples/s]49495 examples [00:48, 1010.46 examples/s]49600 examples [00:48, 1020.72 examples/s]49703 examples [00:48, 1021.37 examples/s]49812 examples [00:48, 1039.33 examples/s]49920 examples [00:48, 1049.80 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7073/50000 [00:00<00:00, 70729.06 examples/s] 40%|███▉      | 19945/50000 [00:00<00:00, 81782.13 examples/s] 65%|██████▌   | 32584/50000 [00:00<00:00, 91466.74 examples/s] 89%|████████▉ | 44571/50000 [00:00<00:00, 98463.64 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 852.47 examples/s]192 examples [00:00, 903.79 examples/s]298 examples [00:00, 944.04 examples/s]404 examples [00:00, 974.96 examples/s]509 examples [00:00, 994.16 examples/s]615 examples [00:00, 1010.53 examples/s]720 examples [00:00, 1019.35 examples/s]825 examples [00:00, 1028.13 examples/s]924 examples [00:00, 1013.14 examples/s]1023 examples [00:01, 995.76 examples/s]1122 examples [00:01, 992.34 examples/s]1222 examples [00:01, 992.01 examples/s]1325 examples [00:01, 1002.26 examples/s]1428 examples [00:01, 1010.26 examples/s]1540 examples [00:01, 1039.23 examples/s]1651 examples [00:01, 1058.13 examples/s]1763 examples [00:01, 1073.95 examples/s]1875 examples [00:01, 1084.40 examples/s]1984 examples [00:01, 1082.65 examples/s]2093 examples [00:02, 1075.08 examples/s]2201 examples [00:02, 1040.42 examples/s]2306 examples [00:02, 1024.73 examples/s]2410 examples [00:02, 1028.86 examples/s]2517 examples [00:02, 1040.40 examples/s]2626 examples [00:02, 1054.09 examples/s]2732 examples [00:02, 1054.70 examples/s]2842 examples [00:02, 1065.34 examples/s]2953 examples [00:02, 1078.09 examples/s]3061 examples [00:02, 1077.73 examples/s]3173 examples [00:03, 1087.96 examples/s]3282 examples [00:03, 1088.02 examples/s]3391 examples [00:03, 1078.10 examples/s]3501 examples [00:03, 1082.80 examples/s]3610 examples [00:03, 1082.37 examples/s]3722 examples [00:03, 1092.45 examples/s]3832 examples [00:03, 1094.33 examples/s]3942 examples [00:03, 1077.15 examples/s]4050 examples [00:03, 1066.83 examples/s]4157 examples [00:03, 1051.46 examples/s]4268 examples [00:04, 1066.44 examples/s]4377 examples [00:04, 1072.62 examples/s]4485 examples [00:04, 1061.51 examples/s]4598 examples [00:04, 1080.74 examples/s]4710 examples [00:04, 1090.05 examples/s]4823 examples [00:04, 1101.11 examples/s]4935 examples [00:04, 1106.34 examples/s]5047 examples [00:04, 1108.46 examples/s]5160 examples [00:04, 1114.74 examples/s]5272 examples [00:04, 1109.20 examples/s]5383 examples [00:05, 1074.93 examples/s]5491 examples [00:05, 1045.28 examples/s]5605 examples [00:05, 1070.61 examples/s]5716 examples [00:05, 1080.22 examples/s]5826 examples [00:05, 1083.55 examples/s]5935 examples [00:05, 1077.90 examples/s]6044 examples [00:05, 1080.62 examples/s]6155 examples [00:05, 1087.92 examples/s]6264 examples [00:05, 1073.88 examples/s]6372 examples [00:06, 1065.61 examples/s]6479 examples [00:06, 1047.05 examples/s]6584 examples [00:06, 1029.55 examples/s]6694 examples [00:06, 1049.27 examples/s]6806 examples [00:06, 1069.31 examples/s]6914 examples [00:06, 1059.21 examples/s]7021 examples [00:06, 1055.68 examples/s]7127 examples [00:06, 1052.48 examples/s]7237 examples [00:06, 1065.65 examples/s]7348 examples [00:06, 1078.53 examples/s]7459 examples [00:07, 1085.17 examples/s]7568 examples [00:07, 1070.36 examples/s]7676 examples [00:07, 1056.81 examples/s]7782 examples [00:07, 1026.12 examples/s]7885 examples [00:07, 1017.39 examples/s]7994 examples [00:07, 1035.83 examples/s]8100 examples [00:07, 1041.54 examples/s]8206 examples [00:07, 1044.97 examples/s]8317 examples [00:07, 1061.37 examples/s]8428 examples [00:07, 1073.81 examples/s]8536 examples [00:08, 1071.07 examples/s]8644 examples [00:08, 1039.30 examples/s]8754 examples [00:08, 1055.68 examples/s]8864 examples [00:08, 1066.17 examples/s]8971 examples [00:08, 1064.84 examples/s]9080 examples [00:08, 1069.65 examples/s]9190 examples [00:08, 1076.56 examples/s]9298 examples [00:08, 1070.70 examples/s]9406 examples [00:08, 1050.37 examples/s]9512 examples [00:08, 1048.31 examples/s]9617 examples [00:09, 1047.94 examples/s]9722 examples [00:09, 1047.50 examples/s]9827 examples [00:09, 1045.96 examples/s]9933 examples [00:09, 1047.88 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePKJXEM/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePKJXEM/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5c02938620>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5c02938620>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5c02938620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.5082615447044373 	 Accuracy: 34
Train Epoch: 1 	 Loss: 3.907748889923096 	 Accuracy: 220
model saves at 220 accuracy
Train Epoch: 2 	 Loss: 0.4465182638168335 	 Accuracy: 56
Train Epoch: 2 	 Loss: 9.320826721191406 	 Accuracy: 260
model saves at 260 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5baa0479d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5baa0479d8>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f5baa0479d8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ba8d659d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ba8d659d8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ba8d659d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([7, 9, 9, 8, 0, 9, 0, 7, 8, 8]), array([0, 1, 9, 0, 5, 9, 2, 3, 8, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f5c0b754390> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f5c08013128> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ba6798488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ba6798488>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ba6798488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:38, 266688.44it/s]  0%|          | 131072/26421880 [00:00<01:20, 325079.98it/s]  2%|▏         | 425984/26421880 [00:00<01:00, 432917.05it/s]  4%|▍         | 1081344/26421880 [00:00<00:42, 597546.21it/s] 13%|█▎        | 3473408/26421880 [00:01<00:27, 839798.73it/s] 28%|██▊       | 7446528/26421880 [00:01<00:15, 1186454.31it/s] 46%|████▌     | 12173312/26421880 [00:01<00:08, 1676886.47it/s] 58%|█████▊    | 15450112/26421880 [00:01<00:04, 2320624.71it/s] 74%|███████▎  | 19431424/26421880 [00:01<00:02, 3216747.90it/s] 90%|█████████ | 23887872/26421880 [00:01<00:00, 4457348.43it/s]26427392it [00:01, 15297285.89it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 95879.77it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 266555.34it/s]  4%|▍         | 172032/4422102 [00:00<00:12, 344336.25it/s] 10%|█         | 458752/4422102 [00:00<00:08, 453685.34it/s] 33%|███▎      | 1458176/4422102 [00:00<00:04, 634158.27it/s] 85%|████████▍ | 3743744/4422102 [00:01<00:00, 888629.61it/s]4423680it [00:01, 4240097.30it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 34047.79it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0037386736671129865 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.024368998169898987 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0027271166642506917 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.027509510040283205 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 3 	 Loss: 0.0020742928584416708 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.022769553303718566 	 Accuracy: 4
Train Epoch: 4 	 Loss: 0.0027469590504964193 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.011925776302814484 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.00210107151667277 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.014358626067638398 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ba6796bf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ba6796bf8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ba6796bf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([2, 7, 2, 0, 2, 2, 5, 2, 2, 9]), array([6, 7, 2, 0, 3, 4, 7, 4, 4, 9]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f5ba6831470> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ba8bc8378>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ba8bc8378>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ba8bc8378> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110197.30it/s]  0%|          | 40960/18165135 [00:00<02:28, 122392.25it/s]  1%|          | 98304/18165135 [00:00<01:57, 153896.07it/s]  1%|          | 212992/18165135 [00:01<01:28, 202483.89it/s]  2%|▏         | 425984/18165135 [00:01<01:05, 272704.36it/s]  5%|▍         | 860160/18165135 [00:01<00:46, 374555.69it/s] 10%|▉         | 1728512/18165135 [00:01<00:31, 520738.94it/s] 19%|█▉        | 3465216/18165135 [00:01<00:20, 729922.58it/s] 34%|███▎      | 6119424/18165135 [00:01<00:11, 1030594.11it/s] 40%|████      | 7299072/18165135 [00:01<00:08, 1354174.95it/s] 46%|████▌     | 8380416/18165135 [00:02<00:05, 1668867.31it/s] 54%|█████▎    | 9748480/18165135 [00:02<00:04, 2061528.53it/s] 67%|██████▋   | 12173312/18165135 [00:02<00:02, 2828807.60it/s] 73%|███████▎  | 13303808/18165135 [00:02<00:01, 3191685.35it/s] 78%|███████▊  | 14237696/18165135 [00:03<00:01, 3886603.17it/s] 86%|████████▌ | 15630336/18165135 [00:03<00:00, 4839740.08it/s] 91%|█████████▏| 16596992/18165135 [00:03<00:00, 5137464.29it/s] 96%|█████████▌| 17448960/18165135 [00:03<00:00, 4891995.72it/s]18169856it [00:03, 5117977.20it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110140.43it/s]32768it [00:00, 49509.82it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110101.96it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122261.12it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153699.19it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 201012.22it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 271375.35it/s] 28%|██▊       | 851968/3041136 [00:01<00:05, 372531.52it/s] 56%|█████▌    | 1703936/3041136 [00:01<00:02, 517709.80it/s]3047424it [00:01, 1953109.54it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17399.97it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.00344781231880188 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.024767687559127807 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0034434856375058494 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.029167978525161745 	 Accuracy: 3
Train Epoch: 3 	 Loss: 0.0035006187160809836 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.05065762376785278 	 Accuracy: 2
Train Epoch: 4 	 Loss: 0.002771424094835917 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.02859197378158569 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.002799211104710897 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.021919716596603394 	 Accuracy: 3

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5ba8bc1ae8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5ba8bc1ae8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f5ba8bc1ae8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([1, 5, 8, 8, 5, 5, 6, 5, 5, 8]), array([6, 2, 8, 2, 0, 3, 0, 8, 5, 5]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

