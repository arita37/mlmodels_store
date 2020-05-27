
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '458b0439a169873cbce08726558e091efacd7d2f', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/458b0439a169873cbce08726558e091efacd7d2f

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f9274980358> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92739b9400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92739b9400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f92739b9400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:01, 161758.73it/s] 59%|█████▊    | 5808128/9912422 [00:00<00:17, 230807.61it/s]9920512it [00:00, 41120108.27it/s]                           
0it [00:00, ?it/s]32768it [00:00, 921537.03it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 2897907.09it/s]          
0it [00:00, ?it/s]8192it [00:00, 137042.63it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0037055222590764362 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.026204327344894408 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.00277428271373113 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.01888710927963257 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.002191435801486174 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.014025117158889771 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 4 	 Loss: 0.0015745282371838888 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.009593263506889342 	 Accuracy: 7
model saves at 7 accuracy
Train Epoch: 5 	 Loss: 0.0017582816630601883 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01053841921687126 	 Accuracy: 6

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92739a4f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92739a4f28>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f92739a4f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([2, 1, 9, 6, 9, 8, 9, 5, 7, 8]), array([2, 1, 5, 6, 9, 9, 9, 5, 2, 2]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f9275831518> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9270d1ab70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9270d1ab70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f9270d1ab70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.00229115358988444 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.018306681871414186 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.002082378149032593 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.05006631088256836 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9270d1a400>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9270d1a400>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f9270d1a400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([7, 6, 2, 3, 2, 3, 3, 7, 2, 3]))

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
  0%|          | 0.00/44.7M [00:00<?, ?B/s]  4%|▎         | 1.62M/44.7M [00:00<00:02, 16.9MB/s]  9%|▉         | 4.00M/44.7M [00:00<00:02, 18.7MB/s] 12%|█▏        | 5.44M/44.7M [00:00<00:02, 17.4MB/s] 15%|█▍        | 6.62M/44.7M [00:00<00:02, 15.5MB/s] 21%|██        | 9.19M/44.7M [00:00<00:02, 17.7MB/s] 24%|██▍       | 10.8M/44.7M [00:00<00:02, 14.3MB/s] 27%|██▋       | 12.2M/44.7M [00:00<00:02, 13.9MB/s] 31%|███       | 13.8M/44.7M [00:00<00:02, 14.7MB/s] 34%|███▍      | 15.2M/44.7M [00:00<00:02, 14.3MB/s] 37%|███▋      | 16.6M/44.7M [00:01<00:02, 11.5MB/s] 43%|████▎     | 19.2M/44.7M [00:01<00:01, 13.9MB/s] 47%|████▋     | 20.9M/44.7M [00:01<00:01, 14.9MB/s] 52%|█████▏    | 23.4M/44.7M [00:01<00:01, 17.0MB/s] 58%|█████▊    | 25.8M/44.7M [00:01<00:01, 18.9MB/s] 63%|██████▎   | 28.1M/44.7M [00:01<00:00, 20.1MB/s] 68%|██████▊   | 30.5M/44.7M [00:01<00:00, 20.1MB/s] 74%|███████▍  | 33.1M/44.7M [00:01<00:00, 21.7MB/s] 79%|███████▉  | 35.5M/44.7M [00:02<00:00, 22.7MB/s] 85%|████████▍ | 37.8M/44.7M [00:02<00:00, 23.1MB/s] 91%|█████████ | 40.5M/44.7M [00:02<00:00, 24.3MB/s] 96%|█████████▌| 42.9M/44.7M [00:02<00:00, 24.7MB/s]100%|██████████| 44.7M/44.7M [00:02<00:00, 19.7MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f92e1b9e390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f92739891e0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f92739891e0>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f92739891e0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:52,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:37,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:25,  5.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:25,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:25,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:17,  7.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:17,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:17,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:17,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:12, 10.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:08, 14.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 18.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 23.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 29.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 29.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 35.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 41.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 47.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 47.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 51.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 54.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 58.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 60.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 62.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 65.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 65.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 64.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 64.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 64.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 64.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.80s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.80s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 64.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  2.80s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 64.87 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.49s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.80s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 64.87 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.49s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.49s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.50 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.49s/ url]
0 examples [00:00, ? examples/s]2020-05-27 00:09:45.467362: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-27 00:09:45.513662: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-27 00:09:45.513981: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bec3eaba80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 00:09:45.514003: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.21 examples/s]89 examples [00:00,  3.15 examples/s]177 examples [00:00,  4.49 examples/s]264 examples [00:00,  6.41 examples/s]354 examples [00:00,  9.12 examples/s]444 examples [00:00, 12.98 examples/s]533 examples [00:01, 18.42 examples/s]620 examples [00:01, 26.08 examples/s]705 examples [00:01, 36.77 examples/s]792 examples [00:01, 51.59 examples/s]883 examples [00:01, 71.94 examples/s]971 examples [00:01, 99.29 examples/s]1060 examples [00:01, 135.36 examples/s]1148 examples [00:01, 181.36 examples/s]1236 examples [00:01, 236.89 examples/s]1328 examples [00:01, 304.72 examples/s]1419 examples [00:02, 380.16 examples/s]1513 examples [00:02, 462.34 examples/s]1606 examples [00:02, 544.43 examples/s]1697 examples [00:02, 611.34 examples/s]1787 examples [00:02, 675.83 examples/s]1877 examples [00:02, 722.06 examples/s]1968 examples [00:02, 769.52 examples/s]2058 examples [00:02, 791.04 examples/s]2146 examples [00:02, 790.30 examples/s]2238 examples [00:02, 823.57 examples/s]2329 examples [00:03, 846.04 examples/s]2417 examples [00:03, 843.95 examples/s]2504 examples [00:03, 823.00 examples/s]2590 examples [00:03, 831.32 examples/s]2682 examples [00:03, 856.00 examples/s]2775 examples [00:03, 874.67 examples/s]2864 examples [00:03, 862.24 examples/s]2951 examples [00:03, 862.22 examples/s]3038 examples [00:03, 857.14 examples/s]3128 examples [00:04, 869.03 examples/s]3225 examples [00:04, 894.74 examples/s]3315 examples [00:04, 879.17 examples/s]3404 examples [00:04, 871.20 examples/s]3494 examples [00:04, 876.43 examples/s]3584 examples [00:04, 881.47 examples/s]3673 examples [00:04, 875.09 examples/s]3761 examples [00:04, 843.89 examples/s]3847 examples [00:04, 845.86 examples/s]3932 examples [00:04, 840.97 examples/s]4021 examples [00:05, 852.48 examples/s]4111 examples [00:05, 864.27 examples/s]4198 examples [00:05, 857.83 examples/s]4284 examples [00:05, 839.74 examples/s]4369 examples [00:05, 838.05 examples/s]4457 examples [00:05, 849.64 examples/s]4543 examples [00:05, 835.04 examples/s]4635 examples [00:05, 856.89 examples/s]4721 examples [00:05, 851.91 examples/s]4809 examples [00:05, 857.58 examples/s]4897 examples [00:06, 862.33 examples/s]4987 examples [00:06, 873.14 examples/s]5075 examples [00:06, 875.13 examples/s]5166 examples [00:06, 885.04 examples/s]5255 examples [00:06, 874.88 examples/s]5343 examples [00:06, 868.88 examples/s]5430 examples [00:06, 847.74 examples/s]5522 examples [00:06, 868.19 examples/s]5610 examples [00:06, 853.66 examples/s]5698 examples [00:07, 860.70 examples/s]5785 examples [00:07, 863.15 examples/s]5874 examples [00:07, 870.12 examples/s]5963 examples [00:07, 872.89 examples/s]6051 examples [00:07, 865.66 examples/s]6143 examples [00:07, 880.73 examples/s]6232 examples [00:07, 876.09 examples/s]6320 examples [00:07, 876.41 examples/s]6410 examples [00:07, 880.66 examples/s]6499 examples [00:07, 838.91 examples/s]6584 examples [00:08, 834.67 examples/s]6673 examples [00:08, 848.15 examples/s]6764 examples [00:08, 863.50 examples/s]6854 examples [00:08, 872.08 examples/s]6947 examples [00:08, 888.14 examples/s]7037 examples [00:08, 890.72 examples/s]7127 examples [00:08, 888.70 examples/s]7217 examples [00:08, 890.24 examples/s]7307 examples [00:08, 891.58 examples/s]7397 examples [00:08, 862.33 examples/s]7490 examples [00:09, 879.78 examples/s]7581 examples [00:09, 887.83 examples/s]7670 examples [00:09, 886.98 examples/s]7759 examples [00:09, 864.74 examples/s]7846 examples [00:09, 845.59 examples/s]7935 examples [00:09, 856.56 examples/s]8021 examples [00:09, 855.83 examples/s]8107 examples [00:09, 841.97 examples/s]8192 examples [00:09, 839.94 examples/s]8283 examples [00:09, 857.52 examples/s]8369 examples [00:10, 853.30 examples/s]8455 examples [00:10, 852.85 examples/s]8545 examples [00:10, 865.35 examples/s]8632 examples [00:10, 836.49 examples/s]8717 examples [00:10, 839.01 examples/s]8806 examples [00:10, 852.05 examples/s]8893 examples [00:10, 857.14 examples/s]8982 examples [00:10, 865.38 examples/s]9069 examples [00:10, 861.74 examples/s]9157 examples [00:11, 865.60 examples/s]9244 examples [00:11, 858.34 examples/s]9333 examples [00:11, 867.14 examples/s]9421 examples [00:11, 869.17 examples/s]9509 examples [00:11, 869.89 examples/s]9598 examples [00:11, 875.66 examples/s]9686 examples [00:11, 871.43 examples/s]9774 examples [00:11, 872.21 examples/s]9862 examples [00:11, 873.34 examples/s]9950 examples [00:11, 859.70 examples/s]10037 examples [00:12, 812.43 examples/s]10125 examples [00:12, 828.99 examples/s]10209 examples [00:12, 826.15 examples/s]10299 examples [00:12, 846.58 examples/s]10387 examples [00:12, 854.76 examples/s]10473 examples [00:12, 854.64 examples/s]10561 examples [00:12, 862.07 examples/s]10648 examples [00:12, 858.67 examples/s]10736 examples [00:12, 861.85 examples/s]10823 examples [00:12, 863.84 examples/s]10913 examples [00:13, 872.60 examples/s]11003 examples [00:13, 880.06 examples/s]11092 examples [00:13, 876.12 examples/s]11181 examples [00:13, 880.24 examples/s]11272 examples [00:13, 888.89 examples/s]11362 examples [00:13, 890.22 examples/s]11454 examples [00:13, 895.59 examples/s]11544 examples [00:13, 890.92 examples/s]11634 examples [00:13, 839.49 examples/s]11719 examples [00:13, 827.23 examples/s]11803 examples [00:14, 811.94 examples/s]11892 examples [00:14, 833.19 examples/s]11983 examples [00:14, 852.72 examples/s]12077 examples [00:14, 875.61 examples/s]12171 examples [00:14, 890.86 examples/s]12263 examples [00:14, 898.16 examples/s]12357 examples [00:14, 908.47 examples/s]12449 examples [00:14, 904.60 examples/s]12540 examples [00:14, 896.92 examples/s]12630 examples [00:15, 895.10 examples/s]12721 examples [00:15, 897.04 examples/s]12811 examples [00:15, 896.92 examples/s]12902 examples [00:15, 900.17 examples/s]12993 examples [00:15, 893.52 examples/s]13083 examples [00:15, 888.41 examples/s]13172 examples [00:15, 887.82 examples/s]13263 examples [00:15, 893.90 examples/s]13358 examples [00:15, 909.23 examples/s]13452 examples [00:15, 915.87 examples/s]13544 examples [00:16, 914.68 examples/s]13637 examples [00:16, 917.18 examples/s]13729 examples [00:16, 906.32 examples/s]13820 examples [00:16, 886.81 examples/s]13911 examples [00:16, 893.43 examples/s]14005 examples [00:16, 905.71 examples/s]14096 examples [00:16, 906.94 examples/s]14187 examples [00:16, 907.11 examples/s]14281 examples [00:16, 916.62 examples/s]14373 examples [00:16, 908.41 examples/s]14464 examples [00:17, 898.49 examples/s]14554 examples [00:17, 873.67 examples/s]14651 examples [00:17, 898.23 examples/s]14742 examples [00:17, 895.35 examples/s]14837 examples [00:17, 909.45 examples/s]14930 examples [00:17, 914.51 examples/s]15022 examples [00:17, 879.12 examples/s]15115 examples [00:17, 892.58 examples/s]15205 examples [00:17, 893.17 examples/s]15295 examples [00:17, 881.15 examples/s]15389 examples [00:18, 896.61 examples/s]15482 examples [00:18, 904.66 examples/s]15574 examples [00:18, 906.76 examples/s]15667 examples [00:18, 912.90 examples/s]15765 examples [00:18, 930.24 examples/s]15859 examples [00:18, 930.49 examples/s]15956 examples [00:18, 941.53 examples/s]16055 examples [00:18, 954.21 examples/s]16152 examples [00:18, 956.79 examples/s]16248 examples [00:18, 935.87 examples/s]16342 examples [00:19, 935.69 examples/s]16436 examples [00:19, 936.80 examples/s]16530 examples [00:19, 918.31 examples/s]16622 examples [00:19, 912.17 examples/s]16722 examples [00:19, 935.65 examples/s]16820 examples [00:19, 948.26 examples/s]16916 examples [00:19, 949.14 examples/s]17012 examples [00:19, 940.62 examples/s]17107 examples [00:19, 936.41 examples/s]17203 examples [00:20, 942.57 examples/s]17298 examples [00:20, 936.91 examples/s]17392 examples [00:20, 909.94 examples/s]17488 examples [00:20, 923.78 examples/s]17581 examples [00:20, 907.03 examples/s]17672 examples [00:20, 903.53 examples/s]17763 examples [00:20, 900.99 examples/s]17857 examples [00:20, 909.56 examples/s]17956 examples [00:20, 931.26 examples/s]18052 examples [00:20, 938.53 examples/s]18152 examples [00:21, 954.78 examples/s]18248 examples [00:21, 946.92 examples/s]18343 examples [00:21, 938.98 examples/s]18438 examples [00:21, 938.98 examples/s]18532 examples [00:21, 919.29 examples/s]18625 examples [00:21, 917.17 examples/s]18717 examples [00:21, 903.02 examples/s]18808 examples [00:21, 888.26 examples/s]18899 examples [00:21, 891.97 examples/s]18993 examples [00:21, 904.46 examples/s]19084 examples [00:22, 892.16 examples/s]19174 examples [00:22, 891.25 examples/s]19265 examples [00:22, 896.54 examples/s]19355 examples [00:22, 894.72 examples/s]19448 examples [00:22, 903.29 examples/s]19540 examples [00:22, 907.16 examples/s]19634 examples [00:22, 916.29 examples/s]19730 examples [00:22, 928.70 examples/s]19825 examples [00:22, 932.21 examples/s]19919 examples [00:22, 925.08 examples/s]20012 examples [00:23, 865.06 examples/s]20104 examples [00:23, 878.76 examples/s]20193 examples [00:23, 857.34 examples/s]20282 examples [00:23, 864.72 examples/s]20380 examples [00:23, 895.14 examples/s]20474 examples [00:23, 907.10 examples/s]20566 examples [00:23, 901.62 examples/s]20658 examples [00:23, 905.86 examples/s]20749 examples [00:23, 895.71 examples/s]20840 examples [00:24, 898.60 examples/s]20930 examples [00:24, 881.42 examples/s]21019 examples [00:24, 828.09 examples/s]21103 examples [00:24, 826.48 examples/s]21191 examples [00:24, 841.01 examples/s]21276 examples [00:24, 830.64 examples/s]21360 examples [00:24, 830.70 examples/s]21448 examples [00:24, 842.06 examples/s]21533 examples [00:24, 823.60 examples/s]21622 examples [00:24, 840.50 examples/s]21710 examples [00:25, 850.54 examples/s]21801 examples [00:25, 865.69 examples/s]21894 examples [00:25, 882.88 examples/s]21983 examples [00:25, 872.24 examples/s]22071 examples [00:25, 870.17 examples/s]22164 examples [00:25, 884.95 examples/s]22256 examples [00:25, 895.18 examples/s]22346 examples [00:25, 889.66 examples/s]22436 examples [00:25, 884.79 examples/s]22525 examples [00:25, 880.17 examples/s]22614 examples [00:26, 873.29 examples/s]22702 examples [00:26, 858.84 examples/s]22788 examples [00:26, 840.88 examples/s]22876 examples [00:26, 851.51 examples/s]22968 examples [00:26, 869.80 examples/s]23060 examples [00:26, 883.92 examples/s]23151 examples [00:26, 891.16 examples/s]23246 examples [00:26, 905.41 examples/s]23337 examples [00:26, 903.52 examples/s]23428 examples [00:27, 896.41 examples/s]23518 examples [00:27, 897.41 examples/s]23608 examples [00:27, 883.96 examples/s]23698 examples [00:27, 886.78 examples/s]23787 examples [00:27, 887.62 examples/s]23876 examples [00:27, 882.80 examples/s]23965 examples [00:27, 881.17 examples/s]24054 examples [00:27, 879.49 examples/s]24143 examples [00:27, 880.41 examples/s]24232 examples [00:27, 872.58 examples/s]24323 examples [00:28, 882.84 examples/s]24418 examples [00:28, 899.56 examples/s]24509 examples [00:28, 901.99 examples/s]24600 examples [00:28, 893.63 examples/s]24690 examples [00:28, 893.49 examples/s]24780 examples [00:28, 874.74 examples/s]24869 examples [00:28, 878.73 examples/s]24960 examples [00:28, 885.88 examples/s]25050 examples [00:28, 887.92 examples/s]25139 examples [00:28, 877.37 examples/s]25233 examples [00:29, 893.16 examples/s]25327 examples [00:29, 904.48 examples/s]25419 examples [00:29, 907.52 examples/s]25510 examples [00:29, 874.96 examples/s]25600 examples [00:29, 880.81 examples/s]25692 examples [00:29, 891.79 examples/s]25786 examples [00:29, 903.34 examples/s]25877 examples [00:29, 887.33 examples/s]25966 examples [00:29, 880.74 examples/s]26055 examples [00:29, 863.54 examples/s]26142 examples [00:30, 846.16 examples/s]26234 examples [00:30, 865.74 examples/s]26321 examples [00:30, 864.89 examples/s]26413 examples [00:30, 879.63 examples/s]26504 examples [00:30, 885.30 examples/s]26593 examples [00:30, 876.30 examples/s]26681 examples [00:30, 865.44 examples/s]26773 examples [00:30, 878.65 examples/s]26863 examples [00:30, 884.47 examples/s]26953 examples [00:31, 887.39 examples/s]27042 examples [00:31, 869.70 examples/s]27130 examples [00:31, 865.13 examples/s]27220 examples [00:31, 873.05 examples/s]27308 examples [00:31, 871.21 examples/s]27396 examples [00:31, 856.66 examples/s]27482 examples [00:31, 847.97 examples/s]27571 examples [00:31, 858.57 examples/s]27663 examples [00:31, 875.01 examples/s]27751 examples [00:31, 867.99 examples/s]27843 examples [00:32, 881.55 examples/s]27932 examples [00:32, 867.57 examples/s]28019 examples [00:32, 825.60 examples/s]28107 examples [00:32, 838.98 examples/s]28192 examples [00:32, 817.70 examples/s]28284 examples [00:32, 843.85 examples/s]28369 examples [00:32, 838.42 examples/s]28459 examples [00:32, 853.81 examples/s]28551 examples [00:32, 870.87 examples/s]28643 examples [00:32, 883.58 examples/s]28734 examples [00:33, 889.83 examples/s]28824 examples [00:33, 864.69 examples/s]28911 examples [00:33, 865.10 examples/s]28998 examples [00:33, 858.32 examples/s]29085 examples [00:33, 861.46 examples/s]29174 examples [00:33, 868.23 examples/s]29261 examples [00:33, 860.42 examples/s]29348 examples [00:33, 807.53 examples/s]29436 examples [00:33, 827.58 examples/s]29523 examples [00:34, 839.85 examples/s]29613 examples [00:34, 856.44 examples/s]29700 examples [00:34, 839.59 examples/s]29788 examples [00:34, 850.00 examples/s]29879 examples [00:34, 865.39 examples/s]29969 examples [00:34, 873.10 examples/s]30057 examples [00:34, 796.69 examples/s]30139 examples [00:34, 802.59 examples/s]30228 examples [00:34, 825.72 examples/s]30317 examples [00:34, 843.25 examples/s]30403 examples [00:35, 842.80 examples/s]30495 examples [00:35, 864.26 examples/s]30582 examples [00:35, 856.05 examples/s]30670 examples [00:35, 862.15 examples/s]30757 examples [00:35, 834.25 examples/s]30850 examples [00:35, 859.31 examples/s]30943 examples [00:35, 878.40 examples/s]31035 examples [00:35, 889.20 examples/s]31126 examples [00:35, 893.71 examples/s]31218 examples [00:35, 899.73 examples/s]31315 examples [00:36, 917.25 examples/s]31410 examples [00:36, 925.29 examples/s]31503 examples [00:36, 891.65 examples/s]31593 examples [00:36, 876.85 examples/s]31682 examples [00:36, 880.63 examples/s]31771 examples [00:36, 870.18 examples/s]31859 examples [00:36, 858.79 examples/s]31946 examples [00:36, 851.49 examples/s]32033 examples [00:36, 855.21 examples/s]32119 examples [00:37, 817.49 examples/s]32202 examples [00:37, 816.40 examples/s]32289 examples [00:37, 829.93 examples/s]32373 examples [00:37, 808.54 examples/s]32458 examples [00:37, 818.69 examples/s]32551 examples [00:37, 847.11 examples/s]32645 examples [00:37, 870.96 examples/s]32738 examples [00:37, 887.68 examples/s]32828 examples [00:37, 884.81 examples/s]32917 examples [00:37, 882.95 examples/s]33010 examples [00:38, 894.42 examples/s]33106 examples [00:38, 912.83 examples/s]33198 examples [00:38, 913.55 examples/s]33290 examples [00:38, 898.21 examples/s]33380 examples [00:38, 876.67 examples/s]33468 examples [00:38, 849.78 examples/s]33555 examples [00:38, 853.13 examples/s]33643 examples [00:38, 860.48 examples/s]33730 examples [00:38, 851.02 examples/s]33819 examples [00:38, 860.38 examples/s]33909 examples [00:39, 871.19 examples/s]33998 examples [00:39, 875.76 examples/s]34086 examples [00:39, 875.29 examples/s]34174 examples [00:39, 872.90 examples/s]34267 examples [00:39, 889.04 examples/s]34360 examples [00:39, 898.46 examples/s]34456 examples [00:39, 913.73 examples/s]34548 examples [00:39, 901.25 examples/s]34639 examples [00:39, 889.64 examples/s]34729 examples [00:40, 882.19 examples/s]34818 examples [00:40, 881.14 examples/s]34907 examples [00:40, 879.68 examples/s]34999 examples [00:40, 889.16 examples/s]35088 examples [00:40, 860.43 examples/s]35179 examples [00:40, 871.51 examples/s]35272 examples [00:40, 886.62 examples/s]35361 examples [00:40, 882.43 examples/s]35453 examples [00:40, 892.69 examples/s]35543 examples [00:40, 879.85 examples/s]35634 examples [00:41, 887.25 examples/s]35725 examples [00:41, 893.95 examples/s]35818 examples [00:41, 901.94 examples/s]35909 examples [00:41, 887.10 examples/s]36000 examples [00:41, 891.71 examples/s]36090 examples [00:41, 872.13 examples/s]36178 examples [00:41, 872.12 examples/s]36267 examples [00:41, 876.15 examples/s]36355 examples [00:41, 876.48 examples/s]36443 examples [00:41, 842.19 examples/s]36531 examples [00:42, 850.82 examples/s]36621 examples [00:42, 863.64 examples/s]36713 examples [00:42, 878.82 examples/s]36803 examples [00:42, 882.77 examples/s]36892 examples [00:42, 860.26 examples/s]36983 examples [00:42, 874.47 examples/s]37071 examples [00:42, 868.34 examples/s]37160 examples [00:42, 874.02 examples/s]37252 examples [00:42, 886.20 examples/s]37341 examples [00:42, 870.76 examples/s]37439 examples [00:43, 899.03 examples/s]37534 examples [00:43, 911.73 examples/s]37626 examples [00:43, 913.46 examples/s]37718 examples [00:43, 904.79 examples/s]37809 examples [00:43, 879.96 examples/s]37903 examples [00:43, 896.61 examples/s]37993 examples [00:43, 896.67 examples/s]38087 examples [00:43, 904.58 examples/s]38178 examples [00:43, 904.74 examples/s]38269 examples [00:44, 902.56 examples/s]38365 examples [00:44, 917.44 examples/s]38462 examples [00:44, 931.65 examples/s]38556 examples [00:44, 934.09 examples/s]38650 examples [00:44, 932.93 examples/s]38744 examples [00:44, 919.98 examples/s]38837 examples [00:44, 898.52 examples/s]38935 examples [00:44, 921.35 examples/s]39031 examples [00:44, 930.24 examples/s]39128 examples [00:44, 940.87 examples/s]39223 examples [00:45, 916.98 examples/s]39315 examples [00:45, 913.45 examples/s]39410 examples [00:45, 921.76 examples/s]39503 examples [00:45, 914.20 examples/s]39599 examples [00:45, 924.78 examples/s]39692 examples [00:45, 895.79 examples/s]39782 examples [00:45, 886.98 examples/s]39873 examples [00:45, 893.51 examples/s]39965 examples [00:45, 899.07 examples/s]40056 examples [00:45, 844.53 examples/s]40142 examples [00:46, 829.52 examples/s]40233 examples [00:46, 850.73 examples/s]40322 examples [00:46, 860.58 examples/s]40410 examples [00:46, 865.14 examples/s]40502 examples [00:46, 878.29 examples/s]40591 examples [00:46, 873.71 examples/s]40679 examples [00:46, 863.53 examples/s]40766 examples [00:46, 857.19 examples/s]40852 examples [00:46, 851.78 examples/s]40938 examples [00:47, 838.39 examples/s]41022 examples [00:47, 836.74 examples/s]41106 examples [00:47, 831.23 examples/s]41191 examples [00:47, 834.27 examples/s]41278 examples [00:47, 842.74 examples/s]41370 examples [00:47, 864.25 examples/s]41462 examples [00:47, 878.12 examples/s]41550 examples [00:47, 862.80 examples/s]41643 examples [00:47, 879.64 examples/s]41735 examples [00:47, 887.13 examples/s]41824 examples [00:48, 862.62 examples/s]41915 examples [00:48, 873.93 examples/s]42003 examples [00:48, 875.15 examples/s]42094 examples [00:48, 884.49 examples/s]42187 examples [00:48, 897.24 examples/s]42277 examples [00:48, 897.70 examples/s]42367 examples [00:48, 895.68 examples/s]42457 examples [00:48, 863.78 examples/s]42549 examples [00:48, 879.64 examples/s]42638 examples [00:48, 877.47 examples/s]42726 examples [00:49, 853.16 examples/s]42818 examples [00:49, 870.27 examples/s]42906 examples [00:49, 871.18 examples/s]42994 examples [00:49, 857.49 examples/s]43084 examples [00:49, 869.00 examples/s]43172 examples [00:49, 859.18 examples/s]43263 examples [00:49, 871.61 examples/s]43357 examples [00:49, 888.28 examples/s]43449 examples [00:49, 896.19 examples/s]43539 examples [00:49, 893.93 examples/s]43629 examples [00:50, 895.41 examples/s]43720 examples [00:50, 898.56 examples/s]43814 examples [00:50, 908.62 examples/s]43905 examples [00:50, 900.80 examples/s]44000 examples [00:50, 914.65 examples/s]44092 examples [00:50, 913.55 examples/s]44184 examples [00:50, 892.94 examples/s]44277 examples [00:50, 900.65 examples/s]44368 examples [00:50, 901.50 examples/s]44466 examples [00:50, 922.06 examples/s]44561 examples [00:51, 929.34 examples/s]44655 examples [00:51, 929.45 examples/s]44750 examples [00:51, 933.13 examples/s]44844 examples [00:51, 928.86 examples/s]44937 examples [00:51, 925.32 examples/s]45036 examples [00:51, 942.92 examples/s]45131 examples [00:51, 936.86 examples/s]45227 examples [00:51, 943.31 examples/s]45327 examples [00:51, 957.16 examples/s]45424 examples [00:52, 957.96 examples/s]45520 examples [00:52, 947.95 examples/s]45615 examples [00:52, 938.53 examples/s]45709 examples [00:52, 911.36 examples/s]45801 examples [00:52, 896.59 examples/s]45893 examples [00:52, 902.06 examples/s]45984 examples [00:52, 887.74 examples/s]46074 examples [00:52, 889.09 examples/s]46166 examples [00:52, 898.12 examples/s]46260 examples [00:52, 907.97 examples/s]46351 examples [00:53, 904.46 examples/s]46444 examples [00:53, 911.48 examples/s]46538 examples [00:53, 917.78 examples/s]46632 examples [00:53, 921.95 examples/s]46725 examples [00:53, 922.93 examples/s]46818 examples [00:53, 911.83 examples/s]46910 examples [00:53, 910.68 examples/s]47002 examples [00:53, 898.44 examples/s]47092 examples [00:53, 887.58 examples/s]47181 examples [00:53, 866.60 examples/s]47270 examples [00:54, 872.86 examples/s]47358 examples [00:54, 869.73 examples/s]47449 examples [00:54, 881.13 examples/s]47538 examples [00:54, 880.22 examples/s]47632 examples [00:54, 896.81 examples/s]47723 examples [00:54, 899.44 examples/s]47814 examples [00:54, 895.74 examples/s]47907 examples [00:54, 903.38 examples/s]48000 examples [00:54, 909.68 examples/s]48093 examples [00:54, 915.59 examples/s]48185 examples [00:55, 912.23 examples/s]48277 examples [00:55, 899.94 examples/s]48368 examples [00:55, 847.51 examples/s]48454 examples [00:55, 846.61 examples/s]48543 examples [00:55, 856.73 examples/s]48630 examples [00:55, 855.04 examples/s]48718 examples [00:55, 862.37 examples/s]48805 examples [00:55, 859.76 examples/s]48897 examples [00:55, 876.48 examples/s]48987 examples [00:56, 883.38 examples/s]49076 examples [00:56, 882.34 examples/s]49170 examples [00:56, 897.19 examples/s]49265 examples [00:56, 909.92 examples/s]49361 examples [00:56, 922.18 examples/s]49454 examples [00:56, 922.39 examples/s]49547 examples [00:56, 904.96 examples/s]49638 examples [00:56, 863.73 examples/s]49726 examples [00:56, 866.97 examples/s]49814 examples [00:56, 857.91 examples/s]49907 examples [00:57, 878.06 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 5948/50000 [00:00<00:00, 59479.63 examples/s] 34%|███▍      | 17020/50000 [00:00<00:00, 69068.90 examples/s] 57%|█████▋    | 28390/50000 [00:00<00:00, 78287.41 examples/s] 80%|███████▉  | 39912/50000 [00:00<00:00, 86614.27 examples/s]                                                               0 examples [00:00, ? examples/s]70 examples [00:00, 699.82 examples/s]163 examples [00:00, 754.26 examples/s]252 examples [00:00, 788.82 examples/s]346 examples [00:00, 826.93 examples/s]434 examples [00:00, 841.37 examples/s]524 examples [00:00, 857.68 examples/s]612 examples [00:00, 863.06 examples/s]698 examples [00:00, 859.59 examples/s]791 examples [00:00, 878.98 examples/s]882 examples [00:01, 886.72 examples/s]974 examples [00:01, 893.85 examples/s]1066 examples [00:01, 900.86 examples/s]1156 examples [00:01, 899.96 examples/s]1250 examples [00:01, 910.60 examples/s]1344 examples [00:01, 918.69 examples/s]1436 examples [00:01, 917.59 examples/s]1528 examples [00:01, 885.06 examples/s]1617 examples [00:01, 877.08 examples/s]1709 examples [00:01, 887.46 examples/s]1805 examples [00:02, 907.09 examples/s]1897 examples [00:02, 908.54 examples/s]1989 examples [00:02, 907.64 examples/s]2080 examples [00:02, 898.61 examples/s]2172 examples [00:02, 904.77 examples/s]2264 examples [00:02, 908.33 examples/s]2356 examples [00:02, 911.64 examples/s]2449 examples [00:02, 914.84 examples/s]2541 examples [00:02, 906.03 examples/s]2635 examples [00:02, 914.47 examples/s]2727 examples [00:03, 909.59 examples/s]2819 examples [00:03, 912.22 examples/s]2912 examples [00:03, 916.17 examples/s]3004 examples [00:03, 911.28 examples/s]3096 examples [00:03, 900.12 examples/s]3188 examples [00:03, 904.85 examples/s]3282 examples [00:03, 913.68 examples/s]3376 examples [00:03, 920.75 examples/s]3469 examples [00:03, 919.98 examples/s]3564 examples [00:03, 926.48 examples/s]3657 examples [00:04, 924.82 examples/s]3751 examples [00:04, 928.45 examples/s]3844 examples [00:04, 925.22 examples/s]3941 examples [00:04, 935.78 examples/s]4039 examples [00:04, 946.70 examples/s]4134 examples [00:04, 946.05 examples/s]4229 examples [00:04, 943.26 examples/s]4324 examples [00:04, 928.82 examples/s]4417 examples [00:04, 925.17 examples/s]4516 examples [00:04, 943.46 examples/s]4611 examples [00:05, 928.56 examples/s]4704 examples [00:05, 926.59 examples/s]4797 examples [00:05, 913.91 examples/s]4889 examples [00:05, 882.64 examples/s]4985 examples [00:05, 904.38 examples/s]5081 examples [00:05, 920.29 examples/s]5174 examples [00:05, 916.76 examples/s]5266 examples [00:05, 897.95 examples/s]5357 examples [00:05, 893.54 examples/s]5454 examples [00:06, 913.70 examples/s]5548 examples [00:06, 920.98 examples/s]5641 examples [00:06, 917.69 examples/s]5738 examples [00:06, 932.04 examples/s]5833 examples [00:06, 935.78 examples/s]5930 examples [00:06, 943.27 examples/s]6028 examples [00:06, 951.84 examples/s]6124 examples [00:06, 946.41 examples/s]6219 examples [00:06, 921.75 examples/s]6312 examples [00:06, 904.44 examples/s]6403 examples [00:07, 897.87 examples/s]6493 examples [00:07, 882.37 examples/s]6583 examples [00:07, 885.39 examples/s]6677 examples [00:07, 900.14 examples/s]6770 examples [00:07, 908.17 examples/s]6866 examples [00:07, 920.17 examples/s]6960 examples [00:07, 925.33 examples/s]7053 examples [00:07, 915.33 examples/s]7145 examples [00:07, 902.17 examples/s]7236 examples [00:07, 897.43 examples/s]7326 examples [00:08, 888.80 examples/s]7417 examples [00:08, 892.34 examples/s]7511 examples [00:08, 904.63 examples/s]7602 examples [00:08, 903.14 examples/s]7693 examples [00:08, 893.27 examples/s]7786 examples [00:08, 903.41 examples/s]7877 examples [00:08, 887.61 examples/s]7966 examples [00:08, 885.92 examples/s]8058 examples [00:08, 895.88 examples/s]8151 examples [00:08, 903.02 examples/s]8242 examples [00:09, 896.91 examples/s]8333 examples [00:09, 898.74 examples/s]8423 examples [00:09, 889.59 examples/s]8513 examples [00:09, 891.77 examples/s]8605 examples [00:09, 899.51 examples/s]8695 examples [00:09, 894.69 examples/s]8790 examples [00:09, 908.08 examples/s]8881 examples [00:09, 903.63 examples/s]8972 examples [00:09, 892.99 examples/s]9062 examples [00:09, 890.84 examples/s]9154 examples [00:10, 897.63 examples/s]9244 examples [00:10, 888.86 examples/s]9333 examples [00:10, 878.31 examples/s]9426 examples [00:10, 893.08 examples/s]9518 examples [00:10, 898.17 examples/s]9612 examples [00:10, 907.66 examples/s]9703 examples [00:10, 905.25 examples/s]9794 examples [00:10, 901.83 examples/s]9885 examples [00:10, 900.88 examples/s]9981 examples [00:11, 914.63 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete25FKMA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete25FKMA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f925678ee18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f925678ee18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f925678ee18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.475827693939209 	 Accuracy: 48
Train Epoch: 1 	 Loss: 6.701937484741211 	 Accuracy: 160
model saves at 160 accuracy
Train Epoch: 2 	 Loss: 0.47189024209976194 	 Accuracy: 50
Train Epoch: 2 	 Loss: 16.69913330078125 	 Accuracy: 100

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f920f77ee18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f920f77ee18>
>>>>input_tmp:  None

  function with postional parmater data_info <function tf_dataset_download at 0x7f920f77ee18> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f9213fca950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f9213fca950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f9213fca950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([8, 9, 1, 4, 1, 1, 8, 9, 1, 8]), array([0, 9, 0, 4, 1, 3, 0, 0, 2, 7]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f9274980358> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f9213fec860> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f920ae0c6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f920ae0c6a8>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f920ae0c6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/26421880 [00:00<?, ?it/s]  0%|          | 49152/26421880 [00:00<01:37, 269491.31it/s]  0%|          | 131072/26421880 [00:00<01:19, 328822.03it/s]  2%|▏         | 425984/26421880 [00:00<00:59, 437628.86it/s]  4%|▍         | 1081344/26421880 [00:00<00:41, 604227.36it/s] 12%|█▏        | 3211264/26421880 [00:00<00:27, 852811.59it/s] 22%|██▏       | 5718016/26421880 [00:01<00:17, 1200793.70it/s] 35%|███▍      | 9199616/26421880 [00:01<00:10, 1690169.43it/s] 48%|████▊     | 12582912/26421880 [00:01<00:05, 2350372.52it/s] 63%|██████▎   | 16646144/26421880 [00:01<00:02, 3258811.89it/s] 76%|███████▋  | 20160512/26421880 [00:01<00:01, 4477463.33it/s] 90%|█████████ | 23887872/26421880 [00:01<00:00, 6081481.33it/s]26427392it [00:01, 15303894.46it/s]                             
0it [00:00, ?it/s]  0%|          | 0/29515 [00:00<?, ?it/s]32768it [00:00, 97178.28it/s]            
0it [00:00, ?it/s]  0%|          | 0/4422102 [00:00<?, ?it/s]  1%|          | 49152/4422102 [00:00<00:16, 269709.55it/s]  5%|▍         | 212992/4422102 [00:00<00:11, 352794.97it/s] 11%|█         | 466944/4422102 [00:00<00:08, 464314.28it/s] 40%|████      | 1777664/4422102 [00:00<00:04, 650433.13it/s] 87%|████████▋ | 3833856/4422102 [00:00<00:00, 911465.16it/s]4423680it [00:00, 4456093.23it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5148 [00:00<?, ?it/s]8192it [00:00, 34002.78it/s]            Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/FashionMNIST/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/FashionMNIST/FashionMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.0034828127225240073 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.024749143719673156 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0031765136122703552 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.019475912392139434 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.0023249970277150474 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.028967841982841493 	 Accuracy: 3
Train Epoch: 4 	 Loss: 0.0023805243571599325 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.010162649273872375 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.002382439374923706 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.010559604108333587 	 Accuracy: 5

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:FashionMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f920ae0ae18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f920ae0ae18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f920ae0ae18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.FashionMNIST'> 
(array([3, 1, 6, 2, 0, 8, 7, 5, 5, 5]), array([0, 1, 4, 4, 0, 8, 7, 5, 7, 7]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f9213fef898> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f920ae08598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f920ae08598>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f920ae08598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 

Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
0it [00:00, ?it/s]  0%|          | 0/18165135 [00:00<?, ?it/s]  0%|          | 16384/18165135 [00:00<02:44, 110091.20it/s]  0%|          | 40960/18165135 [00:00<02:28, 122231.48it/s]  1%|          | 98304/18165135 [00:01<01:57, 153700.49it/s]  1%|          | 204800/18165135 [00:01<01:29, 201050.47it/s]  2%|▏         | 417792/18165135 [00:01<01:05, 270889.15it/s]  5%|▍         | 843776/18165135 [00:01<00:46, 371861.87it/s]  9%|▉         | 1703936/18165135 [00:01<00:31, 516940.11it/s] 19%|█▉        | 3416064/18165135 [00:01<00:20, 724515.06it/s] 31%|███       | 5619712/18165135 [00:02<00:12, 1001500.98it/s] 40%|███▉      | 7225344/18165135 [00:02<00:07, 1393442.83it/s] 45%|████▍     | 8167424/18165135 [00:02<00:05, 1756723.12it/s] 49%|████▉     | 8962048/18165135 [00:02<00:04, 1958486.83it/s] 60%|██████    | 10936320/18165135 [00:02<00:02, 2683593.96it/s] 66%|██████▌   | 11976704/18165135 [00:03<00:02, 2887484.71it/s] 77%|███████▋  | 14041088/18165135 [00:03<00:01, 3805189.46it/s] 84%|████████▎ | 15179776/18165135 [00:03<00:00, 4693260.98it/s] 89%|████████▉ | 16211968/18165135 [00:03<00:00, 5436254.84it/s] 95%|█████████▍| 17195008/18165135 [00:03<00:00, 5676798.94it/s]100%|█████████▉| 18112512/18165135 [00:03<00:00, 6403137.33it/s]18169856it [00:03, 4927497.27it/s]                              
0it [00:00, ?it/s]  0%|          | 0/29497 [00:00<?, ?it/s] 56%|█████▌    | 16384/29497 [00:00<00:00, 110241.67it/s]32768it [00:00, 42710.47it/s]                            
0it [00:00, ?it/s]  0%|          | 0/3041136 [00:00<?, ?it/s]  1%|          | 16384/3041136 [00:00<00:27, 110209.14it/s]  1%|▏         | 40960/3041136 [00:00<00:24, 122399.55it/s]  3%|▎         | 98304/3041136 [00:00<00:19, 153859.23it/s]  7%|▋         | 204800/3041136 [00:01<00:14, 201224.70it/s] 14%|█▍        | 425984/3041136 [00:01<00:09, 271646.05it/s] 28%|██▊       | 860160/3041136 [00:01<00:05, 373144.74it/s] 57%|█████▋    | 1720320/3041136 [00:01<00:02, 518672.40it/s] 94%|█████████▍| 2867200/3041136 [00:01<00:00, 720180.41it/s]3047424it [00:01, 1839905.38it/s]                            
0it [00:00, ?it/s]  0%|          | 0/5120 [00:00<?, ?it/s]8192it [00:00, 17573.35it/s]            Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Downloading http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/KMNIST/KMNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/KMNIST/KMNIST/raw
Processing...
Done!
Train Epoch: 1 	 Loss: 0.004183369100093841 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.022910103678703306 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0036478867530822753 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.045354002237319946 	 Accuracy: 2
Train Epoch: 3 	 Loss: 0.0032714315255482993 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.021998926043510437 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 4 	 Loss: 0.0022722687224547068 	 Accuracy: 0
Train Epoch: 4 	 Loss: 0.021658146500587463 	 Accuracy: 3
Train Epoch: 5 	 Loss: 0.002417870799700419 	 Accuracy: 0
Train Epoch: 5 	 Loss: 0.02166904890537262 	 Accuracy: 5
model saves at 5 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:KMNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True, '_comment': "  lasses = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo'] "} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f920ae07d08>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f920ae07d08>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f920ae07d08> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.KMNIST'> 
(array([9, 6, 1, 1, 1, 7, 5, 3, 6, 8]), array([9, 4, 2, 1, 3, 2, 5, 9, 6, 0]))

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

