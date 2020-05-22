
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '51b64e342c7b2661e79b8abaa33db92672ae95c7', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
['/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//model_list_CIFAR.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//charcnn_zhang.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//namentity_crm_bilstm_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_textcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json']





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/charcnn.json 

#####  Load JSON data_pars
{
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels/model_keras/raw/char_cnn/data_utils.py::Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
      }
    }
  ]
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

 #####  get_Data DataLoader 
((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/charcnn_zhang.json 

#####  Load JSON data_pars
{
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::pandasDataset",
      "args": {
        "colX": [
          "colX"
        ],
        "coly": [
          "coly"
        ],
        "encoding": "'ISO-8859-1'",
        "read_csv_parm": {
          "usecols": [
            0,
            1
          ],
          "names": [
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels/model_keras/raw/char_cnn/data_utils.py::Data",
      "args": {
        "data_source": "",
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
      }
    }
  ]
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

 #####  get_Data DataLoader 
((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 

#####  Load JSON data_pars
{
  "data_info": {
    "data_path": "mlmodels/dataset/vision/MNIST",
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
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels.preprocess.generic::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e996e1488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e996e1488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e996e1488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Using TensorFlow backend.
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 132224.55it/s] 84%|████████▍ | 8339456/9912422 [00:00<00:08, 188759.99it/s]9920512it [00:00, 41603153.07it/s]                           
0it [00:00, ?it/s]32768it [00:00, 578746.38it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154195.02it/s]1654784it [00:00, 9749675.37it/s]                          
0it [00:00, ?it/s]8192it [00:00, 200400.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Processing...
Done!

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f0e828fc940>, <torch.utils.data.dataloader.DataLoader object at 0x7f0e82908978>), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 

#####  Load JSON data_pars
{
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
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0e828fe9d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0e828fe9d8>

  function with postional parmater data_info <function tf_dataset_download at 0x7f0e828fe9d8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:17,  2.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:17,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:17,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:16,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:16,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:15,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:15,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:14,  2.08 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:52,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:52,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:51,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:51,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:51,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:50,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:50,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:50,  2.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:35,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:22,  5.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.95 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:15,  7.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.61 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 64.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 68.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
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
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 72.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.26 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.67s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.53s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.26 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.67s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.67s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.70 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.67s/ url]
0 examples [00:00, ? examples/s]2020-05-22 00:08:33.112517: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 00:08:33.116502: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 00:08:33.116644: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557084788560 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 00:08:33.116660: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
66 examples [00:00, 659.06 examples/s]173 examples [00:00, 744.50 examples/s]267 examples [00:00, 792.53 examples/s]367 examples [00:00, 843.31 examples/s]479 examples [00:00, 908.76 examples/s]577 examples [00:00, 926.44 examples/s]666 examples [00:00, 913.24 examples/s]769 examples [00:00, 944.26 examples/s]862 examples [00:00, 926.70 examples/s]955 examples [00:01, 927.44 examples/s]1047 examples [00:01, 910.32 examples/s]1148 examples [00:01, 936.48 examples/s]1254 examples [00:01, 969.97 examples/s]1352 examples [00:01, 895.73 examples/s]1456 examples [00:01, 933.26 examples/s]1551 examples [00:01, 924.22 examples/s]1648 examples [00:01, 936.27 examples/s]1746 examples [00:01, 946.86 examples/s]1842 examples [00:01, 942.31 examples/s]1942 examples [00:02, 958.43 examples/s]2047 examples [00:02, 982.99 examples/s]2152 examples [00:02, 1000.20 examples/s]2253 examples [00:02, 995.62 examples/s] 2353 examples [00:02, 994.00 examples/s]2453 examples [00:02, 985.14 examples/s]2552 examples [00:02, 985.76 examples/s]2653 examples [00:02, 991.45 examples/s]2753 examples [00:02, 972.29 examples/s]2857 examples [00:02, 986.44 examples/s]2963 examples [00:03, 1005.63 examples/s]3070 examples [00:03, 1021.68 examples/s]3175 examples [00:03, 1028.74 examples/s]3279 examples [00:03, 999.27 examples/s] 3380 examples [00:03, 989.60 examples/s]3481 examples [00:03, 993.14 examples/s]3581 examples [00:03, 979.22 examples/s]3686 examples [00:03, 998.03 examples/s]3786 examples [00:03, 993.03 examples/s]3886 examples [00:04, 984.21 examples/s]3989 examples [00:04, 995.54 examples/s]4092 examples [00:04, 1005.21 examples/s]4196 examples [00:04, 1013.59 examples/s]4298 examples [00:04, 998.35 examples/s] 4398 examples [00:04, 983.06 examples/s]4497 examples [00:04, 976.32 examples/s]4596 examples [00:04, 979.14 examples/s]4700 examples [00:04, 995.81 examples/s]4802 examples [00:04, 1000.19 examples/s]4903 examples [00:05, 1001.14 examples/s]5010 examples [00:05, 1018.22 examples/s]5112 examples [00:05, 1017.60 examples/s]5214 examples [00:05, 989.12 examples/s] 5316 examples [00:05, 995.94 examples/s]5418 examples [00:05, 1001.49 examples/s]5521 examples [00:05, 1008.14 examples/s]5624 examples [00:05, 1013.49 examples/s]5727 examples [00:05, 1016.09 examples/s]5829 examples [00:05, 987.62 examples/s] 5930 examples [00:06, 991.22 examples/s]6030 examples [00:06, 989.46 examples/s]6139 examples [00:06, 1015.67 examples/s]6250 examples [00:06, 1041.31 examples/s]6361 examples [00:06, 1060.73 examples/s]6468 examples [00:06, 1062.66 examples/s]6575 examples [00:06, 1043.91 examples/s]6680 examples [00:06, 1039.27 examples/s]6786 examples [00:06, 1044.46 examples/s]6891 examples [00:06, 1020.25 examples/s]6994 examples [00:07, 976.97 examples/s] 7108 examples [00:07, 1019.50 examples/s]7215 examples [00:07, 1030.82 examples/s]7319 examples [00:07, 1030.09 examples/s]7424 examples [00:07, 1035.34 examples/s]7533 examples [00:07, 1049.90 examples/s]7642 examples [00:07, 1060.99 examples/s]7754 examples [00:07, 1075.47 examples/s]7866 examples [00:07, 1086.74 examples/s]7975 examples [00:08, 1043.48 examples/s]8080 examples [00:08, 1029.74 examples/s]8188 examples [00:08, 1041.96 examples/s]8295 examples [00:08, 1050.08 examples/s]8406 examples [00:08, 1064.64 examples/s]8513 examples [00:08, 1048.70 examples/s]8619 examples [00:08, 990.57 examples/s] 8731 examples [00:08, 1024.34 examples/s]8845 examples [00:08, 1054.92 examples/s]8952 examples [00:08, 1042.69 examples/s]9057 examples [00:09, 1018.13 examples/s]9167 examples [00:09, 1039.24 examples/s]9277 examples [00:09, 1055.48 examples/s]9390 examples [00:09, 1075.51 examples/s]9502 examples [00:09, 1088.15 examples/s]9612 examples [00:09, 1058.35 examples/s]9719 examples [00:09, 1039.05 examples/s]9824 examples [00:09, 1029.00 examples/s]9931 examples [00:09, 1039.19 examples/s]10036 examples [00:10, 953.99 examples/s]10150 examples [00:10, 1001.35 examples/s]10258 examples [00:10, 1021.40 examples/s]10362 examples [00:10, 980.43 examples/s] 10471 examples [00:10, 1008.08 examples/s]10580 examples [00:10, 1029.64 examples/s]10691 examples [00:10, 1050.73 examples/s]10803 examples [00:10, 1070.12 examples/s]10913 examples [00:10, 1078.33 examples/s]11022 examples [00:10, 1060.42 examples/s]11129 examples [00:11, 1051.87 examples/s]11237 examples [00:11, 1057.48 examples/s]11345 examples [00:11, 1061.73 examples/s]11455 examples [00:11, 1072.74 examples/s]11566 examples [00:11, 1080.75 examples/s]11675 examples [00:11, 1082.97 examples/s]11784 examples [00:11, 1070.29 examples/s]11894 examples [00:11, 1077.23 examples/s]12008 examples [00:11, 1093.14 examples/s]12118 examples [00:11, 1084.45 examples/s]12227 examples [00:12, 1059.82 examples/s]12338 examples [00:12, 1072.71 examples/s]12449 examples [00:12, 1082.23 examples/s]12564 examples [00:12, 1099.25 examples/s]12678 examples [00:12, 1108.60 examples/s]12791 examples [00:12, 1111.75 examples/s]12903 examples [00:12, 1102.71 examples/s]13014 examples [00:12, 1090.30 examples/s]13126 examples [00:12, 1098.16 examples/s]13241 examples [00:12, 1112.88 examples/s]13353 examples [00:13, 1093.44 examples/s]13463 examples [00:13, 1064.15 examples/s]13576 examples [00:13, 1080.34 examples/s]13689 examples [00:13, 1094.59 examples/s]13799 examples [00:13, 1092.46 examples/s]13909 examples [00:13, 1062.98 examples/s]14016 examples [00:13, 1054.60 examples/s]14128 examples [00:13, 1073.07 examples/s]14236 examples [00:13, 1075.05 examples/s]14344 examples [00:14, 1069.36 examples/s]14452 examples [00:14, 1065.28 examples/s]14559 examples [00:14, 1055.30 examples/s]14675 examples [00:14, 1083.16 examples/s]14788 examples [00:14, 1096.13 examples/s]14902 examples [00:14, 1108.40 examples/s]15014 examples [00:14, 1082.70 examples/s]15123 examples [00:14, 993.64 examples/s] 15224 examples [00:14, 973.17 examples/s]15326 examples [00:14, 983.70 examples/s]15430 examples [00:15, 997.72 examples/s]15531 examples [00:15, 980.02 examples/s]15630 examples [00:15, 940.36 examples/s]15725 examples [00:15, 934.39 examples/s]15833 examples [00:15, 970.90 examples/s]15941 examples [00:15, 998.96 examples/s]16042 examples [00:15, 991.85 examples/s]16144 examples [00:15, 998.27 examples/s]16253 examples [00:15, 1021.69 examples/s]16356 examples [00:15, 1014.80 examples/s]16461 examples [00:16, 1021.09 examples/s]16564 examples [00:16, 1011.90 examples/s]16673 examples [00:16, 1033.60 examples/s]16777 examples [00:16, 1028.99 examples/s]16887 examples [00:16, 1048.07 examples/s]17001 examples [00:16, 1073.09 examples/s]17109 examples [00:16, 1062.24 examples/s]17222 examples [00:16, 1079.72 examples/s]17331 examples [00:16, 1041.19 examples/s]17438 examples [00:17, 1047.92 examples/s]17548 examples [00:17, 1060.57 examples/s]17658 examples [00:17, 1070.59 examples/s]17772 examples [00:17, 1090.03 examples/s]17887 examples [00:17, 1105.96 examples/s]17998 examples [00:17, 1086.03 examples/s]18107 examples [00:17, 1064.57 examples/s]18217 examples [00:17, 1073.25 examples/s]18326 examples [00:17, 1077.49 examples/s]18435 examples [00:17, 1080.55 examples/s]18546 examples [00:18, 1088.91 examples/s]18655 examples [00:18, 1081.13 examples/s]18764 examples [00:18, 1077.98 examples/s]18876 examples [00:18, 1088.77 examples/s]18985 examples [00:18, 1087.33 examples/s]19094 examples [00:18, 1058.31 examples/s]19201 examples [00:18, 1035.58 examples/s]19312 examples [00:18, 1055.84 examples/s]19420 examples [00:18, 1060.54 examples/s]19528 examples [00:18, 1064.42 examples/s]19635 examples [00:19, 1051.36 examples/s]19741 examples [00:19, 1013.16 examples/s]19849 examples [00:19, 1031.19 examples/s]19959 examples [00:19, 1050.17 examples/s]20065 examples [00:19, 975.12 examples/s] 20164 examples [00:19, 970.05 examples/s]20264 examples [00:19, 977.07 examples/s]20374 examples [00:19, 1009.16 examples/s]20489 examples [00:19, 1045.56 examples/s]20602 examples [00:20, 1067.88 examples/s]20710 examples [00:20, 1068.36 examples/s]20818 examples [00:20, 1025.96 examples/s]20923 examples [00:20, 1030.98 examples/s]21033 examples [00:20, 1049.11 examples/s]21145 examples [00:20, 1067.69 examples/s]21256 examples [00:20, 1077.99 examples/s]21365 examples [00:20, 1064.31 examples/s]21474 examples [00:20, 1071.55 examples/s]21582 examples [00:20, 1071.38 examples/s]21692 examples [00:21, 1077.28 examples/s]21800 examples [00:21, 1076.07 examples/s]21908 examples [00:21, 1048.76 examples/s]22016 examples [00:21, 1055.36 examples/s]22122 examples [00:21, 1047.05 examples/s]22227 examples [00:21, 1032.33 examples/s]22331 examples [00:21, 1031.05 examples/s]22435 examples [00:21, 999.80 examples/s] 22536 examples [00:21, 997.43 examples/s]22636 examples [00:21, 977.74 examples/s]22734 examples [00:22, 974.21 examples/s]22832 examples [00:22, 973.41 examples/s]22930 examples [00:22, 947.13 examples/s]23031 examples [00:22, 964.80 examples/s]23133 examples [00:22, 979.07 examples/s]23237 examples [00:22, 994.59 examples/s]23341 examples [00:22, 1006.15 examples/s]23442 examples [00:22, 975.96 examples/s] 23548 examples [00:22, 999.13 examples/s]23649 examples [00:23, 990.65 examples/s]23754 examples [00:23, 1006.69 examples/s]23859 examples [00:23, 1019.24 examples/s]23962 examples [00:23, 1014.30 examples/s]24064 examples [00:23, 1002.79 examples/s]24169 examples [00:23, 1013.72 examples/s]24271 examples [00:23, 962.29 examples/s] 24373 examples [00:23, 976.20 examples/s]24475 examples [00:23, 987.73 examples/s]24580 examples [00:23, 1004.96 examples/s]24681 examples [00:24, 988.27 examples/s] 24781 examples [00:24, 987.63 examples/s]24886 examples [00:24, 1004.40 examples/s]24987 examples [00:24, 995.09 examples/s] 25091 examples [00:24, 1006.74 examples/s]25194 examples [00:24, 1012.28 examples/s]25296 examples [00:24, 977.77 examples/s] 25395 examples [00:24, 978.48 examples/s]25494 examples [00:24, 962.84 examples/s]25598 examples [00:24, 982.98 examples/s]25702 examples [00:25, 997.10 examples/s]25806 examples [00:25, 1008.36 examples/s]25908 examples [00:25, 920.97 examples/s] 26002 examples [00:25, 912.37 examples/s]26095 examples [00:25, 908.32 examples/s]26195 examples [00:25, 932.95 examples/s]26299 examples [00:25, 960.19 examples/s]26396 examples [00:25, 960.33 examples/s]26497 examples [00:25, 973.16 examples/s]26595 examples [00:26, 965.37 examples/s]26692 examples [00:26, 955.49 examples/s]26788 examples [00:26, 955.79 examples/s]26884 examples [00:26, 938.70 examples/s]26986 examples [00:26, 960.76 examples/s]27092 examples [00:26, 987.60 examples/s]27197 examples [00:26, 1003.13 examples/s]27301 examples [00:26, 1013.63 examples/s]27403 examples [00:26, 1006.15 examples/s]27504 examples [00:26, 962.93 examples/s] 27607 examples [00:27, 980.16 examples/s]27718 examples [00:27, 1014.45 examples/s]27827 examples [00:27, 1035.80 examples/s]27932 examples [00:27, 1035.76 examples/s]28042 examples [00:27, 1051.83 examples/s]28154 examples [00:27, 1071.04 examples/s]28262 examples [00:27, 1071.00 examples/s]28370 examples [00:27, 1020.26 examples/s]28473 examples [00:27, 902.76 examples/s] 28567 examples [00:28, 872.09 examples/s]28657 examples [00:28, 858.99 examples/s]28745 examples [00:28, 856.55 examples/s]28832 examples [00:28, 829.41 examples/s]28916 examples [00:28, 828.83 examples/s]29001 examples [00:28, 833.06 examples/s]29089 examples [00:28, 846.20 examples/s]29183 examples [00:28, 857.66 examples/s]29278 examples [00:28, 881.91 examples/s]29374 examples [00:28, 903.84 examples/s]29468 examples [00:29, 914.11 examples/s]29560 examples [00:29, 915.43 examples/s]29653 examples [00:29, 917.24 examples/s]29745 examples [00:29, 916.27 examples/s]29841 examples [00:29, 926.92 examples/s]29939 examples [00:29, 940.37 examples/s]30034 examples [00:29, 886.60 examples/s]30142 examples [00:29, 934.96 examples/s]30245 examples [00:29, 960.11 examples/s]30354 examples [00:29, 995.10 examples/s]30461 examples [00:30, 1015.68 examples/s]30564 examples [00:30, 1015.54 examples/s]30667 examples [00:30, 991.20 examples/s] 30767 examples [00:30, 978.83 examples/s]30866 examples [00:30, 977.61 examples/s]30977 examples [00:30, 1011.88 examples/s]31092 examples [00:30, 1049.26 examples/s]31207 examples [00:30, 1077.43 examples/s]31316 examples [00:30, 1076.96 examples/s]31425 examples [00:31, 1057.13 examples/s]31535 examples [00:31, 1069.16 examples/s]31649 examples [00:31, 1081.12 examples/s]31769 examples [00:31, 1112.25 examples/s]31881 examples [00:31, 1050.37 examples/s]31994 examples [00:31, 1072.72 examples/s]32106 examples [00:31, 1086.34 examples/s]32217 examples [00:31, 1090.91 examples/s]32329 examples [00:31, 1098.36 examples/s]32440 examples [00:31, 1067.18 examples/s]32548 examples [00:32, 1068.23 examples/s]32664 examples [00:32, 1091.96 examples/s]32780 examples [00:32, 1108.59 examples/s]32892 examples [00:32, 1099.03 examples/s]33003 examples [00:32, 1081.33 examples/s]33114 examples [00:32, 1087.82 examples/s]33224 examples [00:32, 1088.85 examples/s]33334 examples [00:32, 1092.12 examples/s]33444 examples [00:32, 1074.31 examples/s]33552 examples [00:32, 1072.88 examples/s]33662 examples [00:33, 1079.96 examples/s]33773 examples [00:33, 1087.69 examples/s]33882 examples [00:33, 1088.32 examples/s]33991 examples [00:33, 1073.74 examples/s]34101 examples [00:33, 1081.48 examples/s]34210 examples [00:33, 1070.21 examples/s]34318 examples [00:33, 1041.91 examples/s]34424 examples [00:33, 1045.20 examples/s]34529 examples [00:33, 1029.29 examples/s]34633 examples [00:34, 985.91 examples/s] 34741 examples [00:34, 1012.25 examples/s]34850 examples [00:34, 1032.05 examples/s]34959 examples [00:34, 1047.79 examples/s]35065 examples [00:34, 1046.83 examples/s]35170 examples [00:34, 1046.03 examples/s]35279 examples [00:34, 1056.39 examples/s]35385 examples [00:34, 1045.18 examples/s]35491 examples [00:34, 1049.00 examples/s]35596 examples [00:34, 1041.42 examples/s]35701 examples [00:35, 1039.32 examples/s]35812 examples [00:35, 1057.10 examples/s]35918 examples [00:35, 1033.05 examples/s]36022 examples [00:35, 997.87 examples/s] 36126 examples [00:35, 1007.72 examples/s]36233 examples [00:35, 1024.38 examples/s]36336 examples [00:35, 1019.63 examples/s]36444 examples [00:35, 1035.05 examples/s]36554 examples [00:35, 1053.50 examples/s]36660 examples [00:35, 1043.55 examples/s]36765 examples [00:36, 1038.12 examples/s]36872 examples [00:36, 1044.63 examples/s]36977 examples [00:36, 1045.71 examples/s]37087 examples [00:36, 1058.99 examples/s]37193 examples [00:36, 1036.28 examples/s]37304 examples [00:36, 1054.95 examples/s]37410 examples [00:36, 1054.21 examples/s]37518 examples [00:36, 1060.19 examples/s]37625 examples [00:36, 1025.58 examples/s]37728 examples [00:37, 998.10 examples/s] 37837 examples [00:37, 1023.58 examples/s]37952 examples [00:37, 1058.29 examples/s]38066 examples [00:37, 1081.29 examples/s]38182 examples [00:37, 1101.58 examples/s]38293 examples [00:37, 1044.12 examples/s]38401 examples [00:37, 1053.18 examples/s]38508 examples [00:37, 1049.72 examples/s]38617 examples [00:37, 1061.37 examples/s]38726 examples [00:37, 1069.16 examples/s]38834 examples [00:38, 1055.84 examples/s]38945 examples [00:38, 1070.42 examples/s]39055 examples [00:38, 1078.40 examples/s]39165 examples [00:38, 1081.96 examples/s]39275 examples [00:38, 1084.70 examples/s]39384 examples [00:38, 1081.49 examples/s]39493 examples [00:38, 1005.20 examples/s]39599 examples [00:38, 1019.97 examples/s]39709 examples [00:38, 1040.26 examples/s]39818 examples [00:38, 1053.45 examples/s]39924 examples [00:39, 1042.14 examples/s]40029 examples [00:39, 984.53 examples/s] 40137 examples [00:39, 1010.74 examples/s]40241 examples [00:39, 1018.95 examples/s]40358 examples [00:39, 1058.70 examples/s]40465 examples [00:39, 1038.18 examples/s]40570 examples [00:39, 1024.78 examples/s]40674 examples [00:39, 1029.29 examples/s]40784 examples [00:39, 1047.37 examples/s]40892 examples [00:40, 1056.04 examples/s]40998 examples [00:40, 1036.85 examples/s]41102 examples [00:40, 1034.09 examples/s]41206 examples [00:40, 987.15 examples/s] 41317 examples [00:40, 1019.43 examples/s]41430 examples [00:40, 1048.69 examples/s]41536 examples [00:40, 1027.21 examples/s]41651 examples [00:40, 1060.70 examples/s]41758 examples [00:40, 1061.99 examples/s]41867 examples [00:40, 1067.61 examples/s]41975 examples [00:41, 1054.49 examples/s]42085 examples [00:41, 1067.10 examples/s]42198 examples [00:41, 1084.47 examples/s]42307 examples [00:41, 1068.38 examples/s]42415 examples [00:41, 1066.72 examples/s]42522 examples [00:41, 1057.13 examples/s]42628 examples [00:41, 1056.84 examples/s]42735 examples [00:41, 1060.27 examples/s]42842 examples [00:41, 1051.94 examples/s]42948 examples [00:41, 1007.21 examples/s]43050 examples [00:42, 1010.76 examples/s]43155 examples [00:42, 1020.58 examples/s]43258 examples [00:42, 1017.83 examples/s]43360 examples [00:42, 979.57 examples/s] 43462 examples [00:42, 988.74 examples/s]43562 examples [00:42, 982.19 examples/s]43669 examples [00:42, 1006.06 examples/s]43777 examples [00:42, 1027.05 examples/s]43886 examples [00:42, 1044.08 examples/s]43991 examples [00:42, 1037.87 examples/s]44096 examples [00:43, 1010.34 examples/s]44198 examples [00:43, 1005.39 examples/s]44301 examples [00:43, 1012.40 examples/s]44406 examples [00:43, 1020.93 examples/s]44509 examples [00:43, 1011.89 examples/s]44611 examples [00:43, 947.71 examples/s] 44718 examples [00:43, 980.07 examples/s]44824 examples [00:43, 1000.65 examples/s]44926 examples [00:43, 1004.28 examples/s]45027 examples [00:44, 994.39 examples/s] 45127 examples [00:44, 962.06 examples/s]45229 examples [00:44, 977.47 examples/s]45333 examples [00:44, 994.51 examples/s]45437 examples [00:44, 1005.15 examples/s]45545 examples [00:44, 1026.00 examples/s]45648 examples [00:44, 1008.78 examples/s]45753 examples [00:44, 1020.27 examples/s]45862 examples [00:44, 1039.82 examples/s]45968 examples [00:44, 1045.56 examples/s]46073 examples [00:45, 1026.50 examples/s]46176 examples [00:45, 988.34 examples/s] 46276 examples [00:45, 942.04 examples/s]46376 examples [00:45, 958.61 examples/s]46479 examples [00:45, 976.38 examples/s]46581 examples [00:45, 986.42 examples/s]46681 examples [00:45, 985.93 examples/s]46784 examples [00:45, 998.75 examples/s]46885 examples [00:45, 994.26 examples/s]46985 examples [00:46, 964.26 examples/s]47082 examples [00:46, 951.89 examples/s]47187 examples [00:46, 977.82 examples/s]47289 examples [00:46, 988.84 examples/s]47391 examples [00:46, 995.20 examples/s]47493 examples [00:46, 1001.30 examples/s]47595 examples [00:46, 1005.92 examples/s]47696 examples [00:46, 970.63 examples/s] 47797 examples [00:46, 980.00 examples/s]47896 examples [00:46, 960.34 examples/s]47993 examples [00:47, 952.31 examples/s]48093 examples [00:47, 965.32 examples/s]48190 examples [00:47, 961.71 examples/s]48290 examples [00:47, 972.26 examples/s]48388 examples [00:47, 971.82 examples/s]48497 examples [00:47, 1003.72 examples/s]48598 examples [00:47, 1004.71 examples/s]48699 examples [00:47, 1000.73 examples/s]48800 examples [00:47, 1001.97 examples/s]48901 examples [00:47, 995.32 examples/s] 49001 examples [00:48, 962.29 examples/s]49101 examples [00:48, 970.98 examples/s]49204 examples [00:48, 987.12 examples/s]49312 examples [00:48, 1010.57 examples/s]49414 examples [00:48, 1010.35 examples/s]49516 examples [00:48, 980.03 examples/s] 49615 examples [00:48, 963.18 examples/s]49729 examples [00:48, 1008.59 examples/s]49845 examples [00:48, 1048.04 examples/s]49951 examples [00:48, 1036.50 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7998/50000 [00:00<00:00, 79977.22 examples/s] 41%|████▏     | 20640/50000 [00:00<00:00, 89882.42 examples/s] 69%|██████▉   | 34663/50000 [00:00<00:00, 100731.42 examples/s] 98%|█████████▊| 48984/50000 [00:00<00:00, 110568.70 examples/s]                                                                0 examples [00:00, ? examples/s]79 examples [00:00, 784.67 examples/s]179 examples [00:00, 837.77 examples/s]271 examples [00:00, 859.04 examples/s]366 examples [00:00, 884.25 examples/s]471 examples [00:00, 927.93 examples/s]579 examples [00:00, 967.86 examples/s]685 examples [00:00, 992.14 examples/s]779 examples [00:00, 955.09 examples/s]880 examples [00:00, 970.73 examples/s]987 examples [00:01, 997.64 examples/s]1096 examples [00:01, 1021.79 examples/s]1198 examples [00:01, 1011.68 examples/s]1299 examples [00:01, 1006.98 examples/s]1401 examples [00:01, 1009.04 examples/s]1502 examples [00:01, 974.72 examples/s] 1601 examples [00:01, 978.00 examples/s]1700 examples [00:01, 978.99 examples/s]1801 examples [00:01, 987.36 examples/s]1915 examples [00:01, 1027.05 examples/s]2024 examples [00:02, 1044.27 examples/s]2129 examples [00:02, 1028.90 examples/s]2233 examples [00:02, 973.17 examples/s] 2335 examples [00:02, 985.62 examples/s]2435 examples [00:02, 987.96 examples/s]2539 examples [00:02, 1001.80 examples/s]2654 examples [00:02, 1041.77 examples/s]2764 examples [00:02, 1057.04 examples/s]2871 examples [00:02, 1045.87 examples/s]2979 examples [00:02, 1055.20 examples/s]3089 examples [00:03, 1066.34 examples/s]3201 examples [00:03, 1079.65 examples/s]3311 examples [00:03, 1084.36 examples/s]3420 examples [00:03, 1054.15 examples/s]3531 examples [00:03, 1068.00 examples/s]3641 examples [00:03, 1074.79 examples/s]3752 examples [00:03, 1083.08 examples/s]3863 examples [00:03, 1089.77 examples/s]3973 examples [00:03, 1073.45 examples/s]4082 examples [00:03, 1077.49 examples/s]4190 examples [00:04, 1069.72 examples/s]4298 examples [00:04, 1050.48 examples/s]4409 examples [00:04, 1065.08 examples/s]4516 examples [00:04, 1055.16 examples/s]4626 examples [00:04, 1067.47 examples/s]4733 examples [00:04, 1017.95 examples/s]4842 examples [00:04, 1035.94 examples/s]4951 examples [00:04, 1048.48 examples/s]5057 examples [00:04, 1037.94 examples/s]5165 examples [00:05, 1049.55 examples/s]5271 examples [00:05, 1030.43 examples/s]5375 examples [00:05, 1030.02 examples/s]5482 examples [00:05, 1041.39 examples/s]5587 examples [00:05, 1030.89 examples/s]5691 examples [00:05, 1005.78 examples/s]5800 examples [00:05, 1026.94 examples/s]5903 examples [00:05, 1002.28 examples/s]6004 examples [00:05, 975.84 examples/s] 6117 examples [00:05, 1015.70 examples/s]6228 examples [00:06, 1041.18 examples/s]6339 examples [00:06, 1059.70 examples/s]6450 examples [00:06, 1073.48 examples/s]6558 examples [00:06, 1058.72 examples/s]6665 examples [00:06, 1032.90 examples/s]6769 examples [00:06, 1031.41 examples/s]6873 examples [00:06, 1002.55 examples/s]6974 examples [00:06, 981.92 examples/s] 7073 examples [00:06, 969.27 examples/s]7179 examples [00:07, 993.38 examples/s]7285 examples [00:07, 1012.36 examples/s]7404 examples [00:07, 1058.27 examples/s]7511 examples [00:07, 1060.97 examples/s]7618 examples [00:07, 1035.27 examples/s]7723 examples [00:07, 1025.59 examples/s]7828 examples [00:07, 1032.22 examples/s]7937 examples [00:07, 1048.80 examples/s]8048 examples [00:07, 1064.40 examples/s]8155 examples [00:07, 1047.36 examples/s]8266 examples [00:08, 1063.35 examples/s]8378 examples [00:08, 1078.66 examples/s]8487 examples [00:08, 1075.94 examples/s]8595 examples [00:08, 1059.58 examples/s]8702 examples [00:08, 1025.28 examples/s]8814 examples [00:08, 1050.19 examples/s]8926 examples [00:08, 1068.77 examples/s]9034 examples [00:08, 1071.11 examples/s]9142 examples [00:08, 1069.46 examples/s]9250 examples [00:08, 1056.47 examples/s]9356 examples [00:09, 1036.36 examples/s]9460 examples [00:09, 1030.38 examples/s]9571 examples [00:09, 1052.14 examples/s]9685 examples [00:09, 1076.03 examples/s]9793 examples [00:09, 1035.46 examples/s]9898 examples [00:09, 1010.79 examples/s]10000 examples [00:09, 1011.99 examples/s]                                            0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePOE7VK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePOE7VK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e4523ca60>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e4523ca60>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e4523ca60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f0e82c9bb70>, <torch.utils.data.dataloader.DataLoader object at 0x7f0e356dff98>), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 

#####  Load JSON data_pars
{
  "data_info": {
    "data_path": "mlmodels/dataset/vision/MNIST",
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
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e356fa598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e356fa598>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e356fa598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f0e829085c0>, <torch.utils.data.dataloader.DataLoader object at 0x7f0e47fc7320>), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/keras_textcnn.json 

#####  Load JSON data_pars
{
  "data_info": {
    "dataset": "mlmodels/dataset/text/imdb",
    "pass_data_pars": false,
    "train": true,
    "maxlen": 40,
    "max_features": 5
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::NumpyDataset",
      "args": {
        "numpy_loader_args": {
          "allow_pickle": true
        },
        "encoding": "'ISO-8859-1'"
      }
    }
  ]
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.NumpyDataset'>
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz

 #####  get_Data DataLoader 
((array([list([10, 432, 2, 216, 11, 17, 233, 311, 100, 109, 27791, 5, 31, 3, 168, 366, 4, 1920, 634, 971, 12, 10, 13, 5523, 5, 64, 9, 85, 36, 48, 10, 694, 4, 13059, 15969, 26, 13, 61, 499, 5, 78, 209, 10, 13, 352, 15969, 253, 1, 106, 4, 3270, 14998, 52, 70, 2, 1839, 11762, 253, 1019, 7655, 16, 138, 12866, 1, 1910, 4, 3, 49, 17, 6, 12, 9, 67, 2885, 16, 260, 1435, 11, 28, 119, 615, 12, 1, 433, 747, 60, 13, 2959, 43, 13, 3080, 31, 2126, 312, 1, 83, 317, 4, 1, 17, 2, 68, 1678, 5, 1671, 312, 1, 330, 317, 134, 14200, 1, 747, 10, 21, 61, 216, 108, 369, 8, 1671, 18, 108, 365, 2068, 346, 14, 70, 266, 2721, 21, 5, 384, 256, 64, 95, 2575, 11, 17, 13, 84, 2, 10, 1464, 12, 22, 137, 64, 9, 156, 22, 1916]),
       list([281, 676, 164, 985, 5696, 1157, 53, 24, 2425, 2013, 1, 3357, 186, 11603, 16, 11, 220, 2572, 2252, 450, 41, 1, 21308, 1203, 587, 908, 118, 3, 182, 295, 47415, 5157, 36, 24, 4486, 975, 5, 294, 426, 24, 7117, 8, 48, 13, 2275, 14, 1, 830, 497, 123, 253, 143, 54, 334, 4, 8891, 2, 131, 10465, 9594, 2252, 1551, 23, 3, 9591, 3, 2517, 88, 1030, 221, 5, 1755, 959, 16, 4628, 2, 2376, 129, 18, 46, 86, 11, 19, 13, 8480, 29, 1, 169, 7, 7, 1, 19, 514, 16, 46, 1515, 633, 895, 835, 3, 51329, 307, 4, 1, 1122, 633, 895, 4, 27000, 49040, 2, 5544, 18, 35402, 364, 1361, 15, 91, 83, 31, 1, 1393, 531, 277, 1, 203, 1099, 5, 1, 1203, 587, 908, 180, 1258, 53, 52, 70, 5696, 124, 3, 324, 289, 2, 284, 3, 9408, 15, 1131, 3664, 15697, 10, 444, 1, 2514, 11836, 4223, 4, 1, 203, 20, 248, 104, 4, 1, 908, 12, 19323, 1, 111, 1034, 39, 760, 46, 2073, 1984, 1134, 5, 1, 3917, 222, 46, 1441, 106, 940, 51, 1, 695, 1332, 6, 2365, 31, 1215, 4, 1, 15171, 8, 325, 3672, 2, 347, 6085, 34, 2727, 24, 220, 17370, 14, 3, 503, 5, 94, 93, 15, 3, 8891, 262, 26, 79, 124, 3, 49, 289, 4, 2006, 5004, 48, 268, 20, 8, 1, 73329, 1825, 464, 5097, 8891, 3, 2146, 354, 4106, 6, 836, 6313, 1236, 130, 1106, 141, 79, 27, 345, 1, 267, 16132, 2, 2295, 2547, 15, 1852, 32, 1725, 807, 415, 838, 4, 1313, 2, 5788, 30, 1, 451, 4, 1, 10257, 1114, 7, 7, 22, 121, 86, 11, 6, 167, 5, 127, 21, 61, 85, 42, 445, 20, 3, 280, 62, 18, 79, 85, 105, 8, 11, 509, 791, 1, 169, 14212, 117, 2, 117, 18, 5696, 1454, 20, 3, 125, 71, 853, 120, 2, 379, 10442, 50, 673, 493, 1, 367, 71, 26, 123, 66, 8, 1008, 4, 9, 463, 1, 4374, 873, 11, 6, 3, 324, 2, 773, 19, 5, 3660, 15, 12, 1012, 5, 166, 32, 308]),
       list([14, 3, 32427, 48732, 16, 46, 1854, 4, 1, 45946, 476, 10, 13, 3518, 16, 4668, 8355, 5, 1, 1338, 4, 704, 8, 8891, 8, 1, 399, 10257, 1114, 1, 17, 2396, 70, 1, 1984, 3350, 12, 1332, 5103, 743, 306, 36, 24, 1545, 7417, 4, 109, 20936, 5, 24, 202, 5147, 5, 986, 12, 3059, 8553, 12, 10005, 87, 36, 109, 3208, 14, 32, 3212, 8, 628, 8891, 923, 4713, 1, 182, 268, 140, 24, 202, 704, 3044, 109, 3, 2688, 47450, 8, 1, 520, 4, 1, 3020, 16377, 1528, 34, 19884, 30, 24, 1024, 5, 2197, 749, 24, 2087, 7, 7, 48, 10, 444, 115, 187, 6, 86, 11, 753, 4, 704, 6, 16957, 8, 1, 102, 4, 843, 24, 333, 6, 3, 777, 704, 12527, 34, 1082, 1, 1104, 4, 251, 154, 18, 6, 22727, 31, 1, 3020, 704, 24, 449, 187, 12159, 38, 6253, 673, 2, 1759, 2, 9087, 87, 5, 6648, 24, 922, 4, 9887, 426, 145, 34, 101, 26, 6, 4551, 7, 7, 414, 1, 8891, 136, 23, 70, 3548, 258, 1, 262, 340, 8, 1, 17, 13, 21, 1, 776, 2135, 4, 1, 1376, 10822, 1, 114, 8863, 620, 31, 907, 78, 21, 6203, 36, 1, 933, 4, 1, 19, 222, 28, 114, 907, 558, 30, 1, 3070, 2699, 897, 1, 526, 124, 21, 63, 101, 907, 1, 274, 14, 8, 4628, 6, 21, 46, 907, 3562, 18, 28, 12, 61, 403, 476, 97, 25, 395]),
       ...,
       list([1, 1118, 509, 6, 3, 705, 14053, 16, 32, 3145, 31366, 3, 23322, 22816, 44, 802, 5, 94, 3, 173, 50, 43, 4, 11, 769, 1, 1835, 4, 1, 422, 8, 658, 5, 718, 41, 29, 1374, 4, 389, 1296, 10724, 2426, 5, 863, 1, 102, 50, 71, 1, 308, 7, 7, 42, 63, 244, 5919, 148, 22816, 149, 303, 925, 5264, 1, 705, 788, 8, 3, 1594, 12, 77, 398, 175, 2221, 9996, 244, 24, 1149, 6, 20, 1, 1512, 20469, 530, 7, 7, 1, 411, 6, 5559, 17993, 130, 20892, 47, 2, 16, 1, 1810, 863, 4, 1879, 13753, 31, 1, 2549, 1588, 148, 47, 6, 54, 117, 1, 1976, 35, 12, 1, 153, 67, 76, 20, 16, 7430, 8, 65, 831, 1, 2147, 600, 188, 984, 3, 1034, 4958, 2, 397, 492, 426, 1, 918, 7245, 4, 345, 267, 1325, 222, 79, 3, 52, 18911, 3325, 10, 132, 9, 695, 232, 5, 1, 1420, 10, 158, 178, 5, 64, 3, 6908, 4, 104, 1109, 652, 704, 81, 1056, 8, 32, 2799, 901, 10, 470, 5, 64, 46, 429, 4, 10282, 16531, 4, 1, 62, 7, 7, 414, 9, 6, 8, 189, 1, 62, 60, 1622, 1, 357, 177, 40, 14, 1, 788, 4, 833, 23, 244, 1032, 35, 1, 62, 6, 32, 3037, 4, 4760, 4, 272, 10523, 2, 15001, 9317, 14, 3, 2193, 15, 106, 5148, 36559, 2, 5105, 1422, 11, 23874, 11639, 16, 49, 351, 2, 305, 40745, 7695, 2179, 5, 36559, 6, 3, 2375, 7183, 15, 73, 4, 1, 7102, 7, 7, 143, 682, 21, 12, 42, 75, 18, 12, 9, 97, 25, 74, 73, 125, 339, 155]),
       list([686, 180, 3701, 69, 14, 5, 11, 19, 4630, 9, 378, 38277, 10300, 4, 1060, 2243, 34, 6, 207, 3, 1739, 5, 103, 9, 941, 527, 159, 9106, 521, 5179, 12748, 9, 6, 811, 8, 9055, 16, 1, 159, 9106, 19, 17147, 65041, 3, 461, 1175, 9, 5, 69, 187, 10, 13, 1251, 682, 1, 223, 766, 6, 1752, 2, 2729, 16, 52, 114, 3470, 113, 6, 475, 18, 5179, 6, 469, 464, 340, 1, 82, 153, 2, 1504, 23, 29, 861, 18, 10, 241, 7742, 16, 95, 29, 1209, 36, 1, 324, 26403, 60, 23, 20671, 466, 1, 17, 2, 10300, 2, 5179, 11, 19, 215, 52, 49, 35, 1, 1512, 4, 1, 62, 6, 89, 103, 9, 891, 22, 63, 178, 5]),
       list([625, 792, 4003, 23, 4840, 70, 395, 2, 7177, 14, 217, 132, 282, 10, 232, 41, 9287, 10159, 11, 2554, 618, 6, 35, 6735, 2, 117, 2651, 117, 1052, 2, 212, 25, 2161, 81794, 30, 5507, 8, 5493, 9, 2319, 5281, 1428, 2509, 12, 6, 328, 9120, 9287, 10159, 212, 25, 74, 33351, 36, 3, 2086, 2177, 12, 298, 1397, 72, 185, 12, 4995, 17441, 229, 2, 12, 478, 4, 225, 5583, 1622, 76, 11, 96, 2, 2672, 117, 3, 3682, 24375, 1, 956, 6, 3, 6759, 4, 14375, 84477, 317, 3, 47285, 46, 2726, 6580, 610, 2, 834, 1393, 4, 3, 402, 9645, 2, 58659, 35, 5453, 151, 3354, 14, 5, 27, 37, 618, 44883, 2, 20629, 21663, 16232, 1, 362, 6, 16326, 5464, 2874, 5, 647, 18, 1, 357, 1, 644, 880, 2803, 264, 15668, 3486, 457, 348, 1, 45365, 209, 2, 1, 14960, 354, 687, 11, 2090, 19, 80, 109, 2224, 10159, 6, 37, 3, 27315, 6300, 8107, 618, 16, 46, 5214, 1798, 5, 6708, 9, 53, 320, 623, 1, 288, 156, 1035, 8069, 2, 147, 6, 109, 414, 2522, 15, 1, 318, 19, 6, 2067, 6, 2, 3420, 269, 6459, 2697, 259, 8, 1, 233, 317, 531, 18, 10159, 6, 10218, 2, 13314, 45900, 2, 10, 261, 63, 553, 122, 1, 8107, 618, 18012, 4, 1, 3349, 2, 90, 3420, 165, 37, 10218, 171, 60, 10, 261, 13, 21, 10764, 3421, 5507, 212, 25, 14910, 43939, 51, 33, 216, 11, 2, 16, 157, 888, 1428, 19870, 8, 1274, 5281, 2, 1688, 1428, 8, 20, 3, 785, 248, 2, 2473, 1428, 8, 2562, 126, 10518, 33, 66, 3, 4115, 14120, 4, 15797, 8723, 16, 13816, 1428, 3143, 80, 467, 105, 16, 52, 8330, 21791, 276, 33, 141, 25, 1800, 81463, 7127, 15940, 36, 20, 3, 785, 248, 10159, 13, 52, 1060, 20, 5115, 83, 763, 8, 3872, 2, 2175, 8, 35532, 6151, 15, 1925, 18, 9, 1193, 277, 43, 8, 1, 16528, 2, 1, 3772, 2, 61, 123, 25474, 100, 12, 20, 28, 311, 1404, 16, 20, 3, 785, 248, 14, 3, 2707, 311, 1402, 1291, 555, 5507, 66, 65, 603, 87603, 419, 28, 1428, 2860, 19, 116, 62, 2, 12, 467, 1428, 2860, 2109, 11855, 1, 3515, 79, 1620, 5, 7733, 29, 1, 13816, 1428, 8, 40, 1, 372, 104, 150, 15, 40, 1045])],
      dtype=object), array([list([23022, 309, 6, 3, 1069, 209, 9, 2175, 30, 1, 169, 55, 14, 46, 82, 5869, 41, 393, 110, 138, 14, 5359, 58, 4477, 150, 8, 1, 5032, 5948, 482, 69, 5, 261, 12, 23022, 73935, 2003, 6, 73, 2436, 5, 632, 71, 6, 5359, 1, 25279, 5, 2004, 10471, 1, 5941, 1534, 34, 67, 64, 205, 140, 65, 1232, 63526, 21145, 1, 49265, 4, 1, 223, 901, 29, 3024, 69, 4, 1, 5863, 10, 694, 2, 65, 1534, 51, 10, 216, 1, 387, 8, 60, 3, 1472, 3724, 802, 5, 3521, 177, 1, 393, 10, 1238, 14030, 30, 309, 3, 353, 344, 2989, 143, 130, 5, 7804, 28, 4, 126, 5359, 1472, 2375, 5, 23022, 309, 10, 532, 12, 108, 1470, 4, 58, 556, 101, 12, 23022, 309, 6, 227, 4187, 48, 3, 2237, 12, 9, 215]),
       list([23777, 39, 81226, 14, 739, 20387, 3428, 44, 74, 32, 1831, 15, 150, 18, 112, 3, 1344, 5, 336, 145, 20, 1, 887, 12, 68, 277, 1189, 403, 34, 119, 282, 36, 167, 5, 393, 154, 39, 2299, 15, 1, 548, 88, 81, 101, 4, 1, 3273, 14, 40, 3, 413, 1200, 134, 8208, 41, 180, 138, 14, 3086, 1, 322, 20, 4930, 28948, 359, 5, 3112, 2128, 1, 20045, 19339, 39, 8208, 45, 3661, 27, 372, 5, 127, 53, 20, 1, 1983, 7, 7, 18, 48, 45, 22, 68, 345, 3, 2131, 5, 409, 20, 1, 1983, 15, 3, 3238, 206, 1, 31645, 22, 277, 66, 36, 3, 341, 1, 719, 729, 3, 3865, 1265, 20, 1, 1510, 3, 1219, 2, 282, 22, 277, 2525, 5, 64, 48, 42, 37, 5, 27, 3273, 12, 6, 23030, 75120, 2034, 7, 7, 3771, 3225, 34, 4186, 34, 378, 14, 12583, 296, 3, 1023, 129, 34, 44, 282, 8, 1, 179, 363, 7067, 5, 94, 3, 2131, 16, 3, 5211, 3005, 15913, 21720, 5, 64, 45, 26, 67, 409, 8, 1, 1983, 15, 3261, 501, 206, 1, 31645, 45, 12583, 2877, 26, 67, 78, 48, 26, 491, 16, 3, 702, 1184, 4, 228, 50, 4505, 1, 43259, 20, 118, 12583, 6, 1373, 20, 1, 887, 16, 3, 20447, 20, 24, 3964, 5, 10455, 24, 172, 844, 118, 26, 188, 1488, 122, 1, 6616, 237, 345, 1, 13891, 32804, 31, 3, 39870, 100, 42, 395, 20, 24, 12130, 118, 12583, 889, 82, 102, 584, 3, 252, 31, 1, 400, 4, 4787, 16974, 1962, 3861, 32, 1230, 3186, 34, 185, 4310, 156, 2325, 38, 341, 2, 38, 9048, 7355, 2231, 4846, 2, 32880, 8938, 2610, 34, 23, 457, 340, 5, 1, 1983, 504, 4355, 12583, 215, 237, 21, 340, 5, 4468, 5996, 34689, 37, 26, 277, 119, 51, 109, 1023, 118, 42, 545, 39, 2814, 513, 39, 27, 553, 7, 7, 134, 1, 116, 2022, 197, 4787, 2, 12583, 283, 1667, 5, 111, 10, 255, 110, 4382, 5, 27, 28, 4, 3771, 12267, 16617, 105, 118, 2597, 5, 109, 3, 209, 9, 284, 3, 4325, 496, 1076, 5, 24, 2761, 154, 138, 14, 7673, 11900, 182, 5276, 39, 20422, 15, 1, 548, 5, 120, 48, 42, 37, 257, 139, 4530, 156, 2325, 9, 1, 372, 248, 39, 20, 1, 82, 505, 228, 3, 376, 2131, 37, 29, 1023, 81, 78, 51, 33, 89, 121, 48, 5, 78, 16, 65, 275, 276, 33, 141, 199, 9, 5, 1, 3273, 302, 4, 769, 9, 37, 17648, 275, 7, 7, 39, 276, 11, 19, 77, 6018, 22, 5, 336, 406]),
       list([527, 117, 113, 31, 16974, 1962, 3861, 115, 902, 22590, 758, 10, 25, 123, 107, 2, 116, 136, 8, 1646, 7972, 23, 330, 5, 597, 1, 6244, 20, 390, 6, 3, 353, 14, 49, 14, 230, 8, 7673, 11900, 1, 190, 20, 9567, 6, 79, 894, 100, 109, 3609, 4, 109, 3, 36605, 3485, 43, 24, 1407, 2, 109, 12646, 1, 2405, 4, 32804, 12583, 26018, 39247, 143, 3, 2405, 26, 557, 286, 160, 712, 4122, 21720, 3, 511, 36, 1, 300, 2793, 7068, 120, 6, 774, 130, 96, 14, 3, 1165, 5285, 34, 491, 5, 4263, 1, 6788, 24, 106, 6, 50, 10557, 71, 641, 1, 1547, 133, 2, 1, 133, 118, 1, 3273, 18223, 3, 14364, 2135, 23, 29, 55, 2236, 165, 15, 1, 2974, 133, 2, 1, 104, 191, 16616, 994, 28, 26998, 11, 17, 211, 125, 254, 55, 10, 64, 9, 60, 6, 176, 397]),
       ...,
       list([10, 216, 77424, 233, 311, 30, 1, 21144, 19, 1410, 2, 9, 13, 28, 663, 1384, 1384, 85, 1, 766, 13, 4613, 973, 1, 10825, 4, 316, 7284, 3994, 8, 3, 4136, 4688, 17, 13, 1124, 2, 109, 3, 334, 931, 30037, 143, 21, 4, 45035, 49280, 1551, 4, 1, 1697, 10, 13, 2961, 5, 132, 52, 1994, 5, 805, 11, 17, 43, 58, 1171, 900, 1228, 5, 1, 2236, 419, 1, 766, 44, 983, 18, 1, 3230, 23, 1032, 1, 153, 2628, 57, 3994, 6, 1893, 46, 59, 132, 12, 42, 3, 205, 2820, 4, 1, 1167, 179, 8, 1, 175, 12, 1, 10732, 4, 1, 102, 2892, 3, 1285, 2, 29, 12, 3979, 18, 9, 40, 163, 1, 223, 17, 41077, 40, 37, 1, 133, 118, 3994, 211, 3537, 9, 612, 1500, 3030, 10, 283, 1014, 230, 63005, 402, 18, 128, 710, 72, 1405, 5, 232, 5051, 15, 38, 10, 158, 21, 15, 3, 783, 56, 13, 35, 832, 29, 1, 93, 2, 10, 329, 12, 1, 1378, 13, 1156, 70, 9, 6, 49, 846, 18, 161, 1562, 2241, 342, 10, 212, 971, 12, 1, 2821, 30, 1, 1410, 283, 35, 49, 35, 276, 10, 1046, 43, 139, 130, 18, 30, 1, 127, 4, 1, 17, 10, 423, 336, 533, 7159, 232, 37, 146, 16173, 69516, 171, 3999, 50, 612, 1, 83, 133, 8, 1, 1330, 6, 1290, 321, 2, 29, 18, 10, 66, 1, 2916, 8773, 4, 146, 3, 1204, 2, 50, 354, 307, 4, 1, 133, 8, 1, 6407, 1446, 748, 1, 295, 2277, 3620, 8, 7642, 42456, 8445, 965, 1132, 16, 15523, 1, 2622, 764, 2, 1333, 1521, 1, 1182, 9467, 225, 1, 828, 16846, 12600, 838, 54, 10, 40, 423, 76, 80, 11, 17, 96, 75]),
       list([46, 105, 12, 22, 1258, 53, 15, 3, 6564, 468, 43, 5, 27, 244, 49, 31940, 1114, 105, 623, 4190, 4, 3717, 1119, 2, 295, 17, 12, 68, 84, 18, 258, 35131, 623, 46, 4956, 105, 2920, 406, 1, 6134, 4, 65, 10561, 6, 592, 37, 1, 862, 6242, 7, 7, 1, 61, 1120, 152, 10, 67, 132, 41, 11, 19, 6, 12, 42, 1279, 748, 14, 613, 14, 1, 7222, 4, 2117, 82, 71, 12, 91, 3, 52, 4113, 6825, 19, 16, 1, 1756, 18424, 4, 3, 27058, 310, 2168, 31, 3, 60431, 7, 7, 42, 74, 3209, 3297, 18, 22, 63, 78, 25, 5, 3247, 41, 3, 19, 12, 17580, 5593, 4, 1, 203, 80, 91, 1106, 717, 35, 31, 1, 55, 9, 211, 5, 1, 862, 3031, 871, 107, 9, 29, 457, 7, 7, 75, 17, 448, 77, 25, 3, 1868, 146, 1, 3067, 1779, 2383, 2494, 2, 1, 9586, 113, 4, 1, 174, 259, 1, 11200, 34, 13, 35, 75, 26, 119, 94, 69, 459, 3, 224, 2, 3597, 5, 35131, 15, 394, 8, 5, 1, 1100, 4, 180, 31, 7015, 3, 2489, 35, 75, 9, 418, 37, 10, 13, 146, 46, 1556, 53, 341, 371, 4, 3, 7790, 1186, 7, 7, 370, 370, 535, 1999, 29, 90, 535, 37, 11, 51, 1999, 1837, 3, 1067, 4, 3, 367, 18, 1138, 278, 14824, 2, 131, 105, 38137, 8, 260, 51902, 1195, 795]),
       list([11, 6, 28, 4, 1, 7051, 105, 204, 123, 107, 9, 7244, 122, 751, 123, 549, 4, 705, 2, 1027, 5, 94, 3, 944, 4, 95, 29, 7, 7, 222, 21, 3, 683, 49, 344, 39, 106, 8, 1, 223, 944, 45, 47, 13, 3, 111, 9, 13, 32, 11620, 2, 14, 227, 14, 113, 268, 222, 161, 49, 5, 132, 35, 1812, 132, 161, 10, 1249, 2485, 388, 86, 11, 549, 4, 1832, 211, 1052, 2, 162, 623, 124, 1840, 1195, 21, 30, 46, 865, 101, 12919, 58, 555, 11, 63, 6, 3, 3814, 4, 62964, 2, 680, 9, 3, 248, 91, 592, 37, 11, 12, 44, 81, 17238, 20043, 1, 1469, 269, 37, 3, 337, 272, 19, 30, 219, 45, 22, 25, 10131, 9, 22, 771, 1050, 126, 55, 39, 275, 89, 434, 126, 55, 11, 6, 1347])],
      dtype=object), array([1, 1, 1, ..., 0, 0, 0]), array([1, 1, 1, ..., 0, 0, 0])), {})





 ####################################################################################################
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/namentity_crm_bilstm_new.json 

#####  Load JSON data_pars
{
  "data_info": {
    "data_path": "dataset/text/",
    "dataset": "ner_dataset.csv",
    "pass_data_pars": false,
    "train": true
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::pandasDataset",
      "args": {
        "read_csv_parm": {
          "encoding": "ISO-8859-1"
        },
        "colX": [],
        "coly": []
      }
    },
    {
      "uri": "mlmodels/preprocess/text_keras.py::Preprocess_namentity",
      "args": {
        "max_len": 75
      },
      "internal_states": [
        "word_count"
      ]
    },
    {
      "name": "split_xy",
      "uri": "mlmodels/dataloader.py::split_xy_from_dict",
      "args": {
        "col_Xinput": [
          "X"
        ],
        "col_yinput": [
          "y"
        ]
      }
    },
    {
      "name": "split_train_test",
      "uri": "sklearn.model_selection::train_test_split",
      "args": {
        "test_size": 0.5
      }
    },
    {
      "name": "saver",
      "uri": "mlmodels/dataloader.py::pickle_dump",
      "args": {
        "path": "mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl"
      }
    }
  ],
  "output": {
    "shape": [
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len": 75
  }
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/preprocess/text_keras.py::Preprocess_namentity {'max_len': 75} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/text_keras.Preprocess_namentity'>
cls_name : Preprocess_namentity

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels/dataloader.py::split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0effe7dea0>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0effe7dea0>

  URL:  sklearn.model_selection::train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f0eebe62ea0>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f0eebe62ea0>

  URL:  mlmodels/dataloader.py::pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0e356ddd90>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0e356ddd90>
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/namentity_crm_bilstm_new.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'



###### Test_run_model  #############################################################



 ####################################################################################################
dataset/json/refactor/charcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.dataloader.charcnn.py",
      "embedding_size": 128,
      "conv_layers": [
        [
          256,
          10
        ],
        [
          256,
          7
        ],
        [
          256,
          5
        ],
        [
          256,
          3
        ]
      ],
      "fully_connected_layers": [
        1024,
        1024
      ],
      "threshold": 1e-06,
      "dropout_p": 0.1,
      "optimizer": "adam",
      "loss": "categorical_crossentropy"
    },
    "data_pars": {
      "data_info": {
        "dataset": "mlmodels/dataset/text/ag_news_csv",
        "train": true,
        "alphabet_size": 69,
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
      },
      "preprocessors": [
        {
          "name": "loader",
          "uri": "mlmodels/preprocess/generic.py::pandasDataset",
          "args": {
            "colX": [
              "colX"
            ],
            "coly": [
              "coly"
            ],
            "encoding": "'ISO-8859-1'",
            "read_csv_parm": {
              "usecols": [
                0,
                1
              ],
              "names": [
                "coly",
                "colX"
              ]
            }
          }
        },
        {
          "name": "tokenizer",
          "uri": "mlmodels/model_keras/raw/char_cnn/data_utils.py::Data",
          "args": {
            "data_source": "",
            "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
            "input_size": 1014,
            "num_of_classes": 4
          }
        }
      ]
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 128
    },
    "out_pars": {
      "path": "ztest/ml_keras/charcnn/charcnn.h5",
      "data_type": "pandas",
      "size": [
        0,
        0,
        6
      ],
      "output_size": [
        0,
        6
      ]
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.dataloader.charcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/dataloader/charcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 503, in test_dataloader
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 324, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "mlmodels/dataloader.py", line 78, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3313: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:3313: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
CharCNNKim model built: 
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sent_input (InputLayer)         (None, 1014)         0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 1014, 128)    8960        sent_input[0][0]                 
__________________________________________________________________________________________________
Conv1D_256_10 (Conv1D)          (None, 1005, 256)    327936      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_7 (Conv1D)           (None, 1008, 256)    229632      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_5 (Conv1D)           (None, 1010, 256)    164096      embedding_1[0][0]                
__________________________________________________________________________________________________
Conv1D_256_3 (Conv1D)           (None, 1012, 256)    98560       embedding_1[0][0]                
__________________________________________________________________________________________________
MaxPoolingOverTime_256_10 (Glob (None, 256)          0           Conv1D_256_10[0][0]              
__________________________________________________________________________________________________
MaxPoolingOverTime_256_7 (Globa (None, 256)          0           Conv1D_256_7[0][0]               
__________________________________________________________________________________________________
MaxPoolingOverTime_256_5 (Globa (None, 256)          0           Conv1D_256_5[0][0]               
__________________________________________________________________________________________________
MaxPoolingOverTime_256_3 (Globa (None, 256)          0           Conv1D_256_3[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1024)         0           MaxPoolingOverTime_256_10[0][0]  
                                                                 MaxPoolingOverTime_256_7[0][0]   
                                                                 MaxPoolingOverTime_256_5[0][0]   
                                                                 MaxPoolingOverTime_256_3[0][0]   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1024)         1049600     concatenate_1[0][0]              
__________________________________________________________________________________________________
alpha_dropout_1 (AlphaDropout)  (None, 1024)         0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         1049600     alpha_dropout_1[0][0]            
__________________________________________________________________________________________________
alpha_dropout_2 (AlphaDropout)  (None, 1024)         0           dense_2[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4)            4100        alpha_dropout_2[0][0]            
==================================================================================================
Total params: 2,932,484
Trainable params: 2,932,484
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.dataloader.charcnn.Model object at 0x7f0e4520c198> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.


 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 12s - loss: 1.6744
256/354 [====================>.........] - ETA: 5s - loss: 1.3482 
354/354 [==============================] - 24s 68ms/step - loss: 1.2060 - val_loss: 0.6937

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
[[1.8112642e-05 1.6432718e-05 1.2212619e-01 8.7783921e-01]
 [1.3829990e-05 1.1101216e-05 1.0677413e-01 8.9320093e-01]
 [1.9103119e-05 1.7255679e-05 1.3183378e-01 8.6812991e-01]
 ...
 [2.9218745e-05 2.4288887e-05 1.2241078e-01 8.7753570e-01]
 [3.3424851e-05 3.0289966e-05 1.3185532e-01 8.6808091e-01]
 [3.0359270e-05 2.6433954e-05 1.2908252e-01 8.7086064e-01]]

  #### Get  metrics   ################################################ 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/charcnn_zhang.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.dataloader.charcnn_zhang.py",
      "embedding_size": 128,
      "conv_layers": [
        [
          256,
          7,
          3
        ],
        [
          256,
          7,
          3
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          -1
        ],
        [
          256,
          3,
          3
        ]
      ],
      "fully_connected_layers": [
        1024,
        1024
      ],
      "threshold": 1e-06,
      "dropout_p": 0.5,
      "optimizer": "adam",
      "loss": "categorical_crossentropy"
    },
    "data_pars": {
      "data_info": {
        "dataset": "mlmodels/dataset/text/ag_news_csv",
        "train": true,
        "alphabet_size": 69,
        "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size": 1014,
        "num_of_classes": 4
      },
      "preprocessors": [
        {
          "name": "loader",
          "uri": "mlmodels/preprocess/generic.py::pandasDataset",
          "args": {
            "colX": [
              "colX"
            ],
            "coly": [
              "coly"
            ],
            "encoding": "'ISO-8859-1'",
            "read_csv_parm": {
              "usecols": [
                0,
                1
              ],
              "names": [
                "coly",
                "colX"
              ]
            }
          }
        },
        {
          "name": "tokenizer",
          "uri": "mlmodels/model_keras/raw/char_cnn/data_utils.py::Data",
          "args": {
            "data_source": "",
            "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
            "input_size": 1014,
            "num_of_classes": 4
          }
        }
      ]
    },
    "compute_pars": {
      "epochs": 1,
      "batch_size": 128
    },
    "out_pars": {
      "path": "ztest/ml_keras/charcnn_zhang/",
      "data_type": "pandas",
      "size": [
        0,
        0,
        6
      ],
      "output_size": [
        0,
        6
      ]
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  <module 'mlmodels.model_keras.dataloader.charcnn_zhang' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/dataloader/charcnn_zhang.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

CharCNNZhang model built: 
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sent_input (InputLayer)      (None, 1014)              0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 1014, 128)         8960      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1008, 256)         229632    
_________________________________________________________________
thresholded_re_lu_1 (Thresho (None, 1008, 256)         0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 336, 256)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 330, 256)          459008    
_________________________________________________________________
thresholded_re_lu_2 (Thresho (None, 330, 256)          0         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 110, 256)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 108, 256)          196864    
_________________________________________________________________
thresholded_re_lu_3 (Thresho (None, 108, 256)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 106, 256)          196864    
_________________________________________________________________
thresholded_re_lu_4 (Thresho (None, 106, 256)          0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 104, 256)          196864    
_________________________________________________________________
thresholded_re_lu_5 (Thresho (None, 104, 256)          0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 102, 256)          196864    
_________________________________________________________________
thresholded_re_lu_6 (Thresho (None, 102, 256)          0         
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 34, 256)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8704)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 1024)              8913920   
_________________________________________________________________
thresholded_re_lu_7 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
thresholded_re_lu_8 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 11,452,676
Trainable params: 11,452,676
Non-trainable params: 0
_________________________________________________________________

  <mlmodels.model_keras.dataloader.charcnn_zhang.Model object at 0x7f0e47f49c50> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 8s - loss: 1.3871
256/354 [====================>.........] - ETA: 3s - loss: 1.2813
354/354 [==============================] - 15s 42ms/step - loss: 1.2744 - val_loss: 0.7386

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/model_keras/raw/char_cnn/data_utils.py::Data {'data_source': '', 'alphabet': 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}', 'input_size': 1014, 'num_of_classes': 4} 

###### load_callable_from_uri LOADED <class 'mlmodels/model_keras/raw/char_cnn/data_utils.Data'>
cls_name : Data

 Object Creation

 Object Compute

 Object get_data
[[0.06904484 0.06650709 0.28834805 0.5761    ]
 [0.06895465 0.06640907 0.28821316 0.5764231 ]
 [0.06936935 0.06684992 0.28858143 0.5751993 ]
 ...
 [0.06979471 0.06728936 0.2889557  0.57396024]
 [0.06982934 0.06732082 0.28903985 0.57381004]
 [0.06983231 0.06732206 0.28903967 0.5738059 ]]

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 



 ####################################################################################################
dataset/json/refactor/keras_textcnn.json
{
  "test": {
    "model_pars": {
      "model_uri": "model_keras.dataloader.textcnn.py",
      "maxlen": 40,
      "max_features": 5,
      "embedding_dims": 50
    },
    "data_pars": {
      "data_info": {
        "dataset": "mlmodels/dataset/text/imdb",
        "pass_data_pars": false,
        "train": true,
        "maxlen": 40,
        "max_features": 5
      },
      "preprocessors": [
        {
          "name": "loader",
          "uri": "mlmodels/preprocess/generic.py::NumpyDataset",
          "args": {
            "numpy_loader_args": {
              "allow_pickle": true
            },
            "encoding": "'ISO-8859-1'"
          }
        }
      ]
    },
    "compute_pars": {
      "engine": "adam",
      "loss": "binary_crossentropy",
      "metrics": [
        "accuracy"
      ],
      "batch_size": 1000,
      "epochs": 1
    },
    "out_pars": {
      "path": "mlmodels/output/textcnn_keras//model.h5",
      "model_path": "mlmodels/output/textcnn_keras/model.h5"
    }
  },
  "prod": {
    "model_pars": {},
    "data_pars": {}
  }
}

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.dataloader.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/dataloader/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Model init   ############################################ 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 38, 128)      19328       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 37, 128)      25728       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 36, 128)      32128       embedding_3[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_7[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_8[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            385         concatenate_2[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.dataloader.textcnn.Model object at 0x7f0e4634a1d0> 

  #### Fit   ######################################################## 
Loading data...

  URL:  mlmodels/preprocess/generic.py::NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.NumpyDataset'>
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz
######## Error dataset/json/refactor/keras_textcnn.json Error when checking input: expected input_1 to have shape (40,) but got array with shape (1,)



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
        "data_path": "mlmodels/dataset/vision/MNIST",
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
      "checkpointdir": "ztest/model_tch/torchhub/restnet18/checkpoints/",
      "path": "ztest/model_tch/torchhub/restnet18/"
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

  <mlmodels.model_tch.torchhub.Model object at 0x7f0e47faceb8> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e2f90d8c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e2f90d8c8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e2f90d8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0018765816688537597 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.010582979202270509 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0013757106065750123 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.011461670160293579 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e2f90d6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e2f90d6a8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e2f90d6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 8, 0, 5, 0, 1, 3, 0, 3, 3]), array([4, 4, 2, 5, 4, 1, 3, 2, 1, 3]))

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
        "data_path": "mlmodels/dataset/vision/MNIST",
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
      "checkpointdir": "ztest/model_tch/torchhub/restnet34/checkpoints/",
      "path": "ztest/model_tch/torchhub/restnet34/"
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

  <mlmodels.model_tch.torchhub.Model object at 0x7f0e37d88390> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e2dd4ce18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e2dd4ce18>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e2dd4ce18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0023586355845133466 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.016932499408721925 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0014622490008672079 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03497331190109253 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e2dd4c6a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e2dd4c6a8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e2dd4c6a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]), array([9, 4, 9, 1, 8, 4, 7, 8, 4, 1]))

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
      "model": "resnet101",
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
      "epochs": 1,
      "learning_rate": 0.001
    },
    "out_pars": {
      "checkpointdir": "ztest/model_tch/torchhub/resnet101/checkpoints/",
      "path": "ztest/model_tch/torchhub/resnet101/"
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
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 403, in test_run_model
    test_module(config['test']['model_pars']['model_uri'], param_pars, fittable = False if x in not_fittable_models else True)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 266, in test_module
    model, sess = module.fit(model, data_pars, compute_pars, out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/dataloader/textcnn.py", line 77, in fit
    validation_data=(Xtest, ytest))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training.py", line 1154, in fit
    batch_size=batch_size)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training.py", line 579, in _standardize_user_data
    exception_prefix='input')
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/engine/training_utils.py", line 145, in standardize_input_data
    str(data_shape))
ValueError: Error when checking input: expected input_1 to have shape (40,) but got array with shape (1,)
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" to /home/runner/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth
  0%|          | 0.00/170M [00:00<?, ?B/s]  3%|▎         | 5.96M/170M [00:00<00:02, 61.7MB/s]  7%|▋         | 11.5M/170M [00:00<00:02, 60.6MB/s] 10%|█         | 17.2M/170M [00:00<00:02, 60.3MB/s] 12%|█▏        | 20.9M/170M [00:00<00:03, 50.3MB/s] 16%|█▋        | 28.0M/170M [00:00<00:02, 55.7MB/s] 20%|██        | 34.6M/170M [00:00<00:02, 59.2MB/s] 23%|██▎       | 39.9M/170M [00:00<00:02, 50.7MB/s] 26%|██▌       | 44.7M/170M [00:00<00:02, 47.5MB/s] 29%|██▉       | 49.2M/170M [00:00<00:02, 45.0MB/s] 32%|███▏      | 53.8M/170M [00:01<00:02, 45.8MB/s] 35%|███▍      | 59.3M/170M [00:01<00:02, 48.9MB/s] 38%|███▊      | 64.3M/170M [00:01<00:02, 49.9MB/s] 42%|████▏     | 70.7M/170M [00:01<00:01, 54.0MB/s] 45%|████▍     | 76.0M/170M [00:01<00:02, 47.3MB/s] 49%|████▊     | 83.0M/170M [00:01<00:01, 52.8MB/s] 53%|█████▎    | 89.6M/170M [00:01<00:01, 56.9MB/s] 56%|█████▌    | 95.7M/170M [00:01<00:01, 57.8MB/s] 60%|█████▉    | 102M/170M [00:01<00:01, 58.7MB/s]  63%|██████▎   | 107M/170M [00:02<00:01, 56.3MB/s] 67%|██████▋   | 113M/170M [00:02<00:01, 58.3MB/s] 70%|██████▉   | 119M/170M [00:02<00:00, 58.5MB/s] 73%|███████▎  | 125M/170M [00:02<00:00, 53.8MB/s] 76%|███████▋  | 130M/170M [00:02<00:00, 50.1MB/s] 79%|███████▉  | 135M/170M [00:02<00:00, 47.2MB/s] 82%|████████▏ | 140M/170M [00:02<00:00, 46.8MB/s] 85%|████████▌ | 145M/170M [00:02<00:00, 49.8MB/s] 89%|████████▊ | 151M/170M [00:02<00:00, 52.3MB/s] 92%|█████████▏| 156M/170M [00:03<00:00, 45.9MB/s] 95%|█████████▌| 163M/170M [00:03<00:00, 45.0MB/s] 99%|█████████▊| 168M/170M [00:03<00:00, 48.6MB/s]100%|██████████| 170M/170M [00:03<00:00, 52.0MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f0e2eb8ff98> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0e2df140d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0e2df140d0>

  function with postional parmater data_info <function tf_dataset_download at 0x7f0e2df140d0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e2df01510>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e2df01510>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e2df01510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.5330309247970582 	 Accuracy: 30
Train Epoch: 1 	 Loss: 214.97723388671875 	 Accuracy: 100
model saves at 100 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0e2df388c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0e2df388c8>

  function with postional parmater data_info <function tf_dataset_download at 0x7f0e2df388c8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0e2fdf76a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0e2fdf76a8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f0e2fdf76a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]), array([7, 2, 5, 2, 9, 0, 0, 7, 8, 1]))

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
        "data_path": "mlmodels/dataset/vision/MNIST",
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

  <mlmodels.model_tch.torchhub.Model object at 0x7f0e47faceb8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
img_01.png
0

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
