
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_dataloader 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51091e23d3d59de539ae132a643e07bbde912e52', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '51091e23d3d59de539ae132a643e07bbde912e52', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51091e23d3d59de539ae132a643e07bbde912e52

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51091e23d3d59de539ae132a643e07bbde912e52

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51091e23d3d59de539ae132a643e07bbde912e52

 ************************************************************************************************************************

  ############Check model ################################ 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  
['/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_textcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//namentity_crm_bilstm_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet34_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//matchzoo_models_new.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//model_list_CIFAR.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//charcnn_zhang.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//keras_reuters_charcnn.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//armdn_dataloader.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//resnet18_benchmark_mnist.json', '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor//torchhub_cnn_dataloader.json']





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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92e918a0d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92e918a0d0>

  function with postional parmater data_info <function get_dataset_torch at 0x7f92e918a0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Using TensorFlow backend.
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 158743.98it/s] 71%|███████   | 7045120/9912422 [00:00<00:12, 226555.29it/s]9920512it [00:00, 43038522.10it/s]                           
0it [00:00, ?it/s]32768it [00:00, 632787.68it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 482197.95it/s]1654784it [00:00, 12161417.93it/s]                         
0it [00:00, ?it/s]8192it [00:00, 197816.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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
((<torch.utils.data.dataloader.DataLoader object at 0x7f92d239a9e8>, <torch.utils.data.dataloader.DataLoader object at 0x7f92d23a5da0>), {})





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

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f92d23c1268>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f92d23c1268>

  function with postional parmater data_info <function tf_dataset_download at 0x7f92d23c1268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:38,  1.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:38,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:38,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:37,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:36,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:36,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:35,  1.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:06,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:05,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:04,  2.30 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:45,  3.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:44,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:44,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:31,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:30,  4.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:01<00:21,  6.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:15,  8.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:15,  8.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:15,  8.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:15,  8.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:14,  8.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:14,  8.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:14,  8.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 14.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 33.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 33.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 33.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 33.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 33.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 33.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:02, 37.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 41.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 41.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 44.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 44.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 44.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 44.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 44.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 44.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 44.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 47.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 47.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 47.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 47.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 47.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 47.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 47.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 48.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 48.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 48.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 48.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 48.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 48.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 48.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 50.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 50.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 50.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 50.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 50.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 51.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 51.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 51.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 51.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 51.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 51.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 52.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 52.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 52.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 53.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 53.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 53.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 53.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 53.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 53.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 53.58 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 53.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 53.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 53.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 53.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 53.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 53.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 53.80 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 53.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 53.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 53.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 53.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 53.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 53.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 53.75 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 54.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 54.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 54.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 54.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 54.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 54.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 54.00 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 54.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.54s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 54.26 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.85s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 54.26 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.85s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.85s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 27.70 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.85s/ url]
0 examples [00:00, ? examples/s]2020-05-21 04:30:57.703960: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 04:30:57.708190: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-05-21 04:30:57.708335: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561c554acbf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 04:30:57.708353: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
42 examples [00:00, 417.21 examples/s]140 examples [00:00, 503.56 examples/s]239 examples [00:00, 589.57 examples/s]336 examples [00:00, 667.81 examples/s]437 examples [00:00, 742.33 examples/s]538 examples [00:00, 804.88 examples/s]638 examples [00:00, 854.91 examples/s]731 examples [00:00, 874.84 examples/s]831 examples [00:00, 907.42 examples/s]931 examples [00:01, 932.96 examples/s]1027 examples [00:01, 940.45 examples/s]1129 examples [00:01, 961.41 examples/s]1231 examples [00:01, 976.07 examples/s]1332 examples [00:01, 984.24 examples/s]1433 examples [00:01, 990.04 examples/s]1533 examples [00:01, 986.87 examples/s]1632 examples [00:01, 987.43 examples/s]1731 examples [00:01, 976.50 examples/s]1830 examples [00:01, 978.28 examples/s]1928 examples [00:02, 978.31 examples/s]2026 examples [00:02, 932.09 examples/s]2125 examples [00:02, 946.39 examples/s]2221 examples [00:02, 946.49 examples/s]2316 examples [00:02, 928.22 examples/s]2417 examples [00:02, 948.79 examples/s]2514 examples [00:02, 952.93 examples/s]2612 examples [00:02, 960.04 examples/s]2709 examples [00:02, 958.75 examples/s]2805 examples [00:02, 955.75 examples/s]2903 examples [00:03, 959.43 examples/s]3000 examples [00:03, 938.16 examples/s]3097 examples [00:03, 945.62 examples/s]3194 examples [00:03, 950.15 examples/s]3292 examples [00:03, 956.34 examples/s]3390 examples [00:03, 961.41 examples/s]3487 examples [00:03, 958.43 examples/s]3586 examples [00:03, 965.14 examples/s]3683 examples [00:03, 948.81 examples/s]3778 examples [00:03, 944.45 examples/s]3876 examples [00:04, 952.32 examples/s]3972 examples [00:04, 953.23 examples/s]4071 examples [00:04, 961.26 examples/s]4169 examples [00:04, 965.60 examples/s]4266 examples [00:04, 955.46 examples/s]4364 examples [00:04, 960.15 examples/s]4461 examples [00:04, 954.23 examples/s]4557 examples [00:04, 948.93 examples/s]4653 examples [00:04, 950.07 examples/s]4755 examples [00:04, 968.37 examples/s]4858 examples [00:05, 985.49 examples/s]4957 examples [00:05, 977.39 examples/s]5059 examples [00:05, 989.27 examples/s]5159 examples [00:05, 990.31 examples/s]5259 examples [00:05, 954.29 examples/s]5357 examples [00:05, 960.26 examples/s]5454 examples [00:05, 960.12 examples/s]5555 examples [00:05, 973.09 examples/s]5656 examples [00:05, 982.56 examples/s]5757 examples [00:06, 988.93 examples/s]5859 examples [00:06, 997.09 examples/s]5959 examples [00:06, 991.81 examples/s]6060 examples [00:06, 995.36 examples/s]6160 examples [00:06, 965.91 examples/s]6260 examples [00:06, 974.36 examples/s]6361 examples [00:06, 983.18 examples/s]6462 examples [00:06, 988.70 examples/s]6565 examples [00:06, 998.27 examples/s]6667 examples [00:06, 1002.27 examples/s]6769 examples [00:07, 1007.39 examples/s]6871 examples [00:07, 1009.21 examples/s]6972 examples [00:07, 995.14 examples/s] 7072 examples [00:07, 988.04 examples/s]7174 examples [00:07, 996.69 examples/s]7276 examples [00:07, 1002.87 examples/s]7377 examples [00:07, 1001.48 examples/s]7478 examples [00:07, 1001.36 examples/s]7579 examples [00:07, 959.95 examples/s] 7676 examples [00:07, 953.50 examples/s]7777 examples [00:08, 968.19 examples/s]7875 examples [00:08, 956.20 examples/s]7973 examples [00:08, 961.24 examples/s]8070 examples [00:08, 962.22 examples/s]8168 examples [00:08, 964.80 examples/s]8265 examples [00:08, 932.39 examples/s]8359 examples [00:08, 933.88 examples/s]8460 examples [00:08, 953.60 examples/s]8561 examples [00:08, 968.80 examples/s]8662 examples [00:08, 979.29 examples/s]8763 examples [00:09, 985.50 examples/s]8862 examples [00:09, 975.93 examples/s]8960 examples [00:09, 976.76 examples/s]9058 examples [00:09, 976.71 examples/s]9156 examples [00:09, 970.91 examples/s]9255 examples [00:09, 974.13 examples/s]9353 examples [00:09, 969.98 examples/s]9451 examples [00:09, 957.77 examples/s]9548 examples [00:09, 959.02 examples/s]9649 examples [00:09, 971.98 examples/s]9748 examples [00:10, 976.63 examples/s]9848 examples [00:10, 983.37 examples/s]9947 examples [00:10, 985.25 examples/s]10046 examples [00:10, 858.72 examples/s]10149 examples [00:10, 901.81 examples/s]10248 examples [00:10, 926.49 examples/s]10349 examples [00:10, 949.89 examples/s]10450 examples [00:10, 965.99 examples/s]10553 examples [00:10, 982.38 examples/s]10657 examples [00:11, 996.53 examples/s]10760 examples [00:11, 1004.29 examples/s]10862 examples [00:11, 1006.38 examples/s]10963 examples [00:11, 1007.01 examples/s]11064 examples [00:11, 972.67 examples/s] 11162 examples [00:11, 960.74 examples/s]11263 examples [00:11, 974.21 examples/s]11361 examples [00:11, 971.22 examples/s]11463 examples [00:11, 983.39 examples/s]11563 examples [00:11, 986.18 examples/s]11665 examples [00:12, 993.37 examples/s]11766 examples [00:12, 997.81 examples/s]11866 examples [00:12, 995.60 examples/s]11967 examples [00:12, 998.98 examples/s]12068 examples [00:12, 1001.66 examples/s]12169 examples [00:12, 994.06 examples/s] 12269 examples [00:12, 975.30 examples/s]12367 examples [00:12, 970.28 examples/s]12465 examples [00:12, 972.69 examples/s]12563 examples [00:12, 970.91 examples/s]12661 examples [00:13, 958.56 examples/s]12757 examples [00:13, 950.23 examples/s]12853 examples [00:13, 949.87 examples/s]12949 examples [00:13, 952.86 examples/s]13047 examples [00:13, 959.78 examples/s]13145 examples [00:13, 964.36 examples/s]13244 examples [00:13, 969.93 examples/s]13342 examples [00:13, 959.59 examples/s]13441 examples [00:13, 966.90 examples/s]13539 examples [00:14, 967.50 examples/s]13638 examples [00:14, 971.32 examples/s]13736 examples [00:14, 972.38 examples/s]13834 examples [00:14, 957.79 examples/s]13932 examples [00:14, 948.12 examples/s]14030 examples [00:14, 956.95 examples/s]14126 examples [00:14, 956.44 examples/s]14226 examples [00:14, 967.04 examples/s]14323 examples [00:14, 936.47 examples/s]14421 examples [00:14, 948.17 examples/s]14517 examples [00:15, 949.45 examples/s]14615 examples [00:15, 957.19 examples/s]14711 examples [00:15, 956.93 examples/s]14807 examples [00:15, 949.37 examples/s]14903 examples [00:15, 950.41 examples/s]15002 examples [00:15, 960.24 examples/s]15103 examples [00:15, 974.55 examples/s]15206 examples [00:15, 990.37 examples/s]15306 examples [00:15, 993.22 examples/s]15408 examples [00:15, 1000.57 examples/s]15510 examples [00:16, 1006.24 examples/s]15613 examples [00:16, 1011.92 examples/s]15715 examples [00:16, 1012.61 examples/s]15817 examples [00:16, 1000.16 examples/s]15918 examples [00:16, 986.63 examples/s] 16017 examples [00:16, 980.99 examples/s]16118 examples [00:16, 988.66 examples/s]16219 examples [00:16, 994.86 examples/s]16319 examples [00:16, 974.45 examples/s]16418 examples [00:16, 977.57 examples/s]16516 examples [00:17, 976.73 examples/s]16614 examples [00:17, 967.15 examples/s]16712 examples [00:17, 969.19 examples/s]16809 examples [00:17, 936.57 examples/s]16906 examples [00:17, 943.72 examples/s]17005 examples [00:17, 955.98 examples/s]17101 examples [00:17, 936.07 examples/s]17195 examples [00:17, 933.42 examples/s]17291 examples [00:17, 939.12 examples/s]17392 examples [00:17, 958.64 examples/s]17489 examples [00:18, 940.01 examples/s]17585 examples [00:18, 944.77 examples/s]17686 examples [00:18, 960.70 examples/s]17785 examples [00:18, 967.55 examples/s]17885 examples [00:18, 976.88 examples/s]17987 examples [00:18, 987.04 examples/s]18089 examples [00:18, 994.06 examples/s]18190 examples [00:18, 997.61 examples/s]18290 examples [00:18, 991.84 examples/s]18390 examples [00:19, 977.14 examples/s]18491 examples [00:19, 985.68 examples/s]18590 examples [00:19, 982.20 examples/s]18690 examples [00:19, 987.28 examples/s]18789 examples [00:19, 981.76 examples/s]18889 examples [00:19, 986.53 examples/s]18990 examples [00:19, 992.25 examples/s]19090 examples [00:19, 957.48 examples/s]19187 examples [00:19, 953.60 examples/s]19285 examples [00:19, 959.25 examples/s]19386 examples [00:20, 972.75 examples/s]19488 examples [00:20, 983.81 examples/s]19590 examples [00:20, 993.54 examples/s]19692 examples [00:20, 998.46 examples/s]19792 examples [00:20, 972.60 examples/s]19892 examples [00:20, 980.35 examples/s]19993 examples [00:20, 986.74 examples/s]20092 examples [00:20, 832.77 examples/s]20191 examples [00:20, 873.14 examples/s]20292 examples [00:21, 908.52 examples/s]20391 examples [00:21, 931.47 examples/s]20492 examples [00:21, 953.14 examples/s]20592 examples [00:21, 964.37 examples/s]20691 examples [00:21, 971.21 examples/s]20790 examples [00:21, 975.16 examples/s]20890 examples [00:21, 982.48 examples/s]20992 examples [00:21, 990.79 examples/s]21092 examples [00:21, 991.70 examples/s]21192 examples [00:21, 991.72 examples/s]21292 examples [00:22, 987.97 examples/s]21391 examples [00:22, 977.80 examples/s]21489 examples [00:22, 963.65 examples/s]21588 examples [00:22, 969.46 examples/s]21686 examples [00:22, 967.90 examples/s]21783 examples [00:22, 955.89 examples/s]21883 examples [00:22, 968.52 examples/s]21984 examples [00:22, 979.00 examples/s]22083 examples [00:22, 980.54 examples/s]22185 examples [00:22, 989.68 examples/s]22286 examples [00:23, 993.11 examples/s]22387 examples [00:23, 995.72 examples/s]22487 examples [00:23, 989.65 examples/s]22589 examples [00:23, 996.01 examples/s]22689 examples [00:23, 994.14 examples/s]22789 examples [00:23, 977.49 examples/s]22887 examples [00:23, 975.76 examples/s]22985 examples [00:23, 974.07 examples/s]23083 examples [00:23, 948.70 examples/s]23182 examples [00:23, 960.59 examples/s]23279 examples [00:24, 955.12 examples/s]23381 examples [00:24, 972.63 examples/s]23482 examples [00:24, 982.26 examples/s]23582 examples [00:24, 987.42 examples/s]23681 examples [00:24, 972.33 examples/s]23779 examples [00:24, 956.55 examples/s]23875 examples [00:24, 948.94 examples/s]23971 examples [00:24, 936.87 examples/s]24072 examples [00:24, 955.36 examples/s]24173 examples [00:24, 969.96 examples/s]24274 examples [00:25, 981.27 examples/s]24376 examples [00:25, 990.27 examples/s]24477 examples [00:25, 994.83 examples/s]24577 examples [00:25, 993.86 examples/s]24677 examples [00:25, 984.34 examples/s]24776 examples [00:25, 979.21 examples/s]24877 examples [00:25, 986.74 examples/s]24978 examples [00:25, 993.57 examples/s]25078 examples [00:25, 984.79 examples/s]25178 examples [00:25, 988.02 examples/s]25277 examples [00:26, 957.42 examples/s]25378 examples [00:26, 971.08 examples/s]25479 examples [00:26, 980.14 examples/s]25578 examples [00:26, 981.24 examples/s]25677 examples [00:26, 945.81 examples/s]25773 examples [00:26, 950.01 examples/s]25871 examples [00:26, 958.31 examples/s]25968 examples [00:26, 949.54 examples/s]26064 examples [00:26, 914.85 examples/s]26160 examples [00:27, 926.67 examples/s]26257 examples [00:27, 937.23 examples/s]26354 examples [00:27, 945.06 examples/s]26452 examples [00:27, 953.29 examples/s]26548 examples [00:27, 954.42 examples/s]26645 examples [00:27, 956.82 examples/s]26745 examples [00:27, 969.04 examples/s]26847 examples [00:27, 983.17 examples/s]26946 examples [00:27, 970.78 examples/s]27046 examples [00:27, 976.96 examples/s]27144 examples [00:28, 965.41 examples/s]27241 examples [00:28, 941.08 examples/s]27336 examples [00:28, 937.34 examples/s]27434 examples [00:28, 949.38 examples/s]27530 examples [00:28, 947.48 examples/s]27629 examples [00:28, 957.22 examples/s]27728 examples [00:28, 965.28 examples/s]27828 examples [00:28, 973.97 examples/s]27926 examples [00:28, 964.61 examples/s]28027 examples [00:28, 977.30 examples/s]28125 examples [00:29, 973.48 examples/s]28225 examples [00:29, 979.46 examples/s]28324 examples [00:29, 948.84 examples/s]28420 examples [00:29, 944.00 examples/s]28515 examples [00:29, 937.80 examples/s]28611 examples [00:29, 943.52 examples/s]28709 examples [00:29, 952.52 examples/s]28805 examples [00:29, 937.37 examples/s]28900 examples [00:29, 938.65 examples/s]28997 examples [00:29, 946.84 examples/s]29092 examples [00:30, 947.47 examples/s]29188 examples [00:30, 944.96 examples/s]29283 examples [00:30, 893.16 examples/s]29373 examples [00:30, 882.00 examples/s]29466 examples [00:30, 893.90 examples/s]29563 examples [00:30, 913.95 examples/s]29662 examples [00:30, 934.39 examples/s]29762 examples [00:30, 951.45 examples/s]29862 examples [00:30, 963.35 examples/s]29959 examples [00:31, 930.63 examples/s]30053 examples [00:31, 845.56 examples/s]30148 examples [00:31, 873.37 examples/s]30238 examples [00:31, 879.46 examples/s]30331 examples [00:31, 892.96 examples/s]30427 examples [00:31, 911.22 examples/s]30522 examples [00:31, 921.30 examples/s]30616 examples [00:31, 924.60 examples/s]30711 examples [00:31, 930.91 examples/s]30809 examples [00:31, 942.50 examples/s]30906 examples [00:32, 948.11 examples/s]31004 examples [00:32, 954.87 examples/s]31106 examples [00:32, 971.31 examples/s]31206 examples [00:32, 978.21 examples/s]31304 examples [00:32, 975.55 examples/s]31403 examples [00:32, 978.33 examples/s]31501 examples [00:32, 969.34 examples/s]31598 examples [00:32, 968.18 examples/s]31695 examples [00:32, 954.17 examples/s]31793 examples [00:32, 959.21 examples/s]31889 examples [00:33, 939.10 examples/s]31984 examples [00:33, 935.90 examples/s]32081 examples [00:33, 943.98 examples/s]32181 examples [00:33, 959.69 examples/s]32281 examples [00:33, 970.53 examples/s]32382 examples [00:33, 979.90 examples/s]32481 examples [00:33, 959.35 examples/s]32584 examples [00:33, 977.04 examples/s]32687 examples [00:33, 990.38 examples/s]32787 examples [00:34, 989.28 examples/s]32889 examples [00:34, 996.46 examples/s]32989 examples [00:34, 973.41 examples/s]33092 examples [00:34, 989.03 examples/s]33194 examples [00:34, 996.12 examples/s]33297 examples [00:34, 1004.14 examples/s]33398 examples [00:34, 989.76 examples/s] 33498 examples [00:34, 968.74 examples/s]33596 examples [00:34, 963.47 examples/s]33693 examples [00:34, 959.85 examples/s]33792 examples [00:35, 967.46 examples/s]33890 examples [00:35, 969.02 examples/s]33987 examples [00:35, 962.46 examples/s]34086 examples [00:35, 968.77 examples/s]34183 examples [00:35, 932.35 examples/s]34284 examples [00:35, 953.44 examples/s]34383 examples [00:35, 962.72 examples/s]34481 examples [00:35, 967.28 examples/s]34580 examples [00:35, 972.79 examples/s]34680 examples [00:35, 978.16 examples/s]34778 examples [00:36, 977.51 examples/s]34878 examples [00:36, 983.10 examples/s]34977 examples [00:36, 980.70 examples/s]35077 examples [00:36, 984.95 examples/s]35176 examples [00:36, 982.44 examples/s]35275 examples [00:36, 964.16 examples/s]35372 examples [00:36, 965.60 examples/s]35469 examples [00:36, 956.51 examples/s]35565 examples [00:36, 947.53 examples/s]35663 examples [00:36, 956.77 examples/s]35761 examples [00:37, 961.32 examples/s]35858 examples [00:37, 961.64 examples/s]35955 examples [00:37, 945.27 examples/s]36052 examples [00:37, 952.22 examples/s]36153 examples [00:37, 966.45 examples/s]36255 examples [00:37, 980.96 examples/s]36355 examples [00:37, 986.55 examples/s]36456 examples [00:37, 991.95 examples/s]36556 examples [00:37, 990.69 examples/s]36659 examples [00:37, 1001.70 examples/s]36760 examples [00:38, 988.24 examples/s] 36862 examples [00:38, 995.92 examples/s]36962 examples [00:38, 995.58 examples/s]37064 examples [00:38, 1002.31 examples/s]37165 examples [00:38, 1002.84 examples/s]37269 examples [00:38, 1010.97 examples/s]37372 examples [00:38, 1014.39 examples/s]37474 examples [00:38, 1004.20 examples/s]37575 examples [00:38, 985.21 examples/s] 37678 examples [00:39, 997.10 examples/s]37778 examples [00:39, 987.96 examples/s]37877 examples [00:39, 982.26 examples/s]37976 examples [00:39, 956.20 examples/s]38072 examples [00:39, 957.12 examples/s]38170 examples [00:39, 962.97 examples/s]38268 examples [00:39, 966.71 examples/s]38365 examples [00:39, 964.14 examples/s]38462 examples [00:39, 961.46 examples/s]38560 examples [00:39, 964.98 examples/s]38657 examples [00:40, 945.19 examples/s]38759 examples [00:40, 964.05 examples/s]38861 examples [00:40, 976.47 examples/s]38960 examples [00:40, 979.66 examples/s]39059 examples [00:40, 976.84 examples/s]39162 examples [00:40, 990.51 examples/s]39265 examples [00:40, 999.23 examples/s]39367 examples [00:40, 1004.69 examples/s]39468 examples [00:40, 998.47 examples/s] 39571 examples [00:40, 1004.85 examples/s]39673 examples [00:41, 1008.57 examples/s]39776 examples [00:41, 1012.48 examples/s]39878 examples [00:41, 989.88 examples/s] 39978 examples [00:41, 971.78 examples/s]40076 examples [00:41, 867.27 examples/s]40174 examples [00:41, 896.33 examples/s]40272 examples [00:41, 919.33 examples/s]40367 examples [00:41, 926.90 examples/s]40468 examples [00:41, 949.37 examples/s]40567 examples [00:42, 960.39 examples/s]40669 examples [00:42, 974.91 examples/s]40770 examples [00:42, 984.68 examples/s]40869 examples [00:42, 984.15 examples/s]40968 examples [00:42, 981.89 examples/s]41070 examples [00:42, 991.20 examples/s]41172 examples [00:42, 997.78 examples/s]41273 examples [00:42, 1000.49 examples/s]41374 examples [00:42, 984.40 examples/s] 41473 examples [00:42, 979.10 examples/s]41571 examples [00:43, 936.06 examples/s]41671 examples [00:43, 953.27 examples/s]41773 examples [00:43, 971.71 examples/s]41872 examples [00:43, 974.72 examples/s]41973 examples [00:43, 983.61 examples/s]42076 examples [00:43, 995.78 examples/s]42177 examples [00:43, 997.25 examples/s]42278 examples [00:43, 998.52 examples/s]42379 examples [00:43, 1001.90 examples/s]42480 examples [00:43, 1002.52 examples/s]42581 examples [00:44, 995.86 examples/s] 42684 examples [00:44, 1004.53 examples/s]42788 examples [00:44, 1012.40 examples/s]42890 examples [00:44, 1009.98 examples/s]42992 examples [00:44, 994.46 examples/s] 43093 examples [00:44, 998.47 examples/s]43194 examples [00:44, 1000.94 examples/s]43295 examples [00:44, 1001.07 examples/s]43396 examples [00:44, 996.94 examples/s] 43497 examples [00:44, 998.17 examples/s]43598 examples [00:45, 1000.69 examples/s]43699 examples [00:45, 997.72 examples/s] 43800 examples [00:45, 1000.02 examples/s]43901 examples [00:45, 983.23 examples/s] 44000 examples [00:45, 972.85 examples/s]44101 examples [00:45, 981.43 examples/s]44202 examples [00:45, 987.80 examples/s]44303 examples [00:45, 992.71 examples/s]44403 examples [00:45, 989.66 examples/s]44503 examples [00:45, 954.57 examples/s]44599 examples [00:46, 923.46 examples/s]44696 examples [00:46, 936.73 examples/s]44794 examples [00:46, 946.91 examples/s]44889 examples [00:46, 927.03 examples/s]44986 examples [00:46, 937.08 examples/s]45083 examples [00:46, 944.51 examples/s]45179 examples [00:46, 946.32 examples/s]45274 examples [00:46, 947.01 examples/s]45369 examples [00:46, 930.65 examples/s]45470 examples [00:47, 951.47 examples/s]45572 examples [00:47, 968.49 examples/s]45674 examples [00:47, 981.80 examples/s]45775 examples [00:47, 989.17 examples/s]45875 examples [00:47, 969.25 examples/s]45975 examples [00:47, 975.02 examples/s]46073 examples [00:47, 966.12 examples/s]46170 examples [00:47, 943.97 examples/s]46267 examples [00:47, 949.12 examples/s]46366 examples [00:47, 959.76 examples/s]46463 examples [00:48, 945.07 examples/s]46558 examples [00:48, 929.69 examples/s]46655 examples [00:48, 939.08 examples/s]46750 examples [00:48, 941.75 examples/s]46845 examples [00:48, 938.52 examples/s]46943 examples [00:48, 948.92 examples/s]47041 examples [00:48, 956.93 examples/s]47137 examples [00:48, 954.20 examples/s]47234 examples [00:48, 958.76 examples/s]47330 examples [00:48, 949.28 examples/s]47426 examples [00:49, 951.38 examples/s]47522 examples [00:49, 912.20 examples/s]47619 examples [00:49, 926.85 examples/s]47712 examples [00:49, 921.06 examples/s]47807 examples [00:49, 927.87 examples/s]47907 examples [00:49, 946.36 examples/s]48008 examples [00:49, 962.04 examples/s]48109 examples [00:49, 973.48 examples/s]48207 examples [00:49, 953.80 examples/s]48305 examples [00:49, 959.14 examples/s]48402 examples [00:50, 957.41 examples/s]48499 examples [00:50, 960.77 examples/s]48596 examples [00:50, 962.17 examples/s]48693 examples [00:50, 952.29 examples/s]48792 examples [00:50, 962.48 examples/s]48890 examples [00:50, 965.32 examples/s]48992 examples [00:50, 979.93 examples/s]49091 examples [00:50, 965.49 examples/s]49189 examples [00:50, 967.82 examples/s]49291 examples [00:51, 981.94 examples/s]49390 examples [00:51, 961.88 examples/s]49488 examples [00:51, 965.41 examples/s]49585 examples [00:51, 962.24 examples/s]49682 examples [00:51, 948.87 examples/s]49781 examples [00:51, 958.32 examples/s]49880 examples [00:51, 967.13 examples/s]49977 examples [00:51, 957.39 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6795/50000 [00:00<00:00, 67944.40 examples/s] 35%|███▌      | 17628/50000 [00:00<00:00, 76499.92 examples/s] 56%|█████▌    | 28008/50000 [00:00<00:00, 83051.89 examples/s] 78%|███████▊  | 39000/50000 [00:00<00:00, 89623.83 examples/s] 99%|█████████▉| 49741/50000 [00:00<00:00, 94306.68 examples/s]                                                               0 examples [00:00, ? examples/s]56 examples [00:00, 557.35 examples/s]140 examples [00:00, 619.39 examples/s]241 examples [00:00, 700.25 examples/s]343 examples [00:00, 771.10 examples/s]440 examples [00:00, 821.05 examples/s]541 examples [00:00, 868.09 examples/s]635 examples [00:00, 887.53 examples/s]733 examples [00:00, 910.66 examples/s]830 examples [00:00, 925.14 examples/s]929 examples [00:01, 941.93 examples/s]1028 examples [00:01, 953.17 examples/s]1124 examples [00:01, 938.95 examples/s]1222 examples [00:01, 950.26 examples/s]1320 examples [00:01, 957.09 examples/s]1418 examples [00:01, 962.14 examples/s]1517 examples [00:01, 968.37 examples/s]1614 examples [00:01, 962.81 examples/s]1711 examples [00:01, 959.50 examples/s]1809 examples [00:01, 962.69 examples/s]1906 examples [00:02, 963.99 examples/s]2004 examples [00:02, 966.29 examples/s]2101 examples [00:02, 962.36 examples/s]2199 examples [00:02, 965.16 examples/s]2296 examples [00:02, 943.63 examples/s]2393 examples [00:02, 951.08 examples/s]2495 examples [00:02, 967.99 examples/s]2593 examples [00:02, 968.85 examples/s]2692 examples [00:02, 972.39 examples/s]2790 examples [00:02, 958.42 examples/s]2886 examples [00:03, 945.17 examples/s]2984 examples [00:03, 954.00 examples/s]3080 examples [00:03, 944.28 examples/s]3178 examples [00:03, 954.46 examples/s]3276 examples [00:03, 961.67 examples/s]3373 examples [00:03, 949.00 examples/s]3470 examples [00:03, 954.88 examples/s]3569 examples [00:03, 963.20 examples/s]3672 examples [00:03, 979.72 examples/s]3773 examples [00:03, 988.42 examples/s]3874 examples [00:04, 993.44 examples/s]3974 examples [00:04, 971.03 examples/s]4073 examples [00:04, 974.96 examples/s]4175 examples [00:04, 987.71 examples/s]4278 examples [00:04, 997.92 examples/s]4381 examples [00:04, 1005.21 examples/s]4484 examples [00:04, 1012.23 examples/s]4586 examples [00:04, 1009.02 examples/s]4690 examples [00:04, 1016.95 examples/s]4793 examples [00:04, 1020.42 examples/s]4896 examples [00:05, 1021.12 examples/s]4999 examples [00:05, 1011.91 examples/s]5101 examples [00:05, 1007.81 examples/s]5205 examples [00:05, 1016.69 examples/s]5308 examples [00:05, 1018.38 examples/s]5411 examples [00:05, 1020.98 examples/s]5514 examples [00:05, 1022.84 examples/s]5617 examples [00:05, 1001.71 examples/s]5718 examples [00:05, 996.71 examples/s] 5818 examples [00:05, 968.66 examples/s]5916 examples [00:06, 968.84 examples/s]6017 examples [00:06, 979.03 examples/s]6118 examples [00:06, 986.11 examples/s]6222 examples [00:06, 999.64 examples/s]6324 examples [00:06, 1004.48 examples/s]6426 examples [00:06, 1006.50 examples/s]6529 examples [00:06, 1013.32 examples/s]6631 examples [00:06, 1012.48 examples/s]6733 examples [00:06, 1013.49 examples/s]6836 examples [00:07, 1014.57 examples/s]6938 examples [00:07, 1014.62 examples/s]7040 examples [00:07, 1007.23 examples/s]7141 examples [00:07, 1002.34 examples/s]7242 examples [00:07, 976.69 examples/s] 7340 examples [00:07, 925.60 examples/s]7437 examples [00:07, 937.20 examples/s]7535 examples [00:07, 947.58 examples/s]7631 examples [00:07, 948.97 examples/s]7730 examples [00:07, 959.32 examples/s]7828 examples [00:08, 965.17 examples/s]7926 examples [00:08, 966.66 examples/s]8024 examples [00:08, 968.03 examples/s]8122 examples [00:08, 971.25 examples/s]8220 examples [00:08, 972.58 examples/s]8320 examples [00:08, 979.34 examples/s]8418 examples [00:08, 970.63 examples/s]8519 examples [00:08, 979.38 examples/s]8618 examples [00:08, 981.51 examples/s]8717 examples [00:08, 982.45 examples/s]8816 examples [00:09, 942.63 examples/s]8912 examples [00:09, 947.05 examples/s]9007 examples [00:09, 932.25 examples/s]9104 examples [00:09, 941.36 examples/s]9206 examples [00:09, 961.75 examples/s]9308 examples [00:09, 977.53 examples/s]9406 examples [00:09, 967.89 examples/s]9504 examples [00:09, 970.36 examples/s]9602 examples [00:09, 954.69 examples/s]9703 examples [00:09, 970.35 examples/s]9806 examples [00:10, 984.70 examples/s]9905 examples [00:10, 985.00 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0ASOGZ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0ASOGZ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92d56592f0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92d56592f0>

  function with postional parmater data_info <function get_dataset_torch at 0x7f92d56592f0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f92d27bdc18>, <torch.utils.data.dataloader.DataLoader object at 0x7f92a434fef0>), {})





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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92a77c3510>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92a77c3510>

  function with postional parmater data_info <function get_dataset_torch at 0x7f92a77c3510> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f92a77c06a0>, <torch.utils.data.dataloader.DataLoader object at 0x7f92a77c0748>), {})





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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f92a77c3510>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f92a77c3510>

  URL:  sklearn.model_selection::train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f933b90cb70>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f933b90cb70>

  URL:  mlmodels/dataloader.py::pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f92a77c39d8>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f92a77c39d8>
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

  <mlmodels.model_keras.dataloader.charcnn.Model object at 0x7f92d1101f28> 

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

2020-05-21 04:32:15.873025: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 66453504 exceeds 10% of system memory.
2020-05-21 04:32:15.885560: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132382720 exceeds 10% of system memory.
2020-05-21 04:32:15.885560: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 131727360 exceeds 10% of system memory.
2020-05-21 04:32:16.925978: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 33095680 exceeds 10% of system memory.
2020-05-21 04:32:16.977263: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 132382720 exceeds 10% of system memory.

 Object Creation

 Object Compute

 Object get_data
Train on 354 samples, validate on 354 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 13s - loss: 1.8962
256/354 [====================>.........] - ETA: 5s - loss: 1.4022 
354/354 [==============================] - 24s 69ms/step - loss: 1.2843 - val_loss: 0.9702

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
[[8.8615607e-06 6.8739209e-06 2.3585629e-02 9.7639865e-01]
 [8.9862124e-06 6.5330341e-06 2.2019207e-02 9.7796530e-01]
 [1.1246868e-05 8.4614039e-06 2.5542917e-02 9.7443742e-01]
 ...
 [1.7517363e-05 1.2202296e-05 2.7288485e-02 9.7268176e-01]
 [1.8780129e-05 1.4902651e-05 2.9536892e-02 9.7042936e-01]
 [1.8575693e-05 1.3557658e-05 2.9318364e-02 9.7064948e-01]]

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

  <mlmodels.model_keras.dataloader.charcnn_zhang.Model object at 0x7f92e82c7d30> 

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

128/354 [=========>....................] - ETA: 7s - loss: 1.3900
256/354 [====================>.........] - ETA: 3s - loss: 1.3152
354/354 [==============================] - 15s 41ms/step - loss: 1.1576 - val_loss: 0.7312

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
[[0.0767356  0.06291159 0.26305124 0.59730154]
 [0.07653479 0.0627426  0.26296365 0.597759  ]
 [0.07682055 0.06298127 0.2630946  0.5971036 ]
 ...
 [0.07737476 0.06346103 0.2633962  0.59576803]
 [0.07740162 0.06347594 0.26339722 0.5957252 ]
 [0.07740042 0.06348375 0.26339513 0.5957207 ]]

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

  <mlmodels.model_keras.dataloader.textcnn.Model object at 0x7f92d10af2e8> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f92a721b668> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f927b66b268>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f927b66b268>

  function with postional parmater data_info <function get_dataset_torch at 0x7f927b66b268> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0019748752117156984 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011013798475265502 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0011682379643122355 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.008919044137001038 	 Accuracy: 2
model saves at 2 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f927b660b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f927b660b70>

  function with postional parmater data_info <function get_dataset_torch at 0x7f927b660b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([3, 7, 3, 7, 7, 7, 3, 3, 3, 3]), array([9, 7, 8, 7, 7, 1, 3, 8, 8, 8]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f92d1055128> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f928b7217b8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f928b7217b8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f928b7217b8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0021080466906229656 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.016676144123077392 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0018517194986343384 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.03915741729736328 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f928b725f28>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f928b725f28>

  function with postional parmater data_info <function get_dataset_torch at 0x7f928b725f28> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9]), array([0, 6, 7, 2, 5, 5, 6, 7, 2, 4]))

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
  0%|          | 0.00/170M [00:00<?, ?B/s]  5%|▌         | 8.87M/170M [00:00<00:01, 92.9MB/s] 12%|█▏        | 20.0M/170M [00:00<00:01, 99.0MB/s] 18%|█▊        | 30.7M/170M [00:00<00:01, 102MB/s]  24%|██▍       | 41.3M/170M [00:00<00:01, 105MB/s] 30%|██▉       | 51.0M/170M [00:00<00:01, 104MB/s] 36%|███▌      | 61.2M/170M [00:00<00:01, 105MB/s] 41%|████      | 70.2M/170M [00:00<00:01, 101MB/s] 47%|████▋     | 80.7M/170M [00:00<00:00, 103MB/s] 53%|█████▎    | 90.1M/170M [00:00<00:00, 94.6MB/s] 59%|█████▊    | 99.9M/170M [00:01<00:00, 91.9MB/s] 64%|██████▍   | 109M/170M [00:01<00:00, 90.3MB/s]  69%|██████▉   | 118M/170M [00:01<00:00, 87.6MB/s] 74%|███████▍  | 127M/170M [00:01<00:00, 84.1MB/s] 79%|███████▉  | 135M/170M [00:01<00:00, 73.0MB/s] 85%|████████▌ | 145M/170M [00:01<00:00, 74.8MB/s] 91%|█████████ | 154M/170M [00:01<00:00, 80.5MB/s] 96%|█████████▋| 164M/170M [00:01<00:00, 86.8MB/s]100%|██████████| 170M/170M [00:01<00:00, 92.6MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f927b666400> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f92897029d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f92897029d8>

  function with postional parmater data_info <function tf_dataset_download at 0x7f92897029d8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92896d5c80>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92896d5c80>

  function with postional parmater data_info <function get_dataset_torch at 0x7f92896d5c80> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.5123633766174316 	 Accuracy: 34
Train Epoch: 1 	 Loss: 704404.19375 	 Accuracy: 120
model saves at 120 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f9289702268>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f9289702268>

  function with postional parmater data_info <function tf_dataset_download at 0x7f9289702268> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f92896d5c80>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f92896d5c80>

  function with postional parmater data_info <function get_dataset_torch at 0x7f92896d5c80> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([2, 3, 5, 7, 9, 9, 3, 0, 2, 6]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f92896e6b38> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
img_01.png
0

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
