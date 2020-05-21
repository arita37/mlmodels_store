
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2398eeb488>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2398eeb488>

  function with postional parmater data_info <function get_dataset_torch at 0x7f2398eeb488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Using TensorFlow backend.
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3842048/9912422 [00:00<00:00, 38395251.19it/s]9920512it [00:00, 33146246.35it/s]                             
0it [00:00, ?it/s]32768it [00:00, 730813.36it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1001254.38it/s]1654784it [00:00, 12552410.59it/s]                           
0it [00:00, ?it/s]8192it [00:00, 236531.43it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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
((<torch.utils.data.dataloader.DataLoader object at 0x7f2382104940>, <torch.utils.data.dataloader.DataLoader object at 0x7f238210f940>), {})





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

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f23821069d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f23821069d8>

  function with postional parmater data_info <function tf_dataset_download at 0x7f23821069d8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:01<?, ? MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:01<03:27,  1.29s/ MiB][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:01<03:27,  1.29s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:01<03:26,  1.29s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<03:25,  1.29s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<03:23,  1.29s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<03:22,  1.29s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<03:21,  1.29s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<02:20,  1.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<02:20,  1.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<02:19,  1.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<02:18,  1.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<02:18,  1.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<02:17,  1.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<02:16,  1.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<01:35,  1.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<01:35,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<01:34,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<01:34,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<01:33,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<01:32,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<01:32,  1.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<01:04,  2.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<01:04,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<01:04,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<01:04,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<01:03,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<01:03,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<01:02,  2.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:01<00:44,  3.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:44,  3.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:43,  3.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:43,  3.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:43,  3.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:43,  3.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:42,  3.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:30,  4.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:30,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:30,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:29,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:29,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:29,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:29,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:28,  4.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  5.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  5.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  5.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:20,  5.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:20,  5.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:20,  5.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:02<00:19,  5.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:02<00:19,  5.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:02<00:14,  8.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:02<00:14,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:14,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:13,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:13,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:13,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:13,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:13,  8.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:02<00:09, 11.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:09, 11.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:02<00:06, 14.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:06, 14.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:02<00:04, 19.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:04, 19.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:03, 24.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:03, 24.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 30.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:02, 30.51 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:02, 36.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:02, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:02, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:02, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 36.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 42.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 42.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 47.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 47.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 50.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 50.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 50.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:01, 50.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:01, 50.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 50.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 50.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 50.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 52.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 54.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 54.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 56.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 56.36 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 57.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 57.27 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 58.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 58.29 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 59.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 59.15 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 58.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.90s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 58.93 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.03s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  3.90s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 58.93 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.03s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.03s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 26.86 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.03s/ url]
0 examples [00:00, ? examples/s]2020-05-21 18:08:43.021154: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 18:08:43.316618: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-21 18:08:43.316858: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c7e29bdb10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 18:08:43.316879: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:03,  3.70s/ examples]108 examples [00:03,  2.59s/ examples]210 examples [00:03,  1.81s/ examples]305 examples [00:04,  1.27s/ examples]411 examples [00:04,  1.12 examples/s]518 examples [00:04,  1.61 examples/s]609 examples [00:04,  2.29 examples/s]710 examples [00:04,  3.27 examples/s]819 examples [00:04,  4.67 examples/s]927 examples [00:04,  6.65 examples/s]1035 examples [00:04,  9.48 examples/s]1145 examples [00:04, 13.49 examples/s]1255 examples [00:04, 19.17 examples/s]1365 examples [00:05, 27.19 examples/s]1473 examples [00:05, 38.42 examples/s]1582 examples [00:05, 54.07 examples/s]1691 examples [00:05, 75.62 examples/s]1799 examples [00:05, 104.86 examples/s]1910 examples [00:05, 143.92 examples/s]2020 examples [00:05, 194.69 examples/s]2129 examples [00:05, 257.44 examples/s]2237 examples [00:05, 333.33 examples/s]2346 examples [00:05, 420.88 examples/s]2455 examples [00:06, 515.85 examples/s]2564 examples [00:06, 608.24 examples/s]2671 examples [00:06, 697.89 examples/s]2779 examples [00:06, 780.49 examples/s]2887 examples [00:06, 843.09 examples/s]2996 examples [00:06, 904.05 examples/s]3104 examples [00:06, 950.47 examples/s]3213 examples [00:06, 986.45 examples/s]3322 examples [00:06, 1014.95 examples/s]3431 examples [00:06, 1034.37 examples/s]3541 examples [00:07, 1051.86 examples/s]3650 examples [00:07, 1054.45 examples/s]3759 examples [00:07, 1062.84 examples/s]3867 examples [00:07, 1067.11 examples/s]3977 examples [00:07, 1074.20 examples/s]4086 examples [00:07, 1057.06 examples/s]4193 examples [00:07, 1053.27 examples/s]4302 examples [00:07, 1063.14 examples/s]4409 examples [00:07, 1042.37 examples/s]4519 examples [00:07, 1056.83 examples/s]4628 examples [00:08, 1064.43 examples/s]4736 examples [00:08, 1065.67 examples/s]4843 examples [00:08, 1055.51 examples/s]4952 examples [00:08, 1063.32 examples/s]5062 examples [00:08, 1071.46 examples/s]5171 examples [00:08, 1075.58 examples/s]5280 examples [00:08, 1077.82 examples/s]5391 examples [00:08, 1085.47 examples/s]5500 examples [00:08, 1085.07 examples/s]5609 examples [00:08, 1085.21 examples/s]5718 examples [00:09, 1083.70 examples/s]5827 examples [00:09, 1081.04 examples/s]5936 examples [00:09, 1077.40 examples/s]6045 examples [00:09, 1078.78 examples/s]6154 examples [00:09, 1080.45 examples/s]6264 examples [00:09, 1083.95 examples/s]6374 examples [00:09, 1086.54 examples/s]6483 examples [00:09, 1085.45 examples/s]6593 examples [00:09, 1087.72 examples/s]6702 examples [00:09, 1084.72 examples/s]6812 examples [00:10, 1087.68 examples/s]6921 examples [00:10, 1084.54 examples/s]7030 examples [00:10, 1085.83 examples/s]7139 examples [00:10, 1078.71 examples/s]7247 examples [00:10, 1078.45 examples/s]7356 examples [00:10, 1080.02 examples/s]7465 examples [00:10, 1049.01 examples/s]7571 examples [00:10, 1051.61 examples/s]7677 examples [00:10, 1054.06 examples/s]7785 examples [00:10, 1059.77 examples/s]7892 examples [00:11, 1052.38 examples/s]7999 examples [00:11, 1055.29 examples/s]8108 examples [00:11, 1063.38 examples/s]8218 examples [00:11, 1073.53 examples/s]8326 examples [00:11, 1073.62 examples/s]8434 examples [00:11, 1070.36 examples/s]8542 examples [00:11, 1066.61 examples/s]8651 examples [00:11, 1071.30 examples/s]8761 examples [00:11, 1077.30 examples/s]8869 examples [00:12, 1059.05 examples/s]8979 examples [00:12, 1069.21 examples/s]9087 examples [00:12, 1063.73 examples/s]9196 examples [00:12, 1070.03 examples/s]9304 examples [00:12, 1072.66 examples/s]9412 examples [00:12, 1068.84 examples/s]9521 examples [00:12, 1073.30 examples/s]9629 examples [00:12, 1073.16 examples/s]9739 examples [00:12, 1080.16 examples/s]9848 examples [00:12, 1080.50 examples/s]9957 examples [00:13, 1077.15 examples/s]10065 examples [00:13, 983.28 examples/s]10173 examples [00:13, 1009.06 examples/s]10285 examples [00:13, 1039.06 examples/s]10396 examples [00:13, 1057.13 examples/s]10504 examples [00:13, 1061.42 examples/s]10616 examples [00:13, 1076.20 examples/s]10725 examples [00:13, 1040.45 examples/s]10835 examples [00:13, 1057.24 examples/s]10942 examples [00:13, 1048.19 examples/s]11053 examples [00:14, 1064.36 examples/s]11160 examples [00:14, 1050.35 examples/s]11266 examples [00:14, 1037.15 examples/s]11376 examples [00:14, 1054.32 examples/s]11485 examples [00:14, 1064.05 examples/s]11594 examples [00:14, 1069.29 examples/s]11704 examples [00:14, 1077.43 examples/s]11812 examples [00:14, 1075.35 examples/s]11922 examples [00:14, 1081.57 examples/s]12033 examples [00:14, 1088.20 examples/s]12142 examples [00:15, 1084.50 examples/s]12252 examples [00:15, 1086.92 examples/s]12361 examples [00:15, 1080.64 examples/s]12472 examples [00:15, 1087.16 examples/s]12583 examples [00:15, 1091.46 examples/s]12694 examples [00:15, 1095.34 examples/s]12805 examples [00:15, 1098.61 examples/s]12915 examples [00:15, 1088.92 examples/s]13026 examples [00:15, 1092.76 examples/s]13136 examples [00:15, 1075.62 examples/s]13244 examples [00:16, 1072.78 examples/s]13355 examples [00:16, 1080.86 examples/s]13464 examples [00:16, 1077.42 examples/s]13574 examples [00:16, 1081.70 examples/s]13685 examples [00:16, 1087.09 examples/s]13794 examples [00:16, 1087.60 examples/s]13903 examples [00:16, 1085.76 examples/s]14012 examples [00:16, 1063.14 examples/s]14119 examples [00:16, 1040.12 examples/s]14228 examples [00:17, 1052.03 examples/s]14334 examples [00:17, 1024.69 examples/s]14443 examples [00:17, 1042.09 examples/s]14549 examples [00:17, 1047.11 examples/s]14657 examples [00:17, 1055.80 examples/s]14763 examples [00:17, 1037.14 examples/s]14867 examples [00:17, 1021.53 examples/s]14978 examples [00:17, 1044.45 examples/s]15083 examples [00:17, 1042.73 examples/s]15191 examples [00:17, 1052.79 examples/s]15297 examples [00:18, 1050.92 examples/s]15407 examples [00:18, 1062.56 examples/s]15518 examples [00:18, 1074.05 examples/s]15626 examples [00:18, 901.14 examples/s] 15721 examples [00:18, 907.88 examples/s]15816 examples [00:18, 883.04 examples/s]15907 examples [00:18, 867.50 examples/s]16000 examples [00:18, 884.47 examples/s]16093 examples [00:18, 896.96 examples/s]16184 examples [00:19, 879.95 examples/s]16277 examples [00:19, 892.56 examples/s]16374 examples [00:19, 913.16 examples/s]16480 examples [00:19, 950.89 examples/s]16585 examples [00:19, 976.32 examples/s]16695 examples [00:19, 1008.00 examples/s]16803 examples [00:19, 1026.88 examples/s]16907 examples [00:19, 985.92 examples/s] 17012 examples [00:19, 1002.11 examples/s]17113 examples [00:19, 1000.02 examples/s]17222 examples [00:20, 1023.86 examples/s]17331 examples [00:20, 1041.86 examples/s]17441 examples [00:20, 1056.28 examples/s]17549 examples [00:20, 1061.51 examples/s]17656 examples [00:20, 1061.00 examples/s]17766 examples [00:20, 1072.28 examples/s]17877 examples [00:20, 1080.82 examples/s]17987 examples [00:20, 1084.10 examples/s]18096 examples [00:20, 1082.44 examples/s]18206 examples [00:20, 1085.58 examples/s]18315 examples [00:21, 1075.22 examples/s]18423 examples [00:21, 1067.47 examples/s]18533 examples [00:21, 1076.27 examples/s]18641 examples [00:21, 1076.98 examples/s]18749 examples [00:21, 1075.80 examples/s]18859 examples [00:21, 1081.89 examples/s]18969 examples [00:21, 1084.93 examples/s]19078 examples [00:21, 1079.71 examples/s]19186 examples [00:21, 1064.61 examples/s]19293 examples [00:22, 1001.08 examples/s]19395 examples [00:22, 1005.18 examples/s]19505 examples [00:22, 1030.41 examples/s]19617 examples [00:22, 1055.55 examples/s]19726 examples [00:22, 1064.79 examples/s]19834 examples [00:22, 1068.68 examples/s]19943 examples [00:22, 1073.03 examples/s]20051 examples [00:22, 963.91 examples/s] 20163 examples [00:22, 1004.51 examples/s]20266 examples [00:22, 1006.08 examples/s]20370 examples [00:23, 1013.42 examples/s]20480 examples [00:23, 1036.66 examples/s]20588 examples [00:23, 1047.75 examples/s]20698 examples [00:23, 1061.58 examples/s]20805 examples [00:23, 1061.54 examples/s]20916 examples [00:23, 1072.84 examples/s]21027 examples [00:23, 1081.46 examples/s]21136 examples [00:23, 1079.53 examples/s]21246 examples [00:23, 1084.87 examples/s]21355 examples [00:23, 1082.61 examples/s]21466 examples [00:24, 1089.74 examples/s]21576 examples [00:24, 1049.34 examples/s]21684 examples [00:24, 1056.01 examples/s]21790 examples [00:24, 1050.60 examples/s]21897 examples [00:24, 1054.64 examples/s]22008 examples [00:24, 1069.55 examples/s]22116 examples [00:24, 1063.81 examples/s]22226 examples [00:24, 1072.27 examples/s]22337 examples [00:24, 1081.22 examples/s]22446 examples [00:24, 1081.92 examples/s]22557 examples [00:25, 1088.27 examples/s]22667 examples [00:25, 1091.22 examples/s]22778 examples [00:25, 1094.27 examples/s]22888 examples [00:25, 1094.40 examples/s]22998 examples [00:25, 1088.05 examples/s]23107 examples [00:25, 1088.18 examples/s]23216 examples [00:25, 1083.29 examples/s]23326 examples [00:25, 1088.23 examples/s]23436 examples [00:25, 1089.18 examples/s]23545 examples [00:25, 1086.13 examples/s]23654 examples [00:26, 1049.40 examples/s]23766 examples [00:26, 1066.85 examples/s]23876 examples [00:26, 1074.37 examples/s]23986 examples [00:26, 1081.24 examples/s]24095 examples [00:26, 1049.98 examples/s]24204 examples [00:26, 1060.60 examples/s]24316 examples [00:26, 1075.46 examples/s]24427 examples [00:26, 1084.04 examples/s]24536 examples [00:26, 1079.11 examples/s]24645 examples [00:27, 1075.72 examples/s]24753 examples [00:27, 1067.58 examples/s]24861 examples [00:27, 1070.34 examples/s]24972 examples [00:27, 1079.44 examples/s]25081 examples [00:27, 1081.29 examples/s]25190 examples [00:27, 1081.88 examples/s]25299 examples [00:27, 1072.41 examples/s]25409 examples [00:27, 1079.02 examples/s]25519 examples [00:27, 1084.31 examples/s]25628 examples [00:27, 1084.89 examples/s]25737 examples [00:28, 1085.41 examples/s]25847 examples [00:28, 1086.67 examples/s]25958 examples [00:28, 1091.71 examples/s]26068 examples [00:28, 1088.83 examples/s]26177 examples [00:28, 1088.57 examples/s]26286 examples [00:28, 1084.69 examples/s]26396 examples [00:28, 1087.22 examples/s]26508 examples [00:28, 1094.71 examples/s]26618 examples [00:28, 1093.19 examples/s]26728 examples [00:28, 1093.42 examples/s]26838 examples [00:29, 1087.32 examples/s]26947 examples [00:29, 1058.99 examples/s]27058 examples [00:29, 1071.73 examples/s]27168 examples [00:29, 1077.44 examples/s]27276 examples [00:29, 1063.33 examples/s]27385 examples [00:29, 1070.48 examples/s]27497 examples [00:29, 1082.65 examples/s]27607 examples [00:29, 1086.24 examples/s]27718 examples [00:29, 1092.46 examples/s]27829 examples [00:29, 1094.92 examples/s]27939 examples [00:30, 1071.42 examples/s]28050 examples [00:30, 1081.73 examples/s]28160 examples [00:30, 1085.79 examples/s]28271 examples [00:30, 1091.22 examples/s]28381 examples [00:30, 1085.25 examples/s]28490 examples [00:30, 1072.13 examples/s]28601 examples [00:30, 1083.20 examples/s]28710 examples [00:30, 1081.66 examples/s]28819 examples [00:30, 1066.44 examples/s]28927 examples [00:30, 1068.79 examples/s]29034 examples [00:31, 1066.78 examples/s]29141 examples [00:31, 1063.57 examples/s]29251 examples [00:31, 1072.42 examples/s]29362 examples [00:31, 1080.75 examples/s]29471 examples [00:31, 1077.48 examples/s]29579 examples [00:31, 1064.22 examples/s]29687 examples [00:31, 1067.75 examples/s]29795 examples [00:31, 1069.45 examples/s]29902 examples [00:31, 1068.95 examples/s]30009 examples [00:32, 987.68 examples/s] 30110 examples [00:32, 986.91 examples/s]30210 examples [00:32, 982.29 examples/s]30320 examples [00:32, 1012.95 examples/s]30422 examples [00:32, 1001.89 examples/s]30526 examples [00:32, 1011.74 examples/s]30637 examples [00:32, 1038.65 examples/s]30748 examples [00:32, 1058.44 examples/s]30859 examples [00:32, 1072.35 examples/s]30967 examples [00:32, 1063.54 examples/s]31076 examples [00:33, 1069.23 examples/s]31186 examples [00:33, 1076.73 examples/s]31296 examples [00:33, 1081.62 examples/s]31407 examples [00:33, 1088.25 examples/s]31517 examples [00:33, 1089.06 examples/s]31626 examples [00:33, 1081.93 examples/s]31735 examples [00:33, 1081.18 examples/s]31847 examples [00:33, 1089.96 examples/s]31959 examples [00:33, 1096.62 examples/s]32069 examples [00:33, 1066.42 examples/s]32176 examples [00:34, 1050.29 examples/s]32285 examples [00:34, 1061.79 examples/s]32395 examples [00:34, 1072.09 examples/s]32506 examples [00:34, 1080.93 examples/s]32616 examples [00:34, 1084.37 examples/s]32726 examples [00:34, 1086.58 examples/s]32836 examples [00:34, 1088.39 examples/s]32946 examples [00:34, 1090.94 examples/s]33056 examples [00:34, 1087.90 examples/s]33165 examples [00:34, 1084.49 examples/s]33274 examples [00:35, 1075.01 examples/s]33384 examples [00:35, 1080.83 examples/s]33493 examples [00:35, 1011.72 examples/s]33600 examples [00:35, 1028.50 examples/s]33711 examples [00:35, 1050.33 examples/s]33821 examples [00:35, 1062.33 examples/s]33931 examples [00:35, 1071.52 examples/s]34039 examples [00:35, 1068.24 examples/s]34151 examples [00:35, 1080.50 examples/s]34263 examples [00:36, 1089.30 examples/s]34373 examples [00:36, 1053.76 examples/s]34483 examples [00:36, 1065.53 examples/s]34590 examples [00:36, 1066.40 examples/s]34697 examples [00:36, 1054.07 examples/s]34803 examples [00:36, 1054.53 examples/s]34909 examples [00:36, 1045.80 examples/s]35016 examples [00:36, 1052.03 examples/s]35127 examples [00:36, 1067.22 examples/s]35238 examples [00:36, 1077.08 examples/s]35346 examples [00:37, 1074.00 examples/s]35454 examples [00:37, 1069.12 examples/s]35564 examples [00:37, 1076.21 examples/s]35674 examples [00:37, 1081.87 examples/s]35783 examples [00:37, 1069.77 examples/s]35891 examples [00:37, 1066.90 examples/s]35998 examples [00:37, 1062.75 examples/s]36109 examples [00:37, 1076.05 examples/s]36217 examples [00:37, 1071.63 examples/s]36327 examples [00:37, 1079.18 examples/s]36435 examples [00:38, 1014.32 examples/s]36540 examples [00:38, 1010.92 examples/s]36644 examples [00:38, 1017.93 examples/s]36747 examples [00:38, 1018.62 examples/s]36855 examples [00:38, 1031.75 examples/s]36964 examples [00:38, 1046.83 examples/s]37072 examples [00:38, 1055.78 examples/s]37183 examples [00:38, 1070.16 examples/s]37294 examples [00:38, 1079.69 examples/s]37406 examples [00:38, 1089.71 examples/s]37516 examples [00:39, 1089.68 examples/s]37626 examples [00:39, 1044.49 examples/s]37738 examples [00:39, 1063.99 examples/s]37850 examples [00:39, 1080.13 examples/s]37959 examples [00:39, 1079.26 examples/s]38070 examples [00:39, 1087.83 examples/s]38181 examples [00:39, 1092.78 examples/s]38291 examples [00:39, 1092.60 examples/s]38402 examples [00:39, 1095.22 examples/s]38512 examples [00:39, 1096.33 examples/s]38622 examples [00:40, 1072.05 examples/s]38730 examples [00:40, 1043.76 examples/s]38838 examples [00:40, 1052.02 examples/s]38946 examples [00:40, 1059.90 examples/s]39053 examples [00:40, 1045.71 examples/s]39158 examples [00:40, 1036.86 examples/s]39268 examples [00:40, 1053.25 examples/s]39378 examples [00:40, 1064.54 examples/s]39487 examples [00:40, 1071.22 examples/s]39595 examples [00:41, 1049.69 examples/s]39705 examples [00:41, 1060.54 examples/s]39814 examples [00:41, 1067.09 examples/s]39921 examples [00:41, 1067.60 examples/s]40028 examples [00:41, 943.55 examples/s] 40128 examples [00:41, 958.79 examples/s]40233 examples [00:41, 983.35 examples/s]40343 examples [00:41, 1012.82 examples/s]40455 examples [00:41, 1041.79 examples/s]40566 examples [00:41, 1061.01 examples/s]40673 examples [00:42, 1033.64 examples/s]40781 examples [00:42, 1045.47 examples/s]40889 examples [00:42, 1053.90 examples/s]40999 examples [00:42, 1065.55 examples/s]41106 examples [00:42, 1039.73 examples/s]41215 examples [00:42, 1053.36 examples/s]41325 examples [00:42, 1064.62 examples/s]41433 examples [00:42, 1068.26 examples/s]41540 examples [00:42, 1002.32 examples/s]41642 examples [00:43, 933.88 examples/s] 41737 examples [00:43, 927.71 examples/s]41831 examples [00:43, 920.84 examples/s]41938 examples [00:43, 960.92 examples/s]42045 examples [00:43, 990.97 examples/s]42152 examples [00:43, 1012.72 examples/s]42260 examples [00:43, 1029.67 examples/s]42369 examples [00:43, 1045.88 examples/s]42478 examples [00:43, 1057.02 examples/s]42588 examples [00:43, 1068.95 examples/s]42697 examples [00:44, 1073.92 examples/s]42807 examples [00:44, 1079.38 examples/s]42918 examples [00:44, 1087.97 examples/s]43027 examples [00:44, 1087.86 examples/s]43136 examples [00:44, 1052.32 examples/s]43242 examples [00:44, 1045.14 examples/s]43350 examples [00:44, 1053.09 examples/s]43456 examples [00:44, 1053.35 examples/s]43562 examples [00:44, 1053.26 examples/s]43668 examples [00:44, 1032.52 examples/s]43773 examples [00:45, 1035.95 examples/s]43879 examples [00:45, 1040.57 examples/s]43986 examples [00:45, 1048.28 examples/s]44092 examples [00:45, 1049.35 examples/s]44199 examples [00:45, 1054.30 examples/s]44306 examples [00:45, 1057.42 examples/s]44413 examples [00:45, 1059.34 examples/s]44520 examples [00:45, 1060.85 examples/s]44627 examples [00:45, 1055.46 examples/s]44733 examples [00:45, 1036.62 examples/s]44837 examples [00:46, 1002.82 examples/s]44938 examples [00:46, 974.70 examples/s] 45046 examples [00:46, 1002.84 examples/s]45150 examples [00:46, 1011.37 examples/s]45252 examples [00:46, 1008.00 examples/s]45355 examples [00:46, 1014.40 examples/s]45461 examples [00:46, 1025.37 examples/s]45564 examples [00:46, 984.45 examples/s] 45664 examples [00:46, 986.50 examples/s]45768 examples [00:47, 999.60 examples/s]45870 examples [00:47, 1005.55 examples/s]45971 examples [00:47, 998.57 examples/s] 46072 examples [00:47, 990.41 examples/s]46172 examples [00:47, 971.73 examples/s]46270 examples [00:47, 944.83 examples/s]46365 examples [00:47, 942.17 examples/s]46460 examples [00:47, 926.88 examples/s]46563 examples [00:47, 955.39 examples/s]46664 examples [00:47, 970.05 examples/s]46769 examples [00:48, 991.50 examples/s]46869 examples [00:48, 976.52 examples/s]46967 examples [00:48, 977.44 examples/s]47072 examples [00:48, 997.74 examples/s]47173 examples [00:48, 995.84 examples/s]47280 examples [00:48, 1014.55 examples/s]47387 examples [00:48, 1028.65 examples/s]47494 examples [00:48, 1039.87 examples/s]47602 examples [00:48, 1050.11 examples/s]47708 examples [00:48, 1047.76 examples/s]47813 examples [00:49, 1010.21 examples/s]47915 examples [00:49, 982.53 examples/s] 48021 examples [00:49, 1002.08 examples/s]48125 examples [00:49, 1011.96 examples/s]48228 examples [00:49, 1015.79 examples/s]48334 examples [00:49, 1026.87 examples/s]48439 examples [00:49, 1031.17 examples/s]48545 examples [00:49, 1038.39 examples/s]48653 examples [00:49, 1049.68 examples/s]48759 examples [00:50, 1047.19 examples/s]48867 examples [00:50, 1054.54 examples/s]48973 examples [00:50, 1035.53 examples/s]49079 examples [00:50, 1040.56 examples/s]49184 examples [00:50, 1009.83 examples/s]49288 examples [00:50, 1017.76 examples/s]49390 examples [00:50, 1006.07 examples/s]49497 examples [00:50, 1023.19 examples/s]49602 examples [00:50, 1028.64 examples/s]49706 examples [00:50, 1026.47 examples/s]49811 examples [00:51, 1030.66 examples/s]49915 examples [00:51, 1031.54 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7364/50000 [00:00<00:00, 73638.31 examples/s] 40%|████      | 20242/50000 [00:00<00:00, 84491.01 examples/s] 67%|██████▋   | 33342/50000 [00:00<00:00, 94561.91 examples/s] 94%|█████████▎| 46789/50000 [00:00<00:00, 103803.12 examples/s]                                                                0 examples [00:00, ? examples/s]69 examples [00:00, 688.58 examples/s]170 examples [00:00, 759.67 examples/s]269 examples [00:00, 816.07 examples/s]356 examples [00:00, 830.74 examples/s]462 examples [00:00, 887.68 examples/s]567 examples [00:00, 928.88 examples/s]674 examples [00:00, 966.69 examples/s]769 examples [00:00, 959.68 examples/s]868 examples [00:00, 967.73 examples/s]963 examples [00:01, 959.07 examples/s]1069 examples [00:01, 986.57 examples/s]1167 examples [00:01, 966.89 examples/s]1273 examples [00:01, 992.65 examples/s]1373 examples [00:01, 994.31 examples/s]1478 examples [00:01, 1008.95 examples/s]1579 examples [00:01, 1001.16 examples/s]1684 examples [00:01, 1013.95 examples/s]1788 examples [00:01, 1019.28 examples/s]1890 examples [00:01, 1017.39 examples/s]1992 examples [00:02, 989.99 examples/s] 2099 examples [00:02, 1010.80 examples/s]2206 examples [00:02, 1026.89 examples/s]2313 examples [00:02, 1039.26 examples/s]2418 examples [00:02, 1038.09 examples/s]2525 examples [00:02, 1047.12 examples/s]2633 examples [00:02, 1054.20 examples/s]2740 examples [00:02, 1058.67 examples/s]2846 examples [00:02, 1052.58 examples/s]2952 examples [00:02, 1050.23 examples/s]3059 examples [00:03, 1053.47 examples/s]3165 examples [00:03, 1042.12 examples/s]3273 examples [00:03, 1052.88 examples/s]3381 examples [00:03, 1059.75 examples/s]3488 examples [00:03, 1046.76 examples/s]3593 examples [00:03, 1018.09 examples/s]3696 examples [00:03, 1021.02 examples/s]3799 examples [00:03, 971.01 examples/s] 3902 examples [00:03, 987.19 examples/s]4008 examples [00:03, 1007.80 examples/s]4115 examples [00:04, 1023.95 examples/s]4221 examples [00:04, 1032.29 examples/s]4328 examples [00:04, 1041.01 examples/s]4433 examples [00:04, 1035.32 examples/s]4540 examples [00:04, 1043.18 examples/s]4645 examples [00:04, 1037.88 examples/s]4749 examples [00:04, 1021.99 examples/s]4852 examples [00:04, 1001.39 examples/s]4956 examples [00:04, 1011.83 examples/s]5061 examples [00:04, 1022.38 examples/s]5164 examples [00:05, 1006.15 examples/s]5272 examples [00:05, 1025.58 examples/s]5379 examples [00:05, 1037.70 examples/s]5483 examples [00:05, 996.77 examples/s] 5584 examples [00:05, 987.10 examples/s]5684 examples [00:05, 988.52 examples/s]5788 examples [00:05, 1003.08 examples/s]5894 examples [00:05, 1018.10 examples/s]6001 examples [00:05, 1029.85 examples/s]6105 examples [00:06, 1029.81 examples/s]6209 examples [00:06, 1027.60 examples/s]6313 examples [00:06, 1030.35 examples/s]6419 examples [00:06, 1038.06 examples/s]6525 examples [00:06, 1043.64 examples/s]6630 examples [00:06, 1006.49 examples/s]6731 examples [00:06, 994.21 examples/s] 6831 examples [00:06, 990.80 examples/s]6934 examples [00:06, 1000.51 examples/s]7037 examples [00:06, 1009.11 examples/s]7139 examples [00:07, 1009.94 examples/s]7241 examples [00:07, 990.05 examples/s] 7343 examples [00:07, 997.42 examples/s]7448 examples [00:07, 1011.52 examples/s]7550 examples [00:07, 994.14 examples/s] 7656 examples [00:07, 1011.07 examples/s]7763 examples [00:07, 1026.84 examples/s]7866 examples [00:07, 998.71 examples/s] 7967 examples [00:07, 986.98 examples/s]8066 examples [00:07, 982.61 examples/s]8173 examples [00:08, 1005.88 examples/s]8274 examples [00:08, 985.01 examples/s] 8379 examples [00:08, 1003.37 examples/s]8481 examples [00:08, 1005.18 examples/s]8582 examples [00:08, 1006.00 examples/s]8688 examples [00:08, 1021.21 examples/s]8793 examples [00:08, 1027.74 examples/s]8896 examples [00:08, 1001.45 examples/s]8997 examples [00:08, 980.22 examples/s] 9097 examples [00:09, 983.58 examples/s]9204 examples [00:09, 1006.62 examples/s]9310 examples [00:09, 1020.75 examples/s]9415 examples [00:09, 1027.14 examples/s]9521 examples [00:09, 1035.26 examples/s]9625 examples [00:09, 1016.99 examples/s]9727 examples [00:09, 1014.50 examples/s]9832 examples [00:09, 1022.84 examples/s]9935 examples [00:09, 999.25 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete8DU7HW/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete8DU7HW/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f234c97aa60>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f234c97aa60>

  function with postional parmater data_info <function get_dataset_torch at 0x7f234c97aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f2398e367b8>, <torch.utils.data.dataloader.DataLoader object at 0x7f233cf1ee80>), {})





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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f233cf37598>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f233cf37598>

  function with postional parmater data_info <function get_dataset_torch at 0x7f233cf37598> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f238210f588>, <torch.utils.data.dataloader.DataLoader object at 0x7f234f7c4278>), {})





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

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f23ff685ea0>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f23ff685ea0>

  URL:  sklearn.model_selection::train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f23eb66aea0>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f23eb66aea0>

  URL:  mlmodels/dataloader.py::pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f233cf1cd90>

 ######### postional parameteres :  ['t']

 ######### Execute : preprocessor_func <function pickle_dump at 0x7f233cf1cd90>
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

  <mlmodels.model_keras.dataloader.charcnn.Model object at 0x7f2380228358> 

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

128/354 [=========>....................] - ETA: 8s - loss: 1.7577
256/354 [====================>.........] - ETA: 3s - loss: 1.2997
354/354 [==============================] - 16s 44ms/step - loss: 1.2160 - val_loss: 0.7423

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
[[2.2073616e-05 3.3229207e-05 8.9610599e-02 9.1033411e-01]
 [1.7426908e-05 2.4607494e-05 7.9872370e-02 9.2008561e-01]
 [2.5019937e-05 3.4244153e-05 8.8934384e-02 9.1100639e-01]
 ...
 [2.7036560e-05 3.5166409e-05 7.4353531e-02 9.2558426e-01]
 [3.4751421e-05 5.2782802e-05 8.9902043e-02 9.1001040e-01]
 [3.7888742e-05 5.3890424e-05 8.7683730e-02 9.1222447e-01]]

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

  <mlmodels.model_keras.dataloader.charcnn_zhang.Model object at 0x7f234f7ceac8> 

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

128/354 [=========>....................] - ETA: 5s - loss: 1.3857
256/354 [====================>.........] - ETA: 2s - loss: 1.1906
354/354 [==============================] - 10s 28ms/step - loss: 1.4517 - val_loss: 0.6230

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
[[0.01941205 0.01652954 0.2148158  0.7492426 ]
 [0.01941641 0.01653493 0.21484321 0.7492054 ]
 [0.01950827 0.01661271 0.21510714 0.74877185]
 ...
 [0.01974653 0.01680477 0.21581936 0.7476293 ]
 [0.01975478 0.01681243 0.21584892 0.74758387]
 [0.01973157 0.01679455 0.21574856 0.74772525]]

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

  <mlmodels.model_keras.dataloader.textcnn.Model object at 0x7f2381d22550> 

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f232eb14f98> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f232f1178c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f232f1178c8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f232f1178c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0017448192834854125 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.0110143301486969 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0014660540819168091 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.009039243936538696 	 Accuracy: 2
model saves at 2 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f232f1176a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f232f1176a8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f232f1176a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([8, 6, 2, 8, 8, 7, 5, 8, 2, 9]), array([1, 4, 2, 3, 6, 7, 9, 0, 2, 8]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f2380d5e828> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f232b300e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f232b300e18>

  function with postional parmater data_info <function get_dataset_torch at 0x7f232b300e18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0019999088843663535 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01683736276626587 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0022164207100868224 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.048943851470947264 	 Accuracy: 0

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f232b3006a8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f232b3006a8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f232b3006a8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
(array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([3, 6, 5, 8, 1, 6, 2, 6, 7, 0]))

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
  0%|          | 0.00/170M [00:00<?, ?B/s]  7%|▋         | 11.9M/170M [00:00<00:01, 124MB/s] 12%|█▏        | 21.2M/170M [00:00<00:01, 115MB/s] 17%|█▋        | 29.8M/170M [00:00<00:01, 106MB/s] 23%|██▎       | 39.5M/170M [00:00<00:01, 105MB/s] 29%|██▉       | 49.6M/170M [00:00<00:01, 105MB/s] 35%|███▍      | 59.3M/170M [00:00<00:01, 104MB/s] 41%|████      | 70.1M/170M [00:00<00:00, 107MB/s] 47%|████▋     | 80.7M/170M [00:00<00:00, 108MB/s] 53%|█████▎    | 90.4M/170M [00:00<00:00, 106MB/s] 60%|█████▉    | 101M/170M [00:01<00:00, 109MB/s]  65%|██████▌   | 112M/170M [00:01<00:00, 106MB/s] 71%|███████▏  | 121M/170M [00:01<00:00, 105MB/s] 78%|███████▊  | 133M/170M [00:01<00:00, 109MB/s] 84%|████████▍ | 143M/170M [00:01<00:00, 109MB/s] 90%|█████████ | 154M/170M [00:01<00:00, 109MB/s] 97%|█████████▋| 166M/170M [00:01<00:00, 113MB/s]100%|██████████| 170M/170M [00:01<00:00, 108MB/s]
  <mlmodels.model_tch.torchhub.Model object at 0x7f2380d5ee10> 

  #### Fit   ######################################################## 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f233f4a60d0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f233f4a60d0>

  function with postional parmater data_info <function tf_dataset_download at 0x7f233f4a60d0> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f233f4a6950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f233f4a6950>

  function with postional parmater data_info <function get_dataset_torch at 0x7f233f4a6950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
Train Epoch: 1 	 Loss: 0.5688253450393677 	 Accuracy: 38
Train Epoch: 1 	 Loss: 4828.533935546875 	 Accuracy: 220
model saves at 220 accuracy

  #### Predict   #################################################### 

  URL:  mlmodels/preprocess/generic.py::tf_dataset_download {} 

###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f233f4878c8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f233f4878c8>

  function with postional parmater data_info <function tf_dataset_download at 0x7f233f4878c8> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

  ############## Saving train dataset ############################### 

  ############## Saving train dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'mlmodels/preprocess/generic.py:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f233f536d90>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f233f536d90>

  function with postional parmater data_info <function get_dataset_torch at 0x7f233f536d90> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'preprocess.generic.NumpyDataset'> 
Dataset File path :  dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  dataset/vision/cifar10/test/cifar10.npz
(array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([8, 3, 7, 4, 2, 7, 1, 2, 0, 0]))

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

  <mlmodels.model_tch.torchhub.Model object at 0x7f237847e240> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
img_01.png
0

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
