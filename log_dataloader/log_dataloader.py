
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6ca6da91408244e26c157e9e6467cc18ede43e71', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6ca6da91408244e26c157e9e6467cc18ede43e71

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6ca6da91408244e26c157e9e6467cc18ede43e71

 ************************************************************************************************************************

  ############Check model ################################ 





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py --do test  

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/ag_news_csv",
    "train":true,
    "alphabet_size":69,
    "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size":1014,
    "num_of_classes":4
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "colX":[
          "colX"
        ],
        "coly":[
          "coly"
        ],
        "encoding":"'ISO-8859-1'",
        "read_csv_parm":{
          "usecols":[
            0,
            1
          ],
          "names":[
            "coly",
            "colX"
          ]
        }
      }
    },
    {
      "name":"tokenizer",
      "uri":"mlmodels.model_keras.raw.char_cnn.data_utils:Data",
      "args":{
        "data_source":"",
        "alphabet":"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size":1014,
        "num_of_classes":4
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': "'ISO-8859-1'", 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX']}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/charcnn_zhang.json [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "dataset":"mlmodels\/dataset\/text\/imdb",
    "pass_data_pars":false,
    "train":true,
    "maxlen":40,
    "max_features":5
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:NumpyDataset",
      "args":{
        "numpy_loader_args":{
          "allow_pickle":true
        },
        "encoding":"'ISO-8859-1'"
      }
    },
    {
      "name":"imdb_process",
      "uri":"mlmodels.preprocess.text_keras:IMDBDataset",
      "args":{
        "num_words":5
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.NumpyDataset'> 
cls_name : NumpyDataset
Dataset File path :  train/mlmodels/dataset/text/imdb.npz
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.json [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/text\/",
    "dataset":"ner_dataset.csv",
    "pass_data_pars":false,
    "train":true
  },
  "preprocessors":[
    {
      "name":"loader",
      "uri":"mlmodels.preprocess.generic:pandasDataset",
      "args":{
        "read_csv_parm":{
          "encoding":"ISO-8859-1"
        },
        "colX":[

        ],
        "coly":[

        ]
      }
    },
    {
      "uri":"mlmodels.preprocess.text_keras:Preprocess_namentity",
      "args":{
        "max_len":75
      },
      "internal_states":[
        "word_count"
      ]
    },
    {
      "name":"split_xy",
      "uri":"mlmodels.dataloader:split_xy_from_dict",
      "args":{
        "col_Xinput":[
          "X"
        ],
        "col_yinput":[
          "y"
        ]
      }
    },
    {
      "name":"split_train_test",
      "uri":"sklearn.model_selection:train_test_split",
      "args":{
        "test_size":0.5
      }
    },
    {
      "name":"saver",
      "uri":"mlmodels.dataloader:pickle_dump",
      "args":{
        "path":"mlmodels\/ztest\/ml_keras\/namentity_crm_bilstm\/data.pkl"
      }
    }
  ],
  "output":{
    "shape":[
      [
        75
      ],
      [
        75,
        18
      ]
    ],
    "max_len":75
  }
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.generic.pandasDataset'> 
cls_name : pandasDataset

  URL:  mlmodels.preprocess.text_keras:Preprocess_namentity {'max_len': 75} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.preprocess.text_keras.Preprocess_namentity'> 
cls_name : Preprocess_namentity

  
 Object Creation 

  
 Object Compute 

  
 Object get_data 

  URL:  mlmodels.dataloader:split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5a2d5d2510> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5a2d5d2510> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5a98282488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5a98282488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5ab74a0ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5ab74a0ea0> 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm.json [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256,
            "path":"dataset\/vision\/MNIST\/"
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a455bd158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a455bd158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a455bd158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 502, in __init__
    df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
    self._reader = parsers.TextReader(src, **kwds)
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv' does not exist: b'mlmodels/dataset/text/ag_news_csv/train/mlmodels/dataset/text/ag_news_csv.csv'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 258, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 301, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 93, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 21%|██        | 2105344/9912422 [00:00<00:00, 21049746.27it/s] 79%|███████▉  | 7864320/9912422 [00:00<00:00, 25912700.07it/s]9920512it [00:00, 27075386.28it/s]                             
0it [00:00, ?it/s]32768it [00:00, 614851.36it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486673.80it/s]1654784it [00:00, 11670930.15it/s]                         
0it [00:00, ?it/s]8192it [00:00, 223898.83it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a439a46d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a43aa3860>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/model_list_CIFAR.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/cifar10\/",
    "dataset":"CIFAR10",
    "data_type":"tf_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tf_dataset_start",
      "uri":"mlmodels.preprocess.generic:tf_dataset_download",
      "arg":{
        "train_samples":2000,
        "test_samples":500,
        "shuffle":true,
        "download":true
      }
    },
    {
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"mlmodels.preprocess.generic:NumpyDataset",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_generic",
          "pass_data_pars":false,
          "arg":{
            "fixed_size":256
          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:tf_dataset_download {} 

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5a455b2d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5a455b2d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5a455b2d90> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:47,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:46,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:45,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:45,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:44,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:43,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:43,  1.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:12,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:11,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:11,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:11,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:10,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:10,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:09,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:09,  2.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:48,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:47,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:47,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:46,  3.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:33,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:32,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:32,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:32,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:32,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:31,  4.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:01<00:22,  5.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:22,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:22,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:22,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:21,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:21,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:21,  5.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:15,  8.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:15,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  8.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 48.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 55.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 55.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 69.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 74.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.78 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.32 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.62s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.32 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.66 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ url]
0 examples [00:00, ? examples/s]2020-08-13 00:08:04.669209: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-13 00:08:04.727648: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-13 00:08:04.727901: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f3c9aed900 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-13 00:08:04.727924: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.60 examples/s]93 examples [00:00,  3.71 examples/s]184 examples [00:00,  5.29 examples/s]276 examples [00:00,  7.54 examples/s]353 examples [00:00, 10.73 examples/s]448 examples [00:00, 15.25 examples/s]541 examples [00:00, 21.63 examples/s]636 examples [00:01, 30.60 examples/s]727 examples [00:01, 43.09 examples/s]821 examples [00:01, 60.37 examples/s]916 examples [00:01, 83.93 examples/s]1009 examples [00:01, 115.44 examples/s]1104 examples [00:01, 156.70 examples/s]1196 examples [00:01, 208.53 examples/s]1289 examples [00:01, 271.57 examples/s]1387 examples [00:01, 346.37 examples/s]1481 examples [00:01, 426.72 examples/s]1577 examples [00:02, 511.31 examples/s]1671 examples [00:02, 591.73 examples/s]1765 examples [00:02, 661.91 examples/s]1859 examples [00:02, 677.69 examples/s]1949 examples [00:02, 730.10 examples/s]2038 examples [00:02, 770.43 examples/s]2128 examples [00:02, 804.16 examples/s]2217 examples [00:02, 826.98 examples/s]2308 examples [00:02, 849.30 examples/s]2398 examples [00:03, 849.89 examples/s]2486 examples [00:03, 845.66 examples/s]2573 examples [00:03, 851.58 examples/s]2664 examples [00:03, 865.43 examples/s]2752 examples [00:03, 864.70 examples/s]2840 examples [00:03, 868.44 examples/s]2929 examples [00:03, 873.47 examples/s]3020 examples [00:03, 881.73 examples/s]3109 examples [00:03, 880.03 examples/s]3198 examples [00:03, 858.33 examples/s]3290 examples [00:04, 873.68 examples/s]3383 examples [00:04, 888.83 examples/s]3473 examples [00:04, 890.49 examples/s]3563 examples [00:04, 872.35 examples/s]3657 examples [00:04, 891.05 examples/s]3748 examples [00:04, 894.45 examples/s]3842 examples [00:04, 905.17 examples/s]3933 examples [00:04, 902.13 examples/s]4024 examples [00:04, 903.03 examples/s]4117 examples [00:04, 908.41 examples/s]4208 examples [00:05, 908.12 examples/s]4300 examples [00:05, 911.33 examples/s]4392 examples [00:05, 912.58 examples/s]4484 examples [00:05, 909.07 examples/s]4575 examples [00:05, 900.25 examples/s]4666 examples [00:05, 901.02 examples/s]4760 examples [00:05, 912.13 examples/s]4852 examples [00:05, 909.97 examples/s]4944 examples [00:05, 905.43 examples/s]5035 examples [00:05, 901.68 examples/s]5128 examples [00:06, 908.46 examples/s]5220 examples [00:06, 909.53 examples/s]5311 examples [00:06, 904.44 examples/s]5402 examples [00:06, 899.58 examples/s]5494 examples [00:06, 903.81 examples/s]5585 examples [00:06, 902.71 examples/s]5677 examples [00:06, 905.98 examples/s]5769 examples [00:06, 908.17 examples/s]5860 examples [00:06, 907.95 examples/s]5951 examples [00:06, 907.25 examples/s]6042 examples [00:07, 906.29 examples/s]6135 examples [00:07, 912.48 examples/s]6227 examples [00:07, 910.28 examples/s]6319 examples [00:07, 908.90 examples/s]6410 examples [00:07, 906.21 examples/s]6501 examples [00:07, 901.27 examples/s]6592 examples [00:07, 903.31 examples/s]6686 examples [00:07, 911.61 examples/s]6778 examples [00:07, 910.66 examples/s]6870 examples [00:08, 910.92 examples/s]6962 examples [00:08, 907.50 examples/s]7054 examples [00:08, 909.90 examples/s]7148 examples [00:08, 917.26 examples/s]7240 examples [00:08, 915.04 examples/s]7332 examples [00:08, 907.59 examples/s]7423 examples [00:08, 906.60 examples/s]7514 examples [00:08, 892.56 examples/s]7604 examples [00:08, 890.55 examples/s]7695 examples [00:08, 893.53 examples/s]7785 examples [00:09, 892.37 examples/s]7875 examples [00:09, 892.38 examples/s]7965 examples [00:09, 885.63 examples/s]8055 examples [00:09, 887.74 examples/s]8149 examples [00:09, 900.41 examples/s]8243 examples [00:09, 911.57 examples/s]8335 examples [00:09, 912.93 examples/s]8428 examples [00:09, 916.00 examples/s]8520 examples [00:09, 912.75 examples/s]8612 examples [00:09, 902.58 examples/s]8703 examples [00:10, 902.73 examples/s]8794 examples [00:10, 896.02 examples/s]8884 examples [00:10, 894.86 examples/s]8974 examples [00:10, 892.09 examples/s]9067 examples [00:10, 901.10 examples/s]9162 examples [00:10, 913.31 examples/s]9257 examples [00:10, 921.62 examples/s]9351 examples [00:10, 925.57 examples/s]9444 examples [00:10, 918.19 examples/s]9536 examples [00:10, 916.01 examples/s]9628 examples [00:11, 900.34 examples/s]9720 examples [00:11, 904.04 examples/s]9813 examples [00:11, 908.85 examples/s]9904 examples [00:11, 904.10 examples/s]9995 examples [00:11, 894.82 examples/s]10085 examples [00:11, 834.31 examples/s]10176 examples [00:11, 854.20 examples/s]10266 examples [00:11, 865.81 examples/s]10354 examples [00:11, 869.29 examples/s]10442 examples [00:11, 867.84 examples/s]10533 examples [00:12, 880.00 examples/s]10622 examples [00:12, 880.92 examples/s]10714 examples [00:12, 890.57 examples/s]10804 examples [00:12, 889.44 examples/s]10895 examples [00:12, 894.53 examples/s]10986 examples [00:12, 896.78 examples/s]11077 examples [00:12, 899.79 examples/s]11168 examples [00:12, 899.96 examples/s]11259 examples [00:12, 898.59 examples/s]11349 examples [00:13, 891.48 examples/s]11442 examples [00:13, 899.77 examples/s]11534 examples [00:13, 903.63 examples/s]11625 examples [00:13, 902.62 examples/s]11716 examples [00:13, 895.84 examples/s]11807 examples [00:13, 899.40 examples/s]11897 examples [00:13, 897.32 examples/s]11988 examples [00:13, 899.52 examples/s]12079 examples [00:13, 901.13 examples/s]12173 examples [00:13, 910.14 examples/s]12265 examples [00:14, 905.45 examples/s]12356 examples [00:14, 905.94 examples/s]12451 examples [00:14, 917.34 examples/s]12544 examples [00:14, 919.16 examples/s]12637 examples [00:14, 921.80 examples/s]12730 examples [00:14, 911.84 examples/s]12822 examples [00:14, 911.39 examples/s]12915 examples [00:14, 915.24 examples/s]13007 examples [00:14, 909.14 examples/s]13098 examples [00:14, 908.75 examples/s]13189 examples [00:15, 903.25 examples/s]13280 examples [00:15, 903.58 examples/s]13372 examples [00:15, 907.08 examples/s]13463 examples [00:15, 900.82 examples/s]13554 examples [00:15, 897.12 examples/s]13644 examples [00:15, 894.34 examples/s]13734 examples [00:15, 887.55 examples/s]13823 examples [00:15, 887.17 examples/s]13913 examples [00:15, 889.52 examples/s]14003 examples [00:15, 891.49 examples/s]14093 examples [00:16, 886.13 examples/s]14182 examples [00:16, 883.92 examples/s]14271 examples [00:16, 881.05 examples/s]14360 examples [00:16, 877.67 examples/s]14449 examples [00:16, 880.30 examples/s]14538 examples [00:16, 878.67 examples/s]14626 examples [00:16, 875.62 examples/s]14717 examples [00:16, 884.21 examples/s]14806 examples [00:16, 875.91 examples/s]14895 examples [00:16, 878.02 examples/s]14984 examples [00:17, 879.37 examples/s]15073 examples [00:17, 879.33 examples/s]15163 examples [00:17, 883.31 examples/s]15252 examples [00:17, 880.38 examples/s]15342 examples [00:17, 883.98 examples/s]15431 examples [00:17, 879.89 examples/s]15520 examples [00:17, 876.16 examples/s]15608 examples [00:17, 863.27 examples/s]15695 examples [00:17, 857.65 examples/s]15782 examples [00:17, 860.74 examples/s]15869 examples [00:18, 862.23 examples/s]15962 examples [00:18, 879.45 examples/s]16052 examples [00:18, 882.91 examples/s]16143 examples [00:18, 888.86 examples/s]16233 examples [00:18, 890.06 examples/s]16323 examples [00:18, 892.59 examples/s]16415 examples [00:18, 898.50 examples/s]16505 examples [00:18, 896.29 examples/s]16598 examples [00:18, 904.74 examples/s]16689 examples [00:19, 849.52 examples/s]16775 examples [00:19, 849.47 examples/s]16862 examples [00:19, 854.17 examples/s]16951 examples [00:19, 862.72 examples/s]17043 examples [00:19, 876.38 examples/s]17131 examples [00:19, 873.63 examples/s]17219 examples [00:19, 866.68 examples/s]17311 examples [00:19, 881.31 examples/s]17401 examples [00:19, 886.74 examples/s]17490 examples [00:19, 886.66 examples/s]17580 examples [00:20, 888.90 examples/s]17670 examples [00:20, 891.83 examples/s]17761 examples [00:20, 895.26 examples/s]17851 examples [00:20, 890.69 examples/s]17941 examples [00:20, 885.62 examples/s]18030 examples [00:20, 876.49 examples/s]18118 examples [00:20, 872.27 examples/s]18208 examples [00:20, 878.52 examples/s]18299 examples [00:20, 887.60 examples/s]18390 examples [00:20, 892.08 examples/s]18484 examples [00:21, 903.30 examples/s]18575 examples [00:21, 903.78 examples/s]18666 examples [00:21, 904.94 examples/s]18757 examples [00:21, 905.95 examples/s]18848 examples [00:21, 899.85 examples/s]18939 examples [00:21, 900.73 examples/s]19030 examples [00:21, 900.27 examples/s]19123 examples [00:21, 907.22 examples/s]19217 examples [00:21, 914.47 examples/s]19309 examples [00:21, 908.82 examples/s]19401 examples [00:22, 910.81 examples/s]19493 examples [00:22, 907.53 examples/s]19584 examples [00:22, 903.54 examples/s]19677 examples [00:22, 908.97 examples/s]19769 examples [00:22, 909.55 examples/s]19861 examples [00:22, 910.80 examples/s]19953 examples [00:22, 907.98 examples/s]20044 examples [00:22, 853.52 examples/s]20134 examples [00:22, 866.27 examples/s]20227 examples [00:22, 884.24 examples/s]20317 examples [00:23, 888.42 examples/s]20412 examples [00:23, 904.12 examples/s]20508 examples [00:23, 917.97 examples/s]20601 examples [00:23, 884.99 examples/s]20692 examples [00:23, 890.38 examples/s]20782 examples [00:23, 889.54 examples/s]20872 examples [00:23, 876.23 examples/s]20964 examples [00:23, 886.54 examples/s]21055 examples [00:23, 891.02 examples/s]21145 examples [00:24, 885.94 examples/s]21234 examples [00:24, 871.04 examples/s]21323 examples [00:24, 876.02 examples/s]21415 examples [00:24, 886.83 examples/s]21507 examples [00:24, 895.44 examples/s]21597 examples [00:24, 885.30 examples/s]21686 examples [00:24, 874.19 examples/s]21774 examples [00:24, 870.16 examples/s]21866 examples [00:24, 883.49 examples/s]21957 examples [00:24, 889.41 examples/s]22048 examples [00:25, 892.92 examples/s]22138 examples [00:25, 892.77 examples/s]22228 examples [00:25, 891.40 examples/s]22319 examples [00:25, 895.63 examples/s]22413 examples [00:25, 907.35 examples/s]22504 examples [00:25, 903.65 examples/s]22595 examples [00:25, 894.07 examples/s]22685 examples [00:25, 893.15 examples/s]22777 examples [00:25, 900.48 examples/s]22870 examples [00:25, 908.13 examples/s]22961 examples [00:26, 903.19 examples/s]23054 examples [00:26, 911.06 examples/s]23147 examples [00:26, 916.33 examples/s]23239 examples [00:26, 916.59 examples/s]23331 examples [00:26, 915.44 examples/s]23423 examples [00:26, 873.57 examples/s]23511 examples [00:26, 873.39 examples/s]23599 examples [00:26, 870.08 examples/s]23690 examples [00:26, 879.50 examples/s]23782 examples [00:26, 890.80 examples/s]23872 examples [00:27, 889.41 examples/s]23962 examples [00:27, 888.58 examples/s]24056 examples [00:27, 900.81 examples/s]24149 examples [00:27, 908.14 examples/s]24240 examples [00:27, 898.51 examples/s]24330 examples [00:27, 892.61 examples/s]24420 examples [00:27, 893.91 examples/s]24510 examples [00:27, 893.98 examples/s]24600 examples [00:27, 873.53 examples/s]24689 examples [00:27, 876.20 examples/s]24777 examples [00:28, 874.54 examples/s]24868 examples [00:28, 883.26 examples/s]24960 examples [00:28, 891.10 examples/s]25050 examples [00:28, 888.74 examples/s]25141 examples [00:28, 892.56 examples/s]25234 examples [00:28, 901.02 examples/s]25327 examples [00:28, 906.98 examples/s]25418 examples [00:28, 895.74 examples/s]25513 examples [00:28, 909.53 examples/s]25605 examples [00:28, 911.53 examples/s]25697 examples [00:29, 911.51 examples/s]25791 examples [00:29, 918.32 examples/s]25883 examples [00:29, 892.10 examples/s]25973 examples [00:29, 887.02 examples/s]26062 examples [00:29, 884.81 examples/s]26152 examples [00:29, 888.55 examples/s]26241 examples [00:29, 884.82 examples/s]26330 examples [00:29, 871.99 examples/s]26419 examples [00:29, 876.85 examples/s]26509 examples [00:30, 882.57 examples/s]26598 examples [00:30, 884.48 examples/s]26687 examples [00:30, 881.11 examples/s]26778 examples [00:30, 887.71 examples/s]26871 examples [00:30, 899.60 examples/s]26962 examples [00:30, 896.31 examples/s]27053 examples [00:30, 899.98 examples/s]27144 examples [00:30, 893.83 examples/s]27234 examples [00:30, 873.44 examples/s]27322 examples [00:30, 840.94 examples/s]27412 examples [00:31, 855.26 examples/s]27505 examples [00:31, 874.04 examples/s]27595 examples [00:31, 880.84 examples/s]27687 examples [00:31, 891.43 examples/s]27778 examples [00:31, 893.84 examples/s]27870 examples [00:31, 901.40 examples/s]27962 examples [00:31, 906.14 examples/s]28053 examples [00:31, 903.89 examples/s]28144 examples [00:31, 890.11 examples/s]28234 examples [00:31, 847.91 examples/s]28326 examples [00:32, 867.88 examples/s]28415 examples [00:32, 872.21 examples/s]28505 examples [00:32, 878.17 examples/s]28597 examples [00:32, 889.98 examples/s]28687 examples [00:32, 887.47 examples/s]28776 examples [00:32, 886.54 examples/s]28867 examples [00:32, 892.77 examples/s]28957 examples [00:32, 888.02 examples/s]29050 examples [00:32, 897.66 examples/s]29141 examples [00:32, 900.81 examples/s]29232 examples [00:33, 895.09 examples/s]29322 examples [00:33, 895.39 examples/s]29412 examples [00:33, 889.12 examples/s]29505 examples [00:33, 899.15 examples/s]29597 examples [00:33, 903.97 examples/s]29689 examples [00:33, 906.77 examples/s]29780 examples [00:33, 902.71 examples/s]29871 examples [00:33, 898.63 examples/s]29963 examples [00:33, 902.59 examples/s]30054 examples [00:34, 845.12 examples/s]30145 examples [00:34, 862.10 examples/s]30234 examples [00:34, 869.82 examples/s]30323 examples [00:34, 873.60 examples/s]30412 examples [00:34, 878.12 examples/s]30501 examples [00:34, 858.66 examples/s]30588 examples [00:34, 847.10 examples/s]30677 examples [00:34, 857.19 examples/s]30763 examples [00:34, 856.41 examples/s]30851 examples [00:34, 863.24 examples/s]30940 examples [00:35, 869.31 examples/s]31035 examples [00:35, 889.81 examples/s]31125 examples [00:35, 890.10 examples/s]31215 examples [00:35, 877.51 examples/s]31304 examples [00:35, 881.01 examples/s]31395 examples [00:35, 887.40 examples/s]31485 examples [00:35, 890.59 examples/s]31577 examples [00:35, 896.24 examples/s]31667 examples [00:35, 896.73 examples/s]31757 examples [00:35, 886.49 examples/s]31847 examples [00:36, 888.70 examples/s]31941 examples [00:36, 902.51 examples/s]32032 examples [00:36, 900.97 examples/s]32123 examples [00:36, 894.95 examples/s]32213 examples [00:36, 895.39 examples/s]32303 examples [00:36, 886.45 examples/s]32395 examples [00:36, 895.16 examples/s]32485 examples [00:36, 889.20 examples/s]32575 examples [00:36, 892.02 examples/s]32666 examples [00:36, 895.09 examples/s]32756 examples [00:37, 895.67 examples/s]32846 examples [00:37, 891.31 examples/s]32936 examples [00:37, 892.48 examples/s]33026 examples [00:37, 891.16 examples/s]33116 examples [00:37, 889.64 examples/s]33207 examples [00:37, 893.33 examples/s]33297 examples [00:37, 889.90 examples/s]33388 examples [00:37, 895.26 examples/s]33478 examples [00:37, 895.78 examples/s]33569 examples [00:37, 898.32 examples/s]33659 examples [00:38, 896.57 examples/s]33750 examples [00:38, 899.13 examples/s]33842 examples [00:38, 904.49 examples/s]33933 examples [00:38, 904.30 examples/s]34025 examples [00:38, 906.90 examples/s]34116 examples [00:38, 883.95 examples/s]34205 examples [00:38, 881.45 examples/s]34295 examples [00:38, 885.45 examples/s]34384 examples [00:38, 879.55 examples/s]34473 examples [00:39, 876.84 examples/s]34561 examples [00:39, 875.86 examples/s]34652 examples [00:39, 883.40 examples/s]34741 examples [00:39, 878.78 examples/s]34830 examples [00:39, 879.35 examples/s]34920 examples [00:39, 883.04 examples/s]35016 examples [00:39, 904.53 examples/s]35110 examples [00:39, 905.25 examples/s]35201 examples [00:39, 872.94 examples/s]35289 examples [00:39, 871.41 examples/s]35380 examples [00:40, 880.78 examples/s]35471 examples [00:40, 887.64 examples/s]35561 examples [00:40, 889.97 examples/s]35655 examples [00:40, 903.01 examples/s]35746 examples [00:40, 896.50 examples/s]35836 examples [00:40, 883.70 examples/s]35925 examples [00:40, 881.46 examples/s]36014 examples [00:40, 873.13 examples/s]36104 examples [00:40, 880.93 examples/s]36193 examples [00:40, 869.72 examples/s]36281 examples [00:41, 868.39 examples/s]36368 examples [00:41, 865.28 examples/s]36455 examples [00:41, 860.00 examples/s]36542 examples [00:41, 852.59 examples/s]36628 examples [00:41, 837.88 examples/s]36712 examples [00:41, 795.71 examples/s]36797 examples [00:41, 811.19 examples/s]36883 examples [00:41, 822.38 examples/s]36967 examples [00:41, 826.84 examples/s]37050 examples [00:41, 826.24 examples/s]37133 examples [00:42, 810.43 examples/s]37215 examples [00:42, 796.33 examples/s]37299 examples [00:42, 808.34 examples/s]37390 examples [00:42, 834.45 examples/s]37474 examples [00:42, 834.84 examples/s]37561 examples [00:42, 844.62 examples/s]37651 examples [00:42, 858.89 examples/s]37738 examples [00:42, 846.43 examples/s]37828 examples [00:42, 860.16 examples/s]37922 examples [00:43, 881.20 examples/s]38015 examples [00:43, 893.81 examples/s]38108 examples [00:43, 902.07 examples/s]38201 examples [00:43, 908.62 examples/s]38292 examples [00:43, 896.37 examples/s]38384 examples [00:43, 901.64 examples/s]38477 examples [00:43, 907.61 examples/s]38569 examples [00:43, 908.83 examples/s]38660 examples [00:43, 908.40 examples/s]38751 examples [00:43, 904.29 examples/s]38844 examples [00:44, 910.99 examples/s]38936 examples [00:44, 912.56 examples/s]39028 examples [00:44, 903.40 examples/s]39120 examples [00:44, 905.51 examples/s]39211 examples [00:44, 895.88 examples/s]39303 examples [00:44, 901.93 examples/s]39394 examples [00:44, 876.25 examples/s]39482 examples [00:44, 873.82 examples/s]39572 examples [00:44, 880.30 examples/s]39661 examples [00:44, 865.95 examples/s]39748 examples [00:45, 864.45 examples/s]39838 examples [00:45, 873.43 examples/s]39929 examples [00:45, 881.91 examples/s]40018 examples [00:45, 832.87 examples/s]40107 examples [00:45, 847.64 examples/s]40193 examples [00:45, 840.76 examples/s]40285 examples [00:45, 861.49 examples/s]40375 examples [00:45, 871.78 examples/s]40463 examples [00:45, 870.91 examples/s]40551 examples [00:45, 872.40 examples/s]40642 examples [00:46, 881.43 examples/s]40734 examples [00:46, 892.06 examples/s]40829 examples [00:46, 906.16 examples/s]40921 examples [00:46, 909.29 examples/s]41013 examples [00:46, 906.36 examples/s]41105 examples [00:46, 908.86 examples/s]41199 examples [00:46, 916.12 examples/s]41291 examples [00:46, 916.97 examples/s]41383 examples [00:46, 913.36 examples/s]41475 examples [00:46, 890.74 examples/s]41568 examples [00:47, 902.07 examples/s]41660 examples [00:47, 904.87 examples/s]41751 examples [00:47, 906.20 examples/s]41843 examples [00:47, 908.90 examples/s]41934 examples [00:47, 895.52 examples/s]42025 examples [00:47, 898.40 examples/s]42115 examples [00:47, 896.70 examples/s]42205 examples [00:47, 896.38 examples/s]42295 examples [00:47, 895.04 examples/s]42385 examples [00:48, 885.77 examples/s]42474 examples [00:48, 883.57 examples/s]42563 examples [00:48, 882.82 examples/s]42652 examples [00:48, 871.92 examples/s]42740 examples [00:48, 871.74 examples/s]42828 examples [00:48, 871.74 examples/s]42918 examples [00:48, 879.79 examples/s]43007 examples [00:48, 855.09 examples/s]43095 examples [00:48, 861.27 examples/s]43185 examples [00:48, 871.75 examples/s]43273 examples [00:49, 860.52 examples/s]43361 examples [00:49, 865.13 examples/s]43449 examples [00:49, 868.92 examples/s]43536 examples [00:49, 849.33 examples/s]43622 examples [00:49, 836.97 examples/s]43706 examples [00:49, 829.69 examples/s]43791 examples [00:49, 834.91 examples/s]43877 examples [00:49, 840.32 examples/s]43963 examples [00:49, 844.91 examples/s]44049 examples [00:49, 846.58 examples/s]44134 examples [00:50, 809.61 examples/s]44217 examples [00:50, 812.94 examples/s]44306 examples [00:50, 834.07 examples/s]44395 examples [00:50, 849.19 examples/s]44481 examples [00:50, 847.21 examples/s]44566 examples [00:50, 832.13 examples/s]44655 examples [00:50, 847.95 examples/s]44747 examples [00:50, 867.53 examples/s]44835 examples [00:50, 868.40 examples/s]44929 examples [00:50, 887.37 examples/s]45018 examples [00:51, 872.46 examples/s]45107 examples [00:51, 874.61 examples/s]45196 examples [00:51, 876.60 examples/s]45290 examples [00:51, 892.91 examples/s]45383 examples [00:51, 902.87 examples/s]45474 examples [00:51, 901.51 examples/s]45565 examples [00:51, 900.01 examples/s]45656 examples [00:51, 889.54 examples/s]45748 examples [00:51, 896.51 examples/s]45839 examples [00:52, 900.16 examples/s]45930 examples [00:52, 899.30 examples/s]46021 examples [00:52, 899.93 examples/s]46112 examples [00:52, 898.92 examples/s]46202 examples [00:52, 898.06 examples/s]46292 examples [00:52, 897.95 examples/s]46382 examples [00:52, 897.38 examples/s]46472 examples [00:52, 897.13 examples/s]46562 examples [00:52, 893.80 examples/s]46653 examples [00:52, 896.30 examples/s]46743 examples [00:53, 895.85 examples/s]46833 examples [00:53, 886.30 examples/s]46922 examples [00:53, 869.64 examples/s]47012 examples [00:53, 877.19 examples/s]47102 examples [00:53, 881.59 examples/s]47192 examples [00:53, 885.09 examples/s]47281 examples [00:53, 885.65 examples/s]47371 examples [00:53, 888.02 examples/s]47463 examples [00:53, 895.55 examples/s]47560 examples [00:53, 914.91 examples/s]47652 examples [00:54, 913.56 examples/s]47745 examples [00:54, 915.88 examples/s]47837 examples [00:54, 907.93 examples/s]47928 examples [00:54, 904.01 examples/s]48019 examples [00:54, 898.61 examples/s]48110 examples [00:54, 899.73 examples/s]48201 examples [00:54, 899.62 examples/s]48291 examples [00:54, 873.29 examples/s]48381 examples [00:54, 878.80 examples/s]48473 examples [00:54, 889.07 examples/s]48563 examples [00:55, 886.54 examples/s]48656 examples [00:55, 898.57 examples/s]48746 examples [00:55, 894.26 examples/s]48836 examples [00:55, 890.73 examples/s]48926 examples [00:55, 883.68 examples/s]49015 examples [00:55, 884.63 examples/s]49104 examples [00:55, 876.43 examples/s]49193 examples [00:55, 878.65 examples/s]49282 examples [00:55, 881.04 examples/s]49371 examples [00:55, 882.01 examples/s]49460 examples [00:56, 878.56 examples/s]49549 examples [00:56, 879.27 examples/s]49640 examples [00:56, 886.80 examples/s]49732 examples [00:56, 894.04 examples/s]49824 examples [00:56, 901.33 examples/s]49915 examples [00:56, 902.24 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6112/50000 [00:00<00:00, 61118.02 examples/s] 34%|███▎      | 16786/50000 [00:00<00:00, 70106.61 examples/s] 55%|█████▍    | 27457/50000 [00:00<00:00, 78147.41 examples/s] 77%|███████▋  | 38425/50000 [00:00<00:00, 85522.71 examples/s] 99%|█████████▉| 49410/50000 [00:00<00:00, 91607.50 examples/s]                                                               0 examples [00:00, ? examples/s]68 examples [00:00, 676.38 examples/s]161 examples [00:00, 735.63 examples/s]254 examples [00:00, 782.97 examples/s]346 examples [00:00, 818.56 examples/s]438 examples [00:00, 844.38 examples/s]530 examples [00:00, 865.07 examples/s]622 examples [00:00, 878.99 examples/s]712 examples [00:00, 884.60 examples/s]798 examples [00:00, 843.01 examples/s]881 examples [00:01, 822.26 examples/s]963 examples [00:01, 806.18 examples/s]1045 examples [00:01, 807.27 examples/s]1126 examples [00:01, 795.46 examples/s]1206 examples [00:01, 787.75 examples/s]1286 examples [00:01, 789.96 examples/s]1367 examples [00:01, 793.81 examples/s]1449 examples [00:01, 800.68 examples/s]1530 examples [00:01, 801.49 examples/s]1622 examples [00:01, 831.65 examples/s]1706 examples [00:02, 833.89 examples/s]1792 examples [00:02, 838.76 examples/s]1881 examples [00:02, 851.55 examples/s]1972 examples [00:02, 864.74 examples/s]2059 examples [00:02, 853.17 examples/s]2145 examples [00:02, 847.41 examples/s]2230 examples [00:02, 841.74 examples/s]2317 examples [00:02, 847.13 examples/s]2406 examples [00:02, 859.04 examples/s]2496 examples [00:02, 870.29 examples/s]2584 examples [00:03, 870.75 examples/s]2673 examples [00:03, 873.76 examples/s]2761 examples [00:03, 868.60 examples/s]2855 examples [00:03, 886.74 examples/s]2952 examples [00:03, 907.79 examples/s]3043 examples [00:03, 881.03 examples/s]3137 examples [00:03, 895.13 examples/s]3229 examples [00:03, 900.26 examples/s]3320 examples [00:03, 894.33 examples/s]3411 examples [00:03, 897.32 examples/s]3501 examples [00:04, 886.96 examples/s]3593 examples [00:04, 894.60 examples/s]3683 examples [00:04, 894.38 examples/s]3775 examples [00:04, 899.85 examples/s]3866 examples [00:04, 865.05 examples/s]3957 examples [00:04, 876.53 examples/s]4049 examples [00:04, 886.61 examples/s]4139 examples [00:04, 888.59 examples/s]4230 examples [00:04, 893.86 examples/s]4320 examples [00:05, 890.06 examples/s]4411 examples [00:05, 893.74 examples/s]4501 examples [00:05, 893.21 examples/s]4591 examples [00:05, 865.06 examples/s]4678 examples [00:05, 865.71 examples/s]4767 examples [00:05, 872.82 examples/s]4856 examples [00:05, 876.09 examples/s]4946 examples [00:05, 882.42 examples/s]5038 examples [00:05, 891.10 examples/s]5131 examples [00:05, 899.38 examples/s]5222 examples [00:06, 895.96 examples/s]5312 examples [00:06, 892.02 examples/s]5404 examples [00:06, 898.88 examples/s]5496 examples [00:06, 903.75 examples/s]5590 examples [00:06, 912.70 examples/s]5682 examples [00:06, 909.38 examples/s]5773 examples [00:06, 907.85 examples/s]5868 examples [00:06, 918.02 examples/s]5960 examples [00:06, 915.81 examples/s]6052 examples [00:06, 906.76 examples/s]6143 examples [00:07, 897.92 examples/s]6235 examples [00:07, 903.56 examples/s]6326 examples [00:07, 900.36 examples/s]6417 examples [00:07, 894.76 examples/s]6507 examples [00:07, 895.12 examples/s]6597 examples [00:07, 892.29 examples/s]6687 examples [00:07, 892.13 examples/s]6777 examples [00:07, 882.36 examples/s]6866 examples [00:07, 875.02 examples/s]6958 examples [00:07, 887.61 examples/s]7047 examples [00:08, 884.13 examples/s]7136 examples [00:08, 879.52 examples/s]7224 examples [00:08, 872.63 examples/s]7312 examples [00:08, 873.23 examples/s]7401 examples [00:08, 873.88 examples/s]7489 examples [00:08, 872.15 examples/s]7579 examples [00:08, 878.92 examples/s]7669 examples [00:08, 884.11 examples/s]7758 examples [00:08, 883.50 examples/s]7847 examples [00:08, 885.36 examples/s]7936 examples [00:09, 875.14 examples/s]8024 examples [00:09, 848.03 examples/s]8112 examples [00:09, 854.24 examples/s]8201 examples [00:09, 862.12 examples/s]8288 examples [00:09, 860.51 examples/s]8375 examples [00:09, 860.00 examples/s]8464 examples [00:09, 867.30 examples/s]8553 examples [00:09, 871.89 examples/s]8646 examples [00:09, 886.79 examples/s]8735 examples [00:09, 883.89 examples/s]8825 examples [00:10, 888.65 examples/s]8914 examples [00:10, 884.30 examples/s]9004 examples [00:10, 886.17 examples/s]9096 examples [00:10, 893.52 examples/s]9187 examples [00:10, 897.69 examples/s]9278 examples [00:10, 898.54 examples/s]9368 examples [00:10, 892.55 examples/s]9460 examples [00:10, 898.89 examples/s]9551 examples [00:10, 899.47 examples/s]9641 examples [00:11, 898.85 examples/s]9733 examples [00:11, 903.85 examples/s]9825 examples [00:11, 905.88 examples/s]9916 examples [00:11, 900.11 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]100%|█████████▉| 9962/10000 [00:00<00:00, 99608.93 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9AHR5X/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete9AHR5X/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a455bd158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a455bd158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a455bd158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a90f9b0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a04627f28>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/resnet34_benchmark_mnist.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/vision\/MNIST\/",
    "dataset":"MNIST",
    "data_type":"tch_dataset",
    "batch_size":10,
    "train":true
  },
  "preprocessors":[
    {
      "name":"tch_dataset_start",
      "uri":"mlmodels.preprocess.generic:get_dataset_torch",
      "args":{
        "dataloader":"torchvision.datasets:MNIST",
        "to_image":true,
        "transform":{
          "uri":"mlmodels.preprocess.image:torch_transform_mnist",
          "pass_data_pars":false,
          "arg":{

          }
        },
        "shuffle":true,
        "download":true
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5a455bd158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5a455bd158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5a455bd158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a43aa3320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5a43aa3a58>), {}) 

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{
    "data_path":"dataset\/recommender\/",
    "dataset":"IMDB_sample.txt",
    "data_type":"csv_dataset",
    "batch_size":64,
    "train":true
  },
  "preprocessors":[
    {
      "uri":"mlmodels.model_tch.textcnn:split_train_valid",
      "args":{
        "frac":0.99
      }
    },
    {
      "uri":"mlmodels.model_tch.textcnn:create_tabular_dataset",
      "args":{
        "lang":"en",
        "pretrained_emb":"glove.6B.300d"
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.textcnn:split_train_valid {'frac': 0.99} 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn.json Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range

  




 #################################################################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json 
 

  #####  Load JSON data_pars 

  {
  "data_info":{

  },
  "preprocessors":[
    {
      "name":"DataReader",
      "uri":"mlmodels.model_tch.util_transformer:TransformerDataReader",
      "args":{
        "train":{
          "uri":"sentence_transformers.readers:NLIDataReader",
          "dataset":"dataset\/text\/AllNLI\/train"
        },
        "test":{
          "uri":"sentence_transformers.readers:STSBenchmarkDataReader",
          "dataset":"dataset\/text\/stsbenchmark\/val"
        }
      }
    }
  ]
} 

  
 #####  Load DataLoader  

  
 #####  compute DataLoader  

  URL:  mlmodels.model_tch.util_transformer:TransformerDataReader {'train': {'uri': 'sentence_transformers.readers:NLIDataReader', 'dataset': 'dataset/text/AllNLI/train'}, 'test': {'uri': 'sentence_transformers.readers:STSBenchmarkDataReader', 'dataset': 'dataset/text/stsbenchmark/val'}} 

  
###### load_callable_from_uri LOADED <class 'mlmodels.model_tch.util_transformer.TransformerDataReader'> 
cls_name : TransformerDataReader

  
 Object Creation 

  
 Object Compute 
Error /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.json Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libtorch_cpu.so: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 248, in compute
    preprocessor_func = load_function(uri)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['mlmodels.model_tch.textcnn', 'split_train_valid'] notfound, libtorch_cpu.so: cannot open shared object file: No such file or directory, tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 672, in load_function_uri
    return  getattr(importlib.import_module(package), name)
AttributeError: module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 683, in load_function_uri
    package_name = str(Path(package).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 452, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 275, in compute
    obj_preprocessor.compute(input_tmp)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/util_transformer.py", line 275, in compute
    test_func = load_function(self.test_reader)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 688, in load_function_uri
    raise NameError(f"Module {pkg} notfound, {e1}, {e2}")
NameError: Module ['sentence_transformers.readers', 'STSBenchmarkDataReader'] notfound, module 'sentence_transformers.readers' has no attribute 'STSBenchmarkDataReader', tuple index out of range





 ********************************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py --do test  

  #### Test unit Dataloader/Dataset   #################################### 

  


 #################### tf_dataset_download 

  tf_dataset_download mlmodels/preprocess/generic:tf_dataset_download {'train_samples': 500, 'test_samples': 500} 

  MNIST 

  Dataset Name is :  mnist 
WARNING:absl:Dataset mnist is hosted on GCS. It will automatically be downloaded to your
local data directory. If you'd instead prefer to read directly from our public
GCS bucket (recommended if you're running on GCP), you can instead set
data_dir=gs://tfds-data/datasets.

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.09 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.09 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.09 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.01 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.88 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.49 file/s]2020-08-13 00:09:21.435982: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-13 00:09:21.440716: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-13 00:09:21.440956: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55699f3982a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-13 00:09:21.440974: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist2', 'train', 'test', 'cifar10', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 474698.95it/s] 91%|█████████ | 9003008/9912422 [00:00<00:01, 676603.40it/s]9920512it [00:00, 46965759.25it/s]                           
0it [00:00, ?it/s]32768it [00:00, 667626.63it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 470153.23it/s]1654784it [00:00, 11708791.46it/s]                         
0it [00:00, ?it/s]8192it [00:00, 218430.28it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/raw
Processing...
Done!

  


 #################### PandasDataset 

  PandasDataset mlmodels/preprocess/generic:pandasDataset {'colX': ['colX'], 'coly': ['coly'], 'encoding': 'ISO-8859-1', 'read_csv_parm': {'usecols': [0, 1], 'names': ['coly', 'colX'], 'encoding': 'ISO-8859-1'}} 

  


 #################### NumpyDataset 

  NumpyDataset mlmodels/preprocess/generic:NumpyDataset {'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'numpy_loader_args': {}} 

Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/train/mnist.npz
