
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '061c074e0ea8d028c7c86b76298dc9fc3ebb6845', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/061c074e0ea8d028c7c86b76298dc9fc3ebb6845

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff1c64139d8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff1c64139d8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff23107e510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff23107e510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff2502adea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff2502adea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff1de3b0950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff1de3b0950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff1de3b0950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 19%|█▊        | 1843200/9912422 [00:00<00:00, 14662956.42it/s]9920512it [00:00, 28774373.48it/s]                             
0it [00:00, ?it/s]32768it [00:00, 686367.69it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 489205.25it/s]1654784it [00:00, 12170118.31it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 53315.63it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff1bce4c6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff1bce5e9b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff1de3b0598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff1de3b0598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff1de3b0598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:57,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:56,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.71 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:18,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:18,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:18,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:13,  9.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:13,  9.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:13,  9.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:13,  9.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:13,  9.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:13,  9.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 12.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 16.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 16.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 20.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 20.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 20.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 20.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 20.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 20.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 20.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 25.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 25.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 30.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 30.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 30.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 30.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 34.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 34.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 34.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 34.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 34.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 34.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 34.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 39.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 39.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 39.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 42.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 42.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 42.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 42.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 42.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 42.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 42.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 45.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 45.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 45.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 45.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 45.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 45.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 45.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 47.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 47.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 47.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 47.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 47.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 47.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 47.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 49.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 49.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 49.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 49.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 49.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 49.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 49.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 50.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 50.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 50.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 50.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 50.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 51.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 51.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 51.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:01, 51.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:01, 51.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 51.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 51.39 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 52.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 52.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 52.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 52.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 52.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 52.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 52.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 52.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 52.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 53.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 53.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 53.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 53.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 55.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 55.60 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 56.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 56.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 56.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 56.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 56.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 56.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 56.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 53.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 53.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 53.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 53.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 53.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 53.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 53.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 49.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 49.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.56s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.56s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 49.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.56s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 49.42 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.87s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.56s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 49.42 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.87s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.87s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 27.57 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.88s/ url]
0 examples [00:00, ? examples/s]2020-07-22 18:10:12.620158: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-22 18:10:12.624194: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-22 18:10:12.624370: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557857cc1030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 18:10:12.624387: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
65 examples [00:00, 646.94 examples/s]158 examples [00:00, 711.50 examples/s]256 examples [00:00, 773.49 examples/s]352 examples [00:00, 819.85 examples/s]444 examples [00:00, 846.69 examples/s]540 examples [00:00, 877.36 examples/s]635 examples [00:00, 895.76 examples/s]721 examples [00:00, 873.74 examples/s]809 examples [00:00, 873.71 examples/s]895 examples [00:01, 869.33 examples/s]982 examples [00:01, 868.60 examples/s]1068 examples [00:01, 862.45 examples/s]1154 examples [00:01, 822.14 examples/s]1241 examples [00:01, 835.07 examples/s]1327 examples [00:01, 841.65 examples/s]1417 examples [00:01, 857.66 examples/s]1506 examples [00:01, 867.01 examples/s]1602 examples [00:01, 891.59 examples/s]1695 examples [00:01, 899.78 examples/s]1786 examples [00:02, 887.64 examples/s]1875 examples [00:02, 870.61 examples/s]1963 examples [00:02, 867.80 examples/s]2052 examples [00:02, 872.07 examples/s]2144 examples [00:02, 883.99 examples/s]2233 examples [00:02, 872.78 examples/s]2321 examples [00:02, 870.20 examples/s]2411 examples [00:02, 876.33 examples/s]2503 examples [00:02, 887.97 examples/s]2597 examples [00:02, 900.85 examples/s]2689 examples [00:03, 904.21 examples/s]2780 examples [00:03, 876.73 examples/s]2873 examples [00:03, 890.68 examples/s]2967 examples [00:03, 902.82 examples/s]3058 examples [00:03, 900.58 examples/s]3149 examples [00:03, 890.63 examples/s]3239 examples [00:03, 884.67 examples/s]3328 examples [00:03, 885.97 examples/s]3417 examples [00:03, 885.34 examples/s]3506 examples [00:03, 884.14 examples/s]3595 examples [00:04, 881.30 examples/s]3684 examples [00:04, 869.55 examples/s]3772 examples [00:04, 867.07 examples/s]3859 examples [00:04, 865.10 examples/s]3946 examples [00:04, 856.42 examples/s]4033 examples [00:04, 860.13 examples/s]4120 examples [00:04, 862.70 examples/s]4207 examples [00:04, 860.61 examples/s]4294 examples [00:04, 862.35 examples/s]4386 examples [00:04, 878.04 examples/s]4479 examples [00:05, 891.31 examples/s]4571 examples [00:05, 898.97 examples/s]4661 examples [00:05, 895.59 examples/s]4754 examples [00:05, 903.71 examples/s]4848 examples [00:05, 913.56 examples/s]4942 examples [00:05, 921.32 examples/s]5035 examples [00:05, 893.45 examples/s]5128 examples [00:05, 903.30 examples/s]5221 examples [00:05, 908.40 examples/s]5317 examples [00:06, 920.29 examples/s]5414 examples [00:06, 933.84 examples/s]5509 examples [00:06, 937.14 examples/s]5603 examples [00:06, 935.63 examples/s]5697 examples [00:06, 926.46 examples/s]5790 examples [00:06, 916.35 examples/s]5882 examples [00:06, 888.98 examples/s]5972 examples [00:06, 889.29 examples/s]6062 examples [00:06, 877.44 examples/s]6156 examples [00:06, 895.11 examples/s]6248 examples [00:07, 902.02 examples/s]6339 examples [00:07, 903.91 examples/s]6434 examples [00:07, 917.19 examples/s]6527 examples [00:07, 920.80 examples/s]6621 examples [00:07, 925.04 examples/s]6714 examples [00:07, 896.66 examples/s]6804 examples [00:07, 882.28 examples/s]6893 examples [00:07, 876.93 examples/s]6981 examples [00:07, 850.87 examples/s]7071 examples [00:07, 863.63 examples/s]7158 examples [00:08, 859.03 examples/s]7249 examples [00:08, 872.00 examples/s]7337 examples [00:08, 849.22 examples/s]7423 examples [00:08, 850.67 examples/s]7509 examples [00:08, 844.15 examples/s]7594 examples [00:08, 834.53 examples/s]7679 examples [00:08, 837.67 examples/s]7765 examples [00:08, 840.43 examples/s]7850 examples [00:08, 839.93 examples/s]7939 examples [00:09, 851.03 examples/s]8027 examples [00:09, 856.60 examples/s]8116 examples [00:09, 864.42 examples/s]8203 examples [00:09, 857.50 examples/s]8292 examples [00:09, 866.10 examples/s]8380 examples [00:09, 866.89 examples/s]8468 examples [00:09, 870.58 examples/s]8556 examples [00:09, 862.80 examples/s]8645 examples [00:09, 868.04 examples/s]8732 examples [00:09, 864.34 examples/s]8819 examples [00:10, 865.37 examples/s]8909 examples [00:10, 874.89 examples/s]8998 examples [00:10, 879.34 examples/s]9086 examples [00:10, 876.56 examples/s]9174 examples [00:10, 867.23 examples/s]9261 examples [00:10, 863.04 examples/s]9348 examples [00:10, 811.38 examples/s]9430 examples [00:10, 749.34 examples/s]9507 examples [00:10, 746.61 examples/s]9592 examples [00:10, 773.97 examples/s]9675 examples [00:11, 788.65 examples/s]9769 examples [00:11, 827.56 examples/s]9863 examples [00:11, 856.27 examples/s]9951 examples [00:11, 860.90 examples/s]10038 examples [00:11, 828.82 examples/s]10131 examples [00:11, 856.74 examples/s]10224 examples [00:11, 875.53 examples/s]10316 examples [00:11, 888.41 examples/s]10409 examples [00:11, 898.19 examples/s]10505 examples [00:11, 913.84 examples/s]10601 examples [00:12, 925.39 examples/s]10697 examples [00:12, 934.27 examples/s]10794 examples [00:12, 942.86 examples/s]10889 examples [00:12, 936.74 examples/s]10983 examples [00:12, 922.59 examples/s]11076 examples [00:12, 910.12 examples/s]11171 examples [00:12, 919.60 examples/s]11268 examples [00:12, 934.00 examples/s]11362 examples [00:12, 926.52 examples/s]11460 examples [00:13, 939.34 examples/s]11557 examples [00:13, 946.45 examples/s]11652 examples [00:13, 943.66 examples/s]11747 examples [00:13, 937.71 examples/s]11841 examples [00:13, 924.87 examples/s]11934 examples [00:13, 906.08 examples/s]12025 examples [00:13, 893.28 examples/s]12118 examples [00:13, 903.14 examples/s]12213 examples [00:13, 914.33 examples/s]12305 examples [00:13, 913.76 examples/s]12397 examples [00:14, 910.84 examples/s]12491 examples [00:14, 917.73 examples/s]12583 examples [00:14, 917.35 examples/s]12677 examples [00:14, 923.57 examples/s]12770 examples [00:14, 917.66 examples/s]12863 examples [00:14, 919.27 examples/s]12955 examples [00:14, 898.36 examples/s]13045 examples [00:14, 896.11 examples/s]13137 examples [00:14, 900.53 examples/s]13228 examples [00:14, 901.32 examples/s]13321 examples [00:15, 909.15 examples/s]13417 examples [00:15, 923.75 examples/s]13511 examples [00:15, 928.38 examples/s]13604 examples [00:15, 928.84 examples/s]13697 examples [00:15, 918.51 examples/s]13789 examples [00:15, 859.31 examples/s]13876 examples [00:15, 805.97 examples/s]13960 examples [00:15, 813.98 examples/s]14050 examples [00:15, 836.47 examples/s]14135 examples [00:16, 765.11 examples/s]14214 examples [00:16, 732.82 examples/s]14289 examples [00:16, 727.32 examples/s]14382 examples [00:16, 776.75 examples/s]14473 examples [00:16, 810.80 examples/s]14563 examples [00:16, 835.23 examples/s]14656 examples [00:16, 860.52 examples/s]14751 examples [00:16, 885.11 examples/s]14847 examples [00:16, 903.94 examples/s]14940 examples [00:16, 909.39 examples/s]15033 examples [00:17, 913.37 examples/s]15129 examples [00:17, 926.54 examples/s]15225 examples [00:17, 934.01 examples/s]15319 examples [00:17, 918.68 examples/s]15412 examples [00:17, 919.24 examples/s]15505 examples [00:17, 909.29 examples/s]15597 examples [00:17, 891.43 examples/s]15687 examples [00:17, 862.03 examples/s]15777 examples [00:17, 871.18 examples/s]15866 examples [00:18, 875.91 examples/s]15958 examples [00:18, 887.72 examples/s]16052 examples [00:18, 900.27 examples/s]16147 examples [00:18, 911.75 examples/s]16241 examples [00:18, 918.15 examples/s]16333 examples [00:18, 892.37 examples/s]16428 examples [00:18, 907.47 examples/s]16523 examples [00:18, 919.14 examples/s]16616 examples [00:18, 913.22 examples/s]16710 examples [00:18, 920.64 examples/s]16803 examples [00:19, 894.23 examples/s]16899 examples [00:19, 910.66 examples/s]16993 examples [00:19, 916.57 examples/s]17085 examples [00:19, 911.69 examples/s]17180 examples [00:19, 922.17 examples/s]17273 examples [00:19, 910.85 examples/s]17366 examples [00:19, 915.87 examples/s]17459 examples [00:19, 919.34 examples/s]17553 examples [00:19, 924.73 examples/s]17649 examples [00:19, 933.29 examples/s]17743 examples [00:20, 928.10 examples/s]17837 examples [00:20, 930.60 examples/s]17933 examples [00:20, 936.68 examples/s]18028 examples [00:20, 933.07 examples/s]18122 examples [00:20, 911.19 examples/s]18214 examples [00:20, 859.46 examples/s]18301 examples [00:20, 861.25 examples/s]18388 examples [00:20, 860.98 examples/s]18475 examples [00:20, 861.74 examples/s]18563 examples [00:20, 863.94 examples/s]18650 examples [00:21, 862.17 examples/s]18741 examples [00:21, 873.93 examples/s]18829 examples [00:21, 857.53 examples/s]18915 examples [00:21, 835.77 examples/s]19002 examples [00:21, 845.71 examples/s]19090 examples [00:21, 853.40 examples/s]19181 examples [00:21, 868.25 examples/s]19270 examples [00:21, 874.46 examples/s]19360 examples [00:21, 880.07 examples/s]19452 examples [00:22, 889.32 examples/s]19544 examples [00:22, 897.43 examples/s]19636 examples [00:22, 901.94 examples/s]19729 examples [00:22, 910.02 examples/s]19822 examples [00:22, 912.90 examples/s]19914 examples [00:22, 907.49 examples/s]20005 examples [00:22, 833.87 examples/s]20095 examples [00:22, 851.06 examples/s]20186 examples [00:22, 866.98 examples/s]20277 examples [00:22, 877.00 examples/s]20366 examples [00:23, 879.99 examples/s]20458 examples [00:23, 889.49 examples/s]20551 examples [00:23, 898.98 examples/s]20643 examples [00:23, 904.57 examples/s]20736 examples [00:23, 911.14 examples/s]20828 examples [00:23, 906.20 examples/s]20920 examples [00:23, 907.68 examples/s]21014 examples [00:23, 914.11 examples/s]21106 examples [00:23, 888.70 examples/s]21196 examples [00:23, 880.23 examples/s]21286 examples [00:24, 883.51 examples/s]21377 examples [00:24, 888.56 examples/s]21469 examples [00:24, 895.21 examples/s]21561 examples [00:24, 900.70 examples/s]21654 examples [00:24, 909.06 examples/s]21746 examples [00:24, 910.57 examples/s]21839 examples [00:24, 914.30 examples/s]21931 examples [00:24, 908.69 examples/s]22022 examples [00:24, 907.45 examples/s]22114 examples [00:24, 908.73 examples/s]22205 examples [00:25, 900.61 examples/s]22296 examples [00:25, 895.19 examples/s]22388 examples [00:25, 902.06 examples/s]22479 examples [00:25, 892.83 examples/s]22569 examples [00:25, 883.03 examples/s]22658 examples [00:25, 878.46 examples/s]22746 examples [00:25, 863.10 examples/s]22835 examples [00:25, 868.64 examples/s]22924 examples [00:25, 874.85 examples/s]23015 examples [00:26, 883.83 examples/s]23104 examples [00:26, 880.48 examples/s]23196 examples [00:26, 890.54 examples/s]23287 examples [00:26, 895.31 examples/s]23377 examples [00:26, 896.40 examples/s]23467 examples [00:26, 853.76 examples/s]23553 examples [00:26, 853.13 examples/s]23639 examples [00:26, 852.89 examples/s]23725 examples [00:26, 849.27 examples/s]23811 examples [00:26, 840.89 examples/s]23902 examples [00:27, 859.78 examples/s]23989 examples [00:27, 860.10 examples/s]24076 examples [00:27, 851.20 examples/s]24169 examples [00:27, 871.47 examples/s]24257 examples [00:27, 873.44 examples/s]24349 examples [00:27, 883.70 examples/s]24438 examples [00:27, 857.72 examples/s]24529 examples [00:27, 872.72 examples/s]24621 examples [00:27, 883.93 examples/s]24710 examples [00:27, 854.23 examples/s]24800 examples [00:28, 865.60 examples/s]24887 examples [00:28, 865.56 examples/s]24976 examples [00:28, 872.73 examples/s]25064 examples [00:28, 869.64 examples/s]25152 examples [00:28, 854.66 examples/s]25242 examples [00:28, 866.79 examples/s]25329 examples [00:28, 857.15 examples/s]25415 examples [00:28, 857.51 examples/s]25501 examples [00:28, 847.68 examples/s]25589 examples [00:28, 855.35 examples/s]25680 examples [00:29, 868.79 examples/s]25767 examples [00:29, 844.30 examples/s]25852 examples [00:29, 836.88 examples/s]25939 examples [00:29, 846.43 examples/s]26025 examples [00:29, 850.17 examples/s]26116 examples [00:29, 866.98 examples/s]26204 examples [00:29, 869.03 examples/s]26295 examples [00:29, 879.41 examples/s]26384 examples [00:29, 877.96 examples/s]26474 examples [00:30, 883.90 examples/s]26564 examples [00:30, 886.75 examples/s]26653 examples [00:30, 877.42 examples/s]26745 examples [00:30, 886.85 examples/s]26837 examples [00:30, 896.04 examples/s]26927 examples [00:30, 891.11 examples/s]27019 examples [00:30, 899.31 examples/s]27109 examples [00:30, 892.27 examples/s]27201 examples [00:30, 899.95 examples/s]27292 examples [00:30, 898.53 examples/s]27382 examples [00:31, 871.75 examples/s]27470 examples [00:31, 865.38 examples/s]27557 examples [00:31, 852.33 examples/s]27647 examples [00:31, 865.82 examples/s]27734 examples [00:31, 865.67 examples/s]27821 examples [00:31, 863.68 examples/s]27913 examples [00:31, 879.39 examples/s]28006 examples [00:31, 891.44 examples/s]28096 examples [00:31, 878.30 examples/s]28189 examples [00:31, 890.32 examples/s]28280 examples [00:32, 895.04 examples/s]28370 examples [00:32, 890.37 examples/s]28460 examples [00:32, 883.87 examples/s]28554 examples [00:32, 898.54 examples/s]28649 examples [00:32, 911.69 examples/s]28741 examples [00:32, 911.96 examples/s]28833 examples [00:32, 903.61 examples/s]28924 examples [00:32, 900.07 examples/s]29017 examples [00:32, 908.27 examples/s]29108 examples [00:32, 898.45 examples/s]29202 examples [00:33, 910.46 examples/s]29295 examples [00:33, 914.43 examples/s]29387 examples [00:33, 911.53 examples/s]29481 examples [00:33, 918.87 examples/s]29573 examples [00:33, 909.63 examples/s]29665 examples [00:33, 908.39 examples/s]29756 examples [00:33, 905.83 examples/s]29850 examples [00:33, 913.90 examples/s]29942 examples [00:33, 915.41 examples/s]30034 examples [00:34, 846.63 examples/s]30127 examples [00:34, 869.41 examples/s]30219 examples [00:34, 882.94 examples/s]30309 examples [00:34, 885.95 examples/s]30399 examples [00:34, 849.10 examples/s]30491 examples [00:34, 866.97 examples/s]30583 examples [00:34, 881.96 examples/s]30673 examples [00:34, 884.83 examples/s]30763 examples [00:34, 888.67 examples/s]30853 examples [00:34, 891.62 examples/s]30945 examples [00:35, 897.79 examples/s]31038 examples [00:35, 905.96 examples/s]31129 examples [00:35, 897.88 examples/s]31219 examples [00:35, 897.18 examples/s]31312 examples [00:35, 904.89 examples/s]31403 examples [00:35, 905.14 examples/s]31498 examples [00:35, 915.86 examples/s]31591 examples [00:35, 918.00 examples/s]31684 examples [00:35, 920.90 examples/s]31780 examples [00:35, 931.24 examples/s]31875 examples [00:36, 933.34 examples/s]31973 examples [00:36, 944.87 examples/s]32068 examples [00:36, 945.12 examples/s]32163 examples [00:36, 940.74 examples/s]32258 examples [00:36, 930.69 examples/s]32352 examples [00:36, 922.29 examples/s]32445 examples [00:36, 907.06 examples/s]32536 examples [00:36, 907.40 examples/s]32633 examples [00:36, 924.39 examples/s]32729 examples [00:36, 933.65 examples/s]32823 examples [00:37, 930.43 examples/s]32921 examples [00:37, 943.44 examples/s]33016 examples [00:37, 938.60 examples/s]33115 examples [00:37, 952.15 examples/s]33215 examples [00:37, 965.19 examples/s]33312 examples [00:37, 958.81 examples/s]33408 examples [00:37, 939.00 examples/s]33503 examples [00:37, 938.40 examples/s]33598 examples [00:37, 939.25 examples/s]33693 examples [00:37, 938.53 examples/s]33787 examples [00:38, 926.56 examples/s]33881 examples [00:38, 928.73 examples/s]33974 examples [00:38, 898.16 examples/s]34066 examples [00:38, 902.25 examples/s]34157 examples [00:38, 892.00 examples/s]34250 examples [00:38, 902.78 examples/s]34343 examples [00:38, 909.31 examples/s]34435 examples [00:38, 905.29 examples/s]34528 examples [00:38, 911.42 examples/s]34621 examples [00:38, 914.99 examples/s]34713 examples [00:39, 912.55 examples/s]34805 examples [00:39, 904.74 examples/s]34897 examples [00:39, 907.30 examples/s]34988 examples [00:39, 878.15 examples/s]35081 examples [00:39, 890.66 examples/s]35176 examples [00:39, 906.14 examples/s]35271 examples [00:39, 916.90 examples/s]35363 examples [00:39, 900.38 examples/s]35454 examples [00:39, 882.27 examples/s]35546 examples [00:40, 893.24 examples/s]35640 examples [00:40, 906.22 examples/s]35733 examples [00:40, 912.90 examples/s]35825 examples [00:40, 914.26 examples/s]35921 examples [00:40, 925.18 examples/s]36014 examples [00:40, 919.32 examples/s]36109 examples [00:40, 926.90 examples/s]36202 examples [00:40, 926.96 examples/s]36295 examples [00:40, 914.94 examples/s]36390 examples [00:40, 924.49 examples/s]36487 examples [00:41, 935.41 examples/s]36583 examples [00:41, 940.68 examples/s]36681 examples [00:41, 951.98 examples/s]36777 examples [00:41, 949.14 examples/s]36872 examples [00:41, 949.32 examples/s]36967 examples [00:41, 943.05 examples/s]37066 examples [00:41, 954.92 examples/s]37163 examples [00:41, 958.45 examples/s]37259 examples [00:41, 946.65 examples/s]37354 examples [00:41, 920.54 examples/s]37447 examples [00:42, 922.85 examples/s]37543 examples [00:42, 932.46 examples/s]37638 examples [00:42, 936.51 examples/s]37734 examples [00:42, 941.01 examples/s]37829 examples [00:42, 938.89 examples/s]37923 examples [00:42, 934.42 examples/s]38020 examples [00:42, 943.86 examples/s]38118 examples [00:42, 951.89 examples/s]38214 examples [00:42, 941.12 examples/s]38311 examples [00:42, 947.27 examples/s]38406 examples [00:43, 938.24 examples/s]38505 examples [00:43, 950.86 examples/s]38601 examples [00:43, 946.62 examples/s]38696 examples [00:43, 936.84 examples/s]38793 examples [00:43, 944.53 examples/s]38890 examples [00:43, 949.17 examples/s]38988 examples [00:43, 956.01 examples/s]39084 examples [00:43, 953.59 examples/s]39180 examples [00:43, 939.06 examples/s]39274 examples [00:43, 934.78 examples/s]39368 examples [00:44, 922.89 examples/s]39461 examples [00:44, 910.05 examples/s]39553 examples [00:44, 899.08 examples/s]39644 examples [00:44, 868.07 examples/s]39740 examples [00:44, 891.49 examples/s]39836 examples [00:44, 909.07 examples/s]39932 examples [00:44, 923.37 examples/s]40025 examples [00:44, 864.44 examples/s]40124 examples [00:44, 897.25 examples/s]40219 examples [00:45, 912.07 examples/s]40316 examples [00:45, 928.18 examples/s]40410 examples [00:45, 920.33 examples/s]40504 examples [00:45, 924.79 examples/s]40600 examples [00:45, 934.55 examples/s]40697 examples [00:45, 944.79 examples/s]40794 examples [00:45, 949.61 examples/s]40890 examples [00:45, 951.59 examples/s]40986 examples [00:45, 941.72 examples/s]41083 examples [00:45, 947.97 examples/s]41181 examples [00:46, 955.55 examples/s]41281 examples [00:46, 965.78 examples/s]41380 examples [00:46, 971.70 examples/s]41478 examples [00:46, 960.38 examples/s]41575 examples [00:46, 945.42 examples/s]41670 examples [00:46, 946.59 examples/s]41765 examples [00:46, 945.88 examples/s]41860 examples [00:46, 946.60 examples/s]41955 examples [00:46, 898.83 examples/s]42046 examples [00:46, 886.83 examples/s]42141 examples [00:47, 903.14 examples/s]42236 examples [00:47, 915.16 examples/s]42330 examples [00:47, 922.45 examples/s]42423 examples [00:47, 911.86 examples/s]42515 examples [00:47, 898.28 examples/s]42605 examples [00:47, 891.47 examples/s]42695 examples [00:47, 885.91 examples/s]42786 examples [00:47, 892.67 examples/s]42876 examples [00:47, 893.79 examples/s]42966 examples [00:48, 886.36 examples/s]43055 examples [00:48, 880.33 examples/s]43144 examples [00:48, 876.55 examples/s]43233 examples [00:48, 880.49 examples/s]43322 examples [00:48, 883.00 examples/s]43416 examples [00:48, 898.57 examples/s]43509 examples [00:48, 906.69 examples/s]43603 examples [00:48, 913.70 examples/s]43696 examples [00:48, 916.58 examples/s]43789 examples [00:48, 920.41 examples/s]43882 examples [00:49, 915.20 examples/s]43974 examples [00:49, 911.41 examples/s]44066 examples [00:49, 904.83 examples/s]44157 examples [00:49, 900.85 examples/s]44248 examples [00:49, 881.56 examples/s]44337 examples [00:49, 882.22 examples/s]44429 examples [00:49, 890.83 examples/s]44520 examples [00:49, 896.22 examples/s]44612 examples [00:49, 901.77 examples/s]44703 examples [00:49, 899.87 examples/s]44794 examples [00:50, 901.17 examples/s]44885 examples [00:50, 903.00 examples/s]44977 examples [00:50, 905.10 examples/s]45068 examples [00:50, 902.28 examples/s]45159 examples [00:50, 894.30 examples/s]45249 examples [00:50, 890.25 examples/s]45341 examples [00:50, 897.72 examples/s]45432 examples [00:50, 900.19 examples/s]45523 examples [00:50, 895.36 examples/s]45613 examples [00:50, 891.33 examples/s]45703 examples [00:51, 891.92 examples/s]45793 examples [00:51, 882.11 examples/s]45882 examples [00:51, 862.81 examples/s]45975 examples [00:51, 881.75 examples/s]46066 examples [00:51, 889.22 examples/s]46159 examples [00:51, 900.64 examples/s]46253 examples [00:51, 910.61 examples/s]46345 examples [00:51, 905.92 examples/s]46436 examples [00:51, 900.67 examples/s]46527 examples [00:52, 854.97 examples/s]46619 examples [00:52, 871.09 examples/s]46711 examples [00:52, 867.23 examples/s]46802 examples [00:52, 877.95 examples/s]46896 examples [00:52, 895.40 examples/s]46986 examples [00:52, 884.06 examples/s]47075 examples [00:52, 881.07 examples/s]47164 examples [00:52, 873.04 examples/s]47252 examples [00:52, 865.30 examples/s]47339 examples [00:52, 858.25 examples/s]47428 examples [00:53, 865.23 examples/s]47515 examples [00:53, 864.03 examples/s]47602 examples [00:53, 864.96 examples/s]47689 examples [00:53, 855.95 examples/s]47777 examples [00:53, 862.94 examples/s]47865 examples [00:53, 865.83 examples/s]47952 examples [00:53, 842.71 examples/s]48042 examples [00:53, 858.76 examples/s]48133 examples [00:53, 870.80 examples/s]48228 examples [00:53, 892.85 examples/s]48323 examples [00:54, 908.51 examples/s]48417 examples [00:54, 916.24 examples/s]48512 examples [00:54, 923.94 examples/s]48605 examples [00:54, 924.84 examples/s]48702 examples [00:54, 936.36 examples/s]48796 examples [00:54, 927.38 examples/s]48893 examples [00:54, 938.50 examples/s]48991 examples [00:54, 947.98 examples/s]49089 examples [00:54, 957.10 examples/s]49185 examples [00:54, 955.12 examples/s]49281 examples [00:55, 944.81 examples/s]49376 examples [00:55, 928.51 examples/s]49473 examples [00:55, 938.79 examples/s]49569 examples [00:55, 944.97 examples/s]49667 examples [00:55, 954.40 examples/s]49763 examples [00:55, 951.64 examples/s]49861 examples [00:55, 958.00 examples/s]49958 examples [00:55, 959.75 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7141/50000 [00:00<00:00, 71409.73 examples/s] 37%|███▋      | 18717/50000 [00:00<00:00, 80682.45 examples/s] 61%|██████▏   | 30732/50000 [00:00<00:00, 89501.55 examples/s] 86%|████████▌ | 43021/50000 [00:00<00:00, 97443.12 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 744.75 examples/s]167 examples [00:00, 788.36 examples/s]263 examples [00:00, 832.35 examples/s]360 examples [00:00, 868.12 examples/s]456 examples [00:00, 893.17 examples/s]555 examples [00:00, 918.44 examples/s]651 examples [00:00, 927.99 examples/s]745 examples [00:00, 928.16 examples/s]837 examples [00:00, 923.88 examples/s]933 examples [00:01, 933.23 examples/s]1025 examples [00:01, 921.81 examples/s]1119 examples [00:01, 926.71 examples/s]1211 examples [00:01, 915.56 examples/s]1308 examples [00:01, 928.22 examples/s]1404 examples [00:01, 935.17 examples/s]1501 examples [00:01, 943.24 examples/s]1599 examples [00:01, 953.52 examples/s]1697 examples [00:01, 958.52 examples/s]1795 examples [00:01, 963.28 examples/s]1894 examples [00:02, 968.26 examples/s]1991 examples [00:02, 951.90 examples/s]2089 examples [00:02, 960.07 examples/s]2186 examples [00:02, 927.46 examples/s]2281 examples [00:02, 931.13 examples/s]2378 examples [00:02, 940.20 examples/s]2475 examples [00:02, 948.50 examples/s]2571 examples [00:02, 949.52 examples/s]2667 examples [00:02, 943.90 examples/s]2765 examples [00:02, 952.64 examples/s]2862 examples [00:03, 955.69 examples/s]2959 examples [00:03, 959.64 examples/s]3056 examples [00:03, 957.76 examples/s]3152 examples [00:03, 947.12 examples/s]3247 examples [00:03, 946.77 examples/s]3342 examples [00:03, 945.34 examples/s]3438 examples [00:03, 947.36 examples/s]3533 examples [00:03, 941.18 examples/s]3628 examples [00:03, 919.70 examples/s]3721 examples [00:03, 921.45 examples/s]3816 examples [00:04, 929.00 examples/s]3909 examples [00:04, 925.09 examples/s]4007 examples [00:04, 938.40 examples/s]4101 examples [00:04, 931.00 examples/s]4199 examples [00:04, 943.03 examples/s]4294 examples [00:04, 943.13 examples/s]4390 examples [00:04, 947.28 examples/s]4485 examples [00:04, 946.36 examples/s]4580 examples [00:04, 936.63 examples/s]4679 examples [00:04, 948.96 examples/s]4776 examples [00:05, 952.54 examples/s]4872 examples [00:05, 939.38 examples/s]4967 examples [00:05, 933.32 examples/s]5061 examples [00:05, 927.82 examples/s]5154 examples [00:05, 925.59 examples/s]5251 examples [00:05, 936.10 examples/s]5347 examples [00:05, 941.82 examples/s]5443 examples [00:05, 945.37 examples/s]5538 examples [00:05, 928.99 examples/s]5632 examples [00:05, 929.50 examples/s]5726 examples [00:06, 930.16 examples/s]5821 examples [00:06, 933.15 examples/s]5917 examples [00:06, 940.45 examples/s]6012 examples [00:06, 939.17 examples/s]6111 examples [00:06, 953.37 examples/s]6209 examples [00:06, 960.46 examples/s]6308 examples [00:06, 968.37 examples/s]6407 examples [00:06, 974.03 examples/s]6505 examples [00:06, 969.00 examples/s]6604 examples [00:07, 973.73 examples/s]6702 examples [00:07, 961.25 examples/s]6799 examples [00:07, 959.83 examples/s]6896 examples [00:07, 944.95 examples/s]6991 examples [00:07, 933.05 examples/s]7088 examples [00:07, 941.72 examples/s]7183 examples [00:07, 916.74 examples/s]7275 examples [00:07, 914.90 examples/s]7375 examples [00:07, 937.01 examples/s]7469 examples [00:07, 923.55 examples/s]7562 examples [00:08, 923.66 examples/s]7661 examples [00:08, 940.48 examples/s]7756 examples [00:08, 929.96 examples/s]7850 examples [00:08, 923.00 examples/s]7944 examples [00:08, 927.18 examples/s]8038 examples [00:08, 929.90 examples/s]8136 examples [00:08, 941.93 examples/s]8235 examples [00:08, 954.45 examples/s]8334 examples [00:08, 962.78 examples/s]8431 examples [00:08, 960.20 examples/s]8530 examples [00:09, 968.47 examples/s]8627 examples [00:09, 967.82 examples/s]8725 examples [00:09, 968.18 examples/s]8825 examples [00:09, 976.07 examples/s]8923 examples [00:09, 969.59 examples/s]9021 examples [00:09, 970.28 examples/s]9119 examples [00:09, 959.43 examples/s]9221 examples [00:09, 976.11 examples/s]9321 examples [00:09, 979.82 examples/s]9420 examples [00:09, 974.06 examples/s]9518 examples [00:10, 966.26 examples/s]9617 examples [00:10, 970.42 examples/s]9715 examples [00:10, 968.01 examples/s]9812 examples [00:10, 952.25 examples/s]9908 examples [00:10, 946.24 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteR4D2F1/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteR4D2F1/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff1de3b0950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff1de3b0950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff1de3b0950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff1dc48ec50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff2498b8f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff1de3b0950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff1de3b0950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff1de3b0950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff1bce4c6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff1bce5e518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff156d57268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff156d57268> 

  function with postional parmater data_info <function split_train_valid at 0x7ff156d57268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff156d57378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff156d57378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff156d57378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=cdb660d367479db7a67e1848df738a23422aa0de619f39ea536326914439fbec
  Stored in directory: /tmp/pip-ephem-wheel-cache-pj44icq9/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:17:39, 11.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:26:41, 16.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:09:55, 23.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:07:28, 33.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:58:29, 47.9kB/s].vector_cache/glove.6B.zip:   1%|          | 9.29M/862M [00:01<3:27:38, 68.5kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:24:52, 97.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.5M/862M [00:01<1:40:49, 139kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<1:10:11, 199kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:01<49:05, 283kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:01<34:13, 404kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.5M/862M [00:01<23:57, 574kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<16:45, 816kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<11:50, 1.15MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<08:18, 1.63MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:02<06:44, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<06:37, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:59, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:28, 2.45MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<05:55, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:33, 2.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:51, 2.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<07:12, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<04:09, 3.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<08:42, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<07:31, 1.76MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<05:33, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<06:54, 1.90MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<06:11, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<04:36, 2.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:20, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<05:46, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<04:19, 3.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<06:07, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<06:58, 1.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:16<05:27, 2.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<03:58, 3.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<10:53, 1.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<08:58, 1.44MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<06:36, 1.95MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<07:37, 1.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<07:59, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<06:10, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:21<04:26, 2.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<17:53, 716kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<13:49, 925kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<09:59, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<09:58, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<08:20, 1.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<06:08, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:16, 1.74MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:23, 1.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:47, 2.64MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:19, 1.99MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<07:00, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:32, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:53, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:24, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:06, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:14, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<03:48, 3.26MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:55:26, 17.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<8:21:50, 24.7kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<5:50:52, 35.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<4:07:47, 49.8kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<2:55:55, 70.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<2:03:34, 99.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<1:26:22, 142kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<1:08:20, 179kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<49:04, 250kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<34:35, 353kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<27:00, 451kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<20:09, 604kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<14:23, 845kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:54, 939kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<10:16, 1.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:29, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<05:53, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:15:11, 19.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<7:11:32, 27.9kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<5:01:39, 39.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<3:33:36, 56.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:31:56, 78.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<1:46:50, 112kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:16:24, 156kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<54:29, 218kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<38:22, 310kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<26:56, 440kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<41:54, 283kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<31:45, 373kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<22:47, 519kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<17:47, 662kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<13:40, 860kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<09:48, 1.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:36, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:06, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:52, 1.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:58, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:43, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:07, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:12, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<03:48, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<33:03, 350kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<24:18, 476kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<17:16, 668kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<14:47, 778kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<11:31, 998kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<08:20, 1.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:31, 1.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<07:08, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:17, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:22, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:37, 2.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:10, 2.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:35, 2.02MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:13, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:52, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:30, 3.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<27:53, 402kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<20:41, 542kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<14:41, 762kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<12:52, 867kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:19, 985kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<08:29, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:42, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:31, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:48, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:57, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:27, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:05, 2.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<03:40, 2.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<44:40, 245kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<32:23, 338kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<22:54, 477kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<18:31, 588kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<15:10, 717kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<11:05, 980kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<07:52, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<11:03, 977kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<08:52, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<06:28, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:02, 1.53MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:01, 1.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:28, 2.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:39, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:02, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:47, 2.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:08, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:40, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:32, 2.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:56, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:32, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:26, 3.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:29, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:24, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:29, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:23, 2.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:11, 3.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:16:00, 136kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<54:15, 190kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<38:09, 270kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<28:59, 354kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<22:23, 458kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<16:11, 633kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<11:24, 894kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<1:20:49, 126kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<57:25, 177kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<40:19, 252kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:37<28:17, 358kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<1:06:06, 153kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<48:19, 210kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<34:14, 295kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<24:00, 420kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<22:43, 443kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<16:55, 594kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<12:04, 830kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<10:45, 928kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<09:39, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:16, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<05:10, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<37:21, 266kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<27:10, 365kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<19:13, 514kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<15:43, 627kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<12:01, 819kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<08:39, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:19, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:49, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:54, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<04:16, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<07:03, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:55, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:23, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:19, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:42, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:29, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:14, 2.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<39:07, 245kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<28:21, 338kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<20:00, 477kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<16:11, 587kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<13:16, 717kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<09:46, 972kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<06:54, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<1:05:17, 145kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<46:38, 202kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<32:48, 287kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<25:03, 374kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<18:30, 506kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<13:09, 710kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<11:21, 819kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<08:54, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:25, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:39, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:35, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:07, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:01, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<05:23, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:14, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:04, 2.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<1:07:00, 136kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<47:48, 190kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<33:34, 270kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<25:32, 354kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<19:44, 457kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<14:13, 634kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<10:00, 897kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<21:38, 414kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<16:05, 557kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<11:25, 781kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<10:03, 885kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:52, 1.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:39, 1.33MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:14<04:45, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<1:15:02, 118kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<53:23, 165kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<37:27, 235kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<28:09, 311kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<21:30, 407kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<15:25, 567kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<10:52, 801kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<11:48, 736kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<09:11, 945kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<06:38, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:38, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:31, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<04:04, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:52, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:17, 1.99MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:13, 2.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:13, 2.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:43, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<03:43, 2.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<02:41, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<07:53, 1.07MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<06:22, 1.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:40, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:12, 1.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:25, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:12, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<03:00, 2.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<15:35, 531kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<11:46, 702kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<08:26, 977kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:46, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:11, 1.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:27, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<03:54, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<1:00:43, 134kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<43:19, 188kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<30:26, 266kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<23:06, 349kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<17:01, 474kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<12:04, 665kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<10:17, 777kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:53, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:40, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<04:03, 1.95MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<34:50, 228kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<25:11, 315kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<17:46, 444kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<14:13, 553kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<10:45, 731kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:42, 1.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<07:12, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:50, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:16, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:49, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<04:57, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:51, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:47, 2.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<1:04:23, 119kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<45:49, 167kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<32:09, 237kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<24:10, 314kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<18:33, 409kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<13:22, 566kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<09:34, 789kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<08:33, 880kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<08:02, 936kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:02, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:22, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:51, 1.54MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:19, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:11, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<03:02, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:56, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:59, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:33, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:17, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:51, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:13, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:00, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:57, 2.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:48, 1.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:21, 1.66MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:28, 2.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:31, 2.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<06:06, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<05:57, 1.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:35, 1.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:18, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:53, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:25, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<04:54, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:31, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<07:25, 947kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<06:47, 1.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<05:08, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:40, 1.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<07:20, 950kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<06:37, 1.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<05:00, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:35, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<07:28, 923kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<06:47, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:03, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:39, 1.87MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<04:27, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:34, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:33, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:34, 2.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<07:36, 888kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<07:08, 946kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<05:25, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:52, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:01, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<05:38, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:17, 1.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:05, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<08:09, 812kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<07:25, 892kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<05:35, 1.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<03:59, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<06:22, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<05:44, 1.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<04:18, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<03:04, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:58, 929kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<06:12, 1.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:38, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:23<03:17, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<08:18, 772kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:27, 859kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:37, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<04:00, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:05, 1.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:45, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:20, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<03:08, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:58, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:14, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:20, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:24, 2.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:15, 1.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:03, 1.23MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:51, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:46, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:51, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<05:20, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:02, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:33<02:53, 2.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<11:08, 544kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<09:12, 659kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<06:48, 890kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<04:48, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<06:58, 861kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<06:04, 987kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<04:32, 1.32MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<03:14, 1.83MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<15:47, 376kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:39<12:26, 477kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<09:02, 655kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<06:22, 922kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:04, 726kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<07:01, 835kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:12, 1.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<03:42, 1.57MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:54, 1.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:47, 1.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:37, 1.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:37, 2.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:35, 1.59MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:50, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:01, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:11, 2.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:08, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:45, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:35, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<02:35, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<08:26, 661kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<07:16, 768kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<05:25, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:49<03:51, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:44, 961kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:08, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:51, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<02:45, 1.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<08:05, 672kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<07:00, 777kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:13, 1.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<03:43, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:46, 930kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:18, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:57, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:49, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:22, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:21, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:22, 1.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:25, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:35, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:18, 1.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:19, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:24, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:14, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:16, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:32, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<01:49, 2.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<06:15, 814kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<05:25, 940kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:00, 1.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:50, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:44, 1.06MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<04:32, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:29, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<02:29, 2.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<04:23, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<04:02, 1.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:03, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<02:11, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<08:21, 585kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:48, 718kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<04:58, 980kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<03:31, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:28, 881kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<04:49, 1.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:36, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<02:34, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<07:49, 607kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<06:24, 741kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:42, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<03:20, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<08:01, 584kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<06:31, 717kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<04:47, 976kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<03:22, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<08:51, 522kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<07:19, 630kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<05:24, 853kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<03:49, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:19, 853kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:35, 990kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:25, 1.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<02:25, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<13:53, 322kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<10:49, 414kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<07:49, 571kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<05:29, 806kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<06:35, 670kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<05:35, 789kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:08, 1.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:55, 1.48MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:50, 744kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<04:53, 886kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:35, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:32, 1.68MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:06, 1.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:51, 1.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:55, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:05, 2.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:37, 910kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<04:15, 986kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:11, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:16, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:15, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<03:00, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:16, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:16, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:29, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:56, 2.09MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:24, 2.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:19, 1.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:17, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:45, 2.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:54, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<03:05, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:35, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:54, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:08, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:06, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:35, 2.40MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:09, 3.29MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<05:17, 716kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<04:17, 880kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<03:08, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:50, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:32, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:53, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:42<01:20, 2.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<07:20, 497kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<05:50, 624kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<04:15, 852kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:44<02:58, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<53:54, 66.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<38:25, 93.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<26:58, 132kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<18:38, 189kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<1:04:45, 54.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<46:00, 76.3kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<32:14, 108kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<22:23, 155kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<17:17, 199kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<12:45, 270kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<09:04, 378kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<06:48, 496kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<05:25, 622kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<03:57, 850kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:15, 1.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:57, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:12, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:34, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:40, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:19, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:43, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:47, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:49, 1.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:25, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:30, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:36, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:15, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:22, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:30, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:11, 2.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:19, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:27, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:09, 2.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:16, 2.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:06, 2.61MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:48, 3.54MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:06, 1.34MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:59, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:31, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:30, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:33, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:12, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:17, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:23, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:04, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:46, 3.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:07, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:58, 1.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:28, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:03, 2.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:02, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:54, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:24, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:27, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:06, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:47, 3.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<04:44, 508kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:46, 638kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:43, 880kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:55, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:18, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<02:02, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:30, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<01:03, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<03:29, 652kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:51, 797kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:05, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:48, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:39, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:14, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:52, 2.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:51, 555kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:04, 693kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:14, 946kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:52, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:41, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:14, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:14, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:56, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:40, 2.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:11, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:11, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:54, 2.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:38, 2.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<16:02, 116kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<11:32, 161kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<08:05, 228kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<05:35, 324kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<04:42, 381kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<03:37, 495kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<02:35, 685kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<01:47, 967kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:36, 661kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:09, 802kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:33, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<01:05, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:05, 796kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:46, 937kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:17, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<00:54, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:41, 943kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:28, 1.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:05, 1.45MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:46, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:07, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:04, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:48, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:47, 1.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:48, 1.78MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:37, 2.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:47<00:26, 3.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<06:08, 225kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<04:32, 304kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<03:11, 427kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<02:10, 606kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<03:30, 374kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<02:41, 487kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:55, 674kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:30, 829kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:16, 975kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:56, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:49, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:47, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:36, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:35, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:36, 1.81MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:19, 3.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:01, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:53, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:39, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<00:18, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:47, 505kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:25, 635kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:00, 875kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:41, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:13, 682kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<01:00, 824kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:43, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:36, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:34, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:25, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:17, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:50, 828kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:42, 974kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:31, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:26, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:25, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:18, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:12, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:24, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:23, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:15, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:12, 2.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:12, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:06, 3.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<02:41, 132kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<01:55, 182kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:18, 257kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:47, 366kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<01:04, 266kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:47, 355kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:32, 496kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:20, 633kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:16, 777kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:11, 1.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:07, 1.20MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:06, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:03, 1.29MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.57MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.71MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<58:26:12,  1.90it/s]  0%|          | 756/400000 [00:00<40:49:58,  2.72it/s]  0%|          | 1522/400000 [00:00<28:31:57,  3.88it/s]  1%|          | 2237/400000 [00:00<19:56:29,  5.54it/s]  1%|          | 2964/400000 [00:00<13:56:17,  7.91it/s]  1%|          | 3681/400000 [00:01<9:44:37, 11.30it/s]   1%|          | 4440/400000 [00:01<6:48:42, 16.13it/s]  1%|▏         | 5201/400000 [00:01<4:45:48, 23.02it/s]  1%|▏         | 5933/400000 [00:01<3:19:57, 32.84it/s]  2%|▏         | 6670/400000 [00:01<2:19:58, 46.83it/s]  2%|▏         | 7418/400000 [00:01<1:38:03, 66.72it/s]  2%|▏         | 8175/400000 [00:01<1:08:46, 94.96it/s]  2%|▏         | 8915/400000 [00:01<48:18, 134.91it/s]   2%|▏         | 9678/400000 [00:01<34:00, 191.28it/s]  3%|▎         | 10423/400000 [00:01<24:01, 270.29it/s]  3%|▎         | 11167/400000 [00:02<17:03, 379.98it/s]  3%|▎         | 11902/400000 [00:02<12:10, 531.00it/s]  3%|▎         | 12658/400000 [00:02<08:46, 736.39it/s]  3%|▎         | 13404/400000 [00:02<06:23, 1009.28it/s]  4%|▎         | 14147/400000 [00:02<04:43, 1359.74it/s]  4%|▎         | 14882/400000 [00:02<03:34, 1795.00it/s]  4%|▍         | 15618/400000 [00:02<02:45, 2321.49it/s]  4%|▍         | 16358/400000 [00:02<02:11, 2922.99it/s]  4%|▍         | 17099/400000 [00:02<01:47, 3571.79it/s]  4%|▍         | 17834/400000 [00:02<01:30, 4207.93it/s]  5%|▍         | 18565/400000 [00:03<01:19, 4772.56it/s]  5%|▍         | 19332/400000 [00:03<01:10, 5382.16it/s]  5%|▌         | 20066/400000 [00:03<01:05, 5816.36it/s]  5%|▌         | 20810/400000 [00:03<01:00, 6222.78it/s]  5%|▌         | 21544/400000 [00:03<00:58, 6491.55it/s]  6%|▌         | 22275/400000 [00:03<00:56, 6715.26it/s]  6%|▌         | 23005/400000 [00:03<00:55, 6805.58it/s]  6%|▌         | 23727/400000 [00:03<00:55, 6785.53it/s]  6%|▌         | 24457/400000 [00:03<00:54, 6930.63it/s]  6%|▋         | 25191/400000 [00:03<00:53, 7045.40it/s]  6%|▋         | 25919/400000 [00:04<00:52, 7111.20it/s]  7%|▋         | 26654/400000 [00:04<00:52, 7178.95it/s]  7%|▋         | 27402/400000 [00:04<00:51, 7264.67it/s]  7%|▋         | 28152/400000 [00:04<00:50, 7330.96it/s]  7%|▋         | 28890/400000 [00:04<00:51, 7272.70it/s]  7%|▋         | 29621/400000 [00:04<00:52, 7098.41it/s]  8%|▊         | 30356/400000 [00:04<00:51, 7170.76it/s]  8%|▊         | 31076/400000 [00:04<00:53, 6867.56it/s]  8%|▊         | 31806/400000 [00:04<00:52, 6989.39it/s]  8%|▊         | 32542/400000 [00:04<00:51, 7094.45it/s]  8%|▊         | 33255/400000 [00:05<00:51, 7069.94it/s]  8%|▊         | 33976/400000 [00:05<00:51, 7109.38it/s]  9%|▊         | 34709/400000 [00:05<00:50, 7171.99it/s]  9%|▉         | 35430/400000 [00:05<00:50, 7182.34it/s]  9%|▉         | 36150/400000 [00:05<00:50, 7183.57it/s]  9%|▉         | 36869/400000 [00:05<00:50, 7184.69it/s]  9%|▉         | 37613/400000 [00:05<00:49, 7257.93it/s] 10%|▉         | 38340/400000 [00:05<00:50, 7200.22it/s] 10%|▉         | 39061/400000 [00:05<00:50, 7202.96it/s] 10%|▉         | 39806/400000 [00:05<00:49, 7273.87it/s] 10%|█         | 40534/400000 [00:06<00:49, 7255.10it/s] 10%|█         | 41296/400000 [00:06<00:48, 7360.36it/s] 11%|█         | 42066/400000 [00:06<00:47, 7458.45it/s] 11%|█         | 42813/400000 [00:06<00:48, 7407.52it/s] 11%|█         | 43569/400000 [00:06<00:47, 7452.20it/s] 11%|█         | 44315/400000 [00:06<00:47, 7421.84it/s] 11%|█▏        | 45058/400000 [00:06<00:47, 7396.92it/s] 11%|█▏        | 45798/400000 [00:06<00:48, 7266.46it/s] 12%|█▏        | 46526/400000 [00:06<00:48, 7220.00it/s] 12%|█▏        | 47249/400000 [00:07<00:49, 7154.92it/s] 12%|█▏        | 47966/400000 [00:07<00:49, 7152.38it/s] 12%|█▏        | 48691/400000 [00:07<00:48, 7179.49it/s] 12%|█▏        | 49437/400000 [00:07<00:48, 7259.13it/s] 13%|█▎        | 50174/400000 [00:07<00:47, 7292.03it/s] 13%|█▎        | 50904/400000 [00:07<00:48, 7197.74it/s] 13%|█▎        | 51625/400000 [00:07<00:48, 7170.56it/s] 13%|█▎        | 52353/400000 [00:07<00:48, 7202.04it/s] 13%|█▎        | 53074/400000 [00:07<00:48, 7173.71it/s] 13%|█▎        | 53807/400000 [00:07<00:47, 7218.51it/s] 14%|█▎        | 54530/400000 [00:08<00:50, 6901.46it/s] 14%|█▍        | 55226/400000 [00:08<00:49, 6917.54it/s] 14%|█▍        | 55920/400000 [00:08<00:50, 6817.67it/s] 14%|█▍        | 56617/400000 [00:08<00:50, 6860.76it/s] 14%|█▍        | 57347/400000 [00:08<00:49, 6986.38it/s] 15%|█▍        | 58085/400000 [00:08<00:48, 7098.82it/s] 15%|█▍        | 58797/400000 [00:08<00:48, 7104.87it/s] 15%|█▍        | 59533/400000 [00:08<00:47, 7178.67it/s] 15%|█▌        | 60257/400000 [00:08<00:47, 7195.56it/s] 15%|█▌        | 60987/400000 [00:08<00:46, 7225.00it/s] 15%|█▌        | 61710/400000 [00:09<00:47, 7131.94it/s] 16%|█▌        | 62424/400000 [00:09<00:48, 7001.02it/s] 16%|█▌        | 63148/400000 [00:09<00:47, 7069.93it/s] 16%|█▌        | 63859/400000 [00:09<00:47, 7080.71it/s] 16%|█▌        | 64596/400000 [00:09<00:46, 7163.70it/s] 16%|█▋        | 65329/400000 [00:09<00:46, 7210.55it/s] 17%|█▋        | 66054/400000 [00:09<00:46, 7221.17it/s] 17%|█▋        | 66800/400000 [00:09<00:45, 7289.30it/s] 17%|█▋        | 67538/400000 [00:09<00:45, 7313.81it/s] 17%|█▋        | 68276/400000 [00:09<00:45, 7332.57it/s] 17%|█▋        | 69026/400000 [00:10<00:44, 7381.34it/s] 17%|█▋        | 69765/400000 [00:10<00:44, 7350.10it/s] 18%|█▊        | 70501/400000 [00:10<00:44, 7323.51it/s] 18%|█▊        | 71234/400000 [00:10<00:45, 7302.25it/s] 18%|█▊        | 71970/400000 [00:10<00:44, 7316.92it/s] 18%|█▊        | 72702/400000 [00:10<00:45, 7267.11it/s] 18%|█▊        | 73429/400000 [00:10<00:45, 7253.94it/s] 19%|█▊        | 74160/400000 [00:10<00:44, 7269.70it/s] 19%|█▊        | 74934/400000 [00:10<00:43, 7403.47it/s] 19%|█▉        | 75676/400000 [00:10<00:43, 7395.93it/s] 19%|█▉        | 76433/400000 [00:11<00:43, 7446.25it/s] 19%|█▉        | 77179/400000 [00:11<00:43, 7428.70it/s] 19%|█▉        | 77923/400000 [00:11<00:43, 7420.90it/s] 20%|█▉        | 78666/400000 [00:11<00:43, 7323.82it/s] 20%|█▉        | 79413/400000 [00:11<00:43, 7366.18it/s] 20%|██        | 80163/400000 [00:11<00:43, 7404.67it/s] 20%|██        | 80904/400000 [00:11<00:43, 7337.33it/s] 20%|██        | 81639/400000 [00:11<00:43, 7328.80it/s] 21%|██        | 82407/400000 [00:11<00:42, 7428.72it/s] 21%|██        | 83151/400000 [00:11<00:42, 7379.35it/s] 21%|██        | 83890/400000 [00:12<00:43, 7241.44it/s] 21%|██        | 84616/400000 [00:12<00:44, 7152.99it/s] 21%|██▏       | 85333/400000 [00:12<00:44, 7147.95it/s] 22%|██▏       | 86078/400000 [00:12<00:43, 7233.06it/s] 22%|██▏       | 86802/400000 [00:12<00:43, 7220.78it/s] 22%|██▏       | 87530/400000 [00:12<00:43, 7235.78it/s] 22%|██▏       | 88257/400000 [00:12<00:43, 7244.49it/s] 22%|██▏       | 88982/400000 [00:12<00:43, 7213.86it/s] 22%|██▏       | 89705/400000 [00:12<00:42, 7218.40it/s] 23%|██▎       | 90443/400000 [00:12<00:42, 7263.85it/s] 23%|██▎       | 91170/400000 [00:13<00:42, 7219.40it/s] 23%|██▎       | 91893/400000 [00:13<00:43, 7104.31it/s] 23%|██▎       | 92644/400000 [00:13<00:42, 7219.31it/s] 23%|██▎       | 93367/400000 [00:13<00:42, 7219.57it/s] 24%|██▎       | 94111/400000 [00:13<00:42, 7282.70it/s] 24%|██▎       | 94863/400000 [00:13<00:41, 7352.13it/s] 24%|██▍       | 95643/400000 [00:13<00:40, 7479.36it/s] 24%|██▍       | 96392/400000 [00:13<00:40, 7476.01it/s] 24%|██▍       | 97141/400000 [00:13<00:40, 7440.29it/s] 24%|██▍       | 97906/400000 [00:13<00:40, 7501.52it/s] 25%|██▍       | 98657/400000 [00:14<00:40, 7486.64it/s] 25%|██▍       | 99416/400000 [00:14<00:39, 7515.08it/s] 25%|██▌       | 100178/400000 [00:14<00:39, 7546.23it/s] 25%|██▌       | 100933/400000 [00:14<00:39, 7478.25it/s] 25%|██▌       | 101697/400000 [00:14<00:39, 7525.13it/s] 26%|██▌       | 102450/400000 [00:14<00:39, 7523.30it/s] 26%|██▌       | 103203/400000 [00:14<00:39, 7424.14it/s] 26%|██▌       | 103946/400000 [00:14<00:40, 7303.11it/s] 26%|██▌       | 104678/400000 [00:14<00:40, 7232.48it/s] 26%|██▋       | 105417/400000 [00:15<00:40, 7277.19it/s] 27%|██▋       | 106174/400000 [00:15<00:39, 7360.72it/s] 27%|██▋       | 106911/400000 [00:15<00:39, 7335.66it/s] 27%|██▋       | 107646/400000 [00:15<00:41, 7030.09it/s] 27%|██▋       | 108362/400000 [00:15<00:41, 7063.75it/s] 27%|██▋       | 109106/400000 [00:15<00:40, 7171.95it/s] 27%|██▋       | 109862/400000 [00:15<00:39, 7284.03it/s] 28%|██▊       | 110611/400000 [00:15<00:39, 7344.06it/s] 28%|██▊       | 111347/400000 [00:15<00:39, 7321.19it/s] 28%|██▊       | 112090/400000 [00:15<00:39, 7352.91it/s] 28%|██▊       | 112849/400000 [00:16<00:38, 7422.12it/s] 28%|██▊       | 113592/400000 [00:16<00:38, 7363.62it/s] 29%|██▊       | 114329/400000 [00:16<00:38, 7330.05it/s] 29%|██▉       | 115063/400000 [00:16<00:39, 7164.21it/s] 29%|██▉       | 115784/400000 [00:16<00:39, 7176.09it/s] 29%|██▉       | 116510/400000 [00:16<00:39, 7199.93it/s] 29%|██▉       | 117249/400000 [00:16<00:38, 7254.61it/s] 29%|██▉       | 117975/400000 [00:16<00:39, 7178.28it/s] 30%|██▉       | 118694/400000 [00:16<00:39, 7152.06it/s] 30%|██▉       | 119432/400000 [00:16<00:38, 7216.66it/s] 30%|███       | 120167/400000 [00:17<00:38, 7255.58it/s] 30%|███       | 120895/400000 [00:17<00:38, 7259.54it/s] 30%|███       | 121622/400000 [00:17<00:38, 7250.21it/s] 31%|███       | 122348/400000 [00:17<00:39, 7114.60it/s] 31%|███       | 123062/400000 [00:17<00:38, 7121.09it/s] 31%|███       | 123775/400000 [00:17<00:39, 7078.31it/s] 31%|███       | 124484/400000 [00:17<00:38, 7064.97it/s] 31%|███▏      | 125191/400000 [00:17<00:39, 6960.24it/s] 31%|███▏      | 125888/400000 [00:17<00:39, 6941.88it/s] 32%|███▏      | 126605/400000 [00:17<00:39, 7005.64it/s] 32%|███▏      | 127323/400000 [00:18<00:38, 7054.57it/s] 32%|███▏      | 128045/400000 [00:18<00:38, 7103.35it/s] 32%|███▏      | 128779/400000 [00:18<00:37, 7171.97it/s] 32%|███▏      | 129517/400000 [00:18<00:37, 7230.65it/s] 33%|███▎      | 130241/400000 [00:18<00:37, 7220.00it/s] 33%|███▎      | 130964/400000 [00:18<00:37, 7207.83it/s] 33%|███▎      | 131689/400000 [00:18<00:37, 7219.63it/s] 33%|███▎      | 132412/400000 [00:18<00:38, 6972.09it/s] 33%|███▎      | 133112/400000 [00:18<00:38, 6889.36it/s] 33%|███▎      | 133844/400000 [00:18<00:37, 7011.26it/s] 34%|███▎      | 134547/400000 [00:19<00:37, 6991.28it/s] 34%|███▍      | 135259/400000 [00:19<00:37, 7029.05it/s] 34%|███▍      | 136000/400000 [00:19<00:36, 7138.71it/s] 34%|███▍      | 136741/400000 [00:19<00:36, 7216.28it/s] 34%|███▍      | 137500/400000 [00:19<00:35, 7323.83it/s] 35%|███▍      | 138258/400000 [00:19<00:35, 7395.37it/s] 35%|███▍      | 139045/400000 [00:19<00:34, 7529.04it/s] 35%|███▍      | 139800/400000 [00:19<00:34, 7489.41it/s] 35%|███▌      | 140550/400000 [00:19<00:34, 7443.00it/s] 35%|███▌      | 141301/400000 [00:19<00:34, 7461.48it/s] 36%|███▌      | 142048/400000 [00:20<00:34, 7391.49it/s] 36%|███▌      | 142813/400000 [00:20<00:34, 7465.47it/s] 36%|███▌      | 143573/400000 [00:20<00:34, 7502.82it/s] 36%|███▌      | 144324/400000 [00:20<00:34, 7470.40it/s] 36%|███▋      | 145072/400000 [00:20<00:34, 7338.94it/s] 36%|███▋      | 145828/400000 [00:20<00:34, 7400.01it/s] 37%|███▋      | 146569/400000 [00:20<00:34, 7361.02it/s] 37%|███▋      | 147312/400000 [00:20<00:34, 7380.51it/s] 37%|███▋      | 148051/400000 [00:20<00:34, 7285.49it/s] 37%|███▋      | 148781/400000 [00:21<00:35, 7141.42it/s] 37%|███▋      | 149523/400000 [00:21<00:34, 7220.21it/s] 38%|███▊      | 150297/400000 [00:21<00:33, 7368.05it/s] 38%|███▊      | 151063/400000 [00:21<00:33, 7450.58it/s] 38%|███▊      | 151833/400000 [00:21<00:32, 7522.14it/s] 38%|███▊      | 152603/400000 [00:21<00:32, 7568.78it/s] 38%|███▊      | 153395/400000 [00:21<00:32, 7669.49it/s] 39%|███▊      | 154215/400000 [00:21<00:31, 7818.50it/s] 39%|███▊      | 154999/400000 [00:21<00:31, 7724.26it/s] 39%|███▉      | 155773/400000 [00:21<00:32, 7488.27it/s] 39%|███▉      | 156525/400000 [00:22<00:32, 7497.00it/s] 39%|███▉      | 157277/400000 [00:22<00:32, 7493.06it/s] 40%|███▉      | 158028/400000 [00:22<00:32, 7447.78it/s] 40%|███▉      | 158780/400000 [00:22<00:32, 7468.01it/s] 40%|███▉      | 159528/400000 [00:22<00:32, 7459.24it/s] 40%|████      | 160285/400000 [00:22<00:32, 7489.84it/s] 40%|████      | 161035/400000 [00:22<00:32, 7322.02it/s] 40%|████      | 161784/400000 [00:22<00:32, 7370.80it/s] 41%|████      | 162530/400000 [00:22<00:32, 7395.11it/s] 41%|████      | 163277/400000 [00:22<00:31, 7415.99it/s] 41%|████      | 164037/400000 [00:23<00:31, 7469.60it/s] 41%|████      | 164793/400000 [00:23<00:31, 7495.72it/s] 41%|████▏     | 165543/400000 [00:23<00:31, 7458.23it/s] 42%|████▏     | 166315/400000 [00:23<00:31, 7531.30it/s] 42%|████▏     | 167069/400000 [00:23<00:31, 7502.38it/s] 42%|████▏     | 167820/400000 [00:23<00:31, 7431.75it/s] 42%|████▏     | 168569/400000 [00:23<00:31, 7448.69it/s] 42%|████▏     | 169317/400000 [00:23<00:30, 7456.43it/s] 43%|████▎     | 170082/400000 [00:23<00:30, 7512.58it/s] 43%|████▎     | 170835/400000 [00:23<00:30, 7515.82it/s] 43%|████▎     | 171594/400000 [00:24<00:30, 7537.74it/s] 43%|████▎     | 172360/400000 [00:24<00:30, 7571.08it/s] 43%|████▎     | 173118/400000 [00:24<00:30, 7517.40it/s] 43%|████▎     | 173886/400000 [00:24<00:29, 7564.12it/s] 44%|████▎     | 174643/400000 [00:24<00:29, 7519.15it/s] 44%|████▍     | 175447/400000 [00:24<00:29, 7666.04it/s] 44%|████▍     | 176243/400000 [00:24<00:28, 7750.53it/s] 44%|████▍     | 177019/400000 [00:24<00:28, 7731.24it/s] 44%|████▍     | 177795/400000 [00:24<00:28, 7739.07it/s] 45%|████▍     | 178570/400000 [00:24<00:29, 7634.27it/s] 45%|████▍     | 179336/400000 [00:25<00:28, 7640.24it/s] 45%|████▌     | 180117/400000 [00:25<00:28, 7689.80it/s] 45%|████▌     | 180907/400000 [00:25<00:28, 7749.77it/s] 45%|████▌     | 181683/400000 [00:25<00:28, 7737.68it/s] 46%|████▌     | 182458/400000 [00:25<00:28, 7547.48it/s] 46%|████▌     | 183214/400000 [00:25<00:28, 7513.05it/s] 46%|████▌     | 183989/400000 [00:25<00:28, 7580.40it/s] 46%|████▌     | 184755/400000 [00:25<00:28, 7600.30it/s] 46%|████▋     | 185516/400000 [00:25<00:28, 7534.42it/s] 47%|████▋     | 186270/400000 [00:25<00:28, 7450.82it/s] 47%|████▋     | 187028/400000 [00:26<00:28, 7486.88it/s] 47%|████▋     | 187800/400000 [00:26<00:28, 7553.45it/s] 47%|████▋     | 188581/400000 [00:26<00:27, 7626.90it/s] 47%|████▋     | 189345/400000 [00:26<00:27, 7550.99it/s] 48%|████▊     | 190101/400000 [00:26<00:28, 7486.44it/s] 48%|████▊     | 190851/400000 [00:26<00:28, 7466.41it/s] 48%|████▊     | 191598/400000 [00:26<00:28, 7397.05it/s] 48%|████▊     | 192342/400000 [00:26<00:28, 7406.94it/s] 48%|████▊     | 193086/400000 [00:26<00:27, 7415.07it/s] 48%|████▊     | 193828/400000 [00:26<00:28, 7242.71it/s] 49%|████▊     | 194563/400000 [00:27<00:28, 7274.39it/s] 49%|████▉     | 195317/400000 [00:27<00:27, 7352.09it/s] 49%|████▉     | 196072/400000 [00:27<00:27, 7409.91it/s] 49%|████▉     | 196814/400000 [00:27<00:27, 7358.36it/s] 49%|████▉     | 197551/400000 [00:27<00:27, 7336.11it/s] 50%|████▉     | 198299/400000 [00:27<00:27, 7377.85it/s] 50%|████▉     | 199038/400000 [00:27<00:27, 7361.38it/s] 50%|████▉     | 199783/400000 [00:27<00:27, 7386.69it/s] 50%|█████     | 200522/400000 [00:27<00:27, 7380.58it/s] 50%|█████     | 201261/400000 [00:28<00:26, 7361.93it/s] 50%|█████     | 201998/400000 [00:28<00:26, 7351.46it/s] 51%|█████     | 202734/400000 [00:28<00:26, 7337.16it/s] 51%|█████     | 203487/400000 [00:28<00:26, 7391.04it/s] 51%|█████     | 204241/400000 [00:28<00:26, 7433.15it/s] 51%|█████     | 204985/400000 [00:28<00:26, 7405.21it/s] 51%|█████▏    | 205726/400000 [00:28<00:26, 7404.74it/s] 52%|█████▏    | 206499/400000 [00:28<00:25, 7498.56it/s] 52%|█████▏    | 207250/400000 [00:28<00:25, 7498.80it/s] 52%|█████▏    | 208001/400000 [00:28<00:25, 7441.65it/s] 52%|█████▏    | 208746/400000 [00:29<00:25, 7380.95it/s] 52%|█████▏    | 209485/400000 [00:29<00:26, 7297.78it/s] 53%|█████▎    | 210237/400000 [00:29<00:25, 7361.39it/s] 53%|█████▎    | 210999/400000 [00:29<00:25, 7436.17it/s] 53%|█████▎    | 211744/400000 [00:29<00:25, 7418.05it/s] 53%|█████▎    | 212487/400000 [00:29<00:25, 7326.89it/s] 53%|█████▎    | 213232/400000 [00:29<00:25, 7362.71it/s] 53%|█████▎    | 213992/400000 [00:29<00:25, 7431.28it/s] 54%|█████▎    | 214772/400000 [00:29<00:24, 7537.90it/s] 54%|█████▍    | 215554/400000 [00:29<00:24, 7618.79it/s] 54%|█████▍    | 216317/400000 [00:30<00:24, 7511.69it/s] 54%|█████▍    | 217102/400000 [00:30<00:24, 7607.77it/s] 54%|█████▍    | 217871/400000 [00:30<00:23, 7630.20it/s] 55%|█████▍    | 218635/400000 [00:30<00:23, 7627.85it/s] 55%|█████▍    | 219428/400000 [00:30<00:23, 7714.46it/s] 55%|█████▌    | 220200/400000 [00:30<00:23, 7624.32it/s] 55%|█████▌    | 220966/400000 [00:30<00:23, 7633.43it/s] 55%|█████▌    | 221730/400000 [00:30<00:23, 7592.84it/s] 56%|█████▌    | 222490/400000 [00:30<00:23, 7577.10it/s] 56%|█████▌    | 223260/400000 [00:30<00:23, 7610.88it/s] 56%|█████▌    | 224022/400000 [00:31<00:23, 7516.07it/s] 56%|█████▌    | 224775/400000 [00:31<00:23, 7485.12it/s] 56%|█████▋    | 225524/400000 [00:31<00:23, 7461.78it/s] 57%|█████▋    | 226271/400000 [00:31<00:23, 7408.37it/s] 57%|█████▋    | 227042/400000 [00:31<00:23, 7494.43it/s] 57%|█████▋    | 227793/400000 [00:31<00:22, 7497.70it/s] 57%|█████▋    | 228564/400000 [00:31<00:22, 7557.87it/s] 57%|█████▋    | 229341/400000 [00:31<00:22, 7619.05it/s] 58%|█████▊    | 230104/400000 [00:31<00:22, 7610.47it/s] 58%|█████▊    | 230866/400000 [00:31<00:22, 7494.53it/s] 58%|█████▊    | 231627/400000 [00:32<00:22, 7528.47it/s] 58%|█████▊    | 232381/400000 [00:32<00:22, 7439.44it/s] 58%|█████▊    | 233138/400000 [00:32<00:22, 7475.64it/s] 58%|█████▊    | 233888/400000 [00:32<00:22, 7481.71it/s] 59%|█████▊    | 234650/400000 [00:32<00:21, 7520.72it/s] 59%|█████▉    | 235415/400000 [00:32<00:21, 7558.75it/s] 59%|█████▉    | 236172/400000 [00:32<00:21, 7533.66it/s] 59%|█████▉    | 236926/400000 [00:32<00:21, 7525.58it/s] 59%|█████▉    | 237679/400000 [00:32<00:21, 7502.36it/s] 60%|█████▉    | 238430/400000 [00:32<00:21, 7468.92it/s] 60%|█████▉    | 239177/400000 [00:33<00:21, 7434.62it/s] 60%|█████▉    | 239926/400000 [00:33<00:21, 7448.30it/s] 60%|██████    | 240671/400000 [00:33<00:21, 7433.34it/s] 60%|██████    | 241415/400000 [00:33<00:21, 7397.97it/s] 61%|██████    | 242156/400000 [00:33<00:21, 7398.10it/s] 61%|██████    | 242896/400000 [00:33<00:21, 7370.97it/s] 61%|██████    | 243646/400000 [00:33<00:21, 7404.15it/s] 61%|██████    | 244387/400000 [00:33<00:21, 7390.71it/s] 61%|██████▏   | 245127/400000 [00:33<00:20, 7380.83it/s] 61%|██████▏   | 245866/400000 [00:33<00:21, 7067.34it/s] 62%|██████▏   | 246605/400000 [00:34<00:21, 7160.73it/s] 62%|██████▏   | 247363/400000 [00:34<00:20, 7281.23it/s] 62%|██████▏   | 248116/400000 [00:34<00:20, 7351.75it/s] 62%|██████▏   | 248853/400000 [00:34<00:20, 7264.70it/s] 62%|██████▏   | 249581/400000 [00:34<00:20, 7203.42it/s] 63%|██████▎   | 250346/400000 [00:34<00:20, 7329.66it/s] 63%|██████▎   | 251103/400000 [00:34<00:20, 7398.87it/s] 63%|██████▎   | 251848/400000 [00:34<00:19, 7413.21it/s] 63%|██████▎   | 252603/400000 [00:34<00:19, 7451.73it/s] 63%|██████▎   | 253349/400000 [00:34<00:19, 7371.70it/s] 64%|██████▎   | 254087/400000 [00:35<00:19, 7372.55it/s] 64%|██████▎   | 254825/400000 [00:35<00:19, 7281.64it/s] 64%|██████▍   | 255578/400000 [00:35<00:19, 7354.26it/s] 64%|██████▍   | 256348/400000 [00:35<00:19, 7453.08it/s] 64%|██████▍   | 257095/400000 [00:35<00:19, 7414.31it/s] 64%|██████▍   | 257837/400000 [00:35<00:19, 7400.56it/s] 65%|██████▍   | 258593/400000 [00:35<00:18, 7445.63it/s] 65%|██████▍   | 259343/400000 [00:35<00:18, 7460.31it/s] 65%|██████▌   | 260090/400000 [00:35<00:18, 7419.14it/s] 65%|██████▌   | 260833/400000 [00:36<00:18, 7343.70it/s] 65%|██████▌   | 261568/400000 [00:36<00:18, 7329.89it/s] 66%|██████▌   | 262307/400000 [00:36<00:18, 7345.45it/s] 66%|██████▌   | 263042/400000 [00:36<00:18, 7346.41it/s] 66%|██████▌   | 263777/400000 [00:36<00:19, 7131.47it/s] 66%|██████▌   | 264492/400000 [00:36<00:19, 7114.80it/s] 66%|██████▋   | 265205/400000 [00:36<00:19, 6989.23it/s] 66%|██████▋   | 265921/400000 [00:36<00:19, 7038.42it/s] 67%|██████▋   | 266626/400000 [00:36<00:18, 7025.06it/s] 67%|██████▋   | 267355/400000 [00:36<00:18, 7100.50it/s] 67%|██████▋   | 268069/400000 [00:37<00:18, 7111.46it/s] 67%|██████▋   | 268805/400000 [00:37<00:18, 7176.97it/s] 67%|██████▋   | 269554/400000 [00:37<00:17, 7266.46it/s] 68%|██████▊   | 270297/400000 [00:37<00:17, 7311.69it/s] 68%|██████▊   | 271029/400000 [00:37<00:17, 7311.08it/s] 68%|██████▊   | 271761/400000 [00:37<00:17, 7310.16it/s] 68%|██████▊   | 272503/400000 [00:37<00:17, 7342.27it/s] 68%|██████▊   | 273247/400000 [00:37<00:17, 7370.14it/s] 69%|██████▊   | 274009/400000 [00:37<00:16, 7442.27it/s] 69%|██████▊   | 274754/400000 [00:37<00:16, 7409.40it/s] 69%|██████▉   | 275508/400000 [00:38<00:16, 7445.48it/s] 69%|██████▉   | 276263/400000 [00:38<00:16, 7474.14it/s] 69%|██████▉   | 277036/400000 [00:38<00:16, 7548.29it/s] 69%|██████▉   | 277807/400000 [00:38<00:16, 7594.27it/s] 70%|██████▉   | 278567/400000 [00:38<00:16, 7586.25it/s] 70%|██████▉   | 279326/400000 [00:38<00:15, 7543.70it/s] 70%|███████   | 280081/400000 [00:38<00:16, 7477.82it/s] 70%|███████   | 280834/400000 [00:38<00:15, 7492.03it/s] 70%|███████   | 281593/400000 [00:38<00:15, 7519.83it/s] 71%|███████   | 282346/400000 [00:38<00:15, 7471.84it/s] 71%|███████   | 283094/400000 [00:39<00:15, 7347.48it/s] 71%|███████   | 283830/400000 [00:39<00:15, 7330.86it/s] 71%|███████   | 284569/400000 [00:39<00:15, 7346.82it/s] 71%|███████▏  | 285318/400000 [00:39<00:15, 7387.08it/s] 72%|███████▏  | 286057/400000 [00:39<00:15, 7372.15it/s] 72%|███████▏  | 286802/400000 [00:39<00:15, 7394.10it/s] 72%|███████▏  | 287542/400000 [00:39<00:15, 7366.44it/s] 72%|███████▏  | 288279/400000 [00:39<00:15, 7357.49it/s] 72%|███████▏  | 289015/400000 [00:39<00:15, 7336.38it/s] 72%|███████▏  | 289749/400000 [00:39<00:15, 7236.57it/s] 73%|███████▎  | 290474/400000 [00:40<00:15, 7210.04it/s] 73%|███████▎  | 291196/400000 [00:40<00:15, 7184.04it/s] 73%|███████▎  | 291915/400000 [00:40<00:15, 7157.07it/s] 73%|███████▎  | 292649/400000 [00:40<00:14, 7210.19it/s] 73%|███████▎  | 293371/400000 [00:40<00:14, 7187.39it/s] 74%|███████▎  | 294115/400000 [00:40<00:14, 7260.16it/s] 74%|███████▎  | 294848/400000 [00:40<00:14, 7279.44it/s] 74%|███████▍  | 295577/400000 [00:40<00:14, 7268.59it/s] 74%|███████▍  | 296350/400000 [00:40<00:14, 7398.77it/s] 74%|███████▍  | 297098/400000 [00:40<00:13, 7422.34it/s] 74%|███████▍  | 297851/400000 [00:41<00:13, 7453.53it/s] 75%|███████▍  | 298597/400000 [00:41<00:13, 7281.64it/s] 75%|███████▍  | 299352/400000 [00:41<00:13, 7358.08it/s] 75%|███████▌  | 300089/400000 [00:41<00:13, 7326.14it/s] 75%|███████▌  | 300861/400000 [00:41<00:13, 7438.03it/s] 75%|███████▌  | 301632/400000 [00:41<00:13, 7517.39it/s] 76%|███████▌  | 302385/400000 [00:41<00:13, 7487.73it/s] 76%|███████▌  | 303140/400000 [00:41<00:12, 7506.06it/s] 76%|███████▌  | 303894/400000 [00:41<00:12, 7514.17it/s] 76%|███████▌  | 304646/400000 [00:41<00:12, 7491.75it/s] 76%|███████▋  | 305402/400000 [00:42<00:12, 7511.92it/s] 77%|███████▋  | 306154/400000 [00:42<00:12, 7469.75it/s] 77%|███████▋  | 306945/400000 [00:42<00:12, 7596.30it/s] 77%|███████▋  | 307706/400000 [00:42<00:12, 7549.46it/s] 77%|███████▋  | 308462/400000 [00:42<00:12, 7509.47it/s] 77%|███████▋  | 309218/400000 [00:42<00:12, 7523.42it/s] 77%|███████▋  | 309971/400000 [00:42<00:12, 7441.76it/s] 78%|███████▊  | 310716/400000 [00:42<00:12, 7436.69it/s] 78%|███████▊  | 311472/400000 [00:42<00:11, 7471.01it/s] 78%|███████▊  | 312239/400000 [00:42<00:11, 7528.35it/s] 78%|███████▊  | 313023/400000 [00:43<00:11, 7617.21it/s] 78%|███████▊  | 313786/400000 [00:43<00:11, 7570.09it/s] 79%|███████▊  | 314554/400000 [00:43<00:11, 7601.24it/s] 79%|███████▉  | 315315/400000 [00:43<00:11, 7586.76it/s] 79%|███████▉  | 316074/400000 [00:43<00:11, 7470.06it/s] 79%|███████▉  | 316822/400000 [00:43<00:11, 7438.00it/s] 79%|███████▉  | 317567/400000 [00:43<00:11, 7394.74it/s] 80%|███████▉  | 318329/400000 [00:43<00:10, 7457.41it/s] 80%|███████▉  | 319093/400000 [00:43<00:10, 7510.91it/s] 80%|███████▉  | 319850/400000 [00:43<00:10, 7526.31it/s] 80%|████████  | 320614/400000 [00:44<00:10, 7557.11it/s] 80%|████████  | 321370/400000 [00:44<00:10, 7497.78it/s] 81%|████████  | 322121/400000 [00:44<00:10, 7465.87it/s] 81%|████████  | 322868/400000 [00:44<00:10, 7440.63it/s] 81%|████████  | 323626/400000 [00:44<00:10, 7480.55it/s] 81%|████████  | 324398/400000 [00:44<00:10, 7549.03it/s] 81%|████████▏ | 325154/400000 [00:44<00:09, 7504.79it/s] 81%|████████▏ | 325905/400000 [00:44<00:09, 7492.59it/s] 82%|████████▏ | 326655/400000 [00:44<00:09, 7472.27it/s] 82%|████████▏ | 327403/400000 [00:45<00:09, 7411.63it/s] 82%|████████▏ | 328145/400000 [00:45<00:09, 7211.40it/s] 82%|████████▏ | 328868/400000 [00:45<00:09, 7163.09it/s] 82%|████████▏ | 329595/400000 [00:45<00:09, 7194.67it/s] 83%|████████▎ | 330326/400000 [00:45<00:09, 7227.19it/s] 83%|████████▎ | 331064/400000 [00:45<00:09, 7270.36it/s] 83%|████████▎ | 331792/400000 [00:45<00:09, 7177.24it/s] 83%|████████▎ | 332537/400000 [00:45<00:09, 7253.34it/s] 83%|████████▎ | 333294/400000 [00:45<00:09, 7344.71it/s] 84%|████████▎ | 334052/400000 [00:45<00:08, 7412.25it/s] 84%|████████▎ | 334831/400000 [00:46<00:08, 7520.44it/s] 84%|████████▍ | 335622/400000 [00:46<00:08, 7632.25it/s] 84%|████████▍ | 336387/400000 [00:46<00:08, 7624.51it/s] 84%|████████▍ | 337151/400000 [00:46<00:08, 7602.63it/s] 84%|████████▍ | 337936/400000 [00:46<00:08, 7674.78it/s] 85%|████████▍ | 338704/400000 [00:46<00:08, 7652.61it/s] 85%|████████▍ | 339470/400000 [00:46<00:08, 7468.29it/s] 85%|████████▌ | 340219/400000 [00:46<00:08, 7439.08it/s] 85%|████████▌ | 340983/400000 [00:46<00:07, 7497.85it/s] 85%|████████▌ | 341777/400000 [00:46<00:07, 7624.37it/s] 86%|████████▌ | 342568/400000 [00:47<00:07, 7706.77it/s] 86%|████████▌ | 343340/400000 [00:47<00:07, 7702.55it/s] 86%|████████▌ | 344126/400000 [00:47<00:07, 7748.28it/s] 86%|████████▌ | 344922/400000 [00:47<00:07, 7807.43it/s] 86%|████████▋ | 345704/400000 [00:47<00:06, 7784.88it/s] 87%|████████▋ | 346484/400000 [00:47<00:06, 7788.90it/s] 87%|████████▋ | 347264/400000 [00:47<00:06, 7751.64it/s] 87%|████████▋ | 348040/400000 [00:47<00:06, 7684.93it/s] 87%|████████▋ | 348809/400000 [00:47<00:06, 7647.14it/s] 87%|████████▋ | 349600/400000 [00:47<00:06, 7722.30it/s] 88%|████████▊ | 350377/400000 [00:48<00:06, 7734.70it/s] 88%|████████▊ | 351151/400000 [00:48<00:06, 7622.19it/s] 88%|████████▊ | 351914/400000 [00:48<00:06, 7560.21it/s] 88%|████████▊ | 352705/400000 [00:48<00:06, 7659.44it/s] 88%|████████▊ | 353472/400000 [00:48<00:06, 7546.78it/s] 89%|████████▊ | 354228/400000 [00:48<00:06, 7459.81it/s] 89%|████████▊ | 354982/400000 [00:48<00:06, 7480.79it/s] 89%|████████▉ | 355731/400000 [00:48<00:05, 7419.09it/s] 89%|████████▉ | 356474/400000 [00:48<00:05, 7329.75it/s] 89%|████████▉ | 357229/400000 [00:48<00:05, 7393.92it/s] 89%|████████▉ | 357969/400000 [00:49<00:05, 7392.33it/s] 90%|████████▉ | 358709/400000 [00:49<00:05, 7393.96it/s] 90%|████████▉ | 359449/400000 [00:49<00:05, 7378.32it/s] 90%|█████████ | 360228/400000 [00:49<00:05, 7496.37it/s] 90%|█████████ | 360984/400000 [00:49<00:05, 7512.32it/s] 90%|█████████ | 361736/400000 [00:49<00:05, 7496.71it/s] 91%|█████████ | 362502/400000 [00:49<00:04, 7543.70it/s] 91%|█████████ | 363260/400000 [00:49<00:04, 7553.81it/s] 91%|█████████ | 364017/400000 [00:49<00:04, 7556.61it/s] 91%|█████████ | 364803/400000 [00:49<00:04, 7644.31it/s] 91%|█████████▏| 365576/400000 [00:50<00:04, 7666.69it/s] 92%|█████████▏| 366343/400000 [00:50<00:04, 7624.83it/s] 92%|█████████▏| 367106/400000 [00:50<00:04, 7590.88it/s] 92%|█████████▏| 367869/400000 [00:50<00:04, 7601.58it/s] 92%|█████████▏| 368649/400000 [00:50<00:04, 7657.70it/s] 92%|█████████▏| 369415/400000 [00:50<00:03, 7653.39it/s] 93%|█████████▎| 370185/400000 [00:50<00:03, 7664.91it/s] 93%|█████████▎| 370952/400000 [00:50<00:03, 7574.97it/s] 93%|█████████▎| 371710/400000 [00:50<00:03, 7512.08it/s] 93%|█████████▎| 372467/400000 [00:50<00:03, 7527.48it/s] 93%|█████████▎| 373223/400000 [00:51<00:03, 7535.70it/s] 93%|█████████▎| 373992/400000 [00:51<00:03, 7580.74it/s] 94%|█████████▎| 374754/400000 [00:51<00:03, 7590.91it/s] 94%|█████████▍| 375514/400000 [00:51<00:03, 7430.27it/s] 94%|█████████▍| 376263/400000 [00:51<00:03, 7444.35it/s] 94%|█████████▍| 377017/400000 [00:51<00:03, 7472.08it/s] 94%|█████████▍| 377773/400000 [00:51<00:02, 7498.04it/s] 95%|█████████▍| 378524/400000 [00:51<00:02, 7445.00it/s] 95%|█████████▍| 379269/400000 [00:51<00:02, 7415.62it/s] 95%|█████████▌| 380011/400000 [00:51<00:02, 7383.44it/s] 95%|█████████▌| 380772/400000 [00:52<00:02, 7448.55it/s] 95%|█████████▌| 381569/400000 [00:52<00:02, 7594.86it/s] 96%|█████████▌| 382330/400000 [00:52<00:02, 7586.30it/s] 96%|█████████▌| 383090/400000 [00:52<00:02, 7487.52it/s] 96%|█████████▌| 383840/400000 [00:52<00:02, 7458.99it/s] 96%|█████████▌| 384604/400000 [00:52<00:02, 7511.95it/s] 96%|█████████▋| 385357/400000 [00:52<00:01, 7515.52it/s] 97%|█████████▋| 386109/400000 [00:52<00:01, 7398.44it/s] 97%|█████████▋| 386850/400000 [00:52<00:01, 7370.12it/s] 97%|█████████▋| 387640/400000 [00:53<00:01, 7520.15it/s] 97%|█████████▋| 388423/400000 [00:53<00:01, 7608.40it/s] 97%|█████████▋| 389185/400000 [00:53<00:01, 7433.44it/s] 97%|█████████▋| 389930/400000 [00:53<00:01, 7392.76it/s] 98%|█████████▊| 390675/400000 [00:53<00:01, 7408.87it/s] 98%|█████████▊| 391417/400000 [00:53<00:01, 7238.43it/s] 98%|█████████▊| 392143/400000 [00:53<00:01, 7091.12it/s] 98%|█████████▊| 392854/400000 [00:53<00:01, 6902.81it/s] 98%|█████████▊| 393557/400000 [00:53<00:00, 6940.36it/s] 99%|█████████▊| 394262/400000 [00:53<00:00, 6972.84it/s] 99%|█████████▊| 394961/400000 [00:54<00:00, 6425.76it/s] 99%|█████████▉| 395736/400000 [00:54<00:00, 6771.93it/s] 99%|█████████▉| 396510/400000 [00:54<00:00, 7035.20it/s] 99%|█████████▉| 397225/400000 [00:54<00:00, 7047.32it/s] 99%|█████████▉| 397979/400000 [00:54<00:00, 7187.34it/s]100%|█████████▉| 398737/400000 [00:54<00:00, 7298.93it/s]100%|█████████▉| 399506/400000 [00:54<00:00, 7411.73it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7307.72it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff156d74fd0>, <torchtext.data.dataset.TabularDataset object at 0x7ff156d74208>, <torchtext.vocab.Vocab object at 0x7ff156d74128>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.73 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.75 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.38 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.38 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.31 file/s]2020-07-22 18:19:37.661365: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-22 18:19:37.665886: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-22 18:19:37.666166: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555cc8a696e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-22 18:19:37.666184: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▊       | 2842624/9912422 [00:00<00:00, 28282437.53it/s]9920512it [00:00, 31529957.25it/s]                             
0it [00:00, ?it/s]32768it [00:00, 629498.12it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475215.42it/s]1654784it [00:00, 11976620.49it/s]                         
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 27801.37it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
