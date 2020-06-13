
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/653c2d3f72ece79ad968c8d79dcc9c6729246f46', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '653c2d3f72ece79ad968c8d79dcc9c6729246f46', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/653c2d3f72ece79ad968c8d79dcc9c6729246f46

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/653c2d3f72ece79ad968c8d79dcc9c6729246f46

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/653c2d3f72ece79ad968c8d79dcc9c6729246f46

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff592d8a6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff592d8a6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff5fe3520d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff5fe3520d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff6180a3ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff6180a3ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff5abbcc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff5abbcc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff5abbcc950> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144244.10it/s] 78%|███████▊  | 7741440/9912422 [00:00<00:10, 205898.20it/s]9920512it [00:00, 41707823.38it/s]                           
0it [00:00, ?it/s]32768it [00:00, 498250.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 479131.79it/s]1654784it [00:00, 12054226.91it/s]                         
0it [00:00, ?it/s]8192it [00:00, 173840.45it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff5a910fd30>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff5a91099e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff5abbcc598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff5abbcc598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff5abbcc598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:45,  1.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:45,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:45,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:44,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:43,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:43,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:42,  1.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:12,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:11,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:11,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:10,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:10,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:09,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:09,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:08,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:08,  2.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:48,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:47,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:47,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:47,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:46,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:46,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:46,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:45,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:45,  3.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:31,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:31,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:31,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:31,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:30,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:30,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:30,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:30,  4.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  5.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:21,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:21,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:21,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:20,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:20,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:20,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:20,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:20,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:20,  5.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:13,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:13,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:13,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:13,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:13,  8.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:09, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:08, 11.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 15.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 20.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 26.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 33.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 40.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 48.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 48.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 55.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 55.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 62.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.55s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.55s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.55s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.61 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.55s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.61 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.63 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ url]
0 examples [00:00, ? examples/s]2020-06-13 22:37:48.489068: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 22:37:48.540310: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-13 22:37:48.540485: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5580d0941700 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 22:37:48.540507: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.43 examples/s]85 examples [00:00,  2.04 examples/s]173 examples [00:00,  2.91 examples/s]260 examples [00:01,  4.15 examples/s]354 examples [00:01,  5.92 examples/s]454 examples [00:01,  8.44 examples/s]554 examples [00:01, 12.01 examples/s]643 examples [00:01, 17.06 examples/s]737 examples [00:01, 24.18 examples/s]836 examples [00:01, 34.18 examples/s]938 examples [00:01, 48.14 examples/s]1039 examples [00:01, 67.38 examples/s]1140 examples [00:01, 93.56 examples/s]1238 examples [00:02, 128.26 examples/s]1335 examples [00:02, 172.72 examples/s]1430 examples [00:02, 226.23 examples/s]1521 examples [00:02, 289.69 examples/s]1612 examples [00:02, 363.88 examples/s]1703 examples [00:02, 443.54 examples/s]1793 examples [00:02, 517.66 examples/s]1885 examples [00:02, 594.77 examples/s]1983 examples [00:02, 674.01 examples/s]2075 examples [00:02, 730.48 examples/s]2174 examples [00:03, 792.30 examples/s]2268 examples [00:03, 825.31 examples/s]2366 examples [00:03, 866.26 examples/s]2465 examples [00:03, 898.12 examples/s]2564 examples [00:03, 922.25 examples/s]2661 examples [00:03, 908.56 examples/s]2760 examples [00:03, 929.68 examples/s]2859 examples [00:03, 946.92 examples/s]2957 examples [00:03, 953.96 examples/s]3054 examples [00:03, 954.35 examples/s]3155 examples [00:04, 968.13 examples/s]3253 examples [00:04, 966.09 examples/s]3351 examples [00:04, 936.89 examples/s]3446 examples [00:04, 937.55 examples/s]3541 examples [00:04, 935.27 examples/s]3638 examples [00:04, 943.11 examples/s]3735 examples [00:04, 949.72 examples/s]3831 examples [00:04, 948.42 examples/s]3926 examples [00:04, 929.94 examples/s]4025 examples [00:04, 947.08 examples/s]4126 examples [00:05, 964.06 examples/s]4229 examples [00:05, 981.69 examples/s]4331 examples [00:05, 991.23 examples/s]4432 examples [00:05, 994.72 examples/s]4532 examples [00:05, 992.31 examples/s]4633 examples [00:05, 995.33 examples/s]4735 examples [00:05, 1001.39 examples/s]4836 examples [00:05, 996.36 examples/s] 4936 examples [00:05, 980.92 examples/s]5035 examples [00:06, 973.79 examples/s]5138 examples [00:06, 988.28 examples/s]5237 examples [00:06, 982.34 examples/s]5336 examples [00:06, 942.42 examples/s]5431 examples [00:06, 922.05 examples/s]5524 examples [00:06, 914.81 examples/s]5616 examples [00:06, 907.87 examples/s]5714 examples [00:06, 926.23 examples/s]5810 examples [00:06, 934.71 examples/s]5904 examples [00:06, 930.93 examples/s]6003 examples [00:07, 946.56 examples/s]6103 examples [00:07, 960.60 examples/s]6201 examples [00:07, 964.26 examples/s]6298 examples [00:07, 962.60 examples/s]6395 examples [00:07, 951.85 examples/s]6491 examples [00:07, 920.07 examples/s]6584 examples [00:07, 915.96 examples/s]6677 examples [00:07, 918.38 examples/s]6769 examples [00:07, 906.53 examples/s]6860 examples [00:07, 899.49 examples/s]6951 examples [00:08, 882.44 examples/s]7041 examples [00:08, 886.20 examples/s]7136 examples [00:08, 902.79 examples/s]7235 examples [00:08, 925.76 examples/s]7335 examples [00:08, 944.82 examples/s]7432 examples [00:08, 950.31 examples/s]7530 examples [00:08, 956.51 examples/s]7626 examples [00:08, 927.15 examples/s]7720 examples [00:08, 896.51 examples/s]7811 examples [00:09, 897.73 examples/s]7902 examples [00:09, 876.91 examples/s]7991 examples [00:09, 879.33 examples/s]8090 examples [00:09, 907.61 examples/s]8189 examples [00:09, 930.30 examples/s]8285 examples [00:09, 938.35 examples/s]8380 examples [00:09, 925.06 examples/s]8473 examples [00:09, 881.66 examples/s]8564 examples [00:09, 889.72 examples/s]8658 examples [00:09, 903.22 examples/s]8752 examples [00:10, 910.95 examples/s]8844 examples [00:10, 899.27 examples/s]8935 examples [00:10, 875.73 examples/s]9023 examples [00:10, 874.04 examples/s]9111 examples [00:10, 868.50 examples/s]9199 examples [00:10, 866.10 examples/s]9286 examples [00:10, 858.01 examples/s]9376 examples [00:10, 868.18 examples/s]9473 examples [00:10, 894.27 examples/s]9570 examples [00:10, 914.44 examples/s]9666 examples [00:11, 927.16 examples/s]9759 examples [00:11, 921.51 examples/s]9852 examples [00:11, 918.70 examples/s]9944 examples [00:11, 901.40 examples/s]10035 examples [00:11, 837.13 examples/s]10135 examples [00:11, 878.35 examples/s]10227 examples [00:11, 888.99 examples/s]10321 examples [00:11, 903.13 examples/s]10414 examples [00:11, 908.50 examples/s]10506 examples [00:12, 898.01 examples/s]10604 examples [00:12, 919.63 examples/s]10697 examples [00:12, 919.29 examples/s]10797 examples [00:12, 940.42 examples/s]10895 examples [00:12, 949.71 examples/s]10991 examples [00:12, 940.13 examples/s]11090 examples [00:12, 954.33 examples/s]11186 examples [00:12, 923.93 examples/s]11285 examples [00:12, 942.14 examples/s]11385 examples [00:12, 956.72 examples/s]11483 examples [00:13, 961.52 examples/s]11580 examples [00:13, 960.24 examples/s]11677 examples [00:13, 960.12 examples/s]11777 examples [00:13, 969.51 examples/s]11876 examples [00:13, 973.19 examples/s]11975 examples [00:13, 976.94 examples/s]12073 examples [00:13, 973.35 examples/s]12171 examples [00:13, 946.82 examples/s]12266 examples [00:13, 920.21 examples/s]12359 examples [00:13, 916.29 examples/s]12453 examples [00:14, 922.87 examples/s]12547 examples [00:14, 926.98 examples/s]12643 examples [00:14, 936.48 examples/s]12744 examples [00:14, 956.78 examples/s]12843 examples [00:14, 966.40 examples/s]12945 examples [00:14, 980.20 examples/s]13044 examples [00:14, 970.04 examples/s]13143 examples [00:14, 974.17 examples/s]13244 examples [00:14, 981.97 examples/s]13346 examples [00:14, 991.15 examples/s]13448 examples [00:15, 999.27 examples/s]13548 examples [00:15, 996.90 examples/s]13648 examples [00:15, 972.27 examples/s]13746 examples [00:15, 968.20 examples/s]13843 examples [00:15, 956.99 examples/s]13939 examples [00:15, 942.55 examples/s]14034 examples [00:15, 926.67 examples/s]14127 examples [00:15, 916.33 examples/s]14228 examples [00:15, 940.26 examples/s]14330 examples [00:16, 961.06 examples/s]14431 examples [00:16, 974.78 examples/s]14530 examples [00:16, 976.87 examples/s]14628 examples [00:16, 936.89 examples/s]14724 examples [00:16, 942.07 examples/s]14822 examples [00:16, 950.23 examples/s]14918 examples [00:16, 943.95 examples/s]15017 examples [00:16, 956.67 examples/s]15118 examples [00:16, 969.59 examples/s]15221 examples [00:16, 984.78 examples/s]15320 examples [00:17, 979.46 examples/s]15419 examples [00:17, 979.41 examples/s]15518 examples [00:17, 968.06 examples/s]15616 examples [00:17, 969.07 examples/s]15715 examples [00:17, 972.73 examples/s]15815 examples [00:17, 979.69 examples/s]15916 examples [00:17, 985.90 examples/s]16015 examples [00:17, 983.61 examples/s]16115 examples [00:17, 986.93 examples/s]16214 examples [00:17, 972.49 examples/s]16313 examples [00:18, 977.09 examples/s]16414 examples [00:18, 985.83 examples/s]16513 examples [00:18, 983.23 examples/s]16613 examples [00:18, 985.78 examples/s]16713 examples [00:18, 987.62 examples/s]16812 examples [00:18, 984.11 examples/s]16911 examples [00:18, 945.84 examples/s]17006 examples [00:18, 937.61 examples/s]17101 examples [00:18, 891.25 examples/s]17196 examples [00:19, 905.97 examples/s]17288 examples [00:19, 903.80 examples/s]17381 examples [00:19, 908.84 examples/s]17473 examples [00:19, 906.28 examples/s]17567 examples [00:19, 915.38 examples/s]17661 examples [00:19, 920.03 examples/s]17754 examples [00:19, 911.87 examples/s]17851 examples [00:19, 926.97 examples/s]17950 examples [00:19, 943.79 examples/s]18050 examples [00:19, 959.56 examples/s]18151 examples [00:20, 972.54 examples/s]18251 examples [00:20, 979.87 examples/s]18353 examples [00:20, 990.16 examples/s]18454 examples [00:20, 993.61 examples/s]18555 examples [00:20, 995.63 examples/s]18655 examples [00:20, 981.20 examples/s]18754 examples [00:20, 976.99 examples/s]18853 examples [00:20, 979.30 examples/s]18951 examples [00:20, 973.11 examples/s]19050 examples [00:20, 976.10 examples/s]19148 examples [00:21, 975.27 examples/s]19246 examples [00:21, 960.68 examples/s]19343 examples [00:21, 906.99 examples/s]19442 examples [00:21, 930.22 examples/s]19542 examples [00:21, 947.94 examples/s]19644 examples [00:21, 967.74 examples/s]19745 examples [00:21, 979.08 examples/s]19846 examples [00:21, 987.46 examples/s]19946 examples [00:21, 988.99 examples/s]20046 examples [00:21, 917.10 examples/s]20146 examples [00:22, 938.31 examples/s]20248 examples [00:22, 959.11 examples/s]20345 examples [00:22, 951.67 examples/s]20444 examples [00:22, 962.20 examples/s]20547 examples [00:22, 979.28 examples/s]20648 examples [00:22, 985.73 examples/s]20750 examples [00:22, 993.46 examples/s]20850 examples [00:22, 989.76 examples/s]20950 examples [00:22, 975.27 examples/s]21050 examples [00:23, 982.48 examples/s]21149 examples [00:23, 975.68 examples/s]21250 examples [00:23, 985.37 examples/s]21350 examples [00:23, 988.95 examples/s]21449 examples [00:23, 982.86 examples/s]21548 examples [00:23, 973.60 examples/s]21649 examples [00:23, 982.85 examples/s]21750 examples [00:23, 988.41 examples/s]21849 examples [00:23, 978.04 examples/s]21947 examples [00:23, 968.59 examples/s]22044 examples [00:24, 957.32 examples/s]22140 examples [00:24, 947.74 examples/s]22237 examples [00:24, 953.79 examples/s]22336 examples [00:24, 963.54 examples/s]22437 examples [00:24, 975.06 examples/s]22535 examples [00:24, 974.05 examples/s]22634 examples [00:24, 977.77 examples/s]22733 examples [00:24, 980.11 examples/s]22832 examples [00:24, 973.93 examples/s]22930 examples [00:24, 917.44 examples/s]23023 examples [00:25, 902.13 examples/s]23119 examples [00:25, 918.45 examples/s]23215 examples [00:25, 929.90 examples/s]23309 examples [00:25, 931.42 examples/s]23408 examples [00:25, 948.01 examples/s]23504 examples [00:25, 951.51 examples/s]23600 examples [00:25, 933.21 examples/s]23694 examples [00:25, 915.67 examples/s]23786 examples [00:25, 906.46 examples/s]23877 examples [00:25, 906.20 examples/s]23968 examples [00:26, 904.91 examples/s]24059 examples [00:26, 900.72 examples/s]24156 examples [00:26, 918.31 examples/s]24248 examples [00:26, 898.61 examples/s]24344 examples [00:26, 915.54 examples/s]24440 examples [00:26, 927.05 examples/s]24536 examples [00:26, 934.83 examples/s]24632 examples [00:26, 941.10 examples/s]24727 examples [00:26, 938.91 examples/s]24821 examples [00:27, 932.33 examples/s]24915 examples [00:27, 911.23 examples/s]25008 examples [00:27, 914.23 examples/s]25102 examples [00:27, 920.68 examples/s]25197 examples [00:27, 928.85 examples/s]25293 examples [00:27, 935.09 examples/s]25392 examples [00:27, 950.80 examples/s]25488 examples [00:27, 951.34 examples/s]25584 examples [00:27, 934.51 examples/s]25678 examples [00:27, 925.45 examples/s]25771 examples [00:28, 925.28 examples/s]25864 examples [00:28, 900.62 examples/s]25959 examples [00:28, 911.90 examples/s]26054 examples [00:28, 921.17 examples/s]26147 examples [00:28, 922.94 examples/s]26246 examples [00:28, 941.57 examples/s]26346 examples [00:28, 956.60 examples/s]26442 examples [00:28, 954.46 examples/s]26538 examples [00:28, 927.08 examples/s]26631 examples [00:28, 918.08 examples/s]26724 examples [00:29, 920.28 examples/s]26824 examples [00:29, 941.18 examples/s]26922 examples [00:29, 951.42 examples/s]27018 examples [00:29, 931.25 examples/s]27112 examples [00:29, 931.77 examples/s]27209 examples [00:29, 942.35 examples/s]27309 examples [00:29, 956.72 examples/s]27408 examples [00:29, 964.53 examples/s]27505 examples [00:29, 964.87 examples/s]27602 examples [00:29, 956.45 examples/s]27698 examples [00:30, 953.11 examples/s]27796 examples [00:30, 958.59 examples/s]27895 examples [00:30, 965.91 examples/s]27993 examples [00:30, 969.70 examples/s]28091 examples [00:30, 965.27 examples/s]28189 examples [00:30, 967.67 examples/s]28286 examples [00:30, 961.24 examples/s]28383 examples [00:30, 935.28 examples/s]28477 examples [00:30, 935.73 examples/s]28575 examples [00:30, 946.82 examples/s]28670 examples [00:31, 921.34 examples/s]28763 examples [00:31, 849.09 examples/s]28851 examples [00:31, 856.86 examples/s]28938 examples [00:31, 855.91 examples/s]29027 examples [00:31, 865.30 examples/s]29118 examples [00:31, 877.81 examples/s]29209 examples [00:31, 886.50 examples/s]29301 examples [00:31, 895.05 examples/s]29394 examples [00:31, 904.84 examples/s]29491 examples [00:32, 923.28 examples/s]29590 examples [00:32, 938.53 examples/s]29689 examples [00:32, 953.23 examples/s]29785 examples [00:32, 950.14 examples/s]29881 examples [00:32, 936.80 examples/s]29978 examples [00:32, 945.51 examples/s]30073 examples [00:32, 881.00 examples/s]30163 examples [00:32, 848.38 examples/s]30252 examples [00:32, 858.79 examples/s]30346 examples [00:32, 880.30 examples/s]30439 examples [00:33, 892.23 examples/s]30533 examples [00:33, 904.85 examples/s]30624 examples [00:33, 905.91 examples/s]30717 examples [00:33, 912.88 examples/s]30809 examples [00:33, 909.19 examples/s]30903 examples [00:33, 918.08 examples/s]31000 examples [00:33, 930.50 examples/s]31094 examples [00:33, 913.96 examples/s]31186 examples [00:33, 900.42 examples/s]31277 examples [00:34, 890.26 examples/s]31373 examples [00:34, 909.72 examples/s]31469 examples [00:34, 922.05 examples/s]31563 examples [00:34, 925.18 examples/s]31658 examples [00:34, 931.39 examples/s]31753 examples [00:34, 936.86 examples/s]31847 examples [00:34, 922.42 examples/s]31940 examples [00:34, 916.36 examples/s]32032 examples [00:34, 915.87 examples/s]32127 examples [00:34, 922.69 examples/s]32220 examples [00:35, 896.27 examples/s]32311 examples [00:35, 899.74 examples/s]32402 examples [00:35, 901.01 examples/s]32493 examples [00:35, 894.31 examples/s]32583 examples [00:35, 878.80 examples/s]32680 examples [00:35, 901.94 examples/s]32778 examples [00:35, 923.24 examples/s]32881 examples [00:35, 950.62 examples/s]32979 examples [00:35, 956.67 examples/s]33076 examples [00:35, 959.94 examples/s]33173 examples [00:36, 898.56 examples/s]33264 examples [00:36, 872.61 examples/s]33353 examples [00:36, 876.32 examples/s]33444 examples [00:36, 885.64 examples/s]33535 examples [00:36, 891.58 examples/s]33633 examples [00:36, 915.67 examples/s]33733 examples [00:36, 937.35 examples/s]33830 examples [00:36, 945.77 examples/s]33929 examples [00:36, 957.56 examples/s]34029 examples [00:36, 967.63 examples/s]34126 examples [00:37, 946.47 examples/s]34223 examples [00:37, 950.87 examples/s]34319 examples [00:37, 935.38 examples/s]34413 examples [00:37, 915.56 examples/s]34511 examples [00:37, 933.58 examples/s]34609 examples [00:37, 945.95 examples/s]34704 examples [00:37, 940.18 examples/s]34799 examples [00:37, 908.00 examples/s]34891 examples [00:37, 906.96 examples/s]34982 examples [00:38, 903.68 examples/s]35079 examples [00:38, 921.63 examples/s]35177 examples [00:38, 937.11 examples/s]35274 examples [00:38, 945.26 examples/s]35369 examples [00:38, 933.47 examples/s]35463 examples [00:38, 932.71 examples/s]35562 examples [00:38, 947.37 examples/s]35657 examples [00:38, 947.16 examples/s]35752 examples [00:38, 917.98 examples/s]35845 examples [00:38, 909.28 examples/s]35937 examples [00:39, 899.14 examples/s]36028 examples [00:39, 885.73 examples/s]36125 examples [00:39, 909.24 examples/s]36219 examples [00:39, 915.71 examples/s]36311 examples [00:39, 912.06 examples/s]36410 examples [00:39, 932.32 examples/s]36510 examples [00:39, 950.47 examples/s]36612 examples [00:39, 969.59 examples/s]36713 examples [00:39, 979.05 examples/s]36814 examples [00:39, 987.48 examples/s]36913 examples [00:40, 973.91 examples/s]37011 examples [00:40, 971.04 examples/s]37110 examples [00:40, 974.67 examples/s]37209 examples [00:40, 979.12 examples/s]37307 examples [00:40, 952.26 examples/s]37407 examples [00:40, 964.61 examples/s]37508 examples [00:40, 976.93 examples/s]37606 examples [00:40, 969.25 examples/s]37707 examples [00:40, 978.58 examples/s]37809 examples [00:40, 988.03 examples/s]37908 examples [00:41, 988.45 examples/s]38007 examples [00:41, 985.83 examples/s]38106 examples [00:41, 948.48 examples/s]38202 examples [00:41, 942.03 examples/s]38299 examples [00:41, 947.82 examples/s]38394 examples [00:41, 948.13 examples/s]38493 examples [00:41, 958.90 examples/s]38591 examples [00:41, 964.42 examples/s]38691 examples [00:41, 974.61 examples/s]38791 examples [00:42, 981.59 examples/s]38890 examples [00:42, 957.46 examples/s]38986 examples [00:42, 940.54 examples/s]39081 examples [00:42, 922.93 examples/s]39181 examples [00:42, 943.89 examples/s]39281 examples [00:42, 957.51 examples/s]39378 examples [00:42, 959.65 examples/s]39475 examples [00:42, 959.00 examples/s]39576 examples [00:42, 972.39 examples/s]39674 examples [00:42, 967.22 examples/s]39771 examples [00:43, 956.51 examples/s]39867 examples [00:43, 951.73 examples/s]39965 examples [00:43, 956.77 examples/s]40061 examples [00:43, 902.27 examples/s]40154 examples [00:43, 909.23 examples/s]40252 examples [00:43, 928.21 examples/s]40350 examples [00:43, 940.73 examples/s]40445 examples [00:43, 942.41 examples/s]40545 examples [00:43, 956.39 examples/s]40641 examples [00:43, 957.30 examples/s]40737 examples [00:44, 947.54 examples/s]40835 examples [00:44, 955.37 examples/s]40931 examples [00:44, 945.31 examples/s]41033 examples [00:44, 965.76 examples/s]41134 examples [00:44, 977.70 examples/s]41233 examples [00:44, 979.48 examples/s]41332 examples [00:44, 932.96 examples/s]41426 examples [00:44, 907.17 examples/s]41518 examples [00:44, 894.53 examples/s]41611 examples [00:45, 900.94 examples/s]41711 examples [00:45, 926.74 examples/s]41809 examples [00:45, 939.52 examples/s]41904 examples [00:45, 900.49 examples/s]42002 examples [00:45, 920.87 examples/s]42099 examples [00:45, 934.35 examples/s]42193 examples [00:45, 935.88 examples/s]42287 examples [00:45, 933.98 examples/s]42381 examples [00:45, 922.59 examples/s]42476 examples [00:45, 928.75 examples/s]42570 examples [00:46, 915.78 examples/s]42662 examples [00:46, 911.71 examples/s]42755 examples [00:46, 916.85 examples/s]42848 examples [00:46, 918.18 examples/s]42940 examples [00:46, 893.58 examples/s]43030 examples [00:46, 848.53 examples/s]43119 examples [00:46, 859.43 examples/s]43209 examples [00:46, 869.98 examples/s]43297 examples [00:46, 870.49 examples/s]43393 examples [00:46, 894.83 examples/s]43488 examples [00:47, 910.15 examples/s]43580 examples [00:47, 912.96 examples/s]43680 examples [00:47, 937.01 examples/s]43776 examples [00:47, 942.66 examples/s]43871 examples [00:47, 929.08 examples/s]43965 examples [00:47, 917.29 examples/s]44057 examples [00:47, 898.50 examples/s]44148 examples [00:47, 894.86 examples/s]44244 examples [00:47, 913.08 examples/s]44341 examples [00:47, 928.63 examples/s]44436 examples [00:48, 934.08 examples/s]44530 examples [00:48, 921.76 examples/s]44623 examples [00:48, 920.29 examples/s]44716 examples [00:48, 918.89 examples/s]44808 examples [00:48, 912.66 examples/s]44900 examples [00:48, 904.24 examples/s]44997 examples [00:48, 922.15 examples/s]45090 examples [00:48, 920.03 examples/s]45190 examples [00:48, 940.59 examples/s]45292 examples [00:49, 960.75 examples/s]45393 examples [00:49, 972.59 examples/s]45492 examples [00:49, 975.01 examples/s]45590 examples [00:49, 954.53 examples/s]45686 examples [00:49, 941.98 examples/s]45781 examples [00:49, 921.58 examples/s]45874 examples [00:49, 907.96 examples/s]45973 examples [00:49, 927.11 examples/s]46071 examples [00:49, 942.11 examples/s]46172 examples [00:49, 959.35 examples/s]46269 examples [00:50, 957.01 examples/s]46365 examples [00:50, 936.48 examples/s]46459 examples [00:50, 912.07 examples/s]46560 examples [00:50, 937.16 examples/s]46662 examples [00:50, 959.89 examples/s]46762 examples [00:50, 969.70 examples/s]46860 examples [00:50, 966.56 examples/s]46961 examples [00:50, 979.00 examples/s]47060 examples [00:50, 973.90 examples/s]47158 examples [00:50, 971.97 examples/s]47259 examples [00:51, 982.67 examples/s]47358 examples [00:51, 974.47 examples/s]47456 examples [00:51, 974.42 examples/s]47554 examples [00:51, 967.70 examples/s]47651 examples [00:51, 960.92 examples/s]47748 examples [00:51, 958.06 examples/s]47844 examples [00:51, 917.37 examples/s]47937 examples [00:51, 850.47 examples/s]48025 examples [00:51, 857.37 examples/s]48124 examples [00:52, 891.84 examples/s]48225 examples [00:52, 922.48 examples/s]48324 examples [00:52, 941.61 examples/s]48419 examples [00:52, 938.71 examples/s]48517 examples [00:52, 950.18 examples/s]48614 examples [00:52, 953.89 examples/s]48710 examples [00:52, 925.49 examples/s]48808 examples [00:52, 938.44 examples/s]48903 examples [00:52, 939.49 examples/s]48998 examples [00:52, 934.22 examples/s]49092 examples [00:53, 932.67 examples/s]49186 examples [00:53, 927.01 examples/s]49284 examples [00:53, 940.58 examples/s]49382 examples [00:53, 950.22 examples/s]49478 examples [00:53, 950.89 examples/s]49577 examples [00:53, 960.76 examples/s]49676 examples [00:53, 967.13 examples/s]49775 examples [00:53, 973.00 examples/s]49873 examples [00:53, 970.11 examples/s]49973 examples [00:53, 977.47 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6599/50000 [00:00<00:00, 65987.55 examples/s] 36%|███▌      | 17925/50000 [00:00<00:00, 75431.65 examples/s] 57%|█████▋    | 28355/50000 [00:00<00:00, 82260.38 examples/s] 79%|███████▉  | 39464/50000 [00:00<00:00, 88975.40 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 735.77 examples/s]164 examples [00:00, 774.81 examples/s]258 examples [00:00, 815.83 examples/s]350 examples [00:00, 843.96 examples/s]444 examples [00:00, 870.39 examples/s]528 examples [00:00, 858.74 examples/s]622 examples [00:00, 880.42 examples/s]716 examples [00:00, 897.15 examples/s]803 examples [00:00, 887.72 examples/s]904 examples [00:01, 919.18 examples/s]1003 examples [00:01, 937.56 examples/s]1096 examples [00:01, 916.82 examples/s]1194 examples [00:01, 933.15 examples/s]1287 examples [00:01, 920.38 examples/s]1379 examples [00:01, 712.61 examples/s]1473 examples [00:01, 768.22 examples/s]1569 examples [00:01, 815.49 examples/s]1663 examples [00:01, 848.81 examples/s]1759 examples [00:02, 878.62 examples/s]1853 examples [00:02, 895.25 examples/s]1946 examples [00:02, 900.73 examples/s]2038 examples [00:02, 880.38 examples/s]2128 examples [00:02, 882.59 examples/s]2219 examples [00:02, 890.12 examples/s]2318 examples [00:02, 915.81 examples/s]2415 examples [00:02, 930.40 examples/s]2512 examples [00:02, 940.73 examples/s]2608 examples [00:02, 945.15 examples/s]2703 examples [00:03, 941.53 examples/s]2798 examples [00:03, 941.56 examples/s]2893 examples [00:03, 928.88 examples/s]2987 examples [00:03, 923.06 examples/s]3080 examples [00:03, 899.85 examples/s]3177 examples [00:03, 918.53 examples/s]3271 examples [00:03, 922.49 examples/s]3364 examples [00:03, 912.88 examples/s]3459 examples [00:03, 921.88 examples/s]3556 examples [00:03, 933.13 examples/s]3656 examples [00:04, 950.75 examples/s]3757 examples [00:04, 967.24 examples/s]3858 examples [00:04, 977.70 examples/s]3956 examples [00:04, 948.35 examples/s]4052 examples [00:04, 942.21 examples/s]4152 examples [00:04, 957.97 examples/s]4253 examples [00:04, 970.73 examples/s]4351 examples [00:04, 966.73 examples/s]4448 examples [00:04, 961.32 examples/s]4545 examples [00:04, 938.77 examples/s]4640 examples [00:05, 927.09 examples/s]4738 examples [00:05, 941.24 examples/s]4837 examples [00:05, 954.18 examples/s]4935 examples [00:05, 960.32 examples/s]5032 examples [00:05, 936.15 examples/s]5126 examples [00:05, 933.78 examples/s]5220 examples [00:05, 921.66 examples/s]5314 examples [00:05, 924.65 examples/s]5409 examples [00:05, 929.98 examples/s]5503 examples [00:06, 932.57 examples/s]5601 examples [00:06, 946.05 examples/s]5696 examples [00:06, 943.95 examples/s]5791 examples [00:06, 939.93 examples/s]5891 examples [00:06, 955.15 examples/s]5989 examples [00:06, 960.06 examples/s]6090 examples [00:06, 973.96 examples/s]6191 examples [00:06, 980.81 examples/s]6290 examples [00:06, 981.59 examples/s]6390 examples [00:06, 984.74 examples/s]6489 examples [00:07, 972.73 examples/s]6587 examples [00:07, 969.26 examples/s]6684 examples [00:07, 968.75 examples/s]6781 examples [00:07, 946.26 examples/s]6876 examples [00:07, 921.27 examples/s]6969 examples [00:07, 904.72 examples/s]7060 examples [00:07, 894.20 examples/s]7150 examples [00:07, 888.67 examples/s]7240 examples [00:07, 879.59 examples/s]7329 examples [00:07, 876.58 examples/s]7418 examples [00:08, 879.68 examples/s]7509 examples [00:08, 886.63 examples/s]7598 examples [00:08, 879.08 examples/s]7686 examples [00:08, 866.12 examples/s]7783 examples [00:08, 892.84 examples/s]7882 examples [00:08, 918.72 examples/s]7978 examples [00:08, 926.07 examples/s]8071 examples [00:08, 924.04 examples/s]8173 examples [00:08, 948.47 examples/s]8275 examples [00:08, 966.58 examples/s]8374 examples [00:09, 971.31 examples/s]8476 examples [00:09, 983.69 examples/s]8579 examples [00:09, 994.83 examples/s]8680 examples [00:09, 998.53 examples/s]8780 examples [00:09, 994.19 examples/s]8880 examples [00:09, 985.57 examples/s]8983 examples [00:09, 996.40 examples/s]9083 examples [00:09, 988.83 examples/s]9182 examples [00:09, 965.10 examples/s]9279 examples [00:10, 942.93 examples/s]9374 examples [00:10, 930.50 examples/s]9469 examples [00:10, 935.81 examples/s]9563 examples [00:10, 917.36 examples/s]9656 examples [00:10, 919.77 examples/s]9749 examples [00:10, 903.91 examples/s]9840 examples [00:10, 904.13 examples/s]9931 examples [00:10, 896.40 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 96%|█████████▌| 9576/10000 [00:00<00:00, 95754.84 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP2NG7A/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP2NG7A/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff5abbcc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff5abbcc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff5abbcc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff5f75ba400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff535046a20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff5abbcc950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff5abbcc950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff5abbcc950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff592c7c358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff592c7c080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7ff52f0c32f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7ff52f0c32f0> 

  function with postional parmater data_info <function split_train_valid at 0x7ff52f0c32f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7ff52f0c3400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7ff52f0c3400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7ff52f0c3400> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<100:41:24, 2.38kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<70:42:41, 3.39kB/s] .vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<49:32:37, 4.83kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<34:40:01, 6.90kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<24:11:35, 9.86kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:04<16:49:26, 14.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:04<11:42:01, 20.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:04<8:08:15, 28.7kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:04<5:40:45, 41.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.4M/862M [00:04<3:57:13, 58.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:04<2:45:15, 83.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.4M/862M [00:04<1:55:07, 119kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:04<1:20:15, 170kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:04<55:52, 243kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 50.5M/862M [00:05<39:06, 346kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:05<28:19, 477kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:07<21:38, 621kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:07<16:57, 792kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:07<12:16, 1.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:09<11:06, 1.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:09<09:18, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.9M/862M [00:09<06:53, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:10<05:39, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:11<8:36:35, 25.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:11<6:01:22, 36.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:13<4:14:44, 52.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:13<3:00:59, 73.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:13<2:07:13, 104kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:14<1:30:51, 145kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:15<1:04:57, 203kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:15<45:41, 288kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:16<34:55, 375kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:17<25:47, 508kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:17<18:21, 712kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:18<15:52, 821kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:19<13:44, 948kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:19<10:16, 1.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:20<09:16, 1.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:21<07:47, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:21<05:46, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:22<07:04, 1.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:22<06:15, 2.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:23<04:42, 2.73MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:24<06:17, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:24<07:00, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:25<05:27, 2.34MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.2M/862M [00:25<03:57, 3.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:26<16:28, 775kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:26<12:50, 993kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:27<09:18, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<09:26, 1.34MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<07:53, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:28<05:49, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<07:03, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<07:29, 1.68MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<05:52, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<06:08, 2.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<05:36, 2.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<04:12, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<05:51, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<06:39, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<05:11, 2.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:34<03:47, 3.27MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<08:31, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<07:15, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<05:22, 2.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:38<06:39, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<07:10, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<05:38, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:40<05:55, 2.07MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<05:24, 2.27MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<04:05, 3.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<05:43, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:42<06:28, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<05:04, 2.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<03:40, 3.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<14:09, 858kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<11:07, 1.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<08:02, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<08:27, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<08:21, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:21, 1.90MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<04:35, 2.62MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:48<10:04, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:48<08:17, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<06:05, 1.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<05:00, 2.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<8:09:12, 24.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<5:42:54, 34.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<3:59:22, 49.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<2:53:35, 68.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<2:04:31, 95.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<1:27:41, 135kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<1:01:27, 193kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:54<46:20, 255kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:54<33:44, 350kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<23:51, 494kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<19:10, 612kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<14:38, 801kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<10:31, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<10:05, 1.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<09:26, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<07:11, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<06:54, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<06:00, 1.93MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<04:27, 2.59MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:01<05:49, 1.98MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<06:32, 1.76MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<05:11, 2.22MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<03:45, 3.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<39:01, 294kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<28:30, 402kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<20:12, 565kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<16:43, 681kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<12:52, 884kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<09:17, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<09:05, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:07<08:45, 1.29MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<06:38, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<04:45, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<12:49, 877kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<10:10, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<07:23, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<07:44, 1.45MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<06:36, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<04:51, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<05:57, 1.87MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<06:32, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<05:10, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:14<03:44, 2.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<37:40, 293kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<27:29, 402kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<19:29, 565kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<16:07, 681kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:17<13:36, 807kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<10:00, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:17<07:07, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<10:41, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<08:36, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<06:17, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:21<06:53, 1.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<05:57, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<04:26, 2.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<05:35, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:23<05:03, 2.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<03:48, 2.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:25<05:07, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:25<04:43, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<03:32, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:27<04:55, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:27<05:41, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<04:29, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<03:16, 3.23MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<07:33, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<06:24, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<04:42, 2.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<03:59, 2.63MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<7:05:42, 24.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<4:58:19, 35.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:31<3:28:07, 50.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<2:31:33, 68.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<1:48:22, 96.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:33<1:16:20, 137kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<53:19, 195kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<53:37, 193kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<38:37, 268kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<27:15, 379kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<21:19, 483kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<17:06, 602kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<12:30, 823kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:37<08:49, 1.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<15:30, 660kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:39<11:55, 857kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<08:35, 1.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<08:19, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<07:58, 1.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<06:02, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<04:20, 2.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<08:31, 1.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<07:01, 1.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<05:53, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<06:14, 1.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<04:48, 2.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<03:29, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<07:12, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<06:04, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<04:30, 2.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<05:25, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<04:48, 2.06MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<03:36, 2.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<04:47, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<05:26, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<04:19, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<03:08, 3.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<31:31, 309kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<23:05, 422kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<16:22, 594kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<13:38, 710kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<11:35, 835kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<08:36, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<06:07, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<34:48, 276kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<25:20, 379kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<17:54, 535kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:58<14:41, 650kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:58<12:18, 775kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<09:01, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<06:26, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<08:26, 1.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:00<06:53, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<05:03, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<05:41, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<04:57, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<03:40, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:04<04:43, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:04<05:16, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<04:08, 2.25MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<02:59, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:06<16:25, 565kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:06<12:28, 743kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<08:57, 1.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:08<08:21, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:08<07:48, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<05:56, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:15, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<1:08:10, 134kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<48:39, 188kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<34:10, 266kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<25:54, 350kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<20:02, 452kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<14:24, 628kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<10:09, 887kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<09:49, 917kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<5:57:48, 25.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<4:10:08, 35.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<2:55:53, 50.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<2:05:03, 71.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<1:27:54, 101kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<1:01:19, 145kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<52:23, 169kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<37:33, 236kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<26:24, 335kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<20:29, 430kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<16:11, 543kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:20<11:46, 746kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<08:18, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<1:08:12, 128kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<48:36, 179kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<34:07, 255kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<25:47, 336kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<19:52, 436kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<14:16, 606kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<10:05, 854kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<10:17, 835kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<08:05, 1.06MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:26<05:51, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<06:04, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<04:40, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<03:21, 2.51MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<1:02:43, 135kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<44:45, 189kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<31:27, 268kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<23:51, 352kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<17:34, 477kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<12:27, 671kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<10:36, 785kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<09:10, 906kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<06:46, 1.22MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:52, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<06:20, 1.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<05:18, 1.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<03:53, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<04:57, 1.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<03:53, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:38<02:48, 2.88MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:39<59:57, 135kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<42:47, 189kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:39<30:02, 269kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<22:47, 353kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<17:34, 457kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<12:42, 632kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<08:56, 892kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:43<1:03:15, 126kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<45:04, 177kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<31:37, 251kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<23:51, 331kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:45<18:22, 430kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<13:11, 598kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<09:19, 844kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<09:43, 806kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<07:37, 1.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<05:29, 1.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<05:37, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<04:43, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<03:27, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<04:13, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<03:45, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<02:49, 2.72MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<03:43, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<03:22, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<02:33, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<02:14, 3.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<5:12:36, 24.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<3:38:54, 34.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<2:32:36, 49.3kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<1:49:26, 68.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<1:18:05, 96.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<54:53, 136kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<38:21, 194kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<30:10, 246kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<21:52, 340kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:59<15:25, 480kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<12:28, 591kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<10:16, 716kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<07:31, 978kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<05:20, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<07:01, 1.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<05:41, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<04:09, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<04:33, 1.58MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:57, 1.83MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:05<02:56, 2.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<03:41, 1.94MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<04:05, 1.75MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<03:12, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<02:19, 3.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<23:09, 306kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<16:55, 419kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:09<11:57, 590kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<09:58, 705kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<08:28, 829kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<06:17, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<04:27, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<19:09, 363kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<14:07, 492kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<10:00, 692kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<08:34, 803kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<07:27, 922kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<05:33, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<03:57, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<17:52, 382kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<13:12, 516kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<09:23, 723kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<08:05, 834kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<07:06, 950kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<05:19, 1.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<03:46, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<18:44, 357kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<13:47, 484kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:20<09:46, 681kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<08:19, 794kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<07:10, 922kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<05:20, 1.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:48, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<49:28, 132kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:24<35:16, 185kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<24:45, 263kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<18:43, 346kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<13:47, 469kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<09:45, 661kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<08:15, 775kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<07:09, 896kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<05:16, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<03:44, 1.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<09:32, 664kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<07:20, 862kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<05:15, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<05:06, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<04:54, 1.28MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<03:45, 1.67MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<02:54, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<3:58:14, 26.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<2:46:23, 37.2kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<1:55:38, 53.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<2:00:43, 50.9kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<1:25:53, 71.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<1:00:20, 102kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<42:00, 145kB/s]  .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<34:07, 178kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<24:29, 248kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<17:13, 351kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<13:22, 449kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<10:37, 565kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<07:41, 779kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<05:26, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<06:15, 949kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<04:59, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<03:36, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<03:53, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<03:57, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<03:04, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:12, 2.63MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<15:21, 377kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<11:20, 511kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<08:01, 718kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<06:55, 827kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<06:03, 945kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<04:28, 1.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<03:12, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<04:31, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<03:45, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<02:44, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<03:11, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<02:48, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:05, 2.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<02:44, 2.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<02:30, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<01:53, 2.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<02:34, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<02:57, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<02:18, 2.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<01:40, 3.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:57<03:53, 1.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:57<03:16, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:57<02:24, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<02:54, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<03:10, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<02:29, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<01:48, 2.92MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<17:55, 293kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<13:04, 401kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:01<09:14, 565kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<07:37, 680kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<06:23, 811kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<04:41, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<03:21, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<04:02, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:05<03:21, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<02:28, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<02:52, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<03:05, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:07<02:25, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<01:44, 2.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<13:03, 381kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<09:38, 515kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<06:50, 722kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<05:53, 833kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<05:09, 950kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<03:51, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:12<02:56, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:13<2:53:46, 27.9kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:13<2:01:17, 39.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<1:24:57, 56.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<1:00:29, 79.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<42:30, 112kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<29:31, 160kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<26:13, 180kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<18:50, 250kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<13:14, 353kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<10:13, 454kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<08:07, 570kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<05:53, 786kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:19<04:08, 1.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<05:22, 850kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<04:14, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<03:03, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<03:09, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<03:07, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:23<02:23, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<01:42, 2.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<05:03, 877kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<03:58, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<02:53, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<03:01, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<03:00, 1.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<02:17, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<01:39, 2.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<03:01, 1.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<02:33, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<01:53, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<02:18, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<02:30, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<01:56, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<01:24, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:50, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:25, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:33<01:47, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:11, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:23, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<01:53, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<01:21, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<13:02, 308kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<09:32, 421kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<06:44, 593kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<05:35, 708kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<04:44, 833kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<03:31, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<02:29, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<10:59, 353kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<08:05, 479kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<05:43, 673kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<04:51, 787kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<04:12, 908kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<03:08, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:12, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<10:56, 343kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<08:01, 466kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:44<05:40, 654kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<04:47, 767kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<03:43, 985kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<02:41, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<02:41, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<02:39, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<02:02, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<01:27, 2.44MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<09:54, 357kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<07:14, 489kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:50<05:09, 682kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<03:36, 964kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<11:38, 298kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<08:29, 408kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:52<05:59, 575kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<04:22, 781kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<2:20:58, 24.2kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<1:38:35, 34.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:54<1:08:11, 49.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<49:19, 67.8kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<35:14, 94.9kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<24:43, 135kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<17:09, 192kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<13:36, 241kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<09:50, 332kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:58<06:55, 468kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<05:31, 580kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<04:11, 763kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<02:59, 1.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<02:47, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<02:37, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<01:58, 1.59MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<01:24, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<02:30, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<02:04, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<01:30, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:44, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<01:51, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<01:26, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:02, 2.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<09:33, 307kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<06:58, 419kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<04:54, 591kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<04:03, 707kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<03:26, 832kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<02:31, 1.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<01:47, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<02:23, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:58, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:12<01:25, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:36, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:24, 1.93MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:02, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:15<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<01:28, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<01:08, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<00:49, 3.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:41, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:27, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:03, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<01:19, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<01:27, 1.72MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:08, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<00:48, 3.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<06:13, 394kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<04:36, 532kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<03:15, 745kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<02:47, 854kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<02:26, 977kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<01:48, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:16, 1.83MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<02:34, 899kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<02:02, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<01:27, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:31, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:31, 1.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:09, 1.93MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<00:49, 2.66MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:46, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:28, 1.48MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<01:03, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<01:13, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<01:17, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<00:59, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:42, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<02:13, 921kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<01:45, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:16, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:59, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<1:21:46, 24.2kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<57:03, 34.5kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:35<39:05, 49.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<28:12, 67.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<20:07, 94.9kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<14:05, 135kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<09:37, 192kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<11:22, 162kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<08:05, 227kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:39<05:38, 321kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<04:17, 414kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<03:16, 541kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<02:22, 741kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<01:41, 1.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<01:32, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<01:13, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:53, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:37, 2.60MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<03:50, 426kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<03:02, 538kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<02:11, 741kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:30, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<03:18, 475kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<02:28, 633kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:47<01:44, 884kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:32, 978kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:13, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:49<00:53, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<00:55, 1.54MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:50<00:57, 1.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<00:43, 1.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:51<00:30, 2.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:52<03:30, 389kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<02:35, 525kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:53<01:48, 736kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<01:31, 846kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<01:12, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:51, 1.47MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:56<00:52, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:43, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:31, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:37, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:33, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:24, 2.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:31, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:35, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:27, 2.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:19, 3.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<19:55, 51.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<13:57, 72.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<09:33, 103kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<06:39, 143kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<04:50, 196kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<03:23, 276kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<02:16, 392kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<02:15, 392kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<01:39, 529kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<01:08, 742kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:57, 851kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:44, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:31, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:31, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:31, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:23, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:16, 2.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:29, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:24, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:12<00:17, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:20, 1.82MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:21, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:16, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:11, 2.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:16<01:45, 308kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<01:16, 420kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:16<00:51, 591kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:35, 803kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<20:06, 23.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<13:46, 34.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<08:53, 48.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<06:04, 67.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<04:19, 94.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<02:57, 134kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<01:50, 191kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<01:32, 221kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<01:05, 305kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:42, 432kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:30, 540kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:24, 662kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:16, 908kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:10, 1.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<00:11, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:09, 1.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:05, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:05, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:04, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:02, 2.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:02, 1.93MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:30<00:02, 1.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:01, 2.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:30<00:00, 3.06MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<70:03:25,  1.59it/s]  0%|          | 559/400000 [00:00<48:58:38,  2.27it/s]  0%|          | 1251/400000 [00:00<34:13:46,  3.24it/s]  0%|          | 1975/400000 [00:00<23:55:18,  4.62it/s]  1%|          | 2695/400000 [00:01<16:43:10,  6.60it/s]  1%|          | 3330/400000 [00:01<11:41:24,  9.43it/s]  1%|          | 3997/400000 [00:01<8:10:27, 13.46it/s]   1%|          | 4713/400000 [00:01<5:42:58, 19.21it/s]  1%|▏         | 5428/400000 [00:01<3:59:55, 27.41it/s]  2%|▏         | 6152/400000 [00:01<2:47:54, 39.09it/s]  2%|▏         | 6852/400000 [00:01<1:57:36, 55.71it/s]  2%|▏         | 7535/400000 [00:01<1:22:30, 79.28it/s]  2%|▏         | 8198/400000 [00:01<57:57, 112.66it/s]   2%|▏         | 8855/400000 [00:01<40:48, 159.74it/s]  2%|▏         | 9506/400000 [00:02<28:49, 225.75it/s]  3%|▎         | 10211/400000 [00:02<20:25, 318.13it/s]  3%|▎         | 10913/400000 [00:02<14:32, 445.81it/s]  3%|▎         | 11651/400000 [00:02<10:25, 620.79it/s]  3%|▎         | 12345/400000 [00:02<07:35, 850.45it/s]  3%|▎         | 13017/400000 [00:02<05:36, 1149.98it/s]  3%|▎         | 13681/400000 [00:02<04:12, 1527.88it/s]  4%|▎         | 14382/400000 [00:02<03:13, 1996.09it/s]  4%|▍         | 15098/400000 [00:02<02:31, 2547.07it/s]  4%|▍         | 15820/400000 [00:02<02:01, 3160.70it/s]  4%|▍         | 16517/400000 [00:03<01:41, 3780.29it/s]  4%|▍         | 17235/400000 [00:03<01:26, 4405.58it/s]  4%|▍         | 17954/400000 [00:03<01:16, 4984.23it/s]  5%|▍         | 18678/400000 [00:03<01:09, 5497.28it/s]  5%|▍         | 19398/400000 [00:03<01:04, 5915.63it/s]  5%|▌         | 20118/400000 [00:03<01:00, 6248.06it/s]  5%|▌         | 20834/400000 [00:03<01:00, 6268.86it/s]  5%|▌         | 21537/400000 [00:03<00:58, 6478.00it/s]  6%|▌         | 22238/400000 [00:03<00:57, 6627.14it/s]  6%|▌         | 22935/400000 [00:03<00:56, 6644.60it/s]  6%|▌         | 23655/400000 [00:04<00:55, 6800.81it/s]  6%|▌         | 24370/400000 [00:04<00:54, 6901.77it/s]  6%|▋         | 25073/400000 [00:04<00:54, 6848.92it/s]  6%|▋         | 25773/400000 [00:04<00:54, 6887.86it/s]  7%|▋         | 26468/400000 [00:04<00:55, 6732.47it/s]  7%|▋         | 27147/400000 [00:04<00:56, 6567.19it/s]  7%|▋         | 27809/400000 [00:04<00:56, 6546.26it/s]  7%|▋         | 28507/400000 [00:04<00:55, 6668.29it/s]  7%|▋         | 29188/400000 [00:04<00:55, 6709.60it/s]  7%|▋         | 29861/400000 [00:05<00:55, 6715.47it/s]  8%|▊         | 30553/400000 [00:05<00:54, 6772.87it/s]  8%|▊         | 31232/400000 [00:05<00:55, 6654.70it/s]  8%|▊         | 31899/400000 [00:05<00:55, 6616.26it/s]  8%|▊         | 32562/400000 [00:05<00:56, 6547.87it/s]  8%|▊         | 33218/400000 [00:05<00:56, 6546.13it/s]  8%|▊         | 33874/400000 [00:05<00:56, 6506.21it/s]  9%|▊         | 34583/400000 [00:05<00:54, 6669.80it/s]  9%|▉         | 35308/400000 [00:05<00:53, 6832.11it/s]  9%|▉         | 36017/400000 [00:05<00:52, 6906.33it/s]  9%|▉         | 36710/400000 [00:06<00:54, 6717.31it/s]  9%|▉         | 37384/400000 [00:06<00:54, 6597.68it/s] 10%|▉         | 38046/400000 [00:06<00:54, 6602.48it/s] 10%|▉         | 38764/400000 [00:06<00:53, 6764.21it/s] 10%|▉         | 39483/400000 [00:06<00:52, 6884.25it/s] 10%|█         | 40203/400000 [00:06<00:51, 6974.31it/s] 10%|█         | 40903/400000 [00:06<00:53, 6761.62it/s] 10%|█         | 41582/400000 [00:06<00:54, 6576.21it/s] 11%|█         | 42243/400000 [00:06<00:55, 6486.36it/s] 11%|█         | 42894/400000 [00:06<00:55, 6394.23it/s] 11%|█         | 43605/400000 [00:07<00:54, 6592.10it/s] 11%|█         | 44284/400000 [00:07<00:53, 6647.36it/s] 11%|█         | 44982/400000 [00:07<00:52, 6742.63it/s] 11%|█▏        | 45708/400000 [00:07<00:51, 6889.06it/s] 12%|█▏        | 46445/400000 [00:07<00:50, 7024.86it/s] 12%|█▏        | 47150/400000 [00:07<00:50, 7024.87it/s] 12%|█▏        | 47854/400000 [00:07<00:50, 7020.38it/s] 12%|█▏        | 48558/400000 [00:07<00:50, 7022.22it/s] 12%|█▏        | 49266/400000 [00:07<00:49, 7036.46it/s] 12%|█▏        | 49981/400000 [00:07<00:49, 7068.37it/s] 13%|█▎        | 50697/400000 [00:08<00:49, 7094.17it/s] 13%|█▎        | 51407/400000 [00:08<00:49, 6991.89it/s] 13%|█▎        | 52107/400000 [00:08<00:50, 6941.74it/s] 13%|█▎        | 52831/400000 [00:08<00:49, 7027.77it/s] 13%|█▎        | 53546/400000 [00:08<00:49, 7061.46it/s] 14%|█▎        | 54253/400000 [00:08<00:49, 6998.12it/s] 14%|█▎        | 54991/400000 [00:08<00:48, 7106.61it/s] 14%|█▍        | 55703/400000 [00:08<00:49, 6987.33it/s] 14%|█▍        | 56413/400000 [00:08<00:48, 7018.89it/s] 14%|█▍        | 57144/400000 [00:08<00:48, 7101.78it/s] 14%|█▍        | 57855/400000 [00:09<00:48, 7013.90it/s] 15%|█▍        | 58558/400000 [00:09<00:50, 6818.65it/s] 15%|█▍        | 59242/400000 [00:09<00:51, 6662.39it/s] 15%|█▍        | 59911/400000 [00:09<00:53, 6369.21it/s] 15%|█▌        | 60634/400000 [00:09<00:51, 6603.81it/s] 15%|█▌        | 61353/400000 [00:09<00:50, 6767.77it/s] 16%|█▌        | 62085/400000 [00:09<00:48, 6924.48it/s] 16%|█▌        | 62782/400000 [00:09<00:48, 6891.92it/s] 16%|█▌        | 63475/400000 [00:09<00:49, 6823.80it/s] 16%|█▌        | 64160/400000 [00:10<00:51, 6485.54it/s] 16%|█▌        | 64860/400000 [00:10<00:50, 6630.04it/s] 16%|█▋        | 65558/400000 [00:10<00:49, 6730.90it/s] 17%|█▋        | 66283/400000 [00:10<00:48, 6877.13it/s] 17%|█▋        | 66999/400000 [00:10<00:47, 6958.31it/s] 17%|█▋        | 67723/400000 [00:10<00:47, 7039.63it/s] 17%|█▋        | 68429/400000 [00:10<00:47, 6991.13it/s] 17%|█▋        | 69130/400000 [00:10<00:48, 6769.22it/s] 17%|█▋        | 69837/400000 [00:10<00:48, 6856.13it/s] 18%|█▊        | 70562/400000 [00:10<00:47, 6967.39it/s] 18%|█▊        | 71261/400000 [00:11<00:47, 6925.93it/s] 18%|█▊        | 71975/400000 [00:11<00:46, 6987.08it/s] 18%|█▊        | 72675/400000 [00:11<00:47, 6958.08it/s] 18%|█▊        | 73385/400000 [00:11<00:46, 6998.78it/s] 19%|█▊        | 74102/400000 [00:11<00:46, 7048.11it/s] 19%|█▊        | 74808/400000 [00:11<00:46, 7018.07it/s] 19%|█▉        | 75511/400000 [00:11<00:46, 6951.80it/s] 19%|█▉        | 76207/400000 [00:11<00:46, 6952.96it/s] 19%|█▉        | 76903/400000 [00:11<00:46, 6945.90it/s] 19%|█▉        | 77600/400000 [00:11<00:46, 6950.56it/s] 20%|█▉        | 78311/400000 [00:12<00:45, 6993.30it/s] 20%|█▉        | 79032/400000 [00:12<00:45, 7055.80it/s] 20%|█▉        | 79749/400000 [00:12<00:45, 7088.37it/s] 20%|██        | 80459/400000 [00:12<00:45, 7060.93it/s] 20%|██        | 81175/400000 [00:12<00:44, 7089.24it/s] 20%|██        | 81885/400000 [00:12<00:45, 7062.91it/s] 21%|██        | 82592/400000 [00:12<00:45, 7034.11it/s] 21%|██        | 83296/400000 [00:12<00:45, 7016.38it/s] 21%|██        | 84017/400000 [00:12<00:44, 7072.25it/s] 21%|██        | 84742/400000 [00:12<00:44, 7110.01it/s] 21%|██▏       | 85469/400000 [00:13<00:43, 7154.60it/s] 22%|██▏       | 86185/400000 [00:13<00:44, 7044.30it/s] 22%|██▏       | 86913/400000 [00:13<00:44, 7111.23it/s] 22%|██▏       | 87625/400000 [00:13<00:44, 7095.21it/s] 22%|██▏       | 88342/400000 [00:13<00:43, 7114.98it/s] 22%|██▏       | 89070/400000 [00:13<00:43, 7161.24it/s] 22%|██▏       | 89787/400000 [00:13<00:43, 7149.73it/s] 23%|██▎       | 90503/400000 [00:13<00:43, 7038.15it/s] 23%|██▎       | 91208/400000 [00:13<00:44, 6980.91it/s] 23%|██▎       | 91923/400000 [00:14<00:43, 7028.11it/s] 23%|██▎       | 92656/400000 [00:14<00:43, 7115.89it/s] 23%|██▎       | 93369/400000 [00:14<00:43, 7112.27it/s] 24%|██▎       | 94083/400000 [00:14<00:42, 7119.98it/s] 24%|██▎       | 94796/400000 [00:14<00:43, 7056.10it/s] 24%|██▍       | 95515/400000 [00:14<00:42, 7094.99it/s] 24%|██▍       | 96227/400000 [00:14<00:42, 7091.60it/s] 24%|██▍       | 96949/400000 [00:14<00:42, 7128.28it/s] 24%|██▍       | 97663/400000 [00:14<00:42, 7101.03it/s] 25%|██▍       | 98374/400000 [00:14<00:42, 7028.91it/s] 25%|██▍       | 99078/400000 [00:15<00:42, 7025.21it/s] 25%|██▍       | 99794/400000 [00:15<00:42, 7063.23it/s] 25%|██▌       | 100522/400000 [00:15<00:42, 7126.54it/s] 25%|██▌       | 101241/400000 [00:15<00:41, 7143.63it/s] 25%|██▌       | 101956/400000 [00:15<00:42, 7089.66it/s] 26%|██▌       | 102684/400000 [00:15<00:41, 7145.70it/s] 26%|██▌       | 103399/400000 [00:15<00:41, 7140.48it/s] 26%|██▌       | 104126/400000 [00:15<00:41, 7177.97it/s] 26%|██▌       | 104844/400000 [00:15<00:41, 7172.11it/s] 26%|██▋       | 105562/400000 [00:15<00:41, 7146.95it/s] 27%|██▋       | 106295/400000 [00:16<00:40, 7198.81it/s] 27%|██▋       | 107023/400000 [00:16<00:40, 7221.12it/s] 27%|██▋       | 107746/400000 [00:16<00:40, 7143.19it/s] 27%|██▋       | 108469/400000 [00:16<00:40, 7167.38it/s] 27%|██▋       | 109186/400000 [00:16<00:41, 6964.84it/s] 27%|██▋       | 109908/400000 [00:16<00:41, 7037.29it/s] 28%|██▊       | 110631/400000 [00:16<00:40, 7093.46it/s] 28%|██▊       | 111342/400000 [00:16<00:41, 6998.10it/s] 28%|██▊       | 112043/400000 [00:16<00:43, 6651.71it/s] 28%|██▊       | 112713/400000 [00:16<00:43, 6564.14it/s] 28%|██▊       | 113435/400000 [00:17<00:42, 6747.30it/s] 29%|██▊       | 114156/400000 [00:17<00:41, 6877.98it/s] 29%|██▊       | 114880/400000 [00:17<00:40, 6981.85it/s] 29%|██▉       | 115587/400000 [00:17<00:40, 7006.26it/s] 29%|██▉       | 116290/400000 [00:17<00:40, 6931.83it/s] 29%|██▉       | 117006/400000 [00:17<00:40, 6997.15it/s] 29%|██▉       | 117720/400000 [00:17<00:40, 7038.68it/s] 30%|██▉       | 118425/400000 [00:17<00:41, 6860.26it/s] 30%|██▉       | 119127/400000 [00:17<00:40, 6905.56it/s] 30%|██▉       | 119819/400000 [00:17<00:40, 6898.04it/s] 30%|███       | 120518/400000 [00:18<00:40, 6922.71it/s] 30%|███       | 121221/400000 [00:18<00:40, 6953.81it/s] 30%|███       | 121938/400000 [00:18<00:39, 7015.35it/s] 31%|███       | 122659/400000 [00:18<00:39, 7069.75it/s] 31%|███       | 123372/400000 [00:18<00:39, 7086.53it/s] 31%|███       | 124103/400000 [00:18<00:38, 7150.71it/s] 31%|███       | 124819/400000 [00:18<00:39, 7025.01it/s] 31%|███▏      | 125532/400000 [00:18<00:38, 7055.38it/s] 32%|███▏      | 126239/400000 [00:18<00:38, 7052.45it/s] 32%|███▏      | 126961/400000 [00:18<00:38, 7100.78it/s] 32%|███▏      | 127673/400000 [00:19<00:38, 7103.92it/s] 32%|███▏      | 128403/400000 [00:19<00:37, 7160.02it/s] 32%|███▏      | 129120/400000 [00:19<00:38, 7052.94it/s] 32%|███▏      | 129838/400000 [00:19<00:38, 7088.96it/s] 33%|███▎      | 130549/400000 [00:19<00:37, 7095.02it/s] 33%|███▎      | 131261/400000 [00:19<00:37, 7101.87it/s] 33%|███▎      | 131972/400000 [00:19<00:38, 6977.71it/s] 33%|███▎      | 132700/400000 [00:19<00:37, 7063.10it/s] 33%|███▎      | 133408/400000 [00:19<00:37, 7028.31it/s] 34%|███▎      | 134118/400000 [00:19<00:37, 7046.98it/s] 34%|███▎      | 134824/400000 [00:20<00:37, 7034.81it/s] 34%|███▍      | 135545/400000 [00:20<00:37, 7084.06it/s] 34%|███▍      | 136274/400000 [00:20<00:36, 7142.17it/s] 34%|███▍      | 136989/400000 [00:20<00:36, 7124.06it/s] 34%|███▍      | 137702/400000 [00:20<00:37, 7021.97it/s] 35%|███▍      | 138429/400000 [00:20<00:36, 7092.15it/s] 35%|███▍      | 139141/400000 [00:20<00:36, 7100.04it/s] 35%|███▍      | 139859/400000 [00:20<00:36, 7123.82it/s] 35%|███▌      | 140572/400000 [00:20<00:36, 7113.88it/s] 35%|███▌      | 141284/400000 [00:21<00:37, 6983.59it/s] 35%|███▌      | 141984/400000 [00:21<00:36, 6982.06it/s] 36%|███▌      | 142683/400000 [00:21<00:38, 6760.33it/s] 36%|███▌      | 143361/400000 [00:21<00:39, 6448.62it/s] 36%|███▌      | 144011/400000 [00:21<00:40, 6289.61it/s] 36%|███▌      | 144644/400000 [00:21<00:41, 6217.20it/s] 36%|███▋      | 145375/400000 [00:21<00:39, 6508.82it/s] 37%|███▋      | 146095/400000 [00:21<00:37, 6699.91it/s] 37%|███▋      | 146806/400000 [00:21<00:37, 6815.60it/s] 37%|███▋      | 147492/400000 [00:21<00:37, 6777.78it/s] 37%|███▋      | 148210/400000 [00:22<00:36, 6893.60it/s] 37%|███▋      | 148937/400000 [00:22<00:35, 7001.10it/s] 37%|███▋      | 149656/400000 [00:22<00:35, 7055.39it/s] 38%|███▊      | 150365/400000 [00:22<00:35, 7063.70it/s] 38%|███▊      | 151073/400000 [00:22<00:36, 6905.62it/s] 38%|███▊      | 151766/400000 [00:22<00:36, 6855.54it/s] 38%|███▊      | 152480/400000 [00:22<00:35, 6936.45it/s] 38%|███▊      | 153194/400000 [00:22<00:35, 6994.41it/s] 38%|███▊      | 153895/400000 [00:22<00:35, 6984.30it/s] 39%|███▊      | 154595/400000 [00:22<00:36, 6783.43it/s] 39%|███▉      | 155291/400000 [00:23<00:35, 6833.98it/s] 39%|███▉      | 156009/400000 [00:23<00:35, 6929.61it/s] 39%|███▉      | 156731/400000 [00:23<00:34, 7013.80it/s] 39%|███▉      | 157456/400000 [00:23<00:34, 7082.63it/s] 40%|███▉      | 158166/400000 [00:23<00:34, 7017.99it/s] 40%|███▉      | 158869/400000 [00:23<00:34, 7000.76it/s] 40%|███▉      | 159591/400000 [00:23<00:34, 7063.46it/s] 40%|████      | 160316/400000 [00:23<00:33, 7117.67it/s] 40%|████      | 161029/400000 [00:23<00:34, 6970.01it/s] 40%|████      | 161728/400000 [00:23<00:35, 6703.92it/s] 41%|████      | 162402/400000 [00:24<00:35, 6650.38it/s] 41%|████      | 163088/400000 [00:24<00:35, 6709.51it/s] 41%|████      | 163791/400000 [00:24<00:34, 6800.74it/s] 41%|████      | 164504/400000 [00:24<00:34, 6895.01it/s] 41%|████▏     | 165195/400000 [00:24<00:34, 6795.10it/s] 41%|████▏     | 165926/400000 [00:24<00:33, 6939.95it/s] 42%|████▏     | 166653/400000 [00:24<00:33, 7035.71it/s] 42%|████▏     | 167379/400000 [00:24<00:32, 7101.36it/s] 42%|████▏     | 168091/400000 [00:24<00:32, 7100.31it/s] 42%|████▏     | 168802/400000 [00:25<00:32, 7013.39it/s] 42%|████▏     | 169525/400000 [00:25<00:32, 7075.75it/s] 43%|████▎     | 170239/400000 [00:25<00:32, 7092.98it/s] 43%|████▎     | 170966/400000 [00:25<00:32, 7143.75it/s] 43%|████▎     | 171681/400000 [00:25<00:32, 7114.22it/s] 43%|████▎     | 172393/400000 [00:25<00:32, 6914.23it/s] 43%|████▎     | 173115/400000 [00:25<00:32, 7001.41it/s] 43%|████▎     | 173823/400000 [00:25<00:32, 7024.36it/s] 44%|████▎     | 174536/400000 [00:25<00:31, 7054.69it/s] 44%|████▍     | 175258/400000 [00:25<00:31, 7102.94it/s] 44%|████▍     | 175969/400000 [00:26<00:32, 6971.96it/s] 44%|████▍     | 176676/400000 [00:26<00:31, 6998.92it/s] 44%|████▍     | 177393/400000 [00:26<00:31, 7047.16it/s] 45%|████▍     | 178112/400000 [00:26<00:31, 7085.01it/s] 45%|████▍     | 178828/400000 [00:26<00:31, 7104.87it/s] 45%|████▍     | 179539/400000 [00:26<00:31, 7066.97it/s] 45%|████▌     | 180246/400000 [00:26<00:31, 6964.34it/s] 45%|████▌     | 180943/400000 [00:26<00:32, 6660.25it/s] 45%|████▌     | 181670/400000 [00:26<00:31, 6831.01it/s] 46%|████▌     | 182382/400000 [00:26<00:31, 6913.55it/s] 46%|████▌     | 183091/400000 [00:27<00:31, 6965.45it/s] 46%|████▌     | 183790/400000 [00:27<00:31, 6969.33it/s] 46%|████▌     | 184511/400000 [00:27<00:30, 7037.33it/s] 46%|████▋     | 185220/400000 [00:27<00:30, 7052.37it/s] 46%|████▋     | 185930/400000 [00:27<00:30, 7064.88it/s] 47%|████▋     | 186639/400000 [00:27<00:30, 7070.36it/s] 47%|████▋     | 187356/400000 [00:27<00:29, 7098.34it/s] 47%|████▋     | 188067/400000 [00:27<00:30, 6863.31it/s] 47%|████▋     | 188756/400000 [00:27<00:31, 6727.34it/s] 47%|████▋     | 189431/400000 [00:27<00:31, 6648.04it/s] 48%|████▊     | 190138/400000 [00:28<00:31, 6767.30it/s] 48%|████▊     | 190843/400000 [00:28<00:30, 6848.92it/s] 48%|████▊     | 191575/400000 [00:28<00:29, 6981.00it/s] 48%|████▊     | 192275/400000 [00:28<00:30, 6839.78it/s] 48%|████▊     | 192961/400000 [00:28<00:31, 6602.60it/s] 48%|████▊     | 193625/400000 [00:28<00:32, 6402.07it/s] 49%|████▊     | 194284/400000 [00:28<00:31, 6455.07it/s] 49%|████▊     | 194999/400000 [00:28<00:30, 6648.81it/s] 49%|████▉     | 195713/400000 [00:28<00:30, 6787.40it/s] 49%|████▉     | 196434/400000 [00:29<00:29, 6907.11it/s] 49%|████▉     | 197128/400000 [00:29<00:29, 6916.55it/s] 49%|████▉     | 197822/400000 [00:29<00:29, 6892.15it/s] 50%|████▉     | 198556/400000 [00:29<00:28, 7019.34it/s] 50%|████▉     | 199267/400000 [00:29<00:28, 7043.39it/s] 50%|████▉     | 199973/400000 [00:29<00:29, 6671.43it/s] 50%|█████     | 200645/400000 [00:29<00:31, 6392.36it/s] 50%|█████     | 201346/400000 [00:29<00:30, 6564.91it/s] 51%|█████     | 202048/400000 [00:29<00:29, 6693.92it/s] 51%|█████     | 202758/400000 [00:29<00:28, 6808.89it/s] 51%|█████     | 203460/400000 [00:30<00:28, 6869.35it/s] 51%|█████     | 204151/400000 [00:30<00:28, 6879.40it/s] 51%|█████     | 204875/400000 [00:30<00:27, 6983.51it/s] 51%|█████▏    | 205600/400000 [00:30<00:27, 7058.76it/s] 52%|█████▏    | 206308/400000 [00:30<00:27, 6980.62it/s] 52%|█████▏    | 207009/400000 [00:30<00:27, 6987.09it/s] 52%|█████▏    | 207712/400000 [00:30<00:27, 6998.03it/s] 52%|█████▏    | 208413/400000 [00:30<00:27, 6900.98it/s] 52%|█████▏    | 209104/400000 [00:30<00:28, 6775.69it/s] 52%|█████▏    | 209783/400000 [00:30<00:28, 6740.80it/s] 53%|█████▎    | 210494/400000 [00:31<00:27, 6845.13it/s] 53%|█████▎    | 211186/400000 [00:31<00:27, 6865.68it/s] 53%|█████▎    | 211875/400000 [00:31<00:27, 6872.23it/s] 53%|█████▎    | 212593/400000 [00:31<00:26, 6952.97it/s] 53%|█████▎    | 213289/400000 [00:31<00:27, 6752.34it/s] 53%|█████▎    | 213980/400000 [00:31<00:27, 6796.05it/s] 54%|█████▎    | 214661/400000 [00:31<00:27, 6745.93it/s] 54%|█████▍    | 215368/400000 [00:31<00:26, 6839.94it/s] 54%|█████▍    | 216053/400000 [00:31<00:27, 6731.34it/s] 54%|█████▍    | 216728/400000 [00:31<00:27, 6671.27it/s] 54%|█████▍    | 217401/400000 [00:32<00:27, 6687.73it/s] 55%|█████▍    | 218091/400000 [00:32<00:26, 6747.39it/s] 55%|█████▍    | 218781/400000 [00:32<00:26, 6784.67it/s] 55%|█████▍    | 219460/400000 [00:32<00:26, 6691.47it/s] 55%|█████▌    | 220130/400000 [00:32<00:27, 6596.38it/s] 55%|█████▌    | 220851/400000 [00:32<00:26, 6768.28it/s] 55%|█████▌    | 221552/400000 [00:32<00:26, 6838.80it/s] 56%|█████▌    | 222281/400000 [00:32<00:25, 6966.89it/s] 56%|█████▌    | 222980/400000 [00:32<00:25, 6937.31it/s] 56%|█████▌    | 223675/400000 [00:32<00:25, 6926.31it/s] 56%|█████▌    | 224371/400000 [00:33<00:25, 6933.72it/s] 56%|█████▋    | 225065/400000 [00:33<00:25, 6914.85it/s] 56%|█████▋    | 225757/400000 [00:33<00:25, 6840.57it/s] 57%|█████▋    | 226479/400000 [00:33<00:24, 6947.99it/s] 57%|█████▋    | 227175/400000 [00:33<00:24, 6950.28it/s] 57%|█████▋    | 227871/400000 [00:33<00:25, 6769.66it/s] 57%|█████▋    | 228583/400000 [00:33<00:24, 6869.65it/s] 57%|█████▋    | 229306/400000 [00:33<00:24, 6972.04it/s] 58%|█████▊    | 230018/400000 [00:33<00:24, 7015.04it/s] 58%|█████▊    | 230741/400000 [00:34<00:23, 7078.05it/s] 58%|█████▊    | 231450/400000 [00:34<00:23, 7037.58it/s] 58%|█████▊    | 232162/400000 [00:34<00:23, 7061.11it/s] 58%|█████▊    | 232892/400000 [00:34<00:23, 7130.18it/s] 58%|█████▊    | 233624/400000 [00:34<00:23, 7185.41it/s] 59%|█████▊    | 234352/400000 [00:34<00:22, 7212.79it/s] 59%|█████▉    | 235074/400000 [00:34<00:22, 7198.79it/s] 59%|█████▉    | 235795/400000 [00:34<00:22, 7175.14it/s] 59%|█████▉    | 236513/400000 [00:34<00:22, 7139.60it/s] 59%|█████▉    | 237240/400000 [00:34<00:22, 7176.58it/s] 59%|█████▉    | 237958/400000 [00:35<00:22, 7168.77it/s] 60%|█████▉    | 238675/400000 [00:35<00:22, 7154.85it/s] 60%|█████▉    | 239391/400000 [00:35<00:22, 7093.42it/s] 60%|██████    | 240111/400000 [00:35<00:22, 7123.05it/s] 60%|██████    | 240824/400000 [00:35<00:22, 7101.07it/s] 60%|██████    | 241543/400000 [00:35<00:22, 7124.96it/s] 61%|██████    | 242265/400000 [00:35<00:22, 7150.43it/s] 61%|██████    | 242981/400000 [00:35<00:22, 6942.49it/s] 61%|██████    | 243698/400000 [00:35<00:22, 7006.93it/s] 61%|██████    | 244418/400000 [00:35<00:22, 7061.02it/s] 61%|██████▏   | 245125/400000 [00:36<00:22, 6983.54it/s] 61%|██████▏   | 245846/400000 [00:36<00:21, 7048.66it/s] 62%|██████▏   | 246555/400000 [00:36<00:21, 7043.45it/s] 62%|██████▏   | 247269/400000 [00:36<00:21, 7070.20it/s] 62%|██████▏   | 247977/400000 [00:36<00:21, 7065.62it/s] 62%|██████▏   | 248684/400000 [00:36<00:21, 6973.60it/s] 62%|██████▏   | 249400/400000 [00:36<00:21, 7028.23it/s] 63%|██████▎   | 250104/400000 [00:36<00:21, 6947.20it/s] 63%|██████▎   | 250831/400000 [00:36<00:21, 7040.37it/s] 63%|██████▎   | 251548/400000 [00:36<00:20, 7076.47it/s] 63%|██████▎   | 252257/400000 [00:37<00:21, 6962.70it/s] 63%|██████▎   | 252975/400000 [00:37<00:20, 7024.03it/s] 63%|██████▎   | 253679/400000 [00:37<00:20, 7003.18it/s] 64%|██████▎   | 254398/400000 [00:37<00:20, 7056.84it/s] 64%|██████▍   | 255105/400000 [00:37<00:21, 6886.08it/s] 64%|██████▍   | 255829/400000 [00:37<00:20, 6901.54it/s] 64%|██████▍   | 256557/400000 [00:37<00:20, 7009.44it/s] 64%|██████▍   | 257259/400000 [00:37<00:20, 6860.91it/s] 64%|██████▍   | 257982/400000 [00:37<00:20, 6966.16it/s] 65%|██████▍   | 258695/400000 [00:37<00:20, 7012.26it/s] 65%|██████▍   | 259398/400000 [00:38<00:20, 6744.36it/s] 65%|██████▌   | 260134/400000 [00:38<00:20, 6917.01it/s] 65%|██████▌   | 260829/400000 [00:38<00:20, 6770.85it/s] 65%|██████▌   | 261509/400000 [00:38<00:20, 6670.08it/s] 66%|██████▌   | 262179/400000 [00:38<00:21, 6534.39it/s] 66%|██████▌   | 262900/400000 [00:38<00:20, 6723.05it/s] 66%|██████▌   | 263616/400000 [00:38<00:19, 6846.19it/s] 66%|██████▌   | 264307/400000 [00:38<00:19, 6862.42it/s] 66%|██████▋   | 265031/400000 [00:38<00:19, 6969.28it/s] 66%|██████▋   | 265755/400000 [00:39<00:19, 7047.41it/s] 67%|██████▋   | 266462/400000 [00:39<00:18, 7051.94it/s] 67%|██████▋   | 267187/400000 [00:39<00:18, 7108.00it/s] 67%|██████▋   | 267899/400000 [00:39<00:18, 7095.98it/s] 67%|██████▋   | 268610/400000 [00:39<00:18, 7098.66it/s] 67%|██████▋   | 269321/400000 [00:39<00:18, 7026.41it/s] 68%|██████▊   | 270044/400000 [00:39<00:18, 7085.07it/s] 68%|██████▊   | 270753/400000 [00:39<00:18, 7003.64it/s] 68%|██████▊   | 271456/400000 [00:39<00:18, 7011.04it/s] 68%|██████▊   | 272170/400000 [00:39<00:18, 7047.66it/s] 68%|██████▊   | 272886/400000 [00:40<00:17, 7079.04it/s] 68%|██████▊   | 273598/400000 [00:40<00:17, 7090.30it/s] 69%|██████▊   | 274313/400000 [00:40<00:17, 7105.28it/s] 69%|██████▉   | 275024/400000 [00:40<00:17, 7059.50it/s] 69%|██████▉   | 275731/400000 [00:40<00:17, 7032.30it/s] 69%|██████▉   | 276435/400000 [00:40<00:17, 7023.12it/s] 69%|██████▉   | 277138/400000 [00:40<00:17, 7009.51it/s] 69%|██████▉   | 277840/400000 [00:40<00:17, 6825.54it/s] 70%|██████▉   | 278524/400000 [00:40<00:18, 6488.66it/s] 70%|██████▉   | 279178/400000 [00:40<00:19, 6299.39it/s] 70%|██████▉   | 279835/400000 [00:41<00:18, 6376.06it/s] 70%|███████   | 280511/400000 [00:41<00:18, 6484.83it/s] 70%|███████   | 281177/400000 [00:41<00:18, 6534.44it/s] 70%|███████   | 281878/400000 [00:41<00:17, 6667.75it/s] 71%|███████   | 282595/400000 [00:41<00:17, 6808.45it/s] 71%|███████   | 283296/400000 [00:41<00:16, 6867.49it/s] 71%|███████   | 284001/400000 [00:41<00:16, 6918.76it/s] 71%|███████   | 284695/400000 [00:41<00:17, 6773.77it/s] 71%|███████▏  | 285374/400000 [00:41<00:17, 6672.21it/s] 72%|███████▏  | 286101/400000 [00:41<00:16, 6840.28it/s] 72%|███████▏  | 286832/400000 [00:42<00:16, 6974.42it/s] 72%|███████▏  | 287556/400000 [00:42<00:15, 7048.73it/s] 72%|███████▏  | 288263/400000 [00:42<00:16, 6923.06it/s] 72%|███████▏  | 288987/400000 [00:42<00:15, 7012.94it/s] 72%|███████▏  | 289720/400000 [00:42<00:15, 7104.04it/s] 73%|███████▎  | 290443/400000 [00:42<00:15, 7135.75it/s] 73%|███████▎  | 291178/400000 [00:42<00:15, 7198.15it/s] 73%|███████▎  | 291899/400000 [00:42<00:15, 7149.07it/s] 73%|███████▎  | 292615/400000 [00:42<00:15, 6989.13it/s] 73%|███████▎  | 293326/400000 [00:42<00:15, 7024.72it/s] 74%|███████▎  | 294055/400000 [00:43<00:14, 7099.63it/s] 74%|███████▎  | 294766/400000 [00:43<00:14, 7071.47it/s] 74%|███████▍  | 295474/400000 [00:43<00:15, 6879.69it/s] 74%|███████▍  | 296164/400000 [00:43<00:15, 6649.14it/s] 74%|███████▍  | 296851/400000 [00:43<00:15, 6713.31it/s] 74%|███████▍  | 297577/400000 [00:43<00:14, 6866.82it/s] 75%|███████▍  | 298267/400000 [00:43<00:14, 6820.81it/s] 75%|███████▍  | 298951/400000 [00:43<00:15, 6660.61it/s] 75%|███████▍  | 299620/400000 [00:43<00:15, 6573.76it/s] 75%|███████▌  | 300280/400000 [00:44<00:15, 6419.13it/s] 75%|███████▌  | 300937/400000 [00:44<00:15, 6461.99it/s] 75%|███████▌  | 301649/400000 [00:44<00:14, 6645.55it/s] 76%|███████▌  | 302343/400000 [00:44<00:14, 6729.97it/s] 76%|███████▌  | 303046/400000 [00:44<00:14, 6816.73it/s] 76%|███████▌  | 303764/400000 [00:44<00:13, 6920.88it/s] 76%|███████▌  | 304491/400000 [00:44<00:13, 7020.43it/s] 76%|███████▋  | 305210/400000 [00:44<00:13, 7068.38it/s] 76%|███████▋  | 305925/400000 [00:44<00:13, 7092.26it/s] 77%|███████▋  | 306635/400000 [00:44<00:13, 7023.75it/s] 77%|███████▋  | 307359/400000 [00:45<00:13, 7086.12it/s] 77%|███████▋  | 308076/400000 [00:45<00:12, 7110.75it/s] 77%|███████▋  | 308788/400000 [00:45<00:12, 7062.37it/s] 77%|███████▋  | 309508/400000 [00:45<00:12, 7102.71it/s] 78%|███████▊  | 310219/400000 [00:45<00:12, 7005.11it/s] 78%|███████▊  | 310921/400000 [00:45<00:12, 6948.84it/s] 78%|███████▊  | 311617/400000 [00:45<00:12, 6939.27it/s] 78%|███████▊  | 312342/400000 [00:45<00:12, 7027.00it/s] 78%|███████▊  | 313061/400000 [00:45<00:12, 7073.80it/s] 78%|███████▊  | 313772/400000 [00:45<00:12, 7082.57it/s] 79%|███████▊  | 314481/400000 [00:46<00:12, 7044.17it/s] 79%|███████▉  | 315200/400000 [00:46<00:11, 7086.01it/s] 79%|███████▉  | 315919/400000 [00:46<00:11, 7114.94it/s] 79%|███████▉  | 316643/400000 [00:46<00:11, 7150.84it/s] 79%|███████▉  | 317359/400000 [00:46<00:11, 7059.13it/s] 80%|███████▉  | 318074/400000 [00:46<00:11, 7084.34it/s] 80%|███████▉  | 318794/400000 [00:46<00:11, 7118.00it/s] 80%|███████▉  | 319516/400000 [00:46<00:11, 7146.38it/s] 80%|████████  | 320244/400000 [00:46<00:11, 7183.19it/s] 80%|████████  | 320963/400000 [00:46<00:11, 7132.06it/s] 80%|████████  | 321677/400000 [00:47<00:11, 7118.52it/s] 81%|████████  | 322390/400000 [00:47<00:11, 7040.70it/s] 81%|████████  | 323095/400000 [00:47<00:11, 6935.22it/s] 81%|████████  | 323790/400000 [00:47<00:11, 6766.75it/s] 81%|████████  | 324468/400000 [00:47<00:11, 6675.11it/s] 81%|████████▏ | 325137/400000 [00:47<00:11, 6621.42it/s] 81%|████████▏ | 325847/400000 [00:47<00:10, 6755.77it/s] 82%|████████▏ | 326547/400000 [00:47<00:10, 6825.47it/s] 82%|████████▏ | 327255/400000 [00:47<00:10, 6897.98it/s] 82%|████████▏ | 327966/400000 [00:47<00:10, 6958.60it/s] 82%|████████▏ | 328679/400000 [00:48<00:10, 7008.84it/s] 82%|████████▏ | 329402/400000 [00:48<00:09, 7072.23it/s] 83%|████████▎ | 330110/400000 [00:48<00:09, 7020.83it/s] 83%|████████▎ | 330826/400000 [00:48<00:09, 7061.02it/s] 83%|████████▎ | 331533/400000 [00:48<00:09, 7028.13it/s] 83%|████████▎ | 332251/400000 [00:48<00:09, 7071.50it/s] 83%|████████▎ | 332970/400000 [00:48<00:09, 7106.29it/s] 83%|████████▎ | 333681/400000 [00:48<00:09, 6973.29it/s] 84%|████████▎ | 334380/400000 [00:48<00:09, 6594.48it/s] 84%|████████▍ | 335101/400000 [00:49<00:09, 6767.75it/s] 84%|████████▍ | 335799/400000 [00:49<00:09, 6829.08it/s] 84%|████████▍ | 336486/400000 [00:49<00:09, 6737.47it/s] 84%|████████▍ | 337190/400000 [00:49<00:09, 6823.61it/s] 84%|████████▍ | 337901/400000 [00:49<00:08, 6905.91it/s] 85%|████████▍ | 338594/400000 [00:49<00:09, 6790.11it/s] 85%|████████▍ | 339275/400000 [00:49<00:09, 6467.19it/s] 85%|████████▍ | 339927/400000 [00:49<00:09, 6471.85it/s] 85%|████████▌ | 340579/400000 [00:49<00:09, 6483.76it/s] 85%|████████▌ | 341264/400000 [00:49<00:08, 6589.39it/s] 85%|████████▌ | 341925/400000 [00:50<00:08, 6493.35it/s] 86%|████████▌ | 342576/400000 [00:50<00:08, 6483.70it/s] 86%|████████▌ | 343226/400000 [00:50<00:08, 6383.81it/s] 86%|████████▌ | 343879/400000 [00:50<00:08, 6425.72it/s] 86%|████████▌ | 344523/400000 [00:50<00:08, 6412.95it/s] 86%|████████▋ | 345233/400000 [00:50<00:08, 6602.43it/s] 86%|████████▋ | 345961/400000 [00:50<00:07, 6790.65it/s] 87%|████████▋ | 346682/400000 [00:50<00:07, 6909.07it/s] 87%|████████▋ | 347399/400000 [00:50<00:07, 6982.69it/s] 87%|████████▋ | 348100/400000 [00:50<00:07, 6967.99it/s] 87%|████████▋ | 348827/400000 [00:51<00:07, 7054.98it/s] 87%|████████▋ | 349544/400000 [00:51<00:07, 7088.94it/s] 88%|████████▊ | 350274/400000 [00:51<00:06, 7148.74it/s] 88%|████████▊ | 350990/400000 [00:51<00:07, 6786.89it/s] 88%|████████▊ | 351674/400000 [00:51<00:07, 6628.79it/s] 88%|████████▊ | 352341/400000 [00:51<00:07, 6586.73it/s] 88%|████████▊ | 353072/400000 [00:51<00:06, 6787.39it/s] 88%|████████▊ | 353797/400000 [00:51<00:06, 6919.38it/s] 89%|████████▊ | 354492/400000 [00:51<00:06, 6924.25it/s] 89%|████████▉ | 355187/400000 [00:52<00:06, 6530.15it/s] 89%|████████▉ | 355847/400000 [00:52<00:07, 6248.18it/s] 89%|████████▉ | 356576/400000 [00:52<00:06, 6527.25it/s] 89%|████████▉ | 357307/400000 [00:52<00:06, 6742.75it/s] 90%|████████▉ | 358023/400000 [00:52<00:06, 6861.89it/s] 90%|████████▉ | 358723/400000 [00:52<00:05, 6901.67it/s] 90%|████████▉ | 359449/400000 [00:52<00:05, 7004.47it/s] 90%|█████████ | 360153/400000 [00:52<00:05, 7001.14it/s] 90%|█████████ | 360871/400000 [00:52<00:05, 7041.36it/s] 90%|█████████ | 361594/400000 [00:52<00:05, 7095.33it/s] 91%|█████████ | 362305/400000 [00:53<00:05, 7018.95it/s] 91%|█████████ | 363008/400000 [00:53<00:05, 7001.68it/s] 91%|█████████ | 363740/400000 [00:53<00:05, 7091.73it/s] 91%|█████████ | 364450/400000 [00:53<00:05, 7019.74it/s] 91%|█████████▏| 365176/400000 [00:53<00:04, 7089.08it/s] 91%|█████████▏| 365886/400000 [00:53<00:04, 7081.92it/s] 92%|█████████▏| 366617/400000 [00:53<00:04, 7146.57it/s] 92%|█████████▏| 367336/400000 [00:53<00:04, 7158.35it/s] 92%|█████████▏| 368053/400000 [00:53<00:04, 7161.41it/s] 92%|█████████▏| 368771/400000 [00:53<00:04, 7166.42it/s] 92%|█████████▏| 369488/400000 [00:54<00:04, 7090.70it/s] 93%|█████████▎| 370220/400000 [00:54<00:04, 7156.04it/s] 93%|█████████▎| 370951/400000 [00:54<00:04, 7199.41it/s] 93%|█████████▎| 371676/400000 [00:54<00:03, 7212.91it/s] 93%|█████████▎| 372398/400000 [00:54<00:03, 7178.66it/s] 93%|█████████▎| 373117/400000 [00:54<00:03, 7145.04it/s] 93%|█████████▎| 373842/400000 [00:54<00:03, 7175.83it/s] 94%|█████████▎| 374565/400000 [00:54<00:03, 7189.69it/s] 94%|█████████▍| 375295/400000 [00:54<00:03, 7222.37it/s] 94%|█████████▍| 376025/400000 [00:54<00:03, 7242.59it/s] 94%|█████████▍| 376750/400000 [00:55<00:03, 6915.27it/s] 94%|█████████▍| 377445/400000 [00:55<00:03, 6640.67it/s] 95%|█████████▍| 378114/400000 [00:55<00:03, 6511.07it/s] 95%|█████████▍| 378769/400000 [00:55<00:03, 6466.32it/s] 95%|█████████▍| 379419/400000 [00:55<00:03, 6463.71it/s] 95%|█████████▌| 380068/400000 [00:55<00:03, 6438.54it/s] 95%|█████████▌| 380714/400000 [00:55<00:02, 6443.52it/s] 95%|█████████▌| 381360/400000 [00:55<00:02, 6429.66it/s] 96%|█████████▌| 382014/400000 [00:55<00:02, 6459.63it/s] 96%|█████████▌| 382661/400000 [00:55<00:02, 6456.80it/s] 96%|█████████▌| 383315/400000 [00:56<00:02, 6480.28it/s] 96%|█████████▌| 384040/400000 [00:56<00:02, 6692.62it/s] 96%|█████████▌| 384759/400000 [00:56<00:02, 6832.88it/s] 96%|█████████▋| 385475/400000 [00:56<00:02, 6927.53it/s] 97%|█████████▋| 386170/400000 [00:56<00:02, 6707.21it/s] 97%|█████████▋| 386844/400000 [00:56<00:02, 6508.97it/s] 97%|█████████▋| 387549/400000 [00:56<00:01, 6661.17it/s] 97%|█████████▋| 388226/400000 [00:56<00:01, 6691.90it/s] 97%|█████████▋| 388898/400000 [00:56<00:01, 6550.76it/s] 97%|█████████▋| 389607/400000 [00:57<00:01, 6700.74it/s] 98%|█████████▊| 390280/400000 [00:57<00:01, 6670.88it/s] 98%|█████████▊| 390949/400000 [00:57<00:01, 6616.50it/s] 98%|█████████▊| 391612/400000 [00:57<00:01, 6545.58it/s] 98%|█████████▊| 392268/400000 [00:57<00:01, 6502.03it/s] 98%|█████████▊| 392966/400000 [00:57<00:01, 6622.19it/s] 98%|█████████▊| 393635/400000 [00:57<00:00, 6640.85it/s] 99%|█████████▊| 394354/400000 [00:57<00:00, 6796.49it/s] 99%|█████████▉| 395061/400000 [00:57<00:00, 6874.77it/s] 99%|█████████▉| 395787/400000 [00:57<00:00, 6985.41it/s] 99%|█████████▉| 396508/400000 [00:58<00:00, 7049.61it/s] 99%|█████████▉| 397217/400000 [00:58<00:00, 7059.11it/s] 99%|█████████▉| 397924/400000 [00:58<00:00, 7027.41it/s]100%|█████████▉| 398642/400000 [00:58<00:00, 7070.44it/s]100%|█████████▉| 399354/400000 [00:58<00:00, 7083.61it/s]100%|█████████▉| 399999/400000 [00:58<00:00, 6834.64it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7ff52c00b828>, <torchtext.data.dataset.TabularDataset object at 0x7ff52c00b978>, <torchtext.vocab.Vocab object at 0x7ff52c00b898>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 29.08 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 53.96 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 20.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 20.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.18 file/s]2020-06-13 22:47:15.087922: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-13 22:47:15.092776: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-06-13 22:47:15.093291: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5622be7e82e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-13 22:47:15.093637: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3784704/9912422 [00:00<00:00, 37281931.68it/s]9920512it [00:00, 37850346.73it/s]                             
0it [00:00, ?it/s]32768it [00:00, 599293.41it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 483631.18it/s]1654784it [00:00, 12148305.61it/s]                         
0it [00:00, ?it/s]8192it [00:00, 207584.12it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
