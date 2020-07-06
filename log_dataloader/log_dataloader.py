
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '077ac9573b3255f0836baba55f19fb6dbaa40c9d', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/077ac9573b3255f0836baba55f19fb6dbaa40c9d

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/077ac9573b3255f0836baba55f19fb6dbaa40c9d

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f6d0167f048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f6d0167f048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f6d6d2271e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f6d6d2271e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6d8b56eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6d8b56eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6d1a554620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6d1a554620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6d1a554620> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:23, 118128.90it/s] 86%|████████▋ | 8568832/9912422 [00:00<00:07, 168655.72it/s]9920512it [00:00, 39632421.30it/s]                           
0it [00:00, ?it/s]32768it [00:00, 532795.86it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 461524.61it/s]1654784it [00:00, 11297008.13it/s]                         
0it [00:00, ?it/s]8192it [00:00, 211527.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d01594978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d01588c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f6d1a554268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f6d1a554268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f6d1a554268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:01,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:01,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:01,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:00,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:00,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:00,  2.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:42,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:42,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:42,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:41,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:41,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:40,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:40,  3.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:27,  5.09 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:19,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:19,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:19,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:19,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:19,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:18,  7.05 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:13,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:13,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:13,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:13,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:13,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:13,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:12,  9.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 13.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:09, 13.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:09, 13.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:09, 13.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:09, 13.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 13.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 13.15 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 17.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 29.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 36.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 43.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 50.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 63.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 63.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 72.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 72.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 75.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.40s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.40s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.40s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.04 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.69s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.40s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.04 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.69s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.70s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.49 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.70s/ url]
0 examples [00:00, ? examples/s]2020-07-06 06:09:29.041955: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-06 06:09:29.057829: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-06 06:09:29.057998: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562543ee35d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-06 06:09:29.058018: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
15 examples [00:00, 149.37 examples/s]107 examples [00:00, 199.35 examples/s]199 examples [00:00, 260.53 examples/s]283 examples [00:00, 328.42 examples/s]368 examples [00:00, 402.28 examples/s]461 examples [00:00, 484.12 examples/s]554 examples [00:00, 564.75 examples/s]641 examples [00:00, 631.13 examples/s]731 examples [00:00, 691.70 examples/s]823 examples [00:01, 745.35 examples/s]919 examples [00:01, 796.80 examples/s]1008 examples [00:01, 812.68 examples/s]1096 examples [00:01, 824.43 examples/s]1186 examples [00:01, 843.52 examples/s]1275 examples [00:01, 855.46 examples/s]1365 examples [00:01, 867.05 examples/s]1457 examples [00:01, 880.70 examples/s]1547 examples [00:01, 880.20 examples/s]1642 examples [00:01, 898.72 examples/s]1734 examples [00:02, 903.01 examples/s]1829 examples [00:02, 915.85 examples/s]1921 examples [00:02, 912.50 examples/s]2013 examples [00:02, 909.66 examples/s]2105 examples [00:02, 894.96 examples/s]2196 examples [00:02, 897.44 examples/s]2286 examples [00:02, 894.95 examples/s]2376 examples [00:02, 882.11 examples/s]2469 examples [00:02, 892.57 examples/s]2560 examples [00:02, 897.68 examples/s]2650 examples [00:03, 895.40 examples/s]2741 examples [00:03, 897.96 examples/s]2835 examples [00:03, 908.71 examples/s]2929 examples [00:03, 917.34 examples/s]3021 examples [00:03, 909.35 examples/s]3114 examples [00:03, 914.54 examples/s]3206 examples [00:03, 908.55 examples/s]3297 examples [00:03, 901.14 examples/s]3388 examples [00:03, 896.40 examples/s]3480 examples [00:03, 900.97 examples/s]3571 examples [00:04, 865.40 examples/s]3658 examples [00:04, 845.54 examples/s]3747 examples [00:04, 856.82 examples/s]3837 examples [00:04, 869.30 examples/s]3925 examples [00:04, 861.47 examples/s]4016 examples [00:04, 873.06 examples/s]4104 examples [00:04, 848.28 examples/s]4190 examples [00:04, 851.63 examples/s]4282 examples [00:04, 868.74 examples/s]4370 examples [00:05, 853.58 examples/s]4459 examples [00:05, 862.57 examples/s]4550 examples [00:05, 874.47 examples/s]4645 examples [00:05, 895.58 examples/s]4736 examples [00:05, 899.14 examples/s]4827 examples [00:05, 899.60 examples/s]4918 examples [00:05, 893.60 examples/s]5008 examples [00:05, 873.22 examples/s]5101 examples [00:05, 887.41 examples/s]5195 examples [00:05, 895.66 examples/s]5285 examples [00:06, 890.08 examples/s]5375 examples [00:06, 890.93 examples/s]5470 examples [00:06, 907.36 examples/s]5563 examples [00:06, 913.04 examples/s]5655 examples [00:06, 911.36 examples/s]5747 examples [00:06, 909.95 examples/s]5843 examples [00:06, 922.46 examples/s]5938 examples [00:06, 927.54 examples/s]6031 examples [00:06, 927.18 examples/s]6124 examples [00:06, 903.86 examples/s]6215 examples [00:07, 888.93 examples/s]6305 examples [00:07, 869.44 examples/s]6393 examples [00:07, 862.04 examples/s]6480 examples [00:07, 850.48 examples/s]6566 examples [00:07, 849.28 examples/s]6655 examples [00:07, 860.21 examples/s]6746 examples [00:07, 873.56 examples/s]6839 examples [00:07, 887.17 examples/s]6928 examples [00:07, 875.21 examples/s]7016 examples [00:07, 861.20 examples/s]7104 examples [00:08, 864.93 examples/s]7197 examples [00:08, 881.39 examples/s]7286 examples [00:08, 881.62 examples/s]7380 examples [00:08, 897.09 examples/s]7471 examples [00:08, 898.57 examples/s]7561 examples [00:08, 889.81 examples/s]7654 examples [00:08, 901.04 examples/s]7745 examples [00:08, 877.89 examples/s]7840 examples [00:08, 896.07 examples/s]7933 examples [00:08, 904.87 examples/s]8024 examples [00:09, 894.05 examples/s]8117 examples [00:09, 902.55 examples/s]8209 examples [00:09, 906.95 examples/s]8302 examples [00:09, 912.92 examples/s]8394 examples [00:09, 891.20 examples/s]8484 examples [00:09, 887.58 examples/s]8573 examples [00:09, 875.40 examples/s]8663 examples [00:09, 882.63 examples/s]8753 examples [00:09, 886.21 examples/s]8845 examples [00:10, 895.19 examples/s]8935 examples [00:10, 881.12 examples/s]9024 examples [00:10, 852.99 examples/s]9110 examples [00:10, 852.14 examples/s]9202 examples [00:10, 868.84 examples/s]9295 examples [00:10, 886.05 examples/s]9385 examples [00:10, 887.99 examples/s]9477 examples [00:10, 895.82 examples/s]9569 examples [00:10, 902.68 examples/s]9661 examples [00:10, 905.49 examples/s]9753 examples [00:11, 908.78 examples/s]9844 examples [00:11, 897.86 examples/s]9934 examples [00:11, 895.13 examples/s]10024 examples [00:11, 831.38 examples/s]10116 examples [00:11, 851.35 examples/s]10208 examples [00:11, 869.23 examples/s]10296 examples [00:11, 864.24 examples/s]10387 examples [00:11, 876.97 examples/s]10476 examples [00:11, 869.38 examples/s]10569 examples [00:11, 886.18 examples/s]10658 examples [00:12, 870.09 examples/s]10746 examples [00:12, 841.85 examples/s]10840 examples [00:12, 867.88 examples/s]10928 examples [00:12, 871.36 examples/s]11017 examples [00:12, 875.70 examples/s]11106 examples [00:12, 877.89 examples/s]11194 examples [00:12, 867.93 examples/s]11281 examples [00:12, 867.55 examples/s]11370 examples [00:12, 872.48 examples/s]11463 examples [00:13, 887.88 examples/s]11558 examples [00:13, 905.31 examples/s]11649 examples [00:13, 896.92 examples/s]11743 examples [00:13, 908.59 examples/s]11835 examples [00:13, 909.43 examples/s]11927 examples [00:13, 910.56 examples/s]12019 examples [00:13, 898.11 examples/s]12109 examples [00:13, 895.23 examples/s]12204 examples [00:13, 909.14 examples/s]12296 examples [00:13, 888.80 examples/s]12386 examples [00:14, 875.30 examples/s]12474 examples [00:14, 865.46 examples/s]12566 examples [00:14, 878.76 examples/s]12661 examples [00:14, 897.48 examples/s]12754 examples [00:14, 905.34 examples/s]12845 examples [00:14, 903.57 examples/s]12936 examples [00:14, 884.51 examples/s]13025 examples [00:14, 885.92 examples/s]13114 examples [00:14, 884.93 examples/s]13203 examples [00:14, 850.30 examples/s]13290 examples [00:15, 855.45 examples/s]13376 examples [00:15, 838.63 examples/s]13468 examples [00:15, 858.90 examples/s]13557 examples [00:15, 866.47 examples/s]13646 examples [00:15, 873.05 examples/s]13735 examples [00:15, 876.15 examples/s]13823 examples [00:15, 872.98 examples/s]13911 examples [00:15, 873.74 examples/s]14000 examples [00:15, 878.18 examples/s]14093 examples [00:15, 889.95 examples/s]14183 examples [00:16, 883.17 examples/s]14272 examples [00:16, 872.55 examples/s]14360 examples [00:16, 816.51 examples/s]14454 examples [00:16, 848.16 examples/s]14546 examples [00:16, 868.46 examples/s]14635 examples [00:16, 874.27 examples/s]14730 examples [00:16, 894.42 examples/s]14820 examples [00:16, 884.79 examples/s]14909 examples [00:16, 872.62 examples/s]14997 examples [00:17, 871.87 examples/s]15088 examples [00:17, 881.63 examples/s]15177 examples [00:17, 878.20 examples/s]15271 examples [00:17, 894.28 examples/s]15361 examples [00:17, 883.97 examples/s]15451 examples [00:17, 887.14 examples/s]15545 examples [00:17, 901.81 examples/s]15637 examples [00:17, 906.45 examples/s]15729 examples [00:17, 908.20 examples/s]15821 examples [00:17, 911.65 examples/s]15913 examples [00:18, 894.25 examples/s]16003 examples [00:18, 886.72 examples/s]16092 examples [00:18, 879.41 examples/s]16181 examples [00:18, 881.76 examples/s]16270 examples [00:18, 867.47 examples/s]16357 examples [00:18, 865.25 examples/s]16451 examples [00:18, 883.38 examples/s]16540 examples [00:18, 859.09 examples/s]16630 examples [00:18, 870.27 examples/s]16723 examples [00:18, 886.67 examples/s]16814 examples [00:19, 892.44 examples/s]16906 examples [00:19, 898.52 examples/s]16996 examples [00:19, 876.84 examples/s]17088 examples [00:19, 888.00 examples/s]17181 examples [00:19, 898.60 examples/s]17272 examples [00:19, 892.68 examples/s]17365 examples [00:19, 903.25 examples/s]17456 examples [00:19, 904.79 examples/s]17548 examples [00:19, 905.78 examples/s]17639 examples [00:20, 879.37 examples/s]17728 examples [00:20, 865.39 examples/s]17815 examples [00:20, 865.88 examples/s]17902 examples [00:20, 860.38 examples/s]17996 examples [00:20, 879.50 examples/s]18085 examples [00:20, 882.17 examples/s]18175 examples [00:20, 885.43 examples/s]18267 examples [00:20, 893.11 examples/s]18357 examples [00:20, 892.97 examples/s]18447 examples [00:20, 883.06 examples/s]18536 examples [00:21, 873.26 examples/s]18624 examples [00:21, 847.86 examples/s]18715 examples [00:21, 864.01 examples/s]18805 examples [00:21, 873.94 examples/s]18895 examples [00:21, 880.70 examples/s]18988 examples [00:21, 894.54 examples/s]19078 examples [00:21, 889.12 examples/s]19168 examples [00:21, 887.05 examples/s]19260 examples [00:21, 893.75 examples/s]19350 examples [00:21, 894.09 examples/s]19440 examples [00:22, 887.55 examples/s]19535 examples [00:22, 904.72 examples/s]19626 examples [00:22, 889.17 examples/s]19717 examples [00:22, 893.71 examples/s]19809 examples [00:22, 899.61 examples/s]19904 examples [00:22, 911.61 examples/s]19997 examples [00:22, 915.53 examples/s]20089 examples [00:22, 850.95 examples/s]20183 examples [00:22, 873.57 examples/s]20275 examples [00:22, 885.49 examples/s]20365 examples [00:23, 888.66 examples/s]20455 examples [00:23, 887.69 examples/s]20545 examples [00:23, 882.34 examples/s]20634 examples [00:23, 853.04 examples/s]20722 examples [00:23, 858.83 examples/s]20811 examples [00:23, 865.99 examples/s]20903 examples [00:23, 881.10 examples/s]20992 examples [00:23, 878.08 examples/s]21085 examples [00:23, 892.14 examples/s]21178 examples [00:24, 902.06 examples/s]21269 examples [00:24, 896.14 examples/s]21359 examples [00:24, 895.12 examples/s]21449 examples [00:24, 862.41 examples/s]21544 examples [00:24, 885.99 examples/s]21633 examples [00:24, 879.94 examples/s]21725 examples [00:24, 890.59 examples/s]21815 examples [00:24, 883.72 examples/s]21904 examples [00:24, 869.99 examples/s]21993 examples [00:24, 875.02 examples/s]22084 examples [00:25, 882.56 examples/s]22174 examples [00:25, 885.64 examples/s]22263 examples [00:25, 885.44 examples/s]22352 examples [00:25, 886.24 examples/s]22441 examples [00:25, 885.08 examples/s]22533 examples [00:25, 895.20 examples/s]22628 examples [00:25, 908.73 examples/s]22719 examples [00:25, 896.81 examples/s]22809 examples [00:25, 883.56 examples/s]22898 examples [00:25, 884.78 examples/s]22987 examples [00:26, 879.68 examples/s]23076 examples [00:26, 882.28 examples/s]23168 examples [00:26, 891.24 examples/s]23258 examples [00:26, 860.50 examples/s]23350 examples [00:26, 877.32 examples/s]23443 examples [00:26, 891.62 examples/s]23535 examples [00:26, 898.59 examples/s]23626 examples [00:26, 901.75 examples/s]23719 examples [00:26, 907.60 examples/s]23813 examples [00:26, 914.88 examples/s]23905 examples [00:27, 915.90 examples/s]24003 examples [00:27, 932.85 examples/s]24097 examples [00:27, 908.41 examples/s]24190 examples [00:27, 914.77 examples/s]24284 examples [00:27, 921.43 examples/s]24377 examples [00:27, 882.53 examples/s]24470 examples [00:27, 894.22 examples/s]24561 examples [00:27, 898.03 examples/s]24652 examples [00:27, 890.17 examples/s]24742 examples [00:28, 878.98 examples/s]24831 examples [00:28, 871.17 examples/s]24919 examples [00:28, 866.47 examples/s]25006 examples [00:28, 865.93 examples/s]25093 examples [00:28, 848.87 examples/s]25183 examples [00:28, 862.05 examples/s]25273 examples [00:28, 871.10 examples/s]25361 examples [00:28, 867.29 examples/s]25449 examples [00:28, 870.94 examples/s]25540 examples [00:28, 880.34 examples/s]25630 examples [00:29, 885.60 examples/s]25720 examples [00:29, 889.68 examples/s]25810 examples [00:29, 876.88 examples/s]25898 examples [00:29, 876.84 examples/s]25986 examples [00:29, 874.46 examples/s]26074 examples [00:29, 870.35 examples/s]26162 examples [00:29, 871.51 examples/s]26256 examples [00:29, 888.08 examples/s]26345 examples [00:29, 861.35 examples/s]26434 examples [00:29, 867.32 examples/s]26521 examples [00:30, 865.67 examples/s]26613 examples [00:30, 880.02 examples/s]26703 examples [00:30, 883.83 examples/s]26792 examples [00:30, 885.01 examples/s]26886 examples [00:30, 898.11 examples/s]26978 examples [00:30, 903.13 examples/s]27069 examples [00:30, 878.84 examples/s]27162 examples [00:30, 893.25 examples/s]27255 examples [00:30, 900.59 examples/s]27346 examples [00:30, 902.40 examples/s]27437 examples [00:31, 900.92 examples/s]27528 examples [00:31, 885.16 examples/s]27618 examples [00:31, 888.40 examples/s]27707 examples [00:31, 882.51 examples/s]27797 examples [00:31, 886.51 examples/s]27886 examples [00:31, 876.62 examples/s]27977 examples [00:31, 884.45 examples/s]28067 examples [00:31, 887.43 examples/s]28156 examples [00:31, 880.89 examples/s]28249 examples [00:31, 893.08 examples/s]28339 examples [00:32, 887.07 examples/s]28428 examples [00:32, 884.18 examples/s]28518 examples [00:32, 887.52 examples/s]28607 examples [00:32, 872.05 examples/s]28695 examples [00:32, 861.82 examples/s]28786 examples [00:32, 874.04 examples/s]28874 examples [00:32, 866.61 examples/s]28964 examples [00:32, 874.89 examples/s]29052 examples [00:32, 861.26 examples/s]29139 examples [00:33, 862.38 examples/s]29231 examples [00:33, 878.13 examples/s]29325 examples [00:33, 894.28 examples/s]29416 examples [00:33, 897.54 examples/s]29506 examples [00:33, 890.03 examples/s]29599 examples [00:33, 900.54 examples/s]29690 examples [00:33, 888.00 examples/s]29779 examples [00:33, 878.58 examples/s]29877 examples [00:33, 903.85 examples/s]29968 examples [00:33, 898.80 examples/s]30059 examples [00:34, 830.56 examples/s]30147 examples [00:34, 844.64 examples/s]30233 examples [00:34, 825.62 examples/s]30322 examples [00:34, 841.20 examples/s]30412 examples [00:34, 857.55 examples/s]30504 examples [00:34, 873.77 examples/s]30596 examples [00:34, 886.03 examples/s]30685 examples [00:34, 871.82 examples/s]30777 examples [00:34, 885.03 examples/s]30866 examples [00:34, 881.34 examples/s]30960 examples [00:35, 896.52 examples/s]31050 examples [00:35, 890.31 examples/s]31140 examples [00:35, 885.50 examples/s]31229 examples [00:35, 879.34 examples/s]31318 examples [00:35, 875.13 examples/s]31411 examples [00:35, 889.47 examples/s]31501 examples [00:35, 870.32 examples/s]31592 examples [00:35, 879.73 examples/s]31684 examples [00:35, 890.84 examples/s]31778 examples [00:36, 904.09 examples/s]31870 examples [00:36, 907.66 examples/s]31961 examples [00:36, 901.32 examples/s]32052 examples [00:36, 890.14 examples/s]32142 examples [00:36, 880.44 examples/s]32231 examples [00:36, 879.54 examples/s]32320 examples [00:36, 846.21 examples/s]32406 examples [00:36, 849.89 examples/s]32493 examples [00:36, 855.58 examples/s]32582 examples [00:36, 865.47 examples/s]32673 examples [00:37, 875.15 examples/s]32761 examples [00:37, 851.35 examples/s]32852 examples [00:37, 866.90 examples/s]32945 examples [00:37, 882.58 examples/s]33034 examples [00:37, 872.45 examples/s]33122 examples [00:37, 865.83 examples/s]33212 examples [00:37, 874.56 examples/s]33303 examples [00:37, 882.31 examples/s]33392 examples [00:37, 881.99 examples/s]33481 examples [00:37, 849.89 examples/s]33568 examples [00:38, 855.59 examples/s]33655 examples [00:38, 859.16 examples/s]33745 examples [00:38, 869.36 examples/s]33835 examples [00:38, 876.30 examples/s]33925 examples [00:38, 882.22 examples/s]34014 examples [00:38, 882.48 examples/s]34103 examples [00:38, 879.83 examples/s]34192 examples [00:38, 871.65 examples/s]34284 examples [00:38, 884.59 examples/s]34374 examples [00:38, 886.62 examples/s]34468 examples [00:39, 899.33 examples/s]34559 examples [00:39, 899.15 examples/s]34649 examples [00:39, 889.45 examples/s]34739 examples [00:39, 879.36 examples/s]34828 examples [00:39, 870.86 examples/s]34917 examples [00:39, 864.09 examples/s]35006 examples [00:39, 869.64 examples/s]35094 examples [00:39, 841.55 examples/s]35184 examples [00:39, 856.49 examples/s]35270 examples [00:40, 854.48 examples/s]35358 examples [00:40, 860.39 examples/s]35447 examples [00:40, 868.72 examples/s]35534 examples [00:40, 829.75 examples/s]35626 examples [00:40, 854.48 examples/s]35713 examples [00:40, 856.31 examples/s]35805 examples [00:40, 872.32 examples/s]35893 examples [00:40, 846.67 examples/s]35979 examples [00:40, 841.92 examples/s]36066 examples [00:40, 849.95 examples/s]36152 examples [00:41, 848.16 examples/s]36244 examples [00:41, 867.54 examples/s]36332 examples [00:41, 871.12 examples/s]36422 examples [00:41, 877.20 examples/s]36510 examples [00:41, 873.55 examples/s]36598 examples [00:41, 856.11 examples/s]36689 examples [00:41, 870.80 examples/s]36781 examples [00:41, 882.28 examples/s]36871 examples [00:41, 885.67 examples/s]36962 examples [00:41, 891.29 examples/s]37052 examples [00:42, 889.08 examples/s]37141 examples [00:42, 868.32 examples/s]37228 examples [00:42, 866.82 examples/s]37317 examples [00:42, 872.01 examples/s]37406 examples [00:42, 875.47 examples/s]37495 examples [00:42, 878.33 examples/s]37587 examples [00:42, 888.25 examples/s]37678 examples [00:42, 893.09 examples/s]37768 examples [00:42, 870.27 examples/s]37863 examples [00:42, 891.03 examples/s]37954 examples [00:43, 894.91 examples/s]38047 examples [00:43, 904.74 examples/s]38138 examples [00:43, 902.44 examples/s]38232 examples [00:43, 912.89 examples/s]38326 examples [00:43, 918.88 examples/s]38418 examples [00:43, 905.43 examples/s]38509 examples [00:43, 906.18 examples/s]38601 examples [00:43, 908.74 examples/s]38692 examples [00:43, 899.02 examples/s]38783 examples [00:44, 899.63 examples/s]38874 examples [00:44, 889.55 examples/s]38964 examples [00:44, 884.66 examples/s]39059 examples [00:44, 900.12 examples/s]39150 examples [00:44, 898.72 examples/s]39240 examples [00:44, 898.17 examples/s]39330 examples [00:44, 871.77 examples/s]39419 examples [00:44, 875.92 examples/s]39507 examples [00:44, 876.13 examples/s]39600 examples [00:44, 888.31 examples/s]39689 examples [00:45, 884.16 examples/s]39778 examples [00:45, 866.01 examples/s]39867 examples [00:45, 871.80 examples/s]39959 examples [00:45, 882.75 examples/s]40048 examples [00:45, 821.84 examples/s]40143 examples [00:45, 854.47 examples/s]40230 examples [00:45, 856.60 examples/s]40321 examples [00:45, 870.88 examples/s]40417 examples [00:45, 893.99 examples/s]40507 examples [00:45, 864.46 examples/s]40595 examples [00:46, 846.47 examples/s]40682 examples [00:46, 851.67 examples/s]40768 examples [00:46, 826.33 examples/s]40861 examples [00:46, 854.51 examples/s]40952 examples [00:46, 867.54 examples/s]41040 examples [00:46, 857.81 examples/s]41127 examples [00:46, 852.92 examples/s]41220 examples [00:46, 872.74 examples/s]41312 examples [00:46, 885.50 examples/s]41405 examples [00:47, 896.77 examples/s]41495 examples [00:47, 886.87 examples/s]41584 examples [00:47, 881.08 examples/s]41677 examples [00:47, 892.85 examples/s]41770 examples [00:47, 901.06 examples/s]41862 examples [00:47, 903.38 examples/s]41953 examples [00:47, 890.95 examples/s]42045 examples [00:47, 898.31 examples/s]42135 examples [00:47, 875.16 examples/s]42232 examples [00:47, 900.84 examples/s]42323 examples [00:48, 899.50 examples/s]42414 examples [00:48, 890.51 examples/s]42505 examples [00:48, 894.96 examples/s]42597 examples [00:48, 901.51 examples/s]42690 examples [00:48, 906.91 examples/s]42782 examples [00:48, 908.81 examples/s]42873 examples [00:48, 899.57 examples/s]42967 examples [00:48, 909.80 examples/s]43059 examples [00:48, 910.71 examples/s]43151 examples [00:48, 904.40 examples/s]43243 examples [00:49, 907.39 examples/s]43334 examples [00:49, 895.21 examples/s]43424 examples [00:49, 824.95 examples/s]43514 examples [00:49, 845.34 examples/s]43600 examples [00:49, 841.42 examples/s]43689 examples [00:49, 854.66 examples/s]43775 examples [00:49, 853.73 examples/s]43867 examples [00:49, 871.54 examples/s]43955 examples [00:49, 845.47 examples/s]44046 examples [00:50, 862.80 examples/s]44133 examples [00:50, 851.48 examples/s]44219 examples [00:50, 834.96 examples/s]44308 examples [00:50, 848.79 examples/s]44401 examples [00:50, 867.63 examples/s]44493 examples [00:50, 879.69 examples/s]44582 examples [00:50, 882.60 examples/s]44671 examples [00:50, 864.86 examples/s]44758 examples [00:50, 852.25 examples/s]44844 examples [00:50, 850.74 examples/s]44932 examples [00:51, 859.13 examples/s]45020 examples [00:51, 863.33 examples/s]45109 examples [00:51, 870.41 examples/s]45197 examples [00:51, 858.97 examples/s]45287 examples [00:51, 869.80 examples/s]45375 examples [00:51, 870.12 examples/s]45464 examples [00:51, 874.10 examples/s]45552 examples [00:51, 870.94 examples/s]45644 examples [00:51, 883.38 examples/s]45734 examples [00:51, 886.73 examples/s]45823 examples [00:52, 869.34 examples/s]45914 examples [00:52, 880.29 examples/s]46004 examples [00:52, 877.29 examples/s]46094 examples [00:52, 882.16 examples/s]46186 examples [00:52, 890.73 examples/s]46280 examples [00:52, 904.61 examples/s]46371 examples [00:52, 899.31 examples/s]46462 examples [00:52, 892.19 examples/s]46554 examples [00:52, 897.68 examples/s]46645 examples [00:52, 900.81 examples/s]46736 examples [00:53, 895.05 examples/s]46828 examples [00:53, 898.04 examples/s]46918 examples [00:53, 864.61 examples/s]47006 examples [00:53, 868.74 examples/s]47094 examples [00:53, 871.37 examples/s]47182 examples [00:53, 870.39 examples/s]47271 examples [00:53, 873.81 examples/s]47363 examples [00:53, 887.10 examples/s]47455 examples [00:53, 894.66 examples/s]47547 examples [00:53, 900.46 examples/s]47638 examples [00:54, 897.06 examples/s]47728 examples [00:54, 878.20 examples/s]47816 examples [00:54, 870.80 examples/s]47906 examples [00:54, 878.90 examples/s]47999 examples [00:54, 892.75 examples/s]48089 examples [00:54, 885.98 examples/s]48178 examples [00:54, 880.26 examples/s]48267 examples [00:54, 877.32 examples/s]48356 examples [00:54, 880.65 examples/s]48449 examples [00:55, 894.33 examples/s]48539 examples [00:55, 885.11 examples/s]48628 examples [00:55, 866.77 examples/s]48715 examples [00:55, 866.99 examples/s]48810 examples [00:55, 888.44 examples/s]48906 examples [00:55, 906.73 examples/s]48997 examples [00:55, 906.30 examples/s]49089 examples [00:55, 907.94 examples/s]49180 examples [00:55, 896.41 examples/s]49271 examples [00:55, 897.81 examples/s]49364 examples [00:56, 905.70 examples/s]49455 examples [00:56, 904.42 examples/s]49546 examples [00:56, 895.23 examples/s]49636 examples [00:56, 895.62 examples/s]49726 examples [00:56, 894.13 examples/s]49816 examples [00:56, 887.25 examples/s]49905 examples [00:56, 863.31 examples/s]49993 examples [00:56, 864.66 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6271/50000 [00:00<00:00, 62709.91 examples/s] 34%|███▍      | 17192/50000 [00:00<00:00, 71892.56 examples/s] 57%|█████▋    | 28581/50000 [00:00<00:00, 80834.60 examples/s] 81%|████████  | 40477/50000 [00:00<00:00, 89432.82 examples/s]                                                               0 examples [00:00, ? examples/s]57 examples [00:00, 567.40 examples/s]151 examples [00:00, 643.08 examples/s]245 examples [00:00, 709.15 examples/s]335 examples [00:00, 756.53 examples/s]426 examples [00:00, 796.74 examples/s]515 examples [00:00, 821.92 examples/s]608 examples [00:00, 851.00 examples/s]696 examples [00:00, 859.35 examples/s]781 examples [00:00, 843.72 examples/s]873 examples [00:01, 862.84 examples/s]959 examples [00:01, 833.56 examples/s]1049 examples [00:01, 850.24 examples/s]1138 examples [00:01, 859.94 examples/s]1229 examples [00:01, 873.05 examples/s]1318 examples [00:01, 877.77 examples/s]1407 examples [00:01, 880.68 examples/s]1500 examples [00:01, 893.94 examples/s]1590 examples [00:01, 889.56 examples/s]1682 examples [00:01, 898.06 examples/s]1774 examples [00:02, 904.24 examples/s]1865 examples [00:02, 898.45 examples/s]1955 examples [00:02, 896.15 examples/s]2048 examples [00:02, 905.44 examples/s]2140 examples [00:02, 907.89 examples/s]2232 examples [00:02, 910.66 examples/s]2324 examples [00:02, 880.40 examples/s]2413 examples [00:02, 869.91 examples/s]2503 examples [00:02, 877.63 examples/s]2591 examples [00:02, 876.48 examples/s]2679 examples [00:03, 868.29 examples/s]2766 examples [00:03, 859.56 examples/s]2855 examples [00:03, 867.91 examples/s]2942 examples [00:03, 864.06 examples/s]3030 examples [00:03, 866.61 examples/s]3121 examples [00:03, 877.60 examples/s]3212 examples [00:03, 886.80 examples/s]3303 examples [00:03, 891.37 examples/s]3395 examples [00:03, 898.84 examples/s]3485 examples [00:03, 883.52 examples/s]3575 examples [00:04, 887.69 examples/s]3668 examples [00:04, 898.62 examples/s]3762 examples [00:04, 908.41 examples/s]3853 examples [00:04, 904.71 examples/s]3944 examples [00:04, 894.33 examples/s]4034 examples [00:04, 878.60 examples/s]4123 examples [00:04, 881.59 examples/s]4215 examples [00:04, 892.53 examples/s]4309 examples [00:04, 905.53 examples/s]4402 examples [00:04, 912.09 examples/s]4494 examples [00:05, 901.83 examples/s]4585 examples [00:05, 895.68 examples/s]4678 examples [00:05, 903.44 examples/s]4773 examples [00:05, 916.74 examples/s]4865 examples [00:05, 906.97 examples/s]4956 examples [00:05, 897.19 examples/s]5046 examples [00:05, 892.63 examples/s]5136 examples [00:05, 889.25 examples/s]5225 examples [00:05, 880.36 examples/s]5317 examples [00:06, 889.81 examples/s]5410 examples [00:06, 898.58 examples/s]5500 examples [00:06, 896.21 examples/s]5591 examples [00:06, 898.88 examples/s]5684 examples [00:06, 906.05 examples/s]5775 examples [00:06, 890.69 examples/s]5871 examples [00:06, 909.62 examples/s]5963 examples [00:06, 903.00 examples/s]6054 examples [00:06, 895.95 examples/s]6144 examples [00:06, 886.67 examples/s]6233 examples [00:07, 873.76 examples/s]6321 examples [00:07, 857.77 examples/s]6409 examples [00:07, 863.99 examples/s]6501 examples [00:07, 877.80 examples/s]6589 examples [00:07, 856.75 examples/s]6680 examples [00:07, 871.89 examples/s]6776 examples [00:07, 894.69 examples/s]6867 examples [00:07, 896.71 examples/s]6957 examples [00:07, 886.29 examples/s]7048 examples [00:07, 891.92 examples/s]7140 examples [00:08, 899.24 examples/s]7233 examples [00:08, 907.53 examples/s]7324 examples [00:08, 904.75 examples/s]7415 examples [00:08, 900.69 examples/s]7510 examples [00:08, 913.22 examples/s]7602 examples [00:08, 914.70 examples/s]7694 examples [00:08, 914.75 examples/s]7786 examples [00:08, 886.82 examples/s]7875 examples [00:08, 884.90 examples/s]7967 examples [00:08, 894.68 examples/s]8058 examples [00:09, 896.63 examples/s]8151 examples [00:09, 905.53 examples/s]8242 examples [00:09, 903.23 examples/s]8333 examples [00:09, 894.40 examples/s]8427 examples [00:09, 905.65 examples/s]8518 examples [00:09, 905.11 examples/s]8610 examples [00:09, 907.55 examples/s]8702 examples [00:09, 908.86 examples/s]8793 examples [00:09, 906.93 examples/s]8884 examples [00:09, 902.53 examples/s]8975 examples [00:10, 828.29 examples/s]9063 examples [00:10, 841.13 examples/s]9157 examples [00:10, 867.42 examples/s]9247 examples [00:10, 875.36 examples/s]9339 examples [00:10, 886.27 examples/s]9429 examples [00:10, 885.65 examples/s]9518 examples [00:10, 881.52 examples/s]9607 examples [00:10, 871.28 examples/s]9695 examples [00:10, 864.14 examples/s]9783 examples [00:11, 866.25 examples/s]9874 examples [00:11, 876.53 examples/s]9965 examples [00:11, 883.99 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW146E0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW146E0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6d1a554620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6d1a554620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6d1a554620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6cbc315278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6cbc34dd30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6d1a554620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6d1a554620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6d1a554620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d015885f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6d01588390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f6c99c2f1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f6c99c2f1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f6c99c2f1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f6c99c2f2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f6c99c2f2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f6c99c2f2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=00b6b1aea3774ab3ba6af2b63e092baa28b500112099479890a949a710b3257d
  Stored in directory: /tmp/pip-ephem-wheel-cache-84e19y3e/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.0
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:47:10, 11.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:46:51, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:23:58, 23.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:17:17, 32.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:05:20, 46.9kB/s].vector_cache/glove.6B.zip:   1%|          | 8.72M/862M [00:01<3:32:33, 66.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:28:11, 95.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:43:12, 136kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.3M/862M [00:01<1:12:03, 194kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<50:13, 277kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:01<35:08, 395kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.8M/862M [00:01<24:31, 562kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:02<17:12, 797kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<12:02, 1.13MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<08:31, 1.59MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<06:02, 2.24MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.1M/862M [00:03<06:08, 2.19MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.2M/862M [00:03<04:44, 2.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.2M/862M [00:05<05:48, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:05<07:18, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:05<05:57, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:20, 3.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<10:22, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<08:43, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:07<06:23, 2.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<07:23, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:09<07:05, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<05:20, 2.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<04:28, 2.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<03:58, 3.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:11<8:01:55, 27.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:11<5:38:06, 39.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:11<3:56:14, 55.7kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<2:49:42, 77.4kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<2:06:53, 103kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<1:30:45, 145kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:13<1:03:55, 205kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:13<44:45, 292kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:15<1:06:21, 197kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:15<49:56, 262kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<35:50, 364kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:15<25:13, 516kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<22:40, 573kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<19:09, 678kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<14:16, 910kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:17<10:11, 1.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<12:20, 1.05MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<11:56, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<09:10, 1.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:19<06:38, 1.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<10:03, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:21<10:18, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<08:00, 1.60MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:21<05:49, 2.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<09:39, 1.32MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:23<09:54, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<07:36, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:23<05:31, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<07:47, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:25<08:32, 1.49MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<06:45, 1.88MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<04:54, 2.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<09:11, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<09:39, 1.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<07:26, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<08:19, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<08:43, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<06:44, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<04:52, 2.57MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<08:29, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<08:51, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<06:49, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:56, 2.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<08:06, 1.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<08:33, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:36, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<04:47, 2.59MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:13, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:38, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<06:37, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:51, 2.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<07:00, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<07:45, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<06:02, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<04:23, 2.79MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<08:12, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<08:27, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<06:31, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<04:45, 2.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<09:29, 1.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<09:44, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<07:29, 1.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<05:25, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<07:41, 1.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<08:37, 1.40MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:50, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<04:57, 2.43MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<08:13, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<08:49, 1.36MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:49, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<04:58, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<07:01, 1.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<08:08, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<06:21, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<04:37, 2.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<07:13, 1.65MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<08:06, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<06:27, 1.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:41, 2.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<08:01, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<08:29, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<06:41, 1.77MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<04:52, 2.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<08:32, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<08:34, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<06:32, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<04:48, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:55<06:34, 1.78MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<07:09, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:34, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:04, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<09:11, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<08:58, 1.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:55, 1.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:59, 2.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<10:05, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<09:36, 1.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<07:21, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<05:17, 2.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<10:05, 1.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<09:35, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<07:14, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<05:11, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<10:33, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<09:53, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<07:26, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:21, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<09:02, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<04:54, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<09:37, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<09:12, 1.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:03, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:05, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<09:46, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<09:19, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:02, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<05:03, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<08:36, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<09:00, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:03, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:07, 2.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<08:00, 1.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<08:44, 1.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:46, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:55, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<07:41, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:48, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:02, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:22, 2.50MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<09:37, 1.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<09:07, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:53, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<04:57, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<08:06, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<08:02, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:12, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:28, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<09:34, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<08:59, 1.20MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:46, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:56, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<06:30, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<06:56, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:25, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:56, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<12:00, 886kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<10:39, 998kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:40, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<09:03, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:26<09:01, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:52, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:04, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:53, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:22, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:56, 2.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:36, 2.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:16, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<06:35, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<05:04, 2.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<03:40, 2.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<08:49, 1.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<08:21, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<06:18, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:32, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<09:04, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<08:25, 1.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<06:21, 1.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<04:33, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<10:28, 976kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<09:48, 1.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<07:23, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:19, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<07:36, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<5:47:59, 29.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<4:04:02, 41.6kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<2:50:20, 59.4kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<2:03:25, 81.8kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:31:49, 110kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<1:05:28, 154kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<46:02, 219kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<34:02, 294kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<26:22, 380kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<19:03, 525kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<13:25, 742kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<14:12, 701kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<11:51, 839kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<08:46, 1.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<06:15, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<18:11, 543kB/s] .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<15:07, 654kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<11:05, 891kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<07:52, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:57, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:11, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:10, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:48<04:25, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<42:51, 227kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<31:48, 306kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<22:40, 429kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<15:54, 609kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<46:13, 209kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<34:04, 284kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<24:15, 398kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<17:01, 565kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<2:30:13, 64.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<1:46:46, 90.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<1:15:02, 128kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<53:53, 177kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<39:54, 239kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<28:24, 335kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<19:56, 476kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<17:49, 531kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<13:59, 676kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<10:06, 936kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<07:10, 1.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<11:41, 805kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<10:14, 917kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:40, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:00<05:28, 1.70MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<15:10, 615kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<12:03, 773kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<08:47, 1.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<07:47, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<09:15, 1.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:28, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:26, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:49, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:59, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:39, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:20, 2.73MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<50:17, 181kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<37:02, 246kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<26:21, 346kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<18:27, 491kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<1:17:46, 116kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<56:14, 161kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<39:44, 228kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<27:46, 324kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<1:02:47, 143kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<45:10, 199kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<31:50, 282kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<23:51, 374kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<20:00, 446kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<14:51, 600kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<10:34, 840kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<09:29, 932kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<08:17, 1.07MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<06:12, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:46, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:40, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:22, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:49, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:46, 2.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:04, 2.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:03, 1.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:57, 1.74MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:40, 2.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:34, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:51, 1.77MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:47, 2.26MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:03, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:29, 2.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:50, 2.97MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<5:19:55, 26.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<3:43:58, 37.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<2:37:09, 53.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<1:53:21, 73.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<1:20:05, 105kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<56:04, 149kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<41:04, 202kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<30:17, 274kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<21:33, 385kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<16:22, 503kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<14:46, 558kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<11:06, 741kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<07:56, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:27, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:44, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:05, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:53, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<06:40, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:20, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:52, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:37, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:45, 1.69MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:40, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:41, 2.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:56, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:59, 1.59MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:51, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:00, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:43, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:40, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:27, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:15, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:29, 1.74MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:30, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:43, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:02, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:11, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<03:29, 2.20MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<05:18, 1.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:25, 1.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:14, 2.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:07, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:18, 1.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:18, 2.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:24, 3.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<07:04, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<07:45, 972kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:53<06:03, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:24, 1.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:49, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:45, 1.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:41, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:47, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:09, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:25, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:21, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:26, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:16, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:01, 1.81MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:09, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<03:15, 2.23MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:20, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<36:48, 196kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<27:08, 265kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<19:17, 373kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<14:36, 489kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<12:52, 555kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<09:41, 736kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:54, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:30, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:55, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:28, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:16, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:18, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:20, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:24, 2.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<09:47, 708kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:08, 851kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:00, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:19, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<06:28, 1.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:11, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:45, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:13, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<3:48:10, 29.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<2:39:42, 42.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<1:52:01, 60.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<1:21:06, 83.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<57:16, 117kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<40:03, 167kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<29:30, 226kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<21:52, 305kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<15:33, 427kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<10:52, 607kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<1:03:21, 104kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<46:59, 140kB/s]  .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<33:23, 197kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<23:30, 280kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<17:35, 371kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<13:31, 482kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:45, 667kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:22<06:50, 944kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<46:48, 138kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<33:56, 190kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<24:00, 268kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<17:44, 360kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<13:38, 468kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<09:49, 648kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:52, 803kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:41, 945kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:58, 1.27MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:29, 1.39MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<05:29, 1.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:25, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:13, 1.93MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:47, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:47, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:56, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:03, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:26, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<03:37, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:40, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:17, 1.83MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:26, 1.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:41, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:07, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<02:24, 2.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:45, 3.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<05:07, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:57, 991kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:44, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:25, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:46, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:44, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:53, 2.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:57, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:28, 2.33MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:49, 3.14MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:18, 1.72MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:22, 1.69MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:37, 2.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:59, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:18, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<01:40, 3.33MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<05:37, 989kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<06:10, 901kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:46, 1.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:29, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:42, 1.48MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<03:37, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:45, 1.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:58, 2.76MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<10:34, 514kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<09:34, 567kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<07:13, 750kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<05:08, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:51, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:23, 1.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:17, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:20, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<09:40, 547kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<08:54, 594kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<06:44, 784kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<04:48, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:34, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:10, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:08, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:26, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<3:15:54, 26.4kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<2:17:00, 37.6kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:01<1:35:02, 53.7kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<1:13:42, 69.1kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<53:37, 94.9kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<37:59, 134kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<26:33, 190kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<19:39, 256kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<14:41, 342kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<10:27, 479kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<07:18, 679kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<10:45, 461kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<09:30, 521kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<07:04, 699kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<05:01, 980kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:37, 1.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:08, 1.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:07, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:57, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:00, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:19, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:23, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:33, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:00, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:10, 2.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:16, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:40, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:58, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:28, 1.87MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:35, 1.78MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:01, 2.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:09, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:21, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:51, 2.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:01, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:17, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:48, 2.47MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:58, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:09, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:34, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:53, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:19, 1.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:28, 1.75MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:55, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:02, 2.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:13, 1.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:44, 2.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:16, 3.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:55, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:49, 1.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:08, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:33, 2.67MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<02:42, 1.53MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:39, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:02, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:27, 2.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<09:30, 427kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<07:26, 546kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<05:20, 758kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:46, 1.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<04:02, 987kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:36, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:41, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:31, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:29, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:53, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:21, 2.85MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:26, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:50, 1.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:02, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:12, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:28, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:26, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:52, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:54, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:01, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:35, 2.33MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:42, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:39, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:12, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:36, 2.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:23, 2.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<2:06:54, 28.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<1:28:39, 40.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<1:01:41, 57.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<44:34, 78.9kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<31:27, 112kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<21:56, 159kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<16:00, 216kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<11:51, 291kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<08:25, 407kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<06:22, 531kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<05:06, 661kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:43, 904kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:07, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:48, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:07, 1.56MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:59, 1.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:59, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:32, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:35, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:42, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:20, 2.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:26, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:36, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:15, 2.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:23, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:07, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:17, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:35, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:41, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:18, 2.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<00:55, 3.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<19:41, 147kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<14:51, 195kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<10:37, 272kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<07:26, 385kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<05:48, 487kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<04:36, 613kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:20, 841kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:45, 1.00MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:58, 931kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:20, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:40, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:48, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:44, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:20, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:21, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:55, 1.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:35, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:09, 2.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:24, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:28, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:08, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:12, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:45, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:26, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:03, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:18, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:22, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:04, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:07, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:39, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:20, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:59, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:13, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:59, 2.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:33, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:15, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:56, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:03, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:08, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:53, 2.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:57, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:03, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:49, 2.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:53, 2.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:00, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:47, 2.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:51, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:58, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:46, 2.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:36, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<1:12:45, 25.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<50:37, 36.8kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<34:56, 52.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<24:40, 73.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<17:58, 100kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<12:42, 142kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<08:48, 201kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<06:27, 269kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<04:50, 359kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<03:25, 502kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<02:21, 711kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:52, 582kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:40, 622kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:00, 828kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:25, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:21, 1.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:14, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:56, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:53, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:55, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:42, 2.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:43, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:03, 1.39MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:52, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:37, 2.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:45, 1.83MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:47, 1.74MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:37, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:38, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:41, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:32, 2.42MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:34, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:52, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:43, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:31, 2.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:38, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:40, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:31, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:31, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:34, 1.93MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:27, 2.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:28, 2.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:32, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:25, 2.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:17, 3.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:11, 822kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:00, 965kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:44, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:30, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:12, 753kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:11, 767kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:54, 991kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:38, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:37, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:35, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:26, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:25, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:35, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:28, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:23, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:24, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:27, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:22, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:15, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:18, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:19, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:14, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:15, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:11, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:13, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:10, 2.47MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:10, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:08, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 2.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 2.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<03:32, 26.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<02:07, 38.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:28, 53.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:18, 74.5kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:09, 105kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<32:47:06,  3.39it/s]  0%|          | 779/400000 [00:00<22:54:33,  4.84it/s]  0%|          | 1495/400000 [00:00<16:00:44,  6.91it/s]  1%|          | 2275/400000 [00:00<11:11:27,  9.87it/s]  1%|          | 2986/400000 [00:00<7:49:27, 14.09it/s]   1%|          | 3761/400000 [00:00<5:28:14, 20.12it/s]  1%|          | 4560/400000 [00:00<3:49:32, 28.71it/s]  1%|▏         | 5299/400000 [00:00<2:40:39, 40.95it/s]  2%|▏         | 6059/400000 [00:01<1:52:29, 58.36it/s]  2%|▏         | 6837/400000 [00:01<1:18:50, 83.11it/s]  2%|▏         | 7623/400000 [00:01<55:19, 118.19it/s]   2%|▏         | 8391/400000 [00:01<38:54, 167.73it/s]  2%|▏         | 9191/400000 [00:01<27:25, 237.49it/s]  2%|▏         | 9960/400000 [00:01<19:24, 334.81it/s]  3%|▎         | 10728/400000 [00:01<13:49, 469.18it/s]  3%|▎         | 11487/400000 [00:01<09:56, 651.52it/s]  3%|▎         | 12228/400000 [00:01<07:12, 896.83it/s]  3%|▎         | 12969/400000 [00:02<05:17, 1217.50it/s]  3%|▎         | 13748/400000 [00:02<03:56, 1630.05it/s]  4%|▎         | 14499/400000 [00:02<03:02, 2113.86it/s]  4%|▍         | 15282/400000 [00:02<02:22, 2706.45it/s]  4%|▍         | 16029/400000 [00:02<01:55, 3337.27it/s]  4%|▍         | 16788/400000 [00:02<01:35, 4011.00it/s]  4%|▍         | 17551/400000 [00:02<01:21, 4675.65it/s]  5%|▍         | 18340/400000 [00:02<01:11, 5325.31it/s]  5%|▍         | 19103/400000 [00:02<01:05, 5789.76it/s]  5%|▍         | 19895/400000 [00:02<01:00, 6296.58it/s]  5%|▌         | 20659/400000 [00:03<00:57, 6638.25it/s]  5%|▌         | 21464/400000 [00:03<00:54, 7005.40it/s]  6%|▌         | 22271/400000 [00:03<00:51, 7292.10it/s]  6%|▌         | 23056/400000 [00:03<00:51, 7376.36it/s]  6%|▌         | 23851/400000 [00:03<00:49, 7537.61it/s]  6%|▌         | 24653/400000 [00:03<00:48, 7675.90it/s]  6%|▋         | 25441/400000 [00:03<00:48, 7660.46it/s]  7%|▋         | 26247/400000 [00:03<00:48, 7773.89it/s]  7%|▋         | 27035/400000 [00:03<00:48, 7764.25it/s]  7%|▋         | 27819/400000 [00:03<00:47, 7782.09it/s]  7%|▋         | 28620/400000 [00:04<00:47, 7847.06it/s]  7%|▋         | 29409/400000 [00:04<00:47, 7775.87it/s]  8%|▊         | 30221/400000 [00:04<00:46, 7875.19it/s]  8%|▊         | 31011/400000 [00:04<00:46, 7854.60it/s]  8%|▊         | 31799/400000 [00:04<00:47, 7810.81it/s]  8%|▊         | 32598/400000 [00:04<00:46, 7862.13it/s]  8%|▊         | 33386/400000 [00:04<00:48, 7632.44it/s]  9%|▊         | 34152/400000 [00:04<00:48, 7609.99it/s]  9%|▊         | 34936/400000 [00:04<00:47, 7675.58it/s]  9%|▉         | 35728/400000 [00:04<00:47, 7747.14it/s]  9%|▉         | 36505/400000 [00:05<00:46, 7753.94it/s]  9%|▉         | 37282/400000 [00:05<00:47, 7704.62it/s] 10%|▉         | 38087/400000 [00:05<00:46, 7803.30it/s] 10%|▉         | 38869/400000 [00:05<00:46, 7794.17it/s] 10%|▉         | 39649/400000 [00:05<00:46, 7786.04it/s] 10%|█         | 40428/400000 [00:05<00:46, 7747.05it/s] 10%|█         | 41203/400000 [00:05<00:46, 7656.18it/s] 10%|█         | 41970/400000 [00:05<00:47, 7599.54it/s] 11%|█         | 42731/400000 [00:05<00:47, 7596.65it/s] 11%|█         | 43499/400000 [00:05<00:46, 7621.05it/s] 11%|█         | 44262/400000 [00:06<00:46, 7600.18it/s] 11%|█▏        | 45023/400000 [00:06<00:47, 7494.80it/s] 11%|█▏        | 45773/400000 [00:06<00:47, 7482.60it/s] 12%|█▏        | 46536/400000 [00:06<00:46, 7520.79it/s] 12%|█▏        | 47302/400000 [00:06<00:46, 7559.79it/s] 12%|█▏        | 48079/400000 [00:06<00:46, 7619.61it/s] 12%|█▏        | 48842/400000 [00:06<00:46, 7500.91it/s] 12%|█▏        | 49596/400000 [00:06<00:46, 7511.82it/s] 13%|█▎        | 50361/400000 [00:06<00:46, 7550.59it/s] 13%|█▎        | 51170/400000 [00:06<00:45, 7703.01it/s] 13%|█▎        | 51942/400000 [00:07<00:45, 7705.62it/s] 13%|█▎        | 52714/400000 [00:07<00:45, 7599.42it/s] 13%|█▎        | 53492/400000 [00:07<00:45, 7651.07it/s] 14%|█▎        | 54258/400000 [00:07<00:45, 7618.88it/s] 14%|█▍        | 55021/400000 [00:07<00:45, 7573.65it/s] 14%|█▍        | 55779/400000 [00:07<00:45, 7506.15it/s] 14%|█▍        | 56531/400000 [00:07<00:46, 7462.52it/s] 14%|█▍        | 57319/400000 [00:07<00:45, 7580.95it/s] 15%|█▍        | 58078/400000 [00:07<00:46, 7374.72it/s] 15%|█▍        | 58820/400000 [00:08<00:46, 7387.92it/s] 15%|█▍        | 59617/400000 [00:08<00:45, 7551.72it/s] 15%|█▌        | 60374/400000 [00:08<00:44, 7555.24it/s] 15%|█▌        | 61171/400000 [00:08<00:44, 7674.53it/s] 15%|█▌        | 61940/400000 [00:08<00:45, 7461.80it/s] 16%|█▌        | 62689/400000 [00:08<00:46, 7311.72it/s] 16%|█▌        | 63423/400000 [00:08<00:46, 7304.78it/s] 16%|█▌        | 64196/400000 [00:08<00:45, 7427.26it/s] 16%|█▌        | 64969/400000 [00:08<00:44, 7513.39it/s] 16%|█▋        | 65722/400000 [00:08<00:45, 7386.25it/s] 17%|█▋        | 66471/400000 [00:09<00:44, 7416.89it/s] 17%|█▋        | 67214/400000 [00:09<00:45, 7367.33it/s] 17%|█▋        | 67978/400000 [00:09<00:44, 7445.68it/s] 17%|█▋        | 68753/400000 [00:09<00:43, 7532.49it/s] 17%|█▋        | 69519/400000 [00:09<00:43, 7568.09it/s] 18%|█▊        | 70298/400000 [00:09<00:43, 7630.84it/s] 18%|█▊        | 71062/400000 [00:09<00:43, 7480.40it/s] 18%|█▊        | 71844/400000 [00:09<00:43, 7578.90it/s] 18%|█▊        | 72603/400000 [00:09<00:43, 7540.16it/s] 18%|█▊        | 73358/400000 [00:09<00:43, 7504.95it/s] 19%|█▊        | 74124/400000 [00:10<00:43, 7548.19it/s] 19%|█▊        | 74936/400000 [00:10<00:42, 7709.51it/s] 19%|█▉        | 75709/400000 [00:10<00:42, 7645.10it/s] 19%|█▉        | 76508/400000 [00:10<00:41, 7743.34it/s] 19%|█▉        | 77284/400000 [00:10<00:42, 7675.70it/s] 20%|█▉        | 78054/400000 [00:10<00:41, 7682.01it/s] 20%|█▉        | 78878/400000 [00:10<00:40, 7838.86it/s] 20%|█▉        | 79664/400000 [00:10<00:41, 7717.54it/s] 20%|██        | 80438/400000 [00:10<00:42, 7601.85it/s] 20%|██        | 81200/400000 [00:10<00:42, 7529.01it/s] 21%|██        | 82012/400000 [00:11<00:41, 7695.42it/s] 21%|██        | 82816/400000 [00:11<00:40, 7795.08it/s] 21%|██        | 83614/400000 [00:11<00:40, 7847.61it/s] 21%|██        | 84400/400000 [00:11<00:42, 7432.21it/s] 21%|██▏       | 85149/400000 [00:11<00:42, 7373.59it/s] 21%|██▏       | 85906/400000 [00:11<00:42, 7428.73it/s] 22%|██▏       | 86652/400000 [00:11<00:42, 7393.48it/s] 22%|██▏       | 87450/400000 [00:11<00:41, 7559.35it/s] 22%|██▏       | 88209/400000 [00:11<00:41, 7544.64it/s] 22%|██▏       | 88975/400000 [00:11<00:41, 7576.26it/s] 22%|██▏       | 89783/400000 [00:12<00:40, 7720.16it/s] 23%|██▎       | 90557/400000 [00:12<00:40, 7658.94it/s] 23%|██▎       | 91325/400000 [00:12<00:41, 7425.59it/s] 23%|██▎       | 92098/400000 [00:12<00:40, 7513.15it/s] 23%|██▎       | 92852/400000 [00:12<00:41, 7431.85it/s] 23%|██▎       | 93615/400000 [00:12<00:40, 7487.97it/s] 24%|██▎       | 94391/400000 [00:12<00:40, 7565.48it/s] 24%|██▍       | 95173/400000 [00:12<00:39, 7638.67it/s] 24%|██▍       | 95960/400000 [00:12<00:39, 7704.31it/s] 24%|██▍       | 96732/400000 [00:13<00:39, 7627.90it/s] 24%|██▍       | 97496/400000 [00:13<00:39, 7576.05it/s] 25%|██▍       | 98311/400000 [00:13<00:38, 7737.90it/s] 25%|██▍       | 99089/400000 [00:13<00:38, 7749.03it/s] 25%|██▍       | 99865/400000 [00:13<00:39, 7649.90it/s] 25%|██▌       | 100631/400000 [00:13<00:40, 7444.13it/s] 25%|██▌       | 101421/400000 [00:13<00:39, 7573.01it/s] 26%|██▌       | 102235/400000 [00:13<00:38, 7733.09it/s] 26%|██▌       | 103011/400000 [00:13<00:38, 7685.46it/s] 26%|██▌       | 103789/400000 [00:13<00:38, 7709.94it/s] 26%|██▌       | 104562/400000 [00:14<00:38, 7637.00it/s] 26%|██▋       | 105327/400000 [00:14<00:39, 7373.61it/s] 27%|██▋       | 106067/400000 [00:14<00:40, 7340.10it/s] 27%|██▋       | 106817/400000 [00:14<00:39, 7386.15it/s] 27%|██▋       | 107565/400000 [00:14<00:39, 7413.70it/s] 27%|██▋       | 108308/400000 [00:14<00:39, 7356.06it/s] 27%|██▋       | 109050/400000 [00:14<00:39, 7372.94it/s] 27%|██▋       | 109788/400000 [00:14<00:40, 7108.31it/s] 28%|██▊       | 110567/400000 [00:14<00:39, 7299.11it/s] 28%|██▊       | 111300/400000 [00:14<00:39, 7296.64it/s] 28%|██▊       | 112087/400000 [00:15<00:38, 7458.49it/s] 28%|██▊       | 112861/400000 [00:15<00:38, 7539.11it/s] 28%|██▊       | 113667/400000 [00:15<00:37, 7687.07it/s] 29%|██▊       | 114438/400000 [00:15<00:37, 7666.48it/s] 29%|██▉       | 115210/400000 [00:15<00:37, 7679.50it/s] 29%|██▉       | 115979/400000 [00:15<00:37, 7657.85it/s] 29%|██▉       | 116746/400000 [00:15<00:36, 7655.57it/s] 29%|██▉       | 117524/400000 [00:15<00:36, 7692.19it/s] 30%|██▉       | 118294/400000 [00:15<00:36, 7651.40it/s] 30%|██▉       | 119082/400000 [00:15<00:36, 7716.48it/s] 30%|██▉       | 119855/400000 [00:16<00:36, 7682.79it/s] 30%|███       | 120624/400000 [00:16<00:36, 7629.89it/s] 30%|███       | 121429/400000 [00:16<00:35, 7750.38it/s] 31%|███       | 122229/400000 [00:16<00:35, 7820.88it/s] 31%|███       | 123012/400000 [00:16<00:35, 7716.72it/s] 31%|███       | 123785/400000 [00:16<00:36, 7536.86it/s] 31%|███       | 124560/400000 [00:16<00:36, 7597.76it/s] 31%|███▏      | 125345/400000 [00:16<00:35, 7669.61it/s] 32%|███▏      | 126134/400000 [00:16<00:35, 7732.55it/s] 32%|███▏      | 126909/400000 [00:16<00:35, 7727.91it/s] 32%|███▏      | 127683/400000 [00:17<00:36, 7528.74it/s] 32%|███▏      | 128444/400000 [00:17<00:35, 7551.10it/s] 32%|███▏      | 129224/400000 [00:17<00:35, 7622.50it/s] 32%|███▏      | 129988/400000 [00:17<00:36, 7442.76it/s] 33%|███▎      | 130745/400000 [00:17<00:35, 7479.35it/s] 33%|███▎      | 131509/400000 [00:17<00:35, 7525.38it/s] 33%|███▎      | 132294/400000 [00:17<00:35, 7617.20it/s] 33%|███▎      | 133057/400000 [00:17<00:35, 7620.29it/s] 33%|███▎      | 133862/400000 [00:17<00:34, 7743.28it/s] 34%|███▎      | 134638/400000 [00:18<00:34, 7692.59it/s] 34%|███▍      | 135408/400000 [00:18<00:34, 7602.51it/s] 34%|███▍      | 136169/400000 [00:18<00:35, 7493.39it/s] 34%|███▍      | 136978/400000 [00:18<00:34, 7661.33it/s] 34%|███▍      | 137746/400000 [00:18<00:34, 7600.86it/s] 35%|███▍      | 138508/400000 [00:18<00:34, 7519.41it/s] 35%|███▍      | 139261/400000 [00:18<00:35, 7417.47it/s] 35%|███▌      | 140010/400000 [00:18<00:34, 7438.05it/s] 35%|███▌      | 140801/400000 [00:18<00:34, 7573.23it/s] 35%|███▌      | 141560/400000 [00:18<00:34, 7567.27it/s] 36%|███▌      | 142329/400000 [00:19<00:33, 7599.56it/s] 36%|███▌      | 143108/400000 [00:19<00:33, 7653.76it/s] 36%|███▌      | 143874/400000 [00:19<00:33, 7610.54it/s] 36%|███▌      | 144647/400000 [00:19<00:33, 7645.68it/s] 36%|███▋      | 145443/400000 [00:19<00:32, 7736.20it/s] 37%|███▋      | 146221/400000 [00:19<00:32, 7743.23it/s] 37%|███▋      | 146996/400000 [00:19<00:33, 7616.69it/s] 37%|███▋      | 147759/400000 [00:19<00:33, 7555.59it/s] 37%|███▋      | 148517/400000 [00:19<00:33, 7562.86it/s] 37%|███▋      | 149305/400000 [00:19<00:32, 7654.83it/s] 38%|███▊      | 150072/400000 [00:20<00:32, 7597.15it/s] 38%|███▊      | 150833/400000 [00:20<00:33, 7489.02it/s] 38%|███▊      | 151591/400000 [00:20<00:33, 7515.93it/s] 38%|███▊      | 152362/400000 [00:20<00:32, 7571.11it/s] 38%|███▊      | 153120/400000 [00:20<00:33, 7434.76it/s] 38%|███▊      | 153865/400000 [00:20<00:33, 7365.12it/s] 39%|███▊      | 154628/400000 [00:20<00:32, 7441.57it/s] 39%|███▉      | 155408/400000 [00:20<00:32, 7544.57it/s] 39%|███▉      | 156164/400000 [00:20<00:32, 7523.52it/s] 39%|███▉      | 156917/400000 [00:20<00:32, 7474.24it/s] 39%|███▉      | 157665/400000 [00:21<00:32, 7414.96it/s] 40%|███▉      | 158435/400000 [00:21<00:32, 7497.79it/s] 40%|███▉      | 159197/400000 [00:21<00:31, 7533.70it/s] 40%|███▉      | 159951/400000 [00:21<00:31, 7511.75it/s] 40%|████      | 160703/400000 [00:21<00:32, 7423.21it/s] 40%|████      | 161446/400000 [00:21<00:32, 7370.90it/s] 41%|████      | 162209/400000 [00:21<00:31, 7446.73it/s] 41%|████      | 163015/400000 [00:21<00:31, 7619.84it/s] 41%|████      | 163781/400000 [00:21<00:30, 7630.15it/s] 41%|████      | 164545/400000 [00:21<00:30, 7627.41it/s] 41%|████▏     | 165309/400000 [00:22<00:30, 7589.48it/s] 42%|████▏     | 166069/400000 [00:22<00:31, 7530.77it/s] 42%|████▏     | 166837/400000 [00:22<00:30, 7574.04it/s] 42%|████▏     | 167595/400000 [00:22<00:30, 7543.63it/s] 42%|████▏     | 168350/400000 [00:22<00:31, 7432.54it/s] 42%|████▏     | 169094/400000 [00:22<00:31, 7342.23it/s] 42%|████▏     | 169829/400000 [00:22<00:31, 7311.35it/s] 43%|████▎     | 170561/400000 [00:22<00:31, 7309.87it/s] 43%|████▎     | 171339/400000 [00:22<00:30, 7444.16it/s] 43%|████▎     | 172095/400000 [00:22<00:30, 7477.47it/s] 43%|████▎     | 172844/400000 [00:23<00:30, 7423.06it/s] 43%|████▎     | 173630/400000 [00:23<00:29, 7548.16it/s] 44%|████▎     | 174396/400000 [00:23<00:29, 7581.11it/s] 44%|████▍     | 175155/400000 [00:23<00:30, 7483.63it/s] 44%|████▍     | 175921/400000 [00:23<00:29, 7533.44it/s] 44%|████▍     | 176694/400000 [00:23<00:29, 7590.33it/s] 44%|████▍     | 177516/400000 [00:23<00:28, 7768.36it/s] 45%|████▍     | 178300/400000 [00:23<00:28, 7786.77it/s] 45%|████▍     | 179080/400000 [00:23<00:29, 7613.39it/s] 45%|████▍     | 179843/400000 [00:24<00:28, 7603.96it/s] 45%|████▌     | 180605/400000 [00:24<00:29, 7478.63it/s] 45%|████▌     | 181376/400000 [00:24<00:28, 7545.05it/s] 46%|████▌     | 182143/400000 [00:24<00:28, 7581.85it/s] 46%|████▌     | 182902/400000 [00:24<00:28, 7581.58it/s] 46%|████▌     | 183667/400000 [00:24<00:28, 7601.85it/s] 46%|████▌     | 184432/400000 [00:24<00:28, 7615.27it/s] 46%|████▋     | 185194/400000 [00:24<00:28, 7558.54it/s] 46%|████▋     | 185951/400000 [00:24<00:28, 7472.17it/s] 47%|████▋     | 186699/400000 [00:24<00:28, 7471.08it/s] 47%|████▋     | 187482/400000 [00:25<00:28, 7574.46it/s] 47%|████▋     | 188241/400000 [00:25<00:27, 7577.66it/s] 47%|████▋     | 189026/400000 [00:25<00:27, 7655.95it/s] 47%|████▋     | 189808/400000 [00:25<00:27, 7703.49it/s] 48%|████▊     | 190579/400000 [00:25<00:27, 7643.91it/s] 48%|████▊     | 191344/400000 [00:25<00:27, 7459.83it/s] 48%|████▊     | 192115/400000 [00:25<00:27, 7531.40it/s] 48%|████▊     | 192895/400000 [00:25<00:27, 7609.64it/s] 48%|████▊     | 193657/400000 [00:25<00:27, 7482.08it/s] 49%|████▊     | 194429/400000 [00:25<00:27, 7551.37it/s] 49%|████▉     | 195195/400000 [00:26<00:27, 7582.50it/s] 49%|████▉     | 195959/400000 [00:26<00:26, 7598.34it/s] 49%|████▉     | 196739/400000 [00:26<00:26, 7657.16it/s] 49%|████▉     | 197506/400000 [00:26<00:26, 7648.43it/s] 50%|████▉     | 198272/400000 [00:26<00:26, 7483.45it/s] 50%|████▉     | 199064/400000 [00:26<00:26, 7608.54it/s] 50%|████▉     | 199839/400000 [00:26<00:26, 7649.75it/s] 50%|█████     | 200621/400000 [00:26<00:25, 7697.46it/s] 50%|█████     | 201392/400000 [00:26<00:25, 7696.29it/s] 51%|█████     | 202163/400000 [00:26<00:25, 7692.90it/s] 51%|█████     | 202933/400000 [00:27<00:25, 7675.47it/s] 51%|█████     | 203721/400000 [00:27<00:25, 7733.84it/s] 51%|█████     | 204495/400000 [00:27<00:25, 7717.01it/s] 51%|█████▏    | 205267/400000 [00:27<00:25, 7715.36it/s] 52%|█████▏    | 206053/400000 [00:27<00:25, 7756.32it/s] 52%|█████▏    | 206846/400000 [00:27<00:24, 7806.90it/s] 52%|█████▏    | 207635/400000 [00:27<00:24, 7830.91it/s] 52%|█████▏    | 208419/400000 [00:27<00:24, 7760.41it/s] 52%|█████▏    | 209196/400000 [00:27<00:25, 7538.99it/s] 52%|█████▏    | 209952/400000 [00:27<00:25, 7535.35it/s] 53%|█████▎    | 210707/400000 [00:28<00:25, 7526.58it/s] 53%|█████▎    | 211496/400000 [00:28<00:24, 7625.29it/s] 53%|█████▎    | 212267/400000 [00:28<00:24, 7649.33it/s] 53%|█████▎    | 213033/400000 [00:28<00:24, 7553.75it/s] 53%|█████▎    | 213837/400000 [00:28<00:24, 7693.18it/s] 54%|█████▎    | 214608/400000 [00:28<00:24, 7695.39it/s] 54%|█████▍    | 215387/400000 [00:28<00:23, 7722.94it/s] 54%|█████▍    | 216171/400000 [00:28<00:23, 7756.67it/s] 54%|█████▍    | 216950/400000 [00:28<00:23, 7763.93it/s] 54%|█████▍    | 217727/400000 [00:28<00:23, 7734.24it/s] 55%|█████▍    | 218509/400000 [00:29<00:23, 7758.06it/s] 55%|█████▍    | 219307/400000 [00:29<00:23, 7823.05it/s] 55%|█████▌    | 220090/400000 [00:29<00:22, 7823.02it/s] 55%|█████▌    | 220901/400000 [00:29<00:22, 7903.74it/s] 55%|█████▌    | 221692/400000 [00:29<00:22, 7765.20it/s] 56%|█████▌    | 222495/400000 [00:29<00:22, 7840.89it/s] 56%|█████▌    | 223300/400000 [00:29<00:22, 7900.73it/s] 56%|█████▌    | 224091/400000 [00:29<00:22, 7824.92it/s] 56%|█████▌    | 224875/400000 [00:29<00:22, 7782.76it/s] 56%|█████▋    | 225654/400000 [00:29<00:22, 7716.22it/s] 57%|█████▋    | 226451/400000 [00:30<00:22, 7789.10it/s] 57%|█████▋    | 227231/400000 [00:30<00:22, 7772.68it/s] 57%|█████▋    | 228009/400000 [00:30<00:22, 7774.10it/s] 57%|█████▋    | 228813/400000 [00:30<00:21, 7849.84it/s] 57%|█████▋    | 229599/400000 [00:30<00:21, 7849.94it/s] 58%|█████▊    | 230385/400000 [00:30<00:22, 7653.13it/s] 58%|█████▊    | 231160/400000 [00:30<00:21, 7678.32it/s] 58%|█████▊    | 231929/400000 [00:30<00:21, 7646.09it/s] 58%|█████▊    | 232695/400000 [00:30<00:21, 7642.88it/s] 58%|█████▊    | 233460/400000 [00:31<00:22, 7260.26it/s] 59%|█████▊    | 234191/400000 [00:31<00:23, 7143.69it/s] 59%|█████▊    | 234909/400000 [00:31<00:23, 7118.71it/s] 59%|█████▉    | 235624/400000 [00:31<00:23, 7067.58it/s] 59%|█████▉    | 236333/400000 [00:31<00:23, 7038.31it/s] 59%|█████▉    | 237063/400000 [00:31<00:22, 7112.93it/s] 59%|█████▉    | 237820/400000 [00:31<00:22, 7243.16it/s] 60%|█████▉    | 238577/400000 [00:31<00:22, 7335.58it/s] 60%|█████▉    | 239370/400000 [00:31<00:21, 7503.55it/s] 60%|██████    | 240170/400000 [00:31<00:20, 7643.81it/s] 60%|██████    | 240960/400000 [00:32<00:20, 7716.84it/s] 60%|██████    | 241763/400000 [00:32<00:20, 7807.75it/s] 61%|██████    | 242546/400000 [00:32<00:20, 7752.15it/s] 61%|██████    | 243323/400000 [00:32<00:20, 7714.60it/s] 61%|██████    | 244096/400000 [00:32<00:20, 7538.33it/s] 61%|██████    | 244852/400000 [00:32<00:21, 7353.54it/s] 61%|██████▏   | 245631/400000 [00:32<00:20, 7477.48it/s] 62%|██████▏   | 246413/400000 [00:32<00:20, 7576.80it/s] 62%|██████▏   | 247173/400000 [00:32<00:20, 7361.44it/s] 62%|██████▏   | 247960/400000 [00:32<00:20, 7504.60it/s] 62%|██████▏   | 248719/400000 [00:33<00:20, 7528.77it/s] 62%|██████▏   | 249493/400000 [00:33<00:19, 7590.44it/s] 63%|██████▎   | 250308/400000 [00:33<00:19, 7748.06it/s] 63%|██████▎   | 251085/400000 [00:33<00:19, 7651.14it/s] 63%|██████▎   | 251852/400000 [00:33<00:19, 7646.06it/s] 63%|██████▎   | 252618/400000 [00:33<00:19, 7642.00it/s] 63%|██████▎   | 253389/400000 [00:33<00:19, 7659.46it/s] 64%|██████▎   | 254156/400000 [00:33<00:19, 7653.16it/s] 64%|██████▎   | 254941/400000 [00:33<00:18, 7709.00it/s] 64%|██████▍   | 255713/400000 [00:33<00:19, 7556.65it/s] 64%|██████▍   | 256491/400000 [00:34<00:18, 7622.19it/s] 64%|██████▍   | 257254/400000 [00:34<00:18, 7580.72it/s] 65%|██████▍   | 258013/400000 [00:34<00:18, 7581.41it/s] 65%|██████▍   | 258783/400000 [00:34<00:18, 7616.47it/s] 65%|██████▍   | 259551/400000 [00:34<00:18, 7633.93it/s] 65%|██████▌   | 260334/400000 [00:34<00:18, 7688.72it/s] 65%|██████▌   | 261140/400000 [00:34<00:17, 7794.66it/s] 65%|██████▌   | 261921/400000 [00:34<00:17, 7736.50it/s] 66%|██████▌   | 262696/400000 [00:34<00:17, 7635.84it/s] 66%|██████▌   | 263461/400000 [00:34<00:18, 7566.92it/s] 66%|██████▌   | 264219/400000 [00:35<00:18, 7511.31it/s] 66%|██████▋   | 265024/400000 [00:35<00:17, 7663.77it/s] 66%|██████▋   | 265822/400000 [00:35<00:17, 7754.83it/s] 67%|██████▋   | 266608/400000 [00:35<00:17, 7784.78it/s] 67%|██████▋   | 267388/400000 [00:35<00:17, 7611.40it/s] 67%|██████▋   | 268191/400000 [00:35<00:17, 7730.83it/s] 67%|██████▋   | 268969/400000 [00:35<00:16, 7744.52it/s] 67%|██████▋   | 269761/400000 [00:35<00:16, 7793.33it/s] 68%|██████▊   | 270542/400000 [00:35<00:16, 7645.42it/s] 68%|██████▊   | 271336/400000 [00:35<00:16, 7729.45it/s] 68%|██████▊   | 272114/400000 [00:36<00:16, 7743.83it/s] 68%|██████▊   | 272890/400000 [00:36<00:16, 7714.73it/s] 68%|██████▊   | 273663/400000 [00:36<00:16, 7649.12it/s] 69%|██████▊   | 274429/400000 [00:36<00:16, 7513.71it/s] 69%|██████▉   | 275187/400000 [00:36<00:16, 7531.87it/s] 69%|██████▉   | 275954/400000 [00:36<00:16, 7572.58it/s] 69%|██████▉   | 276722/400000 [00:36<00:16, 7603.79it/s] 69%|██████▉   | 277483/400000 [00:36<00:16, 7545.18it/s] 70%|██████▉   | 278238/400000 [00:36<00:16, 7442.59it/s] 70%|██████▉   | 279013/400000 [00:37<00:16, 7531.13it/s] 70%|██████▉   | 279778/400000 [00:37<00:15, 7565.39it/s] 70%|███████   | 280576/400000 [00:37<00:15, 7684.35it/s] 70%|███████   | 281346/400000 [00:37<00:15, 7582.14it/s] 71%|███████   | 282106/400000 [00:37<00:15, 7536.68it/s] 71%|███████   | 282861/400000 [00:37<00:15, 7387.23it/s] 71%|███████   | 283611/400000 [00:37<00:15, 7419.42it/s] 71%|███████   | 284402/400000 [00:37<00:15, 7559.65it/s] 71%|███████▏  | 285199/400000 [00:37<00:14, 7677.18it/s] 71%|███████▏  | 285969/400000 [00:37<00:14, 7669.05it/s] 72%|███████▏  | 286737/400000 [00:38<00:15, 7550.60it/s] 72%|███████▏  | 287494/400000 [00:38<00:14, 7547.20it/s] 72%|███████▏  | 288264/400000 [00:38<00:14, 7590.40it/s] 72%|███████▏  | 289060/400000 [00:38<00:14, 7695.59it/s] 72%|███████▏  | 289831/400000 [00:38<00:14, 7411.35it/s] 73%|███████▎  | 290582/400000 [00:38<00:14, 7440.14it/s] 73%|███████▎  | 291375/400000 [00:38<00:14, 7579.86it/s] 73%|███████▎  | 292136/400000 [00:38<00:14, 7363.81it/s] 73%|███████▎  | 292916/400000 [00:38<00:14, 7488.24it/s] 73%|███████▎  | 293668/400000 [00:38<00:14, 7497.72it/s] 74%|███████▎  | 294457/400000 [00:39<00:13, 7607.84it/s] 74%|███████▍  | 295225/400000 [00:39<00:13, 7627.14it/s] 74%|███████▍  | 295991/400000 [00:39<00:13, 7636.17it/s] 74%|███████▍  | 296760/400000 [00:39<00:13, 7650.80it/s] 74%|███████▍  | 297526/400000 [00:39<00:13, 7636.71it/s] 75%|███████▍  | 298307/400000 [00:39<00:13, 7686.59it/s] 75%|███████▍  | 299077/400000 [00:39<00:13, 7654.52it/s] 75%|███████▍  | 299845/400000 [00:39<00:13, 7661.17it/s] 75%|███████▌  | 300612/400000 [00:39<00:12, 7653.57it/s] 75%|███████▌  | 301380/400000 [00:39<00:12, 7660.52it/s] 76%|███████▌  | 302201/400000 [00:40<00:12, 7816.81it/s] 76%|███████▌  | 302984/400000 [00:40<00:12, 7760.77it/s] 76%|███████▌  | 303761/400000 [00:40<00:12, 7752.07it/s] 76%|███████▌  | 304576/400000 [00:40<00:12, 7866.78it/s] 76%|███████▋  | 305364/400000 [00:40<00:12, 7665.16it/s] 77%|███████▋  | 306133/400000 [00:40<00:12, 7553.79it/s] 77%|███████▋  | 306905/400000 [00:40<00:12, 7601.05it/s] 77%|███████▋  | 307667/400000 [00:40<00:12, 7604.64it/s] 77%|███████▋  | 308429/400000 [00:40<00:12, 7602.26it/s] 77%|███████▋  | 309190/400000 [00:40<00:11, 7583.16it/s] 77%|███████▋  | 309964/400000 [00:41<00:11, 7628.15it/s] 78%|███████▊  | 310744/400000 [00:41<00:11, 7677.37it/s] 78%|███████▊  | 311513/400000 [00:41<00:11, 7573.07it/s] 78%|███████▊  | 312289/400000 [00:41<00:11, 7627.76it/s] 78%|███████▊  | 313053/400000 [00:41<00:11, 7576.17it/s] 78%|███████▊  | 313845/400000 [00:41<00:11, 7676.04it/s] 79%|███████▊  | 314623/400000 [00:41<00:11, 7706.45it/s] 79%|███████▉  | 315395/400000 [00:41<00:11, 7631.31it/s] 79%|███████▉  | 316159/400000 [00:41<00:11, 7497.15it/s] 79%|███████▉  | 316924/400000 [00:41<00:11, 7541.33it/s] 79%|███████▉  | 317699/400000 [00:42<00:10, 7602.49it/s] 80%|███████▉  | 318464/400000 [00:42<00:10, 7611.12it/s] 80%|███████▉  | 319226/400000 [00:42<00:10, 7595.56it/s] 80%|███████▉  | 319986/400000 [00:42<00:10, 7561.23it/s] 80%|████████  | 320757/400000 [00:42<00:10, 7603.67it/s] 80%|████████  | 321549/400000 [00:42<00:10, 7695.26it/s] 81%|████████  | 322351/400000 [00:42<00:09, 7787.82it/s] 81%|████████  | 323146/400000 [00:42<00:09, 7833.30it/s] 81%|████████  | 323930/400000 [00:42<00:09, 7699.25it/s] 81%|████████  | 324722/400000 [00:43<00:09, 7762.53it/s] 81%|████████▏ | 325530/400000 [00:43<00:09, 7853.80it/s] 82%|████████▏ | 326317/400000 [00:43<00:09, 7848.39it/s] 82%|████████▏ | 327133/400000 [00:43<00:09, 7939.23it/s] 82%|████████▏ | 327928/400000 [00:43<00:09, 7850.46it/s] 82%|████████▏ | 328714/400000 [00:43<00:09, 7679.09it/s] 82%|████████▏ | 329502/400000 [00:43<00:09, 7737.98it/s] 83%|████████▎ | 330286/400000 [00:43<00:08, 7767.65it/s] 83%|████████▎ | 331064/400000 [00:43<00:08, 7759.25it/s] 83%|████████▎ | 331888/400000 [00:43<00:08, 7897.03it/s] 83%|████████▎ | 332679/400000 [00:44<00:08, 7762.22it/s] 83%|████████▎ | 333479/400000 [00:44<00:08, 7831.88it/s] 84%|████████▎ | 334264/400000 [00:44<00:08, 7706.05it/s] 84%|████████▍ | 335075/400000 [00:44<00:08, 7820.23it/s] 84%|████████▍ | 335859/400000 [00:44<00:08, 7805.02it/s] 84%|████████▍ | 336641/400000 [00:44<00:08, 7506.41it/s] 84%|████████▍ | 337420/400000 [00:44<00:08, 7584.91it/s] 85%|████████▍ | 338201/400000 [00:44<00:08, 7650.50it/s] 85%|████████▍ | 339006/400000 [00:44<00:07, 7764.22it/s] 85%|████████▍ | 339792/400000 [00:44<00:07, 7791.66it/s] 85%|████████▌ | 340575/400000 [00:45<00:07, 7800.68it/s] 85%|████████▌ | 341367/400000 [00:45<00:07, 7834.49it/s] 86%|████████▌ | 342169/400000 [00:45<00:07, 7887.66it/s] 86%|████████▌ | 342959/400000 [00:45<00:07, 7843.36it/s] 86%|████████▌ | 343767/400000 [00:45<00:07, 7911.16it/s] 86%|████████▌ | 344559/400000 [00:45<00:07, 7819.83it/s] 86%|████████▋ | 345365/400000 [00:45<00:06, 7887.22it/s] 87%|████████▋ | 346195/400000 [00:45<00:06, 8004.94it/s] 87%|████████▋ | 347004/400000 [00:45<00:06, 8028.00it/s] 87%|████████▋ | 347808/400000 [00:45<00:06, 7952.82it/s] 87%|████████▋ | 348604/400000 [00:46<00:06, 7858.72it/s] 87%|████████▋ | 349391/400000 [00:46<00:06, 7673.64it/s] 88%|████████▊ | 350160/400000 [00:46<00:06, 7650.04it/s] 88%|████████▊ | 350947/400000 [00:46<00:06, 7712.72it/s] 88%|████████▊ | 351737/400000 [00:46<00:06, 7767.21it/s] 88%|████████▊ | 352515/400000 [00:46<00:06, 7625.26it/s] 88%|████████▊ | 353279/400000 [00:46<00:06, 7575.18it/s] 89%|████████▊ | 354038/400000 [00:46<00:06, 7503.29it/s] 89%|████████▊ | 354790/400000 [00:46<00:06, 7253.41it/s] 89%|████████▉ | 355579/400000 [00:46<00:05, 7431.37it/s] 89%|████████▉ | 356338/400000 [00:47<00:05, 7477.28it/s] 89%|████████▉ | 357090/400000 [00:47<00:05, 7488.73it/s] 89%|████████▉ | 357841/400000 [00:47<00:05, 7493.43it/s] 90%|████████▉ | 358592/400000 [00:47<00:05, 7477.14it/s] 90%|████████▉ | 359341/400000 [00:47<00:05, 7251.85it/s] 90%|█████████ | 360131/400000 [00:47<00:05, 7433.22it/s] 90%|█████████ | 360949/400000 [00:47<00:05, 7638.24it/s] 90%|█████████ | 361796/400000 [00:47<00:04, 7868.98it/s] 91%|█████████ | 362587/400000 [00:47<00:04, 7732.73it/s] 91%|█████████ | 363364/400000 [00:48<00:04, 7660.88it/s] 91%|█████████ | 364137/400000 [00:48<00:04, 7679.70it/s] 91%|█████████ | 364907/400000 [00:48<00:04, 7665.10it/s] 91%|█████████▏| 365706/400000 [00:48<00:04, 7756.92it/s] 92%|█████████▏| 366483/400000 [00:48<00:04, 7742.86it/s] 92%|█████████▏| 367275/400000 [00:48<00:04, 7789.41it/s] 92%|█████████▏| 368055/400000 [00:48<00:04, 7769.66it/s] 92%|█████████▏| 368843/400000 [00:48<00:03, 7801.62it/s] 92%|█████████▏| 369686/400000 [00:48<00:03, 7978.56it/s] 93%|█████████▎| 370486/400000 [00:48<00:03, 7888.48it/s] 93%|█████████▎| 371276/400000 [00:49<00:03, 7853.74it/s] 93%|█████████▎| 372063/400000 [00:49<00:03, 7850.09it/s] 93%|█████████▎| 372866/400000 [00:49<00:03, 7902.09it/s] 93%|█████████▎| 373698/400000 [00:49<00:03, 8021.16it/s] 94%|█████████▎| 374501/400000 [00:49<00:03, 8020.01it/s] 94%|█████████▍| 375322/400000 [00:49<00:03, 8074.16it/s] 94%|█████████▍| 376130/400000 [00:49<00:02, 8013.60it/s] 94%|█████████▍| 376932/400000 [00:49<00:02, 7781.93it/s] 94%|█████████▍| 377721/400000 [00:49<00:02, 7812.32it/s] 95%|█████████▍| 378504/400000 [00:49<00:02, 7690.45it/s] 95%|█████████▍| 379275/400000 [00:50<00:02, 7435.28it/s] 95%|█████████▌| 380026/400000 [00:50<00:02, 7400.70it/s] 95%|█████████▌| 380816/400000 [00:50<00:02, 7543.02it/s] 95%|█████████▌| 381614/400000 [00:50<00:02, 7668.59it/s] 96%|█████████▌| 382393/400000 [00:50<00:02, 7702.07it/s] 96%|█████████▌| 383165/400000 [00:50<00:02, 7545.95it/s] 96%|█████████▌| 383939/400000 [00:50<00:02, 7602.07it/s] 96%|█████████▌| 384735/400000 [00:50<00:01, 7705.48it/s] 96%|█████████▋| 385548/400000 [00:50<00:01, 7827.43it/s] 97%|█████████▋| 386333/400000 [00:50<00:01, 7797.64it/s] 97%|█████████▋| 387114/400000 [00:51<00:01, 7772.49it/s] 97%|█████████▋| 387906/400000 [00:51<00:01, 7816.04it/s] 97%|█████████▋| 388689/400000 [00:51<00:01, 7632.58it/s] 97%|█████████▋| 389462/400000 [00:51<00:01, 7659.75it/s] 98%|█████████▊| 390229/400000 [00:51<00:01, 7572.02it/s] 98%|█████████▊| 390988/400000 [00:51<00:01, 7572.20it/s] 98%|█████████▊| 391746/400000 [00:51<00:01, 7498.79it/s] 98%|█████████▊| 392497/400000 [00:51<00:01, 7476.18it/s] 98%|█████████▊| 393308/400000 [00:51<00:00, 7653.57it/s] 99%|█████████▊| 394099/400000 [00:51<00:00, 7726.39it/s] 99%|█████████▊| 394877/400000 [00:52<00:00, 7741.57it/s] 99%|█████████▉| 395669/400000 [00:52<00:00, 7790.96it/s] 99%|█████████▉| 396451/400000 [00:52<00:00, 7796.89it/s] 99%|█████████▉| 397264/400000 [00:52<00:00, 7892.90it/s]100%|█████████▉| 398054/400000 [00:52<00:00, 7819.63it/s]100%|█████████▉| 398837/400000 [00:52<00:00, 7735.81it/s]100%|█████████▉| 399612/400000 [00:52<00:00, 7630.04it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7583.66it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f6c99c4ab38>, <torchtext.data.dataset.TabularDataset object at 0x7f6c99c4ac88>, <torchtext.vocab.Vocab object at 0x7f6c99c4aba8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 29.65 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 46.23 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.00 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.08 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.89 file/s]2020-07-06 06:18:54.693436: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-06 06:18:54.697585: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-07-06 06:18:54.698445: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a50688e930 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-06 06:18:54.698464: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['mnist_dataset_small.npy', 'cifar10', 'test', 'train', 'mnist2', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:04, 153653.72it/s] 62%|██████▏   | 6135808/9912422 [00:00<00:17, 219264.84it/s]9920512it [00:00, 40879494.51it/s]                           
0it [00:00, ?it/s]32768it [00:00, 552103.36it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486999.15it/s]1654784it [00:00, 12044102.78it/s]                         
0it [00:00, ?it/s]8192it [00:00, 223076.07it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
