
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '77c2c9f427106a88e922f586a543b21e10cfaf22', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/77c2c9f427106a88e922f586a543b21e10cfaf22

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe2eec4b620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe2eec4b620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe35a2120d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe35a2120d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe373f3eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe373f3eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe307a8f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe307a8f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe307a8f950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3203072/9912422 [00:00<00:00, 32026474.55it/s]9920512it [00:00, 28358197.68it/s]                             
0it [00:00, ?it/s]32768it [00:00, 590338.01it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 440595.14it/s]1654784it [00:00, 11691828.89it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185614.94it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe3050a5b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe30509ada0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe307a8f598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe307a8f598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe307a8f598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:51,  1.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:51,  1.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:51,  1.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 3/162 [00:00<01:20,  1.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:20,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:20,  1.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<00:59,  2.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:59,  2.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<00:58,  2.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:01<00:44,  3.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:44,  3.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:43,  3.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:01<00:33,  4.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:33,  4.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:33,  4.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:26,  5.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:26,  5.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:26,  5.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:01<00:21,  6.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:21,  6.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:21,  6.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:01<00:17,  8.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:17,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:17,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:15,  9.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:13, 10.93 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:02<00:12, 10.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:02<00:11, 11.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:02<00:11, 11.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:02<00:11, 11.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:02<00:10, 12.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:02<00:10, 12.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:02<00:10, 12.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:02<00:10, 13.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:02<00:10, 13.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:02<00:09, 13.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:02<00:09, 14.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:02<00:09, 14.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:02<00:09, 14.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:02<00:09, 14.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:02<00:09, 14.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:02<00:08, 14.73 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:02<00:08, 15.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:02<00:08, 15.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:02<00:08, 15.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:02<00:08, 15.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:02<00:08, 15.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:02<00:08, 15.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:02<00:08, 15.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:02<00:08, 15.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:02<00:07, 15.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:03<00:07, 16.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:03<00:07, 16.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:03<00:07, 16.23 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:03<00:07, 16.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:03<00:07, 16.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:03<00:07, 16.51 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:03<00:07, 16.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:03<00:07, 16.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:03<00:07, 16.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:03<00:07, 16.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:03<00:07, 16.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:03<00:06, 16.88 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:03<00:06, 17.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:03<00:06, 17.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:03<00:06, 17.01 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:03<00:06, 17.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:03<00:06, 17.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:03<00:06, 17.14 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:03<00:06, 17.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:03<00:06, 17.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:03<00:06, 17.30 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:03<00:06, 17.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:03<00:06, 17.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:03<00:06, 17.49 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:03<00:06, 17.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:03<00:06, 17.46 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:04<00:06, 17.46 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:04<00:06, 17.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:04<00:06, 17.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:04<00:06, 17.22 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:04<00:06, 17.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:04<00:06, 17.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:04<00:06, 17.32 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:04<00:05, 17.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:04<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:04<00:05, 17.48 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:04<00:05, 17.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:04<00:05, 17.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:04<00:05, 17.49 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:04<00:05, 17.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:04<00:05, 17.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:04<00:05, 17.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:04<00:05, 17.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:04<00:05, 17.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:04<00:05, 17.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:04<00:05, 17.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:04<00:05, 17.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:04<00:05, 17.86 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:04<00:05, 18.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:04<00:05, 18.38 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:04<00:05, 18.38 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:04<00:04, 18.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:04<00:04, 18.33 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:05<00:04, 18.33 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:05<00:04, 18.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:05<00:04, 18.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:05<00:04, 18.31 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:05<00:04, 18.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:05<00:04, 18.26 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:05<00:04, 18.26 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:05<00:04, 18.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:05<00:04, 18.59 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:05<00:04, 18.59 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:05<00:04, 18.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:05<00:04, 18.34 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:05<00:04, 18.34 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:05<00:04, 18.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:05<00:04, 18.45 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:05<00:04, 18.45 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:05<00:04, 18.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:05<00:04, 18.40 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:05<00:04, 18.40 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:05<00:04, 18.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:05<00:04, 18.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:05<00:04, 18.65 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:05<00:04, 18.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:05<00:04, 18.37 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:05<00:04, 18.37 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:05<00:03, 18.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:05<00:03, 18.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:05<00:03, 18.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:06<00:03, 18.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:06<00:03, 18.40 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:06<00:03, 18.40 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:06<00:03, 18.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:06<00:03, 18.64 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:06<00:03, 18.64 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:06<00:03, 18.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:06<00:03, 18.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:06<00:03, 18.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:06<00:03, 18.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:06<00:03, 18.56 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:06<00:03, 18.56 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:06<00:03, 18.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:06<00:03, 18.47 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:06<00:03, 18.47 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:06<00:03, 18.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:06<00:03, 18.70 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:06<00:03, 18.70 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:06<00:03, 18.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:06<00:03, 18.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:06<00:03, 18.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:06<00:03, 18.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:06<00:03, 18.31 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:06<00:03, 18.31 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:06<00:02, 18.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:06<00:02, 18.60 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:06<00:02, 18.60 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:07<00:02, 18.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:07<00:02, 18.36 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:07<00:02, 18.36 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:07<00:02, 18.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:07<00:02, 18.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:07<00:02, 18.63 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:07<00:02, 18.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:07<00:02, 18.42 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:07<00:02, 18.42 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:07<00:02, 18.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:07<00:02, 18.66 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:07<00:02, 18.66 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:07<00:02, 18.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:07<00:02, 18.38 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:07<00:02, 18.38 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:07<00:02, 18.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:07<00:02, 18.35 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:07<00:02, 18.35 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:07<00:02, 18.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:07<00:02, 18.59 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:07<00:02, 18.59 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:07<00:02, 18.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:07<00:02, 18.74 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:07<00:02, 18.74 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:07<00:01, 19.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:07<00:01, 19.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:07<00:01, 19.02 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:07<00:01, 18.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:07<00:01, 18.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:08<00:01, 18.83 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:08<00:01, 18.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:08<00:01, 18.89 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:08<00:01, 18.89 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:08<00:01, 19.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:08<00:01, 19.17 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:08<00:01, 19.17 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:08<00:01, 19.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:08<00:01, 19.25 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:08<00:01, 19.25 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:08<00:01, 18.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:08<00:01, 18.99 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:08<00:01, 18.99 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:08<00:01, 19.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:08<00:01, 19.20 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:08<00:01, 19.20 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:08<00:01, 19.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:08<00:01, 19.22 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:08<00:01, 19.22 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:08<00:01, 19.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:08<00:01, 19.39 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:08<00:01, 19.39 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:08<00:00, 19.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:08<00:00, 19.51 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:08<00:00, 19.51 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:08<00:00, 19.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:08<00:00, 19.61 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:08<00:00, 19.61 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:09<00:00, 19.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:09<00:00, 19.72 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:09<00:00, 19.72 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:09<00:00, 19.72 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:09<00:00, 19.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:09<00:00, 19.70 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:09<00:00, 19.70 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:09<00:00, 19.70 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:09<00:00, 19.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:09<00:00, 19.94 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:09<00:00, 19.94 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:09<00:00, 19.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:09<00:00, 19.88 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:09<00:00, 19.88 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:09<00:00, 19.88 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:09<00:00, 20.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:09<00:00, 20.11 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:09<00:00, 20.11 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:09<00:00, 20.11 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:09<00:00, 20.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:09<00:00, 20.27 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:09<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 20.27 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.76s/ url]Dl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.76s/ url]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 20.27 MiB/s][A

Extraction completed...: 0 file [00:09, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:09<00:00,  9.76s/ url]
Dl Size...: 100%|██████████| 162/162 [00:09<00:00, 20.27 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:09<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:12<00:00, 12.05s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:12<00:00,  9.76s/ url]
Dl Size...: 100%|██████████| 162/162 [00:12<00:00, 20.27 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:12<00:00, 12.05s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:12<00:00, 12.05s/ file]
Dl Size...: 100%|██████████| 162/162 [00:12<00:00, 13.44 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:12<00:00, 12.05s/ url]
0 examples [00:00, ? examples/s]2020-06-14 06:09:12.500829: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 06:09:12.515019: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-14 06:09:12.515184: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c22508dfe0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 06:09:12.515205: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
34 examples [00:00, 339.66 examples/s]130 examples [00:00, 421.33 examples/s]227 examples [00:00, 507.07 examples/s]328 examples [00:00, 595.77 examples/s]426 examples [00:00, 674.54 examples/s]528 examples [00:00, 749.25 examples/s]628 examples [00:00, 809.18 examples/s]721 examples [00:00, 841.41 examples/s]811 examples [00:00, 857.37 examples/s]907 examples [00:01, 885.34 examples/s]1004 examples [00:01, 906.87 examples/s]1102 examples [00:01, 925.44 examples/s]1197 examples [00:01, 927.70 examples/s]1291 examples [00:01, 919.95 examples/s]1386 examples [00:01, 927.86 examples/s]1481 examples [00:01, 932.88 examples/s]1576 examples [00:01, 936.40 examples/s]1673 examples [00:01, 944.99 examples/s]1768 examples [00:01, 934.45 examples/s]1865 examples [00:02, 944.23 examples/s]1964 examples [00:02, 956.62 examples/s]2065 examples [00:02, 969.52 examples/s]2163 examples [00:02, 971.79 examples/s]2263 examples [00:02, 978.12 examples/s]2363 examples [00:02, 984.51 examples/s]2462 examples [00:02, 976.05 examples/s]2562 examples [00:02, 982.80 examples/s]2662 examples [00:02, 985.33 examples/s]2761 examples [00:02, 975.02 examples/s]2859 examples [00:03, 963.04 examples/s]2960 examples [00:03, 974.28 examples/s]3058 examples [00:03, 967.74 examples/s]3155 examples [00:03, 955.94 examples/s]3251 examples [00:03, 937.25 examples/s]3345 examples [00:03, 934.72 examples/s]3441 examples [00:03, 941.71 examples/s]3536 examples [00:03, 942.59 examples/s]3634 examples [00:03, 953.12 examples/s]3730 examples [00:03, 934.71 examples/s]3828 examples [00:04, 945.49 examples/s]3927 examples [00:04, 956.94 examples/s]4026 examples [00:04, 965.65 examples/s]4123 examples [00:04, 963.61 examples/s]4220 examples [00:04, 962.09 examples/s]4317 examples [00:04, 954.01 examples/s]4413 examples [00:04, 955.25 examples/s]4509 examples [00:04, 954.64 examples/s]4605 examples [00:04, 927.23 examples/s]4699 examples [00:04, 928.85 examples/s]4793 examples [00:05, 918.12 examples/s]4885 examples [00:05, 917.93 examples/s]4977 examples [00:05, 901.06 examples/s]5069 examples [00:05, 905.96 examples/s]5167 examples [00:05, 924.34 examples/s]5260 examples [00:05, 925.16 examples/s]5354 examples [00:05, 927.95 examples/s]5447 examples [00:05, 913.06 examples/s]5546 examples [00:05, 933.26 examples/s]5640 examples [00:06, 914.84 examples/s]5740 examples [00:06, 938.33 examples/s]5835 examples [00:06, 927.28 examples/s]5929 examples [00:06, 930.40 examples/s]6026 examples [00:06, 940.55 examples/s]6124 examples [00:06, 949.62 examples/s]6221 examples [00:06, 953.19 examples/s]6317 examples [00:06, 952.13 examples/s]6413 examples [00:06, 930.94 examples/s]6507 examples [00:06, 927.74 examples/s]6602 examples [00:07, 932.57 examples/s]6699 examples [00:07, 940.91 examples/s]6796 examples [00:07, 949.35 examples/s]6892 examples [00:07, 941.81 examples/s]6987 examples [00:07, 939.38 examples/s]7081 examples [00:07, 939.48 examples/s]7175 examples [00:07, 927.46 examples/s]7272 examples [00:07, 938.43 examples/s]7366 examples [00:07, 933.57 examples/s]7463 examples [00:07, 919.69 examples/s]7559 examples [00:08, 929.70 examples/s]7656 examples [00:08, 941.17 examples/s]7751 examples [00:08, 942.02 examples/s]7846 examples [00:08, 940.81 examples/s]7941 examples [00:08, 937.24 examples/s]8035 examples [00:08, 935.50 examples/s]8129 examples [00:08, 935.18 examples/s]8229 examples [00:08, 952.94 examples/s]8328 examples [00:08, 962.74 examples/s]8428 examples [00:08, 969.60 examples/s]8526 examples [00:09, 921.88 examples/s]8619 examples [00:09, 881.94 examples/s]8717 examples [00:09, 906.77 examples/s]8811 examples [00:09, 915.13 examples/s]8910 examples [00:09, 934.95 examples/s]9004 examples [00:09, 924.61 examples/s]9104 examples [00:09, 944.07 examples/s]9204 examples [00:09, 959.10 examples/s]9301 examples [00:09, 959.89 examples/s]9400 examples [00:09, 968.23 examples/s]9500 examples [00:10, 976.56 examples/s]9599 examples [00:10, 980.30 examples/s]9700 examples [00:10, 986.76 examples/s]9799 examples [00:10, 985.20 examples/s]9900 examples [00:10, 989.96 examples/s]10000 examples [00:10, 969.02 examples/s]10098 examples [00:10, 924.19 examples/s]10197 examples [00:10, 942.13 examples/s]10294 examples [00:10, 949.79 examples/s]10391 examples [00:11, 954.56 examples/s]10489 examples [00:11, 960.04 examples/s]10588 examples [00:11, 967.23 examples/s]10685 examples [00:11, 967.53 examples/s]10784 examples [00:11, 972.36 examples/s]10884 examples [00:11, 979.76 examples/s]10983 examples [00:11, 982.25 examples/s]11082 examples [00:11, 980.65 examples/s]11181 examples [00:11, 971.20 examples/s]11280 examples [00:11, 974.20 examples/s]11379 examples [00:12, 977.11 examples/s]11477 examples [00:12, 957.12 examples/s]11577 examples [00:12, 968.81 examples/s]11677 examples [00:12, 977.11 examples/s]11776 examples [00:12, 980.40 examples/s]11877 examples [00:12, 987.16 examples/s]11976 examples [00:12, 981.76 examples/s]12075 examples [00:12, 974.99 examples/s]12173 examples [00:12, 973.33 examples/s]12271 examples [00:12, 956.54 examples/s]12369 examples [00:13, 960.08 examples/s]12469 examples [00:13, 970.32 examples/s]12569 examples [00:13, 977.69 examples/s]12667 examples [00:13, 963.63 examples/s]12764 examples [00:13, 958.17 examples/s]12865 examples [00:13, 972.15 examples/s]12964 examples [00:13, 976.33 examples/s]13062 examples [00:13, 961.82 examples/s]13159 examples [00:13, 957.75 examples/s]13255 examples [00:13, 943.45 examples/s]13351 examples [00:14, 947.13 examples/s]13449 examples [00:14, 955.73 examples/s]13545 examples [00:14, 948.93 examples/s]13640 examples [00:14, 944.53 examples/s]13735 examples [00:14, 944.69 examples/s]13830 examples [00:14, 941.10 examples/s]13927 examples [00:14, 941.18 examples/s]14028 examples [00:14, 959.50 examples/s]14128 examples [00:14, 970.94 examples/s]14226 examples [00:15, 952.72 examples/s]14324 examples [00:15, 960.42 examples/s]14421 examples [00:15, 932.94 examples/s]14521 examples [00:15, 950.48 examples/s]14617 examples [00:15, 923.04 examples/s]14710 examples [00:15, 921.86 examples/s]14807 examples [00:15, 934.16 examples/s]14901 examples [00:15, 926.09 examples/s]15001 examples [00:15, 945.34 examples/s]15097 examples [00:15, 949.66 examples/s]15195 examples [00:16, 956.31 examples/s]15291 examples [00:16, 950.11 examples/s]15387 examples [00:16, 949.80 examples/s]15488 examples [00:16, 964.74 examples/s]15587 examples [00:16, 970.16 examples/s]15685 examples [00:16, 970.73 examples/s]15783 examples [00:16, 954.70 examples/s]15881 examples [00:16, 961.99 examples/s]15978 examples [00:16, 948.30 examples/s]16073 examples [00:16, 938.92 examples/s]16169 examples [00:17, 943.44 examples/s]16264 examples [00:17, 942.44 examples/s]16359 examples [00:17, 930.92 examples/s]16453 examples [00:17, 916.80 examples/s]16546 examples [00:17, 918.22 examples/s]16643 examples [00:17, 931.99 examples/s]16737 examples [00:17, 915.22 examples/s]16835 examples [00:17, 932.35 examples/s]16930 examples [00:17, 934.90 examples/s]17027 examples [00:17, 944.75 examples/s]17122 examples [00:18, 945.73 examples/s]17217 examples [00:18, 940.62 examples/s]17312 examples [00:18, 913.15 examples/s]17404 examples [00:18, 910.10 examples/s]17496 examples [00:18, 904.06 examples/s]17587 examples [00:18, 901.58 examples/s]17682 examples [00:18, 913.10 examples/s]17774 examples [00:18, 914.07 examples/s]17867 examples [00:18, 916.21 examples/s]17963 examples [00:19, 928.22 examples/s]18056 examples [00:19, 925.99 examples/s]18152 examples [00:19, 935.90 examples/s]18246 examples [00:19, 919.58 examples/s]18340 examples [00:19, 923.29 examples/s]18438 examples [00:19, 937.89 examples/s]18533 examples [00:19, 939.69 examples/s]18628 examples [00:19, 927.37 examples/s]18723 examples [00:19, 932.64 examples/s]18817 examples [00:19, 924.17 examples/s]18910 examples [00:20, 917.73 examples/s]19004 examples [00:20, 923.63 examples/s]19103 examples [00:20, 940.17 examples/s]19198 examples [00:20, 939.56 examples/s]19293 examples [00:20, 933.23 examples/s]19387 examples [00:20, 923.39 examples/s]19480 examples [00:20, 911.56 examples/s]19572 examples [00:20, 912.70 examples/s]19670 examples [00:20, 930.28 examples/s]19767 examples [00:20, 940.73 examples/s]19864 examples [00:21, 949.06 examples/s]19965 examples [00:21, 964.52 examples/s]20062 examples [00:21, 920.03 examples/s]20155 examples [00:21, 905.30 examples/s]20255 examples [00:21, 928.84 examples/s]20349 examples [00:21, 927.93 examples/s]20443 examples [00:21, 888.52 examples/s]20536 examples [00:21, 899.50 examples/s]20628 examples [00:21, 902.25 examples/s]20721 examples [00:21, 909.33 examples/s]20813 examples [00:22, 907.44 examples/s]20907 examples [00:22, 915.62 examples/s]21001 examples [00:22, 922.73 examples/s]21098 examples [00:22, 933.67 examples/s]21194 examples [00:22, 940.53 examples/s]21289 examples [00:22, 935.55 examples/s]21386 examples [00:22, 943.74 examples/s]21482 examples [00:22, 945.62 examples/s]21577 examples [00:22, 939.55 examples/s]21673 examples [00:22, 944.59 examples/s]21768 examples [00:23, 937.63 examples/s]21865 examples [00:23, 946.14 examples/s]21960 examples [00:23, 944.35 examples/s]22055 examples [00:23, 935.12 examples/s]22152 examples [00:23, 944.26 examples/s]22247 examples [00:23, 939.18 examples/s]22347 examples [00:23, 954.93 examples/s]22443 examples [00:23, 943.05 examples/s]22543 examples [00:23, 958.52 examples/s]22641 examples [00:24, 963.95 examples/s]22738 examples [00:24, 947.83 examples/s]22837 examples [00:24, 957.79 examples/s]22933 examples [00:24, 937.72 examples/s]23027 examples [00:24, 913.23 examples/s]23126 examples [00:24, 932.32 examples/s]23220 examples [00:24, 911.32 examples/s]23318 examples [00:24, 928.36 examples/s]23416 examples [00:24, 942.35 examples/s]23516 examples [00:24, 958.00 examples/s]23615 examples [00:25, 967.07 examples/s]23712 examples [00:25, 967.92 examples/s]23809 examples [00:25, 965.60 examples/s]23908 examples [00:25, 970.66 examples/s]24008 examples [00:25, 978.07 examples/s]24108 examples [00:25, 984.39 examples/s]24207 examples [00:25, 975.47 examples/s]24305 examples [00:25, 967.79 examples/s]24403 examples [00:25, 969.79 examples/s]24502 examples [00:25, 975.57 examples/s]24600 examples [00:26, 923.99 examples/s]24693 examples [00:26, 921.43 examples/s]24791 examples [00:26, 938.09 examples/s]24886 examples [00:26, 940.81 examples/s]24986 examples [00:26, 955.85 examples/s]25084 examples [00:26, 961.50 examples/s]25181 examples [00:26, 958.91 examples/s]25278 examples [00:26, 942.99 examples/s]25373 examples [00:26, 943.69 examples/s]25471 examples [00:26, 952.45 examples/s]25569 examples [00:27, 957.81 examples/s]25665 examples [00:27, 951.36 examples/s]25764 examples [00:27, 960.24 examples/s]25863 examples [00:27, 967.54 examples/s]25960 examples [00:27, 944.75 examples/s]26059 examples [00:27, 955.97 examples/s]26155 examples [00:27, 927.47 examples/s]26251 examples [00:27, 933.35 examples/s]26347 examples [00:27, 937.93 examples/s]26441 examples [00:28, 937.64 examples/s]26538 examples [00:28, 944.92 examples/s]26633 examples [00:28, 942.79 examples/s]26732 examples [00:28, 953.57 examples/s]26831 examples [00:28, 962.56 examples/s]26930 examples [00:28, 969.62 examples/s]27028 examples [00:28, 952.73 examples/s]27124 examples [00:28, 947.23 examples/s]27223 examples [00:28, 959.08 examples/s]27323 examples [00:28, 969.61 examples/s]27424 examples [00:29, 980.09 examples/s]27523 examples [00:29, 977.71 examples/s]27621 examples [00:29, 964.61 examples/s]27718 examples [00:29, 963.28 examples/s]27815 examples [00:29, 959.92 examples/s]27912 examples [00:29, 949.53 examples/s]28011 examples [00:29, 958.63 examples/s]28107 examples [00:29, 958.71 examples/s]28203 examples [00:29, 948.37 examples/s]28300 examples [00:29, 954.06 examples/s]28399 examples [00:30, 963.56 examples/s]28498 examples [00:30, 971.14 examples/s]28597 examples [00:30, 974.30 examples/s]28696 examples [00:30, 976.55 examples/s]28794 examples [00:30, 972.08 examples/s]28892 examples [00:30, 963.78 examples/s]28992 examples [00:30, 972.19 examples/s]29090 examples [00:30, 970.93 examples/s]29189 examples [00:30, 975.66 examples/s]29289 examples [00:30, 982.16 examples/s]29388 examples [00:31, 978.93 examples/s]29486 examples [00:31, 978.12 examples/s]29585 examples [00:31, 979.21 examples/s]29686 examples [00:31, 986.47 examples/s]29785 examples [00:31, 975.01 examples/s]29883 examples [00:31, 969.60 examples/s]29980 examples [00:31, 952.73 examples/s]30076 examples [00:31, 912.91 examples/s]30170 examples [00:31, 919.30 examples/s]30263 examples [00:32, 901.36 examples/s]30354 examples [00:32, 901.16 examples/s]30449 examples [00:32, 912.74 examples/s]30544 examples [00:32, 922.13 examples/s]30641 examples [00:32, 935.84 examples/s]30735 examples [00:32, 935.31 examples/s]30834 examples [00:32, 949.84 examples/s]30930 examples [00:32, 926.57 examples/s]31023 examples [00:32, 923.01 examples/s]31119 examples [00:32, 932.71 examples/s]31214 examples [00:33, 937.17 examples/s]31308 examples [00:33, 929.64 examples/s]31402 examples [00:33, 886.35 examples/s]31493 examples [00:33, 892.85 examples/s]31584 examples [00:33, 897.05 examples/s]31674 examples [00:33, 891.11 examples/s]31764 examples [00:33, 888.02 examples/s]31853 examples [00:33, 885.96 examples/s]31953 examples [00:33, 915.87 examples/s]32047 examples [00:33, 920.64 examples/s]32146 examples [00:34, 939.40 examples/s]32244 examples [00:34, 951.00 examples/s]32341 examples [00:34, 954.65 examples/s]32440 examples [00:34, 962.94 examples/s]32537 examples [00:34, 963.06 examples/s]32637 examples [00:34, 973.24 examples/s]32736 examples [00:34, 977.71 examples/s]32834 examples [00:34, 972.60 examples/s]32932 examples [00:34, 923.84 examples/s]33025 examples [00:34, 925.43 examples/s]33119 examples [00:35, 928.55 examples/s]33213 examples [00:35, 922.38 examples/s]33306 examples [00:35, 924.55 examples/s]33404 examples [00:35, 939.15 examples/s]33501 examples [00:35, 945.70 examples/s]33601 examples [00:35, 960.64 examples/s]33699 examples [00:35, 965.87 examples/s]33796 examples [00:35, 930.09 examples/s]33895 examples [00:35, 944.93 examples/s]33994 examples [00:35, 956.54 examples/s]34092 examples [00:36, 963.04 examples/s]34190 examples [00:36, 967.93 examples/s]34289 examples [00:36, 972.28 examples/s]34388 examples [00:36, 977.50 examples/s]34487 examples [00:36, 980.05 examples/s]34586 examples [00:36, 934.66 examples/s]34684 examples [00:36, 946.63 examples/s]34781 examples [00:36, 951.03 examples/s]34880 examples [00:36, 961.26 examples/s]34980 examples [00:37, 972.49 examples/s]35080 examples [00:37, 978.19 examples/s]35179 examples [00:37, 980.56 examples/s]35278 examples [00:37, 947.67 examples/s]35374 examples [00:37, 933.87 examples/s]35469 examples [00:37, 936.86 examples/s]35566 examples [00:37, 943.92 examples/s]35663 examples [00:37, 949.96 examples/s]35759 examples [00:37, 948.70 examples/s]35859 examples [00:37, 962.71 examples/s]35958 examples [00:38, 969.57 examples/s]36056 examples [00:38, 968.58 examples/s]36153 examples [00:38, 957.78 examples/s]36252 examples [00:38, 964.83 examples/s]36352 examples [00:38, 972.36 examples/s]36452 examples [00:38, 979.00 examples/s]36553 examples [00:38, 986.88 examples/s]36654 examples [00:38, 991.11 examples/s]36754 examples [00:38, 988.42 examples/s]36853 examples [00:38, 985.69 examples/s]36952 examples [00:39, 979.86 examples/s]37051 examples [00:39, 975.57 examples/s]37149 examples [00:39, 960.47 examples/s]37246 examples [00:39, 957.15 examples/s]37342 examples [00:39, 957.67 examples/s]37442 examples [00:39, 967.99 examples/s]37539 examples [00:39, 954.70 examples/s]37635 examples [00:39, 953.41 examples/s]37731 examples [00:39, 941.56 examples/s]37830 examples [00:39, 954.79 examples/s]37927 examples [00:40, 957.42 examples/s]38027 examples [00:40, 969.45 examples/s]38125 examples [00:40, 969.99 examples/s]38223 examples [00:40, 971.76 examples/s]38324 examples [00:40, 980.27 examples/s]38424 examples [00:40, 985.07 examples/s]38523 examples [00:40, 982.09 examples/s]38623 examples [00:40, 985.49 examples/s]38722 examples [00:40, 979.73 examples/s]38822 examples [00:40, 983.81 examples/s]38921 examples [00:41, 982.61 examples/s]39022 examples [00:41, 988.98 examples/s]39122 examples [00:41, 989.67 examples/s]39221 examples [00:41, 957.16 examples/s]39318 examples [00:41, 958.87 examples/s]39415 examples [00:41, 958.59 examples/s]39514 examples [00:41, 967.11 examples/s]39613 examples [00:41, 972.47 examples/s]39711 examples [00:41, 957.64 examples/s]39807 examples [00:42, 951.09 examples/s]39908 examples [00:42, 965.34 examples/s]40005 examples [00:42, 908.57 examples/s]40104 examples [00:42, 931.47 examples/s]40198 examples [00:42, 929.46 examples/s]40298 examples [00:42, 948.06 examples/s]40396 examples [00:42, 957.41 examples/s]40493 examples [00:42, 942.04 examples/s]40588 examples [00:42, 940.86 examples/s]40683 examples [00:42, 924.13 examples/s]40778 examples [00:43, 931.34 examples/s]40874 examples [00:43, 937.38 examples/s]40968 examples [00:43, 938.16 examples/s]41064 examples [00:43, 943.22 examples/s]41159 examples [00:43, 937.45 examples/s]41254 examples [00:43, 938.95 examples/s]41350 examples [00:43, 945.00 examples/s]41450 examples [00:43, 958.43 examples/s]41549 examples [00:43, 965.37 examples/s]41646 examples [00:43, 961.61 examples/s]41745 examples [00:44, 967.41 examples/s]41842 examples [00:44, 957.73 examples/s]41938 examples [00:44, 956.52 examples/s]42034 examples [00:44, 956.40 examples/s]42130 examples [00:44, 938.02 examples/s]42224 examples [00:44, 937.83 examples/s]42318 examples [00:44, 934.10 examples/s]42417 examples [00:44, 947.68 examples/s]42512 examples [00:44, 938.11 examples/s]42610 examples [00:44, 948.38 examples/s]42708 examples [00:45, 955.41 examples/s]42807 examples [00:45, 963.36 examples/s]42907 examples [00:45, 971.90 examples/s]43007 examples [00:45, 978.40 examples/s]43105 examples [00:45, 967.49 examples/s]43202 examples [00:45, 961.24 examples/s]43300 examples [00:45, 964.87 examples/s]43397 examples [00:45, 963.46 examples/s]43494 examples [00:45, 935.44 examples/s]43588 examples [00:45, 935.64 examples/s]43682 examples [00:46, 926.11 examples/s]43780 examples [00:46, 939.12 examples/s]43879 examples [00:46, 951.90 examples/s]43976 examples [00:46, 955.92 examples/s]44072 examples [00:46, 955.21 examples/s]44172 examples [00:46, 967.28 examples/s]44272 examples [00:46, 975.67 examples/s]44372 examples [00:46, 980.28 examples/s]44473 examples [00:46, 986.90 examples/s]44572 examples [00:47, 977.28 examples/s]44673 examples [00:47, 984.11 examples/s]44772 examples [00:47, 984.52 examples/s]44871 examples [00:47, 975.77 examples/s]44970 examples [00:47, 979.30 examples/s]45068 examples [00:47, 976.02 examples/s]45166 examples [00:47, 976.99 examples/s]45266 examples [00:47, 982.80 examples/s]45365 examples [00:47, 975.82 examples/s]45463 examples [00:47, 975.13 examples/s]45561 examples [00:48, 968.22 examples/s]45659 examples [00:48, 970.30 examples/s]45757 examples [00:48, 966.35 examples/s]45858 examples [00:48, 977.42 examples/s]45958 examples [00:48, 984.02 examples/s]46057 examples [00:48, 969.73 examples/s]46158 examples [00:48, 980.69 examples/s]46257 examples [00:48, 930.84 examples/s]46352 examples [00:48, 936.03 examples/s]46447 examples [00:48, 904.51 examples/s]46543 examples [00:49, 919.37 examples/s]46642 examples [00:49, 938.84 examples/s]46741 examples [00:49, 950.98 examples/s]46837 examples [00:49, 948.55 examples/s]46933 examples [00:49, 936.14 examples/s]47027 examples [00:49, 933.78 examples/s]47123 examples [00:49, 940.34 examples/s]47218 examples [00:49, 942.56 examples/s]47313 examples [00:49, 926.67 examples/s]47410 examples [00:49, 937.52 examples/s]47508 examples [00:50, 948.66 examples/s]47609 examples [00:50, 964.59 examples/s]47709 examples [00:50, 973.76 examples/s]47807 examples [00:50, 974.84 examples/s]47907 examples [00:50, 981.88 examples/s]48006 examples [00:50, 977.19 examples/s]48106 examples [00:50, 983.09 examples/s]48205 examples [00:50, 984.66 examples/s]48304 examples [00:50, 970.31 examples/s]48404 examples [00:50, 977.02 examples/s]48502 examples [00:51, 972.91 examples/s]48602 examples [00:51, 978.55 examples/s]48700 examples [00:51, 978.80 examples/s]48798 examples [00:51, 954.38 examples/s]48894 examples [00:51, 955.14 examples/s]48990 examples [00:51, 948.29 examples/s]49089 examples [00:51, 960.19 examples/s]49186 examples [00:51, 949.94 examples/s]49285 examples [00:51, 960.55 examples/s]49382 examples [00:52, 949.73 examples/s]49478 examples [00:52, 944.67 examples/s]49576 examples [00:52, 954.87 examples/s]49676 examples [00:52, 966.96 examples/s]49775 examples [00:52, 973.66 examples/s]49873 examples [00:52, 971.35 examples/s]49971 examples [00:52, 959.11 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6812/50000 [00:00<00:00, 68115.36 examples/s] 38%|███▊      | 18776/50000 [00:00<00:00, 78221.27 examples/s] 61%|██████▏   | 30647/50000 [00:00<00:00, 87135.78 examples/s] 85%|████████▍ | 42397/50000 [00:00<00:00, 94457.53 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 853.50 examples/s]187 examples [00:00, 894.57 examples/s]287 examples [00:00, 921.98 examples/s]387 examples [00:00, 942.45 examples/s]484 examples [00:00, 950.31 examples/s]585 examples [00:00, 965.31 examples/s]687 examples [00:00, 980.36 examples/s]780 examples [00:00, 964.24 examples/s]881 examples [00:00, 977.00 examples/s]976 examples [00:01, 949.68 examples/s]1075 examples [00:01, 958.80 examples/s]1174 examples [00:01, 961.97 examples/s]1270 examples [00:01, 943.13 examples/s]1364 examples [00:01, 938.44 examples/s]1458 examples [00:01, 738.28 examples/s]1557 examples [00:01, 798.61 examples/s]1657 examples [00:01, 849.82 examples/s]1747 examples [00:01, 858.54 examples/s]1843 examples [00:02, 884.40 examples/s]1943 examples [00:02, 914.75 examples/s]2044 examples [00:02, 940.05 examples/s]2144 examples [00:02, 956.70 examples/s]2244 examples [00:02, 966.72 examples/s]2342 examples [00:02, 966.16 examples/s]2442 examples [00:02, 975.24 examples/s]2540 examples [00:02, 967.95 examples/s]2638 examples [00:02, 967.98 examples/s]2736 examples [00:02, 954.57 examples/s]2832 examples [00:03, 945.42 examples/s]2927 examples [00:03, 925.79 examples/s]3020 examples [00:03, 921.94 examples/s]3114 examples [00:03, 925.18 examples/s]3214 examples [00:03, 944.79 examples/s]3312 examples [00:03, 955.02 examples/s]3412 examples [00:03, 966.03 examples/s]3512 examples [00:03, 975.02 examples/s]3613 examples [00:03, 983.68 examples/s]3714 examples [00:03, 989.02 examples/s]3813 examples [00:04, 985.41 examples/s]3912 examples [00:04, 982.75 examples/s]4012 examples [00:04, 986.49 examples/s]4112 examples [00:04, 989.63 examples/s]4211 examples [00:04, 985.21 examples/s]4310 examples [00:04, 951.51 examples/s]4410 examples [00:04, 963.32 examples/s]4510 examples [00:04, 971.52 examples/s]4609 examples [00:04, 975.47 examples/s]4707 examples [00:04, 939.01 examples/s]4802 examples [00:05, 935.44 examples/s]4898 examples [00:05, 941.62 examples/s]4995 examples [00:05, 948.12 examples/s]5092 examples [00:05, 953.23 examples/s]5188 examples [00:05, 947.83 examples/s]5283 examples [00:05, 947.13 examples/s]5381 examples [00:05, 954.26 examples/s]5479 examples [00:05, 959.17 examples/s]5577 examples [00:05, 964.28 examples/s]5674 examples [00:05, 959.98 examples/s]5771 examples [00:06, 957.93 examples/s]5870 examples [00:06, 964.73 examples/s]5970 examples [00:06, 974.42 examples/s]6068 examples [00:06, 958.41 examples/s]6164 examples [00:06, 912.35 examples/s]6256 examples [00:06, 914.47 examples/s]6348 examples [00:06, 908.47 examples/s]6445 examples [00:06, 925.28 examples/s]6542 examples [00:06, 936.51 examples/s]6637 examples [00:07, 939.47 examples/s]6737 examples [00:07, 955.17 examples/s]6837 examples [00:07, 967.62 examples/s]6938 examples [00:07, 978.30 examples/s]7037 examples [00:07, 981.53 examples/s]7136 examples [00:07, 959.80 examples/s]7233 examples [00:07, 958.18 examples/s]7330 examples [00:07, 961.49 examples/s]7428 examples [00:07, 964.08 examples/s]7525 examples [00:07, 953.54 examples/s]7621 examples [00:08, 934.67 examples/s]7715 examples [00:08, 929.67 examples/s]7812 examples [00:08, 938.94 examples/s]7906 examples [00:08, 936.60 examples/s]8003 examples [00:08, 943.66 examples/s]8098 examples [00:08, 936.39 examples/s]8194 examples [00:08, 942.86 examples/s]8291 examples [00:08, 948.02 examples/s]8388 examples [00:08, 952.96 examples/s]8484 examples [00:08, 946.84 examples/s]8579 examples [00:09, 924.91 examples/s]8672 examples [00:09, 925.79 examples/s]8766 examples [00:09, 929.59 examples/s]8861 examples [00:09, 932.79 examples/s]8957 examples [00:09, 939.10 examples/s]9051 examples [00:09, 931.74 examples/s]9148 examples [00:09, 942.56 examples/s]9245 examples [00:09, 946.69 examples/s]9340 examples [00:09, 936.64 examples/s]9434 examples [00:09, 937.39 examples/s]9528 examples [00:10, 925.96 examples/s]9623 examples [00:10, 931.14 examples/s]9719 examples [00:10, 938.00 examples/s]9813 examples [00:10, 914.53 examples/s]9908 examples [00:10, 924.29 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRZ2KUR/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRZ2KUR/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe307a8f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe307a8f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe307a8f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe290f57fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe290f5eef0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe307a8f950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe307a8f950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe307a8f950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe2e5689080>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe2e56890f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe27f4b3268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe27f4b3268> 

  function with postional parmater data_info <function split_train_valid at 0x7fe27f4b3268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe27f4b3378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe27f4b3378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe27f4b3378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=694be6f8f47eedb15e0f9d35925cf0599c17fb33a2b0f676c517af13914612a3
  Stored in directory: /tmp/pip-ephem-wheel-cache-r6wavzoa/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:12:11, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:46:35, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:05:49, 21.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:46:26, 30.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.41M/862M [00:01<5:25:43, 43.9kB/s].vector_cache/glove.6B.zip:   1%|          | 6.76M/862M [00:01<3:47:16, 62.7kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:38:05, 89.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.1M/862M [00:01<1:50:19, 128kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:01<1:16:48, 182kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:01<53:30, 260kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:01<37:17, 371kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.9M/862M [00:02<26:08, 527kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.5M/862M [00:02<18:14, 750kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<12:51, 1.06MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:02<09:01, 1.50MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<06:52, 1.96MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<05:16, 2.55MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:04<11:05:23, 20.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<7:45:33, 28.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:06<5:27:12, 40.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<3:52:05, 57.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<2:42:57, 82.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:06<1:53:53, 117kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<1:27:11, 153kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<1:02:24, 213kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:08<43:54, 302kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<33:40, 393kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<26:23, 502kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<19:03, 694kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:10<13:28, 978kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<16:09, 815kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<12:38, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:12<09:07, 1.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<09:27, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<09:18, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:04, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:14<05:07, 2.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:16<10:00, 1.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<08:20, 1.56MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:16<06:06, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:18<07:19, 1.77MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<07:44, 1.67MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<06:20, 2.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<05:45, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<04:21, 2.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<06:03, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<06:43, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<05:15, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:22<03:49, 3.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:24<11:04, 1.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<09:03, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<06:36, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<07:36, 1.67MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:53, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:04, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:22, 2.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<15:55, 792kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<12:26, 1.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:57, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<09:13, 1.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<07:43, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:40, 2.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:56, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<07:23, 1.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:47, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:03, 2.05MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:53, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:21, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:55, 3.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<08:27, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<07:12, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:20, 2.30MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<06:35, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<07:05, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:29, 2.23MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:58, 3.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<12:33, 973kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<09:49, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<07:10, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:41<07:52, 1.54MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<07:59, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<06:12, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:27, 2.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<11:41:46, 17.2kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<8:12:14, 24.5kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<5:44:08, 35.0kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<4:01:04, 49.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<11:04:46, 18.1kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<7:45:41, 25.8kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:45<5:25:01, 36.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<3:53:46, 51.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<2:46:14, 71.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:56:47, 102kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:47<1:21:40, 146kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<1:02:07, 191kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<44:40, 266kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<31:29, 376kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<24:47, 476kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<19:44, 598kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<14:23, 819kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<11:56, 983kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<10:51, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:11, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<05:51, 1.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<1:27:37, 133kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<1:02:30, 187kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<43:54, 265kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<33:22, 348kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<25:48, 449kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<18:38, 621kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<13:08, 877kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<1:31:33, 126kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:05:13, 177kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<45:48, 251kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<34:40, 331kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<26:35, 431kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<19:10, 597kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<15:12, 749kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<12:56, 880kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<09:33, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<06:49, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<10:10, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<08:17, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:05, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:52, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:05, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:26, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:57, 2.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<08:11, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<06:54, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:03, 2.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:09, 1.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:02, 2.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:27, 2.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<04:56, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<03:43, 2.95MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:12, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:45, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<03:36, 3.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:06, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:46, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:35, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<04:57, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:23, 2.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<03:23, 3.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<02:31, 4.27MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<51:05, 211kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<37:58, 284kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<27:05, 397kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<20:37, 519kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<15:33, 688kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<11:07, 959kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<10:14, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<09:19, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:58, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:00, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<09:24, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<07:41, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<05:38, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:23, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:36, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:08, 2.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:02, 2.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<7:18:13, 23.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<5:07:03, 34.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<3:34:11, 48.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<2:36:22, 66.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<1:51:37, 92.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:18:30, 132kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<54:55, 188kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<42:46, 241kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<30:59, 332kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<21:52, 470kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<17:37, 581kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<13:23, 764kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<09:34, 1.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<09:03, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<08:25, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<06:21, 1.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<04:35, 2.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<07:20, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:10, 1.63MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:31, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:31, 1.81MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:55, 1.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:38, 2.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:51, 2.05MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<04:24, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:17, 3.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<04:38, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:09, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:02, 2.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<02:56, 3.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<08:26, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<06:55, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<05:05, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:50, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:04, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:44, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:52, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:23, 2.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:16, 2.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:33, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:07, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<03:59, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<02:54, 3.29MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<08:45, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<07:05, 1.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<05:11, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:50, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<06:00, 1.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:37, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:19, 2.83MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [01:59<10:23, 906kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<08:14, 1.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:59, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<06:22, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<06:21, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:50, 1.93MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:29, 2.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<07:11, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<05:58, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:22, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:12, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:30, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<03:03, 2.99MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<38:06, 240kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<27:36, 331kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<19:27, 468kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<15:42, 577kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<12:49, 707kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<09:21, 967kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:37, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<11:22, 791kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<08:53, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<06:24, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<05:04, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<6:17:04, 23.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<4:24:07, 33.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<3:04:02, 48.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<2:16:07, 65.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<1:37:07, 91.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<1:08:17, 130kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<47:41, 185kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<39:15, 224kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<28:22, 310kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<20:01, 438kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<16:00, 545kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<12:58, 673kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<09:30, 916kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<08:01, 1.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<06:30, 1.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<05:19, 1.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:27, 1.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:10, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<03:02, 2.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:14, 1.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:13, 1.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:49, 2.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:40, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:58, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:49, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:47, 3.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:56, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:01, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:40, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:31, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:50, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:48, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:59, 2.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:30, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<02:36, 3.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<01:57, 4.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<31:52, 257kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<23:07, 354kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<16:18, 500kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<13:18, 610kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<10:06, 802kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<07:14, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:57, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:30, 1.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:57, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<04:44, 1.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:07, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:04, 2.58MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:00, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:24, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:25, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:29, 3.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:07, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:46, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:13, 1.84MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:46, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:59, 1.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:49, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:45, 2.79MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:32, 1.18MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:22, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:56, 1.95MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:32, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:43, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:39, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:38, 2.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<56:26, 134kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<40:14, 188kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<28:14, 267kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<21:26, 350kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<16:31, 454kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<11:51, 631kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<08:21, 890kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<07:02, 1.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<5:10:27, 23.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<3:37:26, 34.1kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<2:31:25, 48.7kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<1:50:22, 66.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<1:18:43, 93.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<55:21, 133kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<39:40, 184kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<28:31, 256kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<20:03, 362kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<15:38, 462kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<12:24, 582kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<09:02, 798kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<06:21, 1.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<6:58:28, 17.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<4:53:23, 24.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<3:24:48, 34.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<2:24:18, 49.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:41:38, 69.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<1:11:03, 99.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<51:09, 137kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<37:12, 189kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<26:21, 266kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<19:26, 358kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<14:19, 485kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<10:09, 681kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<08:40, 793kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<07:27, 922kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:30, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<03:55, 1.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<06:59, 975kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:34, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:03, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:24, 1.53MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<05:09, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:52, 1.74MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:45, 1.78MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:18, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:28, 2.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:16, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:37, 1.82MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:51, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<02:03, 3.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<6:19:25, 17.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<4:26:00, 24.6kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<3:05:36, 35.1kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<2:10:44, 49.5kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<1:32:46, 69.8kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<1:05:08, 99.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<46:16, 138kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<33:40, 190kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<23:48, 268kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<16:38, 381kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<15:18, 414kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<11:20, 558kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:04, 781kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<07:05, 884kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<05:28, 1.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<04:03, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:54, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<43:08, 144kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<31:26, 197kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<22:13, 278kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<15:31, 396kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<15:23, 398kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<11:22, 538kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<08:05, 754kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<07:03, 859kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<06:09, 983kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:35, 1.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:14, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<20:53, 287kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<15:13, 394kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<10:43, 556kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<07:49, 757kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<4:24:24, 22.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<3:05:04, 32.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<2:08:37, 45.7kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<1:34:28, 62.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<1:07:20, 87.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<47:17, 124kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<32:59, 176kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<25:43, 225kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<18:35, 311kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<13:04, 440kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<10:26, 548kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<08:27, 676kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<06:10, 924kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<04:21, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<5:29:29, 17.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<3:50:58, 24.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<2:41:04, 34.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:53:20, 49.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:20:25, 69.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<56:26, 98.7kB/s]  .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<40:02, 138kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<28:33, 193kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<20:02, 274kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<15:13, 358kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<11:44, 464kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<08:26, 644kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<05:56, 908kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<07:07, 756kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:31, 972kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:59, 1.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:01, 1.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:53, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:56, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:06, 2.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<06:15, 838kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:49, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:32, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:31, 2.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<25:31, 203kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<18:57, 273kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<13:28, 383kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<09:26, 543kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<08:41, 588kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<06:37, 770kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:44, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:24, 1.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:37, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:39, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:57, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:06, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:23, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:44, 2.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:16, 1.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:49, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:05, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:31, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:17, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:43, 2.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:15, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:37, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:05, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:30, 3.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<07:56, 590kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<06:04, 772kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<04:21, 1.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:02, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:50, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:56, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<02:05, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<08:38, 527kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<06:31, 696kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<04:40, 968kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:14, 1.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:26, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:31, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:43, 1.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:53, 1.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:12, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:35, 2.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:01, 1.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:35, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:54, 2.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:16, 1.88MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:30, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:57, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:25, 2.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:38, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:13, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:39, 2.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:12, 3.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<05:01, 823kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<03:57, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:50, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:52, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:27, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:08, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:20, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:19, 2.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<06:50, 574kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<05:08, 763kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:43, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:38, 1.47MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<06:05, 633kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<04:40, 824kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:20, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:09, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:00, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:16, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:37, 2.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:46, 1.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:20, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:43, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:00, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:42, 2.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<01:13, 2.92MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:44, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:18, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:41, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:56, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:08, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:40, 2.09MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:11, 2.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<05:31, 622kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:14, 810kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<03:02, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:51, 1.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:02, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:27, 2.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:42, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:15, 1.46MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:39, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:51, 1.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:59, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:33, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:07, 2.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<05:37, 561kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<04:16, 738kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:03, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<02:48, 1.10MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:38, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:59, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:24, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:53, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:20, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:42, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:50, 1.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:36, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:10, 2.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:27, 1.98MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:39, 1.74MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:17, 2.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<00:55, 3.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:08, 1.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:48, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:19, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:31, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:38, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:16, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:11<00:55, 2.94MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:44, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:30, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:06, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:55, 2.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<1:50:55, 23.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<1:17:25, 33.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<53:13, 47.9kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<40:00, 63.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<28:32, 89.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<19:59, 126kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<13:50, 180kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<10:39, 232kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<07:42, 320kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<05:24, 452kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:15, 565kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:29, 689kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:33, 934kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<01:47, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<04:49, 483kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:37, 643kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:33, 900kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:16, 994kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:04, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:33, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:05, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<07:45, 283kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<05:38, 388kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<03:57, 547kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:12, 664kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:28, 859kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:45, 1.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:39, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:34, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:12, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:50, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<14:48, 134kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<10:32, 188kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<07:20, 267kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<05:28, 351kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<04:13, 453kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<03:02, 626kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<02:05, 886kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<10:23, 178kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<07:27, 248kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<05:11, 351kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<03:57, 450kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<03:07, 569kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:15, 780kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<01:33, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<13:20, 128kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<09:29, 180kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<06:35, 255kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<04:52, 337kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<03:34, 459kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<02:30, 644kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:04, 759kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:47, 873kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:20, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:44<00:55, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<03:02, 495kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:16, 657kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:36, 915kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:25, 1.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:17, 1.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:57, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:40, 2.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:21, 1.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:05, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:46, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:49, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:41, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:30, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:37, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:31, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:22, 3.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:56, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:47, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:34, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:37, 1.74MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:33, 1.95MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:24, 2.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:27, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:20, 2.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:30, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:17, 3.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:29, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:26, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:19, 2.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.81MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:20, 554kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:00, 730kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:41, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:36, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:33, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:16, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:31, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 1.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:18, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:19, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:15, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:10, 2.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:14, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:10, 2.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:12, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:06, 3.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:12, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:10, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 3.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.20MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 3.03MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:07, 422kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:04, 572kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 799kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<19:44:37,  5.63it/s]  0%|          | 772/400000 [00:00<13:47:54,  8.04it/s]  0%|          | 1533/400000 [00:00<9:38:41, 11.48it/s]  1%|          | 2295/400000 [00:00<6:44:34, 16.38it/s]  1%|          | 3080/400000 [00:00<4:42:53, 23.38it/s]  1%|          | 3835/400000 [00:00<3:17:54, 33.36it/s]  1%|          | 4604/400000 [00:00<2:18:31, 47.57it/s]  1%|▏         | 5354/400000 [00:00<1:37:02, 67.78it/s]  2%|▏         | 6133/400000 [00:00<1:08:03, 96.46it/s]  2%|▏         | 6908/400000 [00:01<47:47, 137.07it/s]   2%|▏         | 7647/400000 [00:01<33:39, 194.25it/s]  2%|▏         | 8383/400000 [00:01<23:48, 274.21it/s]  2%|▏         | 9107/400000 [00:01<16:55, 385.06it/s]  2%|▏         | 9822/400000 [00:01<12:05, 537.68it/s]  3%|▎         | 10568/400000 [00:01<08:42, 745.07it/s]  3%|▎         | 11290/400000 [00:01<06:21, 1018.95it/s]  3%|▎         | 12010/400000 [00:01<04:42, 1371.27it/s]  3%|▎         | 12727/400000 [00:01<03:34, 1802.65it/s]  3%|▎         | 13515/400000 [00:02<02:44, 2344.99it/s]  4%|▎         | 14284/400000 [00:02<02:10, 2962.62it/s]  4%|▍         | 15026/400000 [00:02<01:46, 3600.69it/s]  4%|▍         | 15812/400000 [00:02<01:29, 4299.59it/s]  4%|▍         | 16564/400000 [00:02<01:17, 4921.83it/s]  4%|▍         | 17344/400000 [00:02<01:09, 5533.20it/s]  5%|▍         | 18132/400000 [00:02<01:02, 6074.57it/s]  5%|▍         | 18905/400000 [00:02<00:58, 6490.71it/s]  5%|▍         | 19674/400000 [00:02<00:56, 6763.84it/s]  5%|▌         | 20446/400000 [00:02<00:54, 7024.72it/s]  5%|▌         | 21212/400000 [00:03<00:52, 7192.16it/s]  6%|▌         | 22001/400000 [00:03<00:51, 7387.55it/s]  6%|▌         | 22773/400000 [00:03<00:50, 7471.05it/s]  6%|▌         | 23558/400000 [00:03<00:49, 7579.63it/s]  6%|▌         | 24333/400000 [00:03<00:50, 7399.56it/s]  6%|▋         | 25108/400000 [00:03<00:50, 7416.86it/s]  6%|▋         | 25878/400000 [00:03<00:49, 7499.45it/s]  7%|▋         | 26635/400000 [00:03<00:49, 7488.42it/s]  7%|▋         | 27389/400000 [00:03<00:50, 7383.56it/s]  7%|▋         | 28146/400000 [00:03<00:50, 7436.85it/s]  7%|▋         | 28893/400000 [00:04<00:51, 7257.07it/s]  7%|▋         | 29622/400000 [00:04<00:51, 7194.65it/s]  8%|▊         | 30354/400000 [00:04<00:51, 7230.44it/s]  8%|▊         | 31136/400000 [00:04<00:49, 7397.08it/s]  8%|▊         | 31920/400000 [00:04<00:48, 7520.91it/s]  8%|▊         | 32679/400000 [00:04<00:48, 7541.23it/s]  8%|▊         | 33463/400000 [00:04<00:48, 7627.52it/s]  9%|▊         | 34227/400000 [00:04<00:48, 7562.95it/s]  9%|▉         | 35003/400000 [00:04<00:47, 7620.35it/s]  9%|▉         | 35770/400000 [00:04<00:47, 7634.08it/s]  9%|▉         | 36534/400000 [00:05<00:47, 7594.57it/s]  9%|▉         | 37318/400000 [00:05<00:47, 7664.57it/s] 10%|▉         | 38085/400000 [00:05<00:48, 7475.72it/s] 10%|▉         | 38845/400000 [00:05<00:48, 7509.79it/s] 10%|▉         | 39631/400000 [00:05<00:47, 7608.77it/s] 10%|█         | 40393/400000 [00:05<00:47, 7597.43it/s] 10%|█         | 41175/400000 [00:05<00:46, 7661.96it/s] 10%|█         | 41942/400000 [00:05<00:46, 7645.67it/s] 11%|█         | 42708/400000 [00:05<00:46, 7629.08it/s] 11%|█         | 43491/400000 [00:05<00:46, 7687.90it/s] 11%|█         | 44261/400000 [00:06<00:46, 7585.93it/s] 11%|█▏        | 45021/400000 [00:06<00:46, 7581.47it/s] 11%|█▏        | 45801/400000 [00:06<00:46, 7645.46it/s] 12%|█▏        | 46584/400000 [00:06<00:45, 7698.26it/s] 12%|█▏        | 47355/400000 [00:06<00:46, 7662.85it/s] 12%|█▏        | 48124/400000 [00:06<00:45, 7669.27it/s] 12%|█▏        | 48905/400000 [00:06<00:45, 7709.99it/s] 12%|█▏        | 49688/400000 [00:06<00:45, 7744.69it/s] 13%|█▎        | 50473/400000 [00:06<00:44, 7773.24it/s] 13%|█▎        | 51251/400000 [00:06<00:45, 7739.05it/s] 13%|█▎        | 52026/400000 [00:07<00:45, 7701.04it/s] 13%|█▎        | 52809/400000 [00:07<00:44, 7738.36it/s] 13%|█▎        | 53596/400000 [00:07<00:44, 7774.71it/s] 14%|█▎        | 54381/400000 [00:07<00:44, 7794.27it/s] 14%|█▍        | 55161/400000 [00:07<00:44, 7752.32it/s] 14%|█▍        | 55937/400000 [00:07<00:45, 7622.88it/s] 14%|█▍        | 56700/400000 [00:07<00:45, 7603.34it/s] 14%|█▍        | 57473/400000 [00:07<00:44, 7638.37it/s] 15%|█▍        | 58255/400000 [00:07<00:44, 7691.85it/s] 15%|█▍        | 59025/400000 [00:07<00:45, 7531.12it/s] 15%|█▍        | 59780/400000 [00:08<00:46, 7325.47it/s] 15%|█▌        | 60550/400000 [00:08<00:45, 7433.68it/s] 15%|█▌        | 61332/400000 [00:08<00:44, 7544.86it/s] 16%|█▌        | 62111/400000 [00:08<00:44, 7615.75it/s] 16%|█▌        | 62893/400000 [00:08<00:43, 7674.09it/s] 16%|█▌        | 63665/400000 [00:08<00:43, 7686.29it/s] 16%|█▌        | 64435/400000 [00:08<00:46, 7277.01it/s] 16%|█▋        | 65168/400000 [00:08<00:46, 7203.86it/s] 16%|█▋        | 65933/400000 [00:08<00:45, 7331.37it/s] 17%|█▋        | 66692/400000 [00:09<00:45, 7405.31it/s] 17%|█▋        | 67462/400000 [00:09<00:44, 7490.68it/s] 17%|█▋        | 68241/400000 [00:09<00:43, 7576.96it/s] 17%|█▋        | 69001/400000 [00:09<00:43, 7528.56it/s] 17%|█▋        | 69772/400000 [00:09<00:43, 7581.15it/s] 18%|█▊        | 70554/400000 [00:09<00:43, 7648.16it/s] 18%|█▊        | 71324/400000 [00:09<00:42, 7661.21it/s] 18%|█▊        | 72091/400000 [00:09<00:42, 7657.79it/s] 18%|█▊        | 72858/400000 [00:09<00:43, 7484.21it/s] 18%|█▊        | 73608/400000 [00:09<00:44, 7258.09it/s] 19%|█▊        | 74337/400000 [00:10<00:45, 7115.50it/s] 19%|█▉        | 75099/400000 [00:10<00:44, 7259.14it/s] 19%|█▉        | 75859/400000 [00:10<00:44, 7356.86it/s] 19%|█▉        | 76597/400000 [00:10<00:44, 7303.13it/s] 19%|█▉        | 77329/400000 [00:10<00:44, 7181.81it/s] 20%|█▉        | 78049/400000 [00:10<00:45, 7128.52it/s] 20%|█▉        | 78768/400000 [00:10<00:44, 7145.10it/s] 20%|█▉        | 79549/400000 [00:10<00:43, 7331.84it/s] 20%|██        | 80332/400000 [00:10<00:42, 7474.38it/s] 20%|██        | 81116/400000 [00:10<00:42, 7578.21it/s] 20%|██        | 81876/400000 [00:11<00:41, 7574.82it/s] 21%|██        | 82641/400000 [00:11<00:41, 7594.55it/s] 21%|██        | 83416/400000 [00:11<00:41, 7638.91it/s] 21%|██        | 84197/400000 [00:11<00:41, 7688.82it/s] 21%|██        | 84982/400000 [00:11<00:40, 7733.76it/s] 21%|██▏       | 85766/400000 [00:11<00:40, 7763.54it/s] 22%|██▏       | 86543/400000 [00:11<00:40, 7649.37it/s] 22%|██▏       | 87326/400000 [00:11<00:40, 7699.92it/s] 22%|██▏       | 88110/400000 [00:11<00:40, 7739.89it/s] 22%|██▏       | 88886/400000 [00:11<00:40, 7744.25it/s] 22%|██▏       | 89665/400000 [00:12<00:40, 7756.45it/s] 23%|██▎       | 90441/400000 [00:12<00:40, 7597.02it/s] 23%|██▎       | 91207/400000 [00:12<00:40, 7613.28it/s] 23%|██▎       | 91985/400000 [00:12<00:40, 7661.38it/s] 23%|██▎       | 92759/400000 [00:12<00:39, 7682.63it/s] 23%|██▎       | 93542/400000 [00:12<00:39, 7725.77it/s] 24%|██▎       | 94315/400000 [00:12<00:39, 7672.43it/s] 24%|██▍       | 95083/400000 [00:12<00:41, 7307.02it/s] 24%|██▍       | 95818/400000 [00:12<00:42, 7239.01it/s] 24%|██▍       | 96545/400000 [00:12<00:42, 7202.25it/s] 24%|██▍       | 97268/400000 [00:13<00:42, 7104.17it/s] 25%|██▍       | 98023/400000 [00:13<00:41, 7229.69it/s] 25%|██▍       | 98772/400000 [00:13<00:41, 7303.15it/s] 25%|██▍       | 99504/400000 [00:13<00:41, 7177.26it/s] 25%|██▌       | 100267/400000 [00:13<00:41, 7306.61it/s] 25%|██▌       | 101060/400000 [00:13<00:39, 7481.86it/s] 25%|██▌       | 101833/400000 [00:13<00:39, 7554.52it/s] 26%|██▌       | 102610/400000 [00:13<00:39, 7617.22it/s] 26%|██▌       | 103374/400000 [00:13<00:39, 7476.23it/s] 26%|██▌       | 104163/400000 [00:14<00:38, 7595.39it/s] 26%|██▌       | 104951/400000 [00:14<00:38, 7676.82it/s] 26%|██▋       | 105720/400000 [00:14<00:38, 7672.21it/s] 27%|██▋       | 106495/400000 [00:14<00:38, 7693.13it/s] 27%|██▋       | 107280/400000 [00:14<00:37, 7738.76it/s] 27%|██▋       | 108055/400000 [00:14<00:37, 7700.47it/s] 27%|██▋       | 108826/400000 [00:14<00:38, 7634.06it/s] 27%|██▋       | 109590/400000 [00:14<00:38, 7460.21it/s] 28%|██▊       | 110338/400000 [00:14<00:39, 7311.45it/s] 28%|██▊       | 111071/400000 [00:14<00:41, 6999.31it/s] 28%|██▊       | 111775/400000 [00:15<00:41, 6936.95it/s] 28%|██▊       | 112472/400000 [00:15<00:42, 6708.13it/s] 28%|██▊       | 113181/400000 [00:15<00:42, 6816.11it/s] 28%|██▊       | 113964/400000 [00:15<00:40, 7090.24it/s] 29%|██▊       | 114749/400000 [00:15<00:39, 7299.85it/s] 29%|██▉       | 115535/400000 [00:15<00:38, 7457.45it/s] 29%|██▉       | 116297/400000 [00:15<00:37, 7505.44it/s] 29%|██▉       | 117082/400000 [00:15<00:37, 7604.09it/s] 29%|██▉       | 117865/400000 [00:15<00:36, 7669.36it/s] 30%|██▉       | 118635/400000 [00:15<00:36, 7677.58it/s] 30%|██▉       | 119405/400000 [00:16<00:36, 7588.89it/s] 30%|███       | 120166/400000 [00:16<00:37, 7418.15it/s] 30%|███       | 120928/400000 [00:16<00:37, 7475.78it/s] 30%|███       | 121704/400000 [00:16<00:36, 7556.65it/s] 31%|███       | 122490/400000 [00:16<00:36, 7644.61it/s] 31%|███       | 123256/400000 [00:16<00:36, 7635.47it/s] 31%|███       | 124021/400000 [00:16<00:36, 7560.37it/s] 31%|███       | 124794/400000 [00:16<00:36, 7608.68it/s] 31%|███▏      | 125571/400000 [00:16<00:35, 7654.94it/s] 32%|███▏      | 126353/400000 [00:16<00:35, 7702.89it/s] 32%|███▏      | 127135/400000 [00:17<00:35, 7736.79it/s] 32%|███▏      | 127914/400000 [00:17<00:35, 7751.86it/s] 32%|███▏      | 128690/400000 [00:17<00:35, 7732.55it/s] 32%|███▏      | 129464/400000 [00:17<00:35, 7717.89it/s] 33%|███▎      | 130239/400000 [00:17<00:34, 7727.30it/s] 33%|███▎      | 131026/400000 [00:17<00:34, 7767.60it/s] 33%|███▎      | 131810/400000 [00:17<00:34, 7788.31it/s] 33%|███▎      | 132589/400000 [00:17<00:34, 7766.01it/s] 33%|███▎      | 133372/400000 [00:17<00:34, 7781.96it/s] 34%|███▎      | 134151/400000 [00:17<00:34, 7752.03it/s] 34%|███▎      | 134927/400000 [00:18<00:34, 7736.95it/s] 34%|███▍      | 135701/400000 [00:18<00:35, 7425.79it/s] 34%|███▍      | 136447/400000 [00:18<00:36, 7130.90it/s] 34%|███▍      | 137170/400000 [00:18<00:36, 7157.92it/s] 34%|███▍      | 137889/400000 [00:18<00:37, 7064.04it/s] 35%|███▍      | 138608/400000 [00:18<00:36, 7098.53it/s] 35%|███▍      | 139328/400000 [00:18<00:36, 7126.20it/s] 35%|███▌      | 140072/400000 [00:18<00:36, 7216.27it/s] 35%|███▌      | 140849/400000 [00:18<00:35, 7371.55it/s] 35%|███▌      | 141604/400000 [00:19<00:34, 7421.72it/s] 36%|███▌      | 142348/400000 [00:19<00:35, 7220.37it/s] 36%|███▌      | 143073/400000 [00:19<00:36, 7043.64it/s] 36%|███▌      | 143843/400000 [00:19<00:35, 7227.19it/s] 36%|███▌      | 144629/400000 [00:19<00:34, 7404.39it/s] 36%|███▋      | 145405/400000 [00:19<00:33, 7504.95it/s] 37%|███▋      | 146182/400000 [00:19<00:33, 7579.73it/s] 37%|███▋      | 146942/400000 [00:19<00:34, 7382.88it/s] 37%|███▋      | 147692/400000 [00:19<00:34, 7415.01it/s] 37%|███▋      | 148479/400000 [00:19<00:33, 7543.74it/s] 37%|███▋      | 149261/400000 [00:20<00:32, 7623.51it/s] 38%|███▊      | 150025/400000 [00:20<00:32, 7619.91it/s] 38%|███▊      | 150789/400000 [00:20<00:33, 7551.65it/s] 38%|███▊      | 151560/400000 [00:20<00:32, 7593.86it/s] 38%|███▊      | 152339/400000 [00:20<00:32, 7650.77it/s] 38%|███▊      | 153122/400000 [00:20<00:32, 7701.94it/s] 38%|███▊      | 153908/400000 [00:20<00:31, 7748.35it/s] 39%|███▊      | 154687/400000 [00:20<00:31, 7760.38it/s] 39%|███▉      | 155464/400000 [00:20<00:31, 7669.76it/s] 39%|███▉      | 156236/400000 [00:20<00:31, 7684.46it/s] 39%|███▉      | 157005/400000 [00:21<00:32, 7476.35it/s] 39%|███▉      | 157755/400000 [00:21<00:32, 7344.32it/s] 40%|███▉      | 158491/400000 [00:21<00:33, 7224.10it/s] 40%|███▉      | 159215/400000 [00:21<00:34, 6953.73it/s] 40%|███▉      | 159942/400000 [00:21<00:34, 7045.19it/s] 40%|████      | 160661/400000 [00:21<00:33, 7087.60it/s] 40%|████      | 161384/400000 [00:21<00:33, 7129.00it/s] 41%|████      | 162114/400000 [00:21<00:33, 7178.85it/s] 41%|████      | 162845/400000 [00:21<00:32, 7215.27it/s] 41%|████      | 163607/400000 [00:21<00:32, 7330.09it/s] 41%|████      | 164392/400000 [00:22<00:31, 7476.50it/s] 41%|████▏     | 165159/400000 [00:22<00:31, 7531.73it/s] 41%|████▏     | 165944/400000 [00:22<00:30, 7623.29it/s] 42%|████▏     | 166714/400000 [00:22<00:30, 7645.98it/s] 42%|████▏     | 167503/400000 [00:22<00:30, 7716.85it/s] 42%|████▏     | 168276/400000 [00:22<00:30, 7700.26it/s] 42%|████▏     | 169047/400000 [00:22<00:30, 7682.05it/s] 42%|████▏     | 169816/400000 [00:22<00:30, 7527.10it/s] 43%|████▎     | 170570/400000 [00:22<00:31, 7320.68it/s] 43%|████▎     | 171304/400000 [00:23<00:31, 7224.66it/s] 43%|████▎     | 172029/400000 [00:23<00:31, 7125.05it/s] 43%|████▎     | 172794/400000 [00:23<00:31, 7273.93it/s] 43%|████▎     | 173580/400000 [00:23<00:30, 7439.39it/s] 44%|████▎     | 174356/400000 [00:23<00:29, 7531.90it/s] 44%|████▍     | 175112/400000 [00:23<00:29, 7538.06it/s] 44%|████▍     | 175868/400000 [00:23<00:30, 7439.79it/s] 44%|████▍     | 176614/400000 [00:23<00:30, 7296.35it/s] 44%|████▍     | 177346/400000 [00:23<00:30, 7293.68it/s] 45%|████▍     | 178077/400000 [00:23<00:30, 7180.82it/s] 45%|████▍     | 178806/400000 [00:24<00:30, 7212.61it/s] 45%|████▍     | 179529/400000 [00:24<00:30, 7214.39it/s] 45%|████▌     | 180252/400000 [00:24<00:30, 7194.06it/s] 45%|████▌     | 180972/400000 [00:24<00:30, 7144.21it/s] 45%|████▌     | 181716/400000 [00:24<00:30, 7230.10it/s] 46%|████▌     | 182504/400000 [00:24<00:29, 7411.05it/s] 46%|████▌     | 183292/400000 [00:24<00:28, 7538.61it/s] 46%|████▌     | 184080/400000 [00:24<00:28, 7635.98it/s] 46%|████▌     | 184845/400000 [00:24<00:28, 7636.38it/s] 46%|████▋     | 185611/400000 [00:24<00:28, 7643.19it/s] 47%|████▋     | 186401/400000 [00:25<00:27, 7716.08it/s] 47%|████▋     | 187185/400000 [00:25<00:27, 7751.05it/s] 47%|████▋     | 187961/400000 [00:25<00:27, 7703.84it/s] 47%|████▋     | 188732/400000 [00:25<00:27, 7611.04it/s] 47%|████▋     | 189494/400000 [00:25<00:28, 7323.31it/s] 48%|████▊     | 190230/400000 [00:25<00:28, 7301.50it/s] 48%|████▊     | 190963/400000 [00:25<00:28, 7262.85it/s] 48%|████▊     | 191691/400000 [00:25<00:28, 7224.97it/s] 48%|████▊     | 192415/400000 [00:25<00:28, 7206.28it/s] 48%|████▊     | 193137/400000 [00:25<00:28, 7146.67it/s] 48%|████▊     | 193853/400000 [00:26<00:28, 7135.28it/s] 49%|████▊     | 194588/400000 [00:26<00:28, 7196.46it/s] 49%|████▉     | 195309/400000 [00:26<00:28, 7198.27it/s] 49%|████▉     | 196032/400000 [00:26<00:28, 7206.20it/s] 49%|████▉     | 196763/400000 [00:26<00:28, 7236.42it/s] 49%|████▉     | 197551/400000 [00:26<00:27, 7417.41it/s] 50%|████▉     | 198310/400000 [00:26<00:27, 7467.48it/s] 50%|████▉     | 199097/400000 [00:26<00:26, 7582.01it/s] 50%|████▉     | 199884/400000 [00:26<00:26, 7665.65it/s] 50%|█████     | 200652/400000 [00:26<00:26, 7645.80it/s] 50%|█████     | 201437/400000 [00:27<00:25, 7705.86it/s] 51%|█████     | 202209/400000 [00:27<00:25, 7614.88it/s] 51%|█████     | 202988/400000 [00:27<00:25, 7663.83it/s] 51%|█████     | 203773/400000 [00:27<00:25, 7716.93it/s] 51%|█████     | 204546/400000 [00:27<00:25, 7595.08it/s] 51%|█████▏    | 205307/400000 [00:27<00:26, 7399.30it/s] 52%|█████▏    | 206081/400000 [00:27<00:25, 7497.62it/s] 52%|█████▏    | 206833/400000 [00:27<00:26, 7371.76it/s] 52%|█████▏    | 207611/400000 [00:27<00:25, 7487.95it/s] 52%|█████▏    | 208362/400000 [00:28<00:25, 7476.70it/s] 52%|█████▏    | 209111/400000 [00:28<00:25, 7429.40it/s] 52%|█████▏    | 209868/400000 [00:28<00:25, 7469.57it/s] 53%|█████▎    | 210616/400000 [00:28<00:26, 7258.23it/s] 53%|█████▎    | 211344/400000 [00:28<00:26, 7240.61it/s] 53%|█████▎    | 212070/400000 [00:28<00:26, 7092.96it/s] 53%|█████▎    | 212781/400000 [00:28<00:26, 7066.15it/s] 53%|█████▎    | 213558/400000 [00:28<00:25, 7261.68it/s] 54%|█████▎    | 214348/400000 [00:28<00:24, 7440.56it/s] 54%|█████▍    | 215110/400000 [00:28<00:24, 7493.40it/s] 54%|█████▍    | 215883/400000 [00:29<00:24, 7561.04it/s] 54%|█████▍    | 216669/400000 [00:29<00:23, 7646.21it/s] 54%|█████▍    | 217444/400000 [00:29<00:23, 7676.15it/s] 55%|█████▍    | 218228/400000 [00:29<00:23, 7723.59it/s] 55%|█████▍    | 219002/400000 [00:29<00:24, 7473.58it/s] 55%|█████▍    | 219752/400000 [00:29<00:24, 7320.59it/s] 55%|█████▌    | 220494/400000 [00:29<00:24, 7349.83it/s] 55%|█████▌    | 221275/400000 [00:29<00:23, 7480.30it/s] 56%|█████▌    | 222057/400000 [00:29<00:23, 7577.31it/s] 56%|█████▌    | 222838/400000 [00:29<00:23, 7643.32it/s] 56%|█████▌    | 223604/400000 [00:30<00:23, 7413.33it/s] 56%|█████▌    | 224379/400000 [00:30<00:23, 7510.88it/s] 56%|█████▋    | 225159/400000 [00:30<00:23, 7595.32it/s] 56%|█████▋    | 225921/400000 [00:30<00:22, 7591.79it/s] 57%|█████▋    | 226704/400000 [00:30<00:22, 7659.47it/s] 57%|█████▋    | 227471/400000 [00:30<00:22, 7630.21it/s] 57%|█████▋    | 228236/400000 [00:30<00:22, 7633.93it/s] 57%|█████▋    | 229018/400000 [00:30<00:22, 7686.50it/s] 57%|█████▋    | 229803/400000 [00:30<00:22, 7731.70it/s] 58%|█████▊    | 230584/400000 [00:30<00:21, 7752.32it/s] 58%|█████▊    | 231360/400000 [00:31<00:22, 7643.68it/s] 58%|█████▊    | 232127/400000 [00:31<00:21, 7650.38it/s] 58%|█████▊    | 232901/400000 [00:31<00:21, 7675.62it/s] 58%|█████▊    | 233679/400000 [00:31<00:21, 7706.40it/s] 59%|█████▊    | 234461/400000 [00:31<00:21, 7738.01it/s] 59%|█████▉    | 235235/400000 [00:31<00:21, 7677.45it/s] 59%|█████▉    | 236010/400000 [00:31<00:21, 7698.58it/s] 59%|█████▉    | 236781/400000 [00:31<00:21, 7690.53it/s] 59%|█████▉    | 237569/400000 [00:31<00:20, 7743.77it/s] 60%|█████▉    | 238351/400000 [00:31<00:20, 7764.78it/s] 60%|█████▉    | 239128/400000 [00:32<00:20, 7712.77it/s] 60%|█████▉    | 239910/400000 [00:32<00:20, 7744.52it/s] 60%|██████    | 240685/400000 [00:32<00:20, 7631.37it/s] 60%|██████    | 241467/400000 [00:32<00:20, 7684.66it/s] 61%|██████    | 242236/400000 [00:32<00:20, 7523.80it/s] 61%|██████    | 242990/400000 [00:32<00:21, 7322.35it/s] 61%|██████    | 243725/400000 [00:32<00:21, 7307.57it/s] 61%|██████    | 244458/400000 [00:32<00:21, 7228.67it/s] 61%|██████▏   | 245183/400000 [00:32<00:21, 7138.18it/s] 61%|██████▏   | 245912/400000 [00:32<00:21, 7181.58it/s] 62%|██████▏   | 246631/400000 [00:33<00:21, 7128.28it/s] 62%|██████▏   | 247407/400000 [00:33<00:20, 7305.61it/s] 62%|██████▏   | 248195/400000 [00:33<00:20, 7468.21it/s] 62%|██████▏   | 248973/400000 [00:33<00:19, 7557.30it/s] 62%|██████▏   | 249750/400000 [00:33<00:19, 7618.35it/s] 63%|██████▎   | 250514/400000 [00:33<00:20, 7442.27it/s] 63%|██████▎   | 251298/400000 [00:33<00:19, 7556.28it/s] 63%|██████▎   | 252056/400000 [00:33<00:19, 7514.77it/s] 63%|██████▎   | 252809/400000 [00:33<00:19, 7493.68it/s] 63%|██████▎   | 253570/400000 [00:34<00:19, 7525.40it/s] 64%|██████▎   | 254324/400000 [00:34<00:19, 7526.25it/s] 64%|██████▍   | 255107/400000 [00:34<00:19, 7612.19it/s] 64%|██████▍   | 255888/400000 [00:34<00:18, 7668.32it/s] 64%|██████▍   | 256668/400000 [00:34<00:18, 7707.00it/s] 64%|██████▍   | 257447/400000 [00:34<00:18, 7730.90it/s] 65%|██████▍   | 258221/400000 [00:34<00:18, 7606.36it/s] 65%|██████▍   | 259006/400000 [00:34<00:18, 7677.49it/s] 65%|██████▍   | 259790/400000 [00:34<00:18, 7724.64it/s] 65%|██████▌   | 260575/400000 [00:34<00:17, 7759.86it/s] 65%|██████▌   | 261362/400000 [00:35<00:17, 7790.08it/s] 66%|██████▌   | 262142/400000 [00:35<00:18, 7541.22it/s] 66%|██████▌   | 262899/400000 [00:35<00:18, 7405.53it/s] 66%|██████▌   | 263642/400000 [00:35<00:18, 7320.69it/s] 66%|██████▌   | 264376/400000 [00:35<00:18, 7267.39it/s] 66%|██████▋   | 265104/400000 [00:35<00:18, 7229.82it/s] 66%|██████▋   | 265836/400000 [00:35<00:18, 7256.00it/s] 67%|██████▋   | 266563/400000 [00:35<00:18, 7201.01it/s] 67%|██████▋   | 267284/400000 [00:35<00:18, 7192.28it/s] 67%|██████▋   | 268008/400000 [00:35<00:18, 7205.77it/s] 67%|██████▋   | 268729/400000 [00:36<00:18, 7206.48it/s] 67%|██████▋   | 269451/400000 [00:36<00:18, 7209.72it/s] 68%|██████▊   | 270231/400000 [00:36<00:17, 7374.89it/s] 68%|██████▊   | 270994/400000 [00:36<00:17, 7448.67it/s] 68%|██████▊   | 271780/400000 [00:36<00:16, 7565.51it/s] 68%|██████▊   | 272567/400000 [00:36<00:16, 7653.37it/s] 68%|██████▊   | 273334/400000 [00:36<00:16, 7624.51it/s] 69%|██████▊   | 274120/400000 [00:36<00:16, 7692.54it/s] 69%|██████▊   | 274890/400000 [00:36<00:16, 7570.50it/s] 69%|██████▉   | 275673/400000 [00:36<00:16, 7643.92it/s] 69%|██████▉   | 276455/400000 [00:37<00:16, 7694.12it/s] 69%|██████▉   | 277226/400000 [00:37<00:16, 7654.85it/s] 70%|██████▉   | 278001/400000 [00:37<00:15, 7681.72it/s] 70%|██████▉   | 278777/400000 [00:37<00:15, 7702.78it/s] 70%|██████▉   | 279548/400000 [00:37<00:16, 7520.26it/s] 70%|███████   | 280330/400000 [00:37<00:15, 7606.50it/s] 70%|███████   | 281092/400000 [00:37<00:15, 7568.75it/s] 70%|███████   | 281879/400000 [00:37<00:15, 7653.82it/s] 71%|███████   | 282661/400000 [00:37<00:15, 7700.08it/s] 71%|███████   | 283436/400000 [00:37<00:15, 7714.01it/s] 71%|███████   | 284223/400000 [00:38<00:14, 7758.51it/s] 71%|███████▏  | 285000/400000 [00:38<00:14, 7710.72it/s] 71%|███████▏  | 285781/400000 [00:38<00:14, 7739.84it/s] 72%|███████▏  | 286567/400000 [00:38<00:14, 7774.86it/s] 72%|███████▏  | 287351/400000 [00:38<00:14, 7794.27it/s] 72%|███████▏  | 288131/400000 [00:38<00:14, 7750.18it/s] 72%|███████▏  | 288907/400000 [00:38<00:14, 7669.68it/s] 72%|███████▏  | 289675/400000 [00:38<00:14, 7440.86it/s] 73%|███████▎  | 290421/400000 [00:38<00:14, 7363.79it/s] 73%|███████▎  | 291159/400000 [00:38<00:14, 7289.13it/s] 73%|███████▎  | 291890/400000 [00:39<00:15, 7189.27it/s] 73%|███████▎  | 292611/400000 [00:39<00:14, 7177.93it/s] 73%|███████▎  | 293386/400000 [00:39<00:14, 7338.34it/s] 74%|███████▎  | 294162/400000 [00:39<00:14, 7459.84it/s] 74%|███████▎  | 294944/400000 [00:39<00:13, 7562.09it/s] 74%|███████▍  | 295702/400000 [00:39<00:13, 7456.45it/s] 74%|███████▍  | 296449/400000 [00:39<00:14, 7351.51it/s] 74%|███████▍  | 297186/400000 [00:39<00:14, 7316.93it/s] 74%|███████▍  | 297919/400000 [00:39<00:14, 7261.04it/s] 75%|███████▍  | 298646/400000 [00:40<00:14, 7230.66it/s] 75%|███████▍  | 299370/400000 [00:40<00:13, 7215.27it/s] 75%|███████▌  | 300092/400000 [00:40<00:14, 7073.15it/s] 75%|███████▌  | 300844/400000 [00:40<00:13, 7198.91it/s] 75%|███████▌  | 301611/400000 [00:40<00:13, 7333.43it/s] 76%|███████▌  | 302393/400000 [00:40<00:13, 7471.37it/s] 76%|███████▌  | 303181/400000 [00:40<00:12, 7588.99it/s] 76%|███████▌  | 303942/400000 [00:40<00:12, 7592.29it/s] 76%|███████▌  | 304720/400000 [00:40<00:12, 7647.59it/s] 76%|███████▋  | 305505/400000 [00:40<00:12, 7704.95it/s] 77%|███████▋  | 306277/400000 [00:41<00:12, 7577.46it/s] 77%|███████▋  | 307036/400000 [00:41<00:12, 7404.31it/s] 77%|███████▋  | 307778/400000 [00:41<00:12, 7406.26it/s] 77%|███████▋  | 308560/400000 [00:41<00:12, 7524.06it/s] 77%|███████▋  | 309314/400000 [00:41<00:12, 7469.88it/s] 78%|███████▊  | 310101/400000 [00:41<00:11, 7583.95it/s] 78%|███████▊  | 310887/400000 [00:41<00:11, 7663.81it/s] 78%|███████▊  | 311655/400000 [00:41<00:11, 7650.68it/s] 78%|███████▊  | 312442/400000 [00:41<00:11, 7713.67it/s] 78%|███████▊  | 313214/400000 [00:41<00:11, 7683.52it/s] 79%|███████▊  | 314001/400000 [00:42<00:11, 7736.25it/s] 79%|███████▊  | 314787/400000 [00:42<00:10, 7765.72it/s] 79%|███████▉  | 315564/400000 [00:42<00:11, 7634.72it/s] 79%|███████▉  | 316349/400000 [00:42<00:10, 7695.69it/s] 79%|███████▉  | 317133/400000 [00:42<00:10, 7735.64it/s] 79%|███████▉  | 317908/400000 [00:42<00:10, 7664.19it/s] 80%|███████▉  | 318682/400000 [00:42<00:10, 7685.29it/s] 80%|███████▉  | 319451/400000 [00:42<00:10, 7652.78it/s] 80%|████████  | 320238/400000 [00:42<00:10, 7714.14it/s] 80%|████████  | 321020/400000 [00:42<00:10, 7743.84it/s] 80%|████████  | 321795/400000 [00:43<00:10, 7670.86it/s] 81%|████████  | 322578/400000 [00:43<00:10, 7716.19it/s] 81%|████████  | 323350/400000 [00:43<00:10, 7537.24it/s] 81%|████████  | 324105/400000 [00:43<00:10, 7405.51it/s] 81%|████████  | 324847/400000 [00:43<00:10, 7308.33it/s] 81%|████████▏ | 325588/400000 [00:43<00:10, 7306.06it/s] 82%|████████▏ | 326320/400000 [00:43<00:10, 7267.52it/s] 82%|████████▏ | 327048/400000 [00:43<00:10, 7249.55it/s] 82%|████████▏ | 327834/400000 [00:43<00:09, 7422.31it/s] 82%|████████▏ | 328620/400000 [00:43<00:09, 7547.21it/s] 82%|████████▏ | 329405/400000 [00:44<00:09, 7634.21it/s] 83%|████████▎ | 330170/400000 [00:44<00:09, 7592.07it/s] 83%|████████▎ | 330931/400000 [00:44<00:09, 7420.47it/s] 83%|████████▎ | 331675/400000 [00:44<00:09, 7110.44it/s] 83%|████████▎ | 332452/400000 [00:44<00:09, 7295.68it/s] 83%|████████▎ | 333186/400000 [00:44<00:09, 7303.00it/s] 83%|████████▎ | 333920/400000 [00:44<00:09, 7177.84it/s] 84%|████████▎ | 334641/400000 [00:44<00:09, 7166.94it/s] 84%|████████▍ | 335421/400000 [00:44<00:08, 7343.55it/s] 84%|████████▍ | 336207/400000 [00:45<00:08, 7490.78it/s] 84%|████████▍ | 336990/400000 [00:45<00:08, 7588.50it/s] 84%|████████▍ | 337755/400000 [00:45<00:08, 7604.91it/s] 85%|████████▍ | 338535/400000 [00:45<00:08, 7659.60it/s] 85%|████████▍ | 339302/400000 [00:45<00:07, 7598.07it/s] 85%|████████▌ | 340068/400000 [00:45<00:07, 7616.37it/s] 85%|████████▌ | 340831/400000 [00:45<00:07, 7524.45it/s] 85%|████████▌ | 341585/400000 [00:45<00:07, 7353.44it/s] 86%|████████▌ | 342322/400000 [00:45<00:07, 7314.53it/s] 86%|████████▌ | 343055/400000 [00:45<00:08, 7070.93it/s] 86%|████████▌ | 343769/400000 [00:46<00:07, 7089.36it/s] 86%|████████▌ | 344495/400000 [00:46<00:07, 7137.98it/s] 86%|████████▋ | 345211/400000 [00:46<00:07, 7139.24it/s] 86%|████████▋ | 345990/400000 [00:46<00:07, 7319.96it/s] 87%|████████▋ | 346771/400000 [00:46<00:07, 7459.11it/s] 87%|████████▋ | 347519/400000 [00:46<00:07, 7193.05it/s] 87%|████████▋ | 348243/400000 [00:46<00:07, 7205.54it/s] 87%|████████▋ | 348966/400000 [00:46<00:07, 7108.30it/s] 87%|████████▋ | 349688/400000 [00:46<00:07, 7141.12it/s] 88%|████████▊ | 350410/400000 [00:46<00:06, 7163.34it/s] 88%|████████▊ | 351163/400000 [00:47<00:06, 7267.08it/s] 88%|████████▊ | 351899/400000 [00:47<00:06, 7293.85it/s] 88%|████████▊ | 352630/400000 [00:47<00:06, 7230.38it/s] 88%|████████▊ | 353413/400000 [00:47<00:06, 7398.08it/s] 89%|████████▊ | 354194/400000 [00:47<00:06, 7514.56it/s] 89%|████████▊ | 354976/400000 [00:47<00:05, 7601.66it/s] 89%|████████▉ | 355738/400000 [00:47<00:05, 7497.88it/s] 89%|████████▉ | 356491/400000 [00:47<00:05, 7506.28it/s] 89%|████████▉ | 357274/400000 [00:47<00:05, 7599.18it/s] 90%|████████▉ | 358058/400000 [00:47<00:05, 7669.52it/s] 90%|████████▉ | 358826/400000 [00:48<00:05, 7659.81it/s] 90%|████████▉ | 359594/400000 [00:48<00:05, 7663.95it/s] 90%|█████████ | 360361/400000 [00:48<00:05, 7269.19it/s] 90%|█████████ | 361093/400000 [00:48<00:05, 7269.14it/s] 90%|█████████ | 361824/400000 [00:48<00:05, 7249.21it/s] 91%|█████████ | 362552/400000 [00:48<00:05, 7249.05it/s] 91%|█████████ | 363279/400000 [00:48<00:05, 7249.54it/s] 91%|█████████ | 364006/400000 [00:48<00:04, 7224.39it/s] 91%|█████████ | 364761/400000 [00:48<00:04, 7317.33it/s] 91%|█████████▏| 365546/400000 [00:48<00:04, 7468.54it/s] 92%|█████████▏| 366307/400000 [00:49<00:04, 7508.30it/s] 92%|█████████▏| 367059/400000 [00:49<00:04, 7482.44it/s] 92%|█████████▏| 367808/400000 [00:49<00:04, 7294.85it/s] 92%|█████████▏| 368540/400000 [00:49<00:04, 6942.03it/s] 92%|█████████▏| 369310/400000 [00:49<00:04, 7152.14it/s] 93%|█████████▎| 370083/400000 [00:49<00:04, 7315.25it/s] 93%|█████████▎| 370866/400000 [00:49<00:03, 7462.21it/s] 93%|█████████▎| 371636/400000 [00:49<00:03, 7529.50it/s] 93%|█████████▎| 372420/400000 [00:49<00:03, 7619.41it/s] 93%|█████████▎| 373185/400000 [00:50<00:03, 7586.38it/s] 93%|█████████▎| 373953/400000 [00:50<00:03, 7614.10it/s] 94%|█████████▎| 374722/400000 [00:50<00:03, 7636.38it/s] 94%|█████████▍| 375487/400000 [00:50<00:03, 7543.08it/s] 94%|█████████▍| 376243/400000 [00:50<00:03, 7413.87it/s] 94%|█████████▍| 376986/400000 [00:50<00:03, 7198.94it/s] 94%|█████████▍| 377709/400000 [00:50<00:03, 7184.56it/s] 95%|█████████▍| 378429/400000 [00:50<00:03, 6968.84it/s] 95%|█████████▍| 379166/400000 [00:50<00:02, 7083.79it/s] 95%|█████████▍| 379947/400000 [00:50<00:02, 7285.89it/s] 95%|█████████▌| 380728/400000 [00:51<00:02, 7433.92it/s] 95%|█████████▌| 381475/400000 [00:51<00:02, 7414.62it/s] 96%|█████████▌| 382256/400000 [00:51<00:02, 7526.80it/s] 96%|█████████▌| 383028/400000 [00:51<00:02, 7582.20it/s] 96%|█████████▌| 383807/400000 [00:51<00:02, 7642.58it/s] 96%|█████████▌| 384585/400000 [00:51<00:02, 7683.29it/s] 96%|█████████▋| 385355/400000 [00:51<00:01, 7642.58it/s] 97%|█████████▋| 386120/400000 [00:51<00:01, 7617.32it/s] 97%|█████████▋| 386894/400000 [00:51<00:01, 7652.57it/s] 97%|█████████▋| 387673/400000 [00:51<00:01, 7690.55it/s] 97%|█████████▋| 388460/400000 [00:52<00:01, 7742.66it/s] 97%|█████████▋| 389237/400000 [00:52<00:01, 7750.71it/s] 98%|█████████▊| 390013/400000 [00:52<00:01, 7706.80it/s] 98%|█████████▊| 390784/400000 [00:52<00:01, 7668.47it/s] 98%|█████████▊| 391569/400000 [00:52<00:01, 7721.10it/s] 98%|█████████▊| 392342/400000 [00:52<00:00, 7663.92it/s] 98%|█████████▊| 393109/400000 [00:52<00:00, 7513.38it/s] 98%|█████████▊| 393862/400000 [00:52<00:00, 7324.42it/s] 99%|█████████▊| 394597/400000 [00:52<00:00, 7220.74it/s] 99%|█████████▉| 395373/400000 [00:52<00:00, 7373.27it/s] 99%|█████████▉| 396153/400000 [00:53<00:00, 7493.78it/s] 99%|█████████▉| 396908/400000 [00:53<00:00, 7508.16it/s] 99%|█████████▉| 397686/400000 [00:53<00:00, 7587.04it/s]100%|█████████▉| 398446/400000 [00:53<00:00, 7493.00it/s]100%|█████████▉| 399197/400000 [00:53<00:00, 7487.78it/s]100%|█████████▉| 399981/400000 [00:53<00:00, 7589.83it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7463.17it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe373cc10f0>, <torchtext.data.dataset.TabularDataset object at 0x7fe373cc1240>, <torchtext.vocab.Vocab object at 0x7fe373cc1160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 29.01 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 56.32 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 22.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.25 file/s]2020-06-14 06:18:35.840632: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 06:18:35.844890: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-14 06:18:35.845384: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558623b8bee0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 06:18:35.845701: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 47%|████▋     | 4612096/9912422 [00:00<00:00, 46010428.89it/s]9920512it [00:00, 37053704.88it/s]                             
0it [00:00, ?it/s]32768it [00:00, 609947.91it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 485733.54it/s]1654784it [00:00, 12284801.50it/s]                         
0it [00:00, ?it/s]8192it [00:00, 212947.63it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
