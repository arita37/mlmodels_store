
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f0d79c1a6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f0d79c1a6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0de51e20d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0de51e20d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0dfef0eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0dfef0eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0d92a5d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0d92a5d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0d92a5d950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3153920/9912422 [00:00<00:00, 31534192.79it/s]9920512it [00:00, 35793241.43it/s]                             
0it [00:00, ?it/s]32768it [00:00, 614436.29it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 163221.79it/s]1654784it [00:00, 11632435.54it/s]                         
0it [00:00, ?it/s]8192it [00:00, 204092.20it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d90073a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d90069cf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f0d92a5d598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f0d92a5d598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f0d92a5d598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:14,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:13,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:12,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:12,  2.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:50,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:50,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:49,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:49,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:49,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:48,  3.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:34,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:33,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:22,  5.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  8.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 11.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 14.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 37.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 57.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 65.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 67.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 68.08 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 69.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.75s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.75s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.75s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.75s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 70.70 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.87 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ url]
0 examples [00:00, ? examples/s]2020-06-14 00:09:55.706374: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 00:09:55.752077: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-14 00:09:55.752268: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563b4dea7810 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 00:09:55.752287: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.77 examples/s]91 examples [00:00,  2.52 examples/s]192 examples [00:00,  3.60 examples/s]286 examples [00:00,  5.14 examples/s]380 examples [00:00,  7.32 examples/s]478 examples [00:01, 10.43 examples/s]578 examples [00:01, 14.83 examples/s]679 examples [00:01, 21.05 examples/s]770 examples [00:01, 29.76 examples/s]869 examples [00:01, 41.97 examples/s]965 examples [00:01, 58.86 examples/s]1059 examples [00:01, 81.87 examples/s]1153 examples [00:01, 112.64 examples/s]1246 examples [00:01, 152.67 examples/s]1342 examples [00:01, 204.10 examples/s]1442 examples [00:02, 268.06 examples/s]1541 examples [00:02, 342.94 examples/s]1640 examples [00:02, 426.54 examples/s]1739 examples [00:02, 514.29 examples/s]1837 examples [00:02, 599.41 examples/s]1937 examples [00:02, 680.05 examples/s]2036 examples [00:02, 747.05 examples/s]2135 examples [00:02, 805.19 examples/s]2233 examples [00:02, 845.67 examples/s]2332 examples [00:02, 882.29 examples/s]2431 examples [00:03, 911.07 examples/s]2529 examples [00:03, 929.78 examples/s]2628 examples [00:03, 945.37 examples/s]2726 examples [00:03, 953.25 examples/s]2825 examples [00:03, 961.17 examples/s]2925 examples [00:03, 971.73 examples/s]3024 examples [00:03, 973.08 examples/s]3123 examples [00:03, 966.01 examples/s]3221 examples [00:03, 969.57 examples/s]3319 examples [00:04, 964.75 examples/s]3416 examples [00:04, 941.78 examples/s]3511 examples [00:04, 943.63 examples/s]3606 examples [00:04, 943.14 examples/s]3701 examples [00:04, 933.10 examples/s]3796 examples [00:04, 937.59 examples/s]3895 examples [00:04, 951.28 examples/s]3993 examples [00:04, 958.75 examples/s]4091 examples [00:04, 963.74 examples/s]4188 examples [00:04, 948.89 examples/s]4286 examples [00:05, 955.40 examples/s]4385 examples [00:05, 962.96 examples/s]4483 examples [00:05, 967.21 examples/s]4580 examples [00:05, 942.48 examples/s]4675 examples [00:05, 925.98 examples/s]4768 examples [00:05, 921.74 examples/s]4864 examples [00:05, 932.50 examples/s]4959 examples [00:05, 935.18 examples/s]5055 examples [00:05, 942.09 examples/s]5153 examples [00:05, 952.17 examples/s]5249 examples [00:06, 934.57 examples/s]5346 examples [00:06, 939.82 examples/s]5441 examples [00:06, 933.08 examples/s]5538 examples [00:06, 941.67 examples/s]5635 examples [00:06, 948.48 examples/s]5731 examples [00:06, 951.31 examples/s]5829 examples [00:06, 956.92 examples/s]5925 examples [00:06, 938.34 examples/s]6019 examples [00:06, 921.51 examples/s]6116 examples [00:06, 934.39 examples/s]6213 examples [00:07, 943.37 examples/s]6311 examples [00:07, 951.78 examples/s]6407 examples [00:07, 943.74 examples/s]6505 examples [00:07, 951.70 examples/s]6601 examples [00:07, 932.00 examples/s]6695 examples [00:07, 933.57 examples/s]6789 examples [00:07, 935.26 examples/s]6883 examples [00:07, 909.20 examples/s]6975 examples [00:07, 911.41 examples/s]7067 examples [00:08, 908.82 examples/s]7165 examples [00:08, 927.71 examples/s]7264 examples [00:08, 944.14 examples/s]7359 examples [00:08, 938.34 examples/s]7457 examples [00:08, 950.24 examples/s]7555 examples [00:08, 956.42 examples/s]7652 examples [00:08, 959.65 examples/s]7749 examples [00:08, 958.24 examples/s]7845 examples [00:08, 941.72 examples/s]7944 examples [00:08, 953.11 examples/s]8040 examples [00:09, 950.30 examples/s]8137 examples [00:09, 953.81 examples/s]8233 examples [00:09, 944.15 examples/s]8328 examples [00:09, 931.24 examples/s]8422 examples [00:09, 899.39 examples/s]8513 examples [00:09, 887.27 examples/s]8606 examples [00:09, 897.18 examples/s]8698 examples [00:09, 901.73 examples/s]8794 examples [00:09, 917.90 examples/s]8887 examples [00:09, 919.76 examples/s]8984 examples [00:10, 933.62 examples/s]9079 examples [00:10, 936.01 examples/s]9173 examples [00:10, 911.69 examples/s]9269 examples [00:10, 924.52 examples/s]9367 examples [00:10, 938.31 examples/s]9462 examples [00:10, 932.60 examples/s]9556 examples [00:10, 910.09 examples/s]9649 examples [00:10, 915.31 examples/s]9741 examples [00:10, 905.83 examples/s]9840 examples [00:10, 928.64 examples/s]9936 examples [00:11, 935.96 examples/s]10030 examples [00:11, 881.93 examples/s]10127 examples [00:11, 906.38 examples/s]10222 examples [00:11, 917.16 examples/s]10318 examples [00:11, 927.59 examples/s]10418 examples [00:11, 947.93 examples/s]10514 examples [00:11, 943.91 examples/s]10611 examples [00:11, 950.78 examples/s]10707 examples [00:11, 949.52 examples/s]10803 examples [00:12, 943.23 examples/s]10898 examples [00:12, 936.04 examples/s]10992 examples [00:12, 929.66 examples/s]11086 examples [00:12, 929.78 examples/s]11180 examples [00:12, 922.45 examples/s]11273 examples [00:12, 900.42 examples/s]11371 examples [00:12, 919.84 examples/s]11464 examples [00:12, 918.83 examples/s]11558 examples [00:12, 924.38 examples/s]11652 examples [00:12, 927.43 examples/s]11750 examples [00:13, 940.50 examples/s]11850 examples [00:13, 955.78 examples/s]11948 examples [00:13, 960.68 examples/s]12047 examples [00:13, 968.98 examples/s]12147 examples [00:13, 976.60 examples/s]12246 examples [00:13, 978.13 examples/s]12346 examples [00:13, 982.30 examples/s]12445 examples [00:13, 953.34 examples/s]12541 examples [00:13, 951.34 examples/s]12640 examples [00:13, 959.90 examples/s]12737 examples [00:14, 937.95 examples/s]12836 examples [00:14, 950.50 examples/s]12937 examples [00:14, 967.23 examples/s]13034 examples [00:14, 961.95 examples/s]13131 examples [00:14, 954.30 examples/s]13231 examples [00:14, 964.55 examples/s]13328 examples [00:14, 945.73 examples/s]13423 examples [00:14, 939.31 examples/s]13518 examples [00:14, 930.02 examples/s]13615 examples [00:14, 940.38 examples/s]13714 examples [00:15, 952.17 examples/s]13810 examples [00:15, 952.56 examples/s]13909 examples [00:15, 963.32 examples/s]14008 examples [00:15, 970.23 examples/s]14108 examples [00:15, 978.62 examples/s]14206 examples [00:15, 954.89 examples/s]14302 examples [00:15, 942.87 examples/s]14401 examples [00:15, 954.18 examples/s]14497 examples [00:15, 947.46 examples/s]14592 examples [00:16, 944.58 examples/s]14687 examples [00:16, 936.42 examples/s]14783 examples [00:16, 941.03 examples/s]14879 examples [00:16, 945.21 examples/s]14978 examples [00:16, 955.67 examples/s]15077 examples [00:16, 963.58 examples/s]15174 examples [00:16, 948.50 examples/s]15269 examples [00:16, 944.52 examples/s]15364 examples [00:16, 923.22 examples/s]15457 examples [00:16, 922.60 examples/s]15550 examples [00:17, 920.77 examples/s]15643 examples [00:17, 901.06 examples/s]15738 examples [00:17, 913.60 examples/s]15837 examples [00:17, 933.92 examples/s]15935 examples [00:17, 946.97 examples/s]16034 examples [00:17, 957.81 examples/s]16135 examples [00:17, 971.40 examples/s]16233 examples [00:17, 968.84 examples/s]16330 examples [00:17, 967.26 examples/s]16427 examples [00:17, 945.31 examples/s]16525 examples [00:18, 955.30 examples/s]16624 examples [00:18, 965.20 examples/s]16721 examples [00:18, 961.58 examples/s]16821 examples [00:18, 970.68 examples/s]16919 examples [00:18, 969.53 examples/s]17017 examples [00:18, 967.06 examples/s]17117 examples [00:18, 975.99 examples/s]17215 examples [00:18, 969.77 examples/s]17313 examples [00:18, 970.46 examples/s]17411 examples [00:18, 939.22 examples/s]17508 examples [00:19, 946.53 examples/s]17603 examples [00:19, 935.29 examples/s]17699 examples [00:19, 941.69 examples/s]17796 examples [00:19, 948.99 examples/s]17893 examples [00:19, 954.14 examples/s]17991 examples [00:19, 960.26 examples/s]18089 examples [00:19, 964.17 examples/s]18186 examples [00:19, 942.81 examples/s]18281 examples [00:19, 928.94 examples/s]18379 examples [00:19, 943.09 examples/s]18477 examples [00:20, 951.58 examples/s]18577 examples [00:20, 963.14 examples/s]18674 examples [00:20, 953.31 examples/s]18774 examples [00:20, 964.34 examples/s]18871 examples [00:20, 957.73 examples/s]18967 examples [00:20, 939.46 examples/s]19063 examples [00:20, 944.05 examples/s]19158 examples [00:20, 918.18 examples/s]19251 examples [00:20, 916.08 examples/s]19343 examples [00:21, 915.40 examples/s]19438 examples [00:21, 924.66 examples/s]19531 examples [00:21, 912.24 examples/s]19625 examples [00:21, 919.11 examples/s]19723 examples [00:21, 935.74 examples/s]19822 examples [00:21, 949.09 examples/s]19922 examples [00:21, 961.12 examples/s]20019 examples [00:21, 920.01 examples/s]20112 examples [00:21, 913.36 examples/s]20204 examples [00:21, 899.13 examples/s]20300 examples [00:22, 914.13 examples/s]20395 examples [00:22, 921.91 examples/s]20496 examples [00:22, 944.91 examples/s]20594 examples [00:22, 954.63 examples/s]20695 examples [00:22, 970.43 examples/s]20793 examples [00:22, 969.69 examples/s]20891 examples [00:22, 957.72 examples/s]20987 examples [00:22, 951.11 examples/s]21083 examples [00:22, 916.74 examples/s]21181 examples [00:22, 933.91 examples/s]21282 examples [00:23, 955.17 examples/s]21381 examples [00:23, 963.42 examples/s]21479 examples [00:23, 967.44 examples/s]21576 examples [00:23, 947.09 examples/s]21678 examples [00:23, 966.57 examples/s]21779 examples [00:23, 976.68 examples/s]21879 examples [00:23, 981.45 examples/s]21981 examples [00:23, 991.58 examples/s]22081 examples [00:23, 982.47 examples/s]22182 examples [00:23, 990.03 examples/s]22285 examples [00:24, 998.94 examples/s]22385 examples [00:24, 996.36 examples/s]22486 examples [00:24, 997.87 examples/s]22586 examples [00:24, 983.50 examples/s]22685 examples [00:24, 950.00 examples/s]22781 examples [00:24, 932.69 examples/s]22882 examples [00:24, 954.18 examples/s]22983 examples [00:24, 961.75 examples/s]23080 examples [00:24, 959.10 examples/s]23177 examples [00:25, 962.32 examples/s]23279 examples [00:25, 977.45 examples/s]23379 examples [00:25, 982.00 examples/s]23481 examples [00:25, 990.36 examples/s]23581 examples [00:25, 992.25 examples/s]23681 examples [00:25, 994.23 examples/s]23782 examples [00:25, 998.78 examples/s]23882 examples [00:25, 990.10 examples/s]23982 examples [00:25, 981.54 examples/s]24081 examples [00:25, 944.79 examples/s]24176 examples [00:26, 942.80 examples/s]24276 examples [00:26, 958.54 examples/s]24373 examples [00:26, 939.17 examples/s]24470 examples [00:26, 945.58 examples/s]24571 examples [00:26, 963.34 examples/s]24673 examples [00:26, 979.53 examples/s]24772 examples [00:26, 977.53 examples/s]24871 examples [00:26, 978.99 examples/s]24970 examples [00:26, 958.92 examples/s]25067 examples [00:26, 957.20 examples/s]25168 examples [00:27, 971.43 examples/s]25268 examples [00:27, 977.16 examples/s]25367 examples [00:27, 979.67 examples/s]25466 examples [00:27, 979.58 examples/s]25566 examples [00:27, 983.23 examples/s]25665 examples [00:27, 981.92 examples/s]25764 examples [00:27, 973.50 examples/s]25863 examples [00:27, 976.94 examples/s]25961 examples [00:27, 959.04 examples/s]26061 examples [00:27, 969.30 examples/s]26159 examples [00:28, 944.25 examples/s]26257 examples [00:28, 952.33 examples/s]26354 examples [00:28, 957.14 examples/s]26451 examples [00:28, 958.86 examples/s]26550 examples [00:28, 965.80 examples/s]26647 examples [00:28, 961.44 examples/s]26744 examples [00:28, 963.26 examples/s]26841 examples [00:28, 947.48 examples/s]26936 examples [00:28, 919.86 examples/s]27029 examples [00:29, 910.41 examples/s]27129 examples [00:29, 934.03 examples/s]27223 examples [00:29, 921.86 examples/s]27322 examples [00:29, 939.60 examples/s]27420 examples [00:29, 950.38 examples/s]27516 examples [00:29, 948.47 examples/s]27616 examples [00:29, 960.34 examples/s]27713 examples [00:29, 962.73 examples/s]27812 examples [00:29, 968.66 examples/s]27909 examples [00:29, 910.84 examples/s]28009 examples [00:30, 933.88 examples/s]28107 examples [00:30, 946.51 examples/s]28205 examples [00:30, 954.70 examples/s]28301 examples [00:30, 941.68 examples/s]28398 examples [00:30, 949.88 examples/s]28498 examples [00:30, 963.48 examples/s]28595 examples [00:30, 965.37 examples/s]28692 examples [00:30, 961.98 examples/s]28789 examples [00:30, 930.84 examples/s]28884 examples [00:30, 934.05 examples/s]28981 examples [00:31, 943.85 examples/s]29082 examples [00:31, 962.51 examples/s]29179 examples [00:31, 951.68 examples/s]29275 examples [00:31, 942.74 examples/s]29371 examples [00:31, 947.05 examples/s]29469 examples [00:31, 956.19 examples/s]29567 examples [00:31, 962.15 examples/s]29664 examples [00:31, 954.95 examples/s]29760 examples [00:31, 917.63 examples/s]29853 examples [00:32, 904.95 examples/s]29944 examples [00:32, 897.26 examples/s]30034 examples [00:32, 847.66 examples/s]30131 examples [00:32, 880.88 examples/s]30228 examples [00:32, 905.74 examples/s]30325 examples [00:32, 921.52 examples/s]30422 examples [00:32, 933.48 examples/s]30519 examples [00:32, 943.25 examples/s]30614 examples [00:32, 942.46 examples/s]30709 examples [00:32, 935.31 examples/s]30803 examples [00:33, 932.63 examples/s]30902 examples [00:33, 948.21 examples/s]31001 examples [00:33, 959.26 examples/s]31098 examples [00:33, 961.24 examples/s]31197 examples [00:33, 968.71 examples/s]31294 examples [00:33, 965.04 examples/s]31391 examples [00:33, 950.96 examples/s]31490 examples [00:33, 960.27 examples/s]31587 examples [00:33, 923.73 examples/s]31688 examples [00:33, 947.74 examples/s]31787 examples [00:34, 957.40 examples/s]31885 examples [00:34, 961.92 examples/s]31985 examples [00:34, 970.42 examples/s]32083 examples [00:34, 966.95 examples/s]32184 examples [00:34, 976.97 examples/s]32283 examples [00:34, 980.28 examples/s]32382 examples [00:34, 976.95 examples/s]32480 examples [00:34, 975.67 examples/s]32578 examples [00:34, 918.59 examples/s]32671 examples [00:35, 894.64 examples/s]32767 examples [00:35, 911.86 examples/s]32859 examples [00:35, 897.54 examples/s]32950 examples [00:35, 900.50 examples/s]33047 examples [00:35, 917.86 examples/s]33142 examples [00:35, 924.97 examples/s]33239 examples [00:35, 935.29 examples/s]33335 examples [00:35, 942.17 examples/s]33434 examples [00:35, 955.99 examples/s]33531 examples [00:35, 959.16 examples/s]33629 examples [00:36, 963.79 examples/s]33726 examples [00:36, 941.48 examples/s]33823 examples [00:36, 947.23 examples/s]33923 examples [00:36, 962.29 examples/s]34020 examples [00:36, 962.24 examples/s]34121 examples [00:36, 973.43 examples/s]34219 examples [00:36, 972.19 examples/s]34320 examples [00:36, 981.77 examples/s]34419 examples [00:36, 946.73 examples/s]34514 examples [00:36, 930.96 examples/s]34613 examples [00:37, 945.97 examples/s]34708 examples [00:37, 927.03 examples/s]34807 examples [00:37, 944.17 examples/s]34902 examples [00:37, 945.17 examples/s]35002 examples [00:37, 960.23 examples/s]35103 examples [00:37, 973.06 examples/s]35202 examples [00:37, 977.75 examples/s]35302 examples [00:37, 981.77 examples/s]35401 examples [00:37, 960.26 examples/s]35498 examples [00:37, 953.47 examples/s]35594 examples [00:38, 923.21 examples/s]35687 examples [00:38, 910.03 examples/s]35779 examples [00:38, 898.08 examples/s]35876 examples [00:38, 917.79 examples/s]35974 examples [00:38, 934.68 examples/s]36069 examples [00:38, 936.89 examples/s]36165 examples [00:38, 943.20 examples/s]36264 examples [00:38, 955.06 examples/s]36360 examples [00:38, 950.61 examples/s]36458 examples [00:39, 958.36 examples/s]36555 examples [00:39, 959.36 examples/s]36651 examples [00:39, 942.47 examples/s]36748 examples [00:39, 949.66 examples/s]36844 examples [00:39, 945.03 examples/s]36940 examples [00:39, 947.20 examples/s]37035 examples [00:39, 944.82 examples/s]37130 examples [00:39, 941.46 examples/s]37231 examples [00:39, 958.46 examples/s]37329 examples [00:39, 962.98 examples/s]37426 examples [00:40, 953.37 examples/s]37524 examples [00:40, 958.53 examples/s]37620 examples [00:40, 950.99 examples/s]37716 examples [00:40, 950.42 examples/s]37815 examples [00:40, 959.23 examples/s]37914 examples [00:40, 965.73 examples/s]38014 examples [00:40, 973.46 examples/s]38113 examples [00:40, 976.97 examples/s]38211 examples [00:40, 958.03 examples/s]38307 examples [00:40, 938.63 examples/s]38406 examples [00:41, 952.22 examples/s]38504 examples [00:41, 958.78 examples/s]38602 examples [00:41, 962.03 examples/s]38700 examples [00:41, 966.26 examples/s]38797 examples [00:41, 916.74 examples/s]38893 examples [00:41, 927.09 examples/s]38988 examples [00:41, 931.83 examples/s]39086 examples [00:41, 943.96 examples/s]39184 examples [00:41, 954.46 examples/s]39283 examples [00:41, 963.54 examples/s]39383 examples [00:42, 971.77 examples/s]39481 examples [00:42, 965.42 examples/s]39578 examples [00:42, 946.60 examples/s]39674 examples [00:42, 948.59 examples/s]39774 examples [00:42, 961.21 examples/s]39873 examples [00:42, 968.27 examples/s]39970 examples [00:42, 961.91 examples/s]40067 examples [00:42, 921.47 examples/s]40163 examples [00:42, 931.53 examples/s]40261 examples [00:42, 944.16 examples/s]40356 examples [00:43, 924.75 examples/s]40449 examples [00:43, 925.14 examples/s]40542 examples [00:43, 926.01 examples/s]40637 examples [00:43, 931.48 examples/s]40736 examples [00:43, 947.28 examples/s]40832 examples [00:43, 951.05 examples/s]40928 examples [00:43, 927.16 examples/s]41026 examples [00:43, 940.69 examples/s]41122 examples [00:43, 945.53 examples/s]41222 examples [00:44, 959.16 examples/s]41322 examples [00:44, 968.11 examples/s]41421 examples [00:44, 972.63 examples/s]41519 examples [00:44, 974.79 examples/s]41618 examples [00:44, 977.03 examples/s]41716 examples [00:44, 953.70 examples/s]41814 examples [00:44, 960.93 examples/s]41912 examples [00:44, 966.12 examples/s]42009 examples [00:44, 965.52 examples/s]42106 examples [00:44, 963.09 examples/s]42203 examples [00:45, 964.91 examples/s]42300 examples [00:45, 940.22 examples/s]42397 examples [00:45, 947.72 examples/s]42494 examples [00:45, 951.66 examples/s]42594 examples [00:45, 963.22 examples/s]42691 examples [00:45, 956.75 examples/s]42791 examples [00:45, 968.16 examples/s]42890 examples [00:45, 973.67 examples/s]42988 examples [00:45, 968.98 examples/s]43087 examples [00:45, 973.11 examples/s]43187 examples [00:46, 980.97 examples/s]43287 examples [00:46, 984.46 examples/s]43386 examples [00:46, 946.80 examples/s]43482 examples [00:46, 911.65 examples/s]43581 examples [00:46, 932.73 examples/s]43679 examples [00:46, 944.93 examples/s]43774 examples [00:46, 945.56 examples/s]43870 examples [00:46, 948.94 examples/s]43967 examples [00:46, 953.67 examples/s]44065 examples [00:46, 960.96 examples/s]44162 examples [00:47, 963.06 examples/s]44259 examples [00:47, 957.66 examples/s]44355 examples [00:47, 951.74 examples/s]44451 examples [00:47, 936.41 examples/s]44547 examples [00:47, 938.80 examples/s]44641 examples [00:47, 924.37 examples/s]44740 examples [00:47, 942.44 examples/s]44837 examples [00:47, 949.57 examples/s]44933 examples [00:47, 944.73 examples/s]45031 examples [00:48, 952.81 examples/s]45127 examples [00:48, 954.24 examples/s]45226 examples [00:48, 963.30 examples/s]45323 examples [00:48, 958.18 examples/s]45422 examples [00:48, 967.45 examples/s]45522 examples [00:48, 976.00 examples/s]45620 examples [00:48, 969.25 examples/s]45720 examples [00:48, 977.30 examples/s]45820 examples [00:48, 981.20 examples/s]45919 examples [00:48, 960.17 examples/s]46019 examples [00:49, 970.09 examples/s]46118 examples [00:49, 975.54 examples/s]46216 examples [00:49, 973.37 examples/s]46314 examples [00:49, 971.54 examples/s]46412 examples [00:49, 953.38 examples/s]46510 examples [00:49, 960.30 examples/s]46608 examples [00:49, 963.19 examples/s]46706 examples [00:49, 967.45 examples/s]46803 examples [00:49, 963.79 examples/s]46900 examples [00:49, 938.31 examples/s]46996 examples [00:50, 944.58 examples/s]47093 examples [00:50, 950.06 examples/s]47190 examples [00:50, 948.91 examples/s]47285 examples [00:50, 947.65 examples/s]47380 examples [00:50, 946.84 examples/s]47475 examples [00:50, 941.80 examples/s]47570 examples [00:50, 918.24 examples/s]47667 examples [00:50, 932.09 examples/s]47765 examples [00:50, 943.94 examples/s]47860 examples [00:50, 929.02 examples/s]47954 examples [00:51, 919.64 examples/s]48052 examples [00:51, 934.66 examples/s]48146 examples [00:51, 921.85 examples/s]48243 examples [00:51, 933.32 examples/s]48341 examples [00:51, 945.41 examples/s]48439 examples [00:51, 954.44 examples/s]48535 examples [00:51, 956.00 examples/s]48631 examples [00:51, 927.97 examples/s]48731 examples [00:51, 947.02 examples/s]48828 examples [00:51, 953.76 examples/s]48927 examples [00:52, 961.53 examples/s]49024 examples [00:52, 940.15 examples/s]49119 examples [00:52, 936.82 examples/s]49213 examples [00:52, 935.63 examples/s]49309 examples [00:52, 941.69 examples/s]49407 examples [00:52, 952.82 examples/s]49506 examples [00:52, 963.47 examples/s]49604 examples [00:52, 967.60 examples/s]49702 examples [00:52, 968.22 examples/s]49799 examples [00:53, 961.97 examples/s]49898 examples [00:53, 967.51 examples/s]49995 examples [00:53, 965.32 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6539/50000 [00:00<00:00, 65386.17 examples/s] 36%|███▌      | 18072/50000 [00:00<00:00, 75148.03 examples/s] 59%|█████▉    | 29714/50000 [00:00<00:00, 84090.14 examples/s] 82%|████████▏ | 41249/50000 [00:00<00:00, 91530.55 examples/s]                                                               0 examples [00:00, ? examples/s]82 examples [00:00, 819.07 examples/s]182 examples [00:00, 865.97 examples/s]283 examples [00:00, 902.72 examples/s]383 examples [00:00, 929.85 examples/s]483 examples [00:00, 949.15 examples/s]580 examples [00:00, 955.16 examples/s]681 examples [00:00, 969.10 examples/s]780 examples [00:00, 973.07 examples/s]879 examples [00:00, 975.91 examples/s]978 examples [00:01, 979.14 examples/s]1075 examples [00:01, 973.54 examples/s]1175 examples [00:01, 981.19 examples/s]1273 examples [00:01, 971.96 examples/s]1372 examples [00:01, 777.02 examples/s]1461 examples [00:01, 806.01 examples/s]1559 examples [00:01, 851.22 examples/s]1659 examples [00:01, 888.95 examples/s]1751 examples [00:01, 893.81 examples/s]1849 examples [00:01, 915.63 examples/s]1945 examples [00:02, 928.19 examples/s]2042 examples [00:02, 939.94 examples/s]2137 examples [00:02, 942.44 examples/s]2234 examples [00:02, 949.20 examples/s]2333 examples [00:02, 958.54 examples/s]2430 examples [00:02, 955.36 examples/s]2528 examples [00:02, 961.71 examples/s]2625 examples [00:02, 963.17 examples/s]2724 examples [00:02, 970.70 examples/s]2822 examples [00:03, 928.45 examples/s]2919 examples [00:03, 939.16 examples/s]3018 examples [00:03, 953.05 examples/s]3118 examples [00:03, 965.54 examples/s]3217 examples [00:03, 970.03 examples/s]3316 examples [00:03, 973.58 examples/s]3414 examples [00:03, 954.66 examples/s]3515 examples [00:03, 968.12 examples/s]3613 examples [00:03, 969.22 examples/s]3711 examples [00:03, 968.60 examples/s]3813 examples [00:04, 981.17 examples/s]3912 examples [00:04, 978.75 examples/s]4012 examples [00:04, 984.36 examples/s]4113 examples [00:04, 989.74 examples/s]4213 examples [00:04, 974.07 examples/s]4314 examples [00:04, 982.19 examples/s]4414 examples [00:04, 984.60 examples/s]4515 examples [00:04, 990.76 examples/s]4616 examples [00:04, 994.46 examples/s]4716 examples [00:04, 966.73 examples/s]4813 examples [00:05, 962.58 examples/s]4910 examples [00:05, 957.96 examples/s]5006 examples [00:05, 956.77 examples/s]5102 examples [00:05, 955.01 examples/s]5198 examples [00:05, 954.36 examples/s]5295 examples [00:05, 957.05 examples/s]5391 examples [00:05, 945.64 examples/s]5492 examples [00:05, 962.21 examples/s]5592 examples [00:05, 972.70 examples/s]5691 examples [00:05, 977.01 examples/s]5789 examples [00:06, 951.48 examples/s]5890 examples [00:06, 967.33 examples/s]5987 examples [00:06, 967.85 examples/s]6084 examples [00:06, 968.33 examples/s]6181 examples [00:06, 952.15 examples/s]6281 examples [00:06, 963.66 examples/s]6378 examples [00:06, 959.24 examples/s]6475 examples [00:06, 961.34 examples/s]6575 examples [00:06, 971.28 examples/s]6675 examples [00:06, 977.38 examples/s]6776 examples [00:07, 985.68 examples/s]6875 examples [00:07, 983.19 examples/s]6974 examples [00:07, 977.64 examples/s]7072 examples [00:07, 964.52 examples/s]7170 examples [00:07, 967.59 examples/s]7267 examples [00:07, 965.70 examples/s]7364 examples [00:07, 957.74 examples/s]7464 examples [00:07, 968.85 examples/s]7564 examples [00:07, 975.32 examples/s]7662 examples [00:08, 971.57 examples/s]7760 examples [00:08, 969.28 examples/s]7857 examples [00:08, 946.17 examples/s]7952 examples [00:08, 947.14 examples/s]8052 examples [00:08, 962.22 examples/s]8151 examples [00:08, 969.16 examples/s]8249 examples [00:08, 948.84 examples/s]8345 examples [00:08, 935.39 examples/s]8443 examples [00:08, 948.12 examples/s]8544 examples [00:08, 964.96 examples/s]8645 examples [00:09, 977.50 examples/s]8743 examples [00:09, 917.12 examples/s]8838 examples [00:09, 926.02 examples/s]8937 examples [00:09, 942.80 examples/s]9036 examples [00:09, 956.45 examples/s]9133 examples [00:09, 950.96 examples/s]9229 examples [00:09, 949.92 examples/s]9325 examples [00:09, 939.24 examples/s]9425 examples [00:09, 955.89 examples/s]9524 examples [00:09, 965.34 examples/s]9623 examples [00:10, 972.14 examples/s]9721 examples [00:10, 965.92 examples/s]9818 examples [00:10, 958.85 examples/s]9918 examples [00:10, 966.65 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM11CQJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM11CQJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0d92a5d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0d92a5d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0d92a5d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d1ccb3e48>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d1ccbbe80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f0d92a5d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f0d92a5d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f0d92a5d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d78e1c0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0d900694a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f0d0a4552f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f0d0a4552f0> 

  function with postional parmater data_info <function split_train_valid at 0x7f0d0a4552f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f0d0a455400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f0d0a455400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f0d0a455400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9389de372c8fb1469ff19ea2a942a88f416f6710dadbc866bca2279aaecd3e5f
  Stored in directory: /tmp/pip-ephem-wheel-cache-9keqqbos/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:05:17, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:41:34, 15.3kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:02:17, 21.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:44:04, 30.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:24:01, 44.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.38M/862M [00:01<3:45:24, 63.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:36:50, 90.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<1:49:09, 128kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<1:15:59, 183kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:01<52:55, 261kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.7M/862M [00:02<36:53, 372kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<25:44, 530kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:02<17:59, 754kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:02<13:07, 1.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<11:04, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<11:34, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<09:00, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<06:30, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<10:18, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<08:45, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<06:26, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:19, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<06:31, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<04:51, 2.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<06:26, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<06:48, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:11<05:26, 2.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:11<04:01, 3.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:05, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<07:01, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:13<05:14, 2.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<06:42, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<07:20, 1.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:14<05:47, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<06:09, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<05:37, 2.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:16<04:15, 3.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<06:01, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<06:50, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:18<05:25, 2.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:19<03:55, 3.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<12:28:12, 17.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<8:44:48, 24.5kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:20<6:06:55, 35.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<4:19:08, 49.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<3:04:04, 69.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<2:09:22, 98.9kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:22<1:30:21, 141kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<1:49:22, 116kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<1:17:50, 164kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<54:42, 232kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<41:09, 308kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<31:22, 404kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<22:34, 561kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<17:47, 709kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<13:43, 918kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<09:54, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:52, 1.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<09:27, 1.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:09, 1.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<05:13, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:43, 1.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:41, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:00, 2.49MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:22, 1.94MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:58, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:23, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:58, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:28, 1.65MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:29, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:51, 2.53MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:16, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:53, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<03:55, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:22, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:24, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:44, 2.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:10, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:46, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:21, 2.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:41, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:13, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:54, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:33, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:20, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:57, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:35, 3.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<15:06, 789kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<11:46, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<08:31, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<08:44, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:18, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:24, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:34, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:59, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:24, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:54, 3.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<12:22, 947kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:51, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:08, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:43, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:45, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:58, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:18, 2.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<1:26:41, 134kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<1:01:50, 187kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<43:29, 265kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<33:03, 348kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<25:27, 452kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<18:22, 625kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<12:56, 884kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:06:59, 17.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<7:47:48, 24.4kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<5:27:00, 34.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<3:50:51, 49.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<2:43:51, 69.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<1:55:03, 98.7kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<1:20:24, 141kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:02:59, 179kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<45:12, 250kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<31:51, 354kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<24:53, 451kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<19:42, 570kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<14:20, 782kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<11:48, 945kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:23, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<06:47, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:21, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:23, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:44, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:06, 2.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<10:42:19, 17.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<7:30:29, 24.5kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<5:14:51, 34.9kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<3:42:14, 49.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<2:37:46, 69.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<1:50:51, 98.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:19:00, 138kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<56:22, 193kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<39:38, 274kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<30:12, 358kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<22:16, 486kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<15:50, 681kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<13:28, 798kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:43, 917kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:46, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<06:14, 1.71MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<29:04, 367kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<21:18, 501kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<15:13, 700kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<10:44, 988kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<22:20, 475kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<17:54, 593kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<13:05, 810kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<09:15, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<39:55, 264kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<29:02, 363kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<20:31, 512kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<16:40, 628kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<13:54, 753kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<10:12, 1.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<07:16, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:17, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:37, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<05:33, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:12, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:33, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:03, 2.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:38, 2.82MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<12:10, 844kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<09:36, 1.07MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<06:58, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<07:09, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:05, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:29, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:25, 1.87MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:33, 2.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<04:43, 2.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:25, 1.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:15, 2.36MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:05, 3.23MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:51, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<06:22, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:39, 2.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:25, 2.89MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<17:27, 569kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<14:18, 693kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<10:31, 941kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<07:26, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<28:55, 341kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<21:17, 463kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<15:05, 651kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<12:45, 767kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<10:59, 890kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<08:07, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<05:47, 1.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<08:52, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:14, 1.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:16, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:52, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:08, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:44, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:26, 2.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:25, 1.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:03, 2.35MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:00, 1.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:32, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:17, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:09, 2.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:13, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:38, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:27, 2.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:32, 2.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:09, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:01, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<02:55, 3.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<06:47, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:44, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:15, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:02, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:29, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:04, 2.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<08:10, 1.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:41, 1.37MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:54, 1.86MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:28, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:45, 1.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:25, 2.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:12, 2.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:55, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:48, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:14, 2.12MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:00, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:23, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:10, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:00, 2.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:17, 1.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:35, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:50, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:21, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:22, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:14, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:04, 2.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:23, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<05:25, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:44, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:13, 2.05MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:10, 2.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:09, 2.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:43, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:41, 2.32MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:40, 3.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:25, 1.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:05, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:28, 1.90MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:02, 1.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:25, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:18, 2.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:11, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:42, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<03:40, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<02:41, 3.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:02, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:24, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:15, 2.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:10, 1.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<04:39, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:37, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:37, 3.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<06:41, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:32, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<04:03, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:39, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:58, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:54, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:49, 2.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<26:15, 307kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<19:13, 419kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<13:37, 589kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<11:17, 707kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<09:36, 831kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<07:04, 1.13MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<05:01, 1.58MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<09:01, 878kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<07:08, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<05:11, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:23, 1.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:27, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:10, 1.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<02:59, 2.61MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<23:53, 326kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<17:32, 443kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<12:26, 623kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:24, 741kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<08:56, 863kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<06:35, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<04:41, 1.63MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<07:50, 975kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:17, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:35, 1.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:54, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:58, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:51, 1.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<02:46, 2.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<55:39, 135kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<39:44, 189kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<27:53, 268kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<21:06, 352kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<16:21, 454kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<11:45, 631kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<08:19, 887kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:14, 893kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<06:32, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:45, 1.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:58, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<05:01, 1.45MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:49, 1.90MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:45, 2.63MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:37, 1.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:23, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:55, 1.83MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:21, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:33, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:30, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:31, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:07, 874kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:25, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:38, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:51, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:00, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:59, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:40, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:02, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:08, 2.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:16, 3.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<07:14, 952kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:47, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:13, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:28, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:49, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:50, 2.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:33, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:12, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:24, 2.78MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:11, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:49, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:02, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:20, 2.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:02, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:06, 2.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:17, 2.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<01:42, 3.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<16:48, 389kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<12:27, 524kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:07, 716kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<06:24, 1.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<16:03, 403kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<11:57, 541kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:49, 732kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<06:14, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<22:26, 285kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<16:35, 386kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<11:44, 543kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<08:17, 765kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<27:49, 228kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<20:04, 315kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<14:32, 435kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<10:10, 616kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<18:24, 340kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<13:37, 459kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<09:41, 643kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<08:08, 761kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:19, 979kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:32, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:16, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<20:21, 301kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<14:52, 412kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<10:32, 579kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<08:41, 697kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<06:35, 919kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<04:47, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<03:23, 1.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<26:33, 226kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<19:12, 312kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<13:32, 440kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<10:45, 551kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<08:08, 727kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<05:47, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:24, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:58, 1.18MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:46, 1.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:40, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<32:21, 179kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<23:14, 249kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<16:19, 352kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<12:39, 452kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<09:27, 604kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<06:44, 844kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:57, 947kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<05:21, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:59, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<02:51, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:43, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:52, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:49, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:14, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:26, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:41, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:51<01:56, 2.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<17:43, 307kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<12:58, 419kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<09:09, 591kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<07:36, 706kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:52, 913kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<04:13, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:10, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:02, 1.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:03, 1.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:15, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:52, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:34, 2.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:54, 2.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:30, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:45, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:41, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:39, 1.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:35, 1.98MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:24, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:22, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:20, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:17, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:15, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:12, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:12, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:03, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:43, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:29, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:20, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:07, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:03, 2.46MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<02:06, 2.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:56, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:54, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:53, 2.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:53, 2.68MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:51, 2.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:51, 2.71MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<05:00, 1.00MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:14, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:39, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:05, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:42, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:11, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:04, 2.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:59, 2.50MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:00, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:56, 2.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:53, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<04:22, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:46, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:12, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:55, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:35, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:15, 2.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:12, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:05, 2.36MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:54, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:55, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:53, 2.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:45, 2.79MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:48, 2.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:47, 2.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<05:57, 822kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:54, 996kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:56, 1.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:15, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:45, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:19, 2.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:11, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:01, 2.39MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:49, 2.66MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:49, 2.64MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:45, 2.75MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:37, 2.97MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:24, 890kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<05:01, 960kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<03:57, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:14, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:42, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:20, 2.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:04, 2.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:53, 2.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:44, 2.73MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:39, 2.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:35, 3.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<07:48, 609kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<06:25, 739kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:59, 949kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:53, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:02, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:33, 1.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:05, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:49, 2.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:37, 2.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:46, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<04:24, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:37, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:54, 1.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:20, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:54, 2.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:37, 2.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:26, 3.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:34, 1.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<04:35, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<04:43, 977kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:37, 1.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:46, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:16, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:47, 2.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:32, 2.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:18, 3.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<03:54, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:53, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:03, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:19, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:49, 2.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:27, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:25, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:22, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:32, 1.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:39, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:01, 2.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:33, 2.83MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:44, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:15, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<03:23, 1.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:31, 1.73MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:52, 2.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:47, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:58, 871kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:07, 1.05MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:04, 1.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:16, 1.89MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:40, 2.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<05:59, 712kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<05:24, 789kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:04, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:57, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<02:09, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<06:14, 673kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<06:24, 655kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:59, 841kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:36, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:36, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<03:50, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<04:31, 915kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:36, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:39, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:56, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:57, 1.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<05:14, 775kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:11, 969kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:04, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:15, 1.78MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:51, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<04:25, 902kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:36, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:40, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:57, 2.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:48, 1.40MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:04, 965kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:23, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:28, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:49, 2.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:53, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<04:06, 939kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:17, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:25, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:45, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:21, 2.79MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<48:36, 77.9kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<35:51, 106kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<25:31, 148kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<17:52, 210kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<12:29, 299kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<10:27, 355kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<08:57, 415kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<06:41, 555kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<04:46, 775kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:22, 1.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:03, 900kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:36, 791kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<03:38, 999kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:39, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:45<01:55, 1.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:25, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:58, 899kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:11, 1.12MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:20, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:42, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<03:00, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:39, 960kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:56, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:08, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:33, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:09, 827kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<04:26, 776kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:29, 985kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:31, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:49, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:51, 873kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:12, 802kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:18, 1.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<02:24, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:43, 1.92MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:03, 814kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:18, 768kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:21, 982kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:26, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:45, 1.85MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<04:04, 793kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<04:16, 756kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:19, 969kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<02:24, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:43, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:14, 746kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<04:21, 725kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<03:23, 934kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<02:26, 1.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:45, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:54, 794kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<04:06, 755kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<03:11, 968kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:18, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:39, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:58, 762kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:00, 755kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:07, 967kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:15, 1.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:37, 1.83MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<04:36, 642kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<04:25, 668kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<03:21, 880kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:25, 1.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:44, 1.66MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:09, 695kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<04:04, 709kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<03:09, 914kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<02:16, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:38, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<04:00, 703kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:57, 713kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<03:00, 933kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:11, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<01:34, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:24, 809kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:35, 765kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:48, 979kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:01, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:26, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:49, 950kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<03:09, 849kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:29, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:47, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<01:17, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:31, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:55, 893kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:19, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:40, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<01:12, 2.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<03:22, 755kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<03:23, 749kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:38, 961kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:54, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<01:21, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:31, 548kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:10, 594kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:10, 778kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:15, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<01:36, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:11, 573kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:54, 615kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:58, 806kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<02:07, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<01:30, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<04:11, 558kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<03:52, 603kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:54, 801kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:05, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<01:28, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:28, 654kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<03:16, 692kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:30, 902kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:48, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:16, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<09:18, 236kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<07:20, 300kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<05:17, 414kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<03:43, 582kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<02:35, 821kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<10:10, 209kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<07:54, 269kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<05:41, 373kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<03:59, 525kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:12, 644kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:57, 697kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:14, 914kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<01:35, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:34, 1.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:48, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:25, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:10, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:29, 1.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:11, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:51, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:03, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:20, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:04, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:47, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:59, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:16, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:00, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:44, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:57, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:09, 1.47MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:55, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:40, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:29, 3.38MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<03:13, 509kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:42, 607kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:58, 825kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:23, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:42<00:59, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<11:38, 135kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<08:32, 184kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<06:02, 259kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<04:10, 367kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<03:23, 445kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:43, 551kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:59, 749kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<01:23, 1.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:34, 912kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:27, 990kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:04, 1.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:45, 1.83MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<01:03, 1.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:04, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:49, 1.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:34, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:01, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<01:03, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:48, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:34, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:50, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:55, 1.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:42, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:30, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:47, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:51, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:39, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:27, 2.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:45, 1.44MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:48, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [05:58<00:36, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:26, 2.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:35, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:36, 1.68MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:27, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:19, 2.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<07:03, 135kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<05:08, 185kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<03:35, 261kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<02:24, 370kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:16, 389kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<01:44, 506kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:13, 704kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:49, 993kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:15, 644kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:04, 758kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:47, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:31, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:57, 784kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:49, 898kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:36, 1.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:24, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:34, 1.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:32, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:16, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:29, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:28, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:21, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:14, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:24, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:21, 1.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:15, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:15, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:17, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:14, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.82MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:10, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:04, 2.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:03, 3.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:14, 800kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:11, 997kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:07, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 2.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.64MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.13MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<32:31:29,  3.42it/s]  0%|          | 731/400000 [00:00<22:43:49,  4.88it/s]  0%|          | 1459/400000 [00:00<15:53:12,  6.97it/s]  1%|          | 2185/400000 [00:00<11:06:18,  9.95it/s]  1%|          | 2914/400000 [00:00<7:45:49, 14.21it/s]   1%|          | 3593/400000 [00:00<5:25:49, 20.28it/s]  1%|          | 4325/400000 [00:00<3:47:55, 28.93it/s]  1%|▏         | 5055/400000 [00:00<2:39:31, 41.26it/s]  1%|▏         | 5786/400000 [00:01<1:51:43, 58.81it/s]  2%|▏         | 6513/400000 [00:01<1:18:20, 83.72it/s]  2%|▏         | 7246/400000 [00:01<55:00, 119.01it/s]   2%|▏         | 7954/400000 [00:01<38:42, 168.80it/s]  2%|▏         | 8698/400000 [00:01<27:18, 238.82it/s]  2%|▏         | 9417/400000 [00:01<19:21, 336.38it/s]  3%|▎         | 10151/400000 [00:01<13:47, 471.28it/s]  3%|▎         | 10897/400000 [00:01<09:53, 655.46it/s]  3%|▎         | 11647/400000 [00:01<07:10, 902.54it/s]  3%|▎         | 12383/400000 [00:01<05:16, 1222.88it/s]  3%|▎         | 13116/400000 [00:02<03:57, 1630.34it/s]  3%|▎         | 13860/400000 [00:02<03:01, 2128.93it/s]  4%|▎         | 14594/400000 [00:02<02:22, 2703.10it/s]  4%|▍         | 15327/400000 [00:02<01:56, 3296.00it/s]  4%|▍         | 16080/400000 [00:02<01:36, 3964.65it/s]  4%|▍         | 16807/400000 [00:02<01:23, 4568.78it/s]  4%|▍         | 17543/400000 [00:02<01:14, 5154.32it/s]  5%|▍         | 18287/400000 [00:02<01:07, 5677.06it/s]  5%|▍         | 19032/400000 [00:02<01:02, 6112.52it/s]  5%|▍         | 19771/400000 [00:03<00:58, 6446.75it/s]  5%|▌         | 20508/400000 [00:03<00:58, 6492.70it/s]  5%|▌         | 21222/400000 [00:03<00:57, 6584.17it/s]  5%|▌         | 21960/400000 [00:03<00:55, 6802.73it/s]  6%|▌         | 22700/400000 [00:03<00:54, 6969.47it/s]  6%|▌         | 23436/400000 [00:03<00:53, 7081.95it/s]  6%|▌         | 24174/400000 [00:03<00:52, 7166.38it/s]  6%|▌         | 24910/400000 [00:03<00:51, 7222.93it/s]  6%|▋         | 25642/400000 [00:03<00:52, 7159.59it/s]  7%|▋         | 26393/400000 [00:03<00:51, 7260.94it/s]  7%|▋         | 27143/400000 [00:04<00:50, 7328.77it/s]  7%|▋         | 27887/400000 [00:04<00:50, 7361.18it/s]  7%|▋         | 28626/400000 [00:04<00:50, 7286.20it/s]  7%|▋         | 29357/400000 [00:04<00:50, 7290.85it/s]  8%|▊         | 30088/400000 [00:04<00:52, 7001.28it/s]  8%|▊         | 30837/400000 [00:04<00:51, 7138.75it/s]  8%|▊         | 31587/400000 [00:04<00:50, 7240.56it/s]  8%|▊         | 32314/400000 [00:04<00:51, 7186.10it/s]  8%|▊         | 33041/400000 [00:04<00:50, 7209.44it/s]  8%|▊         | 33775/400000 [00:04<00:50, 7245.31it/s]  9%|▊         | 34517/400000 [00:05<00:50, 7295.28it/s]  9%|▉         | 35248/400000 [00:05<00:50, 7269.83it/s]  9%|▉         | 35992/400000 [00:05<00:49, 7318.15it/s]  9%|▉         | 36740/400000 [00:05<00:49, 7364.68it/s]  9%|▉         | 37495/400000 [00:05<00:48, 7417.51it/s] 10%|▉         | 38238/400000 [00:05<00:49, 7326.04it/s] 10%|▉         | 38978/400000 [00:05<00:49, 7347.15it/s] 10%|▉         | 39720/400000 [00:05<00:48, 7368.23it/s] 10%|█         | 40458/400000 [00:05<00:49, 7318.68it/s] 10%|█         | 41215/400000 [00:05<00:48, 7391.13it/s] 10%|█         | 41955/400000 [00:06<00:48, 7338.63it/s] 11%|█         | 42690/400000 [00:06<00:48, 7307.73it/s] 11%|█         | 43422/400000 [00:06<00:49, 7144.04it/s] 11%|█         | 44163/400000 [00:06<00:49, 7221.25it/s] 11%|█         | 44895/400000 [00:06<00:48, 7248.22it/s] 11%|█▏        | 45660/400000 [00:06<00:48, 7362.10it/s] 12%|█▏        | 46422/400000 [00:06<00:47, 7436.27it/s] 12%|█▏        | 47167/400000 [00:06<00:48, 7321.01it/s] 12%|█▏        | 47925/400000 [00:06<00:47, 7395.73it/s] 12%|█▏        | 48689/400000 [00:06<00:47, 7466.19it/s] 12%|█▏        | 49448/400000 [00:07<00:46, 7502.61it/s] 13%|█▎        | 50208/400000 [00:07<00:46, 7530.01it/s] 13%|█▎        | 50962/400000 [00:07<00:46, 7488.14it/s] 13%|█▎        | 51712/400000 [00:07<00:46, 7416.52it/s] 13%|█▎        | 52474/400000 [00:07<00:46, 7475.84it/s] 13%|█▎        | 53239/400000 [00:07<00:46, 7526.28it/s] 13%|█▎        | 53992/400000 [00:07<00:46, 7402.66it/s] 14%|█▎        | 54733/400000 [00:07<00:46, 7376.63it/s] 14%|█▍        | 55472/400000 [00:07<00:47, 7297.99it/s] 14%|█▍        | 56203/400000 [00:07<00:47, 7288.47it/s] 14%|█▍        | 56933/400000 [00:08<00:47, 7272.85it/s] 14%|█▍        | 57673/400000 [00:08<00:46, 7308.23it/s] 15%|█▍        | 58405/400000 [00:08<00:47, 7264.49it/s] 15%|█▍        | 59132/400000 [00:08<00:47, 7246.07it/s] 15%|█▍        | 59857/400000 [00:08<00:46, 7246.52it/s] 15%|█▌        | 60594/400000 [00:08<00:46, 7281.78it/s] 15%|█▌        | 61347/400000 [00:08<00:46, 7353.14it/s] 16%|█▌        | 62089/400000 [00:08<00:45, 7371.44it/s] 16%|█▌        | 62838/400000 [00:08<00:45, 7404.53it/s] 16%|█▌        | 63579/400000 [00:09<00:45, 7376.89it/s] 16%|█▌        | 64317/400000 [00:09<00:46, 7226.74it/s] 16%|█▋        | 65060/400000 [00:09<00:45, 7285.51it/s] 16%|█▋        | 65795/400000 [00:09<00:45, 7303.63it/s] 17%|█▋        | 66526/400000 [00:09<00:46, 7195.64it/s] 17%|█▋        | 67262/400000 [00:09<00:45, 7243.36it/s] 17%|█▋        | 68001/400000 [00:09<00:45, 7286.24it/s] 17%|█▋        | 68731/400000 [00:09<00:46, 7162.33it/s] 17%|█▋        | 69458/400000 [00:09<00:45, 7189.41it/s] 18%|█▊        | 70211/400000 [00:09<00:45, 7288.19it/s] 18%|█▊        | 70968/400000 [00:10<00:44, 7369.02it/s] 18%|█▊        | 71706/400000 [00:10<00:44, 7337.76it/s] 18%|█▊        | 72452/400000 [00:10<00:44, 7371.89it/s] 18%|█▊        | 73190/400000 [00:10<00:44, 7302.47it/s] 18%|█▊        | 73948/400000 [00:10<00:44, 7381.39it/s] 19%|█▊        | 74704/400000 [00:10<00:43, 7433.48it/s] 19%|█▉        | 75453/400000 [00:10<00:43, 7449.58it/s] 19%|█▉        | 76213/400000 [00:10<00:43, 7493.34it/s] 19%|█▉        | 76963/400000 [00:10<00:43, 7493.88it/s] 19%|█▉        | 77713/400000 [00:10<00:43, 7385.88it/s] 20%|█▉        | 78453/400000 [00:11<00:43, 7388.91it/s] 20%|█▉        | 79193/400000 [00:11<00:43, 7335.13it/s] 20%|█▉        | 79933/400000 [00:11<00:43, 7351.82it/s] 20%|██        | 80669/400000 [00:11<00:43, 7348.69it/s] 20%|██        | 81412/400000 [00:11<00:43, 7370.82it/s] 21%|██        | 82150/400000 [00:11<00:43, 7269.09it/s] 21%|██        | 82878/400000 [00:11<00:44, 7201.55it/s] 21%|██        | 83641/400000 [00:11<00:43, 7323.64it/s] 21%|██        | 84382/400000 [00:11<00:42, 7346.30it/s] 21%|██▏       | 85139/400000 [00:11<00:42, 7411.67it/s] 21%|██▏       | 85881/400000 [00:12<00:42, 7322.21it/s] 22%|██▏       | 86626/400000 [00:12<00:42, 7357.39it/s] 22%|██▏       | 87379/400000 [00:12<00:42, 7405.51it/s] 22%|██▏       | 88120/400000 [00:12<00:42, 7400.24it/s] 22%|██▏       | 88865/400000 [00:12<00:41, 7413.79it/s] 22%|██▏       | 89617/400000 [00:12<00:41, 7443.27it/s] 23%|██▎       | 90362/400000 [00:12<00:42, 7251.56it/s] 23%|██▎       | 91122/400000 [00:12<00:42, 7351.82it/s] 23%|██▎       | 91859/400000 [00:12<00:42, 7228.63it/s] 23%|██▎       | 92584/400000 [00:12<00:43, 7146.37it/s] 23%|██▎       | 93327/400000 [00:13<00:42, 7227.12it/s] 24%|██▎       | 94070/400000 [00:13<00:41, 7285.87it/s] 24%|██▎       | 94800/400000 [00:13<00:42, 7257.96it/s] 24%|██▍       | 95533/400000 [00:13<00:41, 7276.88it/s] 24%|██▍       | 96291/400000 [00:13<00:41, 7364.01it/s] 24%|██▍       | 97033/400000 [00:13<00:41, 7379.77it/s] 24%|██▍       | 97781/400000 [00:13<00:40, 7406.87it/s] 25%|██▍       | 98540/400000 [00:13<00:40, 7458.87it/s] 25%|██▍       | 99287/400000 [00:13<00:41, 7278.81it/s] 25%|██▌       | 100045/400000 [00:13<00:40, 7366.21it/s] 25%|██▌       | 100796/400000 [00:14<00:40, 7407.40it/s] 25%|██▌       | 101545/400000 [00:14<00:40, 7429.26it/s] 26%|██▌       | 102309/400000 [00:14<00:39, 7488.71it/s] 26%|██▌       | 103059/400000 [00:14<00:40, 7399.65it/s] 26%|██▌       | 103803/400000 [00:14<00:39, 7408.93it/s] 26%|██▌       | 104564/400000 [00:14<00:39, 7465.93it/s] 26%|██▋       | 105329/400000 [00:14<00:39, 7519.87it/s] 27%|██▋       | 106090/400000 [00:14<00:38, 7546.15it/s] 27%|██▋       | 106845/400000 [00:14<00:39, 7481.22it/s] 27%|██▋       | 107594/400000 [00:14<00:39, 7404.38it/s] 27%|██▋       | 108335/400000 [00:15<00:39, 7392.69it/s] 27%|██▋       | 109087/400000 [00:15<00:39, 7427.86it/s] 27%|██▋       | 109834/400000 [00:15<00:39, 7437.88it/s] 28%|██▊       | 110578/400000 [00:15<00:39, 7398.70it/s] 28%|██▊       | 111325/400000 [00:15<00:38, 7418.67it/s] 28%|██▊       | 112067/400000 [00:15<00:39, 7350.94it/s] 28%|██▊       | 112828/400000 [00:15<00:38, 7425.09it/s] 28%|██▊       | 113591/400000 [00:15<00:38, 7484.09it/s] 29%|██▊       | 114342/400000 [00:15<00:38, 7491.75it/s] 29%|██▉       | 115092/400000 [00:15<00:38, 7491.43it/s] 29%|██▉       | 115854/400000 [00:16<00:37, 7527.97it/s] 29%|██▉       | 116607/400000 [00:16<00:38, 7423.25it/s] 29%|██▉       | 117372/400000 [00:16<00:37, 7487.41it/s] 30%|██▉       | 118122/400000 [00:16<00:37, 7471.64it/s] 30%|██▉       | 118870/400000 [00:16<00:37, 7450.39it/s] 30%|██▉       | 119625/400000 [00:16<00:37, 7479.53it/s] 30%|███       | 120374/400000 [00:16<00:37, 7428.72it/s] 30%|███       | 121119/400000 [00:16<00:37, 7432.30it/s] 30%|███       | 121880/400000 [00:16<00:37, 7484.52it/s] 31%|███       | 122629/400000 [00:17<00:37, 7421.67it/s] 31%|███       | 123380/400000 [00:17<00:37, 7445.21it/s] 31%|███       | 124128/400000 [00:17<00:37, 7455.24it/s] 31%|███       | 124874/400000 [00:17<00:37, 7384.40it/s] 31%|███▏      | 125622/400000 [00:17<00:37, 7411.26it/s] 32%|███▏      | 126373/400000 [00:17<00:36, 7440.52it/s] 32%|███▏      | 127126/400000 [00:17<00:36, 7465.28it/s] 32%|███▏      | 127883/400000 [00:17<00:36, 7494.17it/s] 32%|███▏      | 128633/400000 [00:17<00:36, 7493.12it/s] 32%|███▏      | 129383/400000 [00:17<00:36, 7394.89it/s] 33%|███▎      | 130137/400000 [00:18<00:36, 7436.64it/s] 33%|███▎      | 130881/400000 [00:18<00:36, 7422.48it/s] 33%|███▎      | 131624/400000 [00:18<00:36, 7415.19it/s] 33%|███▎      | 132376/400000 [00:18<00:35, 7444.24it/s] 33%|███▎      | 133121/400000 [00:18<00:35, 7424.13it/s] 33%|███▎      | 133864/400000 [00:18<00:36, 7218.28it/s] 34%|███▎      | 134623/400000 [00:18<00:36, 7323.50it/s] 34%|███▍      | 135372/400000 [00:18<00:35, 7370.87it/s] 34%|███▍      | 136134/400000 [00:18<00:35, 7442.36it/s] 34%|███▍      | 136890/400000 [00:18<00:35, 7475.00it/s] 34%|███▍      | 137639/400000 [00:19<00:35, 7397.19it/s] 35%|███▍      | 138384/400000 [00:19<00:35, 7412.90it/s] 35%|███▍      | 139136/400000 [00:19<00:35, 7443.47it/s] 35%|███▍      | 139881/400000 [00:19<00:35, 7382.09it/s] 35%|███▌      | 140620/400000 [00:19<00:35, 7306.64it/s] 35%|███▌      | 141363/400000 [00:19<00:35, 7342.63it/s] 36%|███▌      | 142098/400000 [00:19<00:35, 7326.28it/s] 36%|███▌      | 142857/400000 [00:19<00:34, 7401.86it/s] 36%|███▌      | 143607/400000 [00:19<00:34, 7430.49it/s] 36%|███▌      | 144355/400000 [00:19<00:34, 7444.74it/s] 36%|███▋      | 145100/400000 [00:20<00:34, 7432.02it/s] 36%|███▋      | 145844/400000 [00:20<00:34, 7337.43it/s] 37%|███▋      | 146579/400000 [00:20<00:34, 7323.78it/s] 37%|███▋      | 147329/400000 [00:20<00:34, 7374.29it/s] 37%|███▋      | 148075/400000 [00:20<00:34, 7399.47it/s] 37%|███▋      | 148819/400000 [00:20<00:33, 7411.12it/s] 37%|███▋      | 149579/400000 [00:20<00:33, 7465.60it/s] 38%|███▊      | 150326/400000 [00:20<00:33, 7463.40it/s] 38%|███▊      | 151073/400000 [00:20<00:33, 7419.69it/s] 38%|███▊      | 151821/400000 [00:20<00:33, 7435.20it/s] 38%|███▊      | 152565/400000 [00:21<00:33, 7348.61it/s] 38%|███▊      | 153302/400000 [00:21<00:33, 7352.38it/s] 39%|███▊      | 154064/400000 [00:21<00:33, 7428.59it/s] 39%|███▊      | 154808/400000 [00:21<00:33, 7368.88it/s] 39%|███▉      | 155554/400000 [00:21<00:33, 7394.84it/s] 39%|███▉      | 156294/400000 [00:21<00:33, 7354.11it/s] 39%|███▉      | 157030/400000 [00:21<00:33, 7205.90it/s] 39%|███▉      | 157762/400000 [00:21<00:33, 7238.51it/s] 40%|███▉      | 158520/400000 [00:21<00:32, 7336.73it/s] 40%|███▉      | 159255/400000 [00:21<00:33, 7262.95it/s] 40%|███▉      | 159982/400000 [00:22<00:33, 7200.00it/s] 40%|████      | 160709/400000 [00:22<00:33, 7220.43it/s] 40%|████      | 161432/400000 [00:22<00:33, 7191.62it/s] 41%|████      | 162182/400000 [00:22<00:32, 7277.98it/s] 41%|████      | 162933/400000 [00:22<00:32, 7345.62it/s] 41%|████      | 163669/400000 [00:22<00:32, 7236.84it/s] 41%|████      | 164422/400000 [00:22<00:32, 7320.73it/s] 41%|████▏     | 165184/400000 [00:22<00:31, 7406.95it/s] 41%|████▏     | 165926/400000 [00:22<00:31, 7386.66it/s] 42%|████▏     | 166684/400000 [00:22<00:31, 7442.27it/s] 42%|████▏     | 167429/400000 [00:23<00:31, 7397.05it/s] 42%|████▏     | 168170/400000 [00:23<00:32, 7220.55it/s] 42%|████▏     | 168912/400000 [00:23<00:31, 7278.69it/s] 42%|████▏     | 169667/400000 [00:23<00:31, 7357.23it/s] 43%|████▎     | 170415/400000 [00:23<00:31, 7391.07it/s] 43%|████▎     | 171155/400000 [00:23<00:31, 7321.03it/s] 43%|████▎     | 171888/400000 [00:23<00:31, 7307.46it/s] 43%|████▎     | 172647/400000 [00:23<00:30, 7389.88it/s] 43%|████▎     | 173397/400000 [00:23<00:30, 7420.22it/s] 44%|████▎     | 174140/400000 [00:23<00:30, 7334.43it/s] 44%|████▎     | 174877/400000 [00:24<00:30, 7344.21it/s] 44%|████▍     | 175635/400000 [00:24<00:30, 7410.60it/s] 44%|████▍     | 176377/400000 [00:24<00:30, 7269.50it/s] 44%|████▍     | 177105/400000 [00:24<00:30, 7259.14it/s] 44%|████▍     | 177852/400000 [00:24<00:30, 7319.07it/s] 45%|████▍     | 178595/400000 [00:24<00:30, 7351.19it/s] 45%|████▍     | 179341/400000 [00:24<00:29, 7381.15it/s] 45%|████▌     | 180098/400000 [00:24<00:29, 7434.22it/s] 45%|████▌     | 180842/400000 [00:24<00:29, 7332.09it/s] 45%|████▌     | 181576/400000 [00:25<00:29, 7326.87it/s] 46%|████▌     | 182310/400000 [00:25<00:30, 7245.77it/s] 46%|████▌     | 183036/400000 [00:25<00:30, 7151.83it/s] 46%|████▌     | 183790/400000 [00:25<00:29, 7261.37it/s] 46%|████▌     | 184535/400000 [00:25<00:29, 7314.05it/s] 46%|████▋     | 185268/400000 [00:25<00:29, 7279.17it/s] 46%|████▋     | 185997/400000 [00:25<00:29, 7173.93it/s] 47%|████▋     | 186742/400000 [00:25<00:29, 7253.55it/s] 47%|████▋     | 187491/400000 [00:25<00:29, 7321.75it/s] 47%|████▋     | 188248/400000 [00:25<00:28, 7392.14it/s] 47%|████▋     | 188988/400000 [00:26<00:28, 7361.62it/s] 47%|████▋     | 189725/400000 [00:26<00:28, 7337.21it/s] 48%|████▊     | 190483/400000 [00:26<00:28, 7406.13it/s] 48%|████▊     | 191244/400000 [00:26<00:27, 7466.03it/s] 48%|████▊     | 191998/400000 [00:26<00:27, 7486.07it/s] 48%|████▊     | 192755/400000 [00:26<00:27, 7509.99it/s] 48%|████▊     | 193507/400000 [00:26<00:27, 7400.36it/s] 49%|████▊     | 194272/400000 [00:26<00:27, 7472.74it/s] 49%|████▉     | 195033/400000 [00:26<00:27, 7511.55it/s] 49%|████▉     | 195791/400000 [00:26<00:27, 7529.28it/s] 49%|████▉     | 196548/400000 [00:27<00:26, 7540.56it/s] 49%|████▉     | 197303/400000 [00:27<00:27, 7502.22it/s] 50%|████▉     | 198054/400000 [00:27<00:27, 7447.42it/s] 50%|████▉     | 198810/400000 [00:27<00:26, 7480.71it/s] 50%|████▉     | 199566/400000 [00:27<00:26, 7503.64it/s] 50%|█████     | 200317/400000 [00:27<00:26, 7445.18it/s] 50%|█████     | 201062/400000 [00:27<00:26, 7370.46it/s] 50%|█████     | 201800/400000 [00:27<00:26, 7346.84it/s] 51%|█████     | 202535/400000 [00:27<00:27, 7294.07it/s] 51%|█████     | 203289/400000 [00:27<00:26, 7364.27it/s] 51%|█████     | 204033/400000 [00:28<00:26, 7386.62it/s] 51%|█████     | 204776/400000 [00:28<00:26, 7398.25it/s] 51%|█████▏    | 205519/400000 [00:28<00:26, 7407.18it/s] 52%|█████▏    | 206260/400000 [00:28<00:26, 7334.93it/s] 52%|█████▏    | 207015/400000 [00:28<00:26, 7396.76it/s] 52%|█████▏    | 207765/400000 [00:28<00:25, 7424.93it/s] 52%|█████▏    | 208510/400000 [00:28<00:25, 7429.89it/s] 52%|█████▏    | 209254/400000 [00:28<00:25, 7422.79it/s] 53%|█████▎    | 210011/400000 [00:28<00:25, 7463.86it/s] 53%|█████▎    | 210758/400000 [00:28<00:25, 7413.30it/s] 53%|█████▎    | 211500/400000 [00:29<00:25, 7284.91it/s] 53%|█████▎    | 212252/400000 [00:29<00:25, 7353.52it/s] 53%|█████▎    | 213005/400000 [00:29<00:25, 7403.53it/s] 53%|█████▎    | 213758/400000 [00:29<00:25, 7439.35it/s] 54%|█████▎    | 214503/400000 [00:29<00:24, 7439.12it/s] 54%|█████▍    | 215248/400000 [00:29<00:25, 7355.96it/s] 54%|█████▍    | 215984/400000 [00:29<00:25, 7353.23it/s] 54%|█████▍    | 216740/400000 [00:29<00:24, 7413.28it/s] 54%|█████▍    | 217482/400000 [00:29<00:24, 7401.60it/s] 55%|█████▍    | 218223/400000 [00:29<00:24, 7394.02it/s] 55%|█████▍    | 218963/400000 [00:30<00:24, 7261.35it/s] 55%|█████▍    | 219700/400000 [00:30<00:24, 7292.28it/s] 55%|█████▌    | 220453/400000 [00:30<00:24, 7359.89it/s] 55%|█████▌    | 221210/400000 [00:30<00:24, 7421.09it/s] 55%|█████▌    | 221965/400000 [00:30<00:23, 7458.82it/s] 56%|█████▌    | 222723/400000 [00:30<00:23, 7492.06it/s] 56%|█████▌    | 223473/400000 [00:30<00:24, 7284.82it/s] 56%|█████▌    | 224226/400000 [00:30<00:23, 7353.91it/s] 56%|█████▌    | 224963/400000 [00:30<00:23, 7343.93it/s] 56%|█████▋    | 225709/400000 [00:30<00:23, 7378.06it/s] 57%|█████▋    | 226466/400000 [00:31<00:23, 7432.81it/s] 57%|█████▋    | 227223/400000 [00:31<00:23, 7470.93it/s] 57%|█████▋    | 227971/400000 [00:31<00:23, 7374.28it/s] 57%|█████▋    | 228710/400000 [00:31<00:23, 7359.98it/s] 57%|█████▋    | 229457/400000 [00:31<00:23, 7391.17it/s] 58%|█████▊    | 230197/400000 [00:31<00:22, 7393.39it/s] 58%|█████▊    | 230937/400000 [00:31<00:22, 7351.77it/s] 58%|█████▊    | 231681/400000 [00:31<00:22, 7376.27it/s] 58%|█████▊    | 232419/400000 [00:31<00:22, 7326.68it/s] 58%|█████▊    | 233168/400000 [00:31<00:22, 7372.70it/s] 58%|█████▊    | 233906/400000 [00:32<00:22, 7368.13it/s] 59%|█████▊    | 234643/400000 [00:32<00:22, 7360.30it/s] 59%|█████▉    | 235400/400000 [00:32<00:22, 7419.22it/s] 59%|█████▉    | 236143/400000 [00:32<00:22, 7252.92it/s] 59%|█████▉    | 236900/400000 [00:32<00:22, 7343.86it/s] 59%|█████▉    | 237636/400000 [00:32<00:22, 7322.34it/s] 60%|█████▉    | 238390/400000 [00:32<00:21, 7385.83it/s] 60%|█████▉    | 239136/400000 [00:32<00:21, 7406.07it/s] 60%|█████▉    | 239883/400000 [00:32<00:21, 7422.68it/s] 60%|██████    | 240626/400000 [00:33<00:21, 7360.48it/s] 60%|██████    | 241383/400000 [00:33<00:21, 7420.60it/s] 61%|██████    | 242126/400000 [00:33<00:21, 7392.62it/s] 61%|██████    | 242866/400000 [00:33<00:21, 7386.30it/s] 61%|██████    | 243620/400000 [00:33<00:21, 7431.54it/s] 61%|██████    | 244375/400000 [00:33<00:20, 7464.29it/s] 61%|██████▏   | 245122/400000 [00:33<00:21, 7366.16it/s] 61%|██████▏   | 245860/400000 [00:33<00:20, 7368.27it/s] 62%|██████▏   | 246613/400000 [00:33<00:20, 7415.07it/s] 62%|██████▏   | 247368/400000 [00:33<00:20, 7454.36it/s] 62%|██████▏   | 248120/400000 [00:34<00:20, 7471.76it/s] 62%|██████▏   | 248868/400000 [00:34<00:20, 7374.87it/s] 62%|██████▏   | 249606/400000 [00:34<00:20, 7281.47it/s] 63%|██████▎   | 250335/400000 [00:34<00:20, 7270.96it/s] 63%|██████▎   | 251074/400000 [00:34<00:20, 7304.73it/s] 63%|██████▎   | 251822/400000 [00:34<00:20, 7355.57it/s] 63%|██████▎   | 252572/400000 [00:34<00:19, 7396.60it/s] 63%|██████▎   | 253312/400000 [00:34<00:20, 7295.80it/s] 64%|██████▎   | 254064/400000 [00:34<00:19, 7360.10it/s] 64%|██████▎   | 254822/400000 [00:34<00:19, 7422.27it/s] 64%|██████▍   | 255565/400000 [00:35<00:20, 7213.83it/s] 64%|██████▍   | 256288/400000 [00:35<00:19, 7217.45it/s] 64%|██████▍   | 257011/400000 [00:35<00:19, 7183.82it/s] 64%|██████▍   | 257731/400000 [00:35<00:20, 7056.76it/s] 65%|██████▍   | 258482/400000 [00:35<00:19, 7184.59it/s] 65%|██████▍   | 259236/400000 [00:35<00:19, 7287.52it/s] 65%|██████▍   | 259978/400000 [00:35<00:19, 7326.55it/s] 65%|██████▌   | 260722/400000 [00:35<00:18, 7358.14it/s] 65%|██████▌   | 261480/400000 [00:35<00:18, 7422.14it/s] 66%|██████▌   | 262223/400000 [00:35<00:18, 7362.26it/s] 66%|██████▌   | 262960/400000 [00:36<00:18, 7315.74it/s] 66%|██████▌   | 263703/400000 [00:36<00:18, 7347.84it/s] 66%|██████▌   | 264440/400000 [00:36<00:18, 7353.30it/s] 66%|██████▋   | 265198/400000 [00:36<00:18, 7418.08it/s] 66%|██████▋   | 265949/400000 [00:36<00:18, 7442.52it/s] 67%|██████▋   | 266694/400000 [00:36<00:18, 7388.80it/s] 67%|██████▋   | 267452/400000 [00:36<00:17, 7442.64it/s] 67%|██████▋   | 268197/400000 [00:36<00:17, 7375.38it/s] 67%|██████▋   | 268945/400000 [00:36<00:17, 7404.61it/s] 67%|██████▋   | 269686/400000 [00:36<00:18, 7196.87it/s] 68%|██████▊   | 270408/400000 [00:37<00:18, 7047.18it/s] 68%|██████▊   | 271142/400000 [00:37<00:18, 7130.84it/s] 68%|██████▊   | 271880/400000 [00:37<00:17, 7203.68it/s] 68%|██████▊   | 272624/400000 [00:37<00:17, 7271.07it/s] 68%|██████▊   | 273365/400000 [00:37<00:17, 7312.17it/s] 69%|██████▊   | 274120/400000 [00:37<00:17, 7380.36it/s] 69%|██████▊   | 274859/400000 [00:37<00:17, 7291.53it/s] 69%|██████▉   | 275589/400000 [00:37<00:17, 7272.64it/s] 69%|██████▉   | 276349/400000 [00:37<00:16, 7366.47it/s] 69%|██████▉   | 277104/400000 [00:37<00:16, 7418.40it/s] 69%|██████▉   | 277847/400000 [00:38<00:16, 7246.41it/s] 70%|██████▉   | 278601/400000 [00:38<00:16, 7331.91it/s] 70%|██████▉   | 279336/400000 [00:38<00:16, 7261.43it/s] 70%|███████   | 280064/400000 [00:38<00:16, 7192.24it/s] 70%|███████   | 280806/400000 [00:38<00:16, 7257.16it/s] 70%|███████   | 281558/400000 [00:38<00:16, 7333.64it/s] 71%|███████   | 282311/400000 [00:38<00:15, 7390.63it/s] 71%|███████   | 283059/400000 [00:38<00:15, 7414.13it/s] 71%|███████   | 283801/400000 [00:38<00:15, 7383.59it/s] 71%|███████   | 284556/400000 [00:38<00:15, 7430.22it/s] 71%|███████▏  | 285300/400000 [00:39<00:15, 7423.54it/s] 72%|███████▏  | 286043/400000 [00:39<00:15, 7409.42it/s] 72%|███████▏  | 286785/400000 [00:39<00:15, 7285.81it/s] 72%|███████▏  | 287515/400000 [00:39<00:15, 7226.13it/s] 72%|███████▏  | 288268/400000 [00:39<00:15, 7312.57it/s] 72%|███████▏  | 289023/400000 [00:39<00:15, 7381.97it/s] 72%|███████▏  | 289788/400000 [00:39<00:14, 7460.30it/s] 73%|███████▎  | 290535/400000 [00:39<00:14, 7443.87it/s] 73%|███████▎  | 291299/400000 [00:39<00:14, 7500.56it/s] 73%|███████▎  | 292050/400000 [00:40<00:14, 7402.08it/s] 73%|███████▎  | 292791/400000 [00:40<00:14, 7397.86it/s] 73%|███████▎  | 293555/400000 [00:40<00:14, 7466.45it/s] 74%|███████▎  | 294310/400000 [00:40<00:14, 7489.68it/s] 74%|███████▍  | 295073/400000 [00:40<00:13, 7529.34it/s] 74%|███████▍  | 295827/400000 [00:40<00:13, 7524.61it/s] 74%|███████▍  | 296580/400000 [00:40<00:13, 7444.46it/s] 74%|███████▍  | 297346/400000 [00:40<00:13, 7507.26it/s] 75%|███████▍  | 298098/400000 [00:40<00:13, 7423.96it/s] 75%|███████▍  | 298855/400000 [00:40<00:13, 7465.52it/s] 75%|███████▍  | 299615/400000 [00:41<00:13, 7504.71it/s] 75%|███████▌  | 300366/400000 [00:41<00:13, 7343.65it/s] 75%|███████▌  | 301116/400000 [00:41<00:13, 7387.05it/s] 75%|███████▌  | 301856/400000 [00:41<00:13, 7380.42it/s] 76%|███████▌  | 302599/400000 [00:41<00:13, 7394.64it/s] 76%|███████▌  | 303339/400000 [00:41<00:13, 7325.75it/s] 76%|███████▌  | 304072/400000 [00:41<00:13, 7319.93it/s] 76%|███████▌  | 304805/400000 [00:41<00:13, 7211.56it/s] 76%|███████▋  | 305527/400000 [00:41<00:13, 7190.76it/s] 77%|███████▋  | 306273/400000 [00:41<00:12, 7269.02it/s] 77%|███████▋  | 307028/400000 [00:42<00:12, 7348.55it/s] 77%|███████▋  | 307772/400000 [00:42<00:12, 7374.70it/s] 77%|███████▋  | 308522/400000 [00:42<00:12, 7409.03it/s] 77%|███████▋  | 309264/400000 [00:42<00:12, 7250.07it/s] 78%|███████▊  | 310017/400000 [00:42<00:12, 7331.33it/s] 78%|███████▊  | 310770/400000 [00:42<00:12, 7389.34it/s] 78%|███████▊  | 311535/400000 [00:42<00:11, 7465.63it/s] 78%|███████▊  | 312285/400000 [00:42<00:11, 7473.80it/s] 78%|███████▊  | 313033/400000 [00:42<00:11, 7268.10it/s] 78%|███████▊  | 313787/400000 [00:42<00:11, 7346.33it/s] 79%|███████▊  | 314528/400000 [00:43<00:11, 7364.23it/s] 79%|███████▉  | 315282/400000 [00:43<00:11, 7416.05it/s] 79%|███████▉  | 316036/400000 [00:43<00:11, 7451.64it/s] 79%|███████▉  | 316782/400000 [00:43<00:11, 7309.31it/s] 79%|███████▉  | 317514/400000 [00:43<00:11, 7178.15it/s] 80%|███████▉  | 318270/400000 [00:43<00:11, 7287.36it/s] 80%|███████▉  | 319027/400000 [00:43<00:10, 7369.23it/s] 80%|███████▉  | 319789/400000 [00:43<00:10, 7439.95it/s] 80%|████████  | 320534/400000 [00:43<00:11, 7181.93it/s] 80%|████████  | 321255/400000 [00:43<00:11, 7094.27it/s] 80%|████████  | 321967/400000 [00:44<00:11, 6776.81it/s] 81%|████████  | 322718/400000 [00:44<00:11, 6978.30it/s] 81%|████████  | 323460/400000 [00:44<00:10, 7104.96it/s] 81%|████████  | 324208/400000 [00:44<00:10, 7212.33it/s] 81%|████████  | 324955/400000 [00:44<00:10, 7285.86it/s] 81%|████████▏ | 325686/400000 [00:44<00:10, 7274.58it/s] 82%|████████▏ | 326427/400000 [00:44<00:10, 7312.94it/s] 82%|████████▏ | 327167/400000 [00:44<00:09, 7336.72it/s] 82%|████████▏ | 327902/400000 [00:44<00:09, 7251.10it/s] 82%|████████▏ | 328647/400000 [00:45<00:09, 7309.61it/s] 82%|████████▏ | 329379/400000 [00:45<00:09, 7292.48it/s] 83%|████████▎ | 330111/400000 [00:45<00:09, 7300.11it/s] 83%|████████▎ | 330847/400000 [00:45<00:09, 7316.48it/s] 83%|████████▎ | 331579/400000 [00:45<00:09, 7275.69it/s] 83%|████████▎ | 332338/400000 [00:45<00:09, 7366.80it/s] 83%|████████▎ | 333094/400000 [00:45<00:09, 7422.99it/s] 83%|████████▎ | 333854/400000 [00:45<00:08, 7474.57it/s] 84%|████████▎ | 334602/400000 [00:45<00:08, 7369.67it/s] 84%|████████▍ | 335340/400000 [00:45<00:08, 7291.34it/s] 84%|████████▍ | 336090/400000 [00:46<00:08, 7351.75it/s] 84%|████████▍ | 336848/400000 [00:46<00:08, 7417.99it/s] 84%|████████▍ | 337603/400000 [00:46<00:08, 7456.54it/s] 85%|████████▍ | 338361/400000 [00:46<00:08, 7490.33it/s] 85%|████████▍ | 339111/400000 [00:46<00:08, 7331.26it/s] 85%|████████▍ | 339848/400000 [00:46<00:08, 7340.15it/s] 85%|████████▌ | 340601/400000 [00:46<00:08, 7395.18it/s] 85%|████████▌ | 341342/400000 [00:46<00:08, 7324.84it/s] 86%|████████▌ | 342077/400000 [00:46<00:07, 7330.84it/s] 86%|████████▌ | 342811/400000 [00:46<00:07, 7260.80it/s] 86%|████████▌ | 343561/400000 [00:47<00:07, 7330.47it/s] 86%|████████▌ | 344306/400000 [00:47<00:07, 7363.29it/s] 86%|████████▋ | 345050/400000 [00:47<00:07, 7383.79it/s] 86%|████████▋ | 345811/400000 [00:47<00:07, 7447.52it/s] 87%|████████▋ | 346557/400000 [00:47<00:07, 7380.37it/s] 87%|████████▋ | 347296/400000 [00:47<00:07, 7371.80it/s] 87%|████████▋ | 348043/400000 [00:47<00:07, 7398.72it/s] 87%|████████▋ | 348790/400000 [00:47<00:06, 7419.93it/s] 87%|████████▋ | 349533/400000 [00:47<00:06, 7394.83it/s] 88%|████████▊ | 350273/400000 [00:47<00:06, 7379.59it/s] 88%|████████▊ | 351028/400000 [00:48<00:06, 7429.11it/s] 88%|████████▊ | 351772/400000 [00:48<00:06, 7363.65it/s] 88%|████████▊ | 352534/400000 [00:48<00:06, 7435.84it/s] 88%|████████▊ | 353291/400000 [00:48<00:06, 7474.81it/s] 89%|████████▊ | 354039/400000 [00:48<00:06, 7431.66it/s] 89%|████████▊ | 354793/400000 [00:48<00:06, 7463.04it/s] 89%|████████▉ | 355542/400000 [00:48<00:05, 7470.86it/s] 89%|████████▉ | 356290/400000 [00:48<00:05, 7462.08it/s] 89%|████████▉ | 357041/400000 [00:48<00:05, 7473.34it/s] 89%|████████▉ | 357789/400000 [00:48<00:05, 7392.69it/s] 90%|████████▉ | 358529/400000 [00:49<00:05, 7389.39it/s] 90%|████████▉ | 359281/400000 [00:49<00:05, 7427.55it/s] 90%|█████████ | 360024/400000 [00:49<00:05, 7379.76it/s] 90%|█████████ | 360772/400000 [00:49<00:05, 7409.37it/s] 90%|█████████ | 361514/400000 [00:49<00:05, 7290.75it/s] 91%|█████████ | 362271/400000 [00:49<00:05, 7370.90it/s] 91%|█████████ | 363009/400000 [00:49<00:05, 7242.29it/s] 91%|█████████ | 363759/400000 [00:49<00:04, 7316.69it/s] 91%|█████████ | 364492/400000 [00:49<00:04, 7189.72it/s] 91%|█████████▏| 365218/400000 [00:49<00:04, 7210.27it/s] 91%|█████████▏| 365940/400000 [00:50<00:04, 7092.39it/s] 92%|█████████▏| 366688/400000 [00:50<00:04, 7204.37it/s] 92%|█████████▏| 367434/400000 [00:50<00:04, 7278.50it/s] 92%|█████████▏| 368173/400000 [00:50<00:04, 7310.85it/s] 92%|█████████▏| 368905/400000 [00:50<00:04, 7272.35it/s] 92%|█████████▏| 369637/400000 [00:50<00:04, 7284.69it/s] 93%|█████████▎| 370376/400000 [00:50<00:04, 7314.77it/s] 93%|█████████▎| 371128/400000 [00:50<00:03, 7373.20it/s] 93%|█████████▎| 371878/400000 [00:50<00:03, 7410.41it/s] 93%|█████████▎| 372620/400000 [00:50<00:03, 7268.58it/s] 93%|█████████▎| 373382/400000 [00:51<00:03, 7369.10it/s] 94%|█████████▎| 374141/400000 [00:51<00:03, 7432.06it/s] 94%|█████████▎| 374896/400000 [00:51<00:03, 7464.73it/s] 94%|█████████▍| 375647/400000 [00:51<00:03, 7476.31it/s] 94%|█████████▍| 376396/400000 [00:51<00:03, 7439.88it/s] 94%|█████████▍| 377141/400000 [00:51<00:03, 7390.05it/s] 94%|█████████▍| 377897/400000 [00:51<00:02, 7437.50it/s] 95%|█████████▍| 378653/400000 [00:51<00:02, 7473.77it/s] 95%|█████████▍| 379408/400000 [00:51<00:02, 7495.67it/s] 95%|█████████▌| 380158/400000 [00:51<00:02, 7483.11it/s] 95%|█████████▌| 380907/400000 [00:52<00:02, 7478.54it/s] 95%|█████████▌| 381655/400000 [00:52<00:02, 7443.58it/s] 96%|█████████▌| 382412/400000 [00:52<00:02, 7479.45it/s] 96%|█████████▌| 383165/400000 [00:52<00:02, 7491.61it/s] 96%|█████████▌| 383915/400000 [00:52<00:02, 7378.74it/s] 96%|█████████▌| 384671/400000 [00:52<00:02, 7430.70it/s] 96%|█████████▋| 385415/400000 [00:52<00:01, 7321.04it/s] 97%|█████████▋| 386166/400000 [00:52<00:01, 7374.42it/s] 97%|█████████▋| 386918/400000 [00:52<00:01, 7417.18it/s] 97%|█████████▋| 387661/400000 [00:53<00:01, 7402.35it/s] 97%|█████████▋| 388405/400000 [00:53<00:01, 7412.81it/s] 97%|█████████▋| 389149/400000 [00:53<00:01, 7419.54it/s] 97%|█████████▋| 389892/400000 [00:53<00:01, 7278.25it/s] 98%|█████████▊| 390641/400000 [00:53<00:01, 7338.73it/s] 98%|█████████▊| 391376/400000 [00:53<00:01, 7266.15it/s] 98%|█████████▊| 392116/400000 [00:53<00:01, 7305.05it/s] 98%|█████████▊| 392866/400000 [00:53<00:00, 7361.62it/s] 98%|█████████▊| 393603/400000 [00:53<00:00, 7356.25it/s] 99%|█████████▊| 394339/400000 [00:53<00:00, 7311.97it/s] 99%|█████████▉| 395072/400000 [00:54<00:00, 7316.90it/s] 99%|█████████▉| 395815/400000 [00:54<00:00, 7348.71it/s] 99%|█████████▉| 396551/400000 [00:54<00:00, 7185.04it/s] 99%|█████████▉| 397308/400000 [00:54<00:00, 7294.74it/s]100%|█████████▉| 398051/400000 [00:54<00:00, 7333.58it/s]100%|█████████▉| 398786/400000 [00:54<00:00, 7327.25it/s]100%|█████████▉| 399537/400000 [00:54<00:00, 7380.71it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7313.93it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f0dfec900f0>, <torchtext.data.dataset.TabularDataset object at 0x7f0d14059e80>, <torchtext.vocab.Vocab object at 0x7f0dfec90160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.10 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 22.03 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.60 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.60 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.87 file/s]2020-06-14 00:19:17.497404: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 00:19:17.501803: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-06-14 00:19:17.502057: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ca7ed58f00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 00:19:17.502120: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▊       | 2826240/9912422 [00:00<00:00, 28201174.13it/s]9920512it [00:00, 32144203.77it/s]                             
0it [00:00, ?it/s]32768it [00:00, 563810.48it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 154698.34it/s]1654784it [00:00, 11101887.86it/s]                         
0it [00:00, ?it/s]8192it [00:00, 210139.74it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
