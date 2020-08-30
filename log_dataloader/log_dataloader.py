
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '76c59fd9a4bb974b18235d07ef6a03bf2361dc5c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/76c59fd9a4bb974b18235d07ef6a03bf2361dc5c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f00e6642620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f00e6642620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f0151367400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f0151367400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f0170584ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f0170584ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f00fe69e400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f00fe69e400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f00fe69e400> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3284992/9912422 [00:00<00:00, 32822092.89it/s]9920512it [00:00, 33182883.17it/s]                             
0it [00:00, ?it/s]32768it [00:00, 566618.38it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 456316.75it/s]1654784it [00:00, 11745666.71it/s]                         
0it [00:00, ?it/s]8192it [00:00, 231861.17it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f00fcb9af60>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f00e6552ba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f00fe69e0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f00fe69e0d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f00fe69e0d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:07,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:46,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:45,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:44,  3.22 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:30,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:29,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:29,  4.53 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:20,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:19,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:19,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:19,  6.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  8.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:13,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:12,  8.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:09, 12.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:09, 12.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:09, 12.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:09, 12.11 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 12.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 12.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:08, 12.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:08, 12.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:08, 12.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 16.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 21.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 28.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 28.12 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 35.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 43.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 43.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 51.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 59.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 66.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 73.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.46 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.46 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.57 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.43s/ url]
0 examples [00:00, ? examples/s]2020-08-30 18:07:15.326105: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-30 18:07:15.345577: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-30 18:07:15.345971: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5612980903a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-30 18:07:15.345995: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.85 examples/s]104 examples [00:00,  4.07 examples/s]207 examples [00:00,  5.80 examples/s]308 examples [00:00,  8.27 examples/s]411 examples [00:00, 11.77 examples/s]514 examples [00:00, 16.73 examples/s]617 examples [00:00, 23.73 examples/s]722 examples [00:01, 33.58 examples/s]822 examples [00:01, 47.28 examples/s]922 examples [00:01, 66.20 examples/s]1020 examples [00:01, 91.80 examples/s]1120 examples [00:01, 126.15 examples/s]1218 examples [00:01, 168.86 examples/s]1323 examples [00:01, 225.63 examples/s]1428 examples [00:01, 295.13 examples/s]1533 examples [00:01, 376.16 examples/s]1634 examples [00:01, 462.74 examples/s]1735 examples [00:02, 546.72 examples/s]1835 examples [00:02, 632.06 examples/s]1937 examples [00:02, 713.16 examples/s]2037 examples [00:02, 773.05 examples/s]2142 examples [00:02, 839.16 examples/s]2243 examples [00:02, 882.28 examples/s]2344 examples [00:02, 891.40 examples/s]2443 examples [00:02, 917.70 examples/s]2542 examples [00:02, 937.26 examples/s]2641 examples [00:03, 951.00 examples/s]2740 examples [00:03, 954.25 examples/s]2841 examples [00:03, 968.76 examples/s]2946 examples [00:03, 989.14 examples/s]3049 examples [00:03, 999.75 examples/s]3150 examples [00:03, 974.99 examples/s]3249 examples [00:03, 963.79 examples/s]3348 examples [00:03, 968.83 examples/s]3453 examples [00:03, 991.15 examples/s]3558 examples [00:03, 1007.36 examples/s]3660 examples [00:04, 1009.31 examples/s]3762 examples [00:04, 1001.59 examples/s]3863 examples [00:04, 999.26 examples/s] 3964 examples [00:04, 952.63 examples/s]4060 examples [00:04, 951.01 examples/s]4156 examples [00:04, 946.67 examples/s]4258 examples [00:04, 966.37 examples/s]4355 examples [00:04, 953.27 examples/s]4452 examples [00:04, 955.28 examples/s]4553 examples [00:04, 968.81 examples/s]4653 examples [00:05, 976.79 examples/s]4752 examples [00:05, 980.51 examples/s]4851 examples [00:05, 975.21 examples/s]4949 examples [00:05, 943.67 examples/s]5044 examples [00:05, 940.96 examples/s]5143 examples [00:05, 952.68 examples/s]5241 examples [00:05, 957.61 examples/s]5339 examples [00:05, 962.45 examples/s]5436 examples [00:05, 949.43 examples/s]5538 examples [00:05, 967.33 examples/s]5637 examples [00:06, 970.62 examples/s]5735 examples [00:06, 971.31 examples/s]5833 examples [00:06, 960.96 examples/s]5930 examples [00:06, 956.94 examples/s]6031 examples [00:06, 970.18 examples/s]6131 examples [00:06, 978.05 examples/s]6231 examples [00:06, 984.51 examples/s]6333 examples [00:06, 994.47 examples/s]6436 examples [00:06, 1002.36 examples/s]6537 examples [00:07, 992.08 examples/s] 6637 examples [00:07, 973.62 examples/s]6741 examples [00:07, 992.14 examples/s]6841 examples [00:07, 988.61 examples/s]6942 examples [00:07, 993.69 examples/s]7042 examples [00:07, 986.75 examples/s]7141 examples [00:07, 984.95 examples/s]7244 examples [00:07, 995.53 examples/s]7344 examples [00:07, 919.66 examples/s]7445 examples [00:07, 942.20 examples/s]7541 examples [00:08, 927.07 examples/s]7637 examples [00:08, 936.57 examples/s]7732 examples [00:08, 925.31 examples/s]7825 examples [00:08, 926.38 examples/s]7928 examples [00:08, 954.61 examples/s]8026 examples [00:08, 960.98 examples/s]8129 examples [00:08, 979.61 examples/s]8229 examples [00:08, 984.52 examples/s]8331 examples [00:08, 992.49 examples/s]8433 examples [00:08, 998.80 examples/s]8534 examples [00:09, 997.69 examples/s]8634 examples [00:09, 968.67 examples/s]8732 examples [00:09, 961.08 examples/s]8833 examples [00:09, 974.65 examples/s]8936 examples [00:09, 989.89 examples/s]9036 examples [00:09, 973.48 examples/s]9135 examples [00:09, 977.69 examples/s]9233 examples [00:09, 957.14 examples/s]9334 examples [00:09, 970.70 examples/s]9433 examples [00:09, 972.02 examples/s]9531 examples [00:10, 969.01 examples/s]9628 examples [00:10, 947.88 examples/s]9723 examples [00:10, 913.45 examples/s]9822 examples [00:10, 933.65 examples/s]9916 examples [00:10, 918.94 examples/s]10009 examples [00:10, 867.87 examples/s]10107 examples [00:10, 898.25 examples/s]10205 examples [00:10, 920.85 examples/s]10305 examples [00:10, 942.46 examples/s]10406 examples [00:11, 960.59 examples/s]10506 examples [00:11, 969.76 examples/s]10604 examples [00:11, 963.13 examples/s]10701 examples [00:11, 955.65 examples/s]10799 examples [00:11, 960.70 examples/s]10903 examples [00:11, 980.02 examples/s]11007 examples [00:11, 997.15 examples/s]11107 examples [00:11, 983.69 examples/s]11211 examples [00:11, 998.29 examples/s]11312 examples [00:11, 994.63 examples/s]11415 examples [00:12, 1003.63 examples/s]11516 examples [00:12, 987.20 examples/s] 11621 examples [00:12, 1002.67 examples/s]11723 examples [00:12, 1007.24 examples/s]11824 examples [00:12, 983.31 examples/s] 11926 examples [00:12, 993.83 examples/s]12031 examples [00:12, 1008.61 examples/s]12133 examples [00:12, 1005.69 examples/s]12234 examples [00:12, 979.59 examples/s] 12333 examples [00:13, 955.26 examples/s]12437 examples [00:13, 976.37 examples/s]12535 examples [00:13, 971.63 examples/s]12635 examples [00:13, 977.52 examples/s]12734 examples [00:13, 980.27 examples/s]12833 examples [00:13, 973.22 examples/s]12932 examples [00:13, 976.10 examples/s]13030 examples [00:13, 964.07 examples/s]13127 examples [00:13, 962.10 examples/s]13224 examples [00:13, 954.62 examples/s]13320 examples [00:14, 950.04 examples/s]13423 examples [00:14, 971.41 examples/s]13527 examples [00:14, 989.56 examples/s]13632 examples [00:14, 1004.95 examples/s]13733 examples [00:14, 997.94 examples/s] 13833 examples [00:14, 994.86 examples/s]13933 examples [00:14, 987.02 examples/s]14032 examples [00:14, 983.00 examples/s]14134 examples [00:14, 991.70 examples/s]14234 examples [00:14, 991.04 examples/s]14335 examples [00:15, 995.81 examples/s]14436 examples [00:15, 997.32 examples/s]14536 examples [00:15, 986.81 examples/s]14635 examples [00:15, 975.18 examples/s]14733 examples [00:15, 967.04 examples/s]14830 examples [00:15, 911.52 examples/s]14929 examples [00:15, 931.84 examples/s]15033 examples [00:15, 959.85 examples/s]15135 examples [00:15, 975.15 examples/s]15238 examples [00:15, 988.92 examples/s]15338 examples [00:16, 976.39 examples/s]15436 examples [00:16, 961.63 examples/s]15539 examples [00:16, 979.24 examples/s]15641 examples [00:16, 990.63 examples/s]15745 examples [00:16, 1003.82 examples/s]15846 examples [00:16, 964.77 examples/s] 15943 examples [00:16, 964.05 examples/s]16045 examples [00:16, 977.81 examples/s]16149 examples [00:16, 993.51 examples/s]16249 examples [00:17, 991.64 examples/s]16350 examples [00:17, 996.54 examples/s]16450 examples [00:17, 995.61 examples/s]16554 examples [00:17, 1007.93 examples/s]16657 examples [00:17, 1013.37 examples/s]16759 examples [00:17, 1006.28 examples/s]16860 examples [00:17, 1001.80 examples/s]16961 examples [00:17, 970.59 examples/s] 17062 examples [00:17, 979.62 examples/s]17161 examples [00:17, 978.70 examples/s]17260 examples [00:18, 965.14 examples/s]17359 examples [00:18, 969.85 examples/s]17457 examples [00:18, 972.21 examples/s]17555 examples [00:18, 961.01 examples/s]17652 examples [00:18, 953.06 examples/s]17748 examples [00:18, 945.98 examples/s]17847 examples [00:18, 957.96 examples/s]17943 examples [00:18, 957.30 examples/s]18046 examples [00:18, 976.30 examples/s]18144 examples [00:18, 974.47 examples/s]18244 examples [00:19, 980.59 examples/s]18343 examples [00:19, 976.06 examples/s]18445 examples [00:19, 986.87 examples/s]18549 examples [00:19, 999.61 examples/s]18650 examples [00:19, 995.78 examples/s]18750 examples [00:19, 962.21 examples/s]18847 examples [00:19, 951.82 examples/s]18943 examples [00:19, 949.20 examples/s]19039 examples [00:19, 939.36 examples/s]19134 examples [00:19, 934.77 examples/s]19229 examples [00:20, 936.68 examples/s]19324 examples [00:20, 938.02 examples/s]19424 examples [00:20, 954.43 examples/s]19521 examples [00:20, 958.97 examples/s]19622 examples [00:20, 971.78 examples/s]19720 examples [00:20, 908.37 examples/s]19816 examples [00:20, 922.98 examples/s]19910 examples [00:20, 927.70 examples/s]20004 examples [00:20, 909.40 examples/s]20107 examples [00:21, 940.90 examples/s]20212 examples [00:21, 969.73 examples/s]20310 examples [00:21, 951.59 examples/s]20411 examples [00:21, 966.47 examples/s]20514 examples [00:21, 983.60 examples/s]20619 examples [00:21, 1000.82 examples/s]20720 examples [00:21, 959.20 examples/s] 20817 examples [00:21, 956.76 examples/s]20921 examples [00:21, 978.40 examples/s]21026 examples [00:21, 997.31 examples/s]21129 examples [00:22, 1005.16 examples/s]21232 examples [00:22, 1009.42 examples/s]21336 examples [00:22, 1016.20 examples/s]21438 examples [00:22, 1009.93 examples/s]21540 examples [00:22, 964.79 examples/s] 21637 examples [00:22, 951.72 examples/s]21739 examples [00:22, 970.18 examples/s]21839 examples [00:22, 976.92 examples/s]21937 examples [00:22, 949.51 examples/s]22041 examples [00:22, 970.52 examples/s]22139 examples [00:23, 958.19 examples/s]22236 examples [00:23, 956.16 examples/s]22336 examples [00:23, 968.65 examples/s]22440 examples [00:23, 987.90 examples/s]22544 examples [00:23, 1001.27 examples/s]22645 examples [00:23, 989.16 examples/s] 22746 examples [00:23, 994.59 examples/s]22846 examples [00:23, 966.02 examples/s]22950 examples [00:23, 985.93 examples/s]23055 examples [00:24, 1001.79 examples/s]23158 examples [00:24, 1009.49 examples/s]23263 examples [00:24, 1020.19 examples/s]23368 examples [00:24, 1026.25 examples/s]23473 examples [00:24, 1030.71 examples/s]23577 examples [00:24, 1006.05 examples/s]23678 examples [00:24, 993.41 examples/s] 23778 examples [00:24, 987.82 examples/s]23877 examples [00:24, 961.86 examples/s]23980 examples [00:24, 980.50 examples/s]24081 examples [00:25, 987.52 examples/s]24180 examples [00:25, 958.89 examples/s]24283 examples [00:25, 977.83 examples/s]24382 examples [00:25, 951.03 examples/s]24484 examples [00:25, 968.92 examples/s]24588 examples [00:25, 987.37 examples/s]24692 examples [00:25, 1001.92 examples/s]24795 examples [00:25, 1007.95 examples/s]24897 examples [00:25, 994.55 examples/s] 24997 examples [00:25, 978.39 examples/s]25098 examples [00:26, 986.72 examples/s]25201 examples [00:26, 999.17 examples/s]25303 examples [00:26, 1005.05 examples/s]25404 examples [00:26, 989.04 examples/s] 25504 examples [00:26, 990.51 examples/s]25604 examples [00:26, 982.94 examples/s]25705 examples [00:26, 990.78 examples/s]25809 examples [00:26, 1002.90 examples/s]25910 examples [00:26, 1004.05 examples/s]26013 examples [00:26, 1010.81 examples/s]26115 examples [00:27, 994.40 examples/s] 26215 examples [00:27, 969.46 examples/s]26315 examples [00:27, 975.80 examples/s]26419 examples [00:27, 991.70 examples/s]26520 examples [00:27, 997.11 examples/s]26620 examples [00:27, 997.34 examples/s]26720 examples [00:27, 979.52 examples/s]26819 examples [00:27, 976.69 examples/s]26917 examples [00:27, 946.74 examples/s]27015 examples [00:28, 954.65 examples/s]27113 examples [00:28, 961.68 examples/s]27210 examples [00:28, 962.62 examples/s]27311 examples [00:28, 975.95 examples/s]27411 examples [00:28, 982.08 examples/s]27515 examples [00:28, 996.31 examples/s]27619 examples [00:28, 1008.89 examples/s]27723 examples [00:28, 1016.25 examples/s]27825 examples [00:28, 967.82 examples/s] 27923 examples [00:28, 965.66 examples/s]28020 examples [00:29, 953.07 examples/s]28123 examples [00:29, 974.43 examples/s]28221 examples [00:29, 939.50 examples/s]28325 examples [00:29, 966.27 examples/s]28426 examples [00:29, 977.27 examples/s]28525 examples [00:29, 967.09 examples/s]28623 examples [00:29, 961.79 examples/s]28720 examples [00:29, 957.68 examples/s]28819 examples [00:29, 965.30 examples/s]28922 examples [00:29, 983.55 examples/s]29021 examples [00:30, 952.34 examples/s]29122 examples [00:30, 967.76 examples/s]29225 examples [00:30, 983.49 examples/s]29324 examples [00:30, 982.93 examples/s]29427 examples [00:30, 994.35 examples/s]29532 examples [00:30, 1010.20 examples/s]29637 examples [00:30, 1019.59 examples/s]29741 examples [00:30, 1025.52 examples/s]29844 examples [00:30, 991.50 examples/s] 29948 examples [00:30, 1004.10 examples/s]30049 examples [00:31, 963.87 examples/s] 30153 examples [00:31, 983.60 examples/s]30252 examples [00:31, 977.45 examples/s]30353 examples [00:31, 986.92 examples/s]30454 examples [00:31, 992.42 examples/s]30558 examples [00:31, 1005.76 examples/s]30660 examples [00:31, 1009.14 examples/s]30762 examples [00:31, 977.05 examples/s] 30861 examples [00:31, 974.51 examples/s]30964 examples [00:32, 988.33 examples/s]31068 examples [00:32, 1002.74 examples/s]31172 examples [00:32, 1012.20 examples/s]31274 examples [00:32, 1005.28 examples/s]31377 examples [00:32, 1010.31 examples/s]31479 examples [00:32, 1012.75 examples/s]31581 examples [00:32, 1009.91 examples/s]31685 examples [00:32, 1016.80 examples/s]31787 examples [00:32, 1015.63 examples/s]31890 examples [00:32, 1017.38 examples/s]31992 examples [00:33, 953.54 examples/s] 32090 examples [00:33, 961.32 examples/s]32193 examples [00:33, 979.19 examples/s]32295 examples [00:33, 989.88 examples/s]32399 examples [00:33, 1002.36 examples/s]32503 examples [00:33, 1012.58 examples/s]32608 examples [00:33, 1022.71 examples/s]32711 examples [00:33, 1022.18 examples/s]32814 examples [00:33, 1006.20 examples/s]32915 examples [00:33, 1004.85 examples/s]33018 examples [00:34, 1010.15 examples/s]33120 examples [00:34, 986.42 examples/s] 33221 examples [00:34, 988.85 examples/s]33322 examples [00:34, 994.28 examples/s]33425 examples [00:34, 1003.49 examples/s]33526 examples [00:34, 1001.58 examples/s]33627 examples [00:34, 976.32 examples/s] 33728 examples [00:34, 985.66 examples/s]33827 examples [00:34, 965.54 examples/s]33924 examples [00:35, 963.99 examples/s]34026 examples [00:35, 978.03 examples/s]34125 examples [00:35, 979.78 examples/s]34224 examples [00:35, 974.22 examples/s]34323 examples [00:35, 977.75 examples/s]34427 examples [00:35, 995.59 examples/s]34528 examples [00:35, 998.59 examples/s]34628 examples [00:35, 993.05 examples/s]34733 examples [00:35, 1007.44 examples/s]34836 examples [00:35, 1012.69 examples/s]34940 examples [00:36, 1020.66 examples/s]35044 examples [00:36, 1025.81 examples/s]35148 examples [00:36, 1028.55 examples/s]35252 examples [00:36, 1031.95 examples/s]35356 examples [00:36, 1008.96 examples/s]35461 examples [00:36, 1019.02 examples/s]35564 examples [00:36, 1019.39 examples/s]35669 examples [00:36, 1026.79 examples/s]35772 examples [00:36, 1021.41 examples/s]35875 examples [00:36, 1023.16 examples/s]35980 examples [00:37, 1030.60 examples/s]36084 examples [00:37, 1022.89 examples/s]36189 examples [00:37, 1030.28 examples/s]36294 examples [00:37, 1033.22 examples/s]36398 examples [00:37, 1025.23 examples/s]36501 examples [00:37, 1018.37 examples/s]36603 examples [00:37, 1009.70 examples/s]36707 examples [00:37, 1017.13 examples/s]36809 examples [00:37, 1014.54 examples/s]36911 examples [00:37, 1008.08 examples/s]37012 examples [00:38, 959.83 examples/s] 37109 examples [00:38, 940.37 examples/s]37212 examples [00:38, 965.44 examples/s]37316 examples [00:38, 984.52 examples/s]37415 examples [00:38, 966.17 examples/s]37516 examples [00:38, 976.84 examples/s]37614 examples [00:38, 964.98 examples/s]37711 examples [00:38, 919.12 examples/s]37816 examples [00:38, 953.27 examples/s]37920 examples [00:39, 975.63 examples/s]38024 examples [00:39, 993.74 examples/s]38124 examples [00:39, 974.50 examples/s]38222 examples [00:39, 966.51 examples/s]38322 examples [00:39, 976.16 examples/s]38420 examples [00:39, 965.96 examples/s]38517 examples [00:39, 952.50 examples/s]38616 examples [00:39, 963.12 examples/s]38718 examples [00:39, 977.90 examples/s]38816 examples [00:39, 977.82 examples/s]38915 examples [00:40, 979.61 examples/s]39014 examples [00:40, 968.45 examples/s]39113 examples [00:40, 972.00 examples/s]39214 examples [00:40, 980.50 examples/s]39317 examples [00:40, 994.71 examples/s]39417 examples [00:40, 985.33 examples/s]39521 examples [00:40, 1000.81 examples/s]39624 examples [00:40, 1009.28 examples/s]39728 examples [00:40, 1017.01 examples/s]39832 examples [00:40, 1021.10 examples/s]39935 examples [00:41, 1008.10 examples/s]40036 examples [00:41, 955.82 examples/s] 40137 examples [00:41, 969.08 examples/s]40235 examples [00:41, 970.39 examples/s]40340 examples [00:41, 990.54 examples/s]40442 examples [00:41, 997.69 examples/s]40545 examples [00:41, 1006.00 examples/s]40648 examples [00:41, 1012.13 examples/s]40753 examples [00:41, 1020.47 examples/s]40858 examples [00:41, 1028.68 examples/s]40961 examples [00:42, 1021.89 examples/s]41065 examples [00:42, 1024.59 examples/s]41170 examples [00:42, 1030.78 examples/s]41274 examples [00:42, 1031.98 examples/s]41380 examples [00:42, 1038.10 examples/s]41484 examples [00:42, 1029.23 examples/s]41587 examples [00:42, 1028.73 examples/s]41692 examples [00:42, 1032.74 examples/s]41796 examples [00:42, 1034.26 examples/s]41900 examples [00:42, 1034.17 examples/s]42004 examples [00:43, 1024.91 examples/s]42107 examples [00:43, 982.30 examples/s] 42209 examples [00:43, 993.14 examples/s]42309 examples [00:43, 935.61 examples/s]42412 examples [00:43, 961.31 examples/s]42511 examples [00:43, 968.26 examples/s]42610 examples [00:43, 972.71 examples/s]42710 examples [00:43, 979.83 examples/s]42809 examples [00:43, 982.16 examples/s]42911 examples [00:44, 991.25 examples/s]43012 examples [00:44, 996.65 examples/s]43115 examples [00:44, 1005.85 examples/s]43219 examples [00:44, 1014.91 examples/s]43324 examples [00:44, 1022.95 examples/s]43427 examples [00:44, 1005.26 examples/s]43528 examples [00:44, 999.56 examples/s] 43629 examples [00:44, 950.67 examples/s]43727 examples [00:44, 957.54 examples/s]43828 examples [00:44, 970.20 examples/s]43930 examples [00:45, 984.59 examples/s]44035 examples [00:45, 1000.63 examples/s]44136 examples [00:45, 988.92 examples/s] 44236 examples [00:45, 991.07 examples/s]44340 examples [00:45, 1005.22 examples/s]44445 examples [00:45, 1016.52 examples/s]44547 examples [00:45, 996.46 examples/s] 44647 examples [00:45, 978.90 examples/s]44747 examples [00:45, 984.62 examples/s]44851 examples [00:45, 997.82 examples/s]44955 examples [00:46, 1009.69 examples/s]45060 examples [00:46, 1018.98 examples/s]45163 examples [00:46, 1015.27 examples/s]45265 examples [00:46, 1002.98 examples/s]45366 examples [00:46, 998.82 examples/s] 45468 examples [00:46, 1002.92 examples/s]45571 examples [00:46, 1009.79 examples/s]45675 examples [00:46, 1017.26 examples/s]45778 examples [00:46, 1019.38 examples/s]45880 examples [00:46, 1010.87 examples/s]45983 examples [00:47, 1013.64 examples/s]46085 examples [00:47, 1010.99 examples/s]46188 examples [00:47, 1016.60 examples/s]46290 examples [00:47, 997.63 examples/s] 46393 examples [00:47, 1005.20 examples/s]46494 examples [00:47, 1006.10 examples/s]46598 examples [00:47, 1014.42 examples/s]46700 examples [00:47, 1013.37 examples/s]46802 examples [00:47, 1013.36 examples/s]46904 examples [00:48, 984.15 examples/s] 47005 examples [00:48, 989.68 examples/s]47105 examples [00:48, 992.67 examples/s]47206 examples [00:48, 997.63 examples/s]47310 examples [00:48, 1007.29 examples/s]47413 examples [00:48, 1013.57 examples/s]47515 examples [00:48, 1010.39 examples/s]47617 examples [00:48, 1012.74 examples/s]47719 examples [00:48, 1011.51 examples/s]47821 examples [00:48, 1013.77 examples/s]47925 examples [00:49, 1019.31 examples/s]48029 examples [00:49, 1024.58 examples/s]48133 examples [00:49, 1027.23 examples/s]48236 examples [00:49, 1010.11 examples/s]48338 examples [00:49, 994.41 examples/s] 48438 examples [00:49, 978.33 examples/s]48540 examples [00:49, 989.57 examples/s]48640 examples [00:49, 975.02 examples/s]48742 examples [00:49, 987.98 examples/s]48842 examples [00:49, 990.42 examples/s]48946 examples [00:50, 1002.97 examples/s]49050 examples [00:50, 1013.36 examples/s]49152 examples [00:50, 1013.51 examples/s]49254 examples [00:50, 957.31 examples/s] 49351 examples [00:50, 942.62 examples/s]49453 examples [00:50, 962.70 examples/s]49557 examples [00:50, 982.03 examples/s]49656 examples [00:50, 981.80 examples/s]49760 examples [00:50, 998.46 examples/s]49863 examples [00:50, 1005.37 examples/s]49965 examples [00:51, 1007.88 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5445/50000 [00:00<00:00, 54448.36 examples/s] 34%|███▍      | 17120/50000 [00:00<00:00, 64825.29 examples/s] 60%|█████▉    | 29798/50000 [00:00<00:00, 75960.57 examples/s] 85%|████████▌ | 42606/50000 [00:00<00:00, 86522.55 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 857.85 examples/s]181 examples [00:00, 883.02 examples/s]286 examples [00:00, 926.37 examples/s]390 examples [00:00, 956.10 examples/s]494 examples [00:00, 978.61 examples/s]599 examples [00:00, 998.68 examples/s]703 examples [00:00, 1008.23 examples/s]802 examples [00:00, 1000.41 examples/s]906 examples [00:00, 1009.30 examples/s]1005 examples [00:01, 1002.77 examples/s]1106 examples [00:01, 1002.19 examples/s]1205 examples [00:01, 975.64 examples/s] 1307 examples [00:01, 986.19 examples/s]1412 examples [00:01, 1002.68 examples/s]1517 examples [00:01, 1015.20 examples/s]1622 examples [00:01, 1024.68 examples/s]1725 examples [00:01, 1019.75 examples/s]1827 examples [00:01, 1010.10 examples/s]1931 examples [00:01, 1017.89 examples/s]2033 examples [00:02, 1016.88 examples/s]2138 examples [00:02, 1025.81 examples/s]2241 examples [00:02, 1022.60 examples/s]2344 examples [00:02, 985.11 examples/s] 2443 examples [00:02, 972.51 examples/s]2541 examples [00:02, 971.33 examples/s]2646 examples [00:02, 993.09 examples/s]2750 examples [00:02, 1006.60 examples/s]2856 examples [00:02, 1019.83 examples/s]2962 examples [00:02, 1029.41 examples/s]3066 examples [00:03, 1016.70 examples/s]3169 examples [00:03, 1020.38 examples/s]3272 examples [00:03, 964.35 examples/s] 3371 examples [00:03, 970.15 examples/s]3472 examples [00:03, 981.43 examples/s]3574 examples [00:03, 992.22 examples/s]3674 examples [00:03, 972.48 examples/s]3774 examples [00:03, 979.82 examples/s]3873 examples [00:03, 978.95 examples/s]3976 examples [00:03, 991.10 examples/s]4082 examples [00:04, 1008.47 examples/s]4184 examples [00:04, 1011.56 examples/s]4289 examples [00:04, 1021.22 examples/s]4392 examples [00:04, 986.35 examples/s] 4493 examples [00:04, 991.95 examples/s]4598 examples [00:04, 1006.47 examples/s]4699 examples [00:04, 1005.36 examples/s]4800 examples [00:04, 792.03 examples/s] 4905 examples [00:04, 853.90 examples/s]5009 examples [00:05, 900.87 examples/s]5113 examples [00:05, 938.40 examples/s]5215 examples [00:05, 961.47 examples/s]5319 examples [00:05, 981.78 examples/s]5421 examples [00:05, 991.02 examples/s]5526 examples [00:05, 1006.35 examples/s]5631 examples [00:05, 1016.74 examples/s]5736 examples [00:05, 1024.07 examples/s]5842 examples [00:05, 1031.83 examples/s]5947 examples [00:05, 1036.63 examples/s]6051 examples [00:06, 1036.40 examples/s]6155 examples [00:06, 1036.64 examples/s]6259 examples [00:06, 1026.68 examples/s]6364 examples [00:06, 1032.78 examples/s]6468 examples [00:06, 1034.34 examples/s]6572 examples [00:06, 1035.69 examples/s]6676 examples [00:06, 1035.30 examples/s]6780 examples [00:06, 1027.71 examples/s]6883 examples [00:06, 1022.78 examples/s]6988 examples [00:07, 1029.17 examples/s]7092 examples [00:07, 1030.84 examples/s]7196 examples [00:07, 1013.00 examples/s]7298 examples [00:07, 1013.41 examples/s]7400 examples [00:07, 997.42 examples/s] 7502 examples [00:07, 1003.25 examples/s]7605 examples [00:07, 1009.41 examples/s]7707 examples [00:07, 983.46 examples/s] 7806 examples [00:07, 983.45 examples/s]7907 examples [00:07, 990.67 examples/s]8012 examples [00:08, 1005.76 examples/s]8113 examples [00:08, 1005.28 examples/s]8214 examples [00:08, 1003.69 examples/s]8316 examples [00:08, 1007.50 examples/s]8421 examples [00:08, 1017.80 examples/s]8525 examples [00:08, 1021.63 examples/s]8630 examples [00:08, 1027.39 examples/s]8733 examples [00:08, 1004.64 examples/s]8834 examples [00:08, 998.52 examples/s] 8939 examples [00:08, 1011.31 examples/s]9043 examples [00:09, 1018.01 examples/s]9147 examples [00:09, 1023.93 examples/s]9250 examples [00:09, 1012.60 examples/s]9353 examples [00:09, 1014.97 examples/s]9458 examples [00:09, 1024.71 examples/s]9562 examples [00:09, 1029.06 examples/s]9666 examples [00:09, 1032.22 examples/s]9770 examples [00:09, 999.14 examples/s] 9873 examples [00:09, 1006.35 examples/s]9977 examples [00:09, 1013.77 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNPKNTA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNPKNTA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f00fe69e400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f00fe69e400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f00fe69e400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f00fc6ff278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f00849591d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f00fe69e400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f00fe69e400> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f00fe69e400> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f0084959128>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f00e655e1d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.21 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.21 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.21 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.84 file/s]2020-08-30 18:08:24.638911: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-30 18:08:24.643224: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-30 18:08:24.643392: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561c81b25030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-30 18:08:24.643407: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 462588.17it/s] 61%|██████    | 6021120/9912422 [00:00<00:05, 658653.41it/s]9920512it [00:00, 38511935.62it/s]                           
0it [00:00, ?it/s]32768it [00:00, 473876.77it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 140833.61it/s]1654784it [00:00, 8953874.11it/s]                          
0it [00:00, ?it/s]8192it [00:00, 227715.33it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
