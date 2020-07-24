
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6fe8a3b73a576c75c565441d8ffa2353b386e363', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6fe8a3b73a576c75c565441d8ffa2353b386e363

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/6fe8a3b73a576c75c565441d8ffa2353b386e363

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc07f4f9ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc07f4f9ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc0ea16e488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc0ea16e488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc10939aea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc10939aea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc0975159d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc0975159d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc0975159d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:12, 136941.58it/s] 68%|██████▊   | 6782976/9912422 [00:00<00:16, 195438.91it/s]9920512it [00:00, 37998390.16it/s]                           
0it [00:00, ?it/s]32768it [00:00, 606510.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 451954.14it/s]1654784it [00:00, 11551971.64it/s]                         
0it [00:00, ?it/s]8192it [00:00, 264356.52it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc075f396d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc075f49940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc097515620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc097515620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc097515620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:16,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:16,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:15,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:50,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:50,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:50,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:33,  4.04 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:22,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:21,  5.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  7.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:09, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 31.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 38.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 44.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 51.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 56.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 62.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 68.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.97 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.97 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.41s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.97 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.93 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.64s/ url]
0 examples [00:00, ? examples/s]2020-07-24 18:09:24.719745: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-24 18:09:24.733278: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-24 18:09:24.734087: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5644b5724390 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 18:09:24.734111: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.44 examples/s]109 examples [00:00,  4.90 examples/s]209 examples [00:00,  6.99 examples/s]313 examples [00:00,  9.96 examples/s]420 examples [00:00, 14.17 examples/s]520 examples [00:00, 20.12 examples/s]606 examples [00:00, 28.46 examples/s]707 examples [00:00, 40.16 examples/s]811 examples [00:01, 56.44 examples/s]919 examples [00:01, 78.85 examples/s]1028 examples [00:01, 109.24 examples/s]1135 examples [00:01, 149.49 examples/s]1243 examples [00:01, 201.54 examples/s]1348 examples [00:01, 264.87 examples/s]1453 examples [00:01, 341.29 examples/s]1561 examples [00:01, 428.96 examples/s]1669 examples [00:01, 523.48 examples/s]1776 examples [00:02, 618.12 examples/s]1883 examples [00:02, 707.61 examples/s]1992 examples [00:02, 790.83 examples/s]2100 examples [00:02, 859.62 examples/s]2208 examples [00:02, 905.73 examples/s]2315 examples [00:02, 938.37 examples/s]2421 examples [00:02, 959.15 examples/s]2531 examples [00:02, 995.77 examples/s]2640 examples [00:02, 1020.68 examples/s]2748 examples [00:02, 1035.37 examples/s]2856 examples [00:03, 1046.08 examples/s]2963 examples [00:03, 1028.13 examples/s]3072 examples [00:03, 1044.14 examples/s]3178 examples [00:03, 1041.27 examples/s]3283 examples [00:03, 1040.67 examples/s]3391 examples [00:03, 1050.13 examples/s]3497 examples [00:03, 1010.32 examples/s]3599 examples [00:03, 1000.79 examples/s]3700 examples [00:03, 999.95 examples/s] 3801 examples [00:03, 988.00 examples/s]3904 examples [00:04, 998.94 examples/s]4005 examples [00:04, 1000.95 examples/s]4110 examples [00:04, 1013.02 examples/s]4219 examples [00:04, 1033.05 examples/s]4323 examples [00:04, 1009.59 examples/s]4430 examples [00:04, 1025.16 examples/s]4533 examples [00:04, 980.02 examples/s] 4641 examples [00:04, 1007.35 examples/s]4750 examples [00:04, 1028.49 examples/s]4856 examples [00:04, 1034.28 examples/s]4963 examples [00:05, 1042.68 examples/s]5068 examples [00:05, 1023.30 examples/s]5171 examples [00:05, 1022.07 examples/s]5277 examples [00:05, 1031.36 examples/s]5381 examples [00:05, 1018.35 examples/s]5483 examples [00:05, 998.15 examples/s] 5590 examples [00:05, 1017.90 examples/s]5693 examples [00:05, 1021.32 examples/s]5802 examples [00:05, 1040.57 examples/s]5908 examples [00:06, 1046.04 examples/s]6013 examples [00:06, 994.67 examples/s] 6114 examples [00:06, 985.82 examples/s]6220 examples [00:06, 1006.58 examples/s]6327 examples [00:06, 1024.08 examples/s]6432 examples [00:06, 1030.48 examples/s]6536 examples [00:06, 1016.35 examples/s]6641 examples [00:06, 1025.88 examples/s]6749 examples [00:06, 1039.21 examples/s]6854 examples [00:06, 1022.98 examples/s]6957 examples [00:07, 1021.30 examples/s]7060 examples [00:07, 1006.11 examples/s]7161 examples [00:07, 1006.09 examples/s]7265 examples [00:07, 1014.23 examples/s]7368 examples [00:07, 1016.18 examples/s]7472 examples [00:07, 1022.31 examples/s]7575 examples [00:07, 1019.98 examples/s]7682 examples [00:07, 1033.00 examples/s]7789 examples [00:07, 1042.25 examples/s]7896 examples [00:07, 1049.20 examples/s]8001 examples [00:08, 1041.82 examples/s]8109 examples [00:08, 1050.41 examples/s]8215 examples [00:08, 1010.01 examples/s]8317 examples [00:08, 994.01 examples/s] 8417 examples [00:08, 982.16 examples/s]8519 examples [00:08, 992.76 examples/s]8619 examples [00:08, 990.38 examples/s]8723 examples [00:08, 995.23 examples/s]8832 examples [00:08, 1019.76 examples/s]8935 examples [00:08, 1009.44 examples/s]9041 examples [00:09, 1022.45 examples/s]9148 examples [00:09, 1035.72 examples/s]9255 examples [00:09, 1045.13 examples/s]9360 examples [00:09, 1042.56 examples/s]9468 examples [00:09, 1053.52 examples/s]9574 examples [00:09, 1018.26 examples/s]9679 examples [00:09, 1026.69 examples/s]9782 examples [00:09, 1003.11 examples/s]9883 examples [00:09, 980.40 examples/s] 9990 examples [00:10, 1004.90 examples/s]10091 examples [00:10, 967.57 examples/s]10197 examples [00:10, 992.91 examples/s]10306 examples [00:10, 1019.39 examples/s]10415 examples [00:10, 1038.95 examples/s]10524 examples [00:10, 1053.37 examples/s]10630 examples [00:10, 1032.19 examples/s]10736 examples [00:10, 1039.62 examples/s]10844 examples [00:10, 1048.80 examples/s]10953 examples [00:10, 1059.03 examples/s]11060 examples [00:11, 1059.05 examples/s]11167 examples [00:11, 1062.22 examples/s]11276 examples [00:11, 1068.37 examples/s]11383 examples [00:11, 1067.70 examples/s]11490 examples [00:11, 1049.50 examples/s]11596 examples [00:11, 1048.96 examples/s]11701 examples [00:11, 1043.04 examples/s]11806 examples [00:11, 1037.48 examples/s]11914 examples [00:11, 1047.07 examples/s]12019 examples [00:11, 1047.23 examples/s]12127 examples [00:12, 1054.41 examples/s]12234 examples [00:12, 1057.06 examples/s]12341 examples [00:12, 1060.71 examples/s]12449 examples [00:12, 1065.30 examples/s]12556 examples [00:12, 1064.98 examples/s]12665 examples [00:12, 1071.36 examples/s]12773 examples [00:12, 1062.88 examples/s]12882 examples [00:12, 1068.80 examples/s]12990 examples [00:12, 1071.63 examples/s]13098 examples [00:12, 1068.48 examples/s]13205 examples [00:13, 1053.93 examples/s]13311 examples [00:13, 1050.42 examples/s]13420 examples [00:13, 1059.56 examples/s]13527 examples [00:13, 1053.83 examples/s]13633 examples [00:13, 1053.16 examples/s]13740 examples [00:13, 1056.50 examples/s]13846 examples [00:13, 1052.83 examples/s]13952 examples [00:13, 1014.74 examples/s]14061 examples [00:13, 1034.51 examples/s]14170 examples [00:14, 1048.45 examples/s]14278 examples [00:14, 1055.44 examples/s]14384 examples [00:14, 1038.93 examples/s]14492 examples [00:14, 1048.75 examples/s]14599 examples [00:14, 1053.02 examples/s]14705 examples [00:14, 1023.88 examples/s]14812 examples [00:14, 1036.34 examples/s]14916 examples [00:14, 1025.60 examples/s]15023 examples [00:14, 1036.99 examples/s]15130 examples [00:14, 1045.15 examples/s]15238 examples [00:15, 1052.98 examples/s]15347 examples [00:15, 1061.39 examples/s]15454 examples [00:15, 1063.21 examples/s]15563 examples [00:15, 1068.45 examples/s]15671 examples [00:15, 1071.39 examples/s]15779 examples [00:15, 1069.17 examples/s]15886 examples [00:15, 1069.29 examples/s]15993 examples [00:15, 1057.78 examples/s]16101 examples [00:15, 1062.40 examples/s]16208 examples [00:15, 1056.54 examples/s]16314 examples [00:16, 1056.81 examples/s]16423 examples [00:16, 1064.65 examples/s]16530 examples [00:16, 1053.53 examples/s]16638 examples [00:16, 1060.75 examples/s]16745 examples [00:16, 1057.18 examples/s]16851 examples [00:16, 1042.31 examples/s]16956 examples [00:16, 1042.13 examples/s]17061 examples [00:16, 997.69 examples/s] 17170 examples [00:16, 1021.23 examples/s]17279 examples [00:16, 1040.66 examples/s]17388 examples [00:17, 1053.50 examples/s]17498 examples [00:17, 1064.68 examples/s]17606 examples [00:17, 1066.79 examples/s]17715 examples [00:17, 1071.21 examples/s]17823 examples [00:17, 1056.13 examples/s]17930 examples [00:17, 1058.16 examples/s]18036 examples [00:17, 1034.19 examples/s]18140 examples [00:17, 1034.72 examples/s]18245 examples [00:17, 1036.62 examples/s]18353 examples [00:17, 1048.04 examples/s]18458 examples [00:18, 1039.16 examples/s]18565 examples [00:18, 1047.70 examples/s]18670 examples [00:18, 1019.79 examples/s]18777 examples [00:18, 1031.95 examples/s]18884 examples [00:18, 1040.73 examples/s]18990 examples [00:18, 1045.92 examples/s]19098 examples [00:18, 1053.91 examples/s]19204 examples [00:18, 1050.21 examples/s]19311 examples [00:18, 1055.42 examples/s]19420 examples [00:19, 1062.60 examples/s]19528 examples [00:19, 1066.59 examples/s]19637 examples [00:19, 1070.64 examples/s]19745 examples [00:19, 1057.39 examples/s]19851 examples [00:19, 1050.79 examples/s]19959 examples [00:19, 1057.93 examples/s]20065 examples [00:19, 978.50 examples/s] 20167 examples [00:19, 988.84 examples/s]20275 examples [00:19, 1012.79 examples/s]20384 examples [00:19, 1032.87 examples/s]20492 examples [00:20, 1046.19 examples/s]20602 examples [00:20, 1059.95 examples/s]20709 examples [00:20, 1059.62 examples/s]20817 examples [00:20, 1064.52 examples/s]20925 examples [00:20, 1068.55 examples/s]21032 examples [00:20, 1053.48 examples/s]21141 examples [00:20, 1061.50 examples/s]21248 examples [00:20, 1063.69 examples/s]21355 examples [00:20, 1063.73 examples/s]21462 examples [00:20, 1056.11 examples/s]21572 examples [00:21, 1066.39 examples/s]21680 examples [00:21, 1069.11 examples/s]21788 examples [00:21, 1069.97 examples/s]21896 examples [00:21, 1040.57 examples/s]22004 examples [00:21, 1051.92 examples/s]22112 examples [00:21, 1059.68 examples/s]22222 examples [00:21, 1069.25 examples/s]22330 examples [00:21, 1070.55 examples/s]22438 examples [00:21, 1023.99 examples/s]22545 examples [00:21, 1035.24 examples/s]22652 examples [00:22, 1044.58 examples/s]22760 examples [00:22, 1054.01 examples/s]22866 examples [00:22, 1052.08 examples/s]22972 examples [00:22, 1046.61 examples/s]23079 examples [00:22, 1053.12 examples/s]23185 examples [00:22, 1053.90 examples/s]23295 examples [00:22, 1064.92 examples/s]23402 examples [00:22, 1045.44 examples/s]23508 examples [00:22, 1049.12 examples/s]23614 examples [00:23, 1041.94 examples/s]23719 examples [00:23, 1036.59 examples/s]23826 examples [00:23, 1044.90 examples/s]23933 examples [00:23, 1049.64 examples/s]24039 examples [00:23, 1041.37 examples/s]24144 examples [00:23, 1033.22 examples/s]24251 examples [00:23, 1041.94 examples/s]24356 examples [00:23, 1033.73 examples/s]24460 examples [00:23, 1013.72 examples/s]24567 examples [00:23, 1027.54 examples/s]24672 examples [00:24, 1032.73 examples/s]24777 examples [00:24, 1035.21 examples/s]24887 examples [00:24, 1052.86 examples/s]24996 examples [00:24, 1062.80 examples/s]25103 examples [00:24, 1062.31 examples/s]25210 examples [00:24, 1064.57 examples/s]25317 examples [00:24, 1049.95 examples/s]25424 examples [00:24, 1054.90 examples/s]25531 examples [00:24, 1056.83 examples/s]25638 examples [00:24, 1059.47 examples/s]25745 examples [00:25, 1061.27 examples/s]25853 examples [00:25, 1066.03 examples/s]25960 examples [00:25, 1040.88 examples/s]26065 examples [00:25, 1013.79 examples/s]26172 examples [00:25, 1028.37 examples/s]26276 examples [00:25, 1028.57 examples/s]26384 examples [00:25, 1042.89 examples/s]26493 examples [00:25, 1054.09 examples/s]26599 examples [00:25, 1016.60 examples/s]26706 examples [00:25, 1029.59 examples/s]26814 examples [00:26, 1043.97 examples/s]26920 examples [00:26, 1046.72 examples/s]27025 examples [00:26, 1038.38 examples/s]27131 examples [00:26, 1043.56 examples/s]27237 examples [00:26, 1048.00 examples/s]27342 examples [00:26, 1040.46 examples/s]27447 examples [00:26, 1038.35 examples/s]27551 examples [00:26, 996.40 examples/s] 27654 examples [00:26, 1002.20 examples/s]27757 examples [00:26, 1007.52 examples/s]27864 examples [00:27, 1024.13 examples/s]27972 examples [00:27, 1038.73 examples/s]28078 examples [00:27, 1042.36 examples/s]28183 examples [00:27, 1037.50 examples/s]28290 examples [00:27, 1044.35 examples/s]28397 examples [00:27, 1049.68 examples/s]28506 examples [00:27, 1059.71 examples/s]28613 examples [00:27, 1060.42 examples/s]28720 examples [00:27, 1047.17 examples/s]28829 examples [00:28, 1057.24 examples/s]28936 examples [00:28, 1059.36 examples/s]29042 examples [00:28, 1047.86 examples/s]29147 examples [00:28, 1040.78 examples/s]29252 examples [00:28, 1043.19 examples/s]29358 examples [00:28, 1047.33 examples/s]29463 examples [00:28, 1031.83 examples/s]29569 examples [00:28, 1039.38 examples/s]29674 examples [00:28, 1023.76 examples/s]29777 examples [00:28, 1024.01 examples/s]29885 examples [00:29, 1039.05 examples/s]29992 examples [00:29, 1046.73 examples/s]30097 examples [00:29, 973.85 examples/s] 30196 examples [00:29, 967.34 examples/s]30296 examples [00:29, 976.83 examples/s]30402 examples [00:29, 997.89 examples/s]30504 examples [00:29, 1003.14 examples/s]30613 examples [00:29, 1026.34 examples/s]30722 examples [00:29, 1044.39 examples/s]30829 examples [00:29, 1051.63 examples/s]30938 examples [00:30, 1061.55 examples/s]31047 examples [00:30, 1069.11 examples/s]31155 examples [00:30, 1069.88 examples/s]31263 examples [00:30, 1070.13 examples/s]31371 examples [00:30, 1072.26 examples/s]31479 examples [00:30, 1065.67 examples/s]31586 examples [00:30, 1064.82 examples/s]31694 examples [00:30, 1068.90 examples/s]31804 examples [00:30, 1075.82 examples/s]31913 examples [00:30, 1077.37 examples/s]32022 examples [00:31, 1078.55 examples/s]32130 examples [00:31, 1075.18 examples/s]32238 examples [00:31, 1071.78 examples/s]32347 examples [00:31, 1074.87 examples/s]32455 examples [00:31, 1057.37 examples/s]32561 examples [00:31, 1052.50 examples/s]32667 examples [00:31, 1054.39 examples/s]32773 examples [00:31, 1040.56 examples/s]32879 examples [00:31, 1044.53 examples/s]32985 examples [00:31, 1046.33 examples/s]33090 examples [00:32, 1045.74 examples/s]33197 examples [00:32, 1050.72 examples/s]33303 examples [00:32, 1049.11 examples/s]33408 examples [00:32, 1033.24 examples/s]33512 examples [00:32, 986.36 examples/s] 33619 examples [00:32, 1008.93 examples/s]33728 examples [00:32, 1030.67 examples/s]33836 examples [00:32, 1044.27 examples/s]33944 examples [00:32, 1052.69 examples/s]34051 examples [00:33, 1056.56 examples/s]34159 examples [00:33, 1061.61 examples/s]34266 examples [00:33, 1063.57 examples/s]34373 examples [00:33, 1054.09 examples/s]34482 examples [00:33, 1062.33 examples/s]34589 examples [00:33, 1058.99 examples/s]34697 examples [00:33, 1063.21 examples/s]34805 examples [00:33, 1068.16 examples/s]34914 examples [00:33, 1072.25 examples/s]35022 examples [00:33, 1072.30 examples/s]35130 examples [00:34, 1067.80 examples/s]35239 examples [00:34, 1071.52 examples/s]35347 examples [00:34, 1072.42 examples/s]35455 examples [00:34, 1070.69 examples/s]35564 examples [00:34, 1075.93 examples/s]35672 examples [00:34, 1057.03 examples/s]35778 examples [00:34, 995.49 examples/s] 35881 examples [00:34, 1003.61 examples/s]35982 examples [00:34, 1001.41 examples/s]36086 examples [00:34, 1010.58 examples/s]36190 examples [00:35, 1016.63 examples/s]36299 examples [00:35, 1036.13 examples/s]36408 examples [00:35, 1049.67 examples/s]36514 examples [00:35, 1041.57 examples/s]36620 examples [00:35, 1046.98 examples/s]36725 examples [00:35, 1045.89 examples/s]36832 examples [00:35, 1051.25 examples/s]36938 examples [00:35, 1050.13 examples/s]37044 examples [00:35, 1040.45 examples/s]37151 examples [00:35, 1048.60 examples/s]37256 examples [00:36, 1047.67 examples/s]37361 examples [00:36, 1048.24 examples/s]37466 examples [00:36, 1046.82 examples/s]37571 examples [00:36, 1035.45 examples/s]37675 examples [00:36, 1031.77 examples/s]37779 examples [00:36, 1029.54 examples/s]37882 examples [00:36, 1025.01 examples/s]37991 examples [00:36, 1042.53 examples/s]38096 examples [00:36, 1043.32 examples/s]38205 examples [00:36, 1055.40 examples/s]38311 examples [00:37, 1055.05 examples/s]38417 examples [00:37, 1025.12 examples/s]38520 examples [00:37, 1002.86 examples/s]38621 examples [00:37, 982.69 examples/s] 38728 examples [00:37, 1002.09 examples/s]38829 examples [00:37, 996.23 examples/s] 38936 examples [00:37, 1014.82 examples/s]39044 examples [00:37, 1032.84 examples/s]39151 examples [00:37, 1042.41 examples/s]39257 examples [00:38, 1047.30 examples/s]39362 examples [00:38, 1047.74 examples/s]39467 examples [00:38, 1043.79 examples/s]39574 examples [00:38, 1051.36 examples/s]39680 examples [00:38, 1051.96 examples/s]39786 examples [00:38, 1034.96 examples/s]39890 examples [00:38, 1035.53 examples/s]39997 examples [00:38, 1044.96 examples/s]40102 examples [00:38, 994.56 examples/s] 40209 examples [00:38, 1015.17 examples/s]40318 examples [00:39, 1034.94 examples/s]40425 examples [00:39, 1045.20 examples/s]40533 examples [00:39, 1054.07 examples/s]40642 examples [00:39, 1062.35 examples/s]40749 examples [00:39, 1063.57 examples/s]40858 examples [00:39, 1070.20 examples/s]40966 examples [00:39, 1061.98 examples/s]41074 examples [00:39, 1065.39 examples/s]41183 examples [00:39, 1071.81 examples/s]41291 examples [00:39, 1073.88 examples/s]41399 examples [00:40, 1074.15 examples/s]41507 examples [00:40, 1068.84 examples/s]41616 examples [00:40, 1074.32 examples/s]41726 examples [00:40, 1079.00 examples/s]41836 examples [00:40, 1082.06 examples/s]41945 examples [00:40, 1075.70 examples/s]42053 examples [00:40, 1074.16 examples/s]42161 examples [00:40, 1073.70 examples/s]42269 examples [00:40, 1070.91 examples/s]42377 examples [00:40, 1055.40 examples/s]42483 examples [00:41, 1050.90 examples/s]42589 examples [00:41, 1050.30 examples/s]42696 examples [00:41, 1054.77 examples/s]42803 examples [00:41, 1057.97 examples/s]42909 examples [00:41, 1051.92 examples/s]43015 examples [00:41, 1035.51 examples/s]43121 examples [00:41, 1041.11 examples/s]43227 examples [00:41, 1045.64 examples/s]43335 examples [00:41, 1052.85 examples/s]43441 examples [00:41, 1050.71 examples/s]43547 examples [00:42, 1052.17 examples/s]43653 examples [00:42, 1050.61 examples/s]43759 examples [00:42, 1051.37 examples/s]43865 examples [00:42, 1047.43 examples/s]43970 examples [00:42, 1045.39 examples/s]44075 examples [00:42, 1042.49 examples/s]44180 examples [00:42, 1041.52 examples/s]44285 examples [00:42, 1043.17 examples/s]44390 examples [00:42, 1037.72 examples/s]44496 examples [00:42, 1043.10 examples/s]44603 examples [00:43, 1050.32 examples/s]44709 examples [00:43, 1032.10 examples/s]44815 examples [00:43, 1037.81 examples/s]44919 examples [00:43, 1010.44 examples/s]45026 examples [00:43, 1026.98 examples/s]45132 examples [00:43, 1036.03 examples/s]45236 examples [00:43, 1031.98 examples/s]45340 examples [00:43, 1018.29 examples/s]45448 examples [00:43, 1034.96 examples/s]45555 examples [00:44, 1044.23 examples/s]45663 examples [00:44, 1053.99 examples/s]45772 examples [00:44, 1061.60 examples/s]45881 examples [00:44, 1068.01 examples/s]45988 examples [00:44, 1034.62 examples/s]46092 examples [00:44, 1016.45 examples/s]46199 examples [00:44, 1030.06 examples/s]46305 examples [00:44, 1038.76 examples/s]46412 examples [00:44, 1046.86 examples/s]46517 examples [00:44, 1019.50 examples/s]46624 examples [00:45, 1031.80 examples/s]46728 examples [00:45, 1033.95 examples/s]46833 examples [00:45, 1038.54 examples/s]46941 examples [00:45, 1049.79 examples/s]47049 examples [00:45, 1056.53 examples/s]47155 examples [00:45, 965.64 examples/s] 47254 examples [00:45, 956.17 examples/s]47356 examples [00:45, 974.29 examples/s]47455 examples [00:45, 966.81 examples/s]47553 examples [00:45, 951.04 examples/s]47652 examples [00:46, 960.94 examples/s]47753 examples [00:46, 974.27 examples/s]47858 examples [00:46, 995.41 examples/s]47966 examples [00:46, 1018.78 examples/s]48074 examples [00:46, 1035.46 examples/s]48178 examples [00:46, 1036.06 examples/s]48285 examples [00:46, 1043.42 examples/s]48390 examples [00:46, 1034.87 examples/s]48496 examples [00:46, 1040.92 examples/s]48601 examples [00:47, 1029.49 examples/s]48706 examples [00:47, 1034.16 examples/s]48810 examples [00:47, 1030.15 examples/s]48915 examples [00:47, 1035.95 examples/s]49024 examples [00:47, 1049.36 examples/s]49132 examples [00:47, 1058.08 examples/s]49238 examples [00:47, 1049.34 examples/s]49344 examples [00:47, 1052.45 examples/s]49450 examples [00:47, 1049.81 examples/s]49557 examples [00:47, 1053.39 examples/s]49663 examples [00:48, 1023.16 examples/s]49766 examples [00:48, 1008.10 examples/s]49868 examples [00:48, 993.03 examples/s] 49974 examples [00:48, 1010.19 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6070/50000 [00:00<00:00, 60399.24 examples/s] 38%|███▊      | 19117/50000 [00:00<00:00, 71999.43 examples/s] 64%|██████▍   | 32053/50000 [00:00<00:00, 83046.62 examples/s] 90%|█████████ | 45237/50000 [00:00<00:00, 93417.48 examples/s]                                                               0 examples [00:00, ? examples/s]91 examples [00:00, 902.00 examples/s]201 examples [00:00, 951.47 examples/s]311 examples [00:00, 989.95 examples/s]418 examples [00:00, 1011.75 examples/s]526 examples [00:00, 1030.89 examples/s]636 examples [00:00, 1049.71 examples/s]746 examples [00:00, 1061.50 examples/s]855 examples [00:00, 1067.76 examples/s]965 examples [00:00, 1074.87 examples/s]1074 examples [00:01, 1077.01 examples/s]1183 examples [00:01, 1079.69 examples/s]1292 examples [00:01, 1082.30 examples/s]1401 examples [00:01, 1082.13 examples/s]1510 examples [00:01, 1083.76 examples/s]1619 examples [00:01, 1083.30 examples/s]1728 examples [00:01, 1084.97 examples/s]1837 examples [00:01, 1086.36 examples/s]1946 examples [00:01, 1072.77 examples/s]2056 examples [00:01, 1078.30 examples/s]2164 examples [00:02, 1047.12 examples/s]2269 examples [00:02, 1045.26 examples/s]2374 examples [00:02, 1046.34 examples/s]2479 examples [00:02, 1044.68 examples/s]2584 examples [00:02, 1042.49 examples/s]2689 examples [00:02, 1042.34 examples/s]2798 examples [00:02, 1054.90 examples/s]2908 examples [00:02, 1065.58 examples/s]3017 examples [00:02, 1070.67 examples/s]3125 examples [00:02, 1071.76 examples/s]3233 examples [00:03, 1068.92 examples/s]3340 examples [00:03, 1068.43 examples/s]3450 examples [00:03, 1076.91 examples/s]3561 examples [00:03, 1084.31 examples/s]3672 examples [00:03, 1090.97 examples/s]3782 examples [00:03, 1087.64 examples/s]3891 examples [00:03, 1077.65 examples/s]3999 examples [00:03, 1048.31 examples/s]4105 examples [00:03, 1020.93 examples/s]4210 examples [00:03, 1029.09 examples/s]4315 examples [00:04, 1035.20 examples/s]4425 examples [00:04, 1053.00 examples/s]4531 examples [00:04, 1052.36 examples/s]4637 examples [00:04, 1052.26 examples/s]4743 examples [00:04, 1051.23 examples/s]4849 examples [00:04, 1022.20 examples/s]4952 examples [00:04, 1024.49 examples/s]5061 examples [00:04, 1042.19 examples/s]5167 examples [00:04, 1045.14 examples/s]5274 examples [00:04, 1051.31 examples/s]5380 examples [00:05, 1040.87 examples/s]5486 examples [00:05, 1046.32 examples/s]5591 examples [00:05, 1019.03 examples/s]5701 examples [00:05, 1041.19 examples/s]5809 examples [00:05, 1051.97 examples/s]5915 examples [00:05, 1047.83 examples/s]6025 examples [00:05, 1061.35 examples/s]6134 examples [00:05, 1069.17 examples/s]6242 examples [00:05, 1071.16 examples/s]6350 examples [00:05, 1072.70 examples/s]6458 examples [00:06, 1061.31 examples/s]6569 examples [00:06, 1072.81 examples/s]6679 examples [00:06, 1079.92 examples/s]6789 examples [00:06, 1083.42 examples/s]6898 examples [00:06, 1078.41 examples/s]7006 examples [00:06, 1077.52 examples/s]7115 examples [00:06, 1080.79 examples/s]7224 examples [00:06, 1081.77 examples/s]7335 examples [00:06, 1088.28 examples/s]7444 examples [00:07, 1088.59 examples/s]7553 examples [00:07, 1053.57 examples/s]7659 examples [00:07, 1054.39 examples/s]7768 examples [00:07, 1063.33 examples/s]7878 examples [00:07, 1071.21 examples/s]7987 examples [00:07, 1076.29 examples/s]8095 examples [00:07, 1071.89 examples/s]8203 examples [00:07, 1062.77 examples/s]8310 examples [00:07, 1038.90 examples/s]8418 examples [00:07, 1048.64 examples/s]8523 examples [00:08, 1046.55 examples/s]8628 examples [00:08, 1029.65 examples/s]8732 examples [00:08, 1020.93 examples/s]8840 examples [00:08, 1037.86 examples/s]8945 examples [00:08, 1038.73 examples/s]9053 examples [00:08, 1050.75 examples/s]9159 examples [00:08, 1027.58 examples/s]9267 examples [00:08, 1042.46 examples/s]9372 examples [00:08, 1035.02 examples/s]9483 examples [00:08, 1052.93 examples/s]9593 examples [00:09, 1064.43 examples/s]9700 examples [00:09, 1064.78 examples/s]9807 examples [00:09, 1059.68 examples/s]9916 examples [00:09, 1068.51 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete584SS2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete584SS2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc0975159d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc0975159d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc0975159d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc01d83cfd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc01d7f72e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc0975159d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc0975159d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc0975159d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc01d7f7dd8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc075f49518>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc0178a1620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc0178a1620> 

  function with postional parmater data_info <function split_train_valid at 0x7fc0178a1620> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc0178a1730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc0178a1730> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc0178a1730> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=ef203b5503e1388e82ffc82c315e9a14c25463f233bbf477d5ccb2185a16a9b0
  Stored in directory: /tmp/pip-ephem-wheel-cache-7ny1rdcq/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:20:07, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<11:40:06, 20.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:13:20, 29.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:00<5:45:46, 41.5kB/s].vector_cache/glove.6B.zip:   0%|          | 2.79M/862M [00:01<4:01:44, 59.3kB/s].vector_cache/glove.6B.zip:   1%|          | 6.69M/862M [00:01<2:48:33, 84.6kB/s].vector_cache/glove.6B.zip:   1%|          | 10.1M/862M [00:01<1:57:38, 121kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<1:21:56, 172kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<57:04, 246kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:01<39:47, 350kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:01<27:45, 498kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:01<19:23, 708kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:02<13:34, 1.00MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<09:31, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.0M/862M [00:03<07:56, 1.70MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.1M/862M [00:03<06:00, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<06:41, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<07:48, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:05<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:05<04:34, 2.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<04:19, 3.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:07<8:09:58, 27.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.0M/862M [00:07<5:42:48, 38.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<4:01:46, 54.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<2:52:01, 77.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<2:00:57, 110kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:09<1:24:29, 156kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:11<1:21:57, 161kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<58:43, 225kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 71.8M/862M [00:11<41:22, 318kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:13<31:55, 411kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:13<25:01, 525kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<18:05, 725kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:13<12:46, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<20:06, 650kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:15<15:26, 846kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<11:04, 1.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:16<10:47, 1.21MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<08:52, 1.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<06:31, 1.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<07:37, 1.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<07:59, 1.62MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<06:09, 2.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:19<04:27, 2.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<11:22, 1.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:21<09:18, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<06:47, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<07:44, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:23<06:43, 1.90MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:23<05:01, 2.54MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<06:31, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<05:51, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:22, 2.90MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:04, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:49, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:25, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<03:55, 3.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<12:10:30, 17.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<8:32:25, 24.6kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<5:58:16, 35.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<4:13:00, 49.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<2:58:17, 70.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<2:04:49, 100kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:30:03, 138kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:05:34, 190kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<46:23, 268kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<32:34, 381kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<28:16, 438kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<20:20, 608kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<14:36, 845kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<13:04, 942kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<10:25, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<07:33, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<08:09, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:57, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<05:10, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:29, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<04:21, 2.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:54, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:22, 2.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:03, 2.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<03:39, 3.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<7:53:35, 25.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<5:31:20, 36.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<3:51:08, 51.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<3:58:29, 50.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<2:49:51, 70.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<1:59:30, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<1:23:31, 143kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:07:03, 178kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<47:59, 248kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<33:51, 351kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<23:46, 498kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<32:07, 369kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<24:54, 475kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<18:01, 656kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<14:28, 813kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:20, 1.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<08:13, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:29, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:07, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:17, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:25, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:52, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:24, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<03:55, 2.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<1:37:02, 119kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<1:09:04, 167kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<48:30, 238kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<36:32, 315kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<26:44, 430kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<18:58, 604kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<15:56, 717kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<13:29, 847kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<10:01, 1.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<07:06, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<10:59:42, 17.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<7:42:41, 24.5kB/s] .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<5:23:26, 35.0kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<3:48:34, 49.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<2:39:46, 70.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<1:54:04, 98.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<1:20:46, 139kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<56:38, 198kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<39:41, 281kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<1:09:57, 159kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<51:14, 218kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<36:24, 306kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<25:30, 435kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<1:37:02, 114kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<1:09:03, 160kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<48:30, 228kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<36:24, 303kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<26:37, 414kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<18:52, 582kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<15:43, 696kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<12:08, 901kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<08:43, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:38, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<08:16, 1.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:21, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<04:34, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<1:20:20, 135kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<57:19, 188kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<40:19, 267kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<30:37, 351kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<23:37, 454kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<17:04, 628kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:21<12:02, 887kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<1:24:39, 126kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<1:00:20, 177kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<42:24, 251kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<30:17, 351kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<7:27:00, 23.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<5:13:11, 33.9kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<3:38:59, 48.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<2:35:22, 67.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<1:50:58, 95.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<1:18:09, 135kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<54:34, 192kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<1:02:23, 168kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<44:44, 234kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<31:29, 332kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<24:22, 427kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<19:18, 539kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<14:03, 739kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<09:54, 1.04MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<1:12:46, 142kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<51:59, 199kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<36:34, 282kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<27:53, 368kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<21:43, 473kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<15:43, 652kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<11:04, 921kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<1:20:34, 127kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<57:26, 178kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<40:19, 252kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<30:30, 332kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<22:23, 452kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<15:54, 635kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<13:25, 750kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<10:26, 963kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:33, 1.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:35, 1.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:27, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:44, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:07, 2.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<1:13:29, 135kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<52:27, 189kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<36:50, 269kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<28:00, 352kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<20:28, 481kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<14:33, 675kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<12:25, 788kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<09:42, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<07:02, 1.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<07:10, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:02, 1.61MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<04:28, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:21, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:43, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:30, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:15, 2.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<1:10:45, 135kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<50:31, 190kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<35:28, 269kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<26:58, 353kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<20:50, 456kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<15:04, 631kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<10:36, 890kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<1:14:39, 127kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<53:12, 177kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<37:23, 252kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<28:15, 332kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<21:41, 432kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<15:35, 600kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:00<10:59, 848kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<13:10, 707kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<10:02, 927kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<07:23, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<05:36, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<6:15:01, 24.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<4:22:43, 35.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<3:03:07, 50.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<2:14:14, 68.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<1:35:53, 95.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<1:07:26, 136kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<47:10, 194kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<36:30, 250kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<26:29, 344kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<18:43, 485kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<15:09, 597kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<12:32, 721kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:10, 985kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<06:29, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<11:30, 780kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<08:59, 997kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<06:30, 1.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:36, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:24, 1.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:59, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<02:53, 3.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<24:15, 364kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<18:46, 471kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<13:35, 649kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<09:34, 916kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<1:09:13, 127kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<49:11, 178kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<34:34, 253kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<26:07, 333kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<20:03, 434kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<14:24, 603kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<10:09, 851kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<11:09, 774kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<08:42, 991kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<06:18, 1.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<06:22, 1.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<06:13, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:47, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<03:26, 2.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<1:03:00, 135kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<44:58, 189kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<31:36, 268kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<23:59, 351kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<18:30, 455kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<13:22, 629kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<09:24, 889kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<38:47, 216kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<28:00, 298kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<19:43, 422kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<15:41, 528kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<11:50, 699kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<08:28, 974kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<07:49, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:19, 1.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<04:37, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<05:07, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:26, 1.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<03:18, 2.46MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:11, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:35, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:37, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:37<02:37, 3.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<1:07:13, 119kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<47:51, 167kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<33:35, 238kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<25:15, 315kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<19:18, 412kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<13:54, 570kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<10:03, 784kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<5:02:23, 26.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<3:31:24, 37.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<2:28:31, 52.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<1:45:57, 73.8kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<1:14:33, 105kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<52:02, 149kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<40:45, 190kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<29:22, 264kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<20:41, 373kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<16:03, 478kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<12:48, 599kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<09:20, 821kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<06:35, 1.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<33:46, 225kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<24:17, 313kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<17:06, 443kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<12:01, 627kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<23:07, 326kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<17:43, 425kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<12:46, 589kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:53<08:59, 832kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<59:42, 125kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<42:25, 176kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<29:47, 250kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<22:28, 330kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<16:28, 449kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<11:39, 633kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<09:51, 745kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<07:38, 960kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<05:31, 1.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<05:32, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:25, 1.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:10, 1.74MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<02:59, 2.41MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<28:04, 256kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<20:16, 355kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<14:20, 500kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<11:38, 613kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<09:39, 738kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:03, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<05:01, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:21, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:11, 1.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:48, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:16, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:29, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:30, 1.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<02:31, 2.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<51:06, 136kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<36:27, 190kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<25:35, 269kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<19:24, 353kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<15:03, 455kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<10:52, 629kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<07:38, 889kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<29:42, 229kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<21:29, 316kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<15:09, 446kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<12:06, 555kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<09:56, 676kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<07:15, 925kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<05:07, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<07:40, 867kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<06:03, 1.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<04:23, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:35, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:53, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:51, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:31, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:36, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<02:59, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:14, 2.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:00, 2.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:46, 2.32MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:06, 3.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<01:51, 3.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<4:24:33, 24.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<3:05:15, 34.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<2:08:53, 49.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<1:33:41, 67.4kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<1:06:57, 94.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<47:08, 134kB/s]   .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<32:48, 191kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<32:10, 194kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<23:08, 270kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<16:16, 382kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<12:46, 484kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<10:15, 602kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<07:29, 823kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<05:16, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<25:45, 237kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<18:32, 329kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<13:06, 464kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<09:11, 658kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<21:02, 287kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<15:21, 393kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<10:49, 555kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<08:56, 668kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<07:31, 793kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:31, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<03:56, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:51, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:00, 1.47MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:56, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:24, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:38, 1.60MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<02:48, 2.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:01, 2.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<04:21, 1.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:32, 1.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:37, 2.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:08, 1.81MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:25, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:41, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:56, 2.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<41:27, 136kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<29:34, 190kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<20:43, 270kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<15:42, 354kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<11:29, 484kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<08:08, 679kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<06:56, 792kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<06:02, 908kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:28, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<03:10, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<08:42, 623kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<06:39, 814kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<04:46, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:33, 1.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<04:20, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:16, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<02:19, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<09:24, 562kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<07:08, 740kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<05:06, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:45, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:26, 1.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:19, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:25, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:08, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:44, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:01, 2.52MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:35, 1.96MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:51, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:13, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<01:36, 3.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<07:01, 713kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<05:26, 920kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<03:55, 1.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<03:02, 1.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<3:22:40, 24.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<2:21:49, 34.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<1:38:25, 49.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<1:12:34, 67.3kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<51:45, 94.3kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<36:21, 134kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<25:19, 191kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<19:59, 241kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<14:23, 334kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<10:08, 472kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<07:07, 667kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<16:17, 291kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<12:24, 382kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<08:52, 533kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<06:14, 753kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<06:09, 759kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:48, 972kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<03:28, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:28, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:24, 1.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:37, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<01:52, 2.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<38:50, 117kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<27:01, 167kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<19:57, 224kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<14:25, 310kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<10:08, 437kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<08:04, 545kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<06:05, 721kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<04:20, 1.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<04:02, 1.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:42, 1.17MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:48, 1.53MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:59, 2.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<17:50, 239kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<12:54, 330kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<09:04, 466kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<07:17, 576kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<05:31, 758kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:57, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<03:42, 1.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:26, 1.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:37, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:51, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<34:24, 118kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<24:27, 166kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<17:07, 235kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<12:48, 312kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<09:21, 426kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<06:36, 599kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<05:30, 713kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<04:39, 843kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<03:25, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<02:25, 1.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:33, 1.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:53, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<02:06, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:58, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<01:29, 2.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:40<01:04, 3.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<07:46, 479kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<05:49, 638kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<04:08, 890kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:43, 982kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:23, 1.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:31, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:49, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:35, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<2:17:16, 26.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<1:35:38, 37.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<1:06:43, 52.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<47:27, 74.1kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<33:17, 105kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:48<23:01, 150kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<20:24, 169kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<14:36, 236kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<10:13, 334kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<07:52, 429kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<06:12, 544kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:29, 751kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<03:08, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<04:32, 729kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:31, 940kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:31, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:30, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:26, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:50, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:19, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:14, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:54, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:23, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:41, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:50, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<01:01, 2.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:57, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:38, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:14, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<00:53, 3.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<07:48, 381kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<06:05, 487kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<04:22, 675kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<03:04, 949kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<03:11, 909kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<02:31, 1.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:49, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:55, 1.47MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:57, 1.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:30, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:04, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<20:28, 135kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<14:35, 189kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<10:10, 268kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<07:39, 352kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<05:56, 454kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<04:16, 627kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<02:57, 887kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<18:56, 139kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<13:29, 194kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<09:25, 276kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<07:05, 361kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<05:13, 489kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:15<03:40, 688kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:07, 799kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<02:26, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<01:45, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:46, 1.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:29, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<01:05, 2.19MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:18, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:09, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:51, 2.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:07, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:01, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:45, 2.95MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:02, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:11, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:55, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:40, 3.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:17, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:07, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:49, 2.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:42, 2.97MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<1:27:45, 23.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<1:01:13, 33.9kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<41:59, 48.4kB/s]  .vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<30:20, 66.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<21:39, 93.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<15:11, 132kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<10:24, 188kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<09:36, 203kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<06:54, 281kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<04:49, 397kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<03:44, 501kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<03:00, 624kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:11, 852kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:35<01:30, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<08:02, 225kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<05:45, 313kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<04:03, 441kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<02:47, 625kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<06:20, 275kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<04:47, 363kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<03:24, 507kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:39<02:21, 717kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:22, 702kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:50, 908kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:18, 1.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:16, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:13, 1.31MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:55, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:38, 2.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<12:58, 118kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<09:12, 166kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<06:22, 236kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<04:41, 313kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<03:34, 409kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:32, 570kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<01:46, 803kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:44, 803kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:21, 1.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:58, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:58, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:57, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:42, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:31, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:42, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:37, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:27, 2.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:35, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:39, 1.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:31, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<00:21, 3.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<08:16, 136kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<05:52, 190kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<04:03, 270kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<02:59, 354kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<02:11, 480kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<01:31, 675kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:15, 789kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:58, 1.01MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:41, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:40, 1.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:33, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:23, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:02<00:16, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<02:20, 364kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:48, 468kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:17, 650kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:52, 916kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:55, 845kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:43, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:30, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:30, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:25, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:18, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:21, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:13, 2.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 3.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<24:31, 23.8kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<16:52, 33.9kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<10:54, 48.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<07:44, 66.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<05:29, 93.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<03:46, 132kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<02:23, 188kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<02:59, 149kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<02:06, 208kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<01:24, 295kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:59, 383kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:42, 518kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:28, 728kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:22, 836kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:17, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:11, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.82MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:04, 2.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<01:16, 135kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:52, 189kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:31, 268kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:17, 352kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:13, 456kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:08, 632kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:02, 895kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:07, 279kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:04, 382kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 539kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 596/400000 [00:00<01:07, 5958.44it/s]  0%|          | 1428/400000 [00:00<01:01, 6511.82it/s]  1%|          | 2271/400000 [00:00<00:56, 6988.37it/s]  1%|          | 3037/400000 [00:00<00:55, 7176.91it/s]  1%|          | 3855/400000 [00:00<00:53, 7450.90it/s]  1%|          | 4704/400000 [00:00<00:51, 7732.76it/s]  1%|▏         | 5539/400000 [00:00<00:49, 7906.26it/s]  2%|▏         | 6386/400000 [00:00<00:48, 8066.29it/s]  2%|▏         | 7237/400000 [00:00<00:47, 8192.99it/s]  2%|▏         | 8038/400000 [00:01<00:48, 8128.60it/s]  2%|▏         | 8887/400000 [00:01<00:47, 8232.24it/s]  2%|▏         | 9733/400000 [00:01<00:47, 8296.78it/s]  3%|▎         | 10570/400000 [00:01<00:46, 8318.09it/s]  3%|▎         | 11419/400000 [00:01<00:46, 8368.28it/s]  3%|▎         | 12264/400000 [00:01<00:46, 8390.31it/s]  3%|▎         | 13115/400000 [00:01<00:45, 8424.37it/s]  3%|▎         | 13957/400000 [00:01<00:46, 8385.07it/s]  4%|▎         | 14802/400000 [00:01<00:45, 8403.77it/s]  4%|▍         | 15652/400000 [00:01<00:45, 8429.84it/s]  4%|▍         | 16495/400000 [00:02<00:45, 8340.15it/s]  4%|▍         | 17333/400000 [00:02<00:45, 8350.49it/s]  5%|▍         | 18169/400000 [00:02<00:46, 8285.96it/s]  5%|▍         | 19020/400000 [00:02<00:45, 8350.11it/s]  5%|▍         | 19856/400000 [00:02<00:45, 8301.30it/s]  5%|▌         | 20687/400000 [00:02<00:45, 8303.17it/s]  5%|▌         | 21533/400000 [00:02<00:45, 8348.97it/s]  6%|▌         | 22384/400000 [00:02<00:44, 8395.85it/s]  6%|▌         | 23224/400000 [00:02<00:44, 8396.65it/s]  6%|▌         | 24064/400000 [00:02<00:44, 8379.09it/s]  6%|▌         | 24918/400000 [00:03<00:44, 8424.55it/s]  6%|▋         | 25764/400000 [00:03<00:44, 8433.14it/s]  7%|▋         | 26608/400000 [00:03<00:44, 8411.43it/s]  7%|▋         | 27461/400000 [00:03<00:44, 8443.72it/s]  7%|▋         | 28313/400000 [00:03<00:43, 8465.28it/s]  7%|▋         | 29160/400000 [00:03<00:44, 8409.07it/s]  8%|▊         | 30002/400000 [00:03<00:44, 8307.17it/s]  8%|▊         | 30851/400000 [00:03<00:44, 8358.62it/s]  8%|▊         | 31699/400000 [00:03<00:43, 8393.27it/s]  8%|▊         | 32540/400000 [00:03<00:43, 8396.67it/s]  8%|▊         | 33380/400000 [00:04<00:43, 8353.82it/s]  9%|▊         | 34217/400000 [00:04<00:43, 8355.75it/s]  9%|▉         | 35071/400000 [00:04<00:43, 8409.01it/s]  9%|▉         | 35913/400000 [00:04<00:43, 8410.10it/s]  9%|▉         | 36767/400000 [00:04<00:43, 8446.46it/s]  9%|▉         | 37612/400000 [00:04<00:43, 8403.08it/s] 10%|▉         | 38453/400000 [00:04<00:43, 8398.19it/s] 10%|▉         | 39305/400000 [00:04<00:42, 8431.67it/s] 10%|█         | 40149/400000 [00:04<00:42, 8410.52it/s] 10%|█         | 40997/400000 [00:04<00:42, 8430.97it/s] 10%|█         | 41841/400000 [00:05<00:42, 8417.43it/s] 11%|█         | 42683/400000 [00:05<00:43, 8307.78it/s] 11%|█         | 43535/400000 [00:05<00:42, 8369.84it/s] 11%|█         | 44389/400000 [00:05<00:42, 8418.06it/s] 11%|█▏        | 45239/400000 [00:05<00:42, 8442.00it/s] 12%|█▏        | 46086/400000 [00:05<00:41, 8448.46it/s] 12%|█▏        | 46932/400000 [00:05<00:41, 8430.08it/s] 12%|█▏        | 47785/400000 [00:05<00:41, 8457.09it/s] 12%|█▏        | 48637/400000 [00:05<00:41, 8473.79it/s] 12%|█▏        | 49490/400000 [00:05<00:41, 8487.94it/s] 13%|█▎        | 50339/400000 [00:06<00:41, 8448.20it/s] 13%|█▎        | 51184/400000 [00:06<00:41, 8444.99it/s] 13%|█▎        | 52036/400000 [00:06<00:41, 8465.64it/s] 13%|█▎        | 52885/400000 [00:06<00:40, 8472.27it/s] 13%|█▎        | 53739/400000 [00:06<00:40, 8490.31it/s] 14%|█▎        | 54594/400000 [00:06<00:40, 8507.46it/s] 14%|█▍        | 55445/400000 [00:06<00:40, 8488.69it/s] 14%|█▍        | 56299/400000 [00:06<00:40, 8503.23it/s] 14%|█▍        | 57152/400000 [00:06<00:40, 8510.77it/s] 15%|█▍        | 58004/400000 [00:06<00:40, 8357.16it/s] 15%|█▍        | 58841/400000 [00:07<00:41, 8189.44it/s] 15%|█▍        | 59662/400000 [00:07<00:42, 8102.59it/s] 15%|█▌        | 60501/400000 [00:07<00:41, 8184.67it/s] 15%|█▌        | 61342/400000 [00:07<00:41, 8248.89it/s] 16%|█▌        | 62183/400000 [00:07<00:40, 8295.58it/s] 16%|█▌        | 63031/400000 [00:07<00:40, 8349.07it/s] 16%|█▌        | 63867/400000 [00:07<00:40, 8290.45it/s] 16%|█▌        | 64720/400000 [00:07<00:40, 8358.26it/s] 16%|█▋        | 65573/400000 [00:07<00:39, 8406.67it/s] 17%|█▋        | 66426/400000 [00:07<00:39, 8443.24it/s] 17%|█▋        | 67281/400000 [00:08<00:39, 8474.31it/s] 17%|█▋        | 68129/400000 [00:08<00:42, 7894.97it/s] 17%|█▋        | 68964/400000 [00:08<00:41, 8025.45it/s] 17%|█▋        | 69817/400000 [00:08<00:40, 8169.41it/s] 18%|█▊        | 70665/400000 [00:08<00:39, 8258.75it/s] 18%|█▊        | 71497/400000 [00:08<00:39, 8275.09it/s] 18%|█▊        | 72333/400000 [00:08<00:39, 8298.88it/s] 18%|█▊        | 73165/400000 [00:08<00:39, 8286.67it/s] 18%|█▊        | 73995/400000 [00:08<00:39, 8174.40it/s] 19%|█▊        | 74841/400000 [00:08<00:39, 8255.85it/s] 19%|█▉        | 75668/400000 [00:09<00:39, 8181.33it/s] 19%|█▉        | 76488/400000 [00:09<00:39, 8137.95it/s] 19%|█▉        | 77324/400000 [00:09<00:39, 8202.16it/s] 20%|█▉        | 78173/400000 [00:09<00:38, 8284.09it/s] 20%|█▉        | 79006/400000 [00:09<00:38, 8297.30it/s] 20%|█▉        | 79854/400000 [00:09<00:38, 8350.57it/s] 20%|██        | 80690/400000 [00:09<00:38, 8257.26it/s] 20%|██        | 81539/400000 [00:09<00:38, 8324.54it/s] 21%|██        | 82392/400000 [00:09<00:37, 8383.22it/s] 21%|██        | 83237/400000 [00:09<00:37, 8401.28it/s] 21%|██        | 84078/400000 [00:10<00:37, 8363.82it/s] 21%|██        | 84915/400000 [00:10<00:38, 8170.22it/s] 21%|██▏       | 85758/400000 [00:10<00:38, 8245.46it/s] 22%|██▏       | 86598/400000 [00:10<00:37, 8289.20it/s] 22%|██▏       | 87431/400000 [00:10<00:37, 8300.55it/s] 22%|██▏       | 88262/400000 [00:10<00:37, 8276.74it/s] 22%|██▏       | 89091/400000 [00:10<00:37, 8238.18it/s] 22%|██▏       | 89916/400000 [00:10<00:38, 8045.84it/s] 23%|██▎       | 90751/400000 [00:10<00:38, 8134.20it/s] 23%|██▎       | 91574/400000 [00:11<00:37, 8146.81it/s] 23%|██▎       | 92425/400000 [00:11<00:37, 8251.39it/s] 23%|██▎       | 93252/400000 [00:11<00:37, 8256.66it/s] 24%|██▎       | 94079/400000 [00:11<00:37, 8221.78it/s] 24%|██▎       | 94930/400000 [00:11<00:36, 8305.82it/s] 24%|██▍       | 95780/400000 [00:11<00:36, 8361.87it/s] 24%|██▍       | 96617/400000 [00:11<00:36, 8252.92it/s] 24%|██▍       | 97443/400000 [00:11<00:36, 8251.20it/s] 25%|██▍       | 98273/400000 [00:11<00:36, 8264.31it/s] 25%|██▍       | 99121/400000 [00:11<00:36, 8326.25it/s] 25%|██▍       | 99973/400000 [00:12<00:35, 8382.91it/s] 25%|██▌       | 100812/400000 [00:12<00:35, 8349.53it/s] 25%|██▌       | 101663/400000 [00:12<00:35, 8394.34it/s] 26%|██▌       | 102513/400000 [00:12<00:35, 8424.49it/s] 26%|██▌       | 103368/400000 [00:12<00:35, 8458.79it/s] 26%|██▌       | 104221/400000 [00:12<00:34, 8479.32it/s] 26%|██▋       | 105070/400000 [00:12<00:35, 8399.92it/s] 26%|██▋       | 105917/400000 [00:12<00:34, 8418.27it/s] 27%|██▋       | 106763/400000 [00:12<00:34, 8428.37it/s] 27%|██▋       | 107607/400000 [00:12<00:34, 8429.90it/s] 27%|██▋       | 108451/400000 [00:13<00:34, 8371.28it/s] 27%|██▋       | 109296/400000 [00:13<00:34, 8393.91it/s] 28%|██▊       | 110136/400000 [00:13<00:34, 8362.81it/s] 28%|██▊       | 110973/400000 [00:13<00:34, 8258.27it/s] 28%|██▊       | 111809/400000 [00:13<00:34, 8283.81it/s] 28%|██▊       | 112644/400000 [00:13<00:34, 8300.89it/s] 28%|██▊       | 113475/400000 [00:13<00:34, 8214.56it/s] 29%|██▊       | 114297/400000 [00:13<00:34, 8166.26it/s] 29%|██▉       | 115119/400000 [00:13<00:34, 8182.04it/s] 29%|██▉       | 115967/400000 [00:13<00:34, 8267.53it/s] 29%|██▉       | 116812/400000 [00:14<00:34, 8319.01it/s] 29%|██▉       | 117658/400000 [00:14<00:33, 8358.84it/s] 30%|██▉       | 118508/400000 [00:14<00:33, 8399.83it/s] 30%|██▉       | 119349/400000 [00:14<00:34, 8200.23it/s] 30%|███       | 120197/400000 [00:14<00:33, 8280.35it/s] 30%|███       | 121052/400000 [00:14<00:33, 8357.65it/s] 30%|███       | 121904/400000 [00:14<00:33, 8405.05it/s] 31%|███       | 122753/400000 [00:14<00:32, 8429.86it/s] 31%|███       | 123597/400000 [00:14<00:32, 8427.73it/s] 31%|███       | 124450/400000 [00:14<00:32, 8457.03it/s] 31%|███▏      | 125304/400000 [00:15<00:32, 8480.19it/s] 32%|███▏      | 126154/400000 [00:15<00:32, 8485.84it/s] 32%|███▏      | 127004/400000 [00:15<00:32, 8489.21it/s] 32%|███▏      | 127854/400000 [00:15<00:32, 8463.61it/s] 32%|███▏      | 128707/400000 [00:15<00:31, 8480.54it/s] 32%|███▏      | 129559/400000 [00:15<00:31, 8492.33it/s] 33%|███▎      | 130410/400000 [00:15<00:31, 8493.75it/s] 33%|███▎      | 131260/400000 [00:15<00:32, 8340.90it/s] 33%|███▎      | 132095/400000 [00:15<00:32, 8233.23it/s] 33%|███▎      | 132920/400000 [00:15<00:33, 8064.18it/s] 33%|███▎      | 133763/400000 [00:16<00:32, 8168.59it/s] 34%|███▎      | 134611/400000 [00:16<00:32, 8258.11it/s] 34%|███▍      | 135454/400000 [00:16<00:31, 8306.21it/s] 34%|███▍      | 136300/400000 [00:16<00:31, 8350.36it/s] 34%|███▍      | 137136/400000 [00:16<00:31, 8351.50it/s] 34%|███▍      | 137979/400000 [00:16<00:31, 8374.81it/s] 35%|███▍      | 138819/400000 [00:16<00:31, 8382.04it/s] 35%|███▍      | 139660/400000 [00:16<00:31, 8388.50it/s] 35%|███▌      | 140500/400000 [00:16<00:30, 8371.85it/s] 35%|███▌      | 141348/400000 [00:16<00:30, 8401.39it/s] 36%|███▌      | 142189/400000 [00:17<00:30, 8362.17it/s] 36%|███▌      | 143038/400000 [00:17<00:30, 8399.03it/s] 36%|███▌      | 143879/400000 [00:17<00:30, 8367.09it/s] 36%|███▌      | 144726/400000 [00:17<00:30, 8396.21it/s] 36%|███▋      | 145579/400000 [00:17<00:30, 8433.54it/s] 37%|███▋      | 146423/400000 [00:17<00:30, 8398.91it/s] 37%|███▋      | 147275/400000 [00:17<00:29, 8432.74it/s] 37%|███▋      | 148121/400000 [00:17<00:29, 8438.43it/s] 37%|███▋      | 148968/400000 [00:17<00:29, 8447.86it/s] 37%|███▋      | 149820/400000 [00:17<00:29, 8468.15it/s] 38%|███▊      | 150674/400000 [00:18<00:29, 8486.76it/s] 38%|███▊      | 151527/400000 [00:18<00:29, 8497.64it/s] 38%|███▊      | 152377/400000 [00:18<00:29, 8488.95it/s] 38%|███▊      | 153226/400000 [00:18<00:29, 8399.96it/s] 39%|███▊      | 154067/400000 [00:18<00:30, 8099.69it/s] 39%|███▊      | 154910/400000 [00:18<00:29, 8194.49it/s] 39%|███▉      | 155762/400000 [00:18<00:29, 8288.58it/s] 39%|███▉      | 156605/400000 [00:18<00:29, 8328.88it/s] 39%|███▉      | 157441/400000 [00:18<00:29, 8336.08it/s] 40%|███▉      | 158278/400000 [00:18<00:28, 8345.94it/s] 40%|███▉      | 159114/400000 [00:19<00:28, 8330.31it/s] 40%|███▉      | 159968/400000 [00:19<00:28, 8391.54it/s] 40%|████      | 160808/400000 [00:19<00:28, 8375.78it/s] 40%|████      | 161646/400000 [00:19<00:28, 8248.79it/s] 41%|████      | 162486/400000 [00:19<00:28, 8292.17it/s] 41%|████      | 163316/400000 [00:19<00:28, 8253.40it/s] 41%|████      | 164169/400000 [00:19<00:28, 8332.60it/s] 41%|████▏     | 165011/400000 [00:19<00:28, 8356.36it/s] 41%|████▏     | 165851/400000 [00:19<00:27, 8368.22it/s] 42%|████▏     | 166689/400000 [00:20<00:27, 8352.29it/s] 42%|████▏     | 167538/400000 [00:20<00:27, 8392.71it/s] 42%|████▏     | 168388/400000 [00:20<00:27, 8422.50it/s] 42%|████▏     | 169231/400000 [00:20<00:27, 8384.74it/s] 43%|████▎     | 170070/400000 [00:20<00:27, 8375.30it/s] 43%|████▎     | 170921/400000 [00:20<00:27, 8412.46it/s] 43%|████▎     | 171763/400000 [00:20<00:27, 8414.00it/s] 43%|████▎     | 172613/400000 [00:20<00:26, 8438.97it/s] 43%|████▎     | 173462/400000 [00:20<00:26, 8451.45it/s] 44%|████▎     | 174308/400000 [00:20<00:26, 8442.38it/s] 44%|████▍     | 175158/400000 [00:21<00:26, 8458.89it/s] 44%|████▍     | 176004/400000 [00:21<00:26, 8455.89it/s] 44%|████▍     | 176850/400000 [00:21<00:26, 8450.73it/s] 44%|████▍     | 177696/400000 [00:21<00:26, 8422.57it/s] 45%|████▍     | 178539/400000 [00:21<00:26, 8422.44it/s] 45%|████▍     | 179382/400000 [00:21<00:26, 8422.07it/s] 45%|████▌     | 180225/400000 [00:21<00:26, 8396.04it/s] 45%|████▌     | 181078/400000 [00:21<00:25, 8433.68it/s] 45%|████▌     | 181928/400000 [00:21<00:25, 8452.85it/s] 46%|████▌     | 182774/400000 [00:21<00:25, 8418.28it/s] 46%|████▌     | 183627/400000 [00:22<00:25, 8448.22it/s] 46%|████▌     | 184484/400000 [00:22<00:25, 8482.27it/s] 46%|████▋     | 185333/400000 [00:22<00:25, 8483.68it/s] 47%|████▋     | 186182/400000 [00:22<00:25, 8413.47it/s] 47%|████▋     | 187024/400000 [00:22<00:26, 8176.43it/s] 47%|████▋     | 187844/400000 [00:22<00:26, 8101.38it/s] 47%|████▋     | 188656/400000 [00:22<00:26, 8030.14it/s] 47%|████▋     | 189461/400000 [00:22<00:26, 8014.44it/s] 48%|████▊     | 190312/400000 [00:22<00:25, 8156.86it/s] 48%|████▊     | 191145/400000 [00:22<00:25, 8207.52it/s] 48%|████▊     | 191967/400000 [00:23<00:25, 8043.34it/s] 48%|████▊     | 192773/400000 [00:23<00:25, 8006.25it/s] 48%|████▊     | 193575/400000 [00:23<00:25, 7973.40it/s] 49%|████▊     | 194374/400000 [00:23<00:25, 7969.93it/s] 49%|████▉     | 195172/400000 [00:23<00:26, 7843.82it/s] 49%|████▉     | 195998/400000 [00:23<00:25, 7962.15it/s] 49%|████▉     | 196829/400000 [00:23<00:25, 8061.91it/s] 49%|████▉     | 197644/400000 [00:23<00:25, 8073.46it/s] 50%|████▉     | 198487/400000 [00:23<00:24, 8176.70it/s] 50%|████▉     | 199319/400000 [00:23<00:24, 8218.53it/s] 50%|█████     | 200142/400000 [00:24<00:24, 8213.15it/s] 50%|█████     | 200981/400000 [00:24<00:24, 8263.13it/s] 50%|█████     | 201830/400000 [00:24<00:23, 8327.75it/s] 51%|█████     | 202673/400000 [00:24<00:23, 8356.80it/s] 51%|█████     | 203513/400000 [00:24<00:23, 8368.07it/s] 51%|█████     | 204351/400000 [00:24<00:23, 8283.75it/s] 51%|█████▏    | 205195/400000 [00:24<00:23, 8329.51it/s] 52%|█████▏    | 206042/400000 [00:24<00:23, 8370.49it/s] 52%|█████▏    | 206880/400000 [00:24<00:23, 8369.00it/s] 52%|█████▏    | 207718/400000 [00:24<00:23, 8331.10it/s] 52%|█████▏    | 208552/400000 [00:25<00:23, 8318.25it/s] 52%|█████▏    | 209398/400000 [00:25<00:22, 8358.40it/s] 53%|█████▎    | 210234/400000 [00:25<00:22, 8348.34it/s] 53%|█████▎    | 211079/400000 [00:25<00:22, 8376.14it/s] 53%|█████▎    | 211923/400000 [00:25<00:22, 8392.43it/s] 53%|█████▎    | 212768/400000 [00:25<00:22, 8408.80it/s] 53%|█████▎    | 213615/400000 [00:25<00:22, 8426.85it/s] 54%|█████▎    | 214463/400000 [00:25<00:21, 8441.95it/s] 54%|█████▍    | 215308/400000 [00:25<00:21, 8432.65it/s] 54%|█████▍    | 216152/400000 [00:25<00:21, 8427.27it/s] 54%|█████▍    | 216995/400000 [00:26<00:22, 8200.33it/s] 54%|█████▍    | 217837/400000 [00:26<00:22, 8264.88it/s] 55%|█████▍    | 218668/400000 [00:26<00:21, 8276.47it/s] 55%|█████▍    | 219509/400000 [00:26<00:21, 8315.81it/s] 55%|█████▌    | 220342/400000 [00:26<00:22, 7989.05it/s] 55%|█████▌    | 221145/400000 [00:26<00:22, 7882.25it/s] 55%|█████▌    | 221997/400000 [00:26<00:22, 8060.94it/s] 56%|█████▌    | 222844/400000 [00:26<00:21, 8176.97it/s] 56%|█████▌    | 223665/400000 [00:26<00:21, 8186.32it/s] 56%|█████▌    | 224496/400000 [00:26<00:21, 8221.29it/s] 56%|█████▋    | 225332/400000 [00:27<00:21, 8260.74it/s] 57%|█████▋    | 226173/400000 [00:27<00:20, 8304.39it/s] 57%|█████▋    | 227020/400000 [00:27<00:20, 8350.88it/s] 57%|█████▋    | 227867/400000 [00:27<00:20, 8386.08it/s] 57%|█████▋    | 228707/400000 [00:27<00:20, 8358.79it/s] 57%|█████▋    | 229544/400000 [00:27<00:20, 8330.74it/s] 58%|█████▊    | 230393/400000 [00:27<00:20, 8376.89it/s] 58%|█████▊    | 231241/400000 [00:27<00:20, 8404.84it/s] 58%|█████▊    | 232095/400000 [00:27<00:19, 8442.39it/s] 58%|█████▊    | 232940/400000 [00:27<00:19, 8416.85it/s] 58%|█████▊    | 233785/400000 [00:28<00:19, 8425.32it/s] 59%|█████▊    | 234637/400000 [00:28<00:19, 8451.50it/s] 59%|█████▉    | 235483/400000 [00:28<00:19, 8444.42it/s] 59%|█████▉    | 236328/400000 [00:28<00:19, 8391.74it/s] 59%|█████▉    | 237168/400000 [00:28<00:19, 8270.27it/s] 59%|█████▉    | 237999/400000 [00:28<00:19, 8279.96it/s] 60%|█████▉    | 238839/400000 [00:28<00:19, 8314.99it/s] 60%|█████▉    | 239671/400000 [00:28<00:19, 8312.93it/s] 60%|██████    | 240516/400000 [00:28<00:19, 8350.78it/s] 60%|██████    | 241352/400000 [00:28<00:18, 8351.78it/s] 61%|██████    | 242188/400000 [00:29<00:18, 8349.96it/s] 61%|██████    | 243024/400000 [00:29<00:18, 8329.66it/s] 61%|██████    | 243858/400000 [00:29<00:18, 8276.63it/s] 61%|██████    | 244690/400000 [00:29<00:18, 8287.41it/s] 61%|██████▏   | 245530/400000 [00:29<00:18, 8319.34it/s] 62%|██████▏   | 246363/400000 [00:29<00:18, 8315.81it/s] 62%|██████▏   | 247214/400000 [00:29<00:18, 8373.06it/s] 62%|██████▏   | 248054/400000 [00:29<00:18, 8380.01it/s] 62%|██████▏   | 248893/400000 [00:29<00:18, 8350.62it/s] 62%|██████▏   | 249737/400000 [00:30<00:17, 8376.87it/s] 63%|██████▎   | 250575/400000 [00:30<00:18, 8282.15it/s] 63%|██████▎   | 251419/400000 [00:30<00:17, 8326.27it/s] 63%|██████▎   | 252263/400000 [00:30<00:17, 8359.48it/s] 63%|██████▎   | 253111/400000 [00:30<00:17, 8395.24it/s] 63%|██████▎   | 253952/400000 [00:30<00:17, 8399.54it/s] 64%|██████▎   | 254793/400000 [00:30<00:17, 8181.25it/s] 64%|██████▍   | 255640/400000 [00:30<00:17, 8263.60it/s] 64%|██████▍   | 256486/400000 [00:30<00:17, 8318.76it/s] 64%|██████▍   | 257337/400000 [00:30<00:17, 8372.25it/s] 65%|██████▍   | 258175/400000 [00:31<00:16, 8371.16it/s] 65%|██████▍   | 259013/400000 [00:31<00:16, 8371.65it/s] 65%|██████▍   | 259863/400000 [00:31<00:16, 8409.24it/s] 65%|██████▌   | 260711/400000 [00:31<00:16, 8428.77it/s] 65%|██████▌   | 261555/400000 [00:31<00:16, 8335.12it/s] 66%|██████▌   | 262405/400000 [00:31<00:16, 8383.60it/s] 66%|██████▌   | 263244/400000 [00:31<00:16, 8306.30it/s] 66%|██████▌   | 264076/400000 [00:31<00:17, 7963.24it/s] 66%|██████▌   | 264876/400000 [00:31<00:17, 7948.07it/s] 66%|██████▋   | 265710/400000 [00:31<00:16, 8059.47it/s] 67%|██████▋   | 266518/400000 [00:32<00:16, 8021.75it/s] 67%|██████▋   | 267346/400000 [00:32<00:16, 8096.44it/s] 67%|██████▋   | 268189/400000 [00:32<00:16, 8191.02it/s] 67%|██████▋   | 269033/400000 [00:32<00:15, 8263.42it/s] 67%|██████▋   | 269869/400000 [00:32<00:15, 8290.83it/s] 68%|██████▊   | 270699/400000 [00:32<00:15, 8237.41it/s] 68%|██████▊   | 271546/400000 [00:32<00:15, 8304.74it/s] 68%|██████▊   | 272389/400000 [00:32<00:15, 8340.25it/s] 68%|██████▊   | 273233/400000 [00:32<00:15, 8367.65it/s] 69%|██████▊   | 274085/400000 [00:32<00:14, 8410.03it/s] 69%|██████▊   | 274928/400000 [00:33<00:14, 8414.37it/s] 69%|██████▉   | 275770/400000 [00:33<00:14, 8415.46it/s] 69%|██████▉   | 276612/400000 [00:33<00:14, 8379.91it/s] 69%|██████▉   | 277451/400000 [00:33<00:14, 8358.96it/s] 70%|██████▉   | 278295/400000 [00:33<00:14, 8381.29it/s] 70%|██████▉   | 279134/400000 [00:33<00:14, 8351.76it/s] 70%|██████▉   | 279970/400000 [00:33<00:14, 8352.87it/s] 70%|███████   | 280811/400000 [00:33<00:14, 8367.44it/s] 70%|███████   | 281648/400000 [00:33<00:14, 8331.42it/s] 71%|███████   | 282492/400000 [00:33<00:14, 8362.74it/s] 71%|███████   | 283339/400000 [00:34<00:13, 8393.01it/s] 71%|███████   | 284179/400000 [00:34<00:13, 8344.58it/s] 71%|███████▏  | 285016/400000 [00:34<00:13, 8351.40it/s] 71%|███████▏  | 285860/400000 [00:34<00:13, 8375.28it/s] 72%|███████▏  | 286698/400000 [00:34<00:13, 8372.61it/s] 72%|███████▏  | 287541/400000 [00:34<00:13, 8389.72it/s] 72%|███████▏  | 288381/400000 [00:34<00:13, 8287.22it/s] 72%|███████▏  | 289224/400000 [00:34<00:13, 8327.00it/s] 73%|███████▎  | 290063/400000 [00:34<00:13, 8344.24it/s] 73%|███████▎  | 290913/400000 [00:34<00:13, 8390.34it/s] 73%|███████▎  | 291753/400000 [00:35<00:12, 8338.46it/s] 73%|███████▎  | 292593/400000 [00:35<00:12, 8354.58it/s] 73%|███████▎  | 293429/400000 [00:35<00:12, 8352.79it/s] 74%|███████▎  | 294275/400000 [00:35<00:12, 8382.14it/s] 74%|███████▍  | 295121/400000 [00:35<00:12, 8402.59it/s] 74%|███████▍  | 295964/400000 [00:35<00:12, 8410.23it/s] 74%|███████▍  | 296807/400000 [00:35<00:12, 8413.82it/s] 74%|███████▍  | 297654/400000 [00:35<00:12, 8428.36it/s] 75%|███████▍  | 298499/400000 [00:35<00:12, 8431.93it/s] 75%|███████▍  | 299343/400000 [00:35<00:11, 8423.10it/s] 75%|███████▌  | 300188/400000 [00:36<00:11, 8428.64it/s] 75%|███████▌  | 301031/400000 [00:36<00:12, 8227.07it/s] 75%|███████▌  | 301874/400000 [00:36<00:11, 8286.36it/s] 76%|███████▌  | 302722/400000 [00:36<00:11, 8342.34it/s] 76%|███████▌  | 303565/400000 [00:36<00:11, 8365.55it/s] 76%|███████▌  | 304404/400000 [00:36<00:11, 8369.88it/s] 76%|███████▋  | 305242/400000 [00:36<00:11, 8327.50it/s] 77%|███████▋  | 306084/400000 [00:36<00:11, 8353.33it/s] 77%|███████▋  | 306920/400000 [00:36<00:11, 8349.95it/s] 77%|███████▋  | 307756/400000 [00:36<00:11, 8156.15it/s] 77%|███████▋  | 308573/400000 [00:37<00:11, 7855.16it/s] 77%|███████▋  | 309362/400000 [00:37<00:12, 7487.33it/s] 78%|███████▊  | 310117/400000 [00:37<00:12, 7478.08it/s] 78%|███████▊  | 310906/400000 [00:37<00:11, 7596.68it/s] 78%|███████▊  | 311669/400000 [00:37<00:11, 7594.24it/s] 78%|███████▊  | 312515/400000 [00:37<00:11, 7832.78it/s] 78%|███████▊  | 313335/400000 [00:37<00:10, 7938.68it/s] 79%|███████▊  | 314183/400000 [00:37<00:10, 8091.20it/s] 79%|███████▉  | 315003/400000 [00:37<00:10, 8123.43it/s] 79%|███████▉  | 315852/400000 [00:38<00:10, 8228.00it/s] 79%|███████▉  | 316700/400000 [00:38<00:10, 8301.13it/s] 79%|███████▉  | 317544/400000 [00:38<00:09, 8341.45it/s] 80%|███████▉  | 318396/400000 [00:38<00:09, 8392.03it/s] 80%|███████▉  | 319243/400000 [00:38<00:09, 8413.77it/s] 80%|████████  | 320089/400000 [00:38<00:09, 8426.45it/s] 80%|████████  | 320942/400000 [00:38<00:09, 8456.09it/s] 80%|████████  | 321792/400000 [00:38<00:09, 8468.72it/s] 81%|████████  | 322641/400000 [00:38<00:09, 8473.74it/s] 81%|████████  | 323489/400000 [00:38<00:09, 8467.69it/s] 81%|████████  | 324341/400000 [00:39<00:08, 8483.31it/s] 81%|████████▏ | 325190/400000 [00:39<00:09, 8294.67it/s] 82%|████████▏ | 326021/400000 [00:39<00:09, 8214.21it/s] 82%|████████▏ | 326866/400000 [00:39<00:08, 8281.16it/s] 82%|████████▏ | 327708/400000 [00:39<00:08, 8322.15it/s] 82%|████████▏ | 328560/400000 [00:39<00:08, 8377.84it/s] 82%|████████▏ | 329411/400000 [00:39<00:08, 8416.07it/s] 83%|████████▎ | 330253/400000 [00:39<00:08, 8357.39it/s] 83%|████████▎ | 331092/400000 [00:39<00:08, 8366.96it/s] 83%|████████▎ | 331932/400000 [00:39<00:08, 8374.95it/s] 83%|████████▎ | 332774/400000 [00:40<00:08, 8386.15it/s] 83%|████████▎ | 333617/400000 [00:40<00:07, 8396.83it/s] 84%|████████▎ | 334457/400000 [00:40<00:07, 8385.47it/s] 84%|████████▍ | 335302/400000 [00:40<00:07, 8404.02it/s] 84%|████████▍ | 336154/400000 [00:40<00:07, 8435.60it/s] 84%|████████▍ | 336999/400000 [00:40<00:07, 8438.92it/s] 84%|████████▍ | 337843/400000 [00:40<00:07, 8438.72it/s] 85%|████████▍ | 338687/400000 [00:40<00:07, 8406.78it/s] 85%|████████▍ | 339528/400000 [00:40<00:07, 8295.65it/s] 85%|████████▌ | 340370/400000 [00:40<00:07, 8330.06it/s] 85%|████████▌ | 341218/400000 [00:41<00:07, 8372.32it/s] 86%|████████▌ | 342064/400000 [00:41<00:06, 8397.27it/s] 86%|████████▌ | 342904/400000 [00:41<00:06, 8353.08it/s] 86%|████████▌ | 343740/400000 [00:41<00:06, 8295.76it/s] 86%|████████▌ | 344582/400000 [00:41<00:06, 8331.18it/s] 86%|████████▋ | 345432/400000 [00:41<00:06, 8380.06it/s] 87%|████████▋ | 346276/400000 [00:41<00:06, 8397.16it/s] 87%|████████▋ | 347116/400000 [00:41<00:06, 8345.24it/s] 87%|████████▋ | 347951/400000 [00:41<00:06, 8346.07it/s] 87%|████████▋ | 348786/400000 [00:41<00:06, 8250.45it/s] 87%|████████▋ | 349612/400000 [00:42<00:06, 8210.92it/s] 88%|████████▊ | 350434/400000 [00:42<00:06, 8099.61it/s] 88%|████████▊ | 351245/400000 [00:42<00:06, 7968.63it/s] 88%|████████▊ | 352043/400000 [00:42<00:06, 7839.26it/s] 88%|████████▊ | 352834/400000 [00:42<00:06, 7858.83it/s] 88%|████████▊ | 353627/400000 [00:42<00:05, 7879.97it/s] 89%|████████▊ | 354416/400000 [00:42<00:05, 7882.11it/s] 89%|████████▉ | 355205/400000 [00:42<00:05, 7780.96it/s] 89%|████████▉ | 355997/400000 [00:42<00:05, 7820.67it/s] 89%|████████▉ | 356848/400000 [00:42<00:05, 8010.76it/s] 89%|████████▉ | 357699/400000 [00:43<00:05, 8154.07it/s] 90%|████████▉ | 358547/400000 [00:43<00:05, 8247.36it/s] 90%|████████▉ | 359374/400000 [00:43<00:04, 8177.92it/s] 90%|█████████ | 360219/400000 [00:43<00:04, 8256.64it/s] 90%|█████████ | 361046/400000 [00:43<00:04, 8226.49it/s] 90%|█████████ | 361886/400000 [00:43<00:04, 8277.60it/s] 91%|█████████ | 362732/400000 [00:43<00:04, 8331.22it/s] 91%|█████████ | 363566/400000 [00:43<00:04, 8283.18it/s] 91%|█████████ | 364395/400000 [00:43<00:04, 8273.41it/s] 91%|█████████▏| 365236/400000 [00:43<00:04, 8313.18it/s] 92%|█████████▏| 366081/400000 [00:44<00:04, 8351.86it/s] 92%|█████████▏| 366918/400000 [00:44<00:03, 8354.99it/s] 92%|█████████▏| 367754/400000 [00:44<00:03, 8207.77it/s] 92%|█████████▏| 368576/400000 [00:44<00:03, 8082.07it/s] 92%|█████████▏| 369401/400000 [00:44<00:03, 8130.38it/s] 93%|█████████▎| 370243/400000 [00:44<00:03, 8214.15it/s] 93%|█████████▎| 371086/400000 [00:44<00:03, 8277.43it/s] 93%|█████████▎| 371915/400000 [00:44<00:03, 8262.16it/s] 93%|█████████▎| 372763/400000 [00:44<00:03, 8326.14it/s] 93%|█████████▎| 373605/400000 [00:44<00:03, 8354.07it/s] 94%|█████████▎| 374448/400000 [00:45<00:03, 8373.85it/s] 94%|█████████▍| 375286/400000 [00:45<00:02, 8373.11it/s] 94%|█████████▍| 376124/400000 [00:45<00:02, 8293.16it/s] 94%|█████████▍| 376959/400000 [00:45<00:02, 8308.69it/s] 94%|█████████▍| 377791/400000 [00:45<00:02, 8274.27it/s] 95%|█████████▍| 378619/400000 [00:45<00:02, 8119.69it/s] 95%|█████████▍| 379464/400000 [00:45<00:02, 8214.59it/s] 95%|█████████▌| 380288/400000 [00:45<00:02, 8218.86it/s] 95%|█████████▌| 381128/400000 [00:45<00:02, 8269.62it/s] 95%|█████████▌| 381976/400000 [00:45<00:02, 8329.02it/s] 96%|█████████▌| 382820/400000 [00:46<00:02, 8361.58it/s] 96%|█████████▌| 383664/400000 [00:46<00:01, 8383.89it/s] 96%|█████████▌| 384503/400000 [00:46<00:01, 8285.56it/s] 96%|█████████▋| 385349/400000 [00:46<00:01, 8335.13it/s] 97%|█████████▋| 386183/400000 [00:46<00:01, 8316.76it/s] 97%|█████████▋| 387015/400000 [00:46<00:01, 8000.96it/s] 97%|█████████▋| 387858/400000 [00:46<00:01, 8124.99it/s] 97%|█████████▋| 388686/400000 [00:46<00:01, 8170.67it/s] 97%|█████████▋| 389529/400000 [00:46<00:01, 8244.08it/s] 98%|█████████▊| 390373/400000 [00:47<00:01, 8299.57it/s] 98%|█████████▊| 391205/400000 [00:47<00:01, 8253.62it/s] 98%|█████████▊| 392043/400000 [00:47<00:00, 8288.62it/s] 98%|█████████▊| 392873/400000 [00:47<00:00, 8273.09it/s] 98%|█████████▊| 393701/400000 [00:47<00:00, 7944.32it/s] 99%|█████████▊| 394510/400000 [00:47<00:00, 7986.06it/s] 99%|█████████▉| 395326/400000 [00:47<00:00, 8034.89it/s] 99%|█████████▉| 396172/400000 [00:47<00:00, 8155.48it/s] 99%|█████████▉| 397002/400000 [00:47<00:00, 8195.05it/s] 99%|█████████▉| 397823/400000 [00:47<00:00, 8152.41it/s]100%|█████████▉| 398668/400000 [00:48<00:00, 8237.85it/s]100%|█████████▉| 399504/400000 [00:48<00:00, 8271.54it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8300.10it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fc0178ba208>, <torchtext.data.dataset.TabularDataset object at 0x7fc0178ba358>, <torchtext.vocab.Vocab object at 0x7fc0178ba278>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.76 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.66 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.66 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.61 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.61 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.00 file/s]2020-07-24 18:18:34.585490: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-24 18:18:34.589729: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-24 18:18:34.589912: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a1f3e38bb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-24 18:18:34.589928: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:11, 138320.61it/s] 78%|███████▊  | 7716864/9912422 [00:00<00:11, 197448.84it/s]9920512it [00:00, 41906171.23it/s]                           
0it [00:00, ?it/s]32768it [00:00, 695045.30it/s]
0it [00:00, ?it/s]  1%|▏         | 24576/1648877 [00:00<00:06, 245616.18it/s]1654784it [00:00, 11135462.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 211292.41it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
