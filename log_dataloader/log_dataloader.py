
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5fb18faea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5fb18faea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f601cbbb1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f601cbbb1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f603af02e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f603af02e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5fc9ee2488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5fc9ee2488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5fc9ee2488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 151461.57it/s] 81%|████████▏ | 8077312/9912422 [00:00<00:08, 216199.22it/s]9920512it [00:00, 37779483.56it/s]                           
0it [00:00, ?it/s]32768it [00:00, 535442.37it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475841.73it/s]1654784it [00:00, 11488660.01it/s]                         
0it [00:00, ?it/s]8192it [00:00, 174235.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5fb100c3c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5fb0ff5668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5fc9ee20d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5fc9ee20d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5fc9ee20d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:01<03:46,  1.41s/ MiB][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:01<03:46,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:01<03:45,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:01<03:43,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:01<03:42,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:01<03:41,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:01<03:39,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<03:38,  1.41s/ MiB][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<02:32,  1.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<02:32,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<02:31,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<02:30,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<02:29,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<02:28,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<02:27,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<02:26,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<02:25,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<02:24,  1.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:01<01:41,  1.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<01:41,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<01:40,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<01:39,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<01:38,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<01:38,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<01:37,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<01:36,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<01:36,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<01:35,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<01:34,  1.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<01:06,  2.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<01:06,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<01:05,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<01:05,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<01:04,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<01:04,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<01:03,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<01:03,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<01:02,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<01:02,  2.04 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:43,  2.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:43,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:43,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:43,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:42,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:42,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:42,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:41,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:41,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:40,  2.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:28,  4.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:28,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:28,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:28,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:28,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:27,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:27,  4.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:27,  4.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:27,  4.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:26,  4.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:02<00:19,  5.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:19,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:18,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:18,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:18,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:18,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:18,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:17,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:17,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:17,  5.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:02<00:12,  7.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:12,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:12,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:12,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:12,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:12,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:11,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:11,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:11,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:11,  7.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:02<00:08, 10.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:08, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:08, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:08, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:08, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:07, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:07, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:07, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:07, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:07, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:07, 10.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:02<00:05, 14.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:05, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:04, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:04, 14.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:03, 19.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:03, 19.69 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:02, 25.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:02, 25.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 32.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:01, 32.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:01, 40.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:01, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:01, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 40.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 48.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 48.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 56.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 56.21 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 63.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 63.57 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 69.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.24s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 69.87 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 69.87 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.24s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 69.87 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.55 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.30s/ url]
0 examples [00:00, ? examples/s]2020-06-26 18:08:33.221870: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-26 18:08:33.239216: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-26 18:08:33.239429: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a02a3b2fc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-26 18:08:33.239450: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.70 examples/s]114 examples [00:00, 13.80 examples/s]233 examples [00:00, 19.62 examples/s]345 examples [00:00, 27.82 examples/s]454 examples [00:00, 39.31 examples/s]561 examples [00:00, 55.28 examples/s]674 examples [00:00, 77.34 examples/s]787 examples [00:00, 107.32 examples/s]904 examples [00:00, 147.49 examples/s]1021 examples [00:01, 199.81 examples/s]1136 examples [00:01, 265.62 examples/s]1251 examples [00:01, 345.04 examples/s]1370 examples [00:01, 438.14 examples/s]1487 examples [00:01, 539.04 examples/s]1602 examples [00:01, 632.59 examples/s]1720 examples [00:01, 734.04 examples/s]1834 examples [00:01, 820.85 examples/s]1952 examples [00:01, 901.95 examples/s]2069 examples [00:01, 967.18 examples/s]2185 examples [00:02, 989.48 examples/s]2299 examples [00:02, 1030.22 examples/s]2414 examples [00:02, 1062.22 examples/s]2531 examples [00:02, 1090.14 examples/s]2651 examples [00:02, 1118.40 examples/s]2768 examples [00:02, 1131.13 examples/s]2884 examples [00:02, 1137.06 examples/s]3001 examples [00:02, 1145.12 examples/s]3119 examples [00:02, 1152.55 examples/s]3236 examples [00:02, 1141.60 examples/s]3351 examples [00:03, 1125.82 examples/s]3465 examples [00:03, 1121.75 examples/s]3578 examples [00:03, 1114.88 examples/s]3690 examples [00:03, 1102.92 examples/s]3808 examples [00:03, 1123.11 examples/s]3923 examples [00:03, 1130.58 examples/s]4039 examples [00:03, 1137.76 examples/s]4153 examples [00:03, 1135.10 examples/s]4274 examples [00:03, 1156.20 examples/s]4390 examples [00:03, 1137.10 examples/s]4505 examples [00:04, 1138.95 examples/s]4620 examples [00:04, 1135.74 examples/s]4738 examples [00:04, 1145.91 examples/s]4858 examples [00:04, 1159.81 examples/s]4977 examples [00:04, 1166.62 examples/s]5094 examples [00:04, 1147.41 examples/s]5212 examples [00:04, 1155.15 examples/s]5328 examples [00:04, 1142.45 examples/s]5445 examples [00:04, 1148.08 examples/s]5564 examples [00:04, 1159.87 examples/s]5681 examples [00:05, 1138.70 examples/s]5796 examples [00:05, 1132.10 examples/s]5914 examples [00:05, 1144.72 examples/s]6033 examples [00:05, 1156.37 examples/s]6149 examples [00:05, 1156.37 examples/s]6266 examples [00:05, 1158.92 examples/s]6382 examples [00:05, 1147.25 examples/s]6499 examples [00:05, 1153.18 examples/s]6618 examples [00:05, 1161.87 examples/s]6735 examples [00:05, 1158.49 examples/s]6852 examples [00:06, 1160.01 examples/s]6969 examples [00:06, 1154.61 examples/s]7085 examples [00:06, 1149.99 examples/s]7207 examples [00:06, 1168.89 examples/s]7328 examples [00:06, 1179.18 examples/s]7447 examples [00:06, 1181.76 examples/s]7566 examples [00:06, 1166.88 examples/s]7684 examples [00:06, 1168.37 examples/s]7801 examples [00:06, 1167.80 examples/s]7918 examples [00:07, 1164.60 examples/s]8035 examples [00:07, 1156.75 examples/s]8155 examples [00:07, 1168.44 examples/s]8272 examples [00:07, 1142.80 examples/s]8387 examples [00:07, 1137.20 examples/s]8506 examples [00:07, 1151.22 examples/s]8624 examples [00:07, 1157.52 examples/s]8740 examples [00:07, 1145.45 examples/s]8855 examples [00:07, 1146.77 examples/s]8970 examples [00:07, 1133.96 examples/s]9086 examples [00:08, 1139.99 examples/s]9201 examples [00:08, 1121.62 examples/s]9314 examples [00:08, 1101.06 examples/s]9427 examples [00:08, 1107.28 examples/s]9545 examples [00:08, 1127.35 examples/s]9660 examples [00:08, 1133.31 examples/s]9774 examples [00:08, 1133.94 examples/s]9889 examples [00:08, 1135.97 examples/s]10003 examples [00:08, 1071.26 examples/s]10120 examples [00:08, 1097.95 examples/s]10235 examples [00:09, 1112.32 examples/s]10354 examples [00:09, 1132.43 examples/s]10468 examples [00:09, 1130.31 examples/s]10583 examples [00:09, 1135.37 examples/s]10702 examples [00:09, 1148.74 examples/s]10820 examples [00:09, 1157.10 examples/s]10937 examples [00:09, 1160.59 examples/s]11054 examples [00:09, 1149.12 examples/s]11170 examples [00:09, 1138.50 examples/s]11284 examples [00:09, 1135.82 examples/s]11398 examples [00:10, 1130.84 examples/s]11513 examples [00:10, 1133.96 examples/s]11627 examples [00:10, 1134.38 examples/s]11741 examples [00:10, 1127.53 examples/s]11854 examples [00:10, 1126.99 examples/s]11967 examples [00:10, 1119.53 examples/s]12081 examples [00:10, 1125.14 examples/s]12194 examples [00:10, 1108.92 examples/s]12305 examples [00:10, 1105.73 examples/s]12420 examples [00:10, 1116.31 examples/s]12535 examples [00:11, 1123.54 examples/s]12650 examples [00:11, 1131.15 examples/s]12764 examples [00:11, 1085.35 examples/s]12873 examples [00:11, 1057.80 examples/s]12986 examples [00:11, 1076.63 examples/s]13101 examples [00:11, 1095.44 examples/s]13214 examples [00:11, 1104.06 examples/s]13326 examples [00:11, 1107.28 examples/s]13439 examples [00:11, 1113.56 examples/s]13556 examples [00:12, 1128.34 examples/s]13669 examples [00:12, 1125.10 examples/s]13783 examples [00:12, 1128.91 examples/s]13899 examples [00:12, 1137.87 examples/s]14017 examples [00:12, 1148.91 examples/s]14133 examples [00:12, 1149.92 examples/s]14252 examples [00:12, 1160.63 examples/s]14370 examples [00:12, 1163.46 examples/s]14487 examples [00:12, 1165.40 examples/s]14604 examples [00:12, 1166.32 examples/s]14722 examples [00:13, 1167.94 examples/s]14839 examples [00:13, 1167.84 examples/s]14956 examples [00:13, 1149.26 examples/s]15072 examples [00:13, 1150.68 examples/s]15194 examples [00:13, 1168.71 examples/s]15313 examples [00:13, 1173.79 examples/s]15431 examples [00:13, 1171.69 examples/s]15549 examples [00:13, 1156.17 examples/s]15665 examples [00:13, 1136.82 examples/s]15780 examples [00:13, 1137.99 examples/s]15894 examples [00:14, 1086.17 examples/s]16010 examples [00:14, 1106.00 examples/s]16122 examples [00:14, 1105.84 examples/s]16233 examples [00:14, 1087.83 examples/s]16352 examples [00:14, 1115.25 examples/s]16467 examples [00:14, 1124.77 examples/s]16586 examples [00:14, 1141.62 examples/s]16704 examples [00:14, 1150.40 examples/s]16820 examples [00:14, 1145.25 examples/s]16936 examples [00:14, 1147.91 examples/s]17053 examples [00:15, 1152.18 examples/s]17169 examples [00:15, 1148.00 examples/s]17288 examples [00:15, 1157.92 examples/s]17404 examples [00:15, 1157.37 examples/s]17520 examples [00:15, 1154.28 examples/s]17636 examples [00:15, 1155.02 examples/s]17752 examples [00:15, 1146.14 examples/s]17867 examples [00:15, 1140.85 examples/s]17983 examples [00:15, 1144.02 examples/s]18098 examples [00:15, 1137.51 examples/s]18214 examples [00:16, 1142.12 examples/s]18329 examples [00:16, 1127.80 examples/s]18445 examples [00:16, 1134.57 examples/s]18559 examples [00:16, 1128.60 examples/s]18676 examples [00:16, 1138.53 examples/s]18795 examples [00:16, 1151.87 examples/s]18911 examples [00:16, 1150.43 examples/s]19027 examples [00:16, 1144.29 examples/s]19142 examples [00:16, 1122.72 examples/s]19256 examples [00:17, 1126.77 examples/s]19371 examples [00:17, 1132.95 examples/s]19487 examples [00:17, 1140.93 examples/s]19604 examples [00:17, 1141.71 examples/s]19719 examples [00:17, 1110.37 examples/s]19835 examples [00:17, 1123.10 examples/s]19951 examples [00:17, 1131.53 examples/s]20065 examples [00:17, 1081.70 examples/s]20187 examples [00:17, 1119.33 examples/s]20300 examples [00:17, 1121.76 examples/s]20415 examples [00:18, 1128.44 examples/s]20532 examples [00:18, 1138.47 examples/s]20649 examples [00:18, 1145.77 examples/s]20764 examples [00:18, 1138.04 examples/s]20881 examples [00:18, 1145.39 examples/s]20996 examples [00:18, 1135.20 examples/s]21110 examples [00:18, 1116.22 examples/s]21222 examples [00:18, 1069.71 examples/s]21334 examples [00:18, 1082.88 examples/s]21445 examples [00:18, 1089.41 examples/s]21562 examples [00:19, 1111.14 examples/s]21678 examples [00:19, 1123.24 examples/s]21793 examples [00:19, 1130.77 examples/s]21912 examples [00:19, 1147.44 examples/s]22027 examples [00:19, 1148.05 examples/s]22145 examples [00:19, 1155.91 examples/s]22264 examples [00:19, 1164.24 examples/s]22383 examples [00:19, 1170.78 examples/s]22502 examples [00:19, 1174.60 examples/s]22620 examples [00:19, 1157.24 examples/s]22740 examples [00:20, 1168.84 examples/s]22857 examples [00:20, 1157.19 examples/s]22973 examples [00:20, 1154.20 examples/s]23089 examples [00:20, 1127.80 examples/s]23202 examples [00:20, 1124.33 examples/s]23320 examples [00:20, 1138.82 examples/s]23438 examples [00:20, 1149.19 examples/s]23554 examples [00:20, 1150.43 examples/s]23671 examples [00:20, 1154.21 examples/s]23787 examples [00:21, 1146.00 examples/s]23903 examples [00:21, 1147.99 examples/s]24018 examples [00:21, 1143.96 examples/s]24135 examples [00:21, 1149.78 examples/s]24252 examples [00:21, 1152.80 examples/s]24368 examples [00:21, 1154.70 examples/s]24484 examples [00:21, 1154.78 examples/s]24602 examples [00:21, 1161.48 examples/s]24719 examples [00:21, 1152.68 examples/s]24835 examples [00:21, 1142.19 examples/s]24950 examples [00:22, 1136.68 examples/s]25065 examples [00:22, 1138.39 examples/s]25179 examples [00:22, 1129.14 examples/s]25294 examples [00:22, 1135.00 examples/s]25409 examples [00:22, 1137.96 examples/s]25523 examples [00:22, 1136.32 examples/s]25641 examples [00:22, 1147.93 examples/s]25758 examples [00:22, 1153.48 examples/s]25876 examples [00:22, 1159.35 examples/s]25994 examples [00:22, 1164.52 examples/s]26111 examples [00:23, 1150.93 examples/s]26227 examples [00:23, 1140.99 examples/s]26342 examples [00:23, 1125.05 examples/s]26455 examples [00:23, 1110.82 examples/s]26567 examples [00:23, 1103.64 examples/s]26681 examples [00:23, 1113.10 examples/s]26796 examples [00:23, 1123.86 examples/s]26912 examples [00:23, 1131.69 examples/s]27027 examples [00:23, 1136.43 examples/s]27146 examples [00:23, 1149.55 examples/s]27262 examples [00:24, 1134.20 examples/s]27378 examples [00:24, 1140.10 examples/s]27493 examples [00:24, 1136.56 examples/s]27614 examples [00:24, 1157.62 examples/s]27731 examples [00:24, 1161.26 examples/s]27848 examples [00:24, 1152.44 examples/s]27964 examples [00:24, 1152.07 examples/s]28080 examples [00:24, 1147.44 examples/s]28198 examples [00:24, 1156.66 examples/s]28315 examples [00:24, 1159.38 examples/s]28431 examples [00:25, 1156.91 examples/s]28548 examples [00:25, 1158.24 examples/s]28664 examples [00:25, 1157.20 examples/s]28782 examples [00:25, 1162.83 examples/s]28901 examples [00:25, 1169.91 examples/s]29019 examples [00:25, 1160.82 examples/s]29136 examples [00:25, 1152.88 examples/s]29252 examples [00:25, 1145.79 examples/s]29367 examples [00:25, 1134.63 examples/s]29481 examples [00:25, 1135.90 examples/s]29595 examples [00:26, 1118.84 examples/s]29708 examples [00:26, 1118.03 examples/s]29820 examples [00:26, 1114.66 examples/s]29934 examples [00:26, 1122.09 examples/s]30047 examples [00:26, 1045.29 examples/s]30158 examples [00:26, 1061.16 examples/s]30274 examples [00:26, 1086.37 examples/s]30385 examples [00:26, 1089.03 examples/s]30495 examples [00:26, 1071.27 examples/s]30606 examples [00:27, 1082.50 examples/s]30715 examples [00:27, 1082.29 examples/s]30826 examples [00:27, 1088.01 examples/s]30943 examples [00:27, 1110.84 examples/s]31055 examples [00:27, 1095.61 examples/s]31172 examples [00:27, 1115.57 examples/s]31284 examples [00:27, 1115.09 examples/s]31398 examples [00:27, 1122.19 examples/s]31511 examples [00:27, 1119.27 examples/s]31624 examples [00:27, 1069.69 examples/s]31732 examples [00:28, 1028.87 examples/s]31836 examples [00:28, 990.43 examples/s] 31936 examples [00:28, 977.81 examples/s]32035 examples [00:28, 937.67 examples/s]32131 examples [00:28, 942.27 examples/s]32229 examples [00:28, 950.94 examples/s]32326 examples [00:28, 953.02 examples/s]32422 examples [00:28, 925.89 examples/s]32517 examples [00:28, 931.81 examples/s]32631 examples [00:29, 985.81 examples/s]32747 examples [00:29, 1030.11 examples/s]32859 examples [00:29, 1053.68 examples/s]32973 examples [00:29, 1076.17 examples/s]33088 examples [00:29, 1094.82 examples/s]33200 examples [00:29, 1100.13 examples/s]33311 examples [00:29, 1098.25 examples/s]33422 examples [00:29, 1095.58 examples/s]33532 examples [00:29, 1070.63 examples/s]33644 examples [00:29, 1084.18 examples/s]33760 examples [00:30, 1104.36 examples/s]33879 examples [00:30, 1127.92 examples/s]33995 examples [00:30, 1135.18 examples/s]34113 examples [00:30, 1147.84 examples/s]34228 examples [00:30, 1101.54 examples/s]34345 examples [00:30, 1119.70 examples/s]34461 examples [00:30, 1130.29 examples/s]34578 examples [00:30, 1139.71 examples/s]34693 examples [00:30, 1133.31 examples/s]34807 examples [00:30, 1130.53 examples/s]34921 examples [00:31, 1129.09 examples/s]35037 examples [00:31, 1136.28 examples/s]35151 examples [00:31, 1129.48 examples/s]35270 examples [00:31, 1145.23 examples/s]35387 examples [00:31, 1150.51 examples/s]35503 examples [00:31, 1151.16 examples/s]35619 examples [00:31, 1127.49 examples/s]35732 examples [00:31, 1123.62 examples/s]35848 examples [00:31, 1131.55 examples/s]35962 examples [00:31, 1131.85 examples/s]36076 examples [00:32, 1130.51 examples/s]36190 examples [00:32, 1115.51 examples/s]36302 examples [00:32, 1111.42 examples/s]36417 examples [00:32, 1119.77 examples/s]36534 examples [00:32, 1133.62 examples/s]36648 examples [00:32, 1135.34 examples/s]36762 examples [00:32, 1125.30 examples/s]36876 examples [00:32, 1128.49 examples/s]36989 examples [00:32, 1123.93 examples/s]37102 examples [00:32, 1100.34 examples/s]37213 examples [00:33, 1100.91 examples/s]37324 examples [00:33, 1092.49 examples/s]37434 examples [00:33, 1088.99 examples/s]37543 examples [00:33, 1078.10 examples/s]37651 examples [00:33, 1075.28 examples/s]37764 examples [00:33, 1089.01 examples/s]37877 examples [00:33, 1100.08 examples/s]37990 examples [00:33, 1108.61 examples/s]38109 examples [00:33, 1129.61 examples/s]38223 examples [00:33, 1125.54 examples/s]38336 examples [00:34, 1086.68 examples/s]38447 examples [00:34, 1091.15 examples/s]38560 examples [00:34, 1101.58 examples/s]38673 examples [00:34, 1109.55 examples/s]38788 examples [00:34, 1119.23 examples/s]38901 examples [00:34, 1093.18 examples/s]39011 examples [00:34, 1091.71 examples/s]39127 examples [00:34, 1110.18 examples/s]39242 examples [00:34, 1119.46 examples/s]39355 examples [00:35, 1118.06 examples/s]39469 examples [00:35, 1122.69 examples/s]39583 examples [00:35, 1127.40 examples/s]39696 examples [00:35, 1113.81 examples/s]39812 examples [00:35, 1124.54 examples/s]39928 examples [00:35, 1134.70 examples/s]40042 examples [00:35, 1058.23 examples/s]40152 examples [00:35, 1070.35 examples/s]40266 examples [00:35, 1084.34 examples/s]40377 examples [00:35, 1089.86 examples/s]40493 examples [00:36, 1107.85 examples/s]40609 examples [00:36, 1120.98 examples/s]40722 examples [00:36, 1076.96 examples/s]40833 examples [00:36, 1086.20 examples/s]40948 examples [00:36, 1103.26 examples/s]41061 examples [00:36, 1110.23 examples/s]41179 examples [00:36, 1129.53 examples/s]41293 examples [00:36, 1130.97 examples/s]41411 examples [00:36, 1144.67 examples/s]41527 examples [00:36, 1149.16 examples/s]41644 examples [00:37, 1153.20 examples/s]41760 examples [00:37, 1116.69 examples/s]41875 examples [00:37, 1125.23 examples/s]41994 examples [00:37, 1142.05 examples/s]42113 examples [00:37, 1155.29 examples/s]42232 examples [00:37, 1164.60 examples/s]42352 examples [00:37, 1173.60 examples/s]42470 examples [00:37, 1164.47 examples/s]42589 examples [00:37, 1169.83 examples/s]42707 examples [00:37, 1171.58 examples/s]42825 examples [00:38, 1170.88 examples/s]42943 examples [00:38, 1163.92 examples/s]43060 examples [00:38, 1148.55 examples/s]43175 examples [00:38, 1133.20 examples/s]43293 examples [00:38, 1145.08 examples/s]43412 examples [00:38, 1156.08 examples/s]43528 examples [00:38, 1141.91 examples/s]43643 examples [00:38, 1112.39 examples/s]43757 examples [00:38, 1119.25 examples/s]43870 examples [00:39, 1116.26 examples/s]43986 examples [00:39, 1128.03 examples/s]44101 examples [00:39, 1133.94 examples/s]44215 examples [00:39, 1131.97 examples/s]44329 examples [00:39, 1127.19 examples/s]44449 examples [00:39, 1145.92 examples/s]44566 examples [00:39, 1152.59 examples/s]44682 examples [00:39, 1114.54 examples/s]44795 examples [00:39, 1118.70 examples/s]44913 examples [00:39, 1135.30 examples/s]45028 examples [00:40, 1138.70 examples/s]45147 examples [00:40, 1151.76 examples/s]45263 examples [00:40, 1112.50 examples/s]45378 examples [00:40, 1121.07 examples/s]45494 examples [00:40, 1131.21 examples/s]45613 examples [00:40, 1146.88 examples/s]45733 examples [00:40, 1160.88 examples/s]45850 examples [00:40, 1161.33 examples/s]45967 examples [00:40, 1155.92 examples/s]46083 examples [00:40, 1146.99 examples/s]46198 examples [00:41, 1139.74 examples/s]46314 examples [00:41, 1145.06 examples/s]46429 examples [00:41, 1131.44 examples/s]46544 examples [00:41, 1134.33 examples/s]46658 examples [00:41, 1134.94 examples/s]46773 examples [00:41, 1138.94 examples/s]46894 examples [00:41, 1157.92 examples/s]47010 examples [00:41, 1154.92 examples/s]47126 examples [00:41, 1144.46 examples/s]47241 examples [00:41, 1139.68 examples/s]47356 examples [00:42, 1136.61 examples/s]47470 examples [00:42, 1135.86 examples/s]47584 examples [00:42, 1122.95 examples/s]47697 examples [00:42, 1124.38 examples/s]47810 examples [00:42, 1111.57 examples/s]47926 examples [00:42, 1124.52 examples/s]48044 examples [00:42, 1138.34 examples/s]48159 examples [00:42, 1141.42 examples/s]48275 examples [00:42, 1144.27 examples/s]48391 examples [00:42, 1146.36 examples/s]48506 examples [00:43, 1138.06 examples/s]48622 examples [00:43, 1144.48 examples/s]48737 examples [00:43, 1094.61 examples/s]48847 examples [00:43, 1092.74 examples/s]48960 examples [00:43, 1102.89 examples/s]49075 examples [00:43, 1114.43 examples/s]49189 examples [00:43, 1119.66 examples/s]49302 examples [00:43, 1117.34 examples/s]49417 examples [00:43, 1125.69 examples/s]49533 examples [00:44, 1135.39 examples/s]49649 examples [00:44, 1140.17 examples/s]49764 examples [00:44, 1120.02 examples/s]49878 examples [00:44, 1122.93 examples/s]49995 examples [00:44, 1135.05 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7426/50000 [00:00<00:00, 74255.64 examples/s] 42%|████▏     | 20818/50000 [00:00<00:00, 85711.27 examples/s] 68%|██████▊   | 33806/50000 [00:00<00:00, 95447.15 examples/s] 95%|█████████▍| 47414/50000 [00:00<00:00, 104836.72 examples/s]                                                                0 examples [00:00, ? examples/s]88 examples [00:00, 873.27 examples/s]204 examples [00:00, 942.76 examples/s]322 examples [00:00, 1000.65 examples/s]437 examples [00:00, 1039.75 examples/s]551 examples [00:00, 1066.99 examples/s]670 examples [00:00, 1100.67 examples/s]783 examples [00:00, 1108.57 examples/s]899 examples [00:00, 1122.06 examples/s]1014 examples [00:00, 1128.01 examples/s]1128 examples [00:01, 1129.67 examples/s]1239 examples [00:01, 1116.90 examples/s]1353 examples [00:01, 1121.77 examples/s]1466 examples [00:01, 1123.61 examples/s]1578 examples [00:01, 1087.42 examples/s]1687 examples [00:01, 1056.36 examples/s]1806 examples [00:01, 1091.28 examples/s]1926 examples [00:01, 1121.22 examples/s]2046 examples [00:01, 1141.35 examples/s]2161 examples [00:01, 1142.34 examples/s]2276 examples [00:02, 1128.62 examples/s]2390 examples [00:02, 1119.22 examples/s]2507 examples [00:02, 1132.92 examples/s]2621 examples [00:02, 1128.72 examples/s]2736 examples [00:02, 1134.62 examples/s]2850 examples [00:02, 1134.07 examples/s]2965 examples [00:02, 1135.82 examples/s]3079 examples [00:02, 1132.35 examples/s]3196 examples [00:02, 1141.96 examples/s]3314 examples [00:02, 1151.91 examples/s]3430 examples [00:03, 1135.69 examples/s]3544 examples [00:03, 1133.58 examples/s]3658 examples [00:03, 1134.56 examples/s]3772 examples [00:03, 1131.32 examples/s]3891 examples [00:03, 1147.43 examples/s]4006 examples [00:03, 1143.18 examples/s]4124 examples [00:03, 1150.91 examples/s]4240 examples [00:03, 1138.91 examples/s]4356 examples [00:03, 1142.42 examples/s]4473 examples [00:03, 1148.13 examples/s]4588 examples [00:04, 1144.70 examples/s]4703 examples [00:04, 1139.63 examples/s]4821 examples [00:04, 1151.23 examples/s]4938 examples [00:04, 1156.12 examples/s]5055 examples [00:04, 1159.17 examples/s]5171 examples [00:04, 1115.45 examples/s]5287 examples [00:04, 1125.63 examples/s]5405 examples [00:04, 1139.99 examples/s]5524 examples [00:04, 1151.99 examples/s]5642 examples [00:04, 1159.53 examples/s]5759 examples [00:05, 1135.80 examples/s]5873 examples [00:05, 1131.52 examples/s]5992 examples [00:05, 1146.36 examples/s]6107 examples [00:05, 1147.39 examples/s]6223 examples [00:05, 1148.96 examples/s]6338 examples [00:05, 1141.40 examples/s]6453 examples [00:05, 1141.80 examples/s]6568 examples [00:05, 1132.71 examples/s]6682 examples [00:05, 1117.42 examples/s]6794 examples [00:06, 1110.31 examples/s]6907 examples [00:06, 1116.13 examples/s]7019 examples [00:06, 1104.76 examples/s]7132 examples [00:06, 1110.94 examples/s]7245 examples [00:06, 1115.94 examples/s]7361 examples [00:06, 1126.56 examples/s]7474 examples [00:06, 1125.77 examples/s]7587 examples [00:06, 1126.96 examples/s]7700 examples [00:06, 1118.81 examples/s]7817 examples [00:06, 1132.76 examples/s]7931 examples [00:07, 1110.81 examples/s]8045 examples [00:07, 1116.78 examples/s]8157 examples [00:07, 1106.72 examples/s]8268 examples [00:07, 1101.07 examples/s]8382 examples [00:07, 1110.16 examples/s]8494 examples [00:07, 1101.95 examples/s]8605 examples [00:07, 1071.50 examples/s]8721 examples [00:07, 1095.85 examples/s]8836 examples [00:07, 1109.91 examples/s]8948 examples [00:07, 1087.99 examples/s]9058 examples [00:08, 1075.26 examples/s]9173 examples [00:08, 1094.25 examples/s]9289 examples [00:08, 1113.13 examples/s]9402 examples [00:08, 1116.44 examples/s]9516 examples [00:08, 1120.82 examples/s]9629 examples [00:08, 1121.24 examples/s]9742 examples [00:08, 1122.50 examples/s]9857 examples [00:08, 1129.22 examples/s]9970 examples [00:08, 1121.39 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteD2YXOT/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteD2YXOT/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5fc9ee2488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5fc9ee2488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5fc9ee2488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f60344c6a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5f640f9a90>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5fc9ee2488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5fc9ee2488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5fc9ee2488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5f640f99e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5fb0ff50f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f5f4c0391e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f5f4c0391e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f5f4c0391e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f5f4c0392f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f5f4c0392f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f5f4c0392f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=37c82c0a5d7e36c67bcc7888c522c558735b38dc35aadfee6de60169a91361e4
  Stored in directory: /tmp/pip-ephem-wheel-cache-udgjbyk0/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:23:39, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:12:23, 15.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:41:54, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:29:49, 31.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:14:04, 45.6kB/s].vector_cache/glove.6B.zip:   1%|          | 7.77M/862M [00:01<3:38:53, 65.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:32:29, 92.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.4M/862M [00:01<1:46:20, 133kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:14:05, 189kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.1M/862M [00:01<51:45, 270kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.2M/862M [00:01<36:05, 384kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.8M/862M [00:01<25:16, 546kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:02<17:39, 777kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.3M/862M [00:02<12:26, 1.10MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<08:44, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:02<06:11, 2.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:03<06:13, 2.17MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:03<04:28, 3.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<09:21, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<08:46, 1.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.1M/862M [00:05<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<07:00, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<06:41, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:06, 2.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<06:09, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<05:58, 2.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:09<04:35, 2.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<05:52, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<06:48, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<05:25, 2.43MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<05:55, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:13<05:18, 2.48MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:13<04:06, 3.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:13<03:01, 4.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<27:34, 474kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<20:36, 634kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<14:44, 885kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<13:32, 961kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<12:01, 1.08MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<08:59, 1.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<06:26, 2.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<11:12, 1.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<09:11, 1.41MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<06:41, 1.93MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:19<04:50, 2.65MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<51:18, 251kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:21<38:31, 334kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<27:30, 467kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:21<19:26, 660kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<17:44, 722kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<13:42, 933kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<09:54, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<09:54, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<08:46, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:25<06:32, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:45, 2.66MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<10:39, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:16, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<06:53, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<04:58, 2.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:20, 878kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<11:50, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<08:44, 1.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<06:15, 2.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<26:40, 470kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<20:32, 610kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<14:44, 849kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:28, 1.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<14:41, 848kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<12:03, 1.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:49, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<06:19, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<16:48, 737kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<13:32, 914kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<09:55, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:12, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<08:11, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<06:06, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<04:25, 2.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<15:15, 803kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<12:29, 980kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<09:10, 1.33MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:39, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:48, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<05:51, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:15, 2.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:52, 941kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:12, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<08:21, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:52, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:01, 1.50MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:13, 1.93MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:34, 2.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:31, 1.83MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:07, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:56, 2.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:31, 2.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:20, 3.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:09, 979kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:33, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:21, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<06:00, 1.97MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:02, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:00, 1.48MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:29, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:03, 2.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<03:42, 3.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<10:30, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:54, 1.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:11, 1.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:32, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:05, 2.87MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:42, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:41, 1.35MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:33, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:50, 2.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:19, 2.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:11, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:52, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:49, 3.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<02:50, 4.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<11:18, 1.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<09:36, 1.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<07:08, 1.62MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:06, 1.62MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:15, 1.58MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:31, 2.08MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<05:54, 1.93MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:31, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:10, 2.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:06, 2.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:47, 1.96MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:27, 2.54MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:07, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:11, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:02, 2.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:54, 2.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:06, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:58, 2.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:49, 2.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:58, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<03:49, 2.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<02:48, 3.96MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<27:21, 405kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<20:47, 533kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<14:55, 741kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<12:26, 885kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<11:50, 929kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:59, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<06:29, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:21, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:42, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:33, 1.96MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:30, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<04:14, 2.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:57, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:01, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<03:53, 2.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:42, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:49, 2.23MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<03:41, 2.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<04:34, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:48, 2.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<03:40, 2.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<02:43, 3.90MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<09:36, 1.10MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<08:04, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:56, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<04:16, 2.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<16:24, 642kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<14:30, 726kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<10:54, 964kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:48, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<09:18, 1.12MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:04, 1.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:01, 1.73MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:06, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:48, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:26, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:00, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:58, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:48, 2.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:33, 2.25MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<06:17, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:01, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:40, 2.78MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:33, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:21, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<04:04, 2.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<02:57, 3.43MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<32:07, 315kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<23:57, 422kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<17:06, 590kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<13:46, 729kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<11:09, 900kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<08:09, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:31, 1.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:08, 1.22MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<06:20, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:33, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:33, 1.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:45, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:16, 2.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:02, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:31, 1.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:09, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:46, 2.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:22, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<05:09, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:57, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:33, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:35, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:34, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:16, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:45, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:37, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:24, 2.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:51, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<04:11, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:40, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:40, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:34, 2.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<02:38, 3.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:45, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<07:17, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<07:01, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:17, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:19, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:54, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:34, 2.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:19, 2.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:25, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:13, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:59, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<02:55, 3.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:56, 1.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:53, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:15, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:32, 2.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:25, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:52, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:15, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:27, 2.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:00, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:55, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:04, 2.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:21, 3.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:23, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:06, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:00, 2.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:52, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:03, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:59, 2.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:25, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:19, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:57, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:38, 2.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:46, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:36, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:32, 2.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:04, 2.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<04:23, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<03:12, 2.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:26, 1.59MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:05, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:52, 2.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:16, 2.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:59, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:00, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:54, 2.94MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:36, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:20, 1.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:40, 1.81MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:00, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:46, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:35, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<02:36, 3.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<09:31, 881kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:52, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:45, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<04:06, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<16:09, 514kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<13:41, 608kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<10:04, 825kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<07:11, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<07:02, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:08, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<04:31, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:33<03:16, 2.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:16, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:25, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<05:42, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:10, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:41, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:15, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:12, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<03:54, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:01, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:06, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:58, 2.68MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:59, 1.60MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:37, 1.72MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:28, 2.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:32, 3.12MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<07:23, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<06:17, 1.25MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:38, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<06:42, 1.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:51, 1.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:22, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<04:28, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:23, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:19, 2.33MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:30, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:32, 1.69MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:32, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:25, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:30, 3.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:51, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<05:14, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:56, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:08, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:58, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:00, 2.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:29, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:37, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:48, 2.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:18, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:15, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:33, 2.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:35, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:44, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:57, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:08, 3.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<09:32, 763kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:43, 941kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:39, 1.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<05:16, 1.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<05:27, 1.32MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:14, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<07:15, 984kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:57, 1.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:20, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:28, 1.58MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:10, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:10, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:29, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:08, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:18, 2.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:24, 2.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<06:21, 1.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:17, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:52, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:07, 1.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<04:50, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:52, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:48, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:29, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:09, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:08, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:25, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:20, 1.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:26, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:34, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:11, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:01, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:18, 2.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:58, 2.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:40, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:54, 2.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:12, 2.96MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:57, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:44, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:06, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:47, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:27, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:45, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:02, 3.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:20, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:05, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:21, 2.70MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:56, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:58, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:17, 2.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:44, 2.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:41, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:00, 2.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:12, 2.81MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:43, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:30, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:40, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:59, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:49, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:03, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:14, 2.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:08, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:05, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:22, 2.54MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<02:45, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:46, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:06, 2.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<01:33, 3.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<06:06, 964kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<05:57, 989kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:31, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<03:15, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:56, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:35, 1.62MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:43, 2.13MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:57, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:44, 2.10MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:04, 2.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:43, 3.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<4:31:17, 21.0kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<3:10:00, 29.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<2:12:14, 42.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<1:34:18, 59.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<1:07:34, 83.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<47:38, 118kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<33:13, 168kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<25:14, 220kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<18:18, 303kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<12:53, 428kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:50<09:02, 607kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<25:41, 213kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<19:17, 284kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<13:48, 396kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<09:39, 562kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<10:34, 511kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<08:10, 661kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<05:53, 914kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:04, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:04, 1.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:52, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<02:50, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:06, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:56, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:14, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:31, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:22, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:47, 2.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:18, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:10, 1.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:35, 1.98MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:53, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:11, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:57, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:14, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:28, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:28, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:54, 2.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:13, 2.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:17, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:46, 2.77MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:07, 2.28MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:10, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:41, 2.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:03, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:08, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:40, 2.85MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:06, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:36, 2.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:11, 3.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:36, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:07, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:20, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:30, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:24, 1.90MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:49, 2.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:19, 3.43MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<07:29, 602kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:54, 764kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<04:15, 1.06MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<03:00, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<08:07, 547kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<06:20, 700kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:33, 968kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:58, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:57, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:01, 1.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<02:09, 2.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:01, 1.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:43, 1.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:02, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:12, 1.92MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:49, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:13, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:37, 2.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:11, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:07, 1.95MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:38, 2.54MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:53, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:55, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:29, 2.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:46, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:50, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:25, 2.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:43, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:46, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:22, 2.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:40, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:20, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:53, 2.06MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:22, 2.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:19, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:07, 1.79MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:35, 2.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:50, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:21, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:55, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:23, 2.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:20, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:09, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:38, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:48, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:47, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:21, 2.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:36, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:38, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:16, 2.78MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:31, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:33, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:11, 2.90MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:27, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:27, 2.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:05, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:25, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:57, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:36, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:10, 2.81MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:01, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:53, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:25, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:34, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:34, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:12, 2.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:24, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:52, 1.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:32, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:06, 2.76MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:52, 1.63MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:19, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:28, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:27, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:07, 2.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:18, 2.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:20, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:01, 2.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<00:44, 3.85MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<04:59, 572kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<04:19, 659kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<03:13, 880kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<02:21, 1.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:45, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<2:01:50, 22.9kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<1:25:12, 32.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<58:55, 46.5kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<41:51, 64.9kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<30:05, 90.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<21:10, 128kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<14:40, 182kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<11:08, 238kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<08:06, 326kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<05:46, 456kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<04:02, 645kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:43, 691kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<03:23, 760kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:31, 1.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:48, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:52, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:39, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:14, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:18, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:39, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:19, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:57, 2.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:32, 1.53MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:22, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:01, 2.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:45, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:22, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:17, 1.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:58, 2.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:05, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:23, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:06, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:48, 2.73MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:12, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:08, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:52, 2.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:59, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:17, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:01, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:44, 2.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:04, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:02, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:48, 2.50MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:54, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:55, 2.12MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:42, 2.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:50, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:51, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:39, 2.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:47, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:48, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:37, 2.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:44, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:46, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:36, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:43, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:44, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:34, 2.87MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<00:41, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:56, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:45, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:33, 2.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:45, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:34, 2.68MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:41, 2.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:54, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:44, 1.99MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:31, 2.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:53, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:49, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:36, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:39, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:50, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:40, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:28, 2.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:43, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:41, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:30, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:21, 3.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<05:00, 239kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<03:50, 311kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<02:45, 430kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:53, 609kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:41, 664kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<01:20, 833kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:57, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:50, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:53, 1.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:41, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:41, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:37, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:27, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:27, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:20, 2.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:14, 3.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<10:19, 82.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<07:27, 114kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<05:13, 161kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<03:30, 229kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<02:40, 291kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:57, 397kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:20, 559kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<01:02, 684kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:55, 764kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:41, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:28, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:32, 1.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:28, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:20, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:19, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:23, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:18, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:12, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:15, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:11, 2.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:07, 3.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:24, 1.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:20, 1.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:48, 449kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:40, 543kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:28, 740kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:19, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:16, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:13, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:06, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:13, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:10, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.62MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:06, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:05, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.94MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 447/400000 [00:00<01:29, 4469.54it/s]  0%|          | 1317/400000 [00:00<01:16, 5232.41it/s]  1%|          | 2183/400000 [00:00<01:07, 5936.63it/s]  1%|          | 3059/400000 [00:00<01:00, 6571.72it/s]  1%|          | 3923/400000 [00:00<00:55, 7078.67it/s]  1%|          | 4821/400000 [00:00<00:52, 7557.23it/s]  1%|▏         | 5714/400000 [00:00<00:49, 7922.17it/s]  2%|▏         | 6576/400000 [00:00<00:48, 8116.86it/s]  2%|▏         | 7464/400000 [00:00<00:47, 8331.66it/s]  2%|▏         | 8333/400000 [00:01<00:46, 8431.36it/s]  2%|▏         | 9229/400000 [00:01<00:45, 8582.86it/s]  3%|▎         | 10122/400000 [00:01<00:44, 8682.32it/s]  3%|▎         | 10993/400000 [00:01<00:44, 8670.32it/s]  3%|▎         | 11865/400000 [00:01<00:44, 8683.55it/s]  3%|▎         | 12753/400000 [00:01<00:44, 8739.45it/s]  3%|▎         | 13637/400000 [00:01<00:44, 8767.67it/s]  4%|▎         | 14522/400000 [00:01<00:43, 8789.91it/s]  4%|▍         | 15402/400000 [00:01<00:44, 8731.96it/s]  4%|▍         | 16276/400000 [00:01<00:43, 8726.32it/s]  4%|▍         | 17149/400000 [00:02<00:43, 8704.97it/s]  5%|▍         | 18024/400000 [00:02<00:43, 8717.05it/s]  5%|▍         | 18896/400000 [00:02<00:44, 8579.93it/s]  5%|▍         | 19755/400000 [00:02<00:44, 8567.30it/s]  5%|▌         | 20626/400000 [00:02<00:44, 8608.28it/s]  5%|▌         | 21500/400000 [00:02<00:43, 8647.28it/s]  6%|▌         | 22385/400000 [00:02<00:43, 8704.07it/s]  6%|▌         | 23256/400000 [00:02<00:43, 8663.74it/s]  6%|▌         | 24129/400000 [00:02<00:43, 8682.58it/s]  6%|▋         | 25017/400000 [00:02<00:42, 8740.73it/s]  6%|▋         | 25892/400000 [00:03<00:43, 8615.99it/s]  7%|▋         | 26802/400000 [00:03<00:42, 8753.39it/s]  7%|▋         | 27679/400000 [00:03<00:42, 8722.35it/s]  7%|▋         | 28556/400000 [00:03<00:42, 8734.01it/s]  7%|▋         | 29439/400000 [00:03<00:42, 8760.52it/s]  8%|▊         | 30326/400000 [00:03<00:42, 8792.20it/s]  8%|▊         | 31217/400000 [00:03<00:41, 8825.00it/s]  8%|▊         | 32100/400000 [00:03<00:41, 8774.48it/s]  8%|▊         | 32978/400000 [00:03<00:42, 8737.48it/s]  8%|▊         | 33852/400000 [00:03<00:42, 8625.82it/s]  9%|▊         | 34729/400000 [00:04<00:42, 8665.88it/s]  9%|▉         | 35616/400000 [00:04<00:41, 8724.72it/s]  9%|▉         | 36489/400000 [00:04<00:41, 8713.58it/s]  9%|▉         | 37383/400000 [00:04<00:41, 8778.36it/s] 10%|▉         | 38262/400000 [00:04<00:41, 8761.70it/s] 10%|▉         | 39139/400000 [00:04<00:41, 8762.46it/s] 10%|█         | 40016/400000 [00:04<00:41, 8674.66it/s] 10%|█         | 40884/400000 [00:04<00:41, 8666.51it/s] 10%|█         | 41765/400000 [00:04<00:41, 8707.86it/s] 11%|█         | 42637/400000 [00:04<00:41, 8709.08it/s] 11%|█         | 43514/400000 [00:05<00:40, 8725.68it/s] 11%|█         | 44387/400000 [00:05<00:41, 8596.30it/s] 11%|█▏        | 45248/400000 [00:05<00:41, 8515.55it/s] 12%|█▏        | 46109/400000 [00:05<00:41, 8541.56it/s] 12%|█▏        | 46995/400000 [00:05<00:40, 8632.66it/s] 12%|█▏        | 47867/400000 [00:05<00:40, 8657.38it/s] 12%|█▏        | 48748/400000 [00:05<00:40, 8699.92it/s] 12%|█▏        | 49619/400000 [00:05<00:41, 8545.81it/s] 13%|█▎        | 50487/400000 [00:05<00:40, 8584.71it/s] 13%|█▎        | 51347/400000 [00:05<00:40, 8588.85it/s] 13%|█▎        | 52246/400000 [00:06<00:39, 8703.55it/s] 13%|█▎        | 53132/400000 [00:06<00:39, 8748.86it/s] 14%|█▎        | 54008/400000 [00:06<00:39, 8721.54it/s] 14%|█▎        | 54887/400000 [00:06<00:39, 8739.51it/s] 14%|█▍        | 55774/400000 [00:06<00:39, 8776.11it/s] 14%|█▍        | 56652/400000 [00:06<00:39, 8742.50it/s] 14%|█▍        | 57560/400000 [00:06<00:38, 8839.19it/s] 15%|█▍        | 58445/400000 [00:06<00:38, 8790.52it/s] 15%|█▍        | 59331/400000 [00:06<00:38, 8809.27it/s] 15%|█▌        | 60235/400000 [00:06<00:38, 8876.22it/s] 15%|█▌        | 61123/400000 [00:07<00:38, 8866.89it/s] 16%|█▌        | 62020/400000 [00:07<00:37, 8896.94it/s] 16%|█▌        | 62910/400000 [00:07<00:37, 8887.92it/s] 16%|█▌        | 63817/400000 [00:07<00:37, 8939.12it/s] 16%|█▌        | 64718/400000 [00:07<00:37, 8959.30it/s] 16%|█▋        | 65615/400000 [00:07<00:37, 8926.96it/s] 17%|█▋        | 66508/400000 [00:07<00:37, 8918.99it/s] 17%|█▋        | 67400/400000 [00:07<00:37, 8897.89it/s] 17%|█▋        | 68290/400000 [00:07<00:37, 8868.60it/s] 17%|█▋        | 69178/400000 [00:07<00:37, 8869.87it/s] 18%|█▊        | 70070/400000 [00:08<00:37, 8884.24it/s] 18%|█▊        | 70959/400000 [00:08<00:37, 8777.53it/s] 18%|█▊        | 71838/400000 [00:08<00:37, 8719.26it/s] 18%|█▊        | 72735/400000 [00:08<00:37, 8792.16it/s] 18%|█▊        | 73615/400000 [00:08<00:37, 8783.14it/s] 19%|█▊        | 74494/400000 [00:08<00:37, 8733.98it/s] 19%|█▉        | 75389/400000 [00:08<00:36, 8794.87it/s] 19%|█▉        | 76269/400000 [00:08<00:37, 8695.92it/s] 19%|█▉        | 77147/400000 [00:08<00:37, 8718.66it/s] 20%|█▉        | 78037/400000 [00:08<00:36, 8770.78it/s] 20%|█▉        | 78924/400000 [00:09<00:36, 8797.22it/s] 20%|█▉        | 79811/400000 [00:09<00:36, 8816.73it/s] 20%|██        | 80707/400000 [00:09<00:36, 8857.80it/s] 20%|██        | 81612/400000 [00:09<00:35, 8911.55it/s] 21%|██        | 82504/400000 [00:09<00:35, 8913.24it/s] 21%|██        | 83396/400000 [00:09<00:35, 8872.90it/s] 21%|██        | 84284/400000 [00:09<00:35, 8792.73it/s] 21%|██▏       | 85164/400000 [00:09<00:35, 8766.01it/s] 22%|██▏       | 86049/400000 [00:09<00:35, 8789.23it/s] 22%|██▏       | 86929/400000 [00:09<00:35, 8791.18it/s] 22%|██▏       | 87810/400000 [00:10<00:35, 8796.38it/s] 22%|██▏       | 88690/400000 [00:10<00:35, 8693.61it/s] 22%|██▏       | 89581/400000 [00:10<00:35, 8757.31it/s] 23%|██▎       | 90480/400000 [00:10<00:35, 8823.30it/s] 23%|██▎       | 91364/400000 [00:10<00:34, 8827.72it/s] 23%|██▎       | 92248/400000 [00:10<00:35, 8762.52it/s] 23%|██▎       | 93125/400000 [00:10<00:35, 8578.01it/s] 24%|██▎       | 94006/400000 [00:10<00:35, 8646.04it/s] 24%|██▎       | 94906/400000 [00:10<00:34, 8748.40it/s] 24%|██▍       | 95782/400000 [00:10<00:34, 8721.67it/s] 24%|██▍       | 96655/400000 [00:11<00:34, 8690.10it/s] 24%|██▍       | 97537/400000 [00:11<00:34, 8727.30it/s] 25%|██▍       | 98411/400000 [00:11<00:34, 8706.24it/s] 25%|██▍       | 99297/400000 [00:11<00:34, 8749.38it/s] 25%|██▌       | 100173/400000 [00:11<00:34, 8727.81it/s] 25%|██▌       | 101046/400000 [00:11<00:34, 8721.49it/s] 25%|██▌       | 101919/400000 [00:11<00:34, 8651.55it/s] 26%|██▌       | 102795/400000 [00:11<00:34, 8681.61it/s] 26%|██▌       | 103695/400000 [00:11<00:33, 8773.40it/s] 26%|██▌       | 104585/400000 [00:11<00:33, 8809.57it/s] 26%|██▋       | 105467/400000 [00:12<00:33, 8770.98it/s] 27%|██▋       | 106370/400000 [00:12<00:33, 8844.71it/s] 27%|██▋       | 107258/400000 [00:12<00:33, 8854.46it/s] 27%|██▋       | 108144/400000 [00:12<00:33, 8813.88it/s] 27%|██▋       | 109031/400000 [00:12<00:32, 8829.26it/s] 27%|██▋       | 109915/400000 [00:12<00:33, 8779.21it/s] 28%|██▊       | 110794/400000 [00:12<00:33, 8755.43it/s] 28%|██▊       | 111673/400000 [00:12<00:32, 8763.33it/s] 28%|██▊       | 112588/400000 [00:12<00:32, 8875.02it/s] 28%|██▊       | 113485/400000 [00:13<00:32, 8900.60it/s] 29%|██▊       | 114376/400000 [00:13<00:32, 8815.12it/s] 29%|██▉       | 115270/400000 [00:13<00:32, 8849.62it/s] 29%|██▉       | 116168/400000 [00:13<00:31, 8886.27it/s] 29%|██▉       | 117057/400000 [00:13<00:32, 8840.44it/s] 29%|██▉       | 117945/400000 [00:13<00:31, 8851.53it/s] 30%|██▉       | 118831/400000 [00:13<00:32, 8746.63it/s] 30%|██▉       | 119707/400000 [00:13<00:32, 8747.30it/s] 30%|███       | 120586/400000 [00:13<00:31, 8759.55it/s] 30%|███       | 121471/400000 [00:13<00:31, 8784.26it/s] 31%|███       | 122350/400000 [00:14<00:31, 8784.11it/s] 31%|███       | 123229/400000 [00:14<00:31, 8757.58it/s] 31%|███       | 124110/400000 [00:14<00:31, 8771.82it/s] 31%|███       | 124988/400000 [00:14<00:31, 8772.01it/s] 31%|███▏      | 125866/400000 [00:14<00:31, 8746.01it/s] 32%|███▏      | 126741/400000 [00:14<00:31, 8703.63it/s] 32%|███▏      | 127612/400000 [00:14<00:31, 8665.14it/s] 32%|███▏      | 128490/400000 [00:14<00:31, 8697.48it/s] 32%|███▏      | 129366/400000 [00:14<00:31, 8715.02it/s] 33%|███▎      | 130265/400000 [00:14<00:30, 8794.00it/s] 33%|███▎      | 131167/400000 [00:15<00:30, 8860.56it/s] 33%|███▎      | 132054/400000 [00:15<00:30, 8822.36it/s] 33%|███▎      | 132937/400000 [00:15<00:30, 8816.19it/s] 33%|███▎      | 133837/400000 [00:15<00:30, 8870.23it/s] 34%|███▎      | 134725/400000 [00:15<00:29, 8869.21it/s] 34%|███▍      | 135620/400000 [00:15<00:29, 8892.18it/s] 34%|███▍      | 136510/400000 [00:15<00:29, 8857.25it/s] 34%|███▍      | 137396/400000 [00:15<00:29, 8813.64it/s] 35%|███▍      | 138285/400000 [00:15<00:29, 8835.92it/s] 35%|███▍      | 139185/400000 [00:15<00:29, 8883.78it/s] 35%|███▌      | 140085/400000 [00:16<00:29, 8917.79it/s] 35%|███▌      | 140977/400000 [00:16<00:29, 8881.11it/s] 35%|███▌      | 141866/400000 [00:16<00:29, 8861.37it/s] 36%|███▌      | 142753/400000 [00:16<00:29, 8850.63it/s] 36%|███▌      | 143639/400000 [00:16<00:29, 8809.84it/s] 36%|███▌      | 144521/400000 [00:16<00:29, 8752.97it/s] 36%|███▋      | 145413/400000 [00:16<00:28, 8800.53it/s] 37%|███▋      | 146294/400000 [00:16<00:28, 8798.39it/s] 37%|███▋      | 147190/400000 [00:16<00:28, 8844.40it/s] 37%|███▋      | 148076/400000 [00:16<00:28, 8846.84it/s] 37%|███▋      | 148961/400000 [00:17<00:28, 8727.22it/s] 37%|███▋      | 149849/400000 [00:17<00:28, 8771.14it/s] 38%|███▊      | 150742/400000 [00:17<00:28, 8815.59it/s] 38%|███▊      | 151635/400000 [00:17<00:28, 8849.09it/s] 38%|███▊      | 152533/400000 [00:17<00:27, 8887.56it/s] 38%|███▊      | 153422/400000 [00:17<00:28, 8784.40it/s] 39%|███▊      | 154330/400000 [00:17<00:27, 8869.67it/s] 39%|███▉      | 155218/400000 [00:17<00:27, 8871.17it/s] 39%|███▉      | 156106/400000 [00:17<00:27, 8794.40it/s] 39%|███▉      | 156991/400000 [00:17<00:27, 8808.47it/s] 39%|███▉      | 157893/400000 [00:18<00:27, 8870.43it/s] 40%|███▉      | 158790/400000 [00:18<00:27, 8898.61it/s] 40%|███▉      | 159684/400000 [00:18<00:27, 8890.82it/s] 40%|████      | 160587/400000 [00:18<00:26, 8930.19it/s] 40%|████      | 161487/400000 [00:18<00:26, 8948.70it/s] 41%|████      | 162382/400000 [00:18<00:26, 8922.16it/s] 41%|████      | 163285/400000 [00:18<00:26, 8952.95it/s] 41%|████      | 164181/400000 [00:18<00:26, 8950.70it/s] 41%|████▏     | 165089/400000 [00:18<00:26, 8987.81it/s] 41%|████▏     | 165988/400000 [00:18<00:26, 8948.75it/s] 42%|████▏     | 166883/400000 [00:19<00:27, 8449.22it/s] 42%|████▏     | 167734/400000 [00:19<00:27, 8395.76it/s] 42%|████▏     | 168578/400000 [00:19<00:28, 8099.21it/s] 42%|████▏     | 169427/400000 [00:19<00:28, 8210.37it/s] 43%|████▎     | 170325/400000 [00:19<00:27, 8426.71it/s] 43%|████▎     | 171220/400000 [00:19<00:26, 8576.64it/s] 43%|████▎     | 172108/400000 [00:19<00:26, 8663.30it/s] 43%|████▎     | 172986/400000 [00:19<00:26, 8695.24it/s] 43%|████▎     | 173884/400000 [00:19<00:25, 8778.52it/s] 44%|████▎     | 174781/400000 [00:19<00:25, 8833.14it/s] 44%|████▍     | 175680/400000 [00:20<00:25, 8877.11it/s] 44%|████▍     | 176573/400000 [00:20<00:25, 8891.27it/s] 44%|████▍     | 177463/400000 [00:20<00:25, 8818.56it/s] 45%|████▍     | 178346/400000 [00:20<00:25, 8760.80it/s] 45%|████▍     | 179223/400000 [00:20<00:25, 8759.49it/s] 45%|████▌     | 180112/400000 [00:20<00:24, 8797.52it/s] 45%|████▌     | 181000/400000 [00:20<00:24, 8819.24it/s] 45%|████▌     | 181883/400000 [00:20<00:24, 8780.54it/s] 46%|████▌     | 182789/400000 [00:20<00:24, 8860.76it/s] 46%|████▌     | 183676/400000 [00:20<00:24, 8849.34it/s] 46%|████▌     | 184590/400000 [00:21<00:24, 8934.23it/s] 46%|████▋     | 185487/400000 [00:21<00:23, 8943.40it/s] 47%|████▋     | 186382/400000 [00:21<00:24, 8873.42it/s] 47%|████▋     | 187270/400000 [00:21<00:23, 8870.45it/s] 47%|████▋     | 188158/400000 [00:21<00:23, 8857.38it/s] 47%|████▋     | 189056/400000 [00:21<00:23, 8891.63it/s] 47%|████▋     | 189959/400000 [00:21<00:23, 8931.89it/s] 48%|████▊     | 190853/400000 [00:21<00:23, 8913.37it/s] 48%|████▊     | 191753/400000 [00:21<00:23, 8937.12it/s] 48%|████▊     | 192647/400000 [00:21<00:23, 8929.15it/s] 48%|████▊     | 193542/400000 [00:22<00:23, 8934.56it/s] 49%|████▊     | 194436/400000 [00:22<00:23, 8923.68it/s] 49%|████▉     | 195329/400000 [00:22<00:22, 8918.53it/s] 49%|████▉     | 196221/400000 [00:22<00:23, 8764.96it/s] 49%|████▉     | 197105/400000 [00:22<00:23, 8787.24it/s] 49%|████▉     | 197999/400000 [00:22<00:22, 8830.20it/s] 50%|████▉     | 198919/400000 [00:22<00:22, 8936.30it/s] 50%|████▉     | 199819/400000 [00:22<00:22, 8954.49it/s] 50%|█████     | 200715/400000 [00:22<00:22, 8934.67it/s] 50%|█████     | 201629/400000 [00:23<00:22, 8993.48it/s] 51%|█████     | 202532/400000 [00:23<00:21, 9004.01it/s] 51%|█████     | 203433/400000 [00:23<00:21, 9001.82it/s] 51%|█████     | 204334/400000 [00:23<00:22, 8874.18it/s] 51%|█████▏    | 205222/400000 [00:23<00:22, 8812.51it/s] 52%|█████▏    | 206116/400000 [00:23<00:21, 8849.18it/s] 52%|█████▏    | 207007/400000 [00:23<00:21, 8864.75it/s] 52%|█████▏    | 207895/400000 [00:23<00:21, 8866.62it/s] 52%|█████▏    | 208782/400000 [00:23<00:21, 8785.63it/s] 52%|█████▏    | 209661/400000 [00:23<00:21, 8756.97it/s] 53%|█████▎    | 210554/400000 [00:24<00:21, 8805.55it/s] 53%|█████▎    | 211460/400000 [00:24<00:21, 8879.53it/s] 53%|█████▎    | 212356/400000 [00:24<00:21, 8903.45it/s] 53%|█████▎    | 213247/400000 [00:24<00:21, 8883.00it/s] 54%|█████▎    | 214136/400000 [00:24<00:20, 8882.59it/s] 54%|█████▍    | 215043/400000 [00:24<00:20, 8937.93it/s] 54%|█████▍    | 215943/400000 [00:24<00:20, 8954.53it/s] 54%|█████▍    | 216839/400000 [00:24<00:20, 8890.42it/s] 54%|█████▍    | 217729/400000 [00:24<00:20, 8770.80it/s] 55%|█████▍    | 218613/400000 [00:24<00:20, 8790.76it/s] 55%|█████▍    | 219505/400000 [00:25<00:20, 8826.69it/s] 55%|█████▌    | 220408/400000 [00:25<00:20, 8885.40it/s] 55%|█████▌    | 221297/400000 [00:25<00:20, 8846.53it/s] 56%|█████▌    | 222182/400000 [00:25<00:20, 8741.90it/s] 56%|█████▌    | 223073/400000 [00:25<00:20, 8788.90it/s] 56%|█████▌    | 223961/400000 [00:25<00:19, 8815.00it/s] 56%|█████▌    | 224853/400000 [00:25<00:19, 8844.78it/s] 56%|█████▋    | 225742/400000 [00:25<00:19, 8855.12it/s] 57%|█████▋    | 226628/400000 [00:25<00:19, 8821.93it/s] 57%|█████▋    | 227536/400000 [00:25<00:19, 8897.71it/s] 57%|█████▋    | 228443/400000 [00:26<00:19, 8947.38it/s] 57%|█████▋    | 229359/400000 [00:26<00:18, 9007.51it/s] 58%|█████▊    | 230261/400000 [00:26<00:18, 8964.42it/s] 58%|█████▊    | 231158/400000 [00:26<00:18, 8931.60it/s] 58%|█████▊    | 232052/400000 [00:26<00:19, 8806.02it/s] 58%|█████▊    | 232934/400000 [00:26<00:18, 8796.07it/s] 58%|█████▊    | 233814/400000 [00:26<00:18, 8788.75it/s] 59%|█████▊    | 234694/400000 [00:26<00:18, 8731.56it/s] 59%|█████▉    | 235570/400000 [00:26<00:18, 8737.72it/s] 59%|█████▉    | 236466/400000 [00:26<00:18, 8802.66it/s] 59%|█████▉    | 237356/400000 [00:27<00:18, 8830.77it/s] 60%|█████▉    | 238252/400000 [00:27<00:18, 8868.02it/s] 60%|█████▉    | 239139/400000 [00:27<00:18, 8771.92it/s] 60%|██████    | 240017/400000 [00:27<00:18, 8757.67it/s] 60%|██████    | 240894/400000 [00:27<00:18, 8721.14it/s] 60%|██████    | 241767/400000 [00:27<00:18, 8636.59it/s] 61%|██████    | 242631/400000 [00:27<00:18, 8622.78it/s] 61%|██████    | 243513/400000 [00:27<00:18, 8679.66it/s] 61%|██████    | 244382/400000 [00:27<00:18, 8602.75it/s] 61%|██████▏   | 245258/400000 [00:27<00:17, 8648.46it/s] 62%|██████▏   | 246158/400000 [00:28<00:17, 8749.90it/s] 62%|██████▏   | 247061/400000 [00:28<00:17, 8832.08it/s] 62%|██████▏   | 247945/400000 [00:28<00:17, 8830.32it/s] 62%|██████▏   | 248835/400000 [00:28<00:17, 8850.91it/s] 62%|██████▏   | 249721/400000 [00:28<00:17, 8750.12it/s] 63%|██████▎   | 250622/400000 [00:28<00:16, 8825.22it/s] 63%|██████▎   | 251506/400000 [00:28<00:16, 8826.90it/s] 63%|██████▎   | 252409/400000 [00:28<00:16, 8885.65it/s] 63%|██████▎   | 253304/400000 [00:28<00:16, 8904.10it/s] 64%|██████▎   | 254195/400000 [00:28<00:16, 8866.76it/s] 64%|██████▍   | 255082/400000 [00:29<00:16, 8844.85it/s] 64%|██████▍   | 255967/400000 [00:29<00:16, 8790.64it/s] 64%|██████▍   | 256859/400000 [00:29<00:16, 8826.53it/s] 64%|██████▍   | 257744/400000 [00:29<00:16, 8831.20it/s] 65%|██████▍   | 258628/400000 [00:29<00:16, 8779.10it/s] 65%|██████▍   | 259509/400000 [00:29<00:15, 8786.97it/s] 65%|██████▌   | 260388/400000 [00:29<00:15, 8781.74it/s] 65%|██████▌   | 261267/400000 [00:29<00:15, 8767.40it/s] 66%|██████▌   | 262164/400000 [00:29<00:15, 8825.99it/s] 66%|██████▌   | 263062/400000 [00:29<00:15, 8869.47it/s] 66%|██████▌   | 263972/400000 [00:30<00:15, 8937.34it/s] 66%|██████▌   | 264866/400000 [00:30<00:15, 8931.88it/s] 66%|██████▋   | 265772/400000 [00:30<00:14, 8968.71it/s] 67%|██████▋   | 266672/400000 [00:30<00:14, 8977.58it/s] 67%|██████▋   | 267570/400000 [00:30<00:14, 8858.47it/s] 67%|██████▋   | 268458/400000 [00:30<00:14, 8864.86it/s] 67%|██████▋   | 269349/400000 [00:30<00:14, 8876.31it/s] 68%|██████▊   | 270238/400000 [00:30<00:14, 8878.17it/s] 68%|██████▊   | 271126/400000 [00:30<00:14, 8707.73it/s] 68%|██████▊   | 272017/400000 [00:30<00:14, 8766.07it/s] 68%|██████▊   | 272922/400000 [00:31<00:14, 8847.64it/s] 68%|██████▊   | 273829/400000 [00:31<00:14, 8911.75it/s] 69%|██████▊   | 274731/400000 [00:31<00:14, 8941.41it/s] 69%|██████▉   | 275632/400000 [00:31<00:13, 8960.85it/s] 69%|██████▉   | 276529/400000 [00:31<00:13, 8828.84it/s] 69%|██████▉   | 277423/400000 [00:31<00:13, 8859.57it/s] 70%|██████▉   | 278310/400000 [00:31<00:13, 8771.88it/s] 70%|██████▉   | 279217/400000 [00:31<00:13, 8858.47it/s] 70%|███████   | 280119/400000 [00:31<00:13, 8904.52it/s] 70%|███████   | 281010/400000 [00:31<00:13, 8859.57it/s] 70%|███████   | 281897/400000 [00:32<00:13, 8767.33it/s] 71%|███████   | 282775/400000 [00:32<00:13, 8746.38it/s] 71%|███████   | 283664/400000 [00:32<00:13, 8788.54it/s] 71%|███████   | 284552/400000 [00:32<00:13, 8815.10it/s] 71%|███████▏  | 285434/400000 [00:32<00:13, 8703.91it/s] 72%|███████▏  | 286319/400000 [00:32<00:12, 8746.37it/s] 72%|███████▏  | 287222/400000 [00:32<00:12, 8828.21it/s] 72%|███████▏  | 288116/400000 [00:32<00:12, 8861.09it/s] 72%|███████▏  | 289026/400000 [00:32<00:12, 8930.31it/s] 72%|███████▏  | 289920/400000 [00:33<00:12, 8903.01it/s] 73%|███████▎  | 290829/400000 [00:33<00:12, 8957.61it/s] 73%|███████▎  | 291736/400000 [00:33<00:12, 8990.80it/s] 73%|███████▎  | 292636/400000 [00:33<00:11, 8989.97it/s] 73%|███████▎  | 293536/400000 [00:33<00:11, 8917.37it/s] 74%|███████▎  | 294428/400000 [00:33<00:11, 8819.94it/s] 74%|███████▍  | 295319/400000 [00:33<00:11, 8844.95it/s] 74%|███████▍  | 296204/400000 [00:33<00:11, 8811.82it/s] 74%|███████▍  | 297097/400000 [00:33<00:11, 8844.70it/s] 74%|███████▍  | 297996/400000 [00:33<00:11, 8887.20it/s] 75%|███████▍  | 298885/400000 [00:34<00:11, 8867.98it/s] 75%|███████▍  | 299790/400000 [00:34<00:11, 8921.40it/s] 75%|███████▌  | 300683/400000 [00:34<00:11, 8888.53it/s] 75%|███████▌  | 301573/400000 [00:34<00:11, 8845.43it/s] 76%|███████▌  | 302458/400000 [00:34<00:11, 8843.19it/s] 76%|███████▌  | 303343/400000 [00:34<00:11, 8732.51it/s] 76%|███████▌  | 304242/400000 [00:34<00:10, 8805.46it/s] 76%|███████▋  | 305129/400000 [00:34<00:10, 8822.69it/s] 77%|███████▋  | 306012/400000 [00:34<00:10, 8792.62it/s] 77%|███████▋  | 306918/400000 [00:34<00:10, 8869.60it/s] 77%|███████▋  | 307806/400000 [00:35<00:10, 8852.02it/s] 77%|███████▋  | 308692/400000 [00:35<00:10, 8740.82it/s] 77%|███████▋  | 309567/400000 [00:35<00:10, 8709.86it/s] 78%|███████▊  | 310439/400000 [00:35<00:10, 8678.45it/s] 78%|███████▊  | 311319/400000 [00:35<00:10, 8712.22it/s] 78%|███████▊  | 312196/400000 [00:35<00:10, 8728.91it/s] 78%|███████▊  | 313086/400000 [00:35<00:09, 8778.37it/s] 78%|███████▊  | 313966/400000 [00:35<00:09, 8784.00it/s] 79%|███████▊  | 314854/400000 [00:35<00:09, 8811.50it/s] 79%|███████▉  | 315736/400000 [00:35<00:09, 8782.86it/s] 79%|███████▉  | 316619/400000 [00:36<00:09, 8796.16it/s] 79%|███████▉  | 317499/400000 [00:36<00:09, 8722.02it/s] 80%|███████▉  | 318406/400000 [00:36<00:09, 8822.54it/s] 80%|███████▉  | 319304/400000 [00:36<00:09, 8867.49it/s] 80%|████████  | 320199/400000 [00:36<00:08, 8891.92it/s] 80%|████████  | 321089/400000 [00:36<00:08, 8867.13it/s] 80%|████████  | 321991/400000 [00:36<00:08, 8911.70it/s] 81%|████████  | 322884/400000 [00:36<00:08, 8916.10it/s] 81%|████████  | 323776/400000 [00:36<00:08, 8868.88it/s] 81%|████████  | 324664/400000 [00:36<00:08, 8764.90it/s] 81%|████████▏ | 325561/400000 [00:37<00:08, 8825.38it/s] 82%|████████▏ | 326478/400000 [00:37<00:08, 8924.21it/s] 82%|████████▏ | 327371/400000 [00:37<00:08, 8919.04it/s] 82%|████████▏ | 328264/400000 [00:37<00:08, 8886.41it/s] 82%|████████▏ | 329153/400000 [00:37<00:07, 8883.28it/s] 83%|████████▎ | 330042/400000 [00:37<00:07, 8883.57it/s] 83%|████████▎ | 330931/400000 [00:37<00:07, 8883.63it/s] 83%|████████▎ | 331820/400000 [00:37<00:07, 8763.41it/s] 83%|████████▎ | 332697/400000 [00:37<00:07, 8728.27it/s] 83%|████████▎ | 333587/400000 [00:37<00:07, 8777.43it/s] 84%|████████▎ | 334474/400000 [00:38<00:07, 8803.64it/s] 84%|████████▍ | 335355/400000 [00:38<00:07, 8784.35it/s] 84%|████████▍ | 336234/400000 [00:38<00:07, 8740.60it/s] 84%|████████▍ | 337128/400000 [00:38<00:07, 8797.59it/s] 85%|████████▍ | 338008/400000 [00:38<00:07, 8780.87it/s] 85%|████████▍ | 338887/400000 [00:38<00:06, 8774.64it/s] 85%|████████▍ | 339766/400000 [00:38<00:06, 8779.11it/s] 85%|████████▌ | 340661/400000 [00:38<00:06, 8828.73it/s] 85%|████████▌ | 341544/400000 [00:38<00:06, 8816.59it/s] 86%|████████▌ | 342445/400000 [00:38<00:06, 8871.49it/s] 86%|████████▌ | 343340/400000 [00:39<00:06, 8892.25it/s] 86%|████████▌ | 344230/400000 [00:39<00:06, 8881.86it/s] 86%|████████▋ | 345142/400000 [00:39<00:06, 8950.57it/s] 87%|████████▋ | 346038/400000 [00:39<00:06, 8873.04it/s] 87%|████████▋ | 346926/400000 [00:39<00:05, 8850.65it/s] 87%|████████▋ | 347825/400000 [00:39<00:05, 8889.85it/s] 87%|████████▋ | 348717/400000 [00:39<00:05, 8897.37it/s] 87%|████████▋ | 349607/400000 [00:39<00:05, 8747.01it/s] 88%|████████▊ | 350508/400000 [00:39<00:05, 8823.34it/s] 88%|████████▊ | 351424/400000 [00:39<00:05, 8919.14it/s] 88%|████████▊ | 352319/400000 [00:40<00:05, 8925.83it/s] 88%|████████▊ | 353214/400000 [00:40<00:05, 8930.22it/s] 89%|████████▊ | 354108/400000 [00:40<00:05, 8829.43it/s] 89%|████████▉ | 355008/400000 [00:40<00:05, 8877.99it/s] 89%|████████▉ | 355897/400000 [00:40<00:05, 8703.70it/s] 89%|████████▉ | 356785/400000 [00:40<00:04, 8755.54it/s] 89%|████████▉ | 357682/400000 [00:40<00:04, 8818.63it/s] 90%|████████▉ | 358565/400000 [00:40<00:04, 8774.17it/s] 90%|████████▉ | 359446/400000 [00:40<00:04, 8781.65it/s] 90%|█████████ | 360348/400000 [00:40<00:04, 8851.73it/s] 90%|█████████ | 361234/400000 [00:41<00:04, 8847.12it/s] 91%|█████████ | 362119/400000 [00:41<00:04, 8815.95it/s] 91%|█████████ | 363001/400000 [00:41<00:04, 8791.21it/s] 91%|█████████ | 363882/400000 [00:41<00:04, 8796.48it/s] 91%|█████████ | 364771/400000 [00:41<00:03, 8823.82it/s] 91%|█████████▏| 365654/400000 [00:41<00:03, 8783.42it/s] 92%|█████████▏| 366533/400000 [00:41<00:03, 8662.84it/s] 92%|█████████▏| 367400/400000 [00:41<00:03, 8650.06it/s] 92%|█████████▏| 368293/400000 [00:41<00:03, 8730.20it/s] 92%|█████████▏| 369187/400000 [00:41<00:03, 8790.92it/s] 93%|█████████▎| 370077/400000 [00:42<00:03, 8821.39it/s] 93%|█████████▎| 370960/400000 [00:42<00:03, 8808.38it/s] 93%|█████████▎| 371857/400000 [00:42<00:03, 8853.72it/s] 93%|█████████▎| 372752/400000 [00:42<00:03, 8880.14it/s] 93%|█████████▎| 373641/400000 [00:42<00:02, 8882.76it/s] 94%|█████████▎| 374530/400000 [00:42<00:02, 8868.27it/s] 94%|█████████▍| 375417/400000 [00:42<00:02, 8812.66it/s] 94%|█████████▍| 376299/400000 [00:42<00:02, 8809.42it/s] 94%|█████████▍| 377183/400000 [00:42<00:02, 8818.07it/s] 95%|█████████▍| 378065/400000 [00:42<00:02, 8719.60it/s] 95%|█████████▍| 378962/400000 [00:43<00:02, 8792.37it/s] 95%|█████████▍| 379842/400000 [00:43<00:02, 8698.61it/s] 95%|█████████▌| 380727/400000 [00:43<00:02, 8741.25it/s] 95%|█████████▌| 381603/400000 [00:43<00:02, 8743.63it/s] 96%|█████████▌| 382509/400000 [00:43<00:01, 8835.63it/s] 96%|█████████▌| 383408/400000 [00:43<00:01, 8880.09it/s] 96%|█████████▌| 384297/400000 [00:43<00:01, 8767.11it/s] 96%|█████████▋| 385175/400000 [00:43<00:01, 8732.84it/s] 97%|█████████▋| 386055/400000 [00:43<00:01, 8751.47it/s] 97%|█████████▋| 386931/400000 [00:44<00:01, 8752.96it/s] 97%|█████████▋| 387813/400000 [00:44<00:01, 8770.76it/s] 97%|█████████▋| 388693/400000 [00:44<00:01, 8777.71it/s] 97%|█████████▋| 389583/400000 [00:44<00:01, 8810.33it/s] 98%|█████████▊| 390465/400000 [00:44<00:01, 8763.46it/s] 98%|█████████▊| 391358/400000 [00:44<00:00, 8812.14it/s] 98%|█████████▊| 392240/400000 [00:44<00:00, 8768.26it/s] 98%|█████████▊| 393117/400000 [00:44<00:00, 8757.31it/s] 98%|█████████▊| 393993/400000 [00:44<00:00, 8746.25it/s] 99%|█████████▊| 394888/400000 [00:44<00:00, 8803.60it/s] 99%|█████████▉| 395781/400000 [00:45<00:00, 8841.00it/s] 99%|█████████▉| 396675/400000 [00:45<00:00, 8869.10it/s] 99%|█████████▉| 397563/400000 [00:45<00:00, 8857.25it/s]100%|█████████▉| 398461/400000 [00:45<00:00, 8892.32it/s]100%|█████████▉| 399354/400000 [00:45<00:00, 8902.57it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8794.70it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f5f4c04d9e8>, <torchtext.data.dataset.TabularDataset object at 0x7f5f4c04db38>, <torchtext.vocab.Vocab object at 0x7f5f4c04da58>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.33 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 23.18 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.71 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 24.71 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.11 file/s]2020-06-26 18:17:30.500002: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-26 18:17:30.504374: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-06-26 18:17:30.505119: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fcbd212750 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-26 18:17:30.505135: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3694592/9912422 [00:00<00:00, 36896692.79it/s]9920512it [00:00, 35696515.39it/s]                             
0it [00:00, ?it/s]32768it [00:00, 626214.04it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 446636.40it/s]1654784it [00:00, 11700284.30it/s]                         
0it [00:00, ?it/s]8192it [00:00, 184652.34it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
