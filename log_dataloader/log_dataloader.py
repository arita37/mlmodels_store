
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff32782eb70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff32782eb70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff39249e510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff39249e510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff3b16cbea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff3b16cbea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff33f7cea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff33f7cea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff33f7cea60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3579904/9912422 [00:00<00:00, 35795831.07it/s]9920512it [00:00, 33672524.14it/s]                             
0it [00:00, ?it/s]32768it [00:00, 602731.05it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 455117.98it/s]1654784it [00:00, 11306117.34it/s]                         
0it [00:00, ?it/s]8192it [00:00, 159461.92it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff3276ea6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff3276f7978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff33f7ce6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff33f7ce6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff33f7ce6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:05,  2.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   2%|▏         | 4/162 [00:00<00:49,  3.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:49,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:48,  3.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:37,  4.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:37,  4.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:01<00:36,  4.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:01<00:28,  5.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:01<00:28,  5.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:01<00:28,  5.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:01<00:27,  5.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   7%|▋         | 11/162 [00:01<00:21,  7.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:01<00:21,  7.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:01<00:21,  7.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:01<00:21,  7.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  9.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:01<00:16,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:01<00:16,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:01<00:16,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<00:15,  9.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:01<00:12, 11.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<00:12, 11.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:12, 11.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:12, 11.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:12, 11.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:01<00:09, 14.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:09, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:09, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:09, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:09, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:09, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:07, 18.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:07, 18.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:07, 18.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:07, 18.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:07, 18.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:07, 18.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:05, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:05, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:05, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:05, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:05, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:05, 22.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:01<00:04, 27.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:04, 27.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:04, 27.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:04, 27.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:04, 27.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:04, 27.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:04, 27.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:03, 31.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:03, 31.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:03, 31.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:02<00:03, 31.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:02<00:03, 31.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:02<00:03, 31.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:02<00:03, 31.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 36.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:02<00:03, 36.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:02<00:03, 36.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:02<00:03, 36.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:02<00:03, 36.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:02<00:03, 36.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:02<00:02, 39.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:02<00:02, 39.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:02<00:02, 39.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:02<00:02, 39.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:02<00:02, 39.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:02<00:02, 39.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:02<00:02, 39.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 42.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:02<00:02, 42.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:02<00:02, 42.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:02<00:02, 42.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:02<00:02, 42.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:02<00:02, 42.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:02<00:02, 42.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 45.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:02<00:02, 45.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:02, 45.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:02, 45.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:02, 45.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:01, 45.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:01, 45.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 47.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 48.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:01, 48.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 50.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:01, 50.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:02<00:01, 50.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:02<00:01, 50.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:02<00:01, 50.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:02<00:01, 50.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:02<00:01, 50.06 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 51.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:02<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:02<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:02<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:02<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:02<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 51.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 51.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:01, 51.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:01, 51.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:03<00:01, 51.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:03<00:01, 51.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:03<00:01, 51.18 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 51.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:03<00:01, 51.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:03<00:01, 51.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:03<00:01, 51.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:03<00:01, 51.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:03<00:01, 51.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:03<00:01, 51.72 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 52.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:03<00:01, 52.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:03<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:03<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:03<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:03<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:03<00:00, 52.69 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 53.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:03<00:00, 53.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:03<00:00, 53.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:03<00:00, 53.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:03<00:00, 53.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:03<00:00, 53.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:03<00:00, 53.78 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 52.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:03<00:00, 52.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:03<00:00, 52.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:03<00:00, 52.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:03<00:00, 52.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:03<00:00, 52.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:03<00:00, 52.93 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 52.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:03<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:03<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:03<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:03<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:03<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:03<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 52.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:03<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:03<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:03<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:03<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:03<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:03<00:00, 52.91 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 54.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:03<00:00, 54.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:03<00:00, 54.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:03<00:00, 54.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:03<00:00, 54.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:03<00:00, 54.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:03<00:00, 54.38 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 53.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:03<00:00, 53.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:03<00:00, 53.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:03<00:00, 53.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:03<00:00, 53.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:03<00:00, 53.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:03<00:00, 53.74 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 53.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:03<00:00, 53.44 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:04<00:00, 53.44 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:04<00:00, 53.44 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:04<00:00, 53.44 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:04<00:00, 53.44 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:04<00:00, 53.44 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 51.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.20s/ url]Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 51.36 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:04<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.38s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  4.20s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 51.36 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.38s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 25.37 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.39s/ url]
0 examples [00:00, ? examples/s]2020-07-30 12:09:36.799128: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-30 12:09:36.811550: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-30 12:09:36.812283: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d5f81d3020 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-30 12:09:36.812304: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
6 examples [00:00, 59.51 examples/s]111 examples [00:00, 82.99 examples/s]219 examples [00:00, 114.77 examples/s]328 examples [00:00, 156.87 examples/s]432 examples [00:00, 210.43 examples/s]530 examples [00:00, 275.24 examples/s]633 examples [00:00, 352.74 examples/s]732 examples [00:00, 436.84 examples/s]833 examples [00:00, 526.11 examples/s]938 examples [00:01, 618.03 examples/s]1043 examples [00:01, 704.52 examples/s]1147 examples [00:01, 778.97 examples/s]1253 examples [00:01, 844.98 examples/s]1356 examples [00:01, 890.92 examples/s]1459 examples [00:01, 919.96 examples/s]1561 examples [00:01, 945.31 examples/s]1663 examples [00:01, 965.93 examples/s]1765 examples [00:01, 980.46 examples/s]1871 examples [00:01, 1002.72 examples/s]1979 examples [00:02, 1022.81 examples/s]2084 examples [00:02, 1021.74 examples/s]2188 examples [00:02, 1017.74 examples/s]2292 examples [00:02, 1023.98 examples/s]2396 examples [00:02, 1008.11 examples/s]2498 examples [00:02, 1011.26 examples/s]2604 examples [00:02, 1025.37 examples/s]2707 examples [00:02, 1018.44 examples/s]2813 examples [00:02, 1028.21 examples/s]2919 examples [00:02, 1036.12 examples/s]3023 examples [00:03, 1036.48 examples/s]3127 examples [00:03, 1034.85 examples/s]3231 examples [00:03, 1025.43 examples/s]3334 examples [00:03, 1021.81 examples/s]3437 examples [00:03, 1014.48 examples/s]3539 examples [00:03, 1005.95 examples/s]3646 examples [00:03, 1022.77 examples/s]3749 examples [00:03, 1001.76 examples/s]3854 examples [00:03, 1013.97 examples/s]3956 examples [00:03, 995.04 examples/s] 4057 examples [00:04, 997.43 examples/s]4158 examples [00:04, 1001.00 examples/s]4259 examples [00:04, 998.32 examples/s] 4359 examples [00:04, 997.27 examples/s]4462 examples [00:04, 1003.51 examples/s]4567 examples [00:04, 1014.60 examples/s]4669 examples [00:04, 1013.56 examples/s]4771 examples [00:04, 1015.31 examples/s]4878 examples [00:04, 1029.63 examples/s]4982 examples [00:04, 938.54 examples/s] 5078 examples [00:05, 924.07 examples/s]5181 examples [00:05, 949.47 examples/s]5283 examples [00:05, 967.63 examples/s]5391 examples [00:05, 996.48 examples/s]5494 examples [00:05, 1004.75 examples/s]5596 examples [00:05, 1007.52 examples/s]5698 examples [00:05, 990.88 examples/s] 5798 examples [00:05, 988.51 examples/s]5904 examples [00:05, 1007.13 examples/s]6009 examples [00:06, 1018.68 examples/s]6112 examples [00:06, 998.21 examples/s] 6213 examples [00:06, 994.23 examples/s]6313 examples [00:06, 965.75 examples/s]6410 examples [00:06, 963.27 examples/s]6507 examples [00:06, 944.15 examples/s]6602 examples [00:06, 933.36 examples/s]6696 examples [00:06, 917.63 examples/s]6799 examples [00:06, 946.78 examples/s]6903 examples [00:06, 972.87 examples/s]7001 examples [00:07, 964.46 examples/s]7101 examples [00:07, 974.26 examples/s]7204 examples [00:07, 988.03 examples/s]7306 examples [00:07, 995.48 examples/s]7407 examples [00:07, 999.57 examples/s]7509 examples [00:07, 1005.08 examples/s]7617 examples [00:07, 1025.78 examples/s]7722 examples [00:07, 1030.23 examples/s]7826 examples [00:07, 1002.37 examples/s]7927 examples [00:07, 978.15 examples/s] 8031 examples [00:08, 994.02 examples/s]8137 examples [00:08, 1011.92 examples/s]8239 examples [00:08, 1009.63 examples/s]8342 examples [00:08, 1014.14 examples/s]8444 examples [00:08, 1001.00 examples/s]8545 examples [00:08, 989.48 examples/s] 8647 examples [00:08, 998.19 examples/s]8748 examples [00:08, 998.85 examples/s]8850 examples [00:08, 1003.10 examples/s]8956 examples [00:09, 1016.99 examples/s]9062 examples [00:09, 1028.55 examples/s]9167 examples [00:09, 1032.37 examples/s]9272 examples [00:09, 1036.70 examples/s]9376 examples [00:09, 1016.42 examples/s]9478 examples [00:09, 1002.27 examples/s]9579 examples [00:09, 996.58 examples/s] 9679 examples [00:09, 992.48 examples/s]9784 examples [00:09, 1007.25 examples/s]9887 examples [00:09, 1011.60 examples/s]9989 examples [00:10, 992.06 examples/s] 10089 examples [00:10, 932.12 examples/s]10188 examples [00:10, 948.36 examples/s]10290 examples [00:10, 967.11 examples/s]10390 examples [00:10, 974.85 examples/s]10495 examples [00:10, 995.88 examples/s]10597 examples [00:10, 1001.82 examples/s]10702 examples [00:10, 1014.76 examples/s]10808 examples [00:10, 1026.77 examples/s]10912 examples [00:10, 1027.28 examples/s]11016 examples [00:11, 1028.81 examples/s]11121 examples [00:11, 1034.98 examples/s]11225 examples [00:11, 1036.01 examples/s]11331 examples [00:11, 1041.16 examples/s]11436 examples [00:11, 1024.73 examples/s]11539 examples [00:11, 1021.50 examples/s]11646 examples [00:11, 1035.20 examples/s]11755 examples [00:11, 1048.74 examples/s]11862 examples [00:11, 1054.15 examples/s]11968 examples [00:11, 1053.76 examples/s]12074 examples [00:12, 1043.31 examples/s]12184 examples [00:12, 1058.54 examples/s]12296 examples [00:12, 1076.04 examples/s]12405 examples [00:12, 1078.51 examples/s]12513 examples [00:12, 1077.99 examples/s]12630 examples [00:12, 1102.26 examples/s]12745 examples [00:12, 1115.48 examples/s]12862 examples [00:12, 1128.76 examples/s]12976 examples [00:12, 1118.60 examples/s]13088 examples [00:12, 1101.84 examples/s]13205 examples [00:13, 1120.22 examples/s]13318 examples [00:13, 1111.71 examples/s]13437 examples [00:13, 1131.42 examples/s]13551 examples [00:13, 1122.70 examples/s]13667 examples [00:13, 1132.88 examples/s]13782 examples [00:13, 1137.94 examples/s]13896 examples [00:13, 1127.39 examples/s]14010 examples [00:13, 1128.91 examples/s]14130 examples [00:13, 1149.31 examples/s]14246 examples [00:13, 1148.98 examples/s]14366 examples [00:14, 1161.06 examples/s]14483 examples [00:14, 1158.07 examples/s]14602 examples [00:14, 1167.20 examples/s]14719 examples [00:14, 1144.37 examples/s]14834 examples [00:14, 1134.89 examples/s]14948 examples [00:14, 1125.45 examples/s]15061 examples [00:14, 1066.12 examples/s]15177 examples [00:14, 1092.40 examples/s]15290 examples [00:14, 1100.83 examples/s]15401 examples [00:15, 1098.81 examples/s]15515 examples [00:15, 1109.46 examples/s]15627 examples [00:15, 1109.38 examples/s]15741 examples [00:15, 1116.04 examples/s]15853 examples [00:15, 1107.20 examples/s]15964 examples [00:15, 1066.06 examples/s]16072 examples [00:15, 1017.44 examples/s]16184 examples [00:15, 1043.05 examples/s]16293 examples [00:15, 1056.23 examples/s]16404 examples [00:15, 1071.64 examples/s]16512 examples [00:16, 1032.78 examples/s]16623 examples [00:16, 1054.45 examples/s]16730 examples [00:16, 1058.37 examples/s]16837 examples [00:16, 1056.65 examples/s]16951 examples [00:16, 1078.76 examples/s]17064 examples [00:16, 1092.32 examples/s]17177 examples [00:16, 1101.01 examples/s]17288 examples [00:16, 1058.49 examples/s]17399 examples [00:16, 1072.43 examples/s]17508 examples [00:17, 1076.40 examples/s]17616 examples [00:17, 1023.38 examples/s]17722 examples [00:17, 1033.90 examples/s]17836 examples [00:17, 1061.67 examples/s]17950 examples [00:17, 1081.55 examples/s]18064 examples [00:17, 1097.07 examples/s]18175 examples [00:17, 1082.92 examples/s]18284 examples [00:17, 1042.15 examples/s]18389 examples [00:17, 997.56 examples/s] 18490 examples [00:17, 961.05 examples/s]18587 examples [00:18, 947.95 examples/s]18683 examples [00:18, 944.02 examples/s]18778 examples [00:18, 921.64 examples/s]18877 examples [00:18, 940.08 examples/s]18972 examples [00:18, 930.68 examples/s]19066 examples [00:18, 921.17 examples/s]19164 examples [00:18, 936.52 examples/s]19268 examples [00:18, 963.66 examples/s]19374 examples [00:18, 990.62 examples/s]19481 examples [00:19, 1010.37 examples/s]19585 examples [00:19, 1017.65 examples/s]19691 examples [00:19, 1027.58 examples/s]19796 examples [00:19, 1033.16 examples/s]19900 examples [00:19, 1026.96 examples/s]20003 examples [00:19, 944.46 examples/s] 20104 examples [00:19, 961.87 examples/s]20213 examples [00:19, 995.66 examples/s]20319 examples [00:19, 1012.14 examples/s]20424 examples [00:19, 1020.50 examples/s]20531 examples [00:20, 1033.28 examples/s]20635 examples [00:20, 1012.35 examples/s]20737 examples [00:20, 1011.22 examples/s]20841 examples [00:20, 1019.53 examples/s]20946 examples [00:20, 1026.42 examples/s]21049 examples [00:20, 1018.51 examples/s]21152 examples [00:20, 1018.60 examples/s]21254 examples [00:20, 1014.61 examples/s]21356 examples [00:20, 980.81 examples/s] 21456 examples [00:20, 985.45 examples/s]21555 examples [00:21, 973.69 examples/s]21653 examples [00:21, 973.62 examples/s]21755 examples [00:21, 986.24 examples/s]21859 examples [00:21, 1000.39 examples/s]21960 examples [00:21, 1000.28 examples/s]22061 examples [00:21, 939.38 examples/s] 22156 examples [00:21, 880.39 examples/s]22246 examples [00:21, 882.59 examples/s]22345 examples [00:21, 911.27 examples/s]22449 examples [00:22, 943.57 examples/s]22546 examples [00:22, 951.16 examples/s]22647 examples [00:22, 965.38 examples/s]22745 examples [00:22, 961.59 examples/s]22848 examples [00:22, 978.34 examples/s]22947 examples [00:22, 973.96 examples/s]23045 examples [00:22, 907.39 examples/s]23137 examples [00:22, 874.72 examples/s]23226 examples [00:22, 856.63 examples/s]23323 examples [00:22, 885.67 examples/s]23421 examples [00:23, 911.32 examples/s]23522 examples [00:23, 938.30 examples/s]23619 examples [00:23, 946.58 examples/s]23715 examples [00:23, 945.69 examples/s]23810 examples [00:23, 945.89 examples/s]23911 examples [00:23, 962.88 examples/s]24013 examples [00:23, 979.06 examples/s]24112 examples [00:23, 976.85 examples/s]24210 examples [00:23, 940.42 examples/s]24313 examples [00:23, 965.41 examples/s]24418 examples [00:24, 988.36 examples/s]24518 examples [00:24, 978.12 examples/s]24617 examples [00:24, 936.17 examples/s]24723 examples [00:24, 968.06 examples/s]24827 examples [00:24, 986.43 examples/s]24927 examples [00:24, 984.98 examples/s]25028 examples [00:24, 981.56 examples/s]25127 examples [00:24, 970.77 examples/s]25229 examples [00:24, 983.19 examples/s]25335 examples [00:25, 1003.43 examples/s]25436 examples [00:25, 953.39 examples/s] 25533 examples [00:25, 949.68 examples/s]25639 examples [00:25, 980.02 examples/s]25743 examples [00:25, 996.27 examples/s]25851 examples [00:25, 1019.79 examples/s]25954 examples [00:25, 1015.51 examples/s]26060 examples [00:25, 1027.19 examples/s]26169 examples [00:25, 1044.52 examples/s]26274 examples [00:25, 1028.22 examples/s]26378 examples [00:26, 1008.84 examples/s]26482 examples [00:26, 1015.28 examples/s]26590 examples [00:26, 1032.31 examples/s]26694 examples [00:26, 1029.93 examples/s]26798 examples [00:26, 1032.42 examples/s]26904 examples [00:26, 1037.90 examples/s]27009 examples [00:26, 1040.22 examples/s]27114 examples [00:26, 1025.58 examples/s]27218 examples [00:26, 1029.32 examples/s]27323 examples [00:26, 1034.36 examples/s]27429 examples [00:27, 1040.38 examples/s]27534 examples [00:27, 1021.72 examples/s]27642 examples [00:27, 1036.39 examples/s]27746 examples [00:27, 1010.73 examples/s]27848 examples [00:27, 985.79 examples/s] 27947 examples [00:27, 980.30 examples/s]28050 examples [00:27, 993.48 examples/s]28152 examples [00:27, 999.80 examples/s]28258 examples [00:27, 1014.89 examples/s]28361 examples [00:28, 1019.31 examples/s]28465 examples [00:28, 1024.04 examples/s]28568 examples [00:28, 1020.77 examples/s]28671 examples [00:28, 1022.67 examples/s]28774 examples [00:28, 1022.59 examples/s]28882 examples [00:28, 1038.83 examples/s]28990 examples [00:28, 1050.21 examples/s]29096 examples [00:28, 1049.58 examples/s]29204 examples [00:28, 1058.00 examples/s]29311 examples [00:28, 1060.07 examples/s]29419 examples [00:29, 1064.66 examples/s]29526 examples [00:29, 1059.26 examples/s]29632 examples [00:29, 1026.72 examples/s]29736 examples [00:29, 1030.38 examples/s]29843 examples [00:29, 1039.74 examples/s]29948 examples [00:29, 1039.08 examples/s]30053 examples [00:29, 995.15 examples/s] 30153 examples [00:29, 990.08 examples/s]30253 examples [00:29, 983.39 examples/s]30352 examples [00:29, 947.99 examples/s]30460 examples [00:30, 982.88 examples/s]30569 examples [00:30, 1010.96 examples/s]30675 examples [00:30, 1024.45 examples/s]30784 examples [00:30, 1041.14 examples/s]30892 examples [00:30, 1052.30 examples/s]30999 examples [00:30, 1056.82 examples/s]31105 examples [00:30, 1014.60 examples/s]31207 examples [00:30, 982.57 examples/s] 31312 examples [00:30, 999.49 examples/s]31414 examples [00:30, 1004.80 examples/s]31516 examples [00:31, 1006.94 examples/s]31624 examples [00:31, 1025.21 examples/s]31731 examples [00:31, 1035.93 examples/s]31838 examples [00:31, 1043.99 examples/s]31945 examples [00:31, 1050.78 examples/s]32051 examples [00:31, 1049.75 examples/s]32157 examples [00:31, 1049.60 examples/s]32264 examples [00:31, 1053.19 examples/s]32372 examples [00:31, 1058.57 examples/s]32481 examples [00:32, 1066.49 examples/s]32590 examples [00:32, 1071.54 examples/s]32698 examples [00:32, 1059.70 examples/s]32805 examples [00:32, 1039.41 examples/s]32910 examples [00:32, 1019.94 examples/s]33015 examples [00:32, 1026.73 examples/s]33118 examples [00:32, 1000.94 examples/s]33222 examples [00:32, 1011.21 examples/s]33330 examples [00:32, 1029.34 examples/s]33438 examples [00:32, 1041.80 examples/s]33543 examples [00:33, 1036.38 examples/s]33649 examples [00:33, 1041.50 examples/s]33757 examples [00:33, 1050.30 examples/s]33863 examples [00:33, 1038.30 examples/s]33967 examples [00:33, 1020.43 examples/s]34070 examples [00:33, 992.05 examples/s] 34170 examples [00:33, 985.43 examples/s]34269 examples [00:33, 982.84 examples/s]34368 examples [00:33, 976.14 examples/s]34468 examples [00:33, 981.39 examples/s]34573 examples [00:34, 998.54 examples/s]34679 examples [00:34, 1014.06 examples/s]34785 examples [00:34, 1026.31 examples/s]34888 examples [00:34, 1016.17 examples/s]34997 examples [00:34, 1035.83 examples/s]35103 examples [00:34, 1041.80 examples/s]35210 examples [00:34, 1048.94 examples/s]35315 examples [00:34, 1044.54 examples/s]35421 examples [00:34, 1046.72 examples/s]35529 examples [00:34, 1053.47 examples/s]35635 examples [00:35, 1034.98 examples/s]35742 examples [00:35, 1043.24 examples/s]35848 examples [00:35, 1046.21 examples/s]35955 examples [00:35, 1050.13 examples/s]36061 examples [00:35, 1039.69 examples/s]36166 examples [00:35, 1013.98 examples/s]36271 examples [00:35, 1023.47 examples/s]36377 examples [00:35, 1032.70 examples/s]36484 examples [00:35, 1042.53 examples/s]36593 examples [00:36, 1056.18 examples/s]36699 examples [00:36, 1039.25 examples/s]36805 examples [00:36, 1045.01 examples/s]36910 examples [00:36, 1042.54 examples/s]37017 examples [00:36, 1049.28 examples/s]37122 examples [00:36, 1041.76 examples/s]37227 examples [00:36, 1042.77 examples/s]37332 examples [00:36, 992.14 examples/s] 37439 examples [00:36, 1013.83 examples/s]37541 examples [00:36, 1013.01 examples/s]37647 examples [00:37, 1026.00 examples/s]37751 examples [00:37, 1029.45 examples/s]37855 examples [00:37, 1009.88 examples/s]37959 examples [00:37, 1015.79 examples/s]38061 examples [00:37, 988.51 examples/s] 38165 examples [00:37, 1001.37 examples/s]38273 examples [00:37, 1023.66 examples/s]38379 examples [00:37, 1032.05 examples/s]38487 examples [00:37, 1043.59 examples/s]38592 examples [00:37, 1045.11 examples/s]38697 examples [00:38, 1042.92 examples/s]38802 examples [00:38, 1040.91 examples/s]38910 examples [00:38, 1050.19 examples/s]39016 examples [00:38, 1028.69 examples/s]39120 examples [00:38, 978.02 examples/s] 39222 examples [00:38, 989.82 examples/s]39327 examples [00:38, 1006.45 examples/s]39430 examples [00:38, 1012.00 examples/s]39534 examples [00:38, 1020.12 examples/s]39641 examples [00:38, 1033.38 examples/s]39750 examples [00:39, 1047.66 examples/s]39859 examples [00:39, 1058.73 examples/s]39966 examples [00:39, 1059.14 examples/s]40073 examples [00:39, 996.60 examples/s] 40179 examples [00:39, 1012.52 examples/s]40284 examples [00:39, 1021.76 examples/s]40389 examples [00:39, 1028.78 examples/s]40493 examples [00:39, 1013.91 examples/s]40595 examples [00:39, 1015.51 examples/s]40699 examples [00:40, 1020.92 examples/s]40802 examples [00:40, 1019.68 examples/s]40905 examples [00:40, 1000.85 examples/s]41008 examples [00:40, 1008.70 examples/s]41109 examples [00:40, 1000.73 examples/s]41215 examples [00:40, 1015.36 examples/s]41317 examples [00:40, 1001.93 examples/s]41419 examples [00:40, 1005.20 examples/s]41522 examples [00:40, 1011.68 examples/s]41629 examples [00:40, 1027.95 examples/s]41739 examples [00:41, 1047.68 examples/s]41849 examples [00:41, 1060.69 examples/s]41957 examples [00:41, 1063.91 examples/s]42064 examples [00:41, 1054.85 examples/s]42170 examples [00:41, 1051.18 examples/s]42278 examples [00:41, 1057.31 examples/s]42388 examples [00:41, 1068.13 examples/s]42497 examples [00:41, 1073.16 examples/s]42606 examples [00:41, 1076.44 examples/s]42714 examples [00:41, 1069.76 examples/s]42822 examples [00:42, 1050.25 examples/s]42932 examples [00:42, 1063.85 examples/s]43040 examples [00:42, 1065.57 examples/s]43147 examples [00:42, 1029.59 examples/s]43251 examples [00:42, 1002.33 examples/s]43354 examples [00:42, 1008.44 examples/s]43462 examples [00:42, 1028.72 examples/s]43571 examples [00:42, 1045.89 examples/s]43678 examples [00:42, 1050.59 examples/s]43784 examples [00:42, 1028.08 examples/s]43891 examples [00:43, 1037.89 examples/s]43995 examples [00:43, 1035.34 examples/s]44103 examples [00:43, 1046.96 examples/s]44208 examples [00:43, 1037.28 examples/s]44316 examples [00:43, 1049.37 examples/s]44423 examples [00:43, 1053.40 examples/s]44531 examples [00:43, 1059.57 examples/s]44638 examples [00:43, 1061.88 examples/s]44745 examples [00:43, 1038.81 examples/s]44852 examples [00:44, 1047.75 examples/s]44958 examples [00:44, 1051.19 examples/s]45068 examples [00:44, 1063.86 examples/s]45177 examples [00:44, 1069.10 examples/s]45285 examples [00:44, 1070.85 examples/s]45393 examples [00:44, 1070.62 examples/s]45501 examples [00:44, 1065.95 examples/s]45608 examples [00:44, 1066.82 examples/s]45715 examples [00:44, 1066.00 examples/s]45825 examples [00:44, 1073.33 examples/s]45933 examples [00:45, 1071.94 examples/s]46041 examples [00:45, 1074.01 examples/s]46149 examples [00:45, 1041.28 examples/s]46254 examples [00:45, 1043.65 examples/s]46364 examples [00:45, 1057.09 examples/s]46472 examples [00:45, 1062.23 examples/s]46579 examples [00:45, 1060.28 examples/s]46686 examples [00:45, 1062.28 examples/s]46793 examples [00:45, 1064.40 examples/s]46902 examples [00:45, 1070.09 examples/s]47010 examples [00:46, 1070.74 examples/s]47118 examples [00:46, 1073.50 examples/s]47226 examples [00:46, 1064.16 examples/s]47333 examples [00:46, 1055.58 examples/s]47439 examples [00:46, 1038.33 examples/s]47544 examples [00:46, 1041.78 examples/s]47651 examples [00:46, 1048.60 examples/s]47759 examples [00:46, 1055.65 examples/s]47868 examples [00:46, 1063.82 examples/s]47976 examples [00:46, 1066.87 examples/s]48083 examples [00:47, 1065.01 examples/s]48190 examples [00:47, 1048.08 examples/s]48297 examples [00:47, 1051.82 examples/s]48403 examples [00:47, 1053.00 examples/s]48509 examples [00:47, 1051.93 examples/s]48615 examples [00:47, 1051.35 examples/s]48723 examples [00:47, 1059.07 examples/s]48830 examples [00:47, 1059.31 examples/s]48936 examples [00:47, 1052.08 examples/s]49044 examples [00:47, 1057.71 examples/s]49150 examples [00:48, 1055.31 examples/s]49258 examples [00:48, 1059.92 examples/s]49367 examples [00:48, 1067.17 examples/s]49474 examples [00:48, 1038.28 examples/s]49580 examples [00:48, 1042.01 examples/s]49685 examples [00:48, 1042.28 examples/s]49790 examples [00:48, 1044.42 examples/s]49895 examples [00:48, 1028.21 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6434/50000 [00:00<00:00, 64337.15 examples/s] 38%|███▊      | 19213/50000 [00:00<00:00, 75597.46 examples/s] 64%|██████▍   | 31959/50000 [00:00<00:00, 86107.77 examples/s] 91%|█████████ | 45380/50000 [00:00<00:00, 96481.66 examples/s]                                                               0 examples [00:00, ? examples/s]91 examples [00:00, 901.54 examples/s]200 examples [00:00, 948.95 examples/s]308 examples [00:00, 984.08 examples/s]419 examples [00:00, 1017.16 examples/s]518 examples [00:00, 1008.68 examples/s]618 examples [00:00, 1004.93 examples/s]714 examples [00:00, 988.64 examples/s] 821 examples [00:00, 1011.66 examples/s]926 examples [00:00, 1019.96 examples/s]1025 examples [00:01, 1007.22 examples/s]1134 examples [00:01, 1029.82 examples/s]1241 examples [00:01, 1041.03 examples/s]1345 examples [00:01, 1029.68 examples/s]1454 examples [00:01, 1046.92 examples/s]1564 examples [00:01, 1062.01 examples/s]1672 examples [00:01, 1066.60 examples/s]1779 examples [00:01, 1066.74 examples/s]1889 examples [00:01, 1075.52 examples/s]1998 examples [00:01, 1078.26 examples/s]2107 examples [00:02, 1080.36 examples/s]2216 examples [00:02, 1080.27 examples/s]2325 examples [00:02, 1080.64 examples/s]2435 examples [00:02, 1085.40 examples/s]2544 examples [00:02, 1082.20 examples/s]2655 examples [00:02, 1088.12 examples/s]2764 examples [00:02, 1087.03 examples/s]2873 examples [00:02, 1038.64 examples/s]2978 examples [00:02, 1034.19 examples/s]3084 examples [00:02, 1039.36 examples/s]3193 examples [00:03, 1053.23 examples/s]3299 examples [00:03, 1042.18 examples/s]3409 examples [00:03, 1058.31 examples/s]3519 examples [00:03, 1069.34 examples/s]3627 examples [00:03, 1058.06 examples/s]3737 examples [00:03, 1068.32 examples/s]3844 examples [00:03, 1067.09 examples/s]3951 examples [00:03, 1064.29 examples/s]4063 examples [00:03, 1078.37 examples/s]4171 examples [00:03, 1075.68 examples/s]4279 examples [00:04, 1064.92 examples/s]4389 examples [00:04, 1074.59 examples/s]4498 examples [00:04, 1077.19 examples/s]4607 examples [00:04, 1080.81 examples/s]4716 examples [00:04, 1078.15 examples/s]4826 examples [00:04, 1082.74 examples/s]4937 examples [00:04, 1090.35 examples/s]5047 examples [00:04, 1090.74 examples/s]5159 examples [00:04, 1096.78 examples/s]5269 examples [00:04, 1089.64 examples/s]5378 examples [00:05, 1082.84 examples/s]5487 examples [00:05, 1074.82 examples/s]5595 examples [00:05, 1067.57 examples/s]5702 examples [00:05, 1055.72 examples/s]5811 examples [00:05, 1065.27 examples/s]5919 examples [00:05, 1068.34 examples/s]6026 examples [00:05, 1052.13 examples/s]6132 examples [00:05, 1053.00 examples/s]6238 examples [00:05, 1052.99 examples/s]6344 examples [00:05, 1045.26 examples/s]6449 examples [00:06, 1033.09 examples/s]6558 examples [00:06, 1047.16 examples/s]6663 examples [00:06, 1038.48 examples/s]6770 examples [00:06, 1046.75 examples/s]6881 examples [00:06, 1062.76 examples/s]6988 examples [00:06, 1057.41 examples/s]7097 examples [00:06, 1066.34 examples/s]7205 examples [00:06, 1068.44 examples/s]7316 examples [00:06, 1079.26 examples/s]7426 examples [00:06, 1084.47 examples/s]7536 examples [00:07, 1087.60 examples/s]7647 examples [00:07, 1091.47 examples/s]7757 examples [00:07, 1074.53 examples/s]7869 examples [00:07, 1084.91 examples/s]7980 examples [00:07, 1089.94 examples/s]8090 examples [00:07, 1083.44 examples/s]8200 examples [00:07, 1088.13 examples/s]8310 examples [00:07, 1090.45 examples/s]8421 examples [00:07, 1093.90 examples/s]8531 examples [00:08, 1095.14 examples/s]8641 examples [00:08, 1095.13 examples/s]8752 examples [00:08, 1097.11 examples/s]8862 examples [00:08, 1093.62 examples/s]8972 examples [00:08, 1091.66 examples/s]9083 examples [00:08, 1096.08 examples/s]9193 examples [00:08, 1087.55 examples/s]9303 examples [00:08, 1089.60 examples/s]9412 examples [00:08, 1056.47 examples/s]9518 examples [00:08, 988.33 examples/s] 9623 examples [00:09, 995.86 examples/s]9724 examples [00:09, 999.18 examples/s]9831 examples [00:09, 1019.15 examples/s]9939 examples [00:09, 1036.06 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTFX927/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTFX927/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff33f7cea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff33f7cea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff33f7cea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff38b1b20f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff2c5a1e828>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff33f7cea60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff33f7cea60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff33f7cea60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff3276f7358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff3276f71d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.94 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.94 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.94 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.03 file/s]2020-07-30 12:10:43.251528: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-30 12:10:43.255749: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-30 12:10:43.255913: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bf7da8ce40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-30 12:10:43.255927: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:21, 457530.99it/s] 79%|███████▊  | 7790592/9912422 [00:00<00:03, 651964.17it/s]9920512it [00:00, 44122133.96it/s]                           
0it [00:00, ?it/s]32768it [00:00, 533772.53it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 464357.39it/s]1654784it [00:00, 11459391.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 164982.44it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
