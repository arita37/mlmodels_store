
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fd94a5cc6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fd94a5cc6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fd9b5b940d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fd9b5b940d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fd9cf8c1ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fd9cf8c1ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd96340d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd96340d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd96340d950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3809280/9912422 [00:00<00:00, 38069328.34it/s]9920512it [00:00, 35280558.02it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 256846.75it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 7668609.95it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 66596.25it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd960951b00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd960955da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fd96340d598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fd96340d598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fd96340d598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:53,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:51,  3.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:36,  4.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:25,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  5.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:17,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:17,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.10 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 11.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 14.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 19.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 25.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 31.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 37.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 52.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 52.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 58.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:00, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 58.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 64.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 64.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 68.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 68.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 70.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 73.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.48 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.48 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.42s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.43s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.48 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.42s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.42s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 36.62 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.42s/ url]
0 examples [00:00, ? examples/s]2020-06-10 06:09:58.084537: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-10 06:09:58.100250: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-10 06:09:58.100551: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559f1d760240 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-10 06:09:58.100570: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
20 examples [00:00, 198.71 examples/s]109 examples [00:00, 259.04 examples/s]212 examples [00:00, 333.43 examples/s]312 examples [00:00, 416.77 examples/s]421 examples [00:00, 511.29 examples/s]531 examples [00:00, 608.75 examples/s]620 examples [00:00, 672.42 examples/s]722 examples [00:00, 748.90 examples/s]818 examples [00:00, 800.80 examples/s]912 examples [00:01, 834.14 examples/s]1012 examples [00:01, 876.34 examples/s]1109 examples [00:01, 900.84 examples/s]1212 examples [00:01, 935.57 examples/s]1317 examples [00:01, 961.64 examples/s]1417 examples [00:01, 971.63 examples/s]1523 examples [00:01, 994.06 examples/s]1625 examples [00:01, 976.42 examples/s]1724 examples [00:01, 958.68 examples/s]1829 examples [00:01, 983.72 examples/s]1929 examples [00:02, 984.31 examples/s]2032 examples [00:02, 997.20 examples/s]2133 examples [00:02, 991.44 examples/s]2237 examples [00:02, 1004.48 examples/s]2338 examples [00:02, 1000.97 examples/s]2439 examples [00:02, 984.56 examples/s] 2538 examples [00:02, 983.13 examples/s]2637 examples [00:02, 937.74 examples/s]2735 examples [00:02, 948.58 examples/s]2839 examples [00:02, 973.42 examples/s]2943 examples [00:03, 990.98 examples/s]3051 examples [00:03, 1013.52 examples/s]3153 examples [00:03, 1005.60 examples/s]3257 examples [00:03, 1013.21 examples/s]3359 examples [00:03, 973.08 examples/s] 3459 examples [00:03, 979.98 examples/s]3558 examples [00:03, 940.11 examples/s]3654 examples [00:03, 944.37 examples/s]3762 examples [00:03, 980.31 examples/s]3867 examples [00:03, 999.74 examples/s]3970 examples [00:04, 1008.60 examples/s]4072 examples [00:04, 1011.36 examples/s]4174 examples [00:04, 1004.70 examples/s]4276 examples [00:04, 1008.76 examples/s]4382 examples [00:04, 1022.70 examples/s]4485 examples [00:04, 1022.23 examples/s]4589 examples [00:04, 1027.16 examples/s]4692 examples [00:04, 997.60 examples/s] 4793 examples [00:04, 993.36 examples/s]4893 examples [00:05, 979.33 examples/s]4992 examples [00:05, 975.60 examples/s]5090 examples [00:05, 974.12 examples/s]5188 examples [00:05, 946.52 examples/s]5295 examples [00:05, 980.03 examples/s]5394 examples [00:05, 978.15 examples/s]5493 examples [00:05, 974.85 examples/s]5591 examples [00:05, 959.36 examples/s]5689 examples [00:05, 964.01 examples/s]5796 examples [00:05, 988.61 examples/s]5901 examples [00:06, 1004.41 examples/s]6002 examples [00:06, 1002.73 examples/s]6103 examples [00:06, 928.29 examples/s] 6200 examples [00:06, 938.10 examples/s]6295 examples [00:06, 940.30 examples/s]6398 examples [00:06, 964.39 examples/s]6500 examples [00:06, 980.02 examples/s]6599 examples [00:06, 954.18 examples/s]6698 examples [00:06, 962.43 examples/s]6804 examples [00:06, 988.67 examples/s]6906 examples [00:07, 997.60 examples/s]7007 examples [00:07, 989.76 examples/s]7107 examples [00:07, 973.80 examples/s]7209 examples [00:07, 986.36 examples/s]7308 examples [00:07, 968.83 examples/s]7406 examples [00:07, 938.89 examples/s]7501 examples [00:07, 914.79 examples/s]7593 examples [00:07, 896.91 examples/s]7684 examples [00:07, 896.52 examples/s]7775 examples [00:08, 898.51 examples/s]7879 examples [00:08, 936.72 examples/s]7974 examples [00:08, 934.36 examples/s]8068 examples [00:08, 932.35 examples/s]8170 examples [00:08, 954.98 examples/s]8266 examples [00:08, 956.03 examples/s]8368 examples [00:08, 972.23 examples/s]8469 examples [00:08, 980.99 examples/s]8568 examples [00:08, 965.69 examples/s]8665 examples [00:08, 944.59 examples/s]8770 examples [00:09, 972.25 examples/s]8878 examples [00:09, 1001.11 examples/s]8984 examples [00:09, 1015.48 examples/s]9086 examples [00:09, 1014.70 examples/s]9189 examples [00:09, 1018.63 examples/s]9292 examples [00:09, 1003.45 examples/s]9400 examples [00:09, 1023.72 examples/s]9510 examples [00:09, 1043.47 examples/s]9615 examples [00:09, 1037.59 examples/s]9725 examples [00:09, 1053.47 examples/s]9831 examples [00:10, 1023.91 examples/s]9934 examples [00:10, 999.21 examples/s] 10035 examples [00:10, 880.22 examples/s]10131 examples [00:10, 900.28 examples/s]10239 examples [00:10, 945.43 examples/s]10336 examples [00:10, 950.21 examples/s]10445 examples [00:10, 986.00 examples/s]10545 examples [00:10, 972.51 examples/s]10644 examples [00:10, 957.59 examples/s]10745 examples [00:11, 970.24 examples/s]10845 examples [00:11, 977.84 examples/s]10946 examples [00:11, 987.14 examples/s]11046 examples [00:11, 986.77 examples/s]11149 examples [00:11, 996.59 examples/s]11250 examples [00:11, 999.03 examples/s]11354 examples [00:11, 1009.69 examples/s]11456 examples [00:11, 1010.92 examples/s]11558 examples [00:11, 977.30 examples/s] 11657 examples [00:11, 972.15 examples/s]11755 examples [00:12, 964.08 examples/s]11863 examples [00:12, 995.50 examples/s]11968 examples [00:12, 1010.13 examples/s]12070 examples [00:12, 996.60 examples/s] 12172 examples [00:12, 1002.13 examples/s]12273 examples [00:12, 989.99 examples/s] 12373 examples [00:12, 979.73 examples/s]12475 examples [00:12, 991.15 examples/s]12577 examples [00:12, 997.78 examples/s]12684 examples [00:12, 1016.37 examples/s]12786 examples [00:13, 1006.33 examples/s]12887 examples [00:13, 985.63 examples/s] 12986 examples [00:13, 986.81 examples/s]13085 examples [00:13, 970.76 examples/s]13183 examples [00:13, 942.86 examples/s]13278 examples [00:13, 921.81 examples/s]13371 examples [00:13, 922.48 examples/s]13469 examples [00:13, 937.31 examples/s]13567 examples [00:13, 948.05 examples/s]13664 examples [00:14, 952.20 examples/s]13760 examples [00:14, 951.69 examples/s]13859 examples [00:14, 961.90 examples/s]13956 examples [00:14, 959.16 examples/s]14052 examples [00:14, 938.98 examples/s]14155 examples [00:14, 963.12 examples/s]14252 examples [00:14, 962.96 examples/s]14349 examples [00:14, 928.26 examples/s]14451 examples [00:14, 952.56 examples/s]14551 examples [00:14, 965.91 examples/s]14648 examples [00:15, 938.79 examples/s]14743 examples [00:15, 918.05 examples/s]14845 examples [00:15, 943.69 examples/s]14948 examples [00:15, 967.87 examples/s]15048 examples [00:15, 976.13 examples/s]15146 examples [00:15, 972.39 examples/s]15249 examples [00:15, 986.90 examples/s]15355 examples [00:15, 1006.26 examples/s]15461 examples [00:15, 1020.66 examples/s]15564 examples [00:15, 1002.91 examples/s]15665 examples [00:16, 1003.01 examples/s]15771 examples [00:16, 1017.96 examples/s]15878 examples [00:16, 1031.03 examples/s]15982 examples [00:16, 1009.69 examples/s]16084 examples [00:16, 999.73 examples/s] 16185 examples [00:16, 1001.43 examples/s]16286 examples [00:16, 998.04 examples/s] 16386 examples [00:16, 972.05 examples/s]16484 examples [00:16, 971.89 examples/s]16583 examples [00:17, 977.08 examples/s]16687 examples [00:17, 994.60 examples/s]16791 examples [00:17, 1007.40 examples/s]16892 examples [00:17, 988.67 examples/s] 16992 examples [00:17, 967.77 examples/s]17090 examples [00:17, 970.13 examples/s]17188 examples [00:17, 967.67 examples/s]17286 examples [00:17, 970.30 examples/s]17389 examples [00:17, 986.49 examples/s]17489 examples [00:17, 989.07 examples/s]17591 examples [00:18, 997.32 examples/s]17698 examples [00:18, 1016.67 examples/s]17800 examples [00:18, 1001.62 examples/s]17901 examples [00:18, 974.36 examples/s] 17999 examples [00:18, 959.51 examples/s]18097 examples [00:18, 963.61 examples/s]18194 examples [00:18, 953.18 examples/s]18290 examples [00:18, 951.11 examples/s]18392 examples [00:18, 969.99 examples/s]18491 examples [00:18, 973.70 examples/s]18596 examples [00:19, 994.48 examples/s]18703 examples [00:19, 1015.75 examples/s]18805 examples [00:19, 1015.84 examples/s]18909 examples [00:19, 1022.58 examples/s]19012 examples [00:19, 1021.41 examples/s]19115 examples [00:19, 1022.30 examples/s]19219 examples [00:19, 1024.87 examples/s]19322 examples [00:19, 1020.94 examples/s]19425 examples [00:19, 1020.96 examples/s]19528 examples [00:19, 995.93 examples/s] 19628 examples [00:20, 993.42 examples/s]19728 examples [00:20, 995.12 examples/s]19832 examples [00:20, 1007.67 examples/s]19934 examples [00:20, 1009.81 examples/s]20036 examples [00:20, 951.45 examples/s] 20140 examples [00:20, 974.52 examples/s]20243 examples [00:20, 988.45 examples/s]20343 examples [00:20, 987.66 examples/s]20443 examples [00:20, 939.44 examples/s]20540 examples [00:21, 945.63 examples/s]20643 examples [00:21, 967.33 examples/s]20742 examples [00:21, 973.81 examples/s]20845 examples [00:21, 976.82 examples/s]20951 examples [00:21, 998.18 examples/s]21052 examples [00:21, 975.87 examples/s]21161 examples [00:21, 1005.16 examples/s]21262 examples [00:21, 985.21 examples/s] 21366 examples [00:21, 999.97 examples/s]21473 examples [00:21, 1016.96 examples/s]21575 examples [00:22, 987.87 examples/s] 21675 examples [00:22, 989.83 examples/s]21779 examples [00:22, 1002.80 examples/s]21880 examples [00:22, 977.92 examples/s] 21981 examples [00:22, 985.54 examples/s]22080 examples [00:22, 965.69 examples/s]22177 examples [00:22, 963.60 examples/s]22276 examples [00:22, 970.21 examples/s]22382 examples [00:22, 993.56 examples/s]22486 examples [00:22, 1006.72 examples/s]22587 examples [00:23, 985.99 examples/s] 22690 examples [00:23, 996.32 examples/s]22796 examples [00:23, 1012.98 examples/s]22899 examples [00:23, 1016.41 examples/s]23001 examples [00:23, 966.90 examples/s] 23099 examples [00:23, 952.62 examples/s]23203 examples [00:23, 976.46 examples/s]23310 examples [00:23, 1001.32 examples/s]23412 examples [00:23, 1005.78 examples/s]23515 examples [00:24, 1011.82 examples/s]23617 examples [00:24, 999.15 examples/s] 23724 examples [00:24, 1017.53 examples/s]23826 examples [00:24, 1010.70 examples/s]23930 examples [00:24, 1018.28 examples/s]24032 examples [00:24, 1012.85 examples/s]24134 examples [00:24, 989.11 examples/s] 24234 examples [00:24, 986.13 examples/s]24333 examples [00:24, 951.36 examples/s]24429 examples [00:24, 948.43 examples/s]24531 examples [00:25, 966.93 examples/s]24628 examples [00:25, 965.81 examples/s]24729 examples [00:25, 976.40 examples/s]24835 examples [00:25, 998.52 examples/s]24943 examples [00:25, 1020.33 examples/s]25046 examples [00:25, 1014.80 examples/s]25148 examples [00:25, 996.23 examples/s] 25255 examples [00:25, 1016.78 examples/s]25357 examples [00:25, 1016.13 examples/s]25459 examples [00:25, 947.98 examples/s] 25555 examples [00:26, 938.00 examples/s]25656 examples [00:26, 958.22 examples/s]25754 examples [00:26, 962.20 examples/s]25851 examples [00:26, 936.12 examples/s]25946 examples [00:26, 929.35 examples/s]26040 examples [00:26, 925.04 examples/s]26145 examples [00:26, 958.24 examples/s]26242 examples [00:26, 947.02 examples/s]26341 examples [00:26, 956.79 examples/s]26437 examples [00:27, 957.21 examples/s]26535 examples [00:27, 961.88 examples/s]26635 examples [00:27, 970.52 examples/s]26744 examples [00:27, 1002.08 examples/s]26850 examples [00:27, 1018.60 examples/s]26954 examples [00:27, 1023.04 examples/s]27057 examples [00:27, 1003.55 examples/s]27158 examples [00:27, 946.78 examples/s] 27254 examples [00:27, 923.72 examples/s]27350 examples [00:27, 932.71 examples/s]27451 examples [00:28, 952.27 examples/s]27548 examples [00:28, 957.50 examples/s]27647 examples [00:28, 966.26 examples/s]27754 examples [00:28, 993.03 examples/s]27854 examples [00:28, 982.60 examples/s]27953 examples [00:28, 981.36 examples/s]28052 examples [00:28, 936.87 examples/s]28147 examples [00:28, 934.68 examples/s]28242 examples [00:28, 938.70 examples/s]28347 examples [00:28, 968.79 examples/s]28445 examples [00:29, 941.99 examples/s]28540 examples [00:29, 943.70 examples/s]28649 examples [00:29, 981.25 examples/s]28752 examples [00:29, 994.10 examples/s]28857 examples [00:29, 1007.95 examples/s]28959 examples [00:29, 1008.63 examples/s]29061 examples [00:29, 975.57 examples/s] 29159 examples [00:29, 969.27 examples/s]29258 examples [00:29, 975.23 examples/s]29356 examples [00:30, 935.26 examples/s]29451 examples [00:30, 911.30 examples/s]29543 examples [00:30, 910.32 examples/s]29635 examples [00:30, 897.86 examples/s]29730 examples [00:30, 912.57 examples/s]29827 examples [00:30, 926.30 examples/s]29928 examples [00:30, 948.53 examples/s]30024 examples [00:30, 895.23 examples/s]30115 examples [00:30, 858.66 examples/s]30202 examples [00:30, 839.39 examples/s]30287 examples [00:31, 830.80 examples/s]30374 examples [00:31, 836.83 examples/s]30474 examples [00:31, 879.66 examples/s]30577 examples [00:31, 919.18 examples/s]30686 examples [00:31, 963.37 examples/s]30787 examples [00:31, 975.73 examples/s]30886 examples [00:31, 963.59 examples/s]30984 examples [00:31, 967.87 examples/s]31087 examples [00:31, 985.08 examples/s]31192 examples [00:32, 999.19 examples/s]31296 examples [00:32, 1010.55 examples/s]31398 examples [00:32, 994.81 examples/s] 31506 examples [00:32, 1016.27 examples/s]31608 examples [00:32, 1014.01 examples/s]31710 examples [00:32, 994.09 examples/s] 31810 examples [00:32, 987.47 examples/s]31909 examples [00:32, 970.86 examples/s]32017 examples [00:32, 999.15 examples/s]32118 examples [00:32, 969.13 examples/s]32218 examples [00:33, 977.74 examples/s]32321 examples [00:33, 991.41 examples/s]32421 examples [00:33, 957.85 examples/s]32521 examples [00:33, 968.59 examples/s]32619 examples [00:33, 970.89 examples/s]32717 examples [00:33, 925.69 examples/s]32820 examples [00:33, 952.26 examples/s]32916 examples [00:33, 952.89 examples/s]33020 examples [00:33, 975.32 examples/s]33118 examples [00:33, 964.75 examples/s]33215 examples [00:34, 952.33 examples/s]33315 examples [00:34, 965.61 examples/s]33412 examples [00:34, 966.10 examples/s]33518 examples [00:34, 992.25 examples/s]33624 examples [00:34, 1010.52 examples/s]33731 examples [00:34, 1025.14 examples/s]33837 examples [00:34, 1033.67 examples/s]33941 examples [00:34, 1013.54 examples/s]34043 examples [00:34, 972.19 examples/s] 34141 examples [00:35, 963.41 examples/s]34242 examples [00:35, 975.98 examples/s]34342 examples [00:35, 982.10 examples/s]34441 examples [00:35, 972.64 examples/s]34547 examples [00:35, 993.99 examples/s]34652 examples [00:35, 1007.64 examples/s]34763 examples [00:35, 1035.44 examples/s]34867 examples [00:35, 1010.49 examples/s]34969 examples [00:35, 1008.27 examples/s]35076 examples [00:35, 1023.83 examples/s]35179 examples [00:36, 1012.53 examples/s]35288 examples [00:36, 1033.46 examples/s]35392 examples [00:36, 974.46 examples/s] 35492 examples [00:36, 981.17 examples/s]35593 examples [00:36, 987.85 examples/s]35698 examples [00:36, 1005.45 examples/s]35804 examples [00:36, 1018.10 examples/s]35907 examples [00:36, 1003.91 examples/s]36008 examples [00:36, 1004.10 examples/s]36113 examples [00:36, 1016.79 examples/s]36215 examples [00:37, 983.45 examples/s] 36314 examples [00:37, 964.53 examples/s]36411 examples [00:37, 950.43 examples/s]36509 examples [00:37, 957.63 examples/s]36608 examples [00:37, 967.00 examples/s]36708 examples [00:37, 974.66 examples/s]36806 examples [00:37, 960.91 examples/s]36903 examples [00:37, 953.34 examples/s]36999 examples [00:37, 944.25 examples/s]37102 examples [00:38, 967.25 examples/s]37202 examples [00:38, 975.39 examples/s]37305 examples [00:38, 990.77 examples/s]37405 examples [00:38, 984.74 examples/s]37505 examples [00:38, 989.02 examples/s]37604 examples [00:38, 984.78 examples/s]37703 examples [00:38, 980.77 examples/s]37807 examples [00:38, 995.42 examples/s]37907 examples [00:38, 984.43 examples/s]38006 examples [00:38, 974.81 examples/s]38106 examples [00:39, 981.89 examples/s]38205 examples [00:39, 978.45 examples/s]38308 examples [00:39, 991.79 examples/s]38408 examples [00:39, 987.27 examples/s]38507 examples [00:39, 980.06 examples/s]38607 examples [00:39, 985.65 examples/s]38712 examples [00:39, 1002.64 examples/s]38815 examples [00:39, 1009.28 examples/s]38917 examples [00:39, 1006.82 examples/s]39018 examples [00:39, 985.06 examples/s] 39117 examples [00:40, 955.32 examples/s]39213 examples [00:40, 954.88 examples/s]39309 examples [00:40, 938.33 examples/s]39404 examples [00:40, 922.06 examples/s]39508 examples [00:40, 953.47 examples/s]39614 examples [00:40, 981.67 examples/s]39718 examples [00:40, 997.30 examples/s]39823 examples [00:40, 1011.12 examples/s]39925 examples [00:40, 988.57 examples/s] 40025 examples [00:41, 905.11 examples/s]40125 examples [00:41, 930.28 examples/s]40228 examples [00:41, 955.78 examples/s]40330 examples [00:41, 972.34 examples/s]40429 examples [00:41, 947.61 examples/s]40525 examples [00:41, 934.15 examples/s]40627 examples [00:41, 957.06 examples/s]40731 examples [00:41, 980.02 examples/s]40830 examples [00:41, 957.72 examples/s]40927 examples [00:41, 951.25 examples/s]41028 examples [00:42, 967.96 examples/s]41133 examples [00:42, 989.51 examples/s]41239 examples [00:42, 1007.35 examples/s]41341 examples [00:42, 992.93 examples/s] 41441 examples [00:42, 947.52 examples/s]41539 examples [00:42, 955.28 examples/s]41640 examples [00:42, 970.15 examples/s]41738 examples [00:42, 964.47 examples/s]41835 examples [00:42, 962.98 examples/s]41932 examples [00:42, 958.12 examples/s]42040 examples [00:43, 988.91 examples/s]42147 examples [00:43, 1010.25 examples/s]42254 examples [00:43, 1027.14 examples/s]42358 examples [00:43, 1004.99 examples/s]42459 examples [00:43, 953.08 examples/s] 42556 examples [00:43, 950.02 examples/s]42660 examples [00:43, 974.86 examples/s]42765 examples [00:43, 995.19 examples/s]42866 examples [00:43, 992.37 examples/s]42969 examples [00:44, 1002.14 examples/s]43076 examples [00:44, 1021.36 examples/s]43179 examples [00:44, 1009.36 examples/s]43281 examples [00:44, 1003.39 examples/s]43382 examples [00:44, 983.22 examples/s] 43485 examples [00:44, 996.22 examples/s]43589 examples [00:44, 1008.87 examples/s]43691 examples [00:44, 1010.22 examples/s]43793 examples [00:44, 1011.10 examples/s]43895 examples [00:44, 950.91 examples/s] 44000 examples [00:45, 978.55 examples/s]44099 examples [00:45, 971.98 examples/s]44197 examples [00:45, 973.30 examples/s]44300 examples [00:45, 987.68 examples/s]44400 examples [00:45, 990.26 examples/s]44503 examples [00:45, 999.73 examples/s]44604 examples [00:45, 1000.46 examples/s]44705 examples [00:45, 992.59 examples/s] 44808 examples [00:45, 1001.96 examples/s]44911 examples [00:45, 1008.54 examples/s]45013 examples [00:46, 1009.72 examples/s]45115 examples [00:46, 998.54 examples/s] 45223 examples [00:46, 1021.18 examples/s]45331 examples [00:46, 1037.81 examples/s]45435 examples [00:46, 992.77 examples/s] 45535 examples [00:46, 990.90 examples/s]45635 examples [00:46, 991.78 examples/s]45736 examples [00:46, 996.18 examples/s]45836 examples [00:46, 963.64 examples/s]45933 examples [00:47, 937.10 examples/s]46031 examples [00:47, 949.07 examples/s]46138 examples [00:47, 982.13 examples/s]46240 examples [00:47, 991.37 examples/s]46344 examples [00:47, 1003.52 examples/s]46445 examples [00:47, 998.24 examples/s] 46550 examples [00:47, 1010.67 examples/s]46652 examples [00:47, 1000.51 examples/s]46756 examples [00:47, 1009.33 examples/s]46858 examples [00:47, 993.98 examples/s] 46958 examples [00:48, 970.11 examples/s]47061 examples [00:48, 985.27 examples/s]47164 examples [00:48, 995.30 examples/s]47264 examples [00:48, 987.79 examples/s]47366 examples [00:48, 995.08 examples/s]47466 examples [00:48, 966.45 examples/s]47563 examples [00:48, 963.22 examples/s]47664 examples [00:48, 975.59 examples/s]47771 examples [00:48, 1002.10 examples/s]47872 examples [00:48, 983.57 examples/s] 47971 examples [00:49, 981.80 examples/s]48072 examples [00:49, 989.97 examples/s]48176 examples [00:49, 1003.78 examples/s]48279 examples [00:49, 1011.05 examples/s]48381 examples [00:49, 977.32 examples/s] 48482 examples [00:49, 985.75 examples/s]48582 examples [00:49, 988.74 examples/s]48682 examples [00:49, 987.97 examples/s]48781 examples [00:49, 982.50 examples/s]48883 examples [00:49, 991.80 examples/s]48983 examples [00:50, 966.22 examples/s]49080 examples [00:50, 959.60 examples/s]49178 examples [00:50, 962.77 examples/s]49283 examples [00:50, 985.29 examples/s]49385 examples [00:50, 994.73 examples/s]49485 examples [00:50, 989.92 examples/s]49592 examples [00:50, 1012.09 examples/s]49696 examples [00:50, 1018.42 examples/s]49802 examples [00:50, 1029.68 examples/s]49906 examples [00:51, 994.81 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▌        | 7730/50000 [00:00<00:00, 77295.65 examples/s] 42%|████▏     | 21155/50000 [00:00<00:00, 88566.78 examples/s] 69%|██████▉   | 34409/50000 [00:00<00:00, 98326.34 examples/s] 96%|█████████▌| 47864/50000 [00:00<00:00, 106964.48 examples/s]                                                                0 examples [00:00, ? examples/s]82 examples [00:00, 814.19 examples/s]183 examples [00:00, 863.88 examples/s]284 examples [00:00, 902.36 examples/s]385 examples [00:00, 930.25 examples/s]488 examples [00:00, 957.87 examples/s]593 examples [00:00, 982.60 examples/s]696 examples [00:00, 994.50 examples/s]790 examples [00:00, 830.47 examples/s]884 examples [00:00, 860.23 examples/s]980 examples [00:01, 887.12 examples/s]1082 examples [00:01, 922.53 examples/s]1183 examples [00:01, 944.34 examples/s]1279 examples [00:01, 942.51 examples/s]1379 examples [00:01, 944.02 examples/s]1477 examples [00:01, 953.27 examples/s]1584 examples [00:01, 983.69 examples/s]1691 examples [00:01, 1005.52 examples/s]1800 examples [00:01, 1029.17 examples/s]1904 examples [00:01, 986.26 examples/s] 2004 examples [00:02, 978.15 examples/s]2108 examples [00:02, 994.86 examples/s]2214 examples [00:02, 1011.82 examples/s]2324 examples [00:02, 1036.19 examples/s]2429 examples [00:02, 1009.17 examples/s]2531 examples [00:02, 1010.91 examples/s]2639 examples [00:02, 1029.95 examples/s]2743 examples [00:02, 1020.09 examples/s]2852 examples [00:02, 1038.87 examples/s]2957 examples [00:03, 1035.79 examples/s]3061 examples [00:03, 964.05 examples/s] 3161 examples [00:03, 972.55 examples/s]3266 examples [00:03, 992.67 examples/s]3370 examples [00:03, 1005.51 examples/s]3472 examples [00:03, 1004.97 examples/s]3574 examples [00:03, 1001.44 examples/s]3675 examples [00:03, 987.40 examples/s] 3779 examples [00:03, 1000.40 examples/s]3881 examples [00:03, 1003.52 examples/s]3982 examples [00:04, 986.64 examples/s] 4088 examples [00:04, 1005.26 examples/s]4189 examples [00:04, 999.35 examples/s] 4290 examples [00:04, 956.82 examples/s]4387 examples [00:04, 917.82 examples/s]4480 examples [00:04, 899.68 examples/s]4578 examples [00:04, 921.51 examples/s]4671 examples [00:04, 919.56 examples/s]4764 examples [00:04, 921.41 examples/s]4857 examples [00:05, 914.94 examples/s]4950 examples [00:05, 917.11 examples/s]5042 examples [00:05, 912.28 examples/s]5143 examples [00:05, 938.82 examples/s]5238 examples [00:05, 924.96 examples/s]5331 examples [00:05, 916.70 examples/s]5426 examples [00:05, 926.39 examples/s]5522 examples [00:05, 935.90 examples/s]5616 examples [00:05, 928.18 examples/s]5729 examples [00:05, 980.72 examples/s]5832 examples [00:06, 992.82 examples/s]5932 examples [00:06, 983.03 examples/s]6034 examples [00:06, 993.17 examples/s]6137 examples [00:06, 1003.46 examples/s]6239 examples [00:06, 1006.18 examples/s]6342 examples [00:06, 1012.25 examples/s]6444 examples [00:06, 1005.23 examples/s]6545 examples [00:06, 992.15 examples/s] 6650 examples [00:06, 1007.76 examples/s]6757 examples [00:06, 1025.31 examples/s]6860 examples [00:07, 963.54 examples/s] 6958 examples [00:07, 943.85 examples/s]7062 examples [00:07, 968.15 examples/s]7169 examples [00:07, 994.90 examples/s]7273 examples [00:07, 1007.94 examples/s]7375 examples [00:07, 993.69 examples/s] 7477 examples [00:07, 999.81 examples/s]7587 examples [00:07, 1025.33 examples/s]7690 examples [00:07, 997.67 examples/s] 7794 examples [00:07, 1007.98 examples/s]7896 examples [00:08, 1003.44 examples/s]7997 examples [00:08, 979.14 examples/s] 8096 examples [00:08, 980.98 examples/s]8200 examples [00:08, 996.05 examples/s]8304 examples [00:08, 1008.49 examples/s]8406 examples [00:08, 1009.09 examples/s]8508 examples [00:08, 960.75 examples/s] 8605 examples [00:08, 960.79 examples/s]8706 examples [00:08, 974.18 examples/s]8812 examples [00:09, 998.16 examples/s]8916 examples [00:09, 1008.48 examples/s]9018 examples [00:09, 1010.26 examples/s]9125 examples [00:09, 1025.93 examples/s]9228 examples [00:09, 1005.30 examples/s]9329 examples [00:09, 993.59 examples/s] 9432 examples [00:09, 1002.71 examples/s]9537 examples [00:09, 1014.86 examples/s]9647 examples [00:09, 1037.63 examples/s]9751 examples [00:09, 1031.37 examples/s]9857 examples [00:10, 1039.68 examples/s]9962 examples [00:10, 1033.93 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYEIH6L/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteYEIH6L/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd96340d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd96340d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd96340d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd9c969db38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd8e8884f60>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fd96340d950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fd96340d950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fd96340d950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd9633fcc50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fd9497ce048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fd8dae052f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fd8dae052f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fd8dae052f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fd8dae05400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fd8dae05400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fd8dae05400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=289fe40e020c6cb85abb83ff16c5bfe0a9a2df9e7501a3c62d18ae8d2d090223
  Stored in directory: /tmp/pip-ephem-wheel-cache-afjr92xu/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<28:47:41, 8.32kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<20:23:19, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<14:19:31, 16.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<10:02:03, 23.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<7:00:18, 34.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.18M/862M [00:01<4:52:23, 48.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<3:23:52, 69.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<2:22:01, 99.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:01<1:39:01, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.3M/862M [00:02<1:08:57, 202kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:02<48:13, 287kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:02<33:42, 409kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.6M/862M [00:02<23:34, 582kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<17:36, 779kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.3M/862M [00:02<12:18, 1.11MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.6M/862M [00:02<08:42, 1.56MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<06:45, 2.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<06:37, 2.02MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<06:50, 1.96MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:20, 2.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:03, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:03, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:07<04:36, 2.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<05:48, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<07:18, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:47, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.6M/862M [00:09<04:15, 3.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<07:07, 1.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<06:27, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<04:52, 2.70MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<06:25, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<07:17, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<05:42, 2.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:13<04:07, 3.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<13:36, 961kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<10:53, 1.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<07:56, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:17<08:35, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<08:39, 1.50MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<06:37, 1.96MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<04:47, 2.70MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<10:34, 1.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<08:44, 1.48MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:19<06:26, 2.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:21<07:31, 1.71MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<06:34, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:21<04:55, 2.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:23<06:27, 1.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:23<05:49, 2.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<04:21, 2.94MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<06:04, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<06:50, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:25<05:25, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:50, 2.17MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:23, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:48, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:39, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:17, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:43, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:18, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<04:01, 3.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:35, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:14, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:39, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<03:55, 3.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:39, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:32, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:12, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:36, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:12, 2.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<03:53, 3.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<02:53, 4.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<1:07:46, 180kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<48:39, 250kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<34:15, 355kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<24:04, 503kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:47:09, 113kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<1:17:29, 156kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<54:48, 221kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<40:09, 300kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<29:20, 410kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<20:48, 577kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<17:19, 692kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:19, 898kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<09:36, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<09:31, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:05, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:55, 1.72MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<04:58, 2.38MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<12:12, 970kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:46, 1.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<07:08, 1.65MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:44, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:37, 1.78MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:55, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:13, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:30, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:09, 2.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:39, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:08, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:53, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:27, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:59, 2.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:46, 3.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:21, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:05, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:50, 2.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:13, 2.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<04:51, 2.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:41, 3.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:14, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:58, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:45, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:08, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:44, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:36, 3.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:09, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:32, 2.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:27, 3.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:04, 2.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<05:49, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:33, 2.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:19, 3.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<10:30, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<08:28, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:10, 1.79MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<06:54, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<05:57, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:26, 2.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:41, 1.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:50, 2.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:15, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:53, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:40, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:00, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:36, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<03:29, 3.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<04:57, 2.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:41, 1.89MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:31, 2.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:16, 3.25MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<43:22, 246kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<31:27, 339kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<22:12, 480kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<17:59, 590kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<14:44, 719kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:50, 977kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<09:15, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<07:34, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:20, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<06:34, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:02, 2.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<03:39, 2.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<08:03, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<06:41, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:56, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:52, 1.76MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:09, 2.00MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:51, 2.66MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:07, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<04:38, 2.21MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<03:29, 2.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:51, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:28, 1.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:20, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<04:40, 2.17MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:18, 2.35MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:16, 3.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:38, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:18, 1.90MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:13, 2.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<03:03, 3.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<1:14:06, 135kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<52:51, 189kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<37:07, 268kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<28:13, 352kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<20:45, 478kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<14:44, 671kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<12:35, 783kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<10:49, 910kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<08:04, 1.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<07:12, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:03, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:28, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:24, 1.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:45, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:31, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:43, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:17, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:14, 2.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:30, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:06, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:00, 2.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<02:53, 3.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<19:32, 487kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<14:39, 648kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<10:28, 905kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<09:31, 991kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<08:40, 1.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:28, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<04:38, 2.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<07:43, 1.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:22, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<04:41, 1.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:26, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:41, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<04:27, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:36, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:09, 2.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:08, 2.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:20, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:54, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:54, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:50, 3.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<1:16:21, 119kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<54:20, 167kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<38:09, 238kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<28:43, 315kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<22:00, 410kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<15:47, 571kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<11:08, 807kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<11:53, 754kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<09:15, 968kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:39, 1.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<06:44, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:39, 1.57MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:09, 2.13MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:19, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<04:10, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:19, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<03:54, 2.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:55, 2.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:04, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:44, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<02:49, 3.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:01, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:41, 2.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:47, 3.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:58, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<03:39, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<02:46, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:55, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:35, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<02:41, 3.14MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:53, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:34, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<02:42, 3.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:51, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:33, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<02:41, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:50, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:31, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<02:38, 3.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:47, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:19, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:26, 2.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:28, 3.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<1:01:04, 133kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<43:32, 187kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<30:33, 265kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<23:12, 348kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<17:52, 451kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<12:54, 625kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<10:16, 779kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<08:01, 998kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:48, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:53, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:44, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:21, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:08, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:14, 1.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:53, 1.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:17, 1.83MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:48, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:58, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:52, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<03:56, 1.96MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:33, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<02:41, 2.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<03:39, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:20, 2.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<02:31, 3.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:33, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:01, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<03:09, 2.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<02:18, 3.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:58, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:16, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:10, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:57, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:31, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:37, 2.83MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:35, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:57, 1.86MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:05, 2.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:14, 3.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<06:03, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<05:47, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:25, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<03:10, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<54:01, 134kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<38:32, 188kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<27:04, 267kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<20:32, 350kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<15:50, 453kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<11:23, 629kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<08:00, 889kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<12:28, 570kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<09:20, 761kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:43, 1.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<04:45, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<31:41, 222kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<23:38, 298kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<16:49, 418kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<11:51, 590kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<10:34, 660kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<08:07, 859kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:50, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<05:41, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<04:40, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:26, 2.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:00, 1.70MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:12, 1.62MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:17, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:23, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:03, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:18, 2.92MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:10, 2.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<02:54, 2.31MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:11, 3.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:06, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:31, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<02:45, 2.40MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:01, 3.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<04:01, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:29, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:35, 2.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:19, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<03:38, 1.78MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<02:49, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:05, 3.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:23, 1.90MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:01, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:16, 2.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:04, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:47, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:06, 3.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<02:57, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<02:42, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:02, 3.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:54, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:39, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:00, 3.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:51, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:14, 1.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:32, 2.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<01:50, 3.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<08:19, 731kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<06:26, 943kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:38, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:38, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<04:29, 1.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:23, 1.77MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:27, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:50, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:17, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:26, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:04, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:21, 1.75MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:35, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:53, 3.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:26, 1.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:42, 1.56MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:42, 2.13MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:14, 1.77MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:50, 2.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:07, 2.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:49, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:05, 1.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:23, 2.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:45, 3.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:46, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<03:13, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:23, 2.33MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<02:57, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:12, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:31, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<02:37, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:23, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<01:48, 3.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:32, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:52, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:16, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<02:26, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:15, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<01:42, 3.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:24, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:45, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:09, 2.42MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:33, 3.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<06:11, 838kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<04:52, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:30, 1.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:38, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<03:35, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:46, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:44, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:25, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:48, 2.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:25, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:42, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:06, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:32, 3.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:41, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<03:05, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:16, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:43, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:23, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<01:47, 2.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:22, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:38, 1.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:03, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:28, 3.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<04:28, 1.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:36, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:37, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:54, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:58, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:16, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:38, 2.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:33, 1.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:57, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<02:09, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:34, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:42, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<02:07, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:31, 2.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<33:08, 134kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<23:37, 188kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<16:33, 266kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<12:31, 349kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<09:38, 453kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<06:57, 626kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:30, 781kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<04:17, 1.00MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:05, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:07, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:03, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:21, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:18, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:02, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:30, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:00, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:16, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:47, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:17, 3.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<29:41, 136kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<21:10, 190kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<14:50, 269kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<11:13, 353kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<08:14, 479kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<05:50, 673kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<04:57, 785kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<04:16, 909kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:09, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:14, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<04:24, 868kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<03:27, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:30, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:36, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:37, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:00, 1.84MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:47, 2.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:19, 2.77MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:46, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:58, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:33, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:07, 3.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<3:23:12, 17.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<2:22:21, 24.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<1:39:00, 35.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<1:09:24, 50.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<48:50, 71.1kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<34:01, 101kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<24:22, 140kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<17:22, 196kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<12:09, 278kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<09:12, 363kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<06:46, 492kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:47, 691kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<04:05, 800kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:31, 927kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:37, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:19, 1.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:57, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:26, 2.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:43, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:31, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:07, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:30, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:41, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:18, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<00:56, 3.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:00, 1.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:42, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:15, 2.36MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:33, 1.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:41, 1.73MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:19, 2.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:22, 2.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:15, 2.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<00:56, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:28, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:10, 2.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:50, 3.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<22:48, 119kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<16:12, 168kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<11:18, 238kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<08:25, 315kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<06:09, 430kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<04:20, 605kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:36, 718kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:47, 927kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:59, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:58, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:53, 1.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:26, 1.73MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<01:01, 2.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<2:21:29, 17.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<1:39:02, 24.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<1:08:39, 35.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<47:53, 49.7kB/s]  .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<33:58, 70.0kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<23:45, 99.6kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<16:18, 142kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<19:24, 119kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<13:47, 167kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<09:36, 238kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<07:08, 314kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<05:12, 429kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<03:39, 604kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<03:02, 716kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:33, 847kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:52, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<01:18, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:51, 735kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:13, 946kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:35, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:34, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:30, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:09, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:06, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:58, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:43, 2.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:56, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:02, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:49, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:35, 3.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<07:28, 246kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<05:23, 338kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<03:46, 478kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:59, 588kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:27, 716kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:47, 975kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:14, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<1:40:22, 16.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<1:10:10, 24.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<48:24, 34.4kB/s]  .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<33:31, 48.5kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<23:32, 68.9kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<16:15, 98.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<11:29, 136kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<08:19, 187kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<05:51, 263kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<03:58, 374kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<1:28:34, 16.8kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<1:01:53, 24.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<42:36, 34.2kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<29:25, 48.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<20:38, 68.6kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<14:12, 97.8kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<10:00, 135kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<07:16, 186kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<05:06, 262kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<03:28, 373kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<03:19, 387kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<02:26, 523kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:42, 733kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:26, 841kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:15, 964kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:55, 1.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:39, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:49, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:41, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:30, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:35, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:31, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:23, 2.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:30, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:32, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:25, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:17, 3.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:38, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:33, 1.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:23, 2.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:28, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:25, 2.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:18, 2.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:14, 3.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:24, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:15, 2.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:19, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:21, 1.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:16, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:15, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:11, 3.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:14, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:16, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:12, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:08, 3.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:19, 1.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:06, 2.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:14, 1.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:12, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:08, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:08, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:09, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:06, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.25MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 3.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:22, 136kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:13, 190kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:04, 270kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<28:54:44,  3.84it/s]  0%|          | 896/400000 [00:00<20:11:49,  5.49it/s]  0%|          | 1823/400000 [00:00<14:06:31,  7.84it/s]  1%|          | 2744/400000 [00:00<9:51:24, 11.20it/s]   1%|          | 3665/400000 [00:00<6:53:14, 15.98it/s]  1%|          | 4637/400000 [00:00<4:48:45, 22.82it/s]  1%|▏         | 5594/400000 [00:00<3:21:51, 32.57it/s]  2%|▏         | 6445/400000 [00:00<2:21:13, 46.45it/s]  2%|▏         | 7354/400000 [00:01<1:38:50, 66.21it/s]  2%|▏         | 8213/400000 [00:01<1:09:17, 94.24it/s]  2%|▏         | 9085/400000 [00:01<48:36, 134.01it/s]   3%|▎         | 10075/400000 [00:01<34:08, 190.34it/s]  3%|▎         | 11004/400000 [00:01<24:03, 269.55it/s]  3%|▎         | 11941/400000 [00:01<17:00, 380.38it/s]  3%|▎         | 12854/400000 [00:01<12:05, 533.54it/s]  3%|▎         | 13789/400000 [00:01<08:39, 744.00it/s]  4%|▎         | 14702/400000 [00:01<06:17, 1019.36it/s]  4%|▍         | 15641/400000 [00:01<04:36, 1391.42it/s]  4%|▍         | 16527/400000 [00:02<03:27, 1850.45it/s]  4%|▍         | 17440/400000 [00:02<02:37, 2432.07it/s]  5%|▍         | 18344/400000 [00:02<02:02, 3115.12it/s]  5%|▍         | 19229/400000 [00:02<01:38, 3851.82it/s]  5%|▌         | 20190/400000 [00:02<01:20, 4695.54it/s]  5%|▌         | 21093/400000 [00:02<01:10, 5365.28it/s]  5%|▌         | 21974/400000 [00:02<01:02, 6077.27it/s]  6%|▌         | 22939/400000 [00:02<00:55, 6836.34it/s]  6%|▌         | 23903/400000 [00:02<00:50, 7489.24it/s]  6%|▌         | 24881/400000 [00:03<00:46, 8053.26it/s]  6%|▋         | 25819/400000 [00:03<00:45, 8272.21it/s]  7%|▋         | 26752/400000 [00:03<00:43, 8561.02it/s]  7%|▋         | 27727/400000 [00:03<00:41, 8885.47it/s]  7%|▋         | 28671/400000 [00:03<00:41, 9042.54it/s]  7%|▋         | 29631/400000 [00:03<00:40, 9200.36it/s]  8%|▊         | 30578/400000 [00:03<00:40, 9053.15it/s]  8%|▊         | 31532/400000 [00:03<00:40, 9192.01it/s]  8%|▊         | 32507/400000 [00:03<00:39, 9351.59it/s]  8%|▊         | 33453/400000 [00:03<00:39, 9380.39it/s]  9%|▊         | 34399/400000 [00:04<00:40, 9106.81it/s]  9%|▉         | 35317/400000 [00:04<00:40, 8949.57it/s]  9%|▉         | 36218/400000 [00:04<00:41, 8852.84it/s]  9%|▉         | 37108/400000 [00:04<00:42, 8535.88it/s]  9%|▉         | 37967/400000 [00:04<00:42, 8544.56it/s] 10%|▉         | 38864/400000 [00:04<00:41, 8666.81it/s] 10%|▉         | 39734/400000 [00:04<00:42, 8384.24it/s] 10%|█         | 40670/400000 [00:04<00:41, 8654.88it/s] 10%|█         | 41574/400000 [00:04<00:40, 8765.79it/s] 11%|█         | 42455/400000 [00:04<00:42, 8483.98it/s] 11%|█         | 43399/400000 [00:05<00:40, 8748.44it/s] 11%|█         | 44280/400000 [00:05<00:40, 8741.28it/s] 11%|█▏        | 45158/400000 [00:05<00:40, 8658.34it/s] 12%|█▏        | 46027/400000 [00:05<00:42, 8401.39it/s] 12%|█▏        | 46871/400000 [00:05<00:44, 7935.86it/s] 12%|█▏        | 47691/400000 [00:05<00:43, 8011.19it/s] 12%|█▏        | 48529/400000 [00:05<00:43, 8116.34it/s] 12%|█▏        | 49395/400000 [00:05<00:42, 8272.06it/s] 13%|█▎        | 50294/400000 [00:05<00:41, 8475.06it/s] 13%|█▎        | 51146/400000 [00:06<00:41, 8337.64it/s] 13%|█▎        | 52032/400000 [00:06<00:40, 8487.72it/s] 13%|█▎        | 52920/400000 [00:06<00:40, 8600.14it/s] 13%|█▎        | 53870/400000 [00:06<00:39, 8850.35it/s] 14%|█▎        | 54835/400000 [00:06<00:38, 9073.81it/s] 14%|█▍        | 55747/400000 [00:06<00:38, 8964.04it/s] 14%|█▍        | 56712/400000 [00:06<00:37, 9157.02it/s] 14%|█▍        | 57631/400000 [00:06<00:37, 9105.79it/s] 15%|█▍        | 58595/400000 [00:06<00:36, 9259.20it/s] 15%|█▍        | 59581/400000 [00:06<00:36, 9430.64it/s] 15%|█▌        | 60527/400000 [00:07<00:36, 9361.71it/s] 15%|█▌        | 61465/400000 [00:07<00:36, 9365.47it/s] 16%|█▌        | 62403/400000 [00:07<00:36, 9137.86it/s] 16%|█▌        | 63319/400000 [00:07<00:37, 9076.25it/s] 16%|█▌        | 64229/400000 [00:07<00:38, 8725.41it/s] 16%|█▋        | 65179/400000 [00:07<00:37, 8943.86it/s] 17%|█▋        | 66078/400000 [00:07<00:37, 8881.55it/s] 17%|█▋        | 66970/400000 [00:07<00:38, 8740.99it/s] 17%|█▋        | 67917/400000 [00:07<00:37, 8945.28it/s] 17%|█▋        | 68829/400000 [00:07<00:36, 8996.18it/s] 17%|█▋        | 69772/400000 [00:08<00:36, 9121.39it/s] 18%|█▊        | 70752/400000 [00:08<00:35, 9312.29it/s] 18%|█▊        | 71686/400000 [00:08<00:36, 9067.20it/s] 18%|█▊        | 72654/400000 [00:08<00:35, 9241.43it/s] 18%|█▊        | 73586/400000 [00:08<00:35, 9263.73it/s] 19%|█▊        | 74535/400000 [00:08<00:34, 9329.11it/s] 19%|█▉        | 75470/400000 [00:08<00:35, 9225.65it/s] 19%|█▉        | 76394/400000 [00:08<00:35, 9079.32it/s] 19%|█▉        | 77343/400000 [00:08<00:35, 9196.29it/s] 20%|█▉        | 78330/400000 [00:08<00:34, 9387.14it/s] 20%|█▉        | 79271/400000 [00:09<00:34, 9361.68it/s] 20%|██        | 80241/400000 [00:09<00:33, 9453.99it/s] 20%|██        | 81188/400000 [00:09<00:35, 8898.28it/s] 21%|██        | 82086/400000 [00:09<00:36, 8700.52it/s] 21%|██        | 83064/400000 [00:09<00:35, 8996.53it/s] 21%|██        | 83971/400000 [00:09<00:35, 8944.11it/s] 21%|██        | 84876/400000 [00:09<00:35, 8972.93it/s] 21%|██▏       | 85812/400000 [00:09<00:34, 9082.83it/s] 22%|██▏       | 86738/400000 [00:09<00:34, 9132.41it/s] 22%|██▏       | 87679/400000 [00:10<00:33, 9212.18it/s] 22%|██▏       | 88649/400000 [00:10<00:33, 9351.97it/s] 22%|██▏       | 89586/400000 [00:10<00:33, 9290.99it/s] 23%|██▎       | 90517/400000 [00:10<00:34, 9067.62it/s] 23%|██▎       | 91426/400000 [00:10<00:34, 8905.60it/s] 23%|██▎       | 92319/400000 [00:10<00:35, 8788.32it/s] 23%|██▎       | 93221/400000 [00:10<00:34, 8856.08it/s] 24%|██▎       | 94138/400000 [00:10<00:34, 8946.28it/s] 24%|██▍       | 95034/400000 [00:10<00:34, 8894.10it/s] 24%|██▍       | 95952/400000 [00:10<00:33, 8975.51it/s] 24%|██▍       | 96851/400000 [00:11<00:33, 8964.04it/s] 24%|██▍       | 97846/400000 [00:11<00:32, 9237.69it/s] 25%|██▍       | 98773/400000 [00:11<00:32, 9196.74it/s] 25%|██▍       | 99698/400000 [00:11<00:32, 9212.54it/s] 25%|██▌       | 100679/400000 [00:11<00:31, 9381.82it/s] 25%|██▌       | 101633/400000 [00:11<00:31, 9427.10it/s] 26%|██▌       | 102577/400000 [00:11<00:31, 9400.49it/s] 26%|██▌       | 103518/400000 [00:11<00:32, 8988.00it/s] 26%|██▌       | 104422/400000 [00:11<00:33, 8930.48it/s] 26%|██▋       | 105385/400000 [00:11<00:32, 9127.85it/s] 27%|██▋       | 106302/400000 [00:12<00:32, 8947.01it/s] 27%|██▋       | 107268/400000 [00:12<00:31, 9147.88it/s] 27%|██▋       | 108187/400000 [00:12<00:32, 9098.03it/s] 27%|██▋       | 109100/400000 [00:12<00:31, 9107.58it/s] 28%|██▊       | 110013/400000 [00:12<00:31, 9077.55it/s] 28%|██▊       | 110922/400000 [00:12<00:32, 9023.42it/s] 28%|██▊       | 111826/400000 [00:12<00:32, 8759.18it/s] 28%|██▊       | 112705/400000 [00:12<00:32, 8709.91it/s] 28%|██▊       | 113616/400000 [00:12<00:32, 8824.36it/s] 29%|██▊       | 114553/400000 [00:13<00:31, 8981.18it/s] 29%|██▉       | 115461/400000 [00:13<00:31, 9008.42it/s] 29%|██▉       | 116429/400000 [00:13<00:30, 9199.38it/s] 29%|██▉       | 117351/400000 [00:13<00:31, 9015.95it/s] 30%|██▉       | 118299/400000 [00:13<00:30, 9149.51it/s] 30%|██▉       | 119216/400000 [00:13<00:30, 9090.78it/s] 30%|███       | 120127/400000 [00:13<00:31, 8980.09it/s] 30%|███       | 121027/400000 [00:13<00:31, 8729.82it/s] 30%|███       | 121953/400000 [00:13<00:31, 8881.68it/s] 31%|███       | 122898/400000 [00:13<00:30, 9042.16it/s] 31%|███       | 123852/400000 [00:14<00:30, 9184.75it/s] 31%|███       | 124773/400000 [00:14<00:30, 9067.45it/s] 31%|███▏      | 125682/400000 [00:14<00:30, 9000.31it/s] 32%|███▏      | 126584/400000 [00:14<00:31, 8745.94it/s] 32%|███▏      | 127462/400000 [00:14<00:31, 8639.99it/s] 32%|███▏      | 128409/400000 [00:14<00:30, 8871.63it/s] 32%|███▏      | 129300/400000 [00:14<00:30, 8744.58it/s] 33%|███▎      | 130277/400000 [00:14<00:29, 9027.09it/s] 33%|███▎      | 131189/400000 [00:14<00:29, 9053.63it/s] 33%|███▎      | 132148/400000 [00:14<00:29, 9205.71it/s] 33%|███▎      | 133136/400000 [00:15<00:28, 9393.69it/s] 34%|███▎      | 134079/400000 [00:15<00:28, 9254.66it/s] 34%|███▍      | 135080/400000 [00:15<00:27, 9466.47it/s] 34%|███▍      | 136030/400000 [00:15<00:28, 9298.68it/s] 34%|███▍      | 136982/400000 [00:15<00:28, 9361.47it/s] 34%|███▍      | 137943/400000 [00:15<00:27, 9434.08it/s] 35%|███▍      | 138894/400000 [00:15<00:27, 9455.69it/s] 35%|███▍      | 139841/400000 [00:15<00:30, 8636.19it/s] 35%|███▌      | 140724/400000 [00:15<00:29, 8690.93it/s] 35%|███▌      | 141604/400000 [00:16<00:29, 8659.96it/s] 36%|███▌      | 142478/400000 [00:16<00:29, 8617.33it/s] 36%|███▌      | 143469/400000 [00:16<00:28, 8966.03it/s] 36%|███▌      | 144466/400000 [00:16<00:27, 9244.27it/s] 36%|███▋      | 145398/400000 [00:16<00:28, 9051.63it/s] 37%|███▋      | 146336/400000 [00:16<00:27, 9145.92it/s] 37%|███▋      | 147279/400000 [00:16<00:27, 9227.97it/s] 37%|███▋      | 148206/400000 [00:16<00:27, 9114.56it/s] 37%|███▋      | 149121/400000 [00:16<00:28, 8881.90it/s] 38%|███▊      | 150013/400000 [00:16<00:28, 8747.40it/s] 38%|███▊      | 150962/400000 [00:17<00:27, 8956.81it/s] 38%|███▊      | 151861/400000 [00:17<00:28, 8754.64it/s] 38%|███▊      | 152740/400000 [00:17<00:28, 8724.61it/s] 38%|███▊      | 153615/400000 [00:17<00:28, 8621.33it/s] 39%|███▊      | 154479/400000 [00:17<00:28, 8500.22it/s] 39%|███▉      | 155406/400000 [00:17<00:28, 8716.20it/s] 39%|███▉      | 156281/400000 [00:17<00:28, 8608.05it/s] 39%|███▉      | 157144/400000 [00:17<00:28, 8435.73it/s] 40%|███▉      | 158087/400000 [00:17<00:27, 8709.75it/s] 40%|███▉      | 158962/400000 [00:17<00:28, 8591.64it/s] 40%|███▉      | 159885/400000 [00:18<00:27, 8772.45it/s] 40%|████      | 160852/400000 [00:18<00:26, 9022.19it/s] 40%|████      | 161830/400000 [00:18<00:25, 9235.10it/s] 41%|████      | 162798/400000 [00:18<00:25, 9361.62it/s] 41%|████      | 163738/400000 [00:18<00:25, 9239.71it/s] 41%|████      | 164679/400000 [00:18<00:25, 9289.45it/s] 41%|████▏     | 165610/400000 [00:18<00:25, 9178.33it/s] 42%|████▏     | 166553/400000 [00:18<00:25, 9250.23it/s] 42%|████▏     | 167514/400000 [00:18<00:24, 9352.26it/s] 42%|████▏     | 168451/400000 [00:18<00:24, 9311.41it/s] 42%|████▏     | 169383/400000 [00:19<00:24, 9313.09it/s] 43%|████▎     | 170315/400000 [00:19<00:24, 9301.16it/s] 43%|████▎     | 171281/400000 [00:19<00:24, 9405.27it/s] 43%|████▎     | 172223/400000 [00:19<00:25, 8819.92it/s] 43%|████▎     | 173142/400000 [00:19<00:25, 8927.64it/s] 44%|████▎     | 174115/400000 [00:19<00:24, 9152.56it/s] 44%|████▍     | 175036/400000 [00:19<00:24, 9132.79it/s] 44%|████▍     | 175954/400000 [00:19<00:24, 9144.23it/s] 44%|████▍     | 176872/400000 [00:19<00:25, 8758.79it/s] 44%|████▍     | 177771/400000 [00:20<00:25, 8825.53it/s] 45%|████▍     | 178685/400000 [00:20<00:24, 8916.53it/s] 45%|████▍     | 179591/400000 [00:20<00:24, 8957.77it/s] 45%|████▌     | 180489/400000 [00:20<00:24, 8944.13it/s] 45%|████▌     | 181405/400000 [00:20<00:24, 9006.28it/s] 46%|████▌     | 182307/400000 [00:20<00:24, 8909.20it/s] 46%|████▌     | 183220/400000 [00:20<00:24, 8971.75it/s] 46%|████▌     | 184142/400000 [00:20<00:23, 9042.75it/s] 46%|████▋     | 185047/400000 [00:20<00:24, 8944.60it/s] 46%|████▋     | 185943/400000 [00:20<00:25, 8523.55it/s] 47%|████▋     | 186801/400000 [00:21<00:26, 7927.79it/s] 47%|████▋     | 187711/400000 [00:21<00:25, 8246.32it/s] 47%|████▋     | 188701/400000 [00:21<00:24, 8681.28it/s] 47%|████▋     | 189648/400000 [00:21<00:23, 8901.82it/s] 48%|████▊     | 190550/400000 [00:21<00:23, 8919.01it/s] 48%|████▊     | 191511/400000 [00:21<00:22, 9113.02it/s] 48%|████▊     | 192499/400000 [00:21<00:22, 9329.46it/s] 48%|████▊     | 193438/400000 [00:21<00:22, 9211.17it/s] 49%|████▊     | 194364/400000 [00:21<00:22, 8984.06it/s] 49%|████▉     | 195267/400000 [00:21<00:23, 8736.75it/s] 49%|████▉     | 196214/400000 [00:22<00:22, 8942.96it/s] 49%|████▉     | 197113/400000 [00:22<00:23, 8746.72it/s] 49%|████▉     | 197992/400000 [00:22<00:23, 8738.60it/s] 50%|████▉     | 198869/400000 [00:22<00:23, 8716.66it/s] 50%|████▉     | 199743/400000 [00:22<00:23, 8695.72it/s] 50%|█████     | 200671/400000 [00:22<00:22, 8858.33it/s] 50%|█████     | 201559/400000 [00:22<00:22, 8755.18it/s] 51%|█████     | 202436/400000 [00:22<00:23, 8579.03it/s] 51%|█████     | 203312/400000 [00:22<00:22, 8628.49it/s] 51%|█████     | 204177/400000 [00:23<00:23, 8342.56it/s] 51%|█████▏    | 205087/400000 [00:23<00:22, 8555.94it/s] 51%|█████▏    | 205975/400000 [00:23<00:22, 8649.20it/s] 52%|█████▏    | 206906/400000 [00:23<00:21, 8837.09it/s] 52%|█████▏    | 207905/400000 [00:23<00:20, 9153.93it/s] 52%|█████▏    | 208826/400000 [00:23<00:20, 9124.03it/s] 52%|█████▏    | 209742/400000 [00:23<00:20, 9109.38it/s] 53%|█████▎    | 210656/400000 [00:23<00:21, 8985.53it/s] 53%|█████▎    | 211611/400000 [00:23<00:20, 9145.12it/s] 53%|█████▎    | 212547/400000 [00:23<00:20, 9208.53it/s] 53%|█████▎    | 213482/400000 [00:24<00:20, 9250.05it/s] 54%|█████▎    | 214463/400000 [00:24<00:19, 9409.09it/s] 54%|█████▍    | 215406/400000 [00:24<00:20, 9033.34it/s] 54%|█████▍    | 216314/400000 [00:24<00:20, 8979.25it/s] 54%|█████▍    | 217215/400000 [00:24<00:21, 8700.23it/s] 55%|█████▍    | 218095/400000 [00:24<00:20, 8727.68it/s] 55%|█████▍    | 218979/400000 [00:24<00:20, 8760.06it/s] 55%|█████▍    | 219964/400000 [00:24<00:19, 9058.56it/s] 55%|█████▌    | 220943/400000 [00:24<00:19, 9264.76it/s] 55%|█████▌    | 221898/400000 [00:24<00:19, 9348.34it/s] 56%|█████▌    | 222836/400000 [00:25<00:19, 9284.00it/s] 56%|█████▌    | 223773/400000 [00:25<00:18, 9309.33it/s] 56%|█████▌    | 224769/400000 [00:25<00:18, 9495.03it/s] 56%|█████▋    | 225755/400000 [00:25<00:18, 9601.41it/s] 57%|█████▋    | 226719/400000 [00:25<00:18, 9611.28it/s] 57%|█████▋    | 227682/400000 [00:25<00:18, 9334.73it/s] 57%|█████▋    | 228668/400000 [00:25<00:18, 9484.07it/s] 57%|█████▋    | 229619/400000 [00:25<00:18, 9410.11it/s] 58%|█████▊    | 230562/400000 [00:25<00:18, 9364.51it/s] 58%|█████▊    | 231503/400000 [00:25<00:17, 9373.56it/s] 58%|█████▊    | 232442/400000 [00:26<00:19, 8746.51it/s] 58%|█████▊    | 233326/400000 [00:26<00:19, 8752.47it/s] 59%|█████▊    | 234273/400000 [00:26<00:18, 8955.85it/s] 59%|█████▉    | 235260/400000 [00:26<00:17, 9209.65it/s] 59%|█████▉    | 236187/400000 [00:26<00:18, 8959.95it/s] 59%|█████▉    | 237145/400000 [00:26<00:17, 9135.05it/s] 60%|█████▉    | 238103/400000 [00:26<00:17, 9263.80it/s] 60%|█████▉    | 239038/400000 [00:26<00:17, 9288.53it/s] 60%|█████▉    | 239982/400000 [00:26<00:17, 9332.51it/s] 60%|██████    | 240918/400000 [00:27<00:17, 9277.93it/s] 60%|██████    | 241848/400000 [00:27<00:17, 9164.53it/s] 61%|██████    | 242828/400000 [00:27<00:16, 9346.29it/s] 61%|██████    | 243765/400000 [00:27<00:16, 9335.51it/s] 61%|██████    | 244700/400000 [00:27<00:17, 9102.14it/s] 61%|██████▏   | 245618/400000 [00:27<00:16, 9124.49it/s] 62%|██████▏   | 246532/400000 [00:27<00:16, 9110.98it/s] 62%|██████▏   | 247445/400000 [00:27<00:16, 9080.68it/s] 62%|██████▏   | 248354/400000 [00:27<00:17, 8873.58it/s] 62%|██████▏   | 249243/400000 [00:27<00:16, 8874.62it/s] 63%|██████▎   | 250208/400000 [00:28<00:16, 9090.26it/s] 63%|██████▎   | 251144/400000 [00:28<00:16, 9167.35it/s] 63%|██████▎   | 252117/400000 [00:28<00:15, 9327.80it/s] 63%|██████▎   | 253069/400000 [00:28<00:15, 9384.30it/s] 64%|██████▎   | 254013/400000 [00:28<00:15, 9398.55it/s] 64%|██████▎   | 254954/400000 [00:28<00:15, 9155.69it/s] 64%|██████▍   | 255872/400000 [00:28<00:16, 8925.75it/s] 64%|██████▍   | 256795/400000 [00:28<00:15, 9013.07it/s] 64%|██████▍   | 257699/400000 [00:28<00:15, 9014.53it/s] 65%|██████▍   | 258602/400000 [00:28<00:15, 8998.31it/s] 65%|██████▍   | 259503/400000 [00:29<00:15, 8988.89it/s] 65%|██████▌   | 260403/400000 [00:29<00:15, 8795.15it/s] 65%|██████▌   | 261294/400000 [00:29<00:15, 8827.32it/s] 66%|██████▌   | 262178/400000 [00:29<00:15, 8784.36it/s] 66%|██████▌   | 263058/400000 [00:29<00:16, 8497.87it/s] 66%|██████▌   | 263954/400000 [00:29<00:15, 8629.57it/s] 66%|██████▌   | 264841/400000 [00:29<00:15, 8698.00it/s] 66%|██████▋   | 265742/400000 [00:29<00:15, 8787.40it/s] 67%|██████▋   | 266623/400000 [00:29<00:15, 8768.60it/s] 67%|██████▋   | 267561/400000 [00:29<00:14, 8942.70it/s] 67%|██████▋   | 268493/400000 [00:30<00:14, 9052.00it/s] 67%|██████▋   | 269400/400000 [00:30<00:14, 8894.04it/s] 68%|██████▊   | 270292/400000 [00:30<00:14, 8837.50it/s] 68%|██████▊   | 271198/400000 [00:30<00:14, 8900.78it/s] 68%|██████▊   | 272151/400000 [00:30<00:14, 9080.10it/s] 68%|██████▊   | 273061/400000 [00:30<00:13, 9080.24it/s] 68%|██████▊   | 273971/400000 [00:30<00:13, 9010.70it/s] 69%|██████▊   | 274923/400000 [00:30<00:13, 9157.10it/s] 69%|██████▉   | 275870/400000 [00:30<00:13, 9247.33it/s] 69%|██████▉   | 276796/400000 [00:30<00:13, 9249.65it/s] 69%|██████▉   | 277722/400000 [00:31<00:13, 8982.70it/s] 70%|██████▉   | 278623/400000 [00:31<00:13, 8697.87it/s] 70%|██████▉   | 279527/400000 [00:31<00:13, 8796.51it/s] 70%|███████   | 280447/400000 [00:31<00:13, 8913.06it/s] 70%|███████   | 281425/400000 [00:31<00:12, 9154.10it/s] 71%|███████   | 282377/400000 [00:31<00:12, 9260.57it/s] 71%|███████   | 283306/400000 [00:31<00:12, 9189.33it/s] 71%|███████   | 284227/400000 [00:31<00:12, 9189.25it/s] 71%|███████▏  | 285155/400000 [00:31<00:12, 9214.38it/s] 72%|███████▏  | 286099/400000 [00:32<00:12, 9280.17it/s] 72%|███████▏  | 287028/400000 [00:32<00:12, 9249.77it/s] 72%|███████▏  | 287954/400000 [00:32<00:12, 9073.58it/s] 72%|███████▏  | 288863/400000 [00:32<00:12, 8848.73it/s] 72%|███████▏  | 289767/400000 [00:32<00:12, 8903.32it/s] 73%|███████▎  | 290725/400000 [00:32<00:12, 9094.26it/s] 73%|███████▎  | 291666/400000 [00:32<00:11, 9185.22it/s] 73%|███████▎  | 292587/400000 [00:32<00:12, 8798.02it/s] 73%|███████▎  | 293472/400000 [00:32<00:12, 8388.31it/s] 74%|███████▎  | 294319/400000 [00:32<00:12, 8332.64it/s] 74%|███████▍  | 295158/400000 [00:33<00:12, 8151.26it/s] 74%|███████▍  | 295978/400000 [00:33<00:13, 7873.49it/s] 74%|███████▍  | 296771/400000 [00:33<00:13, 7499.55it/s] 74%|███████▍  | 297712/400000 [00:33<00:12, 7985.49it/s] 75%|███████▍  | 298695/400000 [00:33<00:11, 8461.21it/s] 75%|███████▍  | 299661/400000 [00:33<00:11, 8787.36it/s] 75%|███████▌  | 300556/400000 [00:33<00:11, 8742.15it/s] 75%|███████▌  | 301549/400000 [00:33<00:10, 9066.59it/s] 76%|███████▌  | 302477/400000 [00:33<00:10, 9129.15it/s] 76%|███████▌  | 303467/400000 [00:34<00:10, 9345.06it/s] 76%|███████▌  | 304409/400000 [00:34<00:10, 9366.28it/s] 76%|███████▋  | 305351/400000 [00:34<00:10, 9193.94it/s] 77%|███████▋  | 306323/400000 [00:34<00:10, 9343.63it/s] 77%|███████▋  | 307273/400000 [00:34<00:09, 9388.84it/s] 77%|███████▋  | 308215/400000 [00:34<00:10, 8878.56it/s] 77%|███████▋  | 309111/400000 [00:34<00:10, 8578.04it/s] 77%|███████▋  | 309977/400000 [00:34<00:10, 8383.72it/s] 78%|███████▊  | 310876/400000 [00:34<00:10, 8555.97it/s] 78%|███████▊  | 311817/400000 [00:34<00:10, 8792.80it/s] 78%|███████▊  | 312744/400000 [00:35<00:09, 8930.12it/s] 78%|███████▊  | 313666/400000 [00:35<00:09, 9014.52it/s] 79%|███████▊  | 314571/400000 [00:35<00:09, 9002.68it/s] 79%|███████▉  | 315550/400000 [00:35<00:09, 9222.27it/s] 79%|███████▉  | 316476/400000 [00:35<00:09, 9162.37it/s] 79%|███████▉  | 317395/400000 [00:35<00:09, 8844.75it/s] 80%|███████▉  | 318374/400000 [00:35<00:08, 9105.19it/s] 80%|███████▉  | 319290/400000 [00:35<00:09, 8959.09it/s] 80%|████████  | 320205/400000 [00:35<00:08, 9013.83it/s] 80%|████████  | 321146/400000 [00:35<00:08, 9128.52it/s] 81%|████████  | 322062/400000 [00:36<00:08, 9072.07it/s] 81%|████████  | 322971/400000 [00:36<00:08, 8982.75it/s] 81%|████████  | 323871/400000 [00:36<00:08, 8966.95it/s] 81%|████████  | 324822/400000 [00:36<00:08, 9121.00it/s] 81%|████████▏ | 325764/400000 [00:36<00:08, 9205.98it/s] 82%|████████▏ | 326686/400000 [00:36<00:07, 9173.75it/s] 82%|████████▏ | 327667/400000 [00:36<00:07, 9355.37it/s] 82%|████████▏ | 328604/400000 [00:36<00:07, 9308.88it/s] 82%|████████▏ | 329573/400000 [00:36<00:07, 9418.69it/s] 83%|████████▎ | 330524/400000 [00:36<00:07, 9444.10it/s] 83%|████████▎ | 331485/400000 [00:37<00:07, 9491.39it/s] 83%|████████▎ | 332441/400000 [00:37<00:07, 9508.93it/s] 83%|████████▎ | 333393/400000 [00:37<00:07, 9376.69it/s] 84%|████████▎ | 334351/400000 [00:37<00:06, 9434.87it/s] 84%|████████▍ | 335296/400000 [00:37<00:06, 9371.67it/s] 84%|████████▍ | 336245/400000 [00:37<00:06, 9406.32it/s] 84%|████████▍ | 337227/400000 [00:37<00:06, 9523.67it/s] 85%|████████▍ | 338180/400000 [00:37<00:06, 9140.88it/s] 85%|████████▍ | 339098/400000 [00:37<00:06, 8929.20it/s] 85%|████████▍ | 339995/400000 [00:38<00:06, 8799.69it/s] 85%|████████▌ | 340957/400000 [00:38<00:06, 9030.00it/s] 85%|████████▌ | 341904/400000 [00:38<00:06, 9153.16it/s] 86%|████████▌ | 342823/400000 [00:38<00:06, 8936.26it/s] 86%|████████▌ | 343776/400000 [00:38<00:06, 9104.76it/s] 86%|████████▌ | 344690/400000 [00:38<00:06, 9095.69it/s] 86%|████████▋ | 345602/400000 [00:38<00:06, 8878.30it/s] 87%|████████▋ | 346496/400000 [00:38<00:06, 8896.45it/s] 87%|████████▋ | 347388/400000 [00:38<00:06, 8763.32it/s] 87%|████████▋ | 348335/400000 [00:38<00:05, 8963.23it/s] 87%|████████▋ | 349306/400000 [00:39<00:05, 9172.60it/s] 88%|████████▊ | 350227/400000 [00:39<00:05, 9087.60it/s] 88%|████████▊ | 351138/400000 [00:39<00:05, 8763.08it/s] 88%|████████▊ | 352020/400000 [00:39<00:05, 8777.73it/s] 88%|████████▊ | 352902/400000 [00:39<00:05, 8788.05it/s] 88%|████████▊ | 353793/400000 [00:39<00:05, 8822.33it/s] 89%|████████▊ | 354731/400000 [00:39<00:05, 8982.36it/s] 89%|████████▉ | 355683/400000 [00:39<00:04, 9135.46it/s] 89%|████████▉ | 356599/400000 [00:39<00:04, 8927.43it/s] 89%|████████▉ | 357513/400000 [00:39<00:04, 8988.56it/s] 90%|████████▉ | 358447/400000 [00:40<00:04, 9089.37it/s] 90%|████████▉ | 359416/400000 [00:40<00:04, 9259.83it/s] 90%|█████████ | 360344/400000 [00:40<00:04, 9154.39it/s] 90%|█████████ | 361261/400000 [00:40<00:04, 9120.16it/s] 91%|█████████ | 362248/400000 [00:40<00:04, 9332.80it/s] 91%|█████████ | 363184/400000 [00:40<00:03, 9300.56it/s] 91%|█████████ | 364149/400000 [00:40<00:03, 9400.58it/s] 91%|█████████▏| 365094/400000 [00:40<00:03, 9412.07it/s] 92%|█████████▏| 366037/400000 [00:40<00:03, 9213.84it/s] 92%|█████████▏| 367011/400000 [00:40<00:03, 9364.72it/s] 92%|█████████▏| 367971/400000 [00:41<00:03, 9433.35it/s] 92%|█████████▏| 368916/400000 [00:41<00:03, 9293.39it/s] 92%|█████████▏| 369852/400000 [00:41<00:03, 9312.43it/s] 93%|█████████▎| 370785/400000 [00:41<00:03, 9174.68it/s] 93%|█████████▎| 371779/400000 [00:41<00:03, 9391.33it/s] 93%|█████████▎| 372721/400000 [00:41<00:02, 9306.84it/s] 93%|█████████▎| 373654/400000 [00:41<00:02, 9055.78it/s] 94%|█████████▎| 374563/400000 [00:41<00:02, 8915.70it/s] 94%|█████████▍| 375457/400000 [00:41<00:02, 8918.68it/s] 94%|█████████▍| 376447/400000 [00:42<00:02, 9191.40it/s] 94%|█████████▍| 377370/400000 [00:42<00:02, 8431.93it/s] 95%|█████████▍| 378327/400000 [00:42<00:02, 8743.20it/s] 95%|█████████▍| 379268/400000 [00:42<00:02, 8931.70it/s] 95%|█████████▌| 380175/400000 [00:42<00:02, 8970.39it/s] 95%|█████████▌| 381127/400000 [00:42<00:02, 9126.65it/s] 96%|█████████▌| 382109/400000 [00:42<00:01, 9323.84it/s] 96%|█████████▌| 383061/400000 [00:42<00:01, 9379.38it/s] 96%|█████████▌| 384003/400000 [00:42<00:01, 9269.13it/s] 96%|█████████▌| 384933/400000 [00:42<00:01, 9081.83it/s] 96%|█████████▋| 385866/400000 [00:43<00:01, 9153.41it/s] 97%|█████████▋| 386799/400000 [00:43<00:01, 9203.07it/s] 97%|█████████▋| 387791/400000 [00:43<00:01, 9406.62it/s] 97%|█████████▋| 388754/400000 [00:43<00:01, 9472.34it/s] 97%|█████████▋| 389703/400000 [00:43<00:01, 9294.91it/s] 98%|█████████▊| 390635/400000 [00:43<00:01, 9248.48it/s] 98%|█████████▊| 391588/400000 [00:43<00:00, 9329.35it/s] 98%|█████████▊| 392523/400000 [00:43<00:00, 9314.33it/s] 98%|█████████▊| 393491/400000 [00:43<00:00, 9420.96it/s] 99%|█████████▊| 394434/400000 [00:43<00:00, 9240.66it/s] 99%|█████████▉| 395414/400000 [00:44<00:00, 9400.20it/s] 99%|█████████▉| 396382/400000 [00:44<00:00, 9480.64it/s] 99%|█████████▉| 397368/400000 [00:44<00:00, 9590.92it/s]100%|█████████▉| 398329/400000 [00:44<00:00, 9530.58it/s]100%|█████████▉| 399283/400000 [00:44<00:00, 9116.57it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8971.47it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fd9cf6440f0>, <torchtext.data.dataset.TabularDataset object at 0x7fd9cf644240>, <torchtext.vocab.Vocab object at 0x7fd9cf644160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.06 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.45 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.45 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.29 file/s]2020-06-10 06:19:07.953081: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-10 06:19:07.956583: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-10 06:19:07.956814: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c6bcd09560 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-10 06:19:07.956861: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'test', 'mnist2', 'cifar10', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 155605.95it/s] 38%|███▊      | 3735552/9912422 [00:00<00:27, 221888.98it/s]9920512it [00:00, 33470005.57it/s]                           
0it [00:00, ?it/s]32768it [00:00, 489702.60it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 137979.00it/s]1654784it [00:00, 10263691.10it/s]                         
0it [00:00, ?it/s]8192it [00:00, 135954.33it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
