
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7d44f3f620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7d44f3f620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7db05060d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7db05060d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f7dca233ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f7dca233ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d5dd81950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d5dd81950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d5dd81950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 148571.09it/s] 79%|███████▉  | 7864320/9912422 [00:00<00:09, 212071.07it/s]9920512it [00:00, 42332125.55it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 267064.66it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8357617.65it/s]          
0it [00:00, ?it/s]8192it [00:00, 217027.15it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7d5b398a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7d5b38dd30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7d5dd81598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7d5dd81598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f7d5dd81598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:56,  2.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:56,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:56,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:55,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:55,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:55,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:54,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:38,  3.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:38,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:38,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:38,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:37,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:37,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:37,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:26,  5.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:25,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 10.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:08, 14.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 14.06 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:00<00:06, 18.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 18.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 23.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 28.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 35.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 41.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 47.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 53.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 56.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 59.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 59.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.99 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 78.13 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 78.13 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 29.89 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.42s/ url]
0 examples [00:00, ? examples/s]2020-06-09 00:09:29.689804: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-09 00:09:29.702549: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394445000 Hz
2020-06-09 00:09:29.702862: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5560f721cb70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-09 00:09:29.703010: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
19 examples [00:00, 187.79 examples/s]87 examples [00:00, 239.84 examples/s]174 examples [00:00, 306.38 examples/s]259 examples [00:00, 378.74 examples/s]343 examples [00:00, 452.81 examples/s]427 examples [00:00, 524.39 examples/s]512 examples [00:00, 592.01 examples/s]594 examples [00:00, 644.69 examples/s]678 examples [00:00, 691.53 examples/s]766 examples [00:01, 737.98 examples/s]848 examples [00:01, 758.08 examples/s]933 examples [00:01, 782.00 examples/s]1023 examples [00:01, 813.22 examples/s]1108 examples [00:01, 822.97 examples/s]1193 examples [00:01, 809.78 examples/s]1277 examples [00:01, 818.43 examples/s]1364 examples [00:01, 833.18 examples/s]1454 examples [00:01, 849.56 examples/s]1545 examples [00:01, 864.80 examples/s]1633 examples [00:02, 867.14 examples/s]1721 examples [00:02, 854.90 examples/s]1807 examples [00:02, 778.72 examples/s]1894 examples [00:02, 802.91 examples/s]1980 examples [00:02, 818.17 examples/s]2072 examples [00:02, 845.60 examples/s]2165 examples [00:02, 868.46 examples/s]2259 examples [00:02, 887.97 examples/s]2352 examples [00:02, 898.07 examples/s]2443 examples [00:02, 894.67 examples/s]2533 examples [00:03, 888.31 examples/s]2623 examples [00:03, 836.58 examples/s]2710 examples [00:03, 845.49 examples/s]2802 examples [00:03, 862.48 examples/s]2894 examples [00:03, 877.22 examples/s]2983 examples [00:03, 862.48 examples/s]3070 examples [00:03, 860.82 examples/s]3157 examples [00:03, 863.03 examples/s]3247 examples [00:03, 872.13 examples/s]3335 examples [00:04, 868.65 examples/s]3423 examples [00:04, 868.67 examples/s]3510 examples [00:04, 861.99 examples/s]3597 examples [00:04, 829.27 examples/s]3681 examples [00:04, 830.58 examples/s]3766 examples [00:04, 824.37 examples/s]3858 examples [00:04, 850.64 examples/s]3945 examples [00:04, 854.26 examples/s]4031 examples [00:04, 852.91 examples/s]4118 examples [00:04, 857.83 examples/s]4204 examples [00:05, 846.09 examples/s]4289 examples [00:05, 833.60 examples/s]4378 examples [00:05, 848.55 examples/s]4464 examples [00:05, 851.92 examples/s]4550 examples [00:05, 840.32 examples/s]4636 examples [00:05, 844.67 examples/s]4725 examples [00:05, 857.63 examples/s]4813 examples [00:05, 863.68 examples/s]4903 examples [00:05, 872.28 examples/s]4997 examples [00:05, 891.06 examples/s]5087 examples [00:06, 886.41 examples/s]5176 examples [00:06, 856.43 examples/s]5262 examples [00:06, 800.08 examples/s]5347 examples [00:06, 812.40 examples/s]5429 examples [00:06, 806.24 examples/s]5511 examples [00:06, 795.32 examples/s]5591 examples [00:06, 790.17 examples/s]5680 examples [00:06, 816.09 examples/s]5763 examples [00:06, 809.67 examples/s]5845 examples [00:07, 807.84 examples/s]5941 examples [00:07, 848.16 examples/s]6036 examples [00:07, 874.11 examples/s]6125 examples [00:07, 872.04 examples/s]6213 examples [00:07, 874.13 examples/s]6307 examples [00:07, 891.27 examples/s]6397 examples [00:07, 876.76 examples/s]6493 examples [00:07, 897.62 examples/s]6585 examples [00:07, 903.03 examples/s]6678 examples [00:07, 910.51 examples/s]6770 examples [00:08, 883.44 examples/s]6862 examples [00:08, 892.24 examples/s]6952 examples [00:08, 854.31 examples/s]7038 examples [00:08, 826.69 examples/s]7126 examples [00:08, 841.16 examples/s]7215 examples [00:08, 852.85 examples/s]7301 examples [00:08, 821.69 examples/s]7387 examples [00:08, 831.14 examples/s]7471 examples [00:08, 818.89 examples/s]7554 examples [00:08, 804.96 examples/s]7641 examples [00:09, 822.81 examples/s]7724 examples [00:09, 821.51 examples/s]7809 examples [00:09, 826.93 examples/s]7892 examples [00:09, 826.57 examples/s]7988 examples [00:09, 859.94 examples/s]8078 examples [00:09, 870.42 examples/s]8166 examples [00:09, 860.52 examples/s]8253 examples [00:09, 842.08 examples/s]8338 examples [00:09, 822.38 examples/s]8424 examples [00:10, 831.48 examples/s]8512 examples [00:10, 844.27 examples/s]8600 examples [00:10, 851.17 examples/s]8686 examples [00:10, 832.26 examples/s]8771 examples [00:10, 835.94 examples/s]8858 examples [00:10, 845.41 examples/s]8943 examples [00:10, 831.82 examples/s]9031 examples [00:10, 845.02 examples/s]9117 examples [00:10, 848.39 examples/s]9202 examples [00:10, 839.82 examples/s]9287 examples [00:11, 841.34 examples/s]9373 examples [00:11, 844.36 examples/s]9458 examples [00:11, 820.43 examples/s]9541 examples [00:11, 806.47 examples/s]9631 examples [00:11, 831.29 examples/s]9720 examples [00:11, 847.82 examples/s]9806 examples [00:11, 847.86 examples/s]9892 examples [00:11, 848.76 examples/s]9978 examples [00:11, 844.58 examples/s]10063 examples [00:11, 775.80 examples/s]10153 examples [00:12, 806.89 examples/s]10248 examples [00:12, 843.61 examples/s]10334 examples [00:12, 820.87 examples/s]10418 examples [00:12, 825.99 examples/s]10510 examples [00:12, 849.96 examples/s]10597 examples [00:12, 844.61 examples/s]10682 examples [00:12, 804.26 examples/s]10769 examples [00:12, 819.48 examples/s]10853 examples [00:12, 825.20 examples/s]10941 examples [00:13, 840.84 examples/s]11030 examples [00:13, 853.73 examples/s]11119 examples [00:13, 862.03 examples/s]11207 examples [00:13, 866.52 examples/s]11294 examples [00:13, 857.51 examples/s]11380 examples [00:13, 858.14 examples/s]11466 examples [00:13, 833.07 examples/s]11550 examples [00:13, 821.31 examples/s]11633 examples [00:13, 822.32 examples/s]11728 examples [00:13, 856.02 examples/s]11824 examples [00:14, 882.72 examples/s]11920 examples [00:14, 902.33 examples/s]12014 examples [00:14, 912.75 examples/s]12106 examples [00:14, 887.10 examples/s]12196 examples [00:14, 852.41 examples/s]12282 examples [00:14, 848.17 examples/s]12368 examples [00:14, 846.58 examples/s]12464 examples [00:14, 875.73 examples/s]12558 examples [00:14, 892.67 examples/s]12648 examples [00:14, 892.85 examples/s]12738 examples [00:15, 890.35 examples/s]12828 examples [00:15, 887.39 examples/s]12917 examples [00:15, 864.98 examples/s]13004 examples [00:15, 834.17 examples/s]13096 examples [00:15, 856.66 examples/s]13191 examples [00:15, 881.13 examples/s]13286 examples [00:15, 898.43 examples/s]13384 examples [00:15, 918.93 examples/s]13477 examples [00:15, 910.89 examples/s]13569 examples [00:16, 893.22 examples/s]13659 examples [00:16, 892.40 examples/s]13749 examples [00:16, 879.14 examples/s]13838 examples [00:16, 833.31 examples/s]13922 examples [00:16, 805.50 examples/s]14011 examples [00:16, 827.16 examples/s]14100 examples [00:16, 844.12 examples/s]14189 examples [00:16, 856.02 examples/s]14275 examples [00:16, 834.68 examples/s]14359 examples [00:16, 832.12 examples/s]14443 examples [00:17, 833.23 examples/s]14534 examples [00:17, 854.41 examples/s]14628 examples [00:17, 877.09 examples/s]14726 examples [00:17, 902.21 examples/s]14821 examples [00:17, 915.44 examples/s]14918 examples [00:17, 930.83 examples/s]15013 examples [00:17, 935.75 examples/s]15107 examples [00:17, 923.43 examples/s]15200 examples [00:17, 911.73 examples/s]15292 examples [00:18, 884.98 examples/s]15381 examples [00:18, 872.53 examples/s]15470 examples [00:18, 874.52 examples/s]15558 examples [00:18, 810.62 examples/s]15641 examples [00:18, 816.02 examples/s]15724 examples [00:18, 799.11 examples/s]15805 examples [00:18, 796.71 examples/s]15886 examples [00:18, 799.72 examples/s]15974 examples [00:18, 821.81 examples/s]16063 examples [00:18, 838.95 examples/s]16148 examples [00:19, 830.38 examples/s]16232 examples [00:19, 825.36 examples/s]16317 examples [00:19, 831.40 examples/s]16402 examples [00:19, 834.64 examples/s]16486 examples [00:19, 827.77 examples/s]16574 examples [00:19, 841.49 examples/s]16669 examples [00:19, 869.63 examples/s]16766 examples [00:19, 897.47 examples/s]16858 examples [00:19, 900.77 examples/s]16949 examples [00:19, 841.75 examples/s]17035 examples [00:20, 845.16 examples/s]17121 examples [00:20, 826.58 examples/s]17205 examples [00:20, 813.05 examples/s]17287 examples [00:20, 804.87 examples/s]17369 examples [00:20, 804.57 examples/s]17453 examples [00:20, 813.88 examples/s]17535 examples [00:20, 800.95 examples/s]17629 examples [00:20, 836.88 examples/s]17722 examples [00:20, 861.13 examples/s]17809 examples [00:21, 838.27 examples/s]17900 examples [00:21, 856.70 examples/s]17991 examples [00:21, 871.10 examples/s]18079 examples [00:21, 863.98 examples/s]18166 examples [00:21, 845.19 examples/s]18251 examples [00:21, 825.45 examples/s]18334 examples [00:21, 812.96 examples/s]18420 examples [00:21, 823.83 examples/s]18517 examples [00:21, 861.48 examples/s]18604 examples [00:21, 850.66 examples/s]18690 examples [00:22, 849.30 examples/s]18777 examples [00:22, 855.17 examples/s]18863 examples [00:22, 841.03 examples/s]18948 examples [00:22, 794.15 examples/s]19041 examples [00:22, 829.08 examples/s]19133 examples [00:22, 852.84 examples/s]19220 examples [00:22, 851.22 examples/s]19306 examples [00:22, 849.56 examples/s]19400 examples [00:22, 870.92 examples/s]19488 examples [00:23, 872.49 examples/s]19583 examples [00:23, 894.34 examples/s]19673 examples [00:23, 881.82 examples/s]19764 examples [00:23, 888.60 examples/s]19854 examples [00:23, 873.65 examples/s]19942 examples [00:23, 846.04 examples/s]20027 examples [00:23, 800.09 examples/s]20122 examples [00:23, 839.41 examples/s]20207 examples [00:23, 833.40 examples/s]20295 examples [00:23, 846.33 examples/s]20381 examples [00:24, 841.52 examples/s]20466 examples [00:24, 829.28 examples/s]20561 examples [00:24, 860.06 examples/s]20648 examples [00:24, 850.51 examples/s]20736 examples [00:24, 856.47 examples/s]20822 examples [00:24, 854.02 examples/s]20908 examples [00:24, 851.54 examples/s]20994 examples [00:24, 836.37 examples/s]21085 examples [00:24, 856.81 examples/s]21180 examples [00:24, 881.89 examples/s]21269 examples [00:25, 856.86 examples/s]21356 examples [00:25, 841.52 examples/s]21443 examples [00:25, 849.63 examples/s]21529 examples [00:25, 846.01 examples/s]21615 examples [00:25, 849.24 examples/s]21704 examples [00:25, 857.86 examples/s]21790 examples [00:25, 844.06 examples/s]21875 examples [00:25, 831.39 examples/s]21959 examples [00:25, 823.37 examples/s]22042 examples [00:26, 805.91 examples/s]22123 examples [00:26, 792.80 examples/s]22203 examples [00:26, 794.86 examples/s]22283 examples [00:26, 778.55 examples/s]22366 examples [00:26, 791.10 examples/s]22452 examples [00:26, 809.96 examples/s]22537 examples [00:26, 820.87 examples/s]22623 examples [00:26, 830.13 examples/s]22707 examples [00:26, 809.07 examples/s]22789 examples [00:26, 791.00 examples/s]22882 examples [00:27, 826.13 examples/s]22972 examples [00:27, 844.12 examples/s]23069 examples [00:27, 877.02 examples/s]23161 examples [00:27, 887.49 examples/s]23251 examples [00:27, 864.84 examples/s]23338 examples [00:27, 864.65 examples/s]23425 examples [00:27, 833.04 examples/s]23509 examples [00:27, 827.59 examples/s]23596 examples [00:27, 837.62 examples/s]23685 examples [00:27, 852.48 examples/s]23779 examples [00:28, 876.11 examples/s]23867 examples [00:28, 870.55 examples/s]23962 examples [00:28, 892.19 examples/s]24052 examples [00:28, 850.89 examples/s]24144 examples [00:28, 870.38 examples/s]24235 examples [00:28, 881.70 examples/s]24324 examples [00:28, 877.12 examples/s]24413 examples [00:28, 868.03 examples/s]24501 examples [00:28, 870.96 examples/s]24591 examples [00:29, 879.35 examples/s]24680 examples [00:29, 879.71 examples/s]24769 examples [00:29, 823.97 examples/s]24853 examples [00:29, 828.30 examples/s]24939 examples [00:29, 837.04 examples/s]25027 examples [00:29, 848.58 examples/s]25113 examples [00:29, 846.24 examples/s]25198 examples [00:29, 844.35 examples/s]25292 examples [00:29, 867.24 examples/s]25380 examples [00:29, 869.54 examples/s]25468 examples [00:30, 852.34 examples/s]25554 examples [00:30, 823.16 examples/s]25637 examples [00:30, 811.35 examples/s]25719 examples [00:30, 806.13 examples/s]25807 examples [00:30, 826.44 examples/s]25898 examples [00:30, 847.92 examples/s]25984 examples [00:30, 843.17 examples/s]26069 examples [00:30, 820.56 examples/s]26154 examples [00:30, 827.60 examples/s]26242 examples [00:31, 842.60 examples/s]26329 examples [00:31, 850.00 examples/s]26416 examples [00:31, 854.68 examples/s]26512 examples [00:31, 882.51 examples/s]26601 examples [00:31, 870.25 examples/s]26696 examples [00:31, 892.24 examples/s]26786 examples [00:31, 891.26 examples/s]26877 examples [00:31, 895.56 examples/s]26967 examples [00:31, 840.18 examples/s]27052 examples [00:31, 811.36 examples/s]27135 examples [00:32, 814.48 examples/s]27222 examples [00:32, 827.78 examples/s]27306 examples [00:32, 819.97 examples/s]27389 examples [00:32, 797.76 examples/s]27480 examples [00:32, 827.72 examples/s]27567 examples [00:32, 839.30 examples/s]27653 examples [00:32, 844.14 examples/s]27738 examples [00:32, 833.24 examples/s]27822 examples [00:32, 799.75 examples/s]27912 examples [00:32, 827.25 examples/s]27996 examples [00:33, 823.23 examples/s]28080 examples [00:33, 826.93 examples/s]28167 examples [00:33, 837.70 examples/s]28263 examples [00:33, 868.73 examples/s]28356 examples [00:33, 883.99 examples/s]28445 examples [00:33, 879.59 examples/s]28534 examples [00:33, 860.53 examples/s]28621 examples [00:33, 846.36 examples/s]28708 examples [00:33, 851.04 examples/s]28796 examples [00:34, 858.76 examples/s]28883 examples [00:34, 834.89 examples/s]28967 examples [00:34, 835.61 examples/s]29056 examples [00:34, 849.78 examples/s]29142 examples [00:34, 770.28 examples/s]29223 examples [00:34, 779.63 examples/s]29309 examples [00:34, 799.54 examples/s]29390 examples [00:34, 793.24 examples/s]29481 examples [00:34, 823.43 examples/s]29578 examples [00:34, 861.20 examples/s]29667 examples [00:35, 869.60 examples/s]29755 examples [00:35, 850.96 examples/s]29841 examples [00:35, 833.65 examples/s]29930 examples [00:35, 848.44 examples/s]30016 examples [00:35, 790.50 examples/s]30097 examples [00:35, 793.17 examples/s]30178 examples [00:35, 767.86 examples/s]30261 examples [00:35, 783.41 examples/s]30342 examples [00:35, 790.56 examples/s]30425 examples [00:36, 801.57 examples/s]30510 examples [00:36, 813.31 examples/s]30592 examples [00:36, 801.32 examples/s]30679 examples [00:36, 820.39 examples/s]30762 examples [00:36, 786.24 examples/s]30847 examples [00:36, 803.14 examples/s]30931 examples [00:36, 811.76 examples/s]31015 examples [00:36, 819.34 examples/s]31098 examples [00:36, 795.28 examples/s]31180 examples [00:36, 802.48 examples/s]31273 examples [00:37, 835.12 examples/s]31358 examples [00:37, 805.95 examples/s]31444 examples [00:37, 818.26 examples/s]31530 examples [00:37, 829.39 examples/s]31616 examples [00:37, 837.14 examples/s]31705 examples [00:37, 849.65 examples/s]31799 examples [00:37, 872.69 examples/s]31891 examples [00:37, 885.55 examples/s]31987 examples [00:37, 904.83 examples/s]32078 examples [00:37, 891.76 examples/s]32168 examples [00:38, 890.15 examples/s]32258 examples [00:38, 874.88 examples/s]32346 examples [00:38, 817.16 examples/s]32429 examples [00:38, 766.01 examples/s]32515 examples [00:38, 791.41 examples/s]32611 examples [00:38, 835.25 examples/s]32707 examples [00:38, 869.03 examples/s]32800 examples [00:38, 885.00 examples/s]32897 examples [00:38, 906.35 examples/s]32992 examples [00:39, 916.43 examples/s]33085 examples [00:39, 911.24 examples/s]33177 examples [00:39, 891.16 examples/s]33267 examples [00:39, 881.99 examples/s]33356 examples [00:39, 853.40 examples/s]33444 examples [00:39, 860.94 examples/s]33533 examples [00:39, 868.67 examples/s]33621 examples [00:39, 862.07 examples/s]33708 examples [00:39, 841.71 examples/s]33794 examples [00:39, 845.83 examples/s]33886 examples [00:40, 864.49 examples/s]33974 examples [00:40, 866.39 examples/s]34061 examples [00:40, 860.65 examples/s]34148 examples [00:40, 843.64 examples/s]34233 examples [00:40, 830.20 examples/s]34321 examples [00:40, 844.22 examples/s]34408 examples [00:40, 849.50 examples/s]34494 examples [00:40, 835.14 examples/s]34578 examples [00:40, 834.36 examples/s]34667 examples [00:41, 848.76 examples/s]34753 examples [00:41, 841.36 examples/s]34845 examples [00:41, 861.81 examples/s]34932 examples [00:41, 859.97 examples/s]35019 examples [00:41, 840.86 examples/s]35104 examples [00:41, 835.15 examples/s]35188 examples [00:41, 825.91 examples/s]35271 examples [00:41, 798.88 examples/s]35352 examples [00:41, 802.17 examples/s]35439 examples [00:41, 819.97 examples/s]35522 examples [00:42, 822.37 examples/s]35609 examples [00:42, 835.44 examples/s]35698 examples [00:42, 849.85 examples/s]35784 examples [00:42, 818.82 examples/s]35867 examples [00:42, 755.95 examples/s]35951 examples [00:42, 777.42 examples/s]36037 examples [00:42, 799.14 examples/s]36130 examples [00:42, 834.16 examples/s]36221 examples [00:42, 855.50 examples/s]36308 examples [00:43, 849.97 examples/s]36397 examples [00:43, 860.96 examples/s]36487 examples [00:43, 870.82 examples/s]36575 examples [00:43, 867.97 examples/s]36663 examples [00:43, 852.47 examples/s]36749 examples [00:43, 807.74 examples/s]36833 examples [00:43, 816.08 examples/s]36923 examples [00:43, 837.02 examples/s]37011 examples [00:43, 847.81 examples/s]37097 examples [00:43, 847.40 examples/s]37182 examples [00:44, 824.87 examples/s]37265 examples [00:44, 814.59 examples/s]37353 examples [00:44, 830.76 examples/s]37439 examples [00:44, 837.39 examples/s]37523 examples [00:44, 801.73 examples/s]37619 examples [00:44, 842.60 examples/s]37710 examples [00:44, 861.22 examples/s]37799 examples [00:44, 869.45 examples/s]37887 examples [00:44, 860.99 examples/s]37977 examples [00:44, 871.73 examples/s]38065 examples [00:45, 846.97 examples/s]38151 examples [00:45, 849.97 examples/s]38238 examples [00:45, 854.78 examples/s]38328 examples [00:45, 862.07 examples/s]38415 examples [00:45, 854.57 examples/s]38502 examples [00:45, 858.99 examples/s]38595 examples [00:45, 879.12 examples/s]38690 examples [00:45, 898.54 examples/s]38781 examples [00:45, 872.46 examples/s]38871 examples [00:46, 879.29 examples/s]38964 examples [00:46, 892.83 examples/s]39055 examples [00:46, 896.22 examples/s]39145 examples [00:46, 849.06 examples/s]39231 examples [00:46, 802.00 examples/s]39325 examples [00:46, 838.61 examples/s]39416 examples [00:46, 857.58 examples/s]39503 examples [00:46, 855.01 examples/s]39590 examples [00:46, 845.64 examples/s]39676 examples [00:46, 833.16 examples/s]39760 examples [00:47, 831.15 examples/s]39849 examples [00:47, 846.48 examples/s]39937 examples [00:47, 856.21 examples/s]40023 examples [00:47, 822.13 examples/s]40106 examples [00:47, 796.13 examples/s]40187 examples [00:47, 774.96 examples/s]40265 examples [00:47, 776.32 examples/s]40353 examples [00:47, 803.94 examples/s]40440 examples [00:47, 822.03 examples/s]40525 examples [00:48, 828.26 examples/s]40612 examples [00:48, 838.33 examples/s]40702 examples [00:48, 855.60 examples/s]40788 examples [00:48, 828.35 examples/s]40872 examples [00:48, 806.17 examples/s]40962 examples [00:48, 831.00 examples/s]41057 examples [00:48, 862.83 examples/s]41149 examples [00:48, 878.97 examples/s]41238 examples [00:48, 874.39 examples/s]41326 examples [00:48, 867.87 examples/s]41424 examples [00:49, 897.43 examples/s]41515 examples [00:49, 882.84 examples/s]41611 examples [00:49, 904.53 examples/s]41705 examples [00:49, 913.60 examples/s]41797 examples [00:49, 903.21 examples/s]41891 examples [00:49, 913.09 examples/s]41987 examples [00:49, 924.85 examples/s]42081 examples [00:49, 928.90 examples/s]42175 examples [00:49, 909.25 examples/s]42267 examples [00:49, 893.30 examples/s]42359 examples [00:50, 897.54 examples/s]42453 examples [00:50, 907.47 examples/s]42550 examples [00:50, 925.23 examples/s]42643 examples [00:50, 895.29 examples/s]42733 examples [00:50, 806.06 examples/s]42816 examples [00:50, 791.03 examples/s]42900 examples [00:50, 801.88 examples/s]42988 examples [00:50, 822.11 examples/s]43078 examples [00:50, 842.03 examples/s]43174 examples [00:51, 873.84 examples/s]43265 examples [00:51, 882.01 examples/s]43354 examples [00:51, 866.21 examples/s]43443 examples [00:51, 871.42 examples/s]43531 examples [00:51, 821.78 examples/s]43626 examples [00:51, 856.06 examples/s]43717 examples [00:51, 871.08 examples/s]43805 examples [00:51, 862.30 examples/s]43892 examples [00:51, 850.44 examples/s]43979 examples [00:51, 855.35 examples/s]44069 examples [00:52, 866.86 examples/s]44165 examples [00:52, 891.41 examples/s]44255 examples [00:52, 885.73 examples/s]44344 examples [00:52, 861.25 examples/s]44431 examples [00:52, 821.66 examples/s]44515 examples [00:52, 825.49 examples/s]44598 examples [00:52, 811.78 examples/s]44680 examples [00:52, 779.24 examples/s]44766 examples [00:52, 799.69 examples/s]44853 examples [00:53, 818.10 examples/s]44943 examples [00:53, 839.08 examples/s]45028 examples [00:53, 839.56 examples/s]45113 examples [00:53, 837.68 examples/s]45199 examples [00:53, 843.17 examples/s]45285 examples [00:53, 846.60 examples/s]45370 examples [00:53, 833.22 examples/s]45458 examples [00:53, 845.94 examples/s]45543 examples [00:53, 838.84 examples/s]45627 examples [00:53, 823.38 examples/s]45713 examples [00:54, 834.03 examples/s]45805 examples [00:54, 855.87 examples/s]45895 examples [00:54, 866.18 examples/s]45983 examples [00:54, 867.73 examples/s]46070 examples [00:54, 813.31 examples/s]46160 examples [00:54, 836.29 examples/s]46257 examples [00:54, 870.33 examples/s]46351 examples [00:54, 889.46 examples/s]46446 examples [00:54, 903.88 examples/s]46537 examples [00:54, 894.53 examples/s]46627 examples [00:55, 884.54 examples/s]46716 examples [00:55, 875.82 examples/s]46804 examples [00:55, 857.21 examples/s]46895 examples [00:55, 871.19 examples/s]46988 examples [00:55, 887.61 examples/s]47079 examples [00:55, 892.79 examples/s]47169 examples [00:55, 862.83 examples/s]47263 examples [00:55, 883.28 examples/s]47357 examples [00:55, 899.36 examples/s]47449 examples [00:56, 903.85 examples/s]47540 examples [00:56, 887.15 examples/s]47629 examples [00:56, 879.77 examples/s]47718 examples [00:56, 874.37 examples/s]47806 examples [00:56, 842.96 examples/s]47891 examples [00:56, 794.52 examples/s]47972 examples [00:56, 774.22 examples/s]48058 examples [00:56, 797.05 examples/s]48146 examples [00:56, 817.45 examples/s]48229 examples [00:56, 817.40 examples/s]48312 examples [00:57, 807.83 examples/s]48406 examples [00:57, 841.87 examples/s]48498 examples [00:57, 862.68 examples/s]48585 examples [00:57, 853.93 examples/s]48672 examples [00:57, 858.67 examples/s]48759 examples [00:57, 851.96 examples/s]48845 examples [00:57, 841.51 examples/s]48930 examples [00:57, 824.45 examples/s]49013 examples [00:57, 819.12 examples/s]49096 examples [00:58, 794.48 examples/s]49185 examples [00:58, 819.47 examples/s]49271 examples [00:58, 830.13 examples/s]49355 examples [00:58, 799.00 examples/s]49445 examples [00:58, 826.50 examples/s]49529 examples [00:58, 814.11 examples/s]49613 examples [00:58, 820.93 examples/s]49696 examples [00:58, 822.90 examples/s]49783 examples [00:58, 835.37 examples/s]49872 examples [00:58, 849.75 examples/s]49962 examples [00:59, 862.22 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5187/50000 [00:00<00:00, 51869.18 examples/s] 31%|███       | 15309/50000 [00:00<00:00, 60738.04 examples/s] 54%|█████▍    | 26950/50000 [00:00<00:00, 70911.72 examples/s] 78%|███████▊  | 38928/50000 [00:00<00:00, 80800.38 examples/s]                                                               0 examples [00:00, ? examples/s]68 examples [00:00, 673.79 examples/s]143 examples [00:00, 693.67 examples/s]229 examples [00:00, 735.19 examples/s]315 examples [00:00, 768.60 examples/s]399 examples [00:00, 788.70 examples/s]496 examples [00:00, 834.91 examples/s]594 examples [00:00, 873.08 examples/s]688 examples [00:00, 891.60 examples/s]775 examples [00:01, 649.36 examples/s]863 examples [00:01, 704.66 examples/s]950 examples [00:01, 745.25 examples/s]1038 examples [00:01, 779.77 examples/s]1121 examples [00:01, 791.13 examples/s]1208 examples [00:01, 811.35 examples/s]1292 examples [00:01, 815.37 examples/s]1376 examples [00:01, 820.05 examples/s]1462 examples [00:01, 829.88 examples/s]1546 examples [00:01, 828.06 examples/s]1631 examples [00:02, 833.13 examples/s]1719 examples [00:02, 845.70 examples/s]1805 examples [00:02, 847.40 examples/s]1890 examples [00:02, 841.44 examples/s]1978 examples [00:02, 850.61 examples/s]2064 examples [00:02, 852.23 examples/s]2150 examples [00:02, 841.69 examples/s]2238 examples [00:02, 850.85 examples/s]2327 examples [00:02, 859.21 examples/s]2414 examples [00:02, 818.19 examples/s]2504 examples [00:03, 838.85 examples/s]2589 examples [00:03, 825.45 examples/s]2676 examples [00:03, 836.21 examples/s]2760 examples [00:03, 835.06 examples/s]2844 examples [00:03, 821.75 examples/s]2927 examples [00:03, 797.84 examples/s]3008 examples [00:03, 764.57 examples/s]3095 examples [00:03, 791.45 examples/s]3176 examples [00:03, 794.63 examples/s]3258 examples [00:04, 801.26 examples/s]3341 examples [00:04, 806.36 examples/s]3433 examples [00:04, 836.83 examples/s]3529 examples [00:04, 868.41 examples/s]3624 examples [00:04, 889.68 examples/s]3720 examples [00:04, 907.25 examples/s]3812 examples [00:04, 896.74 examples/s]3903 examples [00:04, 884.40 examples/s]3992 examples [00:04, 873.57 examples/s]4080 examples [00:04, 794.65 examples/s]4168 examples [00:05, 817.11 examples/s]4262 examples [00:05, 848.93 examples/s]4349 examples [00:05, 850.88 examples/s]4435 examples [00:05, 837.83 examples/s]4528 examples [00:05, 861.18 examples/s]4616 examples [00:05, 864.73 examples/s]4712 examples [00:05, 890.81 examples/s]4806 examples [00:05, 903.61 examples/s]4901 examples [00:05, 916.47 examples/s]4994 examples [00:05, 919.22 examples/s]5087 examples [00:06, 921.59 examples/s]5180 examples [00:06, 873.22 examples/s]5268 examples [00:06, 822.49 examples/s]5353 examples [00:06, 830.43 examples/s]5439 examples [00:06, 838.32 examples/s]5533 examples [00:06, 864.40 examples/s]5626 examples [00:06, 880.62 examples/s]5719 examples [00:06, 891.29 examples/s]5809 examples [00:06, 842.85 examples/s]5895 examples [00:07, 806.77 examples/s]5977 examples [00:07, 798.65 examples/s]6058 examples [00:07, 797.53 examples/s]6148 examples [00:07, 825.07 examples/s]6232 examples [00:07, 778.64 examples/s]6311 examples [00:07, 731.60 examples/s]6396 examples [00:07, 762.53 examples/s]6479 examples [00:07, 779.51 examples/s]6567 examples [00:07, 805.99 examples/s]6655 examples [00:08, 824.92 examples/s]6741 examples [00:08, 832.57 examples/s]6828 examples [00:08, 843.37 examples/s]6916 examples [00:08, 853.72 examples/s]7002 examples [00:08, 848.79 examples/s]7088 examples [00:08, 847.87 examples/s]7173 examples [00:08, 822.75 examples/s]7256 examples [00:08, 787.44 examples/s]7336 examples [00:08, 788.73 examples/s]7431 examples [00:08, 830.27 examples/s]7515 examples [00:09, 812.11 examples/s]7601 examples [00:09, 825.62 examples/s]7686 examples [00:09, 832.63 examples/s]7775 examples [00:09, 846.47 examples/s]7869 examples [00:09, 870.01 examples/s]7964 examples [00:09, 890.35 examples/s]8054 examples [00:09, 892.11 examples/s]8144 examples [00:09, 885.29 examples/s]8233 examples [00:09, 860.31 examples/s]8320 examples [00:09, 841.99 examples/s]8405 examples [00:10, 843.47 examples/s]8490 examples [00:10, 824.80 examples/s]8576 examples [00:10, 834.02 examples/s]8660 examples [00:10, 834.82 examples/s]8744 examples [00:10, 821.18 examples/s]8834 examples [00:10, 841.87 examples/s]8919 examples [00:10, 843.09 examples/s]9004 examples [00:10, 824.95 examples/s]9087 examples [00:10, 821.73 examples/s]9170 examples [00:11, 791.75 examples/s]9252 examples [00:11, 799.44 examples/s]9336 examples [00:11, 809.62 examples/s]9418 examples [00:11, 808.71 examples/s]9504 examples [00:11, 823.39 examples/s]9587 examples [00:11, 813.47 examples/s]9671 examples [00:11, 820.54 examples/s]9759 examples [00:11, 835.47 examples/s]9850 examples [00:11, 855.92 examples/s]9942 examples [00:11, 871.07 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 93%|█████████▎| 9257/10000 [00:00<00:00, 92145.42 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteE16HGB/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteE16HGB/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d5dd81950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d5dd81950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d5dd81950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7dc400fac8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7ce31ecf28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d5dd81950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d5dd81950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d5dd81950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7d441400f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7d441400b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7cd575e268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7cd575e268> 

  function with postional parmater data_info <function split_train_valid at 0x7f7cd575e268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7cd575e378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7cd575e378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7cd575e378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=8b497199b455315106c95e603e12ce8cea56db7a713a51e7a5774c44fb79a35a
  Stored in directory: /tmp/pip-ephem-wheel-cache-77o4rnnp/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:09<267:42:09, 895B/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:09<187:36:57, 1.28kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:09<131:21:37, 1.82kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:09<91:53:37, 2.60kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:09<64:07:26, 3.72kB/s].vector_cache/glove.6B.zip:   1%|          | 9.31M/862M [00:09<44:35:33, 5.31kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:09<31:00:31, 7.59kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:10<21:33:40, 10.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:10<14:59:26, 15.5kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:10<10:25:26, 22.1kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:10<7:14:55, 31.6kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:10<5:02:26, 45.1kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.2M/862M [00:10<3:30:20, 64.4kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:11<2:27:17, 91.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:13<1:44:30, 129kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:13<1:16:45, 175kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:13<54:37, 246kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:13<38:19, 349kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:15<32:59, 405kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:15<26:10, 510kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:15<18:59, 703kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:15<13:27, 990kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:17<14:30, 917kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:17<11:19, 1.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.3M/862M [00:17<08:16, 1.60MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:19<08:43, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:19<08:46, 1.51MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:19<06:47, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:21<06:53, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:21<05:58, 2.20MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:21<04:31, 2.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:23<06:10, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:23<06:58, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:23<05:26, 2.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:23<03:59, 3.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:25<09:15, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:25<07:47, 1.67MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:25<05:46, 2.25MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:27<07:05, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:27<07:35, 1.71MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:27<05:52, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:27<04:14, 3.04MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:29<18:48, 685kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:29<14:28, 890kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:29<10:26, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:31<10:17, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:31<08:29, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:31<06:15, 2.04MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:33<07:23, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:33<06:27, 1.97MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:33<04:50, 2.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:34<06:23, 1.98MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:35<07:02, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:35<05:33, 2.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:36<05:55, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:37<06:48, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:37<05:24, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:37<03:55, 3.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:38<48:08, 260kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:39<34:58, 358kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:39<24:45, 505kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<20:12, 617kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:40<16:40, 748kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:41<12:13, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:41<08:39, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<24:23, 509kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:42<18:19, 677kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:43<13:07, 943kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<12:03, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:44<10:57, 1.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:45<08:14, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:45<05:54, 2.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:46<1:45:20, 116kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:46<1:15:43, 162kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:47<53:52, 227kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:47<39:58, 306kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:47<29:25, 416kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:47<21:21, 573kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:47<16:25, 744kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:47<12:09, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:47<09:47, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<07:27, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:47<06:14, 1.95MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:48<25:34, 477kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:48<19:36, 622kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:49<14:51, 820kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:49<11:09, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:49<08:31, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:49<07:02, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:49<04:43, 2.57MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:50<42:29, 285kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:50<33:08, 366kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:50<24:37, 492kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:51<17:55, 676kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:51<13:20, 907kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:51<09:52, 1.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:51<07:34, 1.59MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:51<06:02, 2.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:52<09:49, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:52<09:17, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:52<07:27, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:53<05:49, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:53<04:31, 2.66MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:53<04:08, 2.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:54<08:08, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:54<10:16, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:54<08:22, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:55<06:52, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:55<05:12, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:55<04:27, 2.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:55<03:37, 3.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:56<07:09, 1.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:56<10:00, 1.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:56<09:06, 1.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:56<07:04, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:57<05:18, 2.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<04:30, 2.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:57<03:51, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:58<07:23, 1.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:58<07:07, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:58<05:26, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:58<04:47, 2.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<03:42, 3.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:59<03:11, 3.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:59<02:52, 4.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:00<20:34, 573kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [01:00<18:43, 629kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [01:00<14:04, 836kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [01:00<10:21, 1.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [01:01<07:39, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:01<05:43, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:02<58:23, 201kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:02<44:39, 262kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:02<32:11, 364kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [01:02<22:56, 509kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [01:03<16:25, 710kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:03<11:50, 983kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:04<4:47:48, 40.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:04<3:24:58, 56.8kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:04<2:24:08, 80.7kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:04<1:41:15, 115kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:04<1:11:05, 163kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:05<49:56, 232kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:06<3:48:25, 50.7kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:06<2:46:55, 69.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:06<1:58:37, 97.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:06<1:23:27, 138kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:06<58:44, 196kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:07<41:21, 278kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:08<36:53, 312kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:08<29:35, 389kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:08<21:35, 532kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:08<15:28, 741kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:08<11:09, 1.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:10<12:45, 897kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:10<12:13, 935kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:10<09:14, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:10<06:50, 1.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:10<05:06, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:10<03:50, 2.96MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:12<23:54, 476kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:12<22:14, 511kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:12<17:03, 666kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:12<12:20, 919kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:12<08:51, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:14<11:00, 1.03MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:14<12:41, 890kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:14<09:59, 1.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:14<07:22, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:14<05:23, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:16<08:10, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:16<10:40, 1.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:16<08:39, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:16<06:21, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:16<04:39, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<15:57, 699kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<15:21, 727kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<11:42, 953kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:18<08:28, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<08:25, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<09:44, 1.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<07:46, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:20<05:40, 1.95MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:20<04:09, 2.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<16:42, 660kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<15:16, 722kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<11:33, 953kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:22<08:17, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<08:45, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<09:15, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:24<07:16, 1.51MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:24<05:18, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:26<07:07, 1.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:26<07:46, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:26<06:09, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:26<04:30, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:28<06:48, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:28<07:31, 1.44MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:28<05:56, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:28<04:20, 2.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:30<07:13, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<07:58, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<06:12, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:30<04:32, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<06:53, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<07:52, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:32<06:10, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:32<04:30, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:34<06:43, 1.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:34<07:44, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:34<06:04, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:34<04:25, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:36<05:53, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:36<06:10, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:36<04:50, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:36<03:31, 2.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:38<14:41, 712kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:38<12:50, 815kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:38<09:31, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:38<06:47, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:40<08:41, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:40<08:28, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:40<06:29, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:40<04:40, 2.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:42<09:48, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:42<09:08, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:42<06:58, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:42<05:01, 2.04MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:44<10:22, 989kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:44<08:55, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:44<06:35, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:44<04:42, 2.17MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:46<28:33, 357kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:46<22:13, 458kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:46<16:00, 636kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:46<11:19, 895kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:48<11:55, 848kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:48<11:14, 900kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:48<08:28, 1.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:48<06:08, 1.64MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:50<06:20, 1.59MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:50<05:23, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:50<03:58, 2.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:51<05:18, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:52<07:26, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:52<06:02, 1.65MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:52<04:28, 2.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:53<05:23, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:54<05:37, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:54<04:24, 2.25MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:55<04:42, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:56<06:59, 1.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:56<05:42, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:56<04:13, 2.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:57<05:16, 1.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:58<05:31, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:58<04:19, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:59<04:37, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:00<05:02, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:00<03:56, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:00<02:50, 3.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<17:50, 540kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [02:01<14:21, 671kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:02<10:25, 923kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:02<07:24, 1.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:03<10:11, 939kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:03<09:00, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:04<06:44, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:05<06:16, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:05<06:09, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:06<04:41, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:06<03:24, 2.78MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:07<07:07, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:07<06:45, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:08<05:10, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:09<05:08, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:09<07:24, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:09<05:57, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:10<04:23, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:11<05:16, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<05:25, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:11<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<04:28, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<06:52, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:13<05:42, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:14<04:10, 2.21MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:15<05:20, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:15<04:09, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:17<04:24, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:17<04:51, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:17<03:49, 2.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:19<04:09, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:19<04:35, 1.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:19<03:35, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:19<02:35, 3.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:21<20:42, 432kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:21<16:09, 554kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:21<11:43, 762kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:23<09:37, 922kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:23<08:28, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:23<06:20, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:25<05:52, 1.50MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:25<05:18, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:25<03:59, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:27<04:28, 1.95MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:27<06:24, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:27<05:16, 1.66MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:27<03:50, 2.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:29<04:56, 1.76MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:29<04:58, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:29<03:51, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:31<04:08, 2.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:31<06:07, 1.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:31<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:31<03:42, 2.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:33<04:49, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:33<04:54, 1.74MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:33<03:48, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:35<04:05, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:35<04:23, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:35<03:28, 2.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:37<03:49, 2.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:37<04:11, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:37<03:18, 2.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:39<03:42, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:39<05:29, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:39<04:35, 1.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:39<03:23, 2.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:41<04:31, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:41<04:39, 1.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:41<03:33, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<02:34, 3.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:43<09:05, 902kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:43<07:49, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:43<05:50, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:45<05:26, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:45<06:37, 1.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:45<05:21, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:45<03:53, 2.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:47<04:49, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:47<04:49, 1.67MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:47<03:43, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:49<03:56, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:49<04:11, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:49<03:17, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:51<03:38, 2.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:51<05:17, 1.50MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:51<04:24, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:51<03:15, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:53<04:19, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:53<04:23, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:53<03:25, 2.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:55<03:42, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:55<03:56, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:55<03:06, 2.50MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:56<03:27, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:57<03:49, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:57<03:01, 2.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:58<03:23, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:59<03:44, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:59<02:57, 2.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:00<03:20, 2.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [03:01<03:42, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:01<02:52, 2.63MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<02:05, 3.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<11:34, 648kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:02<09:12, 815kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:03<06:48, 1.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:03<04:56, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<05:23, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:04<04:45, 1.56MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:05<03:33, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:06<03:56, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:06<05:18, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:07<04:20, 1.69MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:07<03:11, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<04:14, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:08<04:13, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:09<03:13, 2.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:09<02:20, 3.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:10<05:37, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:10<05:11, 1.39MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:11<03:53, 1.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<02:47, 2.56MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:12<08:36, 832kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:12<07:15, 985kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:12<05:19, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:13<03:49, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:14<05:49, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:14<05:18, 1.34MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:14<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:16<03:58, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:16<05:04, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:16<04:05, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:17<03:00, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:18<03:57, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:18<03:58, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:18<03:01, 2.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<02:11, 3.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:20<09:32, 721kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:20<07:54, 871kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:20<05:46, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:20<04:05, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:22<11:39, 585kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:22<10:21, 658kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:22<07:50, 869kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:22<05:36, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:24<05:46, 1.17MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:24<05:12, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:24<03:52, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:24<02:49, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:26<04:38, 1.44MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:26<04:24, 1.52MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:26<03:21, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:28<03:28, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:28<03:32, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:28<02:45, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:30<03:02, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:30<03:15, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:30<02:33, 2.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:32<02:53, 2.24MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:32<03:08, 2.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:32<02:28, 2.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:34<02:49, 2.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:34<03:05, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:34<02:23, 2.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:34<01:44, 3.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:36<05:11, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:36<05:49, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:36<04:31, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:36<03:17, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:38<04:05, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:38<03:57, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:38<03:01, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:40<03:10, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:40<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:40<02:33, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:42<02:50, 2.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:42<04:06, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:42<03:19, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<02:24, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:44<03:21, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:44<03:24, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:44<02:38, 2.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:46<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:46<03:04, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:46<02:22, 2.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:46<01:42, 3.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:48<51:12, 116kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:48<36:51, 161kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:48<25:58, 227kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:50<19:02, 307kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:50<14:20, 408kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:50<10:13, 571kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:50<07:10, 808kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:52<11:16, 513kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:52<08:53, 650kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:52<06:24, 899kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<04:34, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:54<05:21, 1.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:54<04:45, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:54<03:33, 1.60MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:56<03:26, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:56<04:22, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:56<03:32, 1.59MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:56<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:58<03:20, 1.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:58<03:18, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<02:32, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:00<02:42, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:00<02:41, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<02:07, 2.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<01:33, 3.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:01<03:39, 1.49MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:02<03:30, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:02<02:40, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<02:46, 1.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:04<02:44, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:04<02:07, 2.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<01:37, 3.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:05<02:34, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:06<02:43, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:06<02:08, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<02:22, 2.20MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:08<02:34, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:08<02:01, 2.58MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<02:17, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<02:29, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:10<01:57, 2.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:11<02:14, 2.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:11<03:20, 1.52MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:12<02:47, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:12<02:03, 2.46MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:13<02:48, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:13<02:50, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:14<02:10, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:14<01:33, 3.19MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:15<1:14:35, 66.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:15<52:59, 93.5kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:16<37:11, 133kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:17<26:37, 184kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:17<19:52, 246kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:18<14:12, 343kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:18<09:56, 487kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<09:24, 513kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<07:05, 679kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:19<05:05, 941kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<03:36, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<06:18, 752kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<06:04, 783kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:21<04:34, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<03:16, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<03:25, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<03:04, 1.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:23<02:20, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<01:41, 2.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:25<04:56, 935kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:25<04:13, 1.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:25<03:08, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:27<02:57, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:27<02:51, 1.59MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:27<02:09, 2.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:27<01:32, 2.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:29<06:00, 744kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:29<05:45, 777kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:29<04:24, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:29<03:09, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:31<03:24, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:31<03:07, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:31<02:20, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:31<01:40, 2.60MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:33<06:53, 629kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:33<05:32, 781kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:33<04:03, 1.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:35<03:32, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:35<03:13, 1.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:35<02:25, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<02:24, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<02:23, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:37<01:50, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:39<01:59, 2.08MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:39<02:05, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:39<01:37, 2.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:41<01:49, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:41<01:57, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:41<01:32, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:43<01:45, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:43<01:55, 2.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:43<01:28, 2.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:43<01:04, 3.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:45<07:49, 502kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:45<06:09, 637kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<04:27, 876kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:47<03:44, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:47<03:16, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:47<02:26, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<02:20, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<02:16, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:49<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<01:50, 2.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<01:54, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:51<01:29, 2.48MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<01:39, 2.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:53<01:47, 2.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:53<01:24, 2.58MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:55<01:35, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:55<01:42, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:55<01:19, 2.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:55<00:57, 3.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:57<02:32, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:57<02:22, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:57<01:48, 1.94MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:59<01:50, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:59<01:46, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:59<01:23, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<01:00, 3.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:01<03:05, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:01<02:45, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:01<02:02, 1.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:01<01:27, 2.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:03<02:52, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:03<02:34, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:03<01:54, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:03<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:05<03:08, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:05<02:44, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:05<02:02, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:06<01:57, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:07<01:48, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:07<01:23, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:07<01:02, 2.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:08<01:37, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:09<02:12, 1.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:09<01:45, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:09<01:16, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:10<01:41, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:11<01:42, 1.76MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:11<01:19, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:12<01:25, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:13<01:29, 1.98MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:13<01:09, 2.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:14<01:18, 2.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:15<01:23, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:15<01:05, 2.63MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:16<01:14, 2.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:16<01:20, 2.10MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:17<01:03, 2.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:18<01:12, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:18<01:19, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:19<01:01, 2.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:19<00:44, 3.63MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:20<01:47, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:20<01:42, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:21<01:17, 2.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:22<01:20, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:22<01:51, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:23<01:30, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:23<01:06, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:24<01:28, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:24<01:27, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:25<01:07, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:26<01:11, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:26<01:42, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:26<01:22, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:27<01:00, 2.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:28<01:21, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<01:20, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:28<01:01, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:29<00:43, 3.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:30<02:55, 802kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:30<02:26, 961kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:30<01:47, 1.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<01:37, 1.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<01:27, 1.55MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:32<01:06, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:32<00:47, 2.79MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:34<02:30, 879kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:34<02:19, 950kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<01:44, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:34<01:14, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:36<01:23, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:36<01:20, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<01:00, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:38<01:03, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:38<01:27, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<01:11, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:38<00:51, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:40<01:08, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<01:09, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:40<00:52, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:40<00:37, 3.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:42<01:15, 1.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<01:11, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:42<00:54, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<00:56, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<00:58, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:44<00:45, 2.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<00:49, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:46<00:52, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:46<00:40, 2.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<00:28, 3.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<02:52, 598kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:48<02:18, 744kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<01:40, 1.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:50<01:25, 1.16MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:50<01:16, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<00:57, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<00:55, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<00:54, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:52<00:41, 2.25MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:54<00:44, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:54<00:45, 1.97MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:54<00:35, 2.51MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:56<00:39, 2.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:56<00:41, 2.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:56<00:32, 2.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:58<00:36, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:58<00:39, 2.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:58<00:30, 2.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:58<00:21, 3.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<01:06, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<00:59, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:00<00:44, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:42, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:02<00:42, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:02<00:32, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:04<00:33, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:04<00:35, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:04<00:27, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:04<00:19, 3.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:06<01:13, 899kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:06<01:02, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<00:46, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:08<00:41, 1.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:08<00:39, 1.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:08<00:29, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:08<00:20, 2.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:10<01:14, 776kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:10<01:01, 935kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:10<00:44, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:10<00:30, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:12<01:11, 757kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:12<01:08, 786kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:52, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:12<00:36, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:14<00:37, 1.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:14<00:35, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:14<00:25, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:14<00:17, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:15<02:34, 294kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<02:02, 370kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<01:28, 509kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:16<01:00, 715kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:17<00:51, 799kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:18<00:43, 958kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:18<00:31, 1.29MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:19<00:26, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:24, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:20<00:18, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:20<00:12, 2.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:21<00:36, 905kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:22<00:30, 1.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:22<00:22, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:22<00:15, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:23<00:24, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<00:27, 1.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:24<00:21, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:24<00:14, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:25<00:16, 1.54MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:25<00:15, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:26<00:11, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:26<00:07, 2.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:27<06:55, 50.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:27<04:51, 70.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:28<03:15, 100kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:28<02:03, 143kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<01:28, 187kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:29<01:03, 256kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:30<00:42, 360kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:26, 473kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:31<00:20, 605kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:32<00:13, 833kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:08, 987kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:33<00:07, 1.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:33<00:04, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:34<00:02, 2.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:04, 884kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:35<00:03, 1.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:35<00:02, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:37<00:00, 1.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:37<00:00, 1.22MB/s].vector_cache/glove.6B.zip: 862MB [06:37, 2.17MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 127/400000 [00:00<05:15, 1268.37it/s]  0%|          | 842/400000 [00:00<03:57, 1683.77it/s]  0%|          | 1549/400000 [00:00<03:02, 2182.58it/s]  1%|          | 2144/400000 [00:00<02:27, 2693.58it/s]  1%|          | 2710/400000 [00:00<02:04, 3195.94it/s]  1%|          | 3287/400000 [00:00<01:47, 3689.51it/s]  1%|          | 3877/400000 [00:00<01:35, 4155.86it/s]  1%|          | 4480/400000 [00:00<01:26, 4582.72it/s]  1%|▏         | 5038/400000 [00:00<01:21, 4840.80it/s]  1%|▏         | 5644/400000 [00:01<01:16, 5150.11it/s]  2%|▏         | 6350/400000 [00:01<01:10, 5603.54it/s]  2%|▏         | 7087/400000 [00:01<01:05, 6036.50it/s]  2%|▏         | 7793/400000 [00:01<01:02, 6310.01it/s]  2%|▏         | 8516/400000 [00:01<00:59, 6558.69it/s]  2%|▏         | 9229/400000 [00:01<00:58, 6719.22it/s]  2%|▏         | 9921/400000 [00:01<00:59, 6600.58it/s]  3%|▎         | 10664/400000 [00:01<00:57, 6826.90it/s]  3%|▎         | 11364/400000 [00:01<00:56, 6875.61it/s]  3%|▎         | 12061/400000 [00:01<00:59, 6493.35it/s]  3%|▎         | 12721/400000 [00:02<01:00, 6405.52it/s]  3%|▎         | 13453/400000 [00:02<00:58, 6654.21it/s]  4%|▎         | 14127/400000 [00:02<00:59, 6526.81it/s]  4%|▎         | 14829/400000 [00:02<00:57, 6666.16it/s]  4%|▍         | 15555/400000 [00:02<00:56, 6833.59it/s]  4%|▍         | 16286/400000 [00:02<00:55, 6969.68it/s]  4%|▍         | 16987/400000 [00:02<00:58, 6548.37it/s]  4%|▍         | 17650/400000 [00:02<01:00, 6314.87it/s]  5%|▍         | 18289/400000 [00:02<01:02, 6135.28it/s]  5%|▍         | 19013/400000 [00:02<00:59, 6428.52it/s]  5%|▍         | 19690/400000 [00:03<00:58, 6526.64it/s]  5%|▌         | 20415/400000 [00:03<00:56, 6727.48it/s]  5%|▌         | 21113/400000 [00:03<00:55, 6800.11it/s]  5%|▌         | 21845/400000 [00:03<00:54, 6945.81it/s]  6%|▌         | 22581/400000 [00:03<00:53, 7062.63it/s]  6%|▌         | 23317/400000 [00:03<00:52, 7149.17it/s]  6%|▌         | 24049/400000 [00:03<00:52, 7198.30it/s]  6%|▌         | 24771/400000 [00:03<00:53, 6979.73it/s]  6%|▋         | 25475/400000 [00:03<00:53, 6996.70it/s]  7%|▋         | 26210/400000 [00:04<00:52, 7097.18it/s]  7%|▋         | 26944/400000 [00:04<00:52, 7165.48it/s]  7%|▋         | 27677/400000 [00:04<00:51, 7211.47it/s]  7%|▋         | 28414/400000 [00:04<00:51, 7257.02it/s]  7%|▋         | 29141/400000 [00:04<00:51, 7250.65it/s]  7%|▋         | 29867/400000 [00:04<00:54, 6761.38it/s]  8%|▊         | 30551/400000 [00:04<00:57, 6415.18it/s]  8%|▊         | 31284/400000 [00:04<00:55, 6662.41it/s]  8%|▊         | 32023/400000 [00:04<00:53, 6863.54it/s]  8%|▊         | 32748/400000 [00:04<00:52, 6974.12it/s]  8%|▊         | 33476/400000 [00:05<00:51, 7062.29it/s]  9%|▊         | 34187/400000 [00:05<00:53, 6810.36it/s]  9%|▊         | 34873/400000 [00:05<00:54, 6646.98it/s]  9%|▉         | 35563/400000 [00:05<00:54, 6720.38it/s]  9%|▉         | 36239/400000 [00:05<00:56, 6399.57it/s]  9%|▉         | 36887/400000 [00:05<00:56, 6422.52it/s]  9%|▉         | 37617/400000 [00:05<00:54, 6661.84it/s] 10%|▉         | 38337/400000 [00:05<00:53, 6813.18it/s] 10%|▉         | 39070/400000 [00:05<00:51, 6960.34it/s] 10%|▉         | 39809/400000 [00:05<00:50, 7081.21it/s] 10%|█         | 40550/400000 [00:06<00:50, 7175.69it/s] 10%|█         | 41281/400000 [00:06<00:49, 7215.10it/s] 11%|█         | 42026/400000 [00:06<00:49, 7283.69it/s] 11%|█         | 42756/400000 [00:06<00:49, 7277.24it/s] 11%|█         | 43485/400000 [00:06<00:50, 7032.61it/s] 11%|█         | 44191/400000 [00:06<00:52, 6715.20it/s] 11%|█         | 44892/400000 [00:06<00:52, 6799.33it/s] 11%|█▏        | 45626/400000 [00:06<00:50, 6951.71it/s] 12%|█▏        | 46363/400000 [00:06<00:50, 7071.82it/s] 12%|█▏        | 47074/400000 [00:07<00:50, 7010.23it/s] 12%|█▏        | 47778/400000 [00:07<00:52, 6740.00it/s] 12%|█▏        | 48456/400000 [00:07<00:55, 6373.21it/s] 12%|█▏        | 49101/400000 [00:07<00:56, 6185.37it/s] 12%|█▏        | 49726/400000 [00:07<00:56, 6191.44it/s] 13%|█▎        | 50350/400000 [00:07<00:56, 6183.48it/s] 13%|█▎        | 50972/400000 [00:07<00:57, 6037.64it/s] 13%|█▎        | 51579/400000 [00:07<00:59, 5817.57it/s] 13%|█▎        | 52165/400000 [00:07<01:00, 5736.66it/s] 13%|█▎        | 52745/400000 [00:07<01:00, 5739.33it/s] 13%|█▎        | 53332/400000 [00:08<01:00, 5772.77it/s] 13%|█▎        | 53911/400000 [00:08<01:00, 5734.27it/s] 14%|█▎        | 54540/400000 [00:08<00:58, 5888.65it/s] 14%|█▍        | 55160/400000 [00:08<00:57, 5976.44it/s] 14%|█▍        | 55760/400000 [00:08<00:58, 5880.13it/s] 14%|█▍        | 56350/400000 [00:08<00:58, 5855.39it/s] 14%|█▍        | 56983/400000 [00:08<00:57, 5988.21it/s] 14%|█▍        | 57602/400000 [00:08<00:56, 6044.82it/s] 15%|█▍        | 58208/400000 [00:08<00:56, 6005.34it/s] 15%|█▍        | 58810/400000 [00:09<00:57, 5913.68it/s] 15%|█▍        | 59403/400000 [00:09<00:58, 5851.69it/s] 15%|█▌        | 60097/400000 [00:09<00:55, 6138.76it/s] 15%|█▌        | 60825/400000 [00:09<00:52, 6440.99it/s] 15%|█▌        | 61561/400000 [00:09<00:50, 6690.00it/s] 16%|█▌        | 62268/400000 [00:09<00:49, 6798.58it/s] 16%|█▌        | 62954/400000 [00:09<00:52, 6406.59it/s] 16%|█▌        | 63604/400000 [00:09<00:54, 6134.21it/s] 16%|█▌        | 64309/400000 [00:09<00:52, 6382.93it/s] 16%|█▋        | 65048/400000 [00:09<00:50, 6653.05it/s] 16%|█▋        | 65782/400000 [00:10<00:48, 6843.22it/s] 17%|█▋        | 66519/400000 [00:10<00:47, 6990.75it/s] 17%|█▋        | 67253/400000 [00:10<00:46, 7089.66it/s] 17%|█▋        | 67967/400000 [00:10<00:47, 7047.97it/s] 17%|█▋        | 68676/400000 [00:10<00:47, 6950.08it/s] 17%|█▋        | 69406/400000 [00:10<00:46, 7049.30it/s] 18%|█▊        | 70144/400000 [00:10<00:46, 7144.68it/s] 18%|█▊        | 70861/400000 [00:10<00:47, 6903.25it/s] 18%|█▊        | 71584/400000 [00:10<00:46, 6995.97it/s] 18%|█▊        | 72316/400000 [00:10<00:46, 7089.74it/s] 18%|█▊        | 73027/400000 [00:11<00:46, 7041.93it/s] 18%|█▊        | 73733/400000 [00:11<00:48, 6678.68it/s] 19%|█▊        | 74406/400000 [00:11<00:51, 6377.89it/s] 19%|█▉        | 75087/400000 [00:11<00:49, 6500.85it/s] 19%|█▉        | 75820/400000 [00:11<00:48, 6727.71it/s] 19%|█▉        | 76499/400000 [00:11<00:48, 6663.57it/s] 19%|█▉        | 77210/400000 [00:11<00:47, 6790.32it/s] 19%|█▉        | 77946/400000 [00:11<00:46, 6949.63it/s] 20%|█▉        | 78698/400000 [00:11<00:45, 7110.29it/s] 20%|█▉        | 79423/400000 [00:12<00:44, 7148.84it/s] 20%|██        | 80168/400000 [00:12<00:44, 7234.31it/s] 20%|██        | 80914/400000 [00:12<00:43, 7297.46it/s] 20%|██        | 81646/400000 [00:12<00:44, 7208.26it/s] 21%|██        | 82369/400000 [00:12<00:45, 6962.68it/s] 21%|██        | 83113/400000 [00:12<00:44, 7097.71it/s] 21%|██        | 83826/400000 [00:12<00:44, 7058.13it/s] 21%|██        | 84559/400000 [00:12<00:44, 7134.49it/s] 21%|██▏       | 85274/400000 [00:12<00:47, 6581.34it/s] 21%|██▏       | 85962/400000 [00:12<00:47, 6666.16it/s] 22%|██▏       | 86706/400000 [00:13<00:45, 6879.29it/s] 22%|██▏       | 87427/400000 [00:13<00:44, 6974.68it/s] 22%|██▏       | 88161/400000 [00:13<00:44, 7077.44it/s] 22%|██▏       | 88908/400000 [00:13<00:43, 7188.81it/s] 22%|██▏       | 89659/400000 [00:13<00:42, 7282.01it/s] 23%|██▎       | 90390/400000 [00:13<00:46, 6713.80it/s] 23%|██▎       | 91072/400000 [00:13<00:47, 6444.38it/s] 23%|██▎       | 91812/400000 [00:13<00:45, 6702.83it/s] 23%|██▎       | 92538/400000 [00:13<00:44, 6859.11it/s] 23%|██▎       | 93287/400000 [00:14<00:43, 7034.97it/s] 24%|██▎       | 94036/400000 [00:14<00:42, 7164.58it/s] 24%|██▎       | 94758/400000 [00:14<00:42, 7174.78it/s] 24%|██▍       | 95507/400000 [00:14<00:41, 7265.38it/s] 24%|██▍       | 96259/400000 [00:14<00:41, 7337.98it/s] 24%|██▍       | 97007/400000 [00:14<00:41, 7378.87it/s] 24%|██▍       | 97755/400000 [00:14<00:40, 7405.70it/s] 25%|██▍       | 98506/400000 [00:14<00:40, 7434.65it/s] 25%|██▍       | 99251/400000 [00:14<00:41, 7191.33it/s] 25%|██▍       | 99973/400000 [00:14<00:44, 6758.22it/s] 25%|██▌       | 100717/400000 [00:15<00:43, 6947.79it/s] 25%|██▌       | 101418/400000 [00:15<00:45, 6541.30it/s] 26%|██▌       | 102082/400000 [00:15<00:46, 6363.77it/s] 26%|██▌       | 102726/400000 [00:15<00:47, 6280.94it/s] 26%|██▌       | 103360/400000 [00:15<00:48, 6084.25it/s] 26%|██▌       | 103974/400000 [00:15<00:49, 5924.08it/s] 26%|██▌       | 104574/400000 [00:15<00:49, 5945.97it/s] 26%|██▋       | 105172/400000 [00:15<00:50, 5797.04it/s] 26%|██▋       | 105757/400000 [00:15<00:50, 5810.01it/s] 27%|██▋       | 106436/400000 [00:16<00:48, 6072.37it/s] 27%|██▋       | 107172/400000 [00:16<00:45, 6408.28it/s] 27%|██▋       | 107881/400000 [00:16<00:44, 6598.54it/s] 27%|██▋       | 108628/400000 [00:16<00:42, 6835.80it/s] 27%|██▋       | 109319/400000 [00:16<00:44, 6488.31it/s] 27%|██▋       | 109977/400000 [00:16<00:45, 6340.44it/s] 28%|██▊       | 110618/400000 [00:16<00:46, 6264.57it/s] 28%|██▊       | 111250/400000 [00:16<00:46, 6218.87it/s] 28%|██▊       | 111876/400000 [00:16<00:47, 6017.88it/s] 28%|██▊       | 112563/400000 [00:16<00:45, 6250.21it/s] 28%|██▊       | 113283/400000 [00:17<00:44, 6506.76it/s] 28%|██▊       | 114000/400000 [00:17<00:42, 6686.04it/s] 29%|██▊       | 114712/400000 [00:17<00:42, 6778.31it/s] 29%|██▉       | 115458/400000 [00:17<00:40, 6968.97it/s] 29%|██▉       | 116179/400000 [00:17<00:40, 7039.08it/s] 29%|██▉       | 116930/400000 [00:17<00:39, 7172.38it/s] 29%|██▉       | 117678/400000 [00:17<00:38, 7260.78it/s] 30%|██▉       | 118419/400000 [00:17<00:38, 7301.97it/s] 30%|██▉       | 119169/400000 [00:17<00:38, 7357.52it/s] 30%|██▉       | 119921/400000 [00:17<00:37, 7403.28it/s] 30%|███       | 120663/400000 [00:18<00:37, 7352.54it/s] 30%|███       | 121413/400000 [00:18<00:37, 7395.85it/s] 31%|███       | 122157/400000 [00:18<00:37, 7407.59it/s] 31%|███       | 122899/400000 [00:18<00:37, 7400.18it/s] 31%|███       | 123646/400000 [00:18<00:37, 7419.12it/s] 31%|███       | 124395/400000 [00:18<00:37, 7440.16it/s] 31%|███▏      | 125140/400000 [00:18<00:39, 6971.98it/s] 31%|███▏      | 125889/400000 [00:18<00:38, 7117.27it/s] 32%|███▏      | 126606/400000 [00:18<00:40, 6779.08it/s] 32%|███▏      | 127291/400000 [00:19<00:41, 6543.22it/s] 32%|███▏      | 127962/400000 [00:19<00:41, 6591.10it/s] 32%|███▏      | 128664/400000 [00:19<00:40, 6713.53it/s] 32%|███▏      | 129393/400000 [00:19<00:39, 6874.17it/s] 33%|███▎      | 130142/400000 [00:19<00:38, 7045.38it/s] 33%|███▎      | 130894/400000 [00:19<00:37, 7180.53it/s] 33%|███▎      | 131629/400000 [00:19<00:37, 7230.28it/s] 33%|███▎      | 132371/400000 [00:19<00:36, 7284.50it/s] 33%|███▎      | 133102/400000 [00:19<00:38, 6964.36it/s] 33%|███▎      | 133803/400000 [00:19<00:40, 6634.71it/s] 34%|███▎      | 134473/400000 [00:20<00:41, 6388.82it/s] 34%|███▍      | 135213/400000 [00:20<00:39, 6661.84it/s] 34%|███▍      | 135951/400000 [00:20<00:38, 6861.00it/s] 34%|███▍      | 136678/400000 [00:20<00:37, 6977.68it/s] 34%|███▍      | 137426/400000 [00:20<00:36, 7118.52it/s] 35%|███▍      | 138162/400000 [00:20<00:36, 7187.59it/s] 35%|███▍      | 138884/400000 [00:20<00:38, 6818.21it/s] 35%|███▍      | 139573/400000 [00:20<00:40, 6459.48it/s] 35%|███▌      | 140228/400000 [00:20<00:41, 6308.84it/s] 35%|███▌      | 140872/400000 [00:21<00:40, 6347.42it/s] 35%|███▌      | 141620/400000 [00:21<00:38, 6648.53it/s] 36%|███▌      | 142353/400000 [00:21<00:37, 6838.00it/s] 36%|███▌      | 143102/400000 [00:21<00:36, 7020.44it/s] 36%|███▌      | 143850/400000 [00:21<00:35, 7151.60it/s] 36%|███▌      | 144600/400000 [00:21<00:35, 7252.42it/s] 36%|███▋      | 145336/400000 [00:21<00:34, 7284.33it/s] 37%|███▋      | 146080/400000 [00:21<00:34, 7329.62it/s] 37%|███▋      | 146815/400000 [00:21<00:34, 7248.65it/s] 37%|███▋      | 147542/400000 [00:21<00:37, 6664.46it/s] 37%|███▋      | 148219/400000 [00:22<00:38, 6539.46it/s] 37%|███▋      | 148881/400000 [00:22<00:38, 6480.81it/s] 37%|███▋      | 149535/400000 [00:22<00:39, 6362.81it/s] 38%|███▊      | 150239/400000 [00:22<00:38, 6550.88it/s] 38%|███▊      | 150944/400000 [00:22<00:37, 6691.76it/s] 38%|███▊      | 151689/400000 [00:22<00:35, 6901.17it/s] 38%|███▊      | 152439/400000 [00:22<00:35, 7068.03it/s] 38%|███▊      | 153150/400000 [00:22<00:35, 6971.39it/s] 38%|███▊      | 153851/400000 [00:22<00:36, 6671.96it/s] 39%|███▊      | 154524/400000 [00:22<00:37, 6616.60it/s] 39%|███▉      | 155250/400000 [00:23<00:36, 6794.98it/s] 39%|███▉      | 155968/400000 [00:23<00:35, 6905.65it/s] 39%|███▉      | 156662/400000 [00:23<00:35, 6789.24it/s] 39%|███▉      | 157344/400000 [00:23<00:36, 6634.20it/s] 40%|███▉      | 158010/400000 [00:23<00:37, 6439.77it/s] 40%|███▉      | 158658/400000 [00:23<00:39, 6185.36it/s] 40%|███▉      | 159281/400000 [00:23<00:40, 5924.10it/s] 40%|███▉      | 159890/400000 [00:23<00:40, 5970.53it/s] 40%|████      | 160491/400000 [00:23<00:40, 5958.24it/s] 40%|████      | 161120/400000 [00:24<00:39, 6051.65it/s] 40%|████      | 161728/400000 [00:24<00:39, 6028.11it/s] 41%|████      | 162333/400000 [00:24<00:39, 6011.19it/s] 41%|████      | 162985/400000 [00:24<00:38, 6152.67it/s] 41%|████      | 163699/400000 [00:24<00:36, 6418.17it/s] 41%|████      | 164446/400000 [00:24<00:35, 6700.01it/s] 41%|████▏     | 165192/400000 [00:24<00:33, 6909.76it/s] 41%|████▏     | 165889/400000 [00:24<00:33, 6885.69it/s] 42%|████▏     | 166582/400000 [00:24<00:35, 6495.20it/s] 42%|████▏     | 167239/400000 [00:24<00:37, 6276.12it/s] 42%|████▏     | 167971/400000 [00:25<00:35, 6555.29it/s] 42%|████▏     | 168635/400000 [00:25<00:36, 6415.93it/s] 42%|████▏     | 169283/400000 [00:25<00:38, 5989.55it/s] 42%|████▏     | 169903/400000 [00:25<00:38, 6049.54it/s] 43%|████▎     | 170516/400000 [00:25<00:38, 6001.35it/s] 43%|████▎     | 171263/400000 [00:25<00:35, 6376.16it/s] 43%|████▎     | 171982/400000 [00:25<00:34, 6600.02it/s] 43%|████▎     | 172715/400000 [00:25<00:33, 6803.05it/s] 43%|████▎     | 173465/400000 [00:25<00:32, 6995.70it/s] 44%|████▎     | 174211/400000 [00:26<00:31, 7127.82it/s] 44%|████▎     | 174936/400000 [00:26<00:31, 7145.42it/s] 44%|████▍     | 175655/400000 [00:26<00:33, 6729.34it/s] 44%|████▍     | 176336/400000 [00:26<00:33, 6622.79it/s] 44%|████▍     | 177083/400000 [00:26<00:32, 6854.20it/s] 44%|████▍     | 177830/400000 [00:26<00:31, 7027.30it/s] 45%|████▍     | 178582/400000 [00:26<00:30, 7165.88it/s] 45%|████▍     | 179303/400000 [00:26<00:31, 7069.86it/s] 45%|████▌     | 180014/400000 [00:26<00:33, 6651.49it/s] 45%|████▌     | 180687/400000 [00:27<00:34, 6312.97it/s] 45%|████▌     | 181328/400000 [00:27<00:35, 6162.57it/s] 45%|████▌     | 181952/400000 [00:27<00:35, 6104.83it/s] 46%|████▌     | 182568/400000 [00:27<00:36, 6009.11it/s] 46%|████▌     | 183173/400000 [00:27<00:36, 6004.02it/s] 46%|████▌     | 183919/400000 [00:27<00:33, 6376.04it/s] 46%|████▌     | 184656/400000 [00:27<00:32, 6643.82it/s] 46%|████▋     | 185389/400000 [00:27<00:31, 6835.78it/s] 47%|████▋     | 186081/400000 [00:27<00:32, 6541.61it/s] 47%|████▋     | 186753/400000 [00:27<00:32, 6593.81it/s] 47%|████▋     | 187490/400000 [00:28<00:31, 6808.48it/s] 47%|████▋     | 188236/400000 [00:28<00:30, 6990.48it/s] 47%|████▋     | 188975/400000 [00:28<00:29, 7105.65it/s] 47%|████▋     | 189690/400000 [00:28<00:29, 7093.74it/s] 48%|████▊     | 190433/400000 [00:28<00:29, 7190.89it/s] 48%|████▊     | 191155/400000 [00:28<00:29, 7076.55it/s] 48%|████▊     | 191865/400000 [00:28<00:31, 6600.86it/s] 48%|████▊     | 192533/400000 [00:28<00:32, 6338.81it/s] 48%|████▊     | 193175/400000 [00:28<00:33, 6092.01it/s] 48%|████▊     | 193865/400000 [00:29<00:32, 6312.48it/s] 49%|████▊     | 194614/400000 [00:29<00:31, 6624.09it/s] 49%|████▉     | 195347/400000 [00:29<00:30, 6819.02it/s] 49%|████▉     | 196090/400000 [00:29<00:29, 6990.56it/s] 49%|████▉     | 196810/400000 [00:29<00:28, 7050.83it/s] 49%|████▉     | 197550/400000 [00:29<00:28, 7151.52it/s] 50%|████▉     | 198269/400000 [00:29<00:29, 6737.96it/s] 50%|████▉     | 198951/400000 [00:29<00:30, 6509.46it/s] 50%|████▉     | 199609/400000 [00:29<00:31, 6352.95it/s] 50%|█████     | 200349/400000 [00:29<00:30, 6632.37it/s] 50%|█████     | 201091/400000 [00:30<00:29, 6850.45it/s] 50%|█████     | 201834/400000 [00:30<00:28, 7012.80it/s] 51%|█████     | 202562/400000 [00:30<00:27, 7089.56it/s] 51%|█████     | 203295/400000 [00:30<00:27, 7158.20it/s] 51%|█████     | 204014/400000 [00:30<00:29, 6592.52it/s] 51%|█████     | 204685/400000 [00:30<00:30, 6404.03it/s] 51%|█████▏    | 205335/400000 [00:30<00:30, 6326.46it/s] 51%|█████▏    | 205975/400000 [00:30<00:31, 6248.15it/s] 52%|█████▏    | 206605/400000 [00:30<00:31, 6232.80it/s] 52%|█████▏    | 207352/400000 [00:31<00:29, 6558.15it/s] 52%|█████▏    | 208095/400000 [00:31<00:28, 6795.48it/s] 52%|█████▏    | 208825/400000 [00:31<00:27, 6937.92it/s] 52%|█████▏    | 209531/400000 [00:31<00:27, 6971.48it/s] 53%|█████▎    | 210261/400000 [00:31<00:26, 7065.74it/s] 53%|█████▎    | 210991/400000 [00:31<00:26, 7132.25it/s] 53%|█████▎    | 211725/400000 [00:31<00:26, 7191.45it/s] 53%|█████▎    | 212474/400000 [00:31<00:25, 7276.35it/s] 53%|█████▎    | 213204/400000 [00:31<00:26, 7008.38it/s] 53%|█████▎    | 213908/400000 [00:31<00:27, 6831.57it/s] 54%|█████▎    | 214647/400000 [00:32<00:26, 6989.58it/s] 54%|█████▍    | 215370/400000 [00:32<00:26, 7058.46it/s] 54%|█████▍    | 216100/400000 [00:32<00:25, 7128.30it/s] 54%|█████▍    | 216840/400000 [00:32<00:25, 7207.59it/s] 54%|█████▍    | 217563/400000 [00:32<00:25, 7212.24it/s] 55%|█████▍    | 218308/400000 [00:32<00:24, 7281.80it/s] 55%|█████▍    | 219038/400000 [00:32<00:24, 7241.62it/s] 55%|█████▍    | 219788/400000 [00:32<00:24, 7316.59it/s] 55%|█████▌    | 220521/400000 [00:32<00:25, 7057.07it/s] 55%|█████▌    | 221230/400000 [00:32<00:25, 6978.60it/s] 55%|█████▌    | 221965/400000 [00:33<00:25, 7085.57it/s] 56%|█████▌    | 222710/400000 [00:33<00:24, 7188.93it/s] 56%|█████▌    | 223431/400000 [00:33<00:24, 7170.19it/s] 56%|█████▌    | 224154/400000 [00:33<00:24, 7187.95it/s] 56%|█████▌    | 224874/400000 [00:33<00:25, 6970.31it/s] 56%|█████▋    | 225574/400000 [00:33<00:25, 6973.08it/s] 57%|█████▋    | 226273/400000 [00:33<00:24, 6962.33it/s] 57%|█████▋    | 227020/400000 [00:33<00:24, 7104.47it/s] 57%|█████▋    | 227732/400000 [00:33<00:24, 6987.45it/s] 57%|█████▋    | 228433/400000 [00:33<00:25, 6834.18it/s] 57%|█████▋    | 229119/400000 [00:34<00:25, 6685.67it/s] 57%|█████▋    | 229790/400000 [00:34<00:26, 6510.08it/s] 58%|█████▊    | 230444/400000 [00:34<00:26, 6436.48it/s] 58%|█████▊    | 231183/400000 [00:34<00:25, 6694.38it/s] 58%|█████▊    | 231899/400000 [00:34<00:24, 6824.74it/s] 58%|█████▊    | 232644/400000 [00:34<00:23, 6998.94it/s] 58%|█████▊    | 233392/400000 [00:34<00:23, 7134.48it/s] 59%|█████▊    | 234143/400000 [00:34<00:22, 7242.85it/s] 59%|█████▊    | 234892/400000 [00:34<00:22, 7313.82it/s] 59%|█████▉    | 235628/400000 [00:34<00:22, 7327.04it/s] 59%|█████▉    | 236363/400000 [00:35<00:22, 7296.35it/s] 59%|█████▉    | 237094/400000 [00:35<00:23, 7079.44it/s] 59%|█████▉    | 237805/400000 [00:35<00:23, 7043.29it/s] 60%|█████▉    | 238553/400000 [00:35<00:22, 7168.19it/s] 60%|█████▉    | 239290/400000 [00:35<00:22, 7226.55it/s] 60%|██████    | 240024/400000 [00:35<00:22, 7257.21it/s] 60%|██████    | 240751/400000 [00:35<00:22, 7040.79it/s] 60%|██████    | 241458/400000 [00:35<00:23, 6833.58it/s] 61%|██████    | 242145/400000 [00:35<00:23, 6768.49it/s] 61%|██████    | 242880/400000 [00:36<00:22, 6930.84it/s] 61%|██████    | 243628/400000 [00:36<00:22, 7084.68it/s] 61%|██████    | 244358/400000 [00:36<00:21, 7147.01it/s] 61%|██████▏   | 245086/400000 [00:36<00:21, 7184.84it/s] 61%|██████▏   | 245826/400000 [00:36<00:21, 7246.96it/s] 62%|██████▏   | 246552/400000 [00:36<00:22, 6946.23it/s] 62%|██████▏   | 247251/400000 [00:36<00:23, 6615.87it/s] 62%|██████▏   | 247982/400000 [00:36<00:22, 6809.62it/s] 62%|██████▏   | 248669/400000 [00:36<00:22, 6741.49it/s] 62%|██████▏   | 249367/400000 [00:36<00:22, 6809.73it/s] 63%|██████▎   | 250105/400000 [00:37<00:21, 6970.56it/s] 63%|██████▎   | 250853/400000 [00:37<00:20, 7114.10it/s] 63%|██████▎   | 251568/400000 [00:37<00:20, 7080.85it/s] 63%|██████▎   | 252313/400000 [00:37<00:20, 7187.20it/s] 63%|██████▎   | 253062/400000 [00:37<00:20, 7272.92it/s] 63%|██████▎   | 253791/400000 [00:37<00:20, 7208.01it/s] 64%|██████▎   | 254541/400000 [00:37<00:19, 7292.93it/s] 64%|██████▍   | 255286/400000 [00:37<00:19, 7338.99it/s] 64%|██████▍   | 256035/400000 [00:37<00:19, 7381.06it/s] 64%|██████▍   | 256774/400000 [00:37<00:19, 7375.39it/s] 64%|██████▍   | 257512/400000 [00:38<00:19, 7340.26it/s] 65%|██████▍   | 258247/400000 [00:38<00:19, 7310.44it/s] 65%|██████▍   | 258998/400000 [00:38<00:19, 7367.43it/s] 65%|██████▍   | 259736/400000 [00:38<00:19, 7351.72it/s] 65%|██████▌   | 260472/400000 [00:38<00:18, 7353.47it/s] 65%|██████▌   | 261208/400000 [00:38<00:19, 7249.75it/s] 65%|██████▌   | 261936/400000 [00:38<00:19, 7258.73it/s] 66%|██████▌   | 262663/400000 [00:38<00:18, 7251.08it/s] 66%|██████▌   | 263412/400000 [00:38<00:18, 7319.37it/s] 66%|██████▌   | 264160/400000 [00:38<00:18, 7366.67it/s] 66%|██████▌   | 264898/400000 [00:39<00:18, 7369.63it/s] 66%|██████▋   | 265636/400000 [00:39<00:18, 7188.68it/s] 67%|██████▋   | 266356/400000 [00:39<00:18, 7182.30it/s] 67%|██████▋   | 267102/400000 [00:39<00:18, 7262.35it/s] 67%|██████▋   | 267853/400000 [00:39<00:18, 7333.86it/s] 67%|██████▋   | 268592/400000 [00:39<00:17, 7348.00it/s] 67%|██████▋   | 269339/400000 [00:39<00:17, 7382.18it/s] 68%|██████▊   | 270087/400000 [00:39<00:17, 7410.31it/s] 68%|██████▊   | 270829/400000 [00:39<00:17, 7337.74it/s] 68%|██████▊   | 271575/400000 [00:40<00:17, 7371.52it/s] 68%|██████▊   | 272313/400000 [00:40<00:17, 7348.26it/s] 68%|██████▊   | 273049/400000 [00:40<00:18, 7043.76it/s] 68%|██████▊   | 273757/400000 [00:40<00:18, 6910.99it/s] 69%|██████▊   | 274497/400000 [00:40<00:17, 7050.10it/s] 69%|██████▉   | 275226/400000 [00:40<00:17, 7120.14it/s] 69%|██████▉   | 275960/400000 [00:40<00:17, 7184.00it/s] 69%|██████▉   | 276712/400000 [00:40<00:16, 7281.51it/s] 69%|██████▉   | 277442/400000 [00:40<00:17, 6817.49it/s] 70%|██████▉   | 278181/400000 [00:40<00:17, 6979.10it/s] 70%|██████▉   | 278909/400000 [00:41<00:17, 7065.71it/s] 70%|██████▉   | 279640/400000 [00:41<00:16, 7135.38it/s] 70%|███████   | 280389/400000 [00:41<00:16, 7195.03it/s] 70%|███████   | 281111/400000 [00:41<00:16, 7202.17it/s] 70%|███████   | 281859/400000 [00:41<00:16, 7281.96it/s] 71%|███████   | 282610/400000 [00:41<00:15, 7348.70it/s] 71%|███████   | 283346/400000 [00:41<00:16, 7281.28it/s] 71%|███████   | 284096/400000 [00:41<00:15, 7344.46it/s] 71%|███████   | 284837/400000 [00:41<00:15, 7362.89it/s] 71%|███████▏  | 285582/400000 [00:41<00:15, 7386.40it/s] 72%|███████▏  | 286325/400000 [00:42<00:15, 7398.51it/s] 72%|███████▏  | 287066/400000 [00:42<00:15, 7369.30it/s] 72%|███████▏  | 287804/400000 [00:42<00:15, 7143.83it/s] 72%|███████▏  | 288521/400000 [00:42<00:15, 6987.71it/s] 72%|███████▏  | 289222/400000 [00:42<00:16, 6842.53it/s] 72%|███████▏  | 289951/400000 [00:42<00:15, 6969.48it/s] 73%|███████▎  | 290690/400000 [00:42<00:15, 7089.78it/s] 73%|███████▎  | 291437/400000 [00:42<00:15, 7199.62it/s] 73%|███████▎  | 292159/400000 [00:42<00:15, 7097.99it/s] 73%|███████▎  | 292900/400000 [00:42<00:14, 7186.32it/s] 73%|███████▎  | 293621/400000 [00:43<00:14, 7182.84it/s] 74%|███████▎  | 294341/400000 [00:43<00:15, 6819.12it/s] 74%|███████▍  | 295028/400000 [00:43<00:15, 6802.46it/s] 74%|███████▍  | 295712/400000 [00:43<00:15, 6750.25it/s] 74%|███████▍  | 296390/400000 [00:43<00:15, 6518.84it/s] 74%|███████▍  | 297117/400000 [00:43<00:15, 6726.45it/s] 74%|███████▍  | 297866/400000 [00:43<00:14, 6937.05it/s] 75%|███████▍  | 298608/400000 [00:43<00:14, 7073.18it/s] 75%|███████▍  | 299355/400000 [00:43<00:14, 7186.08it/s] 75%|███████▌  | 300097/400000 [00:44<00:13, 7251.45it/s] 75%|███████▌  | 300828/400000 [00:44<00:13, 7267.53it/s] 75%|███████▌  | 301572/400000 [00:44<00:13, 7317.36it/s] 76%|███████▌  | 302322/400000 [00:44<00:13, 7368.65it/s] 76%|███████▌  | 303060/400000 [00:44<00:13, 7266.11it/s] 76%|███████▌  | 303788/400000 [00:44<00:13, 7220.73it/s] 76%|███████▌  | 304511/400000 [00:44<00:13, 7211.25it/s] 76%|███████▋  | 305260/400000 [00:44<00:12, 7291.82it/s] 76%|███████▋  | 305996/400000 [00:44<00:12, 7308.00it/s] 77%|███████▋  | 306728/400000 [00:44<00:12, 7226.68it/s] 77%|███████▋  | 307452/400000 [00:45<00:12, 7176.84it/s] 77%|███████▋  | 308189/400000 [00:45<00:12, 7231.32it/s] 77%|███████▋  | 308913/400000 [00:45<00:12, 7215.57it/s] 77%|███████▋  | 309651/400000 [00:45<00:12, 7262.69it/s] 78%|███████▊  | 310378/400000 [00:45<00:12, 7218.26it/s] 78%|███████▊  | 311127/400000 [00:45<00:12, 7297.67it/s] 78%|███████▊  | 311866/400000 [00:45<00:12, 7322.70it/s] 78%|███████▊  | 312612/400000 [00:45<00:11, 7361.25it/s] 78%|███████▊  | 313349/400000 [00:45<00:11, 7338.79it/s] 79%|███████▊  | 314084/400000 [00:45<00:11, 7272.90it/s] 79%|███████▊  | 314823/400000 [00:46<00:11, 7306.26it/s] 79%|███████▉  | 315554/400000 [00:46<00:11, 7207.67it/s] 79%|███████▉  | 316288/400000 [00:46<00:11, 7244.17it/s] 79%|███████▉  | 317036/400000 [00:46<00:11, 7311.84it/s] 79%|███████▉  | 317768/400000 [00:46<00:11, 7268.95it/s] 80%|███████▉  | 318496/400000 [00:46<00:11, 7177.47it/s] 80%|███████▉  | 319235/400000 [00:46<00:11, 7237.64it/s] 80%|███████▉  | 319978/400000 [00:46<00:10, 7294.25it/s] 80%|████████  | 320714/400000 [00:46<00:10, 7313.84it/s] 80%|████████  | 321446/400000 [00:46<00:10, 7301.17it/s] 81%|████████  | 322177/400000 [00:47<00:10, 7253.21it/s] 81%|████████  | 322913/400000 [00:47<00:10, 7282.42it/s] 81%|████████  | 323653/400000 [00:47<00:10, 7315.72it/s] 81%|████████  | 324392/400000 [00:47<00:10, 7336.41it/s] 81%|████████▏ | 325133/400000 [00:47<00:10, 7357.08it/s] 81%|████████▏ | 325869/400000 [00:47<00:10, 7280.51it/s] 82%|████████▏ | 326598/400000 [00:47<00:10, 7280.09it/s] 82%|████████▏ | 327342/400000 [00:47<00:09, 7326.86it/s] 82%|████████▏ | 328090/400000 [00:47<00:09, 7371.04it/s] 82%|████████▏ | 328836/400000 [00:47<00:09, 7396.59it/s] 82%|████████▏ | 329586/400000 [00:48<00:09, 7426.08it/s] 83%|████████▎ | 330329/400000 [00:48<00:09, 7326.86it/s] 83%|████████▎ | 331063/400000 [00:48<00:09, 7080.24it/s] 83%|████████▎ | 331804/400000 [00:48<00:09, 7174.51it/s] 83%|████████▎ | 332524/400000 [00:48<00:09, 7172.43it/s] 83%|████████▎ | 333243/400000 [00:48<00:09, 7055.96it/s] 83%|████████▎ | 333976/400000 [00:48<00:09, 7133.92it/s] 84%|████████▎ | 334691/400000 [00:48<00:09, 7127.09it/s] 84%|████████▍ | 335439/400000 [00:48<00:08, 7228.67it/s] 84%|████████▍ | 336173/400000 [00:48<00:08, 7260.04it/s] 84%|████████▍ | 336921/400000 [00:49<00:08, 7322.82it/s] 84%|████████▍ | 337660/400000 [00:49<00:08, 7340.67it/s] 85%|████████▍ | 338400/400000 [00:49<00:08, 7356.97it/s] 85%|████████▍ | 339137/400000 [00:49<00:08, 7183.63it/s] 85%|████████▍ | 339879/400000 [00:49<00:08, 7252.21it/s] 85%|████████▌ | 340621/400000 [00:49<00:08, 7298.52it/s] 85%|████████▌ | 341359/400000 [00:49<00:08, 7321.40it/s] 86%|████████▌ | 342092/400000 [00:49<00:08, 7024.79it/s] 86%|████████▌ | 342798/400000 [00:49<00:08, 6830.91it/s] 86%|████████▌ | 343516/400000 [00:50<00:08, 6930.14it/s] 86%|████████▌ | 344249/400000 [00:50<00:07, 7044.76it/s] 86%|████████▌ | 344956/400000 [00:50<00:08, 6758.47it/s] 86%|████████▋ | 345636/400000 [00:50<00:08, 6523.69it/s] 87%|████████▋ | 346293/400000 [00:50<00:08, 6467.12it/s] 87%|████████▋ | 347015/400000 [00:50<00:07, 6673.97it/s] 87%|████████▋ | 347726/400000 [00:50<00:07, 6798.36it/s] 87%|████████▋ | 348463/400000 [00:50<00:07, 6958.75it/s] 87%|████████▋ | 349213/400000 [00:50<00:07, 7110.79it/s] 87%|████████▋ | 349928/400000 [00:50<00:07, 6928.92it/s] 88%|████████▊ | 350669/400000 [00:51<00:06, 7066.54it/s] 88%|████████▊ | 351387/400000 [00:51<00:06, 7099.06it/s] 88%|████████▊ | 352114/400000 [00:51<00:06, 7145.64it/s] 88%|████████▊ | 352831/400000 [00:51<00:06, 7116.05it/s] 88%|████████▊ | 353581/400000 [00:51<00:06, 7227.01it/s] 89%|████████▊ | 354305/400000 [00:51<00:06, 7204.09it/s] 89%|████████▉ | 355054/400000 [00:51<00:06, 7286.67it/s] 89%|████████▉ | 355784/400000 [00:51<00:06, 6943.18it/s] 89%|████████▉ | 356494/400000 [00:51<00:06, 6988.38it/s] 89%|████████▉ | 357237/400000 [00:51<00:06, 7114.32it/s] 89%|████████▉ | 357984/400000 [00:52<00:05, 7216.26it/s] 90%|████████▉ | 358729/400000 [00:52<00:05, 7284.78it/s] 90%|████████▉ | 359465/400000 [00:52<00:05, 7306.68it/s] 90%|█████████ | 360197/400000 [00:52<00:05, 7216.52it/s] 90%|█████████ | 360943/400000 [00:52<00:05, 7287.25it/s] 90%|█████████ | 361676/400000 [00:52<00:05, 7298.56it/s] 91%|█████████ | 362427/400000 [00:52<00:05, 7360.07it/s] 91%|█████████ | 363164/400000 [00:52<00:05, 7148.39it/s] 91%|█████████ | 363917/400000 [00:52<00:04, 7256.10it/s] 91%|█████████ | 364645/400000 [00:52<00:04, 7195.44it/s] 91%|█████████▏| 365388/400000 [00:53<00:04, 7262.53it/s] 92%|█████████▏| 366139/400000 [00:53<00:04, 7332.38it/s] 92%|█████████▏| 366874/400000 [00:53<00:04, 7332.69it/s] 92%|█████████▏| 367608/400000 [00:53<00:04, 7284.05it/s] 92%|█████████▏| 368337/400000 [00:53<00:04, 7266.63it/s] 92%|█████████▏| 369065/400000 [00:53<00:04, 7217.73it/s] 92%|█████████▏| 369816/400000 [00:53<00:04, 7300.94it/s] 93%|█████████▎| 370555/400000 [00:53<00:04, 7325.65it/s] 93%|█████████▎| 371302/400000 [00:53<00:03, 7368.04it/s] 93%|█████████▎| 372050/400000 [00:54<00:03, 7398.58it/s] 93%|█████████▎| 372791/400000 [00:54<00:03, 7238.91it/s] 93%|█████████▎| 373516/400000 [00:54<00:03, 7132.97it/s] 94%|█████████▎| 374257/400000 [00:54<00:03, 7212.46it/s] 94%|█████████▍| 375007/400000 [00:54<00:03, 7294.33it/s] 94%|█████████▍| 375757/400000 [00:54<00:03, 7353.12it/s] 94%|█████████▍| 376494/400000 [00:54<00:03, 7353.63it/s] 94%|█████████▍| 377230/400000 [00:54<00:03, 7259.35it/s] 94%|█████████▍| 377957/400000 [00:54<00:03, 7168.50it/s] 95%|█████████▍| 378704/400000 [00:54<00:02, 7255.59it/s] 95%|█████████▍| 379444/400000 [00:55<00:02, 7296.27it/s] 95%|█████████▌| 380176/400000 [00:55<00:02, 7302.56it/s] 95%|█████████▌| 380907/400000 [00:55<00:02, 7067.83it/s] 95%|█████████▌| 381616/400000 [00:55<00:02, 6966.76it/s] 96%|█████████▌| 382356/400000 [00:55<00:02, 7090.35it/s] 96%|█████████▌| 383094/400000 [00:55<00:02, 7174.06it/s] 96%|█████████▌| 383813/400000 [00:55<00:02, 7125.90it/s] 96%|█████████▌| 384552/400000 [00:55<00:02, 7201.32it/s] 96%|█████████▋| 385284/400000 [00:55<00:02, 7234.39it/s] 97%|█████████▋| 386009/400000 [00:55<00:01, 7229.33it/s] 97%|█████████▋| 386757/400000 [00:56<00:01, 7300.55it/s] 97%|█████████▋| 387488/400000 [00:56<00:01, 7214.93it/s] 97%|█████████▋| 388230/400000 [00:56<00:01, 7275.22it/s] 97%|█████████▋| 388959/400000 [00:56<00:01, 7262.03it/s] 97%|█████████▋| 389686/400000 [00:56<00:01, 7218.25it/s] 98%|█████████▊| 390415/400000 [00:56<00:01, 7237.90it/s] 98%|█████████▊| 391140/400000 [00:56<00:01, 6917.47it/s] 98%|█████████▊| 391835/400000 [00:56<00:01, 6838.83it/s] 98%|█████████▊| 392522/400000 [00:56<00:01, 6786.79it/s] 98%|█████████▊| 393260/400000 [00:56<00:00, 6949.49it/s] 98%|█████████▊| 393969/400000 [00:57<00:00, 6990.37it/s] 99%|█████████▊| 394683/400000 [00:57<00:00, 7032.77it/s] 99%|█████████▉| 395426/400000 [00:57<00:00, 7145.21it/s] 99%|█████████▉| 396142/400000 [00:57<00:00, 7132.48it/s] 99%|█████████▉| 396889/400000 [00:57<00:00, 7229.08it/s] 99%|█████████▉| 397627/400000 [00:57<00:00, 7272.67it/s]100%|█████████▉| 398355/400000 [00:57<00:00, 7218.05it/s]100%|█████████▉| 399103/400000 [00:57<00:00, 7293.88it/s]100%|█████████▉| 399833/400000 [00:57<00:00, 7044.91it/s]100%|█████████▉| 399999/400000 [00:57<00:00, 6906.18it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7dc9fb60f0>, <torchtext.data.dataset.TabularDataset object at 0x7f7dc9fb6240>, <torchtext.vocab.Vocab object at 0x7f7dc9fb6160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.86 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.86 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.86 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.09 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.53 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.53 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.62 file/s]2020-06-09 00:19:16.410952: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-09 00:19:16.415507: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394445000 Hz
2020-06-09 00:19:16.415696: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556479eca7f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-09 00:19:16.415714: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 47%|████▋     | 4636672/9912422 [00:00<00:00, 46361900.66it/s]9920512it [00:00, 31478220.50it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 268990.25it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:01, 1549348.40it/s]          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:01<?, ?it/s]8192it [00:01, 6583.74it/s]             Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
