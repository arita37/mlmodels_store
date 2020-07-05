
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fbca99bd0d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fbca99bd0d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fbd14d071e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fbd14d071e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fbd3304ee18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fbd3304ee18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbcc2034620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbcc2034620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbcc2034620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 141558.88it/s] 79%|███████▉  | 7831552/9912422 [00:00<00:10, 202067.54it/s]9920512it [00:00, 42142663.57it/s]                           
0it [00:00, ?it/s]32768it [00:00, 621296.91it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472981.46it/s]1654784it [00:00, 12041344.60it/s]                         
0it [00:00, ?it/s]8192it [00:00, 200330.81it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbca9073978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbca9068c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fbcc2034268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fbcc2034268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fbcc2034268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.39 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.51 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:11, 10.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 14.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:07, 14.14 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.97 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 31.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 39.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 47.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 55.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 55.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 63.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 63.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 70.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 70.14 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 75.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 75.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 80.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 80.23 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 84.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.13s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.13s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.13s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.13s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.26 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.64 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.19s/ url]
0 examples [00:00, ? examples/s]2020-07-05 00:09:10.872118: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-05 00:09:10.887770: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-05 00:09:10.887961: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5558d5e94000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-05 00:09:10.887978: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
49 examples [00:00, 488.54 examples/s]160 examples [00:00, 586.56 examples/s]271 examples [00:00, 682.30 examples/s]384 examples [00:00, 773.07 examples/s]498 examples [00:00, 855.06 examples/s]609 examples [00:00, 916.55 examples/s]721 examples [00:00, 967.34 examples/s]829 examples [00:00, 996.22 examples/s]939 examples [00:00, 1024.13 examples/s]1054 examples [00:01, 1056.64 examples/s]1164 examples [00:01, 1066.66 examples/s]1272 examples [00:01, 1069.87 examples/s]1380 examples [00:01, 1060.01 examples/s]1487 examples [00:01, 1059.88 examples/s]1599 examples [00:01, 1077.22 examples/s]1709 examples [00:01, 1083.40 examples/s]1819 examples [00:01, 1085.47 examples/s]1933 examples [00:01, 1098.64 examples/s]2044 examples [00:01, 1101.31 examples/s]2155 examples [00:02, 1094.61 examples/s]2266 examples [00:02, 1098.08 examples/s]2378 examples [00:02, 1103.52 examples/s]2489 examples [00:02, 1094.47 examples/s]2599 examples [00:02, 1084.47 examples/s]2710 examples [00:02, 1091.20 examples/s]2822 examples [00:02, 1098.44 examples/s]2933 examples [00:02, 1099.83 examples/s]3044 examples [00:02, 1102.76 examples/s]3156 examples [00:02, 1105.97 examples/s]3267 examples [00:03, 1089.93 examples/s]3377 examples [00:03, 1070.71 examples/s]3485 examples [00:03, 1069.41 examples/s]3595 examples [00:03, 1074.85 examples/s]3703 examples [00:03, 1075.27 examples/s]3811 examples [00:03, 1072.92 examples/s]3920 examples [00:03, 1076.73 examples/s]4028 examples [00:03, 1071.29 examples/s]4139 examples [00:03, 1079.87 examples/s]4248 examples [00:03, 1065.13 examples/s]4355 examples [00:04, 1030.89 examples/s]4464 examples [00:04, 1047.53 examples/s]4572 examples [00:04, 1056.63 examples/s]4680 examples [00:04, 1061.52 examples/s]4793 examples [00:04, 1079.73 examples/s]4907 examples [00:04, 1096.15 examples/s]5017 examples [00:04, 1091.31 examples/s]5127 examples [00:04, 1093.30 examples/s]5237 examples [00:04, 1089.81 examples/s]5347 examples [00:04, 1092.35 examples/s]5457 examples [00:05, 1080.49 examples/s]5566 examples [00:05, 1057.83 examples/s]5677 examples [00:05, 1070.98 examples/s]5788 examples [00:05, 1081.29 examples/s]5901 examples [00:05, 1094.38 examples/s]6013 examples [00:05, 1100.09 examples/s]6126 examples [00:05, 1107.07 examples/s]6239 examples [00:05, 1113.36 examples/s]6352 examples [00:05, 1116.46 examples/s]6466 examples [00:05, 1121.03 examples/s]6579 examples [00:06, 1101.33 examples/s]6690 examples [00:06, 1067.27 examples/s]6801 examples [00:06, 1076.91 examples/s]6909 examples [00:06, 1073.45 examples/s]7020 examples [00:06, 1081.74 examples/s]7129 examples [00:06, 1072.48 examples/s]7237 examples [00:06, 1065.95 examples/s]7348 examples [00:06, 1076.86 examples/s]7458 examples [00:06, 1082.27 examples/s]7567 examples [00:07, 1067.06 examples/s]7676 examples [00:07, 1071.28 examples/s]7785 examples [00:07, 1075.34 examples/s]7894 examples [00:07, 1079.27 examples/s]8004 examples [00:07, 1083.60 examples/s]8114 examples [00:07, 1085.98 examples/s]8223 examples [00:07, 1073.34 examples/s]8335 examples [00:07, 1085.71 examples/s]8445 examples [00:07, 1086.95 examples/s]8555 examples [00:07, 1087.92 examples/s]8667 examples [00:08, 1097.26 examples/s]8778 examples [00:08, 1098.24 examples/s]8888 examples [00:08, 1068.04 examples/s]8996 examples [00:08, 1066.05 examples/s]9105 examples [00:08, 1073.04 examples/s]9218 examples [00:08, 1088.36 examples/s]9330 examples [00:08, 1095.73 examples/s]9441 examples [00:08, 1098.28 examples/s]9552 examples [00:08, 1100.93 examples/s]9663 examples [00:08, 1101.92 examples/s]9776 examples [00:09, 1108.11 examples/s]9887 examples [00:09, 1016.90 examples/s]9998 examples [00:09, 1042.76 examples/s]10104 examples [00:09, 1009.47 examples/s]10217 examples [00:09, 1042.21 examples/s]10328 examples [00:09, 1060.46 examples/s]10439 examples [00:09, 1074.37 examples/s]10548 examples [00:09, 1078.76 examples/s]10657 examples [00:09, 1068.29 examples/s]10765 examples [00:09, 1070.23 examples/s]10876 examples [00:10, 1079.75 examples/s]10987 examples [00:10, 1086.14 examples/s]11096 examples [00:10, 1075.07 examples/s]11205 examples [00:10, 1078.59 examples/s]11316 examples [00:10, 1086.50 examples/s]11429 examples [00:10, 1096.34 examples/s]11540 examples [00:10, 1098.32 examples/s]11650 examples [00:10, 1096.07 examples/s]11760 examples [00:10, 1080.60 examples/s]11869 examples [00:11, 1080.93 examples/s]11980 examples [00:11, 1088.15 examples/s]12091 examples [00:11, 1091.01 examples/s]12201 examples [00:11, 1068.44 examples/s]12308 examples [00:11, 1068.56 examples/s]12415 examples [00:11, 1066.18 examples/s]12529 examples [00:11, 1085.54 examples/s]12644 examples [00:11, 1102.21 examples/s]12755 examples [00:11, 1103.42 examples/s]12866 examples [00:11, 1099.02 examples/s]12976 examples [00:12, 1086.92 examples/s]13085 examples [00:12, 1072.45 examples/s]13193 examples [00:12, 1045.69 examples/s]13301 examples [00:12, 1054.20 examples/s]13413 examples [00:12, 1071.45 examples/s]13525 examples [00:12, 1083.15 examples/s]13635 examples [00:12, 1086.85 examples/s]13748 examples [00:12, 1098.40 examples/s]13858 examples [00:12, 1096.84 examples/s]13968 examples [00:12, 1095.58 examples/s]14078 examples [00:13, 1094.59 examples/s]14189 examples [00:13, 1096.88 examples/s]14299 examples [00:13, 1041.61 examples/s]14409 examples [00:13, 1056.04 examples/s]14517 examples [00:13, 1061.73 examples/s]14631 examples [00:13, 1081.56 examples/s]14742 examples [00:13, 1089.09 examples/s]14852 examples [00:13, 1091.02 examples/s]14965 examples [00:13, 1102.04 examples/s]15079 examples [00:13, 1110.58 examples/s]15194 examples [00:14, 1121.02 examples/s]15308 examples [00:14, 1123.94 examples/s]15421 examples [00:14, 1119.81 examples/s]15534 examples [00:14, 1118.01 examples/s]15646 examples [00:14, 1108.33 examples/s]15758 examples [00:14, 1111.15 examples/s]15872 examples [00:14, 1117.20 examples/s]15986 examples [00:14, 1122.42 examples/s]16099 examples [00:14, 1100.37 examples/s]16210 examples [00:14, 1099.81 examples/s]16321 examples [00:15, 1098.54 examples/s]16431 examples [00:15, 1096.69 examples/s]16541 examples [00:15, 1042.35 examples/s]16652 examples [00:15, 1061.48 examples/s]16759 examples [00:15, 1057.50 examples/s]16867 examples [00:15, 1062.26 examples/s]16979 examples [00:15, 1078.71 examples/s]17092 examples [00:15, 1092.74 examples/s]17202 examples [00:15, 1092.55 examples/s]17314 examples [00:16, 1099.06 examples/s]17426 examples [00:16, 1103.28 examples/s]17538 examples [00:16, 1105.54 examples/s]17649 examples [00:16, 1081.70 examples/s]17758 examples [00:16, 1045.45 examples/s]17865 examples [00:16, 1052.05 examples/s]17975 examples [00:16, 1065.68 examples/s]18082 examples [00:16, 1063.22 examples/s]18192 examples [00:16, 1071.62 examples/s]18302 examples [00:16, 1078.52 examples/s]18410 examples [00:17, 1078.87 examples/s]18522 examples [00:17, 1087.88 examples/s]18631 examples [00:17, 1085.90 examples/s]18740 examples [00:17, 1080.42 examples/s]18849 examples [00:17, 1076.25 examples/s]18959 examples [00:17, 1081.76 examples/s]19068 examples [00:17, 1082.56 examples/s]19179 examples [00:17, 1088.92 examples/s]19288 examples [00:17, 1066.41 examples/s]19396 examples [00:17, 1069.43 examples/s]19505 examples [00:18, 1075.04 examples/s]19617 examples [00:18, 1085.31 examples/s]19726 examples [00:18, 1081.38 examples/s]19835 examples [00:18, 1024.89 examples/s]19939 examples [00:18, 1019.64 examples/s]20042 examples [00:18, 987.62 examples/s] 20151 examples [00:18, 1015.65 examples/s]20254 examples [00:18, 1011.95 examples/s]20366 examples [00:18, 1039.77 examples/s]20471 examples [00:18, 1034.95 examples/s]20581 examples [00:19, 1053.43 examples/s]20696 examples [00:19, 1077.74 examples/s]20808 examples [00:19, 1090.04 examples/s]20921 examples [00:19, 1100.00 examples/s]21032 examples [00:19, 1101.94 examples/s]21143 examples [00:19, 1099.40 examples/s]21257 examples [00:19, 1109.89 examples/s]21370 examples [00:19, 1115.59 examples/s]21483 examples [00:19, 1119.16 examples/s]21596 examples [00:19, 1122.20 examples/s]21709 examples [00:20, 1119.34 examples/s]21822 examples [00:20, 1121.75 examples/s]21935 examples [00:20, 1104.29 examples/s]22046 examples [00:20, 1102.51 examples/s]22157 examples [00:20, 1085.05 examples/s]22268 examples [00:20, 1091.11 examples/s]22380 examples [00:20, 1098.49 examples/s]22490 examples [00:20, 1067.20 examples/s]22597 examples [00:20, 1039.24 examples/s]22707 examples [00:21, 1056.07 examples/s]22815 examples [00:21, 1061.20 examples/s]22923 examples [00:21, 1066.27 examples/s]23032 examples [00:21, 1072.83 examples/s]23140 examples [00:21, 1044.46 examples/s]23245 examples [00:21, 1044.87 examples/s]23356 examples [00:21, 1062.26 examples/s]23464 examples [00:21, 1066.15 examples/s]23574 examples [00:21, 1076.03 examples/s]23685 examples [00:21, 1085.67 examples/s]23794 examples [00:22, 1081.76 examples/s]23903 examples [00:22, 1077.92 examples/s]24011 examples [00:22, 1076.34 examples/s]24123 examples [00:22, 1088.87 examples/s]24232 examples [00:22, 1040.76 examples/s]24338 examples [00:22, 1046.13 examples/s]24443 examples [00:22, 1044.46 examples/s]24552 examples [00:22, 1057.34 examples/s]24662 examples [00:22, 1068.36 examples/s]24770 examples [00:22, 1069.64 examples/s]24879 examples [00:23, 1075.17 examples/s]24989 examples [00:23, 1081.05 examples/s]25098 examples [00:23, 1067.66 examples/s]25205 examples [00:23, 1064.96 examples/s]25315 examples [00:23, 1073.62 examples/s]25423 examples [00:23, 1069.34 examples/s]25533 examples [00:23, 1076.52 examples/s]25645 examples [00:23, 1087.97 examples/s]25756 examples [00:23, 1093.71 examples/s]25866 examples [00:23, 1075.04 examples/s]25977 examples [00:24, 1083.06 examples/s]26086 examples [00:24, 1081.39 examples/s]26195 examples [00:24, 1049.32 examples/s]26305 examples [00:24, 1063.95 examples/s]26412 examples [00:24, 1040.49 examples/s]26523 examples [00:24, 1039.19 examples/s]26628 examples [00:24, 1036.66 examples/s]26734 examples [00:24, 1040.83 examples/s]26845 examples [00:24, 1060.17 examples/s]26956 examples [00:25, 1072.70 examples/s]27066 examples [00:25, 1080.14 examples/s]27177 examples [00:25, 1087.23 examples/s]27287 examples [00:25, 1090.94 examples/s]27397 examples [00:25, 1067.93 examples/s]27508 examples [00:25, 1078.01 examples/s]27618 examples [00:25, 1083.64 examples/s]27727 examples [00:25, 1080.31 examples/s]27836 examples [00:25, 1078.51 examples/s]27947 examples [00:25, 1085.38 examples/s]28056 examples [00:26, 1081.04 examples/s]28166 examples [00:26, 1084.65 examples/s]28276 examples [00:26, 1087.32 examples/s]28385 examples [00:26, 1079.68 examples/s]28495 examples [00:26, 1085.59 examples/s]28605 examples [00:26, 1089.23 examples/s]28716 examples [00:26, 1094.00 examples/s]28827 examples [00:26, 1096.63 examples/s]28937 examples [00:26, 1093.77 examples/s]29047 examples [00:26, 1093.84 examples/s]29157 examples [00:27, 1090.42 examples/s]29267 examples [00:27, 1064.53 examples/s]29374 examples [00:27, 1055.23 examples/s]29485 examples [00:27, 1069.33 examples/s]29595 examples [00:27, 1076.53 examples/s]29703 examples [00:27, 1022.94 examples/s]29811 examples [00:27, 1038.50 examples/s]29916 examples [00:27, 1011.84 examples/s]30018 examples [00:27, 978.53 examples/s] 30127 examples [00:27, 1008.42 examples/s]30233 examples [00:28, 1022.72 examples/s]30337 examples [00:28, 1024.78 examples/s]30440 examples [00:28, 1008.59 examples/s]30544 examples [00:28, 1015.24 examples/s]30646 examples [00:28, 999.97 examples/s] 30752 examples [00:28, 1016.75 examples/s]30857 examples [00:28, 1024.97 examples/s]30960 examples [00:28, 1019.47 examples/s]31063 examples [00:28, 991.68 examples/s] 31168 examples [00:29, 1008.18 examples/s]31270 examples [00:29, 1009.02 examples/s]31372 examples [00:29, 1009.11 examples/s]31479 examples [00:29, 1024.93 examples/s]31583 examples [00:29, 1028.56 examples/s]31693 examples [00:29, 1046.84 examples/s]31801 examples [00:29, 1054.17 examples/s]31908 examples [00:29, 1056.96 examples/s]32014 examples [00:29, 985.71 examples/s] 32114 examples [00:29, 969.46 examples/s]32212 examples [00:30, 962.52 examples/s]32312 examples [00:30, 972.70 examples/s]32411 examples [00:30, 976.62 examples/s]32510 examples [00:30, 980.05 examples/s]32609 examples [00:30, 980.11 examples/s]32708 examples [00:30, 957.09 examples/s]32806 examples [00:30, 962.37 examples/s]32905 examples [00:30, 969.22 examples/s]33009 examples [00:30, 988.32 examples/s]33117 examples [00:30, 1012.77 examples/s]33228 examples [00:31, 1038.42 examples/s]33333 examples [00:31, 1009.56 examples/s]33436 examples [00:31, 1015.22 examples/s]33541 examples [00:31, 1023.20 examples/s]33644 examples [00:31, 981.92 examples/s] 33751 examples [00:31, 1005.30 examples/s]33859 examples [00:31, 1026.01 examples/s]33966 examples [00:31, 1037.02 examples/s]34071 examples [00:31, 1035.63 examples/s]34177 examples [00:32, 1042.51 examples/s]34284 examples [00:32, 1050.18 examples/s]34394 examples [00:32, 1062.46 examples/s]34501 examples [00:32, 1063.78 examples/s]34608 examples [00:32, 1062.56 examples/s]34715 examples [00:32, 1064.50 examples/s]34823 examples [00:32, 1068.61 examples/s]34930 examples [00:32, 1059.50 examples/s]35036 examples [00:32, 1055.12 examples/s]35142 examples [00:32, 1046.38 examples/s]35248 examples [00:33, 1048.63 examples/s]35356 examples [00:33, 1055.71 examples/s]35462 examples [00:33, 1056.91 examples/s]35570 examples [00:33, 1062.84 examples/s]35679 examples [00:33, 1069.07 examples/s]35787 examples [00:33, 1070.42 examples/s]35896 examples [00:33, 1074.68 examples/s]36004 examples [00:33, 1075.17 examples/s]36112 examples [00:33, 1047.28 examples/s]36217 examples [00:33, 1047.55 examples/s]36326 examples [00:34, 1058.42 examples/s]36432 examples [00:34, 1037.19 examples/s]36541 examples [00:34, 1052.29 examples/s]36652 examples [00:34, 1067.65 examples/s]36763 examples [00:34, 1079.93 examples/s]36875 examples [00:34, 1089.10 examples/s]36987 examples [00:34, 1095.40 examples/s]37097 examples [00:34, 1088.49 examples/s]37207 examples [00:34, 1090.75 examples/s]37317 examples [00:34, 1088.30 examples/s]37429 examples [00:35, 1095.44 examples/s]37539 examples [00:35, 1090.18 examples/s]37649 examples [00:35, 1071.05 examples/s]37757 examples [00:35, 1060.72 examples/s]37867 examples [00:35, 1071.04 examples/s]37975 examples [00:35, 1070.67 examples/s]38083 examples [00:35, 1066.16 examples/s]38190 examples [00:35, 1046.74 examples/s]38295 examples [00:35, 1043.43 examples/s]38400 examples [00:35, 1023.48 examples/s]38503 examples [00:36, 1020.69 examples/s]38606 examples [00:36, 979.25 examples/s] 38705 examples [00:36, 970.98 examples/s]38815 examples [00:36, 1004.81 examples/s]38925 examples [00:36, 1029.58 examples/s]39034 examples [00:36, 1045.89 examples/s]39145 examples [00:36, 1064.09 examples/s]39253 examples [00:36, 1061.35 examples/s]39360 examples [00:36, 1035.76 examples/s]39472 examples [00:37, 1058.92 examples/s]39580 examples [00:37, 1064.44 examples/s]39687 examples [00:37, 1050.20 examples/s]39793 examples [00:37, 1040.69 examples/s]39898 examples [00:37, 1034.28 examples/s]40002 examples [00:37, 1005.11 examples/s]40113 examples [00:37, 1032.82 examples/s]40223 examples [00:37, 1049.08 examples/s]40329 examples [00:37, 1048.62 examples/s]40437 examples [00:37, 1057.67 examples/s]40543 examples [00:38, 1057.59 examples/s]40649 examples [00:38, 1045.64 examples/s]40757 examples [00:38, 1053.22 examples/s]40866 examples [00:38, 1062.45 examples/s]40975 examples [00:38, 1069.20 examples/s]41085 examples [00:38, 1075.51 examples/s]41194 examples [00:38, 1077.07 examples/s]41306 examples [00:38, 1089.45 examples/s]41416 examples [00:38, 1081.21 examples/s]41525 examples [00:38, 1077.32 examples/s]41636 examples [00:39, 1086.41 examples/s]41748 examples [00:39, 1095.14 examples/s]41858 examples [00:39, 1091.25 examples/s]41968 examples [00:39, 1088.39 examples/s]42078 examples [00:39, 1091.05 examples/s]42189 examples [00:39, 1094.04 examples/s]42299 examples [00:39, 1089.95 examples/s]42409 examples [00:39, 1087.28 examples/s]42520 examples [00:39, 1093.81 examples/s]42630 examples [00:39, 1086.47 examples/s]42742 examples [00:40, 1093.79 examples/s]42854 examples [00:40, 1099.07 examples/s]42964 examples [00:40, 1054.36 examples/s]43071 examples [00:40, 1057.99 examples/s]43181 examples [00:40, 1068.48 examples/s]43290 examples [00:40, 1074.26 examples/s]43402 examples [00:40, 1085.50 examples/s]43511 examples [00:40, 1086.72 examples/s]43620 examples [00:40, 1074.91 examples/s]43730 examples [00:40, 1081.27 examples/s]43842 examples [00:41, 1089.91 examples/s]43955 examples [00:41, 1100.00 examples/s]44066 examples [00:41, 1102.30 examples/s]44177 examples [00:41, 1084.84 examples/s]44287 examples [00:41, 1088.75 examples/s]44402 examples [00:41, 1103.52 examples/s]44515 examples [00:41, 1108.48 examples/s]44627 examples [00:41, 1111.23 examples/s]44739 examples [00:41, 1110.48 examples/s]44852 examples [00:41, 1113.36 examples/s]44964 examples [00:42, 1112.35 examples/s]45076 examples [00:42, 1109.63 examples/s]45187 examples [00:42, 1096.19 examples/s]45298 examples [00:42, 1100.00 examples/s]45409 examples [00:42, 1101.65 examples/s]45520 examples [00:42, 1099.89 examples/s]45633 examples [00:42, 1106.20 examples/s]45747 examples [00:42, 1115.67 examples/s]45859 examples [00:42, 1088.39 examples/s]45969 examples [00:43, 1083.18 examples/s]46078 examples [00:43, 1085.09 examples/s]46187 examples [00:43, 1033.79 examples/s]46291 examples [00:43, 1017.58 examples/s]46402 examples [00:43, 1042.37 examples/s]46508 examples [00:43, 1046.02 examples/s]46613 examples [00:43, 1029.94 examples/s]46725 examples [00:43, 1053.53 examples/s]46836 examples [00:43, 1069.75 examples/s]46944 examples [00:43, 1070.53 examples/s]47055 examples [00:44, 1079.88 examples/s]47166 examples [00:44, 1088.51 examples/s]47277 examples [00:44, 1091.83 examples/s]47387 examples [00:44, 1091.29 examples/s]47497 examples [00:44, 1092.46 examples/s]47608 examples [00:44, 1096.44 examples/s]47720 examples [00:44, 1101.18 examples/s]47831 examples [00:44, 1103.22 examples/s]47942 examples [00:44, 1096.87 examples/s]48054 examples [00:44, 1102.58 examples/s]48165 examples [00:45, 1090.86 examples/s]48275 examples [00:45, 1088.21 examples/s]48384 examples [00:45, 1085.05 examples/s]48494 examples [00:45, 1088.19 examples/s]48603 examples [00:45, 1088.58 examples/s]48712 examples [00:45, 1088.51 examples/s]48825 examples [00:45, 1099.83 examples/s]48937 examples [00:45, 1105.78 examples/s]49048 examples [00:45, 1097.14 examples/s]49158 examples [00:45, 1093.50 examples/s]49268 examples [00:46, 1092.37 examples/s]49380 examples [00:46, 1100.50 examples/s]49491 examples [00:46, 1094.98 examples/s]49601 examples [00:46, 1053.22 examples/s]49711 examples [00:46, 1066.63 examples/s]49818 examples [00:46, 1027.48 examples/s]49928 examples [00:46, 1046.14 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7340/50000 [00:00<00:00, 73399.37 examples/s] 41%|████      | 20453/50000 [00:00<00:00, 84568.29 examples/s] 67%|██████▋   | 33587/50000 [00:00<00:00, 94683.50 examples/s] 94%|█████████▍| 46879/50000 [00:00<00:00, 103625.99 examples/s]                                                                0 examples [00:00, ? examples/s]86 examples [00:00, 858.84 examples/s]197 examples [00:00, 920.09 examples/s]306 examples [00:00, 964.02 examples/s]419 examples [00:00, 1007.79 examples/s]530 examples [00:00, 1036.09 examples/s]642 examples [00:00, 1058.72 examples/s]752 examples [00:00, 1067.70 examples/s]857 examples [00:00, 1060.43 examples/s]970 examples [00:00, 1077.99 examples/s]1081 examples [00:01, 1086.89 examples/s]1192 examples [00:01, 1092.22 examples/s]1303 examples [00:01, 1096.89 examples/s]1415 examples [00:01, 1102.21 examples/s]1525 examples [00:01, 1092.91 examples/s]1636 examples [00:01, 1097.46 examples/s]1747 examples [00:01, 1099.99 examples/s]1857 examples [00:01, 1085.32 examples/s]1966 examples [00:01, 1083.38 examples/s]2078 examples [00:01, 1092.48 examples/s]2189 examples [00:02, 1095.36 examples/s]2300 examples [00:02, 1099.63 examples/s]2410 examples [00:02, 1057.79 examples/s]2519 examples [00:02, 1065.35 examples/s]2626 examples [00:02, 1010.87 examples/s]2737 examples [00:02, 1036.66 examples/s]2844 examples [00:02, 1034.58 examples/s]2955 examples [00:02, 1055.75 examples/s]3064 examples [00:02, 1065.14 examples/s]3175 examples [00:02, 1077.15 examples/s]3285 examples [00:03, 1082.49 examples/s]3397 examples [00:03, 1092.24 examples/s]3508 examples [00:03, 1096.09 examples/s]3618 examples [00:03, 1095.41 examples/s]3729 examples [00:03, 1099.70 examples/s]3840 examples [00:03, 1102.21 examples/s]3951 examples [00:03, 1100.07 examples/s]4062 examples [00:03, 1101.66 examples/s]4173 examples [00:03, 1102.43 examples/s]4284 examples [00:03, 1089.95 examples/s]4395 examples [00:04, 1094.90 examples/s]4505 examples [00:04, 1093.86 examples/s]4617 examples [00:04, 1098.40 examples/s]4728 examples [00:04, 1098.96 examples/s]4840 examples [00:04, 1103.18 examples/s]4954 examples [00:04, 1111.26 examples/s]5066 examples [00:04, 1107.30 examples/s]5177 examples [00:04, 1105.44 examples/s]5288 examples [00:04, 1094.79 examples/s]5398 examples [00:04, 1089.40 examples/s]5510 examples [00:05, 1096.22 examples/s]5620 examples [00:05, 1087.09 examples/s]5729 examples [00:05, 1049.11 examples/s]5839 examples [00:05, 1063.45 examples/s]5946 examples [00:05, 1034.95 examples/s]6061 examples [00:05, 1066.94 examples/s]6175 examples [00:05, 1085.15 examples/s]6289 examples [00:05, 1098.91 examples/s]6401 examples [00:05, 1102.55 examples/s]6512 examples [00:06, 1085.31 examples/s]6625 examples [00:06, 1096.83 examples/s]6737 examples [00:06, 1102.25 examples/s]6849 examples [00:06, 1105.07 examples/s]6960 examples [00:06, 1102.57 examples/s]7071 examples [00:06, 1073.69 examples/s]7179 examples [00:06, 1072.09 examples/s]7290 examples [00:06, 1082.58 examples/s]7400 examples [00:06, 1087.03 examples/s]7509 examples [00:06, 1085.49 examples/s]7618 examples [00:07, 1057.47 examples/s]7724 examples [00:07, 1047.78 examples/s]7833 examples [00:07, 1059.43 examples/s]7942 examples [00:07, 1067.51 examples/s]8051 examples [00:07, 1072.61 examples/s]8160 examples [00:07, 1075.54 examples/s]8271 examples [00:07, 1085.03 examples/s]8383 examples [00:07, 1093.17 examples/s]8495 examples [00:07, 1099.58 examples/s]8606 examples [00:07, 1099.85 examples/s]8717 examples [00:08, 1082.23 examples/s]8826 examples [00:08, 1076.18 examples/s]8937 examples [00:08, 1083.65 examples/s]9046 examples [00:08, 1061.38 examples/s]9153 examples [00:08, 1043.12 examples/s]9264 examples [00:08, 1060.15 examples/s]9376 examples [00:08, 1075.09 examples/s]9488 examples [00:08, 1086.38 examples/s]9600 examples [00:08, 1093.52 examples/s]9710 examples [00:08, 1087.32 examples/s]9824 examples [00:09, 1100.49 examples/s]9935 examples [00:09, 1097.75 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteBCU15Y/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteBCU15Y/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbcc2034620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbcc2034620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbcc2034620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbc5c256278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbc5c28df98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fbcc2034620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fbcc2034620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fbcc2034620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbca90685f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fbca9068390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fbc39b71158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fbc39b71158> 

  function with postional parmater data_info <function split_train_valid at 0x7fbc39b71158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fbc39b71268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fbc39b71268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fbc39b71268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=943941eb799572155d3cefd331573761ac80a2592ef0adeb3075d39222b75084
  Stored in directory: /tmp/pip-ephem-wheel-cache-8qgwoppw/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:08:47, 13.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<12:56:07, 18.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:06:31, 26.3kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:23:05, 37.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:01<4:27:31, 53.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.29M/862M [00:01<3:06:07, 76.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:09:31, 109kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<1:30:09, 156kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:01<1:02:45, 222kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:01<43:54, 316kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:01<30:35, 451kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:01<21:28, 639kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.7M/862M [00:02<15:02, 908kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<10:34, 1.28MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:02<07:26, 1.81MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:02<08:26, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:04<07:48, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<09:47, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.4M/862M [00:05<07:47, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<05:39, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<07:37, 1.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<07:01, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<05:16, 2.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<06:19, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<07:28, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.0M/862M [00:09<05:53, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<04:19, 3.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<07:40, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<06:45, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<05:00, 2.63MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<06:30, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<07:12, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<05:36, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<04:04, 3.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<10:27, 1.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<08:39, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<06:19, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<07:30, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<07:54, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<06:11, 2.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:17<04:27, 2.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<1:36:49, 134kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<1:09:03, 187kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:18<48:34, 266kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<36:56, 349kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<28:26, 452kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.9M/862M [00:20<20:26, 629kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:21<14:38, 876kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:21<10:43, 1.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<12:41, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<15:46, 811kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<12:48, 999kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:23<09:27, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:23<06:59, 1.83MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<08:45, 1.45MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<12:58, 981kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<10:50, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:25<08:04, 1.58MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<05:57, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:07, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:18, 1.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:22, 1.72MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:29, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<04:11, 3.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:32, 1.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:09, 890kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<11:38, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<08:34, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:21, 1.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:15, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<12:30, 1.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<10:30, 1.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<07:45, 1.61MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:47, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:28, 2.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<10:30, 1.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<14:07, 881kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:33, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:34, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:18, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:15, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:25, 997kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<10:21, 1.20MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<07:39, 1.62MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:42, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:52, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<12:07, 1.02MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:08, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:34, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<05:38, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:50, 1.56MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<12:03, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<10:05, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:31, 1.62MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<05:35, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:56, 1.53MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:08, 1.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<10:08, 1.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<07:33, 1.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:02, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:33, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:39, 1.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<07:12, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<05:19, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:05, 1.49MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<11:29, 1.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:17, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:56, 1.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<05:08, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:23, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<11:40, 1.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<09:33, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<07:04, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<05:11, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:41, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:01, 991kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:46, 1.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<07:13, 1.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<05:16, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:03, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<12:56, 915kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<10:17, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<07:30, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<05:27, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<10:42, 1.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:42, 926kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<10:08, 1.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<07:26, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<05:25, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:52, 844kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<14:25, 811kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<11:07, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<08:05, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<05:51, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:33:19, 125kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:09:38, 167kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<49:38, 234kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<35:00, 331kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<24:45, 468kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<22:11, 521kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<19:30, 592kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<14:38, 789kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<10:30, 1.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<10:02, 1.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<11:01, 1.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<08:30, 1.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<06:13, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:10, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<08:42, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:53, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<05:05, 2.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:25, 1.77MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<08:26, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:49, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<05:00, 2.26MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<06:18, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:37, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:59, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:24, 2.55MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:40, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<07:40, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:58, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:20, 2.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:46, 1.64MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:26, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:52, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:17, 2.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:48, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<08:37, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:40, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:50, 2.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:41, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<07:06, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:28, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:14<03:58, 2.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:01, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:55, 1.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:00, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:19, 2.51MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<09:56, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<09:03, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<06:49, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:18<04:55, 2.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<15:41, 689kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<13:35, 795kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<10:04, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<07:12, 1.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:19, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:00, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:13, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:29, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<08:02, 1.32MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<10:47, 988kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:51, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:30, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<06:35, 1.61MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:57, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<05:24, 1.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:55, 2.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<07:29, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<07:41, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:55, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<04:16, 2.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<07:25, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<07:36, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:57, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:17, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<08:46, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<08:32, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<06:29, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:32<04:41, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:32, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<09:10, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:02, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<05:03, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<09:10, 1.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<07:44, 1.32MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:48, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<04:10, 2.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<12:33, 810kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<12:55, 787kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<10:04, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:15, 1.40MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<07:20, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:32, 1.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:51, 2.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<03:31, 2.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<14:01, 716kB/s] .vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<11:42, 857kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<08:38, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:41, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<07:14, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:30, 1.81MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<05:30, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:34, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<06:09, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:32, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:33, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:39, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:20, 2.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:08, 3.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<10:50, 900kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<11:12, 871kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<08:44, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:17, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:46, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:32, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<05:01, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:35, 2.68MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<49:32, 194kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<36:29, 264kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<25:56, 370kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<19:40, 486kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<15:28, 617kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<11:15, 848kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<09:25, 1.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<10:10, 932kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:51, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<05:39, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:17, 1.50MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<06:06, 1.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:37, 2.03MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:20, 2.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<08:35, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<07:41, 1.22MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:47, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:36, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:35, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:19, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:33, 2.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:55, 1.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:51, 2.38MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:13, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<06:14, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:43, 2.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:43, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:55, 1.84MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:51, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:12, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:33, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<03:32, 2.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<02:33, 3.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<18:23, 486kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<14:28, 617kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<10:27, 853kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<07:24, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<10:09, 873kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<10:25, 851kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<07:58, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<05:44, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<06:03, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:01, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:40, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:54, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:40, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:24, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<03:58, 2.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:53, 1.77MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:59, 1.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:49, 2.26MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:45, 3.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<10:25, 824kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<10:30, 818kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<08:02, 1.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<05:48, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:09, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:49, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:24, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:10, 2.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<07:18, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<08:15, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<06:34, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:45, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:24, 1.55MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:18, 1.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:05, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:14, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<05:51, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:46, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:31, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:30, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:38, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:36, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:53, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:13, 1.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:17, 2.48MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:39, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:35, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:34, 1.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:23, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:19, 1.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:28, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:26, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<03:45, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:24, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:24, 1.81MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:14, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<04:02, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:14, 1.86MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:16, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:23, 3.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:04, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<05:39, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:18, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:18, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:23, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:22, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:28, 3.12MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:51, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:47, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:38, 2.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:37, 2.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<07:43, 988kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<06:46, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:04, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:48, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:58, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:53, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:33, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:20, 1.72MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:23, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<03:24, 2.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:36, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:41, 2.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<02:55, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<02:07, 3.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<08:18, 885kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<07:07, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<05:19, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:55, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<04:33, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:28, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:38, 2.74MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:44, 1.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:54, 1.85MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:03, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:19, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:59, 1.43MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:02, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:59, 2.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:49, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:59, 1.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:06, 2.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:20, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:58, 1.41MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:02, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:59, 2.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:46, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<03:53, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:02, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:16, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:50, 1.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:56, 1.74MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:53, 2.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:33, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:44, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:55, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<02:06, 3.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<14:39, 459kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<11:31, 584kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<08:18, 809kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<05:52, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<08:42, 766kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:43, 980kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<05:46, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<04:15, 1.55MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<03:03, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:06, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:26, 1.20MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:05, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:56, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<05:11, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:07, 1.56MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:00, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:29, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:37, 1.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:49, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:01, 2.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:17, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:35, 2.44MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:51, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:19, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:31, 1.77MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:37, 2.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:01, 2.04MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:14, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:31, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<01:49, 3.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<05:44, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<05:06, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:50, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:41, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:39, 1.65MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:49, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:36<02:01, 2.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<1:29:01, 67.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<1:03:21, 94.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<44:31, 134kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<31:53, 185kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<23:22, 253kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<16:33, 356kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<11:39, 503kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<09:59, 585kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<08:02, 726kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<05:50, 997kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<04:07, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<08:29, 679kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<07:00, 822kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:06, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<03:39, 1.56MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:45, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:23, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:19, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:15, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:19, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<02:34, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:43, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:56, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:16, 2.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:38, 3.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<05:41, 966kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<06:01, 911kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<04:38, 1.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:21, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:41, 1.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:34, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:44, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:48, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:55, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:16, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:28, 2.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:40, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:06, 2.50MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:20, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<03:26, 1.52MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:53, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:07, 2.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:45, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:51, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:13, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:23, 2.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:35, 1.96MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:02, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:15, 2.22MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:28, 2.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:57, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:11, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:24, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:54, 2.58MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:08, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:22, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:50, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<01:20, 3.58MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:36, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:19, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<03:27, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<02:31, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:56, 1.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:56, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:13, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:36, 2.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:25, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:57, 1.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:58, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:11, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:16, 1.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:16, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:42, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:59, 2.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:29, 1.79MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:32, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:56, 2.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:24, 3.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:27, 984kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:54, 1.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:55, 1.50MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:45, 1.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:42, 1.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:04, 2.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:09, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:14, 1.90MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:43, 2.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<01:14, 3.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<09:45, 429kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<08:23, 499kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<06:14, 669kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<04:25, 938kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:06, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:36, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:40, 1.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:54, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<06:26, 628kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<06:00, 673kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:33, 887kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:15, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:14, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:00, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:15, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:13, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:15, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:45, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:51, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:43, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:13, 1.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<01:38, 2.32MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:07, 1.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:39, 2.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:46, 2.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:36, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:09, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:35, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:00, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:04, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:35, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:09, 3.10MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:24, 1.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:19, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:46, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:48, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<02:34, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:07, 1.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:32, 2.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:54, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:56, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:30, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:36, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:43, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:19, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<00:57, 3.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<03:19, 993kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:54, 1.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:10, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:02, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:01, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:33, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:36, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:42, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:20, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:26, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:09, 1.43MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:47, 1.72MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:18, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:39, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:42, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:18, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:24, 2.11MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:04, 1.42MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:43, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:14, 2.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:35, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:38, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:15, 2.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<00:54, 3.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:16, 1.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:37, 1.07MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:05, 1.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:30, 1.83MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:41, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:20, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:57, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:36, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:10, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:27, 1.79MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:29, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:07, 2.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:48, 3.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:16, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:02, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:32, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:28, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:29, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:11, 2.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:15, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:59, 2.42MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:04, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:37, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:20, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:58, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:14, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:16, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:58, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:41, 3.18MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<03:08, 698kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:35, 846kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:53, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:46, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:24, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:09, 1.77MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:50, 2.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:09, 1.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:29, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:13, 1.61MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:05, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:06, 1.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:51, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:53, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:18, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:05, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:47, 2.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:58, 1.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:00, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:46, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:48, 2.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:12, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:00, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:43, 2.31MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:54, 1.82MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:55, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:42, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:45, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:48, 1.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:37, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:44<00:26, 3.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:12, 682kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:06, 715kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:36, 932kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:07, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:07, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:02, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:46, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:32, 2.51MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:09, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:03, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:47, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:32, 2.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:53, 685kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:33, 828kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:08, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:58, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<01:08, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:53, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:43, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:42, 1.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:32, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:32, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:47, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:38, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:27, 2.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:33, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:34, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:26, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:27, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:40, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:29, 1.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:29, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:22, 2.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:34, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:27, 1.75MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:19, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:24, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:25, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:19, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:13, 3.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:53, 757kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:44, 901kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:32, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:27, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:25, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:19, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:17, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:24, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:19, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:13, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:16, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:12, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:17, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:13, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.92MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:05, 2.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:03, 3.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:13, 881kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:13, 858kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:10, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:06, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:05, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 1.94MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:01, 2.67MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 1.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 938kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 1.65MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 273/400000 [00:00<02:26, 2728.29it/s]  0%|          | 1196/400000 [00:00<01:55, 3459.22it/s]  1%|          | 2132/400000 [00:00<01:33, 4265.69it/s]  1%|          | 3024/400000 [00:00<01:18, 5056.80it/s]  1%|          | 3936/400000 [00:00<01:07, 5836.48it/s]  1%|          | 4876/400000 [00:00<01:00, 6584.26it/s]  1%|▏         | 5805/400000 [00:00<00:54, 7213.18it/s]  2%|▏         | 6768/400000 [00:00<00:50, 7799.88it/s]  2%|▏         | 7680/400000 [00:00<00:48, 8152.87it/s]  2%|▏         | 8612/400000 [00:01<00:46, 8470.22it/s]  2%|▏         | 9558/400000 [00:01<00:44, 8742.60it/s]  3%|▎         | 10469/400000 [00:01<00:44, 8826.32it/s]  3%|▎         | 11459/400000 [00:01<00:42, 9122.54it/s]  3%|▎         | 12411/400000 [00:01<00:41, 9235.52it/s]  3%|▎         | 13350/400000 [00:01<00:42, 9033.90it/s]  4%|▎         | 14265/400000 [00:01<00:43, 8925.01it/s]  4%|▍         | 15168/400000 [00:01<00:42, 8954.15it/s]  4%|▍         | 16070/400000 [00:01<00:44, 8713.06it/s]  4%|▍         | 16960/400000 [00:01<00:43, 8740.57it/s]  4%|▍         | 17871/400000 [00:02<00:43, 8847.74it/s]  5%|▍         | 18835/400000 [00:02<00:42, 9071.28it/s]  5%|▍         | 19827/400000 [00:02<00:40, 9308.08it/s]  5%|▌         | 20805/400000 [00:02<00:40, 9442.81it/s]  5%|▌         | 21753/400000 [00:02<00:40, 9297.95it/s]  6%|▌         | 22708/400000 [00:02<00:40, 9371.72it/s]  6%|▌         | 23702/400000 [00:02<00:39, 9534.91it/s]  6%|▌         | 24658/400000 [00:02<00:39, 9525.51it/s]  6%|▋         | 25625/400000 [00:02<00:39, 9566.50it/s]  7%|▋         | 26583/400000 [00:02<00:39, 9526.45it/s]  7%|▋         | 27548/400000 [00:03<00:38, 9561.93it/s]  7%|▋         | 28546/400000 [00:03<00:38, 9681.57it/s]  7%|▋         | 29515/400000 [00:03<00:38, 9653.38it/s]  8%|▊         | 30481/400000 [00:03<00:38, 9597.63it/s]  8%|▊         | 31442/400000 [00:03<00:38, 9580.02it/s]  8%|▊         | 32401/400000 [00:03<00:38, 9510.02it/s]  8%|▊         | 33353/400000 [00:03<00:38, 9404.40it/s]  9%|▊         | 34343/400000 [00:03<00:38, 9545.27it/s]  9%|▉         | 35320/400000 [00:03<00:37, 9610.88it/s]  9%|▉         | 36282/400000 [00:03<00:38, 9389.77it/s]  9%|▉         | 37232/400000 [00:04<00:38, 9421.87it/s] 10%|▉         | 38207/400000 [00:04<00:38, 9516.73it/s] 10%|▉         | 39180/400000 [00:04<00:37, 9577.17it/s] 10%|█         | 40139/400000 [00:04<00:37, 9560.74it/s] 10%|█         | 41096/400000 [00:04<00:37, 9455.67it/s] 11%|█         | 42066/400000 [00:04<00:37, 9524.65it/s] 11%|█         | 43028/400000 [00:04<00:37, 9552.59it/s] 11%|█         | 43984/400000 [00:04<00:37, 9492.70it/s] 11%|█         | 44934/400000 [00:04<00:37, 9377.95it/s] 11%|█▏        | 45873/400000 [00:04<00:38, 9211.24it/s] 12%|█▏        | 46796/400000 [00:05<00:38, 9134.66it/s] 12%|█▏        | 47711/400000 [00:05<00:38, 9129.67it/s] 12%|█▏        | 48648/400000 [00:05<00:38, 9200.16it/s] 12%|█▏        | 49584/400000 [00:05<00:37, 9244.86it/s] 13%|█▎        | 50509/400000 [00:05<00:38, 9089.19it/s] 13%|█▎        | 51479/400000 [00:05<00:37, 9262.23it/s] 13%|█▎        | 52454/400000 [00:05<00:36, 9401.81it/s] 13%|█▎        | 53451/400000 [00:05<00:36, 9563.39it/s] 14%|█▎        | 54413/400000 [00:05<00:36, 9579.68it/s] 14%|█▍        | 55373/400000 [00:05<00:36, 9570.83it/s] 14%|█▍        | 56347/400000 [00:06<00:35, 9619.36it/s] 14%|█▍        | 57320/400000 [00:06<00:35, 9650.82it/s] 15%|█▍        | 58314/400000 [00:06<00:35, 9733.46it/s] 15%|█▍        | 59288/400000 [00:06<00:35, 9659.62it/s] 15%|█▌        | 60255/400000 [00:06<00:35, 9642.77it/s] 15%|█▌        | 61220/400000 [00:06<00:35, 9620.95it/s] 16%|█▌        | 62183/400000 [00:06<00:35, 9513.69it/s] 16%|█▌        | 63149/400000 [00:06<00:35, 9555.15it/s] 16%|█▌        | 64107/400000 [00:06<00:35, 9560.75it/s] 16%|█▋        | 65064/400000 [00:06<00:35, 9479.51it/s] 17%|█▋        | 66013/400000 [00:07<00:35, 9435.62it/s] 17%|█▋        | 66957/400000 [00:07<00:35, 9341.27it/s] 17%|█▋        | 67914/400000 [00:07<00:35, 9408.36it/s] 17%|█▋        | 68886/400000 [00:07<00:34, 9499.62it/s] 17%|█▋        | 69837/400000 [00:07<00:35, 9328.35it/s] 18%|█▊        | 70830/400000 [00:07<00:34, 9500.75it/s] 18%|█▊        | 71790/400000 [00:07<00:34, 9527.87it/s] 18%|█▊        | 72757/400000 [00:07<00:34, 9569.07it/s] 18%|█▊        | 73731/400000 [00:07<00:33, 9616.82it/s] 19%|█▊        | 74694/400000 [00:08<00:34, 9481.57it/s] 19%|█▉        | 75663/400000 [00:08<00:33, 9540.50it/s] 19%|█▉        | 76618/400000 [00:08<00:34, 9497.32it/s] 19%|█▉        | 77582/400000 [00:08<00:33, 9539.38it/s] 20%|█▉        | 78558/400000 [00:08<00:33, 9603.19it/s] 20%|█▉        | 79519/400000 [00:08<00:33, 9528.57it/s] 20%|██        | 80492/400000 [00:08<00:33, 9585.66it/s] 20%|██        | 81451/400000 [00:08<00:33, 9391.89it/s] 21%|██        | 82404/400000 [00:08<00:33, 9431.12it/s] 21%|██        | 83417/400000 [00:08<00:32, 9629.79it/s] 21%|██        | 84382/400000 [00:09<00:32, 9596.85it/s] 21%|██▏       | 85343/400000 [00:09<00:33, 9452.84it/s] 22%|██▏       | 86321/400000 [00:09<00:32, 9548.64it/s] 22%|██▏       | 87313/400000 [00:09<00:32, 9656.68it/s] 22%|██▏       | 88280/400000 [00:09<00:32, 9520.29it/s] 22%|██▏       | 89243/400000 [00:09<00:32, 9552.60it/s] 23%|██▎       | 90216/400000 [00:09<00:32, 9604.71it/s] 23%|██▎       | 91210/400000 [00:09<00:31, 9702.13it/s] 23%|██▎       | 92181/400000 [00:09<00:31, 9698.20it/s] 23%|██▎       | 93152/400000 [00:09<00:31, 9693.81it/s] 24%|██▎       | 94122/400000 [00:10<00:31, 9566.99it/s] 24%|██▍       | 95080/400000 [00:10<00:31, 9530.81it/s] 24%|██▍       | 96066/400000 [00:10<00:31, 9626.74it/s] 24%|██▍       | 97030/400000 [00:10<00:31, 9588.26it/s] 24%|██▍       | 97992/400000 [00:10<00:31, 9597.58it/s] 25%|██▍       | 98953/400000 [00:10<00:31, 9425.55it/s] 25%|██▍       | 99945/400000 [00:10<00:31, 9567.96it/s] 25%|██▌       | 100944/400000 [00:10<00:30, 9690.18it/s] 25%|██▌       | 101915/400000 [00:10<00:31, 9588.84it/s] 26%|██▌       | 102888/400000 [00:10<00:30, 9626.12it/s] 26%|██▌       | 103852/400000 [00:11<00:31, 9381.22it/s] 26%|██▌       | 104813/400000 [00:11<00:31, 9446.10it/s] 26%|██▋       | 105774/400000 [00:11<00:30, 9493.22it/s] 27%|██▋       | 106800/400000 [00:11<00:30, 9708.76it/s] 27%|██▋       | 107773/400000 [00:11<00:31, 9166.04it/s] 27%|██▋       | 108698/400000 [00:11<00:32, 9060.51it/s] 27%|██▋       | 109658/400000 [00:11<00:31, 9212.61it/s] 28%|██▊       | 110584/400000 [00:11<00:31, 9210.04it/s] 28%|██▊       | 111543/400000 [00:11<00:30, 9320.16it/s] 28%|██▊       | 112545/400000 [00:11<00:30, 9517.93it/s] 28%|██▊       | 113500/400000 [00:12<00:30, 9449.77it/s] 29%|██▊       | 114466/400000 [00:12<00:30, 9509.87it/s] 29%|██▉       | 115455/400000 [00:12<00:29, 9619.19it/s] 29%|██▉       | 116440/400000 [00:12<00:29, 9684.90it/s] 29%|██▉       | 117410/400000 [00:12<00:29, 9609.90it/s] 30%|██▉       | 118372/400000 [00:12<00:29, 9531.26it/s] 30%|██▉       | 119390/400000 [00:12<00:28, 9716.00it/s] 30%|███       | 120405/400000 [00:12<00:28, 9840.79it/s] 30%|███       | 121420/400000 [00:12<00:28, 9930.15it/s] 31%|███       | 122415/400000 [00:12<00:28, 9820.87it/s] 31%|███       | 123399/400000 [00:13<00:28, 9579.04it/s] 31%|███       | 124400/400000 [00:13<00:28, 9701.84it/s] 31%|███▏      | 125393/400000 [00:13<00:28, 9769.05it/s] 32%|███▏      | 126372/400000 [00:13<00:28, 9764.23it/s] 32%|███▏      | 127350/400000 [00:13<00:27, 9764.42it/s] 32%|███▏      | 128328/400000 [00:13<00:28, 9677.72it/s] 32%|███▏      | 129297/400000 [00:13<00:28, 9574.08it/s] 33%|███▎      | 130256/400000 [00:13<00:28, 9431.10it/s] 33%|███▎      | 131204/400000 [00:13<00:28, 9442.54it/s] 33%|███▎      | 132163/400000 [00:14<00:28, 9485.91it/s] 33%|███▎      | 133122/400000 [00:14<00:28, 9516.61it/s] 34%|███▎      | 134124/400000 [00:14<00:27, 9660.56it/s] 34%|███▍      | 135116/400000 [00:14<00:27, 9734.11it/s] 34%|███▍      | 136091/400000 [00:14<00:27, 9600.04it/s] 34%|███▍      | 137052/400000 [00:14<00:28, 9312.76it/s] 34%|███▍      | 137989/400000 [00:14<00:28, 9327.26it/s] 35%|███▍      | 138963/400000 [00:14<00:27, 9444.84it/s] 35%|███▍      | 139967/400000 [00:14<00:27, 9614.78it/s] 35%|███▌      | 140982/400000 [00:14<00:26, 9766.27it/s] 35%|███▌      | 141984/400000 [00:15<00:26, 9839.08it/s] 36%|███▌      | 142970/400000 [00:15<00:26, 9662.90it/s] 36%|███▌      | 143961/400000 [00:15<00:26, 9735.00it/s] 36%|███▌      | 144936/400000 [00:15<00:26, 9653.35it/s] 36%|███▋      | 145903/400000 [00:15<00:27, 9402.17it/s] 37%|███▋      | 146889/400000 [00:15<00:26, 9534.98it/s] 37%|███▋      | 147845/400000 [00:15<00:26, 9449.09it/s] 37%|███▋      | 148854/400000 [00:15<00:26, 9632.37it/s] 37%|███▋      | 149837/400000 [00:15<00:25, 9689.02it/s] 38%|███▊      | 150845/400000 [00:15<00:25, 9802.14it/s] 38%|███▊      | 151827/400000 [00:16<00:25, 9718.95it/s] 38%|███▊      | 152800/400000 [00:16<00:25, 9563.07it/s] 38%|███▊      | 153758/400000 [00:16<00:25, 9531.59it/s] 39%|███▊      | 154715/400000 [00:16<00:25, 9542.46it/s] 39%|███▉      | 155670/400000 [00:16<00:25, 9527.23it/s] 39%|███▉      | 156624/400000 [00:16<00:26, 9180.86it/s] 39%|███▉      | 157547/400000 [00:16<00:26, 9194.57it/s] 40%|███▉      | 158481/400000 [00:16<00:26, 9236.64it/s] 40%|███▉      | 159407/400000 [00:16<00:26, 9233.92it/s] 40%|████      | 160362/400000 [00:16<00:25, 9325.89it/s] 40%|████      | 161319/400000 [00:17<00:25, 9397.60it/s] 41%|████      | 162275/400000 [00:17<00:25, 9443.89it/s] 41%|████      | 163262/400000 [00:17<00:24, 9567.35it/s] 41%|████      | 164220/400000 [00:17<00:24, 9468.11it/s] 41%|████▏     | 165168/400000 [00:17<00:25, 9333.39it/s] 42%|████▏     | 166106/400000 [00:17<00:25, 9346.88it/s] 42%|████▏     | 167042/400000 [00:17<00:24, 9335.12it/s] 42%|████▏     | 168000/400000 [00:17<00:24, 9407.07it/s] 42%|████▏     | 168952/400000 [00:17<00:24, 9438.29it/s] 42%|████▏     | 169897/400000 [00:17<00:24, 9354.58it/s] 43%|████▎     | 170833/400000 [00:18<00:24, 9352.84it/s] 43%|████▎     | 171769/400000 [00:18<00:24, 9342.90it/s] 43%|████▎     | 172704/400000 [00:18<00:24, 9333.60it/s] 43%|████▎     | 173638/400000 [00:18<00:24, 9333.42it/s] 44%|████▎     | 174572/400000 [00:18<00:25, 8972.71it/s] 44%|████▍     | 175523/400000 [00:18<00:24, 9126.21it/s] 44%|████▍     | 176473/400000 [00:18<00:24, 9234.15it/s] 44%|████▍     | 177465/400000 [00:18<00:23, 9429.65it/s] 45%|████▍     | 178448/400000 [00:18<00:23, 9544.47it/s] 45%|████▍     | 179405/400000 [00:19<00:23, 9412.27it/s] 45%|████▌     | 180380/400000 [00:19<00:23, 9510.83it/s] 45%|████▌     | 181333/400000 [00:19<00:22, 9511.98it/s] 46%|████▌     | 182286/400000 [00:19<00:23, 9254.47it/s] 46%|████▌     | 183232/400000 [00:19<00:23, 9312.79it/s] 46%|████▌     | 184200/400000 [00:19<00:22, 9417.94it/s] 46%|████▋     | 185152/400000 [00:19<00:22, 9444.89it/s] 47%|████▋     | 186099/400000 [00:19<00:22, 9449.21it/s] 47%|████▋     | 187045/400000 [00:19<00:22, 9334.68it/s] 47%|████▋     | 187980/400000 [00:19<00:22, 9261.78it/s] 47%|████▋     | 188953/400000 [00:20<00:22, 9396.39it/s] 47%|████▋     | 189945/400000 [00:20<00:22, 9545.96it/s] 48%|████▊     | 190911/400000 [00:20<00:21, 9579.02it/s] 48%|████▊     | 191880/400000 [00:20<00:21, 9610.25it/s] 48%|████▊     | 192853/400000 [00:20<00:21, 9645.21it/s] 48%|████▊     | 193819/400000 [00:20<00:21, 9496.66it/s] 49%|████▊     | 194813/400000 [00:20<00:21, 9623.35it/s] 49%|████▉     | 195777/400000 [00:20<00:21, 9515.30it/s] 49%|████▉     | 196748/400000 [00:20<00:21, 9572.24it/s] 49%|████▉     | 197765/400000 [00:20<00:20, 9741.99it/s] 50%|████▉     | 198741/400000 [00:21<00:20, 9679.97it/s] 50%|████▉     | 199715/400000 [00:21<00:20, 9696.94it/s] 50%|█████     | 200686/400000 [00:21<00:20, 9543.32it/s] 50%|█████     | 201707/400000 [00:21<00:20, 9733.21it/s] 51%|█████     | 202682/400000 [00:21<00:20, 9721.98it/s] 51%|█████     | 203666/400000 [00:21<00:20, 9755.83it/s] 51%|█████     | 204643/400000 [00:21<00:20, 9708.59it/s] 51%|█████▏    | 205615/400000 [00:21<00:20, 9695.17it/s] 52%|█████▏    | 206611/400000 [00:21<00:19, 9771.96it/s] 52%|█████▏    | 207629/400000 [00:21<00:19, 9889.73it/s] 52%|█████▏    | 208619/400000 [00:22<00:19, 9745.15it/s] 52%|█████▏    | 209595/400000 [00:22<00:19, 9706.17it/s] 53%|█████▎    | 210567/400000 [00:22<00:19, 9670.40it/s] 53%|█████▎    | 211535/400000 [00:22<00:19, 9615.08it/s] 53%|█████▎    | 212497/400000 [00:22<00:19, 9487.86it/s] 53%|█████▎    | 213447/400000 [00:22<00:19, 9473.67it/s] 54%|█████▎    | 214416/400000 [00:22<00:19, 9535.18it/s] 54%|█████▍    | 215370/400000 [00:22<00:19, 9479.67it/s] 54%|█████▍    | 216319/400000 [00:22<00:19, 9429.97it/s] 54%|█████▍    | 217263/400000 [00:22<00:19, 9422.79it/s] 55%|█████▍    | 218206/400000 [00:23<00:19, 9420.45it/s] 55%|█████▍    | 219216/400000 [00:23<00:18, 9613.22it/s] 55%|█████▌    | 220186/400000 [00:23<00:18, 9637.31it/s] 55%|█████▌    | 221159/400000 [00:23<00:18, 9662.92it/s] 56%|█████▌    | 222133/400000 [00:23<00:18, 9684.23it/s] 56%|█████▌    | 223102/400000 [00:23<00:18, 9668.42it/s] 56%|█████▌    | 224070/400000 [00:23<00:18, 9632.96it/s] 56%|█████▋    | 225034/400000 [00:23<00:18, 9532.17it/s] 56%|█████▋    | 225988/400000 [00:23<00:18, 9517.45it/s] 57%|█████▋    | 226984/400000 [00:23<00:17, 9645.96it/s] 57%|█████▋    | 227950/400000 [00:24<00:17, 9635.52it/s] 57%|█████▋    | 228914/400000 [00:24<00:17, 9622.66it/s] 57%|█████▋    | 229878/400000 [00:24<00:17, 9627.29it/s] 58%|█████▊    | 230841/400000 [00:24<00:17, 9606.72it/s] 58%|█████▊    | 231805/400000 [00:24<00:17, 9614.77it/s] 58%|█████▊    | 232791/400000 [00:24<00:17, 9686.50it/s] 58%|█████▊    | 233760/400000 [00:24<00:17, 9537.49it/s] 59%|█████▊    | 234715/400000 [00:24<00:17, 9308.32it/s] 59%|█████▉    | 235662/400000 [00:24<00:17, 9355.65it/s] 59%|█████▉    | 236602/400000 [00:24<00:17, 9366.39it/s] 59%|█████▉    | 237543/400000 [00:25<00:17, 9378.19it/s] 60%|█████▉    | 238482/400000 [00:25<00:17, 9321.21it/s] 60%|█████▉    | 239415/400000 [00:25<00:17, 9294.08it/s] 60%|██████    | 240376/400000 [00:25<00:17, 9383.51it/s] 60%|██████    | 241315/400000 [00:25<00:16, 9343.69it/s] 61%|██████    | 242283/400000 [00:25<00:16, 9439.59it/s] 61%|██████    | 243228/400000 [00:25<00:16, 9263.02it/s] 61%|██████    | 244156/400000 [00:25<00:16, 9198.88it/s] 61%|██████▏   | 245090/400000 [00:25<00:16, 9239.85it/s] 62%|██████▏   | 246015/400000 [00:26<00:16, 9085.25it/s] 62%|██████▏   | 246925/400000 [00:26<00:16, 9009.03it/s] 62%|██████▏   | 247827/400000 [00:26<00:16, 8975.16it/s] 62%|██████▏   | 248726/400000 [00:26<00:16, 8902.75it/s] 62%|██████▏   | 249617/400000 [00:26<00:16, 8903.11it/s] 63%|██████▎   | 250508/400000 [00:26<00:16, 8882.39it/s] 63%|██████▎   | 251397/400000 [00:26<00:16, 8799.45it/s] 63%|██████▎   | 252281/400000 [00:26<00:16, 8808.77it/s] 63%|██████▎   | 253163/400000 [00:26<00:16, 8767.10it/s] 64%|██████▎   | 254051/400000 [00:26<00:16, 8800.65it/s] 64%|██████▎   | 254953/400000 [00:27<00:16, 8862.99it/s] 64%|██████▍   | 255840/400000 [00:27<00:16, 8500.15it/s] 64%|██████▍   | 256694/400000 [00:27<00:17, 8361.18it/s] 64%|██████▍   | 257533/400000 [00:27<00:18, 7806.44it/s] 65%|██████▍   | 258324/400000 [00:27<00:18, 7760.04it/s] 65%|██████▍   | 259231/400000 [00:27<00:17, 8109.34it/s] 65%|██████▌   | 260142/400000 [00:27<00:16, 8385.57it/s] 65%|██████▌   | 261052/400000 [00:27<00:16, 8585.14it/s] 65%|██████▌   | 261947/400000 [00:27<00:15, 8690.99it/s] 66%|██████▌   | 262881/400000 [00:27<00:15, 8874.61it/s] 66%|██████▌   | 263839/400000 [00:28<00:15, 9071.87it/s] 66%|██████▌   | 264751/400000 [00:28<00:15, 8968.28it/s] 66%|██████▋   | 265704/400000 [00:28<00:14, 9129.12it/s] 67%|██████▋   | 266644/400000 [00:28<00:14, 9208.17it/s] 67%|██████▋   | 267608/400000 [00:28<00:14, 9332.46it/s] 67%|██████▋   | 268544/400000 [00:28<00:14, 9179.51it/s] 67%|██████▋   | 269471/400000 [00:28<00:14, 9206.05it/s] 68%|██████▊   | 270451/400000 [00:28<00:13, 9375.24it/s] 68%|██████▊   | 271391/400000 [00:28<00:13, 9360.37it/s] 68%|██████▊   | 272332/400000 [00:28<00:13, 9374.17it/s] 68%|██████▊   | 273271/400000 [00:29<00:13, 9177.14it/s] 69%|██████▊   | 274191/400000 [00:29<00:13, 9127.68it/s] 69%|██████▉   | 275110/400000 [00:29<00:13, 9144.40it/s] 69%|██████▉   | 276042/400000 [00:29<00:13, 9193.64it/s] 69%|██████▉   | 276978/400000 [00:29<00:13, 9240.65it/s] 69%|██████▉   | 277929/400000 [00:29<00:13, 9319.11it/s] 70%|██████▉   | 278890/400000 [00:29<00:12, 9403.66it/s] 70%|██████▉   | 279838/400000 [00:29<00:12, 9425.33it/s] 70%|███████   | 280781/400000 [00:29<00:12, 9398.42it/s] 70%|███████   | 281722/400000 [00:29<00:12, 9328.98it/s] 71%|███████   | 282664/400000 [00:30<00:12, 9352.93it/s] 71%|███████   | 283600/400000 [00:30<00:12, 9275.43it/s] 71%|███████   | 284537/400000 [00:30<00:12, 9300.60it/s] 71%|███████▏  | 285468/400000 [00:30<00:12, 9236.72it/s] 72%|███████▏  | 286392/400000 [00:30<00:12, 9168.91it/s] 72%|███████▏  | 287365/400000 [00:30<00:12, 9329.02it/s] 72%|███████▏  | 288330/400000 [00:30<00:11, 9422.50it/s] 72%|███████▏  | 289274/400000 [00:30<00:11, 9381.11it/s] 73%|███████▎  | 290223/400000 [00:30<00:11, 9412.18it/s] 73%|███████▎  | 291170/400000 [00:31<00:11, 9427.98it/s] 73%|███████▎  | 292127/400000 [00:31<00:11, 9469.65it/s] 73%|███████▎  | 293075/400000 [00:31<00:11, 9406.49it/s] 74%|███████▎  | 294016/400000 [00:31<00:11, 9236.39it/s] 74%|███████▎  | 294964/400000 [00:31<00:11, 9307.21it/s] 74%|███████▍  | 295896/400000 [00:31<00:11, 9308.47it/s] 74%|███████▍  | 296829/400000 [00:31<00:11, 9314.06it/s] 74%|███████▍  | 297771/400000 [00:31<00:10, 9345.07it/s] 75%|███████▍  | 298728/400000 [00:31<00:10, 9407.59it/s] 75%|███████▍  | 299670/400000 [00:31<00:10, 9311.98it/s] 75%|███████▌  | 300602/400000 [00:32<00:10, 9109.44it/s] 75%|███████▌  | 301515/400000 [00:32<00:11, 8785.90it/s] 76%|███████▌  | 302478/400000 [00:32<00:10, 9022.85it/s] 76%|███████▌  | 303386/400000 [00:32<00:10, 9037.52it/s] 76%|███████▌  | 304294/400000 [00:32<00:10, 9050.22it/s] 76%|███████▋  | 305237/400000 [00:32<00:10, 9158.88it/s] 77%|███████▋  | 306170/400000 [00:32<00:10, 9208.76it/s] 77%|███████▋  | 307093/400000 [00:32<00:10, 9193.71it/s] 77%|███████▋  | 308047/400000 [00:32<00:09, 9294.14it/s] 77%|███████▋  | 309018/400000 [00:32<00:09, 9412.95it/s] 77%|███████▋  | 309961/400000 [00:33<00:09, 9380.54it/s] 78%|███████▊  | 310924/400000 [00:33<00:09, 9452.44it/s] 78%|███████▊  | 311882/400000 [00:33<00:09, 9488.99it/s] 78%|███████▊  | 312841/400000 [00:33<00:09, 9518.66it/s] 78%|███████▊  | 313795/400000 [00:33<00:09, 9523.17it/s] 79%|███████▊  | 314748/400000 [00:33<00:09, 9329.72it/s] 79%|███████▉  | 315683/400000 [00:33<00:09, 9174.76it/s] 79%|███████▉  | 316602/400000 [00:33<00:09, 9061.36it/s] 79%|███████▉  | 317510/400000 [00:33<00:09, 9047.96it/s] 80%|███████▉  | 318416/400000 [00:33<00:09, 8932.68it/s] 80%|███████▉  | 319311/400000 [00:34<00:09, 8884.10it/s] 80%|████████  | 320201/400000 [00:34<00:08, 8879.92it/s] 80%|████████  | 321133/400000 [00:34<00:08, 9005.64it/s] 81%|████████  | 322036/400000 [00:34<00:08, 9009.78it/s] 81%|████████  | 322940/400000 [00:34<00:08, 9016.72it/s] 81%|████████  | 323843/400000 [00:34<00:08, 8962.91it/s] 81%|████████  | 324767/400000 [00:34<00:08, 9042.80it/s] 81%|████████▏ | 325690/400000 [00:34<00:08, 9098.05it/s] 82%|████████▏ | 326665/400000 [00:34<00:07, 9282.95it/s] 82%|████████▏ | 327595/400000 [00:34<00:07, 9210.56it/s] 82%|████████▏ | 328518/400000 [00:35<00:07, 8995.04it/s] 82%|████████▏ | 329420/400000 [00:35<00:07, 8987.47it/s] 83%|████████▎ | 330320/400000 [00:35<00:07, 8820.05it/s] 83%|████████▎ | 331222/400000 [00:35<00:07, 8876.97it/s] 83%|████████▎ | 332111/400000 [00:35<00:07, 8867.99it/s] 83%|████████▎ | 333003/400000 [00:35<00:07, 8881.89it/s] 83%|████████▎ | 333929/400000 [00:35<00:07, 8989.79it/s] 84%|████████▎ | 334876/400000 [00:35<00:07, 9127.03it/s] 84%|████████▍ | 335811/400000 [00:35<00:06, 9189.66it/s] 84%|████████▍ | 336731/400000 [00:35<00:07, 9035.29it/s] 84%|████████▍ | 337636/400000 [00:36<00:06, 8994.89it/s] 85%|████████▍ | 338537/400000 [00:36<00:06, 8975.22it/s] 85%|████████▍ | 339436/400000 [00:36<00:06, 8961.19it/s] 85%|████████▌ | 340344/400000 [00:36<00:06, 8995.96it/s] 85%|████████▌ | 341244/400000 [00:36<00:06, 8875.42it/s] 86%|████████▌ | 342175/400000 [00:36<00:06, 9001.38it/s] 86%|████████▌ | 343081/400000 [00:36<00:06, 9018.72it/s] 86%|████████▌ | 344001/400000 [00:36<00:06, 9069.85it/s] 86%|████████▌ | 344932/400000 [00:36<00:06, 9137.90it/s] 86%|████████▋ | 345847/400000 [00:37<00:05, 9064.75it/s] 87%|████████▋ | 346754/400000 [00:37<00:05, 9061.41it/s] 87%|████████▋ | 347661/400000 [00:37<00:05, 8936.57it/s] 87%|████████▋ | 348556/400000 [00:37<00:05, 8900.32it/s] 87%|████████▋ | 349447/400000 [00:37<00:05, 8797.40it/s] 88%|████████▊ | 350328/400000 [00:37<00:05, 8753.54it/s] 88%|████████▊ | 351219/400000 [00:37<00:05, 8798.38it/s] 88%|████████▊ | 352118/400000 [00:37<00:05, 8854.32it/s] 88%|████████▊ | 353004/400000 [00:37<00:05, 8826.17it/s] 88%|████████▊ | 353898/400000 [00:37<00:05, 8859.81it/s] 89%|████████▊ | 354785/400000 [00:38<00:05, 8785.91it/s] 89%|████████▉ | 355683/400000 [00:38<00:05, 8841.57it/s] 89%|████████▉ | 356598/400000 [00:38<00:04, 8931.16it/s] 89%|████████▉ | 357519/400000 [00:38<00:04, 9010.34it/s] 90%|████████▉ | 358424/400000 [00:38<00:04, 9022.11it/s] 90%|████████▉ | 359327/400000 [00:38<00:04, 8986.75it/s] 90%|█████████ | 360230/400000 [00:38<00:04, 8998.74it/s] 90%|█████████ | 361141/400000 [00:38<00:04, 9029.72it/s] 91%|█████████ | 362045/400000 [00:38<00:04, 8992.03it/s] 91%|█████████ | 362966/400000 [00:38<00:04, 9055.39it/s] 91%|█████████ | 363872/400000 [00:39<00:04, 8979.80it/s] 91%|█████████ | 364772/400000 [00:39<00:03, 8985.55it/s] 91%|█████████▏| 365671/400000 [00:39<00:03, 8942.75it/s] 92%|█████████▏| 366566/400000 [00:39<00:03, 8845.58it/s] 92%|█████████▏| 367467/400000 [00:39<00:03, 8891.97it/s] 92%|█████████▏| 368359/400000 [00:39<00:03, 8898.81it/s] 92%|█████████▏| 369250/400000 [00:39<00:03, 8805.94it/s] 93%|█████████▎| 370171/400000 [00:39<00:03, 8921.25it/s] 93%|█████████▎| 371083/400000 [00:39<00:03, 8978.81it/s] 93%|█████████▎| 372005/400000 [00:39<00:03, 9047.99it/s] 93%|█████████▎| 372911/400000 [00:40<00:02, 9033.62it/s] 93%|█████████▎| 373819/400000 [00:40<00:02, 9047.19it/s] 94%|█████████▎| 374724/400000 [00:40<00:02, 8990.37it/s] 94%|█████████▍| 375624/400000 [00:40<00:02, 8935.13it/s] 94%|█████████▍| 376518/400000 [00:40<00:02, 8920.15it/s] 94%|█████████▍| 377411/400000 [00:40<00:02, 8907.26it/s] 95%|█████████▍| 378335/400000 [00:40<00:02, 9002.40it/s] 95%|█████████▍| 379236/400000 [00:40<00:02, 8987.09it/s] 95%|█████████▌| 380135/400000 [00:40<00:02, 8979.94it/s] 95%|█████████▌| 381066/400000 [00:40<00:02, 9075.66it/s] 95%|█████████▌| 381974/400000 [00:41<00:02, 8989.06it/s] 96%|█████████▌| 382883/400000 [00:41<00:01, 9017.50it/s] 96%|█████████▌| 383791/400000 [00:41<00:01, 9033.40it/s] 96%|█████████▌| 384695/400000 [00:41<00:01, 8860.95it/s] 96%|█████████▋| 385587/400000 [00:41<00:01, 8876.83it/s] 97%|█████████▋| 386476/400000 [00:41<00:01, 8853.84it/s] 97%|█████████▋| 387362/400000 [00:41<00:01, 8763.15it/s] 97%|█████████▋| 388242/400000 [00:41<00:01, 8772.84it/s] 97%|█████████▋| 389133/400000 [00:41<00:01, 8811.21it/s] 98%|█████████▊| 390015/400000 [00:41<00:01, 8638.11it/s] 98%|█████████▊| 390880/400000 [00:42<00:01, 8626.89it/s] 98%|█████████▊| 391744/400000 [00:42<00:00, 8595.09it/s] 98%|█████████▊| 392605/400000 [00:42<00:00, 8288.02it/s] 98%|█████████▊| 393516/400000 [00:42<00:00, 8516.29it/s] 99%|█████████▊| 394441/400000 [00:42<00:00, 8722.07it/s] 99%|█████████▉| 395317/400000 [00:42<00:00, 8721.34it/s] 99%|█████████▉| 396220/400000 [00:42<00:00, 8811.31it/s] 99%|█████████▉| 397180/400000 [00:42<00:00, 9032.11it/s]100%|█████████▉| 398112/400000 [00:42<00:00, 9116.39it/s]100%|█████████▉| 399033/400000 [00:42<00:00, 9143.45it/s]100%|█████████▉| 399949/400000 [00:43<00:00, 9041.60it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9282.49it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fbc39b8cb70>, <torchtext.data.dataset.TabularDataset object at 0x7fbc39b8ccc0>, <torchtext.vocab.Vocab object at 0x7fbc39b8cbe0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.08 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.81 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.81 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.68 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.68 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.77 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.25 file/s]2020-07-05 00:18:10.774356: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-05 00:18:10.778591: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-05 00:18:10.778762: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5636b56af000 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-05 00:18:10.778775: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 143713.78it/s] 80%|███████▉  | 7913472/9912422 [00:00<00:09, 205145.31it/s]9920512it [00:00, 43095444.81it/s]                           
0it [00:00, ?it/s]32768it [00:00, 589765.51it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 480971.73it/s]1654784it [00:00, 11820702.21it/s]                         
0it [00:00, ?it/s]8192it [00:00, 171456.64it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
