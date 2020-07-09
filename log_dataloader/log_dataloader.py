
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f77431140d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f77431140d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f77ae45d1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f77ae45d1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f77cc7a5e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f77cc7a5e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f775b78b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f775b78b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f775b78b620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3694592/9912422 [00:00<00:00, 36944722.07it/s]9920512it [00:00, 38304888.88it/s]                             
0it [00:00, ?it/s]32768it [00:00, 534675.82it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 136193.06it/s]1654784it [00:00, 10122578.11it/s]                         
0it [00:00, ?it/s]8192it [00:00, 194080.05it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7758ce4978>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f77427bfc18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f775b78b268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f775b78b268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f775b78b268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:22,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:20,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:17,  7.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 17.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 22.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 28.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 35.89 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 43.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 51.21 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 64.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 64.24 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 70.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 70.00 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 74.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 74.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 77.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 77.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.59s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 82.29 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.64 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.96s/ url]
0 examples [00:00, ? examples/s]2020-07-09 00:09:44.144138: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-09 00:09:44.158294: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-09 00:09:44.158585: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56335569ba90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-09 00:09:44.158607: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
25 examples [00:00, 248.31 examples/s]110 examples [00:00, 315.01 examples/s]198 examples [00:00, 390.05 examples/s]287 examples [00:00, 467.54 examples/s]374 examples [00:00, 542.29 examples/s]461 examples [00:00, 611.11 examples/s]551 examples [00:00, 675.60 examples/s]640 examples [00:00, 726.49 examples/s]723 examples [00:00, 753.89 examples/s]809 examples [00:01, 781.77 examples/s]900 examples [00:01, 815.77 examples/s]993 examples [00:01, 846.46 examples/s]1083 examples [00:01, 861.18 examples/s]1172 examples [00:01, 857.88 examples/s]1260 examples [00:01, 831.34 examples/s]1345 examples [00:01, 763.19 examples/s]1433 examples [00:01, 794.52 examples/s]1520 examples [00:01, 815.12 examples/s]1604 examples [00:01, 821.59 examples/s]1692 examples [00:02, 836.18 examples/s]1779 examples [00:02, 843.54 examples/s]1866 examples [00:02, 850.28 examples/s]1955 examples [00:02, 861.00 examples/s]2042 examples [00:02, 859.97 examples/s]2130 examples [00:02, 865.31 examples/s]2217 examples [00:02, 855.16 examples/s]2304 examples [00:02, 858.64 examples/s]2398 examples [00:02, 879.56 examples/s]2487 examples [00:02, 868.51 examples/s]2578 examples [00:03, 877.71 examples/s]2667 examples [00:03, 878.87 examples/s]2755 examples [00:03, 861.55 examples/s]2844 examples [00:03, 867.42 examples/s]2931 examples [00:03, 859.48 examples/s]3019 examples [00:03, 863.33 examples/s]3106 examples [00:03, 854.35 examples/s]3192 examples [00:03, 842.08 examples/s]3284 examples [00:03, 862.17 examples/s]3372 examples [00:04, 865.30 examples/s]3461 examples [00:04, 870.98 examples/s]3549 examples [00:04, 869.58 examples/s]3639 examples [00:04, 878.10 examples/s]3728 examples [00:04, 881.56 examples/s]3817 examples [00:04, 869.80 examples/s]3907 examples [00:04, 877.52 examples/s]4001 examples [00:04, 893.51 examples/s]4091 examples [00:04, 888.04 examples/s]4180 examples [00:04, 878.12 examples/s]4268 examples [00:05, 876.26 examples/s]4358 examples [00:05, 881.31 examples/s]4450 examples [00:05, 891.59 examples/s]4540 examples [00:05, 873.30 examples/s]4628 examples [00:05, 860.61 examples/s]4717 examples [00:05, 867.01 examples/s]4804 examples [00:05, 863.14 examples/s]4891 examples [00:05, 857.09 examples/s]4977 examples [00:05, 853.99 examples/s]5066 examples [00:05, 862.83 examples/s]5153 examples [00:06, 853.19 examples/s]5239 examples [00:06, 850.80 examples/s]5325 examples [00:06, 826.07 examples/s]5408 examples [00:06, 817.48 examples/s]5499 examples [00:06, 841.53 examples/s]5584 examples [00:06, 838.96 examples/s]5674 examples [00:06, 854.72 examples/s]5765 examples [00:06, 870.46 examples/s]5855 examples [00:06, 877.01 examples/s]5946 examples [00:06, 884.34 examples/s]6035 examples [00:07, 873.38 examples/s]6123 examples [00:07, 857.89 examples/s]6214 examples [00:07, 870.91 examples/s]6304 examples [00:07, 877.80 examples/s]6395 examples [00:07, 884.50 examples/s]6484 examples [00:07, 879.35 examples/s]6573 examples [00:07, 878.81 examples/s]6661 examples [00:07, 871.57 examples/s]6750 examples [00:07, 876.61 examples/s]6838 examples [00:07, 867.79 examples/s]6925 examples [00:08, 863.73 examples/s]7020 examples [00:08, 886.03 examples/s]7109 examples [00:08, 882.48 examples/s]7198 examples [00:08, 872.44 examples/s]7286 examples [00:08, 856.18 examples/s]7372 examples [00:08, 853.26 examples/s]7458 examples [00:08, 847.64 examples/s]7543 examples [00:08, 812.38 examples/s]7634 examples [00:08, 837.44 examples/s]7721 examples [00:09, 843.92 examples/s]7807 examples [00:09, 847.89 examples/s]7896 examples [00:09, 859.05 examples/s]7983 examples [00:09, 846.19 examples/s]8074 examples [00:09, 861.52 examples/s]8167 examples [00:09, 878.96 examples/s]8256 examples [00:09, 868.55 examples/s]8344 examples [00:09, 867.46 examples/s]8434 examples [00:09, 876.48 examples/s]8523 examples [00:09, 878.83 examples/s]8616 examples [00:10, 892.06 examples/s]8706 examples [00:10, 890.38 examples/s]8796 examples [00:10, 892.66 examples/s]8887 examples [00:10, 895.89 examples/s]8978 examples [00:10, 899.35 examples/s]9068 examples [00:10, 895.97 examples/s]9158 examples [00:10, 885.07 examples/s]9249 examples [00:10, 891.14 examples/s]9339 examples [00:10, 872.18 examples/s]9428 examples [00:10, 875.10 examples/s]9519 examples [00:11, 884.87 examples/s]9608 examples [00:11, 874.69 examples/s]9696 examples [00:11, 866.14 examples/s]9787 examples [00:11, 878.70 examples/s]9876 examples [00:11, 881.86 examples/s]9965 examples [00:11, 877.54 examples/s]10053 examples [00:11, 792.26 examples/s]10134 examples [00:11, 791.55 examples/s]10227 examples [00:11, 827.60 examples/s]10311 examples [00:12, 828.64 examples/s]10398 examples [00:12, 839.42 examples/s]10483 examples [00:12, 829.93 examples/s]10569 examples [00:12, 835.07 examples/s]10660 examples [00:12, 855.62 examples/s]10753 examples [00:12, 874.23 examples/s]10842 examples [00:12, 878.45 examples/s]10931 examples [00:12, 848.84 examples/s]11017 examples [00:12, 848.50 examples/s]11108 examples [00:12, 865.00 examples/s]11201 examples [00:13, 881.45 examples/s]11291 examples [00:13, 884.43 examples/s]11380 examples [00:13, 884.47 examples/s]11469 examples [00:13, 880.81 examples/s]11558 examples [00:13, 876.65 examples/s]11646 examples [00:13, 857.93 examples/s]11733 examples [00:13, 860.25 examples/s]11820 examples [00:13, 846.94 examples/s]11907 examples [00:13, 853.70 examples/s]11999 examples [00:13, 872.13 examples/s]12089 examples [00:14, 879.95 examples/s]12179 examples [00:14, 882.70 examples/s]12268 examples [00:14, 878.85 examples/s]12356 examples [00:14, 876.76 examples/s]12449 examples [00:14, 889.37 examples/s]12542 examples [00:14, 898.70 examples/s]12636 examples [00:14, 910.08 examples/s]12728 examples [00:14, 896.37 examples/s]12818 examples [00:14, 891.22 examples/s]12912 examples [00:14, 904.16 examples/s]13006 examples [00:15, 912.93 examples/s]13098 examples [00:15, 902.92 examples/s]13189 examples [00:15, 889.19 examples/s]13279 examples [00:15, 878.00 examples/s]13371 examples [00:15, 888.56 examples/s]13464 examples [00:15, 899.95 examples/s]13555 examples [00:15, 880.20 examples/s]13645 examples [00:15, 884.29 examples/s]13734 examples [00:15, 885.71 examples/s]13823 examples [00:16, 859.86 examples/s]13910 examples [00:16, 852.22 examples/s]13996 examples [00:16, 853.89 examples/s]14082 examples [00:16, 834.51 examples/s]14170 examples [00:16, 845.98 examples/s]14255 examples [00:16, 838.77 examples/s]14346 examples [00:16, 856.81 examples/s]14432 examples [00:16, 855.31 examples/s]14520 examples [00:16, 861.08 examples/s]14607 examples [00:16, 858.94 examples/s]14694 examples [00:17, 859.48 examples/s]14780 examples [00:17, 826.91 examples/s]14865 examples [00:17, 831.40 examples/s]14956 examples [00:17, 851.81 examples/s]15045 examples [00:17, 860.88 examples/s]15132 examples [00:17, 844.06 examples/s]15217 examples [00:17, 841.21 examples/s]15303 examples [00:17, 846.31 examples/s]15388 examples [00:17, 840.57 examples/s]15473 examples [00:17, 841.22 examples/s]15558 examples [00:18, 828.17 examples/s]15647 examples [00:18, 844.05 examples/s]15732 examples [00:18, 831.01 examples/s]15816 examples [00:18, 822.43 examples/s]15899 examples [00:18, 820.87 examples/s]15989 examples [00:18, 841.86 examples/s]16074 examples [00:18, 841.01 examples/s]16159 examples [00:18, 821.84 examples/s]16247 examples [00:18, 837.76 examples/s]16338 examples [00:18, 856.63 examples/s]16425 examples [00:19, 858.78 examples/s]16512 examples [00:19, 819.65 examples/s]16602 examples [00:19, 840.47 examples/s]16693 examples [00:19, 857.83 examples/s]16780 examples [00:19, 857.76 examples/s]16868 examples [00:19, 861.94 examples/s]16955 examples [00:19, 849.38 examples/s]17041 examples [00:19, 849.46 examples/s]17134 examples [00:19, 870.88 examples/s]17228 examples [00:20, 889.52 examples/s]17318 examples [00:20, 879.51 examples/s]17407 examples [00:20, 875.94 examples/s]17495 examples [00:20, 859.74 examples/s]17582 examples [00:20, 859.11 examples/s]17669 examples [00:20, 802.07 examples/s]17751 examples [00:20, 805.32 examples/s]17833 examples [00:20, 805.08 examples/s]17914 examples [00:20, 783.47 examples/s]18000 examples [00:20, 803.92 examples/s]18087 examples [00:21, 821.86 examples/s]18173 examples [00:21, 832.64 examples/s]18261 examples [00:21, 845.01 examples/s]18346 examples [00:21, 838.02 examples/s]18432 examples [00:21, 842.69 examples/s]18517 examples [00:21, 831.09 examples/s]18606 examples [00:21, 847.75 examples/s]18695 examples [00:21, 857.22 examples/s]18781 examples [00:21, 853.07 examples/s]18868 examples [00:21, 855.75 examples/s]18955 examples [00:22, 859.22 examples/s]19041 examples [00:22, 836.60 examples/s]19126 examples [00:22, 838.87 examples/s]19211 examples [00:22, 840.25 examples/s]19299 examples [00:22, 850.69 examples/s]19389 examples [00:22, 862.94 examples/s]19478 examples [00:22, 869.91 examples/s]19566 examples [00:22, 849.89 examples/s]19652 examples [00:22, 852.17 examples/s]19738 examples [00:23, 829.95 examples/s]19828 examples [00:23, 848.08 examples/s]19915 examples [00:23, 852.19 examples/s]20001 examples [00:23, 774.78 examples/s]20089 examples [00:23, 803.56 examples/s]20171 examples [00:23, 791.82 examples/s]20256 examples [00:23, 808.05 examples/s]20345 examples [00:23, 828.66 examples/s]20431 examples [00:23, 836.66 examples/s]20516 examples [00:23, 827.62 examples/s]20600 examples [00:24, 827.20 examples/s]20685 examples [00:24, 833.28 examples/s]20769 examples [00:24, 833.94 examples/s]20854 examples [00:24, 836.85 examples/s]20943 examples [00:24, 850.50 examples/s]21029 examples [00:24, 850.32 examples/s]21120 examples [00:24, 865.38 examples/s]21207 examples [00:24, 861.72 examples/s]21294 examples [00:24, 857.58 examples/s]21382 examples [00:24, 861.71 examples/s]21472 examples [00:25, 872.43 examples/s]21560 examples [00:25, 867.38 examples/s]21647 examples [00:25, 848.90 examples/s]21733 examples [00:25, 848.47 examples/s]21819 examples [00:25, 850.45 examples/s]21906 examples [00:25, 856.18 examples/s]21994 examples [00:25, 860.51 examples/s]22081 examples [00:25, 858.14 examples/s]22168 examples [00:25, 857.17 examples/s]22254 examples [00:26, 855.23 examples/s]22341 examples [00:26, 856.86 examples/s]22427 examples [00:26, 854.93 examples/s]22514 examples [00:26, 858.93 examples/s]22600 examples [00:26, 855.61 examples/s]22686 examples [00:26, 854.27 examples/s]22773 examples [00:26, 858.18 examples/s]22863 examples [00:26, 868.50 examples/s]22950 examples [00:26, 867.94 examples/s]23037 examples [00:26, 860.92 examples/s]23124 examples [00:27, 840.38 examples/s]23211 examples [00:27, 847.86 examples/s]23301 examples [00:27, 859.38 examples/s]23388 examples [00:27, 852.42 examples/s]23474 examples [00:27, 849.88 examples/s]23563 examples [00:27, 859.20 examples/s]23649 examples [00:27, 850.95 examples/s]23738 examples [00:27, 862.03 examples/s]23826 examples [00:27, 866.89 examples/s]23913 examples [00:27, 854.96 examples/s]23999 examples [00:28, 850.00 examples/s]24085 examples [00:28, 852.86 examples/s]24171 examples [00:28, 845.70 examples/s]24256 examples [00:28, 809.41 examples/s]24338 examples [00:28, 782.75 examples/s]24425 examples [00:28, 806.42 examples/s]24507 examples [00:28, 806.03 examples/s]24588 examples [00:28, 777.19 examples/s]24677 examples [00:28, 805.30 examples/s]24762 examples [00:28, 817.67 examples/s]24850 examples [00:29, 834.58 examples/s]24943 examples [00:29, 860.20 examples/s]25034 examples [00:29, 874.15 examples/s]25123 examples [00:29, 878.12 examples/s]25212 examples [00:29, 864.30 examples/s]25299 examples [00:29, 855.70 examples/s]25387 examples [00:29, 862.71 examples/s]25475 examples [00:29, 867.81 examples/s]25564 examples [00:29, 872.57 examples/s]25652 examples [00:30, 874.11 examples/s]25741 examples [00:30, 878.55 examples/s]25834 examples [00:30, 892.65 examples/s]25924 examples [00:30, 883.50 examples/s]26014 examples [00:30, 887.12 examples/s]26103 examples [00:30, 876.01 examples/s]26194 examples [00:30, 885.15 examples/s]26286 examples [00:30, 894.74 examples/s]26378 examples [00:30, 899.70 examples/s]26469 examples [00:30, 862.62 examples/s]26556 examples [00:31, 863.80 examples/s]26644 examples [00:31, 867.07 examples/s]26734 examples [00:31, 875.24 examples/s]26822 examples [00:31, 848.93 examples/s]26908 examples [00:31, 814.49 examples/s]26990 examples [00:31, 790.96 examples/s]27077 examples [00:31, 812.73 examples/s]27165 examples [00:31, 831.22 examples/s]27250 examples [00:31, 835.51 examples/s]27336 examples [00:31, 840.35 examples/s]27426 examples [00:32, 857.11 examples/s]27514 examples [00:32, 861.43 examples/s]27603 examples [00:32, 869.04 examples/s]27691 examples [00:32, 869.92 examples/s]27779 examples [00:32, 865.63 examples/s]27866 examples [00:32, 854.38 examples/s]27954 examples [00:32, 860.41 examples/s]28041 examples [00:32, 829.37 examples/s]28129 examples [00:32, 843.86 examples/s]28214 examples [00:33, 828.52 examples/s]28298 examples [00:33, 827.26 examples/s]28381 examples [00:33, 791.45 examples/s]28467 examples [00:33, 808.72 examples/s]28555 examples [00:33, 827.54 examples/s]28639 examples [00:33, 826.17 examples/s]28722 examples [00:33, 824.17 examples/s]28808 examples [00:33, 831.57 examples/s]28896 examples [00:33, 842.99 examples/s]28982 examples [00:33, 846.52 examples/s]29067 examples [00:34, 839.54 examples/s]29157 examples [00:34, 855.64 examples/s]29246 examples [00:34, 864.20 examples/s]29334 examples [00:34, 867.46 examples/s]29421 examples [00:34, 854.39 examples/s]29507 examples [00:34, 839.95 examples/s]29593 examples [00:34, 844.52 examples/s]29679 examples [00:34, 848.82 examples/s]29766 examples [00:34, 854.12 examples/s]29852 examples [00:34, 855.30 examples/s]29938 examples [00:35, 849.71 examples/s]30024 examples [00:35, 778.59 examples/s]30111 examples [00:35, 802.79 examples/s]30198 examples [00:35, 820.85 examples/s]30286 examples [00:35, 837.06 examples/s]30371 examples [00:35, 839.35 examples/s]30456 examples [00:35, 840.27 examples/s]30543 examples [00:35, 848.41 examples/s]30630 examples [00:35, 853.98 examples/s]30716 examples [00:35, 847.70 examples/s]30801 examples [00:36, 839.24 examples/s]30886 examples [00:36, 840.89 examples/s]30971 examples [00:36, 828.89 examples/s]31055 examples [00:36, 830.43 examples/s]31139 examples [00:36, 831.30 examples/s]31223 examples [00:36, 798.62 examples/s]31308 examples [00:36, 812.83 examples/s]31395 examples [00:36, 828.58 examples/s]31485 examples [00:36, 846.06 examples/s]31574 examples [00:37, 857.50 examples/s]31660 examples [00:37, 855.58 examples/s]31752 examples [00:37, 871.66 examples/s]31841 examples [00:37, 874.69 examples/s]31930 examples [00:37, 877.97 examples/s]32018 examples [00:37, 861.64 examples/s]32105 examples [00:37, 839.90 examples/s]32191 examples [00:37, 844.76 examples/s]32276 examples [00:37, 845.29 examples/s]32361 examples [00:37, 833.57 examples/s]32449 examples [00:38, 844.59 examples/s]32536 examples [00:38, 850.33 examples/s]32623 examples [00:38, 853.56 examples/s]32710 examples [00:38, 858.41 examples/s]32796 examples [00:38, 853.76 examples/s]32886 examples [00:38, 866.82 examples/s]32973 examples [00:38, 828.16 examples/s]33060 examples [00:38, 837.88 examples/s]33149 examples [00:38, 851.11 examples/s]33238 examples [00:38, 861.73 examples/s]33327 examples [00:39, 867.94 examples/s]33414 examples [00:39, 859.55 examples/s]33501 examples [00:39, 853.42 examples/s]33587 examples [00:39, 846.92 examples/s]33675 examples [00:39, 854.73 examples/s]33761 examples [00:39, 855.67 examples/s]33848 examples [00:39, 856.48 examples/s]33934 examples [00:39, 857.20 examples/s]34020 examples [00:39, 841.84 examples/s]34105 examples [00:39, 823.92 examples/s]34191 examples [00:40, 832.68 examples/s]34277 examples [00:40, 840.29 examples/s]34362 examples [00:40, 840.96 examples/s]34447 examples [00:40, 842.72 examples/s]34532 examples [00:40, 795.37 examples/s]34613 examples [00:40, 793.01 examples/s]34694 examples [00:40, 797.11 examples/s]34779 examples [00:40, 811.06 examples/s]34866 examples [00:40, 826.06 examples/s]34950 examples [00:41, 828.09 examples/s]35038 examples [00:41, 842.65 examples/s]35126 examples [00:41, 850.89 examples/s]35217 examples [00:41, 864.77 examples/s]35307 examples [00:41, 874.82 examples/s]35395 examples [00:41, 871.33 examples/s]35483 examples [00:41, 859.96 examples/s]35570 examples [00:41, 862.16 examples/s]35657 examples [00:41, 857.23 examples/s]35743 examples [00:41, 848.07 examples/s]35828 examples [00:42, 828.73 examples/s]35915 examples [00:42, 840.02 examples/s]36000 examples [00:42, 835.60 examples/s]36084 examples [00:42, 830.42 examples/s]36170 examples [00:42, 838.46 examples/s]36257 examples [00:42, 844.85 examples/s]36343 examples [00:42, 849.24 examples/s]36430 examples [00:42, 854.42 examples/s]36518 examples [00:42, 859.58 examples/s]36604 examples [00:42, 858.62 examples/s]36691 examples [00:43, 861.96 examples/s]36780 examples [00:43, 868.46 examples/s]36867 examples [00:43, 864.53 examples/s]36954 examples [00:43, 857.16 examples/s]37042 examples [00:43, 862.47 examples/s]37129 examples [00:43, 840.92 examples/s]37215 examples [00:43, 845.33 examples/s]37300 examples [00:43, 834.00 examples/s]37386 examples [00:43, 841.13 examples/s]37471 examples [00:43, 835.69 examples/s]37555 examples [00:44, 835.54 examples/s]37641 examples [00:44, 839.44 examples/s]37725 examples [00:44, 826.79 examples/s]37808 examples [00:44, 814.48 examples/s]37894 examples [00:44, 825.76 examples/s]37979 examples [00:44, 830.76 examples/s]38063 examples [00:44, 827.87 examples/s]38147 examples [00:44, 829.27 examples/s]38231 examples [00:44, 829.81 examples/s]38315 examples [00:45, 825.23 examples/s]38401 examples [00:45, 833.82 examples/s]38490 examples [00:45, 847.18 examples/s]38577 examples [00:45, 853.57 examples/s]38666 examples [00:45, 863.15 examples/s]38758 examples [00:45, 877.62 examples/s]38850 examples [00:45, 889.40 examples/s]38940 examples [00:45, 874.84 examples/s]39028 examples [00:45, 871.94 examples/s]39116 examples [00:45, 869.26 examples/s]39203 examples [00:46, 862.28 examples/s]39290 examples [00:46, 838.33 examples/s]39378 examples [00:46, 848.01 examples/s]39465 examples [00:46, 853.15 examples/s]39553 examples [00:46, 858.33 examples/s]39641 examples [00:46, 862.30 examples/s]39728 examples [00:46, 851.53 examples/s]39817 examples [00:46, 862.47 examples/s]39904 examples [00:46, 820.57 examples/s]39987 examples [00:46, 803.44 examples/s]40068 examples [00:47, 745.94 examples/s]40153 examples [00:47, 773.02 examples/s]40235 examples [00:47, 785.69 examples/s]40323 examples [00:47, 811.35 examples/s]40413 examples [00:47, 834.68 examples/s]40502 examples [00:47, 849.76 examples/s]40588 examples [00:47, 849.75 examples/s]40674 examples [00:47, 845.31 examples/s]40760 examples [00:47, 847.45 examples/s]40845 examples [00:48, 848.01 examples/s]40933 examples [00:48, 854.18 examples/s]41019 examples [00:48, 855.67 examples/s]41105 examples [00:48, 806.12 examples/s]41187 examples [00:48, 806.37 examples/s]41271 examples [00:48, 814.78 examples/s]41353 examples [00:48, 808.86 examples/s]41439 examples [00:48, 823.49 examples/s]41523 examples [00:48, 827.37 examples/s]41608 examples [00:48, 832.89 examples/s]41693 examples [00:49, 835.90 examples/s]41783 examples [00:49, 851.53 examples/s]41871 examples [00:49, 859.84 examples/s]41958 examples [00:49, 861.61 examples/s]42045 examples [00:49, 856.22 examples/s]42131 examples [00:49, 855.80 examples/s]42217 examples [00:49, 816.05 examples/s]42304 examples [00:49, 831.37 examples/s]42388 examples [00:49, 827.79 examples/s]42474 examples [00:49, 835.00 examples/s]42558 examples [00:50, 827.16 examples/s]42643 examples [00:50, 833.30 examples/s]42728 examples [00:50, 836.99 examples/s]42814 examples [00:50, 842.16 examples/s]42904 examples [00:50, 856.58 examples/s]42990 examples [00:50, 854.21 examples/s]43078 examples [00:50, 858.73 examples/s]43164 examples [00:50, 850.73 examples/s]43251 examples [00:50, 856.09 examples/s]43337 examples [00:50, 857.26 examples/s]43423 examples [00:51, 848.33 examples/s]43511 examples [00:51, 856.05 examples/s]43600 examples [00:51, 863.17 examples/s]43688 examples [00:51, 865.51 examples/s]43776 examples [00:51, 869.16 examples/s]43865 examples [00:51, 873.28 examples/s]43953 examples [00:51, 857.01 examples/s]44039 examples [00:51, 853.15 examples/s]44125 examples [00:51, 853.47 examples/s]44211 examples [00:51, 848.12 examples/s]44296 examples [00:52, 848.19 examples/s]44381 examples [00:52, 846.57 examples/s]44469 examples [00:52, 853.86 examples/s]44555 examples [00:52, 828.05 examples/s]44642 examples [00:52, 839.60 examples/s]44727 examples [00:52, 842.30 examples/s]44812 examples [00:52, 792.16 examples/s]44895 examples [00:52, 789.64 examples/s]44976 examples [00:52, 793.19 examples/s]45059 examples [00:53, 802.03 examples/s]45143 examples [00:53, 811.96 examples/s]45236 examples [00:53, 841.97 examples/s]45328 examples [00:53, 863.62 examples/s]45415 examples [00:53, 859.57 examples/s]45506 examples [00:53, 873.26 examples/s]45600 examples [00:53, 892.02 examples/s]45690 examples [00:53, 888.66 examples/s]45780 examples [00:53, 859.49 examples/s]45868 examples [00:53, 863.11 examples/s]45956 examples [00:54, 867.64 examples/s]46046 examples [00:54, 875.93 examples/s]46136 examples [00:54, 880.74 examples/s]46225 examples [00:54, 875.37 examples/s]46313 examples [00:54, 859.51 examples/s]46405 examples [00:54, 874.46 examples/s]46499 examples [00:54, 893.13 examples/s]46589 examples [00:54, 886.19 examples/s]46680 examples [00:54, 889.66 examples/s]46770 examples [00:54, 877.25 examples/s]46860 examples [00:55, 881.27 examples/s]46949 examples [00:55, 873.35 examples/s]47039 examples [00:55, 880.79 examples/s]47128 examples [00:55, 882.85 examples/s]47217 examples [00:55, 874.99 examples/s]47305 examples [00:55, 875.13 examples/s]47393 examples [00:55, 872.67 examples/s]47481 examples [00:55, 850.31 examples/s]47570 examples [00:55, 861.11 examples/s]47657 examples [00:56, 859.47 examples/s]47744 examples [00:56, 829.52 examples/s]47829 examples [00:56, 834.79 examples/s]47919 examples [00:56, 853.23 examples/s]48005 examples [00:56, 815.93 examples/s]48092 examples [00:56, 830.00 examples/s]48180 examples [00:56, 843.73 examples/s]48268 examples [00:56, 851.72 examples/s]48357 examples [00:56, 862.16 examples/s]48444 examples [00:56, 854.80 examples/s]48530 examples [00:57, 856.13 examples/s]48616 examples [00:57, 853.09 examples/s]48702 examples [00:57, 853.34 examples/s]48790 examples [00:57, 860.18 examples/s]48877 examples [00:57, 840.14 examples/s]48964 examples [00:57, 847.28 examples/s]49053 examples [00:57, 857.79 examples/s]49139 examples [00:57, 857.39 examples/s]49225 examples [00:57, 853.33 examples/s]49311 examples [00:57, 839.65 examples/s]49396 examples [00:58, 829.23 examples/s]49486 examples [00:58, 847.18 examples/s]49573 examples [00:58, 851.62 examples/s]49662 examples [00:58, 861.57 examples/s]49749 examples [00:58, 845.60 examples/s]49834 examples [00:58, 845.47 examples/s]49922 examples [00:58, 854.71 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6129/50000 [00:00<00:00, 61286.41 examples/s] 32%|███▏      | 16248/50000 [00:00<00:00, 69509.17 examples/s] 53%|█████▎    | 26524/50000 [00:00<00:00, 76981.01 examples/s] 75%|███████▍  | 37417/50000 [00:00<00:00, 84406.42 examples/s] 96%|█████████▌| 47972/50000 [00:00<00:00, 89802.87 examples/s]                                                               0 examples [00:00, ? examples/s]61 examples [00:00, 603.56 examples/s]146 examples [00:00, 661.06 examples/s]234 examples [00:00, 714.22 examples/s]326 examples [00:00, 764.76 examples/s]420 examples [00:00, 807.85 examples/s]512 examples [00:00, 837.38 examples/s]603 examples [00:00, 855.43 examples/s]694 examples [00:00, 870.62 examples/s]783 examples [00:00, 874.43 examples/s]873 examples [00:01, 879.20 examples/s]962 examples [00:01, 881.40 examples/s]1050 examples [00:01, 872.12 examples/s]1137 examples [00:01, 861.69 examples/s]1223 examples [00:01, 843.11 examples/s]1309 examples [00:01, 848.00 examples/s]1401 examples [00:01, 866.53 examples/s]1488 examples [00:01, 867.20 examples/s]1576 examples [00:01, 869.91 examples/s]1668 examples [00:01, 884.05 examples/s]1757 examples [00:02, 885.51 examples/s]1846 examples [00:02, 885.86 examples/s]1935 examples [00:02, 857.01 examples/s]2026 examples [00:02, 869.94 examples/s]2116 examples [00:02, 876.00 examples/s]2204 examples [00:02, 872.60 examples/s]2292 examples [00:02, 843.52 examples/s]2383 examples [00:02, 860.57 examples/s]2470 examples [00:02, 802.09 examples/s]2561 examples [00:02, 829.46 examples/s]2652 examples [00:03, 849.71 examples/s]2742 examples [00:03, 861.79 examples/s]2829 examples [00:03, 860.08 examples/s]2919 examples [00:03, 870.52 examples/s]3007 examples [00:03, 872.73 examples/s]3097 examples [00:03, 878.14 examples/s]3186 examples [00:03, 880.35 examples/s]3278 examples [00:03, 890.99 examples/s]3368 examples [00:03, 890.29 examples/s]3458 examples [00:03, 875.39 examples/s]3546 examples [00:04, 860.48 examples/s]3635 examples [00:04, 866.80 examples/s]3726 examples [00:04, 877.56 examples/s]3818 examples [00:04, 886.93 examples/s]3907 examples [00:04, 856.35 examples/s]4001 examples [00:04, 876.97 examples/s]4090 examples [00:04, 862.70 examples/s]4177 examples [00:04, 847.55 examples/s]4263 examples [00:04, 813.23 examples/s]4350 examples [00:05, 828.66 examples/s]4440 examples [00:05, 848.41 examples/s]4526 examples [00:05, 849.88 examples/s]4612 examples [00:05, 808.33 examples/s]4702 examples [00:05, 831.18 examples/s]4791 examples [00:05, 847.13 examples/s]4877 examples [00:05, 829.58 examples/s]4962 examples [00:05, 835.32 examples/s]5051 examples [00:05, 850.62 examples/s]5140 examples [00:05, 861.90 examples/s]5228 examples [00:06, 865.75 examples/s]5322 examples [00:06, 885.47 examples/s]5411 examples [00:06, 886.55 examples/s]5503 examples [00:06, 895.45 examples/s]5595 examples [00:06, 900.41 examples/s]5686 examples [00:06, 893.01 examples/s]5776 examples [00:06, 837.24 examples/s]5862 examples [00:06, 843.84 examples/s]5955 examples [00:06, 866.02 examples/s]6045 examples [00:07, 867.42 examples/s]6133 examples [00:07, 864.13 examples/s]6220 examples [00:07, 850.34 examples/s]6306 examples [00:07, 842.80 examples/s]6394 examples [00:07, 853.34 examples/s]6482 examples [00:07, 858.35 examples/s]6568 examples [00:07, 842.00 examples/s]6655 examples [00:07, 847.93 examples/s]6740 examples [00:07, 846.65 examples/s]6829 examples [00:07, 856.66 examples/s]6919 examples [00:08, 868.94 examples/s]7007 examples [00:08, 869.30 examples/s]7097 examples [00:08, 878.11 examples/s]7185 examples [00:08, 863.45 examples/s]7272 examples [00:08, 865.33 examples/s]7362 examples [00:08, 874.83 examples/s]7450 examples [00:08, 863.77 examples/s]7537 examples [00:08, 839.49 examples/s]7622 examples [00:08, 838.74 examples/s]7707 examples [00:08, 827.73 examples/s]7797 examples [00:09, 846.99 examples/s]7882 examples [00:09, 842.36 examples/s]7967 examples [00:09, 833.32 examples/s]8051 examples [00:09, 831.39 examples/s]8139 examples [00:09, 844.04 examples/s]8231 examples [00:09, 864.51 examples/s]8322 examples [00:09, 874.12 examples/s]8410 examples [00:09, 860.89 examples/s]8501 examples [00:09, 872.57 examples/s]8594 examples [00:09, 886.95 examples/s]8684 examples [00:10, 890.47 examples/s]8774 examples [00:10, 874.12 examples/s]8862 examples [00:10, 863.70 examples/s]8952 examples [00:10, 872.62 examples/s]9043 examples [00:10, 882.60 examples/s]9136 examples [00:10, 894.78 examples/s]9227 examples [00:10, 898.74 examples/s]9320 examples [00:10, 905.44 examples/s]9411 examples [00:10, 888.00 examples/s]9500 examples [00:11, 870.29 examples/s]9597 examples [00:11, 896.93 examples/s]9688 examples [00:11, 887.64 examples/s]9780 examples [00:11, 895.03 examples/s]9870 examples [00:11, 879.14 examples/s]9959 examples [00:11, 880.29 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete46VW0Q/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete46VW0Q/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f775b78b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f775b78b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f775b78b620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f76e87764e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f76e87b4860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f775b78b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f775b78b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f775b78b620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f77427bf5f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f77427bf358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f76e411d378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f76e411d378> 

  function with postional parmater data_info <function split_train_valid at 0x7f76e411d378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f76e411d488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f76e411d488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f76e411d488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=3a914c3b334089516ca43c268f7df91691df30b71fa9ba3e2dcddd3ed33f9731
  Stored in directory: /tmp/pip-ephem-wheel-cache-1hyqg0kl/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:54:54, 12.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:10:14, 16.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:58:22, 24.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<6:59:13, 34.2kB/s].vector_cache/glove.6B.zip:   0%|          | 2.89M/862M [00:01<4:52:59, 48.9kB/s].vector_cache/glove.6B.zip:   1%|          | 6.73M/862M [00:01<3:24:19, 69.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:22:09, 99.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:01<1:39:12, 142kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:01<1:09:07, 203kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:01<48:14, 289kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.1M/862M [00:01<33:39, 412kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.1M/862M [00:01<23:33, 586kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:01<16:28, 832kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.7M/862M [00:02<11:35, 1.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:02<08:09, 1.67MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:02<05:47, 2.33MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.7M/862M [00:02<05:57, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<06:04, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<08:34, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<06:56, 1.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.3M/862M [00:05<05:07, 2.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<07:24, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<06:50, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<05:09, 2.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<06:15, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<07:02, 1.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:09<05:31, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<06:01, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<05:33, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.2M/862M [00:11<04:13, 3.13MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<06:01, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:12<06:58, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<05:29, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:13<03:57, 3.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<19:12, 681kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<14:47, 884kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<10:39, 1.22MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<10:30, 1.24MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<10:04, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<07:42, 1.69MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:17<05:32, 2.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<1:36:40, 134kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<1:08:56, 188kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:19<48:29, 266kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<36:52, 349kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:20<27:06, 475kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:20<19:15, 667kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<16:27, 778kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:22<14:06, 907kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<10:25, 1.23MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:23<07:25, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<14:51, 857kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<11:40, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<08:28, 1.50MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:53, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:46, 1.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:42, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<04:50, 2.60MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:11:53, 17.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<8:33:20, 24.5kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<5:58:52, 35.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<4:13:27, 49.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<2:59:53, 69.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<2:06:20, 99.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<1:28:18, 141kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:09:52, 178kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<50:09, 248kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:32<35:21, 351kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<27:34, 449kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<21:47, 569kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<15:51, 781kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<13:03, 943kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<10:22, 1.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<07:33, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:09, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:17, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:20, 1.93MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<04:33, 2.67MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<15:49, 770kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<12:19, 988kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<08:55, 1.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<09:02, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:46, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:38, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:48, 2.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:22, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:46, 1.55MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:43, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<06:49, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:58, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:28, 2.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:56, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:35, 1.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:06, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<03:44, 3.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:36, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:32, 1.81MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:51, 2.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:09, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:40, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:15, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:34, 2.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:04, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:48, 3.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:24, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:07, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:51, 2.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:16, 2.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<04:52, 2.37MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:42, 3.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:16, 2.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:00, 1.91MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:47, 2.40MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:27, 3.31MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:05:48, 17.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<7:46:57, 24.5kB/s] .vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<5:26:24, 34.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<3:50:26, 49.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<2:43:32, 69.5kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<1:54:50, 98.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<1:20:14, 141kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:03:48, 177kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<45:46, 247kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<32:15, 349kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<25:08, 447kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<19:51, 565kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<14:21, 780kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<10:11, 1.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:49, 944kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:25, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<06:51, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:23, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:26, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:42, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:05, 2.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<28:51, 382kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<21:18, 517kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<15:06, 727kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<13:08, 834kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<10:17, 1.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<07:25, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:45, 1.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:37, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:52, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:51, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:11, 2.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:52, 2.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:13, 2.06MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:49, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:36, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:56, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:31, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:26, 3.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:53, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:29, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:21, 3.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:52, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:31, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:23, 2.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:45, 2.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:23, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:20, 3.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:46, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:15, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:08, 3.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:55, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:11, 1.99MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:53, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:07, 2.01MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:36, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<03:28, 2.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:50, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:33, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:10, 3.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<06:19, 1.60MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:29, 1.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:27, 1.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<05:48, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:08, 1.96MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:51, 2.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:56, 2.02MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<05:33, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:19, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:09, 3.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<06:56, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:54, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<04:21, 2.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:16, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:44, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:33, 2.75MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:41, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:21, 1.83MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:11, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:02, 3.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<08:52, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:06, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:13, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:47, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:54, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:42, 2.60MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<02:42, 3.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<15:17, 626kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<12:42, 753kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<09:18, 1.03MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<06:37, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<09:00, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:18, 1.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:20, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:51, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:06, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:45, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:26, 2.72MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<30:37, 306kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<22:24, 418kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<15:53, 588kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<13:10, 705kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<10:11, 911kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<07:21, 1.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<07:12, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:59, 1.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:17, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:49, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 313M/862M [02:06<06:33, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<05:32, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:04, 2.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:55, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:22, 1.69MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:14, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:04, 2.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<29:18, 308kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<21:26, 420kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<15:12, 591kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<12:37, 709kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<10:42, 835kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<07:54, 1.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<05:36, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<11:12, 792kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:46, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:21, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:25, 1.37MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:22, 1.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<04:50, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:30, 2.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:02, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<05:09, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<03:49, 2.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:37, 1.87MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:03, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:59, 2.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:53, 2.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<29:22, 293kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<21:27, 401kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<15:12, 564kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<12:31, 682kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<10:33, 809kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<07:45, 1.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<05:31, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<08:25, 1.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<06:46, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<04:55, 1.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:20, 1.57MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:31, 1.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:13, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<03:03, 2.73MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:21, 1.31MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:19, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:54, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<04:35, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<04:56, 1.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:53, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<02:48, 2.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<22:30, 364kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<16:28, 497kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<11:42, 697kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<09:59, 813kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<07:50, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:36<05:40, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:49, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:46, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:24, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:09, 2.53MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<07:24, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<06:01, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<04:22, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:52, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:13, 1.87MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:07, 2.52MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:59, 1.97MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:31, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<02:32, 3.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<26:31, 293kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<19:21, 402kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<13:42, 565kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<11:19, 681kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<09:33, 807kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<07:01, 1.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<04:58, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<09:04, 843kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<07:08, 1.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:09, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:18, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:18, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:06, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<02:56, 2.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<20:09, 372kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<14:53, 504kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<10:33, 708kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<09:01, 824kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:52, 944kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:53, 1.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<04:11, 1.76MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<55:45, 132kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<39:46, 185kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<27:54, 263kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<21:05, 346kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<16:17, 448kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<11:42, 622kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<08:14, 879kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<11:37, 622kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<08:46, 823kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<06:21, 1.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<04:31, 1.58MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<13:45, 521kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<11:07, 643kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<08:08, 877kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<05:44, 1.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<20:28, 346kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<15:04, 470kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<10:40, 662kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:01, 778kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:30, 934kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<05:37, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<04:01, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<05:39, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:39, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:25, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:59, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<04:14, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:17, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:21, 2.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<14:13, 479kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<10:38, 640kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<07:36, 893kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<06:51, 985kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<06:12, 1.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:41, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<03:20, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<17:49, 375kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<13:09, 507kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<09:20, 711kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<08:02, 822kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<07:00, 942kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:14, 1.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<03:43, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<22:19, 293kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<16:18, 401kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<11:32, 564kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<09:31, 680kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<08:01, 807kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<05:53, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<04:11, 1.53MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<06:02, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:53, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<03:34, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:57, 1.60MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:06, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:11, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<02:18, 2.72MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<21:33, 291kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<15:43, 398kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<11:07, 561kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<09:10, 676kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<07:39, 809kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<05:38, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<03:59, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<30:05, 204kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<21:40, 283kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<15:16, 399kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<11:59, 505kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<09:39, 627kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<07:03, 856kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<04:58, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<46:25, 129kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<33:05, 181kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<23:13, 257kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<17:29, 339kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<13:29, 439kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<09:41, 610kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<06:47, 863kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<10:43, 546kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:07, 721kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<05:48, 1.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:20, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:57, 1.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:44, 1.55MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:43<02:39, 2.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<07:30, 762kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:51, 976kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<04:13, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:13, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:08, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:08, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:15, 2.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:28, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:42, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:42, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:08, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:21, 1.64MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:35, 2.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<01:51, 2.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<09:08, 595kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<06:57, 781kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<04:57, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:42, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:26, 1.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:21, 1.59MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<02:24, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<13:24, 395kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<09:56, 533kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<07:02, 749kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:04, 861kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:22, 974kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:00, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<02:51, 1.82MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<04:52, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:58, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:52, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<03:09, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:17, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:31, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:49, 2.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<03:24, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:53, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<02:38, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:54, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:14, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:37, 3.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:10, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:26, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:31, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<02:51, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:00, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:19, 2.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:40, 2.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<04:19, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:26, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:50, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:59, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:19, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:40, 2.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<11:52, 389kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<08:46, 525kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<06:13, 737kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<05:20, 850kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:41, 968kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:31, 1.29MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<02:29, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<15:13, 294kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<11:06, 403kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<07:50, 567kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<06:27, 682kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<05:26, 809kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:00, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:49, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:06, 850kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<04:01, 1.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<02:53, 1.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:09, 1.36MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:15, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:29, 1.69MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:10, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:36, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:04, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:19, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<01:48, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:30<01:17, 3.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<04:33, 893kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<03:36, 1.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<02:36, 1.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:42, 1.48MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:44, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:05, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<01:29, 2.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:08, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:36, 1.50MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:54, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:12, 1.75MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:21, 1.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:50, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<01:19, 2.87MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<12:21, 307kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<09:02, 419kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<06:22, 590kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<05:15, 708kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<04:27, 834kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:16, 1.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<02:19, 1.58MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:20, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:42, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:58, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:10, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:16, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:46, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:46<01:16, 2.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<11:28, 306kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<08:23, 418kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<05:55, 589kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:53, 704kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:46, 912kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:41, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<02:39, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:33, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:56, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:22, 2.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:34, 926kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:50, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:09, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:11, 1.48MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:40, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:11, 2.66MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:29, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:04, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:30, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:45, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:52, 1.65MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:27, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:01, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<05:59, 507kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:29, 673kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:11, 940kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:52, 1.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:38, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:59, 1.48MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:24, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<21:42, 133kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<15:28, 187kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<10:47, 265kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<08:06, 349kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<06:13, 453kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<04:27, 629kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<03:06, 889kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<04:02, 681kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:07, 882kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:13, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:08, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:03, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:33, 1.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:07, 2.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:43, 1.52MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:28, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:05, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:20, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:28, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:09, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:49, 3.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<06:20, 391kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<04:41, 528kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<03:18, 742kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:49, 853kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:28, 970kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:50, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:17, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:57, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:34, 1.48MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:08, 2.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:19, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:23, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:04, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:45, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<12:07, 182kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<08:41, 253kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<06:04, 358kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<04:39, 459kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:42, 576kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:41, 789kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:51, 1.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<07:32, 274kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<05:28, 377kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<03:50, 530kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<03:05, 647kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<02:34, 774kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:53, 1.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<01:18, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<05:46, 334kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<04:13, 455kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<02:57, 640kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:27, 755kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<02:06, 879kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:33, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:04, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<03:16, 547kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<02:28, 723kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<01:44, 1.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:36, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:29, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:07, 1.52MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:39<00:46, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<04:22, 378kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<03:13, 511kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:41<02:16, 716kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:54, 830kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:40, 950kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:13, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:43<00:51, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:43, 876kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:22, 1.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:58, 1.53MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:59, 1.46MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:50, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:36, 2.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:43, 1.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:47, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:36, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:25, 3.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<07:10, 183kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<05:08, 254kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<03:33, 360kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<02:41, 460kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<02:08, 578kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<01:32, 795kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<01:03, 1.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:22, 856kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:04, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:45, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:46, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:46, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:24, 2.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:44, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:37, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:26, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:31, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:26, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:01<00:18, 2.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<05:56, 151kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<04:13, 211kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<02:53, 299kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<02:07, 389kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:39, 498kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:11, 685kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:47, 967kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<02:25, 313kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:50, 410kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:18, 568kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:51, 803kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<05:32, 125kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<03:55, 175kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<02:39, 248kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:53, 328kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<01:22, 447kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:56, 629kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:44, 746kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:37, 870kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:27, 1.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<00:18, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:27, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:22, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:15, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:15, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:15, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:07, 2.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:21, 980kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:16, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:11, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:10, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:10, 1.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:07, 1.96MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:04, 2.71MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:12, 973kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:09, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:06, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.95MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 2.68MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.40MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.24MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<23:08:18,  4.80it/s]  0%|          | 742/400000 [00:00<16:10:16,  6.86it/s]  0%|          | 1517/400000 [00:00<11:18:08,  9.79it/s]  1%|          | 2251/400000 [00:00<7:54:05, 13.98it/s]   1%|          | 2997/400000 [00:00<5:31:30, 19.96it/s]  1%|          | 3732/400000 [00:00<3:51:53, 28.48it/s]  1%|          | 4472/400000 [00:00<2:42:17, 40.62it/s]  1%|▏         | 5260/400000 [00:00<1:53:37, 57.90it/s]  2%|▏         | 6021/400000 [00:01<1:19:38, 82.44it/s]  2%|▏         | 6736/400000 [00:01<55:55, 117.19it/s]   2%|▏         | 7458/400000 [00:01<39:21, 166.26it/s]  2%|▏         | 8211/400000 [00:01<27:45, 235.29it/s]  2%|▏         | 8951/400000 [00:01<19:39, 331.60it/s]  2%|▏         | 9703/400000 [00:01<13:59, 464.93it/s]  3%|▎         | 10448/400000 [00:01<10:02, 646.88it/s]  3%|▎         | 11212/400000 [00:01<07:15, 891.74it/s]  3%|▎         | 11985/400000 [00:01<05:19, 1213.84it/s]  3%|▎         | 12740/400000 [00:01<04:00, 1611.83it/s]  3%|▎         | 13509/400000 [00:02<03:02, 2112.67it/s]  4%|▎         | 14273/400000 [00:02<02:22, 2698.06it/s]  4%|▍         | 15023/400000 [00:02<01:55, 3333.17it/s]  4%|▍         | 15785/400000 [00:02<01:35, 4009.30it/s]  4%|▍         | 16537/400000 [00:02<01:23, 4611.28it/s]  4%|▍         | 17283/400000 [00:02<01:13, 5206.38it/s]  5%|▍         | 18034/400000 [00:02<01:06, 5731.45it/s]  5%|▍         | 18796/400000 [00:02<01:01, 6191.40it/s]  5%|▍         | 19551/400000 [00:02<00:58, 6544.29it/s]  5%|▌         | 20315/400000 [00:02<00:55, 6834.52it/s]  5%|▌         | 21107/400000 [00:03<00:53, 7126.36it/s]  5%|▌         | 21873/400000 [00:03<00:52, 7235.02it/s]  6%|▌         | 22634/400000 [00:03<00:51, 7266.61it/s]  6%|▌         | 23387/400000 [00:03<00:51, 7341.65it/s]  6%|▌         | 24140/400000 [00:03<00:51, 7349.05it/s]  6%|▌         | 24919/400000 [00:03<00:50, 7474.07it/s]  6%|▋         | 25685/400000 [00:03<00:49, 7526.48it/s]  7%|▋         | 26445/400000 [00:03<00:49, 7483.84it/s]  7%|▋         | 27252/400000 [00:03<00:48, 7648.48it/s]  7%|▋         | 28041/400000 [00:03<00:48, 7718.28it/s]  7%|▋         | 28843/400000 [00:04<00:47, 7804.59it/s]  7%|▋         | 29626/400000 [00:04<00:47, 7725.69it/s]  8%|▊         | 30401/400000 [00:04<00:48, 7599.16it/s]  8%|▊         | 31163/400000 [00:04<00:48, 7604.63it/s]  8%|▊         | 31925/400000 [00:04<00:48, 7514.56it/s]  8%|▊         | 32690/400000 [00:04<00:48, 7551.70it/s]  8%|▊         | 33458/400000 [00:04<00:48, 7588.76it/s]  9%|▊         | 34218/400000 [00:04<00:48, 7550.53it/s]  9%|▊         | 34992/400000 [00:04<00:47, 7605.75it/s]  9%|▉         | 35770/400000 [00:04<00:47, 7655.03it/s]  9%|▉         | 36537/400000 [00:05<00:47, 7659.22it/s]  9%|▉         | 37323/400000 [00:05<00:46, 7717.17it/s] 10%|▉         | 38096/400000 [00:05<00:47, 7629.88it/s] 10%|▉         | 38862/400000 [00:05<00:47, 7638.39it/s] 10%|▉         | 39627/400000 [00:05<00:47, 7620.11it/s] 10%|█         | 40450/400000 [00:05<00:46, 7791.64it/s] 10%|█         | 41231/400000 [00:05<00:46, 7652.12it/s] 10%|█         | 41998/400000 [00:05<00:46, 7637.71it/s] 11%|█         | 42795/400000 [00:05<00:46, 7734.29it/s] 11%|█         | 43570/400000 [00:05<00:46, 7641.27it/s] 11%|█         | 44336/400000 [00:06<00:47, 7553.85it/s] 11%|█▏        | 45093/400000 [00:06<00:47, 7488.93it/s] 11%|█▏        | 45843/400000 [00:06<00:47, 7473.05it/s] 12%|█▏        | 46591/400000 [00:06<00:47, 7467.46it/s] 12%|█▏        | 47350/400000 [00:06<00:47, 7502.17it/s] 12%|█▏        | 48101/400000 [00:06<00:47, 7349.56it/s] 12%|█▏        | 48839/400000 [00:06<00:47, 7356.19it/s] 12%|█▏        | 49576/400000 [00:06<00:47, 7350.46it/s] 13%|█▎        | 50352/400000 [00:06<00:46, 7467.07it/s] 13%|█▎        | 51123/400000 [00:06<00:46, 7537.14it/s] 13%|█▎        | 51878/400000 [00:07<00:47, 7375.66it/s] 13%|█▎        | 52652/400000 [00:07<00:46, 7480.95it/s] 13%|█▎        | 53402/400000 [00:07<00:46, 7438.55it/s] 14%|█▎        | 54161/400000 [00:07<00:46, 7480.86it/s] 14%|█▎        | 54915/400000 [00:07<00:46, 7496.11it/s] 14%|█▍        | 55666/400000 [00:07<00:46, 7457.47it/s] 14%|█▍        | 56413/400000 [00:07<00:46, 7448.87it/s] 14%|█▍        | 57159/400000 [00:07<00:46, 7436.69it/s] 14%|█▍        | 57903/400000 [00:07<00:46, 7426.27it/s] 15%|█▍        | 58646/400000 [00:08<00:45, 7424.60it/s] 15%|█▍        | 59416/400000 [00:08<00:45, 7504.92it/s] 15%|█▌        | 60167/400000 [00:08<00:45, 7488.84it/s] 15%|█▌        | 60917/400000 [00:08<00:46, 7334.74it/s] 15%|█▌        | 61666/400000 [00:08<00:45, 7379.13it/s] 16%|█▌        | 62405/400000 [00:08<00:46, 7277.19it/s] 16%|█▌        | 63144/400000 [00:08<00:46, 7309.43it/s] 16%|█▌        | 63892/400000 [00:08<00:45, 7357.96it/s] 16%|█▌        | 64629/400000 [00:08<00:45, 7322.96it/s] 16%|█▋        | 65362/400000 [00:08<00:45, 7300.63it/s] 17%|█▋        | 66141/400000 [00:09<00:44, 7439.44it/s] 17%|█▋        | 66905/400000 [00:09<00:44, 7496.29it/s] 17%|█▋        | 67664/400000 [00:09<00:44, 7522.87it/s] 17%|█▋        | 68417/400000 [00:09<00:44, 7432.80it/s] 17%|█▋        | 69161/400000 [00:09<00:44, 7413.79it/s] 17%|█▋        | 69903/400000 [00:09<00:45, 7322.41it/s] 18%|█▊        | 70636/400000 [00:09<00:45, 7209.53it/s] 18%|█▊        | 71358/400000 [00:09<00:45, 7174.69it/s] 18%|█▊        | 72077/400000 [00:09<00:46, 7001.21it/s] 18%|█▊        | 72808/400000 [00:09<00:46, 7089.92it/s] 18%|█▊        | 73532/400000 [00:10<00:45, 7133.37it/s] 19%|█▊        | 74247/400000 [00:10<00:46, 7060.75it/s] 19%|█▉        | 75013/400000 [00:10<00:44, 7228.37it/s] 19%|█▉        | 75762/400000 [00:10<00:44, 7303.77it/s] 19%|█▉        | 76494/400000 [00:10<00:44, 7264.89it/s] 19%|█▉        | 77222/400000 [00:10<00:44, 7243.89it/s] 19%|█▉        | 77948/400000 [00:10<00:44, 7194.08it/s] 20%|█▉        | 78668/400000 [00:10<00:45, 7118.64it/s] 20%|█▉        | 79410/400000 [00:10<00:44, 7205.59it/s] 20%|██        | 80173/400000 [00:10<00:43, 7327.37it/s] 20%|██        | 80924/400000 [00:11<00:43, 7380.00it/s] 20%|██        | 81692/400000 [00:11<00:42, 7466.51it/s] 21%|██        | 82440/400000 [00:11<00:43, 7313.06it/s] 21%|██        | 83173/400000 [00:11<00:43, 7288.36it/s] 21%|██        | 83912/400000 [00:11<00:43, 7315.56it/s] 21%|██        | 84668/400000 [00:11<00:42, 7384.24it/s] 21%|██▏       | 85424/400000 [00:11<00:42, 7429.52it/s] 22%|██▏       | 86168/400000 [00:11<00:42, 7369.15it/s] 22%|██▏       | 86906/400000 [00:11<00:43, 7275.05it/s] 22%|██▏       | 87641/400000 [00:11<00:42, 7296.97it/s] 22%|██▏       | 88372/400000 [00:12<00:43, 7172.56it/s] 22%|██▏       | 89098/400000 [00:12<00:43, 7196.57it/s] 22%|██▏       | 89819/400000 [00:12<00:43, 7121.60it/s] 23%|██▎       | 90532/400000 [00:12<00:44, 6944.30it/s] 23%|██▎       | 91256/400000 [00:12<00:43, 7026.91it/s] 23%|██▎       | 91998/400000 [00:12<00:43, 7139.74it/s] 23%|██▎       | 92753/400000 [00:12<00:42, 7257.43it/s] 23%|██▎       | 93533/400000 [00:12<00:41, 7410.32it/s] 24%|██▎       | 94276/400000 [00:12<00:41, 7333.57it/s] 24%|██▍       | 95011/400000 [00:12<00:41, 7269.69it/s] 24%|██▍       | 95740/400000 [00:13<00:42, 7127.08it/s] 24%|██▍       | 96455/400000 [00:13<00:43, 7055.95it/s] 24%|██▍       | 97162/400000 [00:13<00:43, 6993.59it/s] 24%|██▍       | 97863/400000 [00:13<00:43, 6955.76it/s] 25%|██▍       | 98591/400000 [00:13<00:42, 7049.47it/s] 25%|██▍       | 99313/400000 [00:13<00:42, 7098.74it/s] 25%|██▌       | 100059/400000 [00:13<00:41, 7201.81it/s] 25%|██▌       | 100813/400000 [00:13<00:41, 7296.48it/s] 25%|██▌       | 101544/400000 [00:13<00:40, 7294.59it/s] 26%|██▌       | 102275/400000 [00:14<00:40, 7292.34it/s] 26%|██▌       | 103050/400000 [00:14<00:40, 7423.71it/s] 26%|██▌       | 103802/400000 [00:14<00:39, 7451.93it/s] 26%|██▌       | 104582/400000 [00:14<00:39, 7551.41it/s] 26%|██▋       | 105338/400000 [00:14<00:39, 7498.28it/s] 27%|██▋       | 106089/400000 [00:14<00:39, 7418.78it/s] 27%|██▋       | 106855/400000 [00:14<00:39, 7486.94it/s] 27%|██▋       | 107649/400000 [00:14<00:38, 7616.29it/s] 27%|██▋       | 108440/400000 [00:14<00:37, 7697.58it/s] 27%|██▋       | 109211/400000 [00:14<00:37, 7656.92it/s] 27%|██▋       | 109995/400000 [00:15<00:37, 7708.42it/s] 28%|██▊       | 110767/400000 [00:15<00:38, 7588.55it/s] 28%|██▊       | 111534/400000 [00:15<00:37, 7610.51it/s] 28%|██▊       | 112296/400000 [00:15<00:38, 7564.24it/s] 28%|██▊       | 113053/400000 [00:15<00:38, 7483.02it/s] 28%|██▊       | 113802/400000 [00:15<00:38, 7408.02it/s] 29%|██▊       | 114558/400000 [00:15<00:38, 7451.23it/s] 29%|██▉       | 115328/400000 [00:15<00:37, 7523.86it/s] 29%|██▉       | 116081/400000 [00:15<00:37, 7525.02it/s] 29%|██▉       | 116834/400000 [00:15<00:37, 7503.70it/s] 29%|██▉       | 117635/400000 [00:16<00:36, 7648.43it/s] 30%|██▉       | 118426/400000 [00:16<00:36, 7724.87it/s] 30%|██▉       | 119200/400000 [00:16<00:36, 7720.91it/s] 30%|██▉       | 119973/400000 [00:16<00:36, 7646.08it/s] 30%|███       | 120739/400000 [00:16<00:36, 7611.48it/s] 30%|███       | 121504/400000 [00:16<00:36, 7622.12it/s] 31%|███       | 122267/400000 [00:16<00:37, 7321.83it/s] 31%|███       | 123015/400000 [00:16<00:37, 7368.01it/s] 31%|███       | 123789/400000 [00:16<00:36, 7475.05it/s] 31%|███       | 124550/400000 [00:16<00:36, 7513.78it/s] 31%|███▏      | 125303/400000 [00:17<00:37, 7390.15it/s] 32%|███▏      | 126044/400000 [00:17<00:37, 7362.42it/s] 32%|███▏      | 126782/400000 [00:17<00:37, 7350.62it/s] 32%|███▏      | 127518/400000 [00:17<00:37, 7321.86it/s] 32%|███▏      | 128251/400000 [00:17<00:37, 7234.78it/s] 32%|███▏      | 128976/400000 [00:17<00:37, 7212.58it/s] 32%|███▏      | 129698/400000 [00:17<00:37, 7180.78it/s] 33%|███▎      | 130422/400000 [00:17<00:37, 7197.04it/s] 33%|███▎      | 131164/400000 [00:17<00:37, 7260.19it/s] 33%|███▎      | 131911/400000 [00:17<00:36, 7317.62it/s] 33%|███▎      | 132656/400000 [00:18<00:36, 7356.26it/s] 33%|███▎      | 133436/400000 [00:18<00:35, 7481.20it/s] 34%|███▎      | 134211/400000 [00:18<00:35, 7559.39it/s] 34%|███▎      | 134968/400000 [00:18<00:35, 7428.59it/s] 34%|███▍      | 135712/400000 [00:18<00:35, 7376.24it/s] 34%|███▍      | 136463/400000 [00:18<00:35, 7414.14it/s] 34%|███▍      | 137214/400000 [00:18<00:35, 7442.11it/s] 34%|███▍      | 137959/400000 [00:18<00:35, 7412.08it/s] 35%|███▍      | 138708/400000 [00:18<00:35, 7433.00it/s] 35%|███▍      | 139452/400000 [00:18<00:35, 7397.78it/s] 35%|███▌      | 140212/400000 [00:19<00:34, 7455.18it/s] 35%|███▌      | 140996/400000 [00:19<00:34, 7565.93it/s] 35%|███▌      | 141785/400000 [00:19<00:33, 7660.09it/s] 36%|███▌      | 142552/400000 [00:19<00:34, 7500.05it/s] 36%|███▌      | 143313/400000 [00:19<00:34, 7531.05it/s] 36%|███▌      | 144067/400000 [00:19<00:34, 7508.81it/s] 36%|███▌      | 144826/400000 [00:19<00:33, 7532.03it/s] 36%|███▋      | 145580/400000 [00:19<00:33, 7532.11it/s] 37%|███▋      | 146334/400000 [00:19<00:33, 7480.19it/s] 37%|███▋      | 147083/400000 [00:20<00:34, 7400.08it/s] 37%|███▋      | 147831/400000 [00:20<00:33, 7422.07it/s] 37%|███▋      | 148600/400000 [00:20<00:33, 7499.72it/s] 37%|███▋      | 149375/400000 [00:20<00:33, 7571.93it/s] 38%|███▊      | 150153/400000 [00:20<00:32, 7629.62it/s] 38%|███▊      | 150917/400000 [00:20<00:32, 7595.08it/s] 38%|███▊      | 151677/400000 [00:20<00:33, 7520.98it/s] 38%|███▊      | 152430/400000 [00:20<00:33, 7364.10it/s] 38%|███▊      | 153185/400000 [00:20<00:33, 7418.09it/s] 38%|███▊      | 153928/400000 [00:20<00:33, 7340.99it/s] 39%|███▊      | 154663/400000 [00:21<00:33, 7289.82it/s] 39%|███▉      | 155393/400000 [00:21<00:34, 7101.26it/s] 39%|███▉      | 156105/400000 [00:21<00:34, 7048.48it/s] 39%|███▉      | 156811/400000 [00:21<00:34, 7049.43it/s] 39%|███▉      | 157578/400000 [00:21<00:33, 7223.93it/s] 40%|███▉      | 158324/400000 [00:21<00:33, 7290.61it/s] 40%|███▉      | 159079/400000 [00:21<00:32, 7365.54it/s] 40%|███▉      | 159817/400000 [00:21<00:33, 7234.14it/s] 40%|████      | 160590/400000 [00:21<00:32, 7375.45it/s] 40%|████      | 161376/400000 [00:21<00:31, 7512.74it/s] 41%|████      | 162159/400000 [00:22<00:31, 7604.38it/s] 41%|████      | 162921/400000 [00:22<00:31, 7593.62it/s] 41%|████      | 163710/400000 [00:22<00:30, 7677.89it/s] 41%|████      | 164479/400000 [00:22<00:30, 7598.61it/s] 41%|████▏     | 165243/400000 [00:22<00:30, 7608.02it/s] 42%|████▏     | 166005/400000 [00:22<00:31, 7542.46it/s] 42%|████▏     | 166760/400000 [00:22<00:31, 7431.53it/s] 42%|████▏     | 167504/400000 [00:22<00:31, 7369.50it/s] 42%|████▏     | 168265/400000 [00:22<00:31, 7440.00it/s] 42%|████▏     | 169038/400000 [00:22<00:30, 7523.54it/s] 42%|████▏     | 169797/400000 [00:23<00:30, 7541.39it/s] 43%|████▎     | 170552/400000 [00:23<00:30, 7513.04it/s] 43%|████▎     | 171355/400000 [00:23<00:29, 7659.76it/s] 43%|████▎     | 172158/400000 [00:23<00:29, 7766.04it/s] 43%|████▎     | 172936/400000 [00:23<00:29, 7745.02it/s] 43%|████▎     | 173712/400000 [00:23<00:29, 7674.00it/s] 44%|████▎     | 174481/400000 [00:23<00:29, 7583.65it/s] 44%|████▍     | 175241/400000 [00:23<00:29, 7573.50it/s] 44%|████▍     | 176019/400000 [00:23<00:29, 7631.99it/s] 44%|████▍     | 176783/400000 [00:23<00:29, 7572.20it/s] 44%|████▍     | 177541/400000 [00:24<00:29, 7423.34it/s] 45%|████▍     | 178285/400000 [00:24<00:30, 7318.83it/s] 45%|████▍     | 179038/400000 [00:24<00:29, 7378.59it/s] 45%|████▍     | 179778/400000 [00:24<00:29, 7384.24it/s] 45%|████▌     | 180517/400000 [00:24<00:29, 7377.23it/s] 45%|████▌     | 181256/400000 [00:24<00:29, 7322.69it/s] 45%|████▌     | 181989/400000 [00:24<00:30, 7263.88it/s] 46%|████▌     | 182716/400000 [00:24<00:30, 7232.12it/s] 46%|████▌     | 183479/400000 [00:24<00:29, 7344.79it/s] 46%|████▌     | 184260/400000 [00:24<00:28, 7477.16it/s] 46%|████▋     | 185058/400000 [00:25<00:28, 7617.87it/s] 46%|████▋     | 185822/400000 [00:25<00:29, 7378.40it/s] 47%|████▋     | 186563/400000 [00:25<00:28, 7386.01it/s] 47%|████▋     | 187304/400000 [00:25<00:28, 7359.68it/s] 47%|████▋     | 188110/400000 [00:25<00:28, 7553.45it/s] 47%|████▋     | 188905/400000 [00:25<00:27, 7667.44it/s] 47%|████▋     | 189674/400000 [00:25<00:28, 7444.34it/s] 48%|████▊     | 190422/400000 [00:25<00:28, 7287.45it/s] 48%|████▊     | 191207/400000 [00:25<00:28, 7446.10it/s] 48%|████▊     | 191996/400000 [00:26<00:27, 7571.57it/s] 48%|████▊     | 192778/400000 [00:26<00:27, 7643.85it/s] 48%|████▊     | 193551/400000 [00:26<00:26, 7668.54it/s] 49%|████▊     | 194329/400000 [00:26<00:26, 7700.81it/s] 49%|████▉     | 195104/400000 [00:26<00:26, 7713.85it/s] 49%|████▉     | 195914/400000 [00:26<00:26, 7825.62it/s] 49%|████▉     | 196698/400000 [00:26<00:26, 7652.01it/s] 49%|████▉     | 197465/400000 [00:26<00:26, 7608.25it/s] 50%|████▉     | 198227/400000 [00:26<00:26, 7511.02it/s] 50%|████▉     | 199010/400000 [00:26<00:26, 7602.04it/s] 50%|████▉     | 199786/400000 [00:27<00:26, 7646.17it/s] 50%|█████     | 200552/400000 [00:27<00:26, 7461.26it/s] 50%|█████     | 201300/400000 [00:27<00:26, 7363.88it/s] 51%|█████     | 202038/400000 [00:27<00:26, 7343.77it/s] 51%|█████     | 202774/400000 [00:27<00:26, 7316.36it/s] 51%|█████     | 203541/400000 [00:27<00:26, 7418.74it/s] 51%|█████     | 204284/400000 [00:27<00:26, 7371.24it/s] 51%|█████▏    | 205063/400000 [00:27<00:26, 7492.05it/s] 51%|█████▏    | 205826/400000 [00:27<00:25, 7532.08it/s] 52%|█████▏    | 206580/400000 [00:27<00:25, 7492.31it/s] 52%|█████▏    | 207330/400000 [00:28<00:26, 7379.20it/s] 52%|█████▏    | 208069/400000 [00:28<00:26, 7323.53it/s] 52%|█████▏    | 208802/400000 [00:28<00:26, 7304.56it/s] 52%|█████▏    | 209588/400000 [00:28<00:25, 7461.24it/s] 53%|█████▎    | 210400/400000 [00:28<00:24, 7646.89it/s] 53%|█████▎    | 211167/400000 [00:28<00:24, 7624.23it/s] 53%|█████▎    | 211931/400000 [00:28<00:24, 7600.30it/s] 53%|█████▎    | 212693/400000 [00:28<00:24, 7585.26it/s] 53%|█████▎    | 213453/400000 [00:28<00:24, 7572.79it/s] 54%|█████▎    | 214212/400000 [00:28<00:24, 7576.96it/s] 54%|█████▎    | 214993/400000 [00:29<00:24, 7644.38it/s] 54%|█████▍    | 215760/400000 [00:29<00:24, 7651.87it/s] 54%|█████▍    | 216526/400000 [00:29<00:24, 7611.76it/s] 54%|█████▍    | 217313/400000 [00:29<00:23, 7685.44it/s] 55%|█████▍    | 218082/400000 [00:29<00:23, 7674.67it/s] 55%|█████▍    | 218850/400000 [00:29<00:23, 7582.07it/s] 55%|█████▍    | 219609/400000 [00:29<00:23, 7535.43it/s] 55%|█████▌    | 220363/400000 [00:29<00:23, 7521.49it/s] 55%|█████▌    | 221135/400000 [00:29<00:23, 7578.69it/s] 55%|█████▌    | 221903/400000 [00:29<00:23, 7605.98it/s] 56%|█████▌    | 222688/400000 [00:30<00:23, 7676.96it/s] 56%|█████▌    | 223473/400000 [00:30<00:22, 7727.46it/s] 56%|█████▌    | 224247/400000 [00:30<00:23, 7582.07it/s] 56%|█████▋    | 225007/400000 [00:30<00:23, 7579.31it/s] 56%|█████▋    | 225770/400000 [00:30<00:22, 7592.25it/s] 57%|█████▋    | 226530/400000 [00:30<00:23, 7315.96it/s] 57%|█████▋    | 227297/400000 [00:30<00:23, 7417.83it/s] 57%|█████▋    | 228046/400000 [00:30<00:23, 7437.41it/s] 57%|█████▋    | 228820/400000 [00:30<00:22, 7524.23it/s] 57%|█████▋    | 229636/400000 [00:30<00:22, 7703.08it/s] 58%|█████▊    | 230423/400000 [00:31<00:21, 7749.83it/s] 58%|█████▊    | 231200/400000 [00:31<00:22, 7617.37it/s] 58%|█████▊    | 231964/400000 [00:31<00:22, 7530.99it/s] 58%|█████▊    | 232719/400000 [00:31<00:22, 7497.46it/s] 58%|█████▊    | 233512/400000 [00:31<00:21, 7621.70it/s] 59%|█████▊    | 234276/400000 [00:31<00:21, 7596.04it/s] 59%|█████▉    | 235037/400000 [00:31<00:21, 7540.38it/s] 59%|█████▉    | 235792/400000 [00:31<00:21, 7483.20it/s] 59%|█████▉    | 236541/400000 [00:31<00:22, 7429.06it/s] 59%|█████▉    | 237285/400000 [00:32<00:22, 7368.67it/s] 60%|█████▉    | 238032/400000 [00:32<00:21, 7397.98it/s] 60%|█████▉    | 238788/400000 [00:32<00:21, 7445.81it/s] 60%|█████▉    | 239589/400000 [00:32<00:21, 7605.24it/s] 60%|██████    | 240351/400000 [00:32<00:21, 7532.52it/s] 60%|██████    | 241117/400000 [00:32<00:20, 7569.68it/s] 60%|██████    | 241943/400000 [00:32<00:20, 7763.34it/s] 61%|██████    | 242722/400000 [00:32<00:20, 7653.27it/s] 61%|██████    | 243541/400000 [00:32<00:20, 7803.26it/s] 61%|██████    | 244346/400000 [00:32<00:19, 7872.49it/s] 61%|██████▏   | 245135/400000 [00:33<00:19, 7789.47it/s] 61%|██████▏   | 245925/400000 [00:33<00:19, 7821.96it/s] 62%|██████▏   | 246709/400000 [00:33<00:19, 7710.59it/s] 62%|██████▏   | 247504/400000 [00:33<00:19, 7778.64it/s] 62%|██████▏   | 248283/400000 [00:33<00:19, 7756.82it/s] 62%|██████▏   | 249060/400000 [00:33<00:19, 7621.83it/s] 62%|██████▏   | 249824/400000 [00:33<00:19, 7589.38it/s] 63%|██████▎   | 250584/400000 [00:33<00:20, 7335.15it/s] 63%|██████▎   | 251367/400000 [00:33<00:19, 7476.55it/s] 63%|██████▎   | 252136/400000 [00:33<00:19, 7538.07it/s] 63%|██████▎   | 252892/400000 [00:34<00:19, 7542.42it/s] 63%|██████▎   | 253648/400000 [00:34<00:19, 7502.74it/s] 64%|██████▎   | 254400/400000 [00:34<00:19, 7376.24it/s] 64%|██████▍   | 255192/400000 [00:34<00:19, 7530.10it/s] 64%|██████▍   | 255971/400000 [00:34<00:18, 7605.56it/s] 64%|██████▍   | 256737/400000 [00:34<00:18, 7619.54it/s] 64%|██████▍   | 257518/400000 [00:34<00:18, 7673.71it/s] 65%|██████▍   | 258287/400000 [00:34<00:18, 7569.90it/s] 65%|██████▍   | 259082/400000 [00:34<00:18, 7679.12it/s] 65%|██████▍   | 259851/400000 [00:34<00:18, 7643.89it/s] 65%|██████▌   | 260654/400000 [00:35<00:17, 7753.19it/s] 65%|██████▌   | 261437/400000 [00:35<00:17, 7775.78it/s] 66%|██████▌   | 262216/400000 [00:35<00:17, 7751.56it/s] 66%|██████▌   | 263012/400000 [00:35<00:17, 7812.47it/s] 66%|██████▌   | 263808/400000 [00:35<00:17, 7853.09it/s] 66%|██████▌   | 264594/400000 [00:35<00:17, 7798.14it/s] 66%|██████▋   | 265375/400000 [00:35<00:17, 7712.67it/s] 67%|██████▋   | 266147/400000 [00:35<00:17, 7629.31it/s] 67%|██████▋   | 266911/400000 [00:35<00:17, 7565.83it/s] 67%|██████▋   | 267669/400000 [00:35<00:17, 7408.47it/s] 67%|██████▋   | 268434/400000 [00:36<00:17, 7477.06it/s] 67%|██████▋   | 269235/400000 [00:36<00:17, 7627.29it/s] 68%|██████▊   | 270000/400000 [00:36<00:17, 7584.10it/s] 68%|██████▊   | 270760/400000 [00:36<00:17, 7586.89it/s] 68%|██████▊   | 271537/400000 [00:36<00:16, 7639.16it/s] 68%|██████▊   | 272318/400000 [00:36<00:16, 7687.51it/s] 68%|██████▊   | 273088/400000 [00:36<00:16, 7497.79it/s] 68%|██████▊   | 273840/400000 [00:36<00:16, 7428.05it/s] 69%|██████▊   | 274584/400000 [00:36<00:17, 7340.54it/s] 69%|██████▉   | 275324/400000 [00:37<00:16, 7357.72it/s] 69%|██████▉   | 276101/400000 [00:37<00:16, 7475.23it/s] 69%|██████▉   | 276866/400000 [00:37<00:16, 7524.36it/s] 69%|██████▉   | 277620/400000 [00:37<00:16, 7517.60it/s] 70%|██████▉   | 278373/400000 [00:37<00:16, 7518.81it/s] 70%|██████▉   | 279171/400000 [00:37<00:15, 7647.24it/s] 70%|██████▉   | 279937/400000 [00:37<00:15, 7583.68it/s] 70%|███████   | 280710/400000 [00:37<00:15, 7626.84it/s] 70%|███████   | 281474/400000 [00:37<00:15, 7582.80it/s] 71%|███████   | 282233/400000 [00:37<00:15, 7442.58it/s] 71%|███████   | 282979/400000 [00:38<00:15, 7365.31it/s] 71%|███████   | 283744/400000 [00:38<00:15, 7448.23it/s] 71%|███████   | 284515/400000 [00:38<00:15, 7523.40it/s] 71%|███████▏  | 285269/400000 [00:38<00:15, 7499.69it/s] 72%|███████▏  | 286023/400000 [00:38<00:15, 7509.53it/s] 72%|███████▏  | 286802/400000 [00:38<00:14, 7589.50it/s] 72%|███████▏  | 287562/400000 [00:38<00:14, 7580.58it/s] 72%|███████▏  | 288321/400000 [00:38<00:14, 7572.50it/s] 72%|███████▏  | 289100/400000 [00:38<00:14, 7635.50it/s] 72%|███████▏  | 289916/400000 [00:38<00:14, 7783.73it/s] 73%|███████▎  | 290696/400000 [00:39<00:14, 7688.39it/s] 73%|███████▎  | 291522/400000 [00:39<00:13, 7850.51it/s] 73%|███████▎  | 292309/400000 [00:39<00:13, 7778.65it/s] 73%|███████▎  | 293089/400000 [00:39<00:13, 7765.88it/s] 73%|███████▎  | 293867/400000 [00:39<00:14, 7519.59it/s] 74%|███████▎  | 294622/400000 [00:39<00:14, 7523.01it/s] 74%|███████▍  | 295376/400000 [00:39<00:14, 7449.40it/s] 74%|███████▍  | 296123/400000 [00:39<00:14, 7291.06it/s] 74%|███████▍  | 296854/400000 [00:39<00:14, 7108.25it/s] 74%|███████▍  | 297581/400000 [00:39<00:14, 7154.11it/s] 75%|███████▍  | 298298/400000 [00:40<00:14, 7023.44it/s] 75%|███████▍  | 299046/400000 [00:40<00:14, 7152.14it/s] 75%|███████▍  | 299814/400000 [00:40<00:13, 7301.17it/s] 75%|███████▌  | 300569/400000 [00:40<00:13, 7372.51it/s] 75%|███████▌  | 301315/400000 [00:40<00:13, 7395.98it/s] 76%|███████▌  | 302056/400000 [00:40<00:13, 7398.66it/s] 76%|███████▌  | 302825/400000 [00:40<00:12, 7482.94it/s] 76%|███████▌  | 303578/400000 [00:40<00:12, 7496.77it/s] 76%|███████▌  | 304329/400000 [00:40<00:13, 7280.58it/s] 76%|███████▋  | 305059/400000 [00:40<00:13, 7211.90it/s] 76%|███████▋  | 305785/400000 [00:41<00:13, 7225.97it/s] 77%|███████▋  | 306558/400000 [00:41<00:12, 7367.20it/s] 77%|███████▋  | 307320/400000 [00:41<00:12, 7439.14it/s] 77%|███████▋  | 308066/400000 [00:41<00:12, 7370.32it/s] 77%|███████▋  | 308844/400000 [00:41<00:12, 7486.80it/s] 77%|███████▋  | 309594/400000 [00:41<00:12, 7427.79it/s] 78%|███████▊  | 310338/400000 [00:41<00:12, 7411.43it/s] 78%|███████▊  | 311129/400000 [00:41<00:11, 7553.64it/s] 78%|███████▊  | 311886/400000 [00:41<00:11, 7490.69it/s] 78%|███████▊  | 312669/400000 [00:42<00:11, 7589.24it/s] 78%|███████▊  | 313452/400000 [00:42<00:11, 7658.52it/s] 79%|███████▊  | 314259/400000 [00:42<00:11, 7776.01it/s] 79%|███████▉  | 315039/400000 [00:42<00:10, 7782.56it/s] 79%|███████▉  | 315818/400000 [00:42<00:11, 7639.25it/s] 79%|███████▉  | 316584/400000 [00:42<00:10, 7585.74it/s] 79%|███████▉  | 317344/400000 [00:42<00:11, 7424.70it/s] 80%|███████▉  | 318089/400000 [00:42<00:11, 7431.99it/s] 80%|███████▉  | 318874/400000 [00:42<00:10, 7550.15it/s] 80%|███████▉  | 319631/400000 [00:42<00:10, 7360.60it/s] 80%|████████  | 320429/400000 [00:43<00:10, 7534.35it/s] 80%|████████  | 321194/400000 [00:43<00:10, 7566.96it/s] 80%|████████  | 321978/400000 [00:43<00:10, 7646.69it/s] 81%|████████  | 322772/400000 [00:43<00:09, 7730.94it/s] 81%|████████  | 323547/400000 [00:43<00:09, 7651.86it/s] 81%|████████  | 324323/400000 [00:43<00:09, 7680.65it/s] 81%|████████▏ | 325109/400000 [00:43<00:09, 7731.71it/s] 81%|████████▏ | 325897/400000 [00:43<00:09, 7761.07it/s] 82%|████████▏ | 326674/400000 [00:43<00:09, 7515.60it/s] 82%|████████▏ | 327428/400000 [00:43<00:09, 7522.01it/s] 82%|████████▏ | 328208/400000 [00:44<00:09, 7603.00it/s] 82%|████████▏ | 328970/400000 [00:44<00:09, 7606.67it/s] 82%|████████▏ | 329743/400000 [00:44<00:09, 7640.83it/s] 83%|████████▎ | 330508/400000 [00:44<00:09, 7563.40it/s] 83%|████████▎ | 331265/400000 [00:44<00:09, 7446.94it/s] 83%|████████▎ | 332028/400000 [00:44<00:09, 7499.73it/s] 83%|████████▎ | 332779/400000 [00:44<00:08, 7499.89it/s] 83%|████████▎ | 333530/400000 [00:44<00:08, 7484.71it/s] 84%|████████▎ | 334280/400000 [00:44<00:08, 7486.55it/s] 84%|████████▍ | 335029/400000 [00:44<00:08, 7383.49it/s] 84%|████████▍ | 335804/400000 [00:45<00:08, 7489.44it/s] 84%|████████▍ | 336588/400000 [00:45<00:08, 7590.52it/s] 84%|████████▍ | 337396/400000 [00:45<00:08, 7728.97it/s] 85%|████████▍ | 338183/400000 [00:45<00:07, 7769.61it/s] 85%|████████▍ | 338961/400000 [00:45<00:07, 7668.14it/s] 85%|████████▍ | 339729/400000 [00:45<00:07, 7624.12it/s] 85%|████████▌ | 340508/400000 [00:45<00:07, 7671.92it/s] 85%|████████▌ | 341278/400000 [00:45<00:07, 7677.63it/s] 86%|████████▌ | 342073/400000 [00:45<00:07, 7756.34it/s] 86%|████████▌ | 342850/400000 [00:45<00:07, 7691.69it/s] 86%|████████▌ | 343620/400000 [00:46<00:07, 7636.54it/s] 86%|████████▌ | 344408/400000 [00:46<00:07, 7705.31it/s] 86%|████████▋ | 345197/400000 [00:46<00:07, 7758.23it/s] 86%|████████▋ | 345974/400000 [00:46<00:07, 7711.77it/s] 87%|████████▋ | 346746/400000 [00:46<00:07, 7601.68it/s] 87%|████████▋ | 347507/400000 [00:46<00:06, 7581.92it/s] 87%|████████▋ | 348266/400000 [00:46<00:07, 7220.45it/s] 87%|████████▋ | 348994/400000 [00:46<00:07, 7237.32it/s] 87%|████████▋ | 349770/400000 [00:46<00:06, 7382.82it/s] 88%|████████▊ | 350511/400000 [00:47<00:06, 7363.07it/s] 88%|████████▊ | 351263/400000 [00:47<00:06, 7405.89it/s] 88%|████████▊ | 352005/400000 [00:47<00:06, 7200.90it/s] 88%|████████▊ | 352800/400000 [00:47<00:06, 7408.80it/s] 88%|████████▊ | 353584/400000 [00:47<00:06, 7530.68it/s] 89%|████████▊ | 354340/400000 [00:47<00:06, 7454.71it/s] 89%|████████▉ | 355088/400000 [00:47<00:06, 7455.82it/s] 89%|████████▉ | 355835/400000 [00:47<00:06, 7264.99it/s] 89%|████████▉ | 356564/400000 [00:47<00:06, 7214.09it/s] 89%|████████▉ | 357318/400000 [00:47<00:05, 7305.84it/s] 90%|████████▉ | 358050/400000 [00:48<00:05, 7217.45it/s] 90%|████████▉ | 358773/400000 [00:48<00:05, 7084.84it/s] 90%|████████▉ | 359555/400000 [00:48<00:05, 7289.49it/s] 90%|█████████ | 360297/400000 [00:48<00:05, 7326.41it/s] 90%|█████████ | 361048/400000 [00:48<00:05, 7380.32it/s] 90%|█████████ | 361791/400000 [00:48<00:05, 7392.70it/s] 91%|█████████ | 362532/400000 [00:48<00:05, 7377.75it/s] 91%|█████████ | 363271/400000 [00:48<00:04, 7379.26it/s] 91%|█████████ | 364054/400000 [00:48<00:04, 7507.65it/s] 91%|█████████ | 364806/400000 [00:48<00:04, 7508.73it/s] 91%|█████████▏| 365558/400000 [00:49<00:04, 7453.60it/s] 92%|█████████▏| 366327/400000 [00:49<00:04, 7522.29it/s] 92%|█████████▏| 367116/400000 [00:49<00:04, 7628.74it/s] 92%|█████████▏| 367905/400000 [00:49<00:04, 7704.44it/s] 92%|█████████▏| 368677/400000 [00:49<00:04, 7634.40it/s] 92%|█████████▏| 369442/400000 [00:49<00:04, 7444.06it/s] 93%|█████████▎| 370188/400000 [00:49<00:04, 7291.18it/s] 93%|█████████▎| 370940/400000 [00:49<00:03, 7356.68it/s] 93%|█████████▎| 371680/400000 [00:49<00:03, 7367.73it/s] 93%|█████████▎| 372418/400000 [00:49<00:03, 7325.86it/s] 93%|█████████▎| 373152/400000 [00:50<00:03, 7242.85it/s] 93%|█████████▎| 373915/400000 [00:50<00:03, 7353.37it/s] 94%|█████████▎| 374682/400000 [00:50<00:03, 7443.22it/s] 94%|█████████▍| 375481/400000 [00:50<00:03, 7598.47it/s] 94%|█████████▍| 376262/400000 [00:50<00:03, 7658.99it/s] 94%|█████████▍| 377030/400000 [00:50<00:03, 7393.16it/s] 94%|█████████▍| 377773/400000 [00:50<00:03, 7373.01it/s] 95%|█████████▍| 378549/400000 [00:50<00:02, 7484.47it/s] 95%|█████████▍| 379315/400000 [00:50<00:02, 7536.18it/s] 95%|█████████▌| 380070/400000 [00:50<00:02, 7456.42it/s] 95%|█████████▌| 380839/400000 [00:51<00:02, 7523.63it/s] 95%|█████████▌| 381593/400000 [00:51<00:02, 7522.86it/s] 96%|█████████▌| 382374/400000 [00:51<00:02, 7606.42it/s] 96%|█████████▌| 383168/400000 [00:51<00:02, 7702.33it/s] 96%|█████████▌| 383949/400000 [00:51<00:02, 7734.05it/s] 96%|█████████▌| 384723/400000 [00:51<00:02, 7603.61it/s] 96%|█████████▋| 385485/400000 [00:51<00:01, 7582.58it/s] 97%|█████████▋| 386244/400000 [00:51<00:01, 7569.60it/s] 97%|█████████▋| 387002/400000 [00:51<00:01, 7538.85it/s] 97%|█████████▋| 387757/400000 [00:51<00:01, 7473.72it/s] 97%|█████████▋| 388508/400000 [00:52<00:01, 7484.39it/s] 97%|█████████▋| 389257/400000 [00:52<00:01, 7383.33it/s] 98%|█████████▊| 390010/400000 [00:52<00:01, 7424.81it/s] 98%|█████████▊| 390759/400000 [00:52<00:01, 7443.98it/s] 98%|█████████▊| 391526/400000 [00:52<00:01, 7507.96it/s] 98%|█████████▊| 392278/400000 [00:52<00:01, 7444.72it/s] 98%|█████████▊| 393067/400000 [00:52<00:00, 7571.52it/s] 98%|█████████▊| 393825/400000 [00:52<00:00, 7573.12it/s] 99%|█████████▊| 394583/400000 [00:52<00:00, 7569.97it/s] 99%|█████████▉| 395380/400000 [00:53<00:00, 7683.09it/s] 99%|█████████▉| 396149/400000 [00:53<00:00, 7678.73it/s] 99%|█████████▉| 396918/400000 [00:53<00:00, 7662.83it/s] 99%|█████████▉| 397686/400000 [00:53<00:00, 7666.82it/s]100%|█████████▉| 398453/400000 [00:53<00:00, 7594.37it/s]100%|█████████▉| 399213/400000 [00:53<00:00, 7585.83it/s]100%|█████████▉| 399972/400000 [00:53<00:00, 7495.59it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7459.78it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f76e4137cc0>, <torchtext.data.dataset.TabularDataset object at 0x7f76e4137e10>, <torchtext.vocab.Vocab object at 0x7f76e4137d30>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 24.55 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 45.25 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  3.53 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  3.53 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  4.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  4.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:01<00:00,  3.93 file/s]2020-07-09 00:19:10.929407: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-09 00:19:10.933994: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-09 00:19:10.934813: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5600ce306f60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-09 00:19:10.934850: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 151230.91it/s] 46%|████▌     | 4530176/9912422 [00:00<00:24, 215733.55it/s]9920512it [00:00, 42816203.79it/s]                           
0it [00:00, ?it/s]32768it [00:00, 584078.61it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477746.10it/s]1654784it [00:00, 11927491.86it/s]                         
0it [00:00, ?it/s]8192it [00:00, 222006.59it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
