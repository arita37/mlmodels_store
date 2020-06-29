
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe606468ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe606468ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe6717281e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe6717281e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe68fa70e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe68fa70e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe61ea51488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe61ea51488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe61ea51488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 482032.22it/s] 73%|███████▎  | 7225344/9912422 [00:00<00:03, 686640.37it/s]9920512it [00:00, 44471138.58it/s]                           
0it [00:00, ?it/s]32768it [00:00, 649808.06it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1060138.41it/s]1654784it [00:00, 12966132.78it/s]                           
0it [00:00, ?it/s]8192it [00:00, 254720.36it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe605b7c358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe605b635f8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe61ea510d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe61ea510d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe61ea510d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:49,  3.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:49,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:49,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:49,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:48,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:48,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:48,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:34,  4.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:34,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:33,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:33,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:33,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:33,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:22,  6.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:22,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:21,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:15,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:14,  8.70 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:10, 11.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:07, 15.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:00<00:05, 20.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 20.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 26.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 32.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 32.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 38.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 44.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 51.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 57.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 57.54 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 62.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 62.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 69.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.80 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.15 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 67.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 70.70 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 70.70 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.26 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.35s/ url]
0 examples [00:00, ? examples/s]2020-06-29 18:09:33.955684: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-29 18:09:33.971103: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-29 18:09:33.971311: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d8fd6fd770 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-29 18:09:33.971331: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
20 examples [00:00, 198.59 examples/s]121 examples [00:00, 261.47 examples/s]212 examples [00:00, 332.50 examples/s]300 examples [00:00, 408.53 examples/s]387 examples [00:00, 483.12 examples/s]474 examples [00:00, 557.37 examples/s]567 examples [00:00, 632.62 examples/s]667 examples [00:00, 710.22 examples/s]756 examples [00:00, 745.83 examples/s]842 examples [00:01, 743.40 examples/s]933 examples [00:01, 786.38 examples/s]1030 examples [00:01, 833.56 examples/s]1133 examples [00:01, 882.33 examples/s]1230 examples [00:01, 906.51 examples/s]1332 examples [00:01, 937.05 examples/s]1435 examples [00:01, 962.94 examples/s]1538 examples [00:01, 981.40 examples/s]1638 examples [00:01, 950.45 examples/s]1735 examples [00:01, 928.36 examples/s]1829 examples [00:02, 921.16 examples/s]1930 examples [00:02, 941.83 examples/s]2032 examples [00:02, 963.55 examples/s]2131 examples [00:02, 967.47 examples/s]2229 examples [00:02, 938.29 examples/s]2324 examples [00:02, 899.22 examples/s]2421 examples [00:02, 918.36 examples/s]2517 examples [00:02, 928.08 examples/s]2611 examples [00:02, 928.68 examples/s]2705 examples [00:02, 926.88 examples/s]2803 examples [00:03, 942.13 examples/s]2898 examples [00:03, 941.84 examples/s]2998 examples [00:03, 956.14 examples/s]3098 examples [00:03, 966.33 examples/s]3195 examples [00:03, 834.15 examples/s]3285 examples [00:03, 851.15 examples/s]3388 examples [00:03, 897.59 examples/s]3481 examples [00:03, 905.94 examples/s]3574 examples [00:03, 889.13 examples/s]3670 examples [00:04, 908.10 examples/s]3764 examples [00:04, 915.45 examples/s]3868 examples [00:04, 947.42 examples/s]3964 examples [00:04, 930.63 examples/s]4058 examples [00:04, 918.49 examples/s]4151 examples [00:04, 869.12 examples/s]4244 examples [00:04, 884.48 examples/s]4339 examples [00:04, 901.52 examples/s]4438 examples [00:04, 926.18 examples/s]4532 examples [00:05, 888.13 examples/s]4622 examples [00:05, 757.49 examples/s]4714 examples [00:05, 798.06 examples/s]4815 examples [00:05, 849.62 examples/s]4908 examples [00:05, 870.14 examples/s]5000 examples [00:05, 883.16 examples/s]5098 examples [00:05, 908.90 examples/s]5201 examples [00:05, 940.90 examples/s]5297 examples [00:05, 941.54 examples/s]5393 examples [00:05, 916.43 examples/s]5486 examples [00:06, 907.43 examples/s]5589 examples [00:06, 939.36 examples/s]5685 examples [00:06, 945.05 examples/s]5780 examples [00:06, 929.50 examples/s]5874 examples [00:06, 925.72 examples/s]5978 examples [00:06, 955.73 examples/s]6078 examples [00:06, 966.82 examples/s]6180 examples [00:06, 981.92 examples/s]6279 examples [00:06, 922.24 examples/s]6381 examples [00:07, 949.38 examples/s]6477 examples [00:07, 929.00 examples/s]6578 examples [00:07, 950.85 examples/s]6674 examples [00:07, 870.37 examples/s]6763 examples [00:07, 862.02 examples/s]6860 examples [00:07, 889.53 examples/s]6957 examples [00:07, 912.21 examples/s]7056 examples [00:07, 934.21 examples/s]7151 examples [00:07, 936.54 examples/s]7248 examples [00:07, 944.53 examples/s]7343 examples [00:08, 894.70 examples/s]7434 examples [00:08, 892.62 examples/s]7532 examples [00:08, 916.84 examples/s]7630 examples [00:08, 934.03 examples/s]7724 examples [00:08, 923.50 examples/s]7817 examples [00:08, 899.95 examples/s]7916 examples [00:08, 923.58 examples/s]8009 examples [00:08, 904.43 examples/s]8105 examples [00:08, 918.97 examples/s]8198 examples [00:09, 863.01 examples/s]8301 examples [00:09, 905.10 examples/s]8398 examples [00:09, 910.74 examples/s]8494 examples [00:09, 924.21 examples/s]8594 examples [00:09, 944.45 examples/s]8689 examples [00:09, 801.51 examples/s]8777 examples [00:09, 822.02 examples/s]8869 examples [00:09, 848.18 examples/s]8957 examples [00:09, 838.27 examples/s]9048 examples [00:10, 857.70 examples/s]9136 examples [00:10, 778.16 examples/s]9236 examples [00:10, 831.80 examples/s]9335 examples [00:10, 872.89 examples/s]9425 examples [00:10, 861.89 examples/s]9518 examples [00:10, 878.87 examples/s]9621 examples [00:10, 918.85 examples/s]9715 examples [00:10, 869.71 examples/s]9805 examples [00:10, 877.86 examples/s]9906 examples [00:10, 913.06 examples/s]10001 examples [00:11, 890.50 examples/s]10091 examples [00:11, 845.23 examples/s]10194 examples [00:11, 891.60 examples/s]10285 examples [00:11, 866.11 examples/s]10373 examples [00:11, 840.06 examples/s]10475 examples [00:11, 885.49 examples/s]10575 examples [00:11, 916.02 examples/s]10678 examples [00:11, 945.71 examples/s]10774 examples [00:11, 947.53 examples/s]10877 examples [00:12, 970.36 examples/s]10975 examples [00:12, 956.02 examples/s]11072 examples [00:12, 864.26 examples/s]11176 examples [00:12, 909.31 examples/s]11269 examples [00:12, 875.86 examples/s]11359 examples [00:12, 874.75 examples/s]11455 examples [00:12, 897.81 examples/s]11546 examples [00:12, 777.58 examples/s]11633 examples [00:12, 802.62 examples/s]11723 examples [00:13, 828.56 examples/s]11824 examples [00:13, 874.98 examples/s]11915 examples [00:13, 865.65 examples/s]12004 examples [00:13, 869.33 examples/s]12093 examples [00:13, 858.32 examples/s]12180 examples [00:13, 857.51 examples/s]12277 examples [00:13, 885.90 examples/s]12367 examples [00:13, 883.43 examples/s]12458 examples [00:13, 890.55 examples/s]12550 examples [00:13, 898.56 examples/s]12646 examples [00:14, 905.39 examples/s]12737 examples [00:14, 882.70 examples/s]12826 examples [00:14, 881.08 examples/s]12927 examples [00:14, 915.44 examples/s]13020 examples [00:14, 906.61 examples/s]13115 examples [00:14, 915.71 examples/s]13208 examples [00:14, 919.45 examples/s]13301 examples [00:14, 902.72 examples/s]13398 examples [00:14, 921.24 examples/s]13499 examples [00:15, 944.57 examples/s]13602 examples [00:15, 966.18 examples/s]13699 examples [00:15, 936.50 examples/s]13798 examples [00:15, 951.51 examples/s]13894 examples [00:15, 839.80 examples/s]13988 examples [00:15, 866.16 examples/s]14086 examples [00:15, 895.73 examples/s]14178 examples [00:15, 901.43 examples/s]14279 examples [00:15, 929.72 examples/s]14374 examples [00:15, 935.47 examples/s]14469 examples [00:16, 925.56 examples/s]14563 examples [00:16, 927.92 examples/s]14657 examples [00:16, 870.23 examples/s]14760 examples [00:16, 912.36 examples/s]14862 examples [00:16, 940.84 examples/s]14958 examples [00:16, 927.13 examples/s]15052 examples [00:16, 913.74 examples/s]15151 examples [00:16, 934.03 examples/s]15248 examples [00:16, 933.94 examples/s]15346 examples [00:17, 947.06 examples/s]15449 examples [00:17, 968.12 examples/s]15548 examples [00:17, 974.38 examples/s]15646 examples [00:17, 949.14 examples/s]15748 examples [00:17, 968.39 examples/s]15846 examples [00:17, 963.07 examples/s]15943 examples [00:17, 790.08 examples/s]16044 examples [00:17, 844.57 examples/s]16135 examples [00:17, 862.75 examples/s]16232 examples [00:18, 880.82 examples/s]16326 examples [00:18, 897.64 examples/s]16423 examples [00:18, 917.17 examples/s]16517 examples [00:18, 912.44 examples/s]16610 examples [00:18, 892.49 examples/s]16701 examples [00:18, 884.75 examples/s]16794 examples [00:18, 895.49 examples/s]16895 examples [00:18, 927.00 examples/s]16997 examples [00:18, 952.78 examples/s]17100 examples [00:18, 973.35 examples/s]17199 examples [00:19, 976.91 examples/s]17298 examples [00:19, 969.96 examples/s]17396 examples [00:19, 940.25 examples/s]17491 examples [00:19, 909.80 examples/s]17593 examples [00:19, 938.48 examples/s]17693 examples [00:19, 954.82 examples/s]17789 examples [00:19, 949.35 examples/s]17885 examples [00:19, 876.59 examples/s]17974 examples [00:19, 867.57 examples/s]18073 examples [00:20, 899.93 examples/s]18164 examples [00:20, 901.34 examples/s]18263 examples [00:20, 925.65 examples/s]18357 examples [00:20, 851.70 examples/s]18444 examples [00:20, 838.60 examples/s]18530 examples [00:20, 832.79 examples/s]18626 examples [00:20, 865.99 examples/s]18727 examples [00:20, 902.88 examples/s]18819 examples [00:20, 897.39 examples/s]18914 examples [00:20, 911.15 examples/s]19017 examples [00:21, 942.28 examples/s]19112 examples [00:21, 924.78 examples/s]19206 examples [00:21, 907.78 examples/s]19306 examples [00:21, 932.96 examples/s]19409 examples [00:21, 958.55 examples/s]19513 examples [00:21, 980.56 examples/s]19612 examples [00:21, 961.88 examples/s]19709 examples [00:21, 940.63 examples/s]19804 examples [00:21, 935.42 examples/s]19898 examples [00:22, 776.47 examples/s]19999 examples [00:22, 832.82 examples/s]20087 examples [00:22, 845.91 examples/s]20181 examples [00:22, 870.36 examples/s]20284 examples [00:22, 912.54 examples/s]20378 examples [00:22, 897.51 examples/s]20470 examples [00:22, 894.28 examples/s]20561 examples [00:22, 852.92 examples/s]20655 examples [00:22, 875.87 examples/s]20747 examples [00:22, 887.05 examples/s]20837 examples [00:23, 788.01 examples/s]20919 examples [00:23, 758.46 examples/s]21019 examples [00:23, 817.22 examples/s]21106 examples [00:23, 831.40 examples/s]21204 examples [00:23, 870.72 examples/s]21297 examples [00:23, 886.95 examples/s]21398 examples [00:23, 918.58 examples/s]21492 examples [00:23, 910.64 examples/s]21589 examples [00:23, 926.23 examples/s]21683 examples [00:24, 924.81 examples/s]21776 examples [00:24, 826.85 examples/s]21871 examples [00:24, 859.13 examples/s]21973 examples [00:24, 900.64 examples/s]22075 examples [00:24, 931.14 examples/s]22170 examples [00:24, 884.52 examples/s]22261 examples [00:24, 880.42 examples/s]22355 examples [00:24, 895.55 examples/s]22448 examples [00:24, 900.21 examples/s]22548 examples [00:25, 927.50 examples/s]22642 examples [00:25, 917.67 examples/s]22735 examples [00:25, 758.03 examples/s]22823 examples [00:25, 790.30 examples/s]22924 examples [00:25, 845.39 examples/s]23021 examples [00:25, 877.10 examples/s]23112 examples [00:25, 798.31 examples/s]23215 examples [00:25, 854.93 examples/s]23315 examples [00:25, 891.76 examples/s]23410 examples [00:26, 906.04 examples/s]23508 examples [00:26, 926.67 examples/s]23603 examples [00:26, 914.84 examples/s]23701 examples [00:26, 931.81 examples/s]23796 examples [00:26, 908.78 examples/s]23888 examples [00:26, 880.27 examples/s]23981 examples [00:26, 882.62 examples/s]24077 examples [00:26, 903.75 examples/s]24168 examples [00:26, 898.57 examples/s]24259 examples [00:27, 863.52 examples/s]24346 examples [00:27, 861.42 examples/s]24433 examples [00:27, 859.93 examples/s]24520 examples [00:27, 829.00 examples/s]24606 examples [00:27, 837.33 examples/s]24691 examples [00:27, 838.98 examples/s]24790 examples [00:27, 876.76 examples/s]24888 examples [00:27, 903.45 examples/s]24988 examples [00:27, 928.77 examples/s]25088 examples [00:27, 948.33 examples/s]25184 examples [00:28, 934.96 examples/s]25280 examples [00:28, 939.81 examples/s]25375 examples [00:28, 886.74 examples/s]25469 examples [00:28, 900.35 examples/s]25560 examples [00:28, 851.88 examples/s]25658 examples [00:28, 884.38 examples/s]25756 examples [00:28, 910.23 examples/s]25848 examples [00:28, 900.51 examples/s]25942 examples [00:28, 911.37 examples/s]26045 examples [00:28, 942.97 examples/s]26140 examples [00:29, 915.45 examples/s]26242 examples [00:29, 942.14 examples/s]26337 examples [00:29, 911.65 examples/s]26434 examples [00:29, 926.09 examples/s]26528 examples [00:29, 783.60 examples/s]26613 examples [00:29, 800.80 examples/s]26703 examples [00:29, 827.00 examples/s]26803 examples [00:29, 871.91 examples/s]26896 examples [00:29, 886.96 examples/s]26987 examples [00:30, 876.30 examples/s]27076 examples [00:30, 851.57 examples/s]27173 examples [00:30, 882.46 examples/s]27274 examples [00:30, 917.06 examples/s]27376 examples [00:30, 945.24 examples/s]27472 examples [00:30, 940.62 examples/s]27567 examples [00:30, 886.23 examples/s]27657 examples [00:30, 849.18 examples/s]27760 examples [00:30, 895.52 examples/s]27859 examples [00:31, 919.61 examples/s]27960 examples [00:31, 942.67 examples/s]28058 examples [00:31, 953.14 examples/s]28155 examples [00:31, 911.43 examples/s]28248 examples [00:31, 910.91 examples/s]28346 examples [00:31, 928.56 examples/s]28446 examples [00:31, 947.46 examples/s]28550 examples [00:31, 973.04 examples/s]28649 examples [00:31, 977.23 examples/s]28748 examples [00:31, 964.28 examples/s]28845 examples [00:32, 908.60 examples/s]28941 examples [00:32, 922.39 examples/s]29043 examples [00:32, 947.49 examples/s]29139 examples [00:32, 950.15 examples/s]29243 examples [00:32, 973.98 examples/s]29341 examples [00:32, 968.83 examples/s]29443 examples [00:32, 983.09 examples/s]29542 examples [00:32, 943.37 examples/s]29637 examples [00:32, 921.79 examples/s]29730 examples [00:33, 838.98 examples/s]29816 examples [00:33, 841.73 examples/s]29902 examples [00:33, 760.65 examples/s]29988 examples [00:33, 787.41 examples/s]30069 examples [00:33, 760.83 examples/s]30161 examples [00:33, 802.40 examples/s]30256 examples [00:33, 841.50 examples/s]30342 examples [00:33, 768.54 examples/s]30439 examples [00:33, 817.82 examples/s]30531 examples [00:34, 844.24 examples/s]30618 examples [00:34, 836.43 examples/s]30704 examples [00:34, 826.85 examples/s]30788 examples [00:34, 809.45 examples/s]30890 examples [00:34, 861.78 examples/s]30986 examples [00:34, 887.36 examples/s]31082 examples [00:34, 906.21 examples/s]31179 examples [00:34, 924.43 examples/s]31273 examples [00:34, 792.25 examples/s]31367 examples [00:35, 830.06 examples/s]31454 examples [00:35, 800.31 examples/s]31543 examples [00:35, 824.03 examples/s]31640 examples [00:35, 862.04 examples/s]31737 examples [00:35, 890.21 examples/s]31828 examples [00:35, 894.40 examples/s]31929 examples [00:35, 925.91 examples/s]32030 examples [00:35, 948.20 examples/s]32133 examples [00:35, 971.25 examples/s]32231 examples [00:35, 971.98 examples/s]32332 examples [00:36, 981.07 examples/s]32431 examples [00:36, 921.08 examples/s]32525 examples [00:36, 892.34 examples/s]32616 examples [00:36, 879.30 examples/s]32705 examples [00:36, 875.78 examples/s]32794 examples [00:36, 796.37 examples/s]32883 examples [00:36, 822.04 examples/s]32980 examples [00:36, 859.41 examples/s]33078 examples [00:36, 889.60 examples/s]33176 examples [00:37, 913.35 examples/s]33269 examples [00:37, 904.33 examples/s]33361 examples [00:37, 869.09 examples/s]33453 examples [00:37, 883.12 examples/s]33542 examples [00:37, 866.71 examples/s]33632 examples [00:37, 875.93 examples/s]33733 examples [00:37, 911.19 examples/s]33825 examples [00:37, 869.53 examples/s]33922 examples [00:37, 897.12 examples/s]34013 examples [00:37, 876.63 examples/s]34109 examples [00:38, 899.05 examples/s]34208 examples [00:38, 911.70 examples/s]34300 examples [00:38, 910.25 examples/s]34392 examples [00:38, 910.16 examples/s]34488 examples [00:38, 922.63 examples/s]34590 examples [00:38, 949.52 examples/s]34686 examples [00:38, 950.60 examples/s]34782 examples [00:38, 944.95 examples/s]34886 examples [00:38, 969.71 examples/s]34984 examples [00:38, 969.91 examples/s]35082 examples [00:39, 964.09 examples/s]35179 examples [00:39, 800.45 examples/s]35268 examples [00:39, 823.41 examples/s]35367 examples [00:39, 865.91 examples/s]35457 examples [00:39, 865.91 examples/s]35561 examples [00:39, 910.84 examples/s]35655 examples [00:39, 781.75 examples/s]35747 examples [00:39, 817.23 examples/s]35845 examples [00:40, 859.66 examples/s]35944 examples [00:40, 894.91 examples/s]36037 examples [00:40, 903.50 examples/s]36130 examples [00:40, 900.67 examples/s]36222 examples [00:40, 842.27 examples/s]36323 examples [00:40, 885.12 examples/s]36414 examples [00:40, 873.53 examples/s]36517 examples [00:40, 914.61 examples/s]36610 examples [00:40, 902.52 examples/s]36702 examples [00:40, 901.91 examples/s]36793 examples [00:41, 884.35 examples/s]36885 examples [00:41, 893.06 examples/s]36980 examples [00:41, 909.00 examples/s]37072 examples [00:41, 893.66 examples/s]37163 examples [00:41, 897.23 examples/s]37253 examples [00:41, 885.74 examples/s]37342 examples [00:41, 867.91 examples/s]37430 examples [00:41, 858.49 examples/s]37517 examples [00:41, 850.18 examples/s]37603 examples [00:42, 801.00 examples/s]37684 examples [00:42, 762.51 examples/s]37777 examples [00:42, 804.56 examples/s]37878 examples [00:42, 856.17 examples/s]37971 examples [00:42, 876.48 examples/s]38061 examples [00:42, 881.98 examples/s]38164 examples [00:42, 920.59 examples/s]38258 examples [00:42, 901.47 examples/s]38355 examples [00:42, 920.10 examples/s]38455 examples [00:42, 942.32 examples/s]38554 examples [00:43, 954.26 examples/s]38650 examples [00:43, 924.85 examples/s]38744 examples [00:43, 888.21 examples/s]38841 examples [00:43, 903.77 examples/s]38932 examples [00:43, 894.96 examples/s]39027 examples [00:43, 908.71 examples/s]39130 examples [00:43, 939.78 examples/s]39225 examples [00:43, 931.92 examples/s]39325 examples [00:43, 949.80 examples/s]39421 examples [00:44, 952.58 examples/s]39517 examples [00:44, 949.50 examples/s]39613 examples [00:44, 931.12 examples/s]39716 examples [00:44, 958.04 examples/s]39813 examples [00:44, 954.60 examples/s]39910 examples [00:44, 957.78 examples/s]40006 examples [00:44, 913.26 examples/s]40098 examples [00:44, 881.08 examples/s]40187 examples [00:44, 844.66 examples/s]40273 examples [00:44, 841.15 examples/s]40365 examples [00:45, 862.09 examples/s]40466 examples [00:45, 900.23 examples/s]40557 examples [00:45, 882.47 examples/s]40646 examples [00:45, 848.33 examples/s]40735 examples [00:45, 858.13 examples/s]40829 examples [00:45, 879.01 examples/s]40927 examples [00:45, 906.59 examples/s]41023 examples [00:45, 920.20 examples/s]41116 examples [00:45, 908.58 examples/s]41216 examples [00:46, 933.82 examples/s]41310 examples [00:46, 928.57 examples/s]41404 examples [00:46, 931.08 examples/s]41499 examples [00:46, 935.62 examples/s]41594 examples [00:46, 937.22 examples/s]41688 examples [00:46, 874.76 examples/s]41777 examples [00:46, 876.37 examples/s]41870 examples [00:46, 891.17 examples/s]41972 examples [00:46, 924.43 examples/s]42066 examples [00:46, 915.59 examples/s]42167 examples [00:47, 940.57 examples/s]42262 examples [00:47, 939.05 examples/s]42357 examples [00:47, 941.40 examples/s]42452 examples [00:47, 825.91 examples/s]42550 examples [00:47, 864.62 examples/s]42647 examples [00:47, 892.11 examples/s]42741 examples [00:47, 905.84 examples/s]42844 examples [00:47, 939.16 examples/s]42946 examples [00:47, 961.37 examples/s]43044 examples [00:48, 932.95 examples/s]43139 examples [00:48, 925.62 examples/s]43233 examples [00:48, 901.65 examples/s]43335 examples [00:48, 931.73 examples/s]43429 examples [00:48, 879.83 examples/s]43528 examples [00:48, 908.44 examples/s]43629 examples [00:48, 922.65 examples/s]43723 examples [00:48, 914.22 examples/s]43824 examples [00:48, 938.68 examples/s]43919 examples [00:49, 817.27 examples/s]44009 examples [00:49, 840.20 examples/s]44096 examples [00:49, 840.53 examples/s]44192 examples [00:49, 872.44 examples/s]44289 examples [00:49, 898.63 examples/s]44381 examples [00:49, 780.59 examples/s]44478 examples [00:49, 827.53 examples/s]44571 examples [00:49, 853.10 examples/s]44659 examples [00:49, 854.93 examples/s]44763 examples [00:49, 901.31 examples/s]44865 examples [00:50, 933.32 examples/s]44962 examples [00:50, 942.84 examples/s]45058 examples [00:50, 862.91 examples/s]45154 examples [00:50, 889.83 examples/s]45256 examples [00:50, 924.49 examples/s]45360 examples [00:50, 954.25 examples/s]45461 examples [00:50, 967.78 examples/s]45565 examples [00:50, 986.65 examples/s]45665 examples [00:50, 932.49 examples/s]45760 examples [00:51, 936.40 examples/s]45855 examples [00:51, 921.79 examples/s]45948 examples [00:51, 918.68 examples/s]46050 examples [00:51, 945.23 examples/s]46146 examples [00:51, 914.66 examples/s]46239 examples [00:51, 891.14 examples/s]46329 examples [00:51, 802.91 examples/s]46425 examples [00:51, 842.51 examples/s]46527 examples [00:51, 886.93 examples/s]46618 examples [00:52, 883.04 examples/s]46715 examples [00:52, 906.69 examples/s]46810 examples [00:52, 917.14 examples/s]46911 examples [00:52, 942.35 examples/s]47009 examples [00:52, 951.91 examples/s]47106 examples [00:52, 956.29 examples/s]47203 examples [00:52, 942.35 examples/s]47298 examples [00:52, 932.60 examples/s]47395 examples [00:52, 942.74 examples/s]47490 examples [00:52, 873.18 examples/s]47593 examples [00:53, 914.88 examples/s]47695 examples [00:53, 942.68 examples/s]47791 examples [00:53, 941.10 examples/s]47886 examples [00:53, 857.91 examples/s]47988 examples [00:53, 898.92 examples/s]48082 examples [00:53, 910.51 examples/s]48175 examples [00:53, 872.29 examples/s]48270 examples [00:53, 892.25 examples/s]48361 examples [00:53, 782.13 examples/s]48443 examples [00:54, 786.89 examples/s]48525 examples [00:54, 795.26 examples/s]48611 examples [00:54, 813.35 examples/s]48705 examples [00:54, 847.15 examples/s]48798 examples [00:54, 869.43 examples/s]48886 examples [00:54, 854.55 examples/s]48973 examples [00:54, 857.89 examples/s]49060 examples [00:54, 859.56 examples/s]49159 examples [00:54, 894.86 examples/s]49260 examples [00:54, 925.87 examples/s]49354 examples [00:55, 913.82 examples/s]49452 examples [00:55, 932.68 examples/s]49546 examples [00:55, 811.04 examples/s]49645 examples [00:55, 856.32 examples/s]49734 examples [00:55, 748.58 examples/s]49826 examples [00:55, 791.22 examples/s]49925 examples [00:55, 840.51 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6324/50000 [00:00<00:00, 63239.31 examples/s] 36%|███▌      | 17793/50000 [00:00<00:00, 73072.49 examples/s] 59%|█████▊    | 29333/50000 [00:00<00:00, 82107.05 examples/s] 82%|████████▏ | 41009/50000 [00:00<00:00, 90130.66 examples/s]                                                               0 examples [00:00, ? examples/s]86 examples [00:00, 854.15 examples/s]187 examples [00:00, 893.49 examples/s]280 examples [00:00, 903.95 examples/s]374 examples [00:00, 913.60 examples/s]463 examples [00:00, 906.35 examples/s]545 examples [00:00, 877.15 examples/s]647 examples [00:00, 915.28 examples/s]733 examples [00:00, 894.42 examples/s]826 examples [00:00, 904.72 examples/s]923 examples [00:01, 920.95 examples/s]1014 examples [00:01, 883.80 examples/s]1105 examples [00:01, 890.31 examples/s]1194 examples [00:01, 805.14 examples/s]1281 examples [00:01, 822.22 examples/s]1381 examples [00:01, 868.07 examples/s]1475 examples [00:01, 888.34 examples/s]1572 examples [00:01, 910.07 examples/s]1664 examples [00:01, 888.69 examples/s]1761 examples [00:01, 910.31 examples/s]1858 examples [00:02, 926.26 examples/s]1952 examples [00:02, 893.07 examples/s]2052 examples [00:02, 920.62 examples/s]2145 examples [00:02, 803.54 examples/s]2243 examples [00:02, 847.36 examples/s]2334 examples [00:02, 863.83 examples/s]2423 examples [00:02, 863.80 examples/s]2511 examples [00:02, 743.23 examples/s]2608 examples [00:02, 797.71 examples/s]2692 examples [00:03, 809.00 examples/s]2776 examples [00:03, 814.10 examples/s]2860 examples [00:03, 819.82 examples/s]2963 examples [00:03, 871.30 examples/s]3052 examples [00:03, 859.40 examples/s]3140 examples [00:03, 859.38 examples/s]3239 examples [00:03, 894.03 examples/s]3338 examples [00:03, 919.29 examples/s]3433 examples [00:03, 925.89 examples/s]3531 examples [00:04, 940.54 examples/s]3626 examples [00:04, 885.21 examples/s]3724 examples [00:04, 910.99 examples/s]3816 examples [00:04, 883.36 examples/s]3910 examples [00:04, 897.39 examples/s]4002 examples [00:04, 903.72 examples/s]4094 examples [00:04, 908.26 examples/s]4187 examples [00:04, 912.82 examples/s]4279 examples [00:04, 898.31 examples/s]4378 examples [00:04, 914.03 examples/s]4470 examples [00:05, 828.22 examples/s]4555 examples [00:05, 820.52 examples/s]4657 examples [00:05, 870.84 examples/s]4751 examples [00:05, 889.99 examples/s]4855 examples [00:05, 929.76 examples/s]4951 examples [00:05, 936.05 examples/s]5046 examples [00:05, 926.55 examples/s]5145 examples [00:05, 942.81 examples/s]5243 examples [00:05, 953.13 examples/s]5339 examples [00:06, 952.46 examples/s]5435 examples [00:06, 825.22 examples/s]5534 examples [00:06, 867.95 examples/s]5627 examples [00:06, 872.07 examples/s]5733 examples [00:06, 919.14 examples/s]5829 examples [00:06, 930.58 examples/s]5924 examples [00:06, 765.06 examples/s]6016 examples [00:06, 803.86 examples/s]6103 examples [00:06, 821.94 examples/s]6195 examples [00:07, 848.21 examples/s]6296 examples [00:07, 889.08 examples/s]6388 examples [00:07, 806.41 examples/s]6478 examples [00:07, 822.85 examples/s]6563 examples [00:07, 822.53 examples/s]6649 examples [00:07, 830.89 examples/s]6746 examples [00:07, 868.02 examples/s]6835 examples [00:07, 834.88 examples/s]6920 examples [00:07, 832.40 examples/s]7011 examples [00:08, 853.95 examples/s]7103 examples [00:08, 870.32 examples/s]7202 examples [00:08, 901.56 examples/s]7299 examples [00:08, 918.87 examples/s]7398 examples [00:08, 936.88 examples/s]7496 examples [00:08, 949.21 examples/s]7596 examples [00:08, 962.83 examples/s]7696 examples [00:08, 972.87 examples/s]7800 examples [00:08, 990.62 examples/s]7900 examples [00:08, 952.44 examples/s]7998 examples [00:09, 960.19 examples/s]8095 examples [00:09, 922.42 examples/s]8196 examples [00:09, 944.98 examples/s]8298 examples [00:09, 964.54 examples/s]8403 examples [00:09, 988.48 examples/s]8503 examples [00:09, 971.55 examples/s]8601 examples [00:09, 965.14 examples/s]8698 examples [00:09, 957.06 examples/s]8800 examples [00:09, 973.72 examples/s]8898 examples [00:09, 939.76 examples/s]8993 examples [00:10, 931.12 examples/s]9095 examples [00:10, 955.46 examples/s]9191 examples [00:10, 944.99 examples/s]9288 examples [00:10, 952.23 examples/s]9394 examples [00:10, 980.45 examples/s]9493 examples [00:10, 937.23 examples/s]9588 examples [00:10, 869.93 examples/s]9684 examples [00:10, 892.47 examples/s]9784 examples [00:10, 920.07 examples/s]9887 examples [00:11, 948.15 examples/s]9983 examples [00:11, 951.67 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4091GF/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete4091GF/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe61ea51488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe61ea51488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe61ea51488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe605bfc588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe5a8c93240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe61ea51488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe61ea51488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe61ea51488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe5a8c93a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe605b630f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe596845158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe596845158> 

  function with postional parmater data_info <function split_train_valid at 0x7fe596845158> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe596845268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe596845268> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe596845268> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=785cf58252f6949f59deaf0bd9448a95b1f0bb90a46833f485b4b594c3afebb2
  Stored in directory: /tmp/pip-ephem-wheel-cache-gd6kqy16/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:23:16, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:12:22, 15.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:41:49, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:29:46, 31.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 2.16M/862M [00:01<5:15:47, 45.4kB/s].vector_cache/glove.6B.zip:   1%|          | 5.23M/862M [00:01<3:40:27, 64.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.84M/862M [00:01<2:33:35, 92.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:01<1:47:00, 132kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:02<1:17:36, 182kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.2M/862M [00:02<54:09, 259kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 22.9M/862M [00:02<37:51, 369kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:02<26:36, 524kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.6M/862M [00:02<18:38, 744kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.1M/862M [00:02<13:10, 1.05MB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.4M/862M [00:02<11:11, 1.24MB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:02<07:54, 1.74MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:03<05:40, 2.42MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.4M/862M [00:03<04:29, 3.04MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.6M/862M [00:03<03:31, 3.85MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.1M/862M [00:04<02:34, 5.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:04<02:41, 5.02MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<01:58, 6.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:06<16:37, 807kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:06<13:45, 976kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:06<10:08, 1.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:08<09:28, 1.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:08<08:20, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:08<06:09, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<04:33, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:10<10:24, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:10<09:02, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:10<07:41, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:10<05:36, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:12<07:25, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:12<06:26, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:12<05:04, 2.60MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<03:43, 3.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:14<10:48, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:14<10:28, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:14<07:57, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:14<05:46, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:16<10:04, 1.30MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:17<12:45, 1.03MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:17<10:17, 1.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:17<07:28, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:18<08:10, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:19<07:02, 1.85MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:19<05:18, 2.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:20<06:41, 1.93MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:21<05:48, 2.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:21<04:24, 2.93MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:21<03:14, 3.97MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:22<26:36, 484kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<20:26, 630kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:23<15:02, 855kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<10:37, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:24<41:57, 305kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:24<30:43, 417kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:25<21:44, 588kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<15:23, 828kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:26<2:14:25, 94.8kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:26<1:35:24, 133kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:27<1:06:58, 190kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<49:44, 255kB/s]   .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<36:05, 351kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<25:56, 488kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<20:09, 625kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<15:22, 819kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<11:01, 1.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<07:54, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<2:39:34, 78.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<1:54:13, 110kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<1:20:33, 155kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<56:21, 221kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<49:03, 254kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<35:37, 350kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<25:12, 493kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<20:28, 605kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<15:35, 795kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<11:13, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<11:11, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<10:35, 1.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<08:06, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:39<05:50, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<09:00, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<07:40, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<05:42, 2.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<06:50, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<07:38, 1.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<05:57, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<04:27, 2.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<06:01, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<05:32, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<04:14, 2.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<05:34, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<06:29, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<05:11, 2.32MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:47<03:50, 3.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<06:39, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<06:09, 1.95MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:49<04:37, 2.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<03:28, 3.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<07:25, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<06:51, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<05:51, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:51<04:20, 2.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<05:55, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<05:40, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<04:18, 2.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<03:10, 3.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<13:55, 846kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<12:24, 949kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<09:17, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<06:55, 1.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<05:01, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<12:32, 934kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<10:39, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<07:55, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<05:39, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<17:36, 661kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<13:40, 850kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:59<09:52, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<07:05, 1.63MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:01<13:35, 851kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<10:32, 1.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<07:56, 1.45MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<05:43, 2.01MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<08:37, 1.33MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<07:36, 1.51MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<05:41, 2.02MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<06:14, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:05<05:41, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<04:22, 2.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<05:16, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<03:58, 2.85MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<02:55, 3.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<07:34, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<07:13, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<06:04, 1.86MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:09<04:35, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<03:22, 3.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<14:00, 801kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<11:17, 993kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<08:37, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<06:15, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<07:59, 1.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<11:40, 955kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<09:40, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<07:05, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<05:16, 2.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<07:37, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<09:20, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<07:19, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<05:23, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<06:53, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<10:13, 1.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<08:32, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<06:24, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<04:39, 2.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<07:28, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<06:31, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<04:51, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<03:31, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<29:13, 372kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<21:37, 503kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:22<15:23, 704kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<13:06, 824kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<10:23, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 216M/862M [01:24<07:34, 1.42MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<05:32, 1.94MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<10:27, 1.03MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<08:49, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<06:32, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<04:41, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<34:47, 307kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<25:47, 413kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:28<18:15, 583kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<13:00, 817kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<13:58, 758kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:29<11:24, 929kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<08:46, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:30<06:25, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<04:43, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<07:46, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<06:41, 1.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<05:27, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<04:01, 2.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<05:26, 1.92MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<05:04, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<03:55, 2.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:34<02:56, 3.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<05:36, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<04:04, 2.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:36<03:16, 3.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<02:24, 4.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<1:03:24, 163kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<45:35, 226kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:38<32:09, 320kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<24:33, 418kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<18:24, 557kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:40<13:04, 782kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<09:16, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<3:39:55, 46.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:42<2:36:14, 65.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:42<1:49:45, 92.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:42<1:16:41, 132kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:44<1:00:06, 168kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:44<46:11, 219kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<33:13, 304kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<23:38, 427kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<16:35, 606kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<25:57, 387kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<18:40, 537kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<13:13, 756kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:48<12:57, 770kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<15:21, 650kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<11:57, 833kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<08:37, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<06:11, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<10:53, 910kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<08:38, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:50<06:28, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<04:43, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:52<06:21, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:52<05:34, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<04:09, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<05:00, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<04:31, 2.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:54<03:46, 2.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<02:48, 3.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:56<05:07, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:56<04:48, 2.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<03:40, 2.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<04:28, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:58<04:19, 2.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<03:17, 2.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<02:26, 3.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<09:34, 998kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:00<08:53, 1.07MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:00<06:42, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<04:46, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<14:13, 667kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:02<10:56, 867kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:02<07:51, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:04<07:39, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:04<06:20, 1.49MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<04:39, 2.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<05:28, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<04:47, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<03:34, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<04:41, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<05:10, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:08<04:05, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<05:16, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:10<04:10, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<03:02, 3.01MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<06:05, 1.50MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<05:11, 1.76MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:12<03:51, 2.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<04:51, 1.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<04:17, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:14<03:27, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<02:31, 3.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<08:59, 1.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:16<07:12, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:16<05:21, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<03:50, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<16:21, 547kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<12:21, 724kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:18<08:50, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<06:17, 1.41MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<19:39, 452kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<15:34, 570kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:20<11:20, 781kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<08:01, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<10:12, 862kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<08:12, 1.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:22<05:57, 1.47MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<06:10, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<05:13, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:24<03:52, 2.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<04:10, 2.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:26<03:08, 2.75MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<04:13, 2.03MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<03:53, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:28<02:53, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:28<02:08, 3.99MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<1:04:38, 132kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:30<46:05, 185kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:30<32:23, 263kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:31<24:35, 344kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:32<17:57, 471kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:32<13:00, 649kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<09:15, 911kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:33<08:53, 945kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<07:12, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:34<05:34, 1.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<03:58, 2.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:35<08:17, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<06:26, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:36<04:59, 1.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:36<03:38, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<05:30, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<05:37, 1.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<04:19, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:38<03:11, 2.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:40<04:42, 1.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:40<04:46, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:40<04:09, 1.97MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:40<03:04, 2.65MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:42<03:58, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:42<04:19, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:42<03:21, 2.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:44<03:53, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:44<04:30, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:44<03:33, 2.26MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<02:34, 3.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:46<06:39, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:46<05:29, 1.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:46<04:00, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:48<04:40, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:48<04:08, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:48<03:06, 2.53MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:50<03:52, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:50<03:52, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:50<03:38, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:50<02:41, 2.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:52<03:50, 2.03MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:52<03:47, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:52<03:06, 2.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<02:18, 3.34MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:54<03:49, 2.02MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:54<03:39, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:54<03:00, 2.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<02:13, 3.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:56<04:11, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:56<03:43, 2.05MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:56<03:08, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<02:25, 3.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:58<03:46, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:58<04:22, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:58<03:21, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:58<02:32, 2.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:00<03:33, 2.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:00<03:16, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:00<02:26, 3.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:02<03:27, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:02<03:02, 2.44MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:02<02:34, 2.88MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<01:56, 3.81MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:04<03:41, 1.99MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [03:04<03:20, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:04<02:40, 2.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:06<03:09, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:06<03:12, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:06<02:29, 2.92MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:06<01:52, 3.88MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:08<04:46, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:08<04:06, 1.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:08<03:03, 2.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:10<03:48, 1.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:10<04:06, 1.74MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:10<03:14, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:12<03:28, 2.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:12<03:59, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:12<03:10, 2.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<02:16, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:14<27:17, 257kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:14<19:49, 354kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:14<14:00, 499kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:16<11:24, 610kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:16<09:32, 729kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:16<07:00, 989kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:18<05:59, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:18<04:54, 1.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:18<03:36, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:20<04:10, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:20<04:18, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:20<03:21, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<02:26, 2.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:22<04:43, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:22<04:00, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:22<02:57, 2.27MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:24<03:51, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:24<04:13, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:24<03:20, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:24<02:25, 2.74MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:26<04:57, 1.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:26<04:08, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:26<03:04, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:28<03:43, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:28<03:58, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:28<03:04, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:28<02:22, 2.74MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:30<03:06, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:30<02:52, 2.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:30<02:10, 2.97MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:32<03:00, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:32<02:51, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:32<02:09, 2.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:34<02:58, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:34<02:37, 2.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:34<02:05, 3.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:34<01:31, 4.09MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:36<25:58, 241kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:36<19:30, 321kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:36<13:56, 448kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:36<09:44, 636kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:38<27:05, 229kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:38<20:13, 306kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:38<14:31, 426kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:38<10:13, 602kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:40<08:53, 690kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:40<06:55, 884kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:40<05:00, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:42<04:49, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:42<04:43, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:42<03:38, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:42<02:36, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:44<06:03, 989kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:44<06:40, 898kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:45<05:17, 1.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:45<03:49, 1.56MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:45<02:45, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:46<11:16, 525kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:46<08:00, 736kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:48<06:49, 858kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:48<05:52, 995kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:48<04:22, 1.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:48<03:06, 1.86MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:50<08:13, 703kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:50<06:28, 893kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:50<04:40, 1.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:52<04:28, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:52<03:42, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:52<02:45, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:54<03:14, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:54<02:50, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:54<02:07, 2.65MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<02:48, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:56<03:08, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:56<02:28, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:58<02:36, 2.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:58<02:24, 2.29MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:58<01:48, 3.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<02:32, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:00<02:20, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:00<01:44, 3.10MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:00<01:17, 4.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<22:20, 240kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:02<16:16, 330kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:02<11:27, 466kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<09:14, 574kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:04<07:32, 703kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:04<05:32, 956kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<04:40, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:06<03:48, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:06<02:46, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:07<03:08, 1.64MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:08<03:15, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:08<02:30, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:08<01:54, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:09<02:29, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:10<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:10<01:41, 3.00MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:11<02:21, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:12<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:12<02:08, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:12<01:32, 3.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:13<11:04, 448kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:13<08:14, 602kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:14<05:51, 841kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:15<05:13, 936kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:15<04:08, 1.18MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:16<03:00, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:17<03:12, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:17<02:43, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:18<02:02, 2.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:19<02:32, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:19<02:14, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:19<01:41, 2.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:21<02:17, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:21<02:32, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:21<01:58, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:22<01:25, 3.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:23<05:42, 809kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:23<04:28, 1.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:23<03:15, 1.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:25<03:11, 1.43MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:25<03:08, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:25<02:25, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:27<02:24, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:27<02:09, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:27<01:36, 2.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:29<02:09, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:29<02:24, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:29<01:53, 2.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:31<02:01, 2.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:31<01:51, 2.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:31<01:22, 3.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:33<01:58, 2.17MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:33<02:14, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:33<01:44, 2.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:33<01:15, 3.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:35<04:07, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:35<03:18, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:35<02:24, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:37<02:38, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:37<02:40, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:37<02:02, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:37<01:28, 2.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:39<03:40, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:39<02:59, 1.36MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:39<02:20, 1.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:39<01:40, 2.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:41<24:24, 164kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:41<17:27, 229kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:41<12:14, 324kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:43<09:25, 417kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:43<07:23, 532kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:43<05:21, 732kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:45<04:19, 894kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:45<03:25, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:45<02:28, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:47<02:36, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:47<02:35, 1.46MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:47<01:58, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:47<01:24, 2.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:49<03:26, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:49<02:47, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:49<02:02, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:51<02:16, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:51<02:19, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:51<01:47, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:51<01:21, 2.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:53<01:45, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:53<01:36, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:53<01:12, 2.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:55<01:37, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:55<01:50, 1.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:55<01:26, 2.43MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:55<01:02, 3.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:57<02:37, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:57<02:07, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:57<01:33, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:57<01:07, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<20:59, 161kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:59<15:26, 219kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:59<10:56, 308kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:59<07:34, 438kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:00<3:15:34, 17.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:01<2:16:58, 24.2kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:01<1:35:27, 34.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:02<1:06:33, 48.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:03<47:12, 68.7kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:03<33:04, 97.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:03<22:49, 139kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:04<27:34, 115kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:05<20:02, 159kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:05<14:08, 224kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:05<09:49, 318kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:06<08:09, 382kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:07<06:01, 516kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:07<04:15, 721kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:08<03:40, 829kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:08<02:51, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:09<02:03, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:10<02:08, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:10<02:05, 1.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:11<01:35, 1.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:11<01:08, 2.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:12<01:54, 1.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:12<01:37, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:13<01:11, 2.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:14<01:29, 1.89MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:14<01:40, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:14<01:17, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:15<00:55, 3.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:16<03:43, 744kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:16<02:52, 958kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:16<02:04, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:18<02:03, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:18<01:54, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:18<01:27, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:18<01:03, 2.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:20<01:38, 1.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:20<01:24, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:20<01:02, 2.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:22<01:19, 1.93MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:22<01:26, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:22<01:07, 2.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:24<01:10, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:24<01:04, 2.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:24<00:48, 3.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:26<01:07, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:26<01:01, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:26<00:46, 3.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:28<01:05, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:28<00:57, 2.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:28<00:43, 3.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:28<00:32, 4.28MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:30<10:50, 211kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:30<07:49, 291kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:30<05:28, 412kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:32<04:17, 517kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:32<03:23, 655kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:32<02:27, 899kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:32<01:46, 1.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:34<01:43, 1.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:34<01:25, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:34<01:02, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:36<01:12, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:36<01:22, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:36<01:03, 1.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:38<01:02, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:38<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:38<00:41, 2.86MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:40<00:56, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:40<00:51, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:40<00:37, 3.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:42<00:52, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:42<00:48, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:42<00:35, 3.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:42<00:26, 4.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:44<10:56, 165kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:44<07:48, 230kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:44<05:26, 327kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<04:08, 420kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<03:42, 469kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:46<02:45, 627kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:46<01:56, 876kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<01:43, 973kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:48<01:20, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:48<00:58, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:48<00:41, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:50<04:24, 363kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:50<03:14, 492kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:50<02:16, 692kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:52<01:53, 807kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:52<01:38, 928kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:52<01:13, 1.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<00:50, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:53<03:06, 471kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:54<02:18, 630kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:54<01:37, 880kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:55<01:26, 970kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:56<01:08, 1.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:56<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:57<00:52, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:58<00:52, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:58<00:40, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:58<00:28, 2.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:59<00:50, 1.50MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:00<00:43, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:00<00:31, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<00:22, 3.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:01<1:10:37, 16.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:01<49:38, 23.9kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:02<34:25, 34.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<23:02, 48.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:03<24:19, 46.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:03<17:02, 65.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:04<11:40, 93.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:05<08:09, 129kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:05<05:46, 181kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:06<03:58, 257kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:07<02:54, 338kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:07<02:07, 460kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<01:28, 645kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:09<01:12, 757kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:09<01:01, 886kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:09<00:44, 1.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:10<00:31, 1.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:11<00:37, 1.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:11<00:31, 1.61MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:11<00:23, 2.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:11<00:16, 2.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:13<01:04, 721kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:13<00:55, 842kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:13<00:40, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:13<00:27, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:15<00:36, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:15<00:29, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:21, 1.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:17<00:23, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:17<00:23, 1.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:17<00:18, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:19<00:17, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:19<00:15, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:19<00:11, 2.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:19<00:07, 3.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:21<03:00, 168kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:21<02:07, 234kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:21<01:25, 331kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:23<01:01, 425kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:23<00:47, 541kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:23<00:33, 746kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:23<00:22, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:25<00:24, 906kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:25<00:18, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:25<00:12, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:27<00:12, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:27<00:15, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:27<00:11, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:27<00:07, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:29<00:08, 1.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:29<00:07, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:29<00:05, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:31<00:04, 2.00MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:31<00:05, 1.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:31<00:03, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:31<00:01, 3.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:33<05:16, 17.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:33<03:26, 24.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:33<01:40, 35.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:35<00:27, 49.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:35<00:13, 70.6kB/s].vector_cache/glove.6B.zip: 862MB [06:35, 2.18MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 335/400000 [00:00<01:59, 3346.26it/s]  0%|          | 1095/400000 [00:00<01:39, 4020.89it/s]  0%|          | 1808/400000 [00:00<01:26, 4625.32it/s]  1%|          | 2557/400000 [00:00<01:16, 5224.53it/s]  1%|          | 3326/400000 [00:00<01:08, 5780.51it/s]  1%|          | 4100/400000 [00:00<01:03, 6255.06it/s]  1%|          | 4865/400000 [00:00<00:59, 6615.17it/s]  1%|▏         | 5624/400000 [00:00<00:57, 6879.14it/s]  2%|▏         | 6355/400000 [00:00<00:56, 6992.71it/s]  2%|▏         | 7068/400000 [00:01<00:56, 6912.52it/s]  2%|▏         | 7827/400000 [00:01<00:55, 7102.21it/s]  2%|▏         | 8579/400000 [00:01<00:54, 7221.62it/s]  2%|▏         | 9321/400000 [00:01<00:53, 7278.02it/s]  3%|▎         | 10087/400000 [00:01<00:52, 7387.52it/s]  3%|▎         | 10830/400000 [00:01<00:52, 7367.58it/s]  3%|▎         | 11593/400000 [00:01<00:52, 7441.64it/s]  3%|▎         | 12345/400000 [00:01<00:51, 7463.25it/s]  3%|▎         | 13093/400000 [00:01<00:52, 7387.53it/s]  3%|▎         | 13846/400000 [00:01<00:51, 7429.62it/s]  4%|▎         | 14590/400000 [00:02<00:51, 7416.34it/s]  4%|▍         | 15341/400000 [00:02<00:51, 7442.59it/s]  4%|▍         | 16110/400000 [00:02<00:51, 7514.84it/s]  4%|▍         | 16862/400000 [00:02<00:54, 6968.81it/s]  4%|▍         | 17621/400000 [00:02<00:53, 7142.58it/s]  5%|▍         | 18343/400000 [00:02<00:53, 7162.96it/s]  5%|▍         | 19098/400000 [00:02<00:52, 7271.90it/s]  5%|▍         | 19876/400000 [00:02<00:51, 7415.86it/s]  5%|▌         | 20630/400000 [00:02<00:50, 7450.90it/s]  5%|▌         | 21378/400000 [00:02<00:50, 7458.79it/s]  6%|▌         | 22126/400000 [00:03<00:50, 7419.08it/s]  6%|▌         | 22875/400000 [00:03<00:50, 7438.95it/s]  6%|▌         | 23633/400000 [00:03<00:50, 7479.10it/s]  6%|▌         | 24382/400000 [00:03<00:51, 7336.55it/s]  6%|▋         | 25117/400000 [00:03<00:52, 7180.41it/s]  6%|▋         | 25888/400000 [00:03<00:51, 7330.03it/s]  7%|▋         | 26623/400000 [00:03<00:50, 7322.65it/s]  7%|▋         | 27382/400000 [00:03<00:50, 7399.69it/s]  7%|▋         | 28124/400000 [00:03<00:50, 7375.53it/s]  7%|▋         | 28863/400000 [00:03<00:51, 7205.52it/s]  7%|▋         | 29621/400000 [00:04<00:50, 7311.86it/s]  8%|▊         | 30376/400000 [00:04<00:50, 7381.66it/s]  8%|▊         | 31116/400000 [00:04<00:50, 7304.32it/s]  8%|▊         | 31848/400000 [00:04<00:50, 7222.35it/s]  8%|▊         | 32572/400000 [00:04<00:53, 6928.99it/s]  8%|▊         | 33269/400000 [00:04<00:53, 6909.18it/s]  8%|▊         | 33969/400000 [00:04<00:52, 6935.46it/s]  9%|▊         | 34665/400000 [00:04<00:52, 6896.81it/s]  9%|▉         | 35359/400000 [00:04<00:52, 6907.87it/s]  9%|▉         | 36110/400000 [00:04<00:51, 7075.95it/s]  9%|▉         | 36820/400000 [00:05<00:51, 7025.43it/s]  9%|▉         | 37524/400000 [00:05<00:51, 6976.45it/s] 10%|▉         | 38223/400000 [00:05<00:52, 6907.10it/s] 10%|▉         | 38915/400000 [00:05<00:53, 6705.36it/s] 10%|▉         | 39627/400000 [00:05<00:52, 6823.51it/s] 10%|█         | 40312/400000 [00:05<00:53, 6739.71it/s] 10%|█         | 40988/400000 [00:05<00:53, 6740.43it/s] 10%|█         | 41733/400000 [00:05<00:51, 6936.67it/s] 11%|█         | 42457/400000 [00:05<00:50, 7022.31it/s] 11%|█         | 43201/400000 [00:06<00:49, 7140.76it/s] 11%|█         | 43920/400000 [00:06<00:49, 7154.09it/s] 11%|█         | 44642/400000 [00:06<00:49, 7173.30it/s] 11%|█▏        | 45361/400000 [00:06<00:51, 6944.27it/s] 12%|█▏        | 46058/400000 [00:06<00:51, 6904.05it/s] 12%|█▏        | 46777/400000 [00:06<00:50, 6986.92it/s] 12%|█▏        | 47535/400000 [00:06<00:49, 7153.64it/s] 12%|█▏        | 48253/400000 [00:06<00:49, 7058.99it/s] 12%|█▏        | 48994/400000 [00:06<00:49, 7160.31it/s] 12%|█▏        | 49712/400000 [00:06<00:49, 7147.46it/s] 13%|█▎        | 50428/400000 [00:07<00:50, 6980.30it/s] 13%|█▎        | 51128/400000 [00:07<00:50, 6970.67it/s] 13%|█▎        | 51875/400000 [00:07<00:48, 7112.95it/s] 13%|█▎        | 52633/400000 [00:07<00:47, 7245.57it/s] 13%|█▎        | 53382/400000 [00:07<00:47, 7315.33it/s] 14%|█▎        | 54115/400000 [00:07<00:47, 7305.83it/s] 14%|█▎        | 54858/400000 [00:07<00:47, 7340.55it/s] 14%|█▍        | 55593/400000 [00:07<00:47, 7324.12it/s] 14%|█▍        | 56357/400000 [00:07<00:46, 7414.27it/s] 14%|█▍        | 57125/400000 [00:07<00:45, 7489.60it/s] 14%|█▍        | 57875/400000 [00:08<00:45, 7475.73it/s] 15%|█▍        | 58626/400000 [00:08<00:45, 7485.25it/s] 15%|█▍        | 59392/400000 [00:08<00:45, 7534.44it/s] 15%|█▌        | 60146/400000 [00:08<00:45, 7506.51it/s] 15%|█▌        | 60911/400000 [00:08<00:44, 7547.91it/s] 15%|█▌        | 61667/400000 [00:08<00:45, 7504.09it/s] 16%|█▌        | 62418/400000 [00:08<00:46, 7303.14it/s] 16%|█▌        | 63178/400000 [00:08<00:45, 7388.94it/s] 16%|█▌        | 63945/400000 [00:08<00:44, 7468.74it/s] 16%|█▌        | 64719/400000 [00:08<00:44, 7547.66it/s] 16%|█▋        | 65475/400000 [00:09<00:44, 7546.91it/s] 17%|█▋        | 66245/400000 [00:09<00:43, 7591.42it/s] 17%|█▋        | 67005/400000 [00:09<00:43, 7571.68it/s] 17%|█▋        | 67763/400000 [00:09<00:44, 7461.88it/s] 17%|█▋        | 68516/400000 [00:09<00:44, 7480.24it/s] 17%|█▋        | 69265/400000 [00:09<00:44, 7449.07it/s] 18%|█▊        | 70032/400000 [00:09<00:43, 7511.37it/s] 18%|█▊        | 70785/400000 [00:09<00:43, 7514.78it/s] 18%|█▊        | 71537/400000 [00:09<00:44, 7403.72it/s] 18%|█▊        | 72278/400000 [00:09<00:44, 7319.01it/s] 18%|█▊        | 73011/400000 [00:10<00:46, 7101.77it/s] 18%|█▊        | 73724/400000 [00:10<00:46, 7052.33it/s] 19%|█▊        | 74484/400000 [00:10<00:45, 7206.57it/s] 19%|█▉        | 75216/400000 [00:10<00:44, 7240.19it/s] 19%|█▉        | 75942/400000 [00:10<00:45, 7140.00it/s] 19%|█▉        | 76685/400000 [00:10<00:44, 7224.27it/s] 19%|█▉        | 77432/400000 [00:10<00:44, 7293.46it/s] 20%|█▉        | 78192/400000 [00:10<00:43, 7381.04it/s] 20%|█▉        | 78947/400000 [00:10<00:43, 7429.73it/s] 20%|█▉        | 79691/400000 [00:10<00:43, 7406.69it/s] 20%|██        | 80433/400000 [00:11<00:43, 7329.59it/s] 20%|██        | 81194/400000 [00:11<00:43, 7410.10it/s] 20%|██        | 81965/400000 [00:11<00:42, 7494.92it/s] 21%|██        | 82716/400000 [00:11<00:42, 7486.54it/s] 21%|██        | 83466/400000 [00:11<00:42, 7469.19it/s] 21%|██        | 84214/400000 [00:11<00:43, 7297.80it/s] 21%|██        | 84977/400000 [00:11<00:42, 7392.65it/s] 21%|██▏       | 85729/400000 [00:11<00:42, 7428.30it/s] 22%|██▏       | 86498/400000 [00:11<00:41, 7503.79it/s] 22%|██▏       | 87268/400000 [00:12<00:41, 7560.37it/s] 22%|██▏       | 88025/400000 [00:12<00:42, 7416.64it/s] 22%|██▏       | 88768/400000 [00:12<00:42, 7362.47it/s] 22%|██▏       | 89506/400000 [00:12<00:42, 7283.90it/s] 23%|██▎       | 90253/400000 [00:12<00:42, 7336.04it/s] 23%|██▎       | 91018/400000 [00:12<00:41, 7421.99it/s] 23%|██▎       | 91762/400000 [00:12<00:41, 7426.42it/s] 23%|██▎       | 92523/400000 [00:12<00:41, 7480.13it/s] 23%|██▎       | 93272/400000 [00:12<00:41, 7470.74it/s] 24%|██▎       | 94038/400000 [00:12<00:40, 7526.48it/s] 24%|██▎       | 94799/400000 [00:13<00:40, 7548.44it/s] 24%|██▍       | 95555/400000 [00:13<00:40, 7481.84it/s] 24%|██▍       | 96310/400000 [00:13<00:40, 7500.63it/s] 24%|██▍       | 97061/400000 [00:13<00:40, 7448.28it/s] 24%|██▍       | 97807/400000 [00:13<00:40, 7446.13it/s] 25%|██▍       | 98552/400000 [00:13<00:41, 7348.38it/s] 25%|██▍       | 99306/400000 [00:13<00:40, 7402.72it/s] 25%|██▌       | 100065/400000 [00:13<00:40, 7456.17it/s] 25%|██▌       | 100837/400000 [00:13<00:39, 7531.70it/s] 25%|██▌       | 101591/400000 [00:13<00:40, 7450.16it/s] 26%|██▌       | 102343/400000 [00:14<00:39, 7469.78it/s] 26%|██▌       | 103091/400000 [00:14<00:39, 7440.04it/s] 26%|██▌       | 103859/400000 [00:14<00:39, 7507.75it/s] 26%|██▌       | 104611/400000 [00:14<00:40, 7338.63it/s] 26%|██▋       | 105354/400000 [00:14<00:40, 7365.70it/s] 27%|██▋       | 106099/400000 [00:14<00:39, 7389.66it/s] 27%|██▋       | 106863/400000 [00:14<00:39, 7461.70it/s] 27%|██▋       | 107629/400000 [00:14<00:38, 7518.40it/s] 27%|██▋       | 108382/400000 [00:14<00:38, 7479.24it/s] 27%|██▋       | 109140/400000 [00:14<00:38, 7507.86it/s] 27%|██▋       | 109892/400000 [00:15<00:39, 7405.72it/s] 28%|██▊       | 110650/400000 [00:15<00:38, 7455.58it/s] 28%|██▊       | 111397/400000 [00:15<00:38, 7440.13it/s] 28%|██▊       | 112157/400000 [00:15<00:38, 7487.14it/s] 28%|██▊       | 112907/400000 [00:15<00:38, 7442.10it/s] 28%|██▊       | 113671/400000 [00:15<00:38, 7499.22it/s] 29%|██▊       | 114422/400000 [00:15<00:38, 7327.69it/s] 29%|██▉       | 115183/400000 [00:15<00:38, 7408.76it/s] 29%|██▉       | 115948/400000 [00:15<00:37, 7476.39it/s] 29%|██▉       | 116718/400000 [00:15<00:37, 7540.51it/s] 29%|██▉       | 117483/400000 [00:16<00:37, 7570.23it/s] 30%|██▉       | 118241/400000 [00:16<00:37, 7546.02it/s] 30%|██▉       | 118996/400000 [00:16<00:37, 7475.12it/s] 30%|██▉       | 119744/400000 [00:16<00:37, 7463.58it/s] 30%|███       | 120505/400000 [00:16<00:37, 7506.85it/s] 30%|███       | 121256/400000 [00:16<00:37, 7385.85it/s] 31%|███       | 122015/400000 [00:16<00:37, 7444.12it/s] 31%|███       | 122762/400000 [00:16<00:37, 7450.93it/s] 31%|███       | 123525/400000 [00:16<00:36, 7502.51it/s] 31%|███       | 124298/400000 [00:16<00:36, 7568.22it/s] 31%|███▏      | 125071/400000 [00:17<00:36, 7615.38it/s] 31%|███▏      | 125833/400000 [00:17<00:36, 7488.23it/s] 32%|███▏      | 126599/400000 [00:17<00:36, 7538.45it/s] 32%|███▏      | 127357/400000 [00:17<00:36, 7550.53it/s] 32%|███▏      | 128126/400000 [00:17<00:35, 7591.48it/s] 32%|███▏      | 128901/400000 [00:17<00:35, 7638.22it/s] 32%|███▏      | 129666/400000 [00:17<00:35, 7619.58it/s] 33%|███▎      | 130443/400000 [00:17<00:35, 7663.78it/s] 33%|███▎      | 131215/400000 [00:17<00:35, 7678.52it/s] 33%|███▎      | 131984/400000 [00:17<00:35, 7592.33it/s] 33%|███▎      | 132744/400000 [00:18<00:35, 7444.89it/s] 33%|███▎      | 133490/400000 [00:18<00:35, 7430.06it/s] 34%|███▎      | 134249/400000 [00:18<00:35, 7475.46it/s] 34%|███▍      | 135028/400000 [00:18<00:35, 7566.42it/s] 34%|███▍      | 135786/400000 [00:18<00:35, 7453.16it/s] 34%|███▍      | 136533/400000 [00:18<00:36, 7284.99it/s] 34%|███▍      | 137263/400000 [00:18<00:36, 7228.62it/s] 34%|███▍      | 137989/400000 [00:18<00:36, 7235.29it/s] 35%|███▍      | 138716/400000 [00:18<00:36, 7243.60it/s] 35%|███▍      | 139447/400000 [00:19<00:35, 7261.21it/s] 35%|███▌      | 140201/400000 [00:19<00:35, 7342.28it/s] 35%|███▌      | 140943/400000 [00:19<00:35, 7363.65it/s] 35%|███▌      | 141686/400000 [00:19<00:34, 7382.55it/s] 36%|███▌      | 142459/400000 [00:19<00:34, 7482.67it/s] 36%|███▌      | 143208/400000 [00:19<00:34, 7476.38it/s] 36%|███▌      | 143956/400000 [00:19<00:34, 7414.74it/s] 36%|███▌      | 144698/400000 [00:19<00:34, 7314.02it/s] 36%|███▋      | 145459/400000 [00:19<00:34, 7397.89it/s] 37%|███▋      | 146224/400000 [00:19<00:33, 7471.06it/s] 37%|███▋      | 146972/400000 [00:20<00:33, 7454.40it/s] 37%|███▋      | 147740/400000 [00:20<00:33, 7519.33it/s] 37%|███▋      | 148493/400000 [00:20<00:33, 7492.57it/s] 37%|███▋      | 149243/400000 [00:20<00:33, 7477.40it/s] 38%|███▊      | 150006/400000 [00:20<00:33, 7521.96it/s] 38%|███▊      | 150759/400000 [00:20<00:33, 7437.41it/s] 38%|███▊      | 151524/400000 [00:20<00:33, 7498.44it/s] 38%|███▊      | 152285/400000 [00:20<00:32, 7529.19it/s] 38%|███▊      | 153039/400000 [00:20<00:32, 7489.71it/s] 38%|███▊      | 153806/400000 [00:20<00:32, 7541.64it/s] 39%|███▊      | 154586/400000 [00:21<00:32, 7614.66it/s] 39%|███▉      | 155348/400000 [00:21<00:32, 7585.37it/s] 39%|███▉      | 156107/400000 [00:21<00:32, 7559.29it/s] 39%|███▉      | 156881/400000 [00:21<00:31, 7611.46it/s] 39%|███▉      | 157643/400000 [00:21<00:32, 7542.78it/s] 40%|███▉      | 158400/400000 [00:21<00:32, 7549.39it/s] 40%|███▉      | 159156/400000 [00:21<00:32, 7410.65it/s] 40%|███▉      | 159898/400000 [00:21<00:32, 7347.19it/s] 40%|████      | 160658/400000 [00:21<00:32, 7419.03it/s] 40%|████      | 161421/400000 [00:21<00:31, 7479.11it/s] 41%|████      | 162170/400000 [00:22<00:31, 7458.77it/s] 41%|████      | 162934/400000 [00:22<00:31, 7511.95it/s] 41%|████      | 163686/400000 [00:22<00:32, 7340.54it/s] 41%|████      | 164443/400000 [00:22<00:31, 7406.28it/s] 41%|████▏     | 165193/400000 [00:22<00:31, 7434.13it/s] 41%|████▏     | 165938/400000 [00:22<00:31, 7385.53it/s] 42%|████▏     | 166687/400000 [00:22<00:31, 7413.88it/s] 42%|████▏     | 167451/400000 [00:22<00:31, 7480.14it/s] 42%|████▏     | 168230/400000 [00:22<00:30, 7570.32it/s] 42%|████▏     | 169005/400000 [00:22<00:30, 7620.75it/s] 42%|████▏     | 169768/400000 [00:23<00:30, 7573.79it/s] 43%|████▎     | 170526/400000 [00:23<00:30, 7536.74it/s] 43%|████▎     | 171280/400000 [00:23<00:30, 7511.89it/s] 43%|████▎     | 172055/400000 [00:23<00:30, 7580.49it/s] 43%|████▎     | 172817/400000 [00:23<00:29, 7591.13it/s] 43%|████▎     | 173577/400000 [00:23<00:30, 7519.80it/s] 44%|████▎     | 174351/400000 [00:23<00:29, 7580.26it/s] 44%|████▍     | 175110/400000 [00:23<00:29, 7499.66it/s] 44%|████▍     | 175861/400000 [00:23<00:30, 7448.41it/s] 44%|████▍     | 176632/400000 [00:23<00:29, 7522.39it/s] 44%|████▍     | 177385/400000 [00:24<00:29, 7518.06it/s] 45%|████▍     | 178152/400000 [00:24<00:29, 7562.16it/s] 45%|████▍     | 178909/400000 [00:24<00:29, 7497.92it/s] 45%|████▍     | 179678/400000 [00:24<00:29, 7552.31it/s] 45%|████▌     | 180447/400000 [00:24<00:28, 7590.48it/s] 45%|████▌     | 181212/400000 [00:24<00:28, 7605.68it/s] 45%|████▌     | 181992/400000 [00:24<00:28, 7660.14it/s] 46%|████▌     | 182759/400000 [00:24<00:28, 7618.49it/s] 46%|████▌     | 183522/400000 [00:24<00:28, 7525.93it/s] 46%|████▌     | 184275/400000 [00:24<00:28, 7516.59it/s] 46%|████▋     | 185031/400000 [00:25<00:28, 7529.39it/s] 46%|████▋     | 185801/400000 [00:25<00:28, 7578.42it/s] 47%|████▋     | 186560/400000 [00:25<00:28, 7434.67it/s] 47%|████▋     | 187314/400000 [00:25<00:28, 7464.75it/s] 47%|████▋     | 188068/400000 [00:25<00:28, 7468.98it/s] 47%|████▋     | 188816/400000 [00:25<00:28, 7422.51it/s] 47%|████▋     | 189561/400000 [00:25<00:28, 7429.13it/s] 48%|████▊     | 190305/400000 [00:25<00:28, 7431.37it/s] 48%|████▊     | 191080/400000 [00:25<00:27, 7521.76it/s] 48%|████▊     | 191833/400000 [00:25<00:27, 7519.45it/s] 48%|████▊     | 192602/400000 [00:26<00:27, 7569.22it/s] 48%|████▊     | 193379/400000 [00:26<00:27, 7626.17it/s] 49%|████▊     | 194142/400000 [00:26<00:27, 7577.11it/s] 49%|████▊     | 194908/400000 [00:26<00:26, 7599.19it/s] 49%|████▉     | 195669/400000 [00:26<00:26, 7586.10it/s] 49%|████▉     | 196428/400000 [00:26<00:27, 7515.55it/s] 49%|████▉     | 197210/400000 [00:26<00:26, 7602.43it/s] 49%|████▉     | 197971/400000 [00:26<00:26, 7570.27it/s] 50%|████▉     | 198740/400000 [00:26<00:26, 7605.46it/s] 50%|████▉     | 199514/400000 [00:27<00:26, 7644.00it/s] 50%|█████     | 200279/400000 [00:27<00:26, 7600.77it/s] 50%|█████     | 201044/400000 [00:27<00:26, 7615.29it/s] 50%|█████     | 201806/400000 [00:27<00:26, 7522.65it/s] 51%|█████     | 202576/400000 [00:27<00:26, 7573.98it/s] 51%|█████     | 203341/400000 [00:27<00:25, 7595.60it/s] 51%|█████     | 204111/400000 [00:27<00:25, 7624.66it/s] 51%|█████     | 204874/400000 [00:27<00:25, 7571.06it/s] 51%|█████▏    | 205632/400000 [00:27<00:25, 7566.94it/s] 52%|█████▏    | 206389/400000 [00:27<00:25, 7544.49it/s] 52%|█████▏    | 207151/400000 [00:28<00:25, 7565.96it/s] 52%|█████▏    | 207911/400000 [00:28<00:25, 7556.47it/s] 52%|█████▏    | 208667/400000 [00:28<00:25, 7543.33it/s] 52%|█████▏    | 209422/400000 [00:28<00:25, 7364.15it/s] 53%|█████▎    | 210160/400000 [00:28<00:26, 7047.30it/s] 53%|█████▎    | 210933/400000 [00:28<00:26, 7237.97it/s] 53%|█████▎    | 211692/400000 [00:28<00:25, 7338.78it/s] 53%|█████▎    | 212429/400000 [00:28<00:25, 7344.38it/s] 53%|█████▎    | 213166/400000 [00:28<00:25, 7343.19it/s] 53%|█████▎    | 213921/400000 [00:28<00:25, 7402.28it/s] 54%|█████▎    | 214686/400000 [00:29<00:24, 7472.97it/s] 54%|█████▍    | 215461/400000 [00:29<00:24, 7553.41it/s] 54%|█████▍    | 216230/400000 [00:29<00:24, 7591.12it/s] 54%|█████▍    | 216990/400000 [00:29<00:24, 7560.18it/s] 54%|█████▍    | 217747/400000 [00:29<00:24, 7406.02it/s] 55%|█████▍    | 218513/400000 [00:29<00:24, 7480.28it/s] 55%|█████▍    | 219268/400000 [00:29<00:24, 7499.54it/s] 55%|█████▌    | 220041/400000 [00:29<00:23, 7566.65it/s] 55%|█████▌    | 220799/400000 [00:29<00:23, 7569.40it/s] 55%|█████▌    | 221557/400000 [00:29<00:24, 7393.59it/s] 56%|█████▌    | 222298/400000 [00:30<00:24, 7390.87it/s] 56%|█████▌    | 223046/400000 [00:30<00:23, 7416.44it/s] 56%|█████▌    | 223796/400000 [00:30<00:23, 7441.25it/s] 56%|█████▌    | 224541/400000 [00:30<00:23, 7417.91it/s] 56%|█████▋    | 225291/400000 [00:30<00:23, 7441.52it/s] 57%|█████▋    | 226036/400000 [00:30<00:23, 7351.91it/s] 57%|█████▋    | 226784/400000 [00:30<00:23, 7387.74it/s] 57%|█████▋    | 227524/400000 [00:30<00:23, 7350.75it/s] 57%|█████▋    | 228260/400000 [00:30<00:23, 7313.21it/s] 57%|█████▋    | 228992/400000 [00:30<00:23, 7298.59it/s] 57%|█████▋    | 229754/400000 [00:31<00:23, 7391.04it/s] 58%|█████▊    | 230494/400000 [00:31<00:23, 7354.05it/s] 58%|█████▊    | 231257/400000 [00:31<00:22, 7433.27it/s] 58%|█████▊    | 232002/400000 [00:31<00:22, 7436.11it/s] 58%|█████▊    | 232746/400000 [00:31<00:22, 7409.38it/s] 58%|█████▊    | 233488/400000 [00:31<00:22, 7312.73it/s] 59%|█████▊    | 234239/400000 [00:31<00:22, 7368.41it/s] 59%|█████▊    | 234977/400000 [00:31<00:22, 7316.38it/s] 59%|█████▉    | 235710/400000 [00:31<00:22, 7237.87it/s] 59%|█████▉    | 236482/400000 [00:31<00:22, 7374.46it/s] 59%|█████▉    | 237257/400000 [00:32<00:21, 7482.12it/s] 60%|█████▉    | 238031/400000 [00:32<00:21, 7557.34it/s] 60%|█████▉    | 238788/400000 [00:32<00:21, 7530.85it/s] 60%|█████▉    | 239550/400000 [00:32<00:21, 7556.34it/s] 60%|██████    | 240307/400000 [00:32<00:21, 7497.25it/s] 60%|██████    | 241081/400000 [00:32<00:21, 7566.96it/s] 60%|██████    | 241843/400000 [00:32<00:20, 7582.76it/s] 61%|██████    | 242612/400000 [00:32<00:20, 7611.40it/s] 61%|██████    | 243374/400000 [00:32<00:20, 7490.20it/s] 61%|██████    | 244124/400000 [00:32<00:21, 7361.71it/s] 61%|██████    | 244883/400000 [00:33<00:20, 7427.61it/s] 61%|██████▏   | 245649/400000 [00:33<00:20, 7493.95it/s] 62%|██████▏   | 246416/400000 [00:33<00:20, 7545.70it/s] 62%|██████▏   | 247172/400000 [00:33<00:20, 7478.69it/s] 62%|██████▏   | 247945/400000 [00:33<00:20, 7549.86it/s] 62%|██████▏   | 248720/400000 [00:33<00:19, 7605.92it/s] 62%|██████▏   | 249487/400000 [00:33<00:19, 7623.79it/s] 63%|██████▎   | 250255/400000 [00:33<00:19, 7638.34it/s] 63%|██████▎   | 251020/400000 [00:33<00:19, 7588.01it/s] 63%|██████▎   | 251780/400000 [00:34<00:19, 7533.31it/s] 63%|██████▎   | 252538/400000 [00:34<00:19, 7545.36it/s] 63%|██████▎   | 253311/400000 [00:34<00:19, 7598.98it/s] 64%|██████▎   | 254083/400000 [00:34<00:19, 7632.16it/s] 64%|██████▎   | 254847/400000 [00:34<00:19, 7500.73it/s] 64%|██████▍   | 255598/400000 [00:34<00:19, 7499.26it/s] 64%|██████▍   | 256349/400000 [00:34<00:19, 7493.73it/s] 64%|██████▍   | 257113/400000 [00:34<00:18, 7536.08it/s] 64%|██████▍   | 257876/400000 [00:34<00:18, 7562.28it/s] 65%|██████▍   | 258633/400000 [00:34<00:18, 7525.74it/s] 65%|██████▍   | 259399/400000 [00:35<00:18, 7563.85it/s] 65%|██████▌   | 260156/400000 [00:35<00:18, 7544.80it/s] 65%|██████▌   | 260929/400000 [00:35<00:18, 7599.12it/s] 65%|██████▌   | 261703/400000 [00:35<00:18, 7638.31it/s] 66%|██████▌   | 262468/400000 [00:35<00:18, 7590.25it/s] 66%|██████▌   | 263239/400000 [00:35<00:17, 7624.44it/s] 66%|██████▌   | 264013/400000 [00:35<00:17, 7657.43it/s] 66%|██████▌   | 264779/400000 [00:35<00:18, 7431.37it/s] 66%|██████▋   | 265543/400000 [00:35<00:17, 7490.22it/s] 67%|██████▋   | 266294/400000 [00:35<00:18, 7419.44it/s] 67%|██████▋   | 267062/400000 [00:36<00:17, 7492.07it/s] 67%|██████▋   | 267828/400000 [00:36<00:17, 7539.30it/s] 67%|██████▋   | 268583/400000 [00:36<00:17, 7533.82it/s] 67%|██████▋   | 269343/400000 [00:36<00:17, 7551.95it/s] 68%|██████▊   | 270099/400000 [00:36<00:17, 7529.93it/s] 68%|██████▊   | 270853/400000 [00:36<00:17, 7383.21it/s] 68%|██████▊   | 271623/400000 [00:36<00:17, 7474.38it/s] 68%|██████▊   | 272393/400000 [00:36<00:16, 7538.15it/s] 68%|██████▊   | 273148/400000 [00:36<00:17, 7381.16it/s] 68%|██████▊   | 273901/400000 [00:36<00:16, 7422.45it/s] 69%|██████▊   | 274663/400000 [00:37<00:16, 7478.43it/s] 69%|██████▉   | 275420/400000 [00:37<00:16, 7504.32it/s] 69%|██████▉   | 276185/400000 [00:37<00:16, 7547.16it/s] 69%|██████▉   | 276941/400000 [00:37<00:16, 7519.41it/s] 69%|██████▉   | 277696/400000 [00:37<00:16, 7528.50it/s] 70%|██████▉   | 278471/400000 [00:37<00:16, 7592.71it/s] 70%|██████▉   | 279247/400000 [00:37<00:15, 7641.89it/s] 70%|███████   | 280012/400000 [00:37<00:15, 7603.11it/s] 70%|███████   | 280786/400000 [00:37<00:15, 7642.43it/s] 70%|███████   | 281551/400000 [00:37<00:15, 7520.72it/s] 71%|███████   | 282304/400000 [00:38<00:15, 7511.65it/s] 71%|███████   | 283056/400000 [00:38<00:16, 7249.33it/s] 71%|███████   | 283821/400000 [00:38<00:15, 7364.39it/s] 71%|███████   | 284583/400000 [00:38<00:15, 7439.23it/s] 71%|███████▏  | 285343/400000 [00:38<00:15, 7486.31it/s] 72%|███████▏  | 286093/400000 [00:38<00:15, 7465.62it/s] 72%|███████▏  | 286841/400000 [00:38<00:15, 7408.62it/s] 72%|███████▏  | 287583/400000 [00:38<00:15, 7397.13it/s] 72%|███████▏  | 288324/400000 [00:38<00:15, 7369.14it/s] 72%|███████▏  | 289062/400000 [00:38<00:15, 7218.69it/s] 72%|███████▏  | 289792/400000 [00:39<00:15, 7242.22it/s] 73%|███████▎  | 290537/400000 [00:39<00:14, 7302.70it/s] 73%|███████▎  | 291278/400000 [00:39<00:14, 7333.92it/s] 73%|███████▎  | 292044/400000 [00:39<00:14, 7427.04it/s] 73%|███████▎  | 292794/400000 [00:39<00:14, 7447.51it/s] 73%|███████▎  | 293567/400000 [00:39<00:14, 7528.41it/s] 74%|███████▎  | 294321/400000 [00:39<00:14, 7429.06it/s] 74%|███████▍  | 295092/400000 [00:39<00:13, 7510.04it/s] 74%|███████▍  | 295863/400000 [00:39<00:13, 7566.33it/s] 74%|███████▍  | 296621/400000 [00:39<00:13, 7550.20it/s] 74%|███████▍  | 297384/400000 [00:40<00:13, 7571.15it/s] 75%|███████▍  | 298142/400000 [00:40<00:13, 7446.40it/s] 75%|███████▍  | 298893/400000 [00:40<00:13, 7462.49it/s] 75%|███████▍  | 299663/400000 [00:40<00:13, 7530.16it/s] 75%|███████▌  | 300418/400000 [00:40<00:13, 7533.59it/s] 75%|███████▌  | 301172/400000 [00:40<00:13, 7489.04it/s] 75%|███████▌  | 301930/400000 [00:40<00:13, 7515.28it/s] 76%|███████▌  | 302688/400000 [00:40<00:12, 7533.60it/s] 76%|███████▌  | 303451/400000 [00:40<00:12, 7560.63it/s] 76%|███████▌  | 304208/400000 [00:40<00:12, 7555.29it/s] 76%|███████▌  | 304979/400000 [00:41<00:12, 7599.22it/s] 76%|███████▋  | 305740/400000 [00:41<00:12, 7581.77it/s] 77%|███████▋  | 306499/400000 [00:41<00:12, 7526.28it/s] 77%|███████▋  | 307252/400000 [00:41<00:12, 7373.15it/s] 77%|███████▋  | 308010/400000 [00:41<00:12, 7432.19it/s] 77%|███████▋  | 308780/400000 [00:41<00:12, 7508.36it/s] 77%|███████▋  | 309558/400000 [00:41<00:11, 7585.97it/s] 78%|███████▊  | 310333/400000 [00:41<00:11, 7632.68it/s] 78%|███████▊  | 311097/400000 [00:41<00:11, 7628.90it/s] 78%|███████▊  | 311861/400000 [00:42<00:11, 7542.49it/s] 78%|███████▊  | 312639/400000 [00:42<00:11, 7610.87it/s] 78%|███████▊  | 313401/400000 [00:42<00:11, 7460.58it/s] 79%|███████▊  | 314175/400000 [00:42<00:11, 7539.24it/s] 79%|███████▊  | 314935/400000 [00:42<00:11, 7555.20it/s] 79%|███████▉  | 315692/400000 [00:42<00:11, 7406.09it/s] 79%|███████▉  | 316468/400000 [00:42<00:11, 7508.05it/s] 79%|███████▉  | 317241/400000 [00:42<00:10, 7573.22it/s] 80%|███████▉  | 318016/400000 [00:42<00:10, 7622.97it/s] 80%|███████▉  | 318780/400000 [00:42<00:10, 7623.91it/s] 80%|███████▉  | 319543/400000 [00:43<00:10, 7565.73it/s] 80%|████████  | 320302/400000 [00:43<00:10, 7570.86it/s] 80%|████████  | 321060/400000 [00:43<00:10, 7552.57it/s] 80%|████████  | 321816/400000 [00:43<00:10, 7552.08it/s] 81%|████████  | 322593/400000 [00:43<00:10, 7615.63it/s] 81%|████████  | 323355/400000 [00:43<00:10, 7594.07it/s] 81%|████████  | 324115/400000 [00:43<00:10, 7586.03it/s] 81%|████████  | 324874/400000 [00:43<00:10, 7458.08it/s] 81%|████████▏ | 325645/400000 [00:43<00:09, 7530.92it/s] 82%|████████▏ | 326403/400000 [00:43<00:09, 7544.15it/s] 82%|████████▏ | 327158/400000 [00:44<00:09, 7542.53it/s] 82%|████████▏ | 327916/400000 [00:44<00:09, 7550.78it/s] 82%|████████▏ | 328672/400000 [00:44<00:09, 7479.43it/s] 82%|████████▏ | 329441/400000 [00:44<00:09, 7539.24it/s] 83%|████████▎ | 330209/400000 [00:44<00:09, 7579.29it/s] 83%|████████▎ | 330972/400000 [00:44<00:09, 7591.88it/s] 83%|████████▎ | 331732/400000 [00:44<00:08, 7585.59it/s] 83%|████████▎ | 332491/400000 [00:44<00:08, 7557.18it/s] 83%|████████▎ | 333259/400000 [00:44<00:08, 7593.45it/s] 84%|████████▎ | 334019/400000 [00:44<00:09, 7242.33it/s] 84%|████████▎ | 334747/400000 [00:45<00:09, 7223.50it/s] 84%|████████▍ | 335519/400000 [00:45<00:08, 7363.74it/s] 84%|████████▍ | 336258/400000 [00:45<00:08, 7356.55it/s] 84%|████████▍ | 336996/400000 [00:45<00:08, 7225.09it/s] 84%|████████▍ | 337753/400000 [00:45<00:08, 7324.35it/s] 85%|████████▍ | 338488/400000 [00:45<00:08, 7330.71it/s] 85%|████████▍ | 339256/400000 [00:45<00:08, 7430.32it/s] 85%|████████▌ | 340025/400000 [00:45<00:07, 7505.11it/s] 85%|████████▌ | 340792/400000 [00:45<00:07, 7551.20it/s] 85%|████████▌ | 341548/400000 [00:45<00:07, 7456.52it/s] 86%|████████▌ | 342295/400000 [00:46<00:07, 7334.97it/s] 86%|████████▌ | 343051/400000 [00:46<00:07, 7400.99it/s] 86%|████████▌ | 343799/400000 [00:46<00:07, 7422.01it/s] 86%|████████▌ | 344553/400000 [00:46<00:07, 7454.31it/s] 86%|████████▋ | 345299/400000 [00:46<00:07, 7201.10it/s] 87%|████████▋ | 346029/400000 [00:46<00:07, 7229.61it/s] 87%|████████▋ | 346754/400000 [00:46<00:07, 7151.68it/s] 87%|████████▋ | 347520/400000 [00:46<00:07, 7294.45it/s] 87%|████████▋ | 348273/400000 [00:46<00:07, 7363.34it/s] 87%|████████▋ | 349028/400000 [00:46<00:06, 7415.81it/s] 87%|████████▋ | 349771/400000 [00:47<00:07, 7148.85it/s] 88%|████████▊ | 350522/400000 [00:47<00:06, 7252.14it/s] 88%|████████▊ | 351250/400000 [00:47<00:06, 7248.32it/s] 88%|████████▊ | 352015/400000 [00:47<00:06, 7363.73it/s] 88%|████████▊ | 352770/400000 [00:47<00:06, 7418.07it/s] 88%|████████▊ | 353513/400000 [00:47<00:06, 7390.24it/s] 89%|████████▊ | 354253/400000 [00:47<00:06, 7356.70it/s] 89%|████████▉ | 355024/400000 [00:47<00:06, 7456.96it/s] 89%|████████▉ | 355782/400000 [00:47<00:05, 7492.79it/s] 89%|████████▉ | 356540/400000 [00:48<00:05, 7516.35it/s] 89%|████████▉ | 357293/400000 [00:48<00:05, 7496.30it/s] 90%|████████▉ | 358043/400000 [00:48<00:05, 7476.38it/s] 90%|████████▉ | 358814/400000 [00:48<00:05, 7543.46it/s] 90%|████████▉ | 359587/400000 [00:48<00:05, 7597.76it/s] 90%|█████████ | 360348/400000 [00:48<00:05, 7578.73it/s] 90%|█████████ | 361107/400000 [00:48<00:05, 7516.47it/s] 90%|█████████ | 361874/400000 [00:48<00:05, 7561.31it/s] 91%|█████████ | 362631/400000 [00:48<00:04, 7539.02it/s] 91%|█████████ | 363405/400000 [00:48<00:04, 7597.23it/s] 91%|█████████ | 364165/400000 [00:49<00:04, 7523.68it/s] 91%|█████████ | 364918/400000 [00:49<00:04, 7414.93it/s] 91%|█████████▏| 365661/400000 [00:49<00:04, 7418.62it/s] 92%|█████████▏| 366404/400000 [00:49<00:04, 7344.88it/s] 92%|█████████▏| 367139/400000 [00:49<00:04, 7320.96it/s] 92%|█████████▏| 367910/400000 [00:49<00:04, 7432.66it/s] 92%|█████████▏| 368655/400000 [00:49<00:04, 7435.67it/s] 92%|█████████▏| 369412/400000 [00:49<00:04, 7473.28it/s] 93%|█████████▎| 370172/400000 [00:49<00:03, 7508.82it/s] 93%|█████████▎| 370924/400000 [00:49<00:03, 7451.27it/s] 93%|█████████▎| 371670/400000 [00:50<00:03, 7434.17it/s] 93%|█████████▎| 372414/400000 [00:50<00:03, 7428.72it/s] 93%|█████████▎| 373182/400000 [00:50<00:03, 7501.38it/s] 93%|█████████▎| 373953/400000 [00:50<00:03, 7560.05it/s] 94%|█████████▎| 374724/400000 [00:50<00:03, 7602.73it/s] 94%|█████████▍| 375485/400000 [00:50<00:03, 7598.91it/s] 94%|█████████▍| 376246/400000 [00:50<00:03, 7529.87it/s] 94%|█████████▍| 377000/400000 [00:50<00:03, 7483.92it/s] 94%|█████████▍| 377749/400000 [00:50<00:02, 7424.58it/s] 95%|█████████▍| 378514/400000 [00:50<00:02, 7489.80it/s] 95%|█████████▍| 379266/400000 [00:51<00:02, 7497.96it/s] 95%|█████████▌| 380017/400000 [00:51<00:02, 7453.50it/s] 95%|█████████▌| 380763/400000 [00:51<00:02, 7294.74it/s] 95%|█████████▌| 381533/400000 [00:51<00:02, 7409.25it/s] 96%|█████████▌| 382299/400000 [00:51<00:02, 7481.31it/s] 96%|█████████▌| 383051/400000 [00:51<00:02, 7491.58it/s] 96%|█████████▌| 383801/400000 [00:51<00:02, 7393.46it/s] 96%|█████████▌| 384554/400000 [00:51<00:02, 7433.66it/s] 96%|█████████▋| 385323/400000 [00:51<00:01, 7507.91it/s] 97%|█████████▋| 386094/400000 [00:51<00:01, 7567.19it/s] 97%|█████████▋| 386852/400000 [00:52<00:01, 7433.78it/s] 97%|█████████▋| 387597/400000 [00:52<00:01, 7391.66it/s] 97%|█████████▋| 388343/400000 [00:52<00:01, 7409.71it/s] 97%|█████████▋| 389105/400000 [00:52<00:01, 7469.44it/s] 97%|█████████▋| 389853/400000 [00:52<00:01, 7255.63it/s] 98%|█████████▊| 390610/400000 [00:52<00:01, 7346.67it/s] 98%|█████████▊| 391347/400000 [00:52<00:01, 7347.83it/s] 98%|█████████▊| 392083/400000 [00:52<00:01, 7274.52it/s] 98%|█████████▊| 392812/400000 [00:52<00:00, 7252.30it/s] 98%|█████████▊| 393538/400000 [00:52<00:00, 7090.24it/s] 99%|█████████▊| 394249/400000 [00:53<00:00, 7061.62it/s] 99%|█████████▊| 394972/400000 [00:53<00:00, 7109.76it/s] 99%|█████████▉| 395684/400000 [00:53<00:00, 7044.77it/s] 99%|█████████▉| 396413/400000 [00:53<00:00, 7115.32it/s] 99%|█████████▉| 397177/400000 [00:53<00:00, 7264.16it/s] 99%|█████████▉| 397937/400000 [00:53<00:00, 7361.05it/s]100%|█████████▉| 398675/400000 [00:53<00:00, 7365.65it/s]100%|█████████▉| 399441/400000 [00:53<00:00, 7450.38it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7425.64it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe596861400>, <torchtext.data.dataset.TabularDataset object at 0x7fe596861550>, <torchtext.vocab.Vocab object at 0x7fe596861470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.78 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 12.11 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 12.11 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.23 file/s]2020-06-29 18:19:07.897317: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-29 18:19:07.902554: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-06-29 18:19:07.903584: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b2ddd41460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-29 18:19:07.903607: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 156919.58it/s] 75%|███████▍  | 7397376/9912422 [00:00<00:11, 223966.72it/s]9920512it [00:00, 44734865.61it/s]                           
0it [00:00, ?it/s]32768it [00:00, 695981.53it/s]
0it [00:00, ?it/s]  1%|▏         | 24576/1648877 [00:00<00:06, 245723.33it/s]1654784it [00:00, 12528257.44it/s]                         
0it [00:00, ?it/s]8192it [00:00, 216383.41it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
