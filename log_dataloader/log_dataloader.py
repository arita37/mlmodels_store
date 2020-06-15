
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fa2bd5016a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fa2bd5016a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fa328ac90d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fa328ac90d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fa3427f5ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fa3427f5ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa2d6344950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa2d6344950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa2d6344950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:18, 125731.81it/s] 76%|███████▌  | 7553024/9912422 [00:00<00:13, 179487.81it/s]9920512it [00:00, 39790539.09it/s]                           
0it [00:00, ?it/s]32768it [00:00, 677369.52it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 990018.62it/s]1654784it [00:00, 12252445.22it/s]                          
0it [00:00, ?it/s]8192it [00:00, 192789.63it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2bd3deba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2bd3d0e48>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fa2d6344598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fa2d6344598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fa2d6344598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:20,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:20,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:19,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:19,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:18,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:17,  2.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:54,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:53,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:52,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:51,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:51,  2.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:34,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:33,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:33,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:33,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:23,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:22,  5.60 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:15,  7.75 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:14,  7.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:14,  7.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:09, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:09, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:09, 10.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:06, 14.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 19.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 25.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 32.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 49.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 58.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.43 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 74.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.17 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.26s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 76.17 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.97 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.77s/ url]
0 examples [00:00, ? examples/s]2020-06-15 18:10:11.982547: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 18:10:11.999706: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-15 18:10:12.000056: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e8e63dfc40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 18:10:12.000181: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
18 examples [00:00, 178.31 examples/s]94 examples [00:00, 231.35 examples/s]179 examples [00:00, 295.74 examples/s]266 examples [00:00, 368.66 examples/s]350 examples [00:00, 442.86 examples/s]442 examples [00:00, 524.25 examples/s]515 examples [00:00, 569.13 examples/s]607 examples [00:00, 642.05 examples/s]697 examples [00:00, 702.36 examples/s]779 examples [00:01, 698.63 examples/s]865 examples [00:01, 739.27 examples/s]946 examples [00:01, 717.89 examples/s]1035 examples [00:01, 760.59 examples/s]1124 examples [00:01, 793.12 examples/s]1210 examples [00:01, 809.56 examples/s]1301 examples [00:01, 836.80 examples/s]1387 examples [00:01, 807.69 examples/s]1476 examples [00:01, 830.08 examples/s]1562 examples [00:01, 836.09 examples/s]1654 examples [00:02, 859.04 examples/s]1748 examples [00:02, 880.96 examples/s]1837 examples [00:02, 865.25 examples/s]1935 examples [00:02, 894.75 examples/s]2026 examples [00:02, 885.38 examples/s]2115 examples [00:02, 867.71 examples/s]2203 examples [00:02, 860.50 examples/s]2290 examples [00:02, 845.14 examples/s]2380 examples [00:02, 858.56 examples/s]2472 examples [00:02, 875.07 examples/s]2562 examples [00:03, 880.71 examples/s]2654 examples [00:03, 889.93 examples/s]2744 examples [00:03, 836.20 examples/s]2830 examples [00:03, 842.07 examples/s]2920 examples [00:03, 856.30 examples/s]3007 examples [00:03, 828.36 examples/s]3091 examples [00:03, 816.65 examples/s]3176 examples [00:03, 825.51 examples/s]3270 examples [00:03, 854.69 examples/s]3365 examples [00:04, 878.32 examples/s]3454 examples [00:04, 875.85 examples/s]3543 examples [00:04, 876.21 examples/s]3631 examples [00:04, 825.71 examples/s]3724 examples [00:04, 853.49 examples/s]3817 examples [00:04, 874.88 examples/s]3909 examples [00:04, 886.22 examples/s]4002 examples [00:04, 897.83 examples/s]4093 examples [00:04, 843.77 examples/s]4187 examples [00:04, 870.00 examples/s]4277 examples [00:05, 877.19 examples/s]4366 examples [00:05, 848.78 examples/s]4456 examples [00:05, 861.13 examples/s]4543 examples [00:05, 799.76 examples/s]4637 examples [00:05, 836.02 examples/s]4722 examples [00:05, 836.10 examples/s]4816 examples [00:05, 864.12 examples/s]4904 examples [00:05, 849.03 examples/s]4990 examples [00:05, 848.53 examples/s]5087 examples [00:06, 880.96 examples/s]5176 examples [00:06, 882.08 examples/s]5270 examples [00:06, 897.77 examples/s]5361 examples [00:06, 849.04 examples/s]5448 examples [00:06, 854.06 examples/s]5537 examples [00:06, 862.80 examples/s]5624 examples [00:06, 852.68 examples/s]5710 examples [00:06, 817.90 examples/s]5793 examples [00:06, 819.82 examples/s]5876 examples [00:06, 809.34 examples/s]5970 examples [00:07, 841.82 examples/s]6055 examples [00:07, 838.85 examples/s]6143 examples [00:07, 850.74 examples/s]6232 examples [00:07, 860.07 examples/s]6324 examples [00:07, 877.10 examples/s]6413 examples [00:07, 880.43 examples/s]6502 examples [00:07, 875.24 examples/s]6591 examples [00:07, 879.57 examples/s]6680 examples [00:07, 829.84 examples/s]6771 examples [00:08, 851.81 examples/s]6862 examples [00:08, 866.96 examples/s]6953 examples [00:08, 877.67 examples/s]7042 examples [00:08, 876.10 examples/s]7130 examples [00:08, 834.22 examples/s]7215 examples [00:08, 829.90 examples/s]7299 examples [00:08, 816.36 examples/s]7381 examples [00:08, 805.52 examples/s]7468 examples [00:08, 823.37 examples/s]7551 examples [00:08, 819.12 examples/s]7634 examples [00:09, 815.15 examples/s]7716 examples [00:09, 787.29 examples/s]7811 examples [00:09, 828.83 examples/s]7910 examples [00:09, 869.61 examples/s]7999 examples [00:09, 870.62 examples/s]8095 examples [00:09, 892.93 examples/s]8185 examples [00:09, 894.20 examples/s]8275 examples [00:09, 836.23 examples/s]8360 examples [00:09, 837.79 examples/s]8445 examples [00:10, 814.17 examples/s]8536 examples [00:10, 838.47 examples/s]8624 examples [00:10, 848.76 examples/s]8713 examples [00:10, 858.91 examples/s]8803 examples [00:10, 868.45 examples/s]8891 examples [00:10, 836.88 examples/s]8983 examples [00:10, 859.68 examples/s]9075 examples [00:10, 875.65 examples/s]9163 examples [00:10, 865.77 examples/s]9250 examples [00:10, 864.79 examples/s]9337 examples [00:11, 849.60 examples/s]9427 examples [00:11, 863.09 examples/s]9521 examples [00:11, 884.23 examples/s]9615 examples [00:11, 898.21 examples/s]9706 examples [00:11, 858.57 examples/s]9793 examples [00:11, 831.66 examples/s]9883 examples [00:11, 850.81 examples/s]9969 examples [00:11, 852.65 examples/s]10055 examples [00:11, 795.77 examples/s]10136 examples [00:12, 788.12 examples/s]10216 examples [00:12, 778.88 examples/s]10309 examples [00:12, 816.96 examples/s]10401 examples [00:12, 843.23 examples/s]10494 examples [00:12, 865.88 examples/s]10591 examples [00:12, 892.96 examples/s]10682 examples [00:12, 823.36 examples/s]10771 examples [00:12, 840.44 examples/s]10861 examples [00:12, 855.44 examples/s]10948 examples [00:12, 828.27 examples/s]11036 examples [00:13, 823.16 examples/s]11119 examples [00:13, 812.06 examples/s]11201 examples [00:13, 801.46 examples/s]11287 examples [00:13, 817.05 examples/s]11371 examples [00:13, 823.52 examples/s]11458 examples [00:13, 828.10 examples/s]11546 examples [00:13, 841.41 examples/s]11631 examples [00:13, 842.20 examples/s]11716 examples [00:13, 837.57 examples/s]11813 examples [00:13, 871.50 examples/s]11905 examples [00:14, 884.93 examples/s]11994 examples [00:14, 865.88 examples/s]12083 examples [00:14, 871.74 examples/s]12176 examples [00:14, 886.52 examples/s]12265 examples [00:14, 862.55 examples/s]12352 examples [00:14, 829.63 examples/s]12438 examples [00:14, 836.22 examples/s]12532 examples [00:14, 862.91 examples/s]12623 examples [00:14, 873.61 examples/s]12717 examples [00:15, 889.77 examples/s]12807 examples [00:15, 833.64 examples/s]12901 examples [00:15, 861.67 examples/s]12989 examples [00:15, 844.12 examples/s]13081 examples [00:15, 863.30 examples/s]13172 examples [00:15, 874.66 examples/s]13260 examples [00:15, 873.28 examples/s]13348 examples [00:15, 873.58 examples/s]13436 examples [00:15, 870.85 examples/s]13524 examples [00:15, 820.87 examples/s]13620 examples [00:16, 858.11 examples/s]13707 examples [00:16, 826.77 examples/s]13797 examples [00:16, 846.60 examples/s]13889 examples [00:16, 866.37 examples/s]13980 examples [00:16, 878.52 examples/s]14069 examples [00:16, 851.46 examples/s]14155 examples [00:16, 808.99 examples/s]14240 examples [00:16, 818.60 examples/s]14323 examples [00:16, 801.27 examples/s]14413 examples [00:17, 828.02 examples/s]14497 examples [00:17, 823.45 examples/s]14580 examples [00:17, 810.67 examples/s]14675 examples [00:17, 847.98 examples/s]14773 examples [00:17, 882.84 examples/s]14871 examples [00:17, 908.58 examples/s]14963 examples [00:17, 907.34 examples/s]15055 examples [00:17, 886.42 examples/s]15145 examples [00:17, 888.11 examples/s]15241 examples [00:17, 906.18 examples/s]15332 examples [00:18, 906.46 examples/s]15426 examples [00:18, 914.87 examples/s]15518 examples [00:18, 875.40 examples/s]15608 examples [00:18, 880.71 examples/s]15697 examples [00:18, 882.47 examples/s]15790 examples [00:18, 894.62 examples/s]15884 examples [00:18, 904.00 examples/s]15975 examples [00:18, 856.07 examples/s]16062 examples [00:18, 851.85 examples/s]16148 examples [00:19, 817.14 examples/s]16234 examples [00:19, 827.26 examples/s]16323 examples [00:19, 844.10 examples/s]16408 examples [00:19, 829.75 examples/s]16504 examples [00:19, 863.47 examples/s]16593 examples [00:19, 870.17 examples/s]16681 examples [00:19, 831.95 examples/s]16772 examples [00:19, 852.76 examples/s]16858 examples [00:19, 837.80 examples/s]16946 examples [00:19, 850.00 examples/s]17035 examples [00:20, 861.24 examples/s]17124 examples [00:20, 867.59 examples/s]17214 examples [00:20, 874.83 examples/s]17302 examples [00:20, 825.37 examples/s]17395 examples [00:20, 853.54 examples/s]17483 examples [00:20, 857.76 examples/s]17570 examples [00:20, 843.86 examples/s]17655 examples [00:20, 794.11 examples/s]17747 examples [00:20, 828.00 examples/s]17837 examples [00:21, 845.98 examples/s]17929 examples [00:21, 866.30 examples/s]18023 examples [00:21, 884.91 examples/s]18113 examples [00:21, 827.24 examples/s]18197 examples [00:21, 827.61 examples/s]18281 examples [00:21, 828.41 examples/s]18365 examples [00:21, 826.83 examples/s]18456 examples [00:21, 848.06 examples/s]18542 examples [00:21, 813.95 examples/s]18630 examples [00:21, 831.79 examples/s]18714 examples [00:22, 828.20 examples/s]18798 examples [00:22, 794.13 examples/s]18881 examples [00:22, 803.74 examples/s]18962 examples [00:22, 757.49 examples/s]19045 examples [00:22, 777.33 examples/s]19124 examples [00:22, 775.77 examples/s]19203 examples [00:22, 777.00 examples/s]19290 examples [00:22, 800.75 examples/s]19371 examples [00:22, 792.32 examples/s]19472 examples [00:23, 846.04 examples/s]19562 examples [00:23, 859.41 examples/s]19649 examples [00:23, 843.54 examples/s]19736 examples [00:23, 849.31 examples/s]19822 examples [00:23, 801.04 examples/s]19907 examples [00:23, 813.06 examples/s]20001 examples [00:23, 798.31 examples/s]20100 examples [00:23, 845.97 examples/s]20186 examples [00:23, 848.01 examples/s]20272 examples [00:24, 786.12 examples/s]20355 examples [00:24, 796.56 examples/s]20436 examples [00:24, 779.99 examples/s]20521 examples [00:24, 798.07 examples/s]20614 examples [00:24, 833.19 examples/s]20709 examples [00:24, 863.17 examples/s]20807 examples [00:24, 894.20 examples/s]20903 examples [00:24, 911.41 examples/s]21001 examples [00:24, 928.80 examples/s]21097 examples [00:24, 936.80 examples/s]21192 examples [00:25, 922.65 examples/s]21287 examples [00:25, 930.54 examples/s]21381 examples [00:25, 923.16 examples/s]21474 examples [00:25, 899.57 examples/s]21566 examples [00:25, 873.97 examples/s]21658 examples [00:25, 884.96 examples/s]21749 examples [00:25, 890.16 examples/s]21844 examples [00:25, 905.03 examples/s]21935 examples [00:25, 884.20 examples/s]22024 examples [00:25, 849.32 examples/s]22116 examples [00:26, 868.39 examples/s]22209 examples [00:26, 883.60 examples/s]22298 examples [00:26, 878.55 examples/s]22388 examples [00:26, 884.21 examples/s]22477 examples [00:26, 834.47 examples/s]22566 examples [00:26, 848.34 examples/s]22656 examples [00:26, 860.81 examples/s]22743 examples [00:26, 821.29 examples/s]22826 examples [00:26, 820.56 examples/s]22909 examples [00:27, 801.64 examples/s]22996 examples [00:27, 820.55 examples/s]23088 examples [00:27, 846.35 examples/s]23174 examples [00:27, 847.29 examples/s]23263 examples [00:27, 857.63 examples/s]23350 examples [00:27, 838.32 examples/s]23437 examples [00:27, 846.92 examples/s]23522 examples [00:27, 847.48 examples/s]23608 examples [00:27, 846.32 examples/s]23700 examples [00:27, 867.00 examples/s]23787 examples [00:28, 798.30 examples/s]23873 examples [00:28, 812.18 examples/s]23957 examples [00:28, 818.10 examples/s]24046 examples [00:28, 836.88 examples/s]24131 examples [00:28, 816.49 examples/s]24214 examples [00:28, 726.97 examples/s]24305 examples [00:28, 773.47 examples/s]24399 examples [00:28, 814.61 examples/s]24485 examples [00:28, 827.05 examples/s]24570 examples [00:29, 823.88 examples/s]24654 examples [00:29, 800.12 examples/s]24737 examples [00:29, 807.31 examples/s]24820 examples [00:29, 813.80 examples/s]24903 examples [00:29, 816.41 examples/s]24985 examples [00:29, 773.51 examples/s]25065 examples [00:29, 779.62 examples/s]25155 examples [00:29, 810.96 examples/s]25239 examples [00:29, 819.42 examples/s]25330 examples [00:29, 843.31 examples/s]25415 examples [00:30, 843.43 examples/s]25500 examples [00:30, 816.01 examples/s]25591 examples [00:30, 841.75 examples/s]25681 examples [00:30, 857.34 examples/s]25770 examples [00:30, 864.65 examples/s]25857 examples [00:30, 819.03 examples/s]25940 examples [00:30, 804.74 examples/s]26022 examples [00:30, 806.20 examples/s]26108 examples [00:30, 819.26 examples/s]26191 examples [00:31, 805.33 examples/s]26272 examples [00:31, 750.88 examples/s]26349 examples [00:31, 752.52 examples/s]26425 examples [00:31, 737.56 examples/s]26507 examples [00:31, 758.41 examples/s]26585 examples [00:31, 763.34 examples/s]26665 examples [00:31, 772.98 examples/s]26754 examples [00:31, 803.69 examples/s]26843 examples [00:31, 826.06 examples/s]26932 examples [00:31, 842.95 examples/s]27018 examples [00:32, 846.97 examples/s]27104 examples [00:32, 805.10 examples/s]27189 examples [00:32, 817.83 examples/s]27280 examples [00:32, 842.51 examples/s]27376 examples [00:32, 871.95 examples/s]27464 examples [00:32, 863.71 examples/s]27551 examples [00:32, 833.83 examples/s]27643 examples [00:32, 856.24 examples/s]27730 examples [00:32, 850.46 examples/s]27816 examples [00:33, 833.96 examples/s]27900 examples [00:33, 831.39 examples/s]27984 examples [00:33, 760.50 examples/s]28069 examples [00:33, 784.90 examples/s]28155 examples [00:33, 805.29 examples/s]28237 examples [00:33, 785.93 examples/s]28317 examples [00:33, 785.94 examples/s]28397 examples [00:33, 715.28 examples/s]28485 examples [00:33, 757.74 examples/s]28574 examples [00:33, 790.77 examples/s]28655 examples [00:34, 785.74 examples/s]28746 examples [00:34, 817.11 examples/s]28829 examples [00:34, 813.81 examples/s]28925 examples [00:34, 851.25 examples/s]29017 examples [00:34, 870.29 examples/s]29105 examples [00:34, 812.66 examples/s]29197 examples [00:34, 842.08 examples/s]29283 examples [00:34, 847.32 examples/s]29369 examples [00:34, 821.10 examples/s]29452 examples [00:35, 806.30 examples/s]29536 examples [00:35, 814.70 examples/s]29623 examples [00:35, 828.87 examples/s]29707 examples [00:35, 816.69 examples/s]29789 examples [00:35, 810.35 examples/s]29883 examples [00:35, 843.68 examples/s]29968 examples [00:35, 843.24 examples/s]30053 examples [00:35, 776.88 examples/s]30149 examples [00:35, 822.47 examples/s]30238 examples [00:35, 841.27 examples/s]30335 examples [00:36, 874.94 examples/s]30429 examples [00:36, 892.17 examples/s]30520 examples [00:36, 875.05 examples/s]30609 examples [00:36, 871.98 examples/s]30698 examples [00:36, 876.10 examples/s]30793 examples [00:36, 895.47 examples/s]30886 examples [00:36, 904.69 examples/s]30977 examples [00:36, 870.93 examples/s]31065 examples [00:36, 868.60 examples/s]31153 examples [00:37, 869.36 examples/s]31241 examples [00:37, 855.58 examples/s]31327 examples [00:37, 847.94 examples/s]31412 examples [00:37, 817.12 examples/s]31502 examples [00:37, 838.20 examples/s]31587 examples [00:37, 811.39 examples/s]31673 examples [00:37, 823.66 examples/s]31756 examples [00:37, 781.89 examples/s]31835 examples [00:37, 773.25 examples/s]31918 examples [00:37, 788.51 examples/s]32003 examples [00:38, 804.40 examples/s]32096 examples [00:38, 835.36 examples/s]32189 examples [00:38, 861.57 examples/s]32276 examples [00:38, 821.44 examples/s]32365 examples [00:38, 832.76 examples/s]32458 examples [00:38, 859.00 examples/s]32550 examples [00:38, 874.67 examples/s]32640 examples [00:38, 879.88 examples/s]32729 examples [00:38, 827.83 examples/s]32822 examples [00:39, 855.97 examples/s]32917 examples [00:39, 881.22 examples/s]33006 examples [00:39, 839.66 examples/s]33091 examples [00:39, 777.07 examples/s]33173 examples [00:39, 786.97 examples/s]33256 examples [00:39, 798.46 examples/s]33337 examples [00:39, 785.20 examples/s]33417 examples [00:39, 773.49 examples/s]33505 examples [00:39, 786.23 examples/s]33598 examples [00:39, 824.09 examples/s]33689 examples [00:40, 845.89 examples/s]33783 examples [00:40, 869.97 examples/s]33871 examples [00:40, 856.98 examples/s]33958 examples [00:40, 839.93 examples/s]34055 examples [00:40, 873.62 examples/s]34153 examples [00:40, 898.28 examples/s]34245 examples [00:40, 902.89 examples/s]34336 examples [00:40, 877.31 examples/s]34425 examples [00:40, 862.03 examples/s]34516 examples [00:41, 875.02 examples/s]34604 examples [00:41, 872.73 examples/s]34700 examples [00:41, 895.85 examples/s]34792 examples [00:41, 901.06 examples/s]34883 examples [00:41, 857.07 examples/s]34970 examples [00:41, 851.85 examples/s]35064 examples [00:41, 874.90 examples/s]35152 examples [00:41, 860.86 examples/s]35248 examples [00:41, 888.22 examples/s]35338 examples [00:41, 837.22 examples/s]35432 examples [00:42, 864.20 examples/s]35527 examples [00:42, 886.16 examples/s]35617 examples [00:42, 888.87 examples/s]35707 examples [00:42, 891.59 examples/s]35797 examples [00:42, 883.43 examples/s]35890 examples [00:42, 895.00 examples/s]35986 examples [00:42, 911.05 examples/s]36078 examples [00:42, 911.72 examples/s]36170 examples [00:42, 912.28 examples/s]36262 examples [00:43, 861.71 examples/s]36349 examples [00:43, 863.82 examples/s]36442 examples [00:43, 881.13 examples/s]36531 examples [00:43, 875.91 examples/s]36622 examples [00:43, 884.23 examples/s]36711 examples [00:43, 858.79 examples/s]36798 examples [00:43, 853.40 examples/s]36884 examples [00:43, 847.15 examples/s]36969 examples [00:43, 793.75 examples/s]37061 examples [00:43, 827.72 examples/s]37145 examples [00:44, 812.23 examples/s]37236 examples [00:44, 838.51 examples/s]37332 examples [00:44, 869.34 examples/s]37427 examples [00:44, 891.89 examples/s]37517 examples [00:44, 855.84 examples/s]37604 examples [00:44, 831.21 examples/s]37698 examples [00:44, 860.27 examples/s]37795 examples [00:44, 889.07 examples/s]37885 examples [00:44, 859.72 examples/s]37976 examples [00:45, 874.13 examples/s]38064 examples [00:45, 731.70 examples/s]38160 examples [00:45, 786.18 examples/s]38246 examples [00:45, 804.90 examples/s]38331 examples [00:45, 817.68 examples/s]38415 examples [00:45, 790.20 examples/s]38507 examples [00:45, 824.27 examples/s]38600 examples [00:45, 853.27 examples/s]38689 examples [00:45, 863.49 examples/s]38785 examples [00:45, 888.21 examples/s]38875 examples [00:46, 848.59 examples/s]38961 examples [00:46, 846.91 examples/s]39055 examples [00:46, 871.32 examples/s]39148 examples [00:46, 887.39 examples/s]39241 examples [00:46, 897.66 examples/s]39332 examples [00:46, 862.62 examples/s]39421 examples [00:46, 870.17 examples/s]39509 examples [00:46, 870.56 examples/s]39597 examples [00:46, 863.80 examples/s]39684 examples [00:47, 862.41 examples/s]39771 examples [00:47, 829.79 examples/s]39863 examples [00:47, 853.84 examples/s]39949 examples [00:47, 845.77 examples/s]40034 examples [00:47, 809.54 examples/s]40128 examples [00:47, 844.54 examples/s]40214 examples [00:47, 835.78 examples/s]40305 examples [00:47, 856.20 examples/s]40395 examples [00:47, 867.54 examples/s]40485 examples [00:47, 876.93 examples/s]40574 examples [00:48, 878.66 examples/s]40663 examples [00:48, 835.85 examples/s]40756 examples [00:48, 860.83 examples/s]40843 examples [00:48, 852.17 examples/s]40936 examples [00:48, 872.53 examples/s]41024 examples [00:48, 847.93 examples/s]41110 examples [00:48, 829.63 examples/s]41200 examples [00:48, 847.22 examples/s]41295 examples [00:48, 874.91 examples/s]41395 examples [00:49, 908.81 examples/s]41493 examples [00:49, 926.88 examples/s]41587 examples [00:49, 876.58 examples/s]41676 examples [00:49, 864.65 examples/s]41764 examples [00:49, 827.18 examples/s]41848 examples [00:49, 822.01 examples/s]41931 examples [00:49, 803.99 examples/s]42020 examples [00:49, 827.24 examples/s]42104 examples [00:49, 830.14 examples/s]42188 examples [00:49, 816.82 examples/s]42270 examples [00:50, 790.69 examples/s]42350 examples [00:50, 768.91 examples/s]42446 examples [00:50, 817.49 examples/s]42541 examples [00:50, 852.84 examples/s]42635 examples [00:50, 877.08 examples/s]42732 examples [00:50, 901.83 examples/s]42824 examples [00:50, 884.26 examples/s]42917 examples [00:50, 897.44 examples/s]43008 examples [00:50, 883.22 examples/s]43102 examples [00:51, 897.57 examples/s]43194 examples [00:51, 903.87 examples/s]43285 examples [00:51, 868.26 examples/s]43376 examples [00:51, 878.27 examples/s]43466 examples [00:51, 883.33 examples/s]43561 examples [00:51, 900.06 examples/s]43656 examples [00:51, 911.60 examples/s]43748 examples [00:51, 882.71 examples/s]43837 examples [00:51, 846.95 examples/s]43925 examples [00:51, 854.12 examples/s]44015 examples [00:52, 866.61 examples/s]44107 examples [00:52, 879.50 examples/s]44196 examples [00:52, 837.00 examples/s]44281 examples [00:52, 837.73 examples/s]44366 examples [00:52, 828.20 examples/s]44458 examples [00:52, 852.02 examples/s]44544 examples [00:52, 839.87 examples/s]44629 examples [00:52, 797.11 examples/s]44717 examples [00:52, 818.92 examples/s]44801 examples [00:53, 810.85 examples/s]44892 examples [00:53, 837.07 examples/s]44985 examples [00:53, 861.56 examples/s]45073 examples [00:53, 865.07 examples/s]45166 examples [00:53, 882.67 examples/s]45261 examples [00:53, 899.84 examples/s]45352 examples [00:53, 896.11 examples/s]45447 examples [00:53, 911.22 examples/s]45542 examples [00:53, 921.06 examples/s]45635 examples [00:53, 920.26 examples/s]45730 examples [00:54, 928.78 examples/s]45830 examples [00:54, 946.73 examples/s]45925 examples [00:54, 929.26 examples/s]46019 examples [00:54, 808.42 examples/s]46115 examples [00:54, 848.16 examples/s]46211 examples [00:54, 877.80 examples/s]46303 examples [00:54, 888.77 examples/s]46394 examples [00:54, 872.57 examples/s]46487 examples [00:54, 886.13 examples/s]46577 examples [00:55, 885.27 examples/s]46667 examples [00:55, 886.04 examples/s]46757 examples [00:55, 872.22 examples/s]46845 examples [00:55, 854.45 examples/s]46940 examples [00:55, 879.29 examples/s]47029 examples [00:55, 855.80 examples/s]47116 examples [00:55, 857.45 examples/s]47203 examples [00:55, 855.48 examples/s]47289 examples [00:55, 851.01 examples/s]47375 examples [00:55, 820.81 examples/s]47463 examples [00:56, 835.20 examples/s]47547 examples [00:56, 820.94 examples/s]47641 examples [00:56, 852.97 examples/s]47727 examples [00:56, 794.51 examples/s]47815 examples [00:56, 816.32 examples/s]47903 examples [00:56, 832.43 examples/s]47998 examples [00:56, 862.16 examples/s]48087 examples [00:56, 868.94 examples/s]48175 examples [00:56, 855.96 examples/s]48272 examples [00:57, 884.96 examples/s]48365 examples [00:57, 896.70 examples/s]48456 examples [00:57, 900.53 examples/s]48551 examples [00:57, 912.66 examples/s]48643 examples [00:57, 876.58 examples/s]48742 examples [00:57, 905.07 examples/s]48839 examples [00:57, 922.73 examples/s]48935 examples [00:57, 933.15 examples/s]49033 examples [00:57, 944.15 examples/s]49128 examples [00:57, 836.70 examples/s]49216 examples [00:58, 849.19 examples/s]49310 examples [00:58, 865.62 examples/s]49405 examples [00:58, 887.14 examples/s]49497 examples [00:58, 896.02 examples/s]49588 examples [00:58, 874.84 examples/s]49677 examples [00:58, 858.07 examples/s]49774 examples [00:58, 886.68 examples/s]49872 examples [00:58, 912.40 examples/s]49966 examples [00:58, 918.33 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s]  4%|▎         | 1857/50000 [00:00<00:02, 18569.22 examples/s] 22%|██▏       | 10931/50000 [00:00<00:01, 24388.49 examples/s] 41%|████      | 20271/50000 [00:00<00:00, 31334.01 examples/s] 56%|█████▋    | 28129/50000 [00:00<00:00, 38229.23 examples/s] 71%|███████   | 35519/50000 [00:00<00:00, 44702.27 examples/s] 92%|█████████▏| 45931/50000 [00:00<00:00, 53935.35 examples/s]                                                               0 examples [00:00, ? examples/s]78 examples [00:00, 775.01 examples/s]179 examples [00:00, 831.59 examples/s]281 examples [00:00, 878.76 examples/s]372 examples [00:00, 887.38 examples/s]470 examples [00:00, 912.83 examples/s]571 examples [00:00, 939.52 examples/s]672 examples [00:00, 959.29 examples/s]763 examples [00:00, 650.79 examples/s]857 examples [00:01, 716.65 examples/s]949 examples [00:01, 766.13 examples/s]1041 examples [00:01, 805.11 examples/s]1141 examples [00:01, 852.75 examples/s]1231 examples [00:01, 846.86 examples/s]1324 examples [00:01, 868.89 examples/s]1416 examples [00:01, 883.23 examples/s]1513 examples [00:01, 905.83 examples/s]1605 examples [00:01, 907.99 examples/s]1697 examples [00:01, 887.70 examples/s]1793 examples [00:02, 907.28 examples/s]1892 examples [00:02, 927.95 examples/s]1990 examples [00:02, 941.73 examples/s]2087 examples [00:02, 948.88 examples/s]2183 examples [00:02, 907.61 examples/s]2276 examples [00:02, 913.08 examples/s]2368 examples [00:02, 913.83 examples/s]2466 examples [00:02, 931.65 examples/s]2562 examples [00:02, 938.26 examples/s]2657 examples [00:03, 866.37 examples/s]2752 examples [00:03, 889.61 examples/s]2851 examples [00:03, 915.60 examples/s]2949 examples [00:03, 932.28 examples/s]3044 examples [00:03, 935.93 examples/s]3139 examples [00:03, 905.81 examples/s]3231 examples [00:03, 896.29 examples/s]3328 examples [00:03, 915.99 examples/s]3425 examples [00:03, 931.38 examples/s]3520 examples [00:03, 924.56 examples/s]3613 examples [00:04, 869.38 examples/s]3712 examples [00:04, 901.56 examples/s]3809 examples [00:04, 920.00 examples/s]3906 examples [00:04, 932.43 examples/s]4000 examples [00:04, 888.89 examples/s]4092 examples [00:04, 895.73 examples/s]4183 examples [00:04, 889.68 examples/s]4276 examples [00:04, 901.11 examples/s]4367 examples [00:04, 903.64 examples/s]4458 examples [00:04, 879.56 examples/s]4552 examples [00:05, 895.22 examples/s]4647 examples [00:05, 910.18 examples/s]4745 examples [00:05, 927.53 examples/s]4839 examples [00:05, 923.99 examples/s]4932 examples [00:05, 892.27 examples/s]5027 examples [00:05, 908.36 examples/s]5119 examples [00:05, 881.51 examples/s]5214 examples [00:05, 900.38 examples/s]5312 examples [00:05, 919.89 examples/s]5405 examples [00:06, 874.00 examples/s]5498 examples [00:06, 889.44 examples/s]5590 examples [00:06, 898.37 examples/s]5686 examples [00:06, 915.04 examples/s]5778 examples [00:06, 910.21 examples/s]5870 examples [00:06, 869.15 examples/s]5959 examples [00:06, 875.27 examples/s]6054 examples [00:06, 895.47 examples/s]6149 examples [00:06, 909.86 examples/s]6248 examples [00:06, 930.94 examples/s]6342 examples [00:07, 906.84 examples/s]6438 examples [00:07, 920.51 examples/s]6536 examples [00:07, 935.12 examples/s]6630 examples [00:07, 930.26 examples/s]6728 examples [00:07, 942.40 examples/s]6823 examples [00:07, 881.45 examples/s]6915 examples [00:07, 891.79 examples/s]7013 examples [00:07, 915.94 examples/s]7108 examples [00:07, 924.30 examples/s]7201 examples [00:08, 912.03 examples/s]7293 examples [00:08, 880.10 examples/s]7390 examples [00:08, 903.10 examples/s]7485 examples [00:08, 915.62 examples/s]7583 examples [00:08, 931.95 examples/s]7677 examples [00:08, 903.28 examples/s]7768 examples [00:08, 857.69 examples/s]7855 examples [00:08, 818.51 examples/s]7954 examples [00:08, 862.04 examples/s]8053 examples [00:08, 896.27 examples/s]8146 examples [00:09, 903.85 examples/s]8238 examples [00:09, 867.53 examples/s]8335 examples [00:09, 894.72 examples/s]8426 examples [00:09, 887.14 examples/s]8519 examples [00:09, 898.01 examples/s]8614 examples [00:09, 911.10 examples/s]8706 examples [00:09, 834.48 examples/s]8791 examples [00:09, 827.60 examples/s]8879 examples [00:09, 842.53 examples/s]8975 examples [00:10, 873.93 examples/s]9064 examples [00:10, 826.35 examples/s]9155 examples [00:10, 847.83 examples/s]9249 examples [00:10, 871.10 examples/s]9337 examples [00:10, 866.03 examples/s]9426 examples [00:10, 871.62 examples/s]9514 examples [00:10, 828.38 examples/s]9609 examples [00:10, 859.59 examples/s]9704 examples [00:10, 883.71 examples/s]9801 examples [00:10, 906.63 examples/s]9898 examples [00:11, 923.26 examples/s]9991 examples [00:11, 872.49 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 75%|███████▌  | 7548/10000 [00:00<00:00, 75476.11 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRCV8LY/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteRCV8LY/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa2d6344950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa2d6344950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa2d6344950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2705b6a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2d3572780>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fa2d6344950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fa2d6344950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fa2d6344950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2bc704128>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fa2bc704080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fa24dd5e2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fa24dd5e2f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fa24dd5e2f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fa24dd5e400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fa24dd5e400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fa24dd5e400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=fd2376d30f3a7cd3abbe652e16f5f362f731a2a10f800620b37cb713b7ed5cef
  Stored in directory: /tmp/pip-ephem-wheel-cache-xhi342yd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:06:45, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:42:44, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:03:09, 21.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:44:41, 30.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:24:26, 44.1kB/s].vector_cache/glove.6B.zip:   1%|          | 8.09M/862M [00:01<3:46:01, 63.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:37:33, 89.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:01<1:49:42, 128kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:01<1:16:32, 183kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.2M/862M [00:01<53:21, 261kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.7M/862M [00:01<37:15, 372kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.4M/862M [00:02<26:04, 529kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:02<18:15, 751kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.2M/862M [00:02<12:45, 1.07MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:02<09:01, 1.50MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<06:34, 2.06MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<06:30, 2.07MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<08:15, 1.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<06:34, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:04<05:07, 2.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<06:04, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<05:54, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:32, 2.94MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<05:53, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<06:58, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:34, 2.38MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:08<04:02, 3.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:10<19:33, 677kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<15:04, 877kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:10<10:52, 1.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<10:38, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<08:50, 1.49MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:12<06:30, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<07:37, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<09:28, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:34, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<06:42, 1.95MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:14<06:03, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:15<05:35, 2.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:15<05:14, 2.49MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<04:51, 2.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:15<04:41, 2.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.8M/862M [00:15<04:34, 2.85MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<04:31, 2.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:15<04:20, 3.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<1:09:47, 187kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<50:36, 257kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:16<36:38, 355kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<26:51, 484kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:16<20:01, 649kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<15:13, 854kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:17<11:49, 1.10MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<09:29, 1.37MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:17<07:48, 1.66MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:17<06:38, 1.95MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<13:10, 983kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<12:16, 1.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<09:54, 1.31MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<08:01, 1.61MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:18<06:58, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:18<05:52, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.8M/862M [00:19<05:08, 2.51MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:19<04:56, 2.61MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:19<04:35, 2.81MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.8M/862M [00:19<04:43, 2.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.2M/862M [00:19<04:27, 2.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<09:33, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<09:52, 1.31MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<08:14, 1.56MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:55, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:20<05:58, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:20<05:18, 2.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:21<04:50, 2.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<04:39, 2.75MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:21<04:23, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:21<04:10, 3.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<11:07, 1.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<10:48, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<08:36, 1.49MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<07:07, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:57, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<05:43, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.1M/862M [00:22<04:49, 2.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:23<04:56, 2.59MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:23<04:15, 3.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:23<04:36, 2.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.5M/862M [00:23<04:16, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<10:50, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<10:26, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<08:25, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<06:48, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<05:38, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.2M/862M [00:24<04:47, 2.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<04:26, 2.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<03:51, 3.30MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<03:53, 3.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<03:29, 3.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<19:38, 646kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<19:09, 662kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<14:58, 847kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:27, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<08:59, 1.41MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:15, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:54, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<05:05, 2.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:30, 2.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:14, 2.98MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<23:13, 543kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<18:40, 675kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:07, 893kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:44, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<08:22, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<06:41, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<05:30, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:34, 2.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<04:04, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<46:49, 268kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<36:56, 340kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<27:01, 464kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<19:41, 636kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<14:29, 863kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<10:45, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<08:12, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<06:28, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<7:50:44, 26.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<5:33:05, 37.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<3:53:58, 53.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<2:44:10, 75.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<1:55:27, 108kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<1:21:14, 153kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<57:22, 216kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:24:38, 147kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:05:20, 190kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<47:17, 262kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<33:33, 369kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<24:02, 515kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<17:13, 717kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<17:36, 701kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<17:32, 703kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:31, 912kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<09:57, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<07:23, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<05:34, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<20:04, 612kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<18:23, 667kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:56, 880kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<10:09, 1.20MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<07:28, 1.64MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<10:16, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:56, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:34, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:20, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:42, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<03:45, 3.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<33:32, 362kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<30:57, 392kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<23:16, 521kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:50, 719kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<12:23, 977kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<08:56, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<11:53, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<12:02, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:12, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:43, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<05:12, 2.31MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<03:54, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<29:56, 401kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<24:25, 491kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<18:00, 666kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<12:55, 926kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<09:22, 1.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<12:24, 961kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<17:06, 697kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<13:31, 881kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:56, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:22, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<05:27, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<04:17, 2.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<1:13:38, 161kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<55:14, 215kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<39:36, 299kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<28:01, 422kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<19:52, 594kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<24:29, 481kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<23:39, 498kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<18:10, 649kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<13:12, 890kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<09:33, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<06:58, 1.68MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<56:20, 208kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<42:37, 275kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<30:39, 382kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<21:45, 537kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<15:30, 752kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<16:41, 698kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<18:02, 646kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<14:12, 819kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<10:20, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<07:31, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<09:21, 1.24MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<12:56, 895kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<10:38, 1.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:54, 1.46MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:49, 1.98MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<08:13, 1.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:44, 1.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:51, 1.68MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:06, 2.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:48, 3.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<33:11, 345kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<28:52, 396kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<21:44, 526kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<15:29, 737kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<11:14, 1.02MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:52, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<12:41, 896kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<10:01, 1.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<07:21, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:28, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:04, 2.78MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<19:12, 588kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<18:31, 610kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<14:02, 804kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<10:10, 1.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<07:24, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:43, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<10:47, 1.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:45, 1.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<06:24, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:42, 2.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<09:35, 1.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<11:42, 954kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<09:20, 1.19MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:50, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:07, 1.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:12, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:21, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:29, 2.02MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:02, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:40, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<09:20, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:29, 1.47MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:13<05:31, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:24, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<08:04, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:26, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<04:46, 2.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:35, 3.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:51, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:15, 1.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:02, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<05:50, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:10, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:38, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:50, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:38, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:35, 1.42MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:59, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:26, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:25, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<06:34, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:16, 2.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:51, 2.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:19, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:44, 1.37MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:03, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<04:26, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:01, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:33, 1.61MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:06, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:54, 2.69MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:54, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:40, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:31, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<03:20, 3.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:54, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:03, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:48, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<04:53, 2.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:27, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:57, 1.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:04, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<04:22, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<25:45, 399kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<19:57, 514kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<14:27, 709kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<11:46, 866kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:25, 978kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:55, 1.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<05:38, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<20:00, 506kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<16:16, 623kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<11:55, 848kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<08:26, 1.19MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<15:08, 664kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<12:58, 775kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:36, 1.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<06:50, 1.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<08:49, 1.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<08:20, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:19, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:46, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:54, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<06:07, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<04:25, 2.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:58, 989kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<08:26, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:15, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:10, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:14, 1.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<04:50, 2.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:58, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:23, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<04:15, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:04, 3.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<10:40, 904kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:22, 1.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:58, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:58, 1.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<09:53, 968kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<08:48, 1.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:37, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:10, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:12, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:43, 2.01MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<03:28, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<05:29, 1.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:42, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:26, 2.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<03:16, 2.87MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:20, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<05:35, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:21, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:34, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:01, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:49, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:50, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:10, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:48, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:48, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:44, 3.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<54:48, 166kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<40:10, 226kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<28:31, 318kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<19:57, 453kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<28:09, 321kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<21:29, 420kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<15:29, 582kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<10:53, 824kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<29:14, 306kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<22:13, 403kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<15:55, 561kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<11:12, 794kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<14:43, 604kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<12:05, 735kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<08:53, 998kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:38, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:11, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:27, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:13, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:24, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:12, 2.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:21, 1.99MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:45, 1.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:45, 2.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<02:43, 3.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:56, 1.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<06:33, 1.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:00, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<03:35, 2.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<08:44, 978kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<07:44, 1.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:50, 1.46MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<05:27, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:30, 1.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:16, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:20, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:42, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:42, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:57, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:24, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:29, 2.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<02:31, 3.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<31:53, 260kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<23:56, 346kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<17:08, 482kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<13:16, 618kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<10:54, 752kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<08:01, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:55, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:57, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<04:26, 1.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:38, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:47, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:43, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:56, 2.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<04:12, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<03:15, 2.45MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<02:21, 3.37MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<15:05, 526kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<11:57, 663kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<08:42, 909kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<07:23, 1.06MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<06:11, 1.27MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:32, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:42, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<06:02, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:48, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:29, 2.22MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<04:30, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<04:27, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:27, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:42, 2.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<03:56, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:02, 2.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:12, 3.45MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<20:14, 375kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<15:29, 489kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<11:09, 678kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<09:03, 831kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<07:39, 982kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:40, 1.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:13, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:36, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:51, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:59, 1.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:06, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:38, 2.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:51, 2.54MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:14, 2.24MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:32, 2.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:47, 2.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:10, 2.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:28, 2.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:44, 2.61MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:07, 2.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:27, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<02:40, 2.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:01, 3.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:26, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:38, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:51, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:10, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:24, 2.04MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:40, 2.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:03, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:21, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:36, 2.64MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<01:53, 3.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<10:06, 675kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:13<08:17, 823kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<06:56, 983kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:00, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:51, 1.76MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:57, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:33, 2.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<09:29, 713kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<10:20, 654kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<08:13, 821kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:05, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<04:34, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:31, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:46, 2.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<07:21, 910kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<08:48, 760kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<06:54, 967kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:13, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:58, 1.68MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:01, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:29, 2.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<01:59, 3.32MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<09:50, 673kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<10:11, 650kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<07:56, 833kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:52, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:23, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:21, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:38, 2.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<15:46, 416kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<14:17, 459kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<10:45, 608kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<07:47, 839kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<05:48, 1.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:17, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:20, 1.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:34, 2.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<21:52, 296kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<18:12, 356kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<13:29, 480kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<09:41, 667kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<07:00, 920kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:13, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:59, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:04, 2.09MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<36:18, 177kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<28:16, 227kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<20:26, 314kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<14:38, 437kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<10:29, 609kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<07:41, 829kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:39, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<04:10, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<25:50, 246kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<21:13, 299kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<15:34, 407kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<11:11, 566kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:04, 782kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:54, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<04:23, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<36:45, 171kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<28:31, 220kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<20:36, 304kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<14:45, 424kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<10:31, 593kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:44, 805kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:36, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:18, 1.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<29:06, 213kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<23:09, 268kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<16:56, 366kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<12:06, 511kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<08:42, 709kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<06:23, 964kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:42, 1.31MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<07:35, 809kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<08:20, 737kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<06:32, 937kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:52, 1.26MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<03:39, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:48, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:27, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:07, 992kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:59, 1.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:46, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:53, 2.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:15, 2.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:10, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<05:24, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:25, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:19, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<02:30, 2.38MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<02:04, 2.87MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<01:39, 3.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<06:14, 950kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<06:48, 871kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<05:22, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<04:00, 1.48MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:59, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:17, 2.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<06:05, 963kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<06:29, 904kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:01, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:41, 1.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:51, 2.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<02:19, 2.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<01:48, 3.20MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<10:18, 562kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<09:26, 614kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<07:07, 811kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<05:10, 1.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<03:52, 1.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:51, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:49, 984kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<06:07, 935kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:45, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:30, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:37, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<04:18, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:29, 871kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<05:24, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:01, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:57, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<03:47, 1.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:16, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:24, 1.64MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:30, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:49<01:52, 2.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:57, 1.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<06:29, 851kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<05:15, 1.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:51, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:48, 1.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:30, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<04:30, 1.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:28, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:32, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<03:20, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<04:45, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:55, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:55, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:08, 2.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<05:21, 990kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<05:56, 895kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:41, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:25, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:27, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<10:04, 521kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<09:03, 579kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<06:49, 767kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<04:53, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<04:32, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<05:00, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:53, 1.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:50, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:10, 1.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:47, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<02:59, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:11, 2.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:37, 3.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<05:50, 863kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<05:32, 908kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:14, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<03:03, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:40, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:55, 1.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<03:04, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:13, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<03:12, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<03:34, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:47, 1.75MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:01, 2.39MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:30, 3.20MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<11:29, 420kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<09:13, 523kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<06:44, 714kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<04:46, 1.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<05:13, 910kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:07, 927kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:56, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:50, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<03:28, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:31, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:41, 1.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:56, 2.39MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:03, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:02, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:21, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:42, 2.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<06:57, 654kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<05:59, 760kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:26, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:09, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:50, 1.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:47, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:52, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:05, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:42, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:40, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:03, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:09, 2.01MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:31, 1.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<02:01, 2.13MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:28, 2.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<04:07, 1.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:32, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:38, 1.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:34, 1.63MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:42, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:04, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:31, 2.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:16, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:28, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:56, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:23, 2.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<29:57, 136kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<21:50, 186kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<15:25, 263kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<10:49, 372kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<08:33, 468kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<06:50, 584kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<04:58, 799kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<03:29, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<30:24, 129kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<21:48, 180kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<15:18, 255kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<10:39, 363kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<15:23, 251kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<11:28, 336kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<08:10, 470kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<06:17, 603kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<05:05, 745kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<03:42, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<02:37, 1.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:35, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<03:12, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<02:24, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<01:42, 2.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<04:14, 864kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<03:37, 1.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:40, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:54, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:24, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<03:02, 1.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:16, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:10, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<02:09, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:39, 2.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<01:13, 2.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:45, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:51, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:27, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:34, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:44, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:20, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:00, 3.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:37, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:45, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:22, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:39, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:17, 2.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<00:56, 3.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:38, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:28, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:52, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:48, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:51, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:25, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:00<01:00, 3.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:58, 765kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<03:19, 912kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<02:27, 1.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:44, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:45, 1.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:28, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:50, 1.60MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:18, 2.23MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<04:01, 721kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<03:20, 867kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:27, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:10, 1.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:01, 1.39MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:31, 1.84MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:06, 2.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:58, 1.41MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:52, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:24, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:01, 2.66MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:58, 1.37MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:52, 1.44MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:24, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:00, 2.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<02:17, 1.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:05, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:33, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:08, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:32, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:32, 1.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:11, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:14, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:19, 1.88MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:02, 2.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:06, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:13, 1.97MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<00:57, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:41, 3.45MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<02:49, 834kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<02:23, 978kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:46, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:36, 1.43MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:31, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:09, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:51, 2.62MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:16, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:17, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:59, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:44, 2.95MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:04, 1.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:09, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:54, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:57, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:56, 2.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:43, 2.82MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:32, 3.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:19, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:11, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:52, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:59, 1.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:01, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:47, 2.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:51, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:55, 2.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:43, 2.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:48, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:52, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:40, 2.66MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:37<00:29, 3.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<01:26, 1.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:17, 1.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:57, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<00:40, 2.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<02:45, 602kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:12, 749kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:35, 1.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<01:06, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:47, 886kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:34, 1.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:09, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:02, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:59, 1.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:44, 2.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:45<00:31, 2.78MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<03:18, 440kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:33, 568kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:49, 784kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:28, 939kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:15, 1.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:56, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:39, 2.03MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:36, 816kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:21, 967kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:59, 1.31MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:41, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:13, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:04, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:47, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:44, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:43, 1.64MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:32, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:33, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:34, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:26, 2.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:18, 3.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:02, 1.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:54, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:40, 1.52MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:35, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:19, 2.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:36, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:32, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:23, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:25, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:26, 1.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:19, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:14, 3.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:30, 1.51MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:29, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:22, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:21, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:22, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:17, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:17, 2.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:15, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:16, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:12, 2.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:12, 2.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:14, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:10, 2.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:11, 2.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:12, 1.95MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:09, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:10, 2.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:07, 2.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:07, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:08, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:06, 2.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:06, 2.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:04, 2.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:02, 3.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.93MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:04, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:01, 3.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.53MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.91MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.87MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<31:10:49,  3.56it/s]  0%|          | 655/400000 [00:00<21:47:44,  5.09it/s]  0%|          | 1419/400000 [00:00<15:13:55,  7.27it/s]  1%|          | 2156/400000 [00:00<10:38:50, 10.38it/s]  1%|          | 2865/400000 [00:00<7:26:40, 14.82it/s]   1%|          | 3650/400000 [00:00<5:12:18, 21.15it/s]  1%|          | 4309/400000 [00:00<3:38:32, 30.18it/s]  1%|▏         | 5119/400000 [00:00<2:32:54, 43.04it/s]  1%|▏         | 5932/400000 [00:01<1:47:03, 61.35it/s]  2%|▏         | 6727/400000 [00:01<1:15:02, 87.35it/s]  2%|▏         | 7548/400000 [00:01<52:39, 124.21it/s]   2%|▏         | 8315/400000 [00:01<37:04, 176.08it/s]  2%|▏         | 9115/400000 [00:01<26:08, 249.19it/s]  2%|▏         | 9908/400000 [00:01<18:30, 351.25it/s]  3%|▎         | 10716/400000 [00:01<13:10, 492.59it/s]  3%|▎         | 11527/400000 [00:01<09:26, 685.85it/s]  3%|▎         | 12317/400000 [00:01<06:50, 943.51it/s]  3%|▎         | 13150/400000 [00:01<05:00, 1285.41it/s]  3%|▎         | 13991/400000 [00:02<03:43, 1723.30it/s]  4%|▎         | 14802/400000 [00:02<02:51, 2250.30it/s]  4%|▍         | 15605/400000 [00:02<02:14, 2849.64it/s]  4%|▍         | 16393/400000 [00:02<01:50, 3483.52it/s]  4%|▍         | 17177/400000 [00:02<01:31, 4179.69it/s]  4%|▍         | 17951/400000 [00:02<01:19, 4794.88it/s]  5%|▍         | 18714/400000 [00:02<01:10, 5395.76it/s]  5%|▍         | 19477/400000 [00:02<01:04, 5914.72it/s]  5%|▌         | 20239/400000 [00:02<01:01, 6221.41it/s]  5%|▌         | 21002/400000 [00:03<00:57, 6584.60it/s]  5%|▌         | 21810/400000 [00:03<00:54, 6970.25it/s]  6%|▌         | 22610/400000 [00:03<00:52, 7248.49it/s]  6%|▌         | 23404/400000 [00:03<00:50, 7440.67it/s]  6%|▌         | 24186/400000 [00:03<00:51, 7367.44it/s]  6%|▌         | 24989/400000 [00:03<00:49, 7551.42it/s]  6%|▋         | 25783/400000 [00:03<00:48, 7662.88it/s]  7%|▋         | 26590/400000 [00:03<00:48, 7778.31it/s]  7%|▋         | 27392/400000 [00:03<00:47, 7848.00it/s]  7%|▋         | 28185/400000 [00:03<00:48, 7588.24it/s]  7%|▋         | 28994/400000 [00:04<00:47, 7731.99it/s]  7%|▋         | 29773/400000 [00:04<00:47, 7737.84it/s]  8%|▊         | 30551/400000 [00:04<00:48, 7688.33it/s]  8%|▊         | 31323/400000 [00:04<00:49, 7393.68it/s]  8%|▊         | 32067/400000 [00:04<00:51, 7145.42it/s]  8%|▊         | 32867/400000 [00:04<00:49, 7380.07it/s]  8%|▊         | 33611/400000 [00:04<00:49, 7371.87it/s]  9%|▊         | 34355/400000 [00:04<00:49, 7391.16it/s]  9%|▉         | 35166/400000 [00:04<00:48, 7593.06it/s]  9%|▉         | 35929/400000 [00:04<00:48, 7507.42it/s]  9%|▉         | 36746/400000 [00:05<00:47, 7692.31it/s]  9%|▉         | 37549/400000 [00:05<00:46, 7788.79it/s] 10%|▉         | 38331/400000 [00:05<00:46, 7759.94it/s] 10%|▉         | 39109/400000 [00:05<00:46, 7732.55it/s] 10%|▉         | 39884/400000 [00:05<00:49, 7250.84it/s] 10%|█         | 40660/400000 [00:05<00:48, 7395.50it/s] 10%|█         | 41416/400000 [00:05<00:48, 7442.55it/s] 11%|█         | 42213/400000 [00:05<00:47, 7591.72it/s] 11%|█         | 42976/400000 [00:05<00:48, 7425.49it/s] 11%|█         | 43722/400000 [00:06<00:48, 7333.53it/s] 11%|█         | 44497/400000 [00:06<00:47, 7452.36it/s] 11%|█▏        | 45267/400000 [00:06<00:47, 7524.03it/s] 12%|█▏        | 46108/400000 [00:06<00:45, 7767.36it/s] 12%|█▏        | 46923/400000 [00:06<00:44, 7875.96it/s] 12%|█▏        | 47770/400000 [00:06<00:43, 8043.17it/s] 12%|█▏        | 48578/400000 [00:06<00:43, 8042.73it/s] 12%|█▏        | 49385/400000 [00:06<00:44, 7959.88it/s] 13%|█▎        | 50183/400000 [00:06<00:43, 7963.90it/s] 13%|█▎        | 51011/400000 [00:06<00:43, 8055.68it/s] 13%|█▎        | 51818/400000 [00:07<00:43, 7990.51it/s] 13%|█▎        | 52660/400000 [00:07<00:42, 8114.22it/s] 13%|█▎        | 53473/400000 [00:07<00:43, 7957.52it/s] 14%|█▎        | 54283/400000 [00:07<00:43, 7999.34it/s] 14%|█▍        | 55094/400000 [00:07<00:42, 8030.33it/s] 14%|█▍        | 55898/400000 [00:07<00:43, 7928.31it/s] 14%|█▍        | 56695/400000 [00:07<00:43, 7939.99it/s] 14%|█▍        | 57490/400000 [00:07<00:43, 7933.01it/s] 15%|█▍        | 58284/400000 [00:07<00:43, 7869.44it/s] 15%|█▍        | 59072/400000 [00:07<00:44, 7677.83it/s] 15%|█▍        | 59842/400000 [00:08<00:47, 7224.74it/s] 15%|█▌        | 60613/400000 [00:08<00:46, 7362.27it/s] 15%|█▌        | 61381/400000 [00:08<00:45, 7452.71it/s] 16%|█▌        | 62203/400000 [00:08<00:44, 7665.22it/s] 16%|█▌        | 62997/400000 [00:08<00:43, 7743.57it/s] 16%|█▌        | 63775/400000 [00:08<00:45, 7361.14it/s] 16%|█▌        | 64559/400000 [00:08<00:44, 7498.21it/s] 16%|█▋        | 65314/400000 [00:08<00:44, 7496.48it/s] 17%|█▋        | 66068/400000 [00:08<00:44, 7470.24it/s] 17%|█▋        | 66864/400000 [00:08<00:43, 7605.16it/s] 17%|█▋        | 67627/400000 [00:09<00:44, 7493.10it/s] 17%|█▋        | 68421/400000 [00:09<00:43, 7618.21it/s] 17%|█▋        | 69203/400000 [00:09<00:43, 7676.60it/s] 18%|█▊        | 70009/400000 [00:09<00:42, 7773.48it/s] 18%|█▊        | 70816/400000 [00:09<00:41, 7858.45it/s] 18%|█▊        | 71603/400000 [00:09<00:41, 7849.59it/s] 18%|█▊        | 72400/400000 [00:09<00:41, 7883.71it/s] 18%|█▊        | 73189/400000 [00:09<00:41, 7873.71it/s] 18%|█▊        | 73992/400000 [00:09<00:41, 7917.49it/s] 19%|█▊        | 74785/400000 [00:10<00:41, 7921.23it/s] 19%|█▉        | 75578/400000 [00:10<00:42, 7595.43it/s] 19%|█▉        | 76381/400000 [00:10<00:41, 7718.22it/s] 19%|█▉        | 77162/400000 [00:10<00:41, 7744.97it/s] 19%|█▉        | 77939/400000 [00:10<00:41, 7725.38it/s] 20%|█▉        | 78721/400000 [00:10<00:41, 7752.89it/s] 20%|█▉        | 79498/400000 [00:10<00:42, 7587.21it/s] 20%|██        | 80309/400000 [00:10<00:41, 7735.85it/s] 20%|██        | 81085/400000 [00:10<00:41, 7658.62it/s] 20%|██        | 81853/400000 [00:10<00:42, 7461.46it/s] 21%|██        | 82666/400000 [00:11<00:41, 7649.68it/s] 21%|██        | 83434/400000 [00:11<00:42, 7427.69it/s] 21%|██        | 84221/400000 [00:11<00:41, 7553.34it/s] 21%|██        | 84981/400000 [00:11<00:41, 7564.58it/s] 21%|██▏       | 85741/400000 [00:11<00:41, 7573.17it/s] 22%|██▏       | 86500/400000 [00:11<00:41, 7571.03it/s] 22%|██▏       | 87259/400000 [00:11<00:42, 7420.43it/s] 22%|██▏       | 88094/400000 [00:11<00:40, 7674.57it/s] 22%|██▏       | 88902/400000 [00:11<00:39, 7789.20it/s] 22%|██▏       | 89745/400000 [00:11<00:38, 7969.46it/s] 23%|██▎       | 90566/400000 [00:12<00:38, 8038.37it/s] 23%|██▎       | 91405/400000 [00:12<00:37, 8139.15it/s] 23%|██▎       | 92244/400000 [00:12<00:37, 8211.88it/s] 23%|██▎       | 93067/400000 [00:12<00:37, 8174.06it/s] 23%|██▎       | 93890/400000 [00:12<00:37, 8189.76it/s] 24%|██▎       | 94711/400000 [00:12<00:37, 8193.09it/s] 24%|██▍       | 95554/400000 [00:12<00:36, 8262.03it/s] 24%|██▍       | 96400/400000 [00:12<00:36, 8320.18it/s] 24%|██▍       | 97233/400000 [00:12<00:36, 8298.82it/s] 25%|██▍       | 98065/400000 [00:12<00:36, 8303.64it/s] 25%|██▍       | 98896/400000 [00:13<00:37, 8003.04it/s] 25%|██▍       | 99699/400000 [00:13<00:38, 7897.24it/s] 25%|██▌       | 100491/400000 [00:13<00:38, 7767.88it/s] 25%|██▌       | 101270/400000 [00:13<00:38, 7768.63it/s] 26%|██▌       | 102049/400000 [00:13<00:38, 7735.22it/s] 26%|██▌       | 102853/400000 [00:13<00:37, 7824.10it/s] 26%|██▌       | 103637/400000 [00:13<00:39, 7568.23it/s] 26%|██▌       | 104454/400000 [00:13<00:38, 7736.88it/s] 26%|██▋       | 105257/400000 [00:13<00:37, 7822.27it/s] 27%|██▋       | 106051/400000 [00:14<00:37, 7855.79it/s] 27%|██▋       | 106861/400000 [00:14<00:36, 7927.44it/s] 27%|██▋       | 107689/400000 [00:14<00:36, 8029.34it/s] 27%|██▋       | 108532/400000 [00:14<00:35, 8145.05it/s] 27%|██▋       | 109354/400000 [00:14<00:35, 8167.19it/s] 28%|██▊       | 110196/400000 [00:14<00:35, 8239.66it/s] 28%|██▊       | 111027/400000 [00:14<00:34, 8259.18it/s] 28%|██▊       | 111854/400000 [00:14<00:37, 7776.44it/s] 28%|██▊       | 112652/400000 [00:14<00:36, 7833.84it/s] 28%|██▊       | 113440/400000 [00:14<00:38, 7458.97it/s] 29%|██▊       | 114242/400000 [00:15<00:37, 7615.66it/s] 29%|██▉       | 115054/400000 [00:15<00:36, 7759.82it/s] 29%|██▉       | 115835/400000 [00:15<00:37, 7603.20it/s] 29%|██▉       | 116600/400000 [00:15<00:37, 7578.53it/s] 29%|██▉       | 117393/400000 [00:15<00:36, 7678.73it/s] 30%|██▉       | 118200/400000 [00:15<00:36, 7791.98it/s] 30%|██▉       | 119003/400000 [00:15<00:35, 7860.66it/s] 30%|██▉       | 119791/400000 [00:15<00:37, 7529.77it/s] 30%|███       | 120549/400000 [00:15<00:37, 7531.40it/s] 30%|███       | 121363/400000 [00:15<00:36, 7703.71it/s] 31%|███       | 122160/400000 [00:16<00:35, 7780.62it/s] 31%|███       | 123004/400000 [00:16<00:34, 7965.88it/s] 31%|███       | 123838/400000 [00:16<00:34, 8072.93it/s] 31%|███       | 124648/400000 [00:16<00:34, 8058.96it/s] 31%|███▏      | 125456/400000 [00:16<00:34, 7869.62it/s] 32%|███▏      | 126246/400000 [00:16<00:35, 7775.79it/s] 32%|███▏      | 127026/400000 [00:16<00:35, 7608.98it/s] 32%|███▏      | 127789/400000 [00:16<00:36, 7442.08it/s] 32%|███▏      | 128599/400000 [00:16<00:35, 7627.27it/s] 32%|███▏      | 129373/400000 [00:16<00:35, 7659.35it/s] 33%|███▎      | 130141/400000 [00:17<00:36, 7486.81it/s] 33%|███▎      | 130944/400000 [00:17<00:35, 7641.23it/s] 33%|███▎      | 131711/400000 [00:17<00:37, 7250.53it/s] 33%|███▎      | 132491/400000 [00:17<00:36, 7405.48it/s] 33%|███▎      | 133283/400000 [00:17<00:35, 7550.45it/s] 34%|███▎      | 134068/400000 [00:17<00:34, 7635.32it/s] 34%|███▎      | 134887/400000 [00:17<00:34, 7791.34it/s] 34%|███▍      | 135670/400000 [00:17<00:35, 7542.01it/s] 34%|███▍      | 136428/400000 [00:17<00:34, 7537.25it/s] 34%|███▍      | 137226/400000 [00:18<00:34, 7663.90it/s] 34%|███▍      | 137995/400000 [00:18<00:34, 7663.54it/s] 35%|███▍      | 138801/400000 [00:18<00:33, 7776.19it/s] 35%|███▍      | 139581/400000 [00:18<00:34, 7642.45it/s] 35%|███▌      | 140390/400000 [00:18<00:33, 7771.42it/s] 35%|███▌      | 141220/400000 [00:18<00:32, 7921.09it/s] 36%|███▌      | 142038/400000 [00:18<00:32, 7995.13it/s] 36%|███▌      | 142882/400000 [00:18<00:31, 8120.87it/s] 36%|███▌      | 143726/400000 [00:18<00:31, 8213.08it/s] 36%|███▌      | 144556/400000 [00:18<00:31, 8237.14it/s] 36%|███▋      | 145397/400000 [00:19<00:30, 8286.63it/s] 37%|███▋      | 146227/400000 [00:19<00:30, 8249.67it/s] 37%|███▋      | 147058/400000 [00:19<00:30, 8266.76it/s] 37%|███▋      | 147888/400000 [00:19<00:30, 8275.35it/s] 37%|███▋      | 148716/400000 [00:19<00:30, 8229.52it/s] 37%|███▋      | 149557/400000 [00:19<00:30, 8282.66it/s] 38%|███▊      | 150392/400000 [00:19<00:30, 8301.39it/s] 38%|███▊      | 151223/400000 [00:19<00:30, 8226.56it/s] 38%|███▊      | 152067/400000 [00:19<00:29, 8286.96it/s] 38%|███▊      | 152897/400000 [00:19<00:30, 8212.15it/s] 38%|███▊      | 153719/400000 [00:20<00:30, 8159.04it/s] 39%|███▊      | 154536/400000 [00:20<00:30, 8130.92it/s] 39%|███▉      | 155350/400000 [00:20<00:31, 7891.56it/s] 39%|███▉      | 156153/400000 [00:20<00:30, 7931.58it/s] 39%|███▉      | 156948/400000 [00:20<00:30, 7917.88it/s] 39%|███▉      | 157742/400000 [00:20<00:30, 7924.05it/s] 40%|███▉      | 158536/400000 [00:20<00:30, 7870.71it/s] 40%|███▉      | 159324/400000 [00:20<00:31, 7616.91it/s] 40%|████      | 160088/400000 [00:20<00:32, 7459.44it/s] 40%|████      | 160892/400000 [00:20<00:31, 7623.43it/s] 40%|████      | 161697/400000 [00:21<00:30, 7743.97it/s] 41%|████      | 162508/400000 [00:21<00:30, 7848.78it/s] 41%|████      | 163308/400000 [00:21<00:29, 7890.94it/s] 41%|████      | 164099/400000 [00:21<00:30, 7800.84it/s] 41%|████      | 164891/400000 [00:21<00:30, 7834.28it/s] 41%|████▏     | 165676/400000 [00:21<00:30, 7764.42it/s] 42%|████▏     | 166454/400000 [00:21<00:30, 7678.60it/s] 42%|████▏     | 167250/400000 [00:21<00:29, 7759.06it/s] 42%|████▏     | 168027/400000 [00:21<00:31, 7446.60it/s] 42%|████▏     | 168779/400000 [00:22<00:30, 7466.94it/s] 42%|████▏     | 169528/400000 [00:22<00:31, 7397.82it/s] 43%|████▎     | 170285/400000 [00:22<00:30, 7447.89it/s] 43%|████▎     | 171054/400000 [00:22<00:30, 7517.66it/s] 43%|████▎     | 171821/400000 [00:22<00:30, 7560.69it/s] 43%|████▎     | 172578/400000 [00:22<00:30, 7437.76it/s] 43%|████▎     | 173370/400000 [00:22<00:29, 7575.21it/s] 44%|████▎     | 174138/400000 [00:22<00:29, 7605.40it/s] 44%|████▎     | 174915/400000 [00:22<00:29, 7653.83it/s] 44%|████▍     | 175682/400000 [00:22<00:30, 7468.42it/s] 44%|████▍     | 176480/400000 [00:23<00:29, 7613.69it/s] 44%|████▍     | 177315/400000 [00:23<00:28, 7819.99it/s] 45%|████▍     | 178100/400000 [00:23<00:28, 7806.01it/s] 45%|████▍     | 178916/400000 [00:23<00:27, 7904.27it/s] 45%|████▍     | 179709/400000 [00:23<00:28, 7650.55it/s] 45%|████▌     | 180497/400000 [00:23<00:28, 7715.09it/s] 45%|████▌     | 181281/400000 [00:23<00:28, 7750.07it/s] 46%|████▌     | 182076/400000 [00:23<00:27, 7808.26it/s] 46%|████▌     | 182888/400000 [00:23<00:27, 7898.42it/s] 46%|████▌     | 183679/400000 [00:23<00:28, 7714.80it/s] 46%|████▌     | 184476/400000 [00:24<00:27, 7788.92it/s] 46%|████▋     | 185257/400000 [00:24<00:27, 7771.31it/s] 47%|████▋     | 186066/400000 [00:24<00:27, 7863.45it/s] 47%|████▋     | 186869/400000 [00:24<00:26, 7912.20it/s] 47%|████▋     | 187661/400000 [00:24<00:27, 7735.28it/s] 47%|████▋     | 188469/400000 [00:24<00:27, 7833.19it/s] 47%|████▋     | 189254/400000 [00:24<00:27, 7803.68it/s] 48%|████▊     | 190061/400000 [00:24<00:26, 7880.45it/s] 48%|████▊     | 190870/400000 [00:24<00:26, 7940.05it/s] 48%|████▊     | 191665/400000 [00:24<00:27, 7626.95it/s] 48%|████▊     | 192462/400000 [00:25<00:26, 7724.96it/s] 48%|████▊     | 193272/400000 [00:25<00:26, 7833.05it/s] 49%|████▊     | 194058/400000 [00:25<00:26, 7835.80it/s] 49%|████▊     | 194855/400000 [00:25<00:26, 7873.66it/s] 49%|████▉     | 195644/400000 [00:25<00:26, 7677.47it/s] 49%|████▉     | 196414/400000 [00:25<00:26, 7646.85it/s] 49%|████▉     | 197222/400000 [00:25<00:26, 7769.74it/s] 50%|████▉     | 198007/400000 [00:25<00:25, 7792.71it/s] 50%|████▉     | 198808/400000 [00:25<00:25, 7854.45it/s] 50%|████▉     | 199595/400000 [00:26<00:26, 7494.55it/s] 50%|█████     | 200381/400000 [00:26<00:26, 7600.30it/s] 50%|█████     | 201145/400000 [00:26<00:26, 7568.03it/s] 50%|█████     | 201905/400000 [00:26<00:26, 7521.25it/s] 51%|█████     | 202700/400000 [00:26<00:25, 7642.65it/s] 51%|█████     | 203466/400000 [00:26<00:27, 7251.21it/s] 51%|█████     | 204209/400000 [00:26<00:26, 7302.21it/s] 51%|█████     | 204962/400000 [00:26<00:26, 7366.86it/s] 51%|█████▏    | 205725/400000 [00:26<00:26, 7442.96it/s] 52%|█████▏    | 206483/400000 [00:26<00:25, 7481.95it/s] 52%|█████▏    | 207233/400000 [00:27<00:25, 7429.31it/s] 52%|█████▏    | 208051/400000 [00:27<00:25, 7638.44it/s] 52%|█████▏    | 208893/400000 [00:27<00:24, 7857.11it/s] 52%|█████▏    | 209734/400000 [00:27<00:23, 8012.93it/s] 53%|█████▎    | 210541/400000 [00:27<00:23, 8028.68it/s] 53%|█████▎    | 211376/400000 [00:27<00:23, 8120.60it/s] 53%|█████▎    | 212206/400000 [00:27<00:22, 8172.98it/s] 53%|█████▎    | 213052/400000 [00:27<00:22, 8256.20it/s] 53%|█████▎    | 213894/400000 [00:27<00:22, 8304.27it/s] 54%|█████▎    | 214726/400000 [00:27<00:22, 8255.20it/s] 54%|█████▍    | 215553/400000 [00:28<00:22, 8254.29it/s] 54%|█████▍    | 216379/400000 [00:28<00:22, 8193.77it/s] 54%|█████▍    | 217199/400000 [00:28<00:22, 8187.03it/s] 55%|█████▍    | 218018/400000 [00:28<00:22, 7926.62it/s] 55%|█████▍    | 218813/400000 [00:28<00:23, 7787.13it/s] 55%|█████▍    | 219594/400000 [00:28<00:25, 7202.17it/s] 55%|█████▌    | 220432/400000 [00:28<00:23, 7518.60it/s] 55%|█████▌    | 221276/400000 [00:28<00:22, 7773.17it/s] 56%|█████▌    | 222068/400000 [00:28<00:22, 7814.81it/s] 56%|█████▌    | 222857/400000 [00:28<00:22, 7802.02it/s] 56%|█████▌    | 223642/400000 [00:29<00:23, 7665.96it/s] 56%|█████▌    | 224457/400000 [00:29<00:22, 7804.09it/s] 56%|█████▋    | 225275/400000 [00:29<00:22, 7911.67it/s] 57%|█████▋    | 226104/400000 [00:29<00:21, 8019.08it/s] 57%|█████▋    | 226942/400000 [00:29<00:21, 8122.30it/s] 57%|█████▋    | 227757/400000 [00:29<00:21, 8044.85it/s] 57%|█████▋    | 228563/400000 [00:29<00:22, 7761.11it/s] 57%|█████▋    | 229371/400000 [00:29<00:21, 7852.53it/s] 58%|█████▊    | 230159/400000 [00:29<00:22, 7698.04it/s] 58%|█████▊    | 230932/400000 [00:30<00:22, 7489.94it/s] 58%|█████▊    | 231712/400000 [00:30<00:22, 7578.78it/s] 58%|█████▊    | 232473/400000 [00:30<00:22, 7546.66it/s] 58%|█████▊    | 233230/400000 [00:30<00:22, 7533.41it/s] 59%|█████▊    | 234029/400000 [00:30<00:21, 7664.23it/s] 59%|█████▊    | 234833/400000 [00:30<00:21, 7772.71it/s] 59%|█████▉    | 235622/400000 [00:30<00:21, 7806.88it/s] 59%|█████▉    | 236404/400000 [00:30<00:21, 7658.42it/s] 59%|█████▉    | 237205/400000 [00:30<00:21, 7734.54it/s] 59%|█████▉    | 237995/400000 [00:30<00:20, 7781.47it/s] 60%|█████▉    | 238779/400000 [00:31<00:20, 7798.65it/s] 60%|█████▉    | 239560/400000 [00:31<00:21, 7578.38it/s] 60%|██████    | 240352/400000 [00:31<00:20, 7676.02it/s] 60%|██████    | 241166/400000 [00:31<00:20, 7808.47it/s] 60%|██████    | 241969/400000 [00:31<00:20, 7873.40it/s] 61%|██████    | 242760/400000 [00:31<00:19, 7883.59it/s] 61%|██████    | 243550/400000 [00:31<00:19, 7878.15it/s] 61%|██████    | 244339/400000 [00:31<00:19, 7841.11it/s] 61%|██████▏   | 245124/400000 [00:31<00:19, 7754.39it/s] 61%|██████▏   | 245900/400000 [00:31<00:20, 7697.16it/s] 62%|██████▏   | 246671/400000 [00:32<00:20, 7504.05it/s] 62%|██████▏   | 247423/400000 [00:32<00:20, 7338.87it/s] 62%|██████▏   | 248183/400000 [00:32<00:20, 7414.33it/s] 62%|██████▏   | 248926/400000 [00:32<00:20, 7376.23it/s] 62%|██████▏   | 249690/400000 [00:32<00:20, 7451.67it/s] 63%|██████▎   | 250437/400000 [00:32<00:20, 7402.93it/s] 63%|██████▎   | 251179/400000 [00:32<00:20, 7100.95it/s] 63%|██████▎   | 251985/400000 [00:32<00:20, 7362.03it/s] 63%|██████▎   | 252733/400000 [00:32<00:19, 7395.27it/s] 63%|██████▎   | 253476/400000 [00:32<00:19, 7330.14it/s] 64%|██████▎   | 254283/400000 [00:33<00:19, 7537.24it/s] 64%|██████▍   | 255079/400000 [00:33<00:18, 7657.74it/s] 64%|██████▍   | 255866/400000 [00:33<00:18, 7719.42it/s] 64%|██████▍   | 256680/400000 [00:33<00:18, 7840.16it/s] 64%|██████▍   | 257477/400000 [00:33<00:18, 7876.84it/s] 65%|██████▍   | 258289/400000 [00:33<00:17, 7946.37it/s] 65%|██████▍   | 259085/400000 [00:33<00:18, 7696.34it/s] 65%|██████▍   | 259895/400000 [00:33<00:17, 7811.96it/s] 65%|██████▌   | 260706/400000 [00:33<00:17, 7897.15it/s] 65%|██████▌   | 261509/400000 [00:33<00:17, 7929.65it/s] 66%|██████▌   | 262304/400000 [00:34<00:17, 7862.47it/s] 66%|██████▌   | 263092/400000 [00:34<00:17, 7628.74it/s] 66%|██████▌   | 263903/400000 [00:34<00:17, 7765.59it/s] 66%|██████▌   | 264701/400000 [00:34<00:17, 7720.09it/s] 66%|██████▋   | 265500/400000 [00:34<00:17, 7799.05it/s] 67%|██████▋   | 266282/400000 [00:34<00:17, 7730.74it/s] 67%|██████▋   | 267057/400000 [00:34<00:18, 7367.87it/s] 67%|██████▋   | 267808/400000 [00:34<00:17, 7407.38it/s] 67%|██████▋   | 268552/400000 [00:34<00:17, 7415.83it/s] 67%|██████▋   | 269332/400000 [00:35<00:17, 7525.70it/s] 68%|██████▊   | 270122/400000 [00:35<00:17, 7632.38it/s] 68%|██████▊   | 270887/400000 [00:35<00:17, 7568.98it/s] 68%|██████▊   | 271665/400000 [00:35<00:16, 7630.10it/s] 68%|██████▊   | 272430/400000 [00:35<00:17, 7355.74it/s] 68%|██████▊   | 273186/400000 [00:35<00:17, 7415.20it/s] 68%|██████▊   | 273997/400000 [00:35<00:16, 7608.57it/s] 69%|██████▊   | 274761/400000 [00:35<00:16, 7367.44it/s] 69%|██████▉   | 275607/400000 [00:35<00:16, 7661.93it/s] 69%|██████▉   | 276449/400000 [00:35<00:15, 7872.53it/s] 69%|██████▉   | 277291/400000 [00:36<00:15, 8026.94it/s] 70%|██████▉   | 278133/400000 [00:36<00:14, 8139.13it/s] 70%|██████▉   | 278951/400000 [00:36<00:15, 8050.46it/s] 70%|██████▉   | 279794/400000 [00:36<00:14, 8159.84it/s] 70%|███████   | 280613/400000 [00:36<00:14, 8022.03it/s] 70%|███████   | 281419/400000 [00:36<00:14, 8032.93it/s] 71%|███████   | 282224/400000 [00:36<00:14, 8011.88it/s] 71%|███████   | 283027/400000 [00:36<00:15, 7666.58it/s] 71%|███████   | 283798/400000 [00:36<00:15, 7654.55it/s] 71%|███████   | 284605/400000 [00:36<00:14, 7772.60it/s] 71%|███████▏  | 285385/400000 [00:37<00:14, 7768.59it/s] 72%|███████▏  | 286170/400000 [00:37<00:14, 7790.68it/s] 72%|███████▏  | 286951/400000 [00:37<00:15, 7402.69it/s] 72%|███████▏  | 287728/400000 [00:37<00:14, 7507.42it/s] 72%|███████▏  | 288506/400000 [00:37<00:14, 7586.76it/s] 72%|███████▏  | 289312/400000 [00:37<00:14, 7720.36it/s] 73%|███████▎  | 290117/400000 [00:37<00:14, 7815.48it/s] 73%|███████▎  | 290901/400000 [00:37<00:14, 7514.51it/s] 73%|███████▎  | 291676/400000 [00:37<00:14, 7582.20it/s] 73%|███████▎  | 292456/400000 [00:38<00:14, 7645.79it/s] 73%|███████▎  | 293271/400000 [00:38<00:13, 7790.26it/s] 74%|███████▎  | 294053/400000 [00:38<00:13, 7756.37it/s] 74%|███████▎  | 294869/400000 [00:38<00:13, 7871.43it/s] 74%|███████▍  | 295714/400000 [00:38<00:12, 8035.63it/s] 74%|███████▍  | 296521/400000 [00:38<00:12, 8043.73it/s] 74%|███████▍  | 297363/400000 [00:38<00:12, 8152.95it/s] 75%|███████▍  | 298209/400000 [00:38<00:12, 8240.23it/s] 75%|███████▍  | 299035/400000 [00:38<00:12, 7766.58it/s] 75%|███████▍  | 299819/400000 [00:38<00:12, 7715.36it/s] 75%|███████▌  | 300596/400000 [00:39<00:12, 7698.24it/s] 75%|███████▌  | 301370/400000 [00:39<00:12, 7682.15it/s] 76%|███████▌  | 302173/400000 [00:39<00:12, 7782.06it/s] 76%|███████▌  | 302974/400000 [00:39<00:12, 7848.94it/s] 76%|███████▌  | 303783/400000 [00:39<00:12, 7918.11it/s] 76%|███████▌  | 304576/400000 [00:39<00:12, 7871.71it/s] 76%|███████▋  | 305365/400000 [00:39<00:12, 7818.97it/s] 77%|███████▋  | 306163/400000 [00:39<00:11, 7864.79it/s] 77%|███████▋  | 306951/400000 [00:39<00:12, 7663.78it/s] 77%|███████▋  | 307792/400000 [00:39<00:11, 7872.99it/s] 77%|███████▋  | 308628/400000 [00:40<00:11, 8011.34it/s] 77%|███████▋  | 309432/400000 [00:40<00:11, 7901.49it/s] 78%|███████▊  | 310246/400000 [00:40<00:11, 7970.90it/s] 78%|███████▊  | 311045/400000 [00:40<00:11, 7790.75it/s] 78%|███████▊  | 311849/400000 [00:40<00:11, 7863.88it/s] 78%|███████▊  | 312652/400000 [00:40<00:11, 7912.75it/s] 78%|███████▊  | 313445/400000 [00:40<00:10, 7878.47it/s] 79%|███████▊  | 314257/400000 [00:40<00:10, 7948.05it/s] 79%|███████▉  | 315053/400000 [00:40<00:11, 7532.56it/s] 79%|███████▉  | 315872/400000 [00:41<00:10, 7716.40it/s] 79%|███████▉  | 316687/400000 [00:41<00:10, 7836.62it/s] 79%|███████▉  | 317475/400000 [00:41<00:10, 7735.15it/s] 80%|███████▉  | 318264/400000 [00:41<00:10, 7780.03it/s] 80%|███████▉  | 319045/400000 [00:41<00:10, 7433.91it/s] 80%|███████▉  | 319829/400000 [00:41<00:10, 7549.81it/s] 80%|████████  | 320593/400000 [00:41<00:10, 7574.81it/s] 80%|████████  | 321398/400000 [00:41<00:10, 7709.06it/s] 81%|████████  | 322194/400000 [00:41<00:09, 7781.43it/s] 81%|████████  | 322975/400000 [00:41<00:10, 7517.47it/s] 81%|████████  | 323789/400000 [00:42<00:09, 7691.99it/s] 81%|████████  | 324601/400000 [00:42<00:09, 7815.24it/s] 81%|████████▏ | 325401/400000 [00:42<00:09, 7867.02it/s] 82%|████████▏ | 326190/400000 [00:42<00:09, 7754.91it/s] 82%|████████▏ | 326968/400000 [00:42<00:09, 7429.93it/s] 82%|████████▏ | 327780/400000 [00:42<00:09, 7622.24it/s] 82%|████████▏ | 328588/400000 [00:42<00:09, 7752.71it/s] 82%|████████▏ | 329392/400000 [00:42<00:09, 7835.75it/s] 83%|████████▎ | 330179/400000 [00:42<00:09, 7743.73it/s] 83%|████████▎ | 330956/400000 [00:42<00:09, 7494.70it/s] 83%|████████▎ | 331759/400000 [00:43<00:08, 7647.27it/s] 83%|████████▎ | 332561/400000 [00:43<00:08, 7754.32it/s] 83%|████████▎ | 333360/400000 [00:43<00:08, 7822.97it/s] 84%|████████▎ | 334176/400000 [00:43<00:08, 7921.01it/s] 84%|████████▎ | 334970/400000 [00:43<00:08, 7493.67it/s] 84%|████████▍ | 335726/400000 [00:43<00:08, 7386.76it/s] 84%|████████▍ | 336479/400000 [00:43<00:08, 7428.02it/s] 84%|████████▍ | 337225/400000 [00:43<00:08, 7342.95it/s] 85%|████████▍ | 338008/400000 [00:43<00:08, 7481.25it/s] 85%|████████▍ | 338759/400000 [00:44<00:08, 7285.94it/s] 85%|████████▍ | 339599/400000 [00:44<00:07, 7587.17it/s] 85%|████████▌ | 340443/400000 [00:44<00:07, 7823.78it/s] 85%|████████▌ | 341289/400000 [00:44<00:07, 8001.97it/s] 86%|████████▌ | 342095/400000 [00:44<00:07, 7776.79it/s] 86%|████████▌ | 342878/400000 [00:44<00:07, 7429.59it/s] 86%|████████▌ | 343654/400000 [00:44<00:07, 7525.43it/s] 86%|████████▌ | 344412/400000 [00:44<00:07, 7482.65it/s] 86%|████████▋ | 345183/400000 [00:44<00:07, 7544.64it/s] 86%|████████▋ | 345984/400000 [00:44<00:07, 7675.86it/s] 87%|████████▋ | 346754/400000 [00:45<00:07, 7356.05it/s] 87%|████████▋ | 347537/400000 [00:45<00:07, 7490.53it/s] 87%|████████▋ | 348344/400000 [00:45<00:06, 7652.98it/s] 87%|████████▋ | 349144/400000 [00:45<00:06, 7753.83it/s] 87%|████████▋ | 349943/400000 [00:45<00:06, 7822.50it/s] 88%|████████▊ | 350728/400000 [00:45<00:06, 7517.27it/s] 88%|████████▊ | 351510/400000 [00:45<00:06, 7604.31it/s] 88%|████████▊ | 352307/400000 [00:45<00:06, 7709.18it/s] 88%|████████▊ | 353092/400000 [00:45<00:06, 7750.31it/s] 88%|████████▊ | 353869/400000 [00:45<00:06, 7640.50it/s] 89%|████████▊ | 354635/400000 [00:46<00:06, 7320.63it/s] 89%|████████▉ | 355444/400000 [00:46<00:05, 7533.91it/s] 89%|████████▉ | 356209/400000 [00:46<00:05, 7567.26it/s] 89%|████████▉ | 356969/400000 [00:46<00:05, 7571.03it/s] 89%|████████▉ | 357731/400000 [00:46<00:05, 7583.12it/s] 90%|████████▉ | 358491/400000 [00:46<00:05, 7189.95it/s] 90%|████████▉ | 359228/400000 [00:46<00:05, 7240.30it/s] 90%|████████▉ | 359956/400000 [00:46<00:05, 7228.49it/s] 90%|█████████ | 360768/400000 [00:46<00:05, 7470.37it/s] 90%|█████████ | 361541/400000 [00:47<00:05, 7545.20it/s] 91%|█████████ | 362353/400000 [00:47<00:04, 7708.29it/s] 91%|█████████ | 363174/400000 [00:47<00:04, 7849.63it/s] 91%|█████████ | 363962/400000 [00:47<00:04, 7829.67it/s] 91%|█████████ | 364749/400000 [00:47<00:04, 7840.27it/s] 91%|█████████▏| 365565/400000 [00:47<00:04, 7931.50it/s] 92%|█████████▏| 366360/400000 [00:47<00:04, 7596.57it/s] 92%|█████████▏| 367154/400000 [00:47<00:04, 7694.86it/s] 92%|█████████▏| 367953/400000 [00:47<00:04, 7779.37it/s] 92%|█████████▏| 368734/400000 [00:47<00:04, 7723.46it/s] 92%|█████████▏| 369528/400000 [00:48<00:03, 7786.95it/s] 93%|█████████▎| 370330/400000 [00:48<00:03, 7853.64it/s] 93%|█████████▎| 371168/400000 [00:48<00:03, 8002.53it/s] 93%|█████████▎| 372008/400000 [00:48<00:03, 8116.05it/s] 93%|█████████▎| 372821/400000 [00:48<00:03, 8021.57it/s] 93%|█████████▎| 373662/400000 [00:48<00:03, 8132.12it/s] 94%|█████████▎| 374507/400000 [00:48<00:03, 8222.68it/s] 94%|█████████▍| 375331/400000 [00:48<00:03, 8099.73it/s] 94%|█████████▍| 376154/400000 [00:48<00:02, 8138.27it/s] 94%|█████████▍| 376969/400000 [00:48<00:02, 7963.56it/s] 94%|█████████▍| 377783/400000 [00:49<00:02, 8014.40it/s] 95%|█████████▍| 378586/400000 [00:49<00:02, 7709.84it/s] 95%|█████████▍| 379422/400000 [00:49<00:02, 7892.44it/s] 95%|█████████▌| 380249/400000 [00:49<00:02, 7998.56it/s] 95%|█████████▌| 381059/400000 [00:49<00:02, 8025.73it/s] 95%|█████████▌| 381864/400000 [00:49<00:02, 7923.28it/s] 96%|█████████▌| 382697/400000 [00:49<00:02, 8039.48it/s] 96%|█████████▌| 383503/400000 [00:49<00:02, 7918.58it/s] 96%|█████████▌| 384297/400000 [00:49<00:02, 7727.50it/s] 96%|█████████▋| 385072/400000 [00:49<00:01, 7706.11it/s] 96%|█████████▋| 385845/400000 [00:50<00:01, 7626.70it/s] 97%|█████████▋| 386609/400000 [00:50<00:01, 7587.79it/s] 97%|█████████▋| 387436/400000 [00:50<00:01, 7777.91it/s] 97%|█████████▋| 388271/400000 [00:50<00:01, 7940.20it/s] 97%|█████████▋| 389068/400000 [00:50<00:01, 7933.38it/s] 97%|█████████▋| 389863/400000 [00:50<00:01, 7816.82it/s] 98%|█████████▊| 390648/400000 [00:50<00:01, 7826.32it/s] 98%|█████████▊| 391476/400000 [00:50<00:01, 7955.91it/s] 98%|█████████▊| 392310/400000 [00:50<00:00, 8055.42it/s] 98%|█████████▊| 393117/400000 [00:50<00:00, 8006.86it/s] 98%|█████████▊| 393919/400000 [00:51<00:00, 7934.20it/s] 99%|█████████▊| 394714/400000 [00:51<00:00, 7706.26it/s] 99%|█████████▉| 395557/400000 [00:51<00:00, 7908.15it/s] 99%|█████████▉| 396400/400000 [00:51<00:00, 8056.66it/s] 99%|█████████▉| 397209/400000 [00:51<00:00, 7789.85it/s] 99%|█████████▉| 397992/400000 [00:51<00:00, 7575.73it/s]100%|█████████▉| 398754/400000 [00:51<00:00, 7545.92it/s]100%|█████████▉| 399563/400000 [00:51<00:00, 7700.78it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7708.99it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fa342577400>, <torchtext.data.dataset.TabularDataset object at 0x7fa342577550>, <torchtext.vocab.Vocab object at 0x7fa342577470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.09 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.00 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.00 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 13.00 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.75 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.10 file/s]2020-06-15 18:19:46.415840: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-15 18:19:46.422891: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-06-15 18:19:46.423346: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56381a78b870 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-15 18:19:46.423402: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:00, 163517.36it/s] 59%|█████▉    | 5840896/9912422 [00:00<00:17, 233314.67it/s]9920512it [00:00, 42773994.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 648618.20it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 991319.34it/s]1654784it [00:00, 11922881.75it/s]                          
0it [00:00, ?it/s]8192it [00:00, 266954.69it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
