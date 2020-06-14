
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '77c2c9f427106a88e922f586a543b21e10cfaf22', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/77c2c9f427106a88e922f586a543b21e10cfaf22

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/77c2c9f427106a88e922f586a543b21e10cfaf22

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe24f18d6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe24f18d6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe2ba7550d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe2ba7550d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe2d4481ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe2d4481ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe267fd2950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe267fd2950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe267fd2950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 44%|████▍     | 4390912/9912422 [00:00<00:00, 43897230.72it/s]9920512it [00:00, 35688157.08it/s]                             
0it [00:00, ?it/s]32768it [00:00, 654970.23it/s]
0it [00:00, ?it/s]  4%|▍         | 73728/1648877 [00:00<00:02, 719821.89it/s]1654784it [00:00, 11487823.33it/s]                         
0it [00:00, ?it/s]8192it [00:00, 188130.28it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe267fbfb38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe24f05ce80>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe267fd2598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe267fd2598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe267fd2598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:53,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:52,  2.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  3.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  3.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.54 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.08 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.08 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 18.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 29.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 36.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 42.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:01, 53.35 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.12 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 68.32 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.59 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.70 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.90 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.90 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.72s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 77.90 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.49 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.31s/ url]
0 examples [00:00, ? examples/s]2020-06-14 12:11:34.865579: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 12:11:34.972380: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-14 12:11:34.973235: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55aa67236b80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 12:11:34.973254: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  4.14 examples/s]89 examples [00:00,  5.91 examples/s]172 examples [00:00,  8.41 examples/s]246 examples [00:00, 11.96 examples/s]330 examples [00:00, 16.98 examples/s]417 examples [00:00, 24.05 examples/s]507 examples [00:00, 33.97 examples/s]596 examples [00:00, 47.74 examples/s]676 examples [00:01, 66.48 examples/s]762 examples [00:01, 91.89 examples/s]847 examples [00:01, 125.46 examples/s]935 examples [00:01, 168.65 examples/s]1027 examples [00:01, 223.34 examples/s]1113 examples [00:01, 286.71 examples/s]1199 examples [00:01, 357.65 examples/s]1290 examples [00:01, 437.11 examples/s]1380 examples [00:01, 516.78 examples/s]1474 examples [00:01, 597.43 examples/s]1564 examples [00:02, 649.94 examples/s]1652 examples [00:02, 691.39 examples/s]1738 examples [00:02, 726.07 examples/s]1829 examples [00:02, 772.23 examples/s]1918 examples [00:02, 802.80 examples/s]2005 examples [00:02, 819.03 examples/s]2100 examples [00:02, 852.24 examples/s]2195 examples [00:02, 878.99 examples/s]2290 examples [00:02, 898.09 examples/s]2383 examples [00:02, 906.58 examples/s]2476 examples [00:03, 905.73 examples/s]2568 examples [00:03, 903.36 examples/s]2662 examples [00:03, 913.68 examples/s]2754 examples [00:03, 911.51 examples/s]2846 examples [00:03, 894.04 examples/s]2936 examples [00:03, 868.15 examples/s]3029 examples [00:03, 884.46 examples/s]3118 examples [00:03, 859.47 examples/s]3205 examples [00:03, 860.21 examples/s]3293 examples [00:04, 863.14 examples/s]3385 examples [00:04, 878.49 examples/s]3479 examples [00:04, 895.11 examples/s]3576 examples [00:04, 914.42 examples/s]3671 examples [00:04, 921.08 examples/s]3764 examples [00:04, 923.21 examples/s]3862 examples [00:04, 937.38 examples/s]3960 examples [00:04, 949.58 examples/s]4056 examples [00:04, 951.34 examples/s]4153 examples [00:04, 956.52 examples/s]4249 examples [00:05, 956.80 examples/s]4345 examples [00:05, 952.42 examples/s]4442 examples [00:05, 957.42 examples/s]4538 examples [00:05, 955.18 examples/s]4635 examples [00:05, 957.55 examples/s]4731 examples [00:05, 932.85 examples/s]4825 examples [00:05, 912.98 examples/s]4925 examples [00:05, 936.66 examples/s]5021 examples [00:05, 941.86 examples/s]5116 examples [00:05, 939.23 examples/s]5211 examples [00:06, 931.42 examples/s]5305 examples [00:06, 887.63 examples/s]5395 examples [00:06, 882.44 examples/s]5484 examples [00:06, 857.53 examples/s]5571 examples [00:06, 860.83 examples/s]5666 examples [00:06, 884.57 examples/s]5755 examples [00:06, 866.21 examples/s]5859 examples [00:06, 910.88 examples/s]5956 examples [00:06, 926.93 examples/s]6053 examples [00:06, 937.72 examples/s]6148 examples [00:07, 935.97 examples/s]6242 examples [00:07, 917.82 examples/s]6337 examples [00:07, 924.36 examples/s]6436 examples [00:07, 941.81 examples/s]6531 examples [00:07, 941.62 examples/s]6626 examples [00:07, 918.51 examples/s]6720 examples [00:07, 923.63 examples/s]6813 examples [00:07, 916.21 examples/s]6908 examples [00:07, 924.73 examples/s]7001 examples [00:08, 921.29 examples/s]7097 examples [00:08, 932.30 examples/s]7192 examples [00:08, 937.47 examples/s]7291 examples [00:08, 951.30 examples/s]7387 examples [00:08, 930.42 examples/s]7481 examples [00:08, 917.56 examples/s]7577 examples [00:08, 927.77 examples/s]7670 examples [00:08, 881.42 examples/s]7763 examples [00:08, 893.10 examples/s]7853 examples [00:08, 873.74 examples/s]7944 examples [00:09, 881.78 examples/s]8033 examples [00:09, 884.22 examples/s]8127 examples [00:09, 898.03 examples/s]8223 examples [00:09, 913.78 examples/s]8315 examples [00:09, 906.15 examples/s]8406 examples [00:09, 906.18 examples/s]8497 examples [00:09, 876.08 examples/s]8585 examples [00:09, 871.52 examples/s]8673 examples [00:09, 863.17 examples/s]8768 examples [00:09, 886.41 examples/s]8859 examples [00:10, 892.60 examples/s]8955 examples [00:10, 911.28 examples/s]9056 examples [00:10, 936.58 examples/s]9157 examples [00:10, 954.37 examples/s]9255 examples [00:10, 960.79 examples/s]9352 examples [00:10, 911.65 examples/s]9444 examples [00:10, 908.85 examples/s]9546 examples [00:10, 937.59 examples/s]9641 examples [00:10, 936.13 examples/s]9736 examples [00:11, 926.22 examples/s]9836 examples [00:11, 945.07 examples/s]9931 examples [00:11, 941.51 examples/s]10026 examples [00:11, 875.05 examples/s]10118 examples [00:11, 886.86 examples/s]10208 examples [00:11, 888.31 examples/s]10301 examples [00:11, 900.09 examples/s]10395 examples [00:11, 910.96 examples/s]10499 examples [00:11, 944.96 examples/s]10597 examples [00:11, 952.16 examples/s]10693 examples [00:12, 935.54 examples/s]10787 examples [00:12, 910.79 examples/s]10879 examples [00:12, 888.91 examples/s]10973 examples [00:12, 897.13 examples/s]11064 examples [00:12, 900.48 examples/s]11157 examples [00:12, 908.78 examples/s]11257 examples [00:12, 933.62 examples/s]11351 examples [00:12, 912.32 examples/s]11443 examples [00:12, 908.96 examples/s]11539 examples [00:12, 922.35 examples/s]11632 examples [00:13, 923.88 examples/s]11727 examples [00:13, 930.27 examples/s]11821 examples [00:13, 928.10 examples/s]11914 examples [00:13, 926.68 examples/s]12010 examples [00:13, 935.65 examples/s]12104 examples [00:13, 935.88 examples/s]12202 examples [00:13, 945.68 examples/s]12297 examples [00:13, 918.11 examples/s]12390 examples [00:13, 883.80 examples/s]12479 examples [00:14, 868.54 examples/s]12574 examples [00:14, 889.67 examples/s]12669 examples [00:14, 906.73 examples/s]12762 examples [00:14, 911.62 examples/s]12860 examples [00:14, 928.66 examples/s]12954 examples [00:14, 914.36 examples/s]13047 examples [00:14, 916.52 examples/s]13139 examples [00:14, 915.65 examples/s]13231 examples [00:14, 916.68 examples/s]13323 examples [00:14, 894.74 examples/s]13418 examples [00:15, 909.22 examples/s]13515 examples [00:15, 925.60 examples/s]13615 examples [00:15, 946.61 examples/s]13711 examples [00:15, 948.60 examples/s]13808 examples [00:15, 953.25 examples/s]13904 examples [00:15, 893.37 examples/s]13998 examples [00:15, 904.13 examples/s]14093 examples [00:15, 916.07 examples/s]14186 examples [00:15, 919.43 examples/s]14281 examples [00:15, 926.92 examples/s]14374 examples [00:16, 917.37 examples/s]14467 examples [00:16, 920.10 examples/s]14560 examples [00:16, 913.89 examples/s]14652 examples [00:16, 910.15 examples/s]14748 examples [00:16, 923.76 examples/s]14841 examples [00:16, 911.02 examples/s]14935 examples [00:16, 918.70 examples/s]15029 examples [00:16, 924.67 examples/s]15126 examples [00:16, 936.20 examples/s]15228 examples [00:16, 959.84 examples/s]15325 examples [00:17, 931.44 examples/s]15419 examples [00:17, 890.83 examples/s]15510 examples [00:17, 893.90 examples/s]15608 examples [00:17, 918.02 examples/s]15705 examples [00:17, 930.91 examples/s]15799 examples [00:17, 919.09 examples/s]15897 examples [00:17, 935.48 examples/s]15991 examples [00:17, 900.13 examples/s]16085 examples [00:17, 910.89 examples/s]16177 examples [00:18, 910.35 examples/s]16274 examples [00:18, 926.18 examples/s]16372 examples [00:18, 939.93 examples/s]16469 examples [00:18, 946.93 examples/s]16567 examples [00:18, 954.44 examples/s]16663 examples [00:18, 904.39 examples/s]16765 examples [00:18, 936.07 examples/s]16860 examples [00:18, 929.27 examples/s]16954 examples [00:18, 883.00 examples/s]17055 examples [00:18, 917.60 examples/s]17148 examples [00:19, 919.97 examples/s]17244 examples [00:19, 930.83 examples/s]17348 examples [00:19, 960.30 examples/s]17445 examples [00:19, 962.85 examples/s]17552 examples [00:19, 992.38 examples/s]17652 examples [00:19, 974.66 examples/s]17750 examples [00:19, 976.23 examples/s]17848 examples [00:19, 958.63 examples/s]17945 examples [00:19, 941.11 examples/s]18044 examples [00:20, 952.94 examples/s]18140 examples [00:20, 951.56 examples/s]18236 examples [00:20, 945.51 examples/s]18331 examples [00:20, 933.38 examples/s]18425 examples [00:20, 919.63 examples/s]18518 examples [00:20, 859.25 examples/s]18609 examples [00:20, 872.28 examples/s]18706 examples [00:20, 897.44 examples/s]18804 examples [00:20, 920.45 examples/s]18897 examples [00:20, 895.11 examples/s]18988 examples [00:21, 898.59 examples/s]19085 examples [00:21, 917.86 examples/s]19182 examples [00:21, 930.90 examples/s]19276 examples [00:21, 918.08 examples/s]19369 examples [00:21, 881.40 examples/s]19461 examples [00:21, 889.88 examples/s]19562 examples [00:21, 922.66 examples/s]19659 examples [00:21, 933.94 examples/s]19753 examples [00:21, 927.71 examples/s]19849 examples [00:21, 936.59 examples/s]19943 examples [00:22, 913.97 examples/s]20035 examples [00:22, 843.03 examples/s]20124 examples [00:22, 854.69 examples/s]20215 examples [00:22, 870.51 examples/s]20303 examples [00:22, 868.23 examples/s]20391 examples [00:22, 854.47 examples/s]20489 examples [00:22, 887.25 examples/s]20588 examples [00:22, 914.17 examples/s]20681 examples [00:22, 901.42 examples/s]20779 examples [00:23, 920.90 examples/s]20877 examples [00:23, 935.59 examples/s]20971 examples [00:23, 931.51 examples/s]21071 examples [00:23, 949.35 examples/s]21167 examples [00:23, 951.54 examples/s]21263 examples [00:23, 935.99 examples/s]21357 examples [00:23, 936.94 examples/s]21456 examples [00:23, 949.50 examples/s]21552 examples [00:23, 890.50 examples/s]21642 examples [00:23, 885.27 examples/s]21734 examples [00:24, 893.40 examples/s]21825 examples [00:24, 897.62 examples/s]21916 examples [00:24, 894.61 examples/s]22010 examples [00:24, 906.05 examples/s]22103 examples [00:24, 911.10 examples/s]22195 examples [00:24, 908.69 examples/s]22293 examples [00:24, 928.95 examples/s]22388 examples [00:24, 933.52 examples/s]22483 examples [00:24, 937.99 examples/s]22577 examples [00:25, 903.07 examples/s]22668 examples [00:25, 895.55 examples/s]22765 examples [00:25, 915.98 examples/s]22857 examples [00:25, 881.29 examples/s]22946 examples [00:25, 867.67 examples/s]23034 examples [00:25, 829.47 examples/s]23118 examples [00:25, 822.38 examples/s]23204 examples [00:25, 832.52 examples/s]23289 examples [00:25, 836.74 examples/s]23376 examples [00:25, 845.93 examples/s]23461 examples [00:26, 840.90 examples/s]23547 examples [00:26, 843.82 examples/s]23634 examples [00:26, 848.58 examples/s]23719 examples [00:26, 815.09 examples/s]23804 examples [00:26, 821.50 examples/s]23888 examples [00:26, 824.23 examples/s]23976 examples [00:26, 838.98 examples/s]24064 examples [00:26, 848.35 examples/s]24153 examples [00:26, 859.93 examples/s]24240 examples [00:26, 861.30 examples/s]24327 examples [00:27, 849.59 examples/s]24413 examples [00:27, 840.90 examples/s]24498 examples [00:27, 813.03 examples/s]24589 examples [00:27, 837.75 examples/s]24682 examples [00:27, 862.95 examples/s]24769 examples [00:27, 858.57 examples/s]24856 examples [00:27, 857.33 examples/s]24948 examples [00:27, 874.59 examples/s]25040 examples [00:27, 887.37 examples/s]25132 examples [00:28, 895.04 examples/s]25222 examples [00:28, 889.72 examples/s]25316 examples [00:28, 902.68 examples/s]25407 examples [00:28, 887.51 examples/s]25499 examples [00:28, 895.49 examples/s]25592 examples [00:28, 903.86 examples/s]25683 examples [00:28, 892.90 examples/s]25776 examples [00:28, 901.12 examples/s]25868 examples [00:28, 906.38 examples/s]25959 examples [00:28, 906.62 examples/s]26050 examples [00:29, 885.50 examples/s]26139 examples [00:29, 856.91 examples/s]26233 examples [00:29, 879.20 examples/s]26322 examples [00:29, 882.20 examples/s]26418 examples [00:29, 902.30 examples/s]26516 examples [00:29, 920.85 examples/s]26609 examples [00:29, 909.86 examples/s]26709 examples [00:29, 934.57 examples/s]26810 examples [00:29, 954.85 examples/s]26909 examples [00:29, 963.09 examples/s]27006 examples [00:30, 962.97 examples/s]27103 examples [00:30, 941.94 examples/s]27198 examples [00:30, 936.87 examples/s]27292 examples [00:30, 932.73 examples/s]27386 examples [00:30, 914.35 examples/s]27478 examples [00:30, 857.35 examples/s]27570 examples [00:30, 874.49 examples/s]27662 examples [00:30, 887.48 examples/s]27764 examples [00:30, 922.84 examples/s]27862 examples [00:30, 936.40 examples/s]27957 examples [00:31, 938.64 examples/s]28052 examples [00:31, 932.59 examples/s]28149 examples [00:31, 940.92 examples/s]28244 examples [00:31, 929.45 examples/s]28343 examples [00:31, 946.64 examples/s]28438 examples [00:31, 944.63 examples/s]28533 examples [00:31, 896.47 examples/s]28629 examples [00:31, 914.62 examples/s]28727 examples [00:31, 931.53 examples/s]28825 examples [00:32, 944.50 examples/s]28920 examples [00:32, 921.23 examples/s]29013 examples [00:32, 876.44 examples/s]29106 examples [00:32, 889.59 examples/s]29203 examples [00:32, 910.92 examples/s]29305 examples [00:32, 937.17 examples/s]29400 examples [00:32, 940.45 examples/s]29495 examples [00:32, 930.63 examples/s]29591 examples [00:32, 938.55 examples/s]29690 examples [00:32, 951.90 examples/s]29789 examples [00:33, 962.25 examples/s]29886 examples [00:33, 943.63 examples/s]29981 examples [00:33, 934.67 examples/s]30075 examples [00:33, 842.64 examples/s]30165 examples [00:33, 858.61 examples/s]30259 examples [00:33, 880.91 examples/s]30352 examples [00:33, 893.21 examples/s]30450 examples [00:33, 916.69 examples/s]30543 examples [00:33, 919.43 examples/s]30638 examples [00:34, 928.00 examples/s]30732 examples [00:34, 884.83 examples/s]30822 examples [00:34, 874.26 examples/s]30915 examples [00:34, 889.19 examples/s]31005 examples [00:34, 873.78 examples/s]31104 examples [00:34, 903.37 examples/s]31195 examples [00:34, 895.03 examples/s]31287 examples [00:34, 900.42 examples/s]31384 examples [00:34, 919.07 examples/s]31478 examples [00:34, 925.19 examples/s]31571 examples [00:35, 912.97 examples/s]31663 examples [00:35, 894.17 examples/s]31759 examples [00:35, 912.37 examples/s]31859 examples [00:35, 934.32 examples/s]31954 examples [00:35, 937.72 examples/s]32048 examples [00:35, 894.84 examples/s]32141 examples [00:35, 903.22 examples/s]32238 examples [00:35, 919.48 examples/s]32331 examples [00:35, 900.76 examples/s]32423 examples [00:35, 905.08 examples/s]32517 examples [00:36, 913.31 examples/s]32609 examples [00:36, 915.21 examples/s]32701 examples [00:36, 909.71 examples/s]32793 examples [00:36, 905.24 examples/s]32886 examples [00:36, 912.51 examples/s]32978 examples [00:36, 895.72 examples/s]33068 examples [00:36, 892.53 examples/s]33164 examples [00:36, 910.98 examples/s]33261 examples [00:36, 926.91 examples/s]33356 examples [00:37, 932.76 examples/s]33450 examples [00:37, 913.86 examples/s]33542 examples [00:37, 890.70 examples/s]33632 examples [00:37, 871.73 examples/s]33728 examples [00:37, 893.48 examples/s]33821 examples [00:37, 903.63 examples/s]33918 examples [00:37, 921.20 examples/s]34011 examples [00:37, 918.02 examples/s]34103 examples [00:37, 915.27 examples/s]34202 examples [00:37, 935.54 examples/s]34304 examples [00:38, 957.81 examples/s]34401 examples [00:38, 949.61 examples/s]34497 examples [00:38, 948.16 examples/s]34592 examples [00:38, 920.10 examples/s]34685 examples [00:38, 914.31 examples/s]34779 examples [00:38, 921.64 examples/s]34872 examples [00:38, 918.71 examples/s]34974 examples [00:38, 945.24 examples/s]35069 examples [00:38, 931.86 examples/s]35163 examples [00:38, 902.27 examples/s]35254 examples [00:39, 892.55 examples/s]35344 examples [00:39, 888.88 examples/s]35439 examples [00:39, 906.37 examples/s]35536 examples [00:39, 924.00 examples/s]35629 examples [00:39, 924.09 examples/s]35725 examples [00:39, 932.69 examples/s]35819 examples [00:39, 932.65 examples/s]35919 examples [00:39, 951.46 examples/s]36015 examples [00:39, 945.35 examples/s]36112 examples [00:39, 952.31 examples/s]36208 examples [00:40, 938.53 examples/s]36303 examples [00:40, 941.41 examples/s]36398 examples [00:40, 938.47 examples/s]36492 examples [00:40, 937.44 examples/s]36586 examples [00:40, 925.01 examples/s]36679 examples [00:40, 924.86 examples/s]36772 examples [00:40, 909.88 examples/s]36868 examples [00:40, 922.41 examples/s]36961 examples [00:40, 922.99 examples/s]37060 examples [00:41, 941.74 examples/s]37156 examples [00:41, 945.00 examples/s]37251 examples [00:41, 944.64 examples/s]37346 examples [00:41, 936.56 examples/s]37440 examples [00:41, 933.94 examples/s]37534 examples [00:41, 934.47 examples/s]37628 examples [00:41, 910.49 examples/s]37720 examples [00:41, 903.38 examples/s]37815 examples [00:41, 914.68 examples/s]37911 examples [00:41, 925.98 examples/s]38012 examples [00:42, 947.84 examples/s]38107 examples [00:42, 925.76 examples/s]38200 examples [00:42, 898.23 examples/s]38291 examples [00:42, 886.83 examples/s]38382 examples [00:42, 891.76 examples/s]38473 examples [00:42, 894.50 examples/s]38563 examples [00:42, 893.38 examples/s]38660 examples [00:42, 913.31 examples/s]38756 examples [00:42, 923.88 examples/s]38852 examples [00:42, 931.13 examples/s]38949 examples [00:43, 941.48 examples/s]39044 examples [00:43, 942.51 examples/s]39139 examples [00:43, 913.73 examples/s]39231 examples [00:43, 908.45 examples/s]39323 examples [00:43, 901.85 examples/s]39414 examples [00:43, 897.94 examples/s]39508 examples [00:43, 909.72 examples/s]39600 examples [00:43, 903.27 examples/s]39691 examples [00:43, 889.25 examples/s]39781 examples [00:43, 886.99 examples/s]39877 examples [00:44, 905.81 examples/s]39970 examples [00:44, 912.58 examples/s]40062 examples [00:44, 859.00 examples/s]40151 examples [00:44, 866.74 examples/s]40239 examples [00:44, 866.38 examples/s]40330 examples [00:44, 876.40 examples/s]40426 examples [00:44, 899.09 examples/s]40524 examples [00:44, 920.73 examples/s]40618 examples [00:44, 922.87 examples/s]40713 examples [00:45, 929.96 examples/s]40807 examples [00:45, 929.63 examples/s]40901 examples [00:45, 922.15 examples/s]40998 examples [00:45, 935.13 examples/s]41094 examples [00:45, 940.42 examples/s]41189 examples [00:45, 925.50 examples/s]41282 examples [00:45, 888.07 examples/s]41375 examples [00:45, 899.66 examples/s]41466 examples [00:45, 897.23 examples/s]41556 examples [00:45, 893.29 examples/s]41657 examples [00:46, 925.18 examples/s]41754 examples [00:46, 935.38 examples/s]41850 examples [00:46, 941.83 examples/s]41945 examples [00:46, 942.24 examples/s]42040 examples [00:46, 930.24 examples/s]42137 examples [00:46, 940.20 examples/s]42232 examples [00:46, 912.45 examples/s]42329 examples [00:46, 927.90 examples/s]42427 examples [00:46, 942.05 examples/s]42523 examples [00:46, 946.17 examples/s]42620 examples [00:47, 951.06 examples/s]42716 examples [00:47, 921.31 examples/s]42809 examples [00:47, 919.86 examples/s]42902 examples [00:47, 921.43 examples/s]42997 examples [00:47, 928.98 examples/s]43091 examples [00:47, 926.49 examples/s]43184 examples [00:47, 922.83 examples/s]43280 examples [00:47, 932.14 examples/s]43377 examples [00:47, 941.17 examples/s]43472 examples [00:48, 924.55 examples/s]43569 examples [00:48, 936.59 examples/s]43664 examples [00:48, 938.84 examples/s]43762 examples [00:48, 947.93 examples/s]43857 examples [00:48, 943.39 examples/s]43952 examples [00:48, 922.81 examples/s]44045 examples [00:48, 922.46 examples/s]44138 examples [00:48, 924.69 examples/s]44234 examples [00:48, 934.79 examples/s]44328 examples [00:48, 897.59 examples/s]44419 examples [00:49, 895.32 examples/s]44512 examples [00:49, 904.40 examples/s]44609 examples [00:49, 921.33 examples/s]44706 examples [00:49, 933.85 examples/s]44808 examples [00:49, 956.12 examples/s]44904 examples [00:49, 943.62 examples/s]44999 examples [00:49, 928.61 examples/s]45099 examples [00:49, 948.88 examples/s]45195 examples [00:49, 935.51 examples/s]45289 examples [00:49, 934.23 examples/s]45383 examples [00:50, 935.53 examples/s]45477 examples [00:50, 933.40 examples/s]45571 examples [00:50, 928.21 examples/s]45666 examples [00:50, 932.54 examples/s]45766 examples [00:50, 949.11 examples/s]45862 examples [00:50, 937.00 examples/s]45956 examples [00:50, 905.31 examples/s]46047 examples [00:50, 906.41 examples/s]46138 examples [00:50, 904.63 examples/s]46235 examples [00:50, 922.08 examples/s]46328 examples [00:51, 922.93 examples/s]46422 examples [00:51, 926.18 examples/s]46515 examples [00:51, 926.70 examples/s]46612 examples [00:51, 938.68 examples/s]46711 examples [00:51, 953.01 examples/s]46807 examples [00:51, 937.71 examples/s]46901 examples [00:51, 917.35 examples/s]47000 examples [00:51, 937.46 examples/s]47097 examples [00:51, 944.20 examples/s]47197 examples [00:51, 958.04 examples/s]47293 examples [00:52, 949.93 examples/s]47389 examples [00:52, 943.36 examples/s]47484 examples [00:52, 920.50 examples/s]47577 examples [00:52, 921.82 examples/s]47670 examples [00:52, 908.22 examples/s]47764 examples [00:52, 916.43 examples/s]47856 examples [00:52, 888.71 examples/s]47952 examples [00:52, 907.64 examples/s]48048 examples [00:52, 921.36 examples/s]48142 examples [00:53, 926.06 examples/s]48235 examples [00:53, 923.77 examples/s]48331 examples [00:53, 932.74 examples/s]48425 examples [00:53, 916.30 examples/s]48523 examples [00:53, 934.27 examples/s]48617 examples [00:53, 930.41 examples/s]48711 examples [00:53, 922.48 examples/s]48810 examples [00:53, 938.71 examples/s]48905 examples [00:53, 931.42 examples/s]48999 examples [00:53, 923.99 examples/s]49093 examples [00:54, 927.32 examples/s]49186 examples [00:54, 889.93 examples/s]49276 examples [00:54, 883.03 examples/s]49371 examples [00:54, 900.87 examples/s]49469 examples [00:54, 921.41 examples/s]49562 examples [00:54, 921.49 examples/s]49655 examples [00:54, 909.78 examples/s]49756 examples [00:54, 936.56 examples/s]49851 examples [00:54, 938.99 examples/s]49950 examples [00:54, 953.17 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6873/50000 [00:00<00:00, 68665.40 examples/s] 39%|███▊      | 19338/50000 [00:00<00:00, 79356.96 examples/s] 61%|██████    | 30286/50000 [00:00<00:00, 86491.49 examples/s] 87%|████████▋ | 43334/50000 [00:00<00:00, 96222.58 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 735.82 examples/s]154 examples [00:00, 752.50 examples/s]249 examples [00:00, 801.26 examples/s]345 examples [00:00, 841.77 examples/s]446 examples [00:00, 884.16 examples/s]542 examples [00:00, 902.93 examples/s]635 examples [00:00, 910.31 examples/s]733 examples [00:00, 929.85 examples/s]830 examples [00:00, 939.14 examples/s]929 examples [00:01, 950.84 examples/s]1026 examples [00:01, 955.87 examples/s]1121 examples [00:01, 931.34 examples/s]1220 examples [00:01, 946.21 examples/s]1316 examples [00:01, 948.60 examples/s]1411 examples [00:01, 777.90 examples/s]1500 examples [00:01, 807.44 examples/s]1593 examples [00:01, 840.03 examples/s]1687 examples [00:01, 867.18 examples/s]1782 examples [00:01, 888.32 examples/s]1873 examples [00:02, 894.21 examples/s]1964 examples [00:02, 896.41 examples/s]2057 examples [00:02, 904.41 examples/s]2151 examples [00:02, 914.31 examples/s]2251 examples [00:02, 935.65 examples/s]2347 examples [00:02, 940.53 examples/s]2442 examples [00:02, 930.65 examples/s]2536 examples [00:02, 924.72 examples/s]2639 examples [00:02, 950.87 examples/s]2739 examples [00:03, 963.41 examples/s]2838 examples [00:03, 968.92 examples/s]2936 examples [00:03, 944.34 examples/s]3031 examples [00:03, 929.38 examples/s]3130 examples [00:03, 944.83 examples/s]3225 examples [00:03, 937.28 examples/s]3319 examples [00:03, 893.47 examples/s]3413 examples [00:03, 905.42 examples/s]3511 examples [00:03, 925.63 examples/s]3611 examples [00:03, 946.61 examples/s]3714 examples [00:04, 969.96 examples/s]3812 examples [00:04, 965.96 examples/s]3909 examples [00:04, 965.32 examples/s]4006 examples [00:04, 965.37 examples/s]4103 examples [00:04, 962.64 examples/s]4202 examples [00:04, 969.21 examples/s]4300 examples [00:04, 958.78 examples/s]4396 examples [00:04, 942.74 examples/s]4491 examples [00:04, 938.10 examples/s]4589 examples [00:04, 947.97 examples/s]4684 examples [00:05, 932.22 examples/s]4778 examples [00:05, 907.36 examples/s]4869 examples [00:05, 896.17 examples/s]4962 examples [00:05, 906.02 examples/s]5060 examples [00:05, 925.05 examples/s]5159 examples [00:05, 940.86 examples/s]5254 examples [00:05, 937.39 examples/s]5348 examples [00:05, 930.92 examples/s]5442 examples [00:05, 924.29 examples/s]5538 examples [00:05, 933.59 examples/s]5632 examples [00:06, 929.90 examples/s]5728 examples [00:06, 938.56 examples/s]5822 examples [00:06, 916.41 examples/s]5916 examples [00:06, 922.39 examples/s]6017 examples [00:06, 944.60 examples/s]6112 examples [00:06, 938.11 examples/s]6206 examples [00:06, 920.59 examples/s]6303 examples [00:06, 933.80 examples/s]6401 examples [00:06, 946.72 examples/s]6496 examples [00:07, 937.13 examples/s]6592 examples [00:07, 943.58 examples/s]6687 examples [00:07, 941.06 examples/s]6785 examples [00:07, 950.38 examples/s]6881 examples [00:07, 917.57 examples/s]6979 examples [00:07, 934.10 examples/s]7078 examples [00:07, 948.87 examples/s]7174 examples [00:07, 949.27 examples/s]7270 examples [00:07, 941.52 examples/s]7365 examples [00:07, 928.63 examples/s]7458 examples [00:08, 924.17 examples/s]7551 examples [00:08, 919.55 examples/s]7646 examples [00:08, 928.23 examples/s]7740 examples [00:08, 930.51 examples/s]7839 examples [00:08, 946.27 examples/s]7939 examples [00:08, 958.63 examples/s]8035 examples [00:08, 906.17 examples/s]8127 examples [00:08, 894.56 examples/s]8220 examples [00:08, 904.36 examples/s]8319 examples [00:08, 928.28 examples/s]8416 examples [00:09, 939.05 examples/s]8511 examples [00:09, 934.38 examples/s]8613 examples [00:09, 957.94 examples/s]8710 examples [00:09, 926.07 examples/s]8806 examples [00:09, 933.27 examples/s]8905 examples [00:09, 949.19 examples/s]9001 examples [00:09, 942.95 examples/s]9097 examples [00:09, 943.81 examples/s]9192 examples [00:09, 930.57 examples/s]9288 examples [00:10, 938.10 examples/s]9385 examples [00:10, 947.05 examples/s]9480 examples [00:10, 942.68 examples/s]9575 examples [00:10, 908.14 examples/s]9669 examples [00:10, 916.94 examples/s]9761 examples [00:10, 916.64 examples/s]9853 examples [00:10, 917.51 examples/s]9945 examples [00:10, 915.77 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO90ICF/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteO90ICF/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe267fd2950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe267fd2950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe267fd2950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe1ed47ef28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe1ed486f28>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe267fd2950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe267fd2950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe267fd2950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe1ed38d588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe24e390048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe1df9c62f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe1df9c62f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fe1df9c62f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe1df9c6400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe1df9c6400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe1df9c6400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5fb2e153c4a1ca0fbace0d470ced4c8c44072ef0cb589c7817bdc2c38ab502df
  Stored in directory: /tmp/pip-ephem-wheel-cache-hr6b2n2_/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<49:45:44, 4.81kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<35:04:00, 6.83kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<24:35:52, 9.73kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<17:13:09, 13.9kB/s].vector_cache/glove.6B.zip:   0%|          | 3.57M/862M [00:02<12:01:08, 19.8kB/s].vector_cache/glove.6B.zip:   1%|          | 7.71M/862M [00:02<8:22:28, 28.3kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:02<5:49:53, 40.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:02<4:03:24, 57.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:02<2:49:56, 82.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:02<1:58:25, 118kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.5M/862M [00:02<1:22:36, 168kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:02<57:34, 240kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:03<40:14, 341kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:03<28:05, 486kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:03<19:41, 691kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:03<13:47, 980kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<11:04, 1.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<09:38, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<08:40, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:06<06:27, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<04:40, 2.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:07<31:40, 422kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<23:33, 567kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:08<16:48, 793kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:09<14:49, 897kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<11:44, 1.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<08:32, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<09:06, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<07:43, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:11<05:42, 2.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<04:09, 3.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:13<1:40:41, 131kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<1:13:09, 180kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:13<51:45, 254kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:13<36:17, 361kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:15<32:56, 397kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<24:10, 541kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<17:09, 761kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<12:09, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<57:37, 226kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:17<41:39, 312kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:17<29:25, 441kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:19<23:37, 548kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<17:51, 725kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:19<12:46, 1.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:21<11:58, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<09:42, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:21<07:06, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:23<07:59, 1.60MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<06:54, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:23<05:05, 2.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<06:35, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:25<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:25<04:28, 2.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:07, 2.07MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:54, 1.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:29, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:52, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:24, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<04:04, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:47, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:05, 2.46MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<03:50, 3.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<02:51, 4.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<48:57, 255kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<35:31, 351kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<25:08, 495kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<20:29, 605kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<16:54, 733kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<12:22, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:55, 1.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<09:33, 1.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<07:56, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<05:50, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<06:56, 1.77MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:07, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:35, 2.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<06:05, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<05:31, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:10, 2.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:18, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<04:01, 3.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:39, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:27, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<05:03, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<03:39, 3.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<14:32, 825kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<11:24, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<08:16, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:34, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<08:29, 1.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:32, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:43, 2.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<10:03, 1.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<08:17, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<06:02, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:59, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<06:07, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<04:32, 2.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:55, 1.98MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:20, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<04:02, 2.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:34, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:07, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<03:52, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<05:26, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:01, 2.31MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<03:48, 3.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:09, 1.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<04:47, 2.40MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<03:35, 3.19MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:37, 2.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:07, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:50, 2.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:21, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:05, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:44, 2.39MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<03:29, 3.24MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:51, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:00, 1.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:29, 2.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:44, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<06:21, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:57, 2.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<03:36, 3.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<08:08, 1.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<06:53, 1.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:05, 2.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<06:09, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:28, 2.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:06, 2.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:28, 2.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<04:58, 2.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:43, 2.96MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<02:44, 4.00MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<1:14:28, 147kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<53:03, 206kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<37:20, 293kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<28:38, 380kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<21:09, 515kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<15:03, 721kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<13:04, 828kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<10:03, 1.08MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<07:34, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<05:25, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<10:21, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<09:29, 1.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<07:06, 1.51MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:10, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<06:21, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:29, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:02, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<05:16, 2.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<04:54, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<03:44, 2.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:52, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<05:34, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:21, 2.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<03:13, 3.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:55, 1.77MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:15, 1.99MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<03:54, 2.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:08, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:40, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<03:30, 2.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:54, 2.11MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<05:33, 1.86MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:25, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<03:12, 3.20MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<09:49, 1.04MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<08:58, 1.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<06:49, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<04:52, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<42:33, 240kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<30:49, 331kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<21:46, 467kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<17:34, 577kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<13:20, 759kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<09:34, 1.05MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<09:01, 1.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<08:24, 1.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<06:23, 1.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<04:34, 2.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<10:56:52, 15.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<7:38:32, 21.7kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<5:21:59, 30.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<3:46:15, 43.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<2:38:10, 62.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<1:52:41, 87.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<1:19:49, 123kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<55:56, 176kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<41:21, 237kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<29:56, 327kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<21:09, 461kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<17:03, 570kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<12:55, 752kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<09:16, 1.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<08:44, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<08:05, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<05:50, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<04:56, 1.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<03:46, 2.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<02:44, 3.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<19:51, 479kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<14:52, 639kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<10:37, 893kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<09:38, 980kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<07:42, 1.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<05:35, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:06, 1.54MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<05:13, 1.79MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<03:53, 2.40MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:54, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<05:21, 1.74MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:03, 3.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<8:53:34, 17.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<6:14:11, 24.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<4:21:25, 35.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<3:04:25, 49.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<2:09:47, 70.6kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<1:30:56, 101kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<1:03:28, 143kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<1:18:44, 116kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<56:58, 160kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<40:18, 225kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<28:08, 321kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<1:03:34, 142kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<45:24, 199kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<31:55, 282kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<24:21, 368kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<17:58, 499kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<12:46, 699kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<10:59, 809kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<09:31, 934kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<07:02, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:04, 1.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<06:21, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:21, 1.64MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:58, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:49, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:16, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:10, 2.74MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:17, 2.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:53, 2.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<02:54, 2.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<04:04, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:44, 2.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:48, 3.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<03:59, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<03:40, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<02:44, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:37, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<02:44, 3.08MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<03:35, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<02:43, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:33, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<02:41, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<03:50, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:31, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<02:40, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:48, 2.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:30, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<02:39, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<03:46, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<03:28, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<02:38, 3.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:44, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:18, 2.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<02:30, 3.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<01:50, 4.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<31:34, 254kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<23:44, 337kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<16:55, 472kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<11:57, 666kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<10:52, 730kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<08:25, 941kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<06:05, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<06:04, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<05:03, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<03:43, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<04:25, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<03:55, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<02:56, 2.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<03:51, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<03:29, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<02:38, 2.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<03:39, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<03:20, 2.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<02:31, 3.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:33, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<03:15, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<02:28, 3.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:30, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<03:13, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:25, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<02:22, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:25, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<03:08, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<02:23, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:23, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:08, 2.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<02:22, 3.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:22, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:05, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<02:18, 3.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<03:19, 2.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:03, 2.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<02:19, 3.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<03:18, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:03, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<02:18, 3.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<03:00, 2.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:16, 3.07MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<03:14, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<02:58, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:14, 3.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<03:11, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<02:50, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<02:21, 2.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<01:42, 3.99MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<11:50, 578kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<09:42, 704kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<07:06, 960kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:01, 1.35MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<09:01, 751kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<07:00, 965kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:03, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<05:05, 1.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<04:15, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:08, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<03:45, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:18, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:27, 2.68MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<03:16, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:52, 2.29MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:08, 3.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<01:34, 4.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<27:12, 239kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<19:41, 329kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<13:53, 465kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<11:11, 574kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<08:29, 756kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<06:05, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<05:44, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<04:40, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:25, 1.85MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<03:52, 1.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<04:02, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:07, 2.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:14, 2.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<08:05, 770kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<06:18, 984kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:32, 1.37MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<04:36, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<03:51, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:49, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:24, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<03:01, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:13, 2.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:59, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<03:20, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:38, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<02:47, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:34, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<01:56, 3.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<01:52, 3.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:41, 2.15MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:28, 2.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<01:51, 3.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:39, 2.16MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:27, 2.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<01:51, 3.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:37, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<02:25, 2.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<01:50, 3.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:36, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<02:59, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<02:20, 2.38MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<01:42, 3.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:28, 1.59MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<03:01, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:16, 2.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<01:38, 3.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<30:36, 179kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<22:33, 242kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<15:58, 341kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<11:17, 481kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<09:07, 592kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<06:55, 778kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<04:57, 1.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<04:41, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<03:49, 1.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:47, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<03:11, 1.65MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:46, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:04, 2.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:39, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:24, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<01:47, 2.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:27, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<02:47, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:12, 2.31MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<01:37, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<03:03, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:57, 2.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:32, 1.96MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:11, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<01:34, 3.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<04:19, 1.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<03:31, 1.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:34, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:55, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:33, 1.90MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<01:52, 2.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:26, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:11, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<01:39, 2.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<02:15, 2.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:03, 2.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<01:33, 3.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<02:11, 2.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<01:59, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<01:30, 3.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<01:58, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:29, 3.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<02:04, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<01:54, 2.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<01:26, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:03, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<01:53, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:24, 3.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<01:51, 2.34MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:23, 3.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<01:50, 2.33MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:22, 3.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<01:57, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<01:48, 2.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:21, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<01:55, 2.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<01:46, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<01:19, 3.11MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<01:53, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<01:44, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:19, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<01:52, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<01:43, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:17, 3.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<01:50, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<01:41, 2.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:15, 3.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<00:55, 4.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<46:50, 83.0kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<33:08, 117kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<23:08, 167kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<16:56, 225kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<12:14, 312kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<08:36, 440kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<06:51, 547kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<05:06, 732kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<03:39, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:34, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<16:37, 222kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<12:23, 297kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<08:47, 417kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<06:09, 590kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<05:42, 633kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<04:22, 826kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<03:06, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<02:59, 1.19MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:49, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<02:07, 1.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:31, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:14, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:51, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:23, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<01:00, 3.39MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<14:26, 236kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<10:26, 326kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:52<07:19, 461kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<05:51, 570kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<04:26, 752kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:09, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:58, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:44, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:29, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:55, 1.67MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<01:40, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:13, 2.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:35, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:25, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:04, 2.89MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:28, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<01:20, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:00, 3.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:24, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:17, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<00:57, 3.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:21, 2.17MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:14, 2.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<00:56, 3.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:19, 2.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:10, 2.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<00:53, 3.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<00:39, 4.28MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<11:42, 239kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<08:45, 319kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<06:13, 446kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<04:21, 631kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<04:01, 676kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<03:05, 877kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:13, 1.21MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<02:09, 1.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:46, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:18, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:30, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:19, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<00:58, 2.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<01:16, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<01:08, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<00:51, 2.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<01:10, 2.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:03, 2.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<00:48, 3.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:06, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<01:01, 2.32MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<00:45, 3.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<01:04, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<00:59, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<00:44, 3.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<00:57, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:42, 3.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:00, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<00:53, 2.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<00:40, 3.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:29, 4.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<08:17, 254kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<05:59, 350kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<04:11, 495kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<03:22, 605kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<02:33, 793kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:49, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:43, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:24, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<01:01, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:08, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:59, 1.90MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:44, 2.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<00:56, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:50, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:37, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:50, 2.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:46, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<00:34, 3.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:47, 2.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:43, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<00:32, 3.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:45, 2.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:51, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:40, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:29, 3.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:55, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:35, 2.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:45, 1.96MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:40, 2.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:30, 2.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:40, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:37, 2.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:27, 3.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<00:34, 2.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:25, 3.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:35, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:32, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:24, 3.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:33, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:31, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:22, 3.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:36, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:28, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:29, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:27, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:20, 3.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:27, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:24, 2.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:17, 3.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:12, 4.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<03:56, 239kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<02:49, 330kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:56, 467kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<01:30, 576kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<01:08, 759kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:47, 1.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:43, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:35, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:24, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:27, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:27, 1.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:21, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:14, 2.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:27, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:23, 1.69MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:16, 2.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:13<00:19, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:17, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:12, 2.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:15<00:15, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:13, 2.24MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:09, 3.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:13, 2.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:14, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:11, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:10, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:19<00:07, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<00:08, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:05, 3.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:06, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:04, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:04, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:02, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:02, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:01, 3.08MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 2.34MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 3.08MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<18:04:41,  6.15it/s]  0%|          | 877/400000 [00:00<12:37:50,  8.78it/s]  0%|          | 1708/400000 [00:00<8:49:37, 12.53it/s]  1%|          | 2564/400000 [00:00<6:10:10, 17.89it/s]  1%|          | 3335/400000 [00:00<4:18:52, 25.54it/s]  1%|          | 4083/400000 [00:00<3:01:08, 36.43it/s]  1%|          | 4873/400000 [00:00<2:06:47, 51.94it/s]  1%|▏         | 5606/400000 [00:00<1:28:51, 73.97it/s]  2%|▏         | 6359/400000 [00:00<1:02:20, 105.23it/s]  2%|▏         | 7224/400000 [00:01<43:46, 149.55it/s]    2%|▏         | 8020/400000 [00:01<30:49, 211.94it/s]  2%|▏         | 8848/400000 [00:01<21:46, 299.48it/s]  2%|▏         | 9647/400000 [00:01<15:27, 421.06it/s]  3%|▎         | 10440/400000 [00:01<11:03, 587.45it/s]  3%|▎         | 11259/400000 [00:01<07:57, 814.19it/s]  3%|▎         | 12063/400000 [00:01<05:48, 1114.72it/s]  3%|▎         | 12942/400000 [00:01<04:16, 1510.34it/s]  3%|▎         | 13763/400000 [00:01<03:13, 1999.51it/s]  4%|▎         | 14583/400000 [00:01<02:29, 2581.26it/s]  4%|▍         | 15430/400000 [00:02<01:57, 3261.39it/s]  4%|▍         | 16255/400000 [00:02<01:36, 3967.00it/s]  4%|▍         | 17085/400000 [00:02<01:21, 4703.58it/s]  4%|▍         | 17981/400000 [00:02<01:09, 5484.73it/s]  5%|▍         | 18825/400000 [00:02<01:03, 6020.75it/s]  5%|▍         | 19657/400000 [00:02<00:57, 6563.90it/s]  5%|▌         | 20511/400000 [00:02<00:53, 7053.49it/s]  5%|▌         | 21404/400000 [00:02<00:50, 7527.86it/s]  6%|▌         | 22257/400000 [00:02<00:48, 7724.48it/s]  6%|▌         | 23101/400000 [00:02<00:48, 7816.29it/s]  6%|▌         | 23940/400000 [00:03<00:47, 7977.15it/s]  6%|▌         | 24774/400000 [00:03<00:48, 7776.08it/s]  6%|▋         | 25605/400000 [00:03<00:47, 7926.49it/s]  7%|▋         | 26458/400000 [00:03<00:46, 8098.08it/s]  7%|▋         | 27350/400000 [00:03<00:44, 8327.33it/s]  7%|▋         | 28195/400000 [00:03<00:45, 8254.60it/s]  7%|▋         | 29029/400000 [00:03<00:46, 8014.71it/s]  7%|▋         | 29838/400000 [00:03<00:46, 8019.85it/s]  8%|▊         | 30665/400000 [00:03<00:45, 8090.67it/s]  8%|▊         | 31525/400000 [00:04<00:44, 8234.27it/s]  8%|▊         | 32397/400000 [00:04<00:43, 8370.78it/s]  8%|▊         | 33301/400000 [00:04<00:42, 8560.06it/s]  9%|▊         | 34198/400000 [00:04<00:42, 8676.78it/s]  9%|▉         | 35069/400000 [00:04<00:42, 8561.79it/s]  9%|▉         | 35928/400000 [00:04<00:42, 8470.27it/s]  9%|▉         | 36809/400000 [00:04<00:42, 8568.33it/s]  9%|▉         | 37668/400000 [00:04<00:43, 8417.53it/s] 10%|▉         | 38512/400000 [00:04<00:43, 8263.52it/s] 10%|▉         | 39341/400000 [00:04<00:44, 8110.51it/s] 10%|█         | 40181/400000 [00:05<00:43, 8193.09it/s] 10%|█         | 41035/400000 [00:05<00:43, 8293.55it/s] 10%|█         | 41866/400000 [00:05<00:43, 8214.10it/s] 11%|█         | 42689/400000 [00:05<00:44, 8065.70it/s] 11%|█         | 43502/400000 [00:05<00:44, 8081.94it/s] 11%|█         | 44361/400000 [00:05<00:43, 8227.01it/s] 11%|█▏        | 45186/400000 [00:05<00:43, 8205.09it/s] 12%|█▏        | 46008/400000 [00:05<00:43, 8204.66it/s] 12%|█▏        | 46892/400000 [00:05<00:42, 8383.63it/s] 12%|█▏        | 47732/400000 [00:05<00:43, 8183.08it/s] 12%|█▏        | 48556/400000 [00:06<00:42, 8199.38it/s] 12%|█▏        | 49460/400000 [00:06<00:41, 8433.81it/s] 13%|█▎        | 50307/400000 [00:06<00:41, 8368.66it/s] 13%|█▎        | 51151/400000 [00:06<00:41, 8387.51it/s] 13%|█▎        | 51992/400000 [00:06<00:42, 8156.52it/s] 13%|█▎        | 52811/400000 [00:06<00:42, 8083.12it/s] 13%|█▎        | 53622/400000 [00:06<00:43, 7990.09it/s] 14%|█▎        | 54450/400000 [00:06<00:42, 8074.40it/s] 14%|█▍        | 55310/400000 [00:06<00:41, 8224.43it/s] 14%|█▍        | 56135/400000 [00:06<00:42, 8090.05it/s] 14%|█▍        | 56995/400000 [00:07<00:41, 8235.98it/s] 14%|█▍        | 57876/400000 [00:07<00:40, 8398.09it/s] 15%|█▍        | 58718/400000 [00:07<00:43, 7804.95it/s] 15%|█▍        | 59595/400000 [00:07<00:42, 8070.51it/s] 15%|█▌        | 60412/400000 [00:07<00:42, 7954.82it/s] 15%|█▌        | 61228/400000 [00:07<00:42, 8014.55it/s] 16%|█▌        | 62064/400000 [00:07<00:41, 8114.33it/s] 16%|█▌        | 62966/400000 [00:07<00:40, 8364.46it/s] 16%|█▌        | 63822/400000 [00:07<00:39, 8421.76it/s] 16%|█▌        | 64668/400000 [00:08<00:41, 8093.71it/s] 16%|█▋        | 65502/400000 [00:08<00:40, 8165.40it/s] 17%|█▋        | 66323/400000 [00:08<00:40, 8146.41it/s] 17%|█▋        | 67212/400000 [00:08<00:39, 8352.92it/s] 17%|█▋        | 68168/400000 [00:08<00:38, 8679.13it/s] 17%|█▋        | 69042/400000 [00:08<00:39, 8372.36it/s] 17%|█▋        | 69888/400000 [00:08<00:39, 8398.28it/s] 18%|█▊        | 70733/400000 [00:08<00:40, 8220.17it/s] 18%|█▊        | 71642/400000 [00:08<00:38, 8461.31it/s] 18%|█▊        | 72493/400000 [00:08<00:38, 8419.60it/s] 18%|█▊        | 73339/400000 [00:09<00:39, 8344.04it/s] 19%|█▊        | 74182/400000 [00:09<00:38, 8369.51it/s] 19%|█▉        | 75068/400000 [00:09<00:38, 8508.33it/s] 19%|█▉        | 76001/400000 [00:09<00:37, 8738.83it/s] 19%|█▉        | 76878/400000 [00:09<00:37, 8710.58it/s] 19%|█▉        | 77752/400000 [00:09<00:38, 8439.56it/s] 20%|█▉        | 78600/400000 [00:09<00:38, 8294.38it/s] 20%|█▉        | 79433/400000 [00:09<00:39, 8136.75it/s] 20%|██        | 80250/400000 [00:09<00:39, 8014.59it/s] 20%|██        | 81092/400000 [00:10<00:39, 8130.98it/s] 20%|██        | 81908/400000 [00:10<00:40, 7768.42it/s] 21%|██        | 82710/400000 [00:10<00:40, 7839.92it/s] 21%|██        | 83519/400000 [00:10<00:40, 7894.45it/s] 21%|██        | 84392/400000 [00:10<00:38, 8127.64it/s] 21%|██▏       | 85254/400000 [00:10<00:38, 8266.83it/s] 22%|██▏       | 86084/400000 [00:10<00:38, 8200.32it/s] 22%|██▏       | 86907/400000 [00:10<00:38, 8131.87it/s] 22%|██▏       | 87723/400000 [00:10<00:38, 8139.47it/s] 22%|██▏       | 88598/400000 [00:10<00:37, 8313.48it/s] 22%|██▏       | 89432/400000 [00:11<00:37, 8303.61it/s] 23%|██▎       | 90264/400000 [00:11<00:38, 8124.03it/s] 23%|██▎       | 91079/400000 [00:11<00:39, 7920.82it/s] 23%|██▎       | 91882/400000 [00:11<00:38, 7953.07it/s] 23%|██▎       | 92736/400000 [00:11<00:37, 8117.01it/s] 23%|██▎       | 93550/400000 [00:11<00:38, 7939.54it/s] 24%|██▎       | 94347/400000 [00:11<00:38, 7863.71it/s] 24%|██▍       | 95136/400000 [00:11<00:39, 7740.83it/s] 24%|██▍       | 95983/400000 [00:11<00:38, 7943.95it/s] 24%|██▍       | 96780/400000 [00:11<00:39, 7770.70it/s] 24%|██▍       | 97560/400000 [00:12<00:39, 7621.16it/s] 25%|██▍       | 98325/400000 [00:12<00:39, 7549.21it/s] 25%|██▍       | 99082/400000 [00:12<00:41, 7328.40it/s] 25%|██▍       | 99842/400000 [00:12<00:40, 7406.73it/s] 25%|██▌       | 100585/400000 [00:12<00:41, 7250.40it/s] 25%|██▌       | 101313/400000 [00:12<00:41, 7180.67it/s] 26%|██▌       | 102033/400000 [00:12<00:41, 7165.56it/s] 26%|██▌       | 102785/400000 [00:12<00:40, 7266.76it/s] 26%|██▌       | 103514/400000 [00:12<00:40, 7272.76it/s] 26%|██▌       | 104251/400000 [00:13<00:40, 7301.29it/s] 26%|██▌       | 104982/400000 [00:13<00:40, 7206.99it/s] 26%|██▋       | 105704/400000 [00:13<00:42, 6963.31it/s] 27%|██▋       | 106403/400000 [00:13<00:42, 6929.66it/s] 27%|██▋       | 107158/400000 [00:13<00:41, 7104.15it/s] 27%|██▋       | 107895/400000 [00:13<00:40, 7180.58it/s] 27%|██▋       | 108630/400000 [00:13<00:40, 7229.14it/s] 27%|██▋       | 109397/400000 [00:13<00:39, 7355.92it/s] 28%|██▊       | 110144/400000 [00:13<00:39, 7387.54it/s] 28%|██▊       | 110887/400000 [00:13<00:39, 7399.96it/s] 28%|██▊       | 111699/400000 [00:14<00:37, 7601.05it/s] 28%|██▊       | 112529/400000 [00:14<00:36, 7797.45it/s] 28%|██▊       | 113390/400000 [00:14<00:35, 8022.64it/s] 29%|██▊       | 114232/400000 [00:14<00:35, 8137.47it/s] 29%|██▉       | 115049/400000 [00:14<00:35, 7933.60it/s] 29%|██▉       | 115911/400000 [00:14<00:34, 8126.17it/s] 29%|██▉       | 116773/400000 [00:14<00:34, 8266.26it/s] 29%|██▉       | 117603/400000 [00:14<00:34, 8160.64it/s] 30%|██▉       | 118457/400000 [00:14<00:34, 8270.10it/s] 30%|██▉       | 119287/400000 [00:14<00:34, 8084.77it/s] 30%|███       | 120098/400000 [00:15<00:35, 7902.67it/s] 30%|███       | 120913/400000 [00:15<00:35, 7973.76it/s] 30%|███       | 121713/400000 [00:15<00:34, 7954.47it/s] 31%|███       | 122569/400000 [00:15<00:34, 8124.76it/s] 31%|███       | 123384/400000 [00:15<00:34, 8077.75it/s] 31%|███       | 124238/400000 [00:15<00:33, 8207.60it/s] 31%|███▏      | 125133/400000 [00:15<00:32, 8416.76it/s] 31%|███▏      | 125978/400000 [00:15<00:32, 8416.79it/s] 32%|███▏      | 126822/400000 [00:15<00:32, 8359.36it/s] 32%|███▏      | 127713/400000 [00:15<00:31, 8515.33it/s] 32%|███▏      | 128567/400000 [00:16<00:31, 8490.45it/s] 32%|███▏      | 129418/400000 [00:16<00:32, 8219.75it/s] 33%|███▎      | 130253/400000 [00:16<00:32, 8256.77it/s] 33%|███▎      | 131147/400000 [00:16<00:31, 8449.48it/s] 33%|███▎      | 132032/400000 [00:16<00:31, 8563.57it/s] 33%|███▎      | 132891/400000 [00:16<00:31, 8366.79it/s] 33%|███▎      | 133746/400000 [00:16<00:31, 8419.28it/s] 34%|███▎      | 134590/400000 [00:16<00:31, 8363.03it/s] 34%|███▍      | 135451/400000 [00:16<00:31, 8434.15it/s] 34%|███▍      | 136296/400000 [00:16<00:31, 8355.87it/s] 34%|███▍      | 137133/400000 [00:17<00:32, 8112.72it/s] 34%|███▍      | 137947/400000 [00:17<00:34, 7696.92it/s] 35%|███▍      | 138763/400000 [00:17<00:33, 7827.30it/s] 35%|███▍      | 139621/400000 [00:17<00:32, 8035.70it/s] 35%|███▌      | 140525/400000 [00:17<00:31, 8311.12it/s] 35%|███▌      | 141362/400000 [00:17<00:31, 8302.27it/s] 36%|███▌      | 142197/400000 [00:17<00:31, 8167.46it/s] 36%|███▌      | 143038/400000 [00:17<00:31, 8237.40it/s] 36%|███▌      | 143865/400000 [00:17<00:31, 8175.75it/s] 36%|███▌      | 144685/400000 [00:18<00:31, 8178.76it/s] 36%|███▋      | 145505/400000 [00:18<00:32, 7870.24it/s] 37%|███▋      | 146296/400000 [00:18<00:32, 7728.24it/s] 37%|███▋      | 147114/400000 [00:18<00:32, 7857.13it/s] 37%|███▋      | 147969/400000 [00:18<00:31, 8048.65it/s] 37%|███▋      | 148834/400000 [00:18<00:30, 8217.12it/s] 37%|███▋      | 149685/400000 [00:18<00:30, 8302.56it/s] 38%|███▊      | 150518/400000 [00:18<00:30, 8281.79it/s] 38%|███▊      | 151348/400000 [00:18<00:30, 8145.71it/s] 38%|███▊      | 152172/400000 [00:18<00:30, 8172.50it/s] 38%|███▊      | 152991/400000 [00:19<00:30, 8016.99it/s] 38%|███▊      | 153795/400000 [00:19<00:31, 7910.25it/s] 39%|███▊      | 154599/400000 [00:19<00:30, 7948.44it/s] 39%|███▉      | 155442/400000 [00:19<00:30, 8085.59it/s] 39%|███▉      | 156322/400000 [00:19<00:29, 8285.71it/s] 39%|███▉      | 157153/400000 [00:19<00:29, 8258.11it/s] 39%|███▉      | 157981/400000 [00:19<00:29, 8249.63it/s] 40%|███▉      | 158829/400000 [00:19<00:29, 8315.26it/s] 40%|███▉      | 159662/400000 [00:19<00:29, 8259.15it/s] 40%|████      | 160489/400000 [00:19<00:29, 8007.52it/s] 40%|████      | 161317/400000 [00:20<00:29, 8085.67it/s] 41%|████      | 162148/400000 [00:20<00:29, 8149.72it/s] 41%|████      | 162965/400000 [00:20<00:29, 8072.36it/s] 41%|████      | 163794/400000 [00:20<00:29, 8135.29it/s] 41%|████      | 164636/400000 [00:20<00:28, 8218.37it/s] 41%|████▏     | 165488/400000 [00:20<00:28, 8292.35it/s] 42%|████▏     | 166318/400000 [00:20<00:28, 8080.18it/s] 42%|████▏     | 167145/400000 [00:20<00:28, 8135.57it/s] 42%|████▏     | 168007/400000 [00:20<00:28, 8273.08it/s] 42%|████▏     | 168866/400000 [00:20<00:27, 8364.29it/s] 42%|████▏     | 169754/400000 [00:21<00:27, 8510.33it/s] 43%|████▎     | 170607/400000 [00:21<00:28, 8146.70it/s] 43%|████▎     | 171440/400000 [00:21<00:27, 8199.63it/s] 43%|████▎     | 172306/400000 [00:21<00:27, 8330.66it/s] 43%|████▎     | 173142/400000 [00:21<00:27, 8310.97it/s] 43%|████▎     | 173975/400000 [00:21<00:27, 8230.65it/s] 44%|████▎     | 174800/400000 [00:21<00:28, 8005.72it/s] 44%|████▍     | 175646/400000 [00:21<00:27, 8135.27it/s] 44%|████▍     | 176481/400000 [00:21<00:27, 8197.67it/s] 44%|████▍     | 177310/400000 [00:22<00:27, 8224.59it/s] 45%|████▍     | 178177/400000 [00:22<00:26, 8353.15it/s] 45%|████▍     | 179014/400000 [00:22<00:27, 8145.64it/s] 45%|████▍     | 179832/400000 [00:22<00:27, 8153.85it/s] 45%|████▌     | 180704/400000 [00:22<00:26, 8313.77it/s] 45%|████▌     | 181566/400000 [00:22<00:25, 8403.00it/s] 46%|████▌     | 182452/400000 [00:22<00:25, 8534.41it/s] 46%|████▌     | 183307/400000 [00:22<00:26, 8176.27it/s] 46%|████▌     | 184139/400000 [00:22<00:26, 8217.08it/s] 46%|████▌     | 184990/400000 [00:22<00:25, 8301.00it/s] 46%|████▋     | 185824/400000 [00:23<00:25, 8311.90it/s] 47%|████▋     | 186678/400000 [00:23<00:25, 8377.51it/s] 47%|████▋     | 187518/400000 [00:23<00:26, 8159.27it/s] 47%|████▋     | 188337/400000 [00:23<00:26, 8094.89it/s] 47%|████▋     | 189149/400000 [00:23<00:26, 8043.10it/s] 48%|████▊     | 190040/400000 [00:23<00:25, 8283.12it/s] 48%|████▊     | 190891/400000 [00:23<00:25, 8348.94it/s] 48%|████▊     | 191728/400000 [00:23<00:25, 8152.13it/s] 48%|████▊     | 192618/400000 [00:23<00:24, 8361.32it/s] 48%|████▊     | 193492/400000 [00:23<00:24, 8469.82it/s] 49%|████▊     | 194348/400000 [00:24<00:24, 8494.59it/s] 49%|████▉     | 195224/400000 [00:24<00:23, 8572.40it/s] 49%|████▉     | 196083/400000 [00:24<00:23, 8536.33it/s] 49%|████▉     | 196969/400000 [00:24<00:23, 8628.47it/s] 49%|████▉     | 197834/400000 [00:24<00:23, 8632.76it/s] 50%|████▉     | 198719/400000 [00:24<00:23, 8695.45it/s] 50%|████▉     | 199629/400000 [00:24<00:22, 8811.12it/s] 50%|█████     | 200511/400000 [00:24<00:23, 8434.81it/s] 50%|█████     | 201370/400000 [00:24<00:23, 8479.31it/s] 51%|█████     | 202221/400000 [00:24<00:23, 8361.75it/s] 51%|█████     | 203117/400000 [00:25<00:23, 8530.52it/s] 51%|█████     | 203973/400000 [00:25<00:23, 8459.59it/s] 51%|█████     | 204821/400000 [00:25<00:23, 8324.98it/s] 51%|█████▏    | 205718/400000 [00:25<00:22, 8507.42it/s] 52%|█████▏    | 206595/400000 [00:25<00:22, 8584.04it/s] 52%|█████▏    | 207487/400000 [00:25<00:22, 8681.72it/s] 52%|█████▏    | 208391/400000 [00:25<00:21, 8786.16it/s] 52%|█████▏    | 209271/400000 [00:25<00:22, 8546.13it/s] 53%|█████▎    | 210180/400000 [00:25<00:21, 8700.92it/s] 53%|█████▎    | 211074/400000 [00:26<00:21, 8764.06it/s] 53%|█████▎    | 211990/400000 [00:26<00:21, 8877.71it/s] 53%|█████▎    | 212880/400000 [00:26<00:21, 8844.50it/s] 53%|█████▎    | 213766/400000 [00:26<00:21, 8570.15it/s] 54%|█████▎    | 214648/400000 [00:26<00:21, 8642.40it/s] 54%|█████▍    | 215515/400000 [00:26<00:22, 8272.33it/s] 54%|█████▍    | 216347/400000 [00:26<00:22, 8097.18it/s] 54%|█████▍    | 217181/400000 [00:26<00:22, 8168.06it/s] 55%|█████▍    | 218046/400000 [00:26<00:21, 8305.00it/s] 55%|█████▍    | 218916/400000 [00:26<00:21, 8419.41it/s] 55%|█████▍    | 219825/400000 [00:27<00:20, 8607.66it/s] 55%|█████▌    | 220746/400000 [00:27<00:20, 8779.29it/s] 55%|█████▌    | 221627/400000 [00:27<00:20, 8734.58it/s] 56%|█████▌    | 222503/400000 [00:27<00:20, 8630.77it/s] 56%|█████▌    | 223378/400000 [00:27<00:20, 8664.84it/s] 56%|█████▌    | 224310/400000 [00:27<00:19, 8850.30it/s] 56%|█████▋    | 225197/400000 [00:27<00:19, 8847.33it/s] 57%|█████▋    | 226083/400000 [00:27<00:20, 8681.32it/s] 57%|█████▋    | 226953/400000 [00:27<00:20, 8648.90it/s] 57%|█████▋    | 227836/400000 [00:27<00:19, 8699.79it/s] 57%|█████▋    | 228715/400000 [00:28<00:19, 8724.95it/s] 57%|█████▋    | 229597/400000 [00:28<00:19, 8753.08it/s] 58%|█████▊    | 230473/400000 [00:28<00:19, 8563.22it/s] 58%|█████▊    | 231331/400000 [00:28<00:19, 8458.44it/s] 58%|█████▊    | 232256/400000 [00:28<00:19, 8680.01it/s] 58%|█████▊    | 233140/400000 [00:28<00:19, 8727.22it/s] 59%|█████▊    | 234038/400000 [00:28<00:18, 8799.73it/s] 59%|█████▊    | 234920/400000 [00:28<00:19, 8585.18it/s] 59%|█████▉    | 235781/400000 [00:28<00:19, 8388.26it/s] 59%|█████▉    | 236623/400000 [00:28<00:19, 8376.94it/s] 59%|█████▉    | 237497/400000 [00:29<00:19, 8480.77it/s] 60%|█████▉    | 238347/400000 [00:29<00:19, 8469.47it/s] 60%|█████▉    | 239196/400000 [00:29<00:19, 8309.19it/s] 60%|██████    | 240029/400000 [00:29<00:19, 8251.13it/s] 60%|██████    | 240856/400000 [00:29<00:19, 8071.56it/s] 60%|██████    | 241675/400000 [00:29<00:19, 8106.14it/s] 61%|██████    | 242601/400000 [00:29<00:18, 8420.22it/s] 61%|██████    | 243447/400000 [00:29<00:19, 8106.64it/s] 61%|██████    | 244287/400000 [00:29<00:19, 8190.79it/s] 61%|██████▏   | 245191/400000 [00:30<00:18, 8426.58it/s] 62%|██████▏   | 246048/400000 [00:30<00:18, 8468.85it/s] 62%|██████▏   | 246962/400000 [00:30<00:17, 8656.94it/s] 62%|██████▏   | 247831/400000 [00:30<00:17, 8569.23it/s] 62%|██████▏   | 248761/400000 [00:30<00:17, 8773.83it/s] 62%|██████▏   | 249669/400000 [00:30<00:16, 8861.43it/s] 63%|██████▎   | 250582/400000 [00:30<00:16, 8939.63it/s] 63%|██████▎   | 251478/400000 [00:30<00:16, 8895.35it/s] 63%|██████▎   | 252369/400000 [00:30<00:16, 8684.25it/s] 63%|██████▎   | 253248/400000 [00:30<00:16, 8714.09it/s] 64%|██████▎   | 254154/400000 [00:31<00:16, 8813.92it/s] 64%|██████▍   | 255037/400000 [00:31<00:16, 8668.59it/s] 64%|██████▍   | 255906/400000 [00:31<00:17, 8419.57it/s] 64%|██████▍   | 256751/400000 [00:31<00:17, 8335.80it/s] 64%|██████▍   | 257589/400000 [00:31<00:17, 8348.83it/s] 65%|██████▍   | 258483/400000 [00:31<00:16, 8514.85it/s] 65%|██████▍   | 259363/400000 [00:31<00:16, 8595.21it/s] 65%|██████▌   | 260276/400000 [00:31<00:15, 8747.01it/s] 65%|██████▌   | 261153/400000 [00:31<00:16, 8659.52it/s] 66%|██████▌   | 262021/400000 [00:31<00:16, 8590.78it/s] 66%|██████▌   | 262882/400000 [00:32<00:16, 8470.66it/s] 66%|██████▌   | 263735/400000 [00:32<00:16, 8486.66it/s] 66%|██████▌   | 264585/400000 [00:32<00:16, 8450.63it/s] 66%|██████▋   | 265431/400000 [00:32<00:15, 8440.80it/s] 67%|██████▋   | 266302/400000 [00:32<00:15, 8518.00it/s] 67%|██████▋   | 267224/400000 [00:32<00:15, 8714.58it/s] 67%|██████▋   | 268099/400000 [00:32<00:15, 8724.35it/s] 67%|██████▋   | 268973/400000 [00:32<00:15, 8615.60it/s] 67%|██████▋   | 269836/400000 [00:32<00:15, 8613.96it/s] 68%|██████▊   | 270699/400000 [00:32<00:15, 8618.06it/s] 68%|██████▊   | 271565/400000 [00:33<00:14, 8629.80it/s] 68%|██████▊   | 272429/400000 [00:33<00:15, 8344.45it/s] 68%|██████▊   | 273266/400000 [00:33<00:15, 8087.93it/s] 69%|██████▊   | 274113/400000 [00:33<00:15, 8196.57it/s] 69%|██████▊   | 274945/400000 [00:33<00:15, 8230.43it/s] 69%|██████▉   | 275822/400000 [00:33<00:14, 8384.84it/s] 69%|██████▉   | 276696/400000 [00:33<00:14, 8485.91it/s] 69%|██████▉   | 277547/400000 [00:33<00:14, 8214.59it/s] 70%|██████▉   | 278372/400000 [00:33<00:15, 7965.77it/s] 70%|██████▉   | 279173/400000 [00:34<00:15, 7746.55it/s] 70%|██████▉   | 279970/400000 [00:34<00:15, 7810.51it/s] 70%|███████   | 280819/400000 [00:34<00:14, 8001.01it/s] 70%|███████   | 281704/400000 [00:34<00:14, 8233.73it/s] 71%|███████   | 282558/400000 [00:34<00:14, 8322.54it/s] 71%|███████   | 283449/400000 [00:34<00:13, 8489.60it/s] 71%|███████   | 284354/400000 [00:34<00:13, 8647.95it/s] 71%|███████▏  | 285258/400000 [00:34<00:13, 8760.37it/s] 72%|███████▏  | 286137/400000 [00:34<00:13, 8279.43it/s] 72%|███████▏  | 286973/400000 [00:34<00:13, 8193.14it/s] 72%|███████▏  | 287835/400000 [00:35<00:13, 8314.71it/s] 72%|███████▏  | 288744/400000 [00:35<00:13, 8532.34it/s] 72%|███████▏  | 289602/400000 [00:35<00:12, 8518.94it/s] 73%|███████▎  | 290469/400000 [00:35<00:12, 8562.67it/s] 73%|███████▎  | 291328/400000 [00:35<00:12, 8497.75it/s] 73%|███████▎  | 292180/400000 [00:35<00:12, 8477.29it/s] 73%|███████▎  | 293029/400000 [00:35<00:12, 8466.05it/s] 73%|███████▎  | 293877/400000 [00:35<00:12, 8405.53it/s] 74%|███████▎  | 294744/400000 [00:35<00:12, 8481.41it/s] 74%|███████▍  | 295593/400000 [00:35<00:12, 8340.05it/s] 74%|███████▍  | 296428/400000 [00:36<00:12, 8339.13it/s] 74%|███████▍  | 297263/400000 [00:36<00:12, 8251.51it/s] 75%|███████▍  | 298098/400000 [00:36<00:12, 8280.57it/s] 75%|███████▍  | 299003/400000 [00:36<00:11, 8494.46it/s] 75%|███████▍  | 299855/400000 [00:36<00:11, 8473.44it/s] 75%|███████▌  | 300704/400000 [00:36<00:11, 8283.05it/s] 75%|███████▌  | 301612/400000 [00:36<00:11, 8505.08it/s] 76%|███████▌  | 302470/400000 [00:36<00:11, 8525.55it/s] 76%|███████▌  | 303356/400000 [00:36<00:11, 8622.62it/s] 76%|███████▌  | 304220/400000 [00:36<00:11, 8388.86it/s] 76%|███████▋  | 305118/400000 [00:37<00:11, 8557.81it/s] 76%|███████▋  | 305987/400000 [00:37<00:10, 8594.86it/s] 77%|███████▋  | 306871/400000 [00:37<00:10, 8666.27it/s] 77%|███████▋  | 307830/400000 [00:37<00:10, 8923.00it/s] 77%|███████▋  | 308726/400000 [00:37<00:10, 8761.52it/s] 77%|███████▋  | 309634/400000 [00:37<00:10, 8852.54it/s] 78%|███████▊  | 310577/400000 [00:37<00:09, 9016.87it/s] 78%|███████▊  | 311481/400000 [00:37<00:09, 8927.00it/s] 78%|███████▊  | 312376/400000 [00:37<00:10, 8750.58it/s] 78%|███████▊  | 313254/400000 [00:38<00:10, 8430.10it/s] 79%|███████▊  | 314115/400000 [00:38<00:10, 8481.15it/s] 79%|███████▊  | 314966/400000 [00:38<00:10, 7980.24it/s] 79%|███████▉  | 315775/400000 [00:38<00:10, 8012.23it/s] 79%|███████▉  | 316636/400000 [00:38<00:10, 8182.05it/s] 79%|███████▉  | 317467/400000 [00:38<00:10, 8218.48it/s] 80%|███████▉  | 318359/400000 [00:38<00:09, 8414.93it/s] 80%|███████▉  | 319209/400000 [00:38<00:09, 8438.30it/s] 80%|████████  | 320074/400000 [00:38<00:09, 8499.28it/s] 80%|████████  | 320926/400000 [00:38<00:09, 8309.20it/s] 80%|████████  | 321772/400000 [00:39<00:09, 8352.58it/s] 81%|████████  | 322670/400000 [00:39<00:09, 8528.87it/s] 81%|████████  | 323561/400000 [00:39<00:08, 8637.42it/s] 81%|████████  | 324427/400000 [00:39<00:08, 8608.15it/s] 81%|████████▏ | 325290/400000 [00:39<00:08, 8556.32it/s] 82%|████████▏ | 326147/400000 [00:39<00:08, 8491.41it/s] 82%|████████▏ | 327081/400000 [00:39<00:08, 8727.23it/s] 82%|████████▏ | 327956/400000 [00:39<00:08, 8660.84it/s] 82%|████████▏ | 328824/400000 [00:39<00:08, 8626.52it/s] 82%|████████▏ | 329688/400000 [00:39<00:08, 8336.37it/s] 83%|████████▎ | 330525/400000 [00:40<00:08, 8337.45it/s] 83%|████████▎ | 331414/400000 [00:40<00:08, 8493.29it/s] 83%|████████▎ | 332266/400000 [00:40<00:08, 8461.80it/s] 83%|████████▎ | 333134/400000 [00:40<00:07, 8525.37it/s] 84%|████████▎ | 334040/400000 [00:40<00:07, 8676.24it/s] 84%|████████▎ | 334929/400000 [00:40<00:07, 8737.93it/s] 84%|████████▍ | 335866/400000 [00:40<00:07, 8917.03it/s] 84%|████████▍ | 336760/400000 [00:40<00:07, 8910.24it/s] 84%|████████▍ | 337653/400000 [00:40<00:06, 8907.07it/s] 85%|████████▍ | 338545/400000 [00:40<00:07, 8519.24it/s] 85%|████████▍ | 339402/400000 [00:41<00:07, 8320.29it/s] 85%|████████▌ | 340242/400000 [00:41<00:07, 8342.02it/s] 85%|████████▌ | 341123/400000 [00:41<00:06, 8476.77it/s] 85%|████████▌ | 341994/400000 [00:41<00:06, 8544.34it/s] 86%|████████▌ | 342851/400000 [00:41<00:06, 8346.09it/s] 86%|████████▌ | 343694/400000 [00:41<00:06, 8368.43it/s] 86%|████████▌ | 344603/400000 [00:41<00:06, 8570.17it/s] 86%|████████▋ | 345463/400000 [00:41<00:06, 8545.23it/s] 87%|████████▋ | 346320/400000 [00:41<00:06, 8445.41it/s] 87%|████████▋ | 347166/400000 [00:42<00:06, 8370.87it/s] 87%|████████▋ | 348028/400000 [00:42<00:06, 8441.40it/s] 87%|████████▋ | 348928/400000 [00:42<00:05, 8600.03it/s] 87%|████████▋ | 349790/400000 [00:42<00:05, 8492.46it/s] 88%|████████▊ | 350688/400000 [00:42<00:05, 8630.08it/s] 88%|████████▊ | 351553/400000 [00:42<00:05, 8505.51it/s] 88%|████████▊ | 352434/400000 [00:42<00:05, 8593.47it/s] 88%|████████▊ | 353297/400000 [00:42<00:05, 8600.54it/s] 89%|████████▊ | 354158/400000 [00:42<00:05, 8519.78it/s] 89%|████████▉ | 355018/400000 [00:42<00:05, 8542.89it/s] 89%|████████▉ | 355873/400000 [00:43<00:05, 8333.41it/s] 89%|████████▉ | 356714/400000 [00:43<00:05, 8356.23it/s] 89%|████████▉ | 357573/400000 [00:43<00:05, 8423.13it/s] 90%|████████▉ | 358417/400000 [00:43<00:05, 8225.09it/s] 90%|████████▉ | 359242/400000 [00:43<00:04, 8223.49it/s] 90%|█████████ | 360066/400000 [00:43<00:04, 8138.47it/s] 90%|█████████ | 360942/400000 [00:43<00:04, 8315.34it/s] 90%|█████████ | 361832/400000 [00:43<00:04, 8481.85it/s] 91%|█████████ | 362683/400000 [00:43<00:04, 8440.12it/s] 91%|█████████ | 363529/400000 [00:43<00:04, 8380.49it/s] 91%|█████████ | 364369/400000 [00:44<00:04, 8195.47it/s] 91%|█████████▏| 365297/400000 [00:44<00:04, 8491.42it/s] 92%|█████████▏| 366173/400000 [00:44<00:03, 8569.16it/s] 92%|█████████▏| 367064/400000 [00:44<00:03, 8668.10it/s] 92%|█████████▏| 367999/400000 [00:44<00:03, 8856.93it/s] 92%|█████████▏| 368888/400000 [00:44<00:03, 8670.99it/s] 92%|█████████▏| 369758/400000 [00:44<00:03, 8644.05it/s] 93%|█████████▎| 370637/400000 [00:44<00:03, 8687.18it/s] 93%|█████████▎| 371508/400000 [00:44<00:03, 8405.50it/s] 93%|█████████▎| 372371/400000 [00:44<00:03, 8469.36it/s] 93%|█████████▎| 373234/400000 [00:45<00:03, 8516.53it/s] 94%|█████████▎| 374088/400000 [00:45<00:03, 8503.13it/s] 94%|█████████▎| 374943/400000 [00:45<00:02, 8517.02it/s] 94%|█████████▍| 375826/400000 [00:45<00:02, 8607.75it/s] 94%|█████████▍| 376755/400000 [00:45<00:02, 8801.62it/s] 94%|█████████▍| 377637/400000 [00:45<00:02, 8465.33it/s] 95%|█████████▍| 378488/400000 [00:45<00:02, 8320.02it/s] 95%|█████████▍| 379324/400000 [00:45<00:02, 7671.13it/s] 95%|█████████▌| 380104/400000 [00:45<00:02, 7658.45it/s] 95%|█████████▌| 380943/400000 [00:46<00:02, 7863.32it/s] 95%|█████████▌| 381760/400000 [00:46<00:02, 7952.17it/s] 96%|█████████▌| 382686/400000 [00:46<00:02, 8301.77it/s] 96%|█████████▌| 383553/400000 [00:46<00:01, 8407.78it/s] 96%|█████████▌| 384419/400000 [00:46<00:01, 8481.37it/s] 96%|█████████▋| 385272/400000 [00:46<00:01, 8259.22it/s] 97%|█████████▋| 386103/400000 [00:46<00:01, 8048.45it/s] 97%|█████████▋| 387010/400000 [00:46<00:01, 8327.97it/s] 97%|█████████▋| 387855/400000 [00:46<00:01, 8360.87it/s] 97%|█████████▋| 388730/400000 [00:46<00:01, 8473.03it/s] 97%|█████████▋| 389611/400000 [00:47<00:01, 8568.40it/s] 98%|█████████▊| 390471/400000 [00:47<00:01, 8364.17it/s] 98%|█████████▊| 391395/400000 [00:47<00:00, 8608.86it/s] 98%|█████████▊| 392287/400000 [00:47<00:00, 8698.63it/s] 98%|█████████▊| 393205/400000 [00:47<00:00, 8833.88it/s] 99%|█████████▊| 394091/400000 [00:47<00:00, 8722.94it/s] 99%|█████████▊| 394966/400000 [00:47<00:00, 8660.22it/s] 99%|█████████▉| 395834/400000 [00:47<00:00, 8649.61it/s] 99%|█████████▉| 396701/400000 [00:47<00:00, 8503.92it/s] 99%|█████████▉| 397601/400000 [00:47<00:00, 8646.66it/s]100%|█████████▉| 398482/400000 [00:48<00:00, 8693.40it/s]100%|█████████▉| 399353/400000 [00:48<00:00, 8535.98it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8287.57it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe2d4202160>, <torchtext.data.dataset.TabularDataset object at 0x7fe2d42022b0>, <torchtext.vocab.Vocab object at 0x7fe2d42021d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.72 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.72 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.36 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.36 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.36 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.53 file/s]2020-06-14 12:20:50.014826: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-14 12:20:50.018802: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-06-14 12:20:50.019139: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561ec34f02e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-14 12:20:50.019244: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███▏      | 3121152/9912422 [00:00<00:00, 31012210.38it/s]9920512it [00:00, 35590536.11it/s]                             
0it [00:00, ?it/s]32768it [00:00, 529425.86it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1049527.02it/s]1654784it [00:00, 12837041.01it/s]                           
0it [00:00, ?it/s]8192it [00:00, 231476.91it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
