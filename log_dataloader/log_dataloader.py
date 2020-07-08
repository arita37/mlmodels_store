
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1f1dac40d0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1f1dac40d0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1f88e0d1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1f88e0d1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1fa7155e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1fa7155e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1f3613b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1f3613b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1f3613b620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3670016/9912422 [00:00<00:00, 36041121.02it/s]9920512it [00:00, 36825202.39it/s]                             
0it [00:00, ?it/s]32768it [00:00, 632778.94it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486447.58it/s]1654784it [00:00, 11870496.02it/s]                         
0it [00:00, ?it/s]8192it [00:00, 230986.73it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f333615f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f1d16fc88>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1f3613b268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1f3613b268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1f3613b268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:00,  2.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:00,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:00,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:59,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:59,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:59,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:58,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:58,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:40,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:40,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:40,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:40,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:39,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:39,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:39,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:39,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:38,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:38,  3.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:27,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:26,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:16,  7.35 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:11, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:10, 10.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 13.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 25.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 32.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 49.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 58.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 67.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 67.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 75.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 75.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 81.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:01<00:00, 81.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 87.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 87.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:01<00:00,  1.96s/ url]Dl Completed...: 100%|██████████| 1/1 [00:01<00:00,  1.96s/ url]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 87.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:01<00:00,  1.96s/ url]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 87.24 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:01<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.76s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  1.96s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 87.24 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.76s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:03<00:00,  3.76s/ file]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 43.06 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.76s/ url]
0 examples [00:00, ? examples/s]2020-07-08 00:07:56.273988: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-08 00:07:56.288595: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2593905000 Hz
2020-07-08 00:07:56.288749: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f2adb7cb00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-08 00:07:56.288763: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
13 examples [00:00, 129.67 examples/s]139 examples [00:00, 177.41 examples/s]271 examples [00:00, 239.61 examples/s]403 examples [00:00, 317.54 examples/s]535 examples [00:00, 411.07 examples/s]667 examples [00:00, 517.94 examples/s]798 examples [00:00, 632.47 examples/s]930 examples [00:00, 749.35 examples/s]1063 examples [00:00, 861.45 examples/s]1194 examples [00:01, 959.97 examples/s]1327 examples [00:01, 1045.81 examples/s]1458 examples [00:01, 1112.58 examples/s]1591 examples [00:01, 1167.72 examples/s]1723 examples [00:01, 1207.83 examples/s]1855 examples [00:01, 1237.15 examples/s]1987 examples [00:01, 1260.17 examples/s]2118 examples [00:01, 1272.28 examples/s]2249 examples [00:01, 1272.42 examples/s]2382 examples [00:01, 1286.89 examples/s]2513 examples [00:02, 1292.49 examples/s]2644 examples [00:02, 1269.29 examples/s]2775 examples [00:02, 1279.01 examples/s]2907 examples [00:02, 1288.36 examples/s]3039 examples [00:02, 1297.23 examples/s]3171 examples [00:02, 1301.15 examples/s]3303 examples [00:02, 1304.73 examples/s]3435 examples [00:02, 1307.92 examples/s]3566 examples [00:02, 1308.36 examples/s]3698 examples [00:02, 1309.80 examples/s]3830 examples [00:03, 1311.50 examples/s]3962 examples [00:03, 1313.47 examples/s]4094 examples [00:03, 1314.10 examples/s]4226 examples [00:03, 1315.10 examples/s]4358 examples [00:03, 1312.44 examples/s]4490 examples [00:03, 1313.52 examples/s]4622 examples [00:03, 1314.06 examples/s]4755 examples [00:03, 1316.22 examples/s]4887 examples [00:03, 1281.62 examples/s]5016 examples [00:03, 1266.09 examples/s]5148 examples [00:04, 1281.56 examples/s]5280 examples [00:04, 1291.66 examples/s]5413 examples [00:04, 1301.34 examples/s]5545 examples [00:04, 1304.11 examples/s]5677 examples [00:04, 1306.77 examples/s]5809 examples [00:04, 1309.43 examples/s]5941 examples [00:04, 1312.30 examples/s]6073 examples [00:04, 1310.21 examples/s]6205 examples [00:04, 1307.48 examples/s]6337 examples [00:04, 1310.74 examples/s]6469 examples [00:05, 1312.20 examples/s]6601 examples [00:05, 1288.45 examples/s]6731 examples [00:05, 1290.56 examples/s]6863 examples [00:05, 1296.55 examples/s]6995 examples [00:05, 1300.83 examples/s]7126 examples [00:05, 1298.01 examples/s]7256 examples [00:05, 1294.95 examples/s]7386 examples [00:05, 1286.68 examples/s]7516 examples [00:05, 1289.77 examples/s]7647 examples [00:05, 1293.10 examples/s]7777 examples [00:06, 1292.64 examples/s]7907 examples [00:06, 1286.28 examples/s]8038 examples [00:06, 1291.41 examples/s]8169 examples [00:06, 1294.64 examples/s]8301 examples [00:06, 1299.62 examples/s]8432 examples [00:06, 1301.41 examples/s]8563 examples [00:06, 1303.86 examples/s]8695 examples [00:06, 1306.11 examples/s]8826 examples [00:06, 1289.66 examples/s]8956 examples [00:06, 1290.89 examples/s]9087 examples [00:07, 1294.58 examples/s]9220 examples [00:07, 1302.40 examples/s]9351 examples [00:07, 1304.48 examples/s]9482 examples [00:07, 1305.17 examples/s]9613 examples [00:07, 1305.43 examples/s]9745 examples [00:07, 1308.28 examples/s]9878 examples [00:07, 1312.47 examples/s]10010 examples [00:07, 1253.36 examples/s]10141 examples [00:07, 1268.40 examples/s]10272 examples [00:07, 1279.28 examples/s]10404 examples [00:08, 1289.16 examples/s]10534 examples [00:08, 1276.60 examples/s]10662 examples [00:08, 1262.18 examples/s]10793 examples [00:08, 1274.11 examples/s]10924 examples [00:08, 1283.90 examples/s]11055 examples [00:08, 1291.57 examples/s]11186 examples [00:08, 1297.04 examples/s]11316 examples [00:08, 1297.53 examples/s]11448 examples [00:08, 1302.33 examples/s]11579 examples [00:09, 1300.31 examples/s]11710 examples [00:09, 1301.45 examples/s]11842 examples [00:09, 1305.69 examples/s]11974 examples [00:09, 1307.40 examples/s]12105 examples [00:09, 1306.97 examples/s]12236 examples [00:09, 1300.81 examples/s]12367 examples [00:09, 1301.26 examples/s]12498 examples [00:09, 1303.19 examples/s]12630 examples [00:09, 1306.75 examples/s]12762 examples [00:09, 1309.79 examples/s]12893 examples [00:10, 1307.02 examples/s]13025 examples [00:10, 1309.79 examples/s]13156 examples [00:10, 1308.58 examples/s]13288 examples [00:10, 1309.86 examples/s]13420 examples [00:10, 1311.35 examples/s]13552 examples [00:10, 1304.73 examples/s]13684 examples [00:10, 1306.67 examples/s]13815 examples [00:10, 1305.37 examples/s]13946 examples [00:10, 1306.44 examples/s]14078 examples [00:10, 1307.89 examples/s]14209 examples [00:11, 1304.62 examples/s]14340 examples [00:11, 1305.09 examples/s]14471 examples [00:11, 1303.63 examples/s]14602 examples [00:11, 1265.94 examples/s]14733 examples [00:11, 1277.08 examples/s]14863 examples [00:11, 1283.32 examples/s]14995 examples [00:11, 1291.74 examples/s]15126 examples [00:11, 1295.92 examples/s]15256 examples [00:11, 1293.93 examples/s]15387 examples [00:11, 1298.42 examples/s]15517 examples [00:12, 1296.02 examples/s]15648 examples [00:12, 1298.47 examples/s]15779 examples [00:12, 1300.38 examples/s]15910 examples [00:12, 1299.38 examples/s]16041 examples [00:12, 1299.99 examples/s]16172 examples [00:12, 1294.02 examples/s]16303 examples [00:12, 1296.26 examples/s]16434 examples [00:12, 1300.10 examples/s]16565 examples [00:12, 1299.84 examples/s]16696 examples [00:12, 1302.31 examples/s]16827 examples [00:13, 1284.88 examples/s]16958 examples [00:13, 1290.72 examples/s]17089 examples [00:13, 1295.78 examples/s]17220 examples [00:13, 1297.90 examples/s]17351 examples [00:13, 1300.02 examples/s]17482 examples [00:13, 1298.26 examples/s]17612 examples [00:13, 1298.66 examples/s]17743 examples [00:13, 1300.64 examples/s]17874 examples [00:13, 1299.56 examples/s]18006 examples [00:13, 1303.56 examples/s]18137 examples [00:14, 1300.74 examples/s]18268 examples [00:14, 1297.13 examples/s]18399 examples [00:14, 1298.69 examples/s]18529 examples [00:14, 1268.13 examples/s]18659 examples [00:14, 1277.23 examples/s]18790 examples [00:14, 1284.42 examples/s]18919 examples [00:14, 1284.39 examples/s]19049 examples [00:14, 1288.26 examples/s]19180 examples [00:14, 1292.32 examples/s]19310 examples [00:14, 1294.50 examples/s]19442 examples [00:15, 1299.12 examples/s]19573 examples [00:15, 1299.81 examples/s]19704 examples [00:15, 1301.60 examples/s]19835 examples [00:15, 1303.27 examples/s]19966 examples [00:15, 1303.94 examples/s]20097 examples [00:15, 1246.78 examples/s]20228 examples [00:15, 1263.97 examples/s]20359 examples [00:15, 1276.49 examples/s]20489 examples [00:15, 1282.47 examples/s]20620 examples [00:15, 1289.27 examples/s]20751 examples [00:16, 1293.37 examples/s]20882 examples [00:16, 1296.47 examples/s]21013 examples [00:16, 1298.94 examples/s]21143 examples [00:16, 1298.22 examples/s]21273 examples [00:16, 1296.82 examples/s]21404 examples [00:16, 1299.37 examples/s]21534 examples [00:16, 1296.60 examples/s]21665 examples [00:16, 1299.71 examples/s]21795 examples [00:16, 1288.30 examples/s]21925 examples [00:16, 1291.60 examples/s]22056 examples [00:17, 1295.44 examples/s]22187 examples [00:17, 1298.16 examples/s]22318 examples [00:17, 1300.61 examples/s]22449 examples [00:17, 1272.27 examples/s]22578 examples [00:17, 1275.65 examples/s]22708 examples [00:17, 1282.36 examples/s]22837 examples [00:17, 1283.04 examples/s]22967 examples [00:17, 1287.65 examples/s]23097 examples [00:17, 1290.82 examples/s]23227 examples [00:17, 1292.48 examples/s]23359 examples [00:18, 1298.62 examples/s]23489 examples [00:18, 1298.29 examples/s]23620 examples [00:18, 1299.09 examples/s]23751 examples [00:18, 1300.42 examples/s]23882 examples [00:18, 1299.95 examples/s]24013 examples [00:18, 1301.95 examples/s]24144 examples [00:18, 1301.39 examples/s]24275 examples [00:18, 1302.31 examples/s]24406 examples [00:18, 1302.29 examples/s]24537 examples [00:19, 1299.52 examples/s]24668 examples [00:19, 1301.39 examples/s]24799 examples [00:19, 1298.29 examples/s]24930 examples [00:19, 1299.79 examples/s]25060 examples [00:19, 1297.89 examples/s]25190 examples [00:19, 1295.62 examples/s]25320 examples [00:19, 1295.82 examples/s]25451 examples [00:19, 1299.08 examples/s]25581 examples [00:19, 1297.08 examples/s]25712 examples [00:19, 1298.68 examples/s]25842 examples [00:20, 1298.29 examples/s]25973 examples [00:20, 1299.84 examples/s]26104 examples [00:20, 1300.41 examples/s]26235 examples [00:20, 1297.83 examples/s]26365 examples [00:20, 1292.72 examples/s]26495 examples [00:20, 1270.40 examples/s]26625 examples [00:20, 1277.90 examples/s]26756 examples [00:20, 1285.99 examples/s]26886 examples [00:20, 1289.40 examples/s]27017 examples [00:20, 1294.23 examples/s]27148 examples [00:21, 1297.03 examples/s]27279 examples [00:21, 1299.01 examples/s]27409 examples [00:21, 1297.10 examples/s]27539 examples [00:21, 1292.75 examples/s]27670 examples [00:21, 1296.59 examples/s]27800 examples [00:21, 1295.38 examples/s]27931 examples [00:21, 1298.97 examples/s]28061 examples [00:21, 1298.91 examples/s]28191 examples [00:21, 1298.53 examples/s]28321 examples [00:21, 1279.61 examples/s]28451 examples [00:22, 1284.47 examples/s]28583 examples [00:22, 1292.27 examples/s]28713 examples [00:22, 1294.52 examples/s]28844 examples [00:22, 1296.82 examples/s]28974 examples [00:22, 1297.40 examples/s]29104 examples [00:22, 1297.11 examples/s]29235 examples [00:22, 1298.99 examples/s]29366 examples [00:22, 1300.51 examples/s]29497 examples [00:22, 1299.63 examples/s]29628 examples [00:22, 1300.65 examples/s]29759 examples [00:23, 1298.24 examples/s]29890 examples [00:23, 1299.70 examples/s]30020 examples [00:23, 1242.30 examples/s]30150 examples [00:23, 1256.65 examples/s]30281 examples [00:23, 1270.59 examples/s]30409 examples [00:23, 1237.70 examples/s]30540 examples [00:23, 1256.87 examples/s]30671 examples [00:23, 1270.10 examples/s]30801 examples [00:23, 1278.06 examples/s]30932 examples [00:23, 1286.06 examples/s]31063 examples [00:24, 1290.63 examples/s]31194 examples [00:24, 1294.67 examples/s]31325 examples [00:24, 1298.56 examples/s]31455 examples [00:24, 1298.46 examples/s]31585 examples [00:24, 1259.90 examples/s]31715 examples [00:24, 1270.70 examples/s]31846 examples [00:24, 1280.05 examples/s]31976 examples [00:24, 1284.92 examples/s]32106 examples [00:24, 1287.45 examples/s]32236 examples [00:24, 1290.97 examples/s]32366 examples [00:25, 1293.56 examples/s]32497 examples [00:25, 1297.10 examples/s]32628 examples [00:25, 1300.88 examples/s]32759 examples [00:25, 1301.69 examples/s]32890 examples [00:25, 1300.30 examples/s]33021 examples [00:25, 1300.97 examples/s]33152 examples [00:25, 1300.86 examples/s]33283 examples [00:25, 1302.70 examples/s]33414 examples [00:25, 1303.65 examples/s]33545 examples [00:25, 1298.63 examples/s]33676 examples [00:26, 1301.33 examples/s]33807 examples [00:26, 1303.64 examples/s]33938 examples [00:26, 1303.49 examples/s]34069 examples [00:26, 1304.63 examples/s]34200 examples [00:26, 1300.95 examples/s]34331 examples [00:26, 1268.22 examples/s]34461 examples [00:26, 1274.98 examples/s]34589 examples [00:26, 1272.69 examples/s]34717 examples [00:26, 1256.19 examples/s]34843 examples [00:27, 1238.89 examples/s]34971 examples [00:27, 1248.60 examples/s]35099 examples [00:27, 1255.92 examples/s]35229 examples [00:27, 1266.69 examples/s]35359 examples [00:27, 1275.59 examples/s]35487 examples [00:27, 1275.74 examples/s]35618 examples [00:27, 1285.46 examples/s]35747 examples [00:27, 1284.67 examples/s]35878 examples [00:27, 1289.48 examples/s]36008 examples [00:27, 1291.00 examples/s]36138 examples [00:28, 1290.13 examples/s]36268 examples [00:28, 1286.47 examples/s]36398 examples [00:28, 1288.31 examples/s]36527 examples [00:28, 1286.77 examples/s]36656 examples [00:28, 1286.70 examples/s]36785 examples [00:28, 1276.10 examples/s]36913 examples [00:28, 1272.33 examples/s]37041 examples [00:28, 1272.12 examples/s]37169 examples [00:28, 1273.60 examples/s]37300 examples [00:28, 1283.25 examples/s]37429 examples [00:29, 1281.85 examples/s]37558 examples [00:29, 1283.32 examples/s]37687 examples [00:29, 1281.63 examples/s]37818 examples [00:29, 1288.79 examples/s]37949 examples [00:29, 1293.16 examples/s]38079 examples [00:29, 1289.56 examples/s]38209 examples [00:29, 1292.41 examples/s]38339 examples [00:29, 1270.89 examples/s]38467 examples [00:29, 1251.47 examples/s]38595 examples [00:29, 1256.94 examples/s]38722 examples [00:30, 1259.53 examples/s]38852 examples [00:30, 1269.78 examples/s]38980 examples [00:30, 1270.96 examples/s]39110 examples [00:30, 1276.97 examples/s]39238 examples [00:30, 1277.09 examples/s]39366 examples [00:30, 1263.63 examples/s]39493 examples [00:30, 1262.94 examples/s]39620 examples [00:30, 1248.41 examples/s]39750 examples [00:30, 1262.39 examples/s]39877 examples [00:30, 1261.18 examples/s]40004 examples [00:31, 1209.44 examples/s]40135 examples [00:31, 1236.17 examples/s]40265 examples [00:31, 1254.57 examples/s]40395 examples [00:31, 1266.95 examples/s]40525 examples [00:31, 1274.07 examples/s]40654 examples [00:31, 1277.37 examples/s]40784 examples [00:31, 1281.70 examples/s]40913 examples [00:31, 1283.82 examples/s]41042 examples [00:31, 1280.23 examples/s]41171 examples [00:31, 1281.77 examples/s]41300 examples [00:32, 1283.28 examples/s]41429 examples [00:32, 1284.97 examples/s]41559 examples [00:32, 1287.24 examples/s]41690 examples [00:32, 1292.71 examples/s]41821 examples [00:32, 1296.45 examples/s]41951 examples [00:32, 1293.38 examples/s]42081 examples [00:32, 1288.54 examples/s]42210 examples [00:32, 1284.97 examples/s]42340 examples [00:32, 1288.06 examples/s]42469 examples [00:32, 1265.76 examples/s]42599 examples [00:33, 1274.14 examples/s]42727 examples [00:33, 1267.30 examples/s]42857 examples [00:33, 1276.21 examples/s]42988 examples [00:33, 1284.47 examples/s]43118 examples [00:33, 1286.32 examples/s]43249 examples [00:33, 1292.59 examples/s]43380 examples [00:33, 1295.54 examples/s]43510 examples [00:33, 1293.80 examples/s]43640 examples [00:33, 1293.67 examples/s]43770 examples [00:33, 1294.81 examples/s]43900 examples [00:34, 1294.85 examples/s]44030 examples [00:34, 1292.70 examples/s]44160 examples [00:34, 1293.48 examples/s]44290 examples [00:34, 1295.04 examples/s]44421 examples [00:34, 1296.56 examples/s]44552 examples [00:34, 1298.54 examples/s]44682 examples [00:34, 1295.84 examples/s]44812 examples [00:34, 1295.12 examples/s]44942 examples [00:34, 1294.67 examples/s]45072 examples [00:34, 1295.17 examples/s]45202 examples [00:35, 1293.64 examples/s]45332 examples [00:35, 1294.38 examples/s]45462 examples [00:35, 1292.38 examples/s]45593 examples [00:35, 1295.74 examples/s]45723 examples [00:35, 1295.84 examples/s]45854 examples [00:35, 1297.35 examples/s]45984 examples [00:35, 1295.63 examples/s]46114 examples [00:35, 1295.79 examples/s]46244 examples [00:35, 1296.53 examples/s]46375 examples [00:35, 1298.95 examples/s]46505 examples [00:36, 1289.62 examples/s]46634 examples [00:36, 1270.42 examples/s]46764 examples [00:36, 1276.31 examples/s]46895 examples [00:36, 1283.47 examples/s]47025 examples [00:36, 1285.89 examples/s]47156 examples [00:36, 1290.56 examples/s]47286 examples [00:36, 1291.38 examples/s]47416 examples [00:36, 1291.15 examples/s]47546 examples [00:36, 1279.08 examples/s]47676 examples [00:37, 1283.00 examples/s]47806 examples [00:37, 1285.71 examples/s]47937 examples [00:37, 1290.05 examples/s]48067 examples [00:37, 1292.27 examples/s]48197 examples [00:37, 1290.71 examples/s]48327 examples [00:37, 1291.94 examples/s]48457 examples [00:37, 1293.87 examples/s]48587 examples [00:37, 1294.64 examples/s]48717 examples [00:37, 1292.33 examples/s]48847 examples [00:37, 1294.55 examples/s]48977 examples [00:38, 1294.92 examples/s]49108 examples [00:38, 1297.18 examples/s]49238 examples [00:38, 1296.33 examples/s]49368 examples [00:38, 1295.17 examples/s]49499 examples [00:38, 1297.40 examples/s]49629 examples [00:38, 1296.99 examples/s]49760 examples [00:38, 1299.09 examples/s]49890 examples [00:38, 1297.12 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 17%|█▋        | 8421/50000 [00:00<00:00, 84209.08 examples/s] 46%|████▌     | 22809/50000 [00:00<00:00, 96173.95 examples/s] 74%|███████▍  | 37223/50000 [00:00<00:00, 106839.31 examples/s]                                                                0 examples [00:00, ? examples/s]110 examples [00:00, 1095.89 examples/s]242 examples [00:00, 1153.76 examples/s]374 examples [00:00, 1197.70 examples/s]507 examples [00:00, 1232.16 examples/s]639 examples [00:00, 1257.16 examples/s]770 examples [00:00, 1271.11 examples/s]903 examples [00:00, 1286.11 examples/s]1035 examples [00:00, 1294.06 examples/s]1168 examples [00:00, 1301.85 examples/s]1300 examples [00:01, 1305.84 examples/s]1432 examples [00:01, 1306.06 examples/s]1565 examples [00:01, 1311.40 examples/s]1697 examples [00:01, 1312.64 examples/s]1830 examples [00:01, 1316.07 examples/s]1961 examples [00:01, 1283.36 examples/s]2093 examples [00:01, 1292.29 examples/s]2224 examples [00:01, 1296.22 examples/s]2356 examples [00:01, 1302.06 examples/s]2488 examples [00:01, 1307.01 examples/s]2620 examples [00:02, 1310.59 examples/s]2753 examples [00:02, 1314.45 examples/s]2885 examples [00:02, 1313.11 examples/s]3017 examples [00:02, 1312.89 examples/s]3150 examples [00:02, 1315.23 examples/s]3283 examples [00:02, 1319.03 examples/s]3415 examples [00:02, 1319.28 examples/s]3547 examples [00:02, 1301.24 examples/s]3679 examples [00:02, 1306.76 examples/s]3812 examples [00:02, 1312.29 examples/s]3944 examples [00:03, 1293.26 examples/s]4074 examples [00:03, 1295.02 examples/s]4204 examples [00:03, 1284.07 examples/s]4336 examples [00:03, 1291.61 examples/s]4468 examples [00:03, 1298.38 examples/s]4601 examples [00:03, 1306.59 examples/s]4734 examples [00:03, 1312.46 examples/s]4866 examples [00:03, 1314.21 examples/s]4998 examples [00:03, 1309.75 examples/s]5129 examples [00:03, 1308.00 examples/s]5261 examples [00:04, 1310.35 examples/s]5393 examples [00:04, 1260.41 examples/s]5526 examples [00:04, 1277.77 examples/s]5657 examples [00:04, 1286.63 examples/s]5789 examples [00:04, 1295.53 examples/s]5922 examples [00:04, 1304.30 examples/s]6055 examples [00:04, 1309.14 examples/s]6187 examples [00:04, 1312.14 examples/s]6319 examples [00:04, 1310.65 examples/s]6451 examples [00:04, 1312.03 examples/s]6583 examples [00:05, 1291.73 examples/s]6715 examples [00:05, 1298.91 examples/s]6848 examples [00:05, 1306.72 examples/s]6979 examples [00:05, 1306.11 examples/s]7111 examples [00:05, 1308.07 examples/s]7243 examples [00:05, 1310.11 examples/s]7376 examples [00:05, 1314.28 examples/s]7508 examples [00:05, 1315.27 examples/s]7640 examples [00:05, 1312.55 examples/s]7773 examples [00:05, 1314.86 examples/s]7905 examples [00:06, 1305.97 examples/s]8036 examples [00:06, 1273.85 examples/s]8167 examples [00:06, 1283.98 examples/s]8298 examples [00:06, 1290.16 examples/s]8430 examples [00:06, 1298.82 examples/s]8563 examples [00:06, 1305.84 examples/s]8696 examples [00:06, 1310.60 examples/s]8828 examples [00:06, 1286.79 examples/s]8959 examples [00:06, 1291.47 examples/s]9091 examples [00:06, 1298.54 examples/s]9223 examples [00:07, 1303.98 examples/s]9355 examples [00:07, 1307.10 examples/s]9487 examples [00:07, 1308.69 examples/s]9618 examples [00:07, 1308.73 examples/s]9749 examples [00:07, 1308.25 examples/s]9881 examples [00:07, 1308.84 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFEJVTN/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteFEJVTN/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1f3613b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1f3613b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1f3613b620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1ec8355278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1ec838dd30>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1f3613b620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1f3613b620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1f3613b620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f1d16f4a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f1d16f550>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1eadca01e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1eadca01e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f1eadca01e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1eadca02f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1eadca02f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1eadca02f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=cc08a4b693b39251f6a30363d76264f9c0dc1ecc3c25090789817de7d0c487dd
  Stored in directory: /tmp/pip-ephem-wheel-cache-z1t111px/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:21:09, 10.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:34:54, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:39:38, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:10:12, 29.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:42:16, 41.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.34M/862M [00:01<3:58:05, 59.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:45:40, 85.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<1:55:17, 122kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<1:20:14, 174kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:02<55:53, 248kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<38:56, 353kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<27:09, 502kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.6M/862M [00:02<18:57, 715kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<14:10, 952kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<11:47, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<10:26, 1.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<07:51, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<07:49, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<07:14, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<05:30, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<06:25, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<07:35, 1.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<06:01, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<04:21, 3.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<10:31, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<08:48, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<06:31, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<07:27, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<07:54, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:13<06:11, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:13<04:27, 2.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<12:40:12, 17.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<8:53:14, 24.5kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.5M/862M [00:15<6:12:49, 35.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<4:23:18, 49.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<3:06:55, 69.6kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:17<2:11:17, 99.0kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<1:31:49, 141kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<1:10:37, 183kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<50:44, 255kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 87.7M/862M [00:19<35:43, 361kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<27:58, 460kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<22:12, 580kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:21<16:11, 794kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<13:21, 958kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:22<10:39, 1.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.9M/862M [00:23<07:43, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<08:23, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:24<07:10, 1.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<05:20, 2.38MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:43, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:00, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:28, 2.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:07, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:51, 1.83MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:26, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<03:55, 3.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:07:10, 17.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<8:30:04, 24.6kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<5:56:36, 35.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<4:11:49, 49.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<2:57:27, 70.2kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<2:04:13, 100kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:29:39, 138kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:05:18, 190kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<46:18, 267kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<34:16, 360kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<25:14, 488kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<17:57, 685kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<15:23, 796kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<12:02, 1.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<08:40, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:57, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:46, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:45, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<04:50, 2.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:31:03, 133kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<1:04:58, 186kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<45:42, 264kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<34:41, 347kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<26:44, 450kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<19:19, 623kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<15:24, 778kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<12:01, 995kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:40, 1.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:48, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:35, 1.39MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:32, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<04:41, 2.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<14:44, 804kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<11:32, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<08:18, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:34, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:12, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:29, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:56, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:27, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<03:55, 2.96MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:38:42, 118kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<1:10:15, 166kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<49:19, 235kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<37:08, 312kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<27:11, 425kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<19:15, 599kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<16:08, 713kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<12:27, 922kB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<08:59, 1.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:59, 1.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<07:27, 1.53MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:29, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:31, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:53, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:20, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:50, 2.94MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<35:55, 314kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<26:18, 429kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<18:40, 603kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<15:39, 717kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<13:15, 847kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:45, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<06:57, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:35, 962kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<09:15, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<06:42, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:17, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:12, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:34, 2.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:49, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:02, 2.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:48, 2.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:55, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:42, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:24, 3.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<1:31:55, 118kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<1:05:25, 166kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<45:55, 236kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<34:35, 313kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<25:07, 430kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<17:49, 605kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<12:34, 855kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<49:24, 218kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<36:47, 292kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<26:16, 408kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<18:24, 580kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:29:36, 17.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<7:21:32, 24.2kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<5:08:36, 34.5kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<3:37:47, 48.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<2:33:26, 69.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<1:47:22, 98.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<1:17:23, 136kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<56:19, 187kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<39:50, 264kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<27:54, 376kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<26:52, 390kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<19:52, 527kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<14:06, 740kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<12:17, 847kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<09:44, 1.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<07:01, 1.48MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<05:13, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:38, 1.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<09:35, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:34, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<05:30, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<04:09, 2.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:56, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:55, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:16, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:53, 2.62MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<06:47, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<08:55, 1.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:04, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:11, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:52, 2.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:37, 1.53MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:46, 1.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<07:10, 1.41MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<05:17, 1.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:39<03:52, 2.60MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<29:28, 341kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<24:45, 406kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<18:17, 549kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<13:02, 768kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<09:17, 1.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<46:59, 213kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<36:40, 272kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<26:28, 377kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<18:53, 528kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<13:22, 743kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<09:37, 1.03MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<46:39, 213kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<36:25, 272kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<26:16, 377kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<18:34, 532kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<13:15, 745kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<13:08, 749kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<12:55, 762kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:55, 991kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<07:11, 1.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<05:12, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<40:52, 239kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<32:37, 300kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<23:37, 414kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<16:42, 584kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<11:57, 814kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<12:10, 797kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<12:30, 776kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<09:42, 1.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<06:59, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<05:01, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<20:35, 468kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<18:03, 534kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<13:26, 717kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<09:38, 996kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<06:54, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<51:23, 186kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<39:36, 242kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<28:38, 334kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<20:15, 471kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<16:14, 585kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<14:58, 635kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<11:13, 845kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<08:05, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<05:48, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<14:25, 654kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<13:24, 703kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<10:15, 918kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:21, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<07:21, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:41, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:55, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:02, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<05:40, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<07:14, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:51, 1.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:19, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:11, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:53, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:27, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:02, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:32, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<05:19, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<03:53, 2.34MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:14, 1.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<06:27, 1.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:05, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<03:42, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:48, 3.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<10:01, 900kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:46, 922kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:25, 1.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:23, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<03:55, 2.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<33:40, 266kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<25:53, 346kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<18:41, 478kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<13:12, 674kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<12:23, 716kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<11:14, 789kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<08:30, 1.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<06:05, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<07:03, 1.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:21, 1.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<05:42, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:07, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:02, 1.45MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<06:36, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<05:08, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:45, 2.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<05:31, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:53, 1.47MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:32, 1.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:17, 2.62MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<05:53, 1.46MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<06:28, 1.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:06, 1.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:41, 2.32MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<05:56, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<06:15, 1.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:49, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:30, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:05, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<05:53, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:35, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:21, 2.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:41, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:27, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:17, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:08, 2.66MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<05:39, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:07, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:44, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:25, 2.41MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:10, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<05:08, 1.60MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<03:53, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:50, 2.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<05:29, 1.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<05:51, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:30, 1.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:16, 2.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:13, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<05:27, 1.48MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<04:12, 1.93MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:02, 2.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:11, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:09, 1.31MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:40, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:23, 2.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:57, 1.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<05:16, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<04:04, 1.96MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:56, 2.70MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:51, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:52, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:29, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<03:14, 2.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:39, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<05:43, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:26, 1.76MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:10, 2.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<09:29, 819kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<08:22, 928kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<06:17, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<04:28, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<10:18, 747kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<09:02, 852kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:42, 1.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:47, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:01, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:51, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:27, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<03:11, 2.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:30, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<06:13, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:38, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:20, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<08:12, 913kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<07:12, 1.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:25, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:58, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<05:00, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<03:53, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:53, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:14, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:18, 2.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<02:23, 3.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<2:21:35, 51.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<1:40:35, 72.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<1:10:39, 103kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<50:16, 144kB/s]  .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<36:40, 197kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<25:59, 277kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<19:13, 372kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<14:55, 479kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<10:47, 661kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:38, 819kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<07:30, 943kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:32, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<03:59, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:07, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<04:33, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<03:23, 2.06MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:26, 2.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<10:44, 647kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<08:46, 791kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:27, 1.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:38, 1.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:12, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:55, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:48, 2.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:26, 1.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:44, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:19, 1.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:08, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:07, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:11, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:20, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:33, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:47, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:03, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:20, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:38, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:55, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:14, 2.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:30, 2.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<01:50, 3.52MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<04:46, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:30, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:24, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:28, 2.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:38, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:27, 1.44MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:21, 1.90MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:25, 2.62MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:43, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:12, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:55, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:46, 1.66MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:49, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:57, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:05, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:20, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:37, 2.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:50, 2.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:06, 1.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:27, 2.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:43, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:59, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:22, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:39, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:56, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:17, 2.61MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:40, 3.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:58, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:33, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:25, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:26, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:39, 878kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:43, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<04:15, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:55, 1.47MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:48, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:52, 2.00MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:04, 2.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<05:22, 1.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<04:48, 1.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<06:10, 924kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:54, 1.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:25, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:20, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:15, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<04:23, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<03:31, 1.58MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:35, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:08, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:12, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:29, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:38, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:50, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:13, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:26, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:41, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:07, 2.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:21, 2.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:39, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:05, 2.52MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:19, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:37, 1.99MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:01, 2.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:30, 3.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:58, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:03, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<02:22, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:29, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:42, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:07, 2.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:18, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:33, 1.96MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:58, 2.52MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:26, 3.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:45, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:05, 1.21MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:02, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:16, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:50, 1.72MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:41, 1.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<03:01, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<02:12, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<02:47, 1.72MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:51, 1.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:10, 2.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:35, 3.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<03:34, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:25, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:33, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:52, 2.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:58, 1.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:57, 1.58MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:14, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:38, 2.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:59, 1.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:57, 1.56MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:15, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:19, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:27, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:56, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:05, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:16, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<01:45, 2.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:18, 3.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:34, 1.71MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:36, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:59, 2.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:26, 3.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:54, 882kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:13, 1.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:07, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:13, 1.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:00, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:34, 1.19MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:39, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:54, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<04:05, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:37, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:41, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:55, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:52, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:27, 1.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:36, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:29, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:29, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:53, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:21, 2.94MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:20, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:05, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:19, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:39, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:49, 812kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<04:07, 950kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:01, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:09, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:43, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<03:19, 1.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:29, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:21, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:22, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:48, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:18, 2.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:49, 1.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:39, 1.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:01, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:00, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:04, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:36, 2.25MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:42, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:51, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:25, 2.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:02, 3.40MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:52, 907kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<03:20, 1.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:27, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:45, 1.97MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:59, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:33, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:52, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:21, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<03:12, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<02:40, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:57, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<02:00, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:57, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:30, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:35, 2.02MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<01:41, 1.92MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:18, 2.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:27, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:36, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:15, 2.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:23, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<01:30, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:10, 2.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:20, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:21, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:04, 2.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<00:47, 3.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:55, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:51, 1.59MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:23, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:00, 2.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:19, 1.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:06, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:34, 1.82MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:07, 2.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:42, 1.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:22, 1.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:45, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:16, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:40, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:17, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:53, 1.42MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:06, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:25, 1.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:26, 1.82MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:05, 2.37MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:47, 3.23MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:59, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:48, 1.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:21, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:22, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:22, 1.79MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:02, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:45, 3.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:27, 981kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:07, 1.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:34, 1.51MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:29, 1.57MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:26, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:05, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:08, 1.99MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:11, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:55, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:00, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:05, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:50, 2.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:36, 3.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:45, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:35, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:10, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:50, 2.47MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<02:35, 796kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:09, 953kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:34, 1.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:25, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:20, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:00, 1.97MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:43, 2.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:22, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:16, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:58, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:58, 1.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:00, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:46, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:50, 2.14MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:53, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:41, 2.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:46, 2.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:49, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:38, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:43, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:47, 2.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:37, 2.62MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:41, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:45, 2.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:35, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:39, 2.29MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:39, 2.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:29, 3.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:37, 2.32MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:56, 1.53MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:46, 1.83MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:33, 2.50MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:45, 1.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:46, 1.78MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:34, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:24, 3.18MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:02, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:56, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:42, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.79MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:41, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:31, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:33, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:35, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:26, 2.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:19, 3.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:50, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:44, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:33, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:23, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:47, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:43, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:32, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:22, 2.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:46, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:42, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:31, 1.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:21, 2.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<03:21, 266kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:29, 357kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:44, 501kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<01:10, 710kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:49, 450kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:24, 580kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:00, 800kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:47, 955kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:40, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:29, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:20, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:46, 885kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:39, 1.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:28, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:19, 1.97MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:53, 688kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:43, 842kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:31, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:25, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:22, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:16, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<00:11, 2.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:20, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:09, 2.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:32, 761kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:26, 913kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:19, 1.24MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:15, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:09, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:05, 2.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:16, 987kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:14, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:10, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.63MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 2.45MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.60MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.53MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 731/400000 [00:00<00:54, 7303.63it/s]  0%|          | 1759/400000 [00:00<00:49, 7997.70it/s]  1%|          | 2789/400000 [00:00<00:46, 8572.41it/s]  1%|          | 3828/400000 [00:00<00:43, 9045.99it/s]  1%|          | 4837/400000 [00:00<00:42, 9333.94it/s]  1%|▏         | 5875/400000 [00:00<00:40, 9624.02it/s]  2%|▏         | 6914/400000 [00:00<00:39, 9841.03it/s]  2%|▏         | 7955/400000 [00:00<00:39, 10004.24it/s]  2%|▏         | 8982/400000 [00:00<00:38, 10081.70it/s]  2%|▏         | 9996/400000 [00:01<00:38, 10096.30it/s]  3%|▎         | 11035/400000 [00:01<00:38, 10182.05it/s]  3%|▎         | 12076/400000 [00:01<00:37, 10249.13it/s]  3%|▎         | 13094/400000 [00:01<00:37, 10222.34it/s]  4%|▎         | 14136/400000 [00:01<00:37, 10277.90it/s]  4%|▍         | 15175/400000 [00:01<00:37, 10310.77it/s]  4%|▍         | 16216/400000 [00:01<00:37, 10339.81it/s]  4%|▍         | 17249/400000 [00:01<00:37, 10178.16it/s]  5%|▍         | 18290/400000 [00:01<00:37, 10244.98it/s]  5%|▍         | 19331/400000 [00:01<00:36, 10293.30it/s]  5%|▌         | 20361/400000 [00:02<00:36, 10284.13it/s]  5%|▌         | 21394/400000 [00:02<00:36, 10295.82it/s]  6%|▌         | 22431/400000 [00:02<00:36, 10316.96it/s]  6%|▌         | 23463/400000 [00:02<00:36, 10302.32it/s]  6%|▌         | 24494/400000 [00:02<00:36, 10298.02it/s]  6%|▋         | 25524/400000 [00:02<00:36, 10177.78it/s]  7%|▋         | 26556/400000 [00:02<00:36, 10218.64it/s]  7%|▋         | 27589/400000 [00:02<00:36, 10249.87it/s]  7%|▋         | 28627/400000 [00:02<00:36, 10287.30it/s]  7%|▋         | 29661/400000 [00:02<00:35, 10302.59it/s]  8%|▊         | 30697/400000 [00:03<00:35, 10316.85it/s]  8%|▊         | 31739/400000 [00:03<00:35, 10344.86it/s]  8%|▊         | 32781/400000 [00:03<00:35, 10365.33it/s]  8%|▊         | 33824/400000 [00:03<00:35, 10381.77it/s]  9%|▊         | 34863/400000 [00:03<00:35, 10319.58it/s]  9%|▉         | 35905/400000 [00:03<00:35, 10346.70it/s]  9%|▉         | 36946/400000 [00:03<00:35, 10363.91it/s]  9%|▉         | 37988/400000 [00:03<00:34, 10378.47it/s] 10%|▉         | 39026/400000 [00:03<00:34, 10344.07it/s] 10%|█         | 40067/400000 [00:03<00:34, 10362.30it/s] 10%|█         | 41106/400000 [00:04<00:34, 10368.88it/s] 11%|█         | 42147/400000 [00:04<00:34, 10379.36it/s] 11%|█         | 43185/400000 [00:04<00:34, 10346.01it/s] 11%|█         | 44220/400000 [00:04<00:34, 10345.21it/s] 11%|█▏        | 45257/400000 [00:04<00:34, 10352.12it/s] 12%|█▏        | 46298/400000 [00:04<00:34, 10368.36it/s] 12%|█▏        | 47340/400000 [00:04<00:33, 10383.21it/s] 12%|█▏        | 48379/400000 [00:04<00:33, 10351.73it/s] 12%|█▏        | 49421/400000 [00:04<00:33, 10369.11it/s] 13%|█▎        | 50461/400000 [00:04<00:33, 10376.50it/s] 13%|█▎        | 51502/400000 [00:05<00:33, 10384.73it/s] 13%|█▎        | 52541/400000 [00:05<00:33, 10316.00it/s] 13%|█▎        | 53583/400000 [00:05<00:33, 10345.14it/s] 14%|█▎        | 54626/400000 [00:05<00:33, 10368.04it/s] 14%|█▍        | 55667/400000 [00:05<00:33, 10379.85it/s] 14%|█▍        | 56706/400000 [00:05<00:33, 10350.41it/s] 14%|█▍        | 57746/400000 [00:05<00:33, 10364.24it/s] 15%|█▍        | 58787/400000 [00:05<00:32, 10377.85it/s] 15%|█▍        | 59829/400000 [00:05<00:32, 10390.33it/s] 15%|█▌        | 60869/400000 [00:05<00:32, 10332.86it/s] 15%|█▌        | 61909/400000 [00:06<00:32, 10350.53it/s] 16%|█▌        | 62949/400000 [00:06<00:32, 10364.83it/s] 16%|█▌        | 63986/400000 [00:06<00:32, 10358.39it/s] 16%|█▋        | 65022/400000 [00:06<00:32, 10339.80it/s] 17%|█▋        | 66065/400000 [00:06<00:32, 10364.58it/s] 17%|█▋        | 67102/400000 [00:06<00:32, 10366.02it/s] 17%|█▋        | 68142/400000 [00:06<00:31, 10373.85it/s] 17%|█▋        | 69180/400000 [00:06<00:31, 10374.73it/s] 18%|█▊        | 70218/400000 [00:06<00:31, 10328.79it/s] 18%|█▊        | 71258/400000 [00:06<00:31, 10349.51it/s] 18%|█▊        | 72298/400000 [00:07<00:31, 10364.44it/s] 18%|█▊        | 73339/400000 [00:07<00:31, 10376.88it/s] 19%|█▊        | 74377/400000 [00:07<00:31, 10356.01it/s] 19%|█▉        | 75417/400000 [00:07<00:31, 10367.00it/s] 19%|█▉        | 76457/400000 [00:07<00:31, 10376.82it/s] 19%|█▉        | 77496/400000 [00:07<00:31, 10377.75it/s] 20%|█▉        | 78534/400000 [00:07<00:31, 10337.09it/s] 20%|█▉        | 79574/400000 [00:07<00:30, 10354.12it/s] 20%|██        | 80615/400000 [00:07<00:30, 10369.78it/s] 20%|██        | 81658/400000 [00:07<00:30, 10386.15it/s] 21%|██        | 82697/400000 [00:08<00:30, 10338.02it/s] 21%|██        | 83739/400000 [00:08<00:30, 10359.91it/s] 21%|██        | 84779/400000 [00:08<00:30, 10371.75it/s] 21%|██▏       | 85820/400000 [00:08<00:30, 10381.00it/s] 22%|██▏       | 86859/400000 [00:08<00:30, 10300.00it/s] 22%|██▏       | 87896/400000 [00:08<00:30, 10320.52it/s] 22%|██▏       | 88936/400000 [00:08<00:30, 10343.58it/s] 22%|██▏       | 89976/400000 [00:08<00:29, 10359.28it/s] 23%|██▎       | 91013/400000 [00:08<00:29, 10303.02it/s] 23%|██▎       | 92053/400000 [00:08<00:29, 10330.08it/s] 23%|██▎       | 93092/400000 [00:09<00:29, 10346.64it/s] 24%|██▎       | 94132/400000 [00:09<00:29, 10361.52it/s] 24%|██▍       | 95169/400000 [00:09<00:29, 10340.01it/s] 24%|██▍       | 96204/400000 [00:09<00:29, 10291.71it/s] 24%|██▍       | 97245/400000 [00:09<00:29, 10324.95it/s] 25%|██▍       | 98283/400000 [00:09<00:29, 10340.98it/s] 25%|██▍       | 99323/400000 [00:09<00:29, 10357.11it/s] 25%|██▌       | 100359/400000 [00:09<00:29, 10287.22it/s] 25%|██▌       | 101401/400000 [00:09<00:28, 10324.47it/s] 26%|██▌       | 102442/400000 [00:09<00:28, 10347.19it/s] 26%|██▌       | 103477/400000 [00:10<00:28, 10334.57it/s] 26%|██▌       | 104511/400000 [00:10<00:28, 10313.38it/s] 26%|██▋       | 105551/400000 [00:10<00:28, 10339.07it/s] 27%|██▋       | 106592/400000 [00:10<00:28, 10359.50it/s] 27%|██▋       | 107634/400000 [00:10<00:28, 10377.00it/s] 27%|██▋       | 108672/400000 [00:10<00:28, 10307.29it/s] 27%|██▋       | 109711/400000 [00:10<00:28, 10330.58it/s] 28%|██▊       | 110751/400000 [00:10<00:27, 10349.08it/s] 28%|██▊       | 111791/400000 [00:10<00:27, 10363.52it/s] 28%|██▊       | 112828/400000 [00:10<00:27, 10341.37it/s] 28%|██▊       | 113869/400000 [00:11<00:27, 10360.96it/s] 29%|██▊       | 114909/400000 [00:11<00:27, 10370.21it/s] 29%|██▉       | 115948/400000 [00:11<00:27, 10375.64it/s] 29%|██▉       | 116986/400000 [00:11<00:27, 10338.90it/s] 30%|██▉       | 118025/400000 [00:11<00:27, 10354.15it/s] 30%|██▉       | 119063/400000 [00:11<00:27, 10361.81it/s] 30%|███       | 120100/400000 [00:11<00:27, 10361.89it/s] 30%|███       | 121137/400000 [00:11<00:27, 10276.22it/s] 31%|███       | 122175/400000 [00:11<00:26, 10304.40it/s] 31%|███       | 123215/400000 [00:11<00:26, 10331.22it/s] 31%|███       | 124254/400000 [00:12<00:26, 10347.42it/s] 31%|███▏      | 125289/400000 [00:12<00:26, 10303.83it/s] 32%|███▏      | 126327/400000 [00:12<00:26, 10323.99it/s] 32%|███▏      | 127361/400000 [00:12<00:26, 10327.22it/s] 32%|███▏      | 128399/400000 [00:12<00:26, 10342.12it/s] 32%|███▏      | 129439/400000 [00:12<00:26, 10357.40it/s] 33%|███▎      | 130475/400000 [00:12<00:26, 10337.99it/s] 33%|███▎      | 131513/400000 [00:12<00:25, 10348.76it/s] 33%|███▎      | 132552/400000 [00:12<00:25, 10358.25it/s] 33%|███▎      | 133592/400000 [00:12<00:25, 10370.58it/s] 34%|███▎      | 134630/400000 [00:13<00:25, 10312.46it/s] 34%|███▍      | 135671/400000 [00:13<00:25, 10339.38it/s] 34%|███▍      | 136709/400000 [00:13<00:25, 10350.99it/s] 34%|███▍      | 137750/400000 [00:13<00:25, 10367.57it/s] 35%|███▍      | 138787/400000 [00:13<00:25, 10347.95it/s] 35%|███▍      | 139826/400000 [00:13<00:25, 10358.07it/s] 35%|███▌      | 140867/400000 [00:13<00:24, 10370.87it/s] 35%|███▌      | 141906/400000 [00:13<00:24, 10374.47it/s] 36%|███▌      | 142944/400000 [00:13<00:24, 10362.92it/s] 36%|███▌      | 143984/400000 [00:13<00:24, 10371.82it/s] 36%|███▋      | 145024/400000 [00:14<00:24, 10379.23it/s] 37%|███▋      | 146063/400000 [00:14<00:24, 10380.70it/s] 37%|███▋      | 147102/400000 [00:14<00:24, 10327.79it/s] 37%|███▋      | 148140/400000 [00:14<00:24, 10342.73it/s] 37%|███▋      | 149180/400000 [00:14<00:24, 10357.37it/s] 38%|███▊      | 150219/400000 [00:14<00:24, 10365.31it/s] 38%|███▊      | 151256/400000 [00:14<00:24, 10346.44it/s] 38%|███▊      | 152293/400000 [00:14<00:23, 10352.91it/s] 38%|███▊      | 153332/400000 [00:14<00:23, 10362.24it/s] 39%|███▊      | 154373/400000 [00:14<00:23, 10373.78it/s] 39%|███▉      | 155411/400000 [00:15<00:23, 10282.66it/s] 39%|███▉      | 156449/400000 [00:15<00:23, 10310.05it/s] 39%|███▉      | 157485/400000 [00:15<00:23, 10323.68it/s] 40%|███▉      | 158525/400000 [00:15<00:23, 10345.72it/s] 40%|███▉      | 159563/400000 [00:15<00:23, 10353.88it/s] 40%|████      | 160599/400000 [00:15<00:23, 10304.17it/s] 40%|████      | 161636/400000 [00:15<00:23, 10323.18it/s] 41%|████      | 162670/400000 [00:15<00:22, 10328.04it/s] 41%|████      | 163705/400000 [00:15<00:22, 10334.59it/s] 41%|████      | 164739/400000 [00:15<00:22, 10322.40it/s] 41%|████▏     | 165777/400000 [00:16<00:22, 10338.69it/s] 42%|████▏     | 166815/400000 [00:16<00:22, 10350.48it/s] 42%|████▏     | 167851/400000 [00:16<00:22, 10352.95it/s] 42%|████▏     | 168887/400000 [00:16<00:22, 10309.17it/s] 42%|████▏     | 169926/400000 [00:16<00:22, 10331.09it/s] 43%|████▎     | 170962/400000 [00:16<00:22, 10338.48it/s] 43%|████▎     | 172000/400000 [00:16<00:22, 10349.30it/s] 43%|████▎     | 173035/400000 [00:16<00:22, 10295.07it/s] 44%|████▎     | 174074/400000 [00:16<00:21, 10321.62it/s] 44%|████▍     | 175114/400000 [00:16<00:21, 10343.93it/s] 44%|████▍     | 176153/400000 [00:17<00:21, 10355.60it/s] 44%|████▍     | 177189/400000 [00:17<00:21, 10321.42it/s] 45%|████▍     | 178228/400000 [00:17<00:21, 10339.42it/s] 45%|████▍     | 179267/400000 [00:17<00:21, 10354.07it/s] 45%|████▌     | 180306/400000 [00:17<00:21, 10363.48it/s] 45%|████▌     | 181343/400000 [00:17<00:21, 10319.90it/s] 46%|████▌     | 182381/400000 [00:17<00:21, 10335.09it/s] 46%|████▌     | 183421/400000 [00:17<00:20, 10353.53it/s] 46%|████▌     | 184458/400000 [00:17<00:20, 10355.86it/s] 46%|████▋     | 185494/400000 [00:17<00:20, 10304.59it/s] 47%|████▋     | 186531/400000 [00:18<00:20, 10323.40it/s] 47%|████▋     | 187568/400000 [00:18<00:20, 10336.84it/s] 47%|████▋     | 188606/400000 [00:18<00:20, 10349.61it/s] 47%|████▋     | 189643/400000 [00:18<00:20, 10355.63it/s] 48%|████▊     | 190679/400000 [00:18<00:20, 10275.29it/s] 48%|████▊     | 191718/400000 [00:18<00:20, 10308.23it/s] 48%|████▊     | 192757/400000 [00:18<00:20, 10331.41it/s] 48%|████▊     | 193798/400000 [00:18<00:19, 10353.12it/s] 49%|████▊     | 194834/400000 [00:18<00:19, 10283.93it/s] 49%|████▉     | 195874/400000 [00:18<00:19, 10316.21it/s] 49%|████▉     | 196913/400000 [00:19<00:19, 10336.15it/s] 49%|████▉     | 197953/400000 [00:19<00:19, 10354.01it/s] 50%|████▉     | 198989/400000 [00:19<00:19, 10322.80it/s] 50%|█████     | 200028/400000 [00:19<00:19, 10341.12it/s] 50%|█████     | 201066/400000 [00:19<00:19, 10351.36it/s] 51%|█████     | 202106/400000 [00:19<00:19, 10364.50it/s] 51%|█████     | 203143/400000 [00:19<00:19, 10323.03it/s] 51%|█████     | 204181/400000 [00:19<00:18, 10338.13it/s] 51%|█████▏    | 205220/400000 [00:19<00:18, 10351.81it/s] 52%|█████▏    | 206258/400000 [00:19<00:18, 10359.97it/s] 52%|█████▏    | 207295/400000 [00:20<00:18, 10163.05it/s] 52%|█████▏    | 208332/400000 [00:20<00:18, 10222.74it/s] 52%|█████▏    | 209369/400000 [00:20<00:18, 10266.18it/s] 53%|█████▎    | 210406/400000 [00:20<00:18, 10295.55it/s] 53%|█████▎    | 211436/400000 [00:20<00:18, 10272.54it/s] 53%|█████▎    | 212475/400000 [00:20<00:18, 10304.92it/s] 53%|█████▎    | 213513/400000 [00:20<00:18, 10326.12it/s] 54%|█████▎    | 214552/400000 [00:20<00:17, 10343.05it/s] 54%|█████▍    | 215587/400000 [00:20<00:17, 10298.90it/s] 54%|█████▍    | 216625/400000 [00:20<00:17, 10322.85it/s] 54%|█████▍    | 217663/400000 [00:21<00:17, 10338.82it/s] 55%|█████▍    | 218702/400000 [00:21<00:17, 10353.10it/s] 55%|█████▍    | 219738/400000 [00:21<00:17, 10296.19it/s] 55%|█████▌    | 220776/400000 [00:21<00:17, 10319.89it/s] 55%|█████▌    | 221813/400000 [00:21<00:17, 10332.86it/s] 56%|█████▌    | 222849/400000 [00:21<00:17, 10339.20it/s] 56%|█████▌    | 223883/400000 [00:21<00:17, 10278.51it/s] 56%|█████▌    | 224914/400000 [00:21<00:17, 10286.49it/s] 56%|█████▋    | 225948/400000 [00:21<00:16, 10300.97it/s] 57%|█████▋    | 226982/400000 [00:21<00:16, 10310.15it/s] 57%|█████▋    | 228019/400000 [00:22<00:16, 10325.59it/s] 57%|█████▋    | 229052/400000 [00:22<00:16, 10288.47it/s] 58%|█████▊    | 230091/400000 [00:22<00:16, 10317.06it/s] 58%|█████▊    | 231129/400000 [00:22<00:16, 10335.44it/s] 58%|█████▊    | 232166/400000 [00:22<00:16, 10343.01it/s] 58%|█████▊    | 233201/400000 [00:22<00:16, 10274.59it/s] 59%|█████▊    | 234240/400000 [00:22<00:16, 10307.46it/s] 59%|█████▉    | 235273/400000 [00:22<00:15, 10313.89it/s] 59%|█████▉    | 236311/400000 [00:22<00:15, 10333.26it/s] 59%|█████▉    | 237345/400000 [00:22<00:15, 10283.05it/s] 60%|█████▉    | 238382/400000 [00:23<00:15, 10306.68it/s] 60%|█████▉    | 239419/400000 [00:23<00:15, 10324.47it/s] 60%|██████    | 240455/400000 [00:23<00:15, 10334.04it/s] 60%|██████    | 241489/400000 [00:23<00:15, 10279.57it/s] 61%|██████    | 242527/400000 [00:23<00:15, 10306.69it/s] 61%|██████    | 243566/400000 [00:23<00:15, 10328.57it/s] 61%|██████    | 244604/400000 [00:23<00:15, 10342.01it/s] 61%|██████▏   | 245639/400000 [00:23<00:14, 10317.61it/s] 62%|██████▏   | 246678/400000 [00:23<00:14, 10336.88it/s] 62%|██████▏   | 247715/400000 [00:24<00:14, 10346.48it/s] 62%|██████▏   | 248752/400000 [00:24<00:14, 10350.66it/s] 62%|██████▏   | 249788/400000 [00:24<00:14, 10300.15it/s] 63%|██████▎   | 250826/400000 [00:24<00:14, 10322.95it/s] 63%|██████▎   | 251864/400000 [00:24<00:14, 10338.91it/s] 63%|██████▎   | 252899/400000 [00:24<00:14, 10340.85it/s] 63%|██████▎   | 253934/400000 [00:24<00:14, 10296.28it/s] 64%|██████▎   | 254970/400000 [00:24<00:14, 10314.22it/s] 64%|██████▍   | 256007/400000 [00:24<00:13, 10327.90it/s] 64%|██████▍   | 257045/400000 [00:24<00:13, 10341.71it/s] 65%|██████▍   | 258080/400000 [00:25<00:13, 10303.57it/s] 65%|██████▍   | 259111/400000 [00:25<00:13, 10252.93it/s] 65%|██████▌   | 260147/400000 [00:25<00:13, 10284.10it/s] 65%|██████▌   | 261184/400000 [00:25<00:13, 10309.30it/s] 66%|██████▌   | 262221/400000 [00:25<00:13, 10325.79it/s] 66%|██████▌   | 263254/400000 [00:25<00:13, 10286.46it/s] 66%|██████▌   | 264290/400000 [00:25<00:13, 10306.67it/s] 66%|██████▋   | 265328/400000 [00:25<00:13, 10326.50it/s] 67%|██████▋   | 266367/400000 [00:25<00:12, 10342.78it/s] 67%|██████▋   | 267402/400000 [00:25<00:12, 10299.32it/s] 67%|██████▋   | 268440/400000 [00:26<00:12, 10322.23it/s] 67%|██████▋   | 269476/400000 [00:26<00:12, 10332.53it/s] 68%|██████▊   | 270514/400000 [00:26<00:12, 10344.40it/s] 68%|██████▊   | 271549/400000 [00:26<00:12, 10304.88it/s] 68%|██████▊   | 272587/400000 [00:26<00:12, 10325.19it/s] 68%|██████▊   | 273624/400000 [00:26<00:12, 10338.08it/s] 69%|██████▊   | 274658/400000 [00:26<00:12, 10336.57it/s] 69%|██████▉   | 275692/400000 [00:26<00:12, 10314.24it/s] 69%|██████▉   | 276730/400000 [00:26<00:11, 10332.75it/s] 69%|██████▉   | 277767/400000 [00:26<00:11, 10343.74it/s] 70%|██████▉   | 278805/400000 [00:27<00:11, 10353.17it/s] 70%|██████▉   | 279841/400000 [00:27<00:11, 10341.53it/s] 70%|███████   | 280877/400000 [00:27<00:11, 10345.29it/s] 70%|███████   | 281913/400000 [00:27<00:11, 10347.84it/s] 71%|███████   | 282949/400000 [00:27<00:11, 10350.58it/s] 71%|███████   | 283985/400000 [00:27<00:11, 10327.42it/s] 71%|███████▏  | 285021/400000 [00:27<00:11, 10334.95it/s] 72%|███████▏  | 286059/400000 [00:27<00:11, 10345.67it/s] 72%|███████▏  | 287096/400000 [00:27<00:10, 10351.75it/s] 72%|███████▏  | 288132/400000 [00:27<00:10, 10320.51it/s] 72%|███████▏  | 289169/400000 [00:28<00:10, 10333.62it/s] 73%|███████▎  | 290204/400000 [00:28<00:10, 10337.44it/s] 73%|███████▎  | 291242/400000 [00:28<00:10, 10349.91it/s] 73%|███████▎  | 292278/400000 [00:28<00:10, 10297.73it/s] 73%|███████▎  | 293308/400000 [00:28<00:10, 10272.53it/s] 74%|███████▎  | 294345/400000 [00:28<00:10, 10301.28it/s] 74%|███████▍  | 295383/400000 [00:28<00:10, 10322.68it/s] 74%|███████▍  | 296417/400000 [00:28<00:10, 10327.06it/s] 74%|███████▍  | 297450/400000 [00:28<00:09, 10298.01it/s] 75%|███████▍  | 298480/400000 [00:28<00:09, 10292.63it/s] 75%|███████▍  | 299517/400000 [00:29<00:09, 10314.77it/s] 75%|███████▌  | 300555/400000 [00:29<00:09, 10333.52it/s] 75%|███████▌  | 301589/400000 [00:29<00:09, 10290.75it/s] 76%|███████▌  | 302626/400000 [00:29<00:09, 10313.94it/s] 76%|███████▌  | 303664/400000 [00:29<00:09, 10332.93it/s] 76%|███████▌  | 304702/400000 [00:29<00:09, 10345.24it/s] 76%|███████▋  | 305737/400000 [00:29<00:09, 10294.05it/s] 77%|███████▋  | 306773/400000 [00:29<00:09, 10312.78it/s] 77%|███████▋  | 307809/400000 [00:29<00:08, 10325.58it/s] 77%|███████▋  | 308846/400000 [00:29<00:08, 10338.58it/s] 77%|███████▋  | 309880/400000 [00:30<00:08, 10319.95it/s] 78%|███████▊  | 310913/400000 [00:30<00:08, 10271.03it/s] 78%|███████▊  | 311949/400000 [00:30<00:08, 10296.59it/s] 78%|███████▊  | 312987/400000 [00:30<00:08, 10321.10it/s] 79%|███████▊  | 314020/400000 [00:30<00:08, 10293.52it/s] 79%|███████▉  | 315058/400000 [00:30<00:08, 10318.94it/s] 79%|███████▉  | 316095/400000 [00:30<00:08, 10333.64it/s] 79%|███████▉  | 317131/400000 [00:30<00:08, 10339.60it/s] 80%|███████▉  | 318165/400000 [00:30<00:07, 10300.12it/s] 80%|███████▉  | 319202/400000 [00:30<00:07, 10319.73it/s] 80%|████████  | 320239/400000 [00:31<00:07, 10334.36it/s] 80%|████████  | 321273/400000 [00:31<00:07, 10322.71it/s] 81%|████████  | 322306/400000 [00:31<00:07, 10283.12it/s] 81%|████████  | 323343/400000 [00:31<00:07, 10308.04it/s] 81%|████████  | 324380/400000 [00:31<00:07, 10324.16it/s] 81%|████████▏ | 325416/400000 [00:31<00:07, 10333.85it/s] 82%|████████▏ | 326450/400000 [00:31<00:07, 10250.56it/s] 82%|████████▏ | 327477/400000 [00:31<00:07, 10256.29it/s] 82%|████████▏ | 328515/400000 [00:31<00:06, 10290.95it/s] 82%|████████▏ | 329552/400000 [00:31<00:06, 10312.82it/s] 83%|████████▎ | 330584/400000 [00:32<00:06, 10314.16it/s] 83%|████████▎ | 331616/400000 [00:32<00:06, 10297.30it/s] 83%|████████▎ | 332652/400000 [00:32<00:06, 10315.77it/s] 83%|████████▎ | 333691/400000 [00:32<00:06, 10336.85it/s] 84%|████████▎ | 334729/400000 [00:32<00:06, 10349.14it/s] 84%|████████▍ | 335764/400000 [00:32<00:06, 10284.58it/s] 84%|████████▍ | 336801/400000 [00:32<00:06, 10307.22it/s] 84%|████████▍ | 337838/400000 [00:32<00:06, 10324.63it/s] 85%|████████▍ | 338875/400000 [00:32<00:05, 10335.80it/s] 85%|████████▍ | 339909/400000 [00:32<00:05, 10288.14it/s] 85%|████████▌ | 340947/400000 [00:33<00:05, 10313.32it/s] 85%|████████▌ | 341986/400000 [00:33<00:05, 10334.19it/s] 86%|████████▌ | 343024/400000 [00:33<00:05, 10345.26it/s] 86%|████████▌ | 344059/400000 [00:33<00:05, 10322.57it/s] 86%|████████▋ | 345097/400000 [00:33<00:05, 10338.87it/s] 87%|████████▋ | 346135/400000 [00:33<00:05, 10349.34it/s] 87%|████████▋ | 347172/400000 [00:33<00:05, 10353.39it/s] 87%|████████▋ | 348208/400000 [00:33<00:05, 10308.01it/s] 87%|████████▋ | 349245/400000 [00:33<00:04, 10325.05it/s] 88%|████████▊ | 350282/400000 [00:33<00:04, 10337.83it/s] 88%|████████▊ | 351319/400000 [00:34<00:04, 10346.54it/s] 88%|████████▊ | 352354/400000 [00:34<00:04, 10280.67it/s] 88%|████████▊ | 353391/400000 [00:34<00:04, 10305.13it/s] 89%|████████▊ | 354426/400000 [00:34<00:04, 10318.49it/s] 89%|████████▉ | 355464/400000 [00:34<00:04, 10335.22it/s] 89%|████████▉ | 356498/400000 [00:34<00:04, 10320.28it/s] 89%|████████▉ | 357535/400000 [00:34<00:04, 10332.87it/s] 90%|████████▉ | 358573/400000 [00:34<00:04, 10345.67it/s] 90%|████████▉ | 359611/400000 [00:34<00:03, 10353.66it/s] 90%|█████████ | 360647/400000 [00:34<00:03, 10316.81it/s] 90%|█████████ | 361685/400000 [00:35<00:03, 10333.17it/s] 91%|█████████ | 362722/400000 [00:35<00:03, 10343.72it/s] 91%|█████████ | 363758/400000 [00:35<00:03, 10347.99it/s] 91%|█████████ | 364793/400000 [00:35<00:03, 10301.28it/s] 91%|█████████▏| 365828/400000 [00:35<00:03, 10315.16it/s] 92%|█████████▏| 366865/400000 [00:35<00:03, 10331.31it/s] 92%|█████████▏| 367902/400000 [00:35<00:03, 10340.82it/s] 92%|█████████▏| 368937/400000 [00:35<00:03, 10319.09it/s] 92%|█████████▏| 369969/400000 [00:35<00:02, 10311.18it/s] 93%|█████████▎| 371005/400000 [00:35<00:02, 10324.62it/s] 93%|█████████▎| 372043/400000 [00:36<00:02, 10339.38it/s] 93%|█████████▎| 373077/400000 [00:36<00:02, 10338.22it/s] 94%|█████████▎| 374111/400000 [00:36<00:02, 10287.06it/s] 94%|█████████▍| 375148/400000 [00:36<00:02, 10310.75it/s] 94%|█████████▍| 376184/400000 [00:36<00:02, 10324.93it/s] 94%|█████████▍| 377222/400000 [00:36<00:02, 10340.05it/s] 95%|█████████▍| 378257/400000 [00:36<00:02, 10284.37it/s] 95%|█████████▍| 379294/400000 [00:36<00:02, 10309.32it/s] 95%|█████████▌| 380332/400000 [00:36<00:01, 10328.91it/s] 95%|█████████▌| 381370/400000 [00:36<00:01, 10341.99it/s] 96%|█████████▌| 382405/400000 [00:37<00:01, 10295.66it/s] 96%|█████████▌| 383443/400000 [00:37<00:01, 10320.31it/s] 96%|█████████▌| 384476/400000 [00:37<00:01, 10317.34it/s] 96%|█████████▋| 385515/400000 [00:37<00:01, 10337.54it/s] 97%|█████████▋| 386549/400000 [00:37<00:01, 10326.88it/s] 97%|█████████▋| 387585/400000 [00:37<00:01, 10336.54it/s] 97%|█████████▋| 388622/400000 [00:37<00:01, 10345.25it/s] 97%|█████████▋| 389658/400000 [00:37<00:00, 10349.27it/s] 98%|█████████▊| 390693/400000 [00:37<00:00, 10337.40it/s] 98%|█████████▊| 391729/400000 [00:37<00:00, 10343.31it/s] 98%|█████████▊| 392767/400000 [00:38<00:00, 10354.09it/s] 98%|█████████▊| 393804/400000 [00:38<00:00, 10356.94it/s] 99%|█████████▊| 394840/400000 [00:38<00:00, 10264.35it/s] 99%|█████████▉| 395878/400000 [00:38<00:00, 10296.47it/s] 99%|█████████▉| 396914/400000 [00:38<00:00, 10313.08it/s] 99%|█████████▉| 397952/400000 [00:38<00:00, 10330.94it/s]100%|█████████▉| 398986/400000 [00:38<00:00, 10328.23it/s]100%|█████████▉| 399999/400000 [00:38<00:00, 10320.86it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1eadcb9128>, <torchtext.data.dataset.TabularDataset object at 0x7f1eadcb9278>, <torchtext.vocab.Vocab object at 0x7f1eadcb9198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 41.73 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 64.12 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 32.58 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.93 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.93 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.92 file/s]2020-07-08 00:16:36.235299: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-08 00:16:36.239236: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2593905000 Hz
2020-07-08 00:16:36.239425: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ccf7078380 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-08 00:16:36.239445: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3424256/9912422 [00:00<00:00, 34240388.50it/s]9920512it [00:00, 35808427.28it/s]                             
0it [00:00, ?it/s]32768it [00:00, 639578.17it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 473046.58it/s]1654784it [00:00, 12069466.06it/s]                         
0it [00:00, ?it/s]8192it [00:00, 225139.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
