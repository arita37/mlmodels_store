
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f7c4ad7e048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f7c4ad7e048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f7cb69281e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f7cb69281e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f7cd4c6eea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f7cd4c6eea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7c63c56620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7c63c56620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7c63c56620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▌      | 3547136/9912422 [00:00<00:00, 34917367.93it/s]9920512it [00:00, 31713724.34it/s]                             
0it [00:00, ?it/s]32768it [00:00, 620791.78it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477867.91it/s]1654784it [00:00, 11733871.54it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202314.85it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c4ac93940>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c4ac8abe0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f7c63c56268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f7c63c56268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f7c63c56268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:13,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:13,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:12,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:12,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:12,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:11,  2.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:50,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:50,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:49,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:49,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:49,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:48,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:48,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:48,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.07 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:33,  4.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:33,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:33,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:33,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:33,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:32,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:31,  4.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:22,  6.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:22,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:22,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:22,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:22,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:22,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:21,  6.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:14,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:14,  8.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 20.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 26.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.69 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 47.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 47.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 54.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 61.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 64.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 66.74 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.44s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.41 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.44s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 77.41 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.90 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ url]
0 examples [00:00, ? examples/s]2020-07-04 00:11:14.708780: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-04 00:11:14.722885: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-04 00:11:14.723966: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e6d0440630 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-04 00:11:14.723994: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  9.43 examples/s]83 examples [00:00, 13.40 examples/s]164 examples [00:00, 19.01 examples/s]244 examples [00:00, 26.88 examples/s]332 examples [00:00, 37.89 examples/s]415 examples [00:00, 53.09 examples/s]498 examples [00:00, 73.82 examples/s]577 examples [00:00, 101.35 examples/s]655 examples [00:00, 137.11 examples/s]734 examples [00:01, 182.27 examples/s]818 examples [00:01, 238.09 examples/s]904 examples [00:01, 303.73 examples/s]989 examples [00:01, 375.79 examples/s]1071 examples [00:01, 448.59 examples/s]1153 examples [00:01, 515.94 examples/s]1235 examples [00:01, 575.32 examples/s]1316 examples [00:01, 621.85 examples/s]1403 examples [00:01, 679.91 examples/s]1487 examples [00:01, 719.21 examples/s]1571 examples [00:02, 749.53 examples/s]1657 examples [00:02, 777.90 examples/s]1742 examples [00:02, 796.03 examples/s]1827 examples [00:02, 809.03 examples/s]1913 examples [00:02, 822.91 examples/s]1998 examples [00:02, 807.97 examples/s]2081 examples [00:02, 792.29 examples/s]2162 examples [00:02, 788.58 examples/s]2248 examples [00:02, 807.67 examples/s]2336 examples [00:02, 826.43 examples/s]2422 examples [00:03, 835.37 examples/s]2506 examples [00:03, 828.11 examples/s]2592 examples [00:03, 836.66 examples/s]2678 examples [00:03, 841.10 examples/s]2763 examples [00:03, 832.61 examples/s]2847 examples [00:03, 811.76 examples/s]2929 examples [00:03, 808.62 examples/s]3012 examples [00:03, 814.67 examples/s]3094 examples [00:03, 793.17 examples/s]3175 examples [00:03, 796.41 examples/s]3255 examples [00:04, 795.96 examples/s]3335 examples [00:04, 791.63 examples/s]3415 examples [00:04, 789.54 examples/s]3495 examples [00:04, 785.28 examples/s]3578 examples [00:04, 797.98 examples/s]3663 examples [00:04, 812.68 examples/s]3751 examples [00:04, 830.65 examples/s]3836 examples [00:04, 835.93 examples/s]3920 examples [00:04, 836.26 examples/s]4007 examples [00:04, 843.35 examples/s]4092 examples [00:05, 838.41 examples/s]4176 examples [00:05, 832.81 examples/s]4262 examples [00:05, 836.93 examples/s]4347 examples [00:05, 840.57 examples/s]4432 examples [00:05, 840.56 examples/s]4518 examples [00:05, 844.65 examples/s]4603 examples [00:05, 818.57 examples/s]4688 examples [00:05, 826.14 examples/s]4771 examples [00:05, 824.58 examples/s]4858 examples [00:06, 835.97 examples/s]4942 examples [00:06, 834.84 examples/s]5030 examples [00:06, 846.62 examples/s]5115 examples [00:06, 833.49 examples/s]5199 examples [00:06, 828.11 examples/s]5283 examples [00:06, 830.55 examples/s]5367 examples [00:06, 762.30 examples/s]5449 examples [00:06, 776.70 examples/s]5537 examples [00:06, 803.57 examples/s]5621 examples [00:06, 811.88 examples/s]5703 examples [00:07, 805.59 examples/s]5784 examples [00:07, 781.04 examples/s]5864 examples [00:07, 784.87 examples/s]5943 examples [00:07, 768.03 examples/s]6021 examples [00:07, 767.71 examples/s]6099 examples [00:07, 770.79 examples/s]6177 examples [00:07, 769.93 examples/s]6262 examples [00:07, 790.37 examples/s]6342 examples [00:07, 787.91 examples/s]6426 examples [00:07, 801.74 examples/s]6507 examples [00:08, 755.43 examples/s]6584 examples [00:08, 758.95 examples/s]6669 examples [00:08, 782.81 examples/s]6751 examples [00:08, 791.41 examples/s]6832 examples [00:08, 794.24 examples/s]6912 examples [00:08, 787.26 examples/s]6998 examples [00:08, 807.56 examples/s]7080 examples [00:08, 787.98 examples/s]7166 examples [00:08, 808.14 examples/s]7248 examples [00:09, 806.25 examples/s]7330 examples [00:09, 810.21 examples/s]7414 examples [00:09, 816.96 examples/s]7496 examples [00:09, 811.67 examples/s]7578 examples [00:09, 807.97 examples/s]7659 examples [00:09, 801.60 examples/s]7741 examples [00:09, 805.43 examples/s]7823 examples [00:09, 808.11 examples/s]7909 examples [00:09, 823.00 examples/s]7995 examples [00:09, 832.80 examples/s]8079 examples [00:10, 832.17 examples/s]8163 examples [00:10, 824.10 examples/s]8247 examples [00:10, 828.75 examples/s]8334 examples [00:10, 838.40 examples/s]8418 examples [00:10, 835.38 examples/s]8504 examples [00:10, 842.36 examples/s]8589 examples [00:10, 833.76 examples/s]8673 examples [00:10, 832.67 examples/s]8757 examples [00:10, 813.00 examples/s]8839 examples [00:10, 798.56 examples/s]8920 examples [00:11, 784.46 examples/s]9000 examples [00:11, 787.66 examples/s]9089 examples [00:11, 814.01 examples/s]9171 examples [00:11, 812.99 examples/s]9256 examples [00:11, 823.24 examples/s]9341 examples [00:11, 828.59 examples/s]9424 examples [00:11, 809.01 examples/s]9506 examples [00:11, 797.50 examples/s]9586 examples [00:11, 790.99 examples/s]9668 examples [00:12, 797.98 examples/s]9751 examples [00:12, 804.69 examples/s]9834 examples [00:12, 809.90 examples/s]9918 examples [00:12, 818.13 examples/s]10000 examples [00:12, 814.55 examples/s]10082 examples [00:12, 750.18 examples/s]10159 examples [00:12, 736.36 examples/s]10247 examples [00:12, 772.01 examples/s]10331 examples [00:12, 789.31 examples/s]10411 examples [00:12, 779.00 examples/s]10494 examples [00:13, 792.84 examples/s]10574 examples [00:13, 788.36 examples/s]10654 examples [00:13, 787.09 examples/s]10740 examples [00:13, 807.09 examples/s]10826 examples [00:13, 819.76 examples/s]10911 examples [00:13, 826.65 examples/s]10995 examples [00:13, 827.74 examples/s]11078 examples [00:13, 827.76 examples/s]11168 examples [00:13, 847.71 examples/s]11253 examples [00:13, 838.77 examples/s]11338 examples [00:14, 838.18 examples/s]11424 examples [00:14, 844.28 examples/s]11509 examples [00:14, 827.36 examples/s]11594 examples [00:14, 831.32 examples/s]11682 examples [00:14, 845.31 examples/s]11767 examples [00:14, 825.54 examples/s]11850 examples [00:14, 817.53 examples/s]11932 examples [00:14, 815.65 examples/s]12014 examples [00:14, 807.34 examples/s]12098 examples [00:14, 815.71 examples/s]12180 examples [00:15, 811.47 examples/s]12265 examples [00:15, 819.74 examples/s]12348 examples [00:15, 816.52 examples/s]12435 examples [00:15, 829.79 examples/s]12519 examples [00:15, 808.01 examples/s]12601 examples [00:15, 809.30 examples/s]12684 examples [00:15, 813.61 examples/s]12771 examples [00:15, 828.29 examples/s]12856 examples [00:15, 833.57 examples/s]12940 examples [00:16, 832.50 examples/s]13026 examples [00:16, 838.14 examples/s]13110 examples [00:16, 834.85 examples/s]13197 examples [00:16, 844.29 examples/s]13282 examples [00:16, 834.92 examples/s]13366 examples [00:16, 833.97 examples/s]13453 examples [00:16, 842.08 examples/s]13538 examples [00:16, 835.22 examples/s]13622 examples [00:16, 824.25 examples/s]13709 examples [00:16, 836.32 examples/s]13794 examples [00:17, 839.44 examples/s]13879 examples [00:17, 834.40 examples/s]13964 examples [00:17, 835.99 examples/s]14049 examples [00:17, 840.07 examples/s]14134 examples [00:17, 832.19 examples/s]14219 examples [00:17, 835.76 examples/s]14303 examples [00:17, 836.92 examples/s]14387 examples [00:17, 825.64 examples/s]14470 examples [00:17, 826.80 examples/s]14553 examples [00:17, 818.99 examples/s]14641 examples [00:18, 834.60 examples/s]14725 examples [00:18, 829.04 examples/s]14808 examples [00:18, 816.11 examples/s]14890 examples [00:18, 806.76 examples/s]14974 examples [00:18, 814.74 examples/s]15056 examples [00:18, 811.60 examples/s]15138 examples [00:18, 785.08 examples/s]15221 examples [00:18, 797.39 examples/s]15308 examples [00:18, 815.93 examples/s]15390 examples [00:18, 814.79 examples/s]15475 examples [00:19, 824.20 examples/s]15558 examples [00:19, 817.60 examples/s]15640 examples [00:19, 797.94 examples/s]15729 examples [00:19, 821.19 examples/s]15812 examples [00:19, 808.54 examples/s]15895 examples [00:19, 812.34 examples/s]15979 examples [00:19, 819.23 examples/s]16062 examples [00:19, 808.61 examples/s]16146 examples [00:19, 817.28 examples/s]16231 examples [00:20, 826.14 examples/s]16317 examples [00:20, 835.87 examples/s]16402 examples [00:20, 838.29 examples/s]16486 examples [00:20, 815.07 examples/s]16568 examples [00:20, 805.60 examples/s]16649 examples [00:20, 793.42 examples/s]16729 examples [00:20, 788.84 examples/s]16813 examples [00:20, 801.83 examples/s]16894 examples [00:20, 797.73 examples/s]16976 examples [00:20, 803.47 examples/s]17057 examples [00:21, 799.26 examples/s]17137 examples [00:21, 776.15 examples/s]17220 examples [00:21, 789.90 examples/s]17304 examples [00:21, 801.95 examples/s]17388 examples [00:21, 811.38 examples/s]17470 examples [00:21, 813.19 examples/s]17552 examples [00:21, 809.19 examples/s]17635 examples [00:21, 814.10 examples/s]17717 examples [00:21, 805.80 examples/s]17801 examples [00:21, 815.04 examples/s]17885 examples [00:22, 819.85 examples/s]17968 examples [00:22, 819.31 examples/s]18050 examples [00:22, 816.26 examples/s]18132 examples [00:22, 807.46 examples/s]18213 examples [00:22, 788.10 examples/s]18292 examples [00:22, 787.76 examples/s]18374 examples [00:22, 794.08 examples/s]18458 examples [00:22, 805.96 examples/s]18541 examples [00:22, 812.36 examples/s]18626 examples [00:22, 821.65 examples/s]18710 examples [00:23, 824.30 examples/s]18796 examples [00:23, 832.09 examples/s]18880 examples [00:23, 827.94 examples/s]18964 examples [00:23, 829.26 examples/s]19048 examples [00:23, 832.13 examples/s]19133 examples [00:23, 835.47 examples/s]19218 examples [00:23, 838.31 examples/s]19303 examples [00:23, 839.15 examples/s]19387 examples [00:23, 829.74 examples/s]19471 examples [00:24, 790.68 examples/s]19551 examples [00:24, 788.97 examples/s]19631 examples [00:24, 783.17 examples/s]19714 examples [00:24, 795.87 examples/s]19794 examples [00:24, 786.72 examples/s]19877 examples [00:24, 797.93 examples/s]19957 examples [00:24, 792.43 examples/s]20037 examples [00:24, 581.05 examples/s]20121 examples [00:24, 639.88 examples/s]20197 examples [00:25, 671.19 examples/s]20281 examples [00:25, 712.72 examples/s]20362 examples [00:25, 738.17 examples/s]20442 examples [00:25, 752.59 examples/s]20520 examples [00:25, 757.06 examples/s]20605 examples [00:25, 781.95 examples/s]20689 examples [00:25, 797.60 examples/s]20773 examples [00:25, 809.59 examples/s]20856 examples [00:25, 814.44 examples/s]20943 examples [00:25, 829.55 examples/s]21030 examples [00:26, 840.13 examples/s]21115 examples [00:26, 800.13 examples/s]21201 examples [00:26, 814.55 examples/s]21284 examples [00:26, 817.77 examples/s]21368 examples [00:26, 821.90 examples/s]21451 examples [00:26, 819.76 examples/s]21538 examples [00:26, 832.30 examples/s]21624 examples [00:26, 838.09 examples/s]21708 examples [00:26, 829.38 examples/s]21794 examples [00:26, 837.40 examples/s]21880 examples [00:27, 842.45 examples/s]21965 examples [00:27, 839.98 examples/s]22050 examples [00:27, 820.83 examples/s]22134 examples [00:27, 824.39 examples/s]22217 examples [00:27, 815.57 examples/s]22299 examples [00:27, 787.90 examples/s]22382 examples [00:27, 798.39 examples/s]22463 examples [00:27, 796.04 examples/s]22543 examples [00:27, 787.65 examples/s]22627 examples [00:28, 800.41 examples/s]22713 examples [00:28, 815.94 examples/s]22798 examples [00:28, 823.60 examples/s]22881 examples [00:28, 800.80 examples/s]22964 examples [00:28, 807.59 examples/s]23049 examples [00:28, 818.81 examples/s]23132 examples [00:28, 815.46 examples/s]23221 examples [00:28, 835.04 examples/s]23306 examples [00:28, 838.09 examples/s]23394 examples [00:28, 849.76 examples/s]23482 examples [00:29, 856.39 examples/s]23570 examples [00:29, 863.33 examples/s]23659 examples [00:29, 870.05 examples/s]23747 examples [00:29, 846.70 examples/s]23832 examples [00:29, 842.80 examples/s]23917 examples [00:29, 833.76 examples/s]24001 examples [00:29, 833.93 examples/s]24087 examples [00:29, 840.61 examples/s]24172 examples [00:29, 837.27 examples/s]24256 examples [00:29, 834.00 examples/s]24346 examples [00:30, 851.60 examples/s]24436 examples [00:30, 864.28 examples/s]24526 examples [00:30, 872.80 examples/s]24614 examples [00:30, 874.91 examples/s]24702 examples [00:30, 851.20 examples/s]24788 examples [00:30, 841.43 examples/s]24874 examples [00:30, 842.95 examples/s]24963 examples [00:30, 854.15 examples/s]25050 examples [00:30, 856.30 examples/s]25136 examples [00:30, 829.92 examples/s]25223 examples [00:31, 841.51 examples/s]25308 examples [00:31, 830.52 examples/s]25392 examples [00:31, 821.12 examples/s]25478 examples [00:31, 823.49 examples/s]25564 examples [00:31, 832.46 examples/s]25650 examples [00:31, 839.06 examples/s]25736 examples [00:31, 842.80 examples/s]25826 examples [00:31, 856.68 examples/s]25912 examples [00:31, 853.60 examples/s]26001 examples [00:32, 861.64 examples/s]26090 examples [00:32, 868.27 examples/s]26180 examples [00:32, 875.37 examples/s]26270 examples [00:32, 879.84 examples/s]26359 examples [00:32, 871.45 examples/s]26448 examples [00:32, 876.69 examples/s]26536 examples [00:32, 860.93 examples/s]26624 examples [00:32, 863.62 examples/s]26711 examples [00:32, 853.29 examples/s]26799 examples [00:32, 860.76 examples/s]26886 examples [00:33, 853.33 examples/s]26975 examples [00:33, 863.48 examples/s]27062 examples [00:33, 864.03 examples/s]27152 examples [00:33, 873.29 examples/s]27241 examples [00:33, 878.09 examples/s]27329 examples [00:33, 835.36 examples/s]27418 examples [00:33, 848.35 examples/s]27507 examples [00:33, 859.56 examples/s]27594 examples [00:33, 850.55 examples/s]27680 examples [00:33, 827.39 examples/s]27765 examples [00:34, 831.69 examples/s]27853 examples [00:34, 841.77 examples/s]27940 examples [00:34, 849.27 examples/s]28029 examples [00:34, 860.28 examples/s]28116 examples [00:34, 857.14 examples/s]28204 examples [00:34, 862.78 examples/s]28293 examples [00:34, 868.67 examples/s]28381 examples [00:34, 870.16 examples/s]28469 examples [00:34, 872.31 examples/s]28557 examples [00:34, 852.03 examples/s]28644 examples [00:35, 856.38 examples/s]28733 examples [00:35, 865.07 examples/s]28822 examples [00:35, 871.93 examples/s]28911 examples [00:35, 876.30 examples/s]28999 examples [00:35, 862.77 examples/s]29086 examples [00:35, 852.09 examples/s]29172 examples [00:35, 853.25 examples/s]29258 examples [00:35, 830.12 examples/s]29347 examples [00:35, 845.92 examples/s]29437 examples [00:36, 860.26 examples/s]29526 examples [00:36, 866.82 examples/s]29614 examples [00:36, 868.50 examples/s]29701 examples [00:36, 863.51 examples/s]29789 examples [00:36, 868.12 examples/s]29877 examples [00:36, 869.25 examples/s]29965 examples [00:36, 871.86 examples/s]30053 examples [00:36, 801.92 examples/s]30142 examples [00:36, 824.48 examples/s]30230 examples [00:36, 839.32 examples/s]30315 examples [00:37, 830.88 examples/s]30400 examples [00:37, 834.80 examples/s]30488 examples [00:37, 845.75 examples/s]30578 examples [00:37, 860.50 examples/s]30665 examples [00:37, 856.97 examples/s]30753 examples [00:37, 861.70 examples/s]30840 examples [00:37, 843.19 examples/s]30926 examples [00:37, 833.56 examples/s]31013 examples [00:37, 842.87 examples/s]31100 examples [00:37, 848.33 examples/s]31188 examples [00:38, 856.78 examples/s]31277 examples [00:38, 864.13 examples/s]31364 examples [00:38, 848.14 examples/s]31450 examples [00:38, 850.90 examples/s]31536 examples [00:38, 844.94 examples/s]31625 examples [00:38, 855.39 examples/s]31713 examples [00:38, 861.88 examples/s]31801 examples [00:38, 866.39 examples/s]31891 examples [00:38, 874.13 examples/s]31979 examples [00:38, 863.74 examples/s]32066 examples [00:39, 864.49 examples/s]32155 examples [00:39, 871.68 examples/s]32245 examples [00:39, 877.23 examples/s]32333 examples [00:39, 876.16 examples/s]32422 examples [00:39, 879.53 examples/s]32511 examples [00:39, 879.93 examples/s]32600 examples [00:39, 878.62 examples/s]32689 examples [00:39, 880.57 examples/s]32778 examples [00:39, 878.78 examples/s]32866 examples [00:40, 875.19 examples/s]32954 examples [00:40, 846.42 examples/s]33043 examples [00:40, 856.96 examples/s]33132 examples [00:40, 864.40 examples/s]33222 examples [00:40, 874.45 examples/s]33311 examples [00:40, 877.14 examples/s]33399 examples [00:40, 860.30 examples/s]33486 examples [00:40, 862.69 examples/s]33575 examples [00:40, 868.14 examples/s]33664 examples [00:40, 874.22 examples/s]33752 examples [00:41, 867.43 examples/s]33840 examples [00:41, 869.00 examples/s]33927 examples [00:41, 865.73 examples/s]34015 examples [00:41, 868.26 examples/s]34103 examples [00:41, 869.31 examples/s]34190 examples [00:41, 864.24 examples/s]34277 examples [00:41, 864.74 examples/s]34364 examples [00:41, 830.78 examples/s]34449 examples [00:41, 836.14 examples/s]34534 examples [00:41, 839.10 examples/s]34621 examples [00:42, 846.00 examples/s]34706 examples [00:42, 841.42 examples/s]34795 examples [00:42, 853.89 examples/s]34884 examples [00:42, 863.93 examples/s]34972 examples [00:42, 866.84 examples/s]35061 examples [00:42, 873.48 examples/s]35149 examples [00:42, 844.27 examples/s]35234 examples [00:42, 841.37 examples/s]35319 examples [00:42, 817.70 examples/s]35403 examples [00:42, 824.05 examples/s]35487 examples [00:43, 828.27 examples/s]35572 examples [00:43, 831.58 examples/s]35658 examples [00:43, 838.61 examples/s]35742 examples [00:43, 836.05 examples/s]35828 examples [00:43, 842.73 examples/s]35916 examples [00:43, 852.85 examples/s]36005 examples [00:43, 863.04 examples/s]36093 examples [00:43, 865.73 examples/s]36180 examples [00:43, 846.46 examples/s]36272 examples [00:43, 866.27 examples/s]36359 examples [00:44, 864.87 examples/s]36446 examples [00:44, 851.75 examples/s]36533 examples [00:44, 855.58 examples/s]36619 examples [00:44, 850.97 examples/s]36710 examples [00:44, 865.78 examples/s]36797 examples [00:44, 851.44 examples/s]36883 examples [00:44, 842.51 examples/s]36969 examples [00:44, 845.29 examples/s]37060 examples [00:44, 862.88 examples/s]37150 examples [00:45, 872.99 examples/s]37238 examples [00:45, 873.05 examples/s]37327 examples [00:45, 875.74 examples/s]37418 examples [00:45, 883.44 examples/s]37507 examples [00:45, 881.80 examples/s]37596 examples [00:45, 869.03 examples/s]37683 examples [00:45, 863.43 examples/s]37773 examples [00:45, 873.99 examples/s]37861 examples [00:45, 842.68 examples/s]37946 examples [00:45, 832.29 examples/s]38032 examples [00:46, 838.92 examples/s]38121 examples [00:46, 851.94 examples/s]38207 examples [00:46, 851.05 examples/s]38293 examples [00:46, 833.63 examples/s]38377 examples [00:46, 828.22 examples/s]38460 examples [00:46, 822.62 examples/s]38545 examples [00:46, 829.00 examples/s]38630 examples [00:46, 833.72 examples/s]38714 examples [00:46, 834.96 examples/s]38800 examples [00:46, 841.10 examples/s]38885 examples [00:47, 839.03 examples/s]38969 examples [00:47, 828.17 examples/s]39056 examples [00:47, 840.00 examples/s]39145 examples [00:47, 852.78 examples/s]39234 examples [00:47, 861.11 examples/s]39322 examples [00:47, 865.83 examples/s]39409 examples [00:47, 854.31 examples/s]39496 examples [00:47, 858.41 examples/s]39582 examples [00:47, 857.29 examples/s]39670 examples [00:47, 863.83 examples/s]39757 examples [00:48, 864.31 examples/s]39844 examples [00:48, 859.70 examples/s]39930 examples [00:48, 841.78 examples/s]40015 examples [00:48, 793.22 examples/s]40102 examples [00:48, 813.02 examples/s]40184 examples [00:48, 811.16 examples/s]40271 examples [00:48, 826.85 examples/s]40358 examples [00:48, 839.15 examples/s]40443 examples [00:48, 835.08 examples/s]40530 examples [00:49, 842.34 examples/s]40617 examples [00:49, 844.52 examples/s]40704 examples [00:49, 849.55 examples/s]40790 examples [00:49, 849.57 examples/s]40876 examples [00:49, 821.86 examples/s]40959 examples [00:49, 784.86 examples/s]41045 examples [00:49, 803.84 examples/s]41126 examples [00:49, 804.73 examples/s]41214 examples [00:49, 825.46 examples/s]41297 examples [00:49, 823.10 examples/s]41386 examples [00:50, 840.25 examples/s]41475 examples [00:50, 853.07 examples/s]41561 examples [00:50, 843.92 examples/s]41652 examples [00:50, 861.65 examples/s]41742 examples [00:50, 871.13 examples/s]41830 examples [00:50, 858.12 examples/s]41916 examples [00:50, 854.32 examples/s]42003 examples [00:50, 858.77 examples/s]42089 examples [00:50, 844.06 examples/s]42174 examples [00:50, 823.81 examples/s]42257 examples [00:51, 823.75 examples/s]42344 examples [00:51, 834.04 examples/s]42432 examples [00:51, 846.14 examples/s]42519 examples [00:51, 851.30 examples/s]42605 examples [00:51, 853.53 examples/s]42691 examples [00:51, 853.33 examples/s]42777 examples [00:51, 851.66 examples/s]42863 examples [00:51, 847.81 examples/s]42948 examples [00:51, 845.90 examples/s]43034 examples [00:51, 848.76 examples/s]43119 examples [00:52, 838.28 examples/s]43206 examples [00:52, 846.40 examples/s]43295 examples [00:52, 858.85 examples/s]43381 examples [00:52, 850.35 examples/s]43467 examples [00:52, 833.89 examples/s]43551 examples [00:52, 834.29 examples/s]43639 examples [00:52, 845.26 examples/s]43726 examples [00:52, 850.20 examples/s]43813 examples [00:52, 855.03 examples/s]43899 examples [00:53, 852.61 examples/s]43985 examples [00:53, 832.17 examples/s]44073 examples [00:53, 843.97 examples/s]44158 examples [00:53, 841.69 examples/s]44244 examples [00:53, 844.70 examples/s]44334 examples [00:53, 858.03 examples/s]44421 examples [00:53, 860.60 examples/s]44508 examples [00:53, 850.79 examples/s]44594 examples [00:53, 816.63 examples/s]44677 examples [00:53, 806.72 examples/s]44760 examples [00:54, 811.91 examples/s]44848 examples [00:54, 830.35 examples/s]44938 examples [00:54, 848.75 examples/s]45027 examples [00:54, 859.71 examples/s]45114 examples [00:54, 855.44 examples/s]45201 examples [00:54, 857.67 examples/s]45289 examples [00:54, 862.21 examples/s]45376 examples [00:54, 862.84 examples/s]45463 examples [00:54, 842.53 examples/s]45549 examples [00:54, 847.58 examples/s]45634 examples [00:55, 843.00 examples/s]45719 examples [00:55, 838.00 examples/s]45809 examples [00:55, 853.87 examples/s]45896 examples [00:55, 858.38 examples/s]45982 examples [00:55, 838.00 examples/s]46066 examples [00:55, 829.92 examples/s]46151 examples [00:55, 833.80 examples/s]46240 examples [00:55, 847.19 examples/s]46329 examples [00:55, 857.82 examples/s]46415 examples [00:55, 857.19 examples/s]46503 examples [00:56, 860.97 examples/s]46591 examples [00:56, 865.52 examples/s]46680 examples [00:56, 871.00 examples/s]46769 examples [00:56, 874.37 examples/s]46857 examples [00:56, 864.40 examples/s]46944 examples [00:56, 842.82 examples/s]47030 examples [00:56, 846.58 examples/s]47116 examples [00:56, 847.71 examples/s]47201 examples [00:56, 844.06 examples/s]47287 examples [00:57, 847.10 examples/s]47376 examples [00:57, 857.33 examples/s]47462 examples [00:57, 853.19 examples/s]47551 examples [00:57, 863.04 examples/s]47638 examples [00:57, 858.94 examples/s]47724 examples [00:57, 851.58 examples/s]47812 examples [00:57, 857.37 examples/s]47898 examples [00:57, 850.08 examples/s]47987 examples [00:57, 859.02 examples/s]48075 examples [00:57, 863.26 examples/s]48162 examples [00:58, 846.39 examples/s]48251 examples [00:58, 857.19 examples/s]48339 examples [00:58, 863.38 examples/s]48427 examples [00:58, 867.89 examples/s]48514 examples [00:58, 864.58 examples/s]48601 examples [00:58, 826.71 examples/s]48685 examples [00:58, 817.95 examples/s]48768 examples [00:58, 807.09 examples/s]48849 examples [00:58, 774.86 examples/s]48927 examples [00:58, 774.12 examples/s]49016 examples [00:59, 803.61 examples/s]49103 examples [00:59, 820.80 examples/s]49186 examples [00:59, 819.88 examples/s]49274 examples [00:59, 837.02 examples/s]49361 examples [00:59, 843.90 examples/s]49450 examples [00:59, 856.52 examples/s]49538 examples [00:59, 860.94 examples/s]49625 examples [00:59, 853.91 examples/s]49712 examples [00:59, 856.77 examples/s]49798 examples [01:00, 839.27 examples/s]49883 examples [01:00, 825.99 examples/s]49969 examples [01:00, 835.64 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5709/50000 [00:00<00:00, 57046.53 examples/s] 32%|███▏      | 16156/50000 [00:00<00:00, 66038.91 examples/s] 53%|█████▎    | 26725/50000 [00:00<00:00, 74406.32 examples/s] 74%|███████▍  | 37235/50000 [00:00<00:00, 81550.41 examples/s] 96%|█████████▌| 47869/50000 [00:00<00:00, 87680.37 examples/s]                                                               0 examples [00:00, ? examples/s]64 examples [00:00, 639.14 examples/s]147 examples [00:00, 686.20 examples/s]233 examples [00:00, 723.01 examples/s]308 examples [00:00, 729.69 examples/s]393 examples [00:00, 761.80 examples/s]483 examples [00:00, 797.74 examples/s]567 examples [00:00, 808.40 examples/s]644 examples [00:00, 794.38 examples/s]724 examples [00:00, 794.58 examples/s]812 examples [00:01, 816.16 examples/s]898 examples [00:01, 826.86 examples/s]987 examples [00:01, 843.21 examples/s]1075 examples [00:01, 853.69 examples/s]1161 examples [00:01, 851.84 examples/s]1248 examples [00:01, 856.42 examples/s]1334 examples [00:01, 852.59 examples/s]1420 examples [00:01, 849.04 examples/s]1505 examples [00:01, 843.40 examples/s]1590 examples [00:01, 844.31 examples/s]1679 examples [00:02, 855.33 examples/s]1766 examples [00:02, 856.11 examples/s]1852 examples [00:02, 840.33 examples/s]1939 examples [00:02, 847.10 examples/s]2025 examples [00:02, 849.47 examples/s]2111 examples [00:02, 851.59 examples/s]2197 examples [00:02, 826.70 examples/s]2282 examples [00:02, 831.62 examples/s]2368 examples [00:02, 838.52 examples/s]2452 examples [00:02, 834.13 examples/s]2542 examples [00:03, 851.79 examples/s]2628 examples [00:03, 837.06 examples/s]2716 examples [00:03, 839.25 examples/s]2806 examples [00:03, 855.49 examples/s]2896 examples [00:03, 866.34 examples/s]2985 examples [00:03, 872.10 examples/s]3075 examples [00:03, 877.93 examples/s]3163 examples [00:03, 872.95 examples/s]3251 examples [00:03, 861.24 examples/s]3338 examples [00:03, 851.71 examples/s]3427 examples [00:04, 861.20 examples/s]3514 examples [00:04, 851.91 examples/s]3601 examples [00:04, 856.44 examples/s]3688 examples [00:04, 857.13 examples/s]3776 examples [00:04, 863.46 examples/s]3863 examples [00:04, 859.94 examples/s]3950 examples [00:04, 857.35 examples/s]4038 examples [00:04, 860.99 examples/s]4125 examples [00:04, 861.35 examples/s]4215 examples [00:04, 870.42 examples/s]4303 examples [00:05, 869.24 examples/s]4390 examples [00:05, 869.14 examples/s]4478 examples [00:05, 870.30 examples/s]4566 examples [00:05, 867.80 examples/s]4654 examples [00:05, 867.29 examples/s]4741 examples [00:05, 857.76 examples/s]4829 examples [00:05, 861.89 examples/s]4918 examples [00:05, 870.05 examples/s]5007 examples [00:05, 875.61 examples/s]5095 examples [00:06, 850.88 examples/s]5183 examples [00:06, 857.75 examples/s]5269 examples [00:06, 845.68 examples/s]5357 examples [00:06, 852.86 examples/s]5443 examples [00:06, 851.62 examples/s]5529 examples [00:06, 829.90 examples/s]5615 examples [00:06, 838.63 examples/s]5700 examples [00:06, 819.99 examples/s]5785 examples [00:06, 825.55 examples/s]5868 examples [00:06, 750.31 examples/s]5949 examples [00:07, 765.16 examples/s]6036 examples [00:07, 792.17 examples/s]6121 examples [00:07, 808.65 examples/s]6203 examples [00:07, 757.83 examples/s]6286 examples [00:07, 775.51 examples/s]6366 examples [00:07, 775.97 examples/s]6445 examples [00:07, 778.75 examples/s]6534 examples [00:07, 807.83 examples/s]6618 examples [00:07, 816.62 examples/s]6708 examples [00:08, 837.98 examples/s]6793 examples [00:08, 814.56 examples/s]6879 examples [00:08, 825.32 examples/s]6965 examples [00:08, 833.59 examples/s]7050 examples [00:08, 836.90 examples/s]7139 examples [00:08, 850.89 examples/s]7227 examples [00:08, 856.86 examples/s]7315 examples [00:08, 863.33 examples/s]7402 examples [00:08, 859.34 examples/s]7489 examples [00:08, 849.48 examples/s]7576 examples [00:09, 855.35 examples/s]7663 examples [00:09, 857.45 examples/s]7749 examples [00:09, 852.00 examples/s]7837 examples [00:09, 858.11 examples/s]7924 examples [00:09, 859.67 examples/s]8010 examples [00:09, 852.43 examples/s]8098 examples [00:09, 860.31 examples/s]8185 examples [00:09, 858.17 examples/s]8271 examples [00:09, 840.84 examples/s]8356 examples [00:09, 842.09 examples/s]8441 examples [00:10, 810.15 examples/s]8528 examples [00:10, 825.13 examples/s]8612 examples [00:10, 827.68 examples/s]8695 examples [00:10, 824.48 examples/s]8778 examples [00:10, 823.64 examples/s]8868 examples [00:10, 842.85 examples/s]8956 examples [00:10, 852.72 examples/s]9048 examples [00:10, 869.66 examples/s]9138 examples [00:10, 877.77 examples/s]9226 examples [00:10, 872.25 examples/s]9314 examples [00:11, 832.10 examples/s]9398 examples [00:11, 828.38 examples/s]9487 examples [00:11, 843.33 examples/s]9575 examples [00:11, 851.53 examples/s]9663 examples [00:11, 857.74 examples/s]9749 examples [00:11, 839.35 examples/s]9834 examples [00:11, 805.92 examples/s]9919 examples [00:11, 817.84 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQTMLTR/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteQTMLTR/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7c63c56620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7c63c56620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7c63c56620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7bed0a3278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7bed0a3128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7c63c56620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7c63c56620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f7c63c56620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7bed0fe128>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f7c4ac8a390>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f7be70e71e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f7be70e71e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f7be70e71e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f7be70e72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f7be70e72f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f7be70e72f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=d690885332adceaaac4f4bb0104745613ac950eac557de4bacf291b6aca2eb8c
  Stored in directory: /tmp/pip-ephem-wheel-cache-whme9lon/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:57:40, 12.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:12:12, 16.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:59:44, 24.0kB/s]  .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:00:12, 34.2kB/s].vector_cache/glove.6B.zip:   0%|          | 3.21M/862M [00:01<4:53:32, 48.8kB/s].vector_cache/glove.6B.zip:   1%|          | 6.73M/862M [00:01<3:24:47, 69.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:22:28, 99.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:39:09, 142kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:01<1:09:04, 202kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:01<48:16, 288kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.3M/862M [00:01<33:42, 411kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:01<23:34, 584kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<16:31, 830kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<11:35, 1.18MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:02<08:10, 1.66MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:19, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<06:19, 2.12MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<08:15, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<06:35, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.4M/862M [00:05<04:59, 2.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:14, 2.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:06<05:59, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:34, 2.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:57, 2.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<06:58, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:28, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<04:01, 3.30MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<08:40, 1.52MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<07:24, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<05:30, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<06:56, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<05:57, 2.21MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:13<04:29, 2.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<06:15, 2.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<07:01, 1.86MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<05:34, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:15<04:01, 3.23MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<12:37:29, 17.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<8:51:17, 24.5kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:16<6:11:29, 34.9kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<4:22:22, 49.3kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:18<3:06:13, 69.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<2:10:48, 98.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<1:31:26, 141kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<1:12:55, 177kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<52:09, 247kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<36:46, 349kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<28:38, 447kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<22:37, 566kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<16:23, 780kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:22<11:35, 1.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<17:59, 708kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<13:52, 918kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:24<10:01, 1.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:57, 1.27MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<09:31, 1.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:13, 1.75MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<05:10, 2.44MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<24:54, 506kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<18:43, 673kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<13:24, 938kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:18, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:08, 1.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:25, 1.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:53, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:46, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:03, 2.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:26, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<07:06, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:36, 2.21MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:04, 3.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:31:07, 135kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<1:05:01, 189kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<45:41, 269kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<34:46, 353kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<25:23, 483kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<18:02, 678kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<15:30, 786kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<13:18, 916kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<09:55, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:53, 1.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:26, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:30, 2.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:40, 1.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:12, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:02, 2.97MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:31, 1.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:54, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:46, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:49, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:11, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:33, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<04:00, 2.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<11:40, 1.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<09:23, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:49, 1.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<07:32, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:38, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:53, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:14, 2.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:28, 809kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<11:18, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:11, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<08:27, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:54, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:07, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:18, 1.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:46, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:15, 2.20MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:48, 3.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<10:11, 1.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:17, 1.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:05, 1.89MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:56, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<07:10, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:30, 2.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:59, 2.85MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<09:15, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:38, 1.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:37, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:34, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:15, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:48, 2.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<10:27, 1.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<08:29, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<06:10, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:55, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<07:06, 1.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:57, 2.81MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<07:38, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<06:29, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:46, 2.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:55, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:22, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:55, 2.24MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<03:34, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<08:31, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:04, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<05:10, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:10, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:31, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:03, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<03:41, 2.94MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:20, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:15, 1.73MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:36, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<05:44, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:11, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:51, 2.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:07, 2.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:42, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:33, 2.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:57, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:36, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:28, 2.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:13, 3.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<1:18:35, 134kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<56:04, 188kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<39:25, 267kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<29:57, 350kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<23:10, 452kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<16:39, 628kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<11:46, 886kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<13:00, 800kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:10, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<07:22, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:34, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:24, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:42, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<04:06, 2.50MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:01:58, 166kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<44:13, 232kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<31:12, 328kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<21:52, 466kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<56:34, 180kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<41:40, 245kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<29:39, 344kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<22:17, 454kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<16:37, 609kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<11:50, 854kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<10:37, 948kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<08:27, 1.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:09, 1.63MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:40, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<06:42, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:12, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:14, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:32, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:30, 2.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<02:33, 3.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<23:49, 414kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<17:40, 558kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<12:35, 781kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<11:06, 881kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<09:45, 1.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:19, 1.34MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<05:14, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:34, 1.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<07:00, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:08, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:49, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:03, 1.90MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:45, 2.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:54, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:23, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<03:16, 2.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:31, 2.10MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:07, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:07, 3.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:25, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:03, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:02, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:21, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:02, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:19, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:51, 3.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<02:07, 4.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<36:26, 254kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<27:22, 338kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<19:34, 472kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:04<13:42, 669kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<1:17:16, 119kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<55:00, 167kB/s]  .vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<38:37, 237kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<29:03, 313kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<21:14, 428kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<15:01, 604kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<12:38, 715kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<10:40, 846kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<07:55, 1.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:02, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:46, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<03:44, 2.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<1:06:12, 134kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<47:13, 188kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<33:10, 267kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<25:11, 350kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<19:26, 454kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<13:58, 631kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<09:53, 888kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<10:18, 849kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<08:07, 1.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<05:51, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:18<04:12, 2.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<35:13, 247kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<26:26, 329kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<18:55, 459kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:20<13:15, 650kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<8:28:25, 17.0kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<5:56:30, 24.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<4:09:01, 34.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<2:55:35, 48.7kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<2:04:36, 68.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<1:27:31, 97.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<1:01:02, 139kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<53:14, 159kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<38:07, 222kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<26:47, 316kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<20:39, 408kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<15:16, 551kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<10:52, 771kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<09:33, 873kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<08:22, 997kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:12, 1.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<04:27, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<06:23, 1.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<05:18, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:54, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:39, 1.76MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:05, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:03, 2.67MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:03, 2.00MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:40, 2.21MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:46, 2.92MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:50, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:22, 2.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:31, 3.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<01:52, 4.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<33:35, 238kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<25:07, 319kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<17:54, 446kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<12:37, 631kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<11:33, 686kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<08:54, 890kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<06:25, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<06:19, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:13, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:50, 2.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:31, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<03:57, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:57, 2.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:53, 1.99MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:17, 1.80MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<03:23, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:26, 3.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<7:23:03, 17.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<5:10:41, 24.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<3:36:55, 35.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<2:32:54, 49.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<1:47:42, 70.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<1:15:19, 100kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<54:16, 139kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<38:42, 194kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<27:09, 276kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<20:42, 360kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<16:00, 466kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<11:30, 647kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<08:06, 913kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<09:39, 765kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<07:24, 998kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<05:21, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<05:26, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:25, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:17, 2.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [02:59<02:23, 3.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<55:46, 130kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<40:29, 179kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<28:39, 253kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:01<20:01, 359kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<19:40, 365kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<14:30, 495kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<10:16, 696kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:49, 806kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:36, 935kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:40, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<04:01, 1.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<6:49:26, 17.2kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<4:47:04, 24.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<3:20:22, 35.0kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<2:21:10, 49.4kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<1:40:11, 69.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<1:10:18, 99.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<49:02, 141kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<38:16, 181kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<27:20, 252kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<19:18, 357kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:11<13:31, 506kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<50:19, 136kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<35:52, 191kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<25:09, 271kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<19:07, 354kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<14:45, 459kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<10:38, 635kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<08:29, 790kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<06:37, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<04:47, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:54, 1.35MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:47, 1.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:40, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:37, 1.81MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:11, 2.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:23, 2.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:11, 2.03MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:33, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:45, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:00, 3.22MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<05:05, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:13, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:06, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:39, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:54, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:03, 2.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:11, 2.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<24:15, 259kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<17:36, 357kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<12:26, 504kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<10:07, 615kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:42, 807kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<05:30, 1.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<05:17, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:56, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:42, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<02:40, 2.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:28, 1.36MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:44, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<02:45, 2.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:19, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:32, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:44, 2.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:59, 3.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:24, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<03:41, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:41, 2.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:16, 1.80MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:28, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:41, 2.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:41<01:55, 3.02MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<15:40, 371kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<11:32, 503kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:10, 708kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<07:02, 816kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<06:03, 948kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<04:28, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<03:11, 1.78MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<05:13, 1.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:14, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:05, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:28, 1.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:59, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:13, 2.50MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:51, 1.93MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:33, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:54, 2.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:36, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:55, 1.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:17, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:38, 3.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<14:26, 374kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<10:39, 506kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<07:33, 710kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<06:30, 819kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:37, 947kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:10, 1.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<02:57, 1.78MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<1:02:50, 83.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<44:27, 118kB/s]   .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<31:06, 168kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<22:50, 227kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<16:29, 315kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<11:36, 445kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<09:17, 552kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<07:00, 730kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<05:00, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<04:41, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:47, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:46, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:06, 1.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:40, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<01:59, 2.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:32, 1.93MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:12, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:39, 2.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:18, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:35, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:01, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:28, 3.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:36, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:00, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:13, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:39, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:19, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:44, 2.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:18, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:33, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:01, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:08, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:25, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:53, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:22, 3.30MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:50, 1.18MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:08, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:17, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:38, 1.68MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:46, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:32, 2.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<04:23, 996kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<03:31, 1.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:33, 1.70MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:47, 1.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:51, 1.50MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:10, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:34, 2.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:15, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:42, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:59, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:21, 1.76MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:04, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:31, 2.70MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:01, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:22, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:54, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:40, 2.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:16, 3.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<00:56, 4.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<22:08, 179kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<15:52, 249kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<11:08, 353kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<08:38, 450kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<06:50, 569kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<04:56, 786kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:29, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:41, 1.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:58, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:09, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:22, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:25, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:51, 2.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:19, 2.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:13, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:38, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:55, 1.90MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:10, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:15, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:45, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:15, 2.81MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<26:14, 135kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<18:42, 190kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<13:05, 269kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<09:53, 352kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<07:36, 457kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<05:28, 634kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<03:49, 896kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<05:57, 572kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:31, 754kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<03:13, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:01, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:47, 1.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:05, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:29, 2.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:04, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:28, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:48, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:00, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:40, 1.91MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:14, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:36, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:26, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:03, 2.92MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:28, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:20, 2.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:00, 3.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:24, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:35, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:14, 2.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:54, 3.28MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:47, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:32, 1.90MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:08, 2.56MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:27, 1.96MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:35, 1.79MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:14, 2.30MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<00:53, 3.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:48, 999kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:14, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:37, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:45, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:30, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:06, 2.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<01:23, 1.90MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:14, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:55, 2.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:15, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:08, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:50, 3.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:11, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:05, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:48, 3.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:08, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:02, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:46, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:06, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:15, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:58, 2.43MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:42, 3.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:25, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:14, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:54, 2.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:09, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:59, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:44, 2.99MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:32, 4.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<16:36, 131kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<11:48, 184kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<08:13, 261kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<06:09, 343kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<04:30, 467kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<03:10, 655kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:39, 766kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:16, 896kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:41, 1.20MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:28, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:25, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<01:04, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:46, 2.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:26, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:11, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:52, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:01, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:05, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:50, 2.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:36, 2.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:06, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:08, 1.55MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:52, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:37, 2.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<12:26, 137kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<09:11, 185kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<06:30, 260kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<04:31, 368kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<03:12, 515kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:52, 568kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:43, 597kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<02:04, 783kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:27, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:01, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:53, 402kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<03:20, 466kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:28, 629kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:45, 874kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:44<01:14, 1.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:30, 985kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:39, 895kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:17, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:55, 1.58MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:40, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:08, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:19, 1.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:02, 1.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:45, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:32, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:04, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:15, 1.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:59, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:42, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<00:30, 2.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:33, 501kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:16, 564kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:41, 752kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:11, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:06, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:11, 1.02MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:56, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:40, 1.77MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:28, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:33, 738kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:30, 757kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:08, 998kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:48, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:34, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:00, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:04, 1.00MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:49, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:35, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:25, 2.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:36, 625kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:28, 680kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:06, 908kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:46, 1.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:33, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:57, 983kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:59, 946kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:45, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:32, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:22, 2.30MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:17, 677kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:11, 724kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:54, 951kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:37, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:37, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:43, 1.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:33, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:23, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:16, 2.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:05, 672kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:00, 719kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:45, 943kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:31, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:31, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:35, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:27, 1.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:19, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:13, 2.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<02:11, 271kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<01:44, 341kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<01:14, 470kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:50, 663kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:34, 927kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:43, 721kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:41, 759kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:31, 988kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:21, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:14, 1.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<25:06, 18.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<17:39, 25.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<12:11, 36.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<08:02, 52.2kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<05:21, 74.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<03:52, 99.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<02:49, 136kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<01:58, 191kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<01:16, 271kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:53, 353kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:44, 428kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:31, 586kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:20, 824kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:13, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:18, 798kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:18, 815kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:13, 1.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:08, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:07, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:09, 1.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:04, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.63MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:01, 2.21MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:00, 2.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:04, 567kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:03, 633kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 838kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.17MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<26:06:48,  4.25it/s]  0%|          | 692/400000 [00:00<18:15:09,  6.08it/s]  0%|          | 1362/400000 [00:00<12:45:37,  8.68it/s]  1%|          | 2097/400000 [00:00<8:55:13, 12.39it/s]   1%|          | 2784/400000 [00:00<6:14:17, 17.69it/s]  1%|          | 3515/400000 [00:00<4:21:47, 25.24it/s]  1%|          | 4235/400000 [00:00<3:03:11, 36.00it/s]  1%|          | 4982/400000 [00:00<2:08:15, 51.33it/s]  1%|▏         | 5652/400000 [00:01<1:29:55, 73.09it/s]  2%|▏         | 6385/400000 [00:01<1:03:05, 103.97it/s]  2%|▏         | 7071/400000 [00:01<44:22, 147.56it/s]    2%|▏         | 7769/400000 [00:01<31:17, 208.91it/s]  2%|▏         | 8486/400000 [00:01<22:08, 294.76it/s]  2%|▏         | 9222/400000 [00:01<15:43, 413.98it/s]  2%|▏         | 9931/400000 [00:01<11:16, 576.49it/s]  3%|▎         | 10634/400000 [00:01<08:09, 794.70it/s]  3%|▎         | 11354/400000 [00:01<05:58, 1083.95it/s]  3%|▎         | 12069/400000 [00:01<04:26, 1453.92it/s]  3%|▎         | 12776/400000 [00:02<03:23, 1902.96it/s]  3%|▎         | 13475/400000 [00:02<02:39, 2429.99it/s]  4%|▎         | 14179/400000 [00:02<02:07, 3024.03it/s]  4%|▎         | 14877/400000 [00:02<01:46, 3616.07it/s]  4%|▍         | 15647/400000 [00:02<01:29, 4300.05it/s]  4%|▍         | 16386/400000 [00:02<01:18, 4915.58it/s]  4%|▍         | 17107/400000 [00:02<01:11, 5364.21it/s]  4%|▍         | 17872/400000 [00:02<01:04, 5890.73it/s]  5%|▍         | 18598/400000 [00:02<01:02, 6125.65it/s]  5%|▍         | 19309/400000 [00:02<00:59, 6383.06it/s]  5%|▌         | 20019/400000 [00:03<00:58, 6526.33it/s]  5%|▌         | 20731/400000 [00:03<00:56, 6691.30it/s]  5%|▌         | 21437/400000 [00:03<00:56, 6668.63it/s]  6%|▌         | 22184/400000 [00:03<00:54, 6889.54it/s]  6%|▌         | 22893/400000 [00:03<00:55, 6841.73it/s]  6%|▌         | 23592/400000 [00:03<00:55, 6825.38it/s]  6%|▌         | 24316/400000 [00:03<00:54, 6943.68it/s]  6%|▋         | 25023/400000 [00:03<00:53, 6980.19it/s]  6%|▋         | 25727/400000 [00:03<00:53, 6942.33it/s]  7%|▋         | 26428/400000 [00:03<00:53, 6961.15it/s]  7%|▋         | 27127/400000 [00:04<00:53, 6951.15it/s]  7%|▋         | 27842/400000 [00:04<00:53, 7007.77it/s]  7%|▋         | 28577/400000 [00:04<00:52, 7105.33it/s]  7%|▋         | 29317/400000 [00:04<00:51, 7189.66it/s]  8%|▊         | 30065/400000 [00:04<00:50, 7274.09it/s]  8%|▊         | 30794/400000 [00:04<00:50, 7263.35it/s]  8%|▊         | 31522/400000 [00:04<00:51, 7161.65it/s]  8%|▊         | 32239/400000 [00:04<00:51, 7143.32it/s]  8%|▊         | 32954/400000 [00:04<00:51, 7060.35it/s]  8%|▊         | 33704/400000 [00:05<00:50, 7186.20it/s]  9%|▊         | 34424/400000 [00:05<00:50, 7171.46it/s]  9%|▉         | 35142/400000 [00:05<00:51, 7145.18it/s]  9%|▉         | 35858/400000 [00:05<00:50, 7145.31it/s]  9%|▉         | 36573/400000 [00:05<00:50, 7143.82it/s]  9%|▉         | 37313/400000 [00:05<00:50, 7217.05it/s] 10%|▉         | 38036/400000 [00:05<00:50, 7198.23it/s] 10%|▉         | 38780/400000 [00:05<00:49, 7267.19it/s] 10%|▉         | 39543/400000 [00:05<00:48, 7369.90it/s] 10%|█         | 40290/400000 [00:05<00:48, 7399.56it/s] 10%|█         | 41031/400000 [00:06<00:49, 7301.12it/s] 10%|█         | 41776/400000 [00:06<00:48, 7344.56it/s] 11%|█         | 42516/400000 [00:06<00:48, 7360.52it/s] 11%|█         | 43253/400000 [00:06<00:48, 7361.47it/s] 11%|█         | 44008/400000 [00:06<00:47, 7416.57it/s] 11%|█         | 44750/400000 [00:06<00:49, 7222.96it/s] 11%|█▏        | 45487/400000 [00:06<00:48, 7265.04it/s] 12%|█▏        | 46224/400000 [00:06<00:48, 7295.03it/s] 12%|█▏        | 46955/400000 [00:06<00:48, 7216.10it/s] 12%|█▏        | 47698/400000 [00:06<00:48, 7277.01it/s] 12%|█▏        | 48457/400000 [00:07<00:47, 7365.94it/s] 12%|█▏        | 49195/400000 [00:07<00:49, 7143.91it/s] 12%|█▏        | 49934/400000 [00:07<00:48, 7214.23it/s] 13%|█▎        | 50684/400000 [00:07<00:47, 7297.36it/s] 13%|█▎        | 51427/400000 [00:07<00:47, 7334.21it/s] 13%|█▎        | 52188/400000 [00:07<00:46, 7414.74it/s] 13%|█▎        | 52931/400000 [00:07<00:47, 7325.05it/s] 13%|█▎        | 53665/400000 [00:07<00:47, 7292.30it/s] 14%|█▎        | 54395/400000 [00:07<00:47, 7233.83it/s] 14%|█▍        | 55119/400000 [00:07<00:49, 7032.96it/s] 14%|█▍        | 55824/400000 [00:08<00:49, 6982.39it/s] 14%|█▍        | 56524/400000 [00:08<00:49, 6942.96it/s] 14%|█▍        | 57256/400000 [00:08<00:48, 7050.02it/s] 14%|█▍        | 57983/400000 [00:08<00:48, 7112.11it/s] 15%|█▍        | 58732/400000 [00:08<00:47, 7219.43it/s] 15%|█▍        | 59476/400000 [00:08<00:46, 7282.22it/s] 15%|█▌        | 60206/400000 [00:08<00:47, 7106.58it/s] 15%|█▌        | 60919/400000 [00:08<00:47, 7081.24it/s] 15%|█▌        | 61649/400000 [00:08<00:47, 7143.51it/s] 16%|█▌        | 62365/400000 [00:08<00:48, 7012.72it/s] 16%|█▌        | 63104/400000 [00:09<00:47, 7121.17it/s] 16%|█▌        | 63834/400000 [00:09<00:46, 7171.70it/s] 16%|█▌        | 64553/400000 [00:09<00:47, 7067.30it/s] 16%|█▋        | 65261/400000 [00:09<00:47, 7061.20it/s] 16%|█▋        | 65977/400000 [00:09<00:47, 7088.05it/s] 17%|█▋        | 66687/400000 [00:09<00:48, 6898.43it/s] 17%|█▋        | 67379/400000 [00:09<00:48, 6843.01it/s] 17%|█▋        | 68096/400000 [00:09<00:47, 6936.91it/s] 17%|█▋        | 68791/400000 [00:09<00:48, 6881.35it/s] 17%|█▋        | 69481/400000 [00:09<00:48, 6786.31it/s] 18%|█▊        | 70213/400000 [00:10<00:47, 6937.11it/s] 18%|█▊        | 70909/400000 [00:10<00:48, 6841.35it/s] 18%|█▊        | 71662/400000 [00:10<00:46, 7032.15it/s] 18%|█▊        | 72417/400000 [00:10<00:45, 7178.77it/s] 18%|█▊        | 73179/400000 [00:10<00:44, 7303.62it/s] 18%|█▊        | 73931/400000 [00:10<00:44, 7365.86it/s] 19%|█▊        | 74670/400000 [00:10<00:44, 7361.92it/s] 19%|█▉        | 75408/400000 [00:10<00:44, 7259.88it/s] 19%|█▉        | 76158/400000 [00:10<00:44, 7329.32it/s] 19%|█▉        | 76897/400000 [00:11<00:43, 7347.06it/s] 19%|█▉        | 77633/400000 [00:11<00:44, 7321.85it/s] 20%|█▉        | 78376/400000 [00:11<00:43, 7352.35it/s] 20%|█▉        | 79115/400000 [00:11<00:43, 7363.06it/s] 20%|█▉        | 79852/400000 [00:11<00:43, 7348.72it/s] 20%|██        | 80593/400000 [00:11<00:43, 7364.83it/s] 20%|██        | 81349/400000 [00:11<00:42, 7421.19it/s] 21%|██        | 82114/400000 [00:11<00:42, 7488.08it/s] 21%|██        | 82864/400000 [00:11<00:42, 7402.73it/s] 21%|██        | 83605/400000 [00:11<00:44, 7184.79it/s] 21%|██        | 84335/400000 [00:12<00:43, 7216.83it/s] 21%|██▏       | 85058/400000 [00:12<00:44, 7101.88it/s] 21%|██▏       | 85771/400000 [00:12<00:44, 7093.08it/s] 22%|██▏       | 86487/400000 [00:12<00:44, 7111.75it/s] 22%|██▏       | 87242/400000 [00:12<00:43, 7236.09it/s] 22%|██▏       | 87979/400000 [00:12<00:42, 7273.48it/s] 22%|██▏       | 88708/400000 [00:12<00:43, 7170.82it/s] 22%|██▏       | 89443/400000 [00:12<00:42, 7222.79it/s] 23%|██▎       | 90179/400000 [00:12<00:42, 7260.52it/s] 23%|██▎       | 90906/400000 [00:12<00:43, 7092.99it/s] 23%|██▎       | 91643/400000 [00:13<00:42, 7173.08it/s] 23%|██▎       | 92362/400000 [00:13<00:43, 7101.80it/s] 23%|██▎       | 93074/400000 [00:13<00:43, 7099.62it/s] 23%|██▎       | 93785/400000 [00:13<00:44, 6883.65it/s] 24%|██▎       | 94476/400000 [00:13<00:44, 6811.55it/s] 24%|██▍       | 95183/400000 [00:13<00:44, 6884.41it/s] 24%|██▍       | 95873/400000 [00:13<00:45, 6671.71it/s] 24%|██▍       | 96591/400000 [00:13<00:44, 6814.90it/s] 24%|██▍       | 97286/400000 [00:13<00:44, 6853.95it/s] 25%|██▍       | 98026/400000 [00:13<00:43, 7008.78it/s] 25%|██▍       | 98739/400000 [00:14<00:42, 7043.07it/s] 25%|██▍       | 99474/400000 [00:14<00:42, 7130.39it/s] 25%|██▌       | 100217/400000 [00:14<00:41, 7217.07it/s] 25%|██▌       | 100940/400000 [00:14<00:41, 7207.93it/s] 25%|██▌       | 101680/400000 [00:14<00:41, 7263.83it/s] 26%|██▌       | 102458/400000 [00:14<00:40, 7409.66it/s] 26%|██▌       | 103201/400000 [00:14<00:40, 7346.29it/s] 26%|██▌       | 103937/400000 [00:14<00:41, 7136.98it/s] 26%|██▌       | 104653/400000 [00:14<00:42, 6979.47it/s] 26%|██▋       | 105383/400000 [00:14<00:41, 7072.00it/s] 27%|██▋       | 106093/400000 [00:15<00:42, 6904.72it/s] 27%|██▋       | 106844/400000 [00:15<00:41, 7074.10it/s] 27%|██▋       | 107554/400000 [00:15<00:41, 7026.25it/s] 27%|██▋       | 108282/400000 [00:15<00:41, 7099.24it/s] 27%|██▋       | 108994/400000 [00:15<00:41, 7060.21it/s] 27%|██▋       | 109702/400000 [00:15<00:41, 7063.24it/s] 28%|██▊       | 110410/400000 [00:15<00:41, 7047.52it/s] 28%|██▊       | 111116/400000 [00:15<00:41, 7007.63it/s] 28%|██▊       | 111818/400000 [00:15<00:41, 6911.81it/s] 28%|██▊       | 112519/400000 [00:16<00:41, 6938.55it/s] 28%|██▊       | 113225/400000 [00:16<00:41, 6972.48it/s] 28%|██▊       | 113934/400000 [00:16<00:40, 7004.94it/s] 29%|██▊       | 114635/400000 [00:16<00:41, 6874.33it/s] 29%|██▉       | 115379/400000 [00:16<00:40, 7033.34it/s] 29%|██▉       | 116116/400000 [00:16<00:39, 7130.24it/s] 29%|██▉       | 116856/400000 [00:16<00:39, 7204.62it/s] 29%|██▉       | 117578/400000 [00:16<00:39, 7069.23it/s] 30%|██▉       | 118287/400000 [00:16<00:40, 6945.03it/s] 30%|██▉       | 118983/400000 [00:16<00:40, 6929.19it/s] 30%|██▉       | 119678/400000 [00:17<00:40, 6934.02it/s] 30%|███       | 120373/400000 [00:17<00:40, 6908.60it/s] 30%|███       | 121089/400000 [00:17<00:39, 6980.90it/s] 30%|███       | 121821/400000 [00:17<00:39, 7077.45it/s] 31%|███       | 122571/400000 [00:17<00:38, 7197.85it/s] 31%|███       | 123317/400000 [00:17<00:38, 7273.65it/s] 31%|███       | 124056/400000 [00:17<00:37, 7305.53it/s] 31%|███       | 124815/400000 [00:17<00:37, 7387.70it/s] 31%|███▏      | 125555/400000 [00:17<00:37, 7380.62it/s] 32%|███▏      | 126317/400000 [00:17<00:36, 7449.57it/s] 32%|███▏      | 127063/400000 [00:18<00:36, 7427.66it/s] 32%|███▏      | 127807/400000 [00:18<00:36, 7409.91it/s] 32%|███▏      | 128549/400000 [00:18<00:36, 7387.53it/s] 32%|███▏      | 129288/400000 [00:18<00:37, 7279.23it/s] 33%|███▎      | 130049/400000 [00:18<00:36, 7373.53it/s] 33%|███▎      | 130798/400000 [00:18<00:36, 7407.82it/s] 33%|███▎      | 131540/400000 [00:18<00:36, 7353.70it/s] 33%|███▎      | 132288/400000 [00:18<00:36, 7390.42it/s] 33%|███▎      | 133028/400000 [00:18<00:36, 7262.17it/s] 33%|███▎      | 133759/400000 [00:18<00:36, 7275.60it/s] 34%|███▎      | 134488/400000 [00:19<00:37, 7170.26it/s] 34%|███▍      | 135210/400000 [00:19<00:36, 7184.90it/s] 34%|███▍      | 135944/400000 [00:19<00:36, 7229.97it/s] 34%|███▍      | 136668/400000 [00:19<00:36, 7191.61it/s] 34%|███▍      | 137398/400000 [00:19<00:36, 7223.77it/s] 35%|███▍      | 138148/400000 [00:19<00:35, 7302.72it/s] 35%|███▍      | 138879/400000 [00:19<00:35, 7302.80it/s] 35%|███▍      | 139610/400000 [00:19<00:35, 7266.71it/s] 35%|███▌      | 140337/400000 [00:19<00:36, 7199.72it/s] 35%|███▌      | 141064/400000 [00:19<00:35, 7217.98it/s] 35%|███▌      | 141789/400000 [00:20<00:35, 7226.64it/s] 36%|███▌      | 142528/400000 [00:20<00:35, 7273.11it/s] 36%|███▌      | 143256/400000 [00:20<00:35, 7270.25it/s] 36%|███▌      | 143984/400000 [00:20<00:35, 7267.85it/s] 36%|███▌      | 144711/400000 [00:20<00:36, 7025.41it/s] 36%|███▋      | 145438/400000 [00:20<00:35, 7094.98it/s] 37%|███▋      | 146157/400000 [00:20<00:35, 7123.17it/s] 37%|███▋      | 146885/400000 [00:20<00:35, 7167.51it/s] 37%|███▋      | 147605/400000 [00:20<00:35, 7176.73it/s] 37%|███▋      | 148326/400000 [00:20<00:35, 7185.73it/s] 37%|███▋      | 149045/400000 [00:21<00:35, 7131.10it/s] 37%|███▋      | 149779/400000 [00:21<00:34, 7190.37it/s] 38%|███▊      | 150518/400000 [00:21<00:34, 7247.29it/s] 38%|███▊      | 151261/400000 [00:21<00:34, 7300.64it/s] 38%|███▊      | 151993/400000 [00:21<00:33, 7304.97it/s] 38%|███▊      | 152726/400000 [00:21<00:33, 7311.29it/s] 38%|███▊      | 153458/400000 [00:21<00:33, 7264.09it/s] 39%|███▊      | 154193/400000 [00:21<00:33, 7288.15it/s] 39%|███▊      | 154922/400000 [00:21<00:33, 7239.40it/s] 39%|███▉      | 155661/400000 [00:21<00:33, 7283.31it/s] 39%|███▉      | 156390/400000 [00:22<00:33, 7244.71it/s] 39%|███▉      | 157115/400000 [00:22<00:33, 7161.80it/s] 39%|███▉      | 157832/400000 [00:22<00:34, 7078.49it/s] 40%|███▉      | 158541/400000 [00:22<00:34, 7076.44it/s] 40%|███▉      | 159249/400000 [00:22<00:34, 6951.42it/s] 40%|███▉      | 159971/400000 [00:22<00:34, 7029.52it/s] 40%|████      | 160675/400000 [00:22<00:34, 6990.97it/s] 40%|████      | 161375/400000 [00:22<00:34, 6930.54it/s] 41%|████      | 162112/400000 [00:22<00:33, 7055.57it/s] 41%|████      | 162819/400000 [00:23<00:33, 7022.16it/s] 41%|████      | 163522/400000 [00:23<00:33, 6980.97it/s] 41%|████      | 164221/400000 [00:23<00:33, 6951.75it/s] 41%|████      | 164919/400000 [00:23<00:33, 6959.85it/s] 41%|████▏     | 165616/400000 [00:23<00:33, 6908.62it/s] 42%|████▏     | 166318/400000 [00:23<00:33, 6940.17it/s] 42%|████▏     | 167042/400000 [00:23<00:33, 7027.03it/s] 42%|████▏     | 167800/400000 [00:23<00:32, 7183.71it/s] 42%|████▏     | 168522/400000 [00:23<00:32, 7192.47it/s] 42%|████▏     | 169243/400000 [00:23<00:32, 7187.65it/s] 42%|████▏     | 169977/400000 [00:24<00:31, 7230.50it/s] 43%|████▎     | 170701/400000 [00:24<00:32, 7131.30it/s] 43%|████▎     | 171415/400000 [00:24<00:32, 7003.95it/s] 43%|████▎     | 172137/400000 [00:24<00:32, 7065.21it/s] 43%|████▎     | 172877/400000 [00:24<00:31, 7161.04it/s] 43%|████▎     | 173594/400000 [00:24<00:31, 7156.11it/s] 44%|████▎     | 174311/400000 [00:24<00:31, 7121.89it/s] 44%|████▍     | 175024/400000 [00:24<00:31, 7113.62it/s] 44%|████▍     | 175736/400000 [00:24<00:31, 7033.55it/s] 44%|████▍     | 176448/400000 [00:24<00:31, 7058.22it/s] 44%|████▍     | 177171/400000 [00:25<00:31, 7107.92it/s] 44%|████▍     | 177902/400000 [00:25<00:30, 7166.02it/s] 45%|████▍     | 178619/400000 [00:25<00:31, 7133.73it/s] 45%|████▍     | 179339/400000 [00:25<00:30, 7151.76it/s] 45%|████▌     | 180072/400000 [00:25<00:30, 7202.40it/s] 45%|████▌     | 180793/400000 [00:25<00:30, 7198.30it/s] 45%|████▌     | 181513/400000 [00:25<00:30, 7155.76it/s] 46%|████▌     | 182229/400000 [00:25<00:30, 7144.69it/s] 46%|████▌     | 182944/400000 [00:25<00:30, 7128.31it/s] 46%|████▌     | 183683/400000 [00:25<00:30, 7204.49it/s] 46%|████▌     | 184404/400000 [00:26<00:30, 7068.54it/s] 46%|████▋     | 185122/400000 [00:26<00:30, 7100.20it/s] 46%|████▋     | 185845/400000 [00:26<00:30, 7136.14it/s] 47%|████▋     | 186590/400000 [00:26<00:29, 7225.05it/s] 47%|████▋     | 187314/400000 [00:26<00:29, 7187.29it/s] 47%|████▋     | 188034/400000 [00:26<00:29, 7173.41it/s] 47%|████▋     | 188753/400000 [00:26<00:29, 7113.59it/s] 47%|████▋     | 189475/400000 [00:26<00:29, 7142.91it/s] 48%|████▊     | 190225/400000 [00:26<00:28, 7246.39it/s] 48%|████▊     | 190957/400000 [00:26<00:28, 7267.75it/s] 48%|████▊     | 191685/400000 [00:27<00:29, 7132.13it/s] 48%|████▊     | 192400/400000 [00:27<00:29, 7080.28it/s] 48%|████▊     | 193130/400000 [00:27<00:28, 7144.14it/s] 48%|████▊     | 193846/400000 [00:27<00:29, 7010.87it/s] 49%|████▊     | 194549/400000 [00:27<00:29, 6882.43it/s] 49%|████▉     | 195294/400000 [00:27<00:29, 7041.97it/s] 49%|████▉     | 196022/400000 [00:27<00:28, 7110.42it/s] 49%|████▉     | 196780/400000 [00:27<00:28, 7241.27it/s] 49%|████▉     | 197550/400000 [00:27<00:27, 7371.54it/s] 50%|████▉     | 198306/400000 [00:27<00:27, 7424.46it/s] 50%|████▉     | 199050/400000 [00:28<00:27, 7364.36it/s] 50%|████▉     | 199788/400000 [00:28<00:27, 7365.73it/s] 50%|█████     | 200526/400000 [00:28<00:27, 7138.31it/s] 50%|█████     | 201254/400000 [00:28<00:27, 7177.93it/s] 50%|█████     | 201995/400000 [00:28<00:27, 7244.65it/s] 51%|█████     | 202721/400000 [00:28<00:27, 7237.30it/s] 51%|█████     | 203446/400000 [00:28<00:27, 7231.96it/s] 51%|█████     | 204170/400000 [00:28<00:27, 7151.60it/s] 51%|█████     | 204896/400000 [00:28<00:27, 7182.58it/s] 51%|█████▏    | 205635/400000 [00:29<00:26, 7241.13it/s] 52%|█████▏    | 206360/400000 [00:29<00:26, 7224.51it/s] 52%|█████▏    | 207094/400000 [00:29<00:26, 7258.74it/s] 52%|█████▏    | 207821/400000 [00:29<00:26, 7210.62it/s] 52%|█████▏    | 208557/400000 [00:29<00:26, 7254.33it/s] 52%|█████▏    | 209303/400000 [00:29<00:26, 7312.67it/s] 53%|█████▎    | 210035/400000 [00:29<00:26, 7262.10it/s] 53%|█████▎    | 210794/400000 [00:29<00:25, 7354.32it/s] 53%|█████▎    | 211553/400000 [00:29<00:25, 7422.89it/s] 53%|█████▎    | 212301/400000 [00:29<00:25, 7437.05it/s] 53%|█████▎    | 213046/400000 [00:30<00:25, 7340.98it/s] 53%|█████▎    | 213781/400000 [00:30<00:25, 7192.56it/s] 54%|█████▎    | 214539/400000 [00:30<00:25, 7304.14it/s] 54%|█████▍    | 215271/400000 [00:30<00:25, 7230.97it/s] 54%|█████▍    | 215996/400000 [00:30<00:25, 7148.72it/s] 54%|█████▍    | 216713/400000 [00:30<00:25, 7153.14it/s] 54%|█████▍    | 217429/400000 [00:30<00:26, 6952.65it/s] 55%|█████▍    | 218159/400000 [00:30<00:25, 7051.79it/s] 55%|█████▍    | 218887/400000 [00:30<00:25, 7117.56it/s] 55%|█████▍    | 219623/400000 [00:30<00:25, 7188.26it/s] 55%|█████▌    | 220343/400000 [00:31<00:25, 7088.58it/s] 55%|█████▌    | 221072/400000 [00:31<00:25, 7145.85it/s] 55%|█████▌    | 221814/400000 [00:31<00:24, 7224.92it/s] 56%|█████▌    | 222538/400000 [00:31<00:24, 7223.01it/s] 56%|█████▌    | 223262/400000 [00:31<00:24, 7226.37it/s] 56%|█████▌    | 224012/400000 [00:31<00:24, 7304.27it/s] 56%|█████▌    | 224746/400000 [00:31<00:23, 7312.50it/s] 56%|█████▋    | 225478/400000 [00:31<00:23, 7303.05it/s] 57%|█████▋    | 226209/400000 [00:31<00:24, 7175.93it/s] 57%|█████▋    | 226947/400000 [00:31<00:23, 7233.74it/s] 57%|█████▋    | 227671/400000 [00:32<00:23, 7192.80it/s] 57%|█████▋    | 228391/400000 [00:32<00:24, 7060.72it/s] 57%|█████▋    | 229098/400000 [00:32<00:24, 7019.74it/s] 57%|█████▋    | 229801/400000 [00:32<00:24, 6840.45it/s] 58%|█████▊    | 230546/400000 [00:32<00:24, 7011.73it/s] 58%|█████▊    | 231250/400000 [00:32<00:24, 6985.76it/s] 58%|█████▊    | 231951/400000 [00:32<00:24, 6954.90it/s] 58%|█████▊    | 232689/400000 [00:32<00:23, 7074.05it/s] 58%|█████▊    | 233451/400000 [00:32<00:23, 7228.09it/s] 59%|█████▊    | 234176/400000 [00:32<00:22, 7225.48it/s] 59%|█████▊    | 234921/400000 [00:33<00:22, 7290.40it/s] 59%|█████▉    | 235652/400000 [00:33<00:22, 7239.72it/s] 59%|█████▉    | 236401/400000 [00:33<00:22, 7311.21it/s] 59%|█████▉    | 237140/400000 [00:33<00:22, 7330.89it/s] 59%|█████▉    | 237898/400000 [00:33<00:21, 7402.60it/s] 60%|█████▉    | 238639/400000 [00:33<00:22, 7318.50it/s] 60%|█████▉    | 239375/400000 [00:33<00:21, 7328.72it/s] 60%|██████    | 240109/400000 [00:33<00:21, 7304.34it/s] 60%|██████    | 240840/400000 [00:33<00:22, 7224.20it/s] 60%|██████    | 241581/400000 [00:33<00:21, 7276.36it/s] 61%|██████    | 242310/400000 [00:34<00:21, 7202.43it/s] 61%|██████    | 243031/400000 [00:34<00:22, 6970.58it/s] 61%|██████    | 243736/400000 [00:34<00:22, 6992.21it/s] 61%|██████    | 244440/400000 [00:34<00:22, 7003.72it/s] 61%|██████▏   | 245142/400000 [00:34<00:22, 6932.60it/s] 61%|██████▏   | 245855/400000 [00:34<00:22, 6988.67it/s] 62%|██████▏   | 246586/400000 [00:34<00:21, 7080.99it/s] 62%|██████▏   | 247295/400000 [00:34<00:21, 6983.73it/s] 62%|██████▏   | 247995/400000 [00:34<00:21, 6962.39it/s] 62%|██████▏   | 248692/400000 [00:35<00:21, 6928.96it/s] 62%|██████▏   | 249387/400000 [00:35<00:21, 6933.98it/s] 63%|██████▎   | 250081/400000 [00:35<00:21, 6881.60it/s] 63%|██████▎   | 250813/400000 [00:35<00:21, 7006.15it/s] 63%|██████▎   | 251528/400000 [00:35<00:21, 7045.16it/s] 63%|██████▎   | 252234/400000 [00:35<00:21, 7024.50it/s] 63%|██████▎   | 252943/400000 [00:35<00:20, 7013.51it/s] 63%|██████▎   | 253645/400000 [00:35<00:21, 6900.01it/s] 64%|██████▎   | 254348/400000 [00:35<00:20, 6937.34it/s] 64%|██████▍   | 255091/400000 [00:35<00:20, 7078.08it/s] 64%|██████▍   | 255800/400000 [00:36<00:20, 6970.18it/s] 64%|██████▍   | 256499/400000 [00:36<00:20, 6972.44it/s] 64%|██████▍   | 257198/400000 [00:36<00:20, 6943.17it/s] 64%|██████▍   | 257893/400000 [00:36<00:20, 6921.65it/s] 65%|██████▍   | 258608/400000 [00:36<00:20, 6987.92it/s] 65%|██████▍   | 259308/400000 [00:36<00:20, 6982.21it/s] 65%|██████▌   | 260007/400000 [00:36<00:20, 6926.18it/s] 65%|██████▌   | 260739/400000 [00:36<00:19, 7039.03it/s] 65%|██████▌   | 261444/400000 [00:36<00:19, 7025.10it/s] 66%|██████▌   | 262147/400000 [00:36<00:19, 6974.74it/s] 66%|██████▌   | 262845/400000 [00:37<00:20, 6790.60it/s] 66%|██████▌   | 263526/400000 [00:37<00:20, 6680.37it/s] 66%|██████▌   | 264196/400000 [00:37<00:21, 6463.44it/s] 66%|██████▌   | 264845/400000 [00:37<00:20, 6454.08it/s] 66%|██████▋   | 265493/400000 [00:37<00:21, 6343.74it/s] 67%|██████▋   | 266164/400000 [00:37<00:20, 6448.29it/s] 67%|██████▋   | 266868/400000 [00:37<00:20, 6614.59it/s] 67%|██████▋   | 267541/400000 [00:37<00:19, 6648.55it/s] 67%|██████▋   | 268208/400000 [00:37<00:20, 6385.04it/s] 67%|██████▋   | 268882/400000 [00:37<00:20, 6485.16it/s] 67%|██████▋   | 269534/400000 [00:38<00:20, 6483.84it/s] 68%|██████▊   | 270257/400000 [00:38<00:19, 6690.60it/s] 68%|██████▊   | 270985/400000 [00:38<00:18, 6856.04it/s] 68%|██████▊   | 271693/400000 [00:38<00:18, 6919.19it/s] 68%|██████▊   | 272388/400000 [00:38<00:18, 6875.38it/s] 68%|██████▊   | 273078/400000 [00:38<00:18, 6755.65it/s] 68%|██████▊   | 273756/400000 [00:38<00:18, 6727.96it/s] 69%|██████▊   | 274453/400000 [00:38<00:18, 6798.48it/s] 69%|██████▉   | 275177/400000 [00:38<00:18, 6922.42it/s] 69%|██████▉   | 275896/400000 [00:38<00:17, 6998.52it/s] 69%|██████▉   | 276618/400000 [00:39<00:17, 7061.43it/s] 69%|██████▉   | 277331/400000 [00:39<00:17, 7080.18it/s] 70%|██████▉   | 278055/400000 [00:39<00:17, 7126.31it/s] 70%|██████▉   | 278769/400000 [00:39<00:17, 7125.49it/s] 70%|██████▉   | 279509/400000 [00:39<00:16, 7204.82it/s] 70%|███████   | 280230/400000 [00:39<00:17, 6977.50it/s] 70%|███████   | 280955/400000 [00:39<00:16, 7054.76it/s] 70%|███████   | 281666/400000 [00:39<00:16, 7069.94it/s] 71%|███████   | 282415/400000 [00:39<00:16, 7190.17it/s] 71%|███████   | 283179/400000 [00:40<00:15, 7318.67it/s] 71%|███████   | 283920/400000 [00:40<00:15, 7343.96it/s] 71%|███████   | 284676/400000 [00:40<00:15, 7406.18it/s] 71%|███████▏  | 285418/400000 [00:40<00:15, 7349.75it/s] 72%|███████▏  | 286159/400000 [00:40<00:15, 7364.73it/s] 72%|███████▏  | 286913/400000 [00:40<00:15, 7416.24it/s] 72%|███████▏  | 287656/400000 [00:40<00:15, 7263.48it/s] 72%|███████▏  | 288384/400000 [00:40<00:15, 7256.69it/s] 72%|███████▏  | 289122/400000 [00:40<00:15, 7290.26it/s] 72%|███████▏  | 289852/400000 [00:40<00:15, 7182.47it/s] 73%|███████▎  | 290571/400000 [00:41<00:15, 7159.73it/s] 73%|███████▎  | 291288/400000 [00:41<00:15, 7085.03it/s] 73%|███████▎  | 292024/400000 [00:41<00:15, 7164.26it/s] 73%|███████▎  | 292761/400000 [00:41<00:14, 7222.41it/s] 73%|███████▎  | 293488/400000 [00:41<00:14, 7234.67it/s] 74%|███████▎  | 294212/400000 [00:41<00:14, 7223.26it/s] 74%|███████▎  | 294936/400000 [00:41<00:14, 7226.95it/s] 74%|███████▍  | 295676/400000 [00:41<00:14, 7277.40it/s] 74%|███████▍  | 296404/400000 [00:41<00:14, 7268.81it/s] 74%|███████▍  | 297134/400000 [00:41<00:14, 7276.22it/s] 74%|███████▍  | 297863/400000 [00:42<00:14, 7277.77it/s] 75%|███████▍  | 298591/400000 [00:42<00:14, 7206.37it/s] 75%|███████▍  | 299319/400000 [00:42<00:13, 7226.51it/s] 75%|███████▌  | 300042/400000 [00:42<00:13, 7153.11it/s] 75%|███████▌  | 300770/400000 [00:42<00:13, 7190.40it/s] 75%|███████▌  | 301510/400000 [00:42<00:13, 7251.37it/s] 76%|███████▌  | 302236/400000 [00:42<00:13, 7218.73it/s] 76%|███████▌  | 302983/400000 [00:42<00:13, 7289.18it/s] 76%|███████▌  | 303714/400000 [00:42<00:13, 7294.06it/s] 76%|███████▌  | 304460/400000 [00:42<00:13, 7341.78it/s] 76%|███████▋  | 305195/400000 [00:43<00:13, 7254.76it/s] 76%|███████▋  | 305923/400000 [00:43<00:12, 7258.95it/s] 77%|███████▋  | 306664/400000 [00:43<00:12, 7301.44it/s] 77%|███████▋  | 307411/400000 [00:43<00:12, 7349.09it/s] 77%|███████▋  | 308147/400000 [00:43<00:12, 7351.06it/s] 77%|███████▋  | 308884/400000 [00:43<00:12, 7356.35it/s] 77%|███████▋  | 309628/400000 [00:43<00:12, 7378.58it/s] 78%|███████▊  | 310385/400000 [00:43<00:12, 7433.23it/s] 78%|███████▊  | 311129/400000 [00:43<00:12, 7318.25it/s] 78%|███████▊  | 311877/400000 [00:43<00:11, 7365.25it/s] 78%|███████▊  | 312614/400000 [00:44<00:12, 7157.49it/s] 78%|███████▊  | 313338/400000 [00:44<00:12, 7181.39it/s] 79%|███████▊  | 314066/400000 [00:44<00:11, 7208.59it/s] 79%|███████▊  | 314788/400000 [00:44<00:12, 7078.40it/s] 79%|███████▉  | 315497/400000 [00:44<00:11, 7042.54it/s] 79%|███████▉  | 316203/400000 [00:44<00:11, 7031.56it/s] 79%|███████▉  | 316907/400000 [00:44<00:12, 6893.54it/s] 79%|███████▉  | 317642/400000 [00:44<00:11, 7022.88it/s] 80%|███████▉  | 318364/400000 [00:44<00:11, 7077.57it/s] 80%|███████▉  | 319073/400000 [00:44<00:11, 7047.79it/s] 80%|███████▉  | 319779/400000 [00:45<00:11, 7012.51it/s] 80%|████████  | 320532/400000 [00:45<00:11, 7158.33it/s] 80%|████████  | 321298/400000 [00:45<00:10, 7300.26it/s] 81%|████████  | 322030/400000 [00:45<00:10, 7281.37it/s] 81%|████████  | 322773/400000 [00:45<00:10, 7323.07it/s] 81%|████████  | 323507/400000 [00:45<00:10, 7084.18it/s] 81%|████████  | 324218/400000 [00:45<00:10, 6917.41it/s] 81%|████████  | 324951/400000 [00:45<00:10, 7036.08it/s] 81%|████████▏ | 325681/400000 [00:45<00:10, 7112.18it/s] 82%|████████▏ | 326424/400000 [00:46<00:10, 7201.01it/s] 82%|████████▏ | 327146/400000 [00:46<00:10, 6823.04it/s] 82%|████████▏ | 327834/400000 [00:46<00:10, 6794.54it/s] 82%|████████▏ | 328518/400000 [00:46<00:10, 6779.16it/s] 82%|████████▏ | 329239/400000 [00:46<00:10, 6902.15it/s] 82%|████████▏ | 329962/400000 [00:46<00:10, 6997.32it/s] 83%|████████▎ | 330687/400000 [00:46<00:09, 7069.61it/s] 83%|████████▎ | 331407/400000 [00:46<00:09, 7108.10it/s] 83%|████████▎ | 332119/400000 [00:46<00:09, 7074.59it/s] 83%|████████▎ | 332833/400000 [00:46<00:09, 7093.68it/s] 83%|████████▎ | 333549/400000 [00:47<00:09, 7110.34it/s] 84%|████████▎ | 334261/400000 [00:47<00:09, 6679.66it/s] 84%|████████▎ | 334955/400000 [00:47<00:09, 6755.67it/s] 84%|████████▍ | 335635/400000 [00:47<00:09, 6532.25it/s] 84%|████████▍ | 336337/400000 [00:47<00:09, 6668.75it/s] 84%|████████▍ | 337047/400000 [00:47<00:09, 6791.35it/s] 84%|████████▍ | 337745/400000 [00:47<00:09, 6845.27it/s] 85%|████████▍ | 338483/400000 [00:47<00:08, 6996.83it/s] 85%|████████▍ | 339228/400000 [00:47<00:08, 7125.36it/s] 85%|████████▍ | 339947/400000 [00:47<00:08, 7142.47it/s] 85%|████████▌ | 340663/400000 [00:48<00:08, 6980.70it/s] 85%|████████▌ | 341364/400000 [00:48<00:08, 6719.82it/s] 86%|████████▌ | 342040/400000 [00:48<00:08, 6676.30it/s] 86%|████████▌ | 342725/400000 [00:48<00:08, 6724.94it/s] 86%|████████▌ | 343427/400000 [00:48<00:08, 6810.17it/s] 86%|████████▌ | 344110/400000 [00:48<00:08, 6638.34it/s] 86%|████████▌ | 344825/400000 [00:48<00:08, 6783.11it/s] 86%|████████▋ | 345506/400000 [00:48<00:08, 6721.51it/s] 87%|████████▋ | 346182/400000 [00:48<00:07, 6732.19it/s] 87%|████████▋ | 346867/400000 [00:48<00:07, 6765.45it/s] 87%|████████▋ | 347551/400000 [00:49<00:07, 6787.09it/s] 87%|████████▋ | 348231/400000 [00:49<00:07, 6729.76it/s] 87%|████████▋ | 348924/400000 [00:49<00:07, 6785.80it/s] 87%|████████▋ | 349604/400000 [00:49<00:07, 6577.94it/s] 88%|████████▊ | 350264/400000 [00:49<00:07, 6580.44it/s] 88%|████████▊ | 350935/400000 [00:49<00:07, 6616.29it/s] 88%|████████▊ | 351600/400000 [00:49<00:07, 6624.29it/s] 88%|████████▊ | 352281/400000 [00:49<00:07, 6676.14it/s] 88%|████████▊ | 352950/400000 [00:49<00:07, 6672.04it/s] 88%|████████▊ | 353618/400000 [00:50<00:07, 6378.84it/s] 89%|████████▊ | 354259/400000 [00:50<00:07, 6280.27it/s] 89%|████████▊ | 354953/400000 [00:50<00:06, 6463.07it/s] 89%|████████▉ | 355603/400000 [00:50<00:06, 6468.49it/s] 89%|████████▉ | 356342/400000 [00:50<00:06, 6717.84it/s] 89%|████████▉ | 357074/400000 [00:50<00:06, 6887.74it/s] 89%|████████▉ | 357788/400000 [00:50<00:06, 6959.69it/s] 90%|████████▉ | 358516/400000 [00:50<00:05, 7051.49it/s] 90%|████████▉ | 359252/400000 [00:50<00:05, 7140.36it/s] 90%|████████▉ | 359997/400000 [00:50<00:05, 7229.30it/s] 90%|█████████ | 360722/400000 [00:51<00:05, 7209.37it/s] 90%|█████████ | 361461/400000 [00:51<00:05, 7259.10it/s] 91%|█████████ | 362188/400000 [00:51<00:05, 7259.60it/s] 91%|█████████ | 362948/400000 [00:51<00:05, 7356.43it/s] 91%|█████████ | 363689/400000 [00:51<00:04, 7369.79it/s] 91%|█████████ | 364427/400000 [00:51<00:04, 7223.59it/s] 91%|█████████▏| 365151/400000 [00:51<00:04, 7205.01it/s] 91%|█████████▏| 365883/400000 [00:51<00:04, 7235.64it/s] 92%|█████████▏| 366608/400000 [00:51<00:04, 7168.05it/s] 92%|█████████▏| 367347/400000 [00:51<00:04, 7231.94it/s] 92%|█████████▏| 368083/400000 [00:52<00:04, 7267.16it/s] 92%|█████████▏| 368811/400000 [00:52<00:04, 7211.44it/s] 92%|█████████▏| 369541/400000 [00:52<00:04, 7237.08it/s] 93%|█████████▎| 370265/400000 [00:52<00:04, 7235.03it/s] 93%|█████████▎| 370989/400000 [00:52<00:04, 7233.52it/s] 93%|█████████▎| 371713/400000 [00:52<00:03, 7207.53it/s] 93%|█████████▎| 372434/400000 [00:52<00:03, 7149.88it/s] 93%|█████████▎| 373164/400000 [00:52<00:03, 7193.48it/s] 93%|█████████▎| 373884/400000 [00:52<00:03, 7161.69it/s] 94%|█████████▎| 374601/400000 [00:52<00:03, 7057.64it/s] 94%|█████████▍| 375308/400000 [00:53<00:03, 6974.72it/s] 94%|█████████▍| 376026/400000 [00:53<00:03, 7033.47it/s] 94%|█████████▍| 376730/400000 [00:53<00:03, 6938.92it/s] 94%|█████████▍| 377425/400000 [00:53<00:03, 6861.19it/s] 95%|█████████▍| 378112/400000 [00:53<00:03, 6787.75it/s] 95%|█████████▍| 378805/400000 [00:53<00:03, 6828.57it/s] 95%|█████████▍| 379499/400000 [00:53<00:02, 6861.50it/s] 95%|█████████▌| 380229/400000 [00:53<00:02, 6986.47it/s] 95%|█████████▌| 380933/400000 [00:53<00:02, 7001.71it/s] 95%|█████████▌| 381673/400000 [00:53<00:02, 7116.30it/s] 96%|█████████▌| 382386/400000 [00:54<00:02, 7020.02it/s] 96%|█████████▌| 383089/400000 [00:54<00:02, 6895.95it/s] 96%|█████████▌| 383781/400000 [00:54<00:02, 6902.01it/s] 96%|█████████▌| 384536/400000 [00:54<00:02, 7082.84it/s] 96%|█████████▋| 385301/400000 [00:54<00:02, 7242.70it/s] 97%|█████████▋| 386038/400000 [00:54<00:01, 7278.38it/s] 97%|█████████▋| 386795/400000 [00:54<00:01, 7363.45it/s] 97%|█████████▋| 387533/400000 [00:54<00:01, 7365.61it/s] 97%|█████████▋| 388271/400000 [00:54<00:01, 7359.75it/s] 97%|█████████▋| 389008/400000 [00:54<00:01, 7309.15it/s] 97%|█████████▋| 389740/400000 [00:55<00:01, 7226.21it/s] 98%|█████████▊| 390475/400000 [00:55<00:01, 7262.88it/s] 98%|█████████▊| 391222/400000 [00:55<00:01, 7321.00it/s] 98%|█████████▊| 391955/400000 [00:55<00:01, 7237.76it/s] 98%|█████████▊| 392680/400000 [00:55<00:01, 7127.85it/s] 98%|█████████▊| 393398/400000 [00:55<00:00, 7141.71it/s] 99%|█████████▊| 394113/400000 [00:55<00:00, 7103.44it/s] 99%|█████████▊| 394875/400000 [00:55<00:00, 7249.70it/s] 99%|█████████▉| 395606/400000 [00:55<00:00, 7262.12it/s] 99%|█████████▉| 396333/400000 [00:56<00:00, 7231.47it/s] 99%|█████████▉| 397070/400000 [00:56<00:00, 7269.88it/s] 99%|█████████▉| 397798/400000 [00:56<00:00, 7219.98it/s]100%|█████████▉| 398551/400000 [00:56<00:00, 7309.21it/s]100%|█████████▉| 399322/400000 [00:56<00:00, 7424.53it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7079.63it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f7be7103b70>, <torchtext.data.dataset.TabularDataset object at 0x7f7be7103cc0>, <torchtext.vocab.Vocab object at 0x7f7be7103be0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.55 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.55 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.55 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.45 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.42 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.76 file/s]2020-07-04 00:20:52.358010: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-04 00:20:52.362791: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-04 00:20:52.362970: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56225354eaf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-04 00:20:52.362989: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 150635.20it/s] 58%|█████▊    | 5791744/9912422 [00:00<00:19, 214952.85it/s]9920512it [00:00, 41031994.74it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1127769.01it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 465310.10it/s]1654784it [00:00, 11413018.06it/s]                         
0it [00:00, ?it/s]8192it [00:00, 183552.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
