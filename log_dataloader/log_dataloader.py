
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fe09c358158> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fe09c358158> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fe107eff2f0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fe107eff2f0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fe126244e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fe126244e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe0b5226730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe0b5226730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe0b5226730> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3047424/9912422 [00:00<00:00, 30430085.48it/s]9920512it [00:00, 33456764.87it/s]                             
0it [00:00, ?it/s]32768it [00:00, 714220.89it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 138919.56it/s]1654784it [00:00, 8548788.26it/s]                          
0it [00:00, ?it/s]8192it [00:00, 203916.57it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe092dd8a90>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe092de8cf8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fe0b5226378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fe0b5226378> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fe0b5226378> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:986: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:23,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:21,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:20,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:20,  1.93 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
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

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:53,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:36,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:36,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:36,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:36,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:35,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:24,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:24,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:24,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:23,  5.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:15,  7.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 38.36 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 59.57 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.93 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.93 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.58 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.47 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.96 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.57s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.46s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 77.96 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.57s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.57s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.47 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.57s/ url]
0 examples [00:00, ? examples/s]2020-07-15 18:12:41.331387: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-15 18:12:41.506313: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-15 18:12:41.506596: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cea790ac00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-15 18:12:41.506615: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.02 examples/s]110 examples [00:00,  4.31 examples/s]219 examples [00:00,  6.14 examples/s]327 examples [00:00,  8.75 examples/s]430 examples [00:00, 12.46 examples/s]533 examples [00:00, 17.71 examples/s]645 examples [00:00, 25.13 examples/s]753 examples [00:01, 35.54 examples/s]859 examples [00:01, 50.05 examples/s]963 examples [00:01, 70.05 examples/s]1067 examples [00:01, 97.25 examples/s]1174 examples [00:01, 133.68 examples/s]1286 examples [00:01, 181.65 examples/s]1395 examples [00:01, 242.14 examples/s]1504 examples [00:01, 315.76 examples/s]1612 examples [00:01, 399.90 examples/s]1722 examples [00:01, 493.76 examples/s]1830 examples [00:02, 589.28 examples/s]1938 examples [00:02, 681.58 examples/s]2049 examples [00:02, 770.05 examples/s]2160 examples [00:02, 846.41 examples/s]2269 examples [00:02, 902.32 examples/s]2379 examples [00:02, 952.35 examples/s]2488 examples [00:02, 984.94 examples/s]2597 examples [00:02, 1010.13 examples/s]2706 examples [00:02, 1030.58 examples/s]2816 examples [00:02, 1049.10 examples/s]2925 examples [00:03, 1059.55 examples/s]3035 examples [00:03, 1068.89 examples/s]3146 examples [00:03, 1080.53 examples/s]3256 examples [00:03, 1085.34 examples/s]3366 examples [00:03, 1083.09 examples/s]3477 examples [00:03, 1089.44 examples/s]3587 examples [00:03, 1085.23 examples/s]3696 examples [00:03, 1079.43 examples/s]3805 examples [00:03, 1081.83 examples/s]3916 examples [00:03, 1088.68 examples/s]4026 examples [00:04, 1092.01 examples/s]4137 examples [00:04, 1095.68 examples/s]4249 examples [00:04, 1100.42 examples/s]4360 examples [00:04, 1097.14 examples/s]4473 examples [00:04, 1106.23 examples/s]4587 examples [00:04, 1113.53 examples/s]4699 examples [00:04, 1113.61 examples/s]4811 examples [00:04, 1072.44 examples/s]4919 examples [00:04, 1069.23 examples/s]5030 examples [00:04, 1079.94 examples/s]5140 examples [00:05, 1084.79 examples/s]5250 examples [00:05, 1088.45 examples/s]5362 examples [00:05, 1094.94 examples/s]5472 examples [00:05, 1089.55 examples/s]5584 examples [00:05, 1096.70 examples/s]5694 examples [00:05, 1097.54 examples/s]5807 examples [00:05, 1104.21 examples/s]5918 examples [00:05, 1100.90 examples/s]6029 examples [00:05, 1100.38 examples/s]6140 examples [00:05, 1102.23 examples/s]6252 examples [00:06, 1105.62 examples/s]6363 examples [00:06, 1068.56 examples/s]6477 examples [00:06, 1086.61 examples/s]6588 examples [00:06, 1091.85 examples/s]6698 examples [00:06, 1084.48 examples/s]6812 examples [00:06, 1098.09 examples/s]6922 examples [00:06, 1082.36 examples/s]7031 examples [00:06, 1082.72 examples/s]7140 examples [00:06, 1075.40 examples/s]7248 examples [00:07, 1074.93 examples/s]7356 examples [00:07, 1062.18 examples/s]7465 examples [00:07, 1069.39 examples/s]7574 examples [00:07, 1074.02 examples/s]7683 examples [00:07, 1076.78 examples/s]7794 examples [00:07, 1084.70 examples/s]7907 examples [00:07, 1095.21 examples/s]8021 examples [00:07, 1105.95 examples/s]8132 examples [00:07, 1100.21 examples/s]8243 examples [00:07, 1066.18 examples/s]8350 examples [00:08, 1059.00 examples/s]8464 examples [00:08, 1081.82 examples/s]8577 examples [00:08, 1093.41 examples/s]8689 examples [00:08, 1098.80 examples/s]8800 examples [00:08, 1090.82 examples/s]8910 examples [00:08, 1088.56 examples/s]9021 examples [00:08, 1094.21 examples/s]9131 examples [00:08, 1090.31 examples/s]9241 examples [00:08, 1089.02 examples/s]9350 examples [00:08, 1076.74 examples/s]9458 examples [00:09, 1075.80 examples/s]9568 examples [00:09, 1080.59 examples/s]9678 examples [00:09, 1084.41 examples/s]9787 examples [00:09, 1077.86 examples/s]9897 examples [00:09, 1082.96 examples/s]10006 examples [00:09, 1032.35 examples/s]10116 examples [00:09, 1050.16 examples/s]10224 examples [00:09, 1058.80 examples/s]10331 examples [00:09, 1054.08 examples/s]10439 examples [00:09, 1061.19 examples/s]10550 examples [00:10, 1074.38 examples/s]10662 examples [00:10, 1085.48 examples/s]10774 examples [00:10, 1073.55 examples/s]10882 examples [00:10, 1030.70 examples/s]10986 examples [00:10, 1016.28 examples/s]11088 examples [00:10, 1014.52 examples/s]11199 examples [00:10, 1040.96 examples/s]11312 examples [00:10, 1064.50 examples/s]11426 examples [00:10, 1083.37 examples/s]11536 examples [00:11, 1087.01 examples/s]11649 examples [00:11, 1099.39 examples/s]11760 examples [00:11, 1099.97 examples/s]11873 examples [00:11, 1107.26 examples/s]11984 examples [00:11, 1103.95 examples/s]12095 examples [00:11, 1095.72 examples/s]12206 examples [00:11, 1098.06 examples/s]12316 examples [00:11, 1083.45 examples/s]12429 examples [00:11, 1094.31 examples/s]12539 examples [00:11, 1095.98 examples/s]12652 examples [00:12, 1103.53 examples/s]12765 examples [00:12, 1111.22 examples/s]12877 examples [00:12, 1095.60 examples/s]12987 examples [00:12, 1067.49 examples/s]13094 examples [00:12, 1065.49 examples/s]13207 examples [00:12, 1082.94 examples/s]13319 examples [00:12, 1093.77 examples/s]13431 examples [00:12, 1101.20 examples/s]13542 examples [00:12, 1102.98 examples/s]13653 examples [00:12, 1102.37 examples/s]13766 examples [00:13, 1107.99 examples/s]13877 examples [00:13, 1093.35 examples/s]13988 examples [00:13, 1097.91 examples/s]14100 examples [00:13, 1101.72 examples/s]14212 examples [00:13, 1106.95 examples/s]14323 examples [00:13, 1105.42 examples/s]14434 examples [00:13, 1104.21 examples/s]14546 examples [00:13, 1107.98 examples/s]14657 examples [00:13, 1107.21 examples/s]14768 examples [00:13, 1088.15 examples/s]14877 examples [00:14, 1068.34 examples/s]14984 examples [00:14, 1053.76 examples/s]15095 examples [00:14, 1069.85 examples/s]15206 examples [00:14, 1081.12 examples/s]15315 examples [00:14, 1071.01 examples/s]15424 examples [00:14, 1075.20 examples/s]15537 examples [00:14, 1089.95 examples/s]15647 examples [00:14, 1089.50 examples/s]15759 examples [00:14, 1096.25 examples/s]15873 examples [00:14, 1107.93 examples/s]15985 examples [00:15, 1108.52 examples/s]16096 examples [00:15, 1102.40 examples/s]16208 examples [00:15, 1107.19 examples/s]16322 examples [00:15, 1115.00 examples/s]16436 examples [00:15, 1121.25 examples/s]16549 examples [00:15, 1114.29 examples/s]16661 examples [00:15, 1097.19 examples/s]16771 examples [00:15, 1075.80 examples/s]16880 examples [00:15, 1079.71 examples/s]16990 examples [00:15, 1083.33 examples/s]17100 examples [00:16, 1086.30 examples/s]17212 examples [00:16, 1093.50 examples/s]17324 examples [00:16, 1098.73 examples/s]17434 examples [00:16, 1097.09 examples/s]17547 examples [00:16, 1105.49 examples/s]17659 examples [00:16, 1107.19 examples/s]17770 examples [00:16, 1104.13 examples/s]17881 examples [00:16, 1105.85 examples/s]17993 examples [00:16, 1108.44 examples/s]18104 examples [00:17, 1094.82 examples/s]18214 examples [00:17, 1093.99 examples/s]18325 examples [00:17, 1098.36 examples/s]18435 examples [00:17, 1076.27 examples/s]18543 examples [00:17, 1066.17 examples/s]18650 examples [00:17, 1059.93 examples/s]18757 examples [00:17, 1059.92 examples/s]18866 examples [00:17, 1063.13 examples/s]18979 examples [00:17, 1081.58 examples/s]19091 examples [00:17, 1092.52 examples/s]19206 examples [00:18, 1106.68 examples/s]19317 examples [00:18, 1092.07 examples/s]19432 examples [00:18, 1105.95 examples/s]19546 examples [00:18, 1115.59 examples/s]19658 examples [00:18, 1112.29 examples/s]19770 examples [00:18, 1108.51 examples/s]19881 examples [00:18, 1095.76 examples/s]19991 examples [00:18, 1093.47 examples/s]20101 examples [00:18, 1039.91 examples/s]20212 examples [00:18, 1058.72 examples/s]20323 examples [00:19, 1071.15 examples/s]20431 examples [00:19, 1054.11 examples/s]20541 examples [00:19, 1066.39 examples/s]20651 examples [00:19, 1073.55 examples/s]20760 examples [00:19, 1076.96 examples/s]20871 examples [00:19, 1085.34 examples/s]20980 examples [00:19, 1081.66 examples/s]21089 examples [00:19, 1058.68 examples/s]21200 examples [00:19, 1071.78 examples/s]21308 examples [00:19, 1072.64 examples/s]21418 examples [00:20, 1078.93 examples/s]21527 examples [00:20, 1080.59 examples/s]21638 examples [00:20, 1086.17 examples/s]21749 examples [00:20, 1092.67 examples/s]21859 examples [00:20, 1093.14 examples/s]21970 examples [00:20, 1098.04 examples/s]22080 examples [00:20, 1093.00 examples/s]22190 examples [00:20, 1074.67 examples/s]22300 examples [00:20, 1080.86 examples/s]22409 examples [00:20, 1083.57 examples/s]22518 examples [00:21, 1084.96 examples/s]22627 examples [00:21, 1071.47 examples/s]22736 examples [00:21, 1075.97 examples/s]22845 examples [00:21, 1079.33 examples/s]22953 examples [00:21, 1079.41 examples/s]23065 examples [00:21, 1088.36 examples/s]23174 examples [00:21, 1075.99 examples/s]23284 examples [00:21, 1081.37 examples/s]23397 examples [00:21, 1093.45 examples/s]23508 examples [00:21, 1096.39 examples/s]23619 examples [00:22, 1097.60 examples/s]23730 examples [00:22, 1099.25 examples/s]23840 examples [00:22, 1082.43 examples/s]23949 examples [00:22, 1079.86 examples/s]24058 examples [00:22, 1042.24 examples/s]24169 examples [00:22, 1059.45 examples/s]24276 examples [00:22, 1051.07 examples/s]24388 examples [00:22, 1068.80 examples/s]24500 examples [00:22, 1083.56 examples/s]24611 examples [00:23, 1091.36 examples/s]24723 examples [00:23, 1099.30 examples/s]24834 examples [00:23, 1102.34 examples/s]24946 examples [00:23, 1105.88 examples/s]25057 examples [00:23, 1103.03 examples/s]25168 examples [00:23, 1096.83 examples/s]25280 examples [00:23, 1101.30 examples/s]25391 examples [00:23, 1103.53 examples/s]25502 examples [00:23, 1102.46 examples/s]25616 examples [00:23, 1110.94 examples/s]25728 examples [00:24, 1113.57 examples/s]25840 examples [00:24, 1107.45 examples/s]25951 examples [00:24, 1104.87 examples/s]26062 examples [00:24, 1084.65 examples/s]26175 examples [00:24, 1095.69 examples/s]26285 examples [00:24, 1059.29 examples/s]26394 examples [00:24, 1067.62 examples/s]26502 examples [00:24, 1066.68 examples/s]26609 examples [00:24, 1064.77 examples/s]26718 examples [00:24, 1070.27 examples/s]26828 examples [00:25, 1077.39 examples/s]26936 examples [00:25, 1064.52 examples/s]27048 examples [00:25, 1079.67 examples/s]27157 examples [00:25, 1065.82 examples/s]27268 examples [00:25, 1077.35 examples/s]27376 examples [00:25, 1066.77 examples/s]27483 examples [00:25, 1067.33 examples/s]27591 examples [00:25, 1070.89 examples/s]27700 examples [00:25, 1074.00 examples/s]27808 examples [00:25, 1016.35 examples/s]27914 examples [00:26, 1028.36 examples/s]28023 examples [00:26, 1045.39 examples/s]28131 examples [00:26, 1054.32 examples/s]28237 examples [00:26, 1040.62 examples/s]28343 examples [00:26, 1045.96 examples/s]28451 examples [00:26, 1055.48 examples/s]28559 examples [00:26, 1060.35 examples/s]28667 examples [00:26, 1064.05 examples/s]28775 examples [00:26, 1066.75 examples/s]28883 examples [00:26, 1070.06 examples/s]28993 examples [00:27, 1078.16 examples/s]29101 examples [00:27, 1032.87 examples/s]29209 examples [00:27, 1044.64 examples/s]29318 examples [00:27, 1055.82 examples/s]29427 examples [00:27, 1065.39 examples/s]29534 examples [00:27, 1047.93 examples/s]29642 examples [00:27, 1055.88 examples/s]29748 examples [00:27, 1030.89 examples/s]29858 examples [00:27, 1048.17 examples/s]29964 examples [00:28, 1048.39 examples/s]30070 examples [00:28, 1005.49 examples/s]30172 examples [00:28, 1000.74 examples/s]30281 examples [00:28, 1025.11 examples/s]30392 examples [00:28, 1048.50 examples/s]30503 examples [00:28, 1063.50 examples/s]30612 examples [00:28, 1071.30 examples/s]30721 examples [00:28, 1076.49 examples/s]30832 examples [00:28, 1084.15 examples/s]30943 examples [00:28, 1091.30 examples/s]31055 examples [00:29, 1097.07 examples/s]31165 examples [00:29, 1091.26 examples/s]31275 examples [00:29, 1091.90 examples/s]31385 examples [00:29, 1087.18 examples/s]31494 examples [00:29, 1076.80 examples/s]31605 examples [00:29, 1085.49 examples/s]31714 examples [00:29, 1075.56 examples/s]31822 examples [00:29, 1071.10 examples/s]31932 examples [00:29, 1078.92 examples/s]32044 examples [00:29, 1088.82 examples/s]32155 examples [00:30, 1093.63 examples/s]32266 examples [00:30, 1097.75 examples/s]32376 examples [00:30, 1092.63 examples/s]32486 examples [00:30, 1088.52 examples/s]32597 examples [00:30, 1093.18 examples/s]32707 examples [00:30, 1094.50 examples/s]32817 examples [00:30, 1094.67 examples/s]32927 examples [00:30, 1084.05 examples/s]33036 examples [00:30, 1064.45 examples/s]33143 examples [00:30, 1063.94 examples/s]33250 examples [00:31, 1035.77 examples/s]33354 examples [00:31, 1007.91 examples/s]33456 examples [00:31, 1009.41 examples/s]33558 examples [00:31, 1007.52 examples/s]33669 examples [00:31, 1035.78 examples/s]33776 examples [00:31, 1045.67 examples/s]33885 examples [00:31, 1056.09 examples/s]33993 examples [00:31, 1058.83 examples/s]34103 examples [00:31, 1068.86 examples/s]34211 examples [00:32, 1069.77 examples/s]34319 examples [00:32, 1064.57 examples/s]34429 examples [00:32, 1072.55 examples/s]34537 examples [00:32, 1073.09 examples/s]34646 examples [00:32, 1076.47 examples/s]34754 examples [00:32, 1058.02 examples/s]34860 examples [00:32, 1026.83 examples/s]34970 examples [00:32, 1046.80 examples/s]35078 examples [00:32, 1054.98 examples/s]35184 examples [00:32, 1030.28 examples/s]35288 examples [00:33, 1027.57 examples/s]35398 examples [00:33, 1048.08 examples/s]35510 examples [00:33, 1068.54 examples/s]35618 examples [00:33, 1069.63 examples/s]35727 examples [00:33, 1075.43 examples/s]35835 examples [00:33, 1076.63 examples/s]35943 examples [00:33, 1077.54 examples/s]36051 examples [00:33, 1043.23 examples/s]36156 examples [00:33, 1028.49 examples/s]36267 examples [00:33, 1050.64 examples/s]36376 examples [00:34, 1059.28 examples/s]36484 examples [00:34, 1064.28 examples/s]36595 examples [00:34, 1076.39 examples/s]36703 examples [00:34, 1070.34 examples/s]36811 examples [00:34, 1058.31 examples/s]36917 examples [00:34, 1038.28 examples/s]37021 examples [00:34, 1024.92 examples/s]37125 examples [00:34, 1027.09 examples/s]37228 examples [00:34, 1006.99 examples/s]37336 examples [00:34, 1026.27 examples/s]37445 examples [00:35, 1043.62 examples/s]37556 examples [00:35, 1060.29 examples/s]37667 examples [00:35, 1072.88 examples/s]37776 examples [00:35, 1076.78 examples/s]37886 examples [00:35, 1080.72 examples/s]37995 examples [00:35, 1079.29 examples/s]38103 examples [00:35, 1074.08 examples/s]38211 examples [00:35, 1060.48 examples/s]38319 examples [00:35, 1065.70 examples/s]38428 examples [00:36, 1072.07 examples/s]38539 examples [00:36, 1080.39 examples/s]38650 examples [00:36, 1087.83 examples/s]38761 examples [00:36, 1091.58 examples/s]38871 examples [00:36, 1083.01 examples/s]38980 examples [00:36, 1081.90 examples/s]39091 examples [00:36, 1087.94 examples/s]39202 examples [00:36, 1093.77 examples/s]39312 examples [00:36, 1075.81 examples/s]39423 examples [00:36, 1083.18 examples/s]39532 examples [00:37, 1075.03 examples/s]39646 examples [00:37, 1091.96 examples/s]39757 examples [00:37, 1095.56 examples/s]39867 examples [00:37, 1077.80 examples/s]39975 examples [00:37, 1064.44 examples/s]40082 examples [00:37, 1033.95 examples/s]40186 examples [00:37, 1033.09 examples/s]40297 examples [00:37, 1052.91 examples/s]40410 examples [00:37, 1072.52 examples/s]40518 examples [00:37, 1048.54 examples/s]40627 examples [00:38, 1058.42 examples/s]40736 examples [00:38, 1066.50 examples/s]40848 examples [00:38, 1080.72 examples/s]40960 examples [00:38, 1091.25 examples/s]41070 examples [00:38, 1092.05 examples/s]41180 examples [00:38, 1090.37 examples/s]41290 examples [00:38, 1092.67 examples/s]41402 examples [00:38, 1099.58 examples/s]41514 examples [00:38, 1104.63 examples/s]41626 examples [00:38, 1107.05 examples/s]41739 examples [00:39, 1112.44 examples/s]41851 examples [00:39, 1100.88 examples/s]41962 examples [00:39, 1094.07 examples/s]42072 examples [00:39, 1091.84 examples/s]42182 examples [00:39, 1092.50 examples/s]42292 examples [00:39, 1091.79 examples/s]42402 examples [00:39, 1086.47 examples/s]42511 examples [00:39, 1085.39 examples/s]42620 examples [00:39, 1081.53 examples/s]42731 examples [00:39, 1088.68 examples/s]42840 examples [00:40, 1087.30 examples/s]42950 examples [00:40, 1090.97 examples/s]43061 examples [00:40, 1094.57 examples/s]43171 examples [00:40, 1094.77 examples/s]43281 examples [00:40, 1092.90 examples/s]43392 examples [00:40, 1096.75 examples/s]43504 examples [00:40, 1101.71 examples/s]43615 examples [00:40, 1103.77 examples/s]43727 examples [00:40, 1107.00 examples/s]43838 examples [00:40, 1089.57 examples/s]43948 examples [00:41, 1070.43 examples/s]44056 examples [00:41, 1060.14 examples/s]44164 examples [00:41, 1064.90 examples/s]44271 examples [00:41, 1046.56 examples/s]44380 examples [00:41, 1056.62 examples/s]44494 examples [00:41, 1078.65 examples/s]44605 examples [00:41, 1085.83 examples/s]44715 examples [00:41, 1088.10 examples/s]44824 examples [00:41, 1079.12 examples/s]44933 examples [00:42, 1079.84 examples/s]45042 examples [00:42, 1068.49 examples/s]45149 examples [00:42, 1052.11 examples/s]45258 examples [00:42, 1062.08 examples/s]45368 examples [00:42, 1072.64 examples/s]45477 examples [00:42, 1077.63 examples/s]45585 examples [00:42, 1069.16 examples/s]45692 examples [00:42, 1066.97 examples/s]45799 examples [00:42, 1064.80 examples/s]45906 examples [00:42, 1065.84 examples/s]46014 examples [00:43, 1067.15 examples/s]46125 examples [00:43, 1079.48 examples/s]46239 examples [00:43, 1095.39 examples/s]46352 examples [00:43, 1105.24 examples/s]46464 examples [00:43, 1107.11 examples/s]46575 examples [00:43, 1087.12 examples/s]46686 examples [00:43, 1093.40 examples/s]46796 examples [00:43, 1092.39 examples/s]46906 examples [00:43, 1092.82 examples/s]47017 examples [00:43, 1097.38 examples/s]47127 examples [00:44, 1090.26 examples/s]47237 examples [00:44, 1083.38 examples/s]47346 examples [00:44, 1060.53 examples/s]47458 examples [00:44, 1076.64 examples/s]47566 examples [00:44, 1072.63 examples/s]47678 examples [00:44, 1083.91 examples/s]47789 examples [00:44, 1090.75 examples/s]47899 examples [00:44, 1091.02 examples/s]48009 examples [00:44, 1093.38 examples/s]48121 examples [00:44, 1099.71 examples/s]48232 examples [00:45, 1094.45 examples/s]48342 examples [00:45, 1090.76 examples/s]48452 examples [00:45, 1085.51 examples/s]48563 examples [00:45, 1090.81 examples/s]48673 examples [00:45, 1087.62 examples/s]48782 examples [00:45, 1086.34 examples/s]48891 examples [00:45, 1085.64 examples/s]49000 examples [00:45, 1077.86 examples/s]49108 examples [00:45, 1075.69 examples/s]49216 examples [00:45, 1075.62 examples/s]49324 examples [00:46, 1069.38 examples/s]49431 examples [00:46, 1032.48 examples/s]49543 examples [00:46, 1055.04 examples/s]49655 examples [00:46, 1071.12 examples/s]49763 examples [00:46, 1067.08 examples/s]49870 examples [00:46, 1065.34 examples/s]49979 examples [00:46, 1070.97 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7042/50000 [00:00<00:00, 70415.87 examples/s] 41%|████▏     | 20669/50000 [00:00<00:00, 82355.27 examples/s] 69%|██████▉   | 34379/50000 [00:00<00:00, 93560.28 examples/s] 96%|█████████▌| 47903/50000 [00:00<00:00, 103091.28 examples/s]                                                                0 examples [00:00, ? examples/s]93 examples [00:00, 924.58 examples/s]196 examples [00:00, 951.60 examples/s]306 examples [00:00, 991.21 examples/s]421 examples [00:00, 1032.01 examples/s]531 examples [00:00, 1050.71 examples/s]643 examples [00:00, 1069.05 examples/s]759 examples [00:00, 1092.61 examples/s]875 examples [00:00, 1110.34 examples/s]990 examples [00:00, 1121.13 examples/s]1101 examples [00:01, 1117.77 examples/s]1213 examples [00:01, 1116.48 examples/s]1324 examples [00:01, 1114.41 examples/s]1437 examples [00:01, 1118.57 examples/s]1549 examples [00:01, 1110.29 examples/s]1660 examples [00:01, 1090.85 examples/s]1769 examples [00:01, 1075.06 examples/s]1878 examples [00:01, 1077.07 examples/s]1988 examples [00:01, 1083.42 examples/s]2103 examples [00:01, 1099.91 examples/s]2214 examples [00:02, 1092.73 examples/s]2329 examples [00:02, 1108.78 examples/s]2444 examples [00:02, 1118.25 examples/s]2560 examples [00:02, 1128.84 examples/s]2673 examples [00:02, 1109.10 examples/s]2785 examples [00:02, 1108.68 examples/s]2896 examples [00:02, 1075.30 examples/s]3009 examples [00:02, 1089.09 examples/s]3122 examples [00:02, 1098.80 examples/s]3235 examples [00:02, 1106.38 examples/s]3346 examples [00:03, 1106.36 examples/s]3457 examples [00:03, 1106.75 examples/s]3568 examples [00:03, 1106.29 examples/s]3680 examples [00:03, 1110.35 examples/s]3793 examples [00:03, 1115.74 examples/s]3905 examples [00:03, 1115.46 examples/s]4017 examples [00:03, 1109.29 examples/s]4128 examples [00:03, 1107.61 examples/s]4240 examples [00:03, 1110.14 examples/s]4353 examples [00:03, 1114.43 examples/s]4465 examples [00:04, 1114.05 examples/s]4580 examples [00:04, 1122.64 examples/s]4693 examples [00:04, 1122.06 examples/s]4806 examples [00:04, 1115.49 examples/s]4918 examples [00:04, 1103.06 examples/s]5029 examples [00:04, 1099.17 examples/s]5139 examples [00:04, 1097.11 examples/s]5249 examples [00:04, 1095.98 examples/s]5360 examples [00:04, 1098.24 examples/s]5470 examples [00:04, 1095.99 examples/s]5582 examples [00:05, 1102.30 examples/s]5696 examples [00:05, 1111.75 examples/s]5808 examples [00:05, 1088.00 examples/s]5921 examples [00:05, 1099.96 examples/s]6034 examples [00:05, 1107.79 examples/s]6147 examples [00:05, 1112.04 examples/s]6259 examples [00:05, 1113.28 examples/s]6372 examples [00:05, 1117.03 examples/s]6484 examples [00:05, 1115.82 examples/s]6596 examples [00:05, 1116.62 examples/s]6708 examples [00:06, 1115.11 examples/s]6820 examples [00:06, 1071.06 examples/s]6934 examples [00:06, 1088.36 examples/s]7047 examples [00:06, 1100.22 examples/s]7159 examples [00:06, 1105.09 examples/s]7270 examples [00:06, 1101.71 examples/s]7381 examples [00:06, 1103.29 examples/s]7497 examples [00:06, 1117.29 examples/s]7612 examples [00:06, 1125.39 examples/s]7725 examples [00:06, 1122.10 examples/s]7838 examples [00:07, 1118.03 examples/s]7950 examples [00:07, 1112.71 examples/s]8062 examples [00:07, 1113.52 examples/s]8175 examples [00:07, 1116.77 examples/s]8287 examples [00:07, 1113.35 examples/s]8399 examples [00:07, 1098.07 examples/s]8509 examples [00:07, 1097.38 examples/s]8619 examples [00:07, 1088.97 examples/s]8728 examples [00:07, 1088.25 examples/s]8837 examples [00:08, 1077.15 examples/s]8947 examples [00:08, 1081.61 examples/s]9058 examples [00:08, 1087.99 examples/s]9170 examples [00:08, 1094.87 examples/s]9280 examples [00:08, 1071.86 examples/s]9392 examples [00:08, 1083.16 examples/s]9501 examples [00:08, 1084.97 examples/s]9610 examples [00:08, 1053.22 examples/s]9721 examples [00:08, 1068.95 examples/s]9832 examples [00:08, 1080.10 examples/s]9943 examples [00:09, 1087.99 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePFRZNA/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompletePFRZNA/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe0b5226730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe0b5226730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe0b5226730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe092dd8a58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe03a68f358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fe0b5226730> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fe0b5226730> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fe0b5226730> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe03a700358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fe092e060b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fe02cce6400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fe02cce6400> 

  function with postional parmater data_info <function split_train_valid at 0x7fe02cce6400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fe02cce6510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fe02cce6510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fe02cce6510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=a4b223cb3164f750b59b10c80be36780d11455f86d1dc98a76914cc6e9aa4b12
  Stored in directory: /tmp/pip-ephem-wheel-cache-g_gs9ugf/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<18:25:54, 13.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:08:25, 18.2kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<9:15:16, 25.9kB/s]  .vector_cache/glove.6B.zip:   0%|          | 868k/862M [00:01<6:29:16, 36.9kB/s].vector_cache/glove.6B.zip:   0%|          | 3.51M/862M [00:01<4:31:52, 52.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.50M/862M [00:01<3:09:03, 75.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<2:12:01, 107kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:31:53, 153kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.9M/862M [00:01<1:03:57, 218kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<44:44, 311kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:01<31:11, 443kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<21:54, 629kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<15:27, 888kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.3M/862M [00:02<10:53, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<07:38, 1.78MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<05:32, 2.43MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<05:47, 2.32MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<05:52, 2.28MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:04<04:34, 2.93MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<05:44, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<06:45, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<05:18, 2.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.9M/862M [00:06<03:53, 3.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<09:44, 1.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<08:13, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<06:05, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<07:19, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<06:29, 2.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<04:52, 2.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<06:29, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<05:39, 2.32MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:12<04:17, 3.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<06:06, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<06:57, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<05:26, 2.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:14<03:57, 3.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<11:24, 1.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<09:19, 1.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:16<06:51, 1.90MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:18<07:49, 1.66MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<06:50, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<05:03, 2.56MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<06:33, 1.97MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<05:54, 2.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<04:27, 2.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<06:08, 2.09MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<05:36, 2.28MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<04:14, 3.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<05:59, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<06:48, 1.87MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:24, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:50, 2.18MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:24, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:06, 3.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<05:47, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<05:22, 2.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:01, 3.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<05:45, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:37, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:10, 2.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<03:47, 3.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<08:18, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:05, 1.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:14, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<06:39, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:13, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:40, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<05:54, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:23, 2.29MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<04:02, 3.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:42, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<05:14, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:58, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:40, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:13, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:39<03:55, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:38, 2.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<06:26, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<05:06, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:42, 3.25MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<11:32:51, 17.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<8:06:00, 24.8kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<5:39:46, 35.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<3:59:52, 50.0kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<2:50:19, 70.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<1:59:40, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<1:23:32, 143kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:39:53, 119kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:11:07, 168kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<49:59, 238kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<37:40, 315kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<28:49, 412kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<20:40, 573kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:49<14:40, 806kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<14:02, 840kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<10:50, 1.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<07:52, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<08:16, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:11, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:14, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<04:33, 2.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:12, 1.62MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:15, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<04:37, 2.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:57, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:09, 2.24MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<03:53, 2.97MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<02:51, 4.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<45:31, 253kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<34:14, 336kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<24:32, 469kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<18:57, 604kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<14:26, 793kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<10:19, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<09:53, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:04, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:56, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:49, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<07:05, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:27, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<03:58, 2.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<07:42, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:20, 1.77MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:40, 2.40MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<03:24, 3.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<47:31, 235kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<34:23, 325kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<24:15, 460kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<19:34, 567kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<14:50, 748kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:38, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<10:02, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<08:09, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<05:58, 1.84MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:46, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:53, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<05:18, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<03:49, 2.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<20:03, 544kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<15:10, 718kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<10:52, 999kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<10:08, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<08:12, 1.32MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:00, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:44, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:48, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:20, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:33, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<06:04, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:47, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:03, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:39, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<03:32, 3.00MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:56, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:32, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<03:26, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<04:51, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<05:33, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:26, 2.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<03:13, 3.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<1:16:32, 136kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<54:37, 191kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<38:24, 271kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<29:13, 354kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<21:30, 481kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<15:16, 676kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<13:03, 787kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<10:12, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<07:21, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<07:32, 1.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<07:21, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<05:36, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:00, 2.53MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<57:10, 178kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<41:01, 247kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<28:52, 350kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<22:31, 448kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<16:47, 600kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<11:58, 839kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<10:43, 934kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:31, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<06:12, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:41, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:42, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:12, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:13, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:39, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:46<03:28, 2.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:44, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:14, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:09, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<03:00, 3.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<9:24:26, 17.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<6:35:51, 24.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<4:36:37, 35.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<3:15:09, 49.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<2:18:33, 69.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<1:37:17, 99.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<1:07:56, 142kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<53:39, 179kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<38:31, 249kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<27:08, 353kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<21:09, 450kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<15:46, 604kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<11:13, 847kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<10:04, 939kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<08:59, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:46, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<06:14, 1.51MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:20, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:56, 2.38MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<04:56, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:24, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:18, 2.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:30, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:05, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:05, 2.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<03:58, 2.31MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:00, 3.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:15, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:54, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:57, 3.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<03:51, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<02:53, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:09, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:44, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:42, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:43, 3.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:29, 1.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:36, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:26, 2.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:30, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:51, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<02:49, 3.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:48, 1.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:57, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:41, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:36, 1.89MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:06, 2.12MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:04, 2.82MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:11, 2.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:43, 2.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:59, 2.15MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:31, 2.43MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:53, 2.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:05, 4.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<35:44, 238kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<25:51, 329kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<18:13, 465kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<14:43, 573kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<11:09, 755kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<08:00, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:33, 1.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<07:01, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:20, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:03, 1.64MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:23, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:16, 2.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:12, 1.95MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:47, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:49, 2.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:54, 2.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:34, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:42, 3.00MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:48, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:29, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:38, 3.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:44, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:17, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:28, 3.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<01:50, 4.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<31:22, 254kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<22:45, 349kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<16:04, 493kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<13:04, 603kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<09:57, 792kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<07:08, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<06:48, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:34, 1.40MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:03, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:40, 1.66MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:51, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:43, 2.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:43, 2.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<05:04, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:14, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:08, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<02:17, 3.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<16:24, 464kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<13:03, 583kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<09:28, 802kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<06:40, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<12:18, 613kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<09:24, 801kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<06:45, 1.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<06:27, 1.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<05:16, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<03:50, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:26, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:52, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:51, 2.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:44, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:21, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:31, 2.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:28, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:10, 2.28MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:24, 3.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:22, 2.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:06, 2.32MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:21, 3.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:19, 2.15MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:00, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:10, 3.24MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<6:46:42, 17.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<4:45:10, 24.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<3:19:04, 35.3kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<2:20:16, 49.9kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<1:39:35, 70.2kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<1:09:55, 99.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<48:43, 142kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<39:47, 174kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<28:32, 243kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<20:03, 344kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<15:34, 440kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<12:14, 560kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<08:54, 768kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<06:15, 1.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<6:36:26, 17.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<4:37:57, 24.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<3:13:59, 34.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<2:16:38, 49.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<1:36:14, 69.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<1:07:16, 99.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<48:25, 137kB/s]   .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<34:33, 192kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<24:15, 273kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<18:26, 357kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<14:14, 462kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<10:14, 642kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<07:13, 905kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<08:18, 784kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<06:29, 1.00MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<04:41, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:46, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:41, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:36, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:24<02:35, 2.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<47:12, 135kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<33:40, 189kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<23:37, 269kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<17:54, 352kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<13:49, 456kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<09:58, 631kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<07:56, 786kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<06:11, 1.01MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<04:28, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:33, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:27, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:24, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:32<02:25, 2.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<34:15, 178kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<24:35, 248kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<17:18, 351kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<13:27, 449kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<10:01, 601kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<07:08, 841kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<06:22, 937kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:41, 1.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:17, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:03, 1.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<44:11, 133kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<31:31, 187kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<22:06, 265kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<16:44, 348kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<12:54, 452kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<09:15, 627kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<06:33, 883kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:32, 880kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<05:10, 1.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:45, 1.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:56, 1.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:55, 1.45MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<02:59, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:09, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:59, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:01, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:48<02:09, 2.57MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<26:18, 211kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<18:57, 293kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<13:19, 414kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<10:34, 519kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<08:31, 643kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<06:13, 878kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<05:11, 1.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:11, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<03:03, 1.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:23, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:30, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:40, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:56, 2.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<03:30, 1.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<03:00, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:12, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:44, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [03:59<02:59, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<02:18, 2.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:40, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:38, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:04, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:15, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:45, 1.84MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:27, 2.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:48, 2.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:27, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<02:44, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:10, 2.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:18, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:07, 2.32MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:36, 3.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:14, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:35, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:03, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:11, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:01, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:31, 3.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:10, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:00, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:29, 3.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:08, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:58, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:28, 3.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:07, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:57, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:28, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<02:05, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:55, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<01:26, 3.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:03, 2.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:21, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:50, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:20, 3.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:26, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:47, 1.57MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:02, 2.14MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<01:28, 2.93MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<18:25, 234kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<13:47, 313kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<09:48, 439kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<06:51, 622kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<08:02, 529kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<06:03, 701kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<04:19, 976kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:58, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:12, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<02:20, 1.77MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:35, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:14, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:39, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:06, 1.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:24, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:55, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:45, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:19, 2.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:50, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:41, 2.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:16, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:47, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:03, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:36, 2.38MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:09, 3.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:55, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:26, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:47, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:06, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:06, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:38, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<01:14, 2.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:42, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:34, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:11, 3.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:39, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:53, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:30, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:05, 3.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<25:41, 136kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<18:18, 191kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<12:48, 271kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<09:42, 354kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<07:04, 483kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:03, 674kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<03:32, 952kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<14:37, 230kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<10:33, 318kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<07:25, 449kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<05:54, 557kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<04:48, 684kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:29, 937kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<02:27, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:48, 847kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:59, 1.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:09, 1.47MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:14, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:12, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:42, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:40, 1.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:29, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<01:06, 2.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:28, 2.05MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:37, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:16, 2.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<00:55, 3.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:40, 1.77MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:28, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:05, 2.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:26, 2.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:35, 1.80MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:14, 2.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:54, 3.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:39, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:27, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:04, 2.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:23, 1.98MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:15, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:56, 2.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:16, 2.10MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:09, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:51, 3.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:13, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:22, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:05, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:46, 3.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:50, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:32, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:08, 2.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:21, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:12, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<00:53, 2.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:10, 2.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:04, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<00:47, 2.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:06, 2.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:00, 2.31MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:45, 3.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:03, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:12, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:56, 2.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:40, 3.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:55, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:34, 1.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:08, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:17, 1.66MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:07, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:49, 2.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:03, 1.96MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:11, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:55, 2.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:41, 2.95MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:58, 2.05MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:53, 2.24MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:39, 2.97MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:54, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:01, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:48, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<00:34, 3.24MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<1:47:33, 17.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<1:15:12, 24.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<51:57, 35.1kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<36:03, 49.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<25:33, 69.7kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<17:51, 99.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<12:24, 138kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<08:49, 194kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<06:07, 275kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<04:34, 360kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<03:32, 465kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:32, 642kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:58, 799kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:32, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:06, 1.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<01:06, 1.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:55, 1.63MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:40, 2.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:48, 1.80MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:42, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:31, 2.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:40, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:36, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:27, 2.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:37, 2.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:33, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:51<00:25, 3.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:34, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:31, 2.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:23, 3.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:16, 4.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<08:52, 132kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<06:16, 185kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<04:22, 262kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<02:57, 373kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<06:13, 177kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<04:33, 240kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<03:12, 337kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<02:18, 447kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:42, 599kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<01:11, 836kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<01:01, 934kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:48, 1.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<00:34, 1.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:35, 1.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:30, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<00:22, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:26, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.10MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:16, 2.82MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:22, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:14, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:21, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:17, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:17, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:15, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 3.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:13, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:08, 3.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<03:34, 135kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<02:31, 188kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<01:41, 267kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<01:10, 351kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:54, 455kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:37, 628kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:23, 889kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<01:36, 215kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<01:08, 298kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:44, 421kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:31, 527kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:25, 651kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:17, 887kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:10, 1.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<01:48, 115kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<01:14, 162kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:45, 229kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:27, 304kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:20, 401kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:13, 559kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:07, 783kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:04, 873kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:03, 1.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.52MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.44MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<14:12:38,  7.82it/s]  0%|          | 857/400000 [00:00<9:55:48, 11.17it/s]  0%|          | 1711/400000 [00:00<6:56:24, 15.94it/s]  1%|          | 2581/400000 [00:00<4:51:04, 22.76it/s]  1%|          | 3470/400000 [00:00<3:23:31, 32.47it/s]  1%|          | 4322/400000 [00:00<2:22:23, 46.31it/s]  1%|▏         | 5192/400000 [00:00<1:39:40, 66.01it/s]  2%|▏         | 6066/400000 [00:00<1:09:50, 94.00it/s]  2%|▏         | 6917/400000 [00:00<49:01, 133.65it/s]   2%|▏         | 7792/400000 [00:01<34:27, 189.69it/s]  2%|▏         | 8639/400000 [00:01<24:18, 268.40it/s]  2%|▏         | 9496/400000 [00:01<17:12, 378.35it/s]  3%|▎         | 10339/400000 [00:01<12:14, 530.27it/s]  3%|▎         | 11190/400000 [00:01<08:46, 737.80it/s]  3%|▎         | 12035/400000 [00:01<06:22, 1013.46it/s]  3%|▎         | 12873/400000 [00:01<04:41, 1376.39it/s]  3%|▎         | 13740/400000 [00:01<03:29, 1840.91it/s]  4%|▎         | 14586/400000 [00:01<02:40, 2405.53it/s]  4%|▍         | 15429/400000 [00:01<02:06, 3048.73it/s]  4%|▍         | 16309/400000 [00:02<01:41, 3791.92it/s]  4%|▍         | 17179/400000 [00:02<01:23, 4564.22it/s]  5%|▍         | 18059/400000 [00:02<01:11, 5333.47it/s]  5%|▍         | 18938/400000 [00:02<01:03, 6046.27it/s]  5%|▍         | 19805/400000 [00:02<00:57, 6571.08it/s]  5%|▌         | 20663/400000 [00:02<00:53, 7065.82it/s]  5%|▌         | 21518/400000 [00:02<00:51, 7418.11it/s]  6%|▌         | 22382/400000 [00:02<00:48, 7744.88it/s]  6%|▌         | 23242/400000 [00:02<00:47, 7981.84it/s]  6%|▌         | 24098/400000 [00:02<00:46, 8135.38it/s]  6%|▌         | 24953/400000 [00:03<00:45, 8246.12it/s]  6%|▋         | 25807/400000 [00:03<00:45, 8291.51it/s]  7%|▋         | 26661/400000 [00:03<00:44, 8362.70it/s]  7%|▋         | 27521/400000 [00:03<00:44, 8431.92it/s]  7%|▋         | 28392/400000 [00:03<00:43, 8510.94it/s]  7%|▋         | 29251/400000 [00:03<00:44, 8417.71it/s]  8%|▊         | 30110/400000 [00:03<00:43, 8468.53it/s]  8%|▊         | 31003/400000 [00:03<00:42, 8599.33it/s]  8%|▊         | 31867/400000 [00:03<00:42, 8566.01it/s]  8%|▊         | 32734/400000 [00:03<00:42, 8596.36it/s]  8%|▊         | 33596/400000 [00:04<00:42, 8549.25it/s]  9%|▊         | 34453/400000 [00:04<00:43, 8362.35it/s]  9%|▉         | 35319/400000 [00:04<00:43, 8446.80it/s]  9%|▉         | 36177/400000 [00:04<00:42, 8483.51it/s]  9%|▉         | 37027/400000 [00:04<00:43, 8431.55it/s]  9%|▉         | 37887/400000 [00:04<00:42, 8480.64it/s] 10%|▉         | 38740/400000 [00:04<00:42, 8492.98it/s] 10%|▉         | 39593/400000 [00:04<00:42, 8502.57it/s] 10%|█         | 40444/400000 [00:04<00:42, 8501.21it/s] 10%|█         | 41295/400000 [00:04<00:42, 8436.39it/s] 11%|█         | 42139/400000 [00:05<00:42, 8421.32it/s] 11%|█         | 42987/400000 [00:05<00:42, 8436.84it/s] 11%|█         | 43831/400000 [00:05<00:42, 8347.72it/s] 11%|█         | 44675/400000 [00:05<00:42, 8373.66it/s] 11%|█▏        | 45517/400000 [00:05<00:42, 8385.45it/s] 12%|█▏        | 46364/400000 [00:05<00:42, 8408.46it/s] 12%|█▏        | 47243/400000 [00:05<00:41, 8516.94it/s] 12%|█▏        | 48096/400000 [00:05<00:41, 8503.91it/s] 12%|█▏        | 48965/400000 [00:05<00:41, 8558.74it/s] 12%|█▏        | 49836/400000 [00:05<00:40, 8603.32it/s] 13%|█▎        | 50697/400000 [00:06<00:40, 8570.70it/s] 13%|█▎        | 51556/400000 [00:06<00:40, 8573.46it/s] 13%|█▎        | 52414/400000 [00:06<00:41, 8335.11it/s] 13%|█▎        | 53279/400000 [00:06<00:41, 8424.51it/s] 14%|█▎        | 54155/400000 [00:06<00:40, 8520.76it/s] 14%|█▍        | 55010/400000 [00:06<00:40, 8527.59it/s] 14%|█▍        | 55864/400000 [00:06<00:41, 8333.47it/s] 14%|█▍        | 56711/400000 [00:06<00:40, 8373.27it/s] 14%|█▍        | 57579/400000 [00:06<00:40, 8461.42it/s] 15%|█▍        | 58450/400000 [00:06<00:40, 8534.27it/s] 15%|█▍        | 59305/400000 [00:07<00:40, 8499.22it/s] 15%|█▌        | 60178/400000 [00:07<00:39, 8566.09it/s] 15%|█▌        | 61046/400000 [00:07<00:39, 8598.82it/s] 15%|█▌        | 61909/400000 [00:07<00:39, 8605.80it/s] 16%|█▌        | 62778/400000 [00:07<00:39, 8630.82it/s] 16%|█▌        | 63642/400000 [00:07<00:39, 8590.14it/s] 16%|█▌        | 64518/400000 [00:07<00:38, 8637.74it/s] 16%|█▋        | 65382/400000 [00:07<00:38, 8600.71it/s] 17%|█▋        | 66262/400000 [00:07<00:38, 8658.34it/s] 17%|█▋        | 67129/400000 [00:08<00:38, 8599.03it/s] 17%|█▋        | 68006/400000 [00:08<00:38, 8646.91it/s] 17%|█▋        | 68881/400000 [00:08<00:38, 8674.90it/s] 17%|█▋        | 69749/400000 [00:08<00:38, 8676.01it/s] 18%|█▊        | 70620/400000 [00:08<00:37, 8685.30it/s] 18%|█▊        | 71489/400000 [00:08<00:38, 8630.60it/s] 18%|█▊        | 72353/400000 [00:08<00:38, 8524.66it/s] 18%|█▊        | 73246/400000 [00:08<00:37, 8641.24it/s] 19%|█▊        | 74124/400000 [00:08<00:37, 8682.09it/s] 19%|█▊        | 74993/400000 [00:08<00:37, 8658.76it/s] 19%|█▉        | 75860/400000 [00:09<00:37, 8597.43it/s] 19%|█▉        | 76721/400000 [00:09<00:37, 8580.34it/s] 19%|█▉        | 77580/400000 [00:09<00:37, 8563.26it/s] 20%|█▉        | 78455/400000 [00:09<00:37, 8616.37it/s] 20%|█▉        | 79346/400000 [00:09<00:36, 8700.98it/s] 20%|██        | 80217/400000 [00:09<00:36, 8697.31it/s] 20%|██        | 81087/400000 [00:09<00:37, 8571.44it/s] 20%|██        | 81945/400000 [00:09<00:37, 8559.32it/s] 21%|██        | 82803/400000 [00:09<00:37, 8562.93it/s] 21%|██        | 83691/400000 [00:09<00:36, 8655.44it/s] 21%|██        | 84563/400000 [00:10<00:36, 8672.15it/s] 21%|██▏       | 85441/400000 [00:10<00:36, 8701.74it/s] 22%|██▏       | 86324/400000 [00:10<00:35, 8737.16it/s] 22%|██▏       | 87198/400000 [00:10<00:36, 8672.26it/s] 22%|██▏       | 88066/400000 [00:10<00:36, 8641.10it/s] 22%|██▏       | 88931/400000 [00:10<00:36, 8490.70it/s] 22%|██▏       | 89781/400000 [00:10<00:37, 8246.28it/s] 23%|██▎       | 90622/400000 [00:10<00:37, 8291.91it/s] 23%|██▎       | 91489/400000 [00:10<00:36, 8400.59it/s] 23%|██▎       | 92368/400000 [00:10<00:36, 8511.79it/s] 23%|██▎       | 93246/400000 [00:11<00:35, 8590.43it/s] 24%|██▎       | 94129/400000 [00:11<00:35, 8658.96it/s] 24%|██▍       | 95023/400000 [00:11<00:34, 8738.97it/s] 24%|██▍       | 95923/400000 [00:11<00:34, 8814.56it/s] 24%|██▍       | 96806/400000 [00:11<00:34, 8804.40it/s] 24%|██▍       | 97687/400000 [00:11<00:34, 8748.90it/s] 25%|██▍       | 98563/400000 [00:11<00:34, 8677.04it/s] 25%|██▍       | 99432/400000 [00:11<00:34, 8660.55it/s] 25%|██▌       | 100316/400000 [00:11<00:34, 8712.38it/s] 25%|██▌       | 101198/400000 [00:11<00:34, 8741.82it/s] 26%|██▌       | 102075/400000 [00:12<00:34, 8747.39it/s] 26%|██▌       | 102962/400000 [00:12<00:33, 8781.16it/s] 26%|██▌       | 103841/400000 [00:12<00:34, 8655.58it/s] 26%|██▌       | 104708/400000 [00:12<00:34, 8621.66it/s] 26%|██▋       | 105582/400000 [00:12<00:34, 8654.00it/s] 27%|██▋       | 106464/400000 [00:12<00:33, 8701.66it/s] 27%|██▋       | 107335/400000 [00:12<00:33, 8670.44it/s] 27%|██▋       | 108203/400000 [00:12<00:33, 8650.57it/s] 27%|██▋       | 109107/400000 [00:12<00:33, 8763.68it/s] 27%|██▋       | 109998/400000 [00:12<00:32, 8804.54it/s] 28%|██▊       | 110879/400000 [00:13<00:32, 8794.43it/s] 28%|██▊       | 111759/400000 [00:13<00:32, 8767.31it/s] 28%|██▊       | 112636/400000 [00:13<00:32, 8714.53it/s] 28%|██▊       | 113508/400000 [00:13<00:32, 8695.55it/s] 29%|██▊       | 114407/400000 [00:13<00:32, 8779.94it/s] 29%|██▉       | 115286/400000 [00:13<00:33, 8616.84it/s] 29%|██▉       | 116149/400000 [00:13<00:32, 8605.85it/s] 29%|██▉       | 117027/400000 [00:13<00:32, 8656.91it/s] 29%|██▉       | 117894/400000 [00:13<00:32, 8647.77it/s] 30%|██▉       | 118760/400000 [00:13<00:32, 8594.83it/s] 30%|██▉       | 119620/400000 [00:14<00:32, 8572.84it/s] 30%|███       | 120478/400000 [00:14<00:32, 8549.67it/s] 30%|███       | 121344/400000 [00:14<00:32, 8580.39it/s] 31%|███       | 122226/400000 [00:14<00:32, 8649.72it/s] 31%|███       | 123097/400000 [00:14<00:31, 8665.87it/s] 31%|███       | 123964/400000 [00:14<00:31, 8652.60it/s] 31%|███       | 124830/400000 [00:14<00:31, 8612.76it/s] 31%|███▏      | 125692/400000 [00:14<00:32, 8498.94it/s] 32%|███▏      | 126555/400000 [00:14<00:32, 8535.45it/s] 32%|███▏      | 127416/400000 [00:14<00:31, 8554.91it/s] 32%|███▏      | 128272/400000 [00:15<00:32, 8340.09it/s] 32%|███▏      | 129154/400000 [00:15<00:31, 8475.64it/s] 33%|███▎      | 130014/400000 [00:15<00:31, 8509.77it/s] 33%|███▎      | 130867/400000 [00:15<00:31, 8510.04it/s] 33%|███▎      | 131727/400000 [00:15<00:31, 8536.34it/s] 33%|███▎      | 132591/400000 [00:15<00:31, 8566.27it/s] 33%|███▎      | 133460/400000 [00:15<00:30, 8600.98it/s] 34%|███▎      | 134328/400000 [00:15<00:30, 8622.72it/s] 34%|███▍      | 135191/400000 [00:15<00:30, 8612.23it/s] 34%|███▍      | 136053/400000 [00:15<00:31, 8464.92it/s] 34%|███▍      | 136912/400000 [00:16<00:30, 8499.16it/s] 34%|███▍      | 137777/400000 [00:16<00:30, 8540.99it/s] 35%|███▍      | 138657/400000 [00:16<00:30, 8616.13it/s] 35%|███▍      | 139523/400000 [00:16<00:30, 8628.21it/s] 35%|███▌      | 140392/400000 [00:16<00:30, 8646.15it/s] 35%|███▌      | 141257/400000 [00:16<00:30, 8590.53it/s] 36%|███▌      | 142124/400000 [00:16<00:29, 8614.06it/s] 36%|███▌      | 142986/400000 [00:16<00:30, 8565.16it/s] 36%|███▌      | 143843/400000 [00:16<00:30, 8534.93it/s] 36%|███▌      | 144697/400000 [00:17<00:29, 8525.41it/s] 36%|███▋      | 145555/400000 [00:17<00:29, 8539.98it/s] 37%|███▋      | 146421/400000 [00:17<00:29, 8574.92it/s] 37%|███▋      | 147293/400000 [00:17<00:29, 8615.77it/s] 37%|███▋      | 148179/400000 [00:17<00:28, 8684.91it/s] 37%|███▋      | 149053/400000 [00:17<00:28, 8701.14it/s] 37%|███▋      | 149939/400000 [00:17<00:28, 8745.30it/s] 38%|███▊      | 150814/400000 [00:17<00:28, 8654.86it/s] 38%|███▊      | 151680/400000 [00:17<00:28, 8587.69it/s] 38%|███▊      | 152573/400000 [00:17<00:28, 8687.20it/s] 38%|███▊      | 153447/400000 [00:18<00:28, 8701.57it/s] 39%|███▊      | 154318/400000 [00:18<00:29, 8432.13it/s] 39%|███▉      | 155164/400000 [00:18<00:29, 8357.94it/s] 39%|███▉      | 156029/400000 [00:18<00:28, 8441.74it/s] 39%|███▉      | 156933/400000 [00:18<00:28, 8612.22it/s] 39%|███▉      | 157799/400000 [00:18<00:28, 8623.22it/s] 40%|███▉      | 158687/400000 [00:18<00:27, 8698.29it/s] 40%|███▉      | 159558/400000 [00:18<00:27, 8661.70it/s] 40%|████      | 160458/400000 [00:18<00:27, 8758.24it/s] 40%|████      | 161335/400000 [00:18<00:27, 8661.78it/s] 41%|████      | 162204/400000 [00:19<00:27, 8669.96it/s] 41%|████      | 163093/400000 [00:19<00:27, 8732.84it/s] 41%|████      | 163976/400000 [00:19<00:26, 8760.32it/s] 41%|████      | 164869/400000 [00:19<00:26, 8808.28it/s] 41%|████▏     | 165751/400000 [00:19<00:26, 8797.65it/s] 42%|████▏     | 166652/400000 [00:19<00:26, 8858.71it/s] 42%|████▏     | 167561/400000 [00:19<00:26, 8925.99it/s] 42%|████▏     | 168454/400000 [00:19<00:26, 8887.98it/s] 42%|████▏     | 169344/400000 [00:19<00:26, 8869.78it/s] 43%|████▎     | 170232/400000 [00:19<00:25, 8848.50it/s] 43%|████▎     | 171133/400000 [00:20<00:25, 8895.89it/s] 43%|████▎     | 172035/400000 [00:20<00:25, 8930.72it/s] 43%|████▎     | 172935/400000 [00:20<00:25, 8950.41it/s] 43%|████▎     | 173831/400000 [00:20<00:25, 8952.56it/s] 44%|████▎     | 174730/400000 [00:20<00:25, 8961.60it/s] 44%|████▍     | 175627/400000 [00:20<00:25, 8905.55it/s] 44%|████▍     | 176518/400000 [00:20<00:25, 8794.64it/s] 44%|████▍     | 177398/400000 [00:20<00:25, 8762.79it/s] 45%|████▍     | 178275/400000 [00:20<00:25, 8741.32it/s] 45%|████▍     | 179150/400000 [00:20<00:25, 8738.17it/s] 45%|████▌     | 180025/400000 [00:21<00:25, 8738.83it/s] 45%|████▌     | 180899/400000 [00:21<00:25, 8647.17it/s] 45%|████▌     | 181767/400000 [00:21<00:25, 8655.94it/s] 46%|████▌     | 182638/400000 [00:21<00:25, 8671.49it/s] 46%|████▌     | 183524/400000 [00:21<00:24, 8724.57it/s] 46%|████▌     | 184397/400000 [00:21<00:25, 8506.33it/s] 46%|████▋     | 185249/400000 [00:21<00:25, 8322.81it/s] 47%|████▋     | 186119/400000 [00:21<00:25, 8431.63it/s] 47%|████▋     | 186998/400000 [00:21<00:24, 8535.64it/s] 47%|████▋     | 187866/400000 [00:21<00:24, 8576.27it/s] 47%|████▋     | 188725/400000 [00:22<00:24, 8569.95it/s] 47%|████▋     | 189583/400000 [00:22<00:24, 8570.22it/s] 48%|████▊     | 190441/400000 [00:22<00:24, 8555.54it/s] 48%|████▊     | 191297/400000 [00:22<00:24, 8499.05it/s] 48%|████▊     | 192152/400000 [00:22<00:24, 8513.99it/s] 48%|████▊     | 193017/400000 [00:22<00:24, 8553.58it/s] 48%|████▊     | 193888/400000 [00:22<00:23, 8596.84it/s] 49%|████▊     | 194760/400000 [00:22<00:23, 8630.75it/s] 49%|████▉     | 195624/400000 [00:22<00:23, 8615.68it/s] 49%|████▉     | 196491/400000 [00:22<00:23, 8630.68it/s] 49%|████▉     | 197355/400000 [00:23<00:23, 8596.39it/s] 50%|████▉     | 198228/400000 [00:23<00:23, 8634.34it/s] 50%|████▉     | 199092/400000 [00:23<00:24, 8348.43it/s] 50%|████▉     | 199947/400000 [00:23<00:23, 8405.68it/s] 50%|█████     | 200814/400000 [00:23<00:23, 8482.48it/s] 50%|█████     | 201667/400000 [00:23<00:23, 8495.98it/s] 51%|█████     | 202518/400000 [00:23<00:23, 8469.24it/s] 51%|█████     | 203376/400000 [00:23<00:23, 8501.71it/s] 51%|█████     | 204242/400000 [00:23<00:22, 8546.79it/s] 51%|█████▏    | 205102/400000 [00:23<00:22, 8560.31it/s] 51%|█████▏    | 205977/400000 [00:24<00:22, 8615.78it/s] 52%|█████▏    | 206839/400000 [00:24<00:22, 8517.01it/s] 52%|█████▏    | 207692/400000 [00:24<00:22, 8517.92it/s] 52%|█████▏    | 208549/400000 [00:24<00:22, 8533.46it/s] 52%|█████▏    | 209405/400000 [00:24<00:22, 8540.12it/s] 53%|█████▎    | 210262/400000 [00:24<00:22, 8547.20it/s] 53%|█████▎    | 211117/400000 [00:24<00:22, 8516.05it/s] 53%|█████▎    | 211983/400000 [00:24<00:21, 8558.33it/s] 53%|█████▎    | 212839/400000 [00:24<00:22, 8459.69it/s] 53%|█████▎    | 213686/400000 [00:24<00:22, 8376.43it/s] 54%|█████▎    | 214525/400000 [00:25<00:22, 8236.86it/s] 54%|█████▍    | 215369/400000 [00:25<00:22, 8296.24it/s] 54%|█████▍    | 216224/400000 [00:25<00:21, 8369.68it/s] 54%|█████▍    | 217090/400000 [00:25<00:21, 8454.58it/s] 54%|█████▍    | 217944/400000 [00:25<00:21, 8478.10it/s] 55%|█████▍    | 218793/400000 [00:25<00:21, 8465.44it/s] 55%|█████▍    | 219640/400000 [00:25<00:21, 8440.62it/s] 55%|█████▌    | 220500/400000 [00:25<00:21, 8485.07it/s] 55%|█████▌    | 221371/400000 [00:25<00:20, 8551.00it/s] 56%|█████▌    | 222238/400000 [00:26<00:20, 8586.07it/s] 56%|█████▌    | 223100/400000 [00:26<00:20, 8595.27it/s] 56%|█████▌    | 223960/400000 [00:26<00:20, 8579.99it/s] 56%|█████▌    | 224838/400000 [00:26<00:20, 8637.08it/s] 56%|█████▋    | 225713/400000 [00:26<00:20, 8670.60it/s] 57%|█████▋    | 226597/400000 [00:26<00:19, 8710.47it/s] 57%|█████▋    | 227469/400000 [00:26<00:19, 8656.48it/s] 57%|█████▋    | 228335/400000 [00:26<00:20, 8516.43it/s] 57%|█████▋    | 229197/400000 [00:26<00:19, 8546.30it/s] 58%|█████▊    | 230083/400000 [00:26<00:19, 8635.42it/s] 58%|█████▊    | 230954/400000 [00:27<00:19, 8655.92it/s] 58%|█████▊    | 231847/400000 [00:27<00:19, 8734.98it/s] 58%|█████▊    | 232721/400000 [00:27<00:19, 8706.35it/s] 58%|█████▊    | 233597/400000 [00:27<00:19, 8721.38it/s] 59%|█████▊    | 234476/400000 [00:27<00:18, 8739.44it/s] 59%|█████▉    | 235377/400000 [00:27<00:18, 8816.39it/s] 59%|█████▉    | 236259/400000 [00:27<00:18, 8791.39it/s] 59%|█████▉    | 237139/400000 [00:27<00:18, 8682.00it/s] 60%|█████▉    | 238024/400000 [00:27<00:18, 8731.13it/s] 60%|█████▉    | 238898/400000 [00:27<00:18, 8699.91it/s] 60%|█████▉    | 239775/400000 [00:28<00:18, 8720.03it/s] 60%|██████    | 240648/400000 [00:28<00:18, 8683.76it/s] 60%|██████    | 241517/400000 [00:28<00:18, 8674.92it/s] 61%|██████    | 242392/400000 [00:28<00:18, 8694.71it/s] 61%|██████    | 243262/400000 [00:28<00:18, 8565.39it/s] 61%|██████    | 244132/400000 [00:28<00:18, 8604.56it/s] 61%|██████▏   | 245057/400000 [00:28<00:17, 8786.39it/s] 61%|██████▏   | 245943/400000 [00:28<00:17, 8806.12it/s] 62%|██████▏   | 246825/400000 [00:28<00:17, 8786.45it/s] 62%|██████▏   | 247705/400000 [00:28<00:17, 8783.62it/s] 62%|██████▏   | 248584/400000 [00:29<00:17, 8614.75it/s] 62%|██████▏   | 249449/400000 [00:29<00:17, 8622.15it/s] 63%|██████▎   | 250325/400000 [00:29<00:17, 8662.29it/s] 63%|██████▎   | 251192/400000 [00:29<00:17, 8613.84it/s] 63%|██████▎   | 252058/400000 [00:29<00:17, 8625.33it/s] 63%|██████▎   | 252921/400000 [00:29<00:17, 8598.11it/s] 63%|██████▎   | 253782/400000 [00:29<00:17, 8577.03it/s] 64%|██████▎   | 254649/400000 [00:29<00:16, 8602.44it/s] 64%|██████▍   | 255522/400000 [00:29<00:16, 8639.83it/s] 64%|██████▍   | 256397/400000 [00:29<00:16, 8670.49it/s] 64%|██████▍   | 257265/400000 [00:30<00:16, 8482.99it/s] 65%|██████▍   | 258115/400000 [00:30<00:16, 8475.40it/s] 65%|██████▍   | 258973/400000 [00:30<00:16, 8504.17it/s] 65%|██████▍   | 259839/400000 [00:30<00:16, 8549.87it/s] 65%|██████▌   | 260695/400000 [00:30<00:16, 8515.69it/s] 65%|██████▌   | 261547/400000 [00:30<00:16, 8500.22it/s] 66%|██████▌   | 262409/400000 [00:30<00:16, 8534.96it/s] 66%|██████▌   | 263263/400000 [00:30<00:16, 8526.09it/s] 66%|██████▌   | 264116/400000 [00:30<00:15, 8510.24it/s] 66%|██████▌   | 264972/400000 [00:30<00:15, 8523.59it/s] 66%|██████▋   | 265828/400000 [00:31<00:15, 8533.00it/s] 67%|██████▋   | 266682/400000 [00:31<00:15, 8485.16it/s] 67%|██████▋   | 267557/400000 [00:31<00:15, 8562.68it/s] 67%|██████▋   | 268467/400000 [00:31<00:15, 8715.14it/s] 67%|██████▋   | 269340/400000 [00:31<00:15, 8682.07it/s] 68%|██████▊   | 270231/400000 [00:31<00:14, 8748.79it/s] 68%|██████▊   | 271107/400000 [00:31<00:14, 8728.64it/s] 68%|██████▊   | 271981/400000 [00:31<00:14, 8702.89it/s] 68%|██████▊   | 272852/400000 [00:31<00:14, 8500.04it/s] 68%|██████▊   | 273704/400000 [00:31<00:15, 8377.92it/s] 69%|██████▊   | 274592/400000 [00:32<00:14, 8522.22it/s] 69%|██████▉   | 275468/400000 [00:32<00:14, 8591.14it/s] 69%|██████▉   | 276356/400000 [00:32<00:14, 8674.42it/s] 69%|██████▉   | 277263/400000 [00:32<00:13, 8788.17it/s] 70%|██████▉   | 278143/400000 [00:32<00:13, 8745.46it/s] 70%|██████▉   | 279019/400000 [00:32<00:14, 8434.32it/s] 70%|██████▉   | 279917/400000 [00:32<00:13, 8589.23it/s] 70%|███████   | 280801/400000 [00:32<00:13, 8660.90it/s] 70%|███████   | 281701/400000 [00:32<00:13, 8758.44it/s] 71%|███████   | 282613/400000 [00:32<00:13, 8861.55it/s] 71%|███████   | 283501/400000 [00:33<00:13, 8694.84it/s] 71%|███████   | 284373/400000 [00:33<00:13, 8606.89it/s] 71%|███████▏  | 285236/400000 [00:33<00:13, 8586.72it/s] 72%|███████▏  | 286096/400000 [00:33<00:13, 8560.60it/s] 72%|███████▏  | 286960/400000 [00:33<00:13, 8583.42it/s] 72%|███████▏  | 287819/400000 [00:33<00:13, 8221.95it/s] 72%|███████▏  | 288664/400000 [00:33<00:13, 8288.98it/s] 72%|███████▏  | 289569/400000 [00:33<00:12, 8501.31it/s] 73%|███████▎  | 290450/400000 [00:33<00:12, 8589.68it/s] 73%|███████▎  | 291321/400000 [00:34<00:12, 8624.64it/s] 73%|███████▎  | 292193/400000 [00:34<00:12, 8652.27it/s] 73%|███████▎  | 293102/400000 [00:34<00:12, 8778.89it/s] 73%|███████▎  | 293982/400000 [00:34<00:12, 8754.12it/s] 74%|███████▎  | 294859/400000 [00:34<00:12, 8722.31it/s] 74%|███████▍  | 295732/400000 [00:34<00:11, 8691.13it/s] 74%|███████▍  | 296602/400000 [00:34<00:11, 8642.39it/s] 74%|███████▍  | 297537/400000 [00:34<00:11, 8841.44it/s] 75%|███████▍  | 298423/400000 [00:34<00:11, 8822.38it/s] 75%|███████▍  | 299307/400000 [00:34<00:11, 8802.71it/s] 75%|███████▌  | 300188/400000 [00:35<00:11, 8792.08it/s] 75%|███████▌  | 301068/400000 [00:35<00:11, 8502.52it/s] 75%|███████▌  | 301923/400000 [00:35<00:11, 8514.06it/s] 76%|███████▌  | 302790/400000 [00:35<00:11, 8556.32it/s] 76%|███████▌  | 303647/400000 [00:35<00:11, 8491.61it/s] 76%|███████▌  | 304498/400000 [00:35<00:11, 8358.44it/s] 76%|███████▋  | 305335/400000 [00:35<00:11, 8274.95it/s] 77%|███████▋  | 306212/400000 [00:35<00:11, 8416.73it/s] 77%|███████▋  | 307089/400000 [00:35<00:10, 8517.25it/s] 77%|███████▋  | 307942/400000 [00:35<00:10, 8478.86it/s] 77%|███████▋  | 308827/400000 [00:36<00:10, 8584.42it/s] 77%|███████▋  | 309689/400000 [00:36<00:10, 8594.42it/s] 78%|███████▊  | 310589/400000 [00:36<00:10, 8711.61it/s] 78%|███████▊  | 311462/400000 [00:36<00:10, 8709.21it/s] 78%|███████▊  | 312334/400000 [00:36<00:10, 8670.36it/s] 78%|███████▊  | 313202/400000 [00:36<00:10, 8601.72it/s] 79%|███████▊  | 314088/400000 [00:36<00:09, 8676.52it/s] 79%|███████▊  | 314963/400000 [00:36<00:09, 8696.45it/s] 79%|███████▉  | 315848/400000 [00:36<00:09, 8741.64it/s] 79%|███████▉  | 316723/400000 [00:36<00:09, 8713.86it/s] 79%|███████▉  | 317595/400000 [00:37<00:09, 8621.76it/s] 80%|███████▉  | 318458/400000 [00:37<00:09, 8599.45it/s] 80%|███████▉  | 319319/400000 [00:37<00:09, 8532.32it/s] 80%|████████  | 320179/400000 [00:37<00:09, 8551.87it/s] 80%|████████  | 321046/400000 [00:37<00:09, 8585.71it/s] 80%|████████  | 321905/400000 [00:37<00:09, 8570.04it/s] 81%|████████  | 322784/400000 [00:37<00:08, 8632.77it/s] 81%|████████  | 323663/400000 [00:37<00:08, 8678.37it/s] 81%|████████  | 324532/400000 [00:37<00:08, 8666.12it/s] 81%|████████▏ | 325399/400000 [00:37<00:08, 8646.80it/s] 82%|████████▏ | 326264/400000 [00:38<00:08, 8569.29it/s] 82%|████████▏ | 327122/400000 [00:38<00:08, 8564.94it/s] 82%|████████▏ | 328011/400000 [00:38<00:08, 8657.83it/s] 82%|████████▏ | 328879/400000 [00:38<00:08, 8662.12it/s] 82%|████████▏ | 329746/400000 [00:38<00:08, 8645.15it/s] 83%|████████▎ | 330611/400000 [00:38<00:08, 8588.93it/s] 83%|████████▎ | 331477/400000 [00:38<00:07, 8608.57it/s] 83%|████████▎ | 332348/400000 [00:38<00:07, 8637.09it/s] 83%|████████▎ | 333220/400000 [00:38<00:07, 8659.92it/s] 84%|████████▎ | 334109/400000 [00:38<00:07, 8725.55it/s] 84%|████████▎ | 334982/400000 [00:39<00:07, 8679.16it/s] 84%|████████▍ | 335867/400000 [00:39<00:07, 8727.02it/s] 84%|████████▍ | 336758/400000 [00:39<00:07, 8781.08it/s] 84%|████████▍ | 337654/400000 [00:39<00:07, 8833.49it/s] 85%|████████▍ | 338548/400000 [00:39<00:06, 8864.35it/s] 85%|████████▍ | 339435/400000 [00:39<00:06, 8850.40it/s] 85%|████████▌ | 340332/400000 [00:39<00:06, 8883.23it/s] 85%|████████▌ | 341221/400000 [00:39<00:06, 8884.97it/s] 86%|████████▌ | 342159/400000 [00:39<00:06, 9027.53it/s] 86%|████████▌ | 343103/400000 [00:39<00:06, 9145.71it/s] 86%|████████▌ | 344019/400000 [00:40<00:06, 9025.97it/s] 86%|████████▌ | 344923/400000 [00:40<00:06, 9022.30it/s] 86%|████████▋ | 345826/400000 [00:40<00:06, 8964.60it/s] 87%|████████▋ | 346723/400000 [00:40<00:05, 8892.23it/s] 87%|████████▋ | 347613/400000 [00:40<00:06, 8587.49it/s] 87%|████████▋ | 348475/400000 [00:40<00:06, 8471.72it/s] 87%|████████▋ | 349339/400000 [00:40<00:05, 8519.34it/s] 88%|████████▊ | 350202/400000 [00:40<00:05, 8551.20it/s] 88%|████████▊ | 351081/400000 [00:40<00:05, 8618.79it/s] 88%|████████▊ | 351969/400000 [00:40<00:05, 8693.91it/s] 88%|████████▊ | 352845/400000 [00:41<00:05, 8712.33it/s] 88%|████████▊ | 353717/400000 [00:41<00:05, 8659.73it/s] 89%|████████▊ | 354591/400000 [00:41<00:05, 8682.52it/s] 89%|████████▉ | 355472/400000 [00:41<00:05, 8718.47it/s] 89%|████████▉ | 356345/400000 [00:41<00:05, 8659.80it/s] 89%|████████▉ | 357247/400000 [00:41<00:04, 8762.18it/s] 90%|████████▉ | 358172/400000 [00:41<00:04, 8902.17it/s] 90%|████████▉ | 359064/400000 [00:41<00:04, 8790.56it/s] 90%|████████▉ | 359969/400000 [00:41<00:04, 8866.41it/s] 90%|█████████ | 360857/400000 [00:42<00:04, 8820.70it/s] 90%|█████████ | 361740/400000 [00:42<00:04, 8819.42it/s] 91%|█████████ | 362640/400000 [00:42<00:04, 8870.88it/s] 91%|█████████ | 363528/400000 [00:42<00:04, 8621.90it/s] 91%|█████████ | 364392/400000 [00:42<00:04, 8491.72it/s] 91%|█████████▏| 365273/400000 [00:42<00:04, 8583.53it/s] 92%|█████████▏| 366178/400000 [00:42<00:03, 8716.61it/s] 92%|█████████▏| 367083/400000 [00:42<00:03, 8811.63it/s] 92%|█████████▏| 367983/400000 [00:42<00:03, 8864.89it/s] 92%|█████████▏| 368897/400000 [00:42<00:03, 8945.54it/s] 92%|█████████▏| 369802/400000 [00:43<00:03, 8976.08it/s] 93%|█████████▎| 370701/400000 [00:43<00:03, 8907.98it/s] 93%|█████████▎| 371597/400000 [00:43<00:03, 8920.85it/s] 93%|█████████▎| 372502/400000 [00:43<00:03, 8957.47it/s] 93%|█████████▎| 373399/400000 [00:43<00:02, 8958.67it/s] 94%|█████████▎| 374301/400000 [00:43<00:02, 8976.68it/s] 94%|█████████▍| 375199/400000 [00:43<00:02, 8927.04it/s] 94%|█████████▍| 376092/400000 [00:43<00:02, 8876.02it/s] 94%|█████████▍| 376980/400000 [00:43<00:02, 8816.86it/s] 94%|█████████▍| 377862/400000 [00:43<00:02, 8748.30it/s] 95%|█████████▍| 378738/400000 [00:44<00:02, 8501.68it/s] 95%|█████████▍| 379606/400000 [00:44<00:02, 8551.95it/s] 95%|█████████▌| 380475/400000 [00:44<00:02, 8591.20it/s] 95%|█████████▌| 381336/400000 [00:44<00:02, 8584.64it/s] 96%|█████████▌| 382200/400000 [00:44<00:02, 8599.64it/s] 96%|█████████▌| 383099/400000 [00:44<00:01, 8710.39it/s] 96%|█████████▌| 383999/400000 [00:44<00:01, 8794.89it/s] 96%|█████████▌| 384905/400000 [00:44<00:01, 8872.14it/s] 96%|█████████▋| 385793/400000 [00:44<00:01, 8779.26it/s] 97%|█████████▋| 386672/400000 [00:44<00:01, 8732.47it/s] 97%|█████████▋| 387546/400000 [00:45<00:01, 8710.89it/s] 97%|█████████▋| 388427/400000 [00:45<00:01, 8738.04it/s] 97%|█████████▋| 389344/400000 [00:45<00:01, 8862.44it/s] 98%|█████████▊| 390231/400000 [00:45<00:01, 8759.05it/s] 98%|█████████▊| 391108/400000 [00:45<00:01, 8569.49it/s] 98%|█████████▊| 391991/400000 [00:45<00:00, 8645.43it/s] 98%|█████████▊| 392857/400000 [00:45<00:00, 8447.71it/s] 98%|█████████▊| 393737/400000 [00:45<00:00, 8548.01it/s] 99%|█████████▊| 394628/400000 [00:45<00:00, 8650.99it/s] 99%|█████████▉| 395495/400000 [00:45<00:00, 8652.13it/s] 99%|█████████▉| 396362/400000 [00:46<00:00, 8646.94it/s] 99%|█████████▉| 397228/400000 [00:46<00:00, 8620.77it/s]100%|█████████▉| 398095/400000 [00:46<00:00, 8634.33it/s]100%|█████████▉| 398959/400000 [00:46<00:00, 8615.89it/s]100%|█████████▉| 399847/400000 [00:46<00:00, 8691.39it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8603.31it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fe02cd00358>, <torchtext.data.dataset.TabularDataset object at 0x7fe02cd004a8>, <torchtext.vocab.Vocab object at 0x7fe02cd003c8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.23 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.89 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 18.89 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 18.89 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.97 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.97 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.55 file/s]2020-07-15 18:21:50.536445: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-15 18:21:50.540895: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-15 18:21:50.541073: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564c6e24e870 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-15 18:21:50.541089: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 38%|███▊      | 3768320/9912422 [00:00<00:00, 36699234.81it/s]9920512it [00:00, 33928335.80it/s]                             
0it [00:00, ?it/s]32768it [00:00, 681423.11it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 143313.97it/s]1654784it [00:00, 10333832.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 149878.25it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
