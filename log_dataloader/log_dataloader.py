
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f232689dea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f232689dea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2391b5c1e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2391b5c1e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f23afea4e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f23afea4e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f233ee84488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f233ee84488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f233ee84488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  7%|▋         | 655360/9912422 [00:00<00:01, 6551763.02it/s]9920512it [00:00, 29962306.05it/s]                           
0it [00:00, ?it/s]32768it [00:00, 581909.82it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 478660.11it/s]1654784it [00:00, 11772502.40it/s]                         
0it [00:00, ?it/s]8192it [00:00, 180879.76it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2326031518>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2325ebc668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f233ee840d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f233ee840d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f233ee840d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:11,  2.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:11,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:10,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:09,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:08,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:08,  2.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:47,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:47,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:46,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:46,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:45,  3.19 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:32,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:31,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:30,  4.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:20,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:14,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:13,  8.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 11.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:09, 11.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 16.00 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 27.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.32 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 34.32 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 42.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 49.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 62.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 73.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 73.28 MiB/s][A

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

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.38s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.87 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.38s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.87 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.31 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.59s/ url]
0 examples [00:00, ? examples/s]2020-06-28 00:08:42.299420: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-28 00:08:42.313445: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-28 00:08:42.313637: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559b75b19f20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-28 00:08:42.313672: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
14 examples [00:00, 138.96 examples/s]103 examples [00:00, 186.05 examples/s]199 examples [00:00, 245.32 examples/s]292 examples [00:00, 314.59 examples/s]386 examples [00:00, 392.77 examples/s]479 examples [00:00, 474.98 examples/s]573 examples [00:00, 557.58 examples/s]671 examples [00:00, 639.78 examples/s]765 examples [00:00, 706.21 examples/s]858 examples [00:01, 760.10 examples/s]954 examples [00:01, 809.23 examples/s]1046 examples [00:01, 814.93 examples/s]1135 examples [00:01, 821.40 examples/s]1224 examples [00:01, 839.11 examples/s]1324 examples [00:01, 879.52 examples/s]1424 examples [00:01, 911.07 examples/s]1518 examples [00:01, 917.58 examples/s]1617 examples [00:01, 935.96 examples/s]1718 examples [00:01, 955.47 examples/s]1822 examples [00:02, 979.13 examples/s]1926 examples [00:02, 994.41 examples/s]2027 examples [00:02, 977.47 examples/s]2126 examples [00:02, 968.18 examples/s]2224 examples [00:02, 956.85 examples/s]2320 examples [00:02, 956.89 examples/s]2416 examples [00:02, 951.26 examples/s]2512 examples [00:02, 952.01 examples/s]2611 examples [00:02, 960.45 examples/s]2708 examples [00:02, 953.75 examples/s]2804 examples [00:03, 942.70 examples/s]2902 examples [00:03, 951.31 examples/s]2998 examples [00:03, 940.19 examples/s]3097 examples [00:03, 951.86 examples/s]3195 examples [00:03, 959.45 examples/s]3292 examples [00:03, 962.06 examples/s]3389 examples [00:03, 954.85 examples/s]3485 examples [00:03, 940.03 examples/s]3584 examples [00:03, 952.36 examples/s]3685 examples [00:03, 967.54 examples/s]3782 examples [00:04, 943.52 examples/s]3885 examples [00:04, 967.84 examples/s]3983 examples [00:04, 962.35 examples/s]4080 examples [00:04, 961.81 examples/s]4178 examples [00:04, 966.52 examples/s]4280 examples [00:04, 980.30 examples/s]4379 examples [00:04, 901.74 examples/s]4471 examples [00:04, 897.74 examples/s]4570 examples [00:04, 921.36 examples/s]4664 examples [00:05, 924.95 examples/s]4758 examples [00:05, 927.86 examples/s]4852 examples [00:05, 901.03 examples/s]4943 examples [00:05, 885.45 examples/s]5032 examples [00:05, 861.46 examples/s]5120 examples [00:05, 865.91 examples/s]5216 examples [00:05, 890.63 examples/s]5316 examples [00:05, 919.35 examples/s]5411 examples [00:05, 926.84 examples/s]5515 examples [00:05, 956.75 examples/s]5612 examples [00:06, 960.29 examples/s]5715 examples [00:06, 978.81 examples/s]5818 examples [00:06, 991.24 examples/s]5918 examples [00:06, 977.28 examples/s]6016 examples [00:06, 950.91 examples/s]6112 examples [00:06, 929.72 examples/s]6207 examples [00:06, 934.43 examples/s]6308 examples [00:06, 954.10 examples/s]6404 examples [00:06, 925.15 examples/s]6500 examples [00:06, 935.32 examples/s]6594 examples [00:07, 936.39 examples/s]6688 examples [00:07, 915.88 examples/s]6780 examples [00:07, 916.69 examples/s]6872 examples [00:07, 886.75 examples/s]6973 examples [00:07, 917.99 examples/s]7072 examples [00:07, 937.14 examples/s]7172 examples [00:07, 953.58 examples/s]7269 examples [00:07, 957.48 examples/s]7366 examples [00:07, 952.25 examples/s]7462 examples [00:08, 949.67 examples/s]7558 examples [00:08, 946.84 examples/s]7655 examples [00:08, 950.14 examples/s]7757 examples [00:08, 969.17 examples/s]7855 examples [00:08, 961.68 examples/s]7953 examples [00:08, 966.58 examples/s]8051 examples [00:08, 969.25 examples/s]8148 examples [00:08, 963.74 examples/s]8245 examples [00:08, 964.82 examples/s]8342 examples [00:08, 949.15 examples/s]8442 examples [00:09, 961.52 examples/s]8539 examples [00:09, 954.33 examples/s]8635 examples [00:09, 954.07 examples/s]8734 examples [00:09, 963.53 examples/s]8831 examples [00:09, 940.93 examples/s]8928 examples [00:09, 946.19 examples/s]9023 examples [00:09, 925.78 examples/s]9116 examples [00:09, 898.98 examples/s]9211 examples [00:09, 911.36 examples/s]9308 examples [00:09, 926.21 examples/s]9407 examples [00:10, 943.15 examples/s]9504 examples [00:10, 948.38 examples/s]9600 examples [00:10, 915.60 examples/s]9692 examples [00:10, 909.03 examples/s]9784 examples [00:10, 909.33 examples/s]9887 examples [00:10, 941.33 examples/s]9982 examples [00:10, 940.48 examples/s]10077 examples [00:10, 889.84 examples/s]10172 examples [00:10, 905.26 examples/s]10273 examples [00:11, 932.18 examples/s]10375 examples [00:11, 956.74 examples/s]10474 examples [00:11, 963.66 examples/s]10571 examples [00:11, 959.40 examples/s]10668 examples [00:11, 918.25 examples/s]10765 examples [00:11, 932.03 examples/s]10862 examples [00:11, 941.94 examples/s]10957 examples [00:11, 927.48 examples/s]11053 examples [00:11, 934.98 examples/s]11147 examples [00:11, 931.11 examples/s]11246 examples [00:12, 946.73 examples/s]11343 examples [00:12, 951.48 examples/s]11443 examples [00:12, 964.18 examples/s]11540 examples [00:12, 954.35 examples/s]11636 examples [00:12, 946.66 examples/s]11731 examples [00:12, 941.01 examples/s]11826 examples [00:12, 930.35 examples/s]11920 examples [00:12, 924.43 examples/s]12016 examples [00:12, 923.94 examples/s]12109 examples [00:12, 917.25 examples/s]12207 examples [00:13, 932.90 examples/s]12302 examples [00:13, 935.75 examples/s]12396 examples [00:13, 917.74 examples/s]12489 examples [00:13, 921.27 examples/s]12582 examples [00:13, 913.28 examples/s]12681 examples [00:13, 934.50 examples/s]12780 examples [00:13, 949.87 examples/s]12876 examples [00:13, 940.64 examples/s]12971 examples [00:13, 923.51 examples/s]13064 examples [00:13, 913.46 examples/s]13156 examples [00:14, 898.72 examples/s]13252 examples [00:14, 914.41 examples/s]13349 examples [00:14, 929.31 examples/s]13446 examples [00:14, 939.51 examples/s]13541 examples [00:14, 915.25 examples/s]13634 examples [00:14, 918.74 examples/s]13728 examples [00:14, 923.91 examples/s]13821 examples [00:14, 916.39 examples/s]13914 examples [00:14, 919.52 examples/s]14008 examples [00:15, 925.38 examples/s]14105 examples [00:15, 936.82 examples/s]14201 examples [00:15, 941.23 examples/s]14296 examples [00:15, 935.17 examples/s]14390 examples [00:15, 919.14 examples/s]14483 examples [00:15, 889.76 examples/s]14574 examples [00:15, 894.49 examples/s]14666 examples [00:15, 899.75 examples/s]14757 examples [00:15, 892.83 examples/s]14852 examples [00:15, 908.58 examples/s]14944 examples [00:16, 905.39 examples/s]15035 examples [00:16, 902.00 examples/s]15134 examples [00:16, 926.58 examples/s]15227 examples [00:16, 913.60 examples/s]15322 examples [00:16, 922.24 examples/s]15415 examples [00:16, 923.99 examples/s]15514 examples [00:16, 942.14 examples/s]15616 examples [00:16, 964.19 examples/s]15713 examples [00:16, 954.63 examples/s]15809 examples [00:16, 955.93 examples/s]15905 examples [00:17, 921.87 examples/s]16002 examples [00:17, 933.90 examples/s]16096 examples [00:17, 926.84 examples/s]16190 examples [00:17, 929.08 examples/s]16288 examples [00:17, 943.48 examples/s]16384 examples [00:17, 945.10 examples/s]16481 examples [00:17, 952.04 examples/s]16577 examples [00:17, 950.40 examples/s]16673 examples [00:17, 946.76 examples/s]16768 examples [00:17, 925.91 examples/s]16870 examples [00:18, 950.47 examples/s]16967 examples [00:18, 955.26 examples/s]17063 examples [00:18, 945.27 examples/s]17161 examples [00:18, 954.42 examples/s]17258 examples [00:18, 958.66 examples/s]17361 examples [00:18, 976.77 examples/s]17459 examples [00:18, 973.76 examples/s]17558 examples [00:18, 975.80 examples/s]17656 examples [00:18, 961.88 examples/s]17753 examples [00:19, 954.03 examples/s]17849 examples [00:19, 934.57 examples/s]17943 examples [00:19, 931.08 examples/s]18041 examples [00:19, 943.23 examples/s]18136 examples [00:19, 906.90 examples/s]18228 examples [00:19, 872.82 examples/s]18324 examples [00:19, 895.19 examples/s]18424 examples [00:19, 922.28 examples/s]18517 examples [00:19, 873.74 examples/s]18606 examples [00:19, 877.06 examples/s]18695 examples [00:20, 870.30 examples/s]18783 examples [00:20, 870.07 examples/s]18871 examples [00:20, 871.78 examples/s]18964 examples [00:20, 885.39 examples/s]19054 examples [00:20, 888.57 examples/s]19144 examples [00:20, 873.76 examples/s]19236 examples [00:20, 884.70 examples/s]19325 examples [00:20, 879.63 examples/s]19427 examples [00:20, 916.65 examples/s]19520 examples [00:20, 919.83 examples/s]19613 examples [00:21, 909.48 examples/s]19705 examples [00:21, 912.12 examples/s]19797 examples [00:21, 895.25 examples/s]19889 examples [00:21, 900.81 examples/s]19980 examples [00:21, 885.46 examples/s]20069 examples [00:21, 832.68 examples/s]20169 examples [00:21, 874.47 examples/s]20268 examples [00:21, 905.36 examples/s]20369 examples [00:21, 934.00 examples/s]20464 examples [00:22, 917.49 examples/s]20557 examples [00:22, 901.06 examples/s]20659 examples [00:22, 933.12 examples/s]20763 examples [00:22, 960.41 examples/s]20860 examples [00:22, 934.50 examples/s]20960 examples [00:22, 950.99 examples/s]21056 examples [00:22, 946.73 examples/s]21152 examples [00:22, 950.31 examples/s]21252 examples [00:22, 960.61 examples/s]21349 examples [00:22, 958.24 examples/s]21445 examples [00:23, 958.22 examples/s]21541 examples [00:23, 942.12 examples/s]21636 examples [00:23, 930.19 examples/s]21732 examples [00:23, 915.23 examples/s]21824 examples [00:23, 905.95 examples/s]21917 examples [00:23, 912.61 examples/s]22013 examples [00:23, 925.07 examples/s]22107 examples [00:23, 927.24 examples/s]22200 examples [00:23, 918.48 examples/s]22294 examples [00:23, 922.75 examples/s]22389 examples [00:24, 929.44 examples/s]22483 examples [00:24, 931.37 examples/s]22580 examples [00:24, 941.55 examples/s]22675 examples [00:24, 931.06 examples/s]22772 examples [00:24, 940.80 examples/s]22869 examples [00:24, 948.78 examples/s]22964 examples [00:24, 938.14 examples/s]23062 examples [00:24, 941.04 examples/s]23157 examples [00:24, 914.71 examples/s]23249 examples [00:25, 903.30 examples/s]23342 examples [00:25, 909.48 examples/s]23435 examples [00:25, 914.84 examples/s]23529 examples [00:25, 922.11 examples/s]23627 examples [00:25, 937.94 examples/s]23721 examples [00:25, 931.36 examples/s]23815 examples [00:25, 899.29 examples/s]23910 examples [00:25, 912.59 examples/s]24007 examples [00:25, 927.33 examples/s]24107 examples [00:25, 945.70 examples/s]24207 examples [00:26, 958.72 examples/s]24307 examples [00:26, 968.15 examples/s]24407 examples [00:26, 975.82 examples/s]24509 examples [00:26, 986.94 examples/s]24610 examples [00:26, 992.78 examples/s]24710 examples [00:26, 985.10 examples/s]24809 examples [00:26, 970.40 examples/s]24907 examples [00:26, 960.89 examples/s]25007 examples [00:26, 969.07 examples/s]25104 examples [00:26, 948.96 examples/s]25200 examples [00:27, 949.61 examples/s]25296 examples [00:27, 944.04 examples/s]25391 examples [00:27, 943.26 examples/s]25486 examples [00:27, 913.46 examples/s]25585 examples [00:27, 934.47 examples/s]25687 examples [00:27, 956.45 examples/s]25783 examples [00:27, 957.19 examples/s]25884 examples [00:27, 969.99 examples/s]25984 examples [00:27, 977.18 examples/s]26086 examples [00:27, 988.64 examples/s]26186 examples [00:28, 988.30 examples/s]26285 examples [00:28, 967.14 examples/s]26382 examples [00:28, 916.30 examples/s]26481 examples [00:28, 936.96 examples/s]26577 examples [00:28, 942.97 examples/s]26678 examples [00:28, 960.20 examples/s]26775 examples [00:28, 956.17 examples/s]26871 examples [00:28, 952.83 examples/s]26970 examples [00:28, 961.55 examples/s]27067 examples [00:29, 960.15 examples/s]27168 examples [00:29, 973.06 examples/s]27266 examples [00:29, 965.78 examples/s]27365 examples [00:29, 970.61 examples/s]27463 examples [00:29, 966.02 examples/s]27561 examples [00:29, 967.51 examples/s]27659 examples [00:29, 968.51 examples/s]27756 examples [00:29, 947.68 examples/s]27851 examples [00:29, 944.07 examples/s]27946 examples [00:29, 904.70 examples/s]28037 examples [00:30, 896.66 examples/s]28131 examples [00:30, 908.87 examples/s]28223 examples [00:30, 911.87 examples/s]28315 examples [00:30, 908.71 examples/s]28406 examples [00:30, 902.84 examples/s]28506 examples [00:30, 927.74 examples/s]28600 examples [00:30, 923.59 examples/s]28693 examples [00:30, 919.88 examples/s]28786 examples [00:30, 920.16 examples/s]28880 examples [00:30, 923.93 examples/s]28984 examples [00:31, 955.12 examples/s]29082 examples [00:31, 960.70 examples/s]29179 examples [00:31, 948.51 examples/s]29275 examples [00:31, 942.23 examples/s]29371 examples [00:31, 947.47 examples/s]29466 examples [00:31, 942.40 examples/s]29561 examples [00:31, 914.23 examples/s]29653 examples [00:31, 913.93 examples/s]29745 examples [00:31, 887.09 examples/s]29845 examples [00:32, 917.24 examples/s]29938 examples [00:32, 916.73 examples/s]30030 examples [00:32, 868.17 examples/s]30130 examples [00:32, 901.97 examples/s]30222 examples [00:32, 902.80 examples/s]30320 examples [00:32, 923.61 examples/s]30417 examples [00:32, 934.27 examples/s]30515 examples [00:32, 945.78 examples/s]30616 examples [00:32, 962.12 examples/s]30718 examples [00:32, 977.09 examples/s]30818 examples [00:33, 982.95 examples/s]30920 examples [00:33, 991.93 examples/s]31020 examples [00:33, 968.13 examples/s]31118 examples [00:33, 958.88 examples/s]31215 examples [00:33, 954.69 examples/s]31311 examples [00:33, 944.59 examples/s]31410 examples [00:33, 955.82 examples/s]31506 examples [00:33, 950.59 examples/s]31602 examples [00:33, 930.25 examples/s]31697 examples [00:33, 935.95 examples/s]31794 examples [00:34, 944.34 examples/s]31889 examples [00:34, 936.22 examples/s]31983 examples [00:34, 918.70 examples/s]32076 examples [00:34, 906.37 examples/s]32167 examples [00:34, 904.94 examples/s]32265 examples [00:34, 925.37 examples/s]32358 examples [00:34, 917.80 examples/s]32450 examples [00:34, 906.23 examples/s]32547 examples [00:34, 923.01 examples/s]32642 examples [00:34, 929.96 examples/s]32736 examples [00:35, 930.40 examples/s]32835 examples [00:35, 946.43 examples/s]32930 examples [00:35, 938.56 examples/s]33026 examples [00:35, 944.57 examples/s]33121 examples [00:35, 946.17 examples/s]33216 examples [00:35, 929.63 examples/s]33311 examples [00:35, 935.44 examples/s]33405 examples [00:35, 934.20 examples/s]33501 examples [00:35, 939.21 examples/s]33595 examples [00:36, 938.59 examples/s]33691 examples [00:36, 944.21 examples/s]33788 examples [00:36, 938.49 examples/s]33882 examples [00:36, 936.57 examples/s]33978 examples [00:36, 942.94 examples/s]34073 examples [00:36, 944.41 examples/s]34168 examples [00:36, 929.77 examples/s]34262 examples [00:36, 921.94 examples/s]34355 examples [00:36, 904.39 examples/s]34446 examples [00:36, 889.82 examples/s]34536 examples [00:37, 863.02 examples/s]34632 examples [00:37, 888.40 examples/s]34731 examples [00:37, 914.91 examples/s]34825 examples [00:37, 920.57 examples/s]34918 examples [00:37, 919.14 examples/s]35014 examples [00:37, 929.86 examples/s]35108 examples [00:37, 932.80 examples/s]35202 examples [00:37, 906.18 examples/s]35298 examples [00:37, 917.97 examples/s]35391 examples [00:37, 911.63 examples/s]35483 examples [00:38, 912.99 examples/s]35578 examples [00:38, 921.81 examples/s]35671 examples [00:38, 916.17 examples/s]35763 examples [00:38, 895.45 examples/s]35853 examples [00:38, 896.70 examples/s]35944 examples [00:38, 898.92 examples/s]36041 examples [00:38, 917.51 examples/s]36139 examples [00:38, 935.19 examples/s]36235 examples [00:38, 942.11 examples/s]36332 examples [00:38, 950.13 examples/s]36432 examples [00:39, 963.23 examples/s]36533 examples [00:39, 976.77 examples/s]36631 examples [00:39, 937.48 examples/s]36727 examples [00:39, 940.80 examples/s]36823 examples [00:39, 944.37 examples/s]36922 examples [00:39, 955.34 examples/s]37023 examples [00:39, 968.98 examples/s]37121 examples [00:39, 967.88 examples/s]37218 examples [00:39, 961.88 examples/s]37321 examples [00:40, 978.95 examples/s]37420 examples [00:40, 969.08 examples/s]37519 examples [00:40, 974.52 examples/s]37622 examples [00:40, 987.62 examples/s]37721 examples [00:40, 977.04 examples/s]37819 examples [00:40, 954.37 examples/s]37920 examples [00:40, 969.73 examples/s]38020 examples [00:40, 977.63 examples/s]38118 examples [00:40, 951.32 examples/s]38214 examples [00:40, 947.95 examples/s]38310 examples [00:41, 950.61 examples/s]38406 examples [00:41, 949.53 examples/s]38502 examples [00:41, 952.64 examples/s]38598 examples [00:41, 948.94 examples/s]38693 examples [00:41, 934.25 examples/s]38787 examples [00:41, 934.98 examples/s]38881 examples [00:41, 934.45 examples/s]38975 examples [00:41, 927.69 examples/s]39076 examples [00:41, 949.13 examples/s]39172 examples [00:41, 940.35 examples/s]39270 examples [00:42, 949.52 examples/s]39369 examples [00:42, 960.69 examples/s]39471 examples [00:42, 976.60 examples/s]39575 examples [00:42, 993.40 examples/s]39675 examples [00:42, 965.30 examples/s]39777 examples [00:42, 978.71 examples/s]39876 examples [00:42, 981.85 examples/s]39975 examples [00:42, 969.28 examples/s]40073 examples [00:42, 898.19 examples/s]40169 examples [00:43, 914.30 examples/s]40269 examples [00:43, 938.08 examples/s]40366 examples [00:43, 944.97 examples/s]40464 examples [00:43, 954.85 examples/s]40560 examples [00:43, 942.86 examples/s]40655 examples [00:43, 926.52 examples/s]40753 examples [00:43, 941.46 examples/s]40853 examples [00:43, 957.47 examples/s]40954 examples [00:43, 971.64 examples/s]41052 examples [00:43, 927.70 examples/s]41147 examples [00:44, 931.82 examples/s]41241 examples [00:44, 886.72 examples/s]41341 examples [00:44, 916.55 examples/s]41442 examples [00:44, 941.44 examples/s]41537 examples [00:44, 933.30 examples/s]41634 examples [00:44, 941.70 examples/s]41734 examples [00:44, 956.76 examples/s]41830 examples [00:44, 934.19 examples/s]41929 examples [00:44, 948.42 examples/s]42025 examples [00:44, 946.07 examples/s]42120 examples [00:45, 944.42 examples/s]42215 examples [00:45, 925.72 examples/s]42309 examples [00:45, 928.28 examples/s]42404 examples [00:45, 934.29 examples/s]42498 examples [00:45, 932.14 examples/s]42594 examples [00:45, 936.83 examples/s]42695 examples [00:45, 956.95 examples/s]42791 examples [00:45, 952.11 examples/s]42887 examples [00:45, 950.05 examples/s]42983 examples [00:45, 952.70 examples/s]43079 examples [00:46, 917.55 examples/s]43172 examples [00:46, 908.46 examples/s]43264 examples [00:46, 871.88 examples/s]43364 examples [00:46, 905.50 examples/s]43464 examples [00:46, 931.86 examples/s]43568 examples [00:46, 960.76 examples/s]43665 examples [00:46, 963.44 examples/s]43764 examples [00:46, 969.93 examples/s]43862 examples [00:46, 964.44 examples/s]43959 examples [00:47, 937.86 examples/s]44054 examples [00:47, 883.71 examples/s]44153 examples [00:47, 909.75 examples/s]44252 examples [00:47, 930.36 examples/s]44355 examples [00:47, 956.76 examples/s]44455 examples [00:47, 967.59 examples/s]44559 examples [00:47, 986.89 examples/s]44661 examples [00:47, 995.47 examples/s]44762 examples [00:47, 997.80 examples/s]44863 examples [00:47, 983.25 examples/s]44963 examples [00:48, 987.16 examples/s]45066 examples [00:48, 998.22 examples/s]45168 examples [00:48, 1003.57 examples/s]45269 examples [00:48, 988.84 examples/s] 45369 examples [00:48, 973.19 examples/s]45467 examples [00:48, 967.92 examples/s]45573 examples [00:48, 993.81 examples/s]45675 examples [00:48, 1000.89 examples/s]45776 examples [00:48, 986.58 examples/s] 45875 examples [00:48, 956.56 examples/s]45972 examples [00:49, 960.10 examples/s]46079 examples [00:49, 989.74 examples/s]46183 examples [00:49, 1001.74 examples/s]46284 examples [00:49, 997.65 examples/s] 46386 examples [00:49, 1004.25 examples/s]46487 examples [00:49, 997.80 examples/s] 46587 examples [00:49, 961.24 examples/s]46685 examples [00:49, 963.92 examples/s]46782 examples [00:49, 942.25 examples/s]46877 examples [00:50, 919.34 examples/s]46970 examples [00:50, 918.12 examples/s]47063 examples [00:50, 917.09 examples/s]47170 examples [00:50, 955.41 examples/s]47270 examples [00:50, 967.58 examples/s]47369 examples [00:50, 971.78 examples/s]47467 examples [00:50, 962.65 examples/s]47567 examples [00:50, 971.15 examples/s]47673 examples [00:50, 994.60 examples/s]47775 examples [00:50, 1001.26 examples/s]47882 examples [00:51, 1020.81 examples/s]47985 examples [00:51, 1005.60 examples/s]48086 examples [00:51, 1003.58 examples/s]48187 examples [00:51, 971.46 examples/s] 48287 examples [00:51, 977.59 examples/s]48391 examples [00:51, 992.44 examples/s]48491 examples [00:51, 989.14 examples/s]48591 examples [00:51, 976.49 examples/s]48689 examples [00:51, 969.34 examples/s]48787 examples [00:51, 959.84 examples/s]48885 examples [00:52, 965.41 examples/s]48982 examples [00:52, 931.58 examples/s]49083 examples [00:52, 953.04 examples/s]49181 examples [00:52, 958.59 examples/s]49283 examples [00:52, 974.86 examples/s]49392 examples [00:52, 1006.34 examples/s]49494 examples [00:52, 965.33 examples/s] 49601 examples [00:52, 992.43 examples/s]49704 examples [00:52, 1001.88 examples/s]49805 examples [00:53, 961.47 examples/s] 49902 examples [00:53, 950.70 examples/s]49998 examples [00:53, 939.08 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6775/50000 [00:00<00:00, 67747.32 examples/s] 37%|███▋      | 18293/50000 [00:00<00:00, 77295.94 examples/s] 61%|██████    | 30453/50000 [00:00<00:00, 86780.12 examples/s] 87%|████████▋ | 43723/50000 [00:00<00:00, 96831.75 examples/s]                                                               0 examples [00:00, ? examples/s]82 examples [00:00, 812.67 examples/s]187 examples [00:00, 871.41 examples/s]284 examples [00:00, 896.47 examples/s]388 examples [00:00, 933.15 examples/s]493 examples [00:00, 964.39 examples/s]599 examples [00:00, 991.02 examples/s]703 examples [00:00, 1001.84 examples/s]806 examples [00:00, 1007.66 examples/s]916 examples [00:00, 1033.08 examples/s]1017 examples [00:01, 993.89 examples/s]1122 examples [00:01, 1007.82 examples/s]1223 examples [00:01, 1008.30 examples/s]1325 examples [00:01, 1010.00 examples/s]1426 examples [00:01, 1009.41 examples/s]1527 examples [00:01, 983.82 examples/s] 1626 examples [00:01, 978.66 examples/s]1726 examples [00:01, 984.19 examples/s]1826 examples [00:01, 986.13 examples/s]1925 examples [00:01, 984.12 examples/s]2030 examples [00:02, 1001.32 examples/s]2131 examples [00:02, 999.25 examples/s] 2231 examples [00:02, 976.25 examples/s]2331 examples [00:02, 983.09 examples/s]2430 examples [00:02, 970.27 examples/s]2528 examples [00:02, 961.06 examples/s]2625 examples [00:02, 951.65 examples/s]2731 examples [00:02, 979.95 examples/s]2830 examples [00:02, 976.10 examples/s]2928 examples [00:02, 975.18 examples/s]3026 examples [00:03, 961.31 examples/s]3123 examples [00:03, 959.55 examples/s]3223 examples [00:03, 970.60 examples/s]3327 examples [00:03, 989.56 examples/s]3432 examples [00:03, 1005.23 examples/s]3534 examples [00:03, 1008.04 examples/s]3639 examples [00:03, 1017.75 examples/s]3741 examples [00:03, 1018.15 examples/s]3848 examples [00:03, 1032.09 examples/s]3955 examples [00:03, 1041.39 examples/s]4060 examples [00:04, 1030.76 examples/s]4164 examples [00:04, 1025.93 examples/s]4268 examples [00:04, 1027.71 examples/s]4374 examples [00:04, 1034.43 examples/s]4478 examples [00:04, 1030.15 examples/s]4582 examples [00:04, 1000.30 examples/s]4683 examples [00:04, 1000.16 examples/s]4784 examples [00:04, 998.91 examples/s] 4887 examples [00:04, 1005.81 examples/s]4994 examples [00:04, 1022.69 examples/s]5097 examples [00:05, 1015.08 examples/s]5202 examples [00:05, 1023.55 examples/s]5305 examples [00:05, 1023.99 examples/s]5411 examples [00:05, 1033.90 examples/s]5515 examples [00:05, 1016.60 examples/s]5617 examples [00:05, 998.99 examples/s] 5718 examples [00:05, 978.77 examples/s]5817 examples [00:05, 936.33 examples/s]5912 examples [00:05, 909.78 examples/s]6004 examples [00:06, 902.64 examples/s]6096 examples [00:06, 905.92 examples/s]6194 examples [00:06, 923.63 examples/s]6294 examples [00:06, 943.82 examples/s]6389 examples [00:06, 941.67 examples/s]6491 examples [00:06, 963.84 examples/s]6588 examples [00:06, 933.44 examples/s]6684 examples [00:06, 939.05 examples/s]6788 examples [00:06, 965.86 examples/s]6891 examples [00:06, 982.01 examples/s]6990 examples [00:07, 974.95 examples/s]7089 examples [00:07, 978.92 examples/s]7188 examples [00:07, 965.26 examples/s]7289 examples [00:07, 978.04 examples/s]7394 examples [00:07, 998.29 examples/s]7495 examples [00:07, 986.84 examples/s]7594 examples [00:07, 971.27 examples/s]7692 examples [00:07, 972.81 examples/s]7793 examples [00:07, 981.97 examples/s]7892 examples [00:07, 969.47 examples/s]7990 examples [00:08, 964.13 examples/s]8087 examples [00:08, 964.79 examples/s]8184 examples [00:08, 964.52 examples/s]8284 examples [00:08, 974.43 examples/s]8382 examples [00:08, 971.12 examples/s]8480 examples [00:08, 962.61 examples/s]8577 examples [00:08, 959.37 examples/s]8674 examples [00:08, 952.84 examples/s]8775 examples [00:08, 967.81 examples/s]8872 examples [00:09, 950.24 examples/s]8968 examples [00:09, 937.95 examples/s]9071 examples [00:09, 961.68 examples/s]9173 examples [00:09, 976.93 examples/s]9278 examples [00:09, 995.03 examples/s]9378 examples [00:09, 976.17 examples/s]9481 examples [00:09, 991.36 examples/s]9581 examples [00:09, 957.18 examples/s]9678 examples [00:09, 927.15 examples/s]9779 examples [00:09, 947.29 examples/s]9880 examples [00:10, 962.93 examples/s]9978 examples [00:10, 966.70 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM86RU3/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteM86RU3/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f233ee84488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f233ee84488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f233ee84488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2326031518>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f22c831bba8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f233ee84488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f233ee84488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f233ee84488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f22c831ba58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2325ebc0b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f22c00351e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f22c00351e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f22c00351e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f22c00352f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f22c00352f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f22c00352f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=92212ccbf06ec1f883b1e535104ad2d03890915a21164b60a755b1f59c7c5d49
  Stored in directory: /tmp/pip-ephem-wheel-cache-szjp4zi0/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:23:44, 10.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:36:26, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:40:40, 20.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:10:55, 29.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.64M/862M [00:01<5:42:44, 41.7kB/s].vector_cache/glove.6B.zip:   1%|          | 7.94M/862M [00:01<3:58:49, 59.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:46:20, 85.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.7M/862M [00:01<1:55:43, 121kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.6M/862M [00:01<1:20:30, 173kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.5M/862M [00:01<56:18, 247kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:02<39:14, 352kB/s].vector_cache/glove.6B.zip:   5%|▍         | 38.9M/862M [00:02<27:21, 502kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.8M/862M [00:02<19:13, 711kB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:02<13:26, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.3M/862M [00:02<09:30, 1.42MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<07:20, 1.84MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<07:02, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<06:48, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:04<05:08, 2.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:04<03:44, 3.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:06<2:42:26, 82.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<1:56:56, 114kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<1:22:34, 162kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:07<57:48, 230kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<50:03, 266kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<36:25, 365kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:08<25:44, 515kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:10<21:01, 629kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<17:22, 761kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<12:49, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:10<09:05, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<12:48:45, 17.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<8:59:13, 24.4kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:12<6:17:01, 34.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<4:26:15, 49.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<3:07:36, 69.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<2:11:23, 99.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<1:34:49, 137kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<1:08:59, 189kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<48:47, 267kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 84.0M/862M [00:16<34:13, 379kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<32:11, 403kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<23:51, 543kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<16:59, 760kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<14:53, 865kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<13:02, 989kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<09:39, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.8M/862M [00:20<06:56, 1.85MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<10:08, 1.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<08:23, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.0M/862M [00:22<06:11, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:19, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<07:41, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<05:55, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<04:18, 2.95MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<10:06, 1.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:21, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:06, 2.07MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:14, 1.74MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:07, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:36, 2.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<03:24, 3.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<55:45, 225kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<41:33, 302kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<29:36, 423kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<20:48, 600kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<22:40, 550kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<17:07, 728kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<12:16, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:29, 1.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:32, 1.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:00, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<05:44, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:44:58, 118kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<1:14:41, 165kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<52:29, 234kB/s]  .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<39:30, 311kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<30:07, 407kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<21:35, 568kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<15:13, 802kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<18:05, 674kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<13:55, 876kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<10:02, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<09:51, 1.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<09:20, 1.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:03, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<05:05, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<09:19, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:44, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<05:43, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<06:48, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:58, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<04:28, 2.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:57, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:22, 2.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:03, 2.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:38, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:08, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:53, 3.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:31, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:02, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:48, 3.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:26, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:10, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<04:49, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<03:30, 3.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<09:10, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:35, 1.53MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<05:33, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:37, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:58, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:27, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:40, 2.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:09, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:51, 2.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:23, 2.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:54, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:40, 3.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<05:16, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:50, 2.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<03:37, 3.13MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:14, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:56, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:43, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:07, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<04:43, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<03:35, 3.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:07, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:50, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:33, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:19, 3.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<08:35, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:09, 1.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:14, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:15, 1.76MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:36, 1.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:11, 2.13MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:23, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<04:43, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:34, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:04, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:40, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:25, 2.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:17<03:13, 3.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<07:56, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:40, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:56, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:58, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:21, 1.69MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:53, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<03:33, 3.01MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:01, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<07:23, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:25, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:17, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:32, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:01, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:38, 2.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<09:44, 1.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:52, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:43, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:27, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:38, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:05, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:40, 2.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<10:35, 985kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<08:28, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:08, 1.69MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:44, 1.54MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<06:48, 1.52MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:12, 1.98MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:45, 2.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<10:26, 985kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<08:20, 1.23MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:05, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:38, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:40, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:10, 2.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:20, 1.90MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:45, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:34, 2.82MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:53, 2.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:27, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<04:14, 2.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:04, 3.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<10:52, 920kB/s] .vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:37, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<06:16, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:42, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:41, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<05:11, 1.91MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:43, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:26, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:12, 3.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<08:55, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<07:04, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<05:11, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:55, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<06:06, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:45, 2.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:53, 1.98MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:23, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<03:16, 2.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:32, 2.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:07, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:00, 2.39MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:54<02:53, 3.29MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<11:55, 799kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<09:19, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<06:43, 1.41MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:54, 1.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:47, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<04:16, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:12, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:32, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:16, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:00<03:07, 2.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<06:01, 1.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:10, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:50, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:15, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:08, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:22, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<05:00, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:53, 2.36MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:06<02:50, 3.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<06:36, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:33, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:07, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:58, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:18, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<02:58, 3.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<07:33, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:12, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:33, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:16, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:34, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:22, 2.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:28, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:55, 1.80MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:53, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:07, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:47, 2.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:52, 3.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:01, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:34, 1.90MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:34, 2.43MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:36, 3.31MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:57, 1.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:44, 2.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:37, 1.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:58, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:54, 2.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:49, 3.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<8:12:39, 17.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<5:45:20, 24.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<4:01:13, 35.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<2:48:18, 50.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<2:18:44, 60.8kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<1:38:46, 85.3kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<1:09:27, 121kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<48:25, 173kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<8:37:07, 16.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<6:02:34, 23.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<4:13:13, 32.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<2:58:26, 46.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<2:06:32, 65.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<1:28:51, 93.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<1:03:08, 130kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<45:50, 179kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<32:24, 253kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<22:40, 360kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<21:20, 382kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<15:46, 517kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<11:10, 727kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<09:40, 836kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<08:24, 962kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<06:13, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:26, 1.81MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<07:00, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:43, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<04:11, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:47, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:10, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:06, 2.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:02, 1.95MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:25, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:25, 2.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:43<02:29, 3.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:57, 1.31MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:57, 1.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:39, 2.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:21, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:37, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:33, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:47<02:34, 2.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:23, 1.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:15, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<03:52, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:29, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:41, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:40, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:51<02:38, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<55:46, 135kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<39:46, 189kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<27:54, 269kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<21:12, 352kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<16:21, 457kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<11:48, 632kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<09:24, 787kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<07:20, 1.01MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<05:18, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:24, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:31, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:19, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:02, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:17, 1.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:22, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:30, 2.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:10, 2.26MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:24, 2.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:21, 2.13MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:05<02:57, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:08, 3.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<07:13, 978kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:46, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<04:12, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:34, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:48, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:49, 2.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:36, 1.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:13, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:25, 2.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:18, 2.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:42, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:56, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:13<02:07, 3.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<6:33:33, 17.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<4:35:55, 24.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<3:12:35, 35.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<2:15:39, 49.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<1:36:16, 69.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<1:07:32, 99.2kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<47:09, 141kB/s]   .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<35:34, 187kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<25:33, 260kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<17:59, 368kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<14:04, 468kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<10:30, 626kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<07:27, 877kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<06:45, 964kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<06:02, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:30, 1.44MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<03:13, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:19, 1.21MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:23, 1.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:12, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:43, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:15, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<02:25, 2.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:11, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:51, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:09, 2.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:58, 2.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:21, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:39, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:14, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:34, 2.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<01:51, 3.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<36:39, 166kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<26:14, 232kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<18:26, 329kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<14:15, 423kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<10:34, 570kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<07:30, 800kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<06:38, 897kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:51, 1.02MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<04:20, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<03:05, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<06:02, 975kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:50, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:29, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:48, 1.53MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:50, 1.52MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:58, 1.95MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<02:59, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<03:15, 1.76MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:31, 2.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:50, 3.11MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:11, 1.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:30, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:35, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:07, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:16, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:33, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:40, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:26, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:50, 2.99MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:34, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:54, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:16, 2.41MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<01:38, 3.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:38, 1.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:47, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:47, 1.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<03:10, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:18, 1.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:34, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:38, 2.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<02:55, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:17, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:38, 3.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<13:56, 374kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<10:16, 506kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<07:17, 710kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<06:16, 820kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:25, 947kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:00, 1.28MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:52, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:50, 1.32MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:12, 1.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:21, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:59, 1.67MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:21, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<02:25, 2.03MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:12, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:39, 2.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:17, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:35, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:03, 2.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:29, 3.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<40:16, 119kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<28:38, 167kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<20:02, 238kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<15:02, 314kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<11:28, 412kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<08:12, 574kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<05:44, 812kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<08:46, 531kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<06:36, 704kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<04:43, 980kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<04:21, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<03:58, 1.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:00, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:49, 1.61MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:26, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<01:48, 2.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:17, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:30, 1.78MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<01:58, 2.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:04, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<01:53, 2.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:25, 3.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:00, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:14, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:46, 2.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:55, 2.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<01:46, 2.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:20, 3.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:54, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:10, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:43, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:14, 3.30MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<30:27, 135kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<21:41, 189kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<15:11, 268kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<11:29, 352kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<08:51, 456kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<06:21, 634kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<04:27, 895kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<05:26, 730kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:12, 944kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<03:01, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:00, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:53, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:11, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:34, 2.45MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:57, 1.30MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:27, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:47, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:08, 1.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:15, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:14, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:28, 1.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:48, 1.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:02, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:15, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:18, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:47, 2.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:49, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:37, 2.18MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:13, 2.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:40, 2.10MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:52, 1.86MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:27, 2.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:03, 3.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:48, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:18, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:41, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:57, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:42, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:16, 2.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:39, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:49, 1.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:26, 2.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:30, 2.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:23, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:02, 3.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:27, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:39, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:17, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<00:55, 3.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:32, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:04, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:31, 2.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:45, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:32, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:07, 2.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:28, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:37, 1.81MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:16, 2.29MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:05<00:54, 3.15MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<2:46:49, 17.2kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<1:56:48, 24.6kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<1:21:05, 35.1kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<56:42, 49.5kB/s]  .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<39:53, 70.3kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<27:43, 100kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<19:48, 138kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<14:24, 190kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<10:10, 268kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<07:25, 360kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<05:26, 489kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<03:50, 688kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:15, 798kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<02:48, 926kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<02:04, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:27, 1.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<19:15, 132kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<13:42, 184kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<09:33, 262kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<07:10, 344kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<05:15, 468kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<03:41, 659kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:07, 768kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:39, 898kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:58, 1.21MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:20<01:22, 1.69MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<2:15:47, 17.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<1:35:02, 24.4kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<1:05:50, 34.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<45:53, 49.2kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<32:33, 69.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<22:46, 98.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:24<15:36, 140kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<26:36, 82.4kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<18:47, 116kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<13:03, 165kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<09:29, 224kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<06:49, 310kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<04:46, 438kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:46, 544kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<03:02, 675kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<02:12, 920kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:50, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:42, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:16, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:53, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:43, 1.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:23, 1.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:00, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:07, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:09, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:53, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:36<00:37, 2.82MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<1:43:25, 17.2kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<1:12:18, 24.5kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<49:55, 35.0kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<34:37, 49.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<24:19, 70.1kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<16:48, 99.9kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<11:54, 138kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<08:28, 193kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<05:52, 274kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<04:23, 358kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:13, 487kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:15, 683kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:54, 793kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:37, 921kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:12, 1.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<01:02, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.22MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:45, 1.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:48, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:36, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:26, 3.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:18, 990kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:02, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:45, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:48, 1.54MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:40, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:29, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:36, 1.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:32, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:23, 2.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:31, 2.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:28, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:32, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:17, 3.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<07:02, 136kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<04:59, 190kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<03:25, 270kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:30, 353kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:56, 456kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:22, 631kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:55, 891kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<06:30, 126kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<04:36, 177kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<03:08, 251kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<02:16, 331kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<01:44, 432kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:13, 601kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:49, 849kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<01:04, 635kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:48, 831kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:33, 1.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:31, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:29, 1.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:21, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:14, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:31, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:24, 1.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:17, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:18, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:09, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:24, 999kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:19, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:13, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.42MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:03, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<11:49, 17.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<08:02, 24.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<04:53, 34.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<02:43, 49.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<01:50, 70.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<01:02, 99.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:28, 138kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:19, 189kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:11, 266kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 379kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<17:04:58,  6.50it/s]  0%|          | 792/400000 [00:00<11:56:19,  9.29it/s]  0%|          | 1658/400000 [00:00<8:20:33, 13.26it/s]  1%|          | 2475/400000 [00:00<5:49:55, 18.93it/s]  1%|          | 3379/400000 [00:00<4:04:36, 27.02it/s]  1%|          | 4203/400000 [00:00<2:51:06, 38.55it/s]  1%|▏         | 5084/400000 [00:00<1:59:44, 54.97it/s]  1%|▏         | 5897/400000 [00:00<1:23:53, 78.30it/s]  2%|▏         | 6721/400000 [00:00<58:50, 111.41it/s]   2%|▏         | 7510/400000 [00:01<41:21, 158.14it/s]  2%|▏         | 8319/400000 [00:01<29:08, 224.03it/s]  2%|▏         | 9120/400000 [00:01<20:35, 316.26it/s]  2%|▏         | 9960/400000 [00:01<14:37, 444.62it/s]  3%|▎         | 10765/400000 [00:01<10:27, 619.95it/s]  3%|▎         | 11659/400000 [00:01<07:31, 860.05it/s]  3%|▎         | 12485/400000 [00:01<05:30, 1172.97it/s]  3%|▎         | 13307/400000 [00:01<04:04, 1579.09it/s]  4%|▎         | 14121/400000 [00:01<03:07, 2060.67it/s]  4%|▎         | 14945/400000 [00:01<02:24, 2658.84it/s]  4%|▍         | 15789/400000 [00:02<01:54, 3346.42it/s]  4%|▍         | 16600/400000 [00:02<01:34, 4053.71it/s]  4%|▍         | 17430/400000 [00:02<01:19, 4787.86it/s]  5%|▍         | 18244/400000 [00:02<01:10, 5443.40it/s]  5%|▍         | 19054/400000 [00:02<01:03, 5964.02it/s]  5%|▍         | 19852/400000 [00:02<00:59, 6428.96it/s]  5%|▌         | 20646/400000 [00:02<00:56, 6714.31it/s]  5%|▌         | 21427/400000 [00:02<00:54, 6980.91it/s]  6%|▌         | 22286/400000 [00:02<00:51, 7395.26it/s]  6%|▌         | 23138/400000 [00:03<00:48, 7698.66it/s]  6%|▌         | 24027/400000 [00:03<00:46, 8019.55it/s]  6%|▌         | 24865/400000 [00:03<00:47, 7944.04it/s]  6%|▋         | 25701/400000 [00:03<00:46, 8063.83it/s]  7%|▋         | 26544/400000 [00:03<00:45, 8168.36it/s]  7%|▋         | 27413/400000 [00:03<00:44, 8315.41it/s]  7%|▋         | 28255/400000 [00:03<00:45, 8260.67it/s]  7%|▋         | 29089/400000 [00:03<00:44, 8282.51it/s]  8%|▊         | 30019/400000 [00:03<00:43, 8562.33it/s]  8%|▊         | 30882/400000 [00:03<00:43, 8577.75it/s]  8%|▊         | 31755/400000 [00:04<00:42, 8620.35it/s]  8%|▊         | 32643/400000 [00:04<00:42, 8695.91it/s]  8%|▊         | 33515/400000 [00:04<00:43, 8462.87it/s]  9%|▊         | 34387/400000 [00:04<00:42, 8536.57it/s]  9%|▉         | 35243/400000 [00:04<00:43, 8410.87it/s]  9%|▉         | 36119/400000 [00:04<00:42, 8510.97it/s]  9%|▉         | 36991/400000 [00:04<00:42, 8571.90it/s]  9%|▉         | 37850/400000 [00:04<00:42, 8477.59it/s] 10%|▉         | 38759/400000 [00:04<00:41, 8650.15it/s] 10%|▉         | 39627/400000 [00:04<00:41, 8657.78it/s] 10%|█         | 40494/400000 [00:05<00:41, 8645.74it/s] 10%|█         | 41360/400000 [00:05<00:42, 8396.96it/s] 11%|█         | 42258/400000 [00:05<00:41, 8562.61it/s] 11%|█         | 43176/400000 [00:05<00:40, 8738.87it/s] 11%|█         | 44112/400000 [00:05<00:39, 8916.22it/s] 11%|█▏        | 45007/400000 [00:05<00:40, 8706.85it/s] 11%|█▏        | 45881/400000 [00:05<00:41, 8589.34it/s] 12%|█▏        | 46755/400000 [00:05<00:40, 8633.09it/s] 12%|█▏        | 47621/400000 [00:05<00:41, 8485.94it/s] 12%|█▏        | 48502/400000 [00:05<00:40, 8579.00it/s] 12%|█▏        | 49362/400000 [00:06<00:41, 8538.45it/s] 13%|█▎        | 50217/400000 [00:06<00:41, 8354.22it/s] 13%|█▎        | 51055/400000 [00:06<00:42, 8209.29it/s] 13%|█▎        | 51878/400000 [00:06<00:42, 8135.78it/s] 13%|█▎        | 52733/400000 [00:06<00:42, 8254.65it/s] 13%|█▎        | 53588/400000 [00:06<00:41, 8338.17it/s] 14%|█▎        | 54423/400000 [00:06<00:41, 8304.81it/s] 14%|█▍        | 55268/400000 [00:06<00:41, 8346.75it/s] 14%|█▍        | 56180/400000 [00:06<00:40, 8563.57it/s] 14%|█▍        | 57078/400000 [00:06<00:39, 8681.57it/s] 14%|█▍        | 57965/400000 [00:07<00:39, 8736.89it/s] 15%|█▍        | 58840/400000 [00:07<00:39, 8737.97it/s] 15%|█▍        | 59737/400000 [00:07<00:38, 8803.64it/s] 15%|█▌        | 60681/400000 [00:07<00:37, 8983.70it/s] 15%|█▌        | 61585/400000 [00:07<00:37, 8998.40it/s] 16%|█▌        | 62486/400000 [00:07<00:37, 8987.84it/s] 16%|█▌        | 63386/400000 [00:07<00:38, 8682.10it/s] 16%|█▌        | 64257/400000 [00:07<00:39, 8563.99it/s] 16%|█▋        | 65156/400000 [00:07<00:38, 8686.24it/s] 17%|█▋        | 66081/400000 [00:08<00:37, 8845.23it/s] 17%|█▋        | 66991/400000 [00:08<00:37, 8918.93it/s] 17%|█▋        | 67885/400000 [00:08<00:38, 8655.73it/s] 17%|█▋        | 68754/400000 [00:08<00:38, 8543.52it/s] 17%|█▋        | 69611/400000 [00:08<00:39, 8412.18it/s] 18%|█▊        | 70455/400000 [00:08<00:39, 8350.53it/s] 18%|█▊        | 71292/400000 [00:08<00:39, 8243.24it/s] 18%|█▊        | 72118/400000 [00:08<00:40, 8118.20it/s] 18%|█▊        | 72986/400000 [00:08<00:39, 8277.11it/s] 18%|█▊        | 73856/400000 [00:08<00:38, 8396.90it/s] 19%|█▊        | 74755/400000 [00:09<00:37, 8564.49it/s] 19%|█▉        | 75663/400000 [00:09<00:37, 8712.76it/s] 19%|█▉        | 76537/400000 [00:09<00:37, 8631.63it/s] 19%|█▉        | 77402/400000 [00:09<00:37, 8629.86it/s] 20%|█▉        | 78271/400000 [00:09<00:37, 8646.42it/s] 20%|█▉        | 79137/400000 [00:09<00:37, 8611.44it/s] 20%|█▉        | 79999/400000 [00:09<00:37, 8475.01it/s] 20%|██        | 80848/400000 [00:09<00:37, 8468.18it/s] 20%|██        | 81696/400000 [00:09<00:37, 8424.97it/s] 21%|██        | 82557/400000 [00:09<00:37, 8477.21it/s] 21%|██        | 83406/400000 [00:10<00:37, 8377.96it/s] 21%|██        | 84248/400000 [00:10<00:37, 8389.85it/s] 21%|██▏       | 85088/400000 [00:10<00:37, 8298.02it/s] 21%|██▏       | 85987/400000 [00:10<00:36, 8494.07it/s] 22%|██▏       | 86838/400000 [00:10<00:37, 8256.44it/s] 22%|██▏       | 87719/400000 [00:10<00:37, 8414.14it/s] 22%|██▏       | 88563/400000 [00:10<00:37, 8394.14it/s] 22%|██▏       | 89426/400000 [00:10<00:36, 8462.76it/s] 23%|██▎       | 90334/400000 [00:10<00:35, 8637.12it/s] 23%|██▎       | 91231/400000 [00:10<00:35, 8733.51it/s] 23%|██▎       | 92106/400000 [00:11<00:35, 8684.49it/s] 23%|██▎       | 92976/400000 [00:11<00:36, 8483.56it/s] 23%|██▎       | 93846/400000 [00:11<00:35, 8544.89it/s] 24%|██▎       | 94735/400000 [00:11<00:35, 8643.30it/s] 24%|██▍       | 95601/400000 [00:11<00:35, 8570.47it/s] 24%|██▍       | 96509/400000 [00:11<00:34, 8715.32it/s] 24%|██▍       | 97382/400000 [00:11<00:36, 8385.89it/s] 25%|██▍       | 98225/400000 [00:11<00:38, 7893.80it/s] 25%|██▍       | 99023/400000 [00:11<00:38, 7853.17it/s] 25%|██▍       | 99850/400000 [00:12<00:37, 7973.64it/s] 25%|██▌       | 100757/400000 [00:12<00:36, 8272.90it/s] 25%|██▌       | 101591/400000 [00:12<00:36, 8097.17it/s] 26%|██▌       | 102482/400000 [00:12<00:35, 8322.64it/s] 26%|██▌       | 103393/400000 [00:12<00:34, 8543.90it/s] 26%|██▌       | 104318/400000 [00:12<00:33, 8742.99it/s] 26%|██▋       | 105232/400000 [00:12<00:33, 8856.74it/s] 27%|██▋       | 106122/400000 [00:12<00:33, 8709.22it/s] 27%|██▋       | 106997/400000 [00:12<00:33, 8656.46it/s] 27%|██▋       | 107929/400000 [00:12<00:33, 8845.09it/s] 27%|██▋       | 108853/400000 [00:13<00:32, 8958.45it/s] 27%|██▋       | 109790/400000 [00:13<00:31, 9076.24it/s] 28%|██▊       | 110700/400000 [00:13<00:32, 8921.15it/s] 28%|██▊       | 111595/400000 [00:13<00:32, 8817.56it/s] 28%|██▊       | 112525/400000 [00:13<00:32, 8953.60it/s] 28%|██▊       | 113422/400000 [00:13<00:32, 8854.53it/s] 29%|██▊       | 114309/400000 [00:13<00:33, 8516.15it/s] 29%|██▉       | 115165/400000 [00:13<00:33, 8443.60it/s] 29%|██▉       | 116013/400000 [00:13<00:33, 8361.30it/s] 29%|██▉       | 116907/400000 [00:13<00:33, 8526.86it/s] 29%|██▉       | 117762/400000 [00:14<00:33, 8460.45it/s] 30%|██▉       | 118628/400000 [00:14<00:33, 8518.29it/s] 30%|██▉       | 119482/400000 [00:14<00:33, 8379.17it/s] 30%|███       | 120322/400000 [00:14<00:33, 8228.42it/s] 30%|███       | 121147/400000 [00:14<00:33, 8210.27it/s] 31%|███       | 122054/400000 [00:14<00:32, 8449.89it/s] 31%|███       | 122946/400000 [00:14<00:32, 8583.22it/s] 31%|███       | 123807/400000 [00:14<00:32, 8472.66it/s] 31%|███       | 124727/400000 [00:14<00:31, 8676.08it/s] 31%|███▏      | 125635/400000 [00:14<00:31, 8792.74it/s] 32%|███▏      | 126517/400000 [00:15<00:31, 8579.15it/s] 32%|███▏      | 127393/400000 [00:15<00:31, 8632.00it/s] 32%|███▏      | 128259/400000 [00:15<00:31, 8621.71it/s] 32%|███▏      | 129176/400000 [00:15<00:30, 8778.89it/s] 33%|███▎      | 130121/400000 [00:15<00:30, 8968.34it/s] 33%|███▎      | 131070/400000 [00:15<00:29, 9116.26it/s] 33%|███▎      | 131984/400000 [00:15<00:29, 9049.44it/s] 33%|███▎      | 132891/400000 [00:15<00:29, 9016.47it/s] 33%|███▎      | 133794/400000 [00:15<00:29, 8891.08it/s] 34%|███▎      | 134698/400000 [00:16<00:29, 8930.81it/s] 34%|███▍      | 135592/400000 [00:16<00:29, 8819.63it/s] 34%|███▍      | 136482/400000 [00:16<00:29, 8842.13it/s] 34%|███▍      | 137367/400000 [00:16<00:29, 8808.88it/s] 35%|███▍      | 138281/400000 [00:16<00:29, 8901.26it/s] 35%|███▍      | 139189/400000 [00:16<00:29, 8952.52it/s] 35%|███▌      | 140085/400000 [00:16<00:29, 8946.29it/s] 35%|███▌      | 141050/400000 [00:16<00:28, 9143.73it/s] 35%|███▌      | 141966/400000 [00:16<00:29, 8895.76it/s] 36%|███▌      | 142858/400000 [00:16<00:28, 8876.03it/s] 36%|███▌      | 143760/400000 [00:17<00:28, 8918.01it/s] 36%|███▌      | 144654/400000 [00:17<00:28, 8921.85it/s] 36%|███▋      | 145599/400000 [00:17<00:28, 9071.55it/s] 37%|███▋      | 146508/400000 [00:17<00:28, 8907.19it/s] 37%|███▋      | 147420/400000 [00:17<00:28, 8968.20it/s] 37%|███▋      | 148350/400000 [00:17<00:27, 9064.29it/s] 37%|███▋      | 149258/400000 [00:17<00:28, 8877.40it/s] 38%|███▊      | 150148/400000 [00:17<00:28, 8855.75it/s] 38%|███▊      | 151035/400000 [00:17<00:28, 8662.68it/s] 38%|███▊      | 151988/400000 [00:17<00:27, 8905.01it/s] 38%|███▊      | 152882/400000 [00:18<00:28, 8825.50it/s] 38%|███▊      | 153767/400000 [00:18<00:29, 8476.57it/s] 39%|███▊      | 154683/400000 [00:18<00:28, 8669.12it/s] 39%|███▉      | 155557/400000 [00:18<00:28, 8687.89it/s] 39%|███▉      | 156468/400000 [00:18<00:27, 8809.87it/s] 39%|███▉      | 157379/400000 [00:18<00:27, 8895.86it/s] 40%|███▉      | 158274/400000 [00:18<00:27, 8909.90it/s] 40%|███▉      | 159167/400000 [00:18<00:27, 8780.11it/s] 40%|████      | 160047/400000 [00:18<00:27, 8690.43it/s] 40%|████      | 160996/400000 [00:18<00:26, 8914.68it/s] 40%|████      | 161907/400000 [00:19<00:26, 8971.51it/s] 41%|████      | 162839/400000 [00:19<00:26, 9071.14it/s] 41%|████      | 163748/400000 [00:19<00:26, 9001.80it/s] 41%|████      | 164650/400000 [00:19<00:28, 8352.68it/s] 41%|████▏     | 165496/400000 [00:19<00:28, 8308.26it/s] 42%|████▏     | 166398/400000 [00:19<00:27, 8507.09it/s] 42%|████▏     | 167303/400000 [00:19<00:26, 8660.54it/s] 42%|████▏     | 168189/400000 [00:19<00:26, 8718.57it/s] 42%|████▏     | 169065/400000 [00:19<00:26, 8672.17it/s] 42%|████▏     | 169948/400000 [00:20<00:26, 8718.85it/s] 43%|████▎     | 170822/400000 [00:20<00:26, 8630.08it/s] 43%|████▎     | 171687/400000 [00:20<00:26, 8546.98it/s] 43%|████▎     | 172543/400000 [00:20<00:27, 8384.90it/s] 43%|████▎     | 173405/400000 [00:20<00:26, 8452.74it/s] 44%|████▎     | 174252/400000 [00:20<00:27, 8344.66it/s] 44%|████▍     | 175088/400000 [00:20<00:27, 8293.97it/s] 44%|████▍     | 175936/400000 [00:20<00:26, 8348.85it/s] 44%|████▍     | 176791/400000 [00:20<00:26, 8406.61it/s] 44%|████▍     | 177633/400000 [00:20<00:26, 8399.46it/s] 45%|████▍     | 178518/400000 [00:21<00:25, 8528.88it/s] 45%|████▍     | 179372/400000 [00:21<00:26, 8269.96it/s] 45%|████▌     | 180202/400000 [00:21<00:26, 8185.01it/s] 45%|████▌     | 181090/400000 [00:21<00:26, 8378.34it/s] 45%|████▌     | 181963/400000 [00:21<00:25, 8477.85it/s] 46%|████▌     | 182813/400000 [00:21<00:25, 8477.08it/s] 46%|████▌     | 183663/400000 [00:21<00:25, 8395.57it/s] 46%|████▌     | 184504/400000 [00:21<00:25, 8397.87it/s] 46%|████▋     | 185345/400000 [00:21<00:26, 8089.98it/s] 47%|████▋     | 186233/400000 [00:21<00:25, 8310.20it/s] 47%|████▋     | 187145/400000 [00:22<00:24, 8536.91it/s] 47%|████▋     | 188038/400000 [00:22<00:24, 8648.78it/s] 47%|████▋     | 188939/400000 [00:22<00:24, 8752.16it/s] 47%|████▋     | 189826/400000 [00:22<00:23, 8784.68it/s] 48%|████▊     | 190707/400000 [00:22<00:24, 8626.88it/s] 48%|████▊     | 191593/400000 [00:22<00:23, 8691.51it/s] 48%|████▊     | 192475/400000 [00:22<00:23, 8729.62it/s] 48%|████▊     | 193350/400000 [00:22<00:24, 8600.23it/s] 49%|████▊     | 194212/400000 [00:22<00:24, 8554.64it/s] 49%|████▉     | 195081/400000 [00:22<00:23, 8594.42it/s] 49%|████▉     | 195992/400000 [00:23<00:23, 8741.21it/s] 49%|████▉     | 196932/400000 [00:23<00:22, 8927.20it/s] 49%|████▉     | 197866/400000 [00:23<00:22, 9046.03it/s] 50%|████▉     | 198773/400000 [00:23<00:22, 8983.82it/s] 50%|████▉     | 199673/400000 [00:23<00:22, 8812.56it/s] 50%|█████     | 200556/400000 [00:23<00:22, 8799.02it/s] 50%|█████     | 201438/400000 [00:23<00:22, 8795.71it/s] 51%|█████     | 202361/400000 [00:23<00:22, 8919.67it/s] 51%|█████     | 203254/400000 [00:23<00:22, 8872.80it/s] 51%|█████     | 204143/400000 [00:24<00:22, 8580.05it/s] 51%|█████▏    | 205032/400000 [00:24<00:22, 8668.83it/s] 51%|█████▏    | 205916/400000 [00:24<00:22, 8717.26it/s] 52%|█████▏    | 206790/400000 [00:24<00:22, 8584.14it/s] 52%|█████▏    | 207650/400000 [00:24<00:22, 8407.54it/s] 52%|█████▏    | 208493/400000 [00:24<00:23, 8294.31it/s] 52%|█████▏    | 209325/400000 [00:24<00:23, 8207.79it/s] 53%|█████▎    | 210148/400000 [00:24<00:23, 8042.44it/s] 53%|█████▎    | 210990/400000 [00:24<00:23, 8150.61it/s] 53%|█████▎    | 211807/400000 [00:24<00:23, 8129.28it/s] 53%|█████▎    | 212698/400000 [00:25<00:22, 8347.20it/s] 53%|█████▎    | 213535/400000 [00:25<00:22, 8112.97it/s] 54%|█████▎    | 214457/400000 [00:25<00:22, 8414.68it/s] 54%|█████▍    | 215363/400000 [00:25<00:21, 8595.66it/s] 54%|█████▍    | 216227/400000 [00:25<00:21, 8444.78it/s] 54%|█████▍    | 217076/400000 [00:25<00:21, 8347.21it/s] 54%|█████▍    | 217914/400000 [00:25<00:22, 7983.63it/s] 55%|█████▍    | 218779/400000 [00:25<00:22, 8172.32it/s] 55%|█████▍    | 219685/400000 [00:25<00:21, 8418.13it/s] 55%|█████▌    | 220533/400000 [00:25<00:21, 8286.52it/s] 55%|█████▌    | 221399/400000 [00:26<00:21, 8392.07it/s] 56%|█████▌    | 222242/400000 [00:26<00:21, 8325.31it/s] 56%|█████▌    | 223157/400000 [00:26<00:20, 8554.93it/s] 56%|█████▌    | 224018/400000 [00:26<00:20, 8569.24it/s] 56%|█████▌    | 224878/400000 [00:26<00:20, 8527.98it/s] 56%|█████▋    | 225791/400000 [00:26<00:20, 8696.65it/s] 57%|█████▋    | 226686/400000 [00:26<00:19, 8766.83it/s] 57%|█████▋    | 227582/400000 [00:26<00:19, 8821.51it/s] 57%|█████▋    | 228466/400000 [00:26<00:19, 8693.50it/s] 57%|█████▋    | 229337/400000 [00:26<00:19, 8543.58it/s] 58%|█████▊    | 230205/400000 [00:27<00:19, 8581.45it/s] 58%|█████▊    | 231065/400000 [00:27<00:19, 8519.57it/s] 58%|█████▊    | 231918/400000 [00:27<00:20, 8329.78it/s] 58%|█████▊    | 232753/400000 [00:27<00:20, 8331.93it/s] 58%|█████▊    | 233608/400000 [00:27<00:19, 8394.83it/s] 59%|█████▊    | 234493/400000 [00:27<00:19, 8525.81it/s] 59%|█████▉    | 235402/400000 [00:27<00:18, 8686.14it/s] 59%|█████▉    | 236273/400000 [00:27<00:19, 8472.91it/s] 59%|█████▉    | 237123/400000 [00:27<00:19, 8345.79it/s] 59%|█████▉    | 237960/400000 [00:28<00:19, 8328.12it/s] 60%|█████▉    | 238839/400000 [00:28<00:19, 8458.66it/s] 60%|█████▉    | 239757/400000 [00:28<00:18, 8660.19it/s] 60%|██████    | 240698/400000 [00:28<00:17, 8869.65it/s] 60%|██████    | 241588/400000 [00:28<00:18, 8779.69it/s] 61%|██████    | 242469/400000 [00:28<00:18, 8504.79it/s] 61%|██████    | 243323/400000 [00:28<00:18, 8464.26it/s] 61%|██████    | 244246/400000 [00:28<00:17, 8680.29it/s] 61%|██████▏   | 245157/400000 [00:28<00:17, 8804.54it/s] 62%|██████▏   | 246041/400000 [00:28<00:17, 8771.25it/s] 62%|██████▏   | 246920/400000 [00:29<00:17, 8760.14it/s] 62%|██████▏   | 247798/400000 [00:29<00:17, 8507.70it/s] 62%|██████▏   | 248675/400000 [00:29<00:17, 8583.49it/s] 62%|██████▏   | 249539/400000 [00:29<00:17, 8597.71it/s] 63%|██████▎   | 250401/400000 [00:29<00:17, 8415.40it/s] 63%|██████▎   | 251256/400000 [00:29<00:17, 8451.02it/s] 63%|██████▎   | 252147/400000 [00:29<00:17, 8583.15it/s] 63%|██████▎   | 253060/400000 [00:29<00:16, 8740.11it/s] 63%|██████▎   | 253951/400000 [00:29<00:16, 8788.93it/s] 64%|██████▎   | 254877/400000 [00:29<00:16, 8923.84it/s] 64%|██████▍   | 255771/400000 [00:30<00:16, 8829.60it/s] 64%|██████▍   | 256656/400000 [00:30<00:16, 8811.23it/s] 64%|██████▍   | 257554/400000 [00:30<00:16, 8858.40it/s] 65%|██████▍   | 258441/400000 [00:30<00:16, 8642.17it/s] 65%|██████▍   | 259352/400000 [00:30<00:16, 8776.79it/s] 65%|██████▌   | 260232/400000 [00:30<00:15, 8780.91it/s] 65%|██████▌   | 261112/400000 [00:30<00:15, 8763.95it/s] 65%|██████▌   | 261990/400000 [00:30<00:15, 8746.34it/s] 66%|██████▌   | 262866/400000 [00:30<00:15, 8694.16it/s] 66%|██████▌   | 263736/400000 [00:30<00:15, 8685.38it/s] 66%|██████▌   | 264605/400000 [00:31<00:16, 8391.96it/s] 66%|██████▋   | 265447/400000 [00:31<00:16, 8103.05it/s] 67%|██████▋   | 266306/400000 [00:31<00:16, 8241.93it/s] 67%|██████▋   | 267248/400000 [00:31<00:15, 8562.19it/s] 67%|██████▋   | 268110/400000 [00:31<00:15, 8352.83it/s] 67%|██████▋   | 268967/400000 [00:31<00:15, 8416.66it/s] 67%|██████▋   | 269837/400000 [00:31<00:15, 8496.76it/s] 68%|██████▊   | 270694/400000 [00:31<00:15, 8515.59it/s] 68%|██████▊   | 271548/400000 [00:31<00:15, 8379.68it/s] 68%|██████▊   | 272411/400000 [00:32<00:15, 8452.47it/s] 68%|██████▊   | 273258/400000 [00:32<00:15, 8437.88it/s] 69%|██████▊   | 274200/400000 [00:32<00:14, 8709.22it/s] 69%|██████▉   | 275074/400000 [00:32<00:14, 8699.61it/s] 69%|██████▉   | 275946/400000 [00:32<00:14, 8595.88it/s] 69%|██████▉   | 276820/400000 [00:32<00:14, 8635.27it/s] 69%|██████▉   | 277728/400000 [00:32<00:13, 8762.43it/s] 70%|██████▉   | 278610/400000 [00:32<00:13, 8778.75it/s] 70%|██████▉   | 279512/400000 [00:32<00:13, 8849.32it/s] 70%|███████   | 280428/400000 [00:32<00:13, 8937.87it/s] 70%|███████   | 281353/400000 [00:33<00:13, 9025.05it/s] 71%|███████   | 282257/400000 [00:33<00:13, 8984.55it/s] 71%|███████   | 283210/400000 [00:33<00:12, 9140.03it/s] 71%|███████   | 284165/400000 [00:33<00:12, 9257.07it/s] 71%|███████▏  | 285092/400000 [00:33<00:12, 8900.64it/s] 71%|███████▏  | 285986/400000 [00:33<00:13, 8689.89it/s] 72%|███████▏  | 286859/400000 [00:33<00:13, 8632.61it/s] 72%|███████▏  | 287725/400000 [00:33<00:13, 8605.78it/s] 72%|███████▏  | 288633/400000 [00:33<00:12, 8740.72it/s] 72%|███████▏  | 289509/400000 [00:33<00:12, 8695.72it/s] 73%|███████▎  | 290380/400000 [00:34<00:12, 8665.23it/s] 73%|███████▎  | 291314/400000 [00:34<00:12, 8854.79it/s] 73%|███████▎  | 292202/400000 [00:34<00:12, 8346.19it/s] 73%|███████▎  | 293044/400000 [00:34<00:12, 8352.41it/s] 73%|███████▎  | 293885/400000 [00:34<00:12, 8255.17it/s] 74%|███████▎  | 294716/400000 [00:34<00:12, 8269.45it/s] 74%|███████▍  | 295671/400000 [00:34<00:12, 8614.60it/s] 74%|███████▍  | 296592/400000 [00:34<00:11, 8782.75it/s] 74%|███████▍  | 297535/400000 [00:34<00:11, 8966.36it/s] 75%|███████▍  | 298436/400000 [00:34<00:11, 8886.55it/s] 75%|███████▍  | 299328/400000 [00:35<00:11, 8733.11it/s] 75%|███████▌  | 300223/400000 [00:35<00:11, 8796.68it/s] 75%|███████▌  | 301172/400000 [00:35<00:10, 8991.79it/s] 76%|███████▌  | 302103/400000 [00:35<00:10, 9083.80it/s] 76%|███████▌  | 303014/400000 [00:35<00:10, 8940.52it/s] 76%|███████▌  | 303910/400000 [00:35<00:11, 8594.72it/s] 76%|███████▌  | 304825/400000 [00:35<00:10, 8751.71it/s] 76%|███████▋  | 305746/400000 [00:35<00:10, 8882.19it/s] 77%|███████▋  | 306688/400000 [00:35<00:10, 9034.43it/s] 77%|███████▋  | 307609/400000 [00:36<00:10, 9083.89it/s] 77%|███████▋  | 308520/400000 [00:36<00:10, 8706.21it/s] 77%|███████▋  | 309396/400000 [00:36<00:10, 8632.10it/s] 78%|███████▊  | 310312/400000 [00:36<00:10, 8782.84it/s] 78%|███████▊  | 311194/400000 [00:36<00:10, 8646.33it/s] 78%|███████▊  | 312062/400000 [00:36<00:10, 8341.08it/s] 78%|███████▊  | 312902/400000 [00:36<00:10, 8356.86it/s] 78%|███████▊  | 313741/400000 [00:36<00:11, 7688.28it/s] 79%|███████▊  | 314523/400000 [00:36<00:11, 7303.00it/s] 79%|███████▉  | 315277/400000 [00:36<00:11, 7370.00it/s] 79%|███████▉  | 316076/400000 [00:37<00:11, 7545.51it/s] 79%|███████▉  | 316992/400000 [00:37<00:10, 7966.33it/s] 79%|███████▉  | 317902/400000 [00:37<00:09, 8273.94it/s] 80%|███████▉  | 318799/400000 [00:37<00:09, 8469.48it/s] 80%|███████▉  | 319655/400000 [00:37<00:09, 8316.57it/s] 80%|████████  | 320495/400000 [00:37<00:09, 8338.43it/s] 80%|████████  | 321449/400000 [00:37<00:09, 8665.52it/s] 81%|████████  | 322339/400000 [00:37<00:08, 8731.67it/s] 81%|████████  | 323222/400000 [00:37<00:08, 8756.73it/s] 81%|████████  | 324102/400000 [00:38<00:08, 8694.55it/s] 81%|████████  | 324974/400000 [00:38<00:08, 8509.46it/s] 81%|████████▏ | 325856/400000 [00:38<00:08, 8599.40it/s] 82%|████████▏ | 326771/400000 [00:38<00:08, 8754.32it/s] 82%|████████▏ | 327649/400000 [00:38<00:08, 8733.57it/s] 82%|████████▏ | 328524/400000 [00:38<00:08, 8410.93it/s] 82%|████████▏ | 329369/400000 [00:38<00:08, 8362.01it/s] 83%|████████▎ | 330208/400000 [00:38<00:08, 8227.69it/s] 83%|████████▎ | 331063/400000 [00:38<00:08, 8320.22it/s] 83%|████████▎ | 331933/400000 [00:38<00:08, 8428.28it/s] 83%|████████▎ | 332778/400000 [00:39<00:08, 8298.20it/s] 83%|████████▎ | 333623/400000 [00:39<00:07, 8337.92it/s] 84%|████████▎ | 334479/400000 [00:39<00:07, 8402.87it/s] 84%|████████▍ | 335412/400000 [00:39<00:07, 8660.59it/s] 84%|████████▍ | 336281/400000 [00:39<00:07, 8437.52it/s] 84%|████████▍ | 337128/400000 [00:39<00:07, 8094.24it/s] 85%|████████▍ | 338083/400000 [00:39<00:07, 8481.86it/s] 85%|████████▍ | 339002/400000 [00:39<00:07, 8682.01it/s] 85%|████████▍ | 339878/400000 [00:39<00:06, 8676.20it/s] 85%|████████▌ | 340764/400000 [00:39<00:06, 8728.84it/s] 85%|████████▌ | 341641/400000 [00:40<00:06, 8553.63it/s] 86%|████████▌ | 342526/400000 [00:40<00:06, 8640.16it/s] 86%|████████▌ | 343408/400000 [00:40<00:06, 8690.48it/s] 86%|████████▌ | 344323/400000 [00:40<00:06, 8821.69it/s] 86%|████████▋ | 345207/400000 [00:40<00:06, 8595.70it/s] 87%|████████▋ | 346070/400000 [00:40<00:06, 8542.19it/s] 87%|████████▋ | 347010/400000 [00:40<00:06, 8782.56it/s] 87%|████████▋ | 347918/400000 [00:40<00:05, 8868.46it/s] 87%|████████▋ | 348829/400000 [00:40<00:05, 8937.75it/s] 87%|████████▋ | 349725/400000 [00:40<00:05, 8898.42it/s] 88%|████████▊ | 350617/400000 [00:41<00:05, 8689.71it/s] 88%|████████▊ | 351488/400000 [00:41<00:05, 8553.00it/s] 88%|████████▊ | 352346/400000 [00:41<00:05, 8467.60it/s] 88%|████████▊ | 353195/400000 [00:41<00:05, 8336.75it/s] 89%|████████▊ | 354031/400000 [00:41<00:05, 8278.61it/s] 89%|████████▊ | 354878/400000 [00:41<00:05, 8332.83it/s] 89%|████████▉ | 355716/400000 [00:41<00:05, 8344.77it/s] 89%|████████▉ | 356552/400000 [00:41<00:05, 8297.47it/s] 89%|████████▉ | 357383/400000 [00:41<00:05, 8196.28it/s] 90%|████████▉ | 358228/400000 [00:42<00:05, 8268.79it/s] 90%|████████▉ | 359088/400000 [00:42<00:04, 8363.01it/s] 90%|████████▉ | 359965/400000 [00:42<00:04, 8480.32it/s] 90%|█████████ | 360896/400000 [00:42<00:04, 8710.80it/s] 90%|█████████ | 361849/400000 [00:42<00:04, 8938.69it/s] 91%|█████████ | 362746/400000 [00:42<00:04, 8865.19it/s] 91%|█████████ | 363635/400000 [00:42<00:04, 8661.11it/s] 91%|█████████ | 364558/400000 [00:42<00:04, 8821.37it/s] 91%|█████████▏| 365463/400000 [00:42<00:03, 8885.47it/s] 92%|█████████▏| 366354/400000 [00:42<00:03, 8623.20it/s] 92%|█████████▏| 367261/400000 [00:43<00:03, 8750.73it/s] 92%|█████████▏| 368139/400000 [00:43<00:03, 8667.50it/s] 92%|█████████▏| 369039/400000 [00:43<00:03, 8764.61it/s] 92%|█████████▏| 369972/400000 [00:43<00:03, 8926.71it/s] 93%|█████████▎| 370867/400000 [00:43<00:03, 8857.10it/s] 93%|█████████▎| 371755/400000 [00:43<00:03, 8657.31it/s] 93%|█████████▎| 372623/400000 [00:43<00:03, 8550.28it/s] 93%|█████████▎| 373538/400000 [00:43<00:03, 8720.51it/s] 94%|█████████▎| 374413/400000 [00:43<00:02, 8560.32it/s] 94%|█████████▍| 375281/400000 [00:43<00:02, 8595.36it/s] 94%|█████████▍| 376201/400000 [00:44<00:02, 8767.17it/s] 94%|█████████▍| 377080/400000 [00:44<00:02, 8754.45it/s] 94%|█████████▍| 377973/400000 [00:44<00:02, 8804.65it/s] 95%|█████████▍| 378855/400000 [00:44<00:02, 8566.72it/s] 95%|█████████▍| 379714/400000 [00:44<00:02, 8571.16it/s] 95%|█████████▌| 380598/400000 [00:44<00:02, 8649.04it/s] 95%|█████████▌| 381465/400000 [00:44<00:02, 8378.94it/s] 96%|█████████▌| 382326/400000 [00:44<00:02, 8444.99it/s] 96%|█████████▌| 383245/400000 [00:44<00:01, 8653.34it/s] 96%|█████████▌| 384156/400000 [00:44<00:01, 8783.28it/s] 96%|█████████▋| 385047/400000 [00:45<00:01, 8817.43it/s] 96%|█████████▋| 385931/400000 [00:45<00:01, 8728.49it/s] 97%|█████████▋| 386837/400000 [00:45<00:01, 8824.19it/s] 97%|█████████▋| 387723/400000 [00:45<00:01, 8831.24it/s] 97%|█████████▋| 388607/400000 [00:45<00:01, 8791.07it/s] 97%|█████████▋| 389530/400000 [00:45<00:01, 8915.70it/s] 98%|█████████▊| 390423/400000 [00:45<00:01, 8647.53it/s] 98%|█████████▊| 391341/400000 [00:45<00:00, 8800.53it/s] 98%|█████████▊| 392224/400000 [00:45<00:00, 8712.81it/s] 98%|█████████▊| 393101/400000 [00:46<00:00, 8727.74it/s] 99%|█████████▊| 394048/400000 [00:46<00:00, 8937.33it/s] 99%|█████████▊| 394944/400000 [00:46<00:00, 8614.59it/s] 99%|█████████▉| 395824/400000 [00:46<00:00, 8667.04it/s] 99%|█████████▉| 396694/400000 [00:46<00:00, 8593.28it/s] 99%|█████████▉| 397556/400000 [00:46<00:00, 8534.67it/s]100%|█████████▉| 398412/400000 [00:46<00:00, 8489.72it/s]100%|█████████▉| 399263/400000 [00:46<00:00, 8333.05it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8543.52it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f22c0052748>, <torchtext.data.dataset.TabularDataset object at 0x7f22c0052898>, <torchtext.vocab.Vocab object at 0x7f22c00527b8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 11.23 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 12.64 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 12.64 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 12.64 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.15 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.53 file/s]2020-06-28 00:17:52.824102: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-28 00:17:52.828221: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-06-28 00:17:52.828426: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dc43e94e60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-28 00:17:52.828459: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 144446.33it/s] 68%|██████▊   | 6782976/9912422 [00:00<00:15, 206161.01it/s]9920512it [00:00, 41812013.31it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1152585.90it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 466743.86it/s]1654784it [00:00, 11839091.09it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191890.60it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
